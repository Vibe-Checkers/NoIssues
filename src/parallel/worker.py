"""Worker loop for BuildAgent v2.0.

Each worker processes one repository: clone → blueprint → agent → cleanup.
All exceptions are caught and logged to the database.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from db.models import RunRecord
from db.writer import DBWriter
from agent.llm import LLMClient
from agent.blueprint import generate_blueprint
from agent.react_loop import run_agent
from agent.docker_ops import DockerOps
from parallel.rate_limiter import GlobalRateLimiter
from parallel.disk_monitor import DiskSpaceMonitor

logger = logging.getLogger(__name__)


def make_slug(repo_url: str) -> str:
    """Convert a GitHub URL to a short slug (e.g., 'owner/repo' → 'owner-repo')."""
    # Strip .git suffix and trailing slash
    url = repo_url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    # Extract owner/repo from URL
    parts = url.split("/")
    if len(parts) >= 2:
        slug = f"{parts[-2]}-{parts[-1]}"
    else:
        slug = parts[-1]
    # Sanitize: keep only alphanumeric, dash, underscore
    slug = re.sub(r"[^a-zA-Z0-9_-]", "-", slug)
    return slug.lower()


def clone_repo(repo_url: str, dest: Path, timeout: int = 120, retries: int = 3) -> None:
    """Shallow-clone a git repository with retry and non-interactive git settings."""
    logger.info("Cloning %s → %s", repo_url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_ASKPASS"] = "echo"

    backoffs = [0, 5, 15]
    last_exc = None
    for attempt in range(1, retries + 1):
        if attempt > 1:
            wait_s = backoffs[min(attempt - 1, len(backoffs) - 1)]
            logger.warning("Clone retry %d/%d for %s after %ds", attempt, retries, repo_url, wait_s)
            time.sleep(wait_s)

        # A failed/partial clone can leave dest populated. Ensure each attempt starts clean.
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)

        t0 = time.monotonic()
        try:
            subprocess.run(
                [
                    "git", "clone",
                    "--depth", "1",
                    "--single-branch",
                    "--filter=blob:none",
                    repo_url,
                    str(dest),
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True,
                env=env,
            )
            logger.info("Clone succeeded for %s on attempt %d in %.1fs", repo_url, attempt, time.monotonic() - t0)
            return
        except subprocess.TimeoutExpired as e:
            last_exc = e
            logger.warning(
                "Clone timeout for %s on attempt %d/%d after %ds",
                repo_url,
                attempt,
                retries,
                timeout,
            )
        except subprocess.CalledProcessError as e:
            last_exc = e
            err = (e.stderr or e.stdout or "")[-500:]
            logger.warning(
                "Clone failed for %s on attempt %d/%d: %s",
                repo_url,
                attempt,
                retries,
                err,
            )

    if last_exc is not None:
        raise last_exc


def worker_loop(
    worker_id: int,
    repo_url: str,
    batch_id: str,
    image_catalog: str,
    rate_limiter: GlobalRateLimiter,
    build_semaphore: threading.Semaphore,
    disk_monitor: DiskSpaceMonitor,
    db: DBWriter,
    workdir: str = "workdir",
    heartbeat_registry: dict[str, dict] | None = None,
    heartbeat_lock: threading.Lock | None = None,
    ablation: str = "default",
) -> None:
    """Process a single repository end-to-end.

    All exceptions are caught — the run is recorded as 'error' in the DB.
    """
    slug = make_slug(repo_url)
    clone_dir = Path(workdir) / batch_id / str(worker_id) / slug
    image_name = f"buildagent-{slug}-{worker_id}"
    llm = LLMClient(rate_limiter, worker_id=worker_id)
    docker_ops = DockerOps(build_semaphore=build_semaphore)

    run_id = db.get_run_id_for_repo(batch_id, slug)
    if run_id:
        run_record = RunRecord(
            id=run_id,
            batch_id=batch_id,
            repo_url=repo_url,
            repo_slug=slug,
            status="running",
            worker_id=worker_id,
            started_at=datetime.now(timezone.utc),
        )
        db.update_run_start(run_record)
    else:
        run_record = RunRecord(
            batch_id=batch_id,
            repo_url=repo_url,
            repo_slug=slug,
            status="running",
            worker_id=worker_id,
        )
        db.write_run_start(run_record)

    t0 = time.monotonic()

    def heartbeat(phase: str, extra: str = "") -> None:
        msg = f"[worker-{worker_id}] {slug}: phase={phase}"
        if extra:
            msg += f" {extra}"
        logger.info(msg)
        if heartbeat_registry is not None and heartbeat_lock is not None:
            with heartbeat_lock:
                heartbeat_registry[run_record.id] = {
                    "worker_id": worker_id,
                    "slug": slug,
                    "phase": phase,
                    "ts": time.monotonic(),
                    "extra": extra,
                }

    try:
        # Disk check — block if low
        heartbeat("disk_check")
        disk_monitor.check_or_wait()

        # Clone
        heartbeat("clone_start", f"repo={repo_url}")
        clone_repo(repo_url, clone_dir)
        heartbeat("clone_done")

        # Phase 1: Blueprint
        if ablation == "no-metaprompt":
            heartbeat("blueprint_skipped")
            blueprint = {}
            collected_context = None
            bp_pt, bp_ct, bp_dur = 0, 0, 0
        else:
            heartbeat("blueprint_start")
            bp_t0 = time.monotonic()
            blueprint, collected_context, bp_pt, bp_ct = generate_blueprint(str(clone_dir), image_catalog, llm)
            bp_dur = int((time.monotonic() - bp_t0) * 1000)
            heartbeat("blueprint_done", f"duration_ms={bp_dur}")

        run_record.context_blueprint = json.dumps(blueprint)
        run_record.detected_language = blueprint.get("language")
        run_record.repo_type = blueprint.get("repo_type")
        run_record.blueprint_tokens_prompt = bp_pt
        run_record.blueprint_tokens_completion = bp_ct
        run_record.blueprint_duration_ms = bp_dur
        db.update_run_blueprint(run_record)

        # Phase 2: Agent loop
        heartbeat("agent_start")
        run_record = run_agent(
            repo_root=clone_dir,
            blueprint=blueprint,
            llm=llm,
            docker_ops=docker_ops,
            image_name=image_name,
            db=db,
            run_record=run_record,
            collected_context=collected_context,
            ablation=ablation,
        )
        heartbeat("agent_done", f"iterations={run_record.iteration_count}")

        # Determine final status
        if run_record.smoke_test_passed:
            run_record.status = "success"
        else:
            run_record.status = "failure"

    except Exception as e:
        run_record.status = "error"
        run_record.error_message = str(e)
        logger.exception("[worker-%d] %s: %s", worker_id, slug, e)

    finally:
        # Cleanup always runs
        heartbeat("cleanup_start")
        elapsed = int((time.monotonic() - t0) * 1000)
        run_record.duration_ms = elapsed
        run_record.finished_at = datetime.now(timezone.utc)

        # Tally totals
        run_record.total_steps = sum(
            it.step_count for it in run_record.iterations
        )
        run_record.total_prompt_tokens = (
            run_record.blueprint_tokens_prompt
            + sum(it.prompt_tokens for it in run_record.iterations)
        )
        run_record.total_completion_tokens = (
            run_record.blueprint_tokens_completion
            + sum(it.completion_tokens for it in run_record.iterations)
        )

        db.write_run_finish(run_record)

        if run_record.final_dockerfile:
            db.write_artifact(run_record.id, "dockerfile", "Dockerfile",
                              content=run_record.final_dockerfile)

        # Clean up filesystem and Docker image
        shutil.rmtree(clone_dir, ignore_errors=True)
        try:
            docker_ops.cleanup(image_name)
        except Exception:
            logger.debug("Cleanup failed for %s (may not exist)", image_name)

        if heartbeat_registry is not None and heartbeat_lock is not None:
            with heartbeat_lock:
                heartbeat_registry.pop(run_record.id, None)

    logger.info(
        "[worker-%d] %s: %s in %.1fs",
        worker_id, slug, run_record.status,
        elapsed / 1000,
    )
