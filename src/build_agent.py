"""Single-repo CLI entrypoint for BuildAgent v2.0.

Usage:
    python src/build_agent.py <repo_url> [--db results.db]

Clones a repository, generates a blueprint, runs the agent loop,
and prints the result.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from db.models import BatchRun, RunRecord
from db.writer import DBWriter
from agent.llm import LLMClient
from agent.blueprint import ImageCatalog, generate_blueprint
from agent.react_loop import run_agent
from agent.docker_ops import DockerOps
from parallel.rate_limiter import GlobalRateLimiter
from parallel.disk_monitor import DiskSpaceMonitor
from parallel.worker import make_slug, clone_repo

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="BuildAgent v2.0 — generate a Dockerfile for a GitHub repository",
    )
    parser.add_argument("repo_url", help="GitHub repository URL")
    parser.add_argument(
        "--db", default=os.environ.get("DATABASE_URL", "sqlite:///results.db"),
        help="Database URL (default: sqlite:///results.db)",
    )
    parser.add_argument(
        "--max-iterations", type=int,
        default=int(os.environ.get("MAX_ITERATIONS", "3")),
        help="Maximum agent iterations (default: 3)",
    )
    parser.add_argument(
        "--workdir", default="workdir",
        help="Working directory for cloned repos (default: workdir)",
    )
    args = parser.parse_args(argv)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    repo_url = args.repo_url
    slug = make_slug(repo_url)

    print(f"BuildAgent v2.0 — processing {slug}")
    print(f"  Repository: {repo_url}")
    print(f"  Database:   {args.db}")
    print()

    # Initialize shared resources
    rate_limiter = GlobalRateLimiter()
    build_semaphore = threading.Semaphore(
        int(os.environ.get("DOCKER_BUILD_CONCURRENCY", "2"))
    )
    disk_monitor = DiskSpaceMonitor()
    db = DBWriter(args.db)
    docker_ops = DockerOps(build_semaphore=build_semaphore)
    llm = LLMClient(rate_limiter)

    # Create batch (even for single repo, keeps DB schema consistent)
    batch = BatchRun(worker_count=1, repo_count=1)
    db.write_batch_start(batch)

    clone_dir = Path(args.workdir) / batch.id / "0" / slug
    image_name = f"buildagent-{slug}-0"

    run_record = RunRecord(
        batch_id=batch.id,
        repo_url=repo_url,
        repo_slug=slug,
        status="running",
        worker_id=0,
    )
    db.write_run_start(run_record)

    t0 = time.monotonic()

    try:
        # Disk check
        disk_monitor.check_or_wait()

        # Clone
        print(f"[1/3] Cloning {repo_url}...")
        clone_repo(repo_url, clone_dir)

        # Phase 0+1: Blueprint
        print("[2/3] Generating blueprint...")
        image_catalog = ImageCatalog().get()

        bp_t0 = time.monotonic()
        blueprint, bp_pt, bp_ct = generate_blueprint(str(clone_dir), image_catalog, llm)
        bp_dur = int((time.monotonic() - bp_t0) * 1000)

        run_record.context_blueprint = json.dumps(blueprint)
        run_record.detected_language = blueprint.get("language")
        run_record.repo_type = blueprint.get("repo_type")
        run_record.blueprint_tokens_prompt = bp_pt
        run_record.blueprint_tokens_completion = bp_ct
        run_record.blueprint_duration_ms = bp_dur
        db.update_run_blueprint(run_record)

        print(f"  Language: {blueprint.get('language', 'unknown')}")
        print(f"  Base image: {blueprint.get('base_image', 'unknown')}")
        print()

        # Phase 2: Agent loop
        print("[3/3] Running agent loop...")
        run_record = run_agent(
            repo_root=clone_dir,
            blueprint=blueprint,
            llm=llm,
            docker_ops=docker_ops,
            image_name=image_name,
            db=db,
            run_record=run_record,
            max_iterations=args.max_iterations,
        )

        if run_record.smoke_test_passed:
            run_record.status = "success"
        else:
            run_record.status = "failure"

    except KeyboardInterrupt:
        run_record.status = "error"
        run_record.error_message = "Interrupted by user"
        print("\nInterrupted.")
    except Exception as e:
        run_record.status = "error"
        run_record.error_message = str(e)
        logger.exception("Fatal error: %s", e)
    finally:
        elapsed = int((time.monotonic() - t0) * 1000)
        run_record.duration_ms = elapsed
        run_record.finished_at = datetime.now(timezone.utc)

        run_record.total_steps = sum(it.step_count for it in run_record.iterations)
        run_record.total_prompt_tokens = (
            run_record.blueprint_tokens_prompt
            + sum(it.prompt_tokens for it in run_record.iterations)
        )
        run_record.total_completion_tokens = (
            run_record.blueprint_tokens_completion
            + sum(it.completion_tokens for it in run_record.iterations)
        )

        db.write_run_finish(run_record)

        # Update batch
        batch.finished_at = datetime.now(timezone.utc)
        batch.success_count = 1 if run_record.status == "success" else 0
        batch.failure_count = 1 if run_record.status != "success" else 0
        batch.total_prompt_tokens = run_record.total_prompt_tokens
        batch.total_completion_tokens = run_record.total_completion_tokens
        db.write_batch_finish(batch)

        # Cleanup
        shutil.rmtree(clone_dir, ignore_errors=True)
        try:
            docker_ops.cleanup(image_name)
        except Exception:
            pass

        db.close()

    # Print result
    print()
    print("=" * 50)
    print(f"Result: {run_record.status.upper()}")
    print(f"Iterations: {run_record.iteration_count}")
    print(f"Total steps: {run_record.total_steps}")
    print(f"Duration: {elapsed / 1000:.1f}s")
    print(f"Tokens: {run_record.total_prompt_tokens:,} prompt, "
          f"{run_record.total_completion_tokens:,} completion")

    if run_record.final_dockerfile:
        print()
        print("Generated Dockerfile:")
        print("-" * 50)
        print(run_record.final_dockerfile)

    if run_record.error_message:
        print(f"\nError: {run_record.error_message}")

    return 0 if run_record.status == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
