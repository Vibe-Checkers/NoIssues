"""Parallel batch runner for BuildAgent v2.0.

Usage:
    python src/batch_runner.py repos.txt [--workers 4] [--db results.db]

Reads a list of GitHub repository URLs (one per line), processes them
in parallel using a thread pool, and writes results to the database.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from db.models import BatchRun
from db.writer import DBWriter
from agent.blueprint import ImageCatalog
from parallel.rate_limiter import GlobalRateLimiter
from parallel.disk_monitor import DiskSpaceMonitor
from parallel.worker import worker_loop, make_slug

logger = logging.getLogger(__name__)


def read_repo_list(path: str) -> list[str]:
    """Read repository URLs from a file (one per line, # comments allowed)."""
    repos = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                repos.append(line)
    return repos


def print_summary(db: DBWriter, batch_id: str, elapsed_s: float) -> None:
    """Print a post-run summary of the batch."""
    rows = db._query(
        "SELECT status, COUNT(*) FROM run WHERE batch_id=? GROUP BY status",
        (batch_id,),
    )
    counts = dict(rows)
    total = sum(counts.values())
    success = counts.get("success", 0)
    failure = counts.get("failure", 0)
    error = counts.get("error", 0)

    # Token totals
    tok_rows = db._query(
        "SELECT SUM(total_prompt_tokens), SUM(total_completion_tokens) FROM run WHERE batch_id=?",
        (batch_id,),
    )
    total_pt = tok_rows[0][0] or 0
    total_ct = tok_rows[0][1] or 0

    # Avg iterations for successful runs
    avg_rows = db._query(
        "SELECT AVG(iteration_count) FROM run WHERE batch_id=? AND status='success'",
        (batch_id,),
    )
    avg_iters = avg_rows[0][0] or 0

    hours = int(elapsed_s // 3600)
    mins = int((elapsed_s % 3600) // 60)
    secs = int(elapsed_s % 60)

    print()
    print("=" * 60)
    print(f"Batch complete: {total} repos in {hours}h {mins}m {secs}s")
    print(f"  Success: {success} ({success * 100 // max(total, 1)}%)")
    print(f"  Failure: {failure} ({failure * 100 // max(total, 1)}%)")
    print(f"  Error:   {error} ({error * 100 // max(total, 1)}%)")
    print(f"  Total tokens: {total_pt:,} prompt, {total_ct:,} completion")
    print(f"  Avg iterations/success: {avg_iters:.1f}")
    if total > 0:
        print(f"  Avg duration/repo: {elapsed_s / total:.0f}s")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="BuildAgent v2.0 — batch process repositories in parallel",
    )
    parser.add_argument("repo_list", help="Path to text file with repository URLs")
    parser.add_argument(
        "--workers", type=int,
        default=int(os.environ.get("WORKERS", "4")),
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--db", default=os.environ.get("DATABASE_URL", "sqlite:///results.db"),
        help="Database URL (default: sqlite:///results.db)",
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

    # Read repos
    repos = read_repo_list(args.repo_list)
    if not repos:
        print("No repositories to process.")
        return 0

    print(f"BuildAgent v2.0 — batch mode")
    print(f"  Repos:   {len(repos)}")
    print(f"  Workers: {args.workers}")
    print(f"  DB:      {args.db}")
    print()

    # Phase 0: Fetch image catalog once
    print("Fetching Docker image catalog...")
    image_catalog = ImageCatalog().get()

    # Initialize shared resources
    rate_limiter = GlobalRateLimiter()
    build_semaphore = threading.Semaphore(
        int(os.environ.get("DOCKER_BUILD_CONCURRENCY", "2"))
    )
    disk_monitor = DiskSpaceMonitor()
    db = DBWriter(args.db)

    # Create batch record
    config = {
        "workers": args.workers,
        "database": args.db,
        "max_iterations": int(os.environ.get("MAX_ITERATIONS", "3")),
    }
    batch = BatchRun(
        worker_count=args.workers,
        repo_count=len(repos),
        config_json=json.dumps(config),
    )
    db.write_batch_start(batch)

    # Crash recovery: skip already-successful repos
    completed = db.get_successful_slugs(batch.id)
    pending = [r for r in repos if make_slug(r) not in completed]
    if len(repos) != len(pending):
        print(f"  Skipping {len(repos) - len(pending)} already-successful repos")
    print(f"  Processing {len(pending)} repos")
    print()

    t0 = time.monotonic()
    done_count = 0
    success_count = 0
    fail_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for i, url in enumerate(pending):
            future = pool.submit(
                worker_loop,
                worker_id=i % args.workers,
                repo_url=url,
                batch_id=batch.id,
                image_catalog=image_catalog,
                rate_limiter=rate_limiter,
                build_semaphore=build_semaphore,
                disk_monitor=disk_monitor,
                db=db,
                workdir=args.workdir,
            )
            futures[future] = url

        for future in as_completed(futures):
            url = futures[future]
            slug = make_slug(url)
            done_count += 1

            try:
                future.result()
            except Exception as e:
                logger.error("Unhandled error for %s: %s", slug, e)
                error_count += 1

            elapsed = time.monotonic() - t0
            active = len(pending) - done_count
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"{done_count}/{len(pending)} complete | "
                f"{active} active | "
                f"{elapsed:.0f}s elapsed"
            )

    total_elapsed = time.monotonic() - t0

    # Finalize batch
    batch.finished_at = datetime.now(timezone.utc)

    # Query final counts from DB
    rows = db._query(
        "SELECT status, COUNT(*) FROM run WHERE batch_id=? GROUP BY status",
        (batch.id,),
    )
    counts = dict(rows)
    batch.success_count = counts.get("success", 0)
    batch.failure_count = (
        counts.get("failure", 0) + counts.get("error", 0)
    )

    tok_rows = db._query(
        "SELECT SUM(total_prompt_tokens), SUM(total_completion_tokens) FROM run WHERE batch_id=?",
        (batch.id,),
    )
    batch.total_prompt_tokens = tok_rows[0][0] or 0
    batch.total_completion_tokens = tok_rows[0][1] or 0

    db.write_batch_finish(batch)

    print_summary(db, batch.id, total_elapsed)

    db.close()

    return 0 if batch.failure_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
