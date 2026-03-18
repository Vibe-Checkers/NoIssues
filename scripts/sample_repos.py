#!/usr/bin/env python3
"""Sample repositories from an input list by their run status.

Usage:
    python scripts/sample_repos.py <repos.txt> --status success --sample 20
    python scripts/sample_repos.py <repos.txt> --status running_stuck
    python scripts/sample_repos.py <repos.txt> --status not_run --sample 10 --seed 42

Statuses:
    success        - runs that completed successfully
    failure        - runs that failed
    error          - runs that errored
    running        - runs currently in progress
    running_stuck  - running runs with no other status, started > --stuck-hours ago
    not_run        - repos from input file with no entries in DB at all
"""

import argparse
import os
import random
import sys
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from db.writer import DBWriter


def normalize_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    return url.lower()


def load_input_urls(path: str) -> list[str]:
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return lines


def get_matching_repos(db: DBWriter, input_urls: list[str], status: str, stuck_hours: int) -> list[str]:
    # Build a map: normalized URL -> original URL (from input file)
    norm_to_orig = {normalize_url(u): u for u in input_urls}
    input_norm = set(norm_to_orig)

    if status == "not_run":
        rows = db._query("SELECT DISTINCT repo_url FROM run")
        db_norm = {normalize_url(row[0]) for row in rows if row[0]}
        matched_norm = input_norm - db_norm
        return [norm_to_orig[n] for n in matched_norm]

    if status == "running_stuck":
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=stuck_hours)).isoformat()
        rows = db._query(
            """
            SELECT DISTINCT r.repo_url FROM run r
            WHERE r.status = 'running'
              AND r.started_at < ?
              AND NOT EXISTS (
                  SELECT 1 FROM run r2
                  WHERE r2.repo_url = r.repo_url
                    AND r2.status != 'running'
              )
            """,
            (cutoff,),
        )
    else:
        rows = db._query(
            "SELECT DISTINCT repo_url FROM run WHERE status = ?",
            (status,),
        )

    db_norm = {normalize_url(row[0]): row[0] for row in rows if row[0]}
    matched = []
    for norm, db_url in db_norm.items():
        if norm in input_norm:
            matched.append(db_url)
    return matched


def main():
    parser = argparse.ArgumentParser(description="Sample repositories by run status.")
    parser.add_argument("input_file", help="Path to .txt file with one repo URL per line")
    parser.add_argument(
        "--status",
        required=True,
        choices=["success", "failure", "error", "running", "running_stuck", "not_run"],
        help="Status to filter by",
    )
    parser.add_argument("--sample", type=int, default=None, help="Number of repos to sample (default: all)")
    parser.add_argument("--stuck-hours", type=int, default=6, help="Hours threshold for running_stuck (default: 6)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    input_urls = load_input_urls(args.input_file)
    if not input_urls:
        print("No URLs found in input file.", file=sys.stderr)
        sys.exit(1)

    db = DBWriter()
    try:
        matched = get_matching_repos(db, input_urls, args.status, args.stuck_hours)
    finally:
        db.close()

    total_input = len(input_urls)
    total_matched = len(matched)

    if args.sample is not None and args.sample < total_matched:
        rng = random.Random(args.seed)
        result = rng.sample(matched, args.sample)
    else:
        result = matched

    print(
        f"Input: {total_input} repos | Matched ({args.status}): {total_matched} | Showing: {len(result)}",
        file=sys.stderr,
    )

    for url in result:
        print(url)


if __name__ == "__main__":
    main()
