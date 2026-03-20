#!/usr/bin/env python3
"""
fetch_all_results.py — Fetch all run data for the 282-repo dataset from PostgreSQL.

For each repo in our_282.txt:
  - Query all runs from the `run` table
  - If ANY run has status='success' → repo is SUCCESS, otherwise FAIL
  - Fetch full hierarchy: run → iterations → steps → verify_build_details
  - Fetch run_artifacts (final Dockerfiles, etc.)
  - Save per-repo JSON to result_analysis/per_repo/<slug>.json
  - Save summary CSV to result_analysis/summary.csv
  - Save aggregate stats to result_analysis/aggregate_stats.json

Usage:
    python result_analysis/fetch_all_results.py
"""

import csv
import json
import os
import sys
from datetime import datetime, date

import psycopg2
import psycopg2.extras

# ── Config ──────────────────────────────────────────────────────────────────

DB_URL = (
    "postgresql://neondb_owner:npg_NRtmXhij2JP8@"
    "ep-curly-rain-al6tqu0g-pooler.c-3.eu-central-1.aws.neon.tech/"
    "neondb?sslmode=require"
)

REPO_LIST = os.path.join(os.path.dirname(__file__), "..", "our_282.txt")
OUTPUT_DIR = os.path.dirname(__file__)
PER_REPO_DIR = os.path.join(OUTPUT_DIR, "per_repo")


# ── Helpers ─────────────────────────────────────────────────────────────────

def json_serial(obj):
    """JSON serializer for objects not serializable by default."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def dict_row(cursor):
    """Convert a cursor row to a dict using column names."""
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def load_repo_list(path: str) -> list[str]:
    """Load repo URLs from file, skip blanks and comments."""
    repos = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                repos.append(line)
    return repos


def url_to_slug(url: str) -> str:
    """Convert GitHub URL to owner--repo slug (matching BuildAgent convention)."""
    parts = url.rstrip("/").split("/")
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return url


# ── Query functions ─────────────────────────────────────────────────────────

def fetch_runs_for_repo(cur, repo_url: str) -> list[dict]:
    """Fetch all runs matching a repo URL."""
    cur.execute(
        "SELECT * FROM run WHERE repo_url = %s ORDER BY started_at",
        (repo_url,),
    )
    return dict_row(cur)


def fetch_iterations_for_run(cur, run_id: str) -> list[dict]:
    """Fetch all iterations for a given run."""
    cur.execute(
        "SELECT * FROM iteration WHERE run_id = %s ORDER BY iteration_number",
        (run_id,),
    )
    return dict_row(cur)


def fetch_steps_for_iteration(cur, iteration_id: str) -> list[dict]:
    """Fetch all steps for a given iteration."""
    cur.execute(
        "SELECT * FROM step WHERE iteration_id = %s ORDER BY step_number",
        (iteration_id,),
    )
    return dict_row(cur)


def fetch_verify_details_for_step(cur, step_id: str) -> list[dict]:
    """Fetch verify_build_detail rows for a step."""
    cur.execute(
        "SELECT * FROM verify_build_detail WHERE step_id = %s",
        (step_id,),
    )
    return dict_row(cur)


def fetch_artifacts_for_run(cur, run_id: str) -> list[dict]:
    """Fetch all artifacts for a run."""
    cur.execute(
        "SELECT * FROM run_artifact WHERE run_id = %s ORDER BY created_at",
        (run_id,),
    )
    return dict_row(cur)


def fetch_batch_info(cur, batch_id: str) -> dict | None:
    """Fetch batch_run info."""
    cur.execute("SELECT * FROM batch_run WHERE id = %s", (batch_id,))
    rows = dict_row(cur)
    return rows[0] if rows else None


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(PER_REPO_DIR, exist_ok=True)

    print("Connecting to PostgreSQL...")
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    # Load repo list
    repos = load_repo_list(REPO_LIST)
    print(f"Loaded {len(repos)} repos from our_282.txt")

    # Also try matching by slug in case URLs differ slightly
    # First, get all distinct repo_urls in the DB for cross-reference
    cur.execute("SELECT DISTINCT repo_url FROM run")
    db_urls = {row[0] for row in cur.fetchall()}
    print(f"Found {len(db_urls)} distinct repo_urls in the run table")

    summary_rows = []
    all_results = {}
    found_count = 0
    not_found_count = 0
    success_count = 0
    fail_count = 0

    for i, repo_url in enumerate(repos, 1):
        slug = url_to_slug(repo_url)
        short_slug = slug.replace("/", "--")

        # Fetch runs
        runs = fetch_runs_for_repo(cur, repo_url)

        # If no exact URL match, try slug-based match
        if not runs:
            cur.execute(
                "SELECT * FROM run WHERE repo_url LIKE %s ORDER BY started_at",
                (f"%{slug}%",),
            )
            runs = dict_row(cur)

        if not runs:
            not_found_count += 1
            summary_rows.append({
                "index": i,
                "repo_url": repo_url,
                "slug": slug,
                "found_in_db": False,
                "run_count": 0,
                "overall_result": "NOT_FOUND",
                "any_success": False,
                "best_status": None,
                "total_iterations": 0,
                "total_steps": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "detected_language": None,
                "has_dockerfile": False,
                "smoke_test_passed": None,
                "total_duration_ms": 0,
            })
            if i % 50 == 0:
                print(f"  [{i}/{len(repos)}] processed...")
            continue

        found_count += 1

        # Determine overall success: any run with status='success' → SUCCESS
        any_success = any(r["status"] == "success" for r in runs)
        if any_success:
            success_count += 1
        else:
            fail_count += 1

        # Find the "best" run (success preferred, then most recent)
        best_run = None
        for r in runs:
            if r["status"] == "success":
                best_run = r
                break
        if not best_run:
            best_run = runs[-1]  # most recent

        # Build full data for each run
        runs_full = []
        for run in runs:
            run_data = dict(run)

            # Fetch iterations
            iterations = fetch_iterations_for_run(cur, run["id"])
            iters_full = []
            for it in iterations:
                it_data = dict(it)

                # Fetch steps
                steps = fetch_steps_for_iteration(cur, it["id"])
                steps_full = []
                for step in steps:
                    step_data = dict(step)

                    # Fetch verify_build_details if this is a VerifyBuild step
                    if step["tool_name"] == "VerifyBuild":
                        vbd = fetch_verify_details_for_step(cur, step["id"])
                        step_data["verify_build_details"] = vbd
                    else:
                        step_data["verify_build_details"] = []

                    steps_full.append(step_data)

                it_data["steps"] = steps_full
                iters_full.append(it_data)

            run_data["iterations"] = iters_full

            # Fetch artifacts
            artifacts = fetch_artifacts_for_run(cur, run["id"])
            run_data["artifacts"] = artifacts

            # Fetch batch info (only once per unique batch_id)
            if run.get("batch_id"):
                batch_info = fetch_batch_info(cur, run["batch_id"])
                run_data["batch_info"] = batch_info

            runs_full.append(run_data)

        # Per-repo record
        repo_record = {
            "repo_url": repo_url,
            "slug": slug,
            "overall_result": "SUCCESS" if any_success else "FAIL",
            "run_count": len(runs),
            "runs": runs_full,
        }

        # Save per-repo JSON
        per_repo_path = os.path.join(PER_REPO_DIR, f"{short_slug}.json")
        with open(per_repo_path, "w") as f:
            json.dump(repo_record, f, indent=2, default=json_serial)

        all_results[repo_url] = repo_record

        # Summary row
        total_iters = sum(r.get("iteration_count", 0) or 0 for r in runs)
        total_steps = sum(r.get("total_steps", 0) or 0 for r in runs)
        total_prompt = sum(r.get("total_prompt_tokens", 0) or 0 for r in runs)
        total_completion = sum(r.get("total_completion_tokens", 0) or 0 for r in runs)
        total_duration = sum(r.get("duration_ms", 0) or 0 for r in runs)

        summary_rows.append({
            "index": i,
            "repo_url": repo_url,
            "slug": slug,
            "found_in_db": True,
            "run_count": len(runs),
            "overall_result": "SUCCESS" if any_success else "FAIL",
            "any_success": any_success,
            "best_status": best_run["status"],
            "total_iterations": total_iters,
            "total_steps": total_steps,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "detected_language": best_run.get("detected_language"),
            "has_dockerfile": best_run.get("final_dockerfile") is not None,
            "smoke_test_passed": best_run.get("smoke_test_passed"),
            "total_duration_ms": total_duration,
        })

        if i % 50 == 0:
            print(f"  [{i}/{len(repos)}] processed...")

    # ── Write summary CSV ───────────────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, "summary.csv")
    fieldnames = [
        "index", "repo_url", "slug", "found_in_db", "run_count",
        "overall_result", "any_success", "best_status",
        "total_iterations", "total_steps",
        "total_prompt_tokens", "total_completion_tokens",
        "detected_language", "has_dockerfile", "smoke_test_passed",
        "total_duration_ms",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    # ── Aggregate stats ─────────────────────────────────────────────────────
    found_repos = [r for r in summary_rows if r["found_in_db"]]

    # Language distribution (from found repos)
    lang_dist = {}
    for r in found_repos:
        lang = r["detected_language"] or "unknown"
        lang_dist[lang] = lang_dist.get(lang, 0) + 1

    # Status distribution across ALL runs (not just per-repo)
    cur.execute("""
        SELECT status, COUNT(*) as cnt
        FROM run
        WHERE repo_url = ANY(%s)
        GROUP BY status
        ORDER BY cnt DESC
    """, ([r["repo_url"] for r in summary_rows],))
    status_dist = {row[0]: row[1] for row in cur.fetchall()}

    # Iteration count distribution for successful repos
    iter_counts = [r["total_iterations"] for r in found_repos if r["overall_result"] == "SUCCESS"]

    aggregate = {
        "total_repos_in_list": len(repos),
        "repos_found_in_db": found_count,
        "repos_not_found_in_db": not_found_count,
        "repos_success": success_count,
        "repos_fail": fail_count,
        "success_rate": round(success_count / found_count * 100, 2) if found_count else 0,
        "language_distribution": dict(sorted(lang_dist.items(), key=lambda x: -x[1])),
        "run_status_distribution": status_dist,
        "avg_iterations_success": round(sum(iter_counts) / len(iter_counts), 2) if iter_counts else 0,
        "avg_steps_all": round(
            sum(r["total_steps"] for r in found_repos) / len(found_repos), 2
        ) if found_repos else 0,
        "total_prompt_tokens": sum(r["total_prompt_tokens"] for r in found_repos),
        "total_completion_tokens": sum(r["total_completion_tokens"] for r in found_repos),
        "avg_duration_ms": round(
            sum(r["total_duration_ms"] for r in found_repos) / len(found_repos), 2
        ) if found_repos else 0,
    }

    agg_path = os.path.join(OUTPUT_DIR, "aggregate_stats.json")
    with open(agg_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    # ── Print summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FETCH COMPLETE")
    print("=" * 60)
    print(f"Repos in list:       {len(repos)}")
    print(f"Found in DB:         {found_count}")
    print(f"Not found in DB:     {not_found_count}")
    print(f"SUCCESS:             {success_count}")
    print(f"FAIL:                {fail_count}")
    if found_count:
        print(f"Success rate:        {success_count/found_count*100:.1f}%")
    print(f"\nOutputs:")
    print(f"  Per-repo JSONs:    {PER_REPO_DIR}/")
    print(f"  Summary CSV:       {csv_path}")
    print(f"  Aggregate stats:   {agg_path}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
