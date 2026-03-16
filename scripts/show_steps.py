#!/usr/bin/env python3
"""View agent step details from the BuildAgent database.

Usage:
    python scripts/show_steps.py                  # list recent runs
    python scripts/show_steps.py <run_id_prefix>  # show steps for that run
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from db.writer import DBWriter

db = DBWriter()


def list_runs():
    rows = db._query("""
        SELECT r.id, r.repo_slug, r.status, r.iteration_count,
               r.total_steps, r.total_prompt_tokens, r.total_completion_tokens,
               r.started_at
        FROM run r
        ORDER BY r.started_at DESC
        LIMIT 20
    """)
    print(f"\n{'SLUG':<35} {'STATUS':<10} {'IT':<4} {'STEPS':<6} {'TOKENS':<10} {'ID[:8]'}")
    print("-" * 85)
    for r in rows:
        tokens = (r[5] or 0) + (r[6] or 0)
        print(f"{(r[1] or ''):<35} {r[2]:<10} {r[3]:<4} {r[4]:<6} {tokens:<10} {r[0][:8]}")
    print()
    print("Run: python scripts/show_steps.py <id_prefix>")


def show_run(prefix: str):
    rows = db._query(
        "SELECT id, repo_slug, status, iteration_count, error_message FROM run WHERE id LIKE ? ORDER BY started_at DESC LIMIT 1",
        (prefix + "%",),
    )
    if not rows:
        print(f"No run found with id starting with '{prefix}'")
        return

    run_id, slug, status, iters, error = rows[0]
    print(f"\nRun: {run_id}")
    print(f"Repo: {slug}  Status: {status}  Iterations: {iters}")
    if error:
        print(f"Error: {error}")

    # Get iterations
    iteration_rows = db._query(
        "SELECT id, iteration_number, status, step_count, error_message FROM iteration WHERE run_id = ? ORDER BY iteration_number",
        (run_id,),
    )

    for it_id, it_num, it_status, step_count, it_error in iteration_rows:
        print(f"\n{'─'*70}")
        print(f"  ITERATION {it_num}  [{it_status}]  {step_count} steps")
        if it_error:
            print(f"  Error: {it_error}")
        print(f"{'─'*70}")

        step_rows = db._query(
            "SELECT step_number, tool_name, tool_input, tool_output, was_summarized FROM step WHERE iteration_id = ? ORDER BY step_number",
            (it_id,),
        )

        for step_num, tool_name, tool_input, tool_output, summarized in step_rows:
            flag = " [summarized]" if summarized else ""
            print(f"\n  Step {step_num}: {tool_name}{flag}")
            if tool_input and tool_input != "{}":
                import json
                try:
                    inp = json.loads(tool_input)
                    for k, v in inp.items():
                        v_str = str(v)
                        print(f"    {k}: {v_str[:120]}")
                except Exception:
                    print(f"    input: {str(tool_input)[:120]}")
            if tool_output:
                lines = tool_output.splitlines()
                preview = "\n    ".join(lines[:8])
                suffix = f"\n    ... ({len(lines)} lines total)" if len(lines) > 8 else ""
                print(f"    output:\n    {preview}{suffix}")


def main():
    if len(sys.argv) < 2:
        list_runs()
    else:
        show_run(sys.argv[1])
    db.close()


if __name__ == "__main__":
    main()
