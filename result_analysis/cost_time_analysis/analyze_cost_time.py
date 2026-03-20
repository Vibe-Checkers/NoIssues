#!/usr/bin/env python3
"""
analyze_cost_time.py — Detailed cost and time analysis for all 282 repos.

For each repo and each run/iteration:
  - Token counts (prompt + completion) broken down by phase
  - Dollar cost using OpenRouter pricing
  - Wall-clock time broken down by phase (blueprint, iterations, build, smoke)
  - Per-iteration and per-step granularity

Pricing (OpenRouter, as of March 2026):
  google/gemini-2.0-flash-001:  $0.10 / 1M input,  $0.40 / 1M output
  anthropic/claude-sonnet-4:    $3.00 / 1M input, $15.00 / 1M output

The agent uses gemini-flash for most calls (agent steps, blueprint, verify review,
error summarization, output summarization) and claude-sonnet for lesson extraction.

Usage:
    python3 result_analysis/cost_time_analysis/analyze_cost_time.py
"""

import csv
import json
import math
import os
import statistics
from collections import defaultdict
from datetime import datetime

# ── Paths ───────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(BASE_DIR)
PER_REPO_DIR = os.path.join(PARENT_DIR, "per_repo")
CHAR_CSV = os.path.join(PARENT_DIR, "stratified_repos_2000_majority_vote.csv")
OUT_DIR = BASE_DIR

# ── Pricing (USD per 1M tokens) ────────────────────────────────────────────

# Gemini 2.0 Flash (used for: agent steps, blueprint, verify review,
# error summarization, output summarization)
GEMINI_INPUT_PER_M = 0.10
GEMINI_OUTPUT_PER_M = 0.40

# Claude Sonnet 4 (used for: lesson extraction between iterations)
SONNET_INPUT_PER_M = 3.00
SONNET_OUTPUT_PER_M = 15.00


def cost_gemini(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens * GEMINI_INPUT_PER_M + completion_tokens * GEMINI_OUTPUT_PER_M) / 1_000_000


def cost_sonnet(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens * SONNET_INPUT_PER_M + completion_tokens * SONNET_OUTPUT_PER_M) / 1_000_000


def ms_to_min(ms) -> float:
    return (ms or 0) / 60_000


def ms_to_sec(ms) -> float:
    return (ms or 0) / 1_000


def safe_int(v) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def percentile(data: list, p: float) -> float:
    if not data:
        return 0.0
    k = (len(data) - 1) * p / 100
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return data[int(k)]
    return data[f] * (c - k) + data[c] * (k - f)


def load_csv_dict(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: list[dict], fieldnames: list[str]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load characterization
    char_data = load_csv_dict(CHAR_CSV)
    char_lookup = {row["repo_link"].rstrip("/"): row for row in char_data}

    # Load all per-repo JSONs
    all_repos = {}
    for fname in os.listdir(PER_REPO_DIR):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(PER_REPO_DIR, fname)) as f:
            data = json.load(f)
        all_repos[data["repo_url"]] = data

    print(f"Loaded {len(all_repos)} repos")

    # ════════════════════════════════════════════════════════════════════════
    # 1. PER-REPO COST & TIME BREAKDOWN
    # ════════════════════════════════════════════════════════════════════════

    per_repo_rows = []
    all_iteration_rows = []

    for url, data in sorted(all_repos.items()):
        slug = data["slug"]
        result = data["overall_result"]
        char = char_lookup.get(url.rstrip("/"), {})

        # Use the "best" run: successful if exists, otherwise last
        best_run = None
        for run in data["runs"]:
            if run["status"] == "success":
                best_run = run
                break
        if not best_run:
            best_run = data["runs"][-1] if data["runs"] else None

        if not best_run:
            continue

        # ── Run-level totals ────────────────────────────────────────────
        run_prompt = safe_int(best_run.get("total_prompt_tokens"))
        run_completion = safe_int(best_run.get("total_completion_tokens"))
        run_duration_ms = safe_int(best_run.get("duration_ms"))
        iteration_count = safe_int(best_run.get("iteration_count"))

        # Blueprint phase
        bp_prompt = safe_int(best_run.get("blueprint_tokens_prompt"))
        bp_completion = safe_int(best_run.get("blueprint_tokens_completion"))
        bp_duration_ms = safe_int(best_run.get("blueprint_duration_ms"))
        bp_cost = cost_gemini(bp_prompt, bp_completion)

        # ── Iteration-level breakdown ───────────────────────────────────
        iterations = best_run.get("iterations", [])
        total_agent_prompt = 0
        total_agent_completion = 0
        total_lesson_prompt = 0
        total_lesson_completion = 0
        total_review_prompt = 0
        total_review_completion = 0
        total_summary_prompt = 0
        total_summary_completion = 0
        total_build_error_summary_prompt = 0
        total_build_error_summary_completion = 0
        total_build_duration_ms = 0
        total_smoke_duration_ms = 0
        total_review_duration_ms = 0
        total_steps = 0
        total_verify_calls = 0

        for it in iterations:
            it_num = it.get("iteration_number", 0)
            it_prompt = safe_int(it.get("prompt_tokens"))
            it_completion = safe_int(it.get("completion_tokens"))
            it_duration_ms = safe_int(it.get("duration_ms"))
            it_step_count = safe_int(it.get("step_count"))

            # Lesson extraction (Claude Sonnet)
            le_prompt = safe_int(it.get("lesson_extraction_tokens_prompt"))
            le_completion = safe_int(it.get("lesson_extraction_tokens_completion"))
            total_lesson_prompt += le_prompt
            total_lesson_completion += le_completion

            # Per-step breakdown
            iter_agent_prompt = 0
            iter_agent_completion = 0
            iter_summary_prompt = 0
            iter_summary_completion = 0
            iter_review_prompt = 0
            iter_review_completion = 0
            iter_build_dur = 0
            iter_smoke_dur = 0
            iter_review_dur = 0
            iter_be_summary_prompt = 0
            iter_be_summary_completion = 0
            iter_verify_calls = 0

            for s in it.get("steps", []):
                # Agent step tokens (Gemini)
                s_prompt = safe_int(s.get("prompt_tokens"))
                s_completion = safe_int(s.get("completion_tokens"))
                iter_agent_prompt += s_prompt
                iter_agent_completion += s_completion

                # Output summarization tokens (Gemini)
                iter_summary_prompt += safe_int(s.get("summary_prompt_tokens"))
                iter_summary_completion += safe_int(s.get("summary_completion_tokens"))

                # VerifyBuild details
                if s.get("tool_name") == "VerifyBuild":
                    iter_verify_calls += 1
                    for v in s.get("verify_build_details", []):
                        iter_review_prompt += safe_int(v.get("review_prompt_tokens"))
                        iter_review_completion += safe_int(v.get("review_completion_tokens"))
                        iter_review_dur += safe_int(v.get("review_duration_ms"))
                        iter_build_dur += safe_int(v.get("build_duration_ms"))
                        iter_smoke_dur += safe_int(v.get("smoke_duration_ms"))
                        iter_be_summary_prompt += safe_int(v.get("build_error_summary_tokens_prompt"))
                        iter_be_summary_completion += safe_int(v.get("build_error_summary_tokens_completion"))

            total_agent_prompt += iter_agent_prompt
            total_agent_completion += iter_agent_completion
            total_summary_prompt += iter_summary_prompt
            total_summary_completion += iter_summary_completion
            total_review_prompt += iter_review_prompt
            total_review_completion += iter_review_completion
            total_build_duration_ms += iter_build_dur
            total_smoke_duration_ms += iter_smoke_dur
            total_review_duration_ms += iter_review_dur
            total_build_error_summary_prompt += iter_be_summary_prompt
            total_build_error_summary_completion += iter_be_summary_completion
            total_steps += it_step_count
            total_verify_calls += iter_verify_calls

            # Cost for this iteration
            iter_gemini_cost = cost_gemini(
                iter_agent_prompt + iter_summary_prompt + iter_review_prompt + iter_be_summary_prompt,
                iter_agent_completion + iter_summary_completion + iter_review_completion + iter_be_summary_completion,
            )
            iter_sonnet_cost = cost_sonnet(le_prompt, le_completion)
            iter_total_cost = iter_gemini_cost + iter_sonnet_cost

            all_iteration_rows.append({
                "slug": slug,
                "repo_url": url,
                "result": result,
                "iteration": it_num,
                "status": it.get("status", ""),
                "verify_result": it.get("verify_result") or "",
                "step_count": it_step_count,
                "verify_calls": iter_verify_calls,
                "duration_ms": it_duration_ms,
                "duration_min": round(ms_to_min(it_duration_ms), 2),
                "agent_prompt_tokens": iter_agent_prompt,
                "agent_completion_tokens": iter_agent_completion,
                "summary_prompt_tokens": iter_summary_prompt,
                "summary_completion_tokens": iter_summary_completion,
                "review_prompt_tokens": iter_review_prompt,
                "review_completion_tokens": iter_review_completion,
                "lesson_prompt_tokens": le_prompt,
                "lesson_completion_tokens": le_completion,
                "build_error_summary_prompt": iter_be_summary_prompt,
                "build_error_summary_completion": iter_be_summary_completion,
                "build_duration_ms": iter_build_dur,
                "smoke_duration_ms": iter_smoke_dur,
                "review_duration_ms": iter_review_dur,
                "gemini_cost_usd": round(iter_gemini_cost, 6),
                "sonnet_cost_usd": round(iter_sonnet_cost, 6),
                "total_cost_usd": round(iter_total_cost, 6),
            })

        # ── Total costs ────────────────────────────────────────────────
        # All Gemini costs
        gemini_prompt_total = bp_prompt + total_agent_prompt + total_summary_prompt + total_review_prompt + total_build_error_summary_prompt
        gemini_completion_total = bp_completion + total_agent_completion + total_summary_completion + total_review_completion + total_build_error_summary_completion
        gemini_cost = cost_gemini(gemini_prompt_total, gemini_completion_total)

        # Sonnet costs (lesson extraction only)
        sonnet_cost = cost_sonnet(total_lesson_prompt, total_lesson_completion)

        total_cost = gemini_cost + sonnet_cost

        per_repo_rows.append({
            "slug": slug,
            "repo_url": url,
            "result": result,
            "domain": char.get("domain", ""),
            "build_type": char.get("build_type", ""),
            "iteration_count": iteration_count,
            "total_steps": total_steps,
            "verify_calls": total_verify_calls,
            # Time
            "total_duration_ms": run_duration_ms,
            "total_duration_min": round(ms_to_min(run_duration_ms), 2),
            "blueprint_duration_ms": bp_duration_ms,
            "blueprint_duration_min": round(ms_to_min(bp_duration_ms), 2),
            "total_build_duration_ms": total_build_duration_ms,
            "total_build_duration_min": round(ms_to_min(total_build_duration_ms), 2),
            "total_smoke_duration_ms": total_smoke_duration_ms,
            "total_review_duration_ms": total_review_duration_ms,
            # Tokens — Gemini (agent/blueprint/review/summary)
            "gemini_prompt_tokens": gemini_prompt_total,
            "gemini_completion_tokens": gemini_completion_total,
            "gemini_cost_usd": round(gemini_cost, 6),
            # Tokens — Sonnet (lesson extraction)
            "sonnet_prompt_tokens": total_lesson_prompt,
            "sonnet_completion_tokens": total_lesson_completion,
            "sonnet_cost_usd": round(sonnet_cost, 6),
            # Totals
            "total_prompt_tokens": gemini_prompt_total + total_lesson_prompt,
            "total_completion_tokens": gemini_completion_total + total_lesson_completion,
            "total_cost_usd": round(total_cost, 6),
            # Breakdown
            "blueprint_prompt_tokens": bp_prompt,
            "blueprint_completion_tokens": bp_completion,
            "blueprint_cost_usd": round(bp_cost, 6),
            "agent_prompt_tokens": total_agent_prompt,
            "agent_completion_tokens": total_agent_completion,
            "review_prompt_tokens": total_review_prompt,
            "review_completion_tokens": total_review_completion,
            "summary_prompt_tokens": total_summary_prompt,
            "summary_completion_tokens": total_summary_completion,
            "lesson_prompt_tokens": total_lesson_prompt,
            "lesson_completion_tokens": total_lesson_completion,
        })

    # ════════════════════════════════════════════════════════════════════════
    # SAVE PER-REPO CSV
    # ════════════════════════════════════════════════════════════════════════
    per_repo_fields = list(per_repo_rows[0].keys())
    write_csv(os.path.join(OUT_DIR, "per_repo_cost_time.csv"), per_repo_rows, per_repo_fields)

    # ════════════════════════════════════════════════════════════════════════
    # SAVE PER-ITERATION CSV
    # ════════════════════════════════════════════════════════════════════════
    iter_fields = list(all_iteration_rows[0].keys()) if all_iteration_rows else []
    write_csv(os.path.join(OUT_DIR, "per_iteration_cost_time.csv"), all_iteration_rows, iter_fields)

    # ════════════════════════════════════════════════════════════════════════
    # AGGREGATE STATISTICS
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("OVERALL STATISTICS (282 repos, best run per repo)")
    print("=" * 70)

    costs = [r["total_cost_usd"] for r in per_repo_rows]
    durations = [r["total_duration_min"] for r in per_repo_rows]
    iters = [r["iteration_count"] for r in per_repo_rows]
    steps = [r["total_steps"] for r in per_repo_rows]
    prompt_tokens = [r["total_prompt_tokens"] for r in per_repo_rows]
    completion_tokens = [r["total_completion_tokens"] for r in per_repo_rows]

    costs_s = sorted(costs)
    durations_s = sorted(durations)

    def stats_block(label, values):
        vs = sorted(values)
        print(f"\n  {label}:")
        print(f"    Total:   {sum(vs):>12,.2f}")
        print(f"    Mean:    {statistics.mean(vs):>12,.2f}")
        print(f"    Median:  {statistics.median(vs):>12,.2f}")
        print(f"    Std Dev: {statistics.stdev(vs):>12,.2f}" if len(vs) > 1 else "")
        print(f"    Min:     {vs[0]:>12,.2f}")
        print(f"    P25:     {percentile(vs, 25):>12,.2f}")
        print(f"    P75:     {percentile(vs, 75):>12,.2f}")
        print(f"    P90:     {percentile(vs, 90):>12,.2f}")
        print(f"    P95:     {percentile(vs, 95):>12,.2f}")
        print(f"    Max:     {vs[-1]:>12,.2f}")

    stats_block("Cost (USD)", costs)
    stats_block("Duration (minutes)", durations)
    stats_block("Iterations", [float(i) for i in iters])
    stats_block("Steps", [float(s) for s in steps])
    stats_block("Prompt tokens", [float(t) for t in prompt_tokens])
    stats_block("Completion tokens", [float(t) for t in completion_tokens])

    # ── Success vs Fail comparison ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUCCESS vs FAIL COMPARISON")
    print("=" * 70)

    for group_name, group in [("SUCCESS", [r for r in per_repo_rows if r["result"] == "SUCCESS"]),
                               ("FAIL", [r for r in per_repo_rows if r["result"] == "FAIL"])]:
        if not group:
            continue
        g_costs = [r["total_cost_usd"] for r in group]
        g_dur = [r["total_duration_min"] for r in group]
        g_iters = [r["iteration_count"] for r in group]
        g_steps = [r["total_steps"] for r in group]
        g_gemini = [r["gemini_cost_usd"] for r in group]
        g_sonnet = [r["sonnet_cost_usd"] for r in group]
        g_bp_dur = [r["blueprint_duration_min"] for r in group]
        g_build_dur = [r["total_build_duration_min"] for r in group]

        print(f"\n  {group_name} (n={len(group)}):")
        print(f"    {'Metric':<30} {'Total':>12} {'Mean':>10} {'Median':>10}")
        print(f"    {'─'*30} {'─'*12} {'─'*10} {'─'*10}")
        for label, vals in [
            ("Cost (USD)", g_costs),
            ("  Gemini cost", g_gemini),
            ("  Sonnet cost", g_sonnet),
            ("Duration (min)", g_dur),
            ("  Blueprint (min)", g_bp_dur),
            ("  Docker build (min)", g_build_dur),
            ("Iterations", [float(i) for i in g_iters]),
            ("Steps", [float(s) for s in g_steps]),
        ]:
            print(f"    {label:<30} {sum(vals):>12,.2f} {statistics.mean(vals):>10,.2f} {statistics.median(vals):>10,.2f}")

    # ── Cost breakdown by phase ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COST BREAKDOWN BY PHASE (all 282 repos)")
    print("=" * 70)

    total_bp_cost = sum(r["blueprint_cost_usd"] for r in per_repo_rows)
    total_gemini_cost = sum(r["gemini_cost_usd"] for r in per_repo_rows)
    total_sonnet_cost = sum(r["sonnet_cost_usd"] for r in per_repo_rows)
    total_total_cost = sum(r["total_cost_usd"] for r in per_repo_rows)

    # Compute agent-step-only cost (gemini minus blueprint minus review minus summary)
    total_agent_only = sum(cost_gemini(r["agent_prompt_tokens"], r["agent_completion_tokens"]) for r in per_repo_rows)
    total_review_only = sum(cost_gemini(r["review_prompt_tokens"], r["review_completion_tokens"]) for r in per_repo_rows)
    total_summary_only = sum(cost_gemini(r["summary_prompt_tokens"], r["summary_completion_tokens"]) for r in per_repo_rows)

    print(f"\n  {'Phase':<35} {'Cost (USD)':>12} {'%':>7}")
    print(f"  {'─'*35} {'─'*12} {'─'*7}")
    for label, val in [
        ("Blueprint (Gemini)", total_bp_cost),
        ("Agent steps (Gemini)", total_agent_only),
        ("Verify review (Gemini)", total_review_only),
        ("Output summarization (Gemini)", total_summary_only),
        ("Lesson extraction (Sonnet)", total_sonnet_cost),
    ]:
        pct = val / total_total_cost * 100 if total_total_cost else 0
        print(f"  {label:<35} ${val:>11,.4f} {pct:>6.1f}%")
    print(f"  {'─'*35} {'─'*12} {'─'*7}")
    print(f"  {'TOTAL':<35} ${total_total_cost:>11,.4f} {'100.0':>6}%")
    print(f"\n  Total Gemini: ${total_gemini_cost:,.4f}  |  Total Sonnet: ${total_sonnet_cost:,.4f}")

    # ── Time breakdown ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TIME BREAKDOWN (all 282 repos)")
    print("=" * 70)

    total_wall = sum(r["total_duration_ms"] for r in per_repo_rows) / 3_600_000
    total_bp_time = sum(r["blueprint_duration_ms"] for r in per_repo_rows) / 3_600_000
    total_build_time = sum(r["total_build_duration_ms"] for r in per_repo_rows) / 3_600_000
    total_smoke_time = sum(r["total_smoke_duration_ms"] for r in per_repo_rows) / 3_600_000
    total_review_time = sum(r["total_review_duration_ms"] for r in per_repo_rows) / 3_600_000

    print(f"\n  {'Phase':<35} {'Hours':>10} {'%':>7}")
    print(f"  {'─'*35} {'─'*10} {'─'*7}")
    for label, val in [
        ("Blueprint generation", total_bp_time),
        ("Docker builds", total_build_time),
        ("Smoke tests", total_smoke_time),
        ("LLM review", total_review_time),
        ("Other (agent reasoning, I/O)", total_wall - total_bp_time - total_build_time - total_smoke_time - total_review_time),
    ]:
        pct = val / total_wall * 100 if total_wall else 0
        print(f"  {label:<35} {val:>10.2f} {pct:>6.1f}%")
    print(f"  {'─'*35} {'─'*10} {'─'*7}")
    print(f"  {'TOTAL WALL-CLOCK':<35} {total_wall:>10.2f} {'100.0':>6}%")

    # ── Per iteration number stats ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("PER ITERATION-NUMBER STATISTICS")
    print("=" * 70)

    iter_by_num = defaultdict(list)
    for ir in all_iteration_rows:
        iter_by_num[ir["iteration"]].append(ir)

    print(f"\n  {'Iter':<5} {'Count':>6} {'AvgCost':>10} {'AvgMin':>8} {'AvgSteps':>9} {'AvgVerify':>10} {'AvgTokens(K)':>13}")
    print(f"  {'─'*5} {'─'*6} {'─'*10} {'─'*8} {'─'*9} {'─'*10} {'─'*13}")

    iter_stats_rows = []
    for it_num in sorted(iter_by_num.keys()):
        group = iter_by_num[it_num]
        n = len(group)
        avg_cost = sum(r["total_cost_usd"] for r in group) / n
        avg_dur = sum(r["duration_min"] for r in group) / n
        avg_steps = sum(r["step_count"] for r in group) / n
        avg_verify = sum(r["verify_calls"] for r in group) / n
        avg_tokens = sum(r["agent_prompt_tokens"] + r["agent_completion_tokens"] for r in group) / n / 1000

        print(f"  {it_num:<5} {n:>6} ${avg_cost:>9.4f} {avg_dur:>7.1f} {avg_steps:>9.1f} {avg_verify:>10.1f} {avg_tokens:>13.1f}")

        iter_stats_rows.append({
            "iteration": it_num,
            "count": n,
            "avg_cost_usd": round(avg_cost, 6),
            "avg_duration_min": round(avg_dur, 2),
            "avg_steps": round(avg_steps, 1),
            "avg_verify_calls": round(avg_verify, 1),
            "avg_tokens_k": round(avg_tokens, 1),
            "total_cost_usd": round(sum(r["total_cost_usd"] for r in group), 4),
        })

    write_csv(os.path.join(OUT_DIR, "per_iteration_number_stats.csv"), iter_stats_rows,
              list(iter_stats_rows[0].keys()) if iter_stats_rows else [])

    # ── Cost by domain and build_type ───────────────────────────────────
    print("\n" + "=" * 70)
    print("COST BY DOMAIN")
    print("=" * 70)

    for group_col in ["domain", "build_type"]:
        grouped = defaultdict(list)
        for r in per_repo_rows:
            grouped[r[group_col] or "unknown"].append(r)

        print(f"\n  {group_col.upper()}")
        print(f"  {'Value':<30} {'N':>4} {'TotalCost':>10} {'AvgCost':>10} {'AvgMin':>8} {'AvgIter':>8}")
        print(f"  {'─'*30} {'─'*4} {'─'*10} {'─'*10} {'─'*8} {'─'*8}")

        group_rows = []
        for val in sorted(grouped.keys(), key=lambda v: -len(grouped[v])):
            g = grouped[val]
            n = len(g)
            tc = sum(r["total_cost_usd"] for r in g)
            ac = tc / n
            ad = sum(r["total_duration_min"] for r in g) / n
            ai = sum(r["iteration_count"] for r in g) / n
            print(f"  {val:<30} {n:>4} ${tc:>9.4f} ${ac:>9.4f} {ad:>7.1f} {ai:>8.2f}")
            group_rows.append({
                group_col: val, "n": n,
                "total_cost_usd": round(tc, 4), "avg_cost_usd": round(ac, 6),
                "avg_duration_min": round(ad, 2), "avg_iterations": round(ai, 2),
            })

        write_csv(os.path.join(OUT_DIR, f"cost_by_{group_col}.csv"), group_rows,
                  list(group_rows[0].keys()) if group_rows else [])

    # ── Top 10 most expensive repos ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("TOP 10 MOST EXPENSIVE REPOS")
    print("=" * 70)

    top_cost = sorted(per_repo_rows, key=lambda r: -r["total_cost_usd"])[:10]
    print(f"\n  {'Slug':<40} {'Cost':>10} {'Min':>8} {'Iter':>5} {'Steps':>6} {'Result':>8}")
    print(f"  {'─'*40} {'─'*10} {'─'*8} {'─'*5} {'─'*6} {'─'*8}")
    for r in top_cost:
        print(f"  {r['slug']:<40} ${r['total_cost_usd']:>9.4f} {r['total_duration_min']:>7.1f} {r['iteration_count']:>5} {r['total_steps']:>6} {r['result']:>8}")

    # ── Top 10 longest repos ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TOP 10 LONGEST REPOS")
    print("=" * 70)

    top_dur = sorted(per_repo_rows, key=lambda r: -r["total_duration_min"])[:10]
    print(f"\n  {'Slug':<40} {'Min':>8} {'Cost':>10} {'Iter':>5} {'Steps':>6} {'Result':>8}")
    print(f"  {'─'*40} {'─'*8} {'─'*10} {'─'*5} {'─'*6} {'─'*8}")
    for r in top_dur:
        print(f"  {r['slug']:<40} {r['total_duration_min']:>7.1f} ${r['total_cost_usd']:>9.4f} {r['iteration_count']:>5} {r['total_steps']:>6} {r['result']:>8}")

    # ════════════════════════════════════════════════════════════════════════
    # SAVE AGGREGATE JSON
    # ════════════════════════════════════════════════════════════════════════

    aggregate = {
        "total_repos": len(per_repo_rows),
        "pricing": {
            "gemini_flash_input_per_M": GEMINI_INPUT_PER_M,
            "gemini_flash_output_per_M": GEMINI_OUTPUT_PER_M,
            "sonnet_input_per_M": SONNET_INPUT_PER_M,
            "sonnet_output_per_M": SONNET_OUTPUT_PER_M,
        },
        "totals": {
            "total_cost_usd": round(sum(costs), 4),
            "total_gemini_cost_usd": round(total_gemini_cost, 4),
            "total_sonnet_cost_usd": round(total_sonnet_cost, 4),
            "total_wall_clock_hours": round(total_wall, 2),
            "total_prompt_tokens": sum(prompt_tokens),
            "total_completion_tokens": sum(completion_tokens),
            "total_tokens": sum(prompt_tokens) + sum(completion_tokens),
        },
        "cost_breakdown_usd": {
            "blueprint": round(total_bp_cost, 4),
            "agent_steps": round(total_agent_only, 4),
            "verify_review": round(total_review_only, 4),
            "output_summarization": round(total_summary_only, 4),
            "lesson_extraction_sonnet": round(total_sonnet_cost, 4),
        },
        "time_breakdown_hours": {
            "blueprint": round(total_bp_time, 2),
            "docker_builds": round(total_build_time, 2),
            "smoke_tests": round(total_smoke_time, 2),
            "llm_review": round(total_review_time, 2),
            "other": round(total_wall - total_bp_time - total_build_time - total_smoke_time - total_review_time, 2),
        },
        "per_repo_stats": {
            "cost": {
                "mean": round(statistics.mean(costs), 6),
                "median": round(statistics.median(costs), 6),
                "std": round(statistics.stdev(costs), 6) if len(costs) > 1 else 0,
                "min": round(min(costs), 6),
                "p25": round(percentile(costs_s, 25), 6),
                "p75": round(percentile(costs_s, 75), 6),
                "p90": round(percentile(costs_s, 90), 6),
                "p95": round(percentile(costs_s, 95), 6),
                "max": round(max(costs), 6),
            },
            "duration_min": {
                "mean": round(statistics.mean(durations), 2),
                "median": round(statistics.median(durations), 2),
                "std": round(statistics.stdev(durations), 2) if len(durations) > 1 else 0,
                "min": round(min(durations), 2),
                "p25": round(percentile(durations_s, 25), 2),
                "p75": round(percentile(durations_s, 75), 2),
                "p90": round(percentile(durations_s, 90), 2),
                "p95": round(percentile(durations_s, 95), 2),
                "max": round(max(durations), 2),
            },
            "iterations": {
                "mean": round(statistics.mean(iters), 2),
                "median": round(statistics.median(iters), 2),
            },
        },
        "success_vs_fail": {},
    }

    for grp_name, grp in [("success", [r for r in per_repo_rows if r["result"] == "SUCCESS"]),
                           ("fail", [r for r in per_repo_rows if r["result"] == "FAIL"])]:
        gc = [r["total_cost_usd"] for r in grp]
        gd = [r["total_duration_min"] for r in grp]
        aggregate["success_vs_fail"][grp_name] = {
            "n": len(grp),
            "total_cost_usd": round(sum(gc), 4),
            "avg_cost_usd": round(statistics.mean(gc), 6) if gc else 0,
            "median_cost_usd": round(statistics.median(gc), 6) if gc else 0,
            "total_duration_hours": round(sum(gd) / 60, 2),
            "avg_duration_min": round(statistics.mean(gd), 2) if gd else 0,
            "median_duration_min": round(statistics.median(gd), 2) if gd else 0,
        }

    with open(os.path.join(OUT_DIR, "cost_time_summary.json"), "w") as f:
        json.dump(aggregate, f, indent=2)

    # ════════════════════════════════════════════════════════════════════════
    # LATEX TABLES
    # ════════════════════════════════════════════════════════════════════════

    latex_path = os.path.join(OUT_DIR, "cost_time_latex_tables.tex")
    with open(latex_path, "w") as f:
        f.write("% Auto-generated LaTeX tables — cost & time analysis\n\n")

        # Overall stats
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Overall Cost and Time Statistics (N=282)}\n")
        f.write("\\label{tab:cost-time-overall}\n")
        f.write("\\begin{tabular}{lrr}\n\\toprule\n")
        f.write("Metric & Total & Per Repo (mean) \\\\\n\\midrule\n")
        f.write(f"Cost (USD) & \\${sum(costs):.2f} & \\${statistics.mean(costs):.4f} \\\\\n")
        f.write(f"\\quad Gemini Flash & \\${total_gemini_cost:.2f} & \\${total_gemini_cost/len(per_repo_rows):.4f} \\\\\n")
        f.write(f"\\quad Claude Sonnet & \\${total_sonnet_cost:.2f} & \\${total_sonnet_cost/len(per_repo_rows):.4f} \\\\\n")
        f.write(f"Wall-clock time (hours) & {total_wall:.1f} & {statistics.mean(durations):.1f} min \\\\\n")
        f.write(f"Prompt tokens & {sum(prompt_tokens):,} & {statistics.mean(prompt_tokens):,.0f} \\\\\n")
        f.write(f"Completion tokens & {sum(completion_tokens):,} & {statistics.mean(completion_tokens):,.0f} \\\\\n")
        f.write(f"Iterations & --- & {statistics.mean(iters):.2f} \\\\\n")
        f.write(f"Steps & --- & {statistics.mean(steps):.1f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

        # Success vs fail
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Cost and Time: Successful vs.\\ Failed Repos}\n")
        f.write("\\label{tab:cost-success-fail}\n")
        f.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
        f.write(" & \\multicolumn{2}{c}{Successful} & \\multicolumn{2}{c}{Failed} \\\\\n")
        f.write("Metric & Mean & Median & Mean & Median \\\\\n\\midrule\n")
        for label, key in [("Cost (USD)", "total_cost_usd"), ("Duration (min)", "total_duration_min"),
                           ("Iterations", "iteration_count"), ("Steps", "total_steps")]:
            sv = [r[key] for r in per_repo_rows if r["result"] == "SUCCESS"]
            fv = [r[key] for r in per_repo_rows if r["result"] == "FAIL"]
            sm = statistics.mean(sv) if sv else 0
            smed = statistics.median(sv) if sv else 0
            fm = statistics.mean(fv) if fv else 0
            fmed = statistics.median(fv) if fv else 0
            if "Cost" in label:
                f.write(f"{label} & \\${sm:.4f} & \\${smed:.4f} & \\${fm:.4f} & \\${fmed:.4f} \\\\\n")
            else:
                f.write(f"{label} & {sm:.1f} & {smed:.1f} & {fm:.1f} & {fmed:.1f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

        # Cost breakdown
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Cost Breakdown by Pipeline Phase}\n")
        f.write("\\label{tab:cost-breakdown}\n")
        f.write("\\begin{tabular}{lrr}\n\\toprule\n")
        f.write("Phase & Cost (USD) & \\% \\\\\n\\midrule\n")
        for label, val in [
            ("Blueprint (Gemini Flash)", total_bp_cost),
            ("Agent steps (Gemini Flash)", total_agent_only),
            ("Verify review (Gemini Flash)", total_review_only),
            ("Output summarization (Gemini Flash)", total_summary_only),
            ("Lesson extraction (Claude Sonnet 4)", total_sonnet_cost),
        ]:
            pct = val / total_total_cost * 100 if total_total_cost else 0
            f.write(f"{label} & \\${val:.4f} & {pct:.1f} \\\\\n")
        f.write(f"\\midrule\nTotal & \\${total_total_cost:.4f} & 100.0 \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

    print(f"\n  LaTeX tables written to {latex_path}")

    # ── File listing ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ALL OUTPUTS SAVED")
    print("=" * 70)
    print(f"\nOutput directory: {OUT_DIR}/")
    for fname in sorted(os.listdir(OUT_DIR)):
        fpath = os.path.join(OUT_DIR, fname)
        size = os.path.getsize(fpath)
        print(f"  {fname:<45} {size:>8} bytes")


if __name__ == "__main__":
    main()
