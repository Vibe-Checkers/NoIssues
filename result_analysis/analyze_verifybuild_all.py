#!/usr/bin/env python3
"""
analyze_verifybuild_all.py -- Analysis of ALL VerifyBuild calls across 282 repos.

Unlike analyze_failures.py which only looks at the 86 failed repos, this script
examines every VerifyBuild invocation including intermediate failures in repos
that eventually succeeded. This gives a fuller picture of what the agent struggles
with and recovers from.

Reads:  result_analysis/per_repo/*.json
        result_analysis/analysis_output/merged_dataset.csv
Writes: result_analysis/failure_analysis/verifybuild_all/
            verifybuild_summary.json
            per_repo_verifybuild.csv
            recovery_patterns.csv
            intermediate_errors_in_success.csv
            review_rejection_analysis.csv
            smoke_vs_build_progression.csv
            verifybuild_all_latex.tex

Usage:
    python3 result_analysis/analyze_verifybuild_all.py
"""

import csv
import json
import os
import re
from collections import Counter, defaultdict

BASE_DIR = os.path.dirname(__file__)
PER_REPO_DIR = os.path.join(BASE_DIR, "per_repo")
MERGED_CSV = os.path.join(BASE_DIR, "analysis_output", "merged_dataset.csv")
OUT_DIR = os.path.join(BASE_DIR, "failure_analysis", "verifybuild_all")

BUILD_ERROR_PATTERNS = [
    ("dependency_not_found", [
        r"no matching (version|package|distribution)",
        r"could not (find|resolve|locate)",
        r"package .+ not found",
        r"unable to locate package",
        r"ModuleNotFoundError",
        r"cannot find module",
        r"module not found",
        r"Could not find artifact",
        r"404 Not Found",
    ]),
    ("version_mismatch", [
        r"version .+ (not compatible|incompatible)",
        r"requires .+ version",
        r"minimum required .+ version",
        r"unsupported .+ version",
        r"does not satisfy",
        r"peer dep",
        r"engine .+ is incompatible",
    ]),
    ("compilation_error", [
        r"error(\[E\d+\])?:.*compile",
        r"fatal error:",
        r"cannot find -l",
        r"undefined reference",
        r"linker command failed",
        r"cc1.*error",
        r"Build FAILED",
        r"error:.*cannot find",
    ]),
    ("missing_file", [
        r"COPY failed",
        r"file not found",
        r"No such file or directory",
        r"not found in build context",
        r"stat .+: no such file",
    ]),
    ("syntax_error", [
        r"Dockerfile parse error",
        r"unknown instruction",
        r"invalid reference format",
        r"unexpected token",
        r"failed to parse",
    ]),
    ("network_error", [
        r"network (is )?unreachable",
        r"Could not connect",
        r"connection (refused|timed out|reset)",
        r"TLS handshake timeout",
        r"temporary failure (in|resolving)",
    ]),
    ("timeout", [
        r"timed? ?out",
        r"exceeded.*timeout",
        r"deadline exceeded",
    ]),
]


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def safe_div(a, b):
    return a / b if b else 0.0


def classify_build_error(error_text):
    if not error_text:
        return []
    categories = []
    for cat, patterns in BUILD_ERROR_PATTERNS:
        for p in patterns:
            if re.search(p, error_text, re.IGNORECASE):
                categories.append(cat)
                break
    return categories


def parse_json_field(val):
    if not val:
        return []
    if isinstance(val, list):
        return val
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return []


def extract_verifybuild_calls(repo_json):
    """Extract all VerifyBuild calls from a repo's JSON, with context."""
    calls = []
    slug = repo_json.get("slug", "unknown")
    overall = repo_json.get("overall_result", "UNKNOWN")

    for run in repo_json.get("runs", []):
        for iteration in run.get("iterations", []):
            iter_num = iteration.get("iteration_number", 0)
            iter_status = iteration.get("status", "unknown")

            for step in iteration.get("steps", []):
                if step.get("tool_name") != "VerifyBuild":
                    continue
                for vbd in step.get("verify_build_details", []):
                    call = {
                        "slug": slug,
                        "overall_result": overall,
                        "iteration": iter_num,
                        "iter_status": iter_status,
                        "step_number": step.get("step_number", 0),
                        "review_approved": vbd.get("review_approved"),
                        "build_attempted": vbd.get("build_attempted"),
                        "build_success": vbd.get("build_success"),
                        "build_duration_ms": vbd.get("build_duration_ms"),
                        "build_error": vbd.get("build_error") or vbd.get("build_error_raw") or "",
                        "smoke_attempted": vbd.get("smoke_attempted"),
                        "smoke_passed": vbd.get("smoke_passed"),
                        "smoke_duration_ms": vbd.get("smoke_duration_ms"),
                        "smoke_results": vbd.get("smoke_results", ""),
                        "smoke_test_commands": vbd.get("smoke_test_commands", ""),
                        "review_concerns": vbd.get("review_concerns", ""),
                    }
                    calls.append(call)
    return calls


def classify_call_outcome(call):
    if not call["review_approved"]:
        return "rejected"
    if not call["build_attempted"]:
        return "no_build"
    if not call["build_success"]:
        return "build_failed"
    if not call["smoke_attempted"]:
        return "no_smoke"
    if call["smoke_passed"]:
        return "accepted"
    return "smoke_failed"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load characterization data
    char_lookup = {}
    if os.path.exists(MERGED_CSV):
        for row in load_csv(MERGED_CSV):
            char_lookup[row["slug"]] = row

    # Process all per-repo JSONs
    all_calls = []
    per_repo_stats = []
    repo_files = sorted(f for f in os.listdir(PER_REPO_DIR) if f.endswith(".json"))

    print(f"Processing {len(repo_files)} repo files...")

    for fname in repo_files:
        with open(os.path.join(PER_REPO_DIR, fname)) as f:
            repo_json = json.load(f)

        calls = extract_verifybuild_calls(repo_json)
        all_calls.extend(calls)

        slug = repo_json.get("slug", fname.replace(".json", "").replace("--", "/"))
        overall = repo_json.get("overall_result", "UNKNOWN")

        outcomes = [classify_call_outcome(c) for c in calls]
        per_repo_stats.append({
            "slug": slug,
            "overall_result": overall,
            "total_verify_calls": len(calls),
            "rejected": outcomes.count("rejected"),
            "build_failed": outcomes.count("build_failed"),
            "smoke_failed": outcomes.count("smoke_failed"),
            "accepted": outcomes.count("accepted"),
            "calls_before_first_accept": next(
                (i for i, o in enumerate(outcomes) if o == "accepted"), len(outcomes)
            ),
        })

    print(f"Total VerifyBuild calls: {len(all_calls)}")

    # Classify each call
    for call in all_calls:
        call["outcome"] = classify_call_outcome(call)
        call["error_categories"] = classify_build_error(call["build_error"]) if call["outcome"] == "build_failed" else []

    # 1. Overall VerifyBuild outcome distribution
    outcome_counts = Counter(c["outcome"] for c in all_calls)
    success_calls = [c for c in all_calls if c["overall_result"] == "SUCCESS"]
    fail_calls = [c for c in all_calls if c["overall_result"] == "FAIL"]

    print("\n" + "=" * 70)
    print("VERIFYBUILD OUTCOME DISTRIBUTION")
    print("=" * 70)
    print(f"\n  {'Outcome':<20} {'All':>6} {'In Success':>12} {'In Fail':>10}")
    print(f"  {'-'*20} {'-'*6} {'-'*12} {'-'*10}")
    for outcome in ["accepted", "build_failed", "smoke_failed", "rejected", "no_build", "no_smoke"]:
        total = outcome_counts.get(outcome, 0)
        in_succ = sum(1 for c in success_calls if c["outcome"] == outcome)
        in_fail = sum(1 for c in fail_calls if c["outcome"] == outcome)
        print(f"  {outcome:<20} {total:>6} {in_succ:>12} {in_fail:>10}")

    total_all = len(all_calls)
    total_succ = len(success_calls)
    total_fail = len(fail_calls)
    print(f"  {'TOTAL':<20} {total_all:>6} {total_succ:>12} {total_fail:>10}")

    # 2. Intermediate failures in successful repos
    print("\n" + "=" * 70)
    print("INTERMEDIATE FAILURES IN SUCCESSFUL REPOS")
    print("=" * 70)

    succ_repos = [r for r in per_repo_stats if r["overall_result"] == "SUCCESS"]
    fail_repos = [r for r in per_repo_stats if r["overall_result"] == "FAIL"]

    avg_calls_succ = safe_div(sum(r["total_verify_calls"] for r in succ_repos), len(succ_repos))
    avg_calls_fail = safe_div(sum(r["total_verify_calls"] for r in fail_repos), len(fail_repos))
    avg_before_accept = safe_div(sum(r["calls_before_first_accept"] for r in succ_repos), len(succ_repos))

    print(f"\n  Avg VerifyBuild calls per successful repo: {avg_calls_succ:.1f}")
    print(f"  Avg VerifyBuild calls per failed repo:     {avg_calls_fail:.1f}")
    print(f"  Avg failed calls before first accept:      {avg_before_accept:.1f}")

    # How many successful repos had zero intermediate failures?
    first_try = sum(1 for r in succ_repos if r["calls_before_first_accept"] == 0)
    needed_retry = len(succ_repos) - first_try
    print(f"\n  Succeeded on first VerifyBuild:  {first_try}/{len(succ_repos)} ({safe_div(first_try, len(succ_repos))*100:.1f}%)")
    print(f"  Needed multiple VerifyBuild calls: {needed_retry}/{len(succ_repos)} ({safe_div(needed_retry, len(succ_repos))*100:.1f}%)")

    # Distribution of calls_before_first_accept for successful repos
    before_dist = Counter(r["calls_before_first_accept"] for r in succ_repos)
    print(f"\n  Failures before first accept (successful repos):")
    for k in sorted(before_dist.keys()):
        print(f"    {k} failures: {before_dist[k]} repos")

    # 3. What errors do successful repos recover from?
    print("\n" + "=" * 70)
    print("ERRORS THAT SUCCESSFUL REPOS RECOVERED FROM")
    print("=" * 70)

    recovered_errors = Counter()
    unrecovered_errors = Counter()

    for call in all_calls:
        if call["outcome"] != "build_failed":
            continue
        cats = call["error_categories"]
        if call["overall_result"] == "SUCCESS":
            for cat in cats:
                recovered_errors[cat] += 1
        else:
            for cat in cats:
                unrecovered_errors[cat] += 1

    all_error_cats = sorted(set(list(recovered_errors.keys()) + list(unrecovered_errors.keys())))
    print(f"\n  {'Error Category':<25} {'Recovered':>10} {'Unrecovered':>12} {'Recovery %':>12}")
    print(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*12}")

    recovery_rows = []
    for cat in all_error_cats:
        rec = recovered_errors.get(cat, 0)
        unrec = unrecovered_errors.get(cat, 0)
        total = rec + unrec
        rate = safe_div(rec, total) * 100
        print(f"  {cat:<25} {rec:>10} {unrec:>12} {rate:>11.1f}%")
        recovery_rows.append({
            "error_category": cat,
            "recovered": rec,
            "unrecovered": unrec,
            "total": total,
            "recovery_rate": round(rate, 1),
        })

    write_csv(os.path.join(OUT_DIR, "recovery_patterns.csv"), recovery_rows,
              ["error_category", "recovered", "unrecovered", "total", "recovery_rate"])

    # 4. Intermediate build errors in repos that eventually succeeded
    print("\n" + "=" * 70)
    print("BUILD ERROR CATEGORIES IN SUCCESSFUL VS FAILED REPOS")
    print("=" * 70)

    succ_build_fails = [c for c in success_calls if c["outcome"] == "build_failed"]
    fail_build_fails = [c for c in fail_calls if c["outcome"] == "build_failed"]

    succ_error_cats = Counter()
    fail_error_cats = Counter()
    for c in succ_build_fails:
        for cat in c["error_categories"]:
            succ_error_cats[cat] += 1
    for c in fail_build_fails:
        for cat in c["error_categories"]:
            fail_error_cats[cat] += 1

    inter_error_rows = []
    for cat in all_error_cats:
        s = succ_error_cats.get(cat, 0)
        f = fail_error_cats.get(cat, 0)
        inter_error_rows.append({
            "error_category": cat,
            "in_successful_repos": s,
            "in_failed_repos": f,
            "total": s + f,
        })
        print(f"  {cat:<25}  success_repos: {s:>4}   fail_repos: {f:>4}")

    write_csv(os.path.join(OUT_DIR, "intermediate_errors_in_success.csv"), inter_error_rows,
              ["error_category", "in_successful_repos", "in_failed_repos", "total"])

    # 5. Review rejection analysis
    print("\n" + "=" * 70)
    print("REVIEW REJECTION ANALYSIS")
    print("=" * 70)

    rejected_calls = [c for c in all_calls if c["outcome"] == "rejected"]
    total_rejections = len(rejected_calls)
    rejections_in_success = sum(1 for c in rejected_calls if c["overall_result"] == "SUCCESS")
    rejections_in_fail = total_rejections - rejections_in_success

    print(f"\n  Total review rejections: {total_rejections}")
    print(f"    In eventually-successful repos: {rejections_in_success}")
    print(f"    In failed repos:                {rejections_in_fail}")

    # Parse and count review concerns
    concern_counter = Counter()
    for c in all_calls:
        if c["outcome"] == "rejected":
            concerns = parse_json_field(c["review_concerns"])
            for concern in concerns:
                if isinstance(concern, str) and len(concern) > 5:
                    # Normalize concern text (take first 80 chars)
                    concern_counter[concern[:100]] += 1

    print(f"\n  Top review concerns (in rejections):")
    for concern, count in concern_counter.most_common(10):
        print(f"    [{count}] {concern}")

    rejection_rows = []
    for concern, count in concern_counter.most_common(20):
        rejection_rows.append({"concern": concern, "count": count})
    write_csv(os.path.join(OUT_DIR, "review_rejection_analysis.csv"), rejection_rows,
              ["concern", "count"])

    # 6. Smoke test failure analysis across all repos
    print("\n" + "=" * 70)
    print("SMOKE TEST FAILURES ACROSS ALL REPOS")
    print("=" * 70)

    smoke_failed_calls = [c for c in all_calls if c["outcome"] == "smoke_failed"]
    smoke_in_success = sum(1 for c in smoke_failed_calls if c["overall_result"] == "SUCCESS")
    smoke_in_fail = len(smoke_failed_calls) - smoke_in_success

    print(f"\n  Total smoke test failures: {len(smoke_failed_calls)}")
    print(f"    In eventually-successful repos: {smoke_in_success}")
    print(f"    In failed repos:                {smoke_in_fail}")

    # Smoke test command patterns
    smoke_cmd_patterns = Counter()
    for c in smoke_failed_calls:
        commands = parse_json_field(c["smoke_test_commands"])
        for cmd in commands:
            if isinstance(cmd, str):
                # Classify command type
                if "import " in cmd or "python" in cmd:
                    smoke_cmd_patterns["python import/run"] += 1
                elif "node " in cmd or "npm " in cmd:
                    smoke_cmd_patterns["node/npm"] += 1
                elif "java " in cmd or "mvn " in cmd or "gradle" in cmd:
                    smoke_cmd_patterns["java/jvm"] += 1
                elif "go " in cmd or "cargo " in cmd:
                    smoke_cmd_patterns["go/rust"] += 1
                elif "--version" in cmd or "--help" in cmd or "-v" in cmd:
                    smoke_cmd_patterns["version/help check"] += 1
                elif "curl " in cmd or "wget " in cmd:
                    smoke_cmd_patterns["http request"] += 1
                elif "test " in cmd or "ls " in cmd:
                    smoke_cmd_patterns["file existence"] += 1
                else:
                    smoke_cmd_patterns["other"] += 1

    print(f"\n  Smoke test command types (in failures):")
    for pattern, count in smoke_cmd_patterns.most_common():
        print(f"    {pattern:<25}: {count}")

    # 7. Progression within iterations: how does outcome change step-by-step?
    print("\n" + "=" * 70)
    print("VERIFYBUILD PROGRESSION WITHIN REPOS")
    print("=" * 70)

    progression_rows = []
    for stat in per_repo_stats:
        slug = stat["slug"]
        repo_calls = [c for c in all_calls if c["slug"] == slug]
        if len(repo_calls) < 2:
            continue

        outcomes_seq = [classify_call_outcome(c) for c in repo_calls]
        first = outcomes_seq[0]
        last = outcomes_seq[-1]

        # Track stage reached at each call (rejected=0, build_failed=1, smoke_failed=2, accepted=3)
        stage_map = {"rejected": 0, "no_build": 0, "build_failed": 1, "no_smoke": 2, "smoke_failed": 2, "accepted": 3}
        stages = [stage_map.get(o, 0) for o in outcomes_seq]
        max_stage = max(stages)
        min_stage = min(stages)
        final_stage = stages[-1]

        # Did the repo ever reach a higher stage than its final stage?
        ever_higher = max_stage > final_stage

        progression_rows.append({
            "slug": slug,
            "overall_result": stat["overall_result"],
            "total_calls": len(repo_calls),
            "first_outcome": first,
            "last_outcome": last,
            "max_stage_reached": max_stage,
            "final_stage": final_stage,
            "regressed_from_peak": ever_higher,
        })

    regressed = [r for r in progression_rows if r["regressed_from_peak"]]
    regressed_in_fail = [r for r in regressed if r["overall_result"] == "FAIL"]
    regressed_in_succ = [r for r in regressed if r["overall_result"] == "SUCCESS"]

    print(f"\n  Repos that regressed from their peak stage: {len(regressed)}/{len(progression_rows)}")
    print(f"    In successful repos: {len(regressed_in_succ)}")
    print(f"    In failed repos:     {len(regressed_in_fail)}")

    # Among failed repos: how many reached smoke stage at some point?
    fail_reached_smoke = [r for r in progression_rows if r["overall_result"] == "FAIL" and r["max_stage_reached"] >= 2]
    fail_reached_build = [r for r in progression_rows if r["overall_result"] == "FAIL" and r["max_stage_reached"] >= 1]
    print(f"\n  Failed repos that reached build success at some point: {len(fail_reached_smoke)}")
    print(f"  Failed repos that got past review at some point:      {len(fail_reached_build)}")

    write_csv(os.path.join(OUT_DIR, "smoke_vs_build_progression.csv"), progression_rows,
              ["slug", "overall_result", "total_calls", "first_outcome", "last_outcome",
               "max_stage_reached", "final_stage", "regressed_from_peak"])

    # 8. Per-repo VerifyBuild stats
    write_csv(os.path.join(OUT_DIR, "per_repo_verifybuild.csv"), per_repo_stats,
              ["slug", "overall_result", "total_verify_calls", "rejected", "build_failed",
               "smoke_failed", "accepted", "calls_before_first_accept"])

    # 9. Summary JSON
    summary = {
        "total_repos": len(per_repo_stats),
        "total_verifybuild_calls": len(all_calls),
        "calls_in_successful_repos": len(success_calls),
        "calls_in_failed_repos": len(fail_calls),
        "outcome_distribution": dict(outcome_counts),
        "avg_calls_per_successful_repo": round(avg_calls_succ, 1),
        "avg_calls_per_failed_repo": round(avg_calls_fail, 1),
        "avg_failures_before_first_accept": round(avg_before_accept, 1),
        "first_try_success_count": first_try,
        "first_try_success_rate": round(safe_div(first_try, len(succ_repos)) * 100, 1),
        "total_review_rejections": total_rejections,
        "rejections_in_successful_repos": rejections_in_success,
        "total_smoke_failures": len(smoke_failed_calls),
        "smoke_failures_in_successful_repos": smoke_in_success,
        "repos_that_regressed": len(regressed),
        "failed_repos_that_reached_smoke": len(fail_reached_smoke),
    }

    with open(os.path.join(OUT_DIR, "verifybuild_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll outputs written to {OUT_DIR}/")
    return summary, per_repo_stats, all_calls, recovery_rows, progression_rows


if __name__ == "__main__":
    main()
