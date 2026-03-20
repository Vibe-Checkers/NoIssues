#!/usr/bin/env python3
"""
analyze_failures.py — Deep failure analysis of the 86 failed repositories.

Examines every failed repo's full run history to determine:
  1. Root cause classification (build error, smoke failure, review rejection, no attempt, crash)
  2. Build error taxonomy (dependency issues, version mismatch, missing files, network, etc.)
  3. Smoke test failure taxonomy
  4. Iteration progression: did the agent improve across retries?
  5. Dockerfile quality signals even in failures
  6. Tool usage patterns in failed vs successful repos
  7. Per-repo failure report cards
  8. Cross-reference with repo characteristics

Outputs go to result_analysis/failure_analysis/

Usage:
    python3 result_analysis/analyze_failures.py
"""

import csv
import json
import os
import re
from collections import Counter, defaultdict

# ── Paths ───────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(__file__)
PER_REPO_DIR = os.path.join(BASE_DIR, "per_repo")
SUMMARY_CSV = os.path.join(BASE_DIR, "summary.csv")
CHAR_CSV = os.path.join(BASE_DIR, "stratified_repos_2000_majority_vote.csv")
MERGED_CSV = os.path.join(BASE_DIR, "analysis_output", "merged_dataset.csv")
OUT_DIR = os.path.join(BASE_DIR, "failure_analysis")


# ── Build error classification patterns ─────────────────────────────────────

BUILD_ERROR_PATTERNS = [
    # Dependency / package resolution
    ("dependency_not_found", [
        r"no matching (version|package|distribution)",
        r"could not (find|resolve|locate)",
        r"package .+ not found",
        r"no such package",
        r"unresolvable dependencies",
        r"failed to (fetch|download|resolve)",
        r"unable to locate package",
        r"ModuleNotFoundError",
        r"ImportError",
        r"cannot find module",
        r"module not found",
        r"not found in any of the sources",
        r"Could not find artifact",
        r"404 Not Found",
        r"dependency .+ was not found",
    ]),
    # Version / compatibility mismatch
    ("version_mismatch", [
        r"version .+ (not compatible|incompatible|is older|is required)",
        r"requires .+ version",
        r"edition.+is required",
        r"this version of .+ is older",
        r"minimum required .+ version",
        r"unsupported .+ version",
        r"feature .+ is required",
        r"requires (at least|minimum)",
        r"engine .+ is incompatible",
        r"does not satisfy",
        r"peer dep",
    ]),
    # Compilation / build errors
    ("compilation_error", [
        r"error(\[E\d+\])?:.*compile",
        r"fatal error:",
        r"cannot find -l",
        r"undefined reference",
        r"linker command failed",
        r"error:.*cannot find",
        r"cc1.*error",
        r"Build FAILED",
        r"LINK : fatal error",
        r"syntax error",
        r"type error",
        r"error TS\d+",
    ]),
    # Missing system libraries / headers
    ("missing_system_dep", [
        r"No such file or directory.*\.h",
        r"fatal error:.*\.h.*No such file",
        r"pkg-config.*not found",
        r"lib\w+ not found",
        r"Could not find .+Config\.cmake",
        r"apt-get.*(install|unable)",
        r"missing (required |)library",
        r"cannot find.*header",
        r"CMake Error.*could not find",
    ]),
    # Network / download failures
    ("network_error", [
        r"(connection|connect) (timed out|refused|reset)",
        r"network (is )?unreachable",
        r"failed to (connect|download|fetch)",
        r"Could not connect to",
        r"curl.*error",
        r"SSL.*error",
        r"timeout",
        r"temporary failure (in|resolving)",
    ]),
    # Dockerfile syntax / instruction errors
    ("dockerfile_syntax", [
        r"failed to process.*Dockerfile",
        r"unknown instruction",
        r"COPY failed",
        r"ADD failed",
        r"file not found.*COPY",
        r"When using COPY with more than one source",
        r"the --from flag must",
        r"failed to compute cache key",
        r"failed to calculate checksum",
        r"source can't be a URL for COPY",
    ]),
    # Permission / access errors
    ("permission_error", [
        r"permission denied",
        r"access denied",
        r"EACCES",
        r"Operation not permitted",
        r"403 Forbidden",
        r"authentication required",
    ]),
    # Out of memory / resources
    ("resource_exhaustion", [
        r"out of memory",
        r"OOM",
        r"Cannot allocate memory",
        r"no space left on device",
        r"killed",
        r"signal: killed",
        r"ENOMEM",
    ]),
    # Platform / architecture issues
    ("platform_issue", [
        r"exec format error",
        r"no matching manifest.*platform",
        r"does not match the detected host platform",
        r"incompatible platform",
        r"unsupported (os|arch|platform)",
        r"linux/arm",
    ]),
    # Missing source files in repo
    ("missing_source_file", [
        r"COPY .+\. no such file",
        r"file not found",
        r"No such file or directory",
        r"does not exist",
        r"not found in build context",
    ]),
]

SMOKE_FAILURE_PATTERNS = [
    ("import_error", [r"ImportError", r"ModuleNotFoundError", r"cannot find module", r"Cannot find module"]),
    ("runtime_crash", [r"Segmentation fault", r"core dumped", r"signal: (aborted|killed|segfault)", r"panic:", r"SIGSEGV"]),
    ("missing_binary", [r"not found", r"No such file or directory", r"command not found", r"exec:.*not found"]),
    ("config_error", [r"configuration error", r"config.*not found", r"missing.*config", r"environment variable.*not set"]),
    ("port_bind", [r"address already in use", r"EADDRINUSE", r"bind.*failed"]),
    ("permission_denied", [r"permission denied", r"EACCES"]),
    ("dependency_missing", [r"no module named", r"require.*not found", r"Class.*not found"]),
    ("timeout", [r"timed? ?out", r"deadline exceeded"]),
    ("version_error", [r"version", r"incompatible", r"unsupported"]),
]


def classify_build_error(error_text: str) -> list[str]:
    """Classify a build error into one or more categories."""
    if not error_text:
        return ["unknown"]
    categories = []
    error_lower = error_text.lower()
    for category, patterns in BUILD_ERROR_PATTERNS:
        for pat in patterns:
            if re.search(pat, error_text, re.IGNORECASE):
                categories.append(category)
                break
    return categories if categories else ["unclassified"]


def classify_smoke_failure(smoke_text: str) -> list[str]:
    """Classify smoke test failure."""
    if not smoke_text:
        return ["unknown"]
    categories = []
    for category, patterns in SMOKE_FAILURE_PATTERNS:
        for pat in patterns:
            if re.search(pat, smoke_text, re.IGNORECASE):
                categories.append(category)
                break
    return categories if categories else ["unclassified"]


def extract_failing_docker_stage(error_text: str) -> str | None:
    """Extract the failing Docker build step."""
    if not error_text:
        return None
    m = re.search(r'\[[\w-]+ (\d+/\d+)\] (RUN .+)', error_text)
    if m:
        return m.group(2)[:120]
    m = re.search(r'Step (\d+/\d+) : (.+)', error_text)
    if m:
        return m.group(2)[:120]
    m = re.search(r'(RUN|COPY|ADD|FROM) .+', error_text)
    if m:
        return m.group(0)[:120]
    return None


def extract_base_image(dockerfile: str) -> str | None:
    """Extract base image from Dockerfile content."""
    if not dockerfile:
        return None
    m = re.search(r'^FROM\s+(\S+)', dockerfile, re.MULTILINE)
    return m.group(1) if m else None


def count_dockerfile_stages(dockerfile: str) -> int:
    """Count multi-stage build stages."""
    if not dockerfile:
        return 0
    return len(re.findall(r'^FROM\s+', dockerfile, re.MULTILINE))


def dockerfile_line_count(dockerfile: str) -> int:
    if not dockerfile:
        return 0
    return len([l for l in dockerfile.strip().split('\n') if l.strip() and not l.strip().startswith('#')])


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

    # Load characterization data
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

    fail_repos = {url: d for url, d in all_repos.items() if d["overall_result"] == "FAIL"}
    success_repos = {url: d for url, d in all_repos.items() if d["overall_result"] == "SUCCESS"}

    print(f"Loaded {len(all_repos)} repos: {len(success_repos)} success, {len(fail_repos)} fail")

    # ════════════════════════════════════════════════════════════════════════
    # 1. ROOT CAUSE CLASSIFICATION
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("1. ROOT CAUSE CLASSIFICATION")
    print("=" * 70)

    repo_reports = []

    for url, data in sorted(fail_repos.items()):
        slug = data["slug"]
        char = char_lookup.get(url.rstrip("/"), {})

        report = {
            "repo_url": url,
            "slug": slug,
            "domain": char.get("domain", ""),
            "build_type": char.get("build_type", ""),
            "automation_level": char.get("automation_level", ""),
            "dependency_transparency": char.get("dependency_transparency", ""),
            "tooling_complexity": char.get("tooling_complexity", ""),
            "run_count": len(data["runs"]),
            "root_cause": "unknown",
            "sub_causes": [],
            "total_iterations": 0,
            "total_verify_attempts": 0,
            "total_build_attempts": 0,
            "total_build_failures": 0,
            "total_smoke_attempts": 0,
            "total_smoke_failures": 0,
            "total_rejections": 0,
            "max_iterations_reached": False,
            "had_any_build_success": False,
            "had_any_smoke_attempt": False,
            "all_build_errors": [],
            "all_smoke_errors": [],
            "all_review_concerns": [],
            "all_dockerfiles": [],
            "all_base_images": [],
            "failing_steps": [],
            "build_error_categories": [],
            "smoke_error_categories": [],
            "iteration_progression": [],
            "tool_usage": Counter(),
            "error_message": None,
            "last_verify_result": None,
        }

        for run in data["runs"]:
            if run.get("error_message"):
                report["error_message"] = run["error_message"]

            iterations = run.get("iterations", [])
            report["total_iterations"] += len(iterations)

            for it in iterations:
                iter_info = {
                    "iteration": it.get("iteration_number", 0),
                    "status": it.get("status", ""),
                    "verify_result": it.get("verify_result"),
                    "step_count": it.get("step_count", 0),
                    "dockerfile_generated": it.get("dockerfile_generated", False),
                    "verify_attempted": it.get("verify_attempted", False),
                    "had_build_success": False,
                    "had_smoke_pass": False,
                    "build_error_cats": [],
                }

                for s in it.get("steps", []):
                    report["tool_usage"][s["tool_name"]] += 1

                    if s["tool_name"] == "VerifyBuild":
                        report["total_verify_attempts"] += 1

                        for v in s.get("verify_build_details", []):
                            # Review concerns
                            concerns = v.get("review_concerns", "")
                            if concerns:
                                if isinstance(concerns, str):
                                    try:
                                        concerns = json.loads(concerns)
                                    except (json.JSONDecodeError, TypeError):
                                        concerns = [concerns]
                                if isinstance(concerns, list):
                                    report["all_review_concerns"].extend(concerns)

                            if v.get("review_approved") == False:
                                report["total_rejections"] += 1

                            # Build
                            if v.get("build_attempted"):
                                report["total_build_attempts"] += 1
                                if v.get("build_success"):
                                    report["had_any_build_success"] = True
                                    iter_info["had_build_success"] = True
                                elif v.get("build_success") == False:
                                    report["total_build_failures"] += 1
                                    err = v.get("build_error") or v.get("build_error_raw") or ""
                                    if err:
                                        report["all_build_errors"].append(err)
                                        cats = classify_build_error(err)
                                        report["build_error_categories"].extend(cats)
                                        iter_info["build_error_cats"].extend(cats)
                                        step_cmd = extract_failing_docker_stage(err)
                                        if step_cmd:
                                            report["failing_steps"].append(step_cmd)

                            # Smoke
                            if v.get("smoke_attempted"):
                                report["total_smoke_attempts"] += 1
                                report["had_any_smoke_attempt"] = True
                                if v.get("smoke_passed") == False:
                                    report["total_smoke_failures"] += 1
                                    sr = v.get("smoke_results") or ""
                                    if isinstance(sr, str):
                                        report["all_smoke_errors"].append(sr)
                                        report["smoke_error_categories"].extend(classify_smoke_failure(sr))
                                    iter_info["had_smoke_pass"] = False
                                elif v.get("smoke_passed"):
                                    iter_info["had_smoke_pass"] = True

                    # Collect dockerfiles from WriteFile calls
                    if s["tool_name"] == "WriteFile":
                        tool_input = s.get("tool_input", "")
                        if isinstance(tool_input, str):
                            try:
                                tool_input = json.loads(tool_input)
                            except (json.JSONDecodeError, TypeError):
                                tool_input = {}
                        if isinstance(tool_input, dict):
                            fname = tool_input.get("file_path", "") or tool_input.get("path", "")
                            if "dockerfile" in fname.lower() or "Dockerfile" in fname:
                                content = tool_input.get("content", "")
                                if content:
                                    report["all_dockerfiles"].append(content)
                                    base = extract_base_image(content)
                                    if base:
                                        report["all_base_images"].append(base)

                report["iteration_progression"].append(iter_info)
                report["last_verify_result"] = it.get("verify_result")

            if len(iterations) >= 3:
                report["max_iterations_reached"] = True

        # ── Determine root cause ────────────────────────────────────────────
        if report["error_message"] and report["total_verify_attempts"] == 0:
            report["root_cause"] = "agent_crash"
        elif report["total_verify_attempts"] == 0:
            if report["total_iterations"] == 0:
                report["root_cause"] = "no_iterations"
            else:
                report["root_cause"] = "no_verify_attempt"
        elif report["total_rejections"] > 0 and report["total_build_attempts"] == 0:
            report["root_cause"] = "all_rejected"
        elif report["total_build_failures"] > 0 and not report["had_any_build_success"]:
            report["root_cause"] = "build_always_failed"
        elif report["had_any_build_success"] and report["total_smoke_failures"] > 0:
            report["root_cause"] = "smoke_test_failure"
        elif report["total_build_failures"] > 0 and report["had_any_build_success"]:
            report["root_cause"] = "intermittent_build"
        elif report["total_rejections"] > 0:
            report["root_cause"] = "mostly_rejected"
        else:
            report["root_cause"] = "other"

        # Dominant build error sub-cause
        if report["build_error_categories"]:
            report["sub_causes"] = [c for c, _ in Counter(report["build_error_categories"]).most_common(3)]
        elif report["smoke_error_categories"]:
            report["sub_causes"] = [c for c, _ in Counter(report["smoke_error_categories"]).most_common(3)]

        repo_reports.append(report)

    # ── Root cause distribution ─────────────────────────────────────────────
    root_cause_counts = Counter(r["root_cause"] for r in repo_reports)
    print(f"\n  {'Root Cause':<30} {'Count':>5} {'%':>6}")
    print(f"  {'─'*30} {'─'*5} {'─'*6}")
    for cause, count in root_cause_counts.most_common():
        print(f"  {cause:<30} {count:>5} {count/len(repo_reports)*100:>5.1f}%")

    # Save
    root_cause_rows = [{"root_cause": c, "count": n, "pct": round(n / len(repo_reports) * 100, 1)}
                       for c, n in root_cause_counts.most_common()]
    write_csv(os.path.join(OUT_DIR, "root_cause_distribution.csv"), root_cause_rows,
              ["root_cause", "count", "pct"])

    # ════════════════════════════════════════════════════════════════════════
    # 2. BUILD ERROR TAXONOMY
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("2. BUILD ERROR TAXONOMY")
    print("=" * 70)

    # Per-repo (deduplicated: count each category once per repo)
    repo_build_cats = Counter()
    for r in repo_reports:
        unique_cats = set(r["build_error_categories"])
        for c in unique_cats:
            repo_build_cats[c] += 1

    total_with_build_err = sum(1 for r in repo_reports if r["build_error_categories"])

    print(f"\n  Build error categories (repos affected, N={total_with_build_err} repos with build errors):")
    print(f"  {'Category':<30} {'Repos':>5} {'%':>6}")
    print(f"  {'─'*30} {'─'*5} {'─'*6}")
    build_cat_rows = []
    for cat, cnt in repo_build_cats.most_common():
        pct = cnt / total_with_build_err * 100 if total_with_build_err else 0
        print(f"  {cat:<30} {cnt:>5} {pct:>5.1f}%")
        build_cat_rows.append({"category": cat, "repos_affected": cnt, "pct": round(pct, 1)})
    write_csv(os.path.join(OUT_DIR, "build_error_taxonomy.csv"), build_cat_rows,
              ["category", "repos_affected", "pct"])

    # Occurrence-level counts (total across all iterations)
    all_build_cats = Counter()
    for r in repo_reports:
        all_build_cats.update(r["build_error_categories"])
    print(f"\n  Total build error occurrences: {sum(all_build_cats.values())}")
    for cat, cnt in all_build_cats.most_common():
        print(f"    {cat:<30} {cnt:>5}")

    # ════════════════════════════════════════════════════════════════════════
    # 3. SMOKE TEST FAILURE TAXONOMY
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("3. SMOKE TEST FAILURE TAXONOMY")
    print("=" * 70)

    repo_smoke_cats = Counter()
    for r in repo_reports:
        unique = set(r["smoke_error_categories"])
        for c in unique:
            repo_smoke_cats[c] += 1

    total_with_smoke = sum(1 for r in repo_reports if r["smoke_error_categories"])
    print(f"\n  Smoke error categories (repos affected, N={total_with_smoke} repos with smoke failures):")
    print(f"  {'Category':<30} {'Repos':>5}")
    print(f"  {'─'*30} {'─'*5}")
    smoke_cat_rows = []
    for cat, cnt in repo_smoke_cats.most_common():
        print(f"  {cat:<30} {cnt:>5}")
        smoke_cat_rows.append({"category": cat, "repos_affected": cnt})
    write_csv(os.path.join(OUT_DIR, "smoke_error_taxonomy.csv"), smoke_cat_rows,
              ["category", "repos_affected"])

    # ════════════════════════════════════════════════════════════════════════
    # 4. ITERATION PROGRESSION — DID RETRIES HELP?
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("4. ITERATION PROGRESSION ANALYSIS")
    print("=" * 70)

    # For repos with multiple iterations, did the error category change?
    progression_stats = {
        "total_multi_iter_repos": 0,
        "error_changed": 0,
        "error_same": 0,
        "got_further": 0,  # e.g. went from build_fail to smoke_fail
        "regressed": 0,
    }

    progress_stages = {"no_verify_attempt": 0, "rejected": 1, "build_failed": 2, "smoke_failed": 3}

    for r in repo_reports:
        iters = r["iteration_progression"]
        if len(iters) < 2:
            continue
        progression_stats["total_multi_iter_repos"] += 1

        # Compare first vs last iteration
        first_vr = iters[0].get("verify_result") or "no_verify_attempt"
        last_vr = iters[-1].get("verify_result") or "no_verify_attempt"

        first_stage = progress_stages.get(first_vr, -1)
        last_stage = progress_stages.get(last_vr, -1)

        if last_stage > first_stage:
            progression_stats["got_further"] += 1
        elif last_stage < first_stage:
            progression_stats["regressed"] += 1

        first_cats = set(iters[0].get("build_error_cats", []))
        last_cats = set(iters[-1].get("build_error_cats", []))
        if first_cats and last_cats:
            if first_cats != last_cats:
                progression_stats["error_changed"] += 1
            else:
                progression_stats["error_same"] += 1

    print(f"\n  Repos with multiple iterations: {progression_stats['total_multi_iter_repos']}")
    print(f"  Agent progressed further:       {progression_stats['got_further']}")
    print(f"  Agent regressed:                {progression_stats['regressed']}")
    print(f"  Build error type changed:       {progression_stats['error_changed']}")
    print(f"  Build error type unchanged:     {progression_stats['error_same']}")

    with open(os.path.join(OUT_DIR, "iteration_progression.json"), "w") as f:
        json.dump(progression_stats, f, indent=2)

    # Verify result distribution per iteration number
    iter_verify_dist = defaultdict(Counter)
    for r in repo_reports:
        for ip in r["iteration_progression"]:
            it_num = ip["iteration"]
            vr = ip.get("verify_result") or "no_result"
            iter_verify_dist[it_num][vr] += 1

    print(f"\n  Verify result distribution by iteration number:")
    print(f"  {'Iter':<6} {'build_failed':>12} {'smoke_failed':>12} {'rejected':>10} {'no_result':>10}")
    print(f"  {'─'*6} {'─'*12} {'─'*12} {'─'*10} {'─'*10}")
    iter_dist_rows = []
    for it_num in sorted(iter_verify_dist.keys()):
        d = iter_verify_dist[it_num]
        print(f"  {it_num:<6} {d.get('build_failed',0):>12} {d.get('smoke_failed',0):>12} {d.get('rejected',0):>10} {d.get('no_result',0):>10}")
        row = {"iteration": it_num}
        row.update(dict(d))
        iter_dist_rows.append(row)

    # ════════════════════════════════════════════════════════════════════════
    # 5. DOCKERFILE ANALYSIS (from failed attempts)
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("5. DOCKERFILE ANALYSIS (failed repos)")
    print("=" * 70)

    base_image_counter = Counter()
    stage_counts = []
    line_counts = []
    repos_with_dockerfiles = 0

    for r in repo_reports:
        if not r["all_dockerfiles"]:
            continue
        repos_with_dockerfiles += 1
        # Use the last dockerfile attempt
        last_df = r["all_dockerfiles"][-1]
        base = extract_base_image(last_df)
        if base:
            # Normalize base image (strip tag for grouping)
            base_clean = base.split(":")[0] if ":" in base else base
            base_image_counter[base_clean] += 1
        stage_counts.append(count_dockerfile_stages(last_df))
        line_counts.append(dockerfile_line_count(last_df))

    print(f"\n  Repos that generated at least one Dockerfile: {repos_with_dockerfiles}/{len(repo_reports)}")
    print(f"  Repos with NO Dockerfile at all: {len(repo_reports) - repos_with_dockerfiles}")

    if stage_counts:
        print(f"\n  Multi-stage builds: {sum(1 for s in stage_counts if s > 1)}/{len(stage_counts)}")
        print(f"  Avg stages: {sum(stage_counts)/len(stage_counts):.1f}")
        print(f"  Avg Dockerfile lines: {sum(line_counts)/len(line_counts):.1f}")

    print(f"\n  Top base images used (failed repos):")
    base_rows = []
    for img, cnt in base_image_counter.most_common(20):
        print(f"    {img:<40} {cnt:>3}")
        base_rows.append({"base_image": img, "count": cnt})
    write_csv(os.path.join(OUT_DIR, "failed_base_images.csv"), base_rows,
              ["base_image", "count"])

    # ════════════════════════════════════════════════════════════════════════
    # 6. FAILING DOCKER STEPS (what commands fail most)
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("6. MOST COMMON FAILING DOCKER COMMANDS")
    print("=" * 70)

    # Normalize failing steps
    step_cats = Counter()
    for r in repo_reports:
        for step_cmd in r["failing_steps"]:
            # Categorize by command type
            cmd = step_cmd.strip()
            if cmd.startswith("RUN pip ") or cmd.startswith("RUN python -m pip"):
                step_cats["RUN pip install"] += 1
            elif cmd.startswith("RUN npm ") or cmd.startswith("RUN yarn ") or cmd.startswith("RUN pnpm "):
                step_cats["RUN npm/yarn/pnpm install"] += 1
            elif cmd.startswith("RUN cargo"):
                step_cats["RUN cargo build/install"] += 1
            elif cmd.startswith("RUN go "):
                step_cats["RUN go build"] += 1
            elif cmd.startswith("RUN make") or cmd.startswith("RUN cmake"):
                step_cats["RUN make/cmake"] += 1
            elif cmd.startswith("RUN apt") or cmd.startswith("RUN apk") or cmd.startswith("RUN yum"):
                step_cats["RUN apt/apk install"] += 1
            elif cmd.startswith("RUN gradle") or cmd.startswith("RUN ./gradlew") or cmd.startswith("RUN mvn"):
                step_cats["RUN gradle/maven"] += 1
            elif cmd.startswith("RUN dotnet"):
                step_cats["RUN dotnet build"] += 1
            elif cmd.startswith("COPY") or cmd.startswith("ADD"):
                step_cats["COPY/ADD"] += 1
            else:
                step_cats["other RUN"] += 1

    print(f"\n  {'Failing command type':<35} {'Count':>5}")
    print(f"  {'─'*35} {'─'*5}")
    fail_step_rows = []
    for cat, cnt in step_cats.most_common():
        print(f"  {cat:<35} {cnt:>5}")
        fail_step_rows.append({"command_type": cat, "count": cnt})
    write_csv(os.path.join(OUT_DIR, "failing_docker_commands.csv"), fail_step_rows,
              ["command_type", "count"])

    # ════════════════════════════════════════════════════════════════════════
    # 7. TOOL USAGE: FAILED vs SUCCESSFUL repos
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("7. TOOL USAGE COMPARISON: FAILED vs SUCCESSFUL")
    print("=" * 70)

    # Gather tool usage for successful repos
    success_tool_usage = Counter()
    success_count = 0
    for url, data in success_repos.items():
        success_count += 1
        for run in data["runs"]:
            if run["status"] != "success":
                continue
            for it in run.get("iterations", []):
                for s in it.get("steps", []):
                    success_tool_usage[s["tool_name"]] += 1
            break  # only first successful run

    fail_tool_usage = Counter()
    for r in repo_reports:
        fail_tool_usage.update(r["tool_usage"])

    all_tools = sorted(set(list(success_tool_usage.keys()) + list(fail_tool_usage.keys())))
    print(f"\n  {'Tool':<25} {'Fail(avg)':>10} {'Success(avg)':>12} {'Diff':>8}")
    print(f"  {'─'*25} {'─'*10} {'─'*12} {'─'*8}")

    tool_rows = []
    for tool in all_tools:
        f_avg = fail_tool_usage[tool] / len(fail_repos) if fail_repos else 0
        s_avg = success_tool_usage[tool] / success_count if success_count else 0
        diff = f_avg - s_avg
        print(f"  {tool:<25} {f_avg:>10.2f} {s_avg:>12.2f} {diff:>+7.2f}")
        tool_rows.append({"tool": tool, "fail_avg": round(f_avg, 2), "success_avg": round(s_avg, 2), "diff": round(diff, 2)})

    write_csv(os.path.join(OUT_DIR, "tool_usage_comparison.csv"), tool_rows,
              ["tool", "fail_avg", "success_avg", "diff"])

    # ════════════════════════════════════════════════════════════════════════
    # 8. REVIEW CONCERNS ANALYSIS
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("8. COMMON REVIEW CONCERNS (LLM reviewer)")
    print("=" * 70)

    # Keyword extraction from review concerns
    concern_keywords = Counter()
    total_concerns = 0
    for r in repo_reports:
        for concern in r["all_review_concerns"]:
            total_concerns += 1
            c_lower = concern.lower()
            # Keyword categories
            if any(w in c_lower for w in ["version", "pin", "tag", "latest"]):
                concern_keywords["version_pinning"] += 1
            if any(w in c_lower for w in ["expose", "port"]):
                concern_keywords["port_expose"] += 1
            if any(w in c_lower for w in ["copy", "path", "wrong path", "missing file"]):
                concern_keywords["copy_path_issues"] += 1
            if any(w in c_lower for w in ["multi-stage", "stage", "builder"]):
                concern_keywords["multi_stage"] += 1
            if any(w in c_lower for w in ["security", "root", "user", "privilege"]):
                concern_keywords["security"] += 1
            if any(w in c_lower for w in ["cache", "layer", "optimize"]):
                concern_keywords["caching_optimization"] += 1
            if any(w in c_lower for w in ["env", "environment", "variable"]):
                concern_keywords["env_variables"] += 1
            if any(w in c_lower for w in ["entrypoint", "cmd", "command"]):
                concern_keywords["entrypoint_cmd"] += 1
            if any(w in c_lower for w in ["workdir", "working dir"]):
                concern_keywords["workdir"] += 1
            if any(w in c_lower for w in ["apt", "apk", "install", "dependency", "package"]):
                concern_keywords["dependency_install"] += 1

    print(f"\n  Total review concerns: {total_concerns}")
    print(f"  {'Concern theme':<30} {'Count':>5}")
    print(f"  {'─'*30} {'─'*5}")
    concern_rows = []
    for kw, cnt in concern_keywords.most_common():
        print(f"  {kw:<30} {cnt:>5}")
        concern_rows.append({"concern_theme": kw, "count": cnt})
    write_csv(os.path.join(OUT_DIR, "review_concerns.csv"), concern_rows,
              ["concern_theme", "count"])

    # ════════════════════════════════════════════════════════════════════════
    # 9. ROOT CAUSE × REPO CHARACTERISTICS CROSS-TAB
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("9. ROOT CAUSE × CHARACTERISTICS")
    print("=" * 70)

    for char_col in ["domain", "build_type"]:
        cross = defaultdict(Counter)
        for r in repo_reports:
            cross[r[char_col]][r["root_cause"]] += 1

        print(f"\n  {char_col.upper()} × ROOT CAUSE:")
        all_causes = sorted(set(c for d in cross.values() for c in d))
        header = f"  {'Value':<25}" + "".join(f" {c[:15]:>15}" for c in all_causes)
        print(header)
        print(f"  {'─'*25}" + "─" * 15 * len(all_causes))

        cross_rows = []
        for val in sorted(cross.keys(), key=lambda v: -sum(cross[v].values())):
            row_str = f"  {val:<25}"
            row_data = {char_col: val}
            for c in all_causes:
                row_str += f" {cross[val].get(c, 0):>15}"
                row_data[c] = cross[val].get(c, 0)
            print(row_str)
            cross_rows.append(row_data)

        write_csv(os.path.join(OUT_DIR, f"cross_rootcause_{char_col}.csv"),
                  cross_rows, [char_col] + all_causes)

    # ════════════════════════════════════════════════════════════════════════
    # 10. PER-REPO FAILURE REPORT CARDS
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("10. WRITING PER-REPO FAILURE REPORTS")
    print("=" * 70)

    report_rows = []
    for r in repo_reports:
        report_rows.append({
            "slug": r["slug"],
            "repo_url": r["repo_url"],
            "domain": r["domain"],
            "build_type": r["build_type"],
            "root_cause": r["root_cause"],
            "sub_causes": "; ".join(r["sub_causes"]),
            "run_count": r["run_count"],
            "total_iterations": r["total_iterations"],
            "total_verify_attempts": r["total_verify_attempts"],
            "total_build_attempts": r["total_build_attempts"],
            "total_build_failures": r["total_build_failures"],
            "total_smoke_attempts": r["total_smoke_attempts"],
            "total_smoke_failures": r["total_smoke_failures"],
            "total_rejections": r["total_rejections"],
            "had_any_build_success": r["had_any_build_success"],
            "max_iterations_reached": r["max_iterations_reached"],
            "num_dockerfiles_generated": len(r["all_dockerfiles"]),
            "last_verify_result": r["last_verify_result"],
            "base_images_tried": "; ".join(set(r["all_base_images"])),
            "error_message": (r["error_message"] or "")[:300],
        })

    write_csv(
        os.path.join(OUT_DIR, "per_repo_failure_report.csv"),
        report_rows,
        list(report_rows[0].keys()),
    )
    print(f"  Wrote {len(report_rows)} repo failure reports")

    # ════════════════════════════════════════════════════════════════════════
    # 11. AGGREGATE SUMMARY JSON
    # ════════════════════════════════════════════════════════════════════════

    # Build error examples (top 3 per category for reference)
    error_examples = defaultdict(list)
    for r in repo_reports:
        for i, err in enumerate(r["all_build_errors"]):
            cats = classify_build_error(err)
            for cat in cats:
                if len(error_examples[cat]) < 3:
                    error_examples[cat].append({
                        "repo": r["slug"],
                        "error_snippet": err[:400],
                    })

    summary = {
        "total_failed_repos": len(repo_reports),
        "root_cause_distribution": dict(root_cause_counts),
        "build_error_taxonomy": dict(repo_build_cats),
        "smoke_error_taxonomy": dict(repo_smoke_cats),
        "iteration_progression": progression_stats,
        "dockerfile_stats": {
            "repos_with_dockerfile": repos_with_dockerfiles,
            "repos_without_dockerfile": len(repo_reports) - repos_with_dockerfiles,
            "avg_stages": round(sum(stage_counts) / len(stage_counts), 1) if stage_counts else 0,
            "avg_lines": round(sum(line_counts) / len(line_counts), 1) if line_counts else 0,
        },
        "top_base_images": dict(base_image_counter.most_common(15)),
        "failing_command_types": dict(step_cats),
        "review_concern_themes": dict(concern_keywords),
        "error_examples": {cat: examples for cat, examples in error_examples.items()},
    }

    with open(os.path.join(OUT_DIR, "failure_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ════════════════════════════════════════════════════════════════════════
    # 12. LaTeX TABLES
    # ════════════════════════════════════════════════════════════════════════

    latex_path = os.path.join(OUT_DIR, "failure_latex_tables.tex")
    with open(latex_path, "w") as f:
        f.write("% Auto-generated LaTeX tables — failure analysis\n\n")

        # Root cause table
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Root Cause Classification of Failed Repositories (N=86)}\n")
        f.write("\\label{tab:root-causes}\n")
        f.write("\\begin{tabular}{lrr}\n\\toprule\n")
        f.write("Root Cause & Count & \\% \\\\\n\\midrule\n")
        for cause, count in root_cause_counts.most_common():
            label = cause.replace("_", " ").title()
            f.write(f"{label} & {count} & {count/len(repo_reports)*100:.1f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

        # Build error taxonomy
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Build Error Categories (repos affected)}\n")
        f.write("\\label{tab:build-errors}\n")
        f.write("\\begin{tabular}{lrr}\n\\toprule\n")
        f.write("Error Category & Repos & \\% \\\\\n\\midrule\n")
        for cat, cnt in repo_build_cats.most_common():
            label = cat.replace("_", " ").title()
            pct = cnt / total_with_build_err * 100 if total_with_build_err else 0
            f.write(f"{label} & {cnt} & {pct:.1f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

        # Tool usage comparison
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Average Tool Usage: Failed vs.\\ Successful Repos}\n")
        f.write("\\label{tab:tool-usage}\n")
        f.write("\\begin{tabular}{lrrr}\n\\toprule\n")
        f.write("Tool & Failed & Successful & $\\Delta$ \\\\\n\\midrule\n")
        for tr in sorted(tool_rows, key=lambda x: -abs(x["diff"])):
            f.write(f"{tr['tool']} & {tr['fail_avg']:.2f} & {tr['success_avg']:.2f} & {tr['diff']:+.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

    print(f"\n  LaTeX tables written to {latex_path}")

    # ── Final file listing ──────────────────────────────────────────────────
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
