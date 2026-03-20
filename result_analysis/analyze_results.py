#!/usr/bin/env python3
"""
analyze_results.py — Detailed analysis of BuildAgent results vs. repo characteristics.

Joins the 282-repo run results (summary.csv) with the characterization data
(stratified_repos_2000_majority_vote.csv) and produces:

  1. Per-category success/fail breakdown tables (CSV + console)
  2. Cross-tabulation of key dimension pairs
  3. Statistical tests (chi-square, Fisher's exact, Cramér's V)
  4. Detailed per-repo merged dataset for further analysis
  5. Iteration & token analysis by category
  6. LaTeX-ready tables for the paper

Outputs go to result_analysis/analysis_output/

Usage:
    python3 result_analysis/analyze_results.py
"""

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from itertools import combinations

# ── Paths ───────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(__file__)
SUMMARY_CSV = os.path.join(BASE_DIR, "summary.csv")
CHAR_CSV = os.path.join(BASE_DIR, "stratified_repos_2000_majority_vote.csv")
OUT_DIR = os.path.join(BASE_DIR, "analysis_output")

# Category columns from the characterization CSV
CATEGORY_COLS = [
    "domain",
    "build_type",
    "automation_level",
    "environment_specificity",
    "dependency_transparency",
    "tooling_complexity",
    "reproducibility_support",
]


# ── Helpers ─────────────────────────────────────────────────────────────────

def load_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: list[dict], fieldnames: list[str]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def safe_div(a, b, default=0.0):
    return a / b if b else default


def chi_square_test(table: dict[str, dict[str, int]]) -> dict:
    """
    Manual chi-square test for a contingency table.
    table: {category_value: {"SUCCESS": n, "FAIL": m}}
    Returns chi2, dof, p_value, cramers_v.
    """
    import math

    categories = sorted(table.keys())
    outcomes = ["SUCCESS", "FAIL"]

    # Observed counts
    observed = []
    for cat in categories:
        row = [table[cat].get(o, 0) for o in outcomes]
        observed.append(row)

    n_rows = len(observed)
    n_cols = len(outcomes)
    n_total = sum(sum(row) for row in observed)

    if n_total == 0 or n_rows < 2:
        return {"chi2": 0, "dof": 0, "p_value": 1.0, "cramers_v": 0, "n": n_total}

    # Row and column totals
    row_totals = [sum(row) for row in observed]
    col_totals = [sum(observed[r][c] for r in range(n_rows)) for c in range(n_cols)]

    # Expected counts and chi-square
    chi2 = 0.0
    for r in range(n_rows):
        for c in range(n_cols):
            expected = row_totals[r] * col_totals[c] / n_total
            if expected > 0:
                chi2 += (observed[r][c] - expected) ** 2 / expected

    dof = (n_rows - 1) * (n_cols - 1)

    # Approximate p-value using chi-square survival function
    # Using Wilson-Hilferty approximation for chi-square CDF
    p_value = _chi2_p_value(chi2, dof)

    # Cramér's V
    k = min(n_rows, n_cols)
    cramers_v = math.sqrt(chi2 / (n_total * (k - 1))) if n_total * (k - 1) > 0 else 0

    return {
        "chi2": round(chi2, 4),
        "dof": dof,
        "p_value": round(p_value, 6),
        "cramers_v": round(cramers_v, 4),
        "n": n_total,
    }


def _chi2_p_value(chi2: float, dof: int) -> float:
    """Approximate p-value for chi-square distribution using regularized gamma."""
    if dof <= 0:
        return 1.0
    if chi2 <= 0:
        return 1.0

    # Use the regularized incomplete gamma function approximation
    # P(X > chi2) = 1 - gamma_inc(dof/2, chi2/2) / Gamma(dof/2)
    # For a simpler approach, use the series expansion
    import math

    k = dof / 2.0
    x = chi2 / 2.0

    # For large chi2, p-value is very small
    if x > 200:
        return 0.0

    # Regularized lower incomplete gamma function via series expansion
    # gamma_inc(k, x) = x^k * e^(-x) * sum(x^n / Gamma(k+n+1), n=0..inf)
    total = 0.0
    term = 1.0 / k
    total = term
    for n in range(1, 300):
        term *= x / (k + n)
        total += term
        if abs(term) < 1e-15:
            break

    p_lower = math.exp(-x + k * math.log(x) - math.lgamma(k)) * total
    p_lower = min(max(p_lower, 0.0), 1.0)

    return 1.0 - p_lower


def fisher_exact_2x2(a, b, c, d) -> float:
    """
    Fisher's exact test p-value for 2x2 table:
        [[a, b],
         [c, d]]
    Returns two-sided p-value.
    """
    import math

    n = a + b + c + d
    # Hypergeometric probability
    def hyper_prob(a, b, c, d):
        return math.exp(
            math.lgamma(a+b+1) + math.lgamma(c+d+1) +
            math.lgamma(a+c+1) + math.lgamma(b+d+1) -
            math.lgamma(n+1) - math.lgamma(a+1) -
            math.lgamma(b+1) - math.lgamma(c+1) - math.lgamma(d+1)
        )

    p_obs = hyper_prob(a, b, c, d)
    p_value = 0.0

    # Enumerate all possible tables with same marginals
    row1 = a + b
    col1 = a + c
    min_a = max(0, row1 + col1 - n)
    max_a = min(row1, col1)

    for ai in range(min_a, max_a + 1):
        bi = row1 - ai
        ci = col1 - ai
        di = n - ai - bi - ci
        if di < 0:
            continue
        p = hyper_prob(ai, bi, ci, di)
        if p <= p_obs + 1e-12:
            p_value += p

    return min(round(p_value, 6), 1.0)


# ── Main Analysis ───────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load data
    summary = load_csv(SUMMARY_CSV)
    char_data = load_csv(CHAR_CSV)

    # Build lookup from characterization: repo_link -> row
    char_lookup = {}
    for row in char_data:
        url = row["repo_link"].rstrip("/")
        char_lookup[url] = row

    # Merge: join summary with characterization
    merged = []
    unmatched = []
    for s in summary:
        url = s["repo_url"].rstrip("/")
        char = char_lookup.get(url)
        if char:
            m = {**s, **{f"char_{k}": char[k] for k in CATEGORY_COLS}}
            m["star_count"] = char.get("star_count", "")
            merged.append(m)
        else:
            unmatched.append(url)

    print(f"Merged: {len(merged)} repos matched, {len(unmatched)} unmatched")
    if unmatched:
        print(f"  Unmatched URLs (first 10): {unmatched[:10]}")

    # Save merged dataset
    merged_fields = list(merged[0].keys()) if merged else []
    write_csv(os.path.join(OUT_DIR, "merged_dataset.csv"), merged, merged_fields)

    # ── 1. Per-category success/fail breakdown ──────────────────────────────
    print("\n" + "=" * 70)
    print("PER-CATEGORY ANALYSIS")
    print("=" * 70)

    all_breakdowns = {}

    for col in CATEGORY_COLS:
        char_col = f"char_{col}"
        breakdown = defaultdict(lambda: {"SUCCESS": 0, "FAIL": 0, "NOT_FOUND": 0})

        for row in merged:
            val = row[char_col]
            result = row["overall_result"]
            breakdown[val][result] += 1

        # Sort by total count descending
        sorted_vals = sorted(breakdown.keys(), key=lambda v: -(breakdown[v]["SUCCESS"] + breakdown[v]["FAIL"] + breakdown[v]["NOT_FOUND"]))

        all_breakdowns[col] = dict(breakdown)

        # Print table
        print(f"\n{'─' * 60}")
        print(f"  {col.upper()}")
        print(f"{'─' * 60}")
        print(f"  {'Value':<35} {'Total':>5} {'OK':>4} {'Fail':>4} {'Rate':>7}")
        print(f"  {'─'*35} {'─'*5} {'─'*4} {'─'*4} {'─'*7}")

        rows_for_csv = []
        for val in sorted_vals:
            d = breakdown[val]
            total = d["SUCCESS"] + d["FAIL"]
            rate = safe_div(d["SUCCESS"], total) * 100 if total else 0
            print(f"  {val:<35} {total:>5} {d['SUCCESS']:>4} {d['FAIL']:>4} {rate:>6.1f}%")
            rows_for_csv.append({
                "category": col,
                "value": val,
                "total": total,
                "success": d["SUCCESS"],
                "fail": d["FAIL"],
                "success_rate": round(rate, 2),
            })

        # Save per-category CSV
        write_csv(
            os.path.join(OUT_DIR, f"breakdown_{col}.csv"),
            rows_for_csv,
            ["category", "value", "total", "success", "fail", "success_rate"],
        )

        # Chi-square test (exclude NOT_FOUND, exclude 'error' categories)
        test_table = {}
        for val in sorted_vals:
            if val == "error":
                continue
            d = breakdown[val]
            total = d["SUCCESS"] + d["FAIL"]
            if total > 0:
                test_table[val] = {"SUCCESS": d["SUCCESS"], "FAIL": d["FAIL"]}

        if len(test_table) >= 2:
            result = chi_square_test(test_table)
            sig = "***" if result["p_value"] < 0.001 else "**" if result["p_value"] < 0.01 else "*" if result["p_value"] < 0.05 else "n.s."
            print(f"\n  Chi-square: X²={result['chi2']}, df={result['dof']}, p={result['p_value']} {sig}")
            print(f"  Cramér's V: {result['cramers_v']}  (n={result['n']})")

    # ── 2. Statistical tests summary ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS SUMMARY")
    print("=" * 70)

    stat_rows = []
    for col in CATEGORY_COLS:
        char_col = f"char_{col}"
        test_table = {}
        for row in merged:
            val = row[char_col]
            if val == "error":
                continue
            result = row["overall_result"]
            if result in ("SUCCESS", "FAIL"):
                if val not in test_table:
                    test_table[val] = {"SUCCESS": 0, "FAIL": 0}
                test_table[val][result] += 1

        if len(test_table) >= 2:
            res = chi_square_test(test_table)
            sig = "***" if res["p_value"] < 0.001 else "**" if res["p_value"] < 0.01 else "*" if res["p_value"] < 0.05 else "n.s."
            print(f"  {col:<30} X²={res['chi2']:>8.2f}  df={res['dof']:>2}  p={res['p_value']:<10}  V={res['cramers_v']:<6}  {sig}")
            stat_rows.append({
                "dimension": col,
                "chi2": res["chi2"],
                "dof": res["dof"],
                "p_value": res["p_value"],
                "cramers_v": res["cramers_v"],
                "significance": sig,
                "n": res["n"],
            })

    write_csv(
        os.path.join(OUT_DIR, "statistical_tests.csv"),
        stat_rows,
        ["dimension", "chi2", "dof", "p_value", "cramers_v", "significance", "n"],
    )

    # ── 3. Pairwise Fisher exact tests for notable 2x2 comparisons ─────────
    print("\n" + "=" * 70)
    print("NOTABLE PAIRWISE COMPARISONS (Fisher's Exact Test)")
    print("=" * 70)

    # For each category, compare highest vs lowest success rate (with n>=5)
    fisher_rows = []
    for col in CATEGORY_COLS:
        char_col = f"char_{col}"
        val_counts = defaultdict(lambda: {"SUCCESS": 0, "FAIL": 0})
        for row in merged:
            val = row[char_col]
            if val == "error":
                continue
            result = row["overall_result"]
            if result in ("SUCCESS", "FAIL"):
                val_counts[val][result] += 1

        # Filter to values with n >= 5
        eligible = {v: c for v, c in val_counts.items() if c["SUCCESS"] + c["FAIL"] >= 5}
        if len(eligible) < 2:
            continue

        # Sort by success rate
        rated = sorted(
            eligible.items(),
            key=lambda x: safe_div(x[1]["SUCCESS"], x[1]["SUCCESS"] + x[1]["FAIL"]),
        )

        worst_val, worst = rated[0]
        best_val, best = rated[-1]

        worst_rate = safe_div(worst["SUCCESS"], worst["SUCCESS"] + worst["FAIL"]) * 100
        best_rate = safe_div(best["SUCCESS"], best["SUCCESS"] + best["FAIL"]) * 100

        # Fisher 2x2
        p = fisher_exact_2x2(best["SUCCESS"], best["FAIL"], worst["SUCCESS"], worst["FAIL"])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."

        print(f"\n  {col}:")
        print(f"    Best:  {best_val:<30} {best_rate:>5.1f}% ({best['SUCCESS']}/{best['SUCCESS']+best['FAIL']})")
        print(f"    Worst: {worst_val:<30} {worst_rate:>5.1f}% ({worst['SUCCESS']}/{worst['SUCCESS']+worst['FAIL']})")
        print(f"    Fisher p={p} {sig}")

        fisher_rows.append({
            "dimension": col,
            "best_value": best_val,
            "best_rate": round(best_rate, 2),
            "best_n": best["SUCCESS"] + best["FAIL"],
            "worst_value": worst_val,
            "worst_rate": round(worst_rate, 2),
            "worst_n": worst["SUCCESS"] + worst["FAIL"],
            "fisher_p": p,
            "significance": sig,
        })

    write_csv(
        os.path.join(OUT_DIR, "fisher_pairwise.csv"),
        fisher_rows,
        ["dimension", "best_value", "best_rate", "best_n", "worst_value", "worst_rate", "worst_n", "fisher_p", "significance"],
    )

    # ── 4. Iteration & token analysis by category ───────────────────────────
    print("\n" + "=" * 70)
    print("RESOURCE USAGE BY CATEGORY (successful repos only)")
    print("=" * 70)

    for col in CATEGORY_COLS:
        char_col = f"char_{col}"
        usage = defaultdict(lambda: {"iters": [], "steps": [], "prompt_tokens": [], "duration_ms": []})

        for row in merged:
            if row["overall_result"] != "SUCCESS":
                continue
            val = row[char_col]
            usage[val]["iters"].append(int(row.get("total_iterations", 0) or 0))
            usage[val]["steps"].append(int(row.get("total_steps", 0) or 0))
            usage[val]["prompt_tokens"].append(int(row.get("total_prompt_tokens", 0) or 0))
            usage[val]["duration_ms"].append(int(row.get("total_duration_ms", 0) or 0))

        sorted_vals = sorted(usage.keys(), key=lambda v: -len(usage[v]["iters"]))

        print(f"\n  {col.upper()} (successful repos)")
        print(f"  {'Value':<30} {'N':>4} {'AvgIter':>8} {'AvgStep':>8} {'AvgTok(K)':>10} {'AvgDur(m)':>10}")
        print(f"  {'─'*30} {'─'*4} {'─'*8} {'─'*8} {'─'*10} {'─'*10}")

        usage_rows = []
        for val in sorted_vals:
            d = usage[val]
            n = len(d["iters"])
            if n == 0:
                continue
            avg_iter = sum(d["iters"]) / n
            avg_step = sum(d["steps"]) / n
            avg_tok = sum(d["prompt_tokens"]) / n / 1000
            avg_dur = sum(d["duration_ms"]) / n / 60000

            print(f"  {val:<30} {n:>4} {avg_iter:>8.2f} {avg_step:>8.1f} {avg_tok:>10.1f} {avg_dur:>10.1f}")

            usage_rows.append({
                "category": col,
                "value": val,
                "n_success": n,
                "avg_iterations": round(avg_iter, 2),
                "avg_steps": round(avg_step, 1),
                "avg_prompt_tokens_k": round(avg_tok, 1),
                "avg_duration_min": round(avg_dur, 1),
            })

        write_csv(
            os.path.join(OUT_DIR, f"usage_{col}.csv"),
            usage_rows,
            ["category", "value", "n_success", "avg_iterations", "avg_steps", "avg_prompt_tokens_k", "avg_duration_min"],
        )

    # ── 5. Cross-tabulation: domain × build_type ────────────────────────────
    print("\n" + "=" * 70)
    print("CROSS-TABULATION: domain × build_type (success rates)")
    print("=" * 70)

    cross_tab = defaultdict(lambda: defaultdict(lambda: {"SUCCESS": 0, "FAIL": 0}))
    for row in merged:
        domain = row["char_domain"]
        bt = row["char_build_type"]
        result = row["overall_result"]
        if result in ("SUCCESS", "FAIL"):
            cross_tab[domain][bt][result] += 1

    cross_rows = []
    for domain in sorted(cross_tab.keys()):
        for bt in sorted(cross_tab[domain].keys()):
            d = cross_tab[domain][bt]
            total = d["SUCCESS"] + d["FAIL"]
            if total == 0:
                continue
            rate = d["SUCCESS"] / total * 100
            cross_rows.append({
                "domain": domain,
                "build_type": bt,
                "total": total,
                "success": d["SUCCESS"],
                "fail": d["FAIL"],
                "success_rate": round(rate, 2),
            })

    write_csv(
        os.path.join(OUT_DIR, "cross_domain_buildtype.csv"),
        cross_rows,
        ["domain", "build_type", "total", "success", "fail", "success_rate"],
    )

    # ── 6. Cross-tabulation: automation_level × tooling_complexity ──────────
    print("\n  (see cross_domain_buildtype.csv)")

    cross_tab2 = defaultdict(lambda: defaultdict(lambda: {"SUCCESS": 0, "FAIL": 0}))
    for row in merged:
        auto = row["char_automation_level"]
        tool = row["char_tooling_complexity"]
        result = row["overall_result"]
        if result in ("SUCCESS", "FAIL") and auto != "error" and tool != "error":
            cross_tab2[auto][tool][result] += 1

    cross_rows2 = []
    for auto in sorted(cross_tab2.keys()):
        for tool in sorted(cross_tab2[auto].keys()):
            d = cross_tab2[auto][tool]
            total = d["SUCCESS"] + d["FAIL"]
            if total == 0:
                continue
            rate = d["SUCCESS"] / total * 100
            cross_rows2.append({
                "automation_level": auto,
                "tooling_complexity": tool,
                "total": total,
                "success": d["SUCCESS"],
                "fail": d["FAIL"],
                "success_rate": round(rate, 2),
            })

    write_csv(
        os.path.join(OUT_DIR, "cross_automation_tooling.csv"),
        cross_rows2,
        ["automation_level", "tooling_complexity", "total", "success", "fail", "success_rate"],
    )

    # ── 7. Failure analysis: most common failure patterns ───────────────────
    print("\n" + "=" * 70)
    print("FAILURE PROFILE ANALYSIS")
    print("=" * 70)

    # For failed repos, what combination of characteristics is most common?
    fail_profiles = []
    for row in merged:
        if row["overall_result"] != "FAIL":
            continue
        profile = tuple(row[f"char_{c}"] for c in CATEGORY_COLS)
        fail_profiles.append(profile)

    # Count by each individual characteristic
    for i, col in enumerate(CATEGORY_COLS):
        fail_by_val = Counter(p[i] for p in fail_profiles)
        total_by_val = Counter(row[f"char_{col}"] for row in merged if row["overall_result"] in ("SUCCESS", "FAIL"))
        print(f"\n  {col}: failures / total")
        for val, fail_n in fail_by_val.most_common():
            total_n = total_by_val[val]
            fail_rate = fail_n / total_n * 100 if total_n else 0
            print(f"    {val:<35} {fail_n:>3} / {total_n:>3} ({fail_rate:>5.1f}% fail)")

    # ── 8. Build type grouped into families ─────────────────────────────────
    print("\n" + "=" * 70)
    print("BUILD TYPE FAMILIES (grouped)")
    print("=" * 70)

    BUILD_FAMILIES = {
        "JavaScript/Node": ["npm", "yarn", "pnpm", "bun", "bower", "grunt"],
        "Python": ["pip", "poetry", "conda", "setuptools", "pdm", "python"],
        "Java/JVM": ["maven", "gradle", "ant", "sbt", "leiningen"],
        "C/C++": ["cmake", "make", "meson", "scons", "waf", "gn", "bazel"],
        "Rust": ["cargo"],
        "Go": ["go-mod"],
        ".NET/C#": ["dotnet", "msbuild", "cake"],
        "Ruby": ["bundler", "gem", "rake", "jekyll"],
        "Swift/iOS": ["swiftpm", "xcode", "cocoapods", "CocoaPods", "carthage"],
        "Dart/Flutter": ["pub", "flutter"],
        "Docker": ["docker", "docker-compose"],
        "Haskell": ["cabal", "stack"],
        "Other": ["nix", "packer", "luarocks", "zig", "mix"],
        "None": ["none"],
    }

    # Invert: build_type -> family
    bt_to_family = {}
    for family, members in BUILD_FAMILIES.items():
        for m in members:
            bt_to_family[m] = family

    family_counts = defaultdict(lambda: {"SUCCESS": 0, "FAIL": 0})
    for row in merged:
        bt = row["char_build_type"]
        family = bt_to_family.get(bt, "Other")
        result = row["overall_result"]
        if result in ("SUCCESS", "FAIL"):
            family_counts[family][result] += 1

    sorted_families = sorted(family_counts.keys(), key=lambda f: -(family_counts[f]["SUCCESS"] + family_counts[f]["FAIL"]))

    print(f"\n  {'Family':<25} {'Total':>5} {'OK':>4} {'Fail':>4} {'Rate':>7}")
    print(f"  {'─'*25} {'─'*5} {'─'*4} {'─'*4} {'─'*7}")

    family_rows = []
    for fam in sorted_families:
        d = family_counts[fam]
        total = d["SUCCESS"] + d["FAIL"]
        rate = d["SUCCESS"] / total * 100 if total else 0
        print(f"  {fam:<25} {total:>5} {d['SUCCESS']:>4} {d['FAIL']:>4} {rate:>6.1f}%")
        family_rows.append({
            "family": fam,
            "total": total,
            "success": d["SUCCESS"],
            "fail": d["FAIL"],
            "success_rate": round(rate, 2),
        })

    write_csv(
        os.path.join(OUT_DIR, "build_type_families.csv"),
        family_rows,
        ["family", "total", "success", "fail", "success_rate"],
    )

    # Chi-square for families
    fam_table = {fam: family_counts[fam] for fam in sorted_families if family_counts[fam]["SUCCESS"] + family_counts[fam]["FAIL"] > 0}
    if len(fam_table) >= 2:
        res = chi_square_test(fam_table)
        sig = "***" if res["p_value"] < 0.001 else "**" if res["p_value"] < 0.01 else "*" if res["p_value"] < 0.05 else "n.s."
        print(f"\n  Chi-square: X²={res['chi2']}, df={res['dof']}, p={res['p_value']} {sig}, V={res['cramers_v']}")

    # ── 9. LaTeX tables ─────────────────────────────────────────────────────
    latex_path = os.path.join(OUT_DIR, "latex_tables.tex")
    with open(latex_path, "w") as f:
        f.write("% Auto-generated LaTeX tables from BuildAgent result analysis\n")
        f.write("% Generated by analyze_results.py\n\n")

        # Table 1: Overall results
        f.write("% ── Table: Overall Results ──\n")
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Overall Dockerfile Generation Results (N=282)}\n")
        f.write("\\label{tab:overall-results}\n")
        f.write("\\begin{tabular}{lr}\n\\toprule\n")
        f.write("Metric & Value \\\\\n\\midrule\n")
        total_found = len([r for r in merged if r["found_in_db"] == "True"])
        total_success = len([r for r in merged if r["overall_result"] == "SUCCESS"])
        total_fail = len([r for r in merged if r["overall_result"] == "FAIL"])
        f.write(f"Total repositories & {len(merged)} \\\\\n")
        f.write(f"Successful & {total_success} ({total_success/len(merged)*100:.1f}\\%) \\\\\n")
        f.write(f"Failed & {total_fail} ({total_fail/len(merged)*100:.1f}\\%) \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

        # Table 2: Per-category breakdown
        for col in CATEGORY_COLS:
            char_col = f"char_{col}"
            breakdown = defaultdict(lambda: {"SUCCESS": 0, "FAIL": 0})
            for row in merged:
                val = row[char_col]
                if val == "error":
                    continue
                result = row["overall_result"]
                if result in ("SUCCESS", "FAIL"):
                    breakdown[val][result] += 1

            sorted_vals = sorted(breakdown.keys(), key=lambda v: -(breakdown[v]["SUCCESS"] + breakdown[v]["FAIL"]))

            col_label = col.replace("_", " ").title()
            f.write(f"% ── Table: {col_label} ──\n")
            f.write("\\begin{table}[h]\n\\centering\n")
            f.write(f"\\caption{{Success Rate by {col_label}}}\n")
            f.write(f"\\label{{tab:{col}}}\n")
            f.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
            f.write(f"{col_label} & N & Success & Fail & Rate (\\%) \\\\\n\\midrule\n")

            for val in sorted_vals:
                d = breakdown[val]
                total = d["SUCCESS"] + d["FAIL"]
                if total == 0:
                    continue
                rate = d["SUCCESS"] / total * 100
                val_escaped = val.replace("_", "\\_").replace("#", "\\#")
                f.write(f"{val_escaped} & {total} & {d['SUCCESS']} & {d['FAIL']} & {rate:.1f} \\\\\n")

            f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

        # Table 3: Statistical tests
        f.write("% ── Table: Statistical Tests ──\n")
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Chi-Square Tests of Independence}\n")
        f.write("\\label{tab:chi-square}\n")
        f.write("\\begin{tabular}{lrrrrl}\n\\toprule\n")
        f.write("Dimension & $\\chi^2$ & df & $p$ & Cram\\'er's $V$ & Sig. \\\\\n\\midrule\n")
        for sr in stat_rows:
            dim = sr["dimension"].replace("_", "\\_")
            f.write(f"{dim} & {sr['chi2']:.2f} & {sr['dof']} & {sr['p_value']:.4f} & {sr['cramers_v']:.3f} & {sr['significance']} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

        # Table 4: Build type families
        f.write("% ── Table: Build Type Families ──\n")
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Success Rate by Build System Family}\n")
        f.write("\\label{tab:build-families}\n")
        f.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
        f.write("Family & N & Success & Fail & Rate (\\%) \\\\\n\\midrule\n")
        for fr in family_rows:
            fam = fr["family"].replace("#", "\\#")
            f.write(f"{fam} & {fr['total']} & {fr['success']} & {fr['fail']} & {fr['success_rate']:.1f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

    print(f"\n  LaTeX tables written to {latex_path}")

    # ── 10. Full summary JSON ───────────────────────────────────────────────
    full_summary = {
        "total_repos": len(merged),
        "success": total_success,
        "fail": total_fail,
        "success_rate_pct": round(total_success / len(merged) * 100, 2),
        "statistical_tests": stat_rows,
        "build_families": family_rows,
        "fisher_pairwise": fisher_rows,
    }

    with open(os.path.join(OUT_DIR, "analysis_summary.json"), "w") as f:
        json.dump(full_summary, f, indent=2)

    print("\n" + "=" * 70)
    print("ALL OUTPUTS SAVED")
    print("=" * 70)
    print(f"\nOutput directory: {OUT_DIR}/")
    print("Files:")
    for fname in sorted(os.listdir(OUT_DIR)):
        fpath = os.path.join(OUT_DIR, fname)
        size = os.path.getsize(fpath)
        print(f"  {fname:<45} {size:>8} bytes")


if __name__ == "__main__":
    main()
