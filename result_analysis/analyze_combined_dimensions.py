#!/usr/bin/env python3
"""
analyze_combined_dimensions.py -- Interaction effects between characterization dimensions.

Tests whether pairs of individually non-significant dimensions become significant
when combined, and identifies specific category combinations that predict
containerization outcome better than either dimension alone.

Reads:  result_analysis/analysis_output/merged_dataset.csv
Writes: result_analysis/analysis_output/combined_dimensions/
            combined_chi_square.csv          -- chi-square for all dimension pairs
            dep_repro_crosstab.csv           -- dep_transparency x reproducibility
            auto_dep_crosstab.csv            -- automation_level x dep_transparency
            repro_domain_crosstab.csv        -- reproducibility x domain
            fisher_contrasts.csv             -- Fisher exact tests on key contrasts
            combined_dimensions_latex.tex    -- LaTeX tables for the paper

Usage:
    python3 result_analysis/analyze_combined_dimensions.py
"""

import csv
import json
import math
import os
from collections import defaultdict
from itertools import combinations

# Paths

BASE_DIR = os.path.dirname(__file__)
MERGED_CSV = os.path.join(BASE_DIR, "analysis_output", "merged_dataset.csv")
OUT_DIR = os.path.join(BASE_DIR, "analysis_output", "combined_dimensions")

DIMS = [
    "char_automation_level",
    "char_environment_specificity",
    "char_dependency_transparency",
    "char_tooling_complexity",
    "char_reproducibility_support",
    "char_domain",
    "char_build_type",
]


def load_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: list[dict], fieldnames: list[str]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def safe_div(a, b):
    return a / b if b else 0.0


def chi_square_test(table: dict[str, dict[str, int]]) -> dict:
    """Chi-square test on {category: {SUCCESS: n, FAIL: m}}."""
    categories = sorted(table.keys())
    outcomes = ["SUCCESS", "FAIL"]

    observed = [[table[c].get(o, 0) for o in outcomes] for c in categories]
    n_rows = len(observed)
    n_cols = len(outcomes)
    n_total = sum(sum(row) for row in observed)

    if n_total == 0 or n_rows < 2:
        return {"chi2": 0, "dof": 0, "p_value": 1.0, "cramers_v": 0, "n": n_total}

    row_totals = [sum(row) for row in observed]
    col_totals = [sum(observed[r][c] for r in range(n_rows)) for c in range(n_cols)]

    chi2 = 0.0
    for r in range(n_rows):
        for c in range(n_cols):
            expected = row_totals[r] * col_totals[c] / n_total
            if expected > 0:
                chi2 += (observed[r][c] - expected) ** 2 / expected

    dof = (n_rows - 1) * (n_cols - 1)
    p_value = _chi2_p_value(chi2, dof)

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
    if dof <= 0 or chi2 <= 0:
        return 1.0
    k = dof / 2.0
    x = chi2 / 2.0
    if x > 200:
        return 0.0

    total = 0.0
    term = 1.0 / k
    total = term
    for n in range(1, 300):
        term *= x / (k + n)
        total += term
        if abs(term) < 1e-15:
            break

    p_lower = math.exp(-x + k * math.log(x) - math.lgamma(k)) * total
    return 1.0 - min(max(p_lower, 0.0), 1.0)


def fisher_exact_2x2(a, b, c, d) -> dict:
    """Fisher's exact test for 2x2 table [[a,b],[c,d]]. Returns p-value and odds ratio."""
    n = a + b + c + d

    def hyper_prob(a, b, c, d):
        return math.exp(
            math.lgamma(a + b + 1) + math.lgamma(c + d + 1)
            + math.lgamma(a + c + 1) + math.lgamma(b + d + 1)
            - math.lgamma(n + 1) - math.lgamma(a + 1)
            - math.lgamma(b + 1) - math.lgamma(c + 1) - math.lgamma(d + 1)
        )

    p_obs = hyper_prob(a, b, c, d)
    p_value = 0.0

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

    odds_ratio = (a * d) / (b * c) if b * c > 0 else float("inf")
    return {"p_value": min(round(p_value, 6), 1.0), "odds_ratio": round(odds_ratio, 4)}


def crosstab(rows, dim1, dim2, min_n=3):
    """Build success/fail counts for each (dim1_val, dim2_val) pair."""
    counts = defaultdict(lambda: {"SUCCESS": 0, "FAIL": 0})
    for r in rows:
        combo = f"{r[dim1]} + {r[dim2]}"
        result = "SUCCESS" if r["overall_result"] == "SUCCESS" else "FAIL"
        counts[combo][result] += 1

    out = []
    for combo, c in sorted(counts.items(), key=lambda x: -safe_div(x[1]["SUCCESS"], x[1]["SUCCESS"] + x[1]["FAIL"])):
        total = c["SUCCESS"] + c["FAIL"]
        if total >= min_n:
            out.append({
                "combination": combo,
                "success": c["SUCCESS"],
                "fail": c["FAIL"],
                "total": total,
                "rate": round(safe_div(c["SUCCESS"], total) * 100, 1),
            })
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    merged = load_csv(MERGED_CSV)
    print(f"Loaded {len(merged)} repos from merged dataset")

    # 1. Chi-square tests for all dimension pairs
    print("\n" + "=" * 70)
    print("CHI-SQUARE TESTS FOR ALL DIMENSION PAIRS")
    print("=" * 70)

    pair_results = []
    for d1, d2 in combinations(DIMS, 2):
        d1_short = d1.replace("char_", "")
        d2_short = d2.replace("char_", "")

        # Build contingency table for combined categories (min 3 per cell)
        combo_counts = defaultdict(int)
        for r in merged:
            combo = f"{r[d1]} | {r[d2]}"
            combo_counts[combo] += 1

        valid_combos = {k for k, v in combo_counts.items() if v >= 3}

        table = defaultdict(lambda: {"SUCCESS": 0, "FAIL": 0})
        n_used = 0
        for r in merged:
            combo = f"{r[d1]} | {r[d2]}"
            if combo in valid_combos:
                result = "SUCCESS" if r["overall_result"] == "SUCCESS" else "FAIL"
                table[combo][result] += 1
                n_used += 1

        if len(table) < 3 or n_used < 20:
            continue

        result = chi_square_test(dict(table))
        sig = "***" if result["p_value"] < 0.001 else "**" if result["p_value"] < 0.01 else "*" if result["p_value"] < 0.05 else "n.s."

        pair_results.append({
            "dim1": d1_short,
            "dim2": d2_short,
            "chi2": result["chi2"],
            "dof": result["dof"],
            "p_value": result["p_value"],
            "cramers_v": result["cramers_v"],
            "n_categories": len(table),
            "n_repos": n_used,
            "significance": sig,
        })

    pair_results.sort(key=lambda x: x["p_value"])

    print(f"\n  {'Dim1':<25} {'Dim2':<25} {'chi2':>8} {'dof':>4} {'p':>10} {'V':>6} {'Sig':>5}")
    print(f"  {'-'*25} {'-'*25} {'-'*8} {'-'*4} {'-'*10} {'-'*6} {'-'*5}")
    for r in pair_results:
        print(f"  {r['dim1']:<25} {r['dim2']:<25} {r['chi2']:>8.2f} {r['dof']:>4} {r['p_value']:>10.4f} {r['cramers_v']:>6.3f} {r['significance']:>5}")

    write_csv(
        os.path.join(OUT_DIR, "combined_chi_square.csv"),
        pair_results,
        ["dim1", "dim2", "chi2", "dof", "p_value", "cramers_v", "n_categories", "n_repos", "significance"],
    )

    # 2. Focused cross-tabulations for the three interesting pairs
    print("\n" + "=" * 70)
    print("DEPENDENCY TRANSPARENCY x REPRODUCIBILITY SUPPORT")
    print("=" * 70)

    ct1 = crosstab(merged, "char_dependency_transparency", "char_reproducibility_support")
    print(f"\n  {'Combination':<55} {'S':>4} {'F':>4} {'N':>4} {'Rate':>7}")
    print(f"  {'-'*55} {'-'*4} {'-'*4} {'-'*4} {'-'*7}")
    for r in ct1:
        print(f"  {r['combination']:<55} {r['success']:>4} {r['fail']:>4} {r['total']:>4} {r['rate']:>6.1f}%")
    write_csv(os.path.join(OUT_DIR, "dep_repro_crosstab.csv"), ct1,
              ["combination", "success", "fail", "total", "rate"])

    print("\n" + "=" * 70)
    print("AUTOMATION LEVEL x DEPENDENCY TRANSPARENCY")
    print("=" * 70)

    ct2 = crosstab(merged, "char_automation_level", "char_dependency_transparency")
    print(f"\n  {'Combination':<55} {'S':>4} {'F':>4} {'N':>4} {'Rate':>7}")
    print(f"  {'-'*55} {'-'*4} {'-'*4} {'-'*4} {'-'*7}")
    for r in ct2:
        print(f"  {r['combination']:<55} {r['success']:>4} {r['fail']:>4} {r['total']:>4} {r['rate']:>6.1f}%")
    write_csv(os.path.join(OUT_DIR, "auto_dep_crosstab.csv"), ct2,
              ["combination", "success", "fail", "total", "rate"])

    print("\n" + "=" * 70)
    print("REPRODUCIBILITY SUPPORT x DOMAIN")
    print("=" * 70)

    ct3 = crosstab(merged, "char_reproducibility_support", "char_domain")
    print(f"\n  {'Combination':<55} {'S':>4} {'F':>4} {'N':>4} {'Rate':>7}")
    print(f"  {'-'*55} {'-'*4} {'-'*4} {'-'*4} {'-'*7}")
    for r in ct3:
        print(f"  {r['combination']:<55} {r['success']:>4} {r['fail']:>4} {r['total']:>4} {r['rate']:>6.1f}%")
    write_csv(os.path.join(OUT_DIR, "repro_domain_crosstab.csv"), ct3,
              ["combination", "success", "fail", "total", "rate"])

    # 3. Fisher exact tests on key contrasts
    print("\n" + "=" * 70)
    print("FISHER EXACT TESTS ON KEY CONTRASTS")
    print("=" * 70)

    contrasts = []

    def run_contrast(label, group_a_filter, group_b_filter, label_a, label_b):
        ga = [r for r in merged if group_a_filter(r)]
        gb = [r for r in merged if group_b_filter(r)]
        sa = sum(1 for r in ga if r["overall_result"] == "SUCCESS")
        sb = sum(1 for r in gb if r["overall_result"] == "SUCCESS")
        fa = len(ga) - sa
        fb = len(gb) - sb
        if len(ga) < 3 or len(gb) < 3:
            return
        result = fisher_exact_2x2(sa, fa, sb, fb)
        rate_a = safe_div(sa, len(ga)) * 100
        rate_b = safe_div(sb, len(gb)) * 100
        print(f"\n  {label}")
        print(f"    {label_a}: {sa}/{len(ga)} = {rate_a:.1f}%")
        print(f"    {label_b}: {sb}/{len(gb)} = {rate_b:.1f}%")
        print(f"    OR = {result['odds_ratio']}, p = {result['p_value']}")
        contrasts.append({
            "contrast": label,
            "group_a": label_a,
            "group_b": label_b,
            "success_a": sa, "n_a": len(ga), "rate_a": round(rate_a, 1),
            "success_b": sb, "n_b": len(gb), "rate_b": round(rate_b, 1),
            "odds_ratio": result["odds_ratio"],
            "p_value": result["p_value"],
        })

    # Contrast 1: loose+CI vs pinned+CI
    run_contrast(
        "Loose+CI vs Pinned+CI",
        lambda r: r["char_dependency_transparency"] == "explicit_loose" and r["char_reproducibility_support"] == "repro_ready",
        lambda r: r["char_dependency_transparency"] == "explicit_machine_readable" and r["char_reproducibility_support"] == "repro_ready",
        "explicit_loose + repro_ready",
        "explicit_machine_readable + repro_ready",
    )

    # Contrast 2: loose+CI vs implicit+noCI
    run_contrast(
        "Loose+CI vs Implicit+NoCI",
        lambda r: r["char_dependency_transparency"] == "explicit_loose" and r["char_reproducibility_support"] == "repro_ready",
        lambda r: r["char_dependency_transparency"] == "implicit" and r["char_reproducibility_support"] == "no_ci_manual_only",
        "explicit_loose + repro_ready",
        "implicit + no_ci_manual_only",
    )

    # Contrast 3: semi_auto+loose vs manual+implicit
    run_contrast(
        "SemiAuto+Loose vs Manual+Implicit",
        lambda r: r["char_automation_level"] == "semi_automated" and r["char_dependency_transparency"] == "explicit_loose",
        lambda r: r["char_automation_level"] == "manual" and r["char_dependency_transparency"] == "implicit",
        "semi_automated + explicit_loose",
        "manual + implicit",
    )

    # Contrast 4: CI+data_science vs noCI+data_science
    run_contrast(
        "CI+DataScience vs NoCI+DataScience",
        lambda r: r["char_reproducibility_support"] == "repro_ready" and r["char_domain"] == "data-science",
        lambda r: r["char_reproducibility_support"] == "no_ci_manual_only" and r["char_domain"] == "data-science",
        "repro_ready + data-science",
        "no_ci_manual_only + data-science",
    )

    # Contrast 5: pinned+noCI vs loose+CI (the paradox from opposite angle)
    run_contrast(
        "Pinned+NoCI vs Loose+CI",
        lambda r: r["char_dependency_transparency"] == "explicit_machine_readable" and r["char_reproducibility_support"] == "no_ci_manual_only",
        lambda r: r["char_dependency_transparency"] == "explicit_loose" and r["char_reproducibility_support"] == "repro_ready",
        "explicit_machine_readable + no_ci_manual_only",
        "explicit_loose + repro_ready",
    )

    write_csv(os.path.join(OUT_DIR, "fisher_contrasts.csv"), contrasts,
              ["contrast", "group_a", "group_b", "success_a", "n_a", "rate_a",
               "success_b", "n_b", "rate_b", "odds_ratio", "p_value"])

    # 4. Summary JSON
    summary = {
        "total_repos": len(merged),
        "dimension_pairs_tested": len(pair_results),
        "significant_pairs_005": sum(1 for r in pair_results if r["p_value"] < 0.05),
        "top_3_pairs": [
            {"dims": f"{r['dim1']} x {r['dim2']}", "p": r["p_value"], "V": r["cramers_v"]}
            for r in pair_results[:3]
        ],
        "fisher_contrasts": contrasts,
    }

    with open(os.path.join(OUT_DIR, "combined_dimensions_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll outputs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
