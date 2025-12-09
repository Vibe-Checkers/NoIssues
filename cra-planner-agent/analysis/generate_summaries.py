#!/usr/bin/env python3
"""
Generate empirical summary files from existing JSON results.
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def generate_summary_from_json(json_file_path: str):
    """Generate a summary file from an empirical results JSON file."""

    json_path = Path(json_file_path)

    if not json_path.exists():
        print(f"[ERROR] JSON file not found: {json_file_path}")
        return False

    # Read JSON data
    print(f"[INFO] Reading {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    timestamp = data.get("timestamp", "unknown")
    total = data.get("total_tested", 0)
    results = data.get("results", [])

    if total == 0:
        print(f"[WARNING] No results found in {json_path}")
        return False

    # Calculate statistics
    successful = sum(1 for r in results if r.get("success", False))
    failed = total - successful

    # Categorize failures
    failure_stages = {}
    failure_types = {}

    for result in results:
        if not result.get("success", False):
            if "dockerfile_test" in result:
                stage = result["dockerfile_test"].get("stage", "UNKNOWN")
                error_type = result["dockerfile_test"].get("error_type", "UNKNOWN")

                failure_stages[stage] = failure_stages.get(stage, 0) + 1
                failure_types[error_type] = failure_types.get(error_type, 0) + 1

    # Calculate durations and tokens
    durations = [r.get("total_duration_seconds", 0) for r in results]
    avg_duration = sum(durations) / len(durations) if durations else 0

    token_totals = []
    for r in results:
        if "agent_analysis" in r and "token_usage" in r["agent_analysis"]:
            token_totals.append(r["agent_analysis"]["token_usage"].get("total", 0))
    avg_tokens = sum(token_totals) / len(token_totals) if token_totals else 0

    # Create summary file path
    summary_path = json_path.parent / f"empirical_summary_{timestamp}.txt"

    # Write summary file
    print(f"[INFO] Generating summary: {summary_path}")

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("EMPIRICAL TESTING SUMMARY\n")
        f.write("="*80 + "\n\n")

        f.write(f"Test Run: {timestamp}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Generated from: {json_path.name}\n\n")

        f.write("OVERALL RESULTS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Repositories Tested: {total}\n")
        f.write(f"Successful: {successful} ({successful/total*100:.1f}%)\n")
        f.write(f"Failed: {failed} ({failed/total*100:.1f}%)\n\n")

        f.write("PERFORMANCE METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Average Duration per Repository: {avg_duration:.2f} seconds\n")
        f.write(f"Average Token Usage per Repository: {avg_tokens:.0f} tokens\n")
        f.write(f"Total Duration: {sum(durations):.2f} seconds ({sum(durations)/60:.2f} minutes)\n\n")

        if failure_stages:
            f.write("FAILURE BREAKDOWN BY STAGE\n")
            f.write("-"*80 + "\n")
            for stage, count in sorted(failure_stages.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{stage:40s}: {count:3d} ({count/failed*100:.1f}%)\n")
            f.write("\n")

        if failure_types:
            f.write("FAILURE BREAKDOWN BY ERROR TYPE\n")
            f.write("-"*80 + "\n")
            for error_type, count in sorted(failure_types.items(), key=lambda x: x[1], reverse=True):
                if error_type != "UNKNOWN":  # Skip None values
                    f.write(f"{error_type:40s}: {count:3d} ({count/failed*100:.1f}%)\n")
            f.write("\n")

        f.write("DETAILED RESULTS\n")
        f.write("-"*80 + "\n")
        for i, result in enumerate(results, 1):
            f.write(f"\n{i}. {result.get('repo_name', 'Unknown')}\n")
            f.write(f"   URL: {result.get('repo_url', 'Unknown')}\n")
            f.write(f"   Success: {'YES' if result.get('success') else 'NO'}\n")
            f.write(f"   Language: {result.get('detected_language', 'Unknown')}\n")
            f.write(f"   Duration: {result.get('total_duration_seconds', 0):.2f}s\n")

            if not result.get("success") and "dockerfile_test" in result:
                docker = result["dockerfile_test"]
                f.write(f"   Failure Stage: {docker.get('stage', 'UNKNOWN')}\n")
                error_type = docker.get('error_type', 'UNKNOWN')
                if error_type and error_type != "UNKNOWN":
                    f.write(f"   Error Type: {error_type}\n")

        f.write("\n" + "="*80 + "\n")

    print(f"[OK] Summary saved to: {summary_path}")

    # Print summary to console
    print(f"\n{'='*80}")
    print(f"SUMMARY FOR: {json_path.name}")
    print(f"{'='*80}")
    print(f"Total Tested: {total}")
    print(f"Successful: {successful} ({successful/total*100:.1f}%)")
    print(f"Failed: {failed} ({failed/total*100:.1f}%)")
    print(f"Average Duration: {avg_duration:.2f} seconds")
    print(f"Average Tokens: {avg_tokens:.0f}")
    print(f"{'='*80}\n")

    return True


def main():
    """Main function."""
    print("="*80)
    print("EMPIRICAL SUMMARY GENERATOR")
    print("="*80)
    print()

    if len(sys.argv) < 2:
        print("Usage: python generate_summaries.py <json_file1> [json_file2] ...")
        print()
        print("Example:")
        print("  python generate_summaries.py empirical_results_20251201_004624.json")
        sys.exit(1)

    json_files = sys.argv[1:]

    print(f"[INFO] Processing {len(json_files)} JSON file(s)\n")

    success_count = 0
    for json_file in json_files:
        if generate_summary_from_json(json_file):
            success_count += 1
        print()

    print(f"\n[DONE] Successfully generated {success_count}/{len(json_files)} summary file(s)")


if __name__ == "__main__":
    main()
