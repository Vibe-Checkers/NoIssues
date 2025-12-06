#!/usr/bin/env python3
"""
Results Analysis Script
Analyzes empirical testing results and generates statistics.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter


def load_results(json_file: str) -> dict:
    """Load results from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)


def analyze_results(results_data: dict):
    """Analyze results and print statistics."""
    results = results_data['results']
    total = len(results)

    print("="*80)
    print("EMPIRICAL RESULTS ANALYSIS")
    print("="*80)
    print(f"\nTotal Repositories: {total}")

    # Success/Failure breakdown
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print(f"\nSuccess Rate: {len(successful)}/{total} ({len(successful)/total*100:.1f}%)")
    print(f"Failure Rate: {len(failed)}/{total} ({len(failed)/total*100:.1f}%)")

    # Language breakdown
    print("\n" + "="*80)
    print("LANGUAGE BREAKDOWN")
    print("="*80)

    lang_stats = defaultdict(lambda: {'total': 0, 'success': 0, 'failed': 0})

    for r in results:
        lang = r.get('detected_language', 'Unknown')
        lang_stats[lang]['total'] += 1
        if r['success']:
            lang_stats[lang]['success'] += 1
        else:
            lang_stats[lang]['failed'] += 1

    for lang, stats in sorted(lang_stats.items(), key=lambda x: x[1]['total'], reverse=True):
        success_rate = stats['success'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"\n{lang}:")
        print(f"  Total: {stats['total']}")
        print(f"  Success: {stats['success']} ({success_rate:.1f}%)")
        print(f"  Failed: {stats['failed']}")

    # Failure analysis
    if failed:
        print("\n" + "="*80)
        print("FAILURE ANALYSIS")
        print("="*80)

        # Failure stages
        stages = Counter()
        error_types = Counter()

        for r in failed:
            if 'dockerfile_test' in r:
                stage = r['dockerfile_test'].get('stage', 'UNKNOWN')
                error_type = r['dockerfile_test'].get('error_type', 'UNKNOWN')
                stages[stage] += 1
                error_types[error_type] += 1

        print("\nFailure Stages:")
        for stage, count in stages.most_common():
            print(f"  {stage:40s}: {count:3d} ({count/len(failed)*100:5.1f}%)")

        print("\nError Types:")
        for error_type, count in error_types.most_common():
            print(f"  {error_type:40s}: {count:3d} ({count/len(failed)*100:5.1f}%)")

    # Performance metrics
    print("\n" + "="*80)
    print("PERFORMANCE METRICS")
    print("="*80)

    durations = [r['total_duration_seconds'] for r in results if 'total_duration_seconds' in r]
    if durations:
        print(f"\nDuration Statistics:")
        print(f"  Average: {sum(durations)/len(durations):.2f} seconds")
        print(f"  Min: {min(durations):.2f} seconds")
        print(f"  Max: {max(durations):.2f} seconds")
        print(f"  Total: {sum(durations):.2f} seconds ({sum(durations)/60:.2f} minutes)")

    # Token usage
    token_totals = []
    token_inputs = []
    token_outputs = []

    for r in results:
        if 'agent_analysis' in r and 'token_usage' in r['agent_analysis']:
            usage = r['agent_analysis']['token_usage']
            token_totals.append(usage.get('total', 0))
            token_inputs.append(usage.get('input', 0))
            token_outputs.append(usage.get('output', 0))

    if token_totals:
        print(f"\nToken Usage Statistics:")
        print(f"  Average Total: {sum(token_totals)/len(token_totals):.0f} tokens")
        print(f"  Average Input: {sum(token_inputs)/len(token_inputs):.0f} tokens")
        print(f"  Average Output: {sum(token_outputs)/len(token_outputs):.0f} tokens")
        print(f"  Total Tokens: {sum(token_totals):,} tokens")
        print(f"  Min: {min(token_totals):,} tokens")
        print(f"  Max: {max(token_totals):,} tokens")

    # Docker build success rates
    print("\n" + "="*80)
    print("DOCKER BUILD SUCCESS RATES")
    print("="*80)

    dockerfile_generated = sum(1 for r in results if 'dockerfile_test' in r and r['dockerfile_test'].get('stage') != 'DOCKERFILE_GENERATION')
    dockerfile_built = sum(1 for r in results if 'dockerfile_test' in r and r['dockerfile_test'].get('success'))

    print(f"\nDockerfiles Generated: {dockerfile_generated}/{total} ({dockerfile_generated/total*100:.1f}%)")
    print(f"Dockerfiles Built Successfully: {dockerfile_built}/{total} ({dockerfile_built/total*100:.1f}%)")

    if dockerfile_generated > 0:
        print(f"Build Success Rate (of generated): {dockerfile_built}/{dockerfile_generated} ({dockerfile_built/dockerfile_generated*100:.1f}%)")

    # Top failures
    print("\n" + "="*80)
    print("FAILED REPOSITORIES")
    print("="*80)

    for r in failed[:10]:  # Show top 10 failures
        print(f"\n{r['repo_name']} ({r.get('detected_language', 'Unknown')})")
        print(f"  URL: {r['repo_url']}")
        if 'dockerfile_test' in r:
            dt = r['dockerfile_test']
            print(f"  Stage: {dt.get('stage', 'UNKNOWN')}")
            print(f"  Error: {dt.get('error_type', 'UNKNOWN')}")
            error_msg = dt.get('error_message', '')
            if error_msg:
                print(f"  Message: {error_msg[:100]}...")

    print("\n" + "="*80)


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <results_json_file>")
        print("\nExample:")
        print("  python analyze_results.py empirical_results/empirical_results_20231115_143022.json")
        sys.exit(1)

    json_file = sys.argv[1]

    if not Path(json_file).exists():
        print(f"[ERROR] Results file not found: {json_file}")
        sys.exit(1)

    results_data = load_results(json_file)
    analyze_results(results_data)


if __name__ == "__main__":
    main()
