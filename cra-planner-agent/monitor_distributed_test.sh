#!/bin/bash

# Monitoring script for distributed parallel_empirical_test across m0, m4, m5
# Single SSH call per machine, all 3 run in parallel for speed

get_status() {
  local host="$1"
  local dir="$2"
  local total="$3"
  local outfile="$4"
  ssh "$host" "
    running=\$(ps aux | grep 'parallel_empirical_test.py retry' | grep -v grep | wc -l)
    if [ \"\$running\" -gt 0 ]; then status=RUNNING; else status=STOPPED; fi
    latest_result=\$(ls -t $dir/results_*.jsonl 2>/dev/null | head -1)
    latest_progress=\$(ls -t $dir/progress_*.txt 2>/dev/null | head -1)
    if [ -n \"\$latest_progress\" ]; then
      pline=\$(tail -1 \"\$latest_progress\" 2>/dev/null)
      pnum=\$(echo \"\$pline\" | grep -oE '[0-9]+/$total' || echo '0/$total')
      plast=\$(echo \"\$pline\" | grep -oE '[✓✗⊘].*' | head -c 60 || echo 'starting...')
    else
      pnum='0/$total'
      plast='no progress file yet'
    fi
    if [ -n \"\$latest_result\" ]; then
      stats=\$(python3 -c \"
import json
rows = [json.loads(l) for l in open('\$latest_result')]
completed = len(rows)
successes = sum(1 for r in rows if r.get('success') is True)
skipped = sum(1 for r in rows if r.get('skip_reason'))
types = {}
for r in rows:
    if r.get('success') or r.get('skip_reason'):
        continue
    err = str(r.get('agent_analysis', {}).get('lessons_learned', ''))
    if 'RateLimitReached' in err or 'rate limit' in err.lower():
        t = 'RATE_LIMIT'
    else:
        t = r.get('dockerfile_test', {}).get('final_result', {}).get('stage', 'AGENT_FAILURE')
    types[t] = types.get(t, 0) + 1
rate_limits = types.pop('RATE_LIMIT', 0)
failure_str = ','.join(f'{k}:{v}' for k,v in sorted(types.items()))
print(f'{completed}|{successes}|{skipped}|{rate_limits}|{failure_str}')
\" 2>/dev/null || echo '0|0|0|0|')
      completed=\$(echo \"\$stats\" | cut -d'|' -f1)
      successes=\$(echo \"\$stats\" | cut -d'|' -f2)
      rate_limits=\$(echo \"\$stats\" | cut -d'|' -f3)
      agent_failures=\$(echo \"\$stats\" | cut -d'|' -f4)
    else
      completed=0
      successes=0
      rate_limits=0
      agent_failures=''
    fi
    disk=\$(df -h / | tail -1 | awk '{print \$5}')
    echo \"\${status}|\${pnum}|\${completed}|\${successes}|\${rate_limits}|\${agent_failures}|\${disk}|\${plast}\"
  " > "$outfile" 2>/dev/null &
}

while true; do
  clear
  echo "=== Distributed Parallel Empirical Test Monitor (Round 5 — 2 workers) ==="
  echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
  echo ""

  tmpdir=$(mktemp -d)
  get_status "m0" "~/noissues/cra-planner-agent/parallel_empirical_results" 69 "$tmpdir/m0"
  get_status "m4" "~/NoIssues/cra-planner-agent/parallel_empirical_results" 69 "$tmpdir/m4"
  get_status "m5" "~/NoIssues/cra-planner-agent/parallel_empirical_results" 68 "$tmpdir/m5"
  wait

  total_completed=0
  total_success=0
  total_ratelimit=0
  for machine in m0 m4 m5; do
    result=$(cat "$tmpdir/$machine" 2>/dev/null)
    if [ -z "$result" ]; then
      echo "[$machine]  Status: ✗ Unreachable"
      echo ""
      continue
    fi
    IFS='|' read -r status progress completed successes skipped rate_limits agent_failures disk last_status <<< "$result"
    total_completed=$((total_completed + completed))
    total_success=$((total_success + successes))
    total_ratelimit=$((total_ratelimit + rate_limits))
    fails=$((completed - successes - skipped))
    if [ "$status" = "RUNNING" ]; then
      echo "[$machine]  Progress: $progress | Done: $completed | ✓ $successes | ✗ $fails | ⊘ $skipped | 🚦 $rate_limits | Disk: $disk | ✓ Running"
    else
      echo "[$machine]  Progress: $progress | Done: $completed | ✓ $successes | ✗ $fails | ⊘ $skipped | 🚦 $rate_limits | Disk: $disk | ✗ Stopped"
    fi
    if [ -n "$agent_failures" ]; then
      echo "  Failures: $agent_failures"
    fi
    echo "  Last: $last_status"
    echo ""
  done

  rm -rf "$tmpdir"

  total_fail=$((total_completed - total_success))
  echo "=== TOTAL: $total_completed / 206 done | ✓ $total_success success | ✗ $total_fail fail | 🚦 $total_ratelimit rate-limited ==="
  echo ""
  echo "Press Ctrl+C to exit. Refreshing in 20 seconds..."
  sleep 20
done
