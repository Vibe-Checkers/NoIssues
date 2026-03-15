#!/usr/bin/env bash
set -euo pipefail

INVENTORY_FILE="${INVENTORY_FILE:-$(dirname "$0")/vm_inventory.tsv}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/noissues-vms}"
RUN_STAMP_FILTER="${RUN_STAMP_FILTER:-}"
WATCH=0
INTERVAL=20

usage() {
  cat <<'EOF'
Usage:
  bash scripts/vm_monitor.sh [--watch] [--interval SEC] [--run-stamp STAMP]

Options:
  --watch              Refresh continuously
  --interval SEC       Refresh interval in seconds (default: 20)
  --run-stamp STAMP    Mark rows that do not match stamp as MISMATCH
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --watch) WATCH=1; shift ;;
    --interval) INTERVAL="${2:-}"; shift 2 ;;
    --run-stamp) RUN_STAMP_FILTER="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

[[ -f "$INVENTORY_FILE" ]] || { echo "Missing inventory file: $INVENTORY_FILE" >&2; exit 1; }
[[ -f "$SSH_KEY" ]] || { echo "Missing SSH key: $SSH_KEY" >&2; exit 1; }

print_once() {
  local ok=0 bad=0

  printf "%-16s %-15s %-7s %-14s %-8s %-8s %-9s %-9s %s\n" \
    "VM" "IP" "SSH" "RUN_ID" "MATCH" "PROC" "SUCCESS" "FAIL" "LAST_LOG"
  printf '%s\n' "---------------------------------------------------------------------------------------------------------------------"

  while IFS=$'\t' read -r vm_name vm_ip ssh_user slot remote_repos_file; do
    [[ -z "${vm_name:-}" || "${vm_name:0:1}" == "#" ]] && continue

    out="$(ssh -n -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$ssh_user@$vm_ip" '
set +e
RUN_ID=$(cat /home/azureuser/vmtest-run-id.txt 2>/dev/null || echo none)
if pgrep -f "src/parallel_empirical_test.py" >/dev/null 2>&1; then PROC=yes; else PROC=no; fi
if [ -f /home/azureuser/vmtest-run.log ]; then
  SUCCESS=$(grep -c "SUCCESS\|✅" /home/azureuser/vmtest-run.log 2>/dev/null || echo 0)
  FAIL=$(grep -c "FAILED\|FAILURE\|ERROR\|❌" /home/azureuser/vmtest-run.log 2>/dev/null || echo 0)
  LAST=$(tail -n 1 /home/azureuser/vmtest-run.log 2>/dev/null | tr "\t" " " | tr -d "\r" | cut -c1-70)
else
  SUCCESS=0
  FAIL=0
  LAST="-"
fi
echo "ok|$RUN_ID|$PROC|$SUCCESS|$FAIL|$LAST"
' 2>/dev/null || echo "ssh_fail|none|no|0|0|-")"

    state="${out%%|*}"; rest="${out#*|}"
    run_id="${rest%%|*}"; rest="${rest#*|}"
    proc="${rest%%|*}"; rest="${rest#*|}"
    success="${rest%%|*}"; rest="${rest#*|}"
    fail_count="${rest%%|*}"; last_log="${rest#*|}"

    match="ok"
    if [[ -n "$RUN_STAMP_FILTER" && "$run_id" != "$RUN_STAMP_FILTER" ]]; then
      match="MISMATCH"
    fi

    ssh_state="ok"
    if [[ "$state" != "ok" ]]; then
      ssh_state="fail"
    fi

    printf "%-16s %-15s %-7s %-14s %-8s %-8s %-9s %-9s %s\n" \
      "$vm_name" "$vm_ip" "$ssh_state" "$run_id" "$match" "$proc" "$success" "$fail_count" "$last_log"

    if [[ "$ssh_state" == "ok" && "$proc" == "yes" && "$match" != "MISMATCH" ]]; then
      ((ok+=1))
    else
      ((bad+=1))
    fi
  done < "$INVENTORY_FILE"

  echo
  echo "Healthy: $ok"
  echo "Unhealthy: $bad"
  [[ -n "$RUN_STAMP_FILTER" ]] && echo "Run stamp filter: $RUN_STAMP_FILTER"
}

if [[ "$WATCH" -eq 1 ]]; then
  while true; do
    clear || true
    echo "UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    print_once
    sleep "$INTERVAL"
  done
else
  print_once
fi
