#!/usr/bin/env bash
set -euo pipefail

# Script 2:
# - connects to each VM
# - copies links txt to VM target path
# - starts run

INVENTORY_FILE="${INVENTORY_FILE:-$(dirname "$0")/vm_inventory.tsv}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/noissues-vms}"
LOCAL_LINKS_FILE="${LOCAL_LINKS_FILE:-/Users/afikbae/Clones/capstone/NoIssues/cra-planner-agent/library_links.txt}"
WORKERS="${WORKERS:-1}"
PER_REPO_TIMEOUT_SECONDS="${PER_REPO_TIMEOUT_SECONDS:-7200}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

[[ -f "$INVENTORY_FILE" ]] || { echo "Missing inventory file: $INVENTORY_FILE" >&2; exit 1; }
[[ -f "$SSH_KEY" ]] || { echo "Missing SSH key: $SSH_KEY" >&2; exit 1; }
[[ -f "$LOCAL_LINKS_FILE" ]] || { echo "Missing local links file: $LOCAL_LINKS_FILE" >&2; exit 1; }

ok=0
fail=0

while IFS=$'\t' read -r vm_name vm_ip ssh_user slot remote_repos_file; do
  [[ -z "${vm_name:-}" || "${vm_name:0:1}" == "#" ]] && continue
  echo "=== [$vm_name] $vm_ip ($slot) ==="

  if ! scp -q -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=15 \
      "$LOCAL_LINKS_FILE" "$ssh_user@$vm_ip:$remote_repos_file"; then
    echo "[ERROR] failed to copy links file to $vm_name"
    ((fail+=1)); echo; continue
  fi

  if ssh -n -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=20 "$ssh_user@$vm_ip" \
    "bash -s -- '$ssh_user' '$remote_repos_file' '$WORKERS' '$PER_REPO_TIMEOUT_SECONDS' '$RUN_STAMP'" <<'EOSSH'; then
set -euo pipefail
SSH_USER="$1"
REPOS_FILE="$2"
WORKERS="$3"
PER_REPO_TIMEOUT_SECONDS="$4"
RUN_STAMP="$5"

AGENT_DIR="/home/${SSH_USER}/NoIssues/cra-planner-agent"

cd "$AGENT_DIR"
pkill -f 'parallel_empirical_test.py' || true

echo "$RUN_STAMP" > "/home/${SSH_USER}/vmtest-run-id.txt"
echo "=== RUN_STAMP: $RUN_STAMP ===" > "/home/${SSH_USER}/vmtest-run.log"
echo "Started: $(date -u)" >> "/home/${SSH_USER}/vmtest-run.log"

nohup env PER_REPO_TIMEOUT_SECONDS="$PER_REPO_TIMEOUT_SECONDS" \
  "$AGENT_DIR/.venv/bin/python" "$AGENT_DIR/src/parallel_empirical_test.py" "$REPOS_FILE" --workers "$WORKERS" \
  >> "/home/${SSH_USER}/vmtest-run.log" 2>&1 < /dev/null &

sleep 2
pgrep -f "parallel_empirical_test.py $REPOS_FILE --workers $WORKERS" >/dev/null

echo "STARTED"
EOSSH
    echo "[OK] started run on $vm_name"
    ((ok+=1))
  else
    echo "[ERROR] failed to start run on $vm_name"
    ((fail+=1))
  fi

  echo
done < "$INVENTORY_FILE"

echo "Run stamp: $RUN_STAMP"
echo "Started OK: $ok"
echo "Started FAIL: $fail"
[[ $fail -eq 0 ]]

