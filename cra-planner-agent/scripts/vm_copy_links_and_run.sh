#!/usr/bin/env bash
set -euo pipefail

# Script 2:
# - connects to each VM
# - copies links txt to VM target path
# - starts run

INVENTORY_FILE="${INVENTORY_FILE:-$(dirname "$0")/vm_inventory.tsv}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/noissues-vms}"
LOCAL_LINKS_FILE="${LOCAL_LINKS_FILE:-/Users/afikbae/Clones/capstone/NoIssues/cra-planner-agent/library_links.txt}"
LOCAL_ENV_FILE="${LOCAL_ENV_FILE:-/Users/afikbae/Clones/capstone/NoIssues/cra-planner-agent/.env}"
WORKERS="${WORKERS:-1}"
PER_REPO_TIMEOUT_SECONDS="${PER_REPO_TIMEOUT_SECONDS:-7200}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

count_non_comment_lines() {
  awk 'NF && $1 !~ /^#/' "$1" | wc -l | tr -d ' '
}

[[ -f "$INVENTORY_FILE" ]] || { echo "Missing inventory file: $INVENTORY_FILE" >&2; exit 1; }
[[ -f "$SSH_KEY" ]] || { echo "Missing SSH key: $SSH_KEY" >&2; exit 1; }
[[ -f "$LOCAL_LINKS_FILE" ]] || { echo "Missing local links file: $LOCAL_LINKS_FILE" >&2; exit 1; }
[[ -f "$LOCAL_ENV_FILE" ]] || { echo "Missing local env file: $LOCAL_ENV_FILE" >&2; exit 1; }

LOCAL_REPO_COUNT="$(count_non_comment_lines "$LOCAL_LINKS_FILE")"
if [[ "$LOCAL_REPO_COUNT" -eq 0 ]]; then
  echo "Links file has no repositories: $LOCAL_LINKS_FILE" >&2
  exit 1
fi

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

  AGENT_DIR="/home/${ssh_user}/NoIssues/cra-planner-agent"
  if ! scp -q -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=15 \
      "$LOCAL_ENV_FILE" "$ssh_user@$vm_ip:$AGENT_DIR/.env"; then
    echo "[ERROR] failed to copy .env to $vm_name"
    ((fail+=1)); echo; continue
  fi

  if ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=20 "$ssh_user@$vm_ip" \
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
rm -f "/home/${SSH_USER}/vmtest-run.pid"

if [[ ! -f "$REPOS_FILE" ]]; then
  echo "Missing repos file on VM: $REPOS_FILE" >&2
  exit 31
fi
if [[ ! -f "$AGENT_DIR/.env" ]]; then
  echo "Missing .env on VM: $AGENT_DIR/.env" >&2
  exit 32
fi

VM_REPO_COUNT=$(awk 'NF && $1 !~ /^#/' "$REPOS_FILE" | wc -l | tr -d ' ')
if [[ "$VM_REPO_COUNT" -eq 0 ]]; then
  echo "Repos file is empty on VM: $REPOS_FILE" >&2
  exit 33
fi

set -a
. "$AGENT_DIR/.env"
set +a

for required_var in AZURE_OPENAI_API_KEY AZURE_OPENAI_ENDPOINT AZURE_OPENAI_DEPLOYMENT; do
  if [[ -z "${!required_var:-}" ]]; then
    echo "Missing required env var in .env: $required_var" >&2
    exit 34
  fi
done

echo "$RUN_STAMP" > "/home/${SSH_USER}/vmtest-run-id.txt"
echo "=== RUN_STAMP: $RUN_STAMP ===" > "/home/${SSH_USER}/vmtest-run.log"
echo "Started: $(date -u)" >> "/home/${SSH_USER}/vmtest-run.log"
echo "Repos count: $VM_REPO_COUNT" >> "/home/${SSH_USER}/vmtest-run.log"

nohup env PER_REPO_TIMEOUT_SECONDS="$PER_REPO_TIMEOUT_SECONDS" \
  "$AGENT_DIR/.venv/bin/python" "$AGENT_DIR/src/parallel_empirical_test.py" "$REPOS_FILE" --workers "$WORKERS" \
  >> "/home/${SSH_USER}/vmtest-run.log" 2>&1 < /dev/null &

echo "$!" > "/home/${SSH_USER}/vmtest-run.pid"

sleep 2
if [[ ! -s "/home/${SSH_USER}/vmtest-run.pid" ]]; then
  echo "Failed to write run pid" >&2
  exit 35
fi
RUN_PID="$(cat /home/${SSH_USER}/vmtest-run.pid)"
if ! kill -0 "$RUN_PID" 2>/dev/null; then
  echo "Run process is not active (pid=$RUN_PID)" >&2
  exit 36
fi

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
