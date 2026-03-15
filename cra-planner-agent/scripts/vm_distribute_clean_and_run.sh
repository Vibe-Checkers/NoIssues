#!/usr/bin/env bash
set -euo pipefail

# Orchestrates:
# 1) distribute remote cleaner
# 2) execute absolute clean (hard wipe under /home/azureuser except .ssh)
# 3) bootstrap repo + deps
# 4) start run

INVENTORY_FILE="${INVENTORY_FILE:-$(dirname "$0")/vm_inventory.tsv}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/noissues-vms}"
LOCAL_ROOT="${LOCAL_ROOT:-/Users/afikbae/Clones/capstone/NoIssues/cra-planner-agent}"
REPO_URL="${REPO_URL:-https://github.com/Vibe-Checkers/NoIssues.git}"
REPO_BRANCH="${REPO_BRANCH:-vm-testing}"
WORKERS="${WORKERS:-1}"
PER_REPO_TIMEOUT_SECONDS="${PER_REPO_TIMEOUT_SECONDS:-7200}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

[[ -f "$INVENTORY_FILE" ]] || { echo "Missing inventory file: $INVENTORY_FILE" >&2; exit 1; }
[[ -f "$SSH_KEY" ]] || { echo "Missing SSH key: $SSH_KEY" >&2; exit 1; }

ok=0
fail=0

while IFS=$'\t' read -r vm_name vm_ip ssh_user slot remote_repos_file; do
  [[ -z "${vm_name:-}" || "${vm_name:0:1}" == "#" ]] && continue

  echo "=== [$vm_name] $vm_ip ($slot) ==="

  # 1) place cleaner
  if ! scp -q -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=15 \
      "$(dirname "$0")/vm_absolute_clean_remote.sh" \
      "$ssh_user@$vm_ip:/home/$ssh_user/vm_absolute_clean_remote.sh"; then
    echo "[ERROR] failed to copy cleaner to $vm_name"
    ((fail+=1)); echo; continue
  fi

  # 2) run absolute clean remotely
  if ! ssh -n -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=15 "$ssh_user@$vm_ip" \
      "chmod +x /home/$ssh_user/vm_absolute_clean_remote.sh && /home/$ssh_user/vm_absolute_clean_remote.sh /home/$ssh_user"; then
    echo "[ERROR] absolute clean failed on $vm_name"
    ((fail+=1)); echo; continue
  fi

  # 3) bootstrap remote env and clone fresh
  if ! ssh -n -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=20 "$ssh_user@$vm_ip" "bash -s" <<'EOSSH'; then
set -euo pipefail
if command -v sudo >/dev/null 2>&1; then SUDO="sudo -n"; else SUDO=""; fi
$SUDO apt-get update -y
$SUDO apt-get install -y python3 python3-venv python3-pip git curl
EOSSH
    echo "[ERROR] bootstrap packages failed on $vm_name"
    ((fail+=1)); echo; continue
  fi

  remote_repo_root="/home/$ssh_user/NoIssues"
  remote_repo="$remote_repo_root/cra-planner-agent"
  if ! ssh -n -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=20 "$ssh_user@$vm_ip" \
      "git clone --depth 1 --branch '$REPO_BRANCH' '$REPO_URL' '$remote_repo_root'"; then
    echo "[ERROR] clone failed on $vm_name"
    ((fail+=1)); echo; continue
  fi

  # 4) overlay local patched files
  if ! ssh -n -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=15 "$ssh_user@$vm_ip" \
      "mkdir -p '$remote_repo/src/agent'"; then
    echo "[ERROR] mkdir src/agent failed on $vm_name"
    ((fail+=1)); echo; continue
  fi

  if ! scp -q -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=20 \
      "$LOCAL_ROOT/.env" \
      "$LOCAL_ROOT/requirements.txt" \
      "$LOCAL_ROOT/src/parallel_empirical_test.py" \
      "$ssh_user@$vm_ip:$remote_repo/"; then
    echo "[ERROR] top-level overlay failed on $vm_name"
    ((fail+=1)); echo; continue
  fi

  if ! scp -q -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=20 \
      "$LOCAL_ROOT/src/agent/tools.py" \
      "$LOCAL_ROOT/src/agent/workflow.py" \
      "$ssh_user@$vm_ip:$remote_repo/src/agent/"; then
    echo "[ERROR] agent overlay failed on $vm_name"
    ((fail+=1)); echo; continue
  fi

  # 5) create venv, install deps, launch run
  if ssh -n -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=20 "$ssh_user@$vm_ip" \
      "bash -s -- '$remote_repo' '$remote_repos_file' '$WORKERS' '$PER_REPO_TIMEOUT_SECONDS' '$RUN_STAMP' '$ssh_user'" <<'EOSSH'; then
set -euo pipefail
REMOTE_REPO="$1"
REMOTE_REPOS_FILE="$2"
WORKERS="$3"
PER_REPO_TIMEOUT_SECONDS="$4"
RUN_STAMP="$5"
SSH_USER="$6"

cd "$REMOTE_REPO"
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

STAMPED_REPOS_FILE="/home/${SSH_USER}/vmtest-repos-${RUN_STAMP}.txt"
cp "$REMOTE_REPOS_FILE" "$STAMPED_REPOS_FILE"

pkill -f 'parallel_empirical_test.py' || true
echo "$RUN_STAMP" > "/home/${SSH_USER}/vmtest-run-id.txt"
echo "=== RUN_STAMP: $RUN_STAMP ===" > "/home/${SSH_USER}/vmtest-run.log"
echo "Started: $(date -u)" >> "/home/${SSH_USER}/vmtest-run.log"

nohup env PER_REPO_TIMEOUT_SECONDS="$PER_REPO_TIMEOUT_SECONDS" \
  "$REMOTE_REPO/.venv/bin/python" "$REMOTE_REPO/src/parallel_empirical_test.py" "$STAMPED_REPOS_FILE" --workers "$WORKERS" \
  >> "/home/${SSH_USER}/vmtest-run.log" 2>&1 < /dev/null &

sleep 2
pgrep -f "parallel_empirical_test.py $STAMPED_REPOS_FILE --workers $WORKERS" >/dev/null
EOSSH
    echo "[OK] clean + run started on $vm_name"
    ((ok+=1))
  else
    echo "[ERROR] final launch failed on $vm_name"
    ((fail+=1))
  fi

  echo
done < "$INVENTORY_FILE"

echo "=================================================="
echo "Run stamp: $RUN_STAMP"
echo "VMs OK:    $ok"
echo "VMs FAIL:  $fail"
echo "=================================================="

if [[ $fail -gt 0 ]]; then
  exit 2
fi
