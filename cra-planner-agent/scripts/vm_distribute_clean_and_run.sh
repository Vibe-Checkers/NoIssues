#!/usr/bin/env bash
set -euo pipefail

# Simple flow requested:
# 1) absolute clean VM
# 2) pull repo from GitHub
# 3) run scripts/setup_vm_and_run.sh on VM

INVENTORY_FILE="${INVENTORY_FILE:-$(dirname "$0")/vm_inventory.tsv}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/noissues-vms}"
REPO_URL="${REPO_URL:-https://github.com/Vibe-Checkers/NoIssues.git}"
REPO_BRANCH="${REPO_BRANCH:-vm-testing}"
WORKERS="${WORKERS:-1}"
PER_REPO_TIMEOUT_SECONDS="${PER_REPO_TIMEOUT_SECONDS:-7200}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
DEFAULT_REPOS_FILE="${DEFAULT_REPOS_FILE:-/Users/afikbae/Clones/capstone/NoIssues/cra-planner-agent/library_links.txt}"

[[ -f "$INVENTORY_FILE" ]] || { echo "Missing inventory file: $INVENTORY_FILE" >&2; exit 1; }
[[ -f "$SSH_KEY" ]] || { echo "Missing SSH key: $SSH_KEY" >&2; exit 1; }

ok=0
fail=0

while IFS=$'\t' read -r vm_name vm_ip ssh_user slot remote_repos_file; do
  [[ -z "${vm_name:-}" || "${vm_name:0:1}" == "#" ]] && continue

  echo "=== [$vm_name] $vm_ip ($slot) ==="

  # 1) copy and execute absolute cleaner
  if ! scp -q -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=15 \
      "$(dirname "$0")/vm_absolute_clean_remote.sh" \
      "$ssh_user@$vm_ip:/home/$ssh_user/vm_absolute_clean_remote.sh"; then
    echo "[ERROR] failed to copy cleaner to $vm_name"
    ((fail+=1)); echo; continue
  fi

  if ! ssh -n -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=20 "$ssh_user@$vm_ip" \
      "chmod +x /home/$ssh_user/vm_absolute_clean_remote.sh && /home/$ssh_user/vm_absolute_clean_remote.sh /home/$ssh_user"; then
    echo "[ERROR] absolute clean failed on $vm_name"
    ((fail+=1)); echo; continue
  fi

  # 2) install base packages + clone repo fresh
  if ! ssh -n -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=25 "$ssh_user@$vm_ip" \
      "bash -s -- '$ssh_user' '$REPO_BRANCH' '$REPO_URL'" <<'EOSSH'; then
set -euo pipefail
SSH_USER="$1"
REPO_BRANCH="$2"
REPO_URL="$3"
if command -v sudo >/dev/null 2>&1; then SUDO="sudo -n"; else SUDO=""; fi




$SUDO apt-get update -y
$SUDO apt-get install -y python3 python3-venv python3-pip git curl

rm -rf "/home/${SSH_USER}/NoIssues"
git clone --depth 1 --branch "$REPO_BRANCH" "$REPO_URL" "/home/${SSH_USER}/NoIssues"
EOSSH
    echo "[ERROR] clone/setup failed on $vm_name"
    ((fail+=1)); echo; continue
  fi

  # ensure repos file exists on VM (use local fallback if missing)
  if ! ssh -n -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=12 "$ssh_user@$vm_ip" \
      "test -f '$remote_repos_file'"; then
    if ! scp -q -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=20 \
        "$DEFAULT_REPOS_FILE" "$ssh_user@$vm_ip:$remote_repos_file"; then
      echo "[ERROR] failed to place repos file on $vm_name"
      ((fail+=1)); echo; continue
    fi
  fi

  # 3) run setup script that lives in the repo
  if ssh -n -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=20 "$ssh_user@$vm_ip" \
      "cd /home/$ssh_user/NoIssues/cra-planner-agent && chmod +x scripts/setup_vm_and_run.sh && WORKERS='$WORKERS' PER_REPO_TIMEOUT_SECONDS='$PER_REPO_TIMEOUT_SECONDS' RUN_STAMP='$RUN_STAMP' bash scripts/setup_vm_and_run.sh '$remote_repos_file'"; then
    echo "[OK] clean + run started on $vm_name"
    ((ok+=1))
  else
    echo "[ERROR] setup/run failed on $vm_name"
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
