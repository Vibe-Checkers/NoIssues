#!/usr/bin/env bash
set -euo pipefail

# Script 1:
# - connects to each VM
# - cleans previous run state
# - clones/pulls repo
# - checks out target branch

INVENTORY_FILE="${INVENTORY_FILE:-$(dirname "$0")/vm_inventory.tsv}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/noissues-vms}"
REPO_URL="${REPO_URL:-https://github.com/Vibe-Checkers/NoIssues.git}"
BRANCH="${BRANCH:-vm-testing}"

[[ -f "$INVENTORY_FILE" ]] || { echo "Missing inventory file: $INVENTORY_FILE" >&2; exit 1; }
[[ -f "$SSH_KEY" ]] || { echo "Missing SSH key: $SSH_KEY" >&2; exit 1; }

ok=0
fail=0

while IFS=$'\t' read -r vm_name vm_ip ssh_user slot remote_repos_file; do
  [[ -z "${vm_name:-}" || "${vm_name:0:1}" == "#" ]] && continue
  echo "=== [$vm_name] $vm_ip ($slot) ==="

  if ssh -n -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=20 "$ssh_user@$vm_ip" \
    "bash -s -- '$ssh_user' '$REPO_URL' '$BRANCH'" <<'EOSSH'; then
set -euo pipefail
SSH_USER="$1"
REPO_URL="$2"
BRANCH="$3"

if command -v sudo >/dev/null 2>&1; then SUDO="sudo -n"; else SUDO=""; fi

# clean processes + logs
pkill -f 'parallel_empirical_test.py' || true
rm -f "/home/${SSH_USER}/vmtest-run.log" "/home/${SSH_USER}/vmtest-run-id.txt"

# lightweight docker clean
docker rm -f $(docker ps -aq) 2>/dev/null || true
docker system prune -af --volumes || true

# clean repo workspace
rm -rf "/home/${SSH_USER}/NoIssues"

# install base deps
$SUDO apt-get update -y
$SUDO apt-get install -y git python3 python3-venv python3-pip

# clone and checkout branch
git clone "$REPO_URL" "/home/${SSH_USER}/NoIssues"
git -C "/home/${SSH_USER}/NoIssues" fetch --all --prune
git -C "/home/${SSH_USER}/NoIssues" checkout "$BRANCH"
git -C "/home/${SSH_USER}/NoIssues" reset --hard "origin/$BRANCH"

# venv + deps for cra-planner-agent
AGENT_DIR="/home/${SSH_USER}/NoIssues/cra-planner-agent"
[ -d "$AGENT_DIR" ] || { echo "Missing agent dir: $AGENT_DIR" >&2; exit 21; }
cd "$AGENT_DIR"
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
[ -x "$AGENT_DIR/.venv/bin/python" ] || { echo "Missing venv python" >&2; exit 22; }

echo "READY"
EOSSH
    echo "[OK] prepared $vm_name"
    ((ok+=1))
  else
    echo "[ERROR] failed $vm_name"
    ((fail+=1))
  fi
  echo
done < "$INVENTORY_FILE"

echo "Prepared OK: $ok"
echo "Prepared FAIL: $fail"
[[ $fail -eq 0 ]]
