#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/setup_vm_and_run.sh [repos_file]
#
# Env overrides:
#   WORKERS=1
#   PER_REPO_TIMEOUT_SECONDS=7200
#   RUN_STAMP=20260315T133000Z
#   LOG_PATH=/home/azureuser/vmtest-run.log

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SSH_USER="$(whoami)"
WORKERS="${WORKERS:-1}"
PER_REPO_TIMEOUT_SECONDS="${PER_REPO_TIMEOUT_SECONDS:-7200}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ID_PATH="/home/${SSH_USER}/vmtest-run-id.txt"
LOG_PATH="${LOG_PATH:-/home/${SSH_USER}/vmtest-run.log}"

INPUT_REPOS_FILE="${1:-/home/${SSH_USER}/vmtest-repos.txt}"
if [[ -f "$INPUT_REPOS_FILE" ]]; then
  REPOS_FILE="$INPUT_REPOS_FILE"
elif [[ -f "$REPO_ROOT/library_links.txt" ]]; then
  REPOS_FILE="$REPO_ROOT/library_links.txt"
else
  echo "[ERROR] No repos file found. Tried '$INPUT_REPOS_FILE' and '$REPO_ROOT/library_links.txt'" >&2
  exit 2
fi

if command -v sudo >/dev/null 2>&1; then
  SUDO="sudo -n"
else
  SUDO=""
fi

$SUDO apt-get update -y
$SUDO apt-get install -y python3 python3-venv python3-pip git curl

cd "$REPO_ROOT"
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

STAMPED_REPOS_FILE="/home/${SSH_USER}/vmtest-repos-${RUN_STAMP}.txt"
cp "$REPOS_FILE" "$STAMPED_REPOS_FILE"

pkill -f 'parallel_empirical_test.py' || true

echo "$RUN_STAMP" > "$RUN_ID_PATH"
echo "=== RUN_STAMP: $RUN_STAMP ===" > "$LOG_PATH"
echo "Started: $(date -u)" >> "$LOG_PATH"
echo "Repo root: $REPO_ROOT" >> "$LOG_PATH"
echo "Repos file: $STAMPED_REPOS_FILE" >> "$LOG_PATH"
echo "Workers: $WORKERS" >> "$LOG_PATH"

nohup env PER_REPO_TIMEOUT_SECONDS="$PER_REPO_TIMEOUT_SECONDS" \
  "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/src/parallel_empirical_test.py" "$STAMPED_REPOS_FILE" --workers "$WORKERS" \
  >> "$LOG_PATH" 2>&1 < /dev/null &

sleep 2
if pgrep -f "parallel_empirical_test.py $STAMPED_REPOS_FILE --workers $WORKERS" >/dev/null 2>&1; then
  echo "[OK] Started run $RUN_STAMP"
  echo "[OK] Log: $LOG_PATH"
else
  echo "[ERROR] Launch failed; check $LOG_PATH" >&2
  exit 3
fi

