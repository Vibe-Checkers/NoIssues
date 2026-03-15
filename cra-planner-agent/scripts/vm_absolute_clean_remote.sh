#!/usr/bin/env bash
set -euo pipefail

# Runs ON REMOTE VM. Hard wipes everything under /home/azureuser except .ssh.

TARGET_HOME="${1:-/home/azureuser}"

if [[ "$TARGET_HOME" != "/home/azureuser" ]]; then
  echo "Refusing to run on unexpected TARGET_HOME=$TARGET_HOME" >&2
  exit 10
fi

echo "[CLEAN] Starting absolute clean on $TARGET_HOME"

# 1) Kill relevant processes
pkill -9 -f 'parallel_empirical_test.py' || true
pkill -9 -f 'python3 src/parallel_empirical_test.py' || true
pkill -9 -f 'azcopy' || true
pkill -9 -f 'docker build' || true
pkill -9 -f 'docker run' || true

# 2) Hard wipe Docker state
if command -v docker >/dev/null 2>&1; then
  docker ps -aq | xargs -r docker rm -f || true
  docker images -aq | xargs -r docker rmi -f || true
  docker volume ls -q | xargs -r docker volume rm -f || true
  docker network prune -f || true
  docker builder prune -af || true
  docker system prune -af --volumes || true
fi

# 3) Remove everything under home except .ssh
find "$TARGET_HOME" -mindepth 1 -maxdepth 1 \
  ! -name '.ssh' \
  -exec rm -rf {} +

# 4) Recreate minimal expected structure
mkdir -p "$TARGET_HOME/NoIssues"

echo "[CLEAN] Absolute clean completed. Preserved: $TARGET_HOME/.ssh"

