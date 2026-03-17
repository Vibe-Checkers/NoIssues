#!/usr/bin/env bash
set -uo pipefail

# --- Docker ---
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin

sudo usermod -aG docker "$USER"

# --- uv ---
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

# --- Repo ---
git clone https://github.com/Vibe-Checkers/NoIssues.git
cd NoIssues
git pull
git checkout optimus-prime

uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

cp .env.example .env

echo ""
echo "Done. Next steps:"
echo "  1. cd NoIssues && source .venv/bin/activate"
echo "  2. Fill in your credentials in .env"
echo "  3. Log out and back in (or run 'newgrp docker') for Docker group to take effect"
