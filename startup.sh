#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"
# When invoked via VAST.ai onstart:
# git clone git@github.com:eric-merritt/ablitMD.git /workspace/ablitMD && bash /workspace/ablitMD/startup.sh

echo "==> Checking MongoDB..."
if ! which mongod &>/dev/null; then
  curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg --dearmor
  echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" \
    | tee /etc/apt/sources.list.d/mongodb-org-7.0.list
  apt-get update -qq && apt-get install -y mongodb-org
  echo "    MongoDB installed"
fi
if ! pgrep -x mongod > /dev/null; then
  mkdir -p /data/db
  mongod --fork --logpath /var/log/mongod.log --bind_ip 127.0.0.1
  sleep 2
  echo "    MongoDB started"
else
  echo "    MongoDB already running"
fi

echo "==> Installing dependencies..."
which uv &>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
npm install --silent
uv sync

echo "==> Seeding database..."
npm run seed --workspace=backend

echo "==> Pre-fetching model weights from HuggingFace..."
export HF_HOME=/workspace/models
mkdir -p /workspace/models
uv run huggingface-cli download Qwen/Qwen3.6-27B \
  --local-dir-use-symlinks False \
  || echo "    Warning: HF download failed — model will download on first load"

echo "==> Starting services..."
exec npm run dev
