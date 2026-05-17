#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"
# When invoked via VAST.ai onstart:
# git clone git@github.com:eric-merritt/ablitMD.git /workspace/ablitMD && bash /workspace/ablitMD/startup.sh

echo "==> Checking MongoDB..."
if ! pgrep -x mongod > /dev/null; then
  mkdir -p /data/db
  mongod --fork --logpath /var/log/mongod.log --bind_ip 127.0.0.1
  sleep 2
  echo "    MongoDB started"
else
  echo "    MongoDB already running"
fi

echo "==> Installing dependencies..."
npm install --silent

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
