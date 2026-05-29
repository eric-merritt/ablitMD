#!/usr/bin/env bash
set -e

[ -f /opt/nvm/nvm.sh ] && source /opt/nvm/nvm.sh

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"
# When invoked via VAST.ai onstart:
# git clone git@github.com:eric-merritt/ablitMD.git /workspace/ablitMD && bash /workspace/ablitMD/startup.sh


echo "==> Restoring env files from image..."
cp /workspace/env-staging/.env           "$PROJECT_DIR/.env"
cp /workspace/env-staging/frontend/.env  "$PROJECT_DIR/frontend/.env"
cp /workspace/env-staging/backend/.env   "$PROJECT_DIR/backend/.env"

echo "==> Installing dependencies..."
which uv &>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
npm install --silent
uv sync

echo "==> Seeding database..."
npm run seed --workspace=backend

echo "==> Pre-fetching model weights from HuggingFace..."
export HF_HOME=/workspace/models
mkdir -p /workspace/models
hf download Qwen/Qwen3.6-27B --local-dir /workspace/models/Qwen3.6-27B \
  || echo "    Warning: HF download failed — model will download on first load"

echo "==> Starting services..."
exec npm run dev
