#!/usr/bin/env bash
set -e

[ -f /opt/nvm/nvm.sh ] && source /opt/nvm/nvm.sh

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"
# When invoked via VAST.ai onstart:
# git clone git@github.com:eric-merritt/ablitMD.git /workspace/ablitMD && bash /workspace/ablitMD/startup.sh


echo "==> Restoring env files from image..."

echo "==> Writing .env.keys from VastAI env vars..."
echo "DOTENV_PRIVATE_KEY=$DOTENV_PRIVATE_KEY_ROOT"     > "$PROJECT_DIR/.env.keys"
echo "DOTENV_PRIVATE_KEY=$DOTENV_PRIVATE_KEY_FRONTEND" > "$PROJECT_DIR/frontend/.env.keys"
echo "DOTENV_PRIVATE_KEY=$DOTENV_PRIVATE_KEY_BACKEND"  > "$PROJECT_DIR/backend/.env.keys"

echo "==> Registering instance (Atlas IP allowlist + ~/.ssh/config on home box)..."
# Vast var names vary by template — check `env | grep -iE 'ipaddr|port|ssh'` and adjust.
INSTANCE_IP="${PUBLIC_IPADDR:-$(curl -s https://ifconfig.me)}"
INSTANCE_PORT="${VAST_TCP_PORT_22:-${SSH_PORT:-22}}"
if [ -n "$REMOTE_DATA_BASE" ] && [ -n "$REMOTE_DATA_KEY" ]; then
  curl -s -X POST "${REMOTE_DATA_BASE}/instance?key=${REMOTE_DATA_KEY}&ip=${INSTANCE_IP}&port=${INSTANCE_PORT}" \
    && echo "    registered ${INSTANCE_IP}:${INSTANCE_PORT}" \
    || echo "    Warning: instance registration failed"
else
  echo "    Skipped: REMOTE_DATA_BASE/REMOTE_DATA_KEY not set"
fi
# Atlas allowlist propagates while deps install below — no explicit wait needed.

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
