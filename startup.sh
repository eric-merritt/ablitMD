#!/usr/bin/env bash
set -e

[ -f /opt/nvm/nvm.sh ] && source /opt/nvm/nvm.sh

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"
# When invoked via VAST.ai onstart:
# git clone git@github.com:eric-merritt/ablitMD.git /workspace/ablitMD && bash /workspace/ablitMD/startup.sh


echo "==> Restoring env files from image (fallback — git clone normally has them)..."
[ -f "$PROJECT_DIR/.env" ]          || cp /workspace/env-staging/.env          "$PROJECT_DIR/.env"          2>/dev/null || true
[ -f "$PROJECT_DIR/frontend/.env" ] || cp /workspace/env-staging/frontend/.env "$PROJECT_DIR/frontend/.env" 2>/dev/null || true
[ -f "$PROJECT_DIR/backend/.env" ]  || cp /workspace/env-staging/backend/.env  "$PROJECT_DIR/backend/.env"  2>/dev/null || true

echo "==> Writing .env.keys from VastAI env vars (skipped when var unset — preserves rsynced keys)..."
if [ -n "$DOTENV_PRIVATE_KEY_ROOT" ]; then
  echo "DOTENV_PRIVATE_KEY=$DOTENV_PRIVATE_KEY_ROOT"     > "$PROJECT_DIR/.env.keys"
fi
if [ -n "$DOTENV_PRIVATE_KEY_FRONTEND" ]; then
  echo "DOTENV_PRIVATE_KEY=$DOTENV_PRIVATE_KEY_FRONTEND" > "$PROJECT_DIR/frontend/.env.keys"
fi
if [ -n "$DOTENV_PRIVATE_KEY_BACKEND" ]; then
  echo "DOTENV_PRIVATE_KEY=$DOTENV_PRIVATE_KEY_BACKEND"  > "$PROJECT_DIR/backend/.env.keys"
fi

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

echo "==> Fetching flash_attn wheel (pyproject path dep — uv sync fails without it)..."
FLASH_ATTN_WHEEL="flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
mkdir -p "$PROJECT_DIR/pkgs"
[ -f "$PROJECT_DIR/pkgs/$FLASH_ATTN_WHEEL" ] || curl -fL -o "$PROJECT_DIR/pkgs/$FLASH_ATTN_WHEEL" \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/$FLASH_ATTN_WHEEL"

echo "==> Installing dependencies..."
which uv &>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
npm install --silent
uv sync

echo "==> Seeding database..."
npm run seed --workspace=backend

echo "==> Pre-fetching model weights from HuggingFace..."
export HF_HOME=/workspace/models
mkdir -p /workspace/models
uv run hf download Qwen/Qwen3.6-27B --local-dir /workspace/models/Qwen3.6-27B \
  || echo "    Warning: HF download failed — model will download on first load"

echo "==> Starting services..."
exec npm run dev
