#!/usr/bin/env bash
set -e

# Install nvm if the image didn't ship it, via the official wget | sh flow.
if [ ! -s /opt/nvm/nvm.sh ]; then
  echo "==> Installing nvm (wget | sh)..."
  export NVM_DIR=/opt/nvm
  wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash || true
fi
[ -f /opt/nvm/nvm.sh ] && source /opt/nvm/nvm.sh

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"
# When invoked via VAST.ai onstart:
# git clone git@github.com:eric-merritt/ablitMD.git /workspace/ablitMD && bash /workspace/ablitMD/startup.sh

echo "==> Restoring env files from image (fallback — git clone normally has them)..."
[ -f "$PROJECT_DIR/.env" ]          || cp /workspace/env-staging/.env          "$PROJECT_DIR/.env"          2>/dev/null || true
[ -f "$PROJECT_DIR/frontend/.env" ] || cp /workspace/env-staging/frontend/.env "$PROJECT_DIR/frontend/.env" 2>/dev/null || true
[ -f "$PROJECT_DIR/backend/.env" ]  || cp /workspace/env-staging/backend/.env  "$PROJECT_DIR/backend/.env"  2>/dev/null || true

# Pull the three dotenvx private-key files from the data service and drop each where it
# belongs under /workspace/ablitMD. VastAI's env won't carry them, and they're gitignored
# so the clone doesn't have them either — the home box (where the data server runs) is
# the only place they live. This MUST run before anything else hits the data server: the
# .env's that git clone brought over are encrypted, and unlocking them needs these keys.
if [ -n "$REMOTE_DATA_BASE" ] && [ -n "$REMOTE_DATA_KEY" ]; then
  APP_DIR="$PROJECT_DIR" bash scripts/fetch_env_keys.sh
else
  echo "    Skipped: REMOTE_DATA_BASE/REMOTE_DATA_KEY not set"
fi

# Capture the two export blocks this script produces and echo them to the console so
# the VastAI log shows exactly what's on disk.
ENV_KEYS_EXPORT=""
[ -f "$PROJECT_DIR/.env.keys" ] && ENV_KEYS_EXPORT="$(cat "$PROJECT_DIR/.env.keys")"
echo "==> .env.keys export block: ${ENV_KEYS_EXPORT:-<none fetched>}"

WHEELS_EXPORT=""
for whl in \
  "flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl" \
  "causal_conv1d-1.7.0-cp312-cp312-linux_x86_64.whl"; do
  [ -f "$PROJECT_DIR/pkgs/$whl" ] && WHEELS_EXPORT="$WHEELS_EXPORT $whl"
done
echo "==> pkgs export block:${WHEELS_EXPORT:- <none found>}"

echo "==> Registering instance (Atlas IP allowlist + ~/.ssh/config on home box)..."
# Vast var names vary by template — check `env | grep -iE 'ipaddr|port|ssh'` and adjust.
INSTANCE_IP="${PUBLIC_IPADDR:-$(curl -s https://ifconfig.me)}"
INSTANCE_PORT="${VAST_TCP_PORT_22:-${SSH_PORT:-22}}"
if [ -n "$REMOTE_DATA_BASE" ] && [ -n "$REMOTE_DATA_KEY" ]; then
  curl -s -X POST "${REMOTE_DATA_BASE}/instance?key=${REMOTE_DATA_KEY}&ip=${INSTANCE_IP}&port=${INSTANCE_PORT}" \
    && echo "    registered ${INSTANCE_IP}:${INSTANCE_PORT}" \
    || echo "    Warning: instance registration failed"
fi
# Atlas allowlist propagates while deps install below — no explicit wait needed.

echo "==> Fetching wheels (flash_attn, causal_conv1d) from data service..."
mkdir -p "$PROJECT_DIR/pkgs"
if [ -n "$REMOTE_DATA_BASE" ] && [ -n "$REMOTE_DATA_KEY" ]; then
  curl -fsS "${REMOTE_DATA_BASE}/pkgs.tar?key=${REMOTE_DATA_KEY}" | tar -x -C "$PROJECT_DIR" \
    || echo "    Warning: wheel fetch failed — uv sync will try PyPI (likely to fail on these)"
fi
for whl in \
  "flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl" \
  "causal_conv1d-1.7.0-cp312-cp312-linux_x86_64.whl"; do
  if [ ! -f "$PROJECT_DIR/pkgs/$whl" ]; then
    echo "ERROR: pkgs/$whl not found after fetch." >&2
    exit 1
  fi
done

echo "==> Installing dependencies..."
which uv &>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
npm install --silent
uv sync

echo "==> Seeding database..."
npm run seed --workspace=backend

echo "==> Pre-fetching model weights from HuggingFace..."
export HF_HOME=/workspace/models
mkdir -p /workspace/models
uv run hf download Qwen/Qwen3.8-27B --local-dir /workspace/models/Qwen3.8-27B \
  || echo "    Warning: HF download failed — model will download on first load"

echo "==> Starting services..."
exec npm run dev
