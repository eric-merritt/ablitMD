#!/usr/bin/env bash
# Bootstrap: pull the three dotenvx private-key files from the ablitmd-data service and
# drop each where it belongs under /workspace/ablitMD. Run this FIRST on a fresh
# instance — before anything else touches the data server — because the .env's that
# git clone brought over are encrypted, and every later step (unlocking them, serving
# run data) needs these keys in place.
#
# The key files themselves are plain text in the repo (gitignored, so not in the clone),
# which is why the data server can serve them straight off disk without unlocking anything.
#
#   REMOTE_DATA_BASE=https://eric-merritt.com/ablitMD/data \
#   REMOTE_DATA_KEY=<secret> bash scripts/fetch_env_keys.sh
set -e

BASE="${REMOTE_DATA_BASE:?REMOTE_DATA_BASE must be set}"
KEY="${REMOTE_DATA_KEY:?REMOTE_DATA_KEY must be set}"
APP_DIR="${APP_DIR:-/workspace/ablitMD}"

echo "==> Fetching .env.keys from ${BASE} ..."
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

curl -fsS "${BASE}/envkeys.tar?key=${KEY}" -o "$tmp/envkeys.tar"

# The tarball carries: .env.keys, frontend/.env.keys, backend/.env.keys (relative to
# the repo root). Extract straight into the app dir so each lands in its own subdir.
mkdir -p "$APP_DIR/frontend" "$APP_DIR/backend"
tar -xf "$tmp/envkeys.tar" -C "$APP_DIR"

for f in ".env.keys" "frontend/.env.keys" "backend/.env.keys"; do
  if [ -f "$APP_DIR/$f" ]; then
    echo "    placed $APP_DIR/$f"
  else
    echo "    WARNING: $APP_DIR/$f not found after extract" >&2
  fi
done
