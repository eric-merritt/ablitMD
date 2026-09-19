#!/usr/bin/env bash
# Sync pre-built wheels to the VastAI instance so `uv sync` can find them.
# Usage: bash scripts/sync_wheels.sh [ssh_host]
set -euo pipefail

HOST="${1:-vastai}"
DEST="/workspace/ablitMD/pkgs"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PKGS_DIR="$PROJECT_DIR/pkgs"

FLASH_ATTN_WHL="flash_attn_3-3.0.0+20260609.cu130torch2120cxx11abitrue.bc58ab-cp310-abi3-linux_x86_64.whl"
CONV1D_WHL="causal_conv1d-1.6.1-cp310-cp310-linux_x86_64.whl"

# flash-attn-3 lives in ~/Downloads, not in the repo
FLASH_ATTN_SRC="$HOME/Downloads/$FLASH_ATTN_WHL"
CONV1D_SRC="$PKGS_DIR/$CONV1D_WHL"

for src in "$FLASH_ATTN_SRC" "$CONV1D_SRC"; do
    if [ ! -f "$src" ]; then
        echo "ERROR: $src not found" >&2
        exit 1
    fi
done

echo "==> Syncing wheels to ${HOST}:${DEST}/"
rsync -avz -e ssh \
    "$FLASH_ATTN_SRC" \
    "$CONV1D_SRC" \
    "${HOST}:${DEST}/"

echo "Done."
