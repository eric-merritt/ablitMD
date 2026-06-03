#!/usr/bin/env bash

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$PROJECT_DIR/.pids"

cd "$PROJECT_DIR"

if [ -f "$PID_FILE" ]; then
  while IFS= read -r pid; do
    [ -z "$pid" ] && continue
    kill "$pid" 2>/dev/null && echo "killed $pid"
  done < "$PID_FILE"
  rm "$PID_FILE"
  sleep 1
fi

npm run dev --workspace=backend > /tmp/ablitmd-backend.log 2>&1 &
echo $! >> "$PID_FILE"

uv run python -m backend.inference.service > /tmp/ablitmd-inference.log 2>&1 &
echo $! >> "$PID_FILE"

npm run dev --workspace=frontend > /tmp/ablitmd-frontend.log 2>&1 &
echo $! >> "$PID_FILE"

echo "started — PIDs: $(tr '\n' ' ' < "$PID_FILE")"
echo "logs: /tmp/ablitmd-{backend,inference,frontend}.log"
