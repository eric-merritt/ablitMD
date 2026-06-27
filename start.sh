#!/usr/bin/env bash

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$PROJECT_DIR/.pids"

cd "$PROJECT_DIR"

# Free a TCP port by killing whatever is currently listening on it. ss is always
# present (iproute2); -p needs root, which the vast box runs as. This is the reliable
# cleanup — .pids can be stale, but the port is the thing that actually collides.
free_port() {
  local port="$1"
  local pids
  pids="$(ss -tlnpH "sport = :$port" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u)"
  if [ -n "$pids" ]; then
    echo "$pids" | xargs -r kill 2>/dev/null
    echo "freed :$port (killed $(echo $pids | tr '\n' ' '))"
  fi
}

if [ -f "$PID_FILE" ]; then
  while IFS= read -r pid; do
    [ -z "$pid" ] && continue
    kill "$pid" 2>/dev/null && echo "killed $pid"
  done < "$PID_FILE"
  rm "$PID_FILE"
fi

# Belt-and-suspenders: clear the ports regardless of what .pids said.
free_port 8237   # backend
free_port 8238   # inference
free_port 5400   # frontend
sleep 1

npm run dev --workspace=backend > /tmp/ablitmd-backend.log 2>&1 &
echo $! >> "$PID_FILE"

uv run python -m backend.inference.service > /tmp/ablitmd-inference.log 2>&1 &
echo $! >> "$PID_FILE"

npm run dev --workspace=frontend > /tmp/ablitmd-frontend.log 2>&1 &
echo $! >> "$PID_FILE"

echo "started — PIDs: $(tr '\n' ' ' < "$PID_FILE")"
echo "logs: /tmp/ablitmd-{backend,inference,frontend}.log"
