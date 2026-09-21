#!/usr/bin/env bash

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$PROJECT_DIR/.pids"

cd "$PROJECT_DIR"

# Pids of processes listening on a TCP port, resolved straight from /proc so it works
# in minimal containers (the pytorch image ships no ss/fuser/lsof).
port_listener_pids() {
  local port="$1"
  local hexPort socketInode
  hexPort="$(printf '%04X' "$port")"
  for socketInode in $(awk -v hexPort=":$hexPort" '$2 ~ hexPort"$" && $4 == "0A" { print $10 }' /proc/net/tcp /proc/net/tcp6 2>/dev/null); do
    find /proc/[0-9]*/fd -lname "socket:\[$socketInode\]" 2>/dev/null | cut -d/ -f3
  done | sort -u
}

# Climb from a listener to the top of its service wrapper chain (node --watch, dotenvx,
# npm, concurrently). Killing only the listener is useless under node --watch — the
# watcher respawns it with its stale env. Stops at anything else (shells, init) so an
# interactive terminal is never killed. Prints the whole chain, listener first.
service_chain_pids() {
  local currentPid="$1"
  local parentPid parentCmd
  echo "$currentPid"
  while :; do
    parentPid="$(awk '/^PPid:/{ print $2 }' /proc/$currentPid/status 2>/dev/null)"
    { [ -z "$parentPid" ] || [ "$parentPid" -le 1 ]; } && break
    parentCmd="$(tr '\0' ' ' < /proc/$parentPid/cmdline 2>/dev/null)"
    case "$parentCmd" in
      *"bin/www"*|*inference.service*|*"/.bin/vite"*|*"dotenvx run"*|*"npm run"*|*concurrently*) echo "$parentPid"; currentPid="$parentPid" ;;
      *) break ;;
    esac
  done
}

# Free a TCP port by killing the listener and its wrapper chain. This is the reliable
# cleanup — .pids can be stale, but the port is the thing that actually collides.
free_port() {
  local port="$1"
  local listenerPid chainPids
  for listenerPid in $(port_listener_pids "$port"); do
    chainPids="$(service_chain_pids "$listenerPid" | sort -u)"
    kill $chainPids 2>/dev/null
    echo "freed :$port (killed $(echo $chainPids | tr '\n' ' '))"
  done
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
