#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

REMOTE_WORKSPACE="${REMOTE_WORKSPACE:-/data/projects/hok1v1}"
PROXY_HOST="${PROXY_HOST:-127.0.0.1}"
PROXY_PORT="${PROXY_PORT:-8765}"
LOG_PATH="${LOG_PATH:-/tmp/kaiwu-proxy-env-server.log}"
PID_PATH="${PID_PATH:-/tmp/kaiwu-proxy-env-server.pid}"
MAX_COMMAND_TIMEOUT="${MAX_COMMAND_TIMEOUT:-600}"

case "${1:-start}" in
  stop)
    if [ -f "$PID_PATH" ] && kill -0 "$(cat "$PID_PATH")" 2>/dev/null; then
      kill "$(cat "$PID_PATH")"
      rm -f "$PID_PATH"
      echo "stopped"
    else
      pkill -f "script/proxy_env_server.py" 2>/dev/null || true
      echo "not running"
    fi
    exit 0
    ;;
  log)
    exec tail -f "$LOG_PATH"
    ;;
esac

if [ -f "$PID_PATH" ] && kill -0 "$(cat "$PID_PATH")" 2>/dev/null; then
  kill "$(cat "$PID_PATH")" 2>/dev/null || true
  sleep 1
fi

mkdir -p "$REMOTE_WORKSPACE"

MAX_COMMAND_TIMEOUT="$MAX_COMMAND_TIMEOUT" \
  nohup python3 -u script/proxy_env_server.py \
    --host "$PROXY_HOST" \
    --port "$PROXY_PORT" \
    --workspace-root "$REMOTE_WORKSPACE" \
    > "$LOG_PATH" 2>&1 &

echo $! > "$PID_PATH"
echo "started pid=$(cat "$PID_PATH") url=http://$PROXY_HOST:$PROXY_PORT workspace=$REMOTE_WORKSPACE log=$LOG_PATH"
sleep 1
tail -20 "$LOG_PATH" || true
