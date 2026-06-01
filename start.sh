#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/flask"
FRONTEND_DIR="$ROOT_DIR/flask/my-app"
BACKEND_PORT="${BACKEND_PORT:-3636}"
FRONTEND_PORT="${PORT:-3000}"
BACKEND_URL="http://127.0.0.1:$BACKEND_PORT"
FRONTEND_URL="http://127.0.0.1:$FRONTEND_PORT"
BACKEND_LOG="${BACKEND_LOG:-${TMPDIR:-/tmp}/system-flask-$BACKEND_PORT.log}"
BACKEND_PID=""

if [[ ! -f "$BACKEND_DIR/main.py" ]]; then
  echo "Backend entry not found: $BACKEND_DIR/main.py" >&2
  exit 1
fi

if [[ ! -f "$FRONTEND_DIR/package.json" ]]; then
  echo "Frontend package.json not found: $FRONTEND_DIR/package.json" >&2
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "python is required but was not found in PATH." >&2
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required but was not found in PATH." >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required but was not found in PATH." >&2
  exit 1
fi

if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
  echo "Installing frontend dependencies..."
  npm --prefix "$FRONTEND_DIR" install
fi

is_ready() {
  curl -fsS "$1" >/dev/null 2>&1
}

cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]] && kill -0 "$BACKEND_PID" >/dev/null 2>&1; then
    echo "Stopping Flask backend..."
    kill "$BACKEND_PID" >/dev/null 2>&1 || true
    wait "$BACKEND_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if is_ready "$BACKEND_URL/"; then
  echo "Flask backend is already running at $BACKEND_URL"
else
  echo "Starting Flask backend at $BACKEND_URL"
  (
    cd "$BACKEND_DIR"
    BACKEND_PORT="$BACKEND_PORT" python - <<'PY'
import os

import main

main.get_predcit()
main.app.run(
    host="127.0.0.1",
    port=int(os.environ.get("BACKEND_PORT", "3636")),
    debug=False,
    use_reloader=False,
)
PY
  ) >"$BACKEND_LOG" 2>&1 &
  BACKEND_PID="$!"
fi

for _ in {1..30}; do
  if is_ready "$BACKEND_URL/"; then
    break
  fi

  if ! kill -0 "$BACKEND_PID" >/dev/null 2>&1; then
    echo "Flask backend failed to start. Log:" >&2
    tail -n 80 "$BACKEND_LOG" >&2 || true
    exit 1
  fi

  sleep 1
done

if ! is_ready "$BACKEND_URL/"; then
  echo "Flask backend did not become ready on port $BACKEND_PORT. Log:" >&2
  tail -n 80 "$BACKEND_LOG" >&2 || true
  exit 1
fi

if is_ready "$FRONTEND_URL/"; then
  echo "Next.js frontend is already running at $FRONTEND_URL"
  if [[ -n "$BACKEND_PID" ]]; then
    echo "Press Ctrl-C to stop the Flask backend started by this script."
    wait "$BACKEND_PID"
  fi
  exit 0
fi

echo "Starting Next.js frontend at $FRONTEND_URL"
echo "Backend log: $BACKEND_LOG"
cd "$FRONTEND_DIR"
npm run dev -- -p "$FRONTEND_PORT"
