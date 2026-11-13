#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export STAMPEDE_OFFLINE="${STAMPEDE_OFFLINE:-1}"
# SVB-50: the demo runs the calibrated engine; STAMPEDE_MODE=fake gives the scripted arc.
export STAMPEDE_MODE="${STAMPEDE_MODE:-real}"
export STAMPEDE_BACKEND="${STAMPEDE_BACKEND:-gpu}"   # CPY-1: CuPy by default; STAMPEDE_BACKEND=cpu for NumPy
export STAMPEDE_PORT="${STAMPEDE_PORT:-8000}"

PYTHON="python3"
if command -v python >/dev/null 2>&1 && python -c "import sys" >/dev/null 2>&1; then
  PYTHON="python"
fi

if ! "$PYTHON" -c "import fastapi" 2>/dev/null; then
  echo "Install server extras: pip install -e '.[server,dev]'"
  exit 1
fi

"$PYTHON" -m ui.server &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 50); do
  if curl -sf "http://127.0.0.1:${STAMPEDE_PORT}/health" >/dev/null; then
    break
  fi
  sleep 0.2
done

(
  cd ui/crowd
  export VITE_CROWD_FAKE=0
  export VITE_STAMPEDE_API="http://127.0.0.1:${STAMPEDE_PORT}"
  npm run dev -- --host 127.0.0.1 --port 5173
) &
CROWD_PID=$!

(
  cd ui/market
  npm run dev -- --host 127.0.0.1 --port 5174
) &
MARKET_PID=$!

echo "Live server: http://127.0.0.1:${STAMPEDE_PORT}/health"
echo "Crowd UI:    http://127.0.0.1:5173"
echo "Market UI:   http://127.0.0.1:5174"

wait "$CROWD_PID" "$MARKET_PID"
