#!/usr/bin/env bash
# Start M29 + M30 Vite dev servers in fake mode and open both monitors in the default browser.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CROWD_PORT=5173
MARKET_PORT=5174
CROWD_URL="http://127.0.0.1:${CROWD_PORT}/"
MARKET_URL="http://127.0.0.1:${MARKET_PORT}/?fake=1"

cd "$ROOT"

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required but not installed." >&2
  exit 1
fi

if [[ ! -d "$ROOT/ui/crowd/node_modules" ]]; then
  echo "Installing crowd UI dependencies…"
  (cd "$ROOT/ui/crowd" && npm install)
fi

if [[ ! -d "$ROOT/ui/market/node_modules" ]]; then
  echo "Installing market UI dependencies…"
  (cd "$ROOT/ui/market" && npm install)
fi

mkdir -p "$ROOT/.demo-logs"

if ! curl -sf "$CROWD_URL" >/dev/null 2>&1; then
  echo "Starting crowd UI on port ${CROWD_PORT}…"
  (cd "$ROOT/ui/crowd" && npm run dev -- --host 127.0.0.1 --port "$CROWD_PORT") \
    >>"$ROOT/.demo-logs/crowd-dev.log" 2>&1 &
  CROWD_PID=$!
else
  CROWD_PID=""
fi

if ! curl -sf "$MARKET_URL" >/dev/null 2>&1; then
  echo "Starting market UI on port ${MARKET_PORT}…"
  (cd "$ROOT/ui/market" && npm run dev -- --host 127.0.0.1 --port "$MARKET_PORT") \
    >>"$ROOT/.demo-logs/market-dev.log" 2>&1 &
  MARKET_PID=$!
else
  MARKET_PID=""
fi

wait_for() {
  local url="$1"
  local label="$2"
  for _ in $(seq 1 60); do
    if curl -sf "$url" >/dev/null 2>&1; then
      echo "${label} ready at ${url}"
      return 0
    fi
    sleep 0.5
  done
  echo "Timed out waiting for ${label}. See .demo-logs/." >&2
  return 1
}

wait_for "$CROWD_URL" "Crowd UI"
wait_for "$MARKET_URL" "Market UI"

if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$CROWD_URL" >/dev/null 2>&1 || true
  xdg-open "$MARKET_URL" >/dev/null 2>&1 || true
else
  echo "Open ${CROWD_URL} and ${MARKET_URL}"
fi

echo "Demo running (fake mode). Logs: .demo-logs/"
if [[ -n "${CROWD_PID:-}" || -n "${MARKET_PID:-}" ]]; then
  echo "Press Ctrl+C to stop background dev servers."
  wait
fi
