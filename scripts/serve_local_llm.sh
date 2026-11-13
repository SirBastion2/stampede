#!/usr/bin/env bash
# Start a local OpenAI-compatible server for STAMPEDE (§7.1, ADR-055).
# Real inference is GPU-only. Unit tests use FakeLocalBackend, not this script.
set -euo pipefail

# Default GGUF directory is ./models/gguf (repo-relative). Override with STAMPEDE_LOCAL_LLM_WEIGHTS.
# Tests never run this script and never open this path.
WEIGHTS="${STAMPEDE_LOCAL_LLM_WEIGHTS:-models/gguf}"
PORT="${STAMPEDE_LOCAL_LLM_PORT:-8080}"
MODEL_NAME="${STAMPEDE_LOCAL_LLM_MODEL:-Meta-Llama-3.1-8B-Instruct-Q4_K_M}"
GGUF_PATH="${STAMPEDE_LOCAL_LLM_GGUF:-${WEIGHTS}/${MODEL_NAME}.gguf}"

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "Real inference requires an NVIDIA GPU (RTX 3060 ~5 GB at Q4). Unit tests use the fake backend." >&2
  exit 1
fi
if [[ ! -f "${GGUF_PATH}" ]]; then
  echo "GGUF not found: ${GGUF_PATH}" >&2
  echo "Set STAMPEDE_LOCAL_LLM_WEIGHTS or STAMPEDE_LOCAL_LLM_GGUF." >&2
  exit 1
fi

export STAMPEDE_LLM=local
export STAMPEDE_LOCAL_LLM_URL="http://127.0.0.1:${PORT}/v1"
export STAMPEDE_LOCAL_LLM_MODEL="${MODEL_NAME}"
export STAMPEDE_LOCAL_LLM_WEIGHTS="${WEIGHTS}"
export STAMPEDE_LOCAL_LLM_GGUF="${GGUF_PATH}"

if command -v llama-server >/dev/null 2>&1; then
  echo "Starting llama-server on :${PORT} (all layers on GPU)…"
  exec llama-server \
    -m "${GGUF_PATH}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    -ngl 99 \
    --alias "${MODEL_NAME}"
fi

echo "Install llama.cpp (llama-server), then re-run. CPU llama.cpp is not supported." >&2
exit 1
