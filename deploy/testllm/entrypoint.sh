#!/bin/sh
# Entrypoint for Dockerfile.testllm: start llama-server with the baked Gemma 4
# E2B weights and multimodal projector. Extra arguments are forwarded through.
set -eu

BIN="${LLAMA_SERVER_BIN:-/app/llama-server}"
if [ ! -x "$BIN" ]; then
  BIN="$(command -v llama-server)"
fi

MODEL="${TESTLLM_MODEL:-/models/gemma-4-E2B-it-Q4_0.gguf}"
MMPROJ="${TESTLLM_MMPROJ:-/models/mmproj-gemma-4-E2B-it-Q8_0.gguf}"

if [ ! -f "$MODEL" ]; then
  echo "testllm: model not found: $MODEL" >&2
  exit 1
fi

set -- \
  --host "${TESTLLM_HOST:-0.0.0.0}" \
  --port "${TESTLLM_PORT:-9010}" \
  -m "$MODEL" \
  --ctx-size "${TESTLLM_CTX:-16384}" \
  --n-gpu-layers "${TESTLLM_NGL:-99}" \
  --parallel 1 \
  --jinja \
  "$@"

if [ -f "$MMPROJ" ]; then
  set -- "$@" --mmproj "$MMPROJ"
fi

exec "$BIN" "$@"
