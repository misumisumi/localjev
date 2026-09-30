#!/bin/sh
# Entrypoint for the SGLang test backend. Extra arguments are forwarded through.
set -eu

exec python3 -m sglang.launch_server \
  --model-path "${TESTLLM_MODEL:-google/gemma-4-E2B-it}" \
  --host "${TESTLLM_HOST:-0.0.0.0}" \
  --port "${TESTLLM_PORT:-9010}" \
  --served-model-name "${TESTLLM_SERVED_NAME:-gemma-4-E2B-it}" \
  --context-length "${TESTLLM_MAX_LEN:-4096}" \
  --mem-fraction-static "${TESTLLM_MEM_FRACTION:-0.6}" \
  --trust-remote-code \
  "$@"
