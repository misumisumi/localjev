#!/bin/sh
# Entrypoint for the vLLM test backend. Extra arguments are forwarded through.
set -eu

exec python3 -m vllm.entrypoints.openai.api_server \
  --model "${TESTLLM_MODEL:-google/gemma-4-E2B-it}" \
  --host "${TESTLLM_HOST:-0.0.0.0}" \
  --port "${TESTLLM_PORT:-9010}" \
  --served-model-name "${TESTLLM_SERVED_NAME:-gemma-4-E2B-it}" \
  --max-model-len "${TESTLLM_MAX_LEN:-4096}" \
  --gpu-memory-utilization "${TESTLLM_GPU_UTIL:-0.6}" \
  --trust-remote-code \
  "$@"
