# Test-only vLLM backend for LocalJev: serves Gemma 4 E2B through vLLM's
# OpenAI-compatible server, whose `/v1/completions` accepts `logprob_token_ids`
# (vLLM >= 0.26.0), so the `selective` readout can be verified. Weights are
# downloaded at run time into a mounted HF cache, not baked.
#
#   podman build -f deploy/testllm-vllm/Dockerfile -t localhost/testllm-vllm:latest .
#   podman volume create testllm-hf
#   podman run -d --name testllm --network llm --device nvidia.com/gpu=all \
#     -p 9010:9010 -v testllm-hf:/hf -e HF_HOME=/hf localhost/testllm-vllm:latest

FROM docker.io/vllm/vllm-openai:latest

COPY --chmod=755 deploy/testllm-vllm/entrypoint.sh /usr/local/bin/testllm-entrypoint

ENV HF_HOME=/hf \
    TESTLLM_MODEL=google/gemma-4-E2B-it \
    TESTLLM_HOST=0.0.0.0 \
    TESTLLM_PORT=9010 \
    TESTLLM_MAX_LEN=4096 \
    TESTLLM_GPU_UTIL=0.6

EXPOSE 9010

ENTRYPOINT ["/usr/local/bin/testllm-entrypoint"]
