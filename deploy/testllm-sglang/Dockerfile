# Test-only SGLang backend for LocalJev: serves Gemma 4 E2B through SGLang's
# native /generate endpoint so the `selective` readout (token_ids_logprob) can be
# verified. Weights are downloaded at run time into a mounted HF cache, not baked.
#
#   podman build -f deploy/testllm-sglang/Dockerfile -t localhost/testllm-sglang:latest .
#   podman volume create testllm-hf
#   podman run -d --name testllm --network llm --device nvidia.com/gpu=all \
#     -p 9010:9010 -v testllm-hf:/hf -e HF_HOME=/hf localhost/testllm-sglang:latest

FROM docker.io/lmsysorg/sglang:latest

COPY --chmod=755 deploy/testllm-sglang/entrypoint.sh /usr/local/bin/testllm-entrypoint

ENV HF_HOME=/hf \
    TESTLLM_MODEL=google/gemma-4-E2B-it \
    TESTLLM_HOST=0.0.0.0 \
    TESTLLM_PORT=9010 \
    TESTLLM_MAX_LEN=4096 \
    TESTLLM_MEM_FRACTION=0.6

EXPOSE 9010

ENTRYPOINT ["/usr/local/bin/testllm-entrypoint"]
