# Test-only inference backend for LocalJev. Serves Google's Gemma 4 E2B (an
# "any-to-any" model with an image/audio projector) through an unpatched
# llama.cpp server, exposing the OpenAI-compatible API LocalJev reads.
#
#   podman build -f deploy/testllm/Dockerfile -t localhost/testllm:latest .
#   podman run --rm --device nvidia.com/gpu=all -p 9010:9010 localhost/testllm:latest
#
# On a CPU-only host drop --device and build the CPU base:
#   podman build -f deploy/testllm/Dockerfile -t localhost/testllm:latest \
#     --build-arg LLAMACPP_IMAGE=ghcr.io/ggml-org/llama.cpp:server .

ARG LLAMACPP_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda

# Bake the GGUF weights + multimodal projector into the image so the container
# needs no network at run time. Override the files to pick another quant.
FROM python:3.12-slim AS models

RUN pip install --no-cache-dir "huggingface_hub[cli]"

ARG MODEL_REPO=ggml-org/gemma-4-E2B-it-GGUF
ARG MODEL_FILE=gemma-4-E2B-it-Q4_0.gguf
ARG MMPROJ_FILE=mmproj-gemma-4-E2B-it-Q8_0.gguf

RUN hf download "${MODEL_REPO}" "${MODEL_FILE}" "${MMPROJ_FILE}" --local-dir /models

# Runtime: the official llama.cpp server image (llama-server lives at /app).
FROM ${LLAMACPP_IMAGE} AS runtime

ARG MODEL_FILE=gemma-4-E2B-it-Q4_0.gguf
ARG MMPROJ_FILE=mmproj-gemma-4-E2B-it-Q8_0.gguf

COPY --from=models /models/ /models/
COPY --chmod=755 deploy/testllm/entrypoint.sh /usr/local/bin/testllm-entrypoint

ENV TESTLLM_MODEL=/models/${MODEL_FILE} \
    TESTLLM_MMPROJ=/models/${MMPROJ_FILE} \
    TESTLLM_HOST=0.0.0.0 \
    TESTLLM_PORT=9010 \
    TESTLLM_CTX=16384 \
    TESTLLM_NGL=99

EXPOSE 9010

ENTRYPOINT ["/usr/local/bin/testllm-entrypoint"]
