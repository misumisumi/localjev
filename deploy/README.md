# Podman deployment

LocalJev runs as a plain pod on the same podman network as the inference server.
The inference server is **not** part of this repo; for a self-contained test
backend use `deploy/testllm/Dockerfile`.

## Build images

```bash
podman build -t localhost/localjev:latest .
podman build -f deploy/testllm/Dockerfile -t localhost/testllm:latest .
```

## Share the network with the inference server

```bash
podman network create llm        # once; skip if it already exists
```

## Production stack

```bash
# llama-server pod is already published on the "llm" network.
podman kube play --replace --network llm \
  --configmap deploy/localjev-config.yaml deploy/localjev-pod.yaml

curl http://127.0.0.1:8081/ready
podman kube down deploy/localjev-pod.yaml
```

## Test stack (Gemma 4 E2B)

`podman kube play` cannot pass GPU/CDI devices, so start the test LLM with
`podman run` when a GPU is available:

```bash
podman run -d --name testllm --network llm --device nvidia.com/gpu=all \
  -p 9010:9010 localhost/testllm:latest

podman kube play --replace --network llm \
  --configmap deploy/localjev-config.test.yaml deploy/localjev-pod.yaml

podman kube down deploy/localjev-pod.yaml
podman rm -f testllm
```

CPU-only equivalent: build with `--build-arg LLAMACPP_IMAGE=ghcr.io/ggml-org/llama.cpp:server`,
then run without `--device` (or use `deploy/testllm/pod.yaml`).

## Test stacks for vLLM / SGLang (selective readout)

To exercise the `selective` readout (`logprob_token_ids` / `token_ids_logprob`),
`deploy/testllm-vllm/` and `deploy/testllm-sglang/` serve the same Gemma 4 E2B
checkpoint through vLLM and SGLang. Weights are downloaded at run time into a
shared HF cache volume, not baked into the image. **Stop `testllm` first** — all
three stacks serve port 9010 and share the GPU.

```bash
podman build -f deploy/testllm-sglang/Dockerfile -t localhost/testllm-sglang:latest .
podman volume create testllm-hf
podman run -d --name testllm --network llm --device nvidia.com/gpu=all \
  -p 9010:9010 -v testllm-hf:/hf -e HF_HOME=/hf localhost/testllm-sglang:latest
# then a configmap with LOCALJEV_BACKEND=sglang (or vllm), e.g. copy
# deploy/localjev-config.test.yaml and edit the backend/readout lines.
```

Both default to `LOCALJEV_READOUT=auto` -> `selective`. `GET /ready` should report
`"effective": "selective"` and a `null` warning; a fallback to `bias` is reported
there and logged.

## Notes

- Do not enable `hostNetwork` and do not override `dnsPolicy`/`dnsConfig`; name
  resolution of the upstream pod relies on aardvark-dns.
- `LOCALJEV_BACKEND` in `deploy/localjev-config*.yaml` must match the inference server
  dialect (`llamacpp`, `vllm`, `sglang`, or `openai`).
- Unpatched llama.cpp ignores `logit_bias` in its pre-sampling logprobs, so the
  `llamacpp` configs use `LOCALJEV_READOUT=vocab` (full-vocabulary read, ~26 MB per
  read for Gemma 4 E2B; in-flight reads are capped lower to bound RAM). Confirm with
  `bun run verify` and switch to `bias` only if the server reflects `logit_bias`.
- `vllm`/`sglang` default to `selective` (native selected-token logprobs, tiny
  payloads). If the endpoint is unavailable, LocalJev logs a one-time warning and
  falls back to `bias`; `GET /ready` reports the effective readout and warning.
- The localjev image installs `curl` so the pod's `httpGet` probes work under
  `podman kube play`, which emulates them with curl.
- `localjev-config.yaml` and `localjev-config.test.yaml` both define the `localjev-config`
  ConfigMap; pass exactly one to `podman kube play`.
