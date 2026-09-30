# LocalJev

A local, Jev-compatible `POST /v1/systemone` API written in TypeScript for
[Bun](https://bun.sh/). LocalJev reads probabilities from **first-token logprobs of a
local llama-server** instead of asking a model to write JSON probabilities.

The defaults target:

- inference server: `http://127.0.0.1:8000` (llama-server, unpatched)
- LocalJev API: `http://127.0.0.1:8081`

## How the read works

[Jev](https://typesafe.ai/) uses a typed decision API rather than an OpenAI chat
API. OpenJev obtains probabilities with a one-step DiffusionGemma structured read
over patched vLLM extensions. LocalJev gets equivalent first-token probabilities from
an **unpatched** llama-server:

1. translate `state` and typed Jev questions into a plain classification prompt
   (state first so the KV prefix is reusable across questions; client question
   keys never reach the prompt);
2. map each option to a single-token label — `A..Z` for choice, `0..9` for score,
   `yes`/`no` for noul — validated once against `/tokenize`;
3. read the first token's logprobs for every label. Three readout strategies are
   available (`LOCALJEV_READOUT`):
   - **`vocab`** (llama.cpp default): send `POST /v1/completions` with
     `max_tokens: 1` and request the whole vocabulary (`logprobs` clamped to
     `n_vocab`), then read each label directly. No patch and no `logit_bias` —
     required because llama.cpp returns **pre-sampling** `log(softmax(logits))`
     that ignores `logit_bias`.
   - **`bias`** (OpenAI default): apply the **same additive `logit_bias` on every
     label token** and request `LOCALJEV_LOGPROBS_K`; works when the engine's
     logprobs reflect `logit_bias` (post-bias) or the labels fall inside top-K.
   - **`selective`** (vLLM/SGLang default): ask the engine for logprobs of exactly
     the label token ids through its native API — vLLM `logprob_token_ids` on
     `POST /v1/completions` (requires **vLLM ≥ 0.26.0**, vllm-project/vllm#43463)
     or SGLang `token_ids_logprob` on `POST /generate`. One request, smallest
     payload, no bias dependence. If the engine or endpoint is unavailable,
     LocalJev logs a one-time warning and falls back to `bias`.
4. restore the restricted softmax over labels from **logprob differences** / raw
   label logprobs (a shared bias term cancels; labels that are absent get
   probability 0). Labels are matched by **token id**, because several ids can
   decode to the same label string;
5. calculate Jev-compatible choices (`choice` = best client key), expected scores
   (`Σ i·pᵢ` with legend), normalized entropy confidence, and return the normal
   Jev response shape.

Engines differ in how they expose logprobs, so the effective readout differs per
backend (see the table below). The returned probabilities are the raw next-token
distribution in every case; the strategies are not interchangeable in cost, though:
`vocab` returns the whole vocabulary, `bias`/`selective` return a handful of
tokens. `GET /ready` reports the configured `backend`, the `readout`, whether the
backend `selectiveSupported`, and any fallback `warning`.

Run `bun run verify` against your server to confirm which strategy it needs:
it reports whether `logit_bias` is reflected in the logprobs and whether every
label is readable under the configured readout.

One question is one read (autoregressive decoding cannot fill several slots in
one forward pass); questions run in parallel under `LOCALJEV_MAX_INFLIGHT`, and
waiting decisions beyond `LOCALJEV_MAX_QUEUE` get HTTP 529.

These probabilities are **uncalibrated next-token distributions** from a
generative model, not trained confidence. Evaluate calibration on your own
workload before relying on them for consequential decisions.

## Run

Requires Bun 1.2+ and a running llama-server (or any OpenAI-compatible server
that exposes `logprobs` and `logit_bias` on `/v1/completions`).

```sh
bun install
cp .env.example .env   # adjust LOCALJEV_UPSTREAM / LOCALJEV_UPSTREAM_MODEL if needed
bun run start
```

Bun loads `.env` automatically. Check readiness:

```bash
curl http://127.0.0.1:8081/ready
```

Make a decision:

```bash
curl http://127.0.0.1:8081/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "jev-latest",
    "state": "Hi, I have been trying to connect Stripe but keep getting a 403 error.",
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
          "billing": "Payment or subscription issues",
          "technical": "Bugs or integration problems",
          "sales": "Pricing or account questions"
        }
      },
      "frustration": {
        "type": "score",
        "instructions": "How frustrated does the customer appear?",
        "criteria": ["Calm", "Frustrated but civil", "Very angry"]
      },
      "urgent": {
        "type": "noul",
        "instructions": "Does this require an immediate response?"
      }
    }
  }'
```

Optional request field: `"permute": true` averages two reads with the option→label
mapping reversed (position-bias mitigation at 2× cost; `LOCALJEV_PERMUTE_DEFAULT`
enables it globally).

## Images and audio

Jev's own contract is text-only, so LocalJev carries media in two optional request
fields next to `state`:

```json
{
  "state": "Look at the photo and the call recording.",
  "images": ["data:image/png;base64,...."],
  "audio": ["data:audio/wav;base64,...."],
  "questions": { "mood": { "type": "choice", "instructions": "How does the caller sound?", "criteria": { "calm": "relaxed", "upset": "frustrated" } } }
}
```

- `images` accepts image URLs or `data:image/...;base64,` URLs.
- `audio` accepts `data:audio/<format>;base64,` URLs (for example `wav`, `mp3`).
- Media are placed **before** the state text, and each question whose read
  includes media is sent to `POST /v1/chat/completions` as a user message with
  content parts (`image_url`, `input_audio`, then the text prompt). Text-only
  questions keep using `POST /v1/completions`.
- Thinking must not run before the label token, so set `LOCALJEV_DISABLE_THINKING=true`
  for reasoning models that would otherwise emit a thought block. This mirrors
  the classification readout: the first generated token has to be the label.
- **Audio requires a non-llama.cpp backend.** llama.cpp does not serve audio
  input; audio requests against `LOCALJEV_BACKEND=llamacpp` return HTTP 400. Images
  work with llama.cpp when the server is started with a projector (`--mmproj`).

## Other inference backends

The readout is OpenAI-shaped, so vLLM, SGLang, and other OpenAI-compatible
servers work by setting `LOCALJEV_BACKEND`:

| Value | `logit_bias` keys | logprobs shape | `/tokenize` | default readout | audio |
|---|---|---|---|---|---|
| `llamacpp` (default) | token **strings** (`"A"`) | OpenAI `content[0].top_logprobs` | root `/tokenize`, `{content}` | `vocab` | no |
| `vllm` | token **IDs** as strings (`"32"`) | legacy `top_logprobs` list of maps | root `/tokenize`, `{model, prompt}` | `selective` | yes |
| `sglang` | token **IDs** as strings (`"32"`) | legacy `top_logprobs` list of maps | root `/tokenize`, `{model, prompt}` | `selective` | yes |
| `openai` | token **IDs** as strings (`"32"`) | legacy `top_logprobs` list of maps | root `/tokenize`, `{model, prompt}` | `bias` | yes |

LocalJev resolves each label to a single tokenizer token with `/tokenize`, then
reads that label's logprob from the first-token distribution (matching by token
id, since a label string can map to several ids). `token_id:NNN` logprob keys are
also recognized.

- **llama.cpp**: uses `vocab`. Stock llama-server computes `/v1` top logprobs from
  **pre-sampling** `log(softmax(logits))` and therefore **ignores `logit_bias`**
  (ggml-org/llama.cpp#10783). `vocab` asks for the full vocabulary so every label
  is present; no patch is needed. A single read is larger (≈26 MB for a 262144-token
  vocabulary) — lower `LOCALJEV_MAX_INFLIGHT` if RAM is tight.
- **vLLM / SGLang**: use `selective`. LocalJev asks for logprobs of exactly the
  label token ids in one request — SGLang `/generate` with `token_ids_logprob`, or
  vLLM `/v1/completions` with `logprob_token_ids` (**requires vLLM ≥ 0.26.0**;
  vllm-project/vllm#43463, merged 2026-07-13). Older vLLM silently ignores the
  field, so LocalJev detects the missing labels, logs a one-time warning, and
  falls back to `bias`. This avoids vLLM v1's raw-logprobs/bias pitfall, the
  max-logprobs limit, and the `/generative_scoring` speculative-decoding crash
  (vllm-project/vllm#42592).
  Note: vLLM returns selected-token logprobs at reduced precision for tokens
  outside the natural top-k (measured ≈0.125 step at logprob ≈ −20, i.e. bf16
  resolution), so labels whose true probabilities differ by less than that step
  come back tied and the choice falls to tie-break order. SGLang returns finer
  (fp32) values. Ties only affect near-equal labels, which LocalJev reports with
  low confidence.
- **OpenAI**: uses `bias` (no selective endpoint). OpenAI caps `top_logprobs` at
  20 and does not return token ids, so it cannot use `vocab`; results are
  best-effort.

The readout differs by engine, so cost and semantics are not identical:
`vocab` is a full-vocabulary read (exact pre-sampling distribution, heavy);
`bias`/`selective` return only a few tokens. `bias` depends on the server's
post-bias logprob semantics; `selective`/`vocab` do not. `bun run verify` reports
which applies to your server.

## Use the TypeSafe SDK

The SDK requires an API-key value. LocalJev accepts any value unless `LOCALJEV_API_KEY` is
configured. Set the SDK environment for your shell:

```sh
# bash/zsh
export TYPESAFE_BASE_URL=http://127.0.0.1:8081
export TYPESAFE_API_KEY=local
```

```python
from typesafe_sdk import TypeSafeClient

client = TypeSafeClient()
response = client.system_one(
    "I was charged twice this month.",
    {
        "billing": {
            "type": "noul",
            "instructions": "Is this a billing issue?",
        }
    },
)
print(response.nouls["billing"].noul)
```

`jev-latest` and `jev-preview` are accepted aliases so SDK defaults work
unchanged.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `LOCALJEV_UPSTREAM` | `http://127.0.0.1:8000` | llama-server base URL, with or without `/v1` |
| `LOCALJEV_UPSTREAM_API_KEY` | empty | Bearer key sent to the inference server |
| `LOCALJEV_UPSTREAM_MODEL` | empty | Upstream model id; when empty, the first model from upstream `/v1/models` |
| `LOCALJEV_API_KEY` | empty | Optional Bearer key required from LocalJev clients |
| `LOCALJEV_BACKEND` | `llamacpp` | Readout dialect: `llamacpp`, `vllm`, `sglang`, or `openai` (see below) |
| `LOCALJEV_READOUT` | `auto` | Label readout: `auto` (backend default), `bias`, `vocab`, or `selective` |
| `LOCALJEV_DISABLE_THINKING` | `false` | Send `chat_template_kwargs: {"enable_thinking": false}` on media (chat) reads |
| `LOCALJEV_HOST` | `127.0.0.1` | Listen address |
| `LOCALJEV_PORT` | `8081` | Listen port |
| `LOCALJEV_TIMEOUT` | `60` | Upstream timeout in seconds |
| `LOCALJEV_MAX_INFLIGHT` | `4` | Concurrent upstream reads |
| `LOCALJEV_MAX_QUEUE` | `64` | Waiting decisions before HTTP 529 |
| `LOCALJEV_LABEL_BIAS` | `10` | Additive logit_bias applied to every label token (`bias` readout only) |
| `LOCALJEV_LOGPROBS_K` | `64` | Top-K size requested from upstream logprobs (`bias` readout only) |
| `LOCALJEV_PERMUTE_DEFAULT` | `false` | Average reversed-label reads by default (2× cost) |

## Development

```bash
bun install
bun test
bun run typecheck
bun run smoke         # live decision through the Engine against your llama-server
bun run verify        # R1/R2/R3 readout checks against a live llama-server
```

`verify` answers the implementation-critical questions against your actual server
build: whether `logit_bias` is reflected in the returned logprobs (R1), whether
every label is readable under the configured readout (R2), and whether every
label candidate is a single tokenizer token (R3). Stock llama-server computes
`/v1` top logprobs from **pre-sampling** `log(softmax(logits))`, so `logit_bias`
has no effect on them; LocalJev therefore reads llama.cpp with the `vocab`
strategy (the returned `logprobs` list is clamped to `n_vocab`), with no patch.

## Evaluate different models

The repeatable bake-off uses public gold labels for news categorization (AG News),
yes/no reading comprehension (BoolQ), and five-level sentiment (SST-5). It runs
the same LocalJev engine against multiple installed models, comparing quality,
calibration, and full-decision latency at two actual input lengths.

```sh
# Quick integration check (30 requests, not a meaningful quality sample)
bun run eval --out eval/runs/pilot --limit 3

# 120 labeled examples × 2 input lengths per model
bun run eval --out eval/runs/my-bakeoff

# Regenerate a completed or partial report without running inference
bun run eval:report eval/runs/my-bakeoff
```

Requires a running llama-server serving the eval models; no running LocalJev HTTP
server or Python is needed. See [the evaluation guide](docs/evaluation.md) for
pinned data sources, methodology, configuration, resuming runs, and limitations.

## Deploy

`Dockerfile` builds the LocalJev image (no GPU or CUDA needed; it only speaks HTTP to
the inference server). `deploy/testllm/Dockerfile` builds a self-contained Gemma 4
E2B test backend on llama.cpp, including the multimodal projector. Kubernetes/Podman
manifests live in [`deploy/`](deploy/README.md).

```sh
podman build -t localhost/localjev:latest .
podman build -f deploy/testllm/Dockerfile -t localhost/testllm:latest .
```
