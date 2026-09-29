# QEv

A local, Jev-compatible `POST /v1/systemone` API written in TypeScript for
[Bun](https://bun.sh/). QEv reads probabilities from **first-token logprobs of a
local llama-server** instead of asking a model to write JSON probabilities.

The defaults target:

- inference server: `http://127.0.0.1:8000` (llama-server, unpatched)
- QEv API: `http://127.0.0.1:8081`

## How the read works

[Jev](https://typesafe.ai/) uses a typed decision API rather than an OpenAI chat
API. OpenJev obtains probabilities with a one-step DiffusionGemma structured read
over patched vLLM extensions. QEv gets equivalent first-token probabilities from
an **unpatched** llama-server:

1. translate `state` and typed Jev questions into a plain classification prompt
   (state first so the KV prefix is reusable across questions; client question
   keys never reach the prompt);
2. map each option to a single-token label — `A..Z` for choice, `0..9` for score,
   `yes`/`no` for noul — validated once against `/v1/tokenize`;
3. send `POST /v1/completions` with `max_tokens: 1`, `logprobs: K`, and the **same
   additive `logit_bias` on every label token**, so all labels are guaranteed to
   appear in the returned top-K;
4. restore the restricted softmax over labels from **logprob differences** (the
   identical bias term cancels; labels absent from top-K get probability 0);
5. calculate Jev-compatible choices (`choice` = best client key), expected scores
   (`Σ i·pᵢ` with legend), normalized entropy confidence, and return the normal
   Jev response shape.

One question is one read (autoregressive decoding cannot fill several slots in
one forward pass); questions run in parallel under `QEV_MAX_INFLIGHT`, and
waiting decisions beyond `QEV_MAX_QUEUE` get HTTP 529.

These probabilities are **uncalibrated next-token distributions** from a
generative model, not trained confidence. Evaluate calibration on your own
workload before relying on them for consequential decisions.

## Run

Requires Bun 1.2+ and a running llama-server (or any OpenAI-compatible server
that exposes `logprobs` and `logit_bias` on `/v1/completions`).

```sh
bun install
cp .env.example .env   # adjust QEV_UPSTREAM / QEV_UPSTREAM_MODEL if needed
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
mapping reversed (position-bias mitigation at 2× cost; `QEV_PERMUTE_DEFAULT`
enables it globally).

## Images and audio

Jev's own contract is text-only, so QEv carries media in two optional request
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
- Thinking must not run before the label token, so set `QEV_DISABLE_THINKING=true`
  for reasoning models that would otherwise emit a thought block. This mirrors
  the classification readout: the first generated token has to be the label.
- **Audio requires a non-llama.cpp backend.** llama.cpp does not serve audio
  input; audio requests against `QEV_BACKEND=llamacpp` return HTTP 400. Images
  work with llama.cpp when the server is started with a projector (`--mmproj`).

## Other inference backends

The readout is OpenAI-shaped, so vLLM, SGLang, and other OpenAI-compatible
servers work by setting `QEV_BACKEND`:

| Value | `logit_bias` keys | logprobs shape | `/tokenize` | audio |
|---|---|---|---|---|
| `llamacpp` (default) | token **strings** (`"A"`) | OpenAI `content[0].top_logprobs` | `/v1/tokenize`, `{content}` | no |
| `vllm`, `sglang`, `openai` | token **IDs** as strings (`"32"`) | legacy `top_logprobs` list of maps | root `/tokenize`, `{model, prompt}` | yes |

QEv resolves each label to a single tokenizer token with `/tokenize`, then forces
the labels into the returned logprobs with the same `logit_bias` value on every
label (the bias cancels in the softmax over labels). `token_id:NNN` logprob keys
are also recognized.

## Use the TypeSafe SDK

The SDK requires an API-key value. QEv accepts any value unless `QEV_API_KEY` is
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
| `QEV_UPSTREAM` | `http://127.0.0.1:8000` | llama-server base URL, with or without `/v1` |
| `QEV_UPSTREAM_API_KEY` | empty | Bearer key sent to the inference server |
| `QEV_UPSTREAM_MODEL` | empty | Upstream model id; when empty, the first model from upstream `/v1/models` |
| `QEV_API_KEY` | empty | Optional Bearer key required from QEv clients |
| `QEV_BACKEND` | `llamacpp` | Readout dialect: `llamacpp`, `vllm`, `sglang`, or `openai` (see below) |
| `QEV_DISABLE_THINKING` | `false` | Send `chat_template_kwargs: {"enable_thinking": false}` on media (chat) reads |
| `QEV_HOST` | `127.0.0.1` | Listen address |
| `QEV_PORT` | `8081` | Listen port |
| `QEV_TIMEOUT` | `60` | Upstream timeout in seconds |
| `QEV_MAX_INFLIGHT` | `4` | Concurrent upstream reads |
| `QEV_MAX_QUEUE` | `64` | Waiting decisions before HTTP 529 |
| `QEV_LABEL_BIAS` | `10` | Additive logit_bias applied to every label token |
| `QEV_LOGPROBS_K` | `64` | Top-K size requested from upstream logprobs |
| `QEV_PERMUTE_DEFAULT` | `false` | Average reversed-label reads by default (2× cost) |

## Development

```bash
bun install
bun test
bun run typecheck
bun run smoke         # live decision through the Engine against your llama-server
bun run verify        # R1/R2/R3 readout checks against a live llama-server
```

`verify` answers the three implementation-critical questions against your actual
server build: whether `logit_bias` cancels out in label logprob differences
(R1), whether all biased labels appear within top-K (R2), and whether every
label candidate is a single tokenizer token (R3). llama-server computes top
logprobs from **pre-sampling** `log(softmax(logits))`, so llama.cpp requires no
patch; it does not expose raw full-vocabulary logits over HTTP, and QEv does not
need them.

## Evaluate different models

The repeatable bake-off uses public gold labels for news categorization (AG News),
yes/no reading comprehension (BoolQ), and five-level sentiment (SST-5). It runs
the same QEv engine against multiple installed models, comparing quality,
calibration, and full-decision latency at two actual input lengths.

```sh
# Quick integration check (30 requests, not a meaningful quality sample)
bun run eval --out eval/runs/pilot --limit 3

# 120 labeled examples × 2 input lengths per model
bun run eval --out eval/runs/my-bakeoff

# Regenerate a completed or partial report without running inference
bun run eval:report eval/runs/my-bakeoff
```

Requires a running llama-server serving the eval models; no running QEv HTTP
server or Python is needed. See [the evaluation guide](docs/evaluation.md) for
pinned data sources, methodology, configuration, resuming runs, and limitations.
