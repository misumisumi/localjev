export type EngineKind = "llamacpp" | "vllm" | "sglang" | "openai";

// How LocalJev guarantees it can read every answer label's logprob:
//  - "bias":      apply the same additive logit_bias to every label and request
//                 the configured top-K. Correct only when the engine returns
//                 logprobs that already reflect logit_bias (post-bias) or when
//                 the labels naturally fall inside top-K.
//  - "vocab":     request the whole vocabulary (`logprobs` capped at n_vocab by
//                 the server) and read each label directly, without relying on
//                 logit_bias. Required for stock llama.cpp, whose `/v1` logprobs
//                 are computed from pre-sampling logits and therefore ignore
//                 logit_bias (PR ggml-org/llama.cpp#10783).
//  - "selective": ask the engine for logprobs of exactly the label token ids via
//                 its native API (vLLM `logprob_token_ids`, SGLang
//                 `token_ids_logprob`). Smallest payload and no bias dependence,
//                 but not part of the OpenAI-compatible surface.
export type ReadoutMode = "bias" | "vocab" | "selective";

// Sentinel requested in "vocab" mode; servers clamp it to their vocabulary size
// (llama.cpp returned n_vocab for 512000/1<<20 requests). Chosen well above
// current model vocabularies (Gemma 4: 262144, Qwen3: ~152k, Llama 3: 128k).
export const FULL_VOCAB_LOGPROBS = 1 << 20;

export class MediaUnsupportedError extends Error {}

export interface LabelToken {
  label: string;
  id: number;
}

export interface SelectiveRequest {
  root: boolean;
  path: string;
  body: unknown;
}

// Engine-native read of logprobs for specific token ids. `request` returns one
// or more upstream requests (vLLM needs one call per label); `parse` maps the
// label id to its logprob. Throwing from parse signals "unsupported".
export interface SelectiveRead {
  request(prompt: string, labels: LabelToken[], model: string): SelectiveRequest[];
  parse(responses: unknown[], labels: LabelToken[]): Map<number, number>;
}

export interface MediaParts {
  images: string[];
  audio: string[];
}

export interface ReadRequest {
  path: string;
  body: Record<string, unknown>;
}

export interface TokenizeRequest {
  root: boolean;
  path: string;
  body: Record<string, unknown>;
}

export interface LogprobEntry {
  token: string;
  logprob: number;
  // Tokenizer id when the upstream exposes one (llama.cpp `id`, vLLM/SGLang
  // legacy maps keyed by token-id strings). Matching by id is required because
  // several token ids can decode to the same label string.
  id?: number;
}

export interface ReadLogprobs {
  generated?: LogprobEntry;
  entries: LogprobEntry[];
}

export interface Dialect {
  readonly kind: EngineKind;
  readonly openai: boolean;
  readonly supportsAudio: boolean;
  readonly defaultReadout: ReadoutMode;
  readonly selective?: SelectiveRead;
  tokenize(label: string, model: string): TokenizeRequest;
  tokenIds(payload: unknown): number[] | null;
  completion(
    prompt: string,
    labels: LabelToken[],
    model: string,
    logprobsK: number,
    bias: number,
    readout: ReadoutMode,
  ): ReadRequest;
  chat(
    prompt: string,
    labels: LabelToken[],
    media: MediaParts,
    model: string,
    logprobsK: number,
    bias: number,
    disableThinking: boolean,
    readout: ReadoutMode,
  ): ReadRequest;
}

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function tokensFrom(payload: unknown): number[] | null {
  if (!record(payload) || !Array.isArray(payload.tokens)) return null;
  return payload.tokens.filter((token): token is number => typeof token === "number");
}

export function audioPart(url: string): Record<string, unknown> {
  const match = /^data:(audio\/[a-z0-9.+-]+);base64,(.+)$/is.exec(url);
  if (!match) {
    throw new MediaUnsupportedError(
      "audio must be a data:audio/<format>;base64 URL (e.g. data:audio/wav;base64,...)",
    );
  }
  return {
    type: "input_audio",
    input_audio: { data: match[2]!, format: match[1]!.slice("audio/".length).toLowerCase() },
  };
}

function mediaContent(media: MediaParts): Record<string, unknown>[] {
  const parts: Record<string, unknown>[] = [];
  for (const url of media.images) {
    parts.push({ type: "image_url", image_url: { url } });
  }
  for (const url of media.audio) {
    parts.push(audioPart(url));
  }
  return parts;
}

function biasObject(
  labels: LabelToken[],
  bias: number,
  key: (entry: LabelToken) => string,
): Record<string, number> {
  return Object.fromEntries(labels.map((entry) => [key(entry), bias]));
}

// In "vocab" mode request the whole vocabulary; local servers clamp this to
// n_vocab. `logit_bias` is omitted because it is not reflected in pre-sampling
// logprobs and the readout no longer depends on it.
function readoutLogprobs(readout: ReadoutMode, logprobsK: number): number {
  return readout === "vocab" ? FULL_VOCAB_LOGPROBS : logprobsK;
}

function biasField(
  readout: ReadoutMode,
  labels: LabelToken[],
  bias: number,
  key: (entry: LabelToken) => string,
): Record<string, unknown> {
  return readout === "bias" ? { logit_bias: biasObject(labels, bias, key) } : {};
}

function chatBody(
  prompt: string,
  labels: LabelToken[],
  media: MediaParts,
  model: string,
  logprobsK: number,
  bias: number,
  disableThinking: boolean,
  readout: ReadoutMode,
  key: (entry: LabelToken) => string,
): Record<string, unknown> {
  return {
    model,
    messages: [
      {
        role: "user",
        content: [...mediaContent(media), { type: "text", text: prompt }],
      },
    ],
    max_tokens: 1,
    temperature: 0,
    logprobs: true,
    top_logprobs: readoutLogprobs(readout, logprobsK),
    ...biasField(readout, labels, bias, key),
    ...(disableThinking
      ? { chat_template_kwargs: { enable_thinking: false } }
      : {}),
  };
}

const llamacpp: Dialect = {
  kind: "llamacpp",
  openai: false,
  supportsAudio: false,
  // Stock llama.cpp returns pre-sampling logprobs that ignore logit_bias, so it
  // reads the full vocabulary (no patch required).
  defaultReadout: "vocab",
  tokenize: (label) => ({ root: true, path: "/tokenize", body: { content: label } }),
  tokenIds: tokensFrom,
  completion: (prompt, labels, model, logprobsK, bias, readout) => ({
    path: "/completions",
    body: {
      model,
      prompt,
      max_tokens: 1,
      temperature: 0,
      logprobs: readoutLogprobs(readout, logprobsK),
      ...biasField(readout, labels, bias, (entry) => entry.label),
    },
  }),
  chat: (prompt, labels, media, model, logprobsK, bias, disableThinking, readout) => ({
    path: "/chat/completions",
    body: chatBody(
      prompt,
      labels,
      media,
      model,
      logprobsK,
      bias,
      disableThinking,
      readout,
      (entry) => entry.label,
    ),
  }),
};

// SGLang native endpoint: one request returns logprobs for every requested id.
// `token_ids_logprob` is a flat list of token ids. The response is
// `output_token_ids_logprobs[0]`, a list of [logprob, token_id, null] tuples
// (older builds emit [logprobs[], token_ids[], null]); both are handled.
function sglangSelective(): SelectiveRead {
  return {
    request: (prompt, labels) => [
      {
        root: true,
        path: "/generate",
        body: {
          text: prompt,
          sampling_params: { max_new_tokens: 1, temperature: 0 },
          return_logprob: true,
          token_ids_logprob: labels.map((label) => label.id),
        },
      },
    ],
    parse: (responses) => {
      const map = new Map<number, number>();
      const response = responses[0];
      const meta =
        record(response) && record(response.meta_info) ? response.meta_info : undefined;
      const perToken =
        record(meta) && Array.isArray(meta.output_token_ids_logprobs)
          ? meta.output_token_ids_logprobs[0]
          : undefined;
      if (!Array.isArray(perToken)) return map;
      for (const entry of perToken) {
        if (!Array.isArray(entry)) continue;
        const [first, second] = entry;
        if (Array.isArray(first) && Array.isArray(second)) {
          // [logprobs[], token_ids[], null]
          for (let index = 0; index < second.length; index += 1) {
            const id = tokenId(second[index]);
            const value = first[index];
            if (id !== undefined && typeof value === "number") map.set(id, value);
          }
        } else {
          // [logprob, token_id, null]
          const id = tokenId(second);
          if (id !== undefined && typeof first === "number") map.set(id, first);
        }
      }
      return map;
    },
  };
}

// vLLM exposes `logprob_token_ids` on the OpenAI-compatible `/v1/completions`
// route (vllm-project/vllm#43463), so one request returns the logprob of every
// label id even when they are far outside the natural top-k. Older servers that
// predate #43463 silently ignore the field; the response then omits some labels
// and the engine falls back to `bias` with a warning.
function vllmSelective(): SelectiveRead {
  return {
    request: (prompt, labels, model) => [
      {
        root: false,
        path: "/completions",
        body: {
          model,
          prompt,
          max_tokens: 1,
          temperature: 0,
          logprobs: labels.length,
          logprob_token_ids: labels.map((label) => label.id),
        },
      },
    ],
    parse: (responses, labels) => {
      const parsed = collectLogprobs(responses[0]);
      const byToken = new Map<string, number>();
      const byId = new Map<number, number>();
      const add = (entry: LogprobEntry) => {
        byToken.set(entry.token, entry.logprob);
        if (entry.id !== undefined) byId.set(entry.id, entry.logprob);
        const match = /^token_id:(\d+)$/.exec(entry.token);
        if (match) byId.set(Number(match[1]), entry.logprob);
      };
      if (parsed.generated) add(parsed.generated);
      for (const entry of parsed.entries) add(entry);
      const map = new Map<number, number>();
      for (const label of labels) {
        const value = byId.get(label.id) ?? byToken.get(label.label);
        if (value !== undefined) map.set(label.id, value);
      }
      return map;
    },
  };
}

function openaiDialect(kind: EngineKind): Dialect {
  const selective =
    kind === "sglang" ? sglangSelective() : kind === "vllm" ? vllmSelective() : undefined;
  return {
    kind,
    openai: true,
    supportsAudio: true,
    defaultReadout: kind === "openai" ? "bias" : "selective",
    ...(selective ? { selective } : {}),
    tokenize: (label, model) => ({ root: true, path: "/tokenize", body: { model, prompt: label } }),
    tokenIds: tokensFrom,
    completion: (prompt, labels, model, logprobsK, bias, readout) => ({
      path: "/completions",
      body: {
        model,
        prompt,
        max_tokens: 1,
        temperature: 0,
        logprobs: readoutLogprobs(readout, logprobsK),
        ...biasField(readout, labels, bias, (entry) => String(entry.id)),
      },
    }),
    chat: (prompt, labels, media, model, logprobsK, bias, disableThinking, readout) => ({
      path: "/chat/completions",
      body: chatBody(
        prompt,
        labels,
        media,
        model,
        logprobsK,
        bias,
        disableThinking,
        readout,
        (entry) => String(entry.id),
      ),
    }),
  };
}

export function resolveDialect(kind: EngineKind): Dialect {
  if (kind === "llamacpp") return llamacpp;
  return openaiDialect(kind);
}

function tokenId(value: unknown): number | undefined {
  return typeof value === "number" && Number.isInteger(value) && value >= 0 ? value : undefined;
}

function entryOf(value: unknown): LogprobEntry | undefined {
  if (!record(value)) return undefined;
  if (typeof value.token !== "string" || typeof value.logprob !== "number") return undefined;
  const id = tokenId(value.id);
  return id === undefined
    ? { token: value.token, logprob: value.logprob }
    : { token: value.token, logprob: value.logprob, id };
}

function entriesFromList(list: unknown): LogprobEntry[] {
  if (!Array.isArray(list)) return [];
  const entries: LogprobEntry[] = [];
  for (const item of list) {
    const entry = entryOf(item);
    if (entry) entries.push(entry);
  }
  return entries;
}

// Legacy OpenAI completions shape (vLLM/SGLang): a map from token (often the
// token id as a string) to logprob. Preserve a numeric key as the token id so
// the engine can match labels by id instead of the (ambiguous) decoded string.
function entriesFromMap(map: unknown): LogprobEntry[] {
  if (!record(map)) return [];
  const entries: LogprobEntry[] = [];
  for (const [token, logprob] of Object.entries(map)) {
    if (typeof logprob !== "number") continue;
    const id = /^\d+$/.test(token) ? tokenId(Number(token)) : undefined;
    entries.push(id === undefined ? { token, logprob } : { token, logprob, id });
  }
  return entries;
}

export function collectLogprobs(payload: unknown): ReadLogprobs {
  const choices = record(payload) && Array.isArray(payload.choices) ? payload.choices : [];
  const choice = record(choices[0]) ? choices[0] : undefined;
  const logprobs = choice && record(choice.logprobs) ? choice.logprobs : undefined;
  if (!logprobs) return { entries: [] };

  if (Array.isArray(logprobs.content)) {
    const first = record(logprobs.content[0]) ? logprobs.content[0] : undefined;
    if (!first) return { entries: [] };
    const generated = entryOf(first);
    return {
      ...(generated ? { generated } : {}),
      entries: entriesFromList(first.top_logprobs),
    };
  }

  if (Array.isArray(logprobs.top_logprobs)) {
    const first = logprobs.top_logprobs[0];
    const entries = Array.isArray(first) ? entriesFromList(first) : entriesFromMap(first);
    const tokens = Array.isArray(logprobs.tokens) ? logprobs.tokens : [];
    const tokenLogprobs = Array.isArray(logprobs.token_logprobs) ? logprobs.token_logprobs : [];
    const generated =
      typeof tokens[0] === "string" && typeof tokenLogprobs[0] === "number"
        ? { token: tokens[0], logprob: tokenLogprobs[0] }
        : undefined;
    return { ...(generated ? { generated } : {}), entries };
  }

  return { entries: [] };
}
