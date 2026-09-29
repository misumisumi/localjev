export type EngineKind = "llamacpp" | "vllm" | "sglang" | "openai";

export class MediaUnsupportedError extends Error {}

export interface LabelToken {
  label: string;
  id: number;
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
}

export interface ReadLogprobs {
  generated?: LogprobEntry;
  entries: LogprobEntry[];
}

export interface Dialect {
  readonly kind: EngineKind;
  readonly openai: boolean;
  readonly supportsAudio: boolean;
  tokenize(label: string, model: string): TokenizeRequest;
  tokenIds(payload: unknown): number[] | null;
  completion(
    prompt: string,
    labels: LabelToken[],
    model: string,
    logprobsK: number,
    bias: number,
  ): ReadRequest;
  chat(
    prompt: string,
    labels: LabelToken[],
    media: MediaParts,
    model: string,
    logprobsK: number,
    bias: number,
    disableThinking: boolean,
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

function chatBody(
  prompt: string,
  labels: LabelToken[],
  media: MediaParts,
  model: string,
  logprobsK: number,
  bias: number,
  disableThinking: boolean,
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
    top_logprobs: logprobsK,
    logit_bias: biasObject(labels, bias, key),
    ...(disableThinking
      ? { chat_template_kwargs: { enable_thinking: false } }
      : {}),
  };
}

const llamacpp: Dialect = {
  kind: "llamacpp",
  openai: false,
  supportsAudio: false,
  tokenize: (label) => ({ root: false, path: "/tokenize", body: { content: label } }),
  tokenIds: tokensFrom,
  completion: (prompt, labels, model, logprobsK, bias) => ({
    path: "/completions",
    body: {
      model,
      prompt,
      max_tokens: 1,
      temperature: 0,
      logprobs: logprobsK,
      logit_bias: biasObject(labels, bias, (entry) => entry.label),
    },
  }),
  chat: (prompt, labels, media, model, logprobsK, bias, disableThinking) => ({
    path: "/chat/completions",
    body: chatBody(prompt, labels, media, model, logprobsK, bias, disableThinking, (entry) => entry.label),
  }),
};

function openaiDialect(kind: EngineKind): Dialect {
  return {
    kind,
    openai: true,
    supportsAudio: true,
    tokenize: (label, model) => ({ root: true, path: "/tokenize", body: { model, prompt: label } }),
    tokenIds: tokensFrom,
    completion: (prompt, labels, model, logprobsK, bias) => ({
      path: "/completions",
      body: {
        model,
        prompt,
        max_tokens: 1,
        temperature: 0,
        logprobs: logprobsK,
        logit_bias: biasObject(labels, bias, (entry) => String(entry.id)),
      },
    }),
    chat: (prompt, labels, media, model, logprobsK, bias, disableThinking) => ({
      path: "/chat/completions",
      body: chatBody(prompt, labels, media, model, logprobsK, bias, disableThinking, (entry) =>
        String(entry.id),
      ),
    }),
  };
}

export function resolveDialect(kind: EngineKind): Dialect {
  if (kind === "llamacpp") return llamacpp;
  return openaiDialect(kind);
}

function entryOf(value: unknown): LogprobEntry | undefined {
  if (!record(value)) return undefined;
  if (typeof value.token !== "string" || typeof value.logprob !== "number") return undefined;
  return { token: value.token, logprob: value.logprob };
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

function entriesFromMap(map: unknown): LogprobEntry[] {
  if (!record(map)) return [];
  const entries: LogprobEntry[] = [];
  for (const [token, logprob] of Object.entries(map)) {
    if (typeof logprob === "number") entries.push({ token, logprob });
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
