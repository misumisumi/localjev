import type { Settings, ReadoutSetting } from "./config";
import { apiBaseUrl, upstreamRoot } from "./config";
import {
  collectLogprobs,
  MediaUnsupportedError,
  resolveDialect,
  type Dialect,
  type EngineKind,
  type LabelToken,
  type LogprobEntry,
  type MediaParts,
  type ReadoutMode,
} from "./backends";
import type { Answer, Described, JsonValue, Question } from "./types";

export { MediaUnsupportedError };

export class OverloadedError extends Error {}
export class BackendProtocolError extends Error {}
export class BackendUnavailableError extends Error {}
export class LabelMappingError extends Error {}

export class UpstreamHttpError extends Error {
  constructor(readonly status: number) {
    super(`inference backend returned HTTP ${status}`);
  }
}

interface PreparedQuestion {
  key: string;
  internalId: string;
  kind: Question["type"];
  instructions: Described;
  choices: [string, JsonValue | undefined][];
  legend?: JsonValue[];
}

export interface DecisionResult {
  answers: Record<string, Answer>;
  inputTokens: number;
  outputTokens: number;
}

export interface DecisionOptions {
  permute?: boolean;
  images?: string[];
  audio?: string[];
}

export interface DecisionEngine {
  decide(
    questions: Record<string, Question>,
    state: JsonValue,
    seed: number,
    options?: DecisionOptions,
  ): Promise<DecisionResult>;
  ready?(): Promise<boolean>;
  close?(): Promise<void>;
  upstreamModelId?(): string | null;
  readoutStatus?(): {
    backend: string;
    configured: string;
    effective: string;
    selectiveSupported: boolean;
    warning: string | null;
  };
}

type Fetch = (input: string | URL | Request, init?: RequestInit) => Promise<Response>;

class Semaphore {
  private active = 0;
  private readonly waiters: (() => void)[] = [];

  constructor(private readonly maximum: number) {}

  async run<T>(operation: () => Promise<T>): Promise<T> {
    if (this.active < this.maximum) {
      this.active += 1;
    } else {
      await new Promise<void>((resolve) => this.waiters.push(resolve));
    }
    try {
      return await operation();
    } finally {
      const next = this.waiters.shift();
      if (next) next();
      else this.active -= 1;
    }
  }
}

function render(value: unknown): string {
  if (value === null || value === undefined) return "No additional instructions.";
  if (typeof value === "string") return value.trim() || "No additional instructions.";
  return JSON.stringify(value);
}

export function confidence(probabilities: number[]): number {
  const entropy = -probabilities.reduce(
    (sum, probability) =>
      probability > 0 ? sum + probability * Math.log(probability) : sum,
    0,
  );
  const value = 1 - entropy / Math.log(probabilities.length);
  return Math.max(0, Math.min(1, value));
}

export function prepareQuestions(
  questions: Record<string, Question>,
): PreparedQuestion[] {
  return Object.entries(questions).map(([key, question], index) => {
    if (question.type === "noul") {
      return {
        key,
        internalId: `q${index + 1}`,
        kind: question.type,
        instructions: question.instructions,
        choices: [
          ["yes", question.criteria?.true ?? undefined],
          ["no", question.criteria?.false ?? undefined],
        ],
      };
    }
    if (question.type === "choice") {
      return {
        key,
        internalId: `q${index + 1}`,
        kind: question.type,
        instructions: question.instructions,
        choices: Object.entries(question.criteria).map(([label, criterion]) => [
          label,
          criterion ?? undefined,
        ]),
      };
    }
    return {
      key,
      internalId: `q${index + 1}`,
      kind: question.type,
      instructions: question.instructions,
      choices: question.criteria.map((criterion, score) => [
        String(score),
        criterion,
      ]),
      legend: question.criteria,
    };
  });
}

export const CHOICE_LABELS = Array.from(
  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
);

export function labelsFor(question: PreparedQuestion): string[] {
  if (question.kind === "noul") return ["yes", "no"];
  if (question.kind === "score") {
    return question.choices.map((_, index) => String(index));
  }
  if (question.choices.length > CHOICE_LABELS.length) {
    throw new LabelMappingError(
      `A choice question supports at most ${CHOICE_LABELS.length} options in LocalJev ` +
        `(one single-token label per option); got ${question.choices.length}.`,
    );
  }
  return CHOICE_LABELS.slice(0, question.choices.length);
}

function stateMessage(state: JsonValue): string {
  const serialized = JSON.stringify(state)
    .replaceAll("<", "\\u003c")
    .replaceAll(">", "\\u003e");
  return `<document>\n${serialized}\n</document>`;
}

export function promptFor(
  state: JsonValue,
  question: PreparedQuestion,
  entries: [string, JsonValue | undefined][],
): string {
  const lines = [
    stateMessage(state),
    "",
    `Question: ${render(question.instructions)}`,
  ];
  const separator = question.kind === "noul" ? ": " : ". ";
  for (const [label, criterion] of entries) {
    lines.push(
      criterion === undefined ? label : `${label}${separator}${render(criterion)}`,
    );
  }
  lines.push("", "Answer: ");
  return lines.join("\n");
}

export function labelEntries(
  question: PreparedQuestion,
  permute: boolean,
): [string, JsonValue | undefined][] {
  const labels = labelsFor(question);
  const choices = permute ? [...question.choices].reverse() : question.choices;
  return choices.map(([, criterion], index) => [labels[index]!, criterion]);
}

export function softmaxFromLogprobs(values: (number | null)[]): number[] {
  const present = values.filter((value): value is number => value !== null);
  if (present.length === 0) {
    throw new BackendProtocolError(
      "no answer label appeared in the upstream logprobs; " +
        "increase LOCALJEV_LOGPROBS_K, or use LOCALJEV_READOUT=vocab " +
        "(stock llama.cpp ignores logit_bias in pre-sampling logprobs)",
    );
  }
  const maximum = Math.max(...present);
  const exponentials = values.map((value) =>
    value === null ? 0 : Math.exp(value - maximum),
  );
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map((value) => value / total);
}

export function averageProbabilities(vectors: number[][]): number[] {
  if (vectors.length === 1) return vectors[0]!;
  const size = vectors[0]!.length;
  return vectors[0]!.map((value, index) => {
    let sum = value;
    // Permuted reads map the same outcomes to labels in reverse order.
    for (let variant = 1; variant < vectors.length; variant += 1) {
      sum += vectors[variant]![size - 1 - index] ?? 0;
    }
    return sum / vectors.length;
  });
}

export function formatAnswer(
  question: PreparedQuestion,
  probabilities: number[],
): Answer {
  if (question.kind === "noul") {
    return { type: "noul", noul: probabilities[0] ?? 0 };
  }
  const probabilityMap = Object.fromEntries(
    question.choices.map(([label], index) => [label, probabilities[index] ?? 0]),
  );
  const certainty = confidence(probabilities);
  if (question.kind === "choice") {
    let best = 0;
    for (let index = 1; index < probabilities.length; index += 1) {
      if ((probabilities[index] ?? 0) > (probabilities[best] ?? 0)) best = index;
    }
    return {
      type: "choice",
      choice: question.choices[best]?.[0] ?? "",
      probabilities: probabilityMap,
      confidence: certainty,
    };
  }
  return {
    type: "score",
    score: probabilities.reduce(
      (sum, probability, index) => sum + index * probability,
      0,
    ),
    legend: Object.fromEntries(
      (question.legend ?? []).map((item, index) => [String(index), item]),
    ),
    probabilities: probabilityMap,
    confidence: certainty,
  };
}

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function numeric(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

interface ReadResult {
  probabilities: number[];
  inputTokens: number;
  outputTokens: number;
}

export class Engine implements DecisionEngine {
  private readonly slots: Semaphore;
  private readonly dialect: Dialect;
  private waiting = 0;
  private resolvedModel: string | null = null;
  private modelPromise: Promise<string> | null = null;
  private readonly labelTokens = new Map<string, Promise<number>>();
  private effectiveReadoutValue: ReadoutMode | null = null;
  private selectiveWarning: string | null = null;

  constructor(
    private readonly settings: Settings,
    private readonly fetchImpl: Fetch = (input, init) => fetch(input, init),
  ) {
    this.slots = new Semaphore(settings.maxInflight);
    this.dialect = resolveDialect(settings.backend);
  }

  upstreamModelId(): string | null {
    return this.resolvedModel;
  }

  private upstreamHeaders(): HeadersInit {
    return {
      "content-type": "application/json",
      ...(this.settings.upstreamApiKey
        ? { authorization: `Bearer ${this.settings.upstreamApiKey}` }
        : {}),
    };
  }

  private async upstreamRequest(
    path: string,
    body?: unknown,
    root = false,
  ): Promise<unknown> {
    let response: Response;
    try {
      const base = root ? upstreamRoot(this.settings) : apiBaseUrl(this.settings);
      response = await this.slots.run(() =>
        this.fetchImpl(`${base}${path}`, {
          method: body === undefined ? "GET" : "POST",
          headers: this.upstreamHeaders(),
          ...(body === undefined ? {} : { body: JSON.stringify(body) }),
          signal: AbortSignal.timeout(this.settings.timeoutMs),
        }),
      );
    } catch (error) {
      throw new BackendUnavailableError(
        `inference backend unavailable: ${error instanceof Error ? error.name : "network error"}`,
      );
    }
    if (!response.ok) throw new UpstreamHttpError(response.status);
    try {
      return await response.json();
    } catch {
      throw new BackendProtocolError("upstream returned a non-JSON response");
    }
  }

  private resolveModel(): Promise<string> {
    if (this.settings.upstreamModel) {
      return Promise.resolve(this.settings.upstreamModel);
    }
    this.modelPromise ??= (async () => {
      const payload = await this.upstreamRequest("/models");
      const data = record(payload) && Array.isArray(payload.data) ? payload.data : [];
      const first = data.find((model) => record(model) && typeof model.id === "string");
      if (!first) {
        throw new BackendProtocolError("upstream /v1/models listed no model");
      }
      this.resolvedModel = String(first.id);
      return this.resolvedModel;
    })();
    return this.modelPromise;
  }

  async ready(): Promise<boolean> {
    const payload = await this.upstreamRequest("/models");
    if (!record(payload) || !Array.isArray(payload.data)) return false;
    if (!this.settings.upstreamModel) return payload.data.length > 0;
    return payload.data.some(
      (model) => record(model) && model.id === this.settings.upstreamModel,
    );
  }

  private labelToken(label: string): Promise<number> {
    let pending = this.labelTokens.get(label);
    if (!pending) {
      pending = (async () => {
        const request = this.dialect.tokenize(label, await this.resolveModel());
        const payload = await this.upstreamRequest(request.path, request.body, request.root);
        const tokens = this.dialect.tokenIds(payload);
        if (!tokens) {
          throw new BackendProtocolError("upstream /tokenize did not return a token list");
        }
        if (tokens.length !== 1) {
          throw new LabelMappingError(
            `label ${JSON.stringify(label)} is not a single token in the upstream tokenizer`,
          );
        }
        return tokens[0]!;
      })();
      this.labelTokens.set(label, pending);
    }
    return pending;
  }

  private readout(): ReadoutMode {
    return this.settings.readout === "auto" ? this.dialect.defaultReadout : this.settings.readout;
  }

  // Observability for `/ready`: the configured backend/readout, whether the
  // backend exposes a selective API, and any one-time fallback warning.
  readoutStatus(): {
    backend: EngineKind;
    configured: ReadoutSetting;
    effective: ReadoutMode;
    selectiveSupported: boolean;
    warning: string | null;
  } {
    return {
      backend: this.settings.backend,
      configured: this.settings.readout,
      effective: this.effectiveReadoutValue ?? this.readout(),
      selectiveSupported: Boolean(this.dialect.selective),
      warning: this.selectiveWarning,
    };
  }

  private noteSelectiveFallback(detail: string): void {
    const message = `LOCALJEV_READOUT=selective unavailable, falling back to bias: ${detail}`;
    if (this.selectiveWarning !== message) {
      this.selectiveWarning = message;
      console.warn(`[localjev] ${message}`);
    }
  }

  private async selectiveRead(
    prompt: string,
    labels: LabelToken[],
    model: string,
  ): Promise<ReadResult> {
    const spec = this.dialect.selective!;
    const requests = spec.request(prompt, labels, model);
    const responses: unknown[] = [];
    let inputTokens = 0;
    let outputTokens = 0;
    for (const request of requests) {
      const payload = await this.upstreamRequest(request.path, request.body, request.root);
      responses.push(payload);
      const usage = record(payload) && record(payload.usage) ? payload.usage : undefined;
      const meta = record(payload) && record(payload.meta_info) ? payload.meta_info : undefined;
      inputTokens += numeric(usage?.prompt_tokens ?? usage?.input_tokens ?? meta?.prompt_tokens);
      outputTokens += numeric(
        usage?.completion_tokens ?? usage?.output_tokens ?? meta?.completion_tokens,
      );
    }
    const map = spec.parse(responses, labels);
    const values = labels.map((label) => map.get(label.id) ?? null);
    // A selective read is only valid if every requested label came back; a
    // partial result means the server ignored the selected-token request (e.g.
    // a vLLM that predates logprob_token_ids), so fall back instead of silently
    // treating the missing labels as zero.
    if (values.some((value) => value === null)) {
      throw new BackendProtocolError("selective readout did not return every label logprob");
    }
    this.effectiveReadoutValue = "selective";
    return { probabilities: softmaxFromLogprobs(values), inputTokens, outputTokens };
  }

  private async readSlot(
    prompt: string,
    labels: LabelToken[],
    media?: MediaParts,
  ): Promise<ReadResult> {
    const model = await this.resolveModel();
    const useChat = Boolean(media && (media.images.length > 0 || media.audio.length > 0));
    let readout = this.readout();

    if (readout === "selective") {
      if (!this.dialect.selective || useChat) {
        this.noteSelectiveFallback(
          useChat
            ? "selective readout is only available for text reads"
            : `backend ${this.settings.backend} exposes no selective logprob API`,
        );
        readout = "bias";
      } else {
        try {
          return await this.selectiveRead(prompt, labels, model);
        } catch (error) {
          this.noteSelectiveFallback(
            error instanceof Error ? error.message : "selective request failed",
          );
          readout = "bias";
        }
      }
    }

    this.effectiveReadoutValue = readout;
    const request = useChat
      ? this.dialect.chat(
          prompt,
          labels,
          media!,
          model,
          this.settings.logprobsK,
          this.settings.labelBias,
          this.settings.disableThinking,
          readout,
        )
      : this.dialect.completion(
          prompt,
          labels,
          model,
          this.settings.logprobsK,
          this.settings.labelBias,
          readout,
        );
    const payload = await this.upstreamRequest(request.path, request.body);
    const parsed = collectLogprobs(payload);
    const byToken = new Map<string, number>();
    const byId = new Map<number, number>();
    const add = (entry: LogprobEntry) => {
      byToken.set(entry.token, entry.logprob);
      if (entry.id !== undefined) byId.set(entry.id, entry.logprob);
      // Some backends surface the id as a `token_id:NNN` token string.
      const match = /^token_id:(\d+)$/.exec(entry.token);
      if (match) byId.set(Number(match[1]), entry.logprob);
    };
    if (parsed.generated) add(parsed.generated);
    for (const entry of parsed.entries) add(entry);
    if (byToken.size === 0) {
      throw new BackendProtocolError(
        "upstream did not return token logprobs; enable logprobs on the inference backend",
      );
    }
    const usage = record(payload) && record(payload.usage) ? payload.usage : {};
    return {
      probabilities: softmaxFromLogprobs(
        // Match by tokenizer id first: several token ids can decode to the same
        // label string, and only the id returned by /tokenize is the one
        // logit_bias targeted and the one the engine asked for.
        labels.map((entry) => byId.get(entry.id) ?? byToken.get(entry.label) ?? null),
      ),
      inputTokens: numeric(usage.prompt_tokens ?? usage.input_tokens),
      outputTokens: numeric(usage.completion_tokens ?? usage.output_tokens),
    };
  }

  async decide(
    questions: Record<string, Question>,
    state: JsonValue,
    seed: number,
    options?: DecisionOptions,
  ): Promise<DecisionResult> {
    if (this.waiting >= this.settings.maxQueue) {
      throw new OverloadedError("LocalJev is at capacity. Retry shortly.");
    }
    this.waiting += 1;
    try {
      const prepared = prepareQuestions(questions);
      const permute = options?.permute ?? this.settings.permuteDefault;
      const media: MediaParts = {
        images: options?.images ?? [],
        audio: options?.audio ?? [],
      };
      const hasMedia = media.images.length > 0 || media.audio.length > 0;
      if (media.audio.length > 0 && !this.dialect.supportsAudio) {
        throw new MediaUnsupportedError(
          "audio input requires an OpenAI-compatible backend (set LOCALJEV_BACKEND=vllm, sglang, or openai)",
        );
      }
      const plans = prepared.map((question) => {
        const base = labelEntries(question, false);
        const variants = permute ? [base, labelEntries(question, true)] : [base];
        return { question, variants };
      });
      const uniqueLabels = new Set<string>();
      for (const plan of plans) {
        for (const entries of plan.variants) {
          for (const [label] of entries) uniqueLabels.add(label);
        }
      }
      const idPairs = await Promise.all(
        [...uniqueLabels].map(
          async (label) => [label, await this.labelToken(label)] as const,
        ),
      );
      const ids = new Map(idPairs);
      const reads = await Promise.all(
        plans.map(async (plan) => {
          const results = await Promise.all(
            plan.variants.map((entries) =>
              this.readSlot(
                promptFor(state, plan.question, entries),
                entries.map(([label]) => ({ label, id: ids.get(label)! })),
                hasMedia ? media : undefined,
              ),
            ),
          );
          return {
            question: plan.question,
            probabilities: averageProbabilities(
              results.map((result) => result.probabilities),
            ),
            inputTokens: results.reduce((sum, result) => sum + result.inputTokens, 0),
            outputTokens: results.reduce((sum, result) => sum + result.outputTokens, 0),
          };
        }),
      );
      const answers: Record<string, Answer> = {};
      let inputTokens = 0;
      let outputTokens = 0;
      for (const read of reads) {
        answers[read.question.key] = formatAnswer(read.question, read.probabilities);
        inputTokens += read.inputTokens;
        outputTokens += read.outputTokens;
      }
      return { answers, inputTokens, outputTokens };
    } finally {
      this.waiting -= 1;
    }
  }
}
