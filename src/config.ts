import type { EngineKind } from "./backends";

export interface Settings {
  upstream: string;
  upstreamApiKey: string;
  upstreamModel: string;
  apiKey: string;
  host: string;
  port: number;
  timeoutMs: number;
  maxInflight: number;
  maxQueue: number;
  labelBias: number;
  logprobsK: number;
  permuteDefault: boolean;
  backend: EngineKind;
  disableThinking: boolean;
}

const ENGINE_KINDS: readonly EngineKind[] = ["llamacpp", "vllm", "sglang", "openai"];

function engineKind(defaultValue: EngineKind): EngineKind {
  const raw = process.env.QEV_BACKEND?.trim().toLowerCase();
  if (raw === undefined || raw === "") return defaultValue;
  if (raw === "llama.cpp" || raw === "llama_cpp") return "llamacpp";
  if ((ENGINE_KINDS as readonly string[]).includes(raw)) return raw as EngineKind;
  throw new Error(
    `QEV_BACKEND must be one of ${ENGINE_KINDS.join(", ")} (or "llama.cpp"); got ${JSON.stringify(raw)}`,
  );
}

function numberSetting(name: string, fallback: number, minimum: number): number {
  const raw = process.env[name];
  const value = raw === undefined ? fallback : Number(raw);
  if (!Number.isFinite(value) || value < minimum) {
    throw new Error(`${name} must be a number greater than or equal to ${minimum}`);
  }
  return value;
}

function integerSetting(name: string, fallback: number, minimum: number): number {
  const value = numberSetting(name, fallback, minimum);
  if (!Number.isInteger(value)) {
    throw new Error(`${name} must be an integer`);
  }
  return value;
}

function booleanSetting(name: string, fallback: boolean): boolean {
  const raw = process.env[name];
  if (raw === undefined) return fallback;
  const normalized = raw.trim().toLowerCase();
  if (normalized === "true" || normalized === "1") return true;
  if (normalized === "false" || normalized === "0") return false;
  throw new Error(`${name} must be true/false or 1/0`);
}

export function loadSettings(
  overrides: Partial<Settings> = {},
): Settings {
  return {
    upstream: process.env.QEV_UPSTREAM ?? "http://127.0.0.1:8000",
    upstreamApiKey: process.env.QEV_UPSTREAM_API_KEY ?? "",
    upstreamModel: process.env.QEV_UPSTREAM_MODEL ?? "",
    apiKey: process.env.QEV_API_KEY ?? "",
    host: process.env.QEV_HOST ?? "127.0.0.1",
    port: integerSetting("QEV_PORT", 8081, 1),
    timeoutMs: numberSetting("QEV_TIMEOUT", 60, 0.001) * 1_000,
    maxInflight: integerSetting("QEV_MAX_INFLIGHT", 4, 1),
    maxQueue: integerSetting("QEV_MAX_QUEUE", 64, 1),
    labelBias: numberSetting("QEV_LABEL_BIAS", 10, 0),
    logprobsK: integerSetting("QEV_LOGPROBS_K", 64, 1),
    permuteDefault: booleanSetting("QEV_PERMUTE_DEFAULT", false),
    backend: engineKind("llamacpp"),
    disableThinking: booleanSetting("QEV_DISABLE_THINKING", false),
    ...overrides,
  };
}

export function apiBaseUrl(settings: Settings): string {
  const base = settings.upstream.replace(/\/+$/, "");
  return base.endsWith("/v1") ? base : `${base}/v1`;
}

export function upstreamRoot(settings: Settings): string {
  const base = settings.upstream.replace(/\/+$/, "");
  return base.endsWith("/v1") ? base.slice(0, -3) : base;
}

export const MODEL_VERSION = "qev-0.1";
export const MODEL_ALIASES = new Set([
  MODEL_VERSION,
  "qev-latest",
  "jev-latest",
  "jev-preview",
]);
export const MODELS = [
  {
    name: "qev-latest",
    description:
      "Alias for QEv 0.1, a Jev-compatible bridge reading first-token logprobs from a local llama-server.",
    release_date: "2026-09-25",
  },
  {
    name: MODEL_VERSION,
    description:
      "Jev-compatible probability inference read from llama-server first-token logprobs.",
    release_date: "2026-09-25",
  },
] as const;
