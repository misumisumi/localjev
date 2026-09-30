import { apiBaseUrl, loadSettings, upstreamRoot } from "../src/config";
import {
  collectLogprobs,
  resolveDialect,
  type LabelToken,
  type LogprobEntry,
  type ReadoutMode,
} from "../src/backends";
import { CHOICE_LABELS } from "../src/engine";

// Pre-implementation verification for LocalJev (design doc section 11):
//   R1: is logit_bias reflected in the returned logprobs? (stock llama.cpp: no)
//   R2: with the configured readout, does every answer label become readable?
//   R3: are all label candidates single tokens in the upstream tokenizer?
// Run against a live server: bun run scripts/verify-readout.ts
// Select the strategy with LOCALJEV_READOUT=bias|vocab (default: dialect default).

const settings = loadSettings();
const base = apiBaseUrl(settings);
const dialect = resolveDialect(settings.backend);
const readout: ReadoutMode =
  settings.readout === "auto" ? dialect.defaultReadout : settings.readout;
const headers: Record<string, string> = {
  "content-type": "application/json",
  ...(settings.upstreamApiKey
    ? { authorization: `Bearer ${settings.upstreamApiKey}` }
    : {}),
};

async function postUrl(url: string, body: unknown): Promise<Record<string, unknown>> {
  const response = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(settings.timeoutMs),
  });
  if (!response.ok) {
    throw new Error(`${url}: HTTP ${response.status}: ${await response.text()}`);
  }
  return (await response.json()) as Record<string, unknown>;
}

const modelsResponse = await fetch(`${base}/models`, { headers });
if (!modelsResponse.ok) {
  console.error(`upstream /v1/models: HTTP ${modelsResponse.status}; is a server running at ${base}?`);
  process.exit(1);
}
const models = (await modelsResponse.json()) as { data?: { id: string }[] };
const model = settings.upstreamModel || models.data?.[0]?.id;
if (!model) {
  console.error("no upstream model available");
  process.exit(1);
}
console.log(
  `upstream: ${base} model: ${model} backend: ${settings.backend} readout: ${readout}` +
    (readout === "selective"
      ? ` (selective ${dialect.selective ? "available" : "UNAVAILABLE"})`
      : ""),
);

// --- R3: single-token labels -------------------------------------------------
async function labelToken(label: string): Promise<LabelToken | null> {
  const request = dialect.tokenize(label, model!);
  const url = `${request.root ? upstreamRoot(settings) : base}${request.path}`;
  const payload = await postUrl(url, request.body);
  const tokens = dialect.tokenIds(payload);
  if (!tokens || tokens.length !== 1) return null;
  return { label, id: tokens[0]! };
}

const required = ["yes", "no", ..."0123456789", ..."ABCDEFGHIJKLMNOPQRSTUVWXYZ"];
const resolved = new Map<string, LabelToken>();
const multiToken: string[] = [];
for (const label of required) {
  const token = await labelToken(label);
  if (token) resolved.set(label, token);
  else multiToken.push(label);
}
console.log(`R3 single-token: ${required.length - multiToken.length}/${required.length} ok`);
if (multiToken.length) console.log(`R3 not single-token: ${multiToken.join(" ")}`);

// --- reads -------------------------------------------------------------------
interface Scores {
  byToken: Map<string, number>;
  byId: Map<number, number>;
}

async function scores(prompt: string, labels: LabelToken[], bias: number): Promise<Scores> {
  if (readout === "selective") {
    const spec = dialect.selective;
    if (!spec) {
      throw new Error(`backend ${settings.backend} exposes no selective logprobs API`);
    }
    const responses: unknown[] = [];
    for (const request of spec.request(prompt, labels, model!)) {
      const url = `${request.root ? upstreamRoot(settings) : base}${request.path}`;
      responses.push(await postUrl(url, request.body));
    }
    return { byToken: new Map(), byId: spec.parse(responses, labels) };
  }
  const request = dialect.completion(
    prompt,
    labels,
    model!,
    settings.logprobsK,
    bias,
    readout,
  );
  const payload = await postUrl(`${base}${request.path}`, request.body);
  const parsed = collectLogprobs(payload);
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
  return { byToken, byId };
}

function pick(result: Scores, label: LabelToken): number | undefined {
  return result.byId.get(label.id) ?? result.byToken.get(label.label);
}

function report(title: string, condition: boolean, detail: string): void {
  console.log(`${title}: ${condition ? "PASS" : "FAIL"} — ${detail}`);
}

// --- R1: is logit_bias reflected, and does it cancel in differences? ---------
const r1Prompt =
  '<document>\n"Q3 ships on Tuesday. Q3 ships on Tuesday."\n</document>\n\n' +
  "Question: Which letter answers the statement?\nA. alpha\nB. beta\n\nAnswer: ";
const r1Labels = ["A", "B"].map((label) => resolved.get(label)).filter((t): t is LabelToken => !!t);
if (readout === "selective") {
  console.log("R1: n/a — the selective readout does not use logit_bias");
} else {
  const withBias = await scores(r1Prompt, r1Labels, settings.labelBias);
  const withoutBias = await scores(r1Prompt, r1Labels, 0);
  const la = pick(withBias, r1Labels[0]!);
  const lb = pick(withBias, r1Labels[1]!);
  const na = pick(withoutBias, r1Labels[0]!);
  const nb = pick(withoutBias, r1Labels[1]!);
  if (la !== undefined && lb !== undefined && na !== undefined && nb !== undefined) {
    const reflected = Math.abs(la - na) > 1e-6 || Math.abs(lb - nb) > 1e-6;
    report(
      "R1 logit_bias reflected in logprobs",
      reflected,
      `label logprobs changed by bias=${settings.labelBias}: ` +
        `${reflected ? "yes" : "no (bias is not reflected; use LOCALJEV_READOUT=vocab on stock llama.cpp)"}`,
    );
    const delta = Math.abs(la - lb - (na - nb));
    report(
      `R1 bias cancels in differences`,
      delta < 1e-3,
      `biased diff=${(la - lb).toFixed(4)} vs unbiased diff=${(na - nb).toFixed(4)}, |delta|=${delta.toExponential(2)}`,
    );
  } else {
    console.log(
      `R1: inconclusive — labels present without bias: A=${na !== undefined} B=${nb !== undefined}` +
        ` (biased run: A=${la !== undefined} B=${lb !== undefined})`,
    );
  }
}

// --- R2: are all labels readable with the configured readout? ----------------
const r2Labels = CHOICE_LABELS.slice(0, 26)
  .map((label) => resolved.get(label))
  .filter((t): t is LabelToken => !!t);
const r2Prompt =
  '<document>\n"a sentence with plenty of ordinary context tokens before the answer\n'.repeat(4) +
  'a"\n</document>\n\nQuestion: Pick one letter.\n' +
  r2Labels.map((label, index) => `${label.label}. option ${index + 1}`).join("\n") +
  "\n\nAnswer: ";
const r2Scores = await scores(r2Prompt, r2Labels, settings.labelBias);
const present = r2Labels.filter((label) => pick(r2Scores, label) !== undefined);
report(
  `R2 labels readable (readout=${readout}, K=${settings.logprobsK})`,
  present.length === r2Labels.length,
  `present=${present.length}/${r2Labels.length}` +
    (present.length === r2Labels.length
      ? ""
      : `; missing: ${r2Labels.filter((l) => pick(r2Scores, l) === undefined).map((l) => l.label).join(" ")}`),
);

// --- vocabulary bias snapshot for noul ---------------------------------------
const noulPrompt =
  '<document>\n"The statement is definitely true and clearly correct."\n</document>\n\n' +
  "Question: Is the statement true?\nyes\nno\n\nAnswer: ";
const noulLabels = ["yes", "no"]
  .map((label) => resolved.get(label))
  .filter((t): t is LabelToken => !!t);
const noulScores = await scores(noulPrompt, noulLabels, settings.labelBias);
const py = pick(noulScores, noulLabels[0]!);
const pn = pick(noulScores, noulLabels[1]!);
if (py !== undefined && pn !== undefined) {
  console.log(
    `noul sample: P(yes)/P(no)=${Math.exp(py - pn).toFixed(3)} (restricted ratio)`,
  );
}
