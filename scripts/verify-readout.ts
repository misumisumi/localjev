import { apiBaseUrl, loadSettings } from "../src/config";
import { CHOICE_LABELS } from "../src/engine";

// Pre-implementation verification for QEv (design doc section 11):
//   R1: are top logprobs computed after logit_bias (so label differences survive bias)?
//   R2: do all biased labels appear within top-K?
//   R3: are all label candidates single tokens in the upstream tokenizer?
// Run against a live llama-server: bun run scripts/verify-readout.ts

const settings = loadSettings();
const base = apiBaseUrl(settings);
const headers: Record<string, string> = {
  "content-type": "application/json",
  ...(settings.upstreamApiKey
    ? { authorization: `Bearer ${settings.upstreamApiKey}` }
    : {}),
};

async function post(path: string, body: unknown): Promise<Record<string, unknown>> {
  const response = await fetch(`${base}${path}`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(settings.timeoutMs),
  });
  if (!response.ok) {
    throw new Error(`${path}: HTTP ${response.status}: ${await response.text()}`);
  }
  return (await response.json()) as Record<string, unknown>;
}

const modelsResponse = await fetch(`${base}/models`, { headers });
if (!modelsResponse.ok) {
  console.error(`upstream /v1/models: HTTP ${modelsResponse.status}; is llama-server running at ${base}?`);
  process.exit(1);
}
const models = (await modelsResponse.json()) as { data?: { id: string }[] };
const model = settings.upstreamModel || models.data?.[0]?.id;
if (!model) {
  console.error("no upstream model available");
  process.exit(1);
}
console.log(`upstream: ${base} model: ${model}`);

async function singleToken(label: string): Promise<boolean> {
  const payload = await post("/tokenize", { content: label });
  return Array.isArray(payload.tokens) && payload.tokens.length === 1;
}

// --- R3: single-token labels ---
const required = ["yes", "no", ..."0123456789", ..."ABCDEFGHIJKLMNOPQRSTUVWXYZ"];
const multiToken: string[] = [];
for (const label of required) {
  if (!(await singleToken(label))) multiToken.push(label);
}
console.log(`R3 single-token: ${required.length - multiToken.length}/${required.length} ok`);
if (multiToken.length) console.log(`R3 not single-token: ${multiToken.join(" ")}`);

async function topScores(
  prompt: string,
  labels: string[],
  bias: number,
): Promise<Map<string, number>> {
  const payload = await post("/completions", {
    model,
    prompt,
    max_tokens: 1,
    temperature: 0,
    logprobs: settings.logprobsK,
    ...(bias > 0 ? { logit_bias: Object.fromEntries(labels.map((l) => [l, bias])) } : {}),
  });
  const choices = payload.choices as Record<string, unknown>[] | undefined;
  const logprobs = choices?.[0]?.logprobs as
    | { content?: { top_logprobs?: { token: string; logprob: number }[] }[] }
    | undefined;
  const top = logprobs?.content?.[0]?.top_logprobs;
  if (!top) throw new Error("upstream returned no top_logprobs; check llama-server build (R5)");
  return new Map(top.map((entry) => [entry.token, entry.logprob]));
}

function report(title: string, condition: boolean, detail: string): void {
  console.log(`${title}: ${condition ? "PASS" : "FAIL"} — ${detail}`);
}

// --- R1: bias cancels out in label logprob differences ---
const r1Prompt =
  '<document>\n"Q3 ships on Tuesday. Q3 ships on Tuesday."\n</document>\n\n' +
  "Question: Which letter answers the statement?\nA. alpha\nB. beta\n\nAnswer: ";
const withBias = await topScores(r1Prompt, ["A", "B"], settings.labelBias);
const withoutBias = await topScores(r1Prompt, ["A", "B"], 0);
const la = withBias.get("A");
const lb = withBias.get("B");
const na = withoutBias.get("A");
const nb = withoutBias.get("B");
if (la !== undefined && lb !== undefined && na !== undefined && nb !== undefined) {
  const delta = Math.abs(la - lb - (na - nb));
  report(
    "R1 bias cancels in differences",
    delta < 1e-3,
    `biased diff=${(la - lb).toFixed(4)} vs unbiasd diff=${(na - nb).toFixed(4)}, |delta|=${delta.toExponential(2)}`,
  );
} else {
  console.log(
    `R1: inconclusive — labels present without bias: A=${na !== undefined} B=${nb !== undefined}` +
      ` (biased run: A=${la !== undefined} B=${lb !== undefined}); differences are only comparable when both runs contain both labels`,
  );
}

// --- R2: all 26 biased labels appear inside top-K ---
const r2Labels = CHOICE_LABELS.slice(0, 26);
const r2Prompt =
  '<document>\n"a sentence with plenty of ordinary context tokens before the answer\n'.repeat(4) +
  'a"\n</document>\n\nQuestion: Pick one letter.\n' +
  r2Labels.map((label, index) => `${label}. option ${index + 1}`).join("\n") +
  "\n\nAnswer: ";
const r2Scores = await topScores(r2Prompt, r2Labels, settings.labelBias);
const present = r2Labels.filter((label) => r2Scores.has(label));
report(
  `R2 labels in top-K (K=${settings.logprobsK})`,
  present.length === 26,
  `present=${present.length}/26` +
    (present.length === 26
      ? ""
      : `; missing: ${r2Labels.filter((l) => !r2Scores.has(l)).join(" ")}`),
);

// --- vocabulary bias snapshot for noul ---
const noulPrompt =
  '<document>\n"The statement is definitely true and clearly correct."\n</document>\n\n' +
  "Question: Is the statement true?\nyes\nno\n\nAnswer: ";
const noulScores = await topScores(noulPrompt, ["yes", "no"], settings.labelBias);
const py = noulScores.get("yes");
const pn = noulScores.get("no");
if (py !== undefined && pn !== undefined) {
  console.log(`noul vocabulary bias sample: P(yes)=${Math.exp(py - Math.max(py, pn)).toFixed(3)} normalized ratio yes/no=${Math.exp(py - pn).toFixed(3)}`);
}
