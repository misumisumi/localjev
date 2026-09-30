import { describe, expect, test } from "bun:test";

import {
  FULL_VOCAB_LOGPROBS,
  audioPart,
  collectLogprobs,
  MediaUnsupportedError,
  resolveDialect,
} from "../src/backends";

describe("logprob parsing", () => {
  test("reads OpenAI-style content logprobs (llama.cpp / chat)", () => {
    const parsed = collectLogprobs({
      choices: [
        {
          logprobs: {
            content: [
              {
                token: "A",
                logprob: -0.1,
                top_logprobs: [
                  { token: "A", logprob: -0.1 },
                  { token: "B", logprob: -2 },
                ],
              },
            ],
          },
        },
      ],
    });
    expect(parsed.generated).toEqual({ token: "A", logprob: -0.1 });
    expect(parsed.entries).toEqual([
      { token: "A", logprob: -0.1 },
      { token: "B", logprob: -2 },
    ]);
  });

  test("reads legacy top_logprobs maps (vLLM/SGLang completions)", () => {
    const parsed = collectLogprobs({
      choices: [
        {
          logprobs: {
            tokens: ["A"],
            token_logprobs: [-0.2],
            top_logprobs: [{ A: -0.2, B: -3 }],
          },
        },
      ],
    });
    expect(parsed.generated).toEqual({ token: "A", logprob: -0.2 });
    expect(parsed.entries).toEqual([
      { token: "A", logprob: -0.2 },
      { token: "B", logprob: -3 },
    ]);
  });

  test("returns nothing when logprobs are absent", () => {
    expect(collectLogprobs({ choices: [{ text: "A" }] })).toEqual({ entries: [] });
    expect(collectLogprobs(null)).toEqual({ entries: [] });
  });

  test("keeps token ids from the OpenAI/llama.cpp content shape", () => {
    const parsed = collectLogprobs({
      choices: [
        {
          logprobs: {
            content: [
              {
                token: "B",
                id: 236799,
                logprob: -0.2,
                top_logprobs: [
                  { token: "A", id: 236776, logprob: -0.1 },
                  { token: "A", id: 303, logprob: -5 },
                ],
              },
            ],
          },
        },
      ],
    });
    expect(parsed.entries).toEqual([
      { token: "A", logprob: -0.1, id: 236776 },
      { token: "A", logprob: -5, id: 303 },
    ]);
  });

  test("reads numeric-keyed legacy maps as token ids", () => {
    const parsed = collectLogprobs({
      choices: [{ logprobs: { top_logprobs: [{ 236776: -0.1, 303: -2 }] } }],
    });
    expect(parsed.entries).toEqual([
      { token: "303", logprob: -2, id: 303 },
      { token: "236776", logprob: -0.1, id: 236776 },
    ]);
  });
});

describe("audio parts", () => {
  test("decodes a base64 data URL into OpenAI input_audio", () => {
    expect(audioPart("data:audio/wav;base64,QUJD")).toEqual({
      type: "input_audio",
      input_audio: { data: "QUJD", format: "wav" },
    });
  });

  test("rejects non data URLs", () => {
    expect(() => audioPart("https://example.com/a.wav")).toThrow(MediaUnsupportedError);
  });
});

describe("dialect readout", () => {
  test("llama.cpp defaults to the full-vocabulary readout without logit_bias", () => {
    const dialect = resolveDialect("llamacpp");
    expect(dialect.defaultReadout).toBe("vocab");
    const request = dialect.completion("p", [{ label: "A", id: 1 }], "m", 64, 10, "vocab");
    expect(request.body.logprobs).toBe(FULL_VOCAB_LOGPROBS);
    expect(request.body.logit_bias).toBeUndefined();
  });

  test("bias readout applies token-string bias and top-K", () => {
    const dialect = resolveDialect("llamacpp");
    const request = dialect.completion("p", [{ label: "A", id: 1 }], "m", 64, 10, "bias");
    expect(request.body.logprobs).toBe(64);
    expect(request.body.logit_bias).toEqual({ A: 10 });
  });

  test("OpenAI dialects default to bias keyed by token id", () => {
    for (const kind of ["vllm", "sglang", "openai"] as const) {
      const dialect = resolveDialect(kind);
      expect(dialect.defaultReadout).toBe(kind === "openai" ? "bias" : "selective");
      const request = dialect.completion("p", [{ label: "A", id: 32 }], "m", 64, 10, "bias");
      expect(request.body.logit_bias).toEqual({ "32": 10 });
    }
  });

  test("openai has no selective API", () => {
    expect(resolveDialect("openai").selective).toBeUndefined();
    expect(resolveDialect("llamacpp").selective).toBeUndefined();
  });

  test("SGLang selective read uses token_ids_logprob and parses the tuples", () => {
    const dialect = resolveDialect("sglang");
    const labels = [
      { label: "A", id: 10 },
      { label: "B", id: 11 },
    ];
    const requests = dialect.selective!.request("prompt", labels, "m");
    expect(requests).toHaveLength(1);
    expect(requests[0]!.path).toBe("/generate");
    expect(requests[0]!.root).toBe(true);
    expect(requests[0]!.body).toMatchObject({ return_logprob: true, token_ids_logprob: [10, 11] });

    // Current builds: [logprob, token_id, null] tuples.
    const tuples = dialect.selective!.parse(
      [{ meta_info: { output_token_ids_logprobs: [[[-0.1, 10, null], [-0.2, 11, null]]] } }],
      labels,
    );
    expect(tuples.get(10)).toBeCloseTo(-0.1);
    expect(tuples.get(11)).toBeCloseTo(-0.2);

    // Older builds: [logprobs[], token_ids[], null].
    const lists = dialect.selective!.parse(
      [{ meta_info: { output_token_ids_logprobs: [[[[-0.3, -0.4], [10, 11], null]]] } }],
      labels,
    );
    expect(lists.get(10)).toBeCloseTo(-0.3);
    expect(lists.get(11)).toBeCloseTo(-0.4);
  });

  test("vLLM selective read uses one /v1/completions request with logprob_token_ids", () => {
    const dialect = resolveDialect("vllm");
    const labels = [
      { label: "A", id: 10 },
      { label: "B", id: 11 },
    ];
    const requests = dialect.selective!.request("prompt", labels, "m");
    expect(requests).toHaveLength(1);
    expect(requests[0]!.path).toBe("/completions");
    expect(requests[0]!.root).toBe(false);
    expect(requests[0]!.body).toMatchObject({ logprob_token_ids: [10, 11] });

    // vLLM returns the legacy completions map keyed by decoded token string.
    const parsed = dialect.selective!.parse(
      [
        {
          choices: [
            {
              logprobs: {
                tokens: ["x"],
                token_logprobs: [-0.1],
                top_logprobs: [{ A: -0.5, B: -1.5 }],
              },
            },
          ],
        },
      ],
      labels,
    );
    expect(parsed.get(10)).toBeCloseTo(-0.5);
    expect(parsed.get(11)).toBeCloseTo(-1.5);
  });
});
