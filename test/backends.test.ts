import { describe, expect, test } from "bun:test";

import { audioPart, collectLogprobs, MediaUnsupportedError } from "../src/backends";

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
