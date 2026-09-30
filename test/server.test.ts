import { describe, expect, test } from "bun:test";

import { loadSettings } from "../src/config";
import type { DecisionEngine } from "../src/engine";
import { LocalJevApp } from "../src/server";

const requestBody = {
  model: "jev-latest",
  state: "A production service is down.",
  questions: {
    urgent: {
      type: "noul",
      instructions: "Does this need an immediate response?",
    },
  },
};

const fakeEngine: DecisionEngine = {
  async decide(questions, state) {
    expect(state).toBe("A production service is down.");
    expect(questions.urgent?.type).toBe("noul");
    return {
      answers: { urgent: { type: "noul", noul: 0.95 } },
      inputTokens: 42,
      outputTokens: 7,
    };
  },
  async ready() {
    return true;
  },
};

function post(body: unknown, headers: HeadersInit = {}): Request {
  return new Request("http://localhost/v1/systemone", {
    method: "POST",
    headers: { "content-type": "application/json", ...headers },
    body: JSON.stringify(body),
  });
}

describe("Jev API", () => {
  test("returns the expected wire shape and request headers", async () => {
    const app = new LocalJevApp(loadSettings(), fakeEngine);
    const response = await app.fetch(post(requestBody));
    expect(response.status).toBe(200);
    expect(response.headers.get("x-typesafe-request-id")).toStartWith("req_");
    expect(response.headers.get("server-timing")).toContain("model;dur=");
    expect(await response.json()).toEqual({
      model: "localjev-0.1",
      answers: { urgent: { type: "noul", noul: 0.95 } },
      usage: { input_tokens: 42, output_tokens: 7 },
    });
  });

  test("serves models, health, and readiness", async () => {
    const app = new LocalJevApp(loadSettings(), fakeEngine);
    const health = await app.fetch(new Request("http://localhost/health"));
    const ready = await app.fetch(new Request("http://localhost/ready"));
    const models = await app.fetch(new Request("http://localhost/v1/models"));
    expect(await health.json()).toEqual({ status: "ok" });
    expect(await ready.json()).toMatchObject({ status: "ready" });
    expect((await models.json()).models[0].name).toBe("localjev-latest");
  });

  test("returns Jev-style validation and model errors", async () => {
    const app = new LocalJevApp(loadSettings(), fakeEngine);
    const missingState = await app.fetch(
      post({ model: "jev-latest", questions: requestBody.questions }),
    );
    const badScore = await app.fetch(
      post({
        model: "jev-latest",
        state: "x",
        questions: { score: { type: "score", criteria: ["only"] } },
      }),
    );
    const badPermute = await app.fetch(
      post({ ...requestBody, permute: "yes" }),
    );
    const badModel = await app.fetch(post({ ...requestBody, model: "gpt-4" }));
    expect(missingState.status).toBe(422);
    expect((await missingState.json()).detail[0].loc).toEqual(["body", "state"]);
    expect(badScore.status).toBe(422);
    expect(badPermute.status).toBe(422);
    expect(badModel.status).toBe(404);
    expect((await badModel.json()).detail.error_type).toBe("not_found_error");
  });

  test("passes permute through to the engine", async () => {
    const seen: unknown[] = [];
    const engine: DecisionEngine = {
      async decide(_questions, _state, _seed, options) {
        seen.push(options?.permute);
        return {
          answers: { urgent: { type: "noul", noul: 0.5 } },
          inputTokens: 1,
          outputTokens: 1,
        };
      },
    };
    const app = new LocalJevApp(loadSettings(), engine);
    await app.fetch(post({ ...requestBody, permute: true }));
    await app.fetch(post(requestBody));
    expect(seen).toEqual([true, undefined]);
  });

  test("passes images and audio through to the engine", async () => {
    const seen: unknown[] = [];
    const engine: DecisionEngine = {
      async decide(_questions, _state, _seed, options) {
        seen.push(options);
        return {
          answers: { urgent: { type: "noul", noul: 0.5 } },
          inputTokens: 1,
          outputTokens: 1,
        };
      },
    };
    const app = new LocalJevApp(loadSettings(), engine);
    await app.fetch(
      post({
        ...requestBody,
        images: ["data:image/png;base64,AA"],
        audio: ["data:audio/wav;base64,AA"],
      }),
    );
    expect(seen[0]).toMatchObject({
      images: ["data:image/png;base64,AA"],
      audio: ["data:audio/wav;base64,AA"],
    });
  });

  test("rejects malformed images and audio arrays", async () => {
    const app = new LocalJevApp(loadSettings(), fakeEngine);
    const badImages = await app.fetch(post({ ...requestBody, images: "nope" }));
    const badAudio = await app.fetch(post({ ...requestBody, audio: [123] }));
    expect(badImages.status).toBe(422);
    expect((await badImages.json()).detail[0].loc).toEqual(["body", "images"]);
    expect(badAudio.status).toBe(422);
  });

  test("supports optional client authentication", async () => {
    const app = new LocalJevApp(
      loadSettings({ apiKey: "local-secret" }),
      fakeEngine,
    );
    const missing = await app.fetch(
      new Request("http://localhost/v1/models"),
    );
    const wrong = await app.fetch(
      new Request("http://localhost/v1/models", {
        headers: { authorization: "Bearer wrong" },
      }),
    );
    const valid = await app.fetch(
      new Request("http://localhost/v1/models", {
        headers: { authorization: "Bearer local-secret" },
      }),
    );
    expect(missing.status).toBe(403);
    expect(wrong.status).toBe(401);
    expect(valid.status).toBe(200);
  });
});
