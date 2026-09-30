# AGENTS.md

## このリポジトリ

`githubnext/localjev` のフォーク。Jev 互換 `POST /v1/systemone` を OpenAI 互換
Chat Completions 上に立てる Bun/TypeScript ブリッジ。

フォークの目的は **LocalJev**: Engine の「JSON に確率を書かせる」経路を廃し、
**llama-server の first-token logprob 読み取り**へ置き換えること。
設計文書: `../DograMagra/docs/specs/2026-09-25-qev-jev-frontend-llamacpp-backend-design.md`

## 構成

| パス | 内容 | 状態 |
|---|---|---|
| `src/engine.ts` | 推論本体(`DecisionEngine` interface + logprob 読み取り Engine) | LocalJev 方式に書き換え済み |
| `src/server.ts` / `src/types.ts` | wire API・検証・エラー形態(live Jev API 照合済み) | 維持(`LocalJevApp`、`permute` 追加) |
| `src/config.ts` | Settings(`LOCALJEV_*` env、設計文書 §8) | LocalJev 向けに再定義済み |
| `src/index.ts` | エントリ | 維持 |
| `eval/` | gold 120(AG News/BoolQ/SST-5)× contexts + ハーネス | 維持(本フォーク vs upstream の対照評価) |
| `test/`, `scripts/`, `Makefile` | 開発用。`scripts/verify-readout.ts` は R1/R2/R3 のライブ検証 | 維持 |

## コマンド

```bash
bun install                # Bun が .env を自動読込
bun run start              # :8081 で起動 (LOCALJEV_UPSTREAM 等の .env 設定)
bun test; bun run typecheck; bun run smoke
bun run verify             # R1/R2/R3 をライブ llama-server に対して検証
bun run eval --out eval/runs/<name> [--limit N]
bun run eval:report
```

## コミット規約

- upstream 踏襲の簡潔な英文(`Implement ...`, `Add ...`)。スコープは任意
- `.env` はコミットしない

## LocalJev の確定した方針

- 1 質問 = 1 読み取り(1 シーケンス)。バッチ読み取りは対象外 — AR では 1 forward で
  複数スロットを同時には読めないため
- 後端は既存の llama-server を**無パッチ**で使うことを第一目標。llama.cpp の GB10
  (SM121)対応は済んでおり、この repo でビルド対応はしない
- 読み取り機構: `/v1/completions` + `max_tokens: 1` + `logprobs`。ラベルは **トークン id**
  で突き合わせる(同一文字列が複数 id に割り当たるため)。`LOCALJEV_READOUT` で3方式:
  - `vocab`(llamacpp 既定・無パッチ): `logprobs` を語彙全体まで要求(サーバが n_vocab に
    clamp)。`logit_bias` に依存せず全ラベルの logprob を直接読む
  - `bias`(openai 既定): ラベル全トークンへ **同一 logit_bias** を掛けて top-K に載せ、
    logprob **差分**から候補間 restricted softmax を復元(bias 項は差分で消える)
  - `selective`(vllm/sglang 既定): ネイティブ API(vLLM `logprob_token_ids` /
    SGLang `token_ids_logprob`)でラベル id の logprob だけ取得。bias 非依存・最小 payload。
    使えない場合は **警告を1回出して `bias` にフォールバック**(vLLM の `logprob_token_ids` は
    OpenAI 経路で v0.26.0+ のみ。旧版は無視するのでラベル欠落→フォールバック)
- 維持する計算: `prepareQuestions` / `confidence = 1 − H(p)/ln K` /
  答案成形(choice 最良選択・score 期待値 Σ i·pᵢ)
- 廃棄する部分: `buildOutputSchema`, `buildSystemPrompt`, `extractJson`,
  malformed リトライループ, `groups()` 逐列チャンク
- **マルチエンジン**: `LOCALJEV_BACKEND` で方言を切替(`src/backends.ts`)。`llamacpp` は
  logit_bias キー=トークン文字列・OpenAI 新 logprobs 形式・`/tokenize` は root・既定 readout
  は `vocab`。`vllm`/`sglang` は キー=トークン ID 文字列・レガシー `top_logprobs`・
  `/tokenize` は root・既定 readout は `selective`(vLLM は `/v1/completions` の
  `logprob_token_ids`(v0.26.0+)を1回、SGLang は `/generate` の `token_ids_logprob` を1回)。
  `openai` は既定 readout `bias`。
  `LOCALJEV_READOUT=auto|bias|vocab|selective` で上書き可。`GET /ready` が `backend`/
  `readout`/`selectiveSupported`/`warning` を返す
- **画像・音声**: Jev 拡張フィールド `images`/`audio`(data URL)。メディア付き読みのみ
  `/v1/chat/completions` の content parts へ切替(テキストは `/v1/completions` 維持)。
  音声は llamacpp 非対応(vllm/sglang/openai のみ)。thinking は分類に不要

## llama-server の事実(設計の前提、再検証しないこと)

- logprobs は **pre-sampling の log(softmax(logits))**。top-N のみ返り、
  **raw logits は HTTP では取得不能**(上流の明示的な設計判断。待っている PR もなし)
- `logit_bias` は**サンプラ**として適用され、**pre-sampling logprobs には反映されない**
  (ggml-org/llama.cpp#10783)。よって「bias でラベルを top-K に入れる」用途では使えない
- `logit_bias` は既存のリクエストパラメータ。`/tokenize` は root でトークン検証可能。
  `logprobs` に語彙数以上を渡すと n_vocab に clamp される(全語彙取得に使える)
- KV 再利用(prefix 一致で prefill スキップ)の条件: state 先頭・質問后缀・byte 一致
- 厳密 logits アクセスが必要になった場合のみ libllama / llama-cpp-python(v2、要設計判断)

## 必須検証(設計文書 §11。`bun run verify` = scripts/verify-readout.ts がライブ検証)

1. **R1**: logit_bias が logprobs の softmax 計算に反映されるか(bias on/off でラベル
   logprob が変わるか)。未パッチ llama.cpp では反映されない → `vocab` 必須
   (`llama-debug --save-logits` と照合可)
2. **R2**: 設定した readout(`vocab`/`bias`/`selective`)で全ラベルが取得できるか
   (top-K 上限・ネイティブ API の有無も確認)
3. **R3**: ラベル(`A..Z`, `0..9`, `yes`, `no`)が対象トークナイザで単一トークンか

## 上流リファレンス

- upstream: `githubnext/localjev`(MIT。upstream 変更は取り込み可能な範囲で同期)
- Jev wire API: TypeSafe Jev(`/v1/systemone`, noul/choice/score)
- OpenJev: `razorback16/openjev` + `vllm-project/vllm#57250`(canvas 読み取りの意味論の目標)
