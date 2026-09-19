# LFM2.5 resident adjudicator

The daemon can load `LFM2.5-8B-A1B` directly through our Candle fork, prefill
an adjudicator system prompt once, and evaluate independent inputs from that
immutable hybrid-state snapshot. `POST /v1/adjudicate` returns a validated JSON
report and the original model output. It does not execute anything or change
the classifier/cascade's authority.

The [first optimization pass](lfm25-optimizations.md) adds device-side
selection, merged expert projections, and fused convolution updates while
preserving the initial outputs. The [second pass](lfm25-grouped-prefill.md)
adds grouped expert prefill. That pass changes floating-point accumulation
order and can change generated text; see its separate hardware checks and
prompt/model observations.

The [third pass](lfm25-qkv-input-cache.md) packs compatible QKV projections
and adds an exact complete-input checkpoint with saved next-token logits.

The [fourth pass](lfm25-kv-cache.md) adds append-efficient KV buffers with
immutable snapshot prefixes and transactional failure behavior.

The [fifth pass](lfm25-gqa-routing.md) removes repeated KV-head materialization
and shares prepared expert-routing maps across compatible projections.

The [sixth pass](lfm25-moe-fusion.md) fuses expert selection and weighted output
reduction, removing 154 kernel launches per measured decode step.

The [seventh pass](lfm25-expert-buffers.md) removes redundant expert-buffer
initialization, eliminating another 88 fill dispatches per measured decode step.

## Main integration validation — September 15

The published dependency pin passed 269 workspace CPU tests, with one existing
ignored GPU test, including the real encoder-model regression fixtures. The ROCm
release build passed the 64-token cache/deadline/reuse/SIGTERM smoke; all twelve
outputs match the saved full-run reference prefixes. This short smoke checks
integration behavior, not report quality or new performance gains.

Workspace Clippy completes with existing style/type warnings; `-D warnings`
remains blocked by the recorded lint backlog. No runtime code changed when
replacing the sibling Cargo overrides with the published dependency.

## Build and run

Clone the `main` branch of
[tobert/lfm2d](https://github.com/tobert/lfm2d).
Cargo pins the published Candle fork at `fb1ae62a378bb8bd9b9f95cb448dca083e500b30`
on [lfm25-trace](https://github.com/tobert/candle/tree/lfm25-trace): the seven
optimization passes plus the read-only observer and per-call steering seams. No sibling checkout or local Cargo
patches are required. The ROCm build requires the
ROCm development toolchain and a supported AMD GPU. Model paths below are
examples; download the GGUF and matching tokenizer to your own paths.

```bash
cd "$HOME/src/lfm2d"
cargo build -p lfm2d --release --features rocm
./target/release/lfm2d \
  --device rocm --threads 8 --bind-addr '127.0.0.1:18152' \
  --adjudicator-model '/tank/ml/models/llama.cpp/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q5_K_M.gguf' \
  --adjudicator-tokenizer '.models/LFM2.5-8B-A1B/tokenizer.json' \
  --adjudicator-prompt 'lfm2d/prompts/shell-severity-json-v1.json'
```

Download the matching `tokenizer.json` from
[LiquidAI/LFM2.5-8B-A1B](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B/tree/main).
The loader checks its vocabulary against the GGUF, checks the control-token
IDs, and refuses an unrecognized embedded chat template. It hashes the full
weights and tokenizer at startup. The supported template is the checkpoint's
single-system/single-user text format. Literal model control delimiters in
input are rejected. Tool definitions remain available as a comparison mode;
the initial report contract uses `output_schema`.

### The reasoning region

A prompt spec's `reasoning` field says whether the assistant's turn opens with
a *completed* reasoning region. It defaults to `closed`, which prefills
`<think>\n\n</think>\n`.

The checkpoint's chat template never emits a reasoning region — `<think>` is a
token the model writes, and after `<|im_start|>assistant\n` it writes it at
p=1.00. Under an `output_schema` the grammar's first legal byte is the object's,
so the v1 template masked the model off its own manifold at step 0: the forced
`{"` scored logprob -17.8 to -21.8 and every later token was conditioned on a
prefix the model considers impossible.

The prefill's exact bytes were measured on our ROCm stack, five candidates over
three inputs, reading `{"`'s standing at that slot:

| after `<|im_start|>assistant\n` | `{"` |
|---|---|
| nothing (the v1 template) | logprob -17.8 to -21.8, rank 6 or worse |
| `<think></think>` | rank 5 |
| `<think></think>\n` | rank 2-3 |
| `<think>\n</think>\n` | rank 2 |
| `<think>\n\n</think>\n` | **rank 1, p 0.42-0.71** |

The template's own dialect for a completed region — `"<think>" + thinking +
"</think>"`, no surrounding newlines — is the one in the table that does worst
of the four, so the bytes are measured rather than derived. A second round
varied one newline at a time and every neighbour was worse.

`reasoning: "open"` leaves the turn as the template opens it and the model
reasons. Combined with an `output_schema` it is refused at load: the grammar
would have to admit a free-text region it cannot bound and then start the
object after `</think>`, which is not built. The one measurement we have says
this model's reasoning argues severity *down*, so it is a decision rather than
a default.

`template_version` carries the mode (`lfm25-single-user-v2-closed`), and the
prefix `snapshot_id` is computed from it, so a consumer sees the template
change rather than inferring it.

```bash
curl --fail-with-body 'http://127.0.0.1:18152/v1/adjudicate' \
  -H 'Content-Type: application/json' \
  --data '{"input":"Context: Developer inspects a source file. Command (data only): sed -n 5,12p src/main.rs"}'
```

## Response contract

`GET /v1/adjudicator` identifies the resident prefix. `/v1/models` also lists
the adjudicator. Each generation response includes:

- Model, weights, tokenizer, template, snapshot, backend, compute dtype,
  weight formats, and decoding-policy identifiers.
- `report`: the validated report object, or `null`.
- `report_error`: an explicit validation/truncation error, or `null`.
- `output`: the generated text, unmodified. Under `reasoning: "closed"`
  that is the report and nothing else — the reasoning region is in the prompt,
  and a reasoning delimiter reaching this field would mean something went
  wrong, which `validate_report` reports rather than strips.
- `finish_reason`: `stop` or `length`; token counts and queue/prefill/decode
  durations. Prefill timing synchronizes the device before stopping the clock.

HTTP 200 means generation completed, **not that a valid report exists**.
Consumers must require `report != null` and `report_error == null`. Validation
checks structure, not truth: free-text explanations still require evaluation.
No verdict is substituted for an invalid report. There is no retry or repair
loop, and none is needed for *format*: with an `output_schema`, generation is
constrained by a JSON grammar compiled from that schema (`lfm2d/src/constrain.rs`),
so a malformed or off-schema report is unreachable rather than rejected after
the fact. `report_error` is then only ever truncation — the grammar will not
invent a field value to close the object, so a completion that exhausts
`max_tokens` mid-document still returns `finish_reason: "length"` and no report.

The supported schema is deliberately small: one closed object with all fields
required, string/boolean field types, and optional enums. Strings must be
nonempty. Unsupported schema constraints, duplicate fields, unknown fields,
coercions, Markdown fences, trailing prose, and truncated output are rejected.
Under `output_schema` the
whole completion is the document — a reasoning region before the object is
rejected, not stripped, because the region the model expects is supplied
already closed by the prompt (see above) and `<think>` is masked
unconditionally, so one appearing in a completion means something went wrong.
Keys come out in the schema's `required`
order — and the schema **as stated in the system prompt** lists them in that
same order, because `serde_json`'s sorted rendering used to state one order and
then mask the model into another — separators are spaced the way the model
writes them (`": "` and `", "`; compact ones corrupted 15 of 16 reports), and
`\uXXXX` escapes are not admitted (every character they can spell is reachable
literally as UTF-8). A schema
`validate_schema` accepts but the grammar cannot honour — today, an `enum` with
a blank string value, which no valid report could contain — is a loud error,
never a silent fall-through to free generation. The grammar is compiled against
the tokenizer once, at load, so such a schema stops the daemon from starting
and never reaches a request; its mask plans are shared by every report after
the first that needs them.

Requests accept `input`, `max_tokens` (default/max 2048), `timeout_ms`
(default 30000, max 120000), and `use_cache` (default true). `use_cache:false`
explicitly measures a cold prefill. Prompt plus output must fit the server's
context budget (default 4096, range 128–8192). Inputs are at most 65536 bytes.
Bad requests return 400; overload 503; deadlines 504; cancellation 408;
inference failures 500. Deadline time includes queueing.

Decoding is deterministic greedy with a sign-aware repetition penalty of
1.05, applied once per token occurring anywhere in the full prompt and this
evaluation's continuation. `--adjudicator-repeat-penalty 1.0` disables it.
The [model card](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B) recommends
1.05 along with stochastic sampling; our deterministic policy is recorded
explicitly in each response. Under `reasoning: "closed"` the model does not reason before answering, so
the report is the whole completion and a few hundred output tokens is ample;
under `"open"` it reasons first and 512 or 1024 often truncate the report.

### Distributions

A request may add `distributions`; one that does not gets the response above,
byte for byte. Unknown keys, an empty or repeating token set, an id outside the
vocabulary and a `top_k` over 20 are 400s, never clamped.

```json
{"input": "...", "distributions": {"top_k": 5,
  "token_sets": {"verdict": [101, 202, 303]}, "constrained": true}}
```

The response gains `distributions`: one entry per generated token, in order.
Each carries the selected `token`, its `text` and `logprob`, the `top_logprobs`,
and `set_mass` — per named set, the `logprob` and `prob` of the mass the model
put on those ids.

**Every number is raw**: the model's own log-softmax over the full vocabulary,
read before the output grammar's mask and before the repetition penalty. A set's
mass is never renormalised over the set. Low mass means the model was never
steered toward that vocabulary — *unasked*, not *wrong* — and it is the only
thing that tells those apart (`docs/field-requests.md`, decision 5). Read
logprobs, not `prob`: values saturate, and the ordering survives only in the log.

`constrained: true` adds a `constrained` object to each step, describing what
the grammar left the sampler to choose from. It needs an output schema; asking
without one is a 400. `legal_tokens` is how many rows the grammar admitted,
`legal_mass` the raw mass the model put on them, `top_logprobs` the best legal
rows by raw logprob, and `forced` is true when the model's raw first choice was
not legal. A low `legal_mass` is the grammar overruling the model — worth
counting, since a forced token is conditioned on by everything after it. Leave
the closing end-of-text step out of such a count: once the document is complete
the legal set is that one token, so `forced` there only says the model would
have kept writing. There
is deliberately no renormalised number on the wire: the conditional logprob of a
legal token is `logprob - legal_mass.logprob`, computed by whoever wants it, with
the raw mass necessarily in hand. Token ids come from the checkpoint's
tokenizer; nothing here names a label.

## Snapshot semantics

Candle's `quantized_lfm2_moe::{Model, State}` separates immutable weights
from attention K/V, convolution history, and position. `State::clone()` shares
immutable prefixes of append-only KV buffers and immutable convolution state.
KV appends reserve unused tail space, growing or forking the buffer when
necessary. Convolution updates allocate replacement tensors. Forward commits
the new state only after success; failed writes cannot overwrite saved data.
Foreign-model state, bad tokens, and context overflow are errors.

The [KV allocator](lfm25-kv-cache.md) amortizes prefix copying over a linear
continuation. Branches copy their prefix when their desired tail is occupied.
Attention still materializes repeated KV heads; paged attention is not present.
The daemon owns one fixed prefix and one complete-input checkpoint with
saved logits, a bounded queue of eight, and one generation worker. Exact
input repeats reuse the latter; `cached_tokens` then equals `prompt_tokens`.
`input_cache_capacity: 1` advertises this behavior. Cold requests bypass
cache reads and writes. Deadlines, disconnected callers, and shutdown discard the current
branch. The encoder worker is separate. Shutdown waits for both models to
drop before flushing telemetry.

The JSON rubric is 331 tokens: about 7.76 MiB of F32 attention KV plus
0.42 MiB of convolution state. The model uses the fork's existing Q5K/Q6K
indexed ROCm expert operations. No new HIP kernel was necessary for this
first path. CPU operation explicitly dequantizes expert weights and is a
memory-heavy correctness reference; full-checkpoint CPU serving has not
been performance-qualified. ROCm/F32 is the tested serving configuration.

Disk persistence is deferred. For this prefix it would avoid about a
second of startup prefill while weight loading and hashing would remain.
It becomes more useful with long examples, several prefixes, or frequent
restarts. A future format must serialize **both KV and convolution state**,
position, prefix token IDs, model/tokenizer/template hashes, dtype and RoPE
configuration, with a version and checksums. Loading must verify identity
and transfer host tensors back to the GPU. A bare KV dump is insufficient.

## Validation and evidence

The Candle fixture has independently generated NumPy logits for a tiny real
GGUF with dense and MoE FFNs, convolution, attention, and grouped queries.
Tests compare full, incremental, and every prefix split; verify A/B/A branch
isolation; reject invalid/foreign state; and inject a failure after the layers
to check transactional commit. A separate explicitly requested ROCm test
checks packed Q5K/Q6K expert operations against dequantized reference math.

```bash
cd "$HOME/src/wt/candle-lfm25"
cargo test -p candle-transformers --lib 'models::quantized_lfm2_moe'
cargo test -p candle-transformers --test lfm2_moe
cargo test -p candle-transformers --release --features rocm --lib \
  'rocm_quantized_experts_match_dequantized_reference' -- --ignored

cd "$HOME/src/lfm2d"
cargo test -p lfm2d
python3 benchmarks/lfm25/evaluate.py \
  --binary './target/release/lfm2d' \
  --model '/tank/ml/models/llama.cpp/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q5_K_M.gguf' \
  --tokenizer '.models/LFM2.5-8B-A1B/tokenizer.json' \
  --out '/tmp/lfm25-evaluation'
```

The daemon's existing real-PII tests also need their normal model fixture;
set `LFM2_TOKEN_CLF_DIR` when using a worktree without that checkpoint.
The benchmark writes all synthetic responses and a summary, verifies repeat
isolation, errors/deadlines, reuse after cancellation, and in-flight SIGTERM.
Its exit status checks runtime invariants; inspect the summary's `passed`
count for schema and severity agreement. Four cases are an integration probe,
not an accuracy certification. The grader does not establish the truth of
free-text effect/reversibility explanations.

On the real Q5_K_M checkpoint, a 40-token greedy smoke continuation matched
llama.cpp exactly. Cold versus cached fixed-input logits for the adjudicator
also matched exactly (max absolute difference and RMS both zero). This is
bounded evidence; different devices, quantizations, shapes, and optimizations
need their own comparisons. The early native-tool prompt produced divergent
cold/cached continuations; that path remains an unqualified comparison mode.

**2026-09-19, the verdict prompt: cold and cached are NOT the same, and the
difference reaches the answer.** Same daemon, same input bytes, greedy, ROCm,
`command-verdict-enum-v1`, 40 val_F rows, the only difference `use_cache`:

- identical generated text on **13 of 40** rows,
- identical verdict on **39 of 40** — one verdict moved on a cache hit alone,
- per-word |delta| at the verdict slot p50 **0.145** nats, p95 1.71, max 3.15.

The resident prefix is 327 tokens and `327 % 128 = 71`, so the cached path's
chunk boundaries sit 71 tokens off the cold path's and even the prefix region is
built with a different last-chunk shape. This does not contradict the
fixed-input logit parity above — that was a bounded measurement on a different
prompt — but it does mean cold/cached equality must be re-measured per prompt and
per optimization, never inherited.

**What this does and does not say about production.** Both production paths are
warm: a new input prefills from the resident prefix (`cached_tokens` = 327), and an
exact repeat of the previous input replays the stored state and logits unchanged.
`use_cache: false` is only ever set by a measurement harness. So production
verdicts are not nondeterministic on this account — every input meets the same
schedule. What the measurement does say is that **a cold reader does not reproduce
what production answered**, which makes `use_cache: false` a poor baseline for
anything meant to describe the daemon, and puts a floor under every cold-path
probe. `lfm25-examine` is a cold reader. `benchmarks/lfm25/prompts/verdict_eval.py
--no-cache` is the arm, and `docs/lfm25-grouped-prefill.md` predicted exactly
this: "Cold and cached chunk schedules now produce different long greedy
generations."

The [2026-09-13 results](../benchmarks/lfm25/results/2026-09-13.json)
record 12/12 schema-and-severity passes (four cases, cold/cached/repeated),
with all corresponding generated text identical. Median synchronized prefill
was 148 ms cached, 1158 ms cold, and 158 ms on repeated branches. Decoding
these 839–1500-token responses took 11.3–22.1 seconds; a later reuse check
slowed to 26.7 seconds on the shared host. Do not infer a throughput win over
Ollama/llama.cpp from this run. In-flight SIGTERM returned 408 and the daemon
exited 0 after model teardown. The full daemon suite passed 148 tests with
one existing ignored test; the final adjudicator suite passed 12 tests.
Candle's three new unit tests, two GGUF integration tests, and explicit ROCm
expert test passed. Targeted Candle clippy and daemon all-targets clippy
(`--no-deps`) passed with warnings denied.

Kaibo's default cast (`crusoe/GLM-5.3`) reviewed whole source files. It found no
snapshot alias/transaction flaw and highlighted model-reference coverage,
template fidelity, worker shutdown, and service tests. Those areas received
additional checks. Review is supporting evidence, not a substitute for tests.

## Follow-up work

- Broader adjudication cases and explanation grading. The reset case can
  name the correct severity while incorrectly claiming discarded uncommitted
  work is recoverable. Schema validation cannot catch that.
- ~~Shorter reasoning and grammar-constrained final JSON~~ — done. The JSON
  is grammar-constrained, and `reasoning: "closed"` removes the reasoning pass
  outright for schema-bearing prompts. Latency is now the report's own tokens.
  Still open for `"open"`, which is not built alongside a schema.
- An append-efficient KV allocator and less GQA materialization; grouped
  quantized MoE prefill is implemented.
- Wider fixed-token reference comparisons, including the native-tool prompt.
- The old dense `lfm2`/`quantized_lfm2` cached multi-token convolution paths
  ignore existing history. This new model fixes its own path; repair and
  regression-test the older modules separately.
