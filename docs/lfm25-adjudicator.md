# LFM2.5 resident adjudicator

The daemon can load `LFM2.5-8B-A1B` directly through our Candle fork, prefill
an adjudicator system prompt once, and evaluate independent inputs from that
immutable hybrid-state snapshot. `POST /v1/adjudicate` returns a validated JSON
report and the original model output. It does not execute anything or change
the classifier/cascade's authority.

The [first optimization pass](lfm25-optimizations.md) adds device-side
selection, merged expert projections, and fused convolution updates while
preserving the initial outputs.

## Development checkout

This initial integration uses sibling worktrees: `lfm2d-lfm25` and
`candle-lfm25`. The root Cargo patch points to `../candle-lfm25/{candle-core,
candle-nn,candle-transformers}`. Keep these adjacent. The fork branch is
`lfm25-moe-snapshots`, commit `3b3f6f9f`, based on
`d9748a8f4622e7d9b66646a96ff23cdbca2fc424`.
Replace the development patch with a published fork revision before merging
this integration into main. No public publication or deployment is part of
this experiment.

```bash
cd "$HOME/src/wt/lfm2d-lfm25"
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
- `output`: original generated text, including reasoning delimiters.
- `finish_reason`: `stop` or `length`; token counts and queue/prefill/decode
  durations. Prefill timing synchronizes the device before stopping the clock.

HTTP 200 means generation completed, **not that a valid report exists**.
Consumers must require `report != null` and `report_error == null`. Validation
checks structure, not truth: free-text explanations still require evaluation.
No verdict is substituted for an invalid report. Generation is not constrained
by a JSON grammar, and there is no retry or repair loop.

The supported schema is deliberately small: one closed object with all fields
required, string/boolean field types, and optional enums. Strings must be
nonempty. Unsupported schema constraints, duplicate fields, unknown fields,
coercions, Markdown fences, trailing prose, and truncated output are rejected.
An optional completed `<think>...</think>` section may precede the object.

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
explicitly in each response. The model reasons before answering, so 512 or
1024 output tokens often truncate the report.

## Snapshot semantics

Candle's new `quantized_lfm2_moe::{Model, State}` separates immutable weights
from attention K/V, convolution history, and position. `State::clone()` shares
immutable tensor storage. Appends allocate replacement tensors; they never
mutate the saved prefix. Forward commits the new state only after success.
Foreign-model state, bad tokens, and context overflow are errors.

This provides CoW semantics at tensor boundaries. It is not a paged KV
allocator: each append still copies prior KV into the new concatenation.
That is a straightforward correctness baseline for subsequent optimization.
The daemon owns one prefix, a bounded queue of eight, and one generation
worker. Deadlines, disconnected callers, and shutdown discard the current
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

cd "$HOME/src/wt/lfm2d-lfm25"
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
- Shorter reasoning and grammar-constrained final JSON; latency is currently
  dominated by autoregressive reasoning, not prefix prefill.
- Batched/grouped quantized MoE prefill and an append-efficient KV allocator.
- Wider fixed-token reference comparisons, including the native-tool prompt.
- The old dense `lfm2`/`quantized_lfm2` cached multi-token convolution paths
  ignore existing history. This new model fixes its own path; repair and
  regression-test the older modules separately.
- Publish/pin the Candle revision when authorized; then remove the sibling
  development patch before integrating into main.
