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
  --adjudicator-prompt 'lfm2d/prompts/command-verdict-enum-v1.json' \
  --opinion-spec 'lfm2d/prompts/command-verdict-opinion-v1.json'
```

`--adjudicator-prompt` is the spec `/v1/adjudicate` generates from and is
always on the `/v1/opinion` menu; `--opinion-spec` (repeatable, or
`LFM2D_OPINION_SPECS` comma-separated) adds further specs to that menu, each
with its own resident prefix. Specs are named by file stem.

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
- `finish_reason`: `stop` or `length` (`opinion` on an opinion read, below);
  token counts and queue/prefill/decode durations. Prefill timing
  synchronizes the device before stopping the clock.

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
(default 30000, max 120000), `use_cache` (default true), `distributions` and
`opinion` (both below). `use_cache:false` explicitly measures a cold prefill. Prompt plus output must fit the server's
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

### Opinion reads

A prompt spec may carry an `opinion` block — `prefill` (the assistant text
the options continue, e.g. `{"verdict": "`), `options` (the answer set, at
least two, distinct, non-empty) and `close` (the bytes written after each
option, e.g. `",` or `"}`; it makes an option that prefixes another score as
a whole word and is part of every score). A request with `opinion: true`
then does one prefill and teacher-forces every option's canonical tokens
plus the close, decoding nothing. The tokenization is split where the
options' full encodings first diverge, so a BPE merge across the prefill
boundary cannot put an option on an off-canonical path; that split depends
on the tokenizer and the spec, never on the input, and a close that cannot
separate the options stops the daemon at load. `opinion` with
`distributions` is a 400; `opinion: true` against a spec without the block is
a 400.

The response then carries `opinion` and differs from a generation:
`output` is empty, `finish_reason` is `opinion`, `completion_tokens` is 0,
`prompt_tokens` counts the prefilled shared prefix and `decode_ms` is the
option scoring. Inside `opinion`: `options` in spec order, each with its
`tokens`, raw sequence `logprob` (full-vocabulary log-softmax summed over the
continuation, close included) and `prob` renormalised over the options;
`sequence_mass`, the log probability that the model writes exactly one of
the options and its close; `first_token_mass`, the raw log mass on the
options' distinct first tokens; `shared_tokens` and `scored_tokens`; and
`rendered_sha256` of the text the options continue. **No winner is picked.**
Read `prob` beside `sequence_mass`: a low mass means the model was never
asked this question here, and a renormalised number over it is noise that
looks like an answer (decision 5, again). The option names are the spec's,
echoed; nothing in the daemon knows a verdict word. The block is part of
`snapshot_id`, so changing it changes the identity a consumer pins.

The continuation is scored as one block where generation decodes it a token
at a time. On ROCm the two schedules can take different kernels
(`lfm25-chunk-kernels.md`), so an opinion's numbers agree with a generation's
verdict slot in token identity, not to the last nat; the CPU equivalence test
(`opinion.rs`) certifies the alignment, not the serving backend. Measure on
the backend you serve.

Two shipped specs read the verdict enum: `command-verdict-opinion-v1.json`
asks only the verdict, so `{"verdict": "` is its canonical first field;
`command-verdict-enum-v1-opinion.json` reads verdict-first off the
describe-first schema, an arm that was measured (it collapses toward `ask`)
rather than a path the model would write.

### The opinion API: describe-then-read (`/v1/opinion`)

The read above stands the model at a slot the spec fixes. On this checkpoint
a verdict read *before* the model has described the command carries nothing
(F9 gold: AUC 0.46–0.55 at 99% mass); the generative path's own verdict
slot, read *after* `effect`/`scope`/`undo`, is the judgement (AUC 0.74,
25/74 recall at 4 false alarms). `/v1/opinion` serves that slot directly.
It is its own endpoint with its own contract (`docs/integration.md`
invariants 8–11): the typed-decision surface, in the README's words the
System 1 read; in code, the opinion read.

```json
POST /v1/opinion
{"spec": "command-verdict-enum-v1",
 "state": {"command": "cargo clean",
           "facts": "Facts about this command from its manual pages and parser:\n..."},
 "context": null,
 "questions": [{"field": "verdict", "options": ["allow", "ask", "review"]}],
 "use_cache": true, "timeout_ms": 30000}
```

- `spec` names a loaded spec; `GET /v1/opinion/specs` lists them with every
  schema field in emission order, its `kind` (`text`, `choice`, `boolean`)
  and a choice field's `options`. **Read the menu at runtime; never
  hard-code a field name or an option.**
- `state` is rendered into the user turn exactly as the evaluation
  harnesses render it — the `facts` block verbatim (an app builds it; the
  daemon never does), then `Command:` and the command — so a paired
  generative run and an opinion read see the same bytes.
- `questions` names one `choice` field of the spec (v1: exactly one; it is
  a list so the shape survives); `options` may narrow the enum and is
  scored in the spec's order. Questions come from the spec on disk, never
  from request text: framing words in a rendered prompt move label tokens
  by an order of magnitude, and a prompt the instruments never scored is not
  one the daemon serves.
- `context` is reserved and must be `null`.

The daemon renders the prompt, generates every field before the question
under the output grammar (greedy, the repetition penalty, exactly the
generative path), stops when the generated text ends with `"<field>": "`,
and teacher-forces every option plus its close there (`",` when another
field follows, `"}` when the field is last). Each option and its close sit
on their own pre-tokens after the slot — the tokenizer's pre-tokenizer
decides that, checked at load on a probe and held at request time — so the
continuations are the tokens the model would have written.

```json
{"model_id": "...", "snapshot_id": "...", "spec": "command-verdict-enum-v1",
 "described": [{"field": "effect", "value": "Removes build artifacts ..."},
               {"field": "scope", "value": "project"}, {"field": "undo", "value": "easy"}],
 "answers": [{"field": "verdict",
   "options": [{"option": "allow", "logprob": -0.1009, "first_logprob": -0.1009, "prob": 0.905, "tokens": [13537, 1377]},
               {"option": "ask", "logprob": -2.556, "first_logprob": -2.556, "prob": 0.078, "tokens": [1767, 1377]},
               {"option": "review", "logprob": -4.029, "first_logprob": -4.029, "prob": 0.018, "tokens": [63202, 1377]}],
   "sequence_mass": -0.0006, "first_token_mass": -0.0006, "margin": 0.827,
   "shared_tokens": 383, "scored_tokens": 6, "rendered_sha256": "..."}],
 "cache": {"prefix": "hit", "state": "miss", "described": "miss"},
 "prompt_tokens": 344, "cached_tokens": 327, "described_tokens": 39,
 "queue_ms": 1.1, "prefill_ms": 105.8, "describe_ms": 511.6, "read_ms": 46.6}
```

- **No winner.** Nothing in the response names a choice; the caller picks,
  from its own thresholds on its own data, per spec, and refits when
  `snapshot_id` changes.
- `described` is the fields the model wrote before the slot, in emission
  order (a list, because `serde_json` sorts object keys). The read is
  conditioned on it, so it is part of the answer and is echoed.
- Per option: `logprob` is the raw sequence logprob (option plus close,
  full-vocabulary denominators), `first_logprob` the raw logprob of the first
  token alone (the number the F9 slot score reads), `prob` renormalised over
  the options asked. `sequence_mass` and `first_token_mass` are the raw log
  mass on the answer set: read `prob` and `margin` beside them, because both
  are blind to "never asked".
- `margin` is `prob[top] - prob[runner-up]`.
- `cache` names each layer's outcome, `hit`, `miss` or `bypass`
  (`use_cache: false`): the spec's resident prefix; the exact rendered
  prompt (one checkpoint per spec, exact repeat); and the *described*
  state — the prompt plus its generated description at the slot, kept per
  spec in a small LRU (`described_cache_capacity` on the menu, 16). Greedy
  decoding under the grammar is a pure function of the rendered prompt on a
  fixed backend, so the description and the model state after it are
  reusable computation, never a cached answer: a hit skips the prefill and
  the description and scores the options fresh, in tens of milliseconds.
  Every layer is keyed under the spec's identity, so a weight, tokenizer,
  template or spec change empties it.

Measured on ROCm, warm, `command-verdict-enum-v1`, bare commands: a miss
reads in ~490–740 ms (prefill ~100 ms, description ~500 ms for 33–53
tokens, read ~45 ms) against ~630–900 ms for the generative report; a hit
in ~41–46 ms. On the same bytes the read's `first_logprob` and the
generative path's slot agree to ~5e-6 nats (the block-vs-token kernel
difference above), the described fields equal the report's field for field,
and a hit returns bit-identical numbers (`lfm2d/tests/opinion_real.rs`,
ignored: it loads the 8B). `benchmarks/lfm25/opinion_demo.py` shows all of
this against a running daemon; `benchmarks/lfm25/prompts/describe_read_eval.py`
runs the F9 gold set through the endpoint paired on a generative run, and
`score_instruments.py` scores it beside the slot score.

Refused with 400, never queued: an unknown spec, a field the spec lacks or
that is not a `choice`, an option outside the enum, fewer than two options,
more or fewer than one question, a non-null `context`, control tokens in
the state. Refused with 500: a description that ends before the slot (the
grammar makes every field required, so this is the context running out),
or an option that does not tokenize on its own pretokens after the slot.

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
  regression-test the older modules separately. Not to be reached for as the
  explanation of a chunk-shaped effect *here* — it was, once, and the answer
  was the quantized matmul's `b_size <= 8` kernel switch
  (`docs/lfm25-chunk-kernels.md`).
