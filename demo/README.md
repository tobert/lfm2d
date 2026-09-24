# LFM2 search and summary experiments

Python 3.10+, standard library only. Run commands from the repository root.
`conversation.txt` and `thinking.txt` are **invented fixtures**, not session
transcripts. They test a changed plan, an untested hypothesis, constraints,
and an unresolved decision.

## Search and source excerpts: lfm2d

```sh
cargo build --release -p lfm2d
target/release/lfm2d \
  --embedder-dir '.models/LFM2.5-Embedding-350M' \
  --bind-addr '127.0.0.1:8088' --threads 8
```

In another terminal:

```sh
# Bundled 105-document retrieval corpus
python3 demo/lfm2.py search 'how do I stop two threads corrupting shared state'

# Search a conversation, retaining source character offsets
python3 demo/lfm2.py search 'storage schema for passage references' \
  --file demo/conversation.txt

# Select three representative passages, penalizing repetition, in source order
python3 demo/lfm2.py extract demo/conversation.txt --count 3
python3 demo/lfm2.py extract demo/conversation.txt --query 'next steps'
```

For the GPU-enabled build and mandatory CPU/GPU test gate, see
[`lfm2d/README.md`](../lfm2d/README.md#building). The daemon's default device
choice is `auto`; its ordinary build has only CPU compiled. Specify
`--device cpu` for a controlled CPU run.

`--endpoint` precedes the subcommand; it accepts an HTTP(S) root URL or
`unix:///absolute/path/to/lfm2d.sock`. The client discovers the embedder,
pins its ID/weight hash, validates every response, and normalizes raw vectors
before cosine scoring. Query and document roles are always explicit.

Files are split at paragraph/whitespace boundaries into at most 384 UTF-8
bytes per passage. This conservative bound fits the current checkpoint's
byte-level BPE below its 512-token window, including role prefix and BOS.
Offsets are Python character offsets, **not byte offsets**. Non-whitespace
source content is never dropped. Queries exceeding this bound are rejected.
Production chunking should use the actual tokenizer and preserve block IDs.

`extract` uses centroid relevance (or a query) and maximum marginal relevance
to choose **verbatim passages**. It does not understand which latest decision
supersedes an earlier one, and it can omit important exceptions. Its output
is a navigation aid, not conversation compaction. The demo index is in memory
and is rebuilt on each invocation.

## Clickable keyphrases from one block

```sh
python3 demo/keyphrases.py demo/keyphrase_blocks.json \
  --block-id search-thinking --html /tmp/lfm2d-keyphrases.html
```

Open the HTML file locally. Each phrase link highlights its exact source
range. The page contains the selected block and needs no external assets or
JavaScript. JSON output also carries `{block_id, text, start, end, score}`;
offsets are Unicode codepoints, requiring conversion for Rust byte indices
or JavaScript UTF-16 indices. Scores are similarities, not confidence values.

The input is a small JSON array of block snapshots, with an explicit block
ID. Only completed, included, non-ephemeral user/model text or thinking prose
is eligible. Other blocks—including system prompts and tool output—cannot
contribute candidates or influence ranking. Markdown code fences and inline
code are masked while retaining offsets. Unmarked logs/code are ambiguous
and are not reliably filtered by these English heuristics.

At most 48 source phrases (one to three words) are chosen across phrase lengths
and source positions, embedded, ranked against the prose centroid, and selected
with a redundancy penalty. It returns **up to** the requested number, never
inventing fillers. The model does not generate text. This uses the existing
`/embed` endpoint; no token-classifier or generation API is needed.

On the synthetic block, both devices selected “Passage retrieval,” “prefix
looks correct,” and “suspect input truncation.” Three end-to-end extraction
runs, model already loaded, f32 and the same source/50 embedding forwards:

| Device | Seconds per run | Median |
| --- | --- | ---: |
| CPU, 8 rayon threads | 3.763, 3.079, 6.960 | 3.763 s |
| Radeon 8060S, ROCm | 0.766, 0.773, 0.754 | 0.766 s |

This is a tiny local probe, not a latency guarantee. Candidate embeddings
dominate cost; caching unchanged block results and tuning the candidate budget
are the next levers. Per-token embeddings might reduce repeated forwards, but
would need a new service representation and quality evaluation.

## Thinking-block previews: separate generation endpoint

lfm2d's own generation path, `/v1/adjudicate`, continues a spec's state; it
is not a free-form summarizer. This demo calls a separate OpenAI-compatible
server instead (llama-server running LFM2.5-8B-A1B, `--generator`, default
`http://127.0.0.1:2031`). Start that service yourself; this demo neither
starts it nor changes its configuration.

```sh
# One sentence, at most 30 whitespace-delimited words
python3 demo/lfm2.py summarize demo/thinking.txt

# At most 100 words, preserving decisions, constraints, and open questions
python3 demo/lfm2.py summarize demo/conversation.txt --style context

# All choices are explicit and independent of the embedding endpoint
python3 demo/lfm2.py summarize demo/thinking.txt \
  --generator 'http://127.0.0.1:2031' --model 'lfm25-8b-a1b' --max-tokens 1024
```

The 1,024-token output allowance is a **request budget**, shared by reasoning
and the final answer. It can be increased. It is unrelated to the encoder's
512-token input window. The demo caps source input at 32,000 UTF-8 bytes and
refuses larger sources rather than silently truncating them. It inherits the
generator's sampling defaults. Non-`stop` completion, empty output, and an
exceeded word limit are errors; output is not repaired or silently retried.
Word limits here suit the English fixtures, not language-independent UI sizing.
The word-count guard does not prove factual accuracy or enforce one sentence.

For kaijutsu, a thinking-block preview should be derived, expandable content:
keep the original block, and key the cached preview by block ID/content hash,
generator revision, and prompt version. Generate once the block completes;
invalidate on edits. Show only the final content, not the summarizer's own
reasoning. Batch local compaction remains a separate workload requiring
retention tests for decisions, corrections, constraints, and provenance.

## Tests

```sh
python3 -m unittest discover -s demo -p 'test_*.py'
cargo build --release -p lfm2d
python3 demo/e2e.py -v
# Optional: requires the separate generator to already be running
python3 demo/e2e_generation.py -v
# GPU hardware gate: requires the selected backend and all four checkpoints
bash demo/test_devices.sh rocm
```

The E2E suite starts the actual compiled daemon with real embedding weights
on a temporary Unix socket, drives the Python client and CLI through HTTP,
and stops the child even on failure. It needs ~1.4 GiB for the f32 model and
permission to bind a local socket. Missing binary/weights **fail**, not skip.
Override `LFM2D_BIN` or `LFM2_MODELS_DIR` as needed. It checks:

- Batch order/single-input equivalence, role asymmetry, model identity headers,
  vector dimensions, and HTTP error propagation.
- Retrieval over the existing 105-document/35-query corpus, including 68
  hard negatives; quality floors are R@1 ≥80%, R@3 ≥95%, negative wins ≤10%.
- The actual CLI preserves all fixture passages, including the final paragraph,
  and extractive output refers back to exact source slices.
- Selected execution device is reported correctly; keyphrase provenance and
  code/prompt exclusions are preserved on the real service.

Fast tests cover malformed vectors, model swaps, input bounds, UTF-8 splitting,
final-newline preservation, diversity selection, and summary output rejection.
The opt-in generation smoke test invokes the actual preview CLI against the
separately running service. It checks completion/word limits and prints the
answer for human review; it does not grade faithfulness. It is not a dependency
of the encoder E2E suite. Override `LFM2_GENERATOR`/`LFM2_GENERATOR_MODEL` to
select another local endpoint/model.

## Measurements and remaining work (2026-09-12)

Real-daemon retrieval reproduced the library baseline: **31/35 R@1 (88.6%),
35/35 R@3 (100%), 1/68 hard-negative wins**. This is a small developer-prose
corpus, not validation on real kaijutsu conversations. In the conversation
fixture, the broad query “What is still undecided?” ranked the dense-retrieval
decision ahead of the intended unresolved storage-schema passage. No model
tuning or recall-floor relaxation was used to hide that miss. The CLI coverage
test requests all six passages; ranking has its own corpus quality test.

Single local LFM2.5 Q5_K_M probes (as-deployed sampling; not a benchmark):

| Probe | Output budget | Result |
| --- | ---: | --- |
| First conversation summary | 512 | Truncated; rejected |
| Thinking preview | 512 | 3.84 s, 391 completion tokens; concise, preserved uncertainty |
| Conversation summary | 1,024 | 7.30 s, 791 completion tokens; useful overview, omitted no-publishing constraint |

The last omission is why these probes support trying a preview, not trusting
compaction. The context prompt now explicitly mentions constraints. A separate
probe including constraints retained “No publishing today” (9.32 s, 785
completion tokens); prompt and stochastic differences prevent attribution.
`chat_template_kwargs.enable_thinking=false` **still emitted reasoning** on
this deployed template, so the demo does not advertise it as a working switch.
One repeated-prompt response reported 142 cached prompt tokens; the older
blanket claim that LFM2.5 cannot reuse any prefix is too strong.

Service readiness, checked in source and against live deployment arguments:

- lfm2d already has the dense embedding HTTP interface kaijutsu's
  `Lfm2dEmbedder` expects. The live deployment currently loads classifier,
  router, and PII models, **no embedder**. Enabling it needs weights/config
  and memory; this demo does not mutate that deployment.
- Kaijutsu already wires that adapter into index startup, pins model identity,
  and normalizes vectors. Its current context index truncates the combined
  conversation into one embedding: passage storage/aggregation and retention
  of late decisions need work before long-context retrieval is reliable.
- The encoder silently truncates beyond 512 tokens. Model discovery does not
  advertise this limit, tokenizer identity, or truncation counts. Record and
  resolve that consumer contract gap before increasing input sizes.
- ColBERT works in the Rust library, with parity/quality evidence, but is not
  served by lfm2d. It would need its own per-token representation/scoring API;
  it cannot substitute directly into the single-vector index interface.
- Existing Rust tests cover model parity, classifier/PII real-engine routing,
  stub HTTP contracts, TCP/UDS serving and shutdown. These Python tests add
  real-weight embedding transport and retrieval coverage. Kaijutsu → index →
  lfm2d with real weights still needs a separate cross-project E2E test.

The new service device/tracing behavior is documented in
[`lfm2d/README.md`](../lfm2d/README.md): explicit CPU/GPU selection, startup-only
fallback, resource metadata, W3C parent/tracestate propagation and export tests.
The hardware gate requires both devices; it does not turn missing hardware into
a passing skip. The default Python E2E remains explicitly CPU (`LFM2D_TEST_DEVICE`
overrides it; `LFM2D_EXPECT_DEVICE` pins the expected result of `auto`).

Review: Kaibo, DeepSeek cast (`deepseek-flash` explorer and synthesis).
Fixed empty-query handling, CRLF boundaries, and malformed generation-response
errors with failing-then-passing tests. Retained deliberate refusal of overlong
summaries; added explicit generation smoke and split-passage CLI coverage.

## System 1 opinion demos (`/v1/opinion`)

Three scripts and `show.py` to run them in order: a read that answers any
question a spec can name, then two looks at the numbers under it. Stdlib
Python; the `email-triage-v1` spec under `specs/` and the input lines under
`inputs/` are invented fixtures, like the ones above. The spec names its own
input (`"input_label": "Email"`), so a request sends `state: {"input": ...}`.

Start a daemon (a throwaway on a free port is the pattern) with the email
spec on its boot menu:

```sh
target/release/lfm2d \
  --adjudicator-model .models/LFM2.5-8B-A1B/LFM2.5-8B-A1B-Q5_K_M.gguf \
  --adjudicator-tokenizer .models/LFM2.5-8B-A1B/tokenizer.json \
  --opinion-spec demo/specs/email-triage-v1.json \
  --adjudicator-context 4096 --device rocm --bind-addr 127.0.0.1:18171 --threads 8
```

The acts find the spec by its boot name, so it must be a boot spec: an
upload's `spec` is its content hash, never a file stem.

- **`show.py`** — the matinee: three acts in one terminal, title cards
  between, items driven through a pty so they echo like someone typed
  them and each next line waits for the child's own prompt. `--auto` skips
  the between-act pauses; `--acts 13` picks a subset (an unknown act is an
  error). It refuses to start when `email-triage-v1` is not on the menu.
- **`blink_anything.py --spec email-triage-v1 --field verdict`** — the
  verdict vocabulary is the app's, not ours. The primitive routes a
  support inbox (`auto_close` / `human_read`) over a spec written as a prop.
- **`xray.py --spec email-triage-v1`** — what the model wrote, and the
  distribution it wrote it from. Asks every choice field in one request,
  puts the full option distribution (prob, raw `first_logprob`, token ids,
  raw mass) under each written value — the last choice field has no later
  field to reveal its write, so it shows the read alone — flags near ties,
  and prints the exact bytes the last read continued (`rendered: true`)
  after checking their sha256 against `rendered_sha256`. A daemon older
  than the flag refuses the request (the request type denies unknown
  fields); `--no-prompt` skips the view, and a response that omits
  `rendered` when asked ends the demo loudly.
- **`asked.py --spec email-triage-v1 --field feeling`** — was the model
  even asked? Each item is read over the full menu, then with each option
  left out once (`item :: a,b` asks a subset). Leaving one out needs a
  field with at least three options (`feeling`; `verdict` has two, and the
  daemon refuses a one-option question). The grammar walks the model to
  the slot, so high full-menu mass proves the question was put, not that
  the input made sense. Measured 2026-09-24 on the author's lfm2d-system1 deployment (image 0.3.0) with
  `feeling` over `inputs/asked.txt` (n=4): three emoji 100.0%, "what is
  the capital of France?" 93.3%, the cancellation email 99.3%, and the
  store-hours email 82.1% — a real email read lower than emoji. On the
  shell spec the acts were built against it was ~99.9% on everything. Narrowing drops the mass by what
  the omitted options held while `prob` renormalises the rest into a
  confident-looking answer. Invariant 9, on one screen.

Every script checks its spec, field and options against
`GET /v1/opinion/specs` at load: no script hard-codes a field name, an
option, or which option means go.
