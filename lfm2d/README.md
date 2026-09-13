# lfm2d

HTTP sidecar/daemon serving `lfm2-encoder` heads over a Unix domain
socket and/or TCP, from ONE process. Built because:

- candle's `from_mmaped_safetensors` copies every tensor into private
  anonymous RSS on load — N processes loading the same 350M checkpoint cost
  N × ~1.4 GiB, not one shared mapping.
- A single forward pass already saturates ~13.6 of this box's cores
  (measured) — per-request concurrency has no headroom to spend, so this
  daemon serves every request through ONE serial inference worker thread by
  design, not as an unoptimized bottleneck.
- `kaibo` cannot link `candle-core` at all (it hard-wires
  `tokenizers`+`onig`, breaking kaibo's musl static-build invariant) — an
  HTTP boundary is the only way kaibo can reach these models.

See `src/lib.rs`'s module docs for the full architecture writeup (worker
thread / channel design, the `InferenceEngine` trait that decouples router
tests from candle, and the two response conventions this API uses).

## Building

```sh
cargo build --release -p lfm2d
```

The default build is CPU-only. A GPU build selects a device at startup:

```sh
# AMD, with the ROCm toolchain/runtime installed
cargo build --release -p lfm2d --features rocm
target/release/lfm2d --device auto --threads 8 \
  --embedder-dir '.models/LFM2.5-Embedding-350M' --bind-addr '127.0.0.1:8088'
```

Optional `cuda` and `metal` features expose those Candle backends too. ROCm
on the Radeon 8060S is hardware-tested here; CUDA/Metal are not verified by
that result. Candle core/nn are pinned together to the published
`tobert/candle` revision in the root manifest because crates.io 0.11 lacks
ROCm. A clean checkout can fetch this revision without the ignored vendor tree.

`auto` tries compiled backends in ROCm/CUDA/Metal order, then uses CPU if
device initialization is unavailable. Every failed probe/reason is logged,
and execution metadata reports the **selected** backend. Use `--device cpu`
to force CPU or `--device rocm` (etc.) to require a GPU; an explicit GPU never
falls back. `--device-index` selects the backend's device ordinal.

Fallback is startup-only: checkpoint/configuration errors, model smoke-test
failures, and inference failures remain errors, without retrying on CPU.
GPU builds still require their linked runtime libraries even in CPU mode;
missing shared libraries prevent startup before selection can run. The
ordinary CPU build remains portable to machines without those libraries.
The container recipes below build the ordinary CPU variant; a GPU container
also needs the runtime libraries and device access.

Changing the execution device can change classifier scores and near-tied
verdicts. Validate deployment calibration before moving a safety head; the
numerical parity gate is not a replacement for traffic evaluation.

Run the mandatory device gate when changing backend selection/execution:

```sh
bash demo/test_devices.sh rocm
```

It builds once with the chosen backend, compares real embedding/classifier/
router/PII heads on CPU and GPU, then runs the real HTTP suite under explicit
CPU, explicit GPU, automatic GPU selection, and automatic CPU fallback after
an invalid GPU ordinal. A missing GPU/driver/weight file fails this gate.
The hardware Rust test is opt-in (`--ignored`, invoked by this script); normal
CPU-only tests cover selection success/failure through an injected probe.

After proving the pinned ROCm patch set locally, polish it for upstream as
separate work. Amy will review the upstream policy and handle posting; the
local service work does not submit changes upstream.

Container image (build context must be the repo ROOT, since `lfm2d`
depends on the parent crate via `path = ".."`):

```sh
podman build -t lfm2d:latest -f lfm2d/Containerfile .
```

See `deploy/lfm2d.container` for an example podman quadlet unit (models
volume-mounted, `CPUWeight=`/`Nice=` set, both socket and TCP examples,
OTLP env wiring) and `deploy/k8s.yaml` for a complete Deployment + Service
(probes, a measured memory recipe, CPU requests without limits, OTLP env
wiring — see "Deploying on k8s" below for the reasoning behind each).

## Configuration

CLI flags, each with an env-var fallback (`clap`'s `env` feature):

| Flag | Env var | Meaning |
| --- | --- | --- |
| `--embedder-dir` | `LFM2D_EMBEDDER_DIR` | `Lfm2Embedding`-shaped checkpoint dir; backs `/embed` |
| `--classifier-dir` | `LFM2D_CLASSIFIER_DIR` | `Lfm2SequenceClassifier`-shaped checkpoint dir; backs `/predict`, `/v1/classify`, `/v1/cascade` |
| `--router-dir` | `LFM2D_ROUTER_DIR` | Prompt-Router checkpoint dir; backs `/v1/route`, `/v1/cascade` |
| `--token-classifier-dir` (repeatable) | `LFM2D_TOKEN_CLASSIFIER_DIR` (comma-separated) | `Lfm2TokenClassifier`-shaped checkpoint dir(s) — REPEATABLE, unlike the three heads above; backs `/v1/spans`, `/v1/spans/credentials` |
| `--candidate-classifier-dir` | `LFM2D_CANDIDATE_CLASSIFIER_DIR` | A SECOND classifier scored shadow-only beside `--classifier-dir` on every `/v1/classify` and `/v1/cascade`. Never listed in `/v1/models`, never changes a response byte; it records an agreement counter (or, on a failed candidate pass, a failure counter) and discards the verdict. This is how a candidate head is measured against live traffic without serving it |
| `--dtype` | `LFM2D_DTYPE` | `f32` (default), `f16` or `bf16`, for every head. Read the flag's own help before reaching for `f16`: these checkpoints ship f32 natively, so f16 is a real loss of resolution at ~1.6× latency, and `LFM2.5-Embedding-350M` ships bf16, where bf16→f16 can produce inf/0 rather than rounding. It is a memory lever, not a speed one |
| `--device` | `LFM2D_DEVICE` | `auto` (default), `cpu`, `rocm`, `cuda`, or `metal`; requires corresponding compiled GPU feature |
| `--device-index` | `LFM2D_DEVICE_INDEX` | GPU ordinal (default `0`); values exceeding the driver's signed 32-bit range are refused |
| `--cascade-route` (repeatable) | `LFM2D_CASCADE_ROUTES` (comma-separated) | Candidate routes for `/v1/cascade` — server-side config, not a request field |
| `--cascade-severe-label` (repeatable) | `LFM2D_CASCADE_SEVERE_LABELS` (comma-separated) | Which classifier labels count toward the severity ranking, **in ascending severity order — position is ordinal rank**. Default `mutating,destructive`, which is an old checkpoint's vocabulary; the deploy manifests pass `situation-normal,data-critical`. Duplicates are refused, and the resolved ranking is echoed at startup |
| `--log-input-hash` | `LFM2D_LOG_INPUT_HASH` | `true`/`false`, must be spelled out (not a bare flag); default `false`. Attaches a hash of `/v1/spans`/`/v1/spans/credentials` request text — never the text itself — to that call's trace/log span; see "Observability" below |
| `--socket-path` | `LFM2D_SOCKET_PATH` | Unix domain socket to serve on |
| `--bind-addr` | `LFM2D_BIND_ADDR` | TCP address to serve on, e.g. `127.0.0.1:8088` |
| `--threads` | `LFM2D_THREADS` | Size of rayon's global thread pool (candle's matmul runs on it transitively), set BEFORE any model load. Defaults to `std::thread::available_parallelism()` |

Standard OTEL env vars also apply (`OTEL_EXPORTER_OTLP_ENDPOINT`,
`OTEL_SERVICE_NAME` default `lfm2d`, ...) — see "Observability" below; none
of these are `clap` flags, they're read directly by the OTLP exporters and
`main.rs`.

At least one of `--socket-path`/`--bind-addr` and at least one of
`--embedder-dir`/`--classifier-dir`/`--router-dir`/`--token-classifier-dir`
are required — startup
fails loudly (`exit 2`, config problem named) otherwise. Model loading
itself is synchronous and happens before the socket is ever bound: a bad
checkpoint path or an incompatible config also fails loudly at startup
(`exit 1`), never lazily on the first request.

Only ONE model per head kind is supported for the embedder/classifier/router
heads — `/predict`, `/v1/classify`, `/v1/route` don't take a model-selection
parameter, so a deployment serving multiple classifiers, say, needs multiple
`lfm2d` processes. Token classifiers are the one exception: `--token-classifier-dir`
is repeatable, and `/v1/spans`/`/v1/spans/credentials` take an optional
`"model"` field to pick among them — see the API section below.

## API (v1)

The semantic contract a consumer programs against — numbered invariants,
cited by number from outside this repo — is `docs/integration.md` at the
repo root. It never renumbers. This section is the reference for the
endpoints themselves; where the two overlap, the contract wins.

```
GET  /healthz                 liveness
GET  /readyz                  readiness
GET  /v1/models               [{id, kind, weight_hash, labels?, hidden_size}]
POST /embed                   TEI-compat  {"inputs": str|[str], "kind"?: document|query}
POST /predict                 TEI-compat  {"inputs": str|[str]}
POST /v1/classify             {"inputs": str|[str]}
POST /v1/route                {"input": str, "routes": [str]}  -> RAW cosines
POST /v1/cascade              {"clauses": [str]}
POST /v1/spans                {"inputs": str|[str], "model"?: str}
POST /v1/spans/credentials    same shape; credential.* entities only
```

The input field is `inputs` (TEI's spelling), not `texts`. `/v1/route`
takes singular `input` because it scores one text against many routes.

- `GET /healthz` → `200 "ok"` — process alive, never touches the worker.
- `GET /readyz` → `200` once every configured model has loaded, `503`
  before.
- `GET /v1/models` → `[{id, kind, weight_hash, labels?, hidden_size}]` —
  every loaded model. `kind` is `embedder`/`classifier`/`router`/
  `token_classifier`. `labels` is present for a classifier (its full label
  set) or a token classifier (its distinct `entity_types()`, BIOES prefix
  stripped and deduped — NOT the raw `id2label`, which on the PII detector
  is 161 entries wide for only 40 distinct entities). `weight_hash` is
  sha256 over the checkpoint's `model.safetensors`, 64 lowercase hex chars
  — the audit trail this whole daemon exists partly to satisfy (kaish
  approval-chain rulings).
- `POST /embed` — TEI-compatible-ish. `{"inputs": "text"}` or
  `{"inputs": ["a", "b"]}`, optional `"kind": "query"|"document"` (default
  `document`; this is the E5-style asymmetric-embedding side selector —
  see `candle_lfm2_encoder::TextKind`'s docs on why picking wrong quietly
  degrades retrieval). Response: `[[f32, ...]]`, one pooled vector per
  input, in input order. `model_id`/`weight_hash` travel as
  `X-Model-Id`/`X-Model-Weight-Hash` response headers, not in the JSON body
  — keeping the body wire-compatible with plain TEI clients that don't
  know these headers exist.
- `POST /predict` — TEI-ish sequence classification. `{"inputs": ...}` →
  per input, `[{"label", "score"}, ...]` covering ALL labels (full
  softmax, never just top-1), sorted by score descending. Audit headers
  same as `/embed`.
- `POST /v1/classify` — our full contract, not TEI-compat: `{"inputs":
  [...]}` → per input `{"scores": {label: prob}, "top": label, "model_id",
  "weight_hash"}`.
- `POST /v1/route` — `{"input": str, "routes": [str, ...]}` → `{"model_id",
  "weight_hash", "routes": [{"route", "cosine"}, ...]}`. **RAW COSINE
  ONLY** — this router's softmax is route-count arithmetic and carries no
  confidence information (see `candle_lfm2_encoder::routing`'s module
  docs), so it is never exposed on this endpoint.
- `POST /v1/cascade` — `{"clauses": [str, ...]}` (pre-decomposed; clause
  decomposition is the caller's job, e.g. via a kaish `Plan`) →
  moderations-FLAVORED but with **no `flagged` boolean and no threshold
  anywhere**:
  ```json
  {
    "winner": {"index": 1, "clause": "rm -rf .", "severity_scores": {"...": 0.9}},
    "lane": {"route": "shell", "cosine": 0.91},
    "clauses": [
      {"index": 0, "clause": "...", "severity_scores": {"...": 0.9}, "top_severity": "informative"},
      {"index": 1, "clause": "rm -rf .", "severity_scores": {"...": 0.9}, "top_severity": "data-critical"}
    ],
    "models": [{"model_id": "...", "weight_hash": "..."}, {"model_id": "...", "weight_hash": "..."}]
  }
  ```
  The label names above are one checkpoint's, shown to make the shape
  readable — read the real ones from `GET /v1/models` at fire time, never
  from this file. `models[0]` is always the severity classifier that
  produced every `severity_scores` map; `models[1]` is the router that
  produced `lane`.
  This handler does NOT reimplement `candle_lfm2_encoder::cascade`'s
  rank-by-severity-route-the-winner aggregation — it calls straight through
  to the library's `Cascade::run`. **No global severity cutoff exists by
  measurement** (benign max 0.3415 vs data-critical min 0.3440 — see
  `src/cascade.rs`'s module docs in the parent crate): consumers must rank
  clauses WITHIN a statement, never threshold any number here against a
  fixed cutoff. `routes` and `severe_labels` are `--cascade-route`/
  `--cascade-severe-label` server config, not request fields — see
  `src/lib.rs`'s "cascade configuration is server-side" section for why
  this was a judgment call, not something the spec pinned down explicitly.
- `POST /v1/spans` — TEI-style, `{"inputs": str | [str]}` plus an OPTIONAL
  `"model"` (which loaded `--token-classifier-dir` head answers; required
  as a 400 naming every loaded id when 2+ are loaded, implicit when
  exactly 1 is) → `[[{"start", "end", "entity", "score"}, ...], ...]`, a
  bare array of arrays, one span list per input, same TEI-shaped
  convention as `/embed`/`/predict`. Audit pair travels as `X-Model-Id`/
  `X-Model-Weight-Hash` headers, same reason.
  - **`start`/`end` are UTF-8 BYTE offsets, not codepoints/chars, not
    UTF-16 code units.** A Rust caller (kaibo, the first consumer) can
    slice `&text[start..end]` directly; a Python/JS caller must re-encode
    to UTF-8 bytes first — indexing a Python `str` or a JS string with
    these numbers directly is wrong on any non-ASCII input.
  - **The matched text is NEVER returned, by design, not as a missing
    feature** — no `word`/`quote`/`text` field, not even opt-in. The
    caller already has the text it just sent; echoing a credential back
    would only create a second copy of it in every log this response
    passes through. Stricter than GCP DLP on purpose (DLP ships a
    no-matched-text mode as an option; here it's the only mode).
  - **`score` is the MINIMUM softmax probability across the span's
    tokens**, passed straight through from `lfm2_encoder::Span::score`.
    Minimum rather than mean because a span is a conjunction of per-token
    decisions — it is wrong if any one token is wrong — so a single
    coin-flip token inside an otherwise confident credential is exactly
    the signal averaging would hide. **These numbers therefore read
    systematically lower than tools that average** (Hugging Face's
    grouped-entity pipeline) or that report a recognizer's own confidence
    (Presidio); do not compare them across tools, and do not threshold
    them against a fixed cutoff, same as every other number this API
    returns.
- `POST /v1/spans/credentials` — identical contract to `/v1/spans`, server-
  side filtered (via `Lfm2TokenClassifier::credentials`) to only the
  `credential.*` entity family. A separate endpoint, not a query flag on
  `/v1/spans` — AWS's precedent of shipping "contains PII" as its own API.

Errors: `{"error": {"message", "type"}}`. `type` is `"bad_request"` (400 —
malformed/empty input, or a call against a head this instance never
loaded) or `"internal"` (500 — a loaded model's forward pass failed, or the
worker thread itself died). Never a silently-wrong `200`.

## Calling lfm2d from another program (non-normative)

Design notes for a client. The promises are in `docs/integration.md`;
these are the consequences of how the daemon is built.

**Assume a queue, not a pool.** One forward pass already saturates most of
a 16-core box, so client-side fan-out buys nothing — requests serialize
through one worker by design. Batching within a request is the lever;
concurrent connections are not. The `lfm2d.worker.queue_depth` gauge makes
overload visible before it is painful.

**Budget ~1.4 GiB of resident memory per head**, and do not expect heads to
share a trunk unless they were trained on a common frozen one: the base
encoder and the shipped Prompt-Router have 0 of 148 trunk tensors in
common, and a full finetune diverges from both.

**Rank within a request; never threshold across requests.** The score
ranges of benign and severe clauses overlap by measurement, so no global
cutoff exists — which is why `/v1/cascade` ships no `flagged` boolean.
If your policy needs a yes/no, that decision belongs in your code, where
it is visible as your choice.

**The ordinal rank that picks the winner is not on the wire.**
`/v1/cascade` returns per-label `severity_scores` and each clause's
`top_severity`; the ranking number itself (`ClauseVerdict::severity_score`
in the library) is the expected ordinal rank, `0.0..=n` for n severe
labels, and is deliberately not repeated in the response. A caller that
wants it derives it from `severity_scores` and its own severe-label set.
It was previously a plain sum over the severe labels, which was not
monotone in severity; recorded values from an older build are on a
different scale.

**A checkpoint and its severe set travel together.**
`--cascade-severe-label`'s ORDER is the ordinal scale, ascending, least
severe first. Reversing it inverts every ranking and raises no error,
because both names are valid — read the startup line that echoes the
resolved ranking after any change.

**Log the `weight_hash` beside any decision you record.** It is the audit
trail: it ties a verdict to the exact weights that produced it, and it is
how a rollback is told apart from a regression after the fact.

**Keep failures loud.** An unknown severe label is refused by name; a dead
worker exits the process rather than serving a healthy-looking 200. Do not
paper over a 5xx with a permissive default — but do not block on the
daemon either. `docs/integration.md` invariant 6 is the rule: proceed with
your own baseline controls and record that you skipped.

### Standards this API follows, and where it stops

`/embed` and `/predict` are TEI-shaped, so a TEI-conformant client works
against them unchanged. The `/v1/*` endpoints have no standard to follow:
TEI has no token-classification endpoint at all, and KServe V2/OIP — the
only real standard here — is tensor-clunky and was deliberately not
adopted. The span shape follows the PII-service convention (Presidio,
Amazon Comprehend, GCP DLP) instead, which is the closest prior art for
this job.

## The advisory hook (`hooks/`)

`hooks/` holds this repo's reference consumer: a `PreToolUse` Bash hook
that scores every shell command through `/v1/cascade` and appends the
comparison to a local JSONL. **It decides nothing.** The regex rules it
carries make every outcome, byte-for-byte identically to the baseline
guard it was cloned from — `test_parity.py` gates that on 33 cases, and
the model's verdict is recorded beside the decision, never enforced.
The point is to learn where the two disagree, and which one is right when
they do.

Clause extraction is **plan-first**. `kaish_plan.py` runs
`kaish --plan-file -` and renders each simple command in the resulting
plan as one clause: canonically quoted argv, redirect operators and
targets explicit, heredoc bodies stripped out of argv and tagged by kind,
pipelines split into their members. `clause_split.py` is the recorded
fallback for input kaish rejects, and every scored row carries
`split_path`, so the two populations stay separable and the live fallback
rate is a count rather than a guess. Compound statements go to
`/v1/cascade` (ranked within the statement); a single clause goes to
`/v1/classify`; past a clause budget the row falls back to one batched
`/v1/classify`, which keeps per-clause truth without inventing a winner
client-side.
Each planned clause also carries the **structured facts** behind its text
— `name` (the verb), `args`, and `redirects` as `{kind, target}` — so a
caller can decide something about a clause without re-parsing the string
it was just handed. This exists for the escalation pattern where a static
check runs *after* the classifier: `echo restored` scoring data-critical
is dismissible only if the caller can see both that the verb is `echo`
and that nothing redirects, and `echo restored` vs `echo restored >
/etc/shadow` differ by exactly that. Matching the rendered text instead
is the prose-reading mistake the plan path exists to remove. The logged
row carries `plan.commands[]` (verb + redirects per sent clause, indexed
the way `heredocs` is); `args` is left out of the log because it is
already in `text` verbatim and would be most of the added bytes.

One trap the facts handle: an fd-dup like `2>&1` encodes its destination
in the *kind*, and kaish still emits a placeholder target for it. Passed
through raw, a consumer would see a redirect that reads as "writes a file
named `null`". The target is nulled when the kind contains `&`, and a
real file — `echo hi > null` — keeps its target. The discriminator is the
kind, never the target's spelling.

`install.sh bootstrap` wires the hook on a machine that has never had one;
`install.sh install` swaps a compatible existing regex hook for it and
refuses when there is no baseline to gate parity against. The daemon
endpoint is written into the installed command string from `LFM2D_URL`;
the hook's own default is loopback, because a remote daemon is a
machine's configuration and not the code's.

Run the tests with `python3 lfm2d/hooks/test_<name>.py` — they are plain
scripts that exit non-zero on failure, not a pytest suite.
`test_advisory_live.py` needs a reachable daemon and refuses to skip
rather than reporting green while testing nothing, so point it at one:
`LFM2D_URL=http://<host>:8088 python3 lfm2d/hooks/test_advisory_live.py`.
Its defaults are loopback, like the hook's.

### Known gaps

- **The renderer is not pinned.** `LFM2D_KAISH_BIN` defaults to whatever
  `kaish` is on `PATH`, so a kaish upgrade changes how clauses render
  mid-stream. That has already happened once, and it changed the argv
  rendering of a whole flag family. A score floor measured across such a
  boundary is not comparable with one measured after it: pin
  `LFM2D_KAISH_BIN`, or re-baseline deliberately once the move is
  accepted.
- **The hook emits no OpenTelemetry.** The daemon is fully instrumented;
  the hook writes only its local JSONL, so the disagreement data — the
  entire product of the advisory phase — is readable only by opening a
  file on the machine that produced it.
- **Command text leaves the machine and is written to disk.** It goes to
  the daemon over whatever `LFM2D_URL` points at, and to a `0600` local
  log. Commands can contain secrets; weigh that before installing this
  in every session on every machine.
- **The circuit breaker leaves holes.** After repeated failures the hook
  stops calling for a cooldown. Those skipped calls are logged as
  `circuit_open` rows rather than omitted, so a quiet stretch is
  distinguishable from a stretch where nothing was asked — but they are
  still gaps when the log is mined as training signal.
- **Enforce mode is deliberately unimplemented.** `LFM2D_HOOK_MODE=enforce`
  prints a refusal and falls back to advisory. Wiring an untested
  enforcement path and leaving it reachable is how a "temporary" mode
  ships.

## Being a good k8s/k3s container citizen

**Graceful shutdown.** On SIGTERM or SIGINT: `/readyz` flips to 503
immediately (so a Service's endpoint controller stops routing new traffic
here), every listener (Unix socket and/or TCP) stops accepting new
connections via axum's `with_graceful_shutdown`, in-flight requests are
allowed to finish (a forward pass here is sub-second), then the process
exits 0. The drain is bounded — `shutdown::DEFAULT_DRAIN_TIMEOUT` (10s) —
so a wedged request can't hang shutdown forever; `serve()` returns `Ok(())`
either way, exit code 0 regardless. Rust runs as PID 1 in the container, so
this handler is installed explicitly (`tokio::signal`) — there's no init
process to translate the signal for us. See `src/shutdown.rs` and
`src/main.rs`'s `install_signal_handlers`; `tests/graceful_shutdown.rs`
drives the sequence programmatically (via `server::begin_shutdown`) and
asserts all three parts: `/readyz` flips, the in-flight request completes
200, `serve()` resolves.

**Exit waits for the engine drop.** After `serve()` returns, `main.rs`'s
`exit_after_serve` waits up to `shutdown::WORKER_EXIT_TIMEOUT` (5s) for the
worker thread to drop its engine (`worker::WorkerExit`), and only then exits
0. `serve()` returning drops the last `WorkerHandle`, which ends the worker
loop, and the engine drops on the worker thread. A GPU engine's drop frees
device memory through the driver, so exiting underneath it runs
libamdhip64's atexit teardown mid-free; a ROCm build segfaulted exactly
there (see Known problems). If a request outlived the drain cap, its
connection task still holds a `WorkerHandle`, so the wait times out, a
warning is logged, and the exit is still 0. That keeps 10s + 5s inside
Kubernetes' default 30s grace period. It also means the fix NARROWS the
race to that already-pathological path rather than eliminating it: exit
then runs under a live engine, and a GPU build can still fault there.
`tests/shutdown_exit.rs` proves the ordering through the real binary with
no GPU: a stub engine whose drop sleeps and then writes a marker must have
written it before the process exits. `demo/e2e.py` (run by
`demo/test_devices.sh`) fails the device gate on any nonzero SIGTERM exit.

**Crash-only on worker death.** A worker-thread panic today would leave the
HTTP server answering 500s forever while `/healthz` still said 200 —
alive-but-useless, and kubelet never restarts a pod whose process hasn't
exited. `main.rs`'s production path spawns the worker via
`WorkerHandle::spawn_crash_on_panic` instead of the plain `spawn` the test
suite uses: a monitor thread calls `std::process::exit(1)` if (and only if)
the worker thread terminates via panic — NOT on the clean channel-closure
exit a graceful shutdown produces (see `worker_thread_outcome_is_a_crash`
in `src/worker.rs`, unit-tested directly). `tests/worker_crash_monitor.rs`
proves this through the REAL compiled binary: it spawns `lfm2d` itself
with `LFM2D_TEST_CRASH_ON_WORKER_PANIC=1` (a test-only env-gated branch in
`main.rs` that swaps in a panic-on-call stub engine instead of loading real
checkpoints — real models aren't needed to prove the process-exit
mechanism), fires one request over a real HTTP connection, and asserts the
CHILD PROCESS itself exits nonzero.

**Startup observability + `--threads`.** Startup logs (via `tracing`, not
`eprintln!`) the full effective config, each loaded model's `{id, kind,
weight_hash}`, `std::thread::available_parallelism()`, and a hand-rolled
container-runtime probe (`/.dockerenv` → docker, `/run/.containerenv` →
podman, `KUBERNETES_SERVICE_HOST` → kubernetes, `container` env var,
`/sys/fs/cgroup/cpu.max` when readable) — see `src/probe.rs`, unit-tested
with injectable root dir + env snapshot, no new dependency. `--threads N`
(env `LFM2D_THREADS`) sizes rayon's GLOBAL thread pool — which candle's
matmul runs on transitively — via
`rayon::ThreadPoolBuilder::num_threads(n).build_global()`, called BEFORE
any model load (rayon's global pool can only be built once, so this has to
win the race against candle's own lazy default); a build failure exits
loudly rather than silently falling back to the default pool size.

**OpenTelemetry: metrics + traces + logs over ONE pipeline** (`tracing` +
OTLP, not bare Prometheus). Human-readable stderr logging is always on
(`tracing-subscriber`'s `fmt` layer); when `OTEL_EXPORTER_OTLP_ENDPOINT` is
set, OTLP export layers/providers are added on top and ONE loud line is
logged either way ("OTLP export disabled (... unset)" or "OTLP export
enabled"). A missing/unreachable collector never crashes the daemon or
slows inference — every exporter is a BATCH exporter (bounded queue, drops
under backpressure) over gRPC/tonic with rustls (never openssl). One span
per HTTP request (`method`, `route`, `status`); within it, a `worker_call`
span carrying `queue_wait_ms` and `inference_ms` — the queue-wait vs.
compute split is the whole point of a serial worker (see `src/worker.rs`'s
module docs on how a `tracing::Span` is created request-side and recorded
onto from the worker OS thread).

Incoming `traceparent` and `tracestate` are extracted using W3C Trace Context.
The HTTP server span adopts the remote parent, and the worker span carries
that trace context across its OS-thread boundary. Remote sampling is honored;
absent/invalid parent headers start a local trace, duplicate `traceparent`
headers are refused as parents, and split `tracestate` headers retain order.
This extraction does not depend on setting a global propagator. Upstream
clients still need to inject their headers; lfm2d has no downstream inference
HTTP calls to inject into.

HTTP spans also carry `http.request.method`, `http.route`, and integer
`http.response.status_code`, with server span kind and error status on 5xx.
Unknown paths use `<unmatched>` so arbitrary path text never enters trace or
metric labels. Export-capture tests in `telemetry.rs` verify actual parent IDs,
tracestate, sampling, worker timing, HTTP errors, and execution resources.

**`/v1/spans`/`/v1/spans/credentials` telemetry is held to a stricter bar.**
This endpoint exists to find live credentials, so request text contains
them by definition. No span, log event, or metric anywhere in this crate
ever records input text, span offsets, or a matched substring — the
`worker_call` span for these two operations carries only `operation`,
`queue_wait_ms`, `inference_ms`, and (opt-in) `input_hash`, same as every
other operation's span, nothing request-shaped added. `--log-input-hash`
(default OFF) attaches a sha256 hash of the request's input text — never
the text itself — to that call's span as a trace/log attribute ONLY; it is
never threaded onto an OTLP METRIC label (unbounded per-input cardinality
would wreck VictoriaMetrics). `tests/spans_telemetry_safety.rs` is the
load-bearing proof: it embeds a known secret string in a request, captures
every field tracing emits via a custom `Layer` (no OTLP collector
required), and asserts the secret — and the opt-in hash's raw bytes —
never appear; a mutation test during development (temporarily logging the
raw input) confirmed this test actually fails when it should.

Resource attributes: `service.name`
(`OTEL_SERVICE_NAME`, default `lfm2d`), `service.version` (crate version),
`lfm2d.execution.device_type` (`cpu`/`gpu`), `lfm2d.execution.backend`
(`cpu`/`rocm`/`cuda`/`metal`), `lfm2d.execution.dtype`, and
`lfm2d.model.<kind>_hash` per loaded model — set once, from
`main.rs`, AFTER models finish loading (weight hashes aren't known any
earlier; see `src/telemetry.rs`'s module docs). The same resource is attached
to traces, metrics, and logs. Device metadata comes from the loaded engine,
not host hardware inventory; optional `lfm2d.execution.device_name` is omitted
until a hardware name is available through the backend interface. Metrics:
`lfm2d.worker.queue_depth` (observable gauge over an `AtomicUsize`,
incremented on send, decremented when the worker picks a command up),
`lfm2d.request.duration` (histogram, by route+status),
`lfm2d.inference.duration` (histogram, by operation kind), `lfm2d.requests`
(counter), and the shadow-classifier pair `lfm2d.candidate.agreement` /
`lfm2d.candidate.failure` (counters, only when
`--candidate-classifier-dir` is set). Read those two together: a candidate
forward-pass failure never reaches the caller, but it IS counted, because
agreement over an unknown denominator is not a measurement — a candidate
failing systematically would otherwise read as near-perfect agreement over
a shrinking set of successes. The two histograms carried a
`_duration_ms` name until the OTel exporter's unit suffix made it
`..._ms_milliseconds`; `src/telemetry.rs` is the source of truth for the
name, and a running image may still be exporting the old one until it is
rebuilt. Verified against a real OTLP/gRPC capture server with a
classifier and the Prompt-Router loaded: spans arrived correctly nested
(`http_request` parenting `worker_call`) carrying `queue_wait_ms`/
`inference_ms`, all four metrics arrived, logs arrived with the effective
config and per-model weight hashes, and SIGTERM still drained and exited 0
cleanly despite transient export failures observed against one candidate
endpoint (see "Problems noted, not fixed" below — worth knowing, not a
code defect).

**Deploying on k8s** — `deploy/k8s.yaml` is a complete Deployment +
Service: `readinessProbe`/`livenessProbe` against `/readyz`/`/healthz`, a
`startupProbe` budgeted for a COLD PVC (local-disk load measured at
~1.4-3.2s for two heads here; a cold network-backed PVC can take far
longer, which is what the startup probe's generous budget is for — the
readiness/liveness probes only start once it succeeds),
`terminationGracePeriodSeconds` comfortably above the 10s drain cap,
`resources.requests` with a MEASURED memory number (not an estimate — see
the manifest's comment for the exact `/proc/<pid>/status` reading), CPU
`requests` WITHOUT `limits` (rationale in-manifest: rayon's thread pool
auto-sizes to the visible core count at startup and doesn't re-check a
limit applied later — a CPU limit just throttles the same thread count
against fewer cores, no benefit; this workload scales by replica count,
not by starving one replica), and OTLP env wiring. lfm2d is STATELESS — N
replicas behind the Service is the horizontal scaling path; a previous
version of this doc's k8s example pinned `replicas: 1` in a way that read
as a requirement, which was wrong and has been fixed (see `deploy/lfm2d.container`'s
trailing comment and `deploy/k8s.yaml` itself).

## Judgment calls worth knowing about

The task spec left a few things implicit; here's what was decided and why
(also in the relevant doc comments):

1. **Audit headers vs. audit body fields.** The spec requires "every
   inference response carries `{model_id, weight_hash}`" AND specifies
   `/embed`/`/predict` as bare TEI-compatible arrays with no room for those
   fields. Resolved by putting them in `X-Model-Id`/`X-Model-Weight-Hash`
   response headers for those two endpoints, and directly in the JSON body
   for `/v1/classify`/`/v1/route`/`/v1/cascade` (which are "our full
   contract," not TEI-compat).
2. **`/v1/cascade`'s `routes`/`severe_labels`.** The spec's request shape
   is `{"clauses": [...]}` only, but the library's `Cascade::run` also
   needs a route set and a severe-label set. Made these server-side startup
   config (`--cascade-route`, `--cascade-severe-label`) rather than
   request fields — a cascade specialist's lane set and severity
   definition are properties of the deployment, not something each caller
   should be re-specifying per call.
3. **`top_severity`'s meaning.** Read as "the classifier's own argmax
   label for this clause" (mirroring `/v1/classify`'s `top`), NOT the
   severe-label-set ranking sum (`ClauseVerdict::severity_score` in the
   library) — the latter isn't exposed as a separate number in v1, since a
   caller can already derive it from `severity_scores` plus their own
   knowledge of which labels they consider severe.
4. **One model per head kind — except token classifiers.** No `/predict`/
   `/v1/classify`/`/v1/route` request carries a model-selection parameter,
   so this daemon serves at most one embedder, one classifier, one router
   at a time. `/v1/spans`/`/v1/spans/credentials` are the deliberate
   exception: `--token-classifier-dir` is repeatable and the request takes
   an optional `"model"` field, because "N secrets/PII detectors behind one
   sidecar" (a general PII head plus a narrower secrets-only head, say) is
   a realistic deployment shape in a way that "N embedders" or "N routers"
   is not for this daemon's current consumers.
5. **`/v1/spans`'s `score` is the weakest token, not the average.** A
   span is a conjunction of per-token decisions, so its trustworthiness is
   its weakest token's; averaging would hide the coin-flip token that
   indicates a misplaced boundary. The consequence for a caller is that
   these numbers are not comparable with other PII services' — see the
   API section's `/v1/spans` entry and `types::SpanResult::score`'s doc
   comment.

6. **No observe-only endpoint.** Asked for a way to report actions a
   consumer handled WITHOUT scoring them — read-only calls that bypass the
   daemon — so coverage is countable. Deliberately not added: an endpoint
   that records without scoring turns a model server into an event sink,
   a different service with different retention and scaling properties,
   and the right sink already exists. This daemon exports traces, metrics
   AND logs over OTLP (see Observability), so a consumer should emit its
   bypassed actions as spans carrying the attributes it would have sent,
   marked `lfm2d.skipped=read_only`. Scored and unscored actions then live
   in one store, coverage accounting is a query rather than a second write
   path, and lfm2d keeps one job.

## Problems noted, not fixed

- Each head loads its own trunk; there is no trunk sharing across heads.
  This is not a gap to close in the general case — the base encoder and
  the shipped Prompt-Router have 0 of 148 trunk tensors in common, and
  only heads trained on a *common frozen trunk* can share one, which costs
  measured accuracy. Budget memory per head.
- The quadlet's `HealthCmd` only proves the binary still execs, not that
  `/healthz` answers — the runtime image ships no `curl`/`wget` on purpose
  (minimal image; see the Containerfile). A real HTTP healthcheck should
  run from OUTSIDE the container against `/healthz`/`/readyz`.
- **An endpoint that accepts TCP is not an endpoint that forwards gRPC.**
  During live OTLP verification a port that connected instantly timed out
  every actual export, even at a 20 s `OTEL_EXPORTER_OTLP_TIMEOUT`, while
  the standard OTLP/gRPC port on the same box captured everything from the
  identical exporter code. The daemon behaved correctly either way — the
  failed exports neither crashed it nor blocked SIGTERM's clean exit 0 —
  but pointing `OTEL_EXPORTER_OTLP_ENDPOINT` at a genuinely reachable
  collector is a precondition this daemon cannot check for you.
- `WorkerHandle::spawn_crash_on_panic`'s monitor mechanism is proven
  end-to-end via `tests/worker_crash_monitor.rs`, but that test needs a
  `LFM2D_TEST_CRASH_ON_WORKER_PANIC`-gated stub-engine branch in `main.rs`
  to avoid depending on real checkpoints being present in CI. The pure
  decision logic (`worker_thread_outcome_is_a_crash`) is ALSO unit-tested
  directly, so the crash-vs-clean-exit distinction has a fast, model-free
  test in addition to the slower real-binary one.
- **`lfm2d.model.token_classifier_hash` collides across 2+ token
  classifiers.** `telemetry::resource()` sets one OTLP resource attribute
  per LOADED KIND (`lfm2d.model.<kind>_hash`), a scheme that predates
  `--token-classifier-dir` being repeatable — with 2+ token classifiers
  loaded, only the last one's hash survives in that attribute (the others
  are silently overwritten by `Resource::builder().with_attribute`'s
  last-write-wins semantics). Every loaded model's real hash is still
  correctly reported per-model in `GET /v1/models` and in every
  `/v1/spans` response's `X-Model-Weight-Hash` header — this only affects
  the OTLP resource-attribute shortcut, not the audit trail itself. Worth
  fixing (e.g. `lfm2d.model.token_classifier.<id>_hash` per head) before
  anyone relies on that specific attribute with 2+ token classifiers
  loaded; not fixed here because it needed a resource-attribute shape
  decision, not a spans-endpoint one.
- **FIXED 2026-09-13: the final OTLP flush never ran.**
  `telemetry::TelemetryGuard` flushes every batch exporter when dropped, but
  `main` has always left through `std::process::exit`, which never unwinds
  its frame. So the guard never dropped, and whatever the exporters had
  buffered was lost on every shutdown. A kaibo review found it.
  `exit_after_serve` now drops the guard explicitly, after the worker wait,
  bounded by `shutdown::TELEMETRY_FLUSH_TIMEOUT` (5s) so a dead collector
  cannot eat the grace period. `tests/shutdown_exit.rs` asserts the flush
  runs, and runs after the engine drop.
- **FIXED 2026-09-13: a ROCm build crashed after a clean SIGTERM
  shutdown.** On 2026-09-12 a ROCm daemon logged `shutdown complete,
  exiting 0` and the harness reported exit 1. On 2026-09-13 the same shape
  reproduced as SIGSEGV in 1 of 11 paired CPU+ROCm shutdowns. The core dump
  put the fault on the `lfm2d-worker` thread: the worker loop ended, dropped
  `RealEngine`, and its last tensor freed the ROCm allocator
  (`RocmAllocator::release_all` -> `hipFree`) while `main`'s `exit(0)` ran
  libamdhip64's atexit teardown. Two contributing factors, two fixes:
  - lfm2d exited without waiting for the engine drop. `main` now waits for
    it (see "Exit waits for the engine drop").
  - The candle fork's exit guard read its flag once, before the free loop.
    tobert/candle `d9748a8f` makes the atexit hook wait for releases
    already under way; its `rocm_exit_race` test went from 24 of 24
    SIGSEGV to 24 of 24 exit 0.
  After the lfm2d fix, 80 of 80 paired shutdowns exited 0 (at the old rate
  a clean 80 is ~0.05% luck). `demo/test_devices.sh` now asserts the exit
  status.

## Resident causal adjudicator

LFM2.5-8B-A1B GGUF inference, reusable hot-prefix state, and validated JSON
reports are available as a development integration. See
[the adjudicator guide](../docs/lfm25-adjudicator.md) for build/run commands,
HTTP semantics, measured limitations, and the sibling Candle worktree setup.
