# Resident DiffusionGemma benchmark

Status (2026-09-06): the full 60-request resident baseline completed with clean
shutdown. Diagnostics exposed historical voting moving EOS earlier in three
converged outputs, truncating completed text. The runtime now commits the latest
argmax directly and this driver rejects finalized tokens differing from the last
denoise step. Early EOS also occurs before finalization; that separate generation
failure and structured-output quality remain under investigation. Empty answers
remain quality failures; missing terminal usage/finish/canvas metadata is fatal.

Machine-local PoC tooling. A Rust driver embeds the vendored mistral.rs library;
Python standard-library tooling schedules requests and reports results. No HTTP
server. The model stays resident, prefix caching is disabled, and streaming
requests run sequentially. The runner manages only its own child process.
There are no tool definitions, tool execution, live advisory inputs, or changes
to the production classifier/gate.

## Run

From the lfm2d root, outside the sandbox for GPU access:

```bash
python3 -m unittest discover -s 'benchmarks/diffusiongemma' -v

# Build in vendor/mistral.rs:
cargo build --release -p mistralrs --features rocm --example diffusion_bench

# Run from the lfm2d root:
python3 'benchmarks/diffusiongemma/benchmark.py' \
  --binary 'vendor/mistral.rs/target/release/examples/diffusion_bench' \
  --model '/home/atobey/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/f7f5b7f5fa82ffc52addd066915886d497f5517b' \
  --label 'tile16-a' --output '/tmp/dg-resident-tile16-a'
```

Defaults: q4k (`--isq q8_0` selects Q8), thinking off (`--thinking on` enables it), startup seed 42, five complete warmup requests,
12 cases x five measured repetitions. Cases are shuffled within each repetition
with a separate fixed order seed. Warmups are saved but excluded from summaries.
Loading and engine initialization are outside request timings. The first five
default cases cover short output, rubric JSON, multi-canvas output, structured
effects, and longer input. Do not run builds or other GPU work alongside timing.

On gfx1151, set `MISTRALRS_ROCM_MOE_Q8_KERNEL='wmma'` to opt into the
experimental integer WMMA implementation for grouped Q8_0 expert projections.
Use `--isq 'q8_0'` for a full Q8 run. The default (or explicit `dp4a`) retains
the existing kernel; other quantization formats keep their existing kernels.
The selection is read once per process and recorded in hardware overrides.
Invalid values fail, and unsupported architectures fail kernel lookup rather
than silently selecting a different implementation. This is a projection
optimization; measure convergence and output quality along with request time.

For a smoke test, add `--only 'pwd,effects-json,long-explanation' --warmup '3'
--repetitions '1'`. New output directories are required; existing runs are never
overwritten. Failures preserve partial results and a typed error record.

For comparison, build this same native driver against each kernel candidate and
preserve the resulting binaries separately. The older `/tmp/dg-750ms-baseline`
is an HTTP/CLI binary and is NOT compatible with this driver interface.
Run baseline then candidate with different label/output directories, and reverse
the order for another round. Compare per-case distributions, output lengths,
pass counts, truncation, and saved answers. Five samples yield descriptive tails,
not a reliable population p95 or evidence of quality equivalence.

## Measurements and limits

- `results.jsonl`: provenance, exact prompts/schedule, every warmup and measured
  answer, engine usage, first nonempty content time, complete-response time,
  finish reason, per-canvas pass counts, and narrow output checks.
- `driver.log`: full process log; not parsed for measurements.
- `requests.jsonl` / `raw.jsonl`: exact native-driver inputs and typed results.
  Each canvas records its finalization receipt timestamp, pass count, convergence
  flag, and intervals between successive non-final progress events. These are
  host-observed intervals including sampling and scheduling, not pure GPU time.
  Finalized canvases also retain raw token IDs and decoded text, including special
  tokens and unused content after the first stop token, for failure analysis.
  `last_step_tokens` and `last_step_text` preserve the latest argmax;
  `tokens` and `text` are the finalized canvas. Token equality is required.
  Older baseline artifacts captured the runtime's historical vote here.
  `converged` means the denoiser met its stability/entropy stopping criteria.
  It is independent of output quality: neither true nor false determines the
  output check result.
  The first pass of each canvas has no preceding progress event and is excluded
  from the interval list. Finalization notifications are not counted as passes.
- `summary.json`: measured requests only, per-case median/nearest-rank p95,
  pass counts, output lengths, truncation counts, field-check results, and
  counts of unconverged canvases and requests containing them, plus
  aggregate output tokens divided by summed response time. This is not an
  autoregressive inter-token decode rate. The first content arrives when a
  block is committed, not when a preliminary canvas guess exists.
  Schema 3 names the aggregate `reported_completion_tokens_per_s`: engine
  completion counts include failed answers and special tokens, so this is not
  quality-adjusted throughput. The previous `useful_output_tokens_per_s` name
  overstated what the metric measures.

Quantization is passed explicitly to the native driver and recorded in both
provenance and its readiness record; a mismatch fails the run. The requested ISQ
format is a policy, not a claim that every tensor has that format: sensitive
tensor promotions and shape-dependent formats still follow the library's plan.

The native driver uses `MultimodalModelBuilder`, `NormalRequest::new_simple`,
and typed `Response` events. The higher-level streaming helper has a one-slot
channel; progress uses best-effort delivery, so the driver instead supplies a
128-slot response channel and rejects dropped/reordered progress or missing
finalization. Kernel geometry is unchanged; the associated runtime correction
removes historical voting at finalization, matching reference latest-argmax
commitment. Use identical finalization behavior for kernel comparisons. No debug
logits dump or profiler may be active during normal timing runs.

**RNG:** diffusion draws use device RNG seeded at model startup. Request seeds
do not reset that generator. Warmups advance it, and different convergence counts
consume different amounts of randomness. Same seed and schedule therefore do
not make cross-build requests identical-randomness pairs. Use repeated runs and
additional startup seeds for robustness; do not claim paired logit parity here.

The native request timeout bounds the full request future. Model startup has a
separate watchdog. The driver sends `Request::Terminate` at completion; the
runner also stops its own child on failure or interruption and enforces a
30-second graceful-stop timeout. It does not discover or stop unrelated processes.

## What we intend to use dGemma for

Recorded sources on Amy's machine:

- `/home/atobey/exomemory/lfm2d/diffusion-classifier.md`: initial experiment,
  negative results, and subsequent reframing.
- `/home/atobey/exomemory/lfm2d/adjudicator-cascade.md`: the authority boundary.
- `training/v10/rubric.md`: current operator-safety rulings, including the
  eventual judge consulting agent guidance for nuanced policy decisions.

Amy asked whether we could use dGemma "sorta like another classifier" with a
templated verdict. The initial proposal was an independent structured read of
effects, scope, reversibility, and severity, potentially informed by execution
history. The later note says single-verdict speed did not compete with the small
classifier and proposes a local adjudicator candidate or shadow judge over the
disagreement set. That is a research direction, not deployment authorization.

The authority rule remains: "adjuticator can approve. the classifier can't ...
it classifies". A classifier contributes evidence, not permission. Failed and
denied actions must not become accepted effects in a trajectory. An eventual
judge can use operator policy that deliberately is not encoded as a classifier
severity label (for example, asking before public GitHub posts).

The case set reflects this with rubric-grounded shell cases, typed effects,
quoted-command-as-data, a failed/denied trajectory, and a policy-judge case.
General explanation, code generation, longer input, and multi-canvas controls
exercise the runtime beyond short verdicts. Commands in prompts are data only.
Expected JSON fields are explicitly scoped sanity checks, not a complete
adjudication evaluation. Ungraded prose remains ungraded; all answers are saved
for human review. The full v9/v10 probe sweep is separate follow-up work.

This harness measures **normal generated answers**. It does not implement
clamped verdict infill, a trained severity head, calibrated class probabilities,
or dependency-ordered slot commitment. The older design note contains superseded
runtime speculation: the later ROCm work identified quantized-weight access,
expert kernels, and host memory queries as contributing factors. Current runtime
findings are in `signoff.md` and `vendor/notes/diffusiongemma-rocm-2026-09-06.md`.


## Direct 512-wide AOTriton experiment

`probe_aotriton_jit.py` bypasses the installed AOT dispatch table and compiles
exported upstream kernel source with Triton. It does not enable a runtime backend.
Export `modules/flash/kernel` from AOTriton commit
`6e00ef3e335b45dfb49065259533b59c68995bfe`, then change only
`IS_JIT_COMPILING = False` to `True` in the exported `fwd_kernel_inner.py`.
This enables upstream constexpr annotations, including specialization of the
unused variable-length branches. Leave the research checkout and system library
unchanged. The script fingerprints the exported Python source in its results.

```bash
.venv-train/bin/python 'benchmarks/diffusiongemma/probe_aotriton_jit.py'   --kernel-dir '/tmp/dg-aotriton-512/modules/flash/kernel'   --output '/tmp/dg-aotriton-512/new-run'   --dtype 'fp16' --bm '32' --bn '32' --warps '8'
```

Requires ROCm GPU access. Each output directory must be new. Defaults: batch 1,
16 query heads / 2 KV heads, head width 512, 256 queries / 1280 KV positions,
noncausal, scale 1. Query strides represent transposed projections; KV is cached
BHSD. Output starts poisoned with NaNs. Two input magnitudes exercise broad and
sharp attention distributions. The FP32 PyTorch math oracle is independent of
the direct fused kernel. Failure of `atol=0.003, rtol=0.02` aborts the run; there
is no fallback. Only passing cases receive timings (three warmups, ten event
samples; kernel and same-dtype math measurements interleaved).

On 2026-09-06, FP16 32x32/eight waves passed at Q/KV lengths 256/1280, 256/256,
and 253/1277. The 256/1280 sharp case measured 2.33 ms versus 5.38 ms math.
BF16 failed the sharp case at one output element; it is not validated by this
probe. These are synthetic attention timings, not Candle or generation speedups.
Full experiment notes: `vendor/notes/diffusiongemma-aotriton-2026-09-06.md`.

## Native experimental FP16 canvas attention

The ROCm build now accepts `MISTRALRS_ROCM_CANVAS_ATTN=triton-fp16` for
DiffusionGemma's bidirectional canvas. This keeps Q8 weights and the existing
integer WMMA expert path. Q/K/V are converted to contiguous FP16 for attention;
the result is converted back to the caller's dtype. Prefix attention continues
through its existing dispatch. Both canvas head widths use the same exported
AOTriton-source kernel interface; the unsupported installed 512 AOT dispatch
table is not used.

Compile a bundle from the pinned kernel export described above:

```bash
.venv-train/bin/python 'benchmarks/diffusiongemma/export_aotriton_canvas.py' \
  --kernel-dir '/tmp/dg-aotriton-512/modules/flash/kernel' \
  --output '/tmp/my-canvas-kernels'
```

The exporter requires gfx1151 and Triton 3.5.1, checks the upstream source hash,
and validates the exact compiled binaries against FP32 before publishing the
manifest. It exports two HIP code objects with runtime query/KV lengths.
Python and Triton are needed to build the bundle, not to run the Rust backend.
The machine-local validated bundle is currently at
`vendor/experimental/aotriton-canvas-gfx1151-v1` (ignored with vendor).

Build the runner as above, then set these variables before using the benchmark:

```bash
export MISTRALRS_ROCM_MOE_Q8_KERNEL='wmma'
export MISTRALRS_ROCM_CANVAS_ATTN='triton-fp16'
export MISTRALRS_ROCM_CANVAS_KERNEL_DIR='/home/atobey/src/lfm2d/vendor/experimental/aotriton-canvas-gfx1151-v1'
```

Use `--isq 'q8_0'`. Set the attention selector to `off` (or unset it) for the
existing backend. Set it to `validate` to run the native kernel and compare
every canvas call with FP32 attention on its FP16 input tensors. Validation
checks every element, including isolated NaNs, with atol .003 / rtol .02 and
aborts on failure. Validation mode adds substantial work; do not use its
latencies to measure the optimized backend.

Scope: gfx1151, batch 1, query heads 16, head256/KV8 or head512/KV2, query length
1..256, KV length 1..8192, BF16 or FP16 inputs, scale 1, no mask/window/softcap/
sinks. The canvas caller already trims local cached KV and supplies noncausal
attention. Unsupported inputs, invalid selectors, missing bundles, checksum
mismatches and HIP launch errors fail explicitly. Selection is read once per
process. The benchmark records both attention environment variables.

Native tests (from `vendor/mistral.rs`, with GPU access and the bundle variable):

```bash
cargo test -p 'mistralrs-core' --features 'rocm' 'rocm_canvas' --lib \
  -- --include-ignored --nocapture
```


Initial native resident comparison (2026-09-06): median of request median step
latencies .713 s native versus .679 s existing attention, about5% slower in a
small fixed-order pilot (four measured requests per mode). The native path is
working and checked, but is not a performance default. Runtime-length kernels
spill more registers than the earlier specialized probe. See
`vendor/notes/diffusiongemma-native-attention-2026-09-06.md` for details.

## Structured accuracy and native report_analysis

`structured_accuracy.py` audits saved runs without changing their original
strict grades. It accepts either a complete raw JSON object or a single enclosing
Markdown JSON fence and reports that format separately from expected-field
correctness. It does not extract arbitrary braces, coerce types, repair JSON,
or accept duplicate keys. Unparseable responses are unscorable, not semantic
passes. Expected-field accuracy does not grade all explanation claims.

```bash
python3 'benchmarks/diffusiongemma/structured_accuracy.py' \
  '/tmp/dg-native-attn-triton-fp16' '/tmp/dg-native-attn-off' \
  --output '/tmp/new-accuracy-audit.json'
```

`tool_cases.jsonl` is a separate eight-case experiment using the canonical
Gemma chat template's typed `report_analysis` declaration. It preserves the
source cases' expected values, replaces their prose-only JSON request with a
request to call the declared reporting tool, and supplies required boolean/string
parameters plus a severity enum. It also adds an explicit output-tool system
instruction; this is a prompting/protocol experiment, not an isolated schema-only
ablation. Original `cases.jsonl` is unchanged.

The Rust driver supplies `tools` with `ToolChoice::Auto`; no strict schema or
forced-tool decoder guarantee is claimed for diffusion sampling. The existing
runtime tool parser returns normalized argument JSON, while finalized canvas
text/token IDs preserve the raw special-token protocol for inspection. The driver
records tool calls, registers no callback, and enables no execution tools. A
missing, wrong, multiple, malformed or schema-invalid call fails the tool check.
The scorer supports this corpus's flat required string/boolean fields and enums;
it is not a general JSON Schema validator.

Run using the ordinary benchmark command with
`--cases 'benchmarks/diffusiongemma/tool_cases.jsonl' --isq 'q8_0'`.
Use Q8 WMMA and `MISTRALRS_ROCM_CANVAS_ATTN='off'` for the accuracy baseline.


Typed pilot (2026-09-06): first version15/16 schema-valid calls and12/16 expected-
field matches. Clearer field descriptions yielded3/4 matches in a targeted
follow-up, with4/4 schema-valid calls. Executable-name extraction improved, but
one failed inspection was still marked completed despite a correct explanation.
V1 is preserved as `tool_cases_v1.jsonl`; the current corpus has the clarified
field descriptions. These are small unpaired experiments, not overall accuracy
estimates. Details: `vendor/notes/diffusiongemma-structured-accuracy-2026-09-06.md`.


Thinking comparisons use `--thinking off` and `--thinking on` with the same
`--max-tokens 1024` override, case corpus, seed, warmups and repetitions. The
override changes scheduled requests without modifying the source corpus. Metadata
records both settings, and the driver readiness record must confirm the thinking
boolean. This controls the checkpoint chat template's thinking mode, not the
number of denoising passes. Compare schema validity, expected-field correctness,
length finishes, reasoning presence and end-to-end latency; a reasoning response
that exhausts its allowance before reporting is a failed answer.

Thinking pilot (2026-09-06): off scored 14/16 expected-field passes; on scored
15/16 and fixed both failed-inspection ledger answers. One thinking response
omitted the opening tool-call marker. Median response latency increased from
4.86 to 12.23 seconds (2.515x); neither mode truncated. Details and limits:
`vendor/notes/diffusiongemma-thinking-2026-09-06.md`.


Expert staging update (2026-09-06): Q8 WMMA now selects shared weight staging
for K>N projections with at most32 routes/expert; larger route batches and down
projections retain direct WMMA. Full-model gate/up time fell339->174ms in a
representative pass; unprofiled median step0.687->0.530s. Long-output throughput
was21.49->25.45tokens/s in a two-repeat pilot, with differing output lengths and
pass counts. Details: `vendor/notes/diffusiongemma-expert-staging-2026-09-06.md`.
