# Shared KV heads and reusable expert routing

This pass removes two repeated operations in LFM2.5's ROCm path. It preserves
the model's immutable snapshot interface, prompt, and sampling policy. No
profiler was run.

## Attention uses shared KV heads directly

The checkpoint has 32 query heads and 8 KV heads. Previously each attention
layer expanded K and V fourfold to match query heads, including on every
decode token. Now the four query heads sharing a KV head become extra GEMM
rows. The original K/V allocations are consumed directly, including their
capacity-padded batch strides.

Scores are reshaped back to query-head order before the unchanged causal mask
and softmax. For the value multiplication, query rows are grouped again and
the result returns to the original head order. This removes K/V materialization
without changing the attention formula. It does not fuse attention or remove
its score/probability tensors. CPU and ROCm use this path; other backends retain
the prior implementation until their layout behavior is validated.

## Expert projections share one prepared map

For grouped ROCm prefill, `GroupedMoeRouting` captures a routing decision as
GPU-packed token/expert pairs and counts. Gate/up and down reuse the same map,
count/status readback, launch bound, and routing scratch. Each projection still
quantizes its own activation values. All participating weight stacks must be
compatible, on the same device/stream and have the same expert count.

The prepared map is an immutable snapshot of assignments. Subsequent writes to
the original ID tensor cannot alter it. Its buffers live through the projections
and never enter the saved model state. This is reuse within a forward pass;
routing decisions are not reused across unrelated layers, inputs, or tokens.

The 32-expert/top-4 checkpoint uses grouped prefill at 64 tokens and above.
Decode and short suffixes keep indexed vector dispatch. Unsupported mixed-format
projection combinations keep their existing per-projection dispatch. No failed
GPU operation is retried through another path.

## Prompt caching is already active

The shared 331-token rubric is processed once at startup. Its saved state holds
attention K/V, convolution history, position, and model ownership. A new request
clones the snapshot and processes only its appended input. An additional
single-entry cache stores a complete input and next-token logits for exact
repeats. Each request gets fresh sampling history and generates a new answer.

Cache reuse requires the same token prefix and loaded model. Arbitrary editing
of already-processed text is not supported. Generated reasoning remains new
work; prompt caching does not make a 1,000-token answer instantaneous.

## Validation and measurements

CPU/GPU attention checks compare with repeated-head attention, with an independent
scalar causal reference on a multi-batch case. Cases cover head ratios 1/3/4,
query/decode lengths, cached prefixes, offsets, and padded cache capacity.
Independent tiny-GGUF NumPy full/incremental/all-split references also pass.
Prepared routing tests cover shared/per-expert activations, Q5K/Q6K, source-ID
mutation, scratch reuse, invalid IDs and shape/device/expert-count mismatches.
Existing grouped numeric and snapshot isolation tests remain green.

The [paired full-length result](../benchmarks/lfm25/results/2026-09-15-gqa-routing.json)
retains twelve before/after measurements, binary hashes, and both run summaries.
All outputs, reports/errors, finish reasons, and completion counts match exactly.
Cached repeats, reuse after cancellation, and in-flight shutdown pass.

| Warm repeated case | Before decode | After decode | Time reduction |
|---|---:|---:|---:|
| reset --hard | 13.01 s | 12.38 s | 4.85% |
| git restore | 16.08 s | 14.72 s | 8.44% |
| rm -i | 14.62 s | 13.47 s | 7.85% |
| sed read | 8.32 s | 7.83 s | 5.87% |

Median repeated-call decode-time reduction: **6.86%**. Median cold prefill is
572.1 → 567.6 ms, which is too small a difference to claim a meaningful win
on this shared host. CPU builds/tests overlapped parts of both runs. These
numbers measure the combined pass; they do not attribute time to either change.
Changing GEMM shapes can select different accumulation algorithms, so exact
text parity is an observation from these twelve runs, not a general guarantee.
Schema validity 10/12 and fixture grades 7/12 are unchanged and remain limited
prompt/model observations, not accuracy certification.

Kaibo DeepSeek (`deepseek-flash`) reviewed the whole files and immediate backend
interfaces. It found no correctness bug. Its missing model-wiring witness led
to a GPU regression covering one prepared map and two merged or three separate
projection uses, with numerical comparison to per-projection packing. Disabling
model reuse makes that test fail; restoring it passes both layouts. Hardware
attention/routing tests were explicitly run; ignored tests are not counted as
passes without execution. The review's assertion that changing GEMM shapes
cannot affect accumulation order was not accepted as a guarantee.

Recorded follow-ups: load-time packed-weight concatenation still stages through
host memory; fused QKV prefill can copy slices once to reshape and again into
head-major layout. One count/status synchronization per grouped MoE layer remains.
The support query selects a policy rather than validating all possible inputs:
large routing-dependent grid overflows remain explicit errors before launch.
No automatic error fallback was added.

CPU lint passes. ROCm lint passes with only the two previously recorded
`unnecessary_cast` findings in core indexing/reduction allowed. Full daemon
regressions pass (152 tests, one existing ignored). Older vector-test GPU skips
remain recorded separately; the new/grouped GPU tests fail on unavailable hardware.

The daemon retains intentional sibling Candle overrides during local unpublished
development. Use `../candle-lfm25` at `819573783dd422204d11dd9794d0e355754b49a3`.
See the adjudicator guide for launch instructions. Final binary is in the
daemon worktree at `target/release/lfm2d`.

The final rebuilt binary passed a separate 64-token cache/cancellation/shutdown
smoke; all twelve short outputs match prefixes of the full-length baseline.
Those deliberately truncated reports are not a quality measurement.
