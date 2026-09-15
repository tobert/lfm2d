# Next LFM2.5 runtime optimizations

Assessment after the September 15 expert-buffer review by **DeepSeek Flash via
Kaibo**, checked against Candle `73818460`. These are candidates, with no measured
speedup yet. Keep the checkpoint, F32 activations, prompt, greedy sampling,
repetition penalty and output cap fixed. The current measured decode step has
417 kernel dispatches; launch counts alone do not identify the time bottleneck.

A subsequent [local and public runtime comparison](lfm25-runtime-comparison.md)
puts the existing local llama.cpp service about 5–6% ahead in decode throughput,
with a larger cold-prefill gap. This is a shared-host snapshot, not an isolated A/B.

## Concrete next work

### 1. Fuse expert combination and residual addition

DeepSeek suggested residual addition in producer kernels. The smallest concrete
application is the existing `lfm2_combine_4` kernel: finish its four-expert weighted
sum, then add the residual before writing the result. The model currently returns
that sum from `Moe::forward` and performs a separate tensor addition in the layer
loop. This could remove **22 launches and intermediate tensors per token**.

Preserve the four-lane reduction tree and separate F32 multiply/add rounding.
Pass the residual explicitly through the feed-forward interface and write fresh
output storage so branching model state remains immutable. Test against the
existing decomposition with values that expose floating-point cancellation,
offsets and invalid shape/device inputs, then run model branch/replay checks and
paired daemon measurements with the same generation settings. This is the next
small structural change to try.

### 2. Have SwiGLU produce the down projection's Q8 activation

DeepSeek proposed batching the 44 quantizations. That does not fit the dependency
chain, but inspecting it suggests a different fusion: calculate SwiGLU values
and quantize them within the same kernel. Each expert's down projection consumes
only this activation; the separate F32 intermediate can disappear on this path.
This could remove another **22 launches per token**.

This is our refinement of the review suggestion. It requires an explicit internal
interface for already-quantized activations between the producer and expert
projection, including dimensions, row stride and device ownership. Preserve the
SwiGLU F32 rounding, Q8 warp reductions, half headers and padded columns. Compare
the fused payload byte-for-byte with SwiGLU followed by the existing quantizer,
including poisoned destinations and partial blocks, before numerical/model tests.

### 3. Tune expert vector-kernel workgroups for gfx1151

This is the review's most useful less-explored hardware suggestion. The indexed
expert kernel hard-codes **four wave32 warps for one output row**. Merged gate/up
has 3,584 rows and down has 2,048 rows, each for four selected experts: 14,336 and
8,192 workgroups respectively. Across 22 MoE layers that is **495,616 expert
workgroups per token**, despite only 44 expert-matvec dispatches. This count is
derived from launch geometry, not evidence that workgroup scheduling dominates.

Compare one/two/four-wave variants and multiple rows per workgroup using the
actual Q5/Q6 shape and format mix. Extend the existing expert microbenchmark to
batch one and alternate baseline/candidate runs after warmup. Rust launch geometry
and kernel constants must agree; changed reduction order needs numerical and
full-model validation. Retain only demonstrated improvements on the target APU.

## Other review suggestions: source-checked disposition

| Suggestion | Assessment |
|---|---|
| Batch/reuse all 44 Q8 quantizations | Already shared across top-4 experts for merged gate/up. Down needs new post-SwiGLU values; layers depend on previous layers. Raw pointer equality cannot establish value identity with pooled storage. |
| Fuse RMSNorm reduction and scale | Already one kernel doing both. Norms across layers are dependent. Residual+norm and Q/K norm+RoPE remain separate, previously identified candidates. |
| Eliminate grouped-routing readback during decode | This readback is absent at batch one. Grouped dispatch starts at 64 tokens for this model. Compact histogram/prefix-sum/scatter packing and GPU-side dispatch are possible prefill work; validate IDs before expert-weight reads and preserve duplicate routes. |
| Cache function handles and scratch | Still plausible host-overhead work. Built-in lookups bind the device and consult a cache; custom-function lookup hashes source each call. The greedy sampler already retains its handles. Removed fill kernels do not prove the remaining loop is launch-bound: allocator locks, maps and reference counting remain. |

Decode attention fusion and removing proven-redundant initialization in dense
matmul paths remain on the earlier list. Prompt and output-length tuning remain
deferred.

## Review limits and deferred validation

DeepSeek found no confirmed bug in the expert-buffer change. Its description of
Q8 padding needs precision: padded quantized values are zero, but only **fully
padded blocks** have zero scale/sum headers. Tests already distinguish these.
MMQ tile definitions were outside the packet; existing grouped tests cover their
integration. Its guess about the Q5 template flag is not additional evidence of
correctness; the independent minimum-correction test supplies that evidence.

Record a low-priority test for grouped column-grid overflow rejection near 65,535
tiles, preferably through a small launch-plan validator rather than a huge GPU
allocation. This is a general API boundary test, outside the daemon's context
limit, and not a newly identified buffer bug.
