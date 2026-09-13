# First LFM2.5 optimization pass

Implemented three established inference patterns before profiling, following
Amy's direction: “We can start profiling once the big hunks are through.”
The checkpoint, prompt, quantization, repetition policy, and output budget
are unchanged. No profiler was run.

## Implementation references and changes

1. **Device-side greedy selection.** llama.cpp's
   [backend greedy sampler](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-sampler.cpp)
   performs argmax through the backend. vLLM applies
   [repetition penalties on tensors](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/penalties.py).
   Candle-NN now has `sampling::GreedySampler`: one evaluation owns a persistent
   token-presence mask and reduction scratch buffers. ROCm combines the
   repetition penalty and argmax in a two-stage reduction and reads back one
   `u32`, replacing 128,000 F32 logits per token plus CPU copying and history
   rebuilding. Exact ties still select the lowest token ID. Raw nonfinite
   logits fail without accepting a token. The daemon creates a fresh sampler
   from the full prompt for each evaluation and warms kernel compilation
   before readiness. Other backends retain a host reference implementation;
   a failed ROCm operation is never retried on CPU.

2. **Merged expert gate/up projection and fused SwiGLU.** vLLM's
   [LFM2 MoE implementation](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/lfm2_moe.py)
   uses merged projections and its fused MoE layer. Candle's new
   `QTensor::cat` concatenates complete packed rows without dequantizing or
   requantizing. Each expert's gate rows precede its up rows. The model can
   then quantize the input activation once, launch one indexed projection,
   and apply SwiGLU in one kernel. GGUFs with different gate/up block formats
   use an explicit separate-projection variant; weights are never converted
   just to enable merging. Packed concatenation stages through host memory
   once during loading, including when weights are on the GPU.

3. **Specialized gated convolution update.** vLLM's
   [ShortConv layer](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/short_conv.py)
   separates prefill from `causal_conv1d_update`. Our one-token ROCm path
   fuses the B*x gate, convolution, C gate, and next-state construction.
   It reads the previous state and writes a fresh allocation. Output and
   next state are contiguous views of that allocation, so snapshots remain
   immutable. The state view retains one extra output vector; it does not
   retain the full request or mutate another branch. Separate F32 multiply
   and add rounding is preserved. Multi-token convolution retains the
   existing reference path.

These are scoped ROCm/F32 optimizations. General batching, paged attention,
speculative decoding, and alternative decoding policies are separate work.
Reference URLs were inspected on 2026-09-13; the llama.cpp checkout was also
read locally. Implementations adapt the patterns to Candle's storage and
snapshot contracts rather than importing either runtime's state manager.

## Results

The [saved comparison](../benchmarks/lfm25/results/2026-09-13-optimizations.json)
contains all twelve paired measurements and binary hashes. Both runs used
the same Q5_K_M checkpoint and four synthetic cases, each cold/cached/repeated.
Every generated output, report, finish reason, and token count matched the
baseline exactly; all twelve schema/severity grades passed in both runs.

Warm repeated evaluations:

| Case | Before decode | After decode | Time reduction |
|---|---:|---:|---:|
| reset --hard | 21.59 s | 19.63 s | 9.1% |
| git restore | 11.57 s | 10.60 s | 8.4% |
| rm -i | 16.09 s | 14.57 s | 9.4% |
| sed read | 13.93 s | 12.71 s | 8.7% |

Median decode-time reduction was 8.9%. This is a shared-host comparison of
the combined changes, not a per-operation attribution or an isolated GPU
benchmark. Initial calls can include warm-up/build contention; the table
uses repeated calls. Prefill did not improve and was generally slightly
slower. Observed startup was 9.18 s before and 10.33 s after; packed-weight
host staging and kernel warm-up add startup work. Explanation correctness
limitations from the initial experiment still apply.

## Correctness checks

- Packed F32/Q5K/Q6K concatenation matches dequantized row concatenation
  exactly, preserves sources and expert boundaries, and rejects incompatible
  shapes, axes, and dtypes.
- Sampler tests cover cross-tile ties, partial tiles, repeated prompt IDs,
  sign-aware penalties, nonfinite rejection, strided/offset inputs, and
  independent histories. Raw-finite behavior is preserved even if applying
  a penalty overflows; the daemon still limits the penalty to 1–2.
- ROCm fused convolution matches both a scalar reference and the original
  tensor decomposition. Tests reuse returned state views and verify prior
  state remains unchanged. Fused SwiGLU equals the existing GPU operations.
- Merged Q5K/Q6K and mixed-format expert paths match separate GPU projections
  exactly for one-token and 17-token batches.
- Independent NumPy tiny-GGUF full/incremental/split tests, daemon regression
  tests, cancellation/reuse, and in-flight SIGTERM all pass. Hardware tests
  were explicitly run with `--features rocm ... -- --ignored`.

Kaibo/DeepSeek reviewed the source. Its missing host-visibility context was
resolved by inspecting `clone_dtoh -> copy_to_host`: that path synchronizes
the owning stream before a blocking HIP copy. Its requested direct ROCm
convolution and negative-input witnesses were added. CPU lint checks and
focused ROCm Candle-NN/daemon lint checks pass. Full ROCm dependency lint
still encounters two pre-existing unnecessary casts in Candle core's
`ops_indexing.rs` and `ops_reduce.rs`; these are recorded for cleanup.

## Next established changes

Continue with grouped expert matrix multiplication for prefill, an
append-efficient KV allocation scheme that preserves snapshot ownership,
and compatible QKV/residual-normalization fusion. The current MoE prefill
still dispatches indexed matrix-vector work per token/expert pair. Profile
after those larger structural changes, as requested.
