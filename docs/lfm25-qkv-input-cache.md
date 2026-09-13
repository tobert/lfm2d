# QKV fusion and reusable input checkpoints

This pass fuses compatible attention projections in Candle and retains one
complete-input checkpoint in the daemon. The published daemon pins Candle
[`936a15a6`](https://github.com/tobert/candle/commit/936a15a6fbcf1f29e38a27058738ffb4ce091753);
no sibling checkout is required. Start with the [adjudicator guide](lfm25-adjudicator.md).

## Attention projections

The Q5_K_M checkpoint has six attention layers. Four use Q5K for all three
projections and now execute one packed QKV multiplication. Layers 10 and 21
use Q6K for V; they execute packed QK plus the original V multiplication.
Weights are concatenated without requantization. Other mixed formats retain
separate projections. This follows the compatible projection packing used in
[vLLM's LFM2 implementation](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/lfm2.py).
Prefill slices can require materialization in attention's existing reshape
path; decode avoids separate projection launches and activation preparation.

## Complete-input cache

The fixed adjudicator prefix stays resident. A second, single-entry cache
holds the exact full token sequence, hybrid model state, and raw next-token
logits after input prefill. An exact repeat clones that state and starts a
fresh sampler. Reports and mutable sampling history are never reused.

`GET /v1/adjudicator` advertises `input_cache_capacity: 1`. Response
`cached_tokens` is zero for cold requests, the fixed prefix length for an
ordinary cached prefill, and the full input length for an exact hit.
`use_cache: false` bypasses both reads and writes. A different cached input
replaces the entry only after successful, synchronized prefill and a final
cancellation check. Failed prefill preserves the old entry; decode cannot
mutate it. Hits still check cancellation and model ownership. Startup and
request checks verify tokenization preserves the cached prefix boundary.

Storage sharing has the existing tensor-level CoW semantics. It does not
remove append copies or introduce paged KV. Capacity is bounded by one input
within the configured context budget. Disk persistence remains deferred.

## Evidence

The [paired artifact](../benchmarks/lfm25/results/2026-09-13-qkv-input-cache.json)
retains all twelve before/after requests and binary hashes. These shared-host
runs deliberately stop at 64 generated tokens; truncated reports are expected
and provide no adjudicator-quality measurement. No profiler was run.

Excluding the first case from warm medians, 64-token decode fell from
740.24 ms to 676.43 ms (8.62%). All twelve before/after continuations match.
Exact-repeat prefill fell to 0.002–0.004 ms. Ordinary cached prefill was about
136–162 ms after warmup; cold prefill remained about 567–574 ms. These are
short smoke measurements, not an isolated throughput benchmark.

CPU and explicit ROCm tests compare full QKV, partial QK, and separate
projections with independent multiplications at sequence lengths 1, 3, and
17. The tiny GGUF's independent NumPy references and state-branch tests pass.
Four daemon cache tests cover logits/state isolation, exact keys, replacement,
cold bypass, failed/cancelled publication, and cross-model rejection. The
full daemon suite passes with its normal PII fixture and socket access.
Live checks cover repeatability, deadlines, reuse after cancellation, and
in-flight SIGTERM. The final published dependency is also built and smoked.

Kaibo DeepSeek (`deepseek-flash`) reviewed whole model and daemon files.
Its ownership concern produced a failing regression test and an explicit
ownership check; its tokenizer-boundary suggestion became a startup guard.
Full review and resolution notes are in `~/exomemory/lfm2d/`.

## Opportunities for this adjudicator

The complete-input checkpoint provides a consistent starting point for
future temperature sweeps and candidate scoring. An end-of-reasoning
checkpoint could compare report alternatives conditional on shared reasoning;
those would not be independent judgments. Candidate log-likelihoods need
multi-token scoring and calibration before margins become useful diagnostics.

Token counts by reasoning/report phase could reveal where the output budget
goes. Hidden-state or routing signatures might help group examples or detect
distribution shift, but similarity cannot justify reusing a verdict. None of
these probes is implemented here. Sampling remains greedy with repetition
penalty 1.05; prompt and report validation are unchanged.

Next hardware work: append-efficient KV with snapshot ownership preserved,
less GQA materialization, and reuse of expert routing metadata/scratch.
Do not extend the input cache by blindly appending user text: the saved token
sequence already includes closing user and opening assistant delimiters.
Different chunk schedules can also change floating-point accumulation.
