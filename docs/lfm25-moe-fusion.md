# Fused ROCm MoE selection and reduction

The LFM2.5 model previously launched separate sigmoid, bias addition, sort,
gather, sum, epsilon addition and division kernels for every MoE layer. It
also materialized the weighted expert outputs before reducing them. These
operations repeat in 22 layers for every generated token.

Candle now specializes F32 32-expert/top-4 selection into one kernel, and
top-4 weighted output reduction into another. The router retains the existing
ROCm bitonic tie ordering and uses unbiased sigmoid scores for the weights.
Both kernels preserve the four-lane reduction tree; multiplication/addition
remain separately rounded. Inputs and snapshots remain immutable. Other shapes
and devices use the explicit tensor decomposition; device errors propagate.

## Measurements

The [saved paired results](../benchmarks/lfm25/results/2026-09-15-moe-fusion.json)
use the same Q5_K_M checkpoint, F32 activations, prompt, greedy sampling,
full-history repetition penalty 1.05 and 2048-token output cap. All twelve
outputs, validated reports/errors, finish reasons and completion counts match
exactly. No prompt or output-length tuning was involved.

| Warm repeated case | Before decode | After decode | Time reduction |
|---|---:|---:|---:|
| reset --hard | 12.079 s | 11.359 s | 5.97% |
| git restore | 14.830 s | 13.923 s | 6.12% |
| rm -i | 13.432 s | 12.808 s | 4.65% |
| sed read | 7.761 s | 7.351 s | 5.27% |

Median per-case warm decode reduction: **5.62%**. Median new-input request time
with the rubric cached: **13.139 → 12.447 s**. Cold prefill median:
**569.5 → 536.9 ms**. These are shared-host observations, with CPU builds
overlapping parts of the baseline. They do not isolate either fusion or promise
the same speedup on other hardware. Fixture grades remain 7/12 and schema
validity 10/12; those fixtures are not a quality oracle.

A separate four-token ROCm kernel/API trace brackets the second-to-third sampler
completion in the first request. The complete decode step falls from **659 to
505 dispatches**, exactly **154 fewer**: six routing launches and one combine
launch removed per MoE layer. The new kernels each appear 22 times. Trace timing
is not used for the latency claim.

## Validation and review

- CPU fused-op positive/negative tests, 13 model unit tests and two independent
  NumPy fixture tests pass.
- Explicit actual-GPU fused-op tests cover tied/saturated scores, 331-row
  prefill, offsets, padded/transposed inputs, non-specialized dispatch, and
  rounding-sensitive values. Six existing model GPU regressions pass.
- Deliberately changing the combine tree to a sequential sum fails the GPU
  comparison; restoring it passes.
- Daemon regression suite: 152 passed, one existing ignored GPU test.
  Full-length runs pass repeatability, deadline/reuse and in-flight SIGTERM
  cancellation with HTTP 408 and process exit 0.
- CPU and ROCm lint pass, retaining only the two previously recorded core
  `unnecessary_cast` allowances for ROCm.

Kaibo DeepSeek (`deepseek-flash`) found no confirmed correctness bug. Review
suggestions added a 32-expert/top-2 dispatch case and transposed combine weights.
Several suggested cases were already covered by the explicit GPU test. An
extreme row count can exceed device launch limits; launch errors propagate,
and the daemon's bounded context is far below those limits.

The first hosted review exhausted its reasoning budget without producing an
answer. The successful retry used a task-local configuration with reasoning
disabled and an 8192-token answer cap; global Kaibo settings were unchanged.

The first candidate trace hit a rocprof SIGTERM-handler conflict. The successful
retry used `--disable-signal-handlers true`, retaining the daemon's shutdown
handler. Unprofiled full-length shutdown checks passed. Sandbox daemon tests
initially could not bind TCP ports; the complete host run passed.

## Local build and next work

Use sibling Candle revision `68fde99c9dbf4916088dfacbf95c37094ededc4c` with the
intentional local Cargo overrides. Build with `--features rocm`; omitting it
produces a binary that correctly refuses ROCm startup. The refreshed daemon
binary is in this worktree's `target/release/lfm2d`.
It is byte-identical to the measured candidate. A final 64-token smoke passes
cache/cancellation/shutdown checks, and all twelve outputs match prefixes of
the full-length baseline; those truncated reports are not quality measurements.

The trace also shows **88 buffer-fill dispatches per decode step**. The indexed
expert path zeroes both its Q8 activation scratch and output for each of 44
projections. Next, prove which buffers are fully overwritten and remove only
redundant initialization. Q8 padding requirements must be preserved. Other
candidates remain residual/normalization fusion, Q/K normalization/rotary fusion,
and decode attention fusion. Prefill QKV layout copies and load-time packed
weight host staging are still recorded. Prompt and output-length tuning remain
deferred.
