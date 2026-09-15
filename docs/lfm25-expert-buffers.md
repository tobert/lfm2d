# Remove redundant expert-buffer initialization

Each ROCm expert projection previously zeroed its Q8 activation scratch and
output buffer before launching their producers. The producers overwrite both
buffers, so the LFM2.5 decode path spent 88 fill dispatches per token initializing
memory that would be replaced immediately.

Candle now allocates those two buffers without initialization. The Q8 quantizer
writes every quantized value and both half-precision header fields in every
block, including the padded tail. The vector kernel assigns one output for each
row/token/slot; the grouped kernel assigns every valid routed pair's rows,
including partial tiles. Neither output kernel accumulates into the destination.
Routing pairs and atomic count/status buffers retain their existing initialization.
There are no kernel, arithmetic, precision, prompt or sampling changes.

## Measurements

The [paired full-length results](../benchmarks/lfm25/results/2026-09-15-expert-buffers.json)
use the same Q5_K_M checkpoint, F32 activations, rubric prompt, greedy sampler,
full-history repetition penalty 1.05 and 2048-token output cap. All twelve
outputs, reports/errors, finish reasons and completion counts match exactly.

| Warm repeated case | Before decode | After decode | Time reduction |
|---|---:|---:|---:|
| reset --hard | 11.379 s | 11.137 s | 2.12% |
| git restore | 13.901 s | 13.615 s | 2.05% |
| rm -i | 12.727 s | 12.458 s | 2.11% |
| sed read | 7.684 s | 7.207 s | 6.20% |

Median per-case warm decode reduction: **2.12%**. The larger fourth-case change
is noisier than the other three. Median new-input request time with the rubric
cached is **12.175 → 11.932 s**. Cold prefill median is **537.6 → 533.9 ms**;
that small difference does not establish a meaningful prefill improvement.
These are shared-host observations, with CPU builds overlapping portions of
the baseline. Fixture grades remain 7/12 and schema validity 10/12; these
fixtures are not a quality oracle.

A separate short kernel/API trace brackets the second-to-third sampler
completion of the first four-token request: **505 → 417 dispatches**. The only
kernel-count difference is **88 removed buffer fills**, leaving zero such fills
in that decode step. Trace timing is excluded from the latency comparison.
Both traces disable rocprof signal handlers so the daemon owns SIGTERM handling.

## Overwrite coverage and validation

The test-only allocation hook can initialize work buffers with zero, `0xff`
(NaN when viewed as F32), or `0x5a`. The dirty-buffer test compares every output
with the zero-initialized reference on the same device. It exercises all ten
vector formats, grouped Q5/Q6, partial tiles, duplicate routes, unused/hot experts,
nonzero view offsets, and shared/per-expert input rows. A direct probe verifies
that the poisoning hook itself is active.

The Q8 test compares payload bytes across all three initialization patterns,
checks padded quant values and fully padded block headers, and verifies a
64-byte guard after the payload. Widths cross quantization-block and 512-column
padding boundaries; 65,536 rows cross the launcher's 65,535-row chunk boundary.
Deliberately omitting the final chunk fails this test. Independently omitting
an expert output row fails the output test. Both mutations were restored.

Validation on restored source:

- One explicit Q8 GPU test and 15 MoE GPU tests pass; the latter includes an
  existing microbenchmark and independent CPU numerical comparisons.
- Six model GPU regressions, 13 CPU model tests and two NumPy fixtures pass.
- Daemon suite passes 152 tests, with one existing ignored GPU test.
- Full-length runs and short traces pass repeatability, deadline/reuse, and
  in-flight SIGTERM cancellation with HTTP 408 and process exit 0.
- CPU and ROCm lint pass; ROCm retains only the two previously recorded
  unrelated core pointer-cast allowances.

Hosted Kaibo review by **DeepSeek Flash** completed on September 15 after
authorization for the prepared source packet. It found no confirmed correctness
bug. The review covered the allocation paths, tests, allocator, and complete
relevant producer functions; it did not independently verify the omitted MMQ
tile definitions. No implementation changes were required. The raw review and
packet hashes are saved in the private handoff.

The review also proposed further optimizations. Source checks ruled out several
assumptions and produced the [next-optimization assessment](lfm25-next-optimizations.md).

## Build and follow-ups

The daemon pins published Candle revision
`738184605809ca52e09f1dc228401d2ea470aab0`; no local Cargo overrides are needed.
Build with `--features rocm`. See the [run guide](lfm25-adjudicator.md).

The same quantizer is used by other dense integer matmul paths, whose allocation
policy was not changed here. Their redundant initialization is a separate
candidate, subject to the same producer-coverage checks. Residual/normalization,
Q/K normalization/rotary, and decode attention fusion remain candidates. Prompt
and output-length tuning stay deferred.
