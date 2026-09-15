# Append-efficient LFM2.5 KV snapshots

LFM2 MoE now appends attention keys and values into growable buffers instead
of allocating and concatenating the complete history at every token. The
public `Model`/`State` interface, sampling policy, and attention arithmetic
are unchanged. The append path is enabled on CPU and ROCm; other backends
retain their original concatenation path. No profiler was run.

## Ownership and failure behavior

Each private cache view holds a committed length and shares a K/V allocation.
A monotonic atomic reservation boundary tracks positions already claimed by
any continuation. An append can reuse the allocation only when its starting
position equals that boundary and the new tokens fit. It reserves the tail
before writing K and V. Existing views expose only their committed prefixes,
so these writes cannot change any token they can read.

An older branch whose tail is already occupied allocates its own buffer and
copies only its valid prefix. Initial capacity rounds the incoming sequence
up to a power of two with a 128-token floor, capped by model context; later
growth uses the same rule. This amortizes allocation and
prefix-copy costs over a linear continuation. Forking still copies a prefix
when necessary; this is not a paged-attention implementation.

Reservations are never rolled back, including when a write or a later layer
fails. A retry from the old state forks rather than overwriting that tail.
`Model::forward` still commits its state clone only after producing logits.
Convolution state retains its existing replacement semantics. Snapshot data
and position remain immutable even though unused allocation tails may change.

The allocator is private: its writable tensors are used only by attention,
and callers receive opaque model-owned states. Device identity is checked
before appending. Concurrent ROCm callers must share one device instance:
its backend queues every operation on the owning stream. The reservation
atomic orders host ownership, not GPU execution. CUDA defaults to a per-thread
stream; append optimization there and on Metal remains deferred until their
stream/lifetime contracts are validated. Their original concatenation path
remains explicit; GPU errors are never retried through another backend.

## Validation

- The concatenation baseline failed the allocation regression with 298 cache
  allocations for a three-token prefix followed by 297 one-token appends.
  The new allocator passes a bound of at most four allocations for that run.
- CPU tests cover multiple batches/heads, growth, old snapshots and tensor
  views, alternate branches, simultaneous branches, strided/offset sources,
  short context limits, invalid pairs, and abandoned speculative appends.
- A real partial-write failure is injected by aliasing the V source: K writes,
  then `slice_set` rejects V. The saved prefix remains intact and a retry
  allocates separate storage. A late model-layer failure also retries different
  tokens/length and compares against an independent cold forward.
- Disabling reservation conflict protection makes four tests fail, including
  branch overwrites and the partial-write retry. The original allocation
  regression also fails the old concatenate-every-token implementation.
- Independent NumPy tiny-GGUF comparisons cover whole, incremental, and every
  prefix split on CPU. That fixture uses F32 experts, which the production
  ROCm indexed-expert kernel does not support; it is not counted as a GPU pass.

The full daemon suite passes (152 tests, one existing ignored). Four explicit
ROCm tests pass, including cache growth/branch isolation/partial-write failure
and the existing QKV, merged expert, and quantized expert references.

## Paired full-checkpoint results

The [saved comparison](../benchmarks/lfm25/results/2026-09-15-kv-cache.json)
contains binary hashes, all twelve measurements, and both run summaries.
Both runs use the same Q5_K_M checkpoint and prompt, greedy decoding with
full-history repetition penalty 1.05, and a 2048-token output cap.

| Warm repeated case | Before decode | After decode | Time reduction |
|---|---:|---:|---:|
| reset --hard | 14.08 s | 13.00 s | 7.65% |
| git restore | 17.49 s | 16.29 s | 6.85% |
| rm -i | 15.96 s | 14.59 s | 8.58% |
| sed read | 9.02 s | 8.30 s | 8.00% |

Median repeated-call decode-time reduction is **7.82%**. All twelve outputs,
reports/errors, finish reasons, and completion lengths match exactly. The
four cases finish in 748–1383 generated tokens. Cold and cached generations
still differ as before; exact-input repeats remain identical. Prefill is not
the target of this change and shows mixed shared-host variation.

These are shared-host wall-clock measurements. CPU compilation/tests overlapped
parts of both runs; one cold case got slower. This is evidence of a useful
improvement, not an isolated GPU measurement or per-operation attribution.
Do not compound it with historical percentages from different output budgets.
Both benchmark processes passed deadline/reuse and in-flight shutdown checks.

Report validity (10/12) and fixture grades (7/12) are unchanged. They remain prompt/model
observations, not proof of adjudication quality. Raw responses and validation
logs are retained in `~/exomemory/lfm2d/lfm25-kv-2026-09-15/`.

Kaibo DeepSeek (`deepseek-flash`) reviewed the whole allocator, model, tests,
and tensor-copy implementation. Accepted feedback added direct storage-identity
checks for concurrent branches, a sibling retry after a failed append, proof
that the first copy occurred, GPU concurrent-branch coverage, and a mid-conv
failure witness. The stream concern led to explicit CPU/ROCm-only append
selection. These follow-ups do not change the measured ROCm copy arithmetic;
the final binary passed a separate 64-token runtime smoke, with all twelve
outputs matching the beginning of the full-length baseline outputs.

The suggested removal of the existing convolution `.copy()` was rejected:
singleton dimensions do not constrain Candle's contiguous-layout test, so a
single-channel narrow view can retain the larger allocation. CPU mixed-format
expert coverage remains a recorded follow-up; the existing explicit ROCm
mixed-format test passes. Full review and resolutions live in exomemory.

## Build during local development

This checkout temporarily patches Candle dependencies to `../candle-lfm25`.
Use the sibling `lfm25-moe-snapshots` worktree at `332ba982caa146f2efb22269725319342c29a445`
for this unpublished KV change.
The previous published daemon/Candle pair remains `e297a6c` / `936a15a6`.
Once this pass is published, replace the local patches with its Candle pin.

## Remaining structural work

Attention still materializes repeated KV heads for grouped-query attention.
Grouped expert projections still rebuild routing metadata and scratch.
These are the next table-stakes candidates before profiling. Prompt/schema
quality, temperature sweeps, and optional disk/phase checkpoints remain
separate work. Greedy decoding and repetition penalty 1.05 are unchanged.

Focused CPU/ROCm model clippy passes. The unchanged daemon cache lookup has
two `collapsible_if` findings under the current toolchain; daemon clippy passes
with that existing lint allowed. Record those conditionals for routine cleanup.
