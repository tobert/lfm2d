# Prefix caching and DiffusionGemma — measured 2026-09-10

Amy asked whether front-loading the prompt (invariant instructions first,
evidence last) would make most of an adjudicator prompt cacheable. The
structure is right; the engine cannot serve it for this model. This is
what was measured and why.

## The number that motivated it

Prefill is **measured by the engine, not estimated**: every request's
usage record carries `total_prompt_time_sec` and `prompt_tokens`.

    ~370 tokens/s, stable across every run

`first_content_s` is useless as a prefill proxy — it comes out
byte-identical to `response_s` on every row, because content arrives only
when a whole canvas commits.

At 370 tok/s a realistic rubric preamble is not a rounding error:

| prompt | prefill | response | share |
|---|---:|---:|---:|
| 397 tok (paraphrased rubric) | 1.06 s | 7.47 s | 14% |
| 2,810 tok (full rubric) | 7.64 s | 16.09 s | 47% |

## Result: the cache never hits on a prompt this size

Runs at q4k defaults, no WMMA env (so these absolute numbers are NOT
comparable to the Sep 6 q8_0+WMMA figures; all runs share the setting).

    cache off vs on, 397-token prompts repeated verbatim 3x:
      prefill 1.059 s vs 1.056 s, 374.4 vs 375.0 tok/s   -- no change

    cache on, 2,810-token tiered prompts:
      8 MISS, 3 DISCARDED, 0 HIT

## Why — two guards, both in `mistralrs-core/src/prefix_cacher.rs`

**1. An exact full-prefix match is thrown away.**

```rust
let new_toks = toks.0[match_len..].to_vec();
if new_toks.is_empty() { return Ok(None); }
```

Correct for a causal model: it needs at least one token to forward to
produce logits. Wrong for a block/canvas model, where the prompt is pure
conditioning and the canvas supplies the positions — so the best possible
hit is the one discarded. This is what a repeated identical prompt gets,
and it is why the 397-token cache-on run measured exactly nothing.

**2. A partial match cannot be rewound.**

```rust
if !v.can_rewind_to(match_len) { continue; }
```

`can_rewind_to` requires every layer to `try_set_len(len)`. Gemma-4 has
`sliding_window: 1024` with 25 of its 30 layers `sliding_attention`, so
those caches are rotating buffers holding only a fixed tail. Once a
sequence passes 1024 tokens the older entries are physically gone and the
cache cannot be rewound to an earlier logical length. Our shared tier-0
preamble is ~2,375 tokens inside ~2,800-token prompts, so every candidate
is skipped — with the good coverage visible in the log and the shared
prefix unusable:

    MISS: query_tokens=2798 candidates=10
          coverages=[1, 2766, 2835, 2809, 2805, 2862, 2817, 2810, 2766, 2835]

Together:

| prompt vs sliding window | partial match | exact match |
|---|---|---|
| <= 1024 tok | rewindable, could hit | discarded by guard 1 |
| > 1024 tok | rewind fails, skipped | discarded by guard 1 |

## What this implies for the prompt design

**Keep the shared preamble under the sliding window (1024 tokens).** Above
it, no partial hit is servable at all. This lines up with an independent
finding from the same runs: feeding the full 8.8 KB
`training/v10/rubric.md` as tier 0 dropped expected-field accuracy from
**23/24 (96%) to 18/24 (75%)**, added 2 denoising passes, and lengthened
outputs. The terse 977-char inline rubric beat the whole document. Both
the cache and the accuracy want a SHORT tier 0.

That is the cheap next experiment, and it needs no engine change: author a
tier 0 under 1024 tokens and re-run. Guard 1 still costs the
exact-repeat case, but an exact repeat is better served by an
adjudication result cache keyed on `(clause, expanded form)` — see
`~/exomemory/lfm2d/adjudicator-cascade.md`, and note that keying on the
clause alone is the recorded failure mode.

## Instrumentation

`prefix_cacher.rs` had NO hit/miss logging, which is why a 0% hit rate was
invisible: `search_for_matching_cache` has four silent `return Ok(None)`
paths. The vendored tree now logs HIT / MISS (with candidate coverages) /
DISCARDED at debug, and `warn!`s when an entry is stored with zero
coverage. Turn it up with:

    RUST_LOG='mistralrs_core::prefix_cacher=debug,info' python3 \
      benchmarks/diffusiongemma/benchmark.py … --prefix-cache-n 16

The harness no longer forces `RUST_LOG=info`; `driver.log` is not parsed
for measurements, so a louder log cannot move a number.

## Reading the runs

`compare_runs.py` prints per-case medians side by side and REFUSES to call
a response-time delta a speedup when pass counts differ (they differed 42%
and 71% across these runs). Pass count is a property of the answer, and
the device RNG is seeded once at model startup, so runs are not paired
random draws. Trust the prefill and token columns.

## Wrong turns, recorded so they are not repeated

- Hypothesised zero KV coverage in the stored entry. **Wrong** — coverage
  is full (436 of 436 on a short prompt).
- Hypothesised the hit was applied but wasted. **Wrong** — the hit was
  never accepted; guard 1 discards it.
- One `tier-on` run was invalidated by starting a cargo build during
  timing. The harness README says not to; it is right.
