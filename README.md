# lfm2d

LiquidAI's **LFM2.5 bidirectional encoders** on [candle] — pure Rust,
CPU-first, no Python, no C++.

Two crates in one workspace: **`lfm2-encoder`** (this README — the library,
at the repo root) and **`lfm2d`** (the HTTP daemon that serves its heads over
a Unix socket and/or TCP — see [`lfm2d/README.md`](lfm2d/README.md)).

Upstream candle-transformers implements the *causal* LFM2
(`models/lfm2.rs`). Nobody implements the encoder branch —
`Lfm2BidirectionalModel` and its task heads. This crate is that gap:

| checkpoint | head | why we want it |
|---|---|---|
| `LFM2.5-Embedding-350M` | pooled embedding (1024-dim) | text embeddings in-process |
| `LFM2.5-Encoder-350M-PII-Detector` | token classification (BIOES, 161 labels **including credentials/secrets**) | boundary screening of foreign prose & outbound payloads |
| `LFM2.5-Encoder-350M-Policy-Linter` | token classification | same lane, policy flavor |
| `LFM2.5-Encoder-350M-Prompt-Router` | sequence **routing** (`rule_proj_dim` — scores prompts against rule projections, zero-shot-shaped) | dynamic model-lane routing |
| `LFM2.5-Encoder-230M/350M` | masked-LM bases | fine-tune substrate |

## Status

**Every head in the table above runs, and each has a parity test against
the reference implementation**: the trunk, pooled embedding, ColBERT late
interaction, token classification (PII and credential spans), sequence
routing, and sequence classification. The parity suites under `tests/` are
what keep those claims honest — each compares against activations dumped
from real weights by the matching script in `tests/reference/`.

Sequence classification is also the substrate for the fine-tuned severity
heads this repo trains (`training/README.md`), and the heads are served
over HTTP by the `lfm2d` daemon in this workspace.

`Lfm2Trunk` reproduces LiquidAI's own
`modeling_lfm2_bidirectional.py` to max|Δ| ≈ 4.6e-5 on f32 CPU, verified
against activations dumped from the real Embedding-350M weights
(`tests/reference/dump_trunk_reference.py`, transformers 4.56.2).

Milestone 1 (config) covers every family checkpoint. The fixtures keep
earning their keep — the 230M base is *shallower* (14 layers) not just
narrower; the PII config ships a literal `"full_attn_idxs": null`; the
Router is not a softmax classifier; `intermediate_size` **disagrees with
the shipped weights** on three of four checkpoints (says 6656, ships
4608 — see `Lfm2EncoderConfig::ffn_dim`); and the Policy-Linter turned
out to be a fifth architecture name, `Lfm2BidirForRuleMatching`.

**Milestone 3 started: text → vector works end to end.** `Lfm2Embedding`
tokenizes, runs the trunk and CLS-pools, matching the reference pipeline
including exact token ids. Try it:

```
cargo run --release --example embed -- .models/LFM2.5-Embedding-350M
```

Semantic search works end to end. On a 105-document hand-authored corpus:
**recall@1 88.6%, recall@3 100%**, and a hard negative outranks the true
positive **1 time in 68**. That last number is the one that means
something — the hard negatives share vocabulary with the query and are
still wrong. `tests/retrieval_quality.rs` keeps those numbers honest on
every `cargo test`.

```
# search the bundled corpus, or point --dir at your own tree
cargo run --release --example search -- .models/LFM2.5-Embedding-350M \
    --query "how do I stop two threads corrupting shared state"
cargo run --release --example search -- .models/LFM2.5-Embedding-350M --eval
cargo run --release --example search -- .models/LFM2.5-Embedding-350M \
    --dir src --query "where is the attention mask built"
```

### What it costs to run

Measured on 32 cores, per 350M checkpoint (`examples/bench_memory.rs`,
`examples/bench_parallel.rs`, `examples/dtype_drift.rs`):

| dtype | 1 model | 2 models | solo | concurrent | throughput |
|---|---|---|---|---|---|
| f32 | 1409 MiB | 2.70 GiB | ~70 ms | ~87 ms (1.25×) | 23 embeds/s |
| f16 | 745 MiB | 1.41 GiB | ~139 ms | ~150 ms (1.08×) | 13 embeds/s |

**f16 halves memory for free**: cosine 0.999996 against f32, with
identical rankings — and it contends *less* than f32, so it scales better
as you add models. `bf16`, despite being the checkpoints' own storage
dtype, is unsupported for `matmul` on candle CPU.

### ColBERT: late interaction, and it wins

`ColbertModel` implements `LFM2.5-ColBERT-350M` — one 128-dim vector per
token, scored with MaxSim, verified against PyLate itself. Head to head on
the same corpus (`cargo run --release --example compare_retrievers`):

| | recall@1 | recall@3 | hard-neg wins | index | query |
|---|---|---|---|---|---|
| single-vector (CLS + cosine) | 88.6% | 100% | 1.5% | 4.1 KB/doc | 67 ms |
| **ColBERT (MaxSim)** | **97.1%** | 100% | **0.0%** | 15.5 KB/doc | 90 ms |

ColBERT never lets a hard negative outrank a true positive, for 3.8× the
storage and 1.4× the query time. On short documents that trade is far
cheaper than ColBERT's usual reputation suggests.

Its conventions came from PyLate's real behaviour, not the config, because
several are surprising: query expansion pads with **EOS** (this checkpoint
has no mask token at all), `[Q]`/`[D]` are **real vocab ids** 64400/64401
— which is why its vocab is 64402 rather than the family's 65536 —
expansion tokens are masked as attention *keys* yet still emitted as
vectors, and documents drop punctuation vectors via a 32-id skiplist.

## This embedding model is asymmetric

Queries and documents take **different prefixes** — `"query: "` and
`"document: "`. This is not cosmetic: the same sentence embedded both
ways lands at cosine ≈ 0.70. Using one prefix for both sides degrades
retrieval and nothing errors, so `TextKind` is a required argument with
no default:

```rust
let model = Lfm2Embedding::from_dir("...")?;
let q = model.embed_normalized("how do I borrow a value?", TextKind::Query)?;
let d = model.embed_normalized(passage, TextKind::Document)?;
```

`embed` returns the raw vector — the checkpoint ships no Normalize
module — and `embed_normalized` L2-normalizes so cosine is a dot product.

## Batching changes your embeddings (read this)

The short conv is deliberately **not** masked, because that is how these
checkpoints were trained: in the eager/sdpa path the reference's
`apply_mask_to_padding_states` is a no-op, so pad states flow through the
conv. With a centered `k = 3` kernel, each conv layer bleeds a pad one
position into its real neighbour.

**Consequence:** a sequence's embedding depends slightly on what it was
batched with. Unlike BERT, padding is not inert here. If you need
reproducible vectors — cache keys, stored embeddings, anything compared
across processes — embed one sequence at a time with no padding. Batch
when throughput matters more than bit-identical results.

`forward()` refuses a multi-row batch with no `attention_mask` rather
than silently treating pad tokens as content.

## Design intents

- **CPU-first.** Consumers embed this in long-lived server processes.
  Measured on a 32-core Strix Halo: **~81 ms** per short text, f32,
  including tokenization. GPU (CUDA/Metal via candle features) is a
  bonus, never a requirement.

  Getting there needed one non-obvious change: LFM2's short conv is
  *depthwise* (`groups = hidden_size = 1024`), and candle's grouped
  `Conv1d` costs a flat **~173 ms per call regardless of sequence
  length** — per-group dispatch overhead, not arithmetic. Across 10 conv
  layers that was the entire ~1.9 s of a single embedding. The trunk
  evaluates the conv as `k` shifted per-channel multiply-adds instead:
  identical arithmetic, ~23× faster end to end, parity unchanged. See
  `cargo run --release --example bench_ops`.
- **Pure Rust.** No C++ toolchain in the dependency tree.
- **Local weights by default.** Point at a directory holding
  `model.safetensors` + `tokenizer.json` + `config.json`; the optional
  `hub` feature adds Hub download.

## License & attribution

MIT OR Apache-2.0, matching candle. Trunk block implementations are
adapted from [candle-transformers]' `lfm2.rs` (© the candle authors, MIT
OR Apache-2.0); attribution retained in source where adapted.

[candle]: https://github.com/huggingface/candle
[candle-transformers]: https://github.com/huggingface/candle/tree/main/candle-transformers
