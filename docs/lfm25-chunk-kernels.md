# A block's size chooses the kernel, and the kernel changes the answer

The 09-17 ribbon found 41 of 733 examined rows off the common depth-1 value,
all of them — and no other row — with a short final prefill chunk. The note
filed against it guessed the "known older dense cached-convolution bug". That
guess was wrong twice over, and the measurement that replaced it is the subject
here.

## What the ribbon actually found

`verdict_ribbon.py` computed `n_tokens % chunk` and printed it as `slot 1-8
tokens into a chunk`. `n_tokens` is a count, so that expression is the **length
of the final prefill chunk**, not the slot's offset inside it. The number was
right; calling a length an offset is what aimed the next step at the
convolution. (The expression is also wrong for a prompt of exactly one chunk,
where it returns 0 and files a full block with the shortest — fixed, with
`test_verdict_ribbon.py` holding it.)

Two things were then true and neither was checked:

- **The short convolution cannot reach eight.** `lfm2moe.shortconv.l_cache` is
  **3** in the GGUF. At depth 1 the last position sees three tokens. A broken
  history carry could move a final chunk of 1 or 2 and nothing longer.
- **The adjudicator does not run the module that note pointed at.**
  `docs/lfm25-adjudicator.md` records that the old dense `lfm2` /
  `quantized_lfm2` cached multi-token convolution paths ignore existing
  history, and in the same breath that *this* model fixes its own path. The
  adjudicator is `quantized_lfm2_moe`, whose `causal_conv` indexes its taps and
  its state tail correctly and has carried a chunk-splitting unit test since it
  was written.

## The measurement

`lfm25-chunk-sweep` (`lfm2d::chunk_sweep`) feeds `tokens[..n-m]` as one block
and the last `m` tokens as a second, then reads every depth at the final
position through the model's own projection. `m = n` is the unsplit reference.

Varying `m` varies the prefix length too — two differences, not one, which is
the confound that cost the 09-19 session a wrong mechanism. So the sweep
carries its own control: tails far larger than any convolution window or
small-batch threshold move the prefix by the same amounts. Those come back
bit-identical while the short ones do not, so the prefix length is not what
moved them.

Neither the daemon nor the examiner ever prefills in one block. The sweep
measures **invariance of the forward pass**, not either schedule.

## Result: a cliff at eight, on every prompt

Three prompts, ROCm, `LFM2.5-8B-A1B-Q5_K_M`, `command-verdict-enum-v1`, read at
the verdict slot. `first` is the first depth whose reading differs from the
unsplit reference at all; depth 1 is the output of layer 0, depth 3 of layer 2
(the first attention layer).

| tail | first depth that differs | max abs delta at the last depth |
|---|---|---|
| 1–8 | **1** — layer 0 | 0.36 – 2.19 nats |
| 9–56 | **3** — the first attention layer | 0.29 – 3.68 nats |
| ≥64 | varies: some bit-identical, others at depth 7, 11 or 22 | up to 1.8 nats |

The 1–8 boundary is sharp: a tail of 8 differs at depth 1 by ~0.7 nats, a tail
of 9 is bit-identical there and at depth 2. It reproduced on all three prompts
and the whole sweep is bit-identical run to run, so this is a function of the
shapes, not noise.

## Mechanism: MMVQ, not the convolution

`candle-core/src/quantized/rocm.rs:275-280`:

```rust
let b_size = b * m;
if b_size <= mmvq::MAX_BATCH {
    if let Some(dst) = mmvq::try_fwd(self, n, k, b_size, storage, layout)? { ... }
}
```

`mmvq::MAX_BATCH` is **8** (`quantized/rocm/mmvq.rs:28`). At eight rows or
fewer every quantized matmul in the model takes MMVQ, the decode fast path,
which **requantizes the f32 activations to `q8_1`** and dots them against the
packed weights in integers. Above eight rows the same matmul takes MMQ or the
dequantized dense path instead. Q5K and Q6K both have MMVQ kernels, so every
weight in this checkpoint is affected.

So it is not the convolution and not a history bug: a prefill block of 1 to 8
tokens computes *every linear layer* — attention projections, the short
convolution's own in/out projections, the FFN — through a different numerical
procedure than a block of nine does.

The second family (tails ≥ 9 first differing at an attention layer) is a
separate, unlocalized effect. Conv layers mix three tokens, so a difference
born at another position can only reach the last one through attention, which
is why every one of those first appears at a depth of 2, 6, 10, 18 or 21.
`dense::preferred` (`rocm.rs:311`) switches MMQ against the dense path on
`b_size` and the weight's shape, which is the obvious suspect and is not yet
pinned. **Split-invariance is not a property this stack has at any block size**
— some large tails are bit-identical and others are not.

## What it means for the daemon

Both facts are production, not probe artifacts:

- **Every generated token is MMVQ.** `adjudicator.rs:739` decodes with
  `forward(&[token], ...)`, so `b_size == 1`.
- **Roughly one input in sixteen prefills a short final block.**
  `adjudicator.rs:389` prefills `full[start..].chunks(CHUNK)` with `CHUNK` 128,
  so the last block holds `(len - start) % 128` tokens. When that falls in
  1..=8 — 8 of 128 lengths, about 6% — the last block of the prompt, the one
  whose final row produces the first generated token's logits, is computed by
  MMVQ while a prompt one token longer is not.

No verdict flipped on any of the three prompts: the reference's top-to-runner-up
margin at the verdict slot was 1.74, 2.13 and 2.78 nats against worst deltas of
0.75, 2.19 and 1.28. One of those three had a delta larger than its margin and
still did not flip. That is reassurance about confident slots only —
`label-margin-does-not-bound-winner` measured live winner margins down to 5e-9,
with 520 live rows under 1e-4, and a perturbation of this size moves those
freely.

## What it means for the examiner

`lfm25-examine` prefills in 128-token blocks and reads a lens. The daemon wrote
the tokens it is reading **one at a time**. So for any slot inside the generated
region, the examiner reads through MMQ what the daemon wrote through MMVQ, on
every row.

That is a concrete candidate for the residual examiner-against-daemon gap that
F5 left open at p50 0.16 nats per word. The 09-19 session's `--chunk` profile
was flat across buckets and the flatness was read as ruling a chunk effect out;
a mechanism that applies to *every* generated token equally predicts exactly a
flat profile. This has not been measured — the test is to have the examiner
replay the generated region token by token and see whether the gap closes.

## Not settled

- The second family: which kernel selection moves at those tails, and why some
  large tails are bit-identical.
- **Whether CPU shows the same cliff. One small reading, no sweep.** The branch
  above lives in `quantized/rocm.rs` and `fast_mmvq.rs` is CUDA, so the code
  says ROCm-only. The memory-debugging run left one CPU sweep behind (found
  afterwards): a 16-token prompt, tails 1 and 2. Depths 1 and 2 came back
  bit-exact — no layer-0 difference — and both tails first differ at depth 3,
  the first attention layer, by ~5e-6, which a changed expert choice grows to
  0.23 and 1.05 nats at the last depth. So the cliff at 8 did not appear on CPU,
  and the second family did: it is not a ROCm artifact. One prompt and two
  tails is a reading, not a rate. Note the
  fixture cannot stand in: `lfm2d/tests/fixtures/lfm2-moe/tiny.gguf` is all
  F32, 33 tensors, no quantized weight at all, so every CPU test in this repo
  runs a dispatch path this finding does not concern. A CPU sweep of the real
  checkpoint was started and abandoned at 73 minutes: it held ~49 GB and ran on
  one core of 32 throughout, with `RAYON_NUM_THREADS=8` set. A cheaper control
  is an ~80-token prompt, since the threshold is a block size and not a prompt
  length — but see the section below first.
- Whether reading the generated region by decoding closes the 0.16 nats.

## Why a CPU sweep of this checkpoint is so expensive

Measured, because the abandoned run raised it. Both causes are in
`quantized_lfm2_moe.rs`, both on the CPU arm only, and the file says why:
"CPU is an explicit reference implementation, not a GPU fallback."

**28.9 GB of weights, at load.** `Experts::new` (`:158`) dequantizes every
expert to f32 when the device is CPU, keeping the GGUF quantized only on GPU.
22 MoE layers x 32 experts x 3 matrices of 2048x1792 is 7.75 B parameters,
28.9 GB in f32. RSS sampled every 0.5 s climbs 0.44 -> ~29.9 GB between t=3.5 s
and t=14 s, before the first reading starts.

**Then a gather that scales with the token count.** `Experts::Cpu::forward`
(`:180`) does `w.index_select(&ids.flatten_all()?, 0)`, materialising the
selected expert weights once per token slot. With gate/up merged the widest of
those is `tokens x topk x hidden x 2*ffn x 4` bytes:

| tokens | predicted transient | observed peak over baseline |
|---|---|---|
| 17 | 1.86 GB | 1.75 GB |
| 144 | 15.75 GB | 16.10 GB |
| 375 | 41 GB | not reached; the run was killed first |

Time is linear in tokens too: 19 s per reading at 17 tokens, 159 s at 144 —
8.4x for 8.5x. So the 375-token sweep that was killed at 73 minutes was about
7 minutes per reading, **2.2 hours** for all 19, and would have peaked near
70 GB.

Neither is a bug against what that code says it is. They do mean a CPU
comparison wants a short prompt and few tails, and that a reference
implementation which gathers per token slot is the thing to change first if
CPU ever needs to be more than a reference.
