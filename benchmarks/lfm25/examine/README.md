# Examiner tooling

What reads a `verdict_eval.py` run back into the model: the logit lens at an
answer slot, expert routing around it, and the pages that show either one.
`lfm25-examine` and `lfm25-expert-map` (in `lfm2d/src/bin/`) do the inference;
everything here prepares their inputs or scores their output.

Outputs go **outside this repo** — they hold per-row verdicts, which are corpus
rows.

One exception to the aggregates-only rule, and it is load-bearing:
`verdict_inputs.py` writes the rendered user turn, command text and all, because
that is what the examiner has to be fed. **Redirect it to a directory outside this
repo**, as the example below does. Every other script here prints counts and row
numbers only, and row names are numbers so corpus text never becomes a file name.

| script | what it does |
|---|---|
| `verdict_inputs.py` | a run → `lfm25-examine --inputs-file` lines, one per row, standing at a named field's opening quote |
| `replay_vs_rebuild.py` | how far a rebuilt input is from a replayed one, on the same rows: what the rebuild was costing |
| `examiner_vs_daemon.py` | the distance in nats between the examiner's reading of a slot and the daemon's own, and what the flips happened at |
| `verdict_ribbon.py` | the per-depth lean at the slot, grouped by gold and by what the daemon wrote; `--chunk N` also reports the chunk-boundary effect |
| `expert_contrast.py` | per-cell expert usage difference between two row groups, with a family-wise permutation test |
| `knockout_report.py` | what a knocked-out expert did to the margin |
| `build_viz.py`, `build_ribbon_viz.py`, `build_contrast_viz.py` | the three pages, from the records above and the templates beside them |

## A replay is not a re-rendering

`verdict_inputs.py` defaults to **replay**: both halves of an examiner input are
the bytes the run recorded, `input` as sent and a slice of `output` as generated.
Nothing is rebuilt, so nothing can fork.

It forked twice before, in the same family as
`harness-improvements-fork-production`:

* **The prefill, rebuilt from `report`.** rows.jsonl stores the parsed report with
  its keys sorted; the daemon emits the schema's `required` order. The first
  version of this tool replayed the stored order, which put `reason` — written
  *after* the verdict, to justify it — in front of the verdict slot on all 733
  rows of a batch. Ordering from the spec fixed that, but a parse has also lost
  the model's own escaping, and that is not fixable by rendering harder: a model
  that writes `é` and a `json.dumps` that writes the character produce the
  same report and different bytes.
* **The user turn, rebuilt from `text`.** `build_facts` reads the host's manual
  pages. Re-rendering later, or on another machine, renders whatever `man` says
  then — with nothing in the output to show it moved.

`--reconstruct` still rebuilds, for runs recorded before `verdict_eval.py` kept
the raw bytes. On a run that has them, a row missing either half is an error
rather than a quiet rebuild.

## Reading a run

```
export EX=~/somewhere/outside/this/repo/examined
python3 verdict_inputs.py "$RUN" --prompt ../../../lfm2d/prompts/command-verdict-enum-v1.json \
        --slot verdict > "$EX/inputs.jsonl"
python3 replay_vs_rebuild.py "$RUN" --prompt ../../../lfm2d/prompts/command-verdict-enum-v1.json \
        --slot verdict --rows "$EX/differing-rows.txt"
```

Then `lfm25-examine --inputs-file "$EX/inputs.jsonl" --prompt <same spec>`, and
`verdict_ribbon.py --rows "$RUN/rows.jsonl" --examined "$EX/examinations.jsonl"`.

**Pass `--device rocm`.** `--device auto` has picked CPU for `lfm25-examine` and
run for six minutes with the GPU idle. The binary needs `--features rocm` to
offer it at all, and says so if it was built without.

## The examiner does not read what the daemon read

Measured 2026-09-19 on the 733-row val_F batch at the verdict slot, ROCm, with
`examiner_vs_daemon.py`: the examiner's final-depth lens and the daemon's own raw
distribution at the same slot differ by a **median 0.16 nats per word**, p95 1.25,
max 4.23. Both are a `log_softmax` over the full vocabulary and `examine.rs` pins
the final depth to a real forward, so these two numbers should be the same number.

The 11 rows (1.5%) where the two disagree on the top word are not a separate
problem: every one of them sits at a daemon margin below 0.57 nats, under the
disagreement itself. Fixing the prefill bytes — which is what replay did — cannot
close them, and on this corpus replay and rebuild produce identical bytes anyway.

Four mechanisms are ruled out, in the order they were cheapest to test:

1. **The prefill bytes.** `replay_vs_rebuild.py` reports replay and rebuild
   byte-identical on all 733 rows. The grammar forbids `\uXXXX`, so the only
   divergent short escape is `\/` and the model never wrote one.
2. **The grammar mask and the repetition penalty.** The daemon's raw argmax within
   the verdict words is what it wrote on 249 of 249 rows, so neither moved a
   verdict. (`examiner_vs_daemon.py` refuses to compare a row where they did.)
3. **A chunk-boundary effect.** `--chunk 128` puts the delta at p50 0.34–0.48 in
   *every* bucket of the slot's offset into a prefill chunk. Flat — read at the
   time as ruling a chunk effect out. It does not: a mechanism that applies to
   every generated token equally predicts a flat profile, and
   `docs/lfm25-chunk-kernels.md` found one. The daemon decodes each generated
   token at `b_size == 1`, which takes candle's MMVQ kernel; this examiner
   prefills them in 128-token blocks, which does not. Untested, and the test is
   to replay the generated region token by token.
4. **Retokenizing the generated text.** Re-encoding each `output` gives the
   daemon's own generated token count on 149 of 150 rows.

And the prompt side lines up exactly: on a 40-row slice the examiner's total token
count equals the daemon's `prompt_tokens` plus the prefill's tokens on **40 of 40
rows**, while the delta replicated at p50 0.40 nats. Same bytes, same token counts,
different numbers.

### It is the cache schedule, and the daemon does it to itself

The first version of this note blamed "a block prefill versus a token-by-token
decode". That was wrong, and a kaibo review caught it: the examiner is cold from
zero while a recorded run is warm, so cache schedule and prefill-versus-decode were
confounded, and the repo already documented the first one
(`docs/lfm25-grouped-prefill.md`: "Cold and cached chunk schedules now produce
different long greedy generations").

Measured 2026-09-19 with `verdict_eval.py --no-cache`, 40 rows, the only difference
`use_cache`:

- identical generated text on **13 of 40** rows,
- identical verdict on **39 of 40** — a cache hit alone moved one verdict,
- per-word |delta| at the verdict slot p50 **0.145** nats, p95 1.71, max 3.15.

That is the same size as the examiner-vs-daemon gap, from the daemon against
itself. So the examiner is not misreading: it is reading a cold schedule while the
run answered on a warm one. The resident prefix is 327 tokens and `327 % 128 = 71`,
so the two schedules' chunk boundaries are 71 tokens apart and even the prefix
region is built with a different last-chunk shape.

**What it does not mean.** Production is always warm — a new input prefills from
the resident prefix, an exact repeat replays the stored state and logits, and
`use_cache: false` is only ever set by a harness. So production verdicts are not
nondeterministic on this account. What it means is narrower and still serious: **a
cold reader does not reproduce what the daemon answered**, and this examiner is a
cold reader. The 0.16 nats is the price of reading the model on a schedule the
daemon never runs, not evidence that the daemon wobbles.

`--chunk` is bucketed by the daemon's offset for this reason; with one shared
prefix its residues are a relabelling of the examiner's, so the flat profile above
transfers.

**This bounds every lens number in this directory.** A per-depth AUC or a lean
curve is read from the examiner, so a 0.16-nat floor sits under all of it. It does
not touch the daemon's own verdicts, which are what the daemon actually emitted.

## Two standing caveats

**A reading whose final prefill chunk is 1–8 tokens is computed by another
kernel.** 41 of 733 rows left the common line at layer 0 by up to 0.32 nats, and
every one of them had a final chunk that short. Not the convolution, whose
window is three: at eight rows or fewer every quantized matmul takes candle's
MMVQ decode path, which requantizes activations to `q8_1`
(`docs/lfm25-chunk-kernels.md`). `verdict_ribbon.py --chunk 128` groups the
off-line rows by final chunk length on every run, and says outright when one of
them is too long for that explanation.

**A first token is not a word.** The lens follows the first token of each word
encoded alone, so `ask` is exact and `easy` is whatever begins with `e`. The
record carries `first_tokens` so a page can say which readings are exact.
