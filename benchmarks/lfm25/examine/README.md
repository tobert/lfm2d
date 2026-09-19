# Examiner tooling

What reads a `verdict_eval.py` run back into the model: the logit lens at an
answer slot, expert routing around it, and the pages that show either one.
`lfm25-examine` and `lfm25-expert-map` (in `lfm2d/src/bin/`) do the inference;
everything here prepares their inputs or scores their output.

Outputs go **outside this repo** — they hold per-row verdicts, which are corpus
rows. These scripts print aggregates and row numbers, never row text.

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
   *every* bucket of the slot's offset into a prefill chunk. Flat, so this is not
   the layer-0 cached-convolution problem.
4. **Retokenizing the generated text.** Re-encoding each `output` gives the
   daemon's own generated token count on 149 of 150 rows.

And the prompt side lines up exactly: on a 40-row slice the examiner's total token
count equals the daemon's `prompt_tokens` plus the prefill's tokens on **40 of 40
rows**, while the delta replicated at p50 0.40 nats. Same bytes, same token counts,
different numbers.

What is left is the arithmetic: the daemon decoded the report token by token onto a
cached prefix, and the examiner prefilled the same tokens in a block. Next probe is
the daemon's own `use_cache: false` against `true` on the same rows, which asks
whether a cache hit alone can move a verdict in production.

**This bounds every lens number in this directory.** A per-depth AUC or a lean
curve is read from the examiner, so a 0.16-nat floor sits under all of it. It does
not touch the daemon's own verdicts, which are what the daemon actually emitted.

## Two standing caveats

**Lens readings near a chunk head are suspect.** 41 of 733 rows left the common
line at layer 0 by up to 0.32 nats, and every one had its slot 1–8 tokens into a
128-token prefill chunk — the two dense-FFN conv layers, the known cached
convolution problem. `verdict_ribbon.py --chunk 128` reports the count on every
run, so a fix shows up as it reaching zero.

**A first token is not a word.** The lens follows the first token of each word
encoded alone, so `ask` is exact and `easy` is whatever begins with `e`. The
record carries `first_tokens` so a page can say which readings are exact.
