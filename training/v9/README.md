# kube_ordinal v9 — corpus status

**This is data and tooling. There is no v9 checkpoint.** No training run has
happened; the deployed classifier is still `kube_ordinal_v8`. Read `PLAN.md`
for the design and the rulings.

```
python3 training/v9/build_v9.py            # merge every slice -> v9.jsonl, gate it
python3 training/v9/build_v9.py --check    # verify it regenerates byte-identical
python3 -m unittest discover training/tests
```

## Where it stands — 547 rows

| slice | what | rows | status |
|---|---|---|---|
| **1** | worktree / branch cleanup → situation-normal | 71 | generated, relabeled |
| **2** | history rewrite → above situation-normal | 76 | generated, relabeled |
| **3** | data position, carrier with bare-command payload | 75 | generated, relabeled |
| 4 | a6 `curl …/reset.sh` | — | not started |
| 5 | `npm publish` / R7 | — | not started, flagged as the one to cut |
| **6** | system administration surface (+6b extensions) | 334 | generated, relabeled |

Merged: **547 rows** after deduping 9 cross-file repeats —
**279 data-critical (51.0%) · 110 informative (20.1%) · 158 situation-normal
(28.9%)**, 18 contested (3.3%), **17 author tags across 10 model families**.

## Gates

- `validate_v8.py` — per-file: 7-key schema, label vocabulary, 1–40 word cap,
  in-file duplicates, canary contamination. Reports aggregates only, never raw
  rows.
- `build_v9.py` — cross-slice: duplicates *between* slices, per-file label skew
  (fails >75% one label), compound share (fails <15%), and **refuses to merge
  on a label conflict** rather than picking a winner.
- `training/tests/` — 96 tests, including the dedup regression below.

## Cross-family agreement is free evidence

9 texts were generated independently by more than one family, **0 disagreed**.
Three of those are cross-*slice* (slice 2's history rows and slice 6's flag
ladder both reached for `git push --force origin main`) — invisible to any
slice-local tool, which is why `build_v9.py` replaced them.

## Rulings encoded since v8

- **Rule 13** — a confirmation prompt is not an interlock. `-i`/`--interactive`
  do not lower severity; `-y`/`--noconfirm` do not raise it. Amy: *"nothing to
  an agent or a `yes | cmd`."*
- **Rule 14** — logs: bounded retention (`--vacuum-time=30d`) is
  situation-normal; wholesale destruction (`truncate -s 0`) is data-critical.
  Rule 10 does not reach logs.
- **Rule 15** — fetch-and-execute (`curl … | bash`, `eval "$(curl …)"`)
  defaults to data-critical: the text cannot show what runs. The other arm of
  rule 8 — carried text becomes a program once piped into an interpreter.
- **Rule 8 now biases UPWARD** — it removes the *payload* from consideration,
  not the carrier (`git commit` still creates a commit), and ambiguity resolves
  to `situation-normal` or above, never down to `informative`. Amy: *"bias
  towards labeling data-critical or situation-normal when they're detected in
  the data and/or it's ambiguous."*

All four came out of blind relabel passes finding the same disagreement
repeatedly. One question Amy left deliberately open — bare `git restore .`,
where she is comfortable with the inconsistency — is recorded in `PLAN.md`.

## What we learned about the generators

**Local `lfm25-8b-a1b`: unusable.** Labeled every `rm`-ladder arm
data-critical including `rm -i` — reproducing the exact bug this corpus exists
to fix. Not routed to since.

**Crusoe families, first use:**

| model | verdict |
|---|---|
| `Qwen3-235B` | strongest of the new families; got rule 12 right unprompted |
| `Nemotron-3-Super-120B` | reliable on large single-shot asks, narrower variety |
| `Kimi-K2.6` | richest coverage, but **reasoning ate a 16k output budget** at 35 rows — needs ≤20 rows + explicit no-CoT |
| `gpt-oss-120b` | clean labels, repetitive (trailing `&& echo done`) |
| `gemma-4-31b` | correct labels, templated notes (6 rows shared one verbatim string) |
| `Llama-3.3-70B` | **weakest** — half its rows were combinatorial padding, ~2/3 discarded |

**gemini-3.5-flash is not self-consistent run-to-run** — an identical prompt
flipped one row's label between two samples. A single relabel sample is not a
stable measurement on ambiguous rows.

## Known-open

- **Labels are PROPOSALS.** Bulk blind labeling still gated on budget. Every
  row carries its `author` tag so a later pass can disagree row by row.
- **18 contested rows (3.3%)** hold their original label with the dissent in
  `note`, per Amy: *"hesitate where gen/relabel disagree."*
- **All 7 files have had a blind cross-family relabel pass** (86.7%–97.2%);
  see `relabel/README.md`. Five open questions came out of it, listed there.
- **Slices 4–5 not started.**
- The severity-probe constraint `root_delete_over_source_file` needs rebuilding
  against a no-interlock target — bare `rm -rf /` has `--preserve-root`.

## A bug this corpus found in its own tooling

The validators normalized `text` with `.lower()` before the duplicate check, so
`git branch -d` collided with `git branch -D` — **opposite ground-truth
labels** — and the second was rejected as a duplicate. Structurally biased
against exactly the interlock-vs-force pairs rules 11–13 turn on. Fixed in all
four validators, regression test in `training/tests/test_validator_dedup.py`.
No prior data was lost; slice 1 was the first to hit it.
