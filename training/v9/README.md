# kube_ordinal v9 — corpus status

**Three local training passes exist (`.models/kube_ordinal_v9`, gitignored;
metrics committed at `severity_probes/baseline_v9{,_round2,_round3}.json`).
None is a rollout candidate** — all three show a form of instability on the
out-of-corpus probe gate (pass 1: raw score saturation; pass 2: baseline-
probe compression toward the ceiling; pass 3: the calibration-vs-margin
picture holds — classic pass/fail 8→15→17→18/23, but the honest
`--delta-margin 0.05` reading is 5→8→8→9/23, and this pass's benign-control
inversions TRIPLED (1→3) including the exact live false-positive shapes
`signoff.md` already flagged in production). See
`severity_probes/DELTA_METRIC.md` for the metric and the full table. The
deployed classifier is still `kube_ordinal_v8`. Read `PLAN.md` for the
design and the rulings.

```
python3 training/v9/build_v9.py            # merge every slice -> v9.jsonl, gate it
python3 training/v9/build_v9.py --check    # verify it regenerates byte-identical
python3 -m unittest discover training/tests
```

## Where it stands — 767 rows

| slice | what | rows | status |
|---|---|---|---|
| **1** | worktree / branch cleanup → situation-normal | 71 | generated, relabeled |
| **2** | history rewrite → above situation-normal | 76 | generated, relabeled |
| **3** | data position, carrier with bare-command payload | 75 | generated, relabeled |
| **4** | a6 `curl …/reset.sh`, fetch-execute | 67 | generated, relabeled; 10 rows contested — see "rule 15 boundary" below |
| **5** | `npm publish` / R7, no-undo-anywhere | 78 | generated, relabeled |
| **6** | system administration surface (+6b extensions) | 328 | generated, relabeled |
| **7** | sysadmin verb ladders, round 2 (`mkfs`/`dd`/`shred`/`chmod`/`truncate`/boot) | 48 | generated (2026-08-15), not yet relabeled |
| **8** | package-manager / fetch-execute breadth | 51 | generated (2026-08-15), not yet relabeled |

Merged: **767 rows** after deduping 13 cross-file repeats, 18 quarantined as
severity probes — **370 data-critical (48.2%) · 145 informative (18.9%) ·
252 situation-normal (32.9%)**, 45 contested (5.9%), **29 author tags across
13+ model families** (added this round: deepseek-v4-pro, Nemotron-3-Super-
120B, qwen3.8-27b, qwen3.8-max, google/gemma-4-31b-it).

## `pkg_install` — a new, independent axis (2026-08-15)

Every row now also carries `pkg_install: bool`, orthogonal to `label`: true
if the statement's primary effect is fetching/installing a software
dependency, regardless of severity. Not used by training or the probe gate
yet — banked for a future v10 multi-task head (the same one-trunk
architecture PLAN.md already records for splitting blast-radius/
recoverability). See `backfill_pkg_install.py` for how the existing 701 rows
were tagged (deterministic regex pass, hand-reviewed).

## Rule 16 — RULED, overturns the 5/5 blind council (2026-08-15)

Does an install/upgrade with code-execution capability (lifecycle scripts:
npm postinstall, pip setup.py, cargo build.rs, gem extconf.rb, deb postinst,
NuGet install.ps1, ...) count the same as direct fetch-and-execute
(`curl | bash`)? Five blind families were unanimous **against** — the
original gemini-3.5-flash relabel plus four more this session
(deepseek-v4-pro, Nemotron-3-Super-120B, qwen3.8-27b, qwen3.8-max). Amy
ruled for the generator's original proposal anyway: *"the attitude that
package managers are harmless and are not at least situation normal is
dangerous and causes a lot of harm... I disagree with the blind families,
unless we know a package is just files and does not do code exec."* Full
quote and reasoning in `PLAN.md`. Applied via `relabel/apply_rule16.py` —
27 label flips across slice4 (10, reinstated), slice6 (3), slice8 (14),
plus 10 contested resolutions and 4 rows left/marked contested where the
ruling doesn't clearly resolve them (`npm ci` — lockfile-pinning is a
question Amy didn't address; `mix deps.get`/`flutter pub get` — build-hook
semantics not verified). **This is the one case in the corpus where blind
council consensus was overturned rather than followed** — recorded because
it's the sharpest illustration yet that relabel agreement informs a ruling,
it doesn't replace one.

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

**New this round (2026-08-15) — Alibaba Qwen3.8 via OpenRouter, an explicit
exception to the usual bulk-work default:**

| model | verdict |
|---|---|
| `qwen3.8-27b` (small) | strongest of this round's new families; 27/28 rows clean on first ask, only generator to correctly reason about direction-of-flow for `dd` (read vs write) unprompted |
| `qwen3.8-max` (big) | good quality but **truncates on a 28-row ask** (16k output cap) — needs ≤16 rows/call with short notes, same shape as Kimi-K2.6's limit |
| `deepseek-v4-pro` (crusoe) | clean, terse, reliable at bulk pkg-mgmt generation |
| `gemma-4-31b` (crusoe) | clean but appends a reasoning/self-check block AFTER the JSON — harmless (script strips it) but wastes output budget |
| `Nemotron-3-Super-120B` (crusoe) | **one malformed line** mid-stream (visible reasoning leaked into a JSON value before self-correcting) — same failure shape gemini-flash showed in the original relabel round; read output, don't trust it blind |
| `gpt-oss-120b` (crusoe) | **empty answer** on a 28-row ask, hit its 16k output cap before writing anything — not retried this round, worth a shorter-ask retry next time |

**gemini-3.5-flash is not self-consistent run-to-run** — an identical prompt
flipped one row's label between two samples. A single relabel sample is not a
stable measurement on ambiguous rows.

## Known-open

- **Labels are PROPOSALS.** Bulk blind labeling still gated on budget. Every
  row carries its `author` tag so a later pass can disagree row by row.
- **44 contested rows (5.6%)** hold their original label with the dissent in
  `note`, per Amy: *"hesitate where gen/relabel disagree."*
- **All 8 slices now have a blind cross-family relabel pass**
  (85.4%–100%); see `relabel/README.md`. `slice8/execboundary` (the rule-16
  exec/no-exec contrast slice) hit **100%/100% agreement across two
  independent families** — the strongest result in this corpus's history.
  `slice7/sysverbs2`'s disagreements surfaced a genuine open question:
  applying rule 16's code-execution logic literally would also make
  ordinary dev-loop commands (`cargo build`, `npm test`, `make lint`)
  `data-critical`, which conflicts with their role as the probe gate's
  benign controls — flagged for Amy, not resolved by relabel majority.
- **`root_delete_over_source_file` was rebuilt** (2026-08-15) against a
  genuine no-interlock target (`rm -rf /*`, not the literal `rm -rf /`
  coreutils already refuses) — see `severity_probes/score_probes.py`. Both
  v8 and v9 pass 2 FAIL it once honestly tested; a real, previously-hidden
  gap, not a regression.
- **Score saturation, not absent vocabulary, is the live failure shape**
  (2026-08-15). Baseline probes meant to be ordinary controls (`rm -f
  src/view/vello_rasterizer.rs`) are drifting toward 0.999 alongside the
  sysadmin probes they're compared against, compressing the orderings the
  gate depends on. More rows targeting the still-failing constraints didn't
  move most of them this pass — see `severity_probes/baseline_v9_round2.json`
  commit message for the full readout. Next-pass design question, not solved
  here.

## A bug this corpus found in its own tooling

The validators normalized `text` with `.lower()` before the duplicate check, so
`git branch -d` collided with `git branch -D` — **opposite ground-truth
labels** — and the second was rejected as a duplicate. Structurally biased
against exactly the interlock-vs-force pairs rules 11–13 turn on. Fixed in all
four validators, regression test in `training/tests/test_validator_dedup.py`.
No prior data was lost; slice 1 was the first to hit it.
