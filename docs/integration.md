# Consumer integration contract

What lfm2d promises the programs that call it. The first consumer is
kaijutsu's `S50-lfm2d.kai` advisory hook (a `pre_call` hook on
`shell_write` that scores commands through `/v1/cascade` and writes the
read into kaijutsu's approval ledger); that script cites this file by
invariant number, so **the numbering below is part of the contract — add
new invariants at the end, never renumber**.

Wire shapes (exact field names and JSON structure) are pinned by
`lfm2d/tests/contract_serde.rs`. This file holds the semantic invariants
that shapes alone can't express.

## Invariants

**1. `GET /v1/models` is the source of truth for label vocabulary.**
Read it at fire time. Never hard-code a label name, count, or position
into a consumer — a baked-in label goes stale silently the next time the
daemon's checkpoint changes.

**2. The label vocabulary changes between checkpoints.** Not
hypothetically: `kube_ordinal_v6` served `destructive` / `informative` /
`mutating`; v8 and later serve `informative` / `situation-normal` /
`data-critical`. A consumer that matches on label strings breaks on
rollback or upgrade; invariant 1 is how it survives.

**3. For severity classifiers (`kind: "classifier"`), the `labels` array
is in ascending severity order — from v8 onward.** Index 0 is the least
severe rung, the last index the most severe; this is what makes an
ordinal-position policy mapping (first → allow, last → deny) correct.
The order on the wire is the checkpoint's `id2label` id order, pinned by
`tests/severity_ladder.rs` against the shipped configs.

> **Rollback hazard**: `kube_ordinal_v6` predates this convention — its
> id order is *alphabetical* (`destructive` at index 0). Pointing an
> ordinal-position consumer at v6 inverts its verdict mapping: the most
> destructive label reads as "allow". v6 stays staged for score-level
> debugging, but **it is not an ordinal-safe rollback target while any
> ordinal consumer is wired** (`signoff.md` "Rollback is one line" carries
> the same warning). The config can't be "fixed" by editing `id2label` —
> label order and classifier weight rows must permute together or
> predictions silently corrupt.

**4. `/v1/cascade`'s `models[0]` is the severity classifier** that
produced every `severity_scores` map in the response (`models[1]` is the
router that produced `lane`). Structural in `engine_real.rs::cascade` —
the array is built in that order, not sorted or filtered.

**5. Scores are advisory.** The classifier is wrong in both directions on
real shell text (measured repeatedly; see `training/v10/PLAN.md`). A
consumer may let a score RAISE a prompt that would not otherwise fire; it
must never let one lower a prompt, auto-decide, or silently allow —
kaijutsu's ledger design (2026-08-17 ruling) is the reference posture.
A verdict is also a function of (weights, input, device): ~3% of verdicts
flip between CPU and GPU on identical weights and input, so byte-identical
reproduction across hosts is not promised.

**6. Fail open, loudly.** When lfm2d is unreachable, slow, or answers
with an unexpected shape, a consumer must proceed with its baseline
controls unchanged and record that it skipped — never block on the
daemon's availability. (The daemon itself has been down for 21-hour
stretches; a guard that denied on outage would have frozen every seat.)

## Operational numbers (measured, dated — re-measure before designing on them)

- **Latency** (2026-08-13, server-side): `/v1/classify` p50 ~110 ms,
  p99 250–750 ms; `/v1/cascade` p50 ~720 ms on real multi-clause
  commands. A 5 s client timeout is comfortable; sub-second is not.
- **Escalate volume under an ordinal mapping** (2026-08-24, 10,751 real
  cascade rows, v9_cal era): winners are 31.6% `informative`, 56.9%
  `situation-normal`, 11.5% `data-critical`. An ordinal policy that
  auto-allows only index 0 prompts a human on **68.4% of commands** —
  run new consumers in log-only mode first (Amy's standing ruling).
- **Precision context**: on live traffic the `data-critical` argmax has
  run at a ~5% precision ceiling (v8-era read; v9_cal's noise profile is
  in `training/v10/PLAN.md`). Treat a firing as a ranking signal to
  enrich a prompt, not as ground truth.
