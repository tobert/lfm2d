# Consumer integration contract

What lfm2d promises the programs that call it. The first consumer is
kaijutsu's `S50-lfm2d.kai` advisory hook (a `pre_call` hook on
`shell_write` that scores commands through `/v1/cascade` and writes the
read into kaijutsu's approval ledger); that script cites this file by
invariant number, so **the numbering below is part of the contract — add
new invariants at the end, never renumber**.

Wire shapes (exact field names and JSON structure) are pinned by
`lfm2d/tests/contract_serde.rs`. This file holds the semantic invariants
that shapes alone can't express. The endpoint reference and the
non-normative client-design notes are in `lfm2d/README.md`.

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
> ordinal consumer is wired**. `tests/severity_ladder.rs` pins v6's real
> (alphabetical) order against its committed config fixture, so the hazard
> is asserted rather than only described. The config can't be "fixed" by
> editing `id2label` —
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

**7. There is no global score threshold, and one cannot be calibrated.**
Rank clauses WITHIN a statement; never compare a score against a fixed
cutoff to decide anything. Measured on real traffic: the highest-scoring
benign clause reached **0.3415** while the lowest-scoring genuinely
data-critical one sat at **0.3440** — the bands touch, so every cutoff
buys a false positive for each true one it catches. The reason is
structural rather than a calibration bug: the axis the head actually
learned is corpus familiarity, not danger. v8 scored `rm -rf /` at
**0.404** and `git reset --hard` at **0.991**, and a `.md` target
suppresses the score even on a path like `/etc/shadow`. `/v1/cascade`
exists to do the supported operation — rank the clauses of one statement
against each other and name a winner — and a consumer that escalates on
the winner's LABEL inherits that ranking instead of inventing a cutoff
the model cannot support. Prior calibration (`tau`) moves where the
argmax falls; it does not create a threshold that separates the classes.

**8. An opinion is a distribution, not a decision.** `POST /v1/opinion`
(`docs/lfm25-adjudicator.md`, "The opinion API") never names a choice. A
consumer that acts on one picks the option itself, from thresholds fitted
on its own data, per spec, and refits when `snapshot_id` changes —
calibration does not transfer between specs, and the failure is silent.
Invariant 5 still holds for scores from the encoder heads; an opinion is a
different instrument with this contract of its own.

**9. Read the raw mass beside the renormalised probability.** `prob` and
`margin` are renormalised over the options asked; `sequence_mass` and
`first_token_mass` are the raw log mass the model put on that answer set.
A low mass means the model was never asked this question here, and a
renormalised number over it is noise that looks like an answer. A consumer
that thresholds `prob` or `margin` alone is thresholding that noise on
low-mass rows; gate on mass first, or record it beside every decision.

**10. An opinion may raise, never lower.** A static denial upstream (the
cascade's stage 2, kaijutsu's ledger) is never lowered by an opinion. An
opinion may escalate to the generative adjudicator; it is not the
adjudicator, and the description it echoes is the model's own text, not
evidence a consumer may act on.

**11. Specs, fields and options are read from the menu at runtime.**
`GET /v1/opinion/specs` lists every loaded spec with its fields in emission
order and each choice field's options. Never hard-code a spec name, a
field name, an option, or an option's position — invariants 1–3 again,
for the opinion vocabulary. `described` is an ordered list of
`{field, value}`, not an object; the order is the spec's. The menu now
also carries uploads (invariant 12) beside the boot-time specs.

**12. A spec's `id` is a content hash; an unknown `spec` means upload it.**
Every menu entry (boot or uploaded) has an `id`: the lowercase hex sha256
of the exact spec bytes, computed over the raw upload — never a
re-serialized/canonicalized form, so two specs differing only in field
order get different ids (a spec's field order is its emission order, which
is meaning, not formatting). `POST /v1/opinion/specs` registers a spec at
runtime; the body IS the spec's bytes. Registering bytes already loaded
(boot or upload) is free — `200`, no work done — which is what makes
"upload any time you're unsure" a correct client strategy, not just a
convenient one. `POST /v1/opinion` and `POST /v1/adjudicate`'s `spec`
field accepts an id or a boot-time spec's name; naming nothing loaded is
now `404`, not `400` — the client's cue to upload (or re-upload) and
retry, never a reason to fall back to a different spec or invent one.
Uploads are held in memory only, under a bounded least-recently-used
cache (`--opinion-spec-capacity`); an evicted upload's next request is
also a clean `404`. Boot-time specs (`--adjudicator-prompt`/
`--opinion-spec`) are never evicted and cannot be deleted at runtime
(`DELETE /v1/opinion/specs/{id}` on one is `403`). See
`docs/lfm25-adjudicator.md` "Runtime spec registration" for the wire, and
`docs/system1-split-plan.md` "Runtime spec registration" for the ruling
this implements (kaijutsu owns the shell specs and uploads them at
startup and on change, rather than lfm2d shipping them).

**13. `/v1/tokenize` and `/v1/probe` are instruments, not judgement
APIs — invariants 5 and 8–11 do not apply to either.** Neither returns a
decision, a calibrated score, or an opinion; `/v1/probe` takes free-form
request text by design (unlike `/v1/opinion`, whose questions come only
from a loaded spec's menu — invariant 11 exists precisely because that
one doesn't take request text). A consumer must not treat a `/v1/probe`
number as comparable to a `/v1/opinion`/`/v1/adjudicate` verdict without
checking first WHICH schedule it ran: with `decode_from` omitted,
`/v1/probe`'s bulk-only resume reproduces
`POST /v1/adjudicate {"opinion": true}`'s read bit-identically, but does
**NOT** reproduce `POST /v1/opinion`'s own read that way — measured
disagreement up to several nats at the same slot on the same input,
because the two paths take different kernels for the same tokens on this
backend. `/v1/probe` CAN reproduce `POST /v1/opinion`'s own read
bit-identically, but only when the caller supplies `decode_from` — the
byte offset where that read's generation actually began — since
`/v1/probe` has no spec/field awareness to infer it itself; without that
offset, or with the wrong one, the numbers are a different, related
computation, not `/v1/opinion`'s answer (`docs/lfm25-adjudicator.md`
"Probe and tokenize"). `/v1/probe` may be disabled entirely
(`--no-probe`/`LFM2D_PROBE=0`) without affecting `/v1/opinion` or
`/v1/adjudicate`; a consumer must not assume its presence. `/v1/tokenize`
carries no invariant beyond the general ones (1–3, for whichever model's
vocabulary is asked about) — it exposes tokenization, not a scored or
judged quantity.

## Operational numbers (measured, dated — re-measure before designing on them)

- **Latency** (2026-09-05, server-side, `--threads=8`): `/v1/classify`
  ~90 ms on a short single input; `/v1/cascade` p50 **673–727 ms** warm on
  a 5-clause statement, and p50 **746 ms** across a day of real hook
  traffic. A 5 s client timeout is comfortable; sub-second is not. The
  daemon serves through ONE serial inference worker, so concurrent callers
  queue rather than parallelise — throughput is ~1.4 req/s, and a caller
  that fans out past that backlogs everyone behind it.
- **Escalate volume** (2026-09-05, 1,224 real advisory rows over 24 h,
  `kube_ordinal_v10`): winners are **55.6% `informative`, 42.8%
  `situation-normal`, 1.10% `data-critical`**. So a policy that escalates
  only on `data-critical` fires on ~15 statements a day, and one that
  auto-allows only `informative` prompts on **43.9%** of commands.
  **These numbers moved a lot with the checkpoint** — the v9_cal-era read
  (2026-08-24, 10,751 rows) was 31.6 / 56.9 / **11.5%**, so data-critical
  fell 10x when v10 shipped. Re-measure against the deployed head; do not
  size a desk or a budget off a number from another checkpoint.
- **Precision context**: on live traffic most `data-critical` firings are
  still false positives. Of the ~15 in the 24 h read, ~9 fell in known
  families (`echo restored`/`echo stopped` ×6, `set -euo` ×2, git
  branch-read forms ×2) and only `rm -f` and `sudo -n` were arguably
  correct. Treat a firing as a ranking signal to enrich a prompt, not as
  ground truth — and expect a static post-filter on the winning clause's
  verb and redirects to remove most of the volume before it reaches a
  human or an adjudicator.
