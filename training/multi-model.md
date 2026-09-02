> **Historical method note.** How the early synthetic corpora were
> generated across model families and cross-validated, from the era before
> the corpus was built from real commands. The transport and the
> blind-relabel discipline still apply; the corpus numbers do not describe
> anything currently shipped.

# Multi-model generation survey — 2026-08-05

Fixes the one-family blind spot: all gen2 rows and both holdouts were
generated AND labeled by claude-haiku-4-5, so holdout numbers measured
same-family generalization. This survey probed five outside families as
generators and cross-validated labels three ways.

## Transport (proven this session)

kaibo consult with `save_artifacts` → `kaibo://cas/<digest>` → copy to
`~/.local/share/lfm2-training-data/incoming/` → **sha256(file) ==
digest** → `validate_incoming.py` (schema gate) → `report_dataset_stats.py`
(dup/length aggregates). Rows never transit an exec-capable agent
context; subagents and read-only kaibo consults do the row work
(aggregates-only reporting).

## Roster and probe design

Standard probes: 30 command-shaped rows, self-chosen persona, house
rubric from `personas.md`. Subtle probes (kimi, glm): two-pass sessions —
a design turn proposing a 12-category boundary taxonomy, then a
generation turn producing 20 training rows + 10 eval-only rows. Locals
were down; anthropic casts skipped (same family as incumbent data).

## Scorecard (labels post-adjudication)

| family | rows | gates | near-dup | self-report | haiku agree | v3 agree | notes |
|---|---|---|---|---|---|---|---|
| deepseek-v4-pro | 30 | 0 err | 0% | exact | 93% | 93% | best chain variety (12 binaries); quoted fragments in answer |
| gpt-5.6-luna | 30 | 0 err | 0% | exact | 97% | 87% | novel persona; cleanest compliance; cheapest probe |
| gemini-3.5-flash | 30 | 0 err | 3.3% | exact | 100% | 93% | costliest probe (96k in + 22k reasoning) |
| qwen3-coder-next | 32 | 0 err | 0% | **wrong** | 88% | 87.5% | ignored "exactly 30"; 3 labels overturned; short rows |
| kimi-k2.7 subtle | 20+10 | 0 err | 0% | exact | 60–80% | 85%/70% | ALL 7 disputed labels upheld — haiku was wrong, not kimi |
| glm-5.2 subtle | 20+10 | 0 err | 0% | exact | 90–100% | 80%/**30%** | eval-only slice crushes v3, labels 10/10 validated |

Total: 182 rows, ~350k in / 85k out tokens across providers (< $1).

## Label adjudication

Blind haiku relabel of all 182 (shuffled, no labels/authors) agreed with
generators on 165 (91%). The 17 disagreements went to a blind
gpt-5.6-luna third vote; majority ruled: **13 generator labels upheld, 4
overturned** (all destructive→mutating: qwen 3, deepseek 1 — the etcd
snapshot-save direction and serviceaccount-adjacent deletes). Overturned
rows patched in `incoming/` with `contested: true` and an adjudication
note; the CAS artifacts remain the pristine originals.

Pattern worth watching: haiku sat alone on 13 of 17 disputes, always on
the lenient side of subtle destructive rows. The gen2 corpus is
haiku-labeled; boundary-row severity may be systematically soft there.

## Findings

1. **v3 generalizes across families on ordinary command-shaped rows**
   (87–97% agreement) — the family blind spot is small for the covered
   surface form, consistent with the form-coverage law.
2. **Adversarial read-shaped chains are an uncovered form.** GLM's
   eval-only probes (buried-destructive-in-long-chain, etcd-level
   finalizer bypass, read-chain-ending-in-write) score **3/10 with five
   severe destructive→informative misses** — the dangerous direction
   (under-flagging). Labels independently validated 10/10 by haiku.
   This slice is v4's first target.
3. **Audit-log deletion is EVAL-ONLY** (Amy's ruling, this session):
   kimi argued training on it teaches a brittle "audit = destructive"
   keyword override; probe first, train only if v4 still fails.
4. **Never trust generator self-reports** — qwen claimed 30 rows @
   12/13/5, delivered 32 @ 12/9/11. Gates on artifacts, always.
5. Kimi/GLM taxonomies converged independently (finalizer surgery, audit
   pruning, destructive-as-data, cascade choices, controller-vs-bare,
   helm edges, exec trailing risk) — evidence the boundary categories
   are real, not one model's idiosyncrasy.

## Disposition

- `incoming/kube_gen_{deepseek,gemini,gpt_luna,qwen}_sample.jsonl` and
  `incoming/kube_subtle_{kimi,glm}_train.jsonl` (142 rows): candidate
  training material for v4, pending the usual `kube_prep_splits.py`
  contradiction/leakage gates against the full corpus.
- `incoming/kube_subtle_{kimi,glm}_evalonly.jsonl` (20 rows): join the
  holdout suite; never train.
- Cross-check machinery: `crosscheck_prep.py` (prep/score),
  `validate_incoming.py` (schema gate). Verdicts and blind files under
  `~/.local/share/lfm2-training-data/crosscheck/`.

Models engaged: deepseek-v4-pro, gpt-5.6-luna, gemini-3.5-flash,
qwen3-coder-next, kimi-k2.7-code, glm-5.2 (generation/design);
claude-haiku-4-5 (blind relabel); gpt-5.6-luna (blind arbitration);
all via kaibo except the haiku subagent.

## Corpus reclassification (same day, Amy's call)

Luna reviewed every haiku label in the 16 kube source files (1,979 rows,
5 chunked consults via `recheck_labels.py`; instructed to say "unsure"
rather than guess). GLM-5.2 ruled on the unsure set only.

- Luna: 1,858 upheld, **91 corrected (4.6%)**, 30 unsure. Corrections
  skew DOWN in severity (41 downgrades vs 15 upgrades): generation-time
  haiku over-flags — the origin of v3's conservative lean. Worst files:
  cluster_admin 31/100, contested slice 15/225, ci_pipeline 13/100.
  storage_stateful concentrated the genuinely hard rows (14/100 unsure).
- GLM: decided 29/30 unsure (18 further moves, all →mutating); 1 row
  (holdout_nl_assistant:40) undecidable by both — label stands, flagged
  contested. One luna patch created a text-identical contradiction
  (ci_pipeline:90 vs an sre_incident read); GLM arbitrated → informative.
- All patches carry `contested: true` + an audit note naming reviewer
  and move; CAS/original provenance unchanged.
- **kube_test moved (8 rows)** — historical v1–v3 numbers were measured
  against a slightly wrong stick. v3 on corrected labels: test 88.6%
  (was 90.9), NL holdout 85.0% (was 83.3), auditor 90.0% (unchanged,
  same 3 severe audit-deletion misses).

v4 splits (post-recheck, incoming included): kube_train_v4 1,632 /
kube_val_v4 186, stratified by (source, label); 19 sources; gates green.

Ops note: 5 parallel luna consults tripped OpenAI's 200k TPM — two jobs
died AFTER saving complete verdict artifacts (CAS made the failures
free), two died mid-write and re-ran as half chunks. Run bulk luna
passes ≤2 at a time.

## v4 result (same recipe as v3, corrected labels, multi-model pool)

val_acc 93.0. Evals (v3-on-corrected-labels vs v4):

| set | v3 | v4 |
|---|---|---|
| kube_test (corrected) | 88.6% | 88.6% |
| NL-assistant holdout | 85.0% | 81.7% (+1 severe) |
| security-auditor holdout | 90.0% (3 severe) | 90.0% (3 severe) |
| glm eval-only (adversarial chains) | 30.0% (5 severe) | 30.0% (5 severe) |
| kimi eval-only (audit/storage/context) | 70.0% | 50.0% |

**The form-coverage law confirmed a second time, from the other
direction.** 161 multi-model rows of already-covered command-shaped
forms bought nothing (test flat), diluted the NL fraction slightly (NL
holdout −3.3), and the adversarial read-shaped-chain forms stayed
broken at 3/10 — because we deliberately reserved them as eval-only, so
the model never saw the FORM. Multi-family variance within a covered
form is worth as little as persona variance was.

**v3 remains the ship candidate.** v4's real value was validating the
pipeline end-to-end on corrected labels.

Path to v5: generate a TRAINING slice of the read-shaped-chain form
(~100-150 fresh rows, multiple families: benign read chains AND chains
ending in writes/exec-trailing risk, correctly labeled) while keeping
the existing eval-only rows as the untouched holdout — cover the form,
hold out the instances. Same play that fixed NL (25%→83%). Also rebalance
NL fraction in the pool.

## v5 result — the law pays out a third time

New training data: 114 read-shaped-chain rows (deepseek 50, gpt-luna 39
post-repair, kimi 25; gemini 503'd out and was not retried) + 60
multi-family NL rows (deepseek, gpt-luna — first non-haiku NL in the
corpus). GLM generated NONE of the chain rows (it authored the eval-only
probes; no teaching to the test). Cross-review with no self-grading:
luna checked deepseek+kimi (2 corrections, 1 unsure), GLM checked luna's
(1 correction — cascade=orphan destructive→mutating), GLM arbitrated the
unsure row (controller-owned force-delete → mutating). Pool: 1,788
train / 204 val, 24 sources. val_acc 90.2.

| set | v3 | v4 | v5 |
|---|---|---|---|
| kube_test (corrected) | 88.6% | 88.6% | 88.6% |
| NL-assistant holdout | 85.0% (0 sev) | 81.7% (1 sev) | 85.0% (1 sev) |
| security-auditor holdout | 90.0% (3 sev) | 90.0% (3 sev) | 90.0% (2 sev) |
| glm eval-only (adversarial chains) | 30.0% (5 sev) | 30.0% (5 sev) | **60.0% (2 sev)** |
| kimi eval-only | 70.0% | 50.0% | 60.0% |

114 rows of the missing FORM doubled the adversarial slice and cut
severe under-flags 5→2, with zero regression elsewhere — where 161 rows
of covered-form variety (v4) moved nothing. **v5 is the new ship
candidate.** Remaining known debts: 2 severe under-flags on adversarial
chains, 2 on audit-deletion (still eval-only by design), 1 severe NL
miss; all eval slices are small (N=10-88) — expanding them is the next
measurement investment.

## v6 result — targeted four-form slice (miss-driven, test never revealed)

81 rows from deepseek(32)/gpt-luna(29)/kimi(20), four coverage areas
derived from characterized v5 misses (pipe-to-interpreter, gated
conditional chains, log/audit-file lifecycle with balanced neighbors, NL
error-diagnostics), each area label-balanced so no keyword shortcut
works. Generator prompts described the FORMS only — no eval rows, no
model weaknesses named. Cross-review: 80/81 upheld, zero corrections, 1
arbitration (upheld). Pool 1,860/213. val_acc 92.5.

| set | v3 | v5 | v6 |
|---|---|---|---|
| kube_test (corrected) | 88.6% | 88.6% | **90.9%** (all 8 misses adjacent, 6 contested) |
| NL-assistant holdout | 85.0% (0 sev) | 85.0% (1 sev) | **90.0% (0 sev)** |
| security-auditor holdout | 90.0% (3 sev) | 90.0% (2 sev) | **91.7% (2 sev)** |
| glm eval-only | 30.0% (5 sev) | 60.0% (2 sev) | **70.0% (1 sev)** |
| kimi eval-only | 70.0% | 60.0% | **80.0%** (both misses contested storage patches, adjacent) |

**v6 is the new ship candidate.** Severe misses corpus-wide: 3 (was 5
at v5, 8+ at v3-on-corrected). The permission-gated destructive chain
and the NL error-diagnostic severe misses are FIXED. Still standing:
- the audit-log deletion pair (46/47) still lands informative top-1 —
  Amy's "too subtle, settle for mutating" hunch was right at this data
  dose; the consumer-side P(informative) gate is the mitigation, or a
  bigger balanced audit slice / cost-sensitive loss are next levers.
- the pipe-to-bash configmap row — gold label itself contested.
Both eval-only slices improved WITHOUT training on them (form transfer,
not instance leakage — the prompts never saw the test).
