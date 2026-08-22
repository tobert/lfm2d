# kube_ordinal v10 — plan

Status: **drafted 2026-08-22, not started.** Written the day of the first
soak eval, from that eval's numbers and v9's open questions. Every claim
below has a source; gaps are marked as gaps. Amy's words are quoted because
the wording is the rubric.

## What changed since v9 — the reason there is a v10

**v9_cal ran a six-day soak (08-16 → 08-22) and we read the record.**
15,079 replayable rows, 1,266 firings, **8.4% of traffic** (6.7–11% per
day, sustained — the ~4% deploy anchor was a whole-text tau sweep; live
runs the clause cascade). A clause-faithful replay instrument
(`training/v9/severity_probes/clause_replay.py`) reproduces the pod on
15,079/15,079 rows, so what follows is the model's behavior, not
instrument noise.

**The firings are mostly wrong.** A random sample of 15 firings read
**~0/15 truly data-critical** (~2/15 even defensible). The shapes:

- **bare `echo "=== ... ==="` clauses winning the cascade at high
  confidence** — 5 of the 15 sample, still firing live at the time of
  writing (08-22 13:09). A clause-level classifier error the cascade then
  promotes to the row verdict.
- **read-only commands firing**: `sed -n`, `grep`, `awk`, `wc -l`,
  `cargo test > file`.
- **payload shapes over-represented at scale** (n≈5 flag retired,
  `shape_impact.py` on 14,255 rows): heredoc **1.73×**, interp `-c`
  **3.72×** (n=45 firings), pipe-into-interp **2.77×**.

**These are ranking failures, not threshold failures.** tau cannot touch
them; raising tau would kill genuine firings without moving an echo
clause's score. Payload exclusion helps the payload share but cannot help
an `echo` at all. The classifier's clause-level behavior on the commonest
real shapes is wrong, and the commonest real shapes are exactly what the
synthetic training mix lacked: real traffic is 92% compound (mining
corpus), most clauses are benign reads/markers, and v9's mix was 52.6%
data-critical single clauses.

**One soak finding Amy resolved by architecture, not by this model**
(2026-08-22): *"I feel ok with the 'python -c' class of thing for now, I
think in our apps we have kaish and can detect that statically, and
redirect the python source to a python specialist, all static and super
fast, so few will come to our shell classifier. same for ruby etc, we'll
use a small llm for quick checks."* Interpreter payloads (the dominant
heredoc family in the flip data: PY/PYEOF were the top delimiters) leave
this surface upstream in kaish-native stacks. v10 therefore does NOT spend
capacity on interpreter-payload detection; it must still handle
shell-facing heredocs and today's bash-hook traffic, which has no parser
in front of it.

## What v10 does NOT change

- **Trunk, cascade, clause splitter, three-rung vocabulary.** The
  architecture is not what misfired; the classifier's clause verdicts did.
  The three-class shape is settled (recoverability re-axis, both blind
  reviewers argued for three).
- **The serving contract.** tau stays a serving-time knob; what changes is
  how it is SET (below).
- **The gate.** severity probes (classic + delta-margin) still gate every
  checkpoint. v10 must not regress: v9_cal was 43/55 severe probes at
  τ=0.5 with the known `dd` raw-device hole (v8 misses it too).

## What v10 changes

### 1. Train on the real shape distribution, not a 52.6%-dc prior

tau=0.5 is a patch over a corpus property (v9 PLAN "Open questions"). v10
fixes the property: the training mix's class balance and clause-shape
distribution should reflect scored live traffic, so tau becomes a trim
tab, not load-bearing. The advisory log is the distribution source —
15,079 clause-faithfully-scored rows with per-clause texts recorded.

### 2. Real traffic forms enter the training data (gated question)

The echo/read-only/compound-read forms v9 misreads are precisely the
forms synthetic generation never produced. v9's PLAN left deliberately
open **whether real mined commands can be used as training text directly**
(Amy's session history: paths, hostnames, possibly secrets; any row that
ships to HF needs review). **That question becomes blocking for v10.** If
direct text is barred, the fallback is generation seeded from mined shapes
with identifiers scrubbed — but the form coverage argument (v9 process rule
2) says the real forms are the point.

### 3. Two-axis multi-task head (the 2026-08-13 design item, now scheduled)

The ordinal collapses **blast radius** and **recoverability**; v8 picked
vocabulary instead of either. The shape is multi-task heads on ONE trunk
(~0 ms, ~24 KiB), NOT a specialist (rejected twice: latency, and a
specialist trained on this corpus inherits the blind spot). The
`pkg_install` boolean axis is already captured in v9's data (105/815 true)
and rides along — data-only until now, by design. **Gated on:** two-axis
relabel of the corpus, which is gated on the labeling-budget question.
Read `ordinal-collapsed-to-a-set-loses-order` first — that bite cost three
checkpoints.

### 4. Calibration becomes a shipped eval metric, because confidence must carry escalation

Amy (recoverability session): *"we will likely use classifier confidence
to decide when to kick something to a haiku-class model for more
intelligence."* v6's confidence CANNOT do that today: mean top-1
probability 0.868 on should-be-confident rows vs 0.808 on rows a
calibrated model should be unsure about — six hundredths, with the errors
running backwards (`rm -rf /data` answered 0.968; `mkfs.ext4` 0.596).
Temperature scaling can't fix rank inversions. The known fix is a data
change: stop deleting text-undecidable rows, keep them, train toward a
flat target, so the model learns to be unsure where the world is unsure.
v10 ships a calibration metric (argmax accuracy is blind to this) next to
the two gates, and the text-undecidable rows stay in the mix.

### 5. Payload exclusion lands as an optimisation, scoped by the routing ruling

Still worth doing for SHELL-facing heredocs and bash-hook traffic: kaish
0.15 exposes exact heredoc spans (`PlannedHeredoc`), option C
(`kaish --plan` from the hook) works today with a ~17% parse floor on
bash input, and the measured clear rate is 91% (350 historical heredocs
against live v9_cal). Design constraint stands: exclusion is an
optimisation over a correct-but-noisy default, never load-bearing; guard
the kaish version explicitly. `-c` argv spans are NO LONGER worth asking
kaish for (upstream routing absorbs that class).

## Eval — what must be true before v10 ships

1. **Both probe gates**: classic ≥ v9_cal's 16/23, delta-margin ≥ 14/23,
   severe probes ≥ 43/55, benign controls 0/7.
2. **A labeled live-traffic sample, drawn by DISTINCT SHAPE** — the
   instrument this project has lacked since v8's 9% precision estimate.
   Sampled from the advisory log's clause texts across shapes (echo/
   read-only/heredoc/-c/mutation/sysadmin), adjudicated the way the
   recoverability worksheet was (no self-grading). This sets tau from a
   target precision instead of from v8 nostalgia, and produces the first
   honest precision number with a confidence interval. Budget is known:
   under $1 per 182 rows measured across six families.
3. **A benign-clause false-firing probe set made from the soak itself**:
   the echo-winner rows and read-only firings become standing probes —
   the complement to the severity probes' deliberate out-of-corpus
   philosophy: these are in-corpus shapes the TRAINING mix missed.
4. **Calibration metric** (item 4 above) with the escalation use named.
5. **Clause-faithful shadow before deploy**: the running shadow scorer
   (`shadow_score.py`, repointed 2026-08-22) scores the v10 candidate
   against live v9_cal on identical clause inputs as traffic arrives;
   `clause_replay.py --compare` backfills the whole soak in one shot.
   Deploy is then the usual one-line `--classifier-dir` swap.

## Slice order (proposal, Amy to reorder)

1. **Labeled-by-shape live sample** (eval item 2) — gates tau, gates the
   precision claim, and produces the seed set for slice 3.
2. **Benign-clause probe set from the soak** (eval item 3).
3. **Training data**: balanced mix + real forms (needs the direct-text
   ruling); keep text-undecidable rows.
4. **Two-axis relabel** for the multi-task head (labeling-budget ruling).
5. **Train, gate, calibrate, shadow, deploy.**

## Open questions (carried, not re-litigated)

- **Direct real-text training + HF publication review** (v9 PLAN; now
  blocking for v10 slice 3).
- **Labeling budget for the two-axis relabel** (open since 2026-08-13).
- **Rule 16 refinements held contested**: lockfile-pinned installs
  (`npm ci`, `pip install -r requirements.txt --require-hashes`) and two
  ecosystems with unverified build-hook semantics (`mix`, `flutter`) —
  need Amy's ruling.
- **Multi-task shape confirmation**: auxiliary heads beside a three-rung
  severity head, or the severity head itself re-axed — confirm before the
  relabel spends money.
- **Escalation wiring** (confidence → haiku-class): consumer-side, but the
  calibration metric must exist first; kaijutsu is the consumer.

## Sources

- Soak eval numbers + annotated misclassification review:
  `~/.cache/claude-hooks/flip-review-2026-08-22-annotated.md` (0600,
  local-only) and the 2026-08-22 signoff entries.
- Replay/shadow instruments: `training/v9/severity_probes/clause_replay.py`
  (+14 tests), `flip_review.py`, `shadow_score.py` (+5 tests),
  `shape_impact.py`.
- v10 design item (two-axis): v9 PLAN "v10 design item — split the
  ordinal, don't add a specialist" (2026-08-13); recoverability re-axis
  signoff history; `pkg_install` axis: v9 session 2026-08-15 signoff
  entry.
- Confidence-cannot-carry-escalation: recoverability signoff history
  (v6 numbers, the flat-target fix).
