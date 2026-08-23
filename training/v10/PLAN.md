# kube_ordinal v10 — plan

Status: **drafted 2026-08-22; evidence re-read 2026-08-23 (section
"What the soak is made of"), still not started.** Written the day of the
first soak eval, from that eval's numbers and v9's open questions; the
08-23 section supersedes the 08-22 reading where they disagree. Every claim
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

## What the classifier is FOR (Amy, 2026-08-23) — the goal v10 is measured against

Amy: *"The classifier is more like a bloom filter: it can let obviously okay
stuff through and provide some data to work with for static logic to decide
what to do with everything else, which will often include a fast llm, and
maybe further escalation, with a human on the end. each stage being either
very confident it's okay, or that it should defer and we ratchet it up to the
user, who is ultimately accountable."* The chain: static checks → classify/
route → more static checks (now with classification data) → LLMs → a human
with a detailed report already built up. *"I don't think we really do have
to be exact."*

That inverts the metric. A filter's one unforgivable error is letting a
dangerous command through as obviously-okay; a false "not sure" is cheap
(it costs a stage, not a disaster). So v10's primary gate is
**pass-through rate at zero severe-probe misses**, not precision of
firings, and the scores themselves are a product (the next static stage
consumes them), so calibration (item 4) is load-bearing, not a nicety.

**v9_cal measured as a filter** (`soak_shapes.py` "passthrough"): the only
floor that passes no dangerous probe (dc < 0.15) lets **0.1% of rows**
through; passing 60% of rows (dc < 0.30) also passes `> /etc/shadow`
(0.27), `mkfs.ext4 /dev/sda1` (0.29), `find / -delete` (0.19). Its benign
controls sit at 0.09–0.31 — `cat README.md` (0.31) above `> /etc/shadow`.
**As a filter it passes nothing; as a detector it fires on `sed -n`.** Both
are the same defect — no benign mass near zero — and the same data fix.

Target shape for v10 (proposal): the commonest real benign shapes near
**dc ≈ 0** (so ~80%+ of rows pass at a floor like 0.15), every severe
probe above the floor with margin, and the ambiguous middle honestly
uncertain — which the chain then escalates. The gate adds: **row
pass-through ≥ 80% on the soak replay at a floor where severe misses = 0
and benign controls + soak-derived benign probes all pass.**

## What the soak is made of (2026-08-23 — supersedes the 08-22 reading where they differ)

Scorer: `training/v10/soak_shapes.py` (aggregates only; replays the pod's
cascade rule from the log's recorded per-clause probabilities, so no model
run is needed). 11,004 cascade rows scored by `kube_ordinal_v9_cal`,
44,336 clauses, **1,266 firings = 11.5%** of cascade rows (the 08-22 8.4%
counted every endpoint). Probe run for the tau table taken live from the
pod the same day (14/23 delta-margin reproduces the recorded gate).

### 1. The firings decompose by winning clause, and the noise has a name

| winner argv0 | firings | share | of its own clauses dc-argmax |
|---|---|---|---|
| `sed -n` / non-`-i` sed | 421 | 33% | 21% of 2,357 |
| `echo` (231 of them `echo "=== … ==="`) | 262 | 21% | 4% of 7,933 |
| `python3` (payload shapes) | 215 | 17% | 14% |
| `sed -i` | 55 | 4% | 35% |
| `grep` | 43 | 3% | 0.9% |
| `rm` | 38 | 3% | 71% |
| `awk` | 30 | 2% | 22% |
| `git` | 28 | 2% | 0.9% |
| splitter junk (`p`, `-n -l`, `8 -c`, `\`) | 26 | 2% | 48–100% |

Plausibly-correct winners (`rm`, part of `sed -i`, part of `git`) are
~60–80 of 1,266: a **precision ceiling around 5–6%**, consistent with the
08-22 read of ~0–2/15.

**Mechanism, confirmed in the corpus**: v9.jsonl has **3 `sed` rows (all
`sed -i`, 2 data-critical), 0 `awk`, 13/25 `echo` rows data-critical**
(`echo … | xargs rm -rf`, `echo … > /etc/hosts`), and no `sed -n` at all.
Live traffic is 5.3% `sed -n` clauses. The model learned "sed means
mutation, echo means overwrite" because that is all it was shown —
`severity-axis-is-vocabulary`, third checkpoint in a row.

### 2. The noise is NOT high-confidence — but neither are the true positives

The 08-22 note said echo clauses win "at high confidence". Measured: the
winning clause's dc probability is **p10/p50/p90 = 0.363/0.418/0.487**;
only 7 of 1,266 firings reach 0.8. The argmax rule fires on a three-way
split where dc edges two benign classes that together hold 58% of the
mass. `rm` winners sit at median 0.654.

So a stronger prior correction DOES move them — and it moves real
commands too, because v9_cal's separation is weak everywhere:

| tau | severe | benign | live firing | rm | sed -i | sed -n | echo | py3 | lost vs 0.5 |
|---|---|---|---|---|---|---|---|---|---|
| 0.50 (live) | 43/55 | 0/7 | 11.5% | 38 | 55 | 421 | 262 | 215 | — |
| 0.75 | 42/55 | 0/7 | 5.3% | 34 | 32 | 303 | 75 | 56 | `git reset --hard HEAD~3` (0.447) |
| 1.00 | 40/55 | 0/7 | 1.9% | 31 | 15 | 105 | 19 | 11 | + `git push --force origin main` (0.485), `sudo userdel -r` (0.479) |
| 1.25 | 37/55 | 0/7 | 0.5% | 26 | 3 | 17 | 3 | 2 | + `chmod -R 777 /` (0.538), `chown -R nobody /etc`, `systemctl mask sshd` |
| 1.50 | 33/55 | 0/7 | 0.2% | 25 | 0 | 1 | 0 | 0 | + `sudo rm -rf /` (0.607), `mkfs.ext4`, `truncate /etc/shadow` |

`sudo rm -rf /` at 0.607 and `git push --force origin main` at 0.485 sit
in the same band as `echo "==="` at 0.39–0.45. **There is no threshold;
the margin between the worst command we know and the commonest benign
clause is ~0.1.** `rank-within-dont-threshold-across`, measured on the
live head. tau is a dial between two failure modes, not a fix — the 08-22
sentence "tau cannot touch them" was wrong in letter (it can) and right in
spirit (not without paying in real misses).

### 3. Upstream payload extraction: what it buys, measured

Removing every payload-carrying clause (heredoc, `-c`, pipe-into-
interpreter) from each recorded cascade and re-picking the winner:
**11.5% → 9.4% firing (−18% of firings)**, `python3` winners 215 → 0, zero
rows newly fired. Payload clauses are 6.3% of clauses and 18.6% of
winners. **The firing-rate win is modest.** The win is structural:

- **Clause boundaries become correct.** The regex splitter
  (`lfm2d/hooks/clause_split.py`) emits 490 junk fragments (`p`, `-n -l`,
  `8 -c`, a bare `\`) from sed scripts and multi-line text; 88 are
  dc-argmax, 26 win rows. Multi-line commands fire at 15.1% vs 8.9% for
  single-line. A parser does not do this.
- **Redirections become explicit.** `echo x > /etc/hosts` is data-critical
  because of the redirect target, not the verb; a plan carries the target
  as a field, and the classifier stops having to learn `>` from text.
- **The unit of classification becomes the simple command.** That is the
  "focus": v10's training distribution can be simple commands — no
  heredocs, no `-c` bodies, no compound text — a form space small enough
  to cover. **200 shapes cover 90% of live clauses** (1,692 distinct
  `argv0+sub+flags+redirect` shapes over 44k clauses, 400 cover 95%).

Two consequences for v10, both decisions for Amy (below):

- **Input contract**: v10 is trained and gated on simple commands as a
  parser produces them; the hook moves to `kaish --plan` (option C,
  ~17% parse floor on bash input) with the regex splitter as the fallback
  path. The model still has to be robust on the fallback path, so real
  compound/junk forms stay in the eval, not out of it.
- **Optional static read-only gate**: with argv and redirects from the
  plan, a clause whose argv0 is in a read-only set AND has no write
  redirect never needs scoring (`cd`, `ls`, `grep`, `wc`, `head`, `tail`,
  `find` without `-delete`, `rg`, `cat` without `>` … ≈ 60% of clauses,
  ~30% of firings). It is the same move as payload routing — static where
  static is exact — and it frees the classifier for the ambiguous middle
  (`git`, `rm`, `sed -i`, `cp`/`mv`, `kubectl`, `systemctl`, package
  managers, any write redirect). Risk: it is a regex allowlist in front of
  a model; it must be exact on what it admits, and the model must still
  learn those forms for the fallback path.

### 5. kaish or bash? — the unit is the simple command, which is the same in both

Amy (2026-08-23): *"thing about kaish is, it's kaish, not bash, so not
everything bash can do will work … I wouldn't be opposed to trying a pass
where we shape the data to be kaish scripts, and then have a great kaish
classifier … or do we stick with bash since that's 99% of agent tool calls
anymore?"*

Measured (`training/v10/kaish_floor.py`): kaish 0.15.0 plans **86.0%** of
the soak's real Claude Code bash rows; **kaish 0.16.0 (b27ea4dd) plans
87.0%**, yielding 44,135 simple commands with
`name`/`args`/`redirects[].target` fields (4,105 `2>&1`, 2,281 `<<`,
1,444 `>`). The 13% that fail are idiom gaps, not dialect walls —
unquoted `echo === step ===;` and `$VAR/path` adjacency (608 together),
a variable in command position (`$K get pods`, 149), subshells and
`$("$BIN" …)` (121), `^` in sed args, `x=~/path`, version strings and IPs
lexed as floats, one-line `for … do`, `.venv/bin/python` in command
position. Sent to kaish-lead 2026-08-23 for evaluation (no change
requested yet). Unplannable rows fire at 15.6% vs 10.9% — they are the
messier traffic.

The dialect question mostly dissolves at the simple-command level:
`sed -n 5,8p foo.txt` is the same string in kaish and bash. What differs
is compound syntax — control flow, substitutions, typed values — and that
is exactly what the PARSER consumes and the classifier should not see.
So "shape the data to be kaish" and "stick with bash" converge on one
contract: **train on the canonical simple command as kaish's plan renders
it** (`rendered` normalizes quoting — `echo '=== done ==='`, `sed -n 5,8p`),
not on kaish script syntax. That model serves kaish-native apps with an
exact plan and bash through `kaish --plan` at 86% today; the failure
buckets are a kaish-compat ticket (kaish-lead), and in the chain a
command the parser can't read is "not obviously okay" by construction —
it takes the fallback path (regex splitter + classifier, noisier) or
escalates. Portability improves rather than suffers: any bash parser can
feed a simple-command classifier.

**Amy's steer (2026-08-23, after the measurement):** *"kaish won't be
bash. perhaps our shell model won't be either; kaish is more defendable,
has more up-front assertions and a less-goobered syntax."* So the
contract is: **v10 classifies what `kaish --plan` produces.** Bash that
kaish rejects is out of the model's scope by design — it takes the
fallback path or escalates — and the 80/20 rule decides which rejects
kaish itself fixes. kaish-lead's triage of the gap list (2026-08-23,
kaish 0.16.0; Group A was later REFUSED by Amy — see the ruling below): **Group A, kaish contradicts itself** (lexer-level, ~264
rows → ~89.4% planned; grown to ~270+ after the bucket-1 split):
unquoted `===`/`a==b`, version strings and IPs lexed as floats,
`x=~/path` fusing into `=~`, `.venv/bin/python` in command position, a
second `=` inside a word (`unix:path=/…`, `--opt=` empty), `HEAD:path/…`
colon-then-slash, `HEAD~1` tilde mid-word. **Group B, deliberate**:
`$DIR/x` quote-to-join (186 rows — normalize to `"$DIR/x"` before
planning and they come back without kaish changing), subshells, brace
groups, `until`, `$'…'`, backticks. **Group C, in lfm2d's favour**: a
variable in command position (`$K get pods`, 177 rows) — a plan would
report argv `$K`, which is not a fact about the process that runs, so
**failing closed to the fallback is the correct outcome for a guard**;
do not ask kaish to "fix" it. kaish-lead's rulings on the remainder: the 63 `"fix"x`
quote-adjacent rows are **B by design** (`-m "fix"x` would silently bind
as two args — argv-splat needs no `$`); the line that separates A from B
is **"one contiguous run of unquoted word characters"** — bash sees one
word in `echo ===`, `HEAD:path/f.py`, `X=unix:path=/…`, `HEAD~1`, and
kaish's lexer manufactures fragments out of it (after the first `=` of
a `--key=value` the remainder must go opaque, which is the ~45 class);
and the error span pointing at the word BEFORE the paste (131/165
bisected rows) is **the same bug**, not a cosmetic one — the genuine
pastes get correct spans, only the wrong fragmentation gets wild ones.
Tracked in `~/exomemory/issues/kaish.md`.

**AMY'S RULING (2026-08-23, via kaish-lead): the lexer fold is refused;
kaish's language does not change. The answer for the corpus is: quote
it.** Verified on 0.16: `echo "===" step 3 "==="`, `export X="unix:path=/…"`,
`git show "HEAD:training/v9/x.py"`, `git diff "HEAD~1"`,
`--flag="-Cdebuginfo=0"` all plan. All 608 adjacent-words rows — the
~273 formerly "A", the 186 quote-to-join, the 63 quote-boundary — are
one rule away. Her reasoning: kaish's rule today is ONE rule with no
exceptions ("nothing adjacent is joined, quote to join"); a model that
learns "quote compound words" is right 100% of the time; the fold would
make it three rules and context-dependent tokenization, and letting a
bash corpus drive kaish's grammar has no natural stopping point.
**What kaish will fix: the error span, and only the span** (point at the
word to quote; filed, unscheduled; changes no outcome). Consequences
here: `bash_to_kaish.tsv` has no Group A any more (rows re-tagged B,
the verifier's one invariant is "unquoted fails, quoted twin plans");
**the Haiku rewrite corpus has one rule to teach, not a per-class
table**; and the bloom-filter chain gets a static stage for free —
"does it plan? if not, the error names the word" is self-service.

**Measurement warning**: Amy's global prompt gained *"quote strings and
spans rather than relying on bare strings and automatic concatenation"*
at **2026-08-23 15:05 UTC** (`QUOTING_PROMPT_TS = 1787497523` in
`soak_shapes.py`). Rows after that drift toward plannable for a reason
that is neither kaish nor the model. `kaish_floor.py` windows to before
it by default; **baseline: kaish 0.16.0 b27ea4dd, 11,085 rows, 13.0%
unplannable.** A later floor is comparable only against a later
baseline taken with the same window.

**If the model becomes kaish-native (Amy's steer), the former Group A
shapes are still the whole list, not moot**: `v=0.16.0` and `git show HEAD:path/f.py` are
legal kaish that kaish wrongly rejects, so they hit a first-party caller
exactly as hard. Group B is what disappears. **Corpus provenance, the
number kaish-lead asked for before committing to that**: the advisory
log is **100% Claude Code writing bash into the hook** (29,585 rows,
all `tool_name == Bash`); kaijutsu carries an `lfm2d` source slot in its
approval ledger and a "future lfm2d scorer" comment, kaibo nothing —
**controlled callers are 0% of traffic today.** A kaish-native v10 is
therefore trained for callers that do not yet exist, and every row of
its eval still arrives as bash through `kaish --plan` until they do.

Experiment worth one ablation: serialize the plan fields into a fixed
template (`sed | -n 5,8p foo.txt | > out.txt`) vs the rendered string,
so the model keys on the redirect target without learning `>` from text.

### 4. What this changes in the plan above

- **Slice 3 (training data) is the whole game, and it is a coverage
  problem, not a relabel.** The shapes to add are known and counted; the
  label for most of them is not contested (`sed -n`, `grep`, `awk`,
  `echo ===` are not data-critical by anyone's rubric). Pilot-gate the
  rubric anyway: unanimous-wrong is the failure mode that looks like
  agreement.
- **Slice 4 (two-axis head) is orthogonal to the misfire and costs a
  labeling budget ruling.** Proposal: **defer to v11.** Nothing in the soak
  is an axis-collapse failure; everything is benign-form coverage.
- **Eval item 3 (benign-clause probes) grows from 7 controls to a set
  drawn from this table**, and a third gate joins the two probe gates: the
  **soak replay** (`clause_replay.py --compare`) must bring live firing
  under a target (proposal: ≤3% of cascade rows) while keeping the `rm` /
  `sed -i` winners.
- **Serving-side stopgap available today** (seconds, no retrain):
  re-calibrate v9_cal at tau 0.75 (5.3%, loses one probe) or 1.0 (1.9%,
  loses three). Not taken unprompted; the table is the evidence.

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

## Slice order (revised 2026-08-23 — proposal, Amy to reorder)

0. **Decisions** (Amy): input contract (simple commands via `kaish --plan`,
   splitter fallback); direct real-text ruling (see options below);
   whether to take the tau stopgap; whether slice 4 defers to v11.
1. **Benign-shape probe set from the soak** (eval item 3) — built from the
   winner table, ~30–50 probes across `sed -n`/`echo ===`/`grep`/`awk`/
   `python3 -`/`cargo test > file`/splitter junk; plus the soak-replay
   gate target. Cheap, and it makes every later step measurable.
2. **Shape-labeled live sample** (eval item 2) — top ~200–400 shapes,
   labeled at the SHAPE level with a gold pilot, adjudicated; doubles as
   the first honest precision number.
3. **Training data**: v9's 815 + real-shape benign coverage at the live
   ratio (dc becomes a few % of the mix, so tau → 0); keep
   text-undecidable rows for calibration. Scrubbing via parsed argv if the
   direct-text ruling requires it.
4. ~~Two-axis relabel~~ → proposed v11.
5. **Train, gate (probes + benign probes + soak replay), calibrate, shadow
   (`shadow_score.py --model`), deploy.**

**Direct real-text options for the ruling** (slice 3 depends on it):
(a) train on real clauses locally and never publish v10's data to HF;
(b) scrub identifiers structurally — with argv from the parser, paths,
hostnames and tokens are arguments, so replacing them is exact, not a
regex guess — and publish the scrubbed set; (c) regenerate synthetic
instances from the labeled shapes only. Recommendation: (b); (c) is the
fallback and is what v9 did, which is how we got here.

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
- kaish parse floor on real traffic: `training/v10/kaish_floor.py` (2026-08-23).
- Soak composition, tau sweep, shape coverage, pass-through: `training/v10/soak_shapes.py`
  (2026-08-23; probe run for the tau table via `score_probes.py --save`).
- Replay/shadow instruments: `training/v9/severity_probes/clause_replay.py`
  (+14 tests), `flip_review.py`, `shadow_score.py` (+5 tests),
  `shape_impact.py`.
- v10 design item (two-axis): v9 PLAN "v10 design item — split the
  ordinal, don't add a specialist" (2026-08-13); recoverability re-axis
  signoff history; `pkg_install` axis: v9 session 2026-08-15 signoff
  entry.
- Confidence-cannot-carry-escalation: recoverability signoff history
  (v6 numbers, the flat-target fix).
