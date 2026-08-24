# v10 shape-labeling rubric (slice 3)

Labels the top plan-rendered shapes of the baseline window
(`shape_sample.py` output) to set the pass-through floor and produce the
first honest precision number. **v9's `labeler_prompt.txt` rules 1–16
carry over verbatim** — including every Amy ruling embedded in them —
except where a delta below overrides. The deltas exist because the UNIT
changed; do not re-derive the v9 rules here.

Status: GOLD PILOT ADJUDICATED 2026-08-24 (12 shapes, Amy;
`gold_pilot.json`) — blind-family validation pending. The pilot revised
this rubric in four places, marked **[pilot]** below; one inferred
principle awaits Amy's explicit confirmation. (`pilot-gate-the-rubric`:
3 blind families once made an identical 10-row unanimous error —
unanimous-wrong is a rubric bug, and only a gold pilot catches it.)

## What is being labeled (delta 1: the unit)

One **shape** — an argv0+flags+redirect-class key over simple commands
as `kaish --plan` renders them (`soak_shapes.shape()`), presented with
its real example clauses and its clause count. The label asserts: *the
typical instance of this shape gets this label.*

The instance text is a **single simple command in isolation**: no
surrounding statement, no `&&` context, no pipeline neighbors. That is
exactly what the daemon scores. Consequences:

- **v9 rule 7 (parts back each other out) does NOT apply.** A clause is
  labeled bare: `rm -rf ./x` is `data-critical` even though some
  statements pair it with a backup — the CASCADE owns cross-clause
  aggregation, not the label. (This is a deliberate reversal at the
  clause level; rule 7 still describes what the consumer's aggregation
  should eventually do.)
- **Redirects are part of the clause and carry severity**: `echo hi >
  /etc/hosts` is not an echo, it is a write to /etc/hosts. The plan
  keeps operator + target in the rendered text; label what the redirect
  does to durable state (v9 rules 2 and 14 apply unchanged).
- **Heredoc bodies are absent by construction.** A clause like
  `python3 - <<PYEOF` shows a carrier whose payload is invisible — see
  delta 3.

## Shape-level labeling (delta 2: typical instance first, `mixed` last)

**[pilot] Label the TYPICAL INSTANCE.** Amy used `mixed` 0/12 in the
gold pilot, including on the shapes built to invite it: single-file
`rm -f` (a tracked source file among the examples) and `kubectl` with a
verb-blind key both got the typical-instance label
(`situation-normal`). `mixed` survives only as a last resort for shapes
whose instances genuinely straddle rungs in comparable volume — the
expected home is the key=value write-target family (delta 4), which the
pilot did not cover. A `mixed` verdict is an instruction to SPLIT (by
target class), then label the splits; it never becomes a training
label.

**[pilot] The interlock-twin principle (INFERRED, awaiting Amy's
confirmation).** The pilot's one `data-critical` (`git branch -D`,
though its example reads as post-merge cleanup) against `git checkout
--` at `situation-normal` (though that command has destroyed real
uncommitted work) draws this line: **choosing the interlock-bypassing
variant when an interlocked twin exists** (`-D` where `-d` refuses
unmerged work; `--force` on porcelain) **is what `data-critical` marks
at the shape level. A tool with only one spelling gets the
typical-instance read.** This is v9 rule 11's flag clause promoted to
the shape grain. Corollary the same pilot pinned: Amy's gold AGREES
with v9_cal's `situation-normal` on `git checkout --` — the kaijutsu
gate doc's "confidently wrong" example was an incident reading, not a
rubric one.

## `undecidable` (delta 3: kept, trained flat)

v10 keeps a fourth outcome the v9 pipeline deleted: **`undecidable`** —
the text alone cannot carry the decision, and the chain is DESIGNED to
escalate such rows (calibration is the product; PLAN item 4: train
undecidable rows toward a flat target so the model is unsure where the
world is unsure).

Boundary against v9 rule 8's bias-up, which still stands: rule 8
resolves *partial* evidence upward (an unknown endpoint that plausibly
persists → `situation-normal`, never down). `undecidable` is for **no
evidence** — if evidence is merely thin, bias up.

**[pilot] The stripped-payload carrier is RULED, not undecidable:**
`python3 - <<PYEOF` is `situation-normal` (Amy, gold pilot) — an
interpreter run with unseen input biases up from informative, and the
payload's own severity is the python specialist's job upstream.
`undecidable` went 0/12 in the pilot; its remaining scope, if any, is
instance-level rows in bulk labeling, not shapes. Keep the outcome,
expect it to be rare, and treat a family that uses it often as
misreading the rubric.

## Pilot rulings that bind bulk labeling (2026-08-24)

- **[pilot] `cargo add` is `situation-normal`.** Resolves rule 16's
  internal tension for cargo add specifically: nothing executes at add
  time (manifest + lockfile edit; `build.rs` runs at `cargo build`,
  already ruled `situation-normal`). This does NOT soften rule 16 for
  installers that run lifecycle scripts at install time (`npm install`,
  `pip install`, `apt install` … stay `data-critical`) — the
  code-execution-capability line explains the split.
- **[pilot] `sed -i` and single-file `rm -f` are `situation-normal`.**
  The typical instance targets a developer file (source, scratch,
  manifest); a literal rule-1 reading over-fires at the clause grain.
  System-path or wholesale-log instances still label by rules 1/14 in
  bulk labeling — the SHAPE label is the typical instance's.
- **[pilot] plain `git push` is `situation-normal`** (new refs, no
  rewrite; rule 3's irreversibility is not triggered by publishing new
  commits/tags).

## Key=value write targets (delta 4, Amy 2026-08-24)

`dd of=/dev/sda`, `mkfs.* /dev/X`, `wipefs`, `blkdiscard`, `sgdisk`:
the write target lives inside a key=value or positional argv word, not
in a redirect. Label by the target class exactly as a redirect would be
labeled — raw device or system path → `data-critical`; a scratch file →
by v9 rules 2/10/14. The shape key won't separate these (same flags,
different target), so these shapes are expected `mixed` → split by
target class.

## Output per shape

```json
{"shape": "<key>", "label": "informative|situation-normal|data-critical|undecidable|mixed",
 "split": "<only for mixed: the split axis, e.g. 'target class'>",
 "contested": true|false}
```

`contested` marks rows where the rubric itself is unsettled (v9's
lockfile-install convention) — they gate nothing until ruled.

## Process (order is load-bearing)

1. **Gold pilot** (~10 shapes, chosen to include likely rubric breaks):
   Amy adjudicates first; her labels are gold.
2. Blind families label the same pilot (DeepSeek bulk per the spend
   posture; no family sees another's answers, none sees gold).
3. Compare: unanimous-wrong against gold = FIX THE RUBRIC and re-pilot;
   2-1 splits = the design working (`measure-disagreement-dont-declare-it`:
   escalate is what annotators disagree about, not what an author
   flags). Only then bulk-label the top 400.
4. Precision + floor come from the labeled sample joined to recorded
   scores; tau is set from target precision, not inherited.
