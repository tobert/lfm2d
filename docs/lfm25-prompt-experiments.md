# LFM2.5 adjudicator: what the prompt is worth

Record of the 2026-09-16 prompt campaign against the LFM2.5-8B-A1B adjudicator.
Every number is val_F (733 rows: 419 informative / 238 situation-normal / 76
data-critical) unless stated. The trivial always-informative floor is **419**.
Raw runs, scripts and per-row verdicts live outside this repo in
`~/exomemory/lfm2d/lfm25-valf-preamble-2026-09-16/`.

Read [`field-requests.md`](field-requests.md) for the design these fed.

## The short version

Prompt style is worth about **58 rows** — more than any other lever tried. But
three separate measurements were confounded before we understood why, and two
of the best-looking numbers earlier in the day turned out to rest on artifacts.
Both facts are in here, because the corrections are the useful part.

## Results

| arm | what it changes | right | inf | sn | dc | dc false alarms | balanced |
|---|---|---|---|---|---|---|---|
| floor | always informative | 419 | 419/419 | 0 | 0 | 0 | .333 |
| `ctl-p0` | the shipped prompt | 436 | 387/419 | 1/238 | 48/76 | 69 | .520 |
| `collision` | `ctl-p0` minus `(data only)` | 446 | 397/419 | 7/238 | 42/76 | 51 | .510 |
| `p0-orig` | `ctl-p0`, severity stated first | 483 | 399/419 | 22/238 | **62/76** | 43 | **.620** |
| `ste` | STE rewrite | 480 | 396/419 | 32/238 | 52/76 | 39 | .588 |
| `ste-labels` | STE + collision-free labels | **494** | **405/419** | **39/238** | 50/76 | — | .596 |
| `anti` | deliberately bad prose | **291** | 134/419 | 96/238 | 61/76 | **188** | .509 |

All arms hold the same facts, the same empty think block, greedy decoding, and
severity emitted last. Prompts are constants in `ste_experiment.py`, and each
rendered system prompt is sha-hashed into its results.

### Prompt style is the largest lever

`ctl-p0` → `ste` is +44 rows with nothing changed but the prose, and it wins
every class: informative +9, situation-normal **+31**, data-critical +4. Adding
collision-free label words on top (`ste-labels`) is another +14, for the highest
raw accuracy measured. The STE prompt is one instruction per sentence, active
voice, the inertness claim made once and aimed at the assistant rather than at
the command, each label word used exactly once, and the rubric written as four
parallel `Grade X when…` clauses instead of one 80-word semicolon chain.

### Making it worse is instructive

`anti` scores **128 rows below the trivial floor** and yet has the second-best
data-critical recall in the campaign (61/76). It gets there by escalating
indiscriminately: 188 false data-critical calls, 25% precision against
`p0-orig`'s 59% at the same recall. **A data-critical recall number without its
false-alarm count beside it is meaningless.**

### Stating an order you do not enforce is worth 47 rows

`ctl-p0` and `p0-orig` differ only in the schema text embedded in the system
prompt: one states severity last, the other states it first. Both *emit* it
last. Stating it first is worth 47 rows, and every class improves. Naming
severity as the first field makes the model attend to it while it writes the
other fields; the grammar still makes it commit at the end.

This was invisible until constrained decoding arrived, because nothing enforced
field order — `validate_schema` checks `required` through a `BTreeSet` and
`validate_report` ignores order. Emitting severity **first** is separately
catastrophic: a degenerate always-informative classifier, 417/733, sn 0/238, dc
0/76.

### Tokens: real, small, local

The user prefix `Command (data only):` raises `data` — the first token of
`data-critical` — **15x** at the severity slot on a no-op clause (0.5364 vs
0.0364). At token level `ctl-p0` and `collision` differ by exactly **one**
occurrence of token 5911, and that one token is worth 6 data-critical rows.

But count tokens, not words: a word-level grep finds `data` 11 times in `anti`
and only **2** are token 5911, the same as `ctl-p0`. So the collision does not
explain `anti`. The effect is real and isolatable at ~6 rows; prose structure
dominates it roughly seven to one.

### A one-call proxy for situation-normal recall

Reading the severity distribution at `{"severity": "` for the no-op clause
`true` gives each prompt's prior before it sees any command. That prior ranks
the arms in **exactly** their order of situation-normal recall over 733 rows:

| arm | sn prior | sn recall |
|---|---|---|
| `ctl-p0` | 0.001 | 1/238 |
| `collision` | 0.001 | 7/238 |
| `p0-orig` | 0.008 | 22/238 |
| `ste` | 0.086 | 32/238 |
| `ste-labels` | 0.117 | 39/238 |

Five for five, one forward pass instead of twelve minutes. It also explains the
collapse mechanically: under the shipped prompt situation-normal has a prior of
**0.001**. The middle rung was never a live hypothesis, so no evidence in the
clause could lift it.

## What went wrong, and what it cost

Three campaigns were confounded before the cause was understood. They are
recorded because the same trap is easy to re-enter.

1. **The harness forked from the shipped artifact.** `holdout_eval.py`
   reordered fields to severity-last with a justifying comment, while the
   shipped prompt said severity-first. A full day of numbers described a
   configuration the daemon never ran.
2. **Editing the prompt file mid-campaign moved the baseline.** `JSON_SYSTEM` is
   built by reading that file at import, so a fix committed at midday silently
   shifted every later run. The `order-eval` control could not reproduce its own
   baseline.
3. **A "frozen" constant dropped 754 characters.** Written specifically to stop
   drift, it replaced the serialised JSON schema with one prose sentence. The
   control collapsed to 419/733, sn 0/238, dc 0/76. Those characters — field
   descriptions and the enum — are load-bearing; without them the model
   degenerates to the floor.

The earlier frame arms (`f-neutral`, `f-active`) carry defects 2 and 3 together
and should not be cited. Their apparent finding — that active framing destroys
data-critical recall — is not supported.

**Standing rule:** diff the harness's *rendered* prompt against the shipped
artifact before trusting a campaign, hash the rendered prompt into every result,
and never edit a prompt artifact while measurements are in flight.

## What has not been tested

- Whether any of this transfers to our own candle runtime. Every number here
  came through llama.cpp.
- The 350M classifier on the same rows for the three-way call; it scores 93.3%
  on val_F, which is 28 points above the best prompt here.
- Whether val_F's labels are themselves biased toward escalation. Three blind
  Sonnet judges agreed with the labels on 124/150 of a balanced sample, which
  argues against it, but that is not the same as a gold pilot.
