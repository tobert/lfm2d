# Field requests

**Status: design in progress.** Amy is still working out the shape and we will
iterate toward it. This records the decisions we have made and the measurements
that forced them, so the next revision argues with evidence instead of memory.
Expect it to be messy while we learn.

## The idea

A consumer does not ask lfm2d for a verdict. It asks for **fields**, and lfm2d
fills in the values.

```
request:  I want { writes, targets, reversibility, severity, route, credentials }
response: one record, one value per field, each carrying where it came from
```

Some fields are answered by static analysis, some by a model, some are derived
from other fields. The consumer does not have to know which, and the mix will
change underneath it as we learn. That is the point: today `writes` is best
answered by the kaish parser and worst answered by the LLM, and we only know
that because we measured it — that finding should be able to change a provider
without changing a consumer.

## Vocabulary

Used precisely throughout; these are not interchangeable.

- **field** — one named value a consumer can ask for (`writes`, `severity`).
- **field request** — the ordered set of fields a consumer asks for. Order is
  load-bearing for model-filled fields; see below.
- **provider** — the thing that produces a field's value. Three kinds:
  **static** (the kaish plan, a lookup table), **model** (an encoder head, the
  LLM), **derived** (a pure function of other fields).
- **provenance** — which provider answered, which weights/binary version, and
  what it cost. Per field, never per record.
- **record** — the assembled result: values, provenance, and the distributions
  behind them.

Providers are a plugin boundary. Fields are the interface. The record is the
wire format, and `docs/integration.md` remains the consumer contract — a field
that ships gets a numbered invariant there.

## Decisions, and what forced them

### 1. Provenance is per field, and static is not privileged

A record mixes static and model-driven values freely. Static providers get the
same provenance and confidence treatment as models — they are not trusted more
for being deterministic.

*Forced by:* a hand-written "does this write" rule (a static provider) was wrong
on 88 of 419 informative rows in val_F — `gh pr view`, `aws ec2
describe-volumes`, `curl -I`, `sleep 30`, `/usr/bin/find` unbasenamed, and
`mkfs.ext4 -n`, which is a dry run. Deterministic is not the same as correct.

### 2. Field order is semantic, not cosmetic

For model-filled fields the request is an **ordered** list, and reordering
changes values. A consumer that reorders has asked a different question.

*Forced by:* reading `severity` with no fields in front of it reproduced the
full answer on 17 of 40 data-critical rows; reading it after
`effect`/`scope`/`reversibility`/`reason` got 40 of 40. The scaffold is what
holds the line on the severe class. Cheaper is not free.

*And it was unenforced until 2026-09-16.* `validate_schema` checks `required`
through a `BTreeSet` and `validate_report` ignores order entirely, so nothing in
the daemon made the order mean anything. Meanwhile
`lfm2d/prompts/shell-severity-json-v1.json` ships `severity` **first** while the
eval harness has always reordered it **last** — so a full day of measurements
described a configuration the daemon does not run. On real weights the shipped
order returns `informative` for a recursive force delete of the filesystem root,
beside its own `effect` field reading *"system-wide destructive change"*.
Constrained decoding makes the order binding, which is what surfaced it. **An
unenforced clause of a contract is one somebody will diverge from silently** — if
a field's order or presence matters, something has to fail when it is wrong.

### 3. Structure never round-trips through prose

A static field's value reaches a model as structure or as a committed field
value — never serialised into an English sentence and pasted into a prompt.

*Forced by:* we handed the model `- redirect > "${D}/f1": truncates and
overwrites that file` and it answered `informative`, whose definition in the
same prompt is "read-only or display-only". Prose facts are ignorable; a field
the model must fill is not. We are currently doing the thing this rule
forbids — the facts block is generated prose — and that is the first thing to
fix.

### 4. The frame is part of the record

Whatever framing text was used is metadata on the record, because framing moves
values.

*Forced by:* the shipped prompt asserts inertness five times — "offline
evaluation: the report records data only", "Never execute commands", "not
authorization to execute anything", the per-row prefix `Command (data only):`,
and in preamble variants "A clause is parsed, never executed". On 94 of the 178
situation-normal rows the model downgraded, its own stated reason denies any
effect at all ("read-only, no side effects"). We may have been reading our own
frame back. An un-auditable prompt is an un-debuggable field.

### 5. Distributions, not argmax — and mass before value

Every model-filled field carries its distribution. Two rules on top:

- **logprobs, not probabilities.** Values saturate (informative at 0.993,
  data-critical at 0.927) and at that end probability floats smear an ordering
  logprobs keep. Live winner margins have been seen at 5e-9.
- **raw mass inside the allowed set is recorded, and low mass means
  *unasked*, not wrong.** With no system prompt the four severity labels held
  ~0.000 mass, and renormalising four near-zero tails produced a confident
  85/733 "accuracy" that measured nothing.

Ranking is within a record. Thresholding across records is not supported and
`docs/integration.md` says so.

### 6. Every field you add is a field the model will fill

**Open: the mechanism below is not settled, and the first version of this
decision over-claimed.** Four added fields were measured and all four cost
data-critical recall against the `p0` baseline of 61/76: integer counts 10/75,
an invented undo command 21/75, 0-10 severity scores 29/75, a `writes` boolean
59/76 (which instead cost 129 informative rows, escalating them to
data-critical). The first draft of this section attributed the damage to
*generative* fields — ones that invent an artifact absent from the input — and
predicted that *descriptive* fields would be safe. The 0-10 scores are
descriptive and lost 32 data-critical rows anyway, so that prediction failed.

A confound runs through all four: each put its new field **first**, ahead of
`effect`/`scope`/`reversibility`/`reason`. Severity stayed last, so decision 2
was satisfied, but the position actually under test never was. The scaffold may
work because it is descriptive *and* because nothing precedes it, and anything
committed ahead of it may anchor the verdict whatever its kind. The untested
cell is a new field placed **after** the scaffold and before severity. Until
that runs, prefer adding nothing.

What does hold, and is worth keeping regardless of mechanism:

A field that asks the model to **invent an artifact not present in the input**
produces an invention, and the invention then anchors every field after it. Such
a field belongs in a static provider, where the artifact is computable, or
nowhere.

A sentinel value is **not** an abstention. If a field offers "NONE" for the
unanswerable case, measure how often that value is actually emitted; near zero
means the field has no way to say "I don't know" and every value it carries is
suspect.

*Forced by:* an `undo_command` field offering `NONE` (nothing can undo this) and
`NOTHING` (nothing to undo) was asked on 75 data-critical rows and used the
sentinel **zero** times. It answered `shred -u ~/.gnupg/secring.gpg` with the
same command back, misspelled; `rm -f ~/.ssh/authorized_keys` with itself;
`git clone` with `false`; and an SQL insert with the string
`situation-normal` — a severity label, in a free-text command field. Severity
then read off that invention and data-critical recall fell to 21/75. Meanwhile
`effect`/`scope`/`reversibility`/`reason`, equally free-text but *descriptive*,
are worth 40/40 against 17/40. A grammar guarantees a field is filled, never
that it is answered.

### 7. What a field is, versus what it may decide

Fusing representations is free; letting a provider silently set a shipped label
is a policy choice. A field may be filled by any provider. Which fields the
daemon is willing to *return as a verdict*, and on whose authority, stays an
explicit decision — not an emergent consequence of a provider swap.

## Inference lives in lfm2d

**Amy's ruling, 2026-09-16:** llama.cpp is demoted to a cheap parity check
against LFM2.5. All inference we depend on runs in our own code, so we can
integrate vertically.

What we already have in `lfm2d/src/adjudicator.rs`: generative LFM2.5 on candle,
greedy sampling, repetition penalty, KV cache, and a `PromptCache` keyed on
**model state checkpoints** rather than llama.cpp's token hashes. `PreparedPrompt`
and `PreparedEvaluation` both carry `logits: Tensor` — the full distribution is
already in hand at every step, just not surfaced. Encoder heads already serve
`/v1/classify`, `/v1/route`, `/v1/spans`, `/v1/cascade`, `/embed`, `/predict`.

Two gaps before the adjudicator can move off llama.cpp:

1. **Constrained decoding.** We have no grammar or JSON-schema decoding. This is
   not optional: unconstrained, the model echoes the schema back as its answer.
   Under a grammar, format failures went to zero.
2. **Surfacing the distribution.** The logits exist; the API does not return
   them. This is field 5 above and it is mostly plumbing.

Known cost of the move: every measurement in
`~/exomemory/lfm2d/lfm25-valf-preamble-2026-09-16/` was taken through llama.cpp,
including the prefix-cache numbers, which will not transfer — our cache is a
different mechanism. Re-measure on our stack rather than porting conclusions.

## Open

- Whether the adjudicator re-grades severity or decides approve/escalate from
  cascade evidence. Unanswered since 2026-09-15 and it shapes the field set.
- Whether val_F's labels are escalate-biased, which a blind panel with a gold
  pilot is meant to settle.
- Which fields are worth asking a model for at all. On val_F the 350M classifier
  scores 93.3% and the LLM 65.8% on the same rows; the LLM's one demonstrated
  edge is data-critical recall among commands that write (80% at a 14% false
  alarm rate).
