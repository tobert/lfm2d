# Plan: lfm2d becomes a clean System 1; shell moves out

A living doc for the move and the cleanup. It is deleted when the work is
done, and its history stays in git. Decisions are marked **ruled** (Amy said
so, quoted) or **proposed** (waiting on Amy).

## Where this is going

Amy, 2026-09-23: "for now I'd like to get lfm2d towards being a clean system1
implementation and start pushing the kaish/shell stuff over to kaijutsu."

- **lfm2d** serves the LFM suite. It carries the encoder heads' runtime and the
  System 1 opinion engine, and the checkpoints and specs it serves come from
  outside it. Nothing in it knows what a shell is.
- **kaijutsu** owns shell judgement: kaish plans, clause facts, the gate, the
  hook, and the shell specs, which it uploads at runtime.
- **~/src/kaish-training-data** (new repo) owns the shell training and
  eval material worth keeping. Amy: "We can drop anything that's not going
  to help with lfm2d's future as a general system1 model built around the
  LFM suite."

## Rulings so far

1. **Specs are registered at runtime by their owner.** Amy: "I think
   kaijutsu should own its specs, and upload them at runtime. should be
   easy for kaijutsu to update it as we iterate without a lot of hassle."
2. **The shell training material gets a new repo.** Amy: "let's make a new
   repo for the shell training stuff, like ~/src/kaish-training-data."
3. **Probe measurement comes first.** It tests whether a mid-layer read is
   worth building as a generic opinion read. Output goes to
   `~/exomemory/lfm2d/lfm25-probe-f9-2026-09-23/`.
4. **This doc lives in `docs/`** until the move is done.

## Runtime spec registration (ruled 2026-09-23)

Amy: "could the id be a content hash so we get some idempotency? whatever
algo we use for kaibo's cas should be fine right?" and "keep specs in memory
for now ... clients should just upload any time they're not sure, and we'll
do the content identity so it's idempotent for free-ish."

Today the menu is fixed at boot: `--opinion-spec` paths, loaded in
`Adjudicator::load` (`lfm2d/src/adjudicator.rs`), with each spec's name taken
from its file stem. `Handle.menu` is an `Arc<Vec<SpecMenuEntry>>`, set once.
`LoadedSpec::load` needs no file: it takes a parsed `PromptSpec`, and every
check it runs is a load-time refusal. Registration reuses it unchanged.

**Identity.** A spec's `id` is the lowercase hex SHA-256 of the exact bytes
uploaded. That is kaibo's CAS digest (`kaibo/src/cas.rs`, `Digest::of_bytes`)
and lfm2d's own `hash.rs`. The hash covers the exact bytes and nothing is
canonicalized first. Parsing into `serde_json::Value` and re-serializing
would sort every object's keys, and a spec's field order is its emission
order (memory: btreemap-sorting-reaches-the-prompt). So two specs that
differ only in field order are different specs and must get different ids.
The cost is that re-formatting a spec's whitespace also gives it a new id,
which does no harm. Boot-time specs get an id the same way, from their file
bytes, and also answer to their file-stem name.

**The wire:**

| call | what it does |
|---|---|
| `POST /v1/opinion/specs` | The body is the spec's bytes. Returns the menu entry with `id` and `snapshot_id`. `200` if that id is already loaded (no work done), `201` if it was loaded now. `400` if the body is not a spec, `422` with the load-time refusal text if the spec cannot be served. |
| `DELETE /v1/opinion/specs/{id}` | Unloads an uploaded spec. Boot-time specs cannot be deleted. |
| `GET /v1/opinion/specs` | Unchanged: the menu, read at runtime (invariant 11). |
| `POST /v1/opinion` | `spec` is an id, or a boot-time name. An unknown spec is `404`, which tells the client to upload it and retry. |
| `POST /v1/adjudicate` | Takes `spec` as well. Amy: "yeah, it takes a spec". Escalation resumes from that spec's described state. Without `spec` it still serves the `--adjudicator-prompt` spec, so existing callers keep working. |

**Memory only**, with a capacity limit for uploaded specs
(`--opinion-spec-capacity`). Past the limit, the least recently used upload
is evicted, and its next request gets a `404` that tells the client to upload
it again. Boot specs are never evicted. Every spec holds a resident prefix
state, which is why there is a limit.

An id cannot change what it means, so no request needs a `409` pin.
`snapshot_id` still tells a consumer when the model under that spec changed,
and invariant 8's rule to refit on a new `snapshot_id` still applies.
Registration is a separate write on its own route, and opinion requests only
name a spec, so "the spec menu never comes from request text" holds. Each
registration and eviction gets its own span and log line, recording the id,
the snapshot_id, the load time and the source address.

**kaijutsu side:** it uploads each spec from its own tree whenever it is
unsure, and always after a `404`. It records `id` and `snapshot_id` beside
every decision.

## Tokenize and probe endpoints (ruled 2026-09-23)

Amy: "if we don't still have a tokenizing endpoint on lfm2d I think we should
still have that. might add a general inference endpoint too so we can use it
to probe the model consistently."

Neither exists today. The nearest things are `/v1/adjudicate`
`distributions` (top-k per generated step, but only under a spec) and the
`lfm25-examine` binary (a cold reader, and not what the daemon runs).

**`POST /v1/tokenize`**: `{model, text, context?}` → `ids`, each token's
string and byte span, and `tokenizer_hash`. `model` is an id from
`/v1/models`, the adjudicator or an encoder head, since they are separate
tokenizers. With `context`, it also returns the tokens `text` takes up
*after* the context, and whether they are stable as a suffix. That is the
same check the spec loader runs on options (see memory
verdict-words-tokenize-in-string-context). It runs on the handler; the
tokenizer is cloned out of the worker, so tokenizing never waits in the
model queue.

**`POST /v1/probe`**: raw inference over exact text, on the daemon's own
stack.
- Input: `text` (exact bytes), or `messages` rendered through the chat
  template with the rendered text returned. Optional `continuations`, each
  teacher-forced and scored the way the opinion read scores options, and
  optional `generate: n` greedy steps.
- Output: top-k at the last position and at each generated step,
  continuation logprobs with raw mass, `cached_tokens`, and an identity
  block (weight/tokenizer hash, backend, dtype, sha256 of the rendered
  input).
- `use_cache: false` for a cold read, because cold and warm schedules
  disagree by about 0.15 nats (memory cold-and-cached-schedules-disagree).
  The response always says which one ran.
- Later: `lens: [layers]` and `routing: true`, moving the examiner's
  instruments into the daemon so probes and production share one stack.
- It is an instrument, not a judgement API: it has no calibration contract,
  and invariants 8–10 do not apply to it. It takes request text by design,
  so it is a separate route from opinion (whose questions come only from
  the menu). Amy: "/v1/probe"; "it can be on by default" (a flag turns
  it off).

## Verdict vocabulary sweep (after registration + probe land)

The discovery pass comes first: `/v1/probe` with a free-string slot, reading
the model's own top-k over about 20 benign and severe commands. Then the
scored arms, each one an uploaded spec, all scored on F9 (AUC, recall at 4
FA, raw mass). The winner is confirmed on a fresh split.

| arm | key | options |
|---|---|---|
| control | `verdict` | allow / ask |
| harness action | `verdict` | allow / block |
| **risk** (Amy: "goes in") | `risk` | low / medium / high; scored as P(low) or an expected value, never a sum of top rungs |
| question in the key | `safe_to_run` | yes / no |
| boolean | `needs_confirmation` | true / false, unquoted (first check that the opinion read supports a non-string choice) |
| traffic light | `status` | green / yellow / red |

## Inventory and disposition (proposed; Amy to rule per row)

"stays" means it stays in lfm2d. "→ kaijutsu" and "→ ktd" (kaish-training-data)
mean it moves. "drop" means it is deleted; its history stays in lfm2d's git.

| path | what it is | proposed |
|---|---|---|
| `src/` trunk, embedding, colbert, sequence/token classification, routing, config, labels | LFM2 encoder runtime | stays |
| `src/cascade.rs`, `tests/cascade.rs`, `/v1/cascade` | v6 severity rank then router: a composite of shell clauses | **drop now** (ruled: "kaijutsu will do it differently, and a lot in .kai scripts") |
| `tests/severity_ladder.rs`, `tests/clause_routing.rs` | shell-specific checkpoint tests | → ktd, or drop |
| `/v1/classify`, `/v1/route`, `/v1/spans*`, `/embed` | general head endpoints | stays. Which checkpoints get served is deploy config |
| the shell severity checkpoint | an output of training | **retire now.** Amy: "the old lfm2d service will stay frozen indefinitely. so we can move on." `lfm2d-1` keeps serving it from its frozen image to the Claude Code hook and kaijutsu `gate.toml`; main stops carrying it |
| `lfm2d/hooks/` | advisory hook, kaish_plan, clause_split, stage4 | **moved to ktd `hooks/`** (`02cf276`, Amy's pick). `~/.claude/settings.json` now runs it from there. The lfm2d copy is deleted at sign-off |
| `lfm2d/prompts/command-verdict-*`, `shell-severity-*` | shell specs | → kaijutsu, registered at runtime |
| `benchmarks/lfm25/results/`, runtime docs `docs/lfm25-*` (kernels, cache, fusion, gqa, prefill) | engine performance record | stays |
| `benchmarks/lfm25/examine/` | lens, routing and knockout tooling | stays (general over slots); its README examples go generic |
| `benchmarks/lfm25/gold/` | F9 rubric, freeze, instruments, slot scores | → ktd |
| `benchmarks/lfm25/prompts/` | 09-16 prompt campaign harnesses, `build_facts` | `build_facts` → kaijutsu; the campaign → ktd, or drop |
| `benchmarks/lfm25/{kaijutsu_shaped,live_opinion}_eval.py` | live-shaped shell evals | → kaijutsu or ktd |
| `docs/lfm25-prompt-experiments.md`, `docs/field-requests.md` | shell prompt findings | → ktd (the method lessons get a short general note in lfm2d) |
| `training/` (v9, v10, coverage_v8, router, kube_*, cascade, recoverability, hf, …) | shell/k8s head training | → ktd; v9 and older → drop unless v10 imports them |
| `demo/` | shell-flavoured acts plus `email-triage-v1` | general acts stay, shell acts → kaijutsu or drop |
| `docs/integration.md` | consumer contract | stays; invariants that name the shell head get rewritten |
| `lfm2d/src/expert_map.rs`, examine binaries | general interpretability | stays |
| `CLAUDE.md`, `AGENTS.md`, READMEs | working context | rewritten last |

**Moving with history:** build ktd with `git filter-repo --path …` on a
fresh clone, so its files keep their log. Then remove the same paths from
lfm2d in a single commit that points to the new repo. Corpora stay where they
are (`~/.local/share/lfm2-training-data/`); they never lived in a repo.

## Sequence

1. **Probe measurement.** Per-depth AUC of the verdict lean against F9 gold.
   Running.
2. **Runtime spec registration in lfm2d**, test first. It includes the
   `409` pin and adjudicate-by-spec-name if Amy agrees. The pod rolls
   promptly once tested.
3. **Stand up `~/src/kaish-training-data`** — DONE 2026-09-23, local only
   (`e42cc06`, no remote). filter-repo copy of every "→ ktd" row plus
   `lfm2d/prompts/`; paths unchanged; 21 scripts still reach into lfm2d by
   path. Nothing removed from lfm2d yet.
4. **kaijutsu takes the consumer side**: spec registration client,
   `build_facts`, the hook, and the shell evals. Repoint `gate.toml`.
5. **Remove shell material from lfm2d**: code, tests, prompts, docs.
   Rewrite the READMEs, CLAUDE.md and AGENTS.md.
5b. **Tokenize + probe endpoints**, after registration merges (both touch
   `Handle`).
6. **Delete this doc.** The kaiseki rename can ride here; see the memory
   note about copying the memory dir first.

## Open questions for Amy

- The severity head is live in two places, so what replaces the advisory
  hook: a kaijutsu gate calling `/v1/opinion`, or retire the hook outright?
- Should `--adjudicator-prompt` become optional, so a daemon can boot with
  no spec and serve only uploads?
- Does ktd get a remote (private), and when?
