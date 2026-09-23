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

## Runtime spec registration (proposed design)

Today the menu is fixed at boot: `--opinion-spec` paths, loaded in
`Adjudicator::load` (`lfm2d/src/adjudicator.rs`), with each spec's name
taken from its file stem. `Handle.menu` is an `Arc<Vec<SpecMenuEntry>>`,
set once with `with_menu`.

Nothing in `LoadedSpec::load` needs a file. It takes a parsed `PromptSpec`,
and every check it runs is a load-time refusal: the grammar compiles, the
options tokenize on their own after the slot, and the opinion block splits.
Registration reuses it unchanged.

**The wire:**

| call | what it does |
|---|---|
| `PUT /v1/opinion/specs/{name}` | Body is the spec JSON. On the worker thread it runs `LoadedSpec::load`: render, compile the grammar, prefill the prefix. It returns the menu entry with its `snapshot_id`. `200` if a spec with identical content was already there (a no-op), `201` if new or replaced, `422` with the load-time refusal text if the spec cannot be served. |
| `DELETE /v1/opinion/specs/{name}` | Frees the spec's prefix state and its described cache. |
| `GET /v1/opinion/specs` | Unchanged: the menu, read at runtime (invariant 11). |

**Contract points** (these become invariants in `docs/integration.md`):

- **A name can change meaning, a `snapshot_id` cannot.** Replacing a spec
  under the same name gives a new `snapshot_id` and drops that spec's
  described cache. Invariant 8 already tells consumers to refit when the
  `snapshot_id` changes.
- **Requests can pin the version they expect.** An optional
  `snapshot_id` on `/v1/opinion` returns `409` if the loaded spec has
  moved. This closes the race where a request is checked against the old
  menu, then queued, then served by the new spec.
- **Registered specs live in memory only** (proposed). A restart forgets
  them. The consumer re-registers when it gets "no loaded spec" or on its
  own start. The source of truth then stays with the owner, and lfm2d holds
  no state that could drift from it.
- **A capacity limit** (`--opinion-spec-capacity`): every spec holds a
  resident prefix state, so registering past the limit is refused. Replacing
  a spec never counts against it.
- The spec menu still never comes from request text, so "spec menu never
  request text" holds. Registration is a separate write on its own route,
  and an opinion request only names a spec.
- **Trust:** the tailnet is the boundary, the same as every other route.
  Registration is a write that changes what other consumers read, so each
  registration gets its own span and log line, recording the name, the
  snapshot_id and the source address.

**Open:** `/v1/adjudicate` always serves `specs[0]`, which comes from the
boot-time `--adjudicator-prompt`. To be a clean System 1, adjudication
should also name a spec, and escalation should resume from that spec's
described state. It then follows that no spec at all is needed at boot.

**kaijutsu side:** at start and on "no loaded spec", it PUTs each spec from
its own tree. It records the returned `snapshot_id` beside every decision.
It sends `snapshot_id` on requests so that a spec swapped under it fails
loudly instead of quietly changing what the thresholds mean.

## Inventory and disposition (proposed; Amy to rule per row)

"stays" means it stays in lfm2d. "→ kaijutsu" and "→ ktd" (kaish-training-data)
mean it moves. "drop" means it is deleted; its history stays in lfm2d's git.

| path | what it is | proposed |
|---|---|---|
| `src/` trunk, embedding, colbert, sequence/token classification, routing, config, labels | LFM2 encoder runtime | stays |
| `src/cascade.rs`, `tests/cascade.rs`, `/v1/cascade` | v6 severity rank then router: a composite of shell clauses | → kaijutsu as logic over `/v1/classify` + `/v1/route`, or drop |
| `tests/severity_ladder.rs`, `tests/clause_routing.rs` | shell-specific checkpoint tests | → ktd, or drop |
| `/v1/classify`, `/v1/route`, `/v1/spans*`, `/embed` | general head endpoints | stays. Which checkpoints get served is deploy config |
| the shell severity checkpoint | an output of training | built in ktd, deployed by config |
| `lfm2d/hooks/` | advisory hook, kaish_plan, clause_split, stage4 | → kaijutsu (`kj/hook_gate.rs`, `kj/plan_clauses.rs` already overlap) |
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
3. **Stand up `~/src/kaish-training-data`** by filter-repo. It stays local
   until Amy decides on a remote.
4. **kaijutsu takes the consumer side**: spec registration client,
   `build_facts`, the hook, and the shell evals. Repoint `gate.toml`.
5. **Remove shell material from lfm2d**: code, tests, prompts, docs.
   Rewrite the READMEs, CLAUDE.md and AGENTS.md.
6. **Delete this doc.** The kaiseki rename can ride here; see the memory
   note about copying the memory dir first.

## Open questions for Amy

- Should registered specs be memory-only with the consumer re-registering
  (proposed), or kept on disk in a spool directory?
- Should adjudication name its spec, so no spec is needed at boot?
- `src/cascade.rs` and `/v1/cascade`: move to kaijutsu, or drop?
- Does the `lfm2d-1` encoder pod keep serving the shell severity head
  from a checkpoint built in ktd, or does that head retire?
- Does ktd get a remote (private), and when?
