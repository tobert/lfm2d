# Adjudicator prompt harnesses

The scorers behind [`docs/lfm25-prompt-experiments.md`](../../../docs/lfm25-prompt-experiments.md).
A metric ships with its scorer, so these live here rather than in a scratch
directory — every number in that document is reproducible from this directory.

## Running them

Most of them drive a llama.cpp server holding the same GGUF the daemon uses
(`LFM2.5-8B-A1B-Q5_K_M`), default `http://127.0.0.1:2031`.

`verdict_eval.py` is the exception and the one to reach for now: it spawns OUR
daemon from `--binary`, one process per arm, and measures on our stack. The same
GGUF gives different distributions on ROCm, CPU and llama.cpp, so a llama.cpp
number is a cross-check and never the result.

```
export LFM2D_EVAL_OUT=~/somewhere/outside/this/repo
python3 ste_experiment.py                       # the whole STE bracket
python3 ste_experiment.py --variants ste,anti   # or named arms
```

**Output directories never default into the repo.** Runs write per-row verdicts,
which are corpus rows, and corpora are not committed here. With no
`LFM2D_EVAL_OUT` and no `--out` the scripts exit with a message rather than
picking somewhere for you.

| env | meaning |
|---|---|
| `LFM2D_EVAL_OUT` | where runs write. Required. Keep it outside the repo. |
| `LFM2D_TRAINING_DIR` | eval splits (default `training/v10`) |
| `KAISH` | kaish binary (default: `$PATH`, then `~/bin/kaish`) |

## What each one produced

| script | the numbers it made |
|---|---|
| `holdout_eval.py` | the shared base: fact extraction from man pages and the kaish plan, prompt rendering, `summarize`. Every other harness imports it. Its facts changed on 2026-09-21 (F7, below). |
| `preamble.py` + `valf_preamble_eval.py` | the cached-preamble length sweep. data-critical 61 → 58 → 56 → 48 as the preamble grows, with penalty controls at both ends. |
| `severity_dist.py` | the four-way distribution at the *scaffolded* severity slot, and the margin split between data-critical hits (0.891) and misses (0.480). |
| `bare_clause.py` | the no-instruction ablation, and the in-set mass that shows two of its three arms measured nothing at all. |
| `writes_field.py` | the `writes` boolean arms. Showed the model answers true 77% of the time regardless, which relocated the whole problem. |
| `clause_shape.py` | parse-only structural classification, no model. The 79% / 4% / 7% separation the LLM could not reproduce. |
| `frame_and_clinical.py` | the frame arms and the counts / undo / scores arms. **Do not cite the frame arms** — they carry two known confounds; see the experiment doc. |
| `shipped_order.py` | severity-first enforcement, and the always-informative collapse it produces. |
| `ste_experiment.py` | the STE bracket. Prompts are constants in the file and each rendered system prompt is sha-hashed into its results. |
| `prompt_anatomy.py` | token-level collision maps, and the no-op-clause prior that ranks situation-normal recall correctly across every arm. |
| `shell_writing.py` | the bash/kaish writing baseline. Plans candidates before running them, refuses verbs outside an allowlist, and executes kaish under `--overlay` so writes are virtual. |
| `sonnet_build_sample.py` + `sonnet_score.py` | the blind larger-model control. The sample is stratified and its key is written separately from the shards. |
| `facts_ab.py` | two `verdict_eval` runs that differ only in the facts they sent, with its own control: an unchanged input that answers differently fails the run. F7's numbers below. |
| `verdict_eval.py` | the allow / ask / review arms on our own daemon, per row: the report, the raw top-k at the verdict's first token with the mass on the verdict words beside it, the forced-step count, and **the bytes sent and generated verbatim** so everything downstream is a replay. `../examine/` reads its runs. |

## F7: the facts block changed (2026-09-21)

`build_facts` no longer states an fd duplication (`2>&1`); describes every
clause of a pipeline or chain (`Clause i: …`), in the fallback path too; adds
one mechanical line naming paths outside the project (home, root, system,
device, temp, remote, unexpanded variable, `../`), silent inside it; unwraps
`sudo -u x`, `env A=1`, `nice -n 10`; and trims flag documentation, stated and
counted, before any clause is lost. Tests: `test_build_facts.py`.

**Every harness that rebuilds facts now renders different input for about 43%
of val_F** (317 of 733 rows). A number made before this date is not reproduced
by re-running its script today; `verdict_eval` runs replay their recorded bytes
and are unaffected.

Measured with `verdict_eval.py`, enum prompt `c9777385…`, val_F as a SMOKE
CHECK (never a scorecard), same binary, baseline at `573c4ba`:

| facts | flagged of 76 gold ask | false alarms of 657 gold allow | precision at val_F's mix |
|---|---|---|---|
| before (`573c4ba`) | 14 | 20 | 0.41 |
| first cut (`ab32c13`) | 16 | 11 | 0.59 |
| as shipped (`0cf0d0d`) | 14 | 12 | 0.54 |

Every comparison carried its control: unchanged inputs answered
byte-identically (416, 718 and 412 rows), and the baseline reproduced the
09-19 run on all 733 rows. **Read it as: false alarms roughly halve, recall
does not move.** The first cut's +2 catches went away when 15 more rows'
facts changed (wrapper pages documenting only their own flags), which is what
noise at this size looks like.

What carries the drop: rows that lost the fd-dup line cleared 14 false alarms
and gained 1 (first cut; 9 of the 14 had written "descriptor" into `effect`).
Rows touched only by the location and clause lines were churn: +7/−5 catches,
+5/−1 false alarms. `scope` follows the location line only partly: home rows
`scope=home` 2 → 22 of 63, system 3 → 9 of 12, devices `system` 2 → 1 of 16
(`a device` maps onto no scope value).

## Two rules these encode

**Prompts are constants, never read from a shipped artifact at runtime.**
`holdout_eval.py` builds its system prompt by reading
`lfm2d/prompts/shell-severity-json-v1.json` at import, so editing that file
mid-campaign silently moved a baseline and cost a control. `ste_experiment.py`
freezes its prompts instead and hashes each one into the results, so a prompt
that changes between runs shows up as a changed hash rather than as a mystery.

**Aggregates only.** These print counts, recalls and confusion matrices. None of
them print row text. If you add one that does, it does not belong in this repo.
