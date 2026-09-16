# Adjudicator prompt harnesses

The scorers behind [`docs/lfm25-prompt-experiments.md`](../../../docs/lfm25-prompt-experiments.md).
A metric ships with its scorer, so these live here rather than in a scratch
directory — every number in that document is reproducible from this directory.

## Running them

They drive a llama.cpp server holding the same GGUF the daemon uses
(`LFM2.5-8B-A1B-Q5_K_M`), default `http://127.0.0.1:2031`.

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
| `holdout_eval.py` | the shared base: fact extraction from man pages and the kaish plan, prompt rendering, `summarize`. Every other harness imports it. |
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

## Two rules these encode

**Prompts are constants, never read from a shipped artifact at runtime.**
`holdout_eval.py` builds its system prompt by reading
`lfm2d/prompts/shell-severity-json-v1.json` at import, so editing that file
mid-campaign silently moved a baseline and cost a control. `ste_experiment.py`
freezes its prompts instead and hashes each one into the results, so a prompt
that changes between runs shows up as a changed hash rather than as a mystery.

**Aggregates only.** These print counts, recalls and confusion matrices. None of
them print row text. If you add one that does, it does not belong in this repo.
