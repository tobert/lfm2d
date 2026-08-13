# slice 6 — system administration surface, batch 1

First generation batch for PLAN.md slice 6 (+6b). Four Sonnet subagents, one
per sub-slice, each briefed from `BRIEF.md` and the full rubric.

```
python3 training/coverage_v8/validate_v8.py incoming/*.jsonl   # per-file gate
python3 training/v9/build_v9.py                                # cross-slice gate + merge
python3 training/v9/build_v9.py --check                        # regenerates identically
```

**Note (2026-08-13):** the slice-local `merge_slice6.py` / `scorecard.py` /
`slice6.jsonl` were retired once slices 1–3 landed. They could not see
cross-*slice* duplicates, which is where the interesting collisions turned out
to be — slice 2's history rows and this slice's flag ladder independently
produced `git push --force origin main`. `../build_v9.py` supersedes all three
and emits `../v9.jsonl`.

## Result — **328 rows, 0 errors, 0 canary hits**

| file | rows | authors |
|---|---|---|
| `syspaths` | 84 | deepseek 84 |
| `sysverbs` | 83 | gemini-flash 72, sonnet 11 |
| `flagladder` | 78 | sonnet 78 |
| `creds_ext` | 69 | deepseek 39, sonnet 22, **gemini-pro 8** |
| `autoconfirm` | 20 | sonnet 20 *(added after the rule-13 ruling)* |
| **merged** | **328** | 8 author tags |

Merged labels: **182 data-critical (55.5%) · 69 informative (21.0%) ·
77 situation-normal (23.5%)** · 142 distinct verbs · 213 distinct resources ·
7 contested (2.1%).

## Rule 13 — a confirmation prompt is not an interlock

Amy, 2026-08-13, after a blind relabel split 3 rows exactly this way:

> *"`--interactive` isn't a protection, it's a UX surface that can aid humans
> in slowing down, but is nothing to an agent or a `yes | cmd`."*

Applied here: `rm -i draft.md`, `git clean -i`, and bare `terraform destroy`
were flipped **situation-normal → data-critical**. Encoded as rule 13 in
`../labeler_prompt.txt` and in PLAN.md's rulings section.

The ruling also exposed a form the corpus had **never contained** —
`yes | cmd`, which defeats a prompt with no flag at all, and is exactly the
shape an agent produces. `autoconfirm.jsonl` covers it, plus the auto-confirm
flags (`-y`, `--noconfirm`, `--auto-approve`) which are **not aggravating
either**: they remove a prompt that was never protection. The teaching pairs
hold the label constant across prompted / auto-confirmed / `yes`-piped
variants of the same action, and contrast all of them against a *real*
interlock (`git worktree remove`, which refuses; `helm --atomic`, which rolls
back; `--dry-run`, which does not execute).

No file exceeded the 75% single-label share that `../build_v9.py` fails on, so
the benign arms are genuinely present — the point being that if every row
mentioning `/etc` were data-critical we would have taught a *new* vocabulary
shortcut instead of consequence, which is the exact failure slice 6 exists to
fix.

## Cross-file agreement — 5 collisions, **0 label conflicts**

Four agents on adjacent surfaces independently produced the same five texts.
`../build_v9.py` keeps one of each and **refuses to merge on a label
conflict** rather than picking a winner. All five agreed:

| text | label | families that agreed |
|---|---|---|
| `mkfs.ext4 /dev/sda1` | data-critical | sonnet, deepseek |
| `rm -rf /etc` | data-critical | sonnet, deepseek |
| `> /etc/shadow` | data-critical | deepseek, gemini-flash |
| `terraform destroy -auto-approve` | data-critical | sonnet, gemini-flash |
| `terraform plan -destroy` | **informative** | sonnet, gemini-flash |

Unplanned inter-annotator agreement across three model families, 5/5. Small,
but it is free evidence the rubric transfers.

## Model routing actually used

Per Amy's budget guidance: deepseek for bulk, gemini-flash for one slice,
**exactly 2** gemini-pro calls on the subtlest judgements, **no OpenRouter**.

**Local (`lfm25-8b-a1b`) was trialled and FAILED.** Given the 7-arm `rm` flag
ladder with the rubric spelled out, it returned schema-valid JSON and labeled
**every row `data-critical`, including `rm -i` and `rm --interactive=never`** —
i.e. it reproduced precisely the flat, non-monotone response this batch exists
to correct, and would have baked the bug back into the training data. It also
invented `verb` values (`"force-recursive-delete"`) instead of the literal
command verb. Not usable for rubric-sensitive work; untested for low-stakes
phrasing variance. (Gemma was not loaded — the local server had lfm25-8b.)

Because of that, `flagladder.jsonl` is **entirely sonnet-authored** — no
cross-family check on the slice with the subtlest rule (rule 4: which
interlocks count). That file is the best candidate for a blind relabel pass.

## Labels are PROPOSALS

Bulk blind labeling is still gated on Amy's budget decision. Every `label`
here is the generator's proposal, carried with its `author` tag so a later
blind pass can disagree with it row by row.

## What generation surfaced that we did not know

**`rm -rf /` has a built-in interlock.** Verified against the binary: GNU
coreutils 9.11 ships `--preserve-root` as the default, so bare `rm -rf /`
refuses. Under Amy's own guardrail ruling that may make v8's low score for it
*defensible*, and it means one of the severity probes is measuring an
interlock rather than blindness. Written up in
`../severity_probes/README.md` under "CORRECTION 2026-08-13". Both rows that
hit this (`sudo rm -rf /`, `chmod -R 000 /`) are marked `contested: true`
rather than forced.

**`blkdiscard` has no dry-run flag** on this util-linux build — checked
rather than assumed, so the destructive row was paired with `lsblk -f`
instead of a fabricated `--dry-run` twin. Worth repeating as a habit: the
cheapest failure mode in synthetic command data is plausible-looking commands
that cannot actually run.
