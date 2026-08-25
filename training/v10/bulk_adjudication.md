# v10 slice 3 — bulk labels, splits for adjudication (2026-08-25)

Three families labeled the top-400 plan-rendered shapes blind, with
`labeler_prompt_v10.txt` verbatim: deepseek-v4-pro, gemini-3.5-flash,
qwen3-235b (via kaibo `oneshot`). Votes, consensus and the twelve-gold
canary live in `bulk_votes.json`; raw replies in `bulk/raw/`; the
scorer is `bulk_label.py`. Shape keys only here — real clause text stays
in the local sample (ruling (b)).

| | |
|---|---|
| shapes | 400 (95.8% of 44,293 plan clauses) |
| unanimous | **367** |
| 2-1 | 32 |
| 3-way | 1 |
| consensus mix | 241 informative / 153 situation-normal / 5 data-critical |
| gold canary | deepseek 10/10 · gemini 10/10 · qwen 9/10 (`rm -f` → dc) |
| escape-hatch votes | 2 (`mixed`), both qwen/gemini on verb-blind keys |

The five data-critical consensi: `git rebase`, `git branch -D`,
`cargo publish -p`, `git worktree --force`, and (2-1) the PR-comment
shapes below.

## What the labeling round found in the TOOLING (fixed, re-voted)

The splits diagnosed the shape key before they diagnosed the rubric:

1. **`shape()` dropped flags ≥ 6 chars and scanned only 5 tokens** —
   `--amend`, `--force`, `--hard`, `--delete` were invisible; `git push
   --force` keyed as `git push`. Gemini's `mixed` on `git commit - -F`
   was the honest read of an example set that mixed `--amend` in.
2. **Redirect detection read inside quotes** — `grep -n '^>>>>>>>'`
   keyed as `grep -n >>` (60 clauses across 4 shapes labeled as writes).
   Gemini caught it in its reasoning; DeepSeek and Qwen followed the key.
3. `> /dev/null` counted as an artifact; `"a -> b"` counted as a flag.

Fixed in `soak_shapes.shape()` with pinned tests; the sample was
re-cut (`v10-shape-sample-r2.jsonl`), 138 delta keys labeled (chunks
09–11), and the 56 kept keys whose examples changed were re-voted
(chunks 12–13; a later chunk's vote supersedes). After the re-vote both
`git commit - -F` shapes went unanimous `situation-normal`.

4. **kaibo review (GLM-5.2) then found four more redirect misreads** in
   the rewrite: `echo … >&2` keyed as a write, `cmd &> f` and `cmd 2> f`
   keyed as no write, `>> /dev/null` as an append. Replaced the
   char-before-`>` heuristic with a redirect tokenizer
   (`redirect_class()`, 7 more tests). Re-cut as `v10-shape-sample-r3`:
   **identical key set to r2** — only two shapes' example order moved —
   so the votes stand; `bulk_votes.json` is scored against r3.

## Splits, clustered — Amy's rulings needed

### A. Qwen alone dissents against a pilot ruling (15 shapes) — recommend: consensus stands

`git commit -- -m -q`, `git -C -m -q` (qwen dc: a plain commit),
`bash -c`, `python3 -c >` (qwen dc: stripped carrier → ruled sn),
`rm`, `rm -f` (qwen dc: single-file developer rm → ruled sn; `rm -f` is
gold), `cargo check -p`, `cargo check --all-targets -p`,
`cargo check --tests -p`, `cargo clippy -- --all-targets -D -p` (qwen
inf: rule 16 refined says build/lint = sn), `kubectl -l -n -o` (qwen
inf: verb-blind bias-up, `kubectl -l -n` is gold sn), `probe`,
`git worktree`, `git branch`, `kaish - --plan-file`.

These are the instruction-following gap the pilot predicted (qwen 9/10
on the canary). No rubric question.

### B. `git fetch` / `-q` / `--quiet` (80 clauses) — recommend: situation-normal

qwen: informative (observes the remote). deepseek+gemini: sn (writes
remote-tracking refs — a durable change with an evident backout). Rule
2's letter favors sn; stakes are nil for the gate either way.

### C. Verb-blind `gh`/`git -C` keys whose flags pin a READ verb (11 shapes, ~130 clauses) — decision needed

`gh pr` (view/checks), `gh pr --state`, `gh pr --json --template`,
`gh run --json -q`, `gh run --json --limit -q`, `gh run --branch --jq
--json --limit`, `gh run --branch --json --limit -q`, `git --porcelain
-C`, `-v` (`command -v`), `timeout --plan`, `xargs -l`.

DeepSeek applies the pilot's *verb-blind key biases UP* ruling → sn.
Gemini and Qwen read the examples (all reads) → informative. The
ruling was written for `kubectl -l -n`, whose flags say nothing about
the verb; here `--json`/`--state`/`--porcelain`/`--plan` all but
guarantee a read. **Question: does bias-up apply when the flag set
pins the verb?** Recommend: no — label the typical instance
(informative) when the key's flags are read-only-shaped; keep bias-up
for keys with no verb signal. Either way the gate is unaffected (no
dc at stake).

### D. `gh pr --merge` (40 clauses) — recommend: situation-normal

qwen: dc. A merge creates a merge commit on the remote's base branch;
it is revertable and rewrites no history. Rule 12's dc floor is for
rewrites, not for publishing new commits (plain `git push` is sn by
pilot ruling). Flag only because it lands on a shared branch.

### E. PR comments — a real rule-3 question (3 shapes, 18 clauses)

- `gh pr - --body-file` (= `gh pr comment --body-file -`): deepseek+gemini
  **dc**, qwen sn.
- `timeout --body-file` (same command under `timeout`): same split.
- `gh pr --body` (edit vs comment; verb lost in the key): 3-way
  dc / mixed / sn.

Rule 3 says a message a human will receive is data-critical (it cannot
be unsent). Gemini's reasoning applied it literally. **Question: is a
PR comment data-critical for this gate?** By the rubric's letter yes;
it also matches "always ask before posting to repos we don't own". If
Amy rules dc, `gh pr create` (`gh pr - --body-file --title`, 17
clauses, unanimous sn) should be re-examined for consistency — a PR is
also a notification. If she rules sn, rule 3 needs a carve-out for
reviewable/deletable posts.

### F. The 3-way: `gh pr --body` — needs the key fix in G

### G. Proposed next key fix (not done — changes the unit again)

`shape()` takes ONE subcommand token, so `gh pr comment` / `gh pr
edit` / `gh pr view` share `gh pr …`, `git worktree add` / `remove`
share `git worktree`, and `kubectl … get` / `delete` share `kubectl -l
-n`. The pilot handled this with bias-up. A two-level table
(`gh <group> <verb>`, `git worktree|stash|remote <verb>`, `cargo insta
<verb>`, kubectl verb scan) would let the key see the verb and retire
most of cluster C and all of F — at the cost of one more re-sample +
delta round (est. 30–40 new keys, 6–9 calls). **Amy's call**: fix the
key now, or keep bias-up and move on to tau.

## After the rulings

1. Apply rulings → `bulk_votes.json` gains a `ruling` field per
   adjudicated shape (script, not hand-edit).
2. Build the v10 training set from the labeled shapes: scrub real
   clauses structurally (parser argv), one row per instance.
3. tau from target precision + the pass-through floor
   (`passthrough_gate.py`), then the soak replay gate.
