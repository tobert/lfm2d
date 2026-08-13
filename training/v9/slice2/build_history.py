#!/usr/bin/env python3
"""Assemble slice2 (history rewrite) incoming file from two Crusoe model
families (qwen3, llama33) plus hand-authored sonnet rows filling gaps:
explicit protected-vs-personal-branch gradient, rule-8 data-position forms,
NL requests, and revert/restore contrasts."""
import json

rows = []

# ---- qwen3-history (crusoe / Qwen3-235B-A22B-Instruct-2507) ----
qwen = [
    ("git push --force origin main", "data-critical", "push", "origin/main", False,
     "Rewrites shared history with no backout stated; force interlock (none here) does not apply."),
    ("git push --force-with-lease origin feature/user-auth", "data-critical", "push", "origin/feature/user-auth", False,
     "Lease interlock guards a concurrent push, not the rewrite; counts same as --force per ruling."),
    ("git push --force-with-lease=main:abc123 origin main", "data-critical", "push", "origin/main", False,
     "Explicit ref:sha lease still rewrites history; the cookie is concurrent-only, not an undo."),
    ("git commit --amend", "data-critical", "commit", "HEAD", False,
     "Text does not say whether the commit was pushed; ambiguous status defaults data-critical per ruling."),
    ("git commit --amend -m 'fixup local commit, never pushed'", "situation-normal", "commit", "HEAD", False,
     "Text explicitly states unpushed; amending unshared history has an evident backout."),
    ("git rebase main", "data-critical", "rebase", "main", False,
     "Rewrites branch history relative to main; no undo stated if any commit was shared."),
    ("git rebase -i HEAD~3", "data-critical", "rebase", "HEAD~3", False,
     "Interactive rebase rewrites history; status unstated, ruling sets a floor to flag."),
    ("git rebase -i HEAD~2 && git push origin feature", "data-critical", "rebase", "origin/feature", False,
     "Rewrite immediately followed by push; compound form removes any doubt the rewrite ships."),
    ("git filter-branch --tree-filter 'rm -f secrets.txt' HEAD~5..HEAD", "data-critical", "filter-branch", "HEAD~5..HEAD", False,
     "Rewrites history to remove a file; irreversible change with no backout in text."),
    ("git filter-repo --path secrets.pem --invert-paths", "data-critical", "filter-repo", "secrets.pem", False,
     "Permanently removes path from every commit; no undo, content is gone forever."),
    ("git reset --hard HEAD~1", "data-critical", "reset", "HEAD~1", False,
     "Discards the last commit and any uncommitted work with no backout stated in text."),
    ("git reset --hard origin/main", "data-critical", "reset", "origin/main", False,
     "Resets local branch to remote tip; any local-only commits are destroyed with no stated recovery."),
    ("git push origin +main", "data-critical", "push", "origin/main", False,
     "The + refspec forces the push exactly like --force; interlock-free rewrite of a shared branch."),
    ("git reflog expire --expire=now --all && git gc --prune=now", "data-critical", "reflog", "reflog", False,
     "Destroys the reflog and prunes unreachable objects, removing the only text-external recovery path."),
    ("git restore .", "situation-normal", "restore", ".", False,
     "Checks out tracked paths from the index; no history rewrite, evident backout via another restore."),
    ("git revert HEAD~2", "situation-normal", "revert", "HEAD~2", False,
     "Adds a new commit undoing the change; backout is the revert itself, no history rewrite."),
    ("git log --grep='force-pushed'", "informative", "log", "commit history", False,
     "Searches commit messages for text; nothing executes, purely observational (rule 8)."),
    ("echo 'we force-pushed to fix the tag'", "informative", "echo", "stdout", False,
     "Destructive words appear as a string payload, not an executed command (rule 8)."),
    ("grep -r 'git push --force' ~/notes/", "informative", "grep", "~/notes/", False,
     "Searches file contents for a pattern; the flag is data being matched, not run (rule 8)."),
    ("git commit --amend && git push --force-with-lease", "data-critical", "commit", "HEAD", False,
     "Amend then lease-push; rewrites and republishes history, interlock doesn't cover the rewrite."),
    ("git rebase origin/main && git push origin feature", "data-critical", "rebase", "origin/main", False,
     "Rebase followed by push; rewrites feature history with no backout stated."),
    ("git push --force origin release/v1.2", "data-critical", "push", "origin/release/v1.2", False,
     "Force-push to a release branch, a clearly shared target; rewrite harm unmitigated."),
    ("git reset --hard HEAD~3", "data-critical", "reset", "HEAD~3", False,
     "Discards three commits; if any were pushed, history is rewritten with nothing surviving."),
    ("git filter-branch --env-filter 'unset GIT_AUTHOR_DATE' HEAD~10", "data-critical", "filter-branch", "HEAD~10", False,
     "Rewrites commit metadata across ten commits; no undo, alters history irreversibly."),
    ("git push origin +feature/new-ui", "data-critical", "push", "origin/feature/new-ui", False,
     "Forced push via +; same risk as --force regardless of the shorthand syntax."),
    ("git commit --amend -m 'fix typo in local-only commit'", "situation-normal", "commit", "HEAD", False,
     "Text states the commit is local-only; amend has an evident backout, no shared history at risk."),
    ("git push --force-with-lease origin main", "data-critical", "push", "origin/main", False,
     "Lease guards a concurrent clobber on main, not the history rewrite itself; scored as --force."),
    ("git revert HEAD~5..HEAD", "situation-normal", "revert", "HEAD~5..HEAD", False,
     "Creates revert commits for the whole range; backout is the reverts themselves, no rewrite."),
    ("git restore --source=HEAD~1 --worktree .", "situation-normal", "restore", ".", False,
     "Restores working tree files from a prior commit; no history change, evident backout."),
]
for t, l, v, r, c, n in qwen:
    rows.append({"text": t, "label": l, "verb": v, "resource": r, "contested": c, "note": n, "author": "qwen3-history"})

# ---- llama33-history (crusoe / meta-llama/Llama-3.3-70B-Instruct) ----
# curated subset: dropped exact duplicates with qwen3, dropped the most
# mechanically-repetitive pairwise flag/delete-branch combos that add no new
# form, kept the shapes that cover distinct family members.
llama = [
    ("git push --force", "data-critical", "push", "current branch", False,
     "No target stated; ambiguous status defaults to data-critical, rewrite harm assumed present."),
    ("git push --force-with-lease=main:HEAD", "data-critical", "push", "main", False,
     "Ref:sha lease form on main; interlock is concurrent-only, rewrite of a shared branch stands."),
    ("git commit --amend -m 'fix typo' && git push", "data-critical", "commit", "HEAD", False,
     "Amend then push publishes the rewrite; no statement the prior commit was unshared."),
    ("git filter-branch --tree-filter 'rm -rf .git' && git push origin +main", "data-critical", "filter-branch", "origin/main", False,
     "Filter-branch rewrite force-pushed to main; largest blast radius in this family, no backout."),
    ("git reset --hard origin/main && git clean -fdx", "data-critical", "reset", "origin/main", False,
     "Hard reset plus untracked-file wipe; local commits and files both gone with no stated recovery."),
    ("git push origin +feature/123 && git push origin +main", "data-critical", "push", "feature/123, main", False,
     "Force-pushes a feature branch then main in one chain; both rewrites are unmitigated."),
    ("git rebase -i HEAD~3 && git commit --amend -m 'fix typo'", "data-critical", "rebase", "HEAD~3", False,
     "Rebase then amend compounds two rewrites in one statement; no unpushed claim in text."),
    ("git push --force main --delete-branch", "data-critical", "push", "main", False,
     "Force-push plus branch deletion on main; rewrite and removal both irreversible from text."),
    ("git rebase -i HEAD~3 --autosquash", "data-critical", "rebase", "HEAD~3", False,
     "Autosquash still performs a full interactive rebase; history rewrite regardless of the flag."),
    ("git restore --source=HEAD~2 file.txt", "situation-normal", "restore", "file.txt", False,
     "Checks out a single file from a prior commit; backout is evident, no history change."),
    ("git revert -n HEAD", "situation-normal", "revert", "HEAD", False,
     "No-commit revert stages the undo; still adds a reversible change, not a rewrite."),
    ("git commit -m 'force-pushed to origin/main earlier today'", "informative", "commit", "commit message", False,
     "Force-push appears only as commit message text; this statement commits normally, nothing rewrites."),
]
for t, l, v, r, c, n in llama:
    rows.append({"text": t, "label": l, "verb": v, "resource": r, "contested": c, "note": n, "author": "llama33-history"})

# ---- sonnet-history: explicit blast-radius gradient, rule-8, NL, contrasts ----
sonnet = [
    # Protected branch, explicit language -> data-critical, note ties to floor + large radius
    ("git push --force origin main, everyone's already pulled today's commits", "data-critical", "push", "origin/main", False,
     "Text states the branch is shared and pulled by others; largest blast radius, no backout stated."),
    ("force push my hotfix branch to main, other people are working off it", "data-critical", "push", "origin/main", False,
     "NL request labeled by the action (rule 5); explicitly shared target, floor applies at full severity."),
    # Personal/unpushed branch, explicit language -> still data-critical (floor) but note acknowledges smaller radius
    ("git push --force-with-lease origin scratch/amy-testing, it's just my personal branch", "data-critical", "push", "origin/scratch/amy-testing", False,
     "Lease guards a concurrent push, not the rewrite itself (rule 12); floor still applies though blast radius is smaller since only this branch is affected."),
    ("git push origin +wip/amy-notes even though nobody else has this branch", "data-critical", "push", "origin/wip/amy-notes", False,
     "Ruling treats history rewrite as a floor regardless of audience; note the radius is contained to one solo branch, unlike a shared-target rewrite."),
    ("git rebase -i HEAD~2 on my local scratch branch that I've never pushed", "situation-normal", "rebase", "HEAD~2", False,
     "Text is explicit the branch was never pushed; rewrite has an evident backout since nothing external depends on it."),
    # Amend shared vs unpushed, explicit
    ("git commit --amend, teammates already pulled this commit yesterday", "data-critical", "commit", "HEAD", False,
     "Text states the commit was pulled by teammates; amending it clobbers their base with no backout shown."),
    ("git commit --amend -m 'typo fix', this commit hasn't left my machine", "situation-normal", "commit", "HEAD", False,
     "Explicitly unpublished commit; amend is normal local git usage with an evident backout."),
    # rebase shared explicit
    ("git rebase main on the release branch that's already been merged and pushed", "data-critical", "rebase", "release branch", False,
     "Text states the branch was already merged and pushed; rebase rewrites history others depend on."),
    # reset --hard on personal branch, still data-critical (no stated backout for local work)
    ("git reset --hard HEAD~1 to drop my last local commit on this throwaway branch", "data-critical", "reset", "HEAD~1", False,
     "Even on a personal branch, reset --hard discards commits with no backout stated in the text (rule 1); small radius, still no recovery path shown."),
    ("git fetch && git reset --hard origin/main", "data-critical", "reset", "origin/main", False,
     "Fetch then hard reset to origin/main; any local-only commits are discarded with no stated recovery."),
    # force-with-lease ref:sha personal branch
    ("git push --force-with-lease=feature/amy-x:9f2a1e origin feature/amy-x, solo branch", "data-critical", "push", "origin/feature/amy-x", False,
     "Explicit ref:sha lease on a solo branch; interlock still only guards concurrent clobber, not the rewrite, so floor applies despite the small radius."),
    # filter-branch/filter-repo on shared repo, explicit
    ("git filter-repo --path credentials.json --invert-paths on the shared team repo", "data-critical", "filter-repo", "credentials.json", False,
     "Text states the repo is shared; rewriting published history to strip a file has no backout for others already synced."),
    ("git filter-branch --force --index-filter 'git rm --cached secrets.env' HEAD", "data-critical", "filter-branch", "HEAD", False,
     "Force flag disables filter-branch's own safety check; history rewrite proceeds with no stated undo."),
    # reflog-adjacent variant
    ("git reflog expire --expire-unreachable=now --all && git gc --aggressive", "data-critical", "reflog", "reflog", False,
     "Expires unreachable reflog entries and aggressively GCs; destroys the last recovery path for any prior rewrite."),
    # rule 8 additional forms
    ("grep -rn -- '--force' deploy/scripts/*.sh", "informative", "grep", "deploy/scripts/*.sh", False,
     "Searches script files for the flag text; nothing is executed, a pure read (rule 8)."),
    ("curl -d '{\"note\":\"remember: never git push --force to main\"}' https://wiki.internal/notes", "informative", "curl", "https://wiki.internal/notes", False,
     "Force-push text is a payload string in a note-taking request; nothing runs it (rule 8)."),
    ("# TODO: document why we had to force-push origin/main last sprint", "informative", "comment", "n/a", False,
     "A source comment mentioning a past force-push; comment text does not execute anything (rule 8)."),
    # revert/restore contrast, incl an informative dry read
    ("git diff main...feature to preview what a rebase would change", "informative", "diff", "main...feature", False,
     "Pure read comparing branches; no rebase is performed, nothing is written or rewritten."),
    ("git restore --staged config.yaml", "situation-normal", "restore", "config.yaml", False,
     "Unstages a file; index change only, evident backout by re-adding, no history touched."),
    ("git status to check if my last commit was already pushed before amending", "informative", "status", "working tree", False,
     "Read-only status check; nothing is amended or rewritten yet, purely observational."),
    # NL contrast, non-rewriting
    ("please amend my last commit, I haven't pushed it anywhere yet", "situation-normal", "commit", "HEAD", False,
     "NL request labeled by the action (rule 5); text is explicit the commit is unpublished, evident backout."),
    ("can you revert the last three commits on main instead of resetting", "situation-normal", "revert", "main", False,
     "NL request for a revert, which adds undoing commits rather than rewriting history; evident backout."),
    # more compound forms to raise compound share toward ~1/3
    ("git fetch origin && git rebase origin/main && git push --force origin feature", "data-critical", "rebase", "origin/feature", False,
     "Chain rewrites feature history against main then force-publishes it; no backout stated for the rewrite."),
    ("git add -A && git commit --amend --no-edit && git push --force-with-lease", "data-critical", "commit", "HEAD", False,
     "Stages, amends, then lease-pushes; ambiguous pushed status defaults data-critical, lease covers concurrency only."),
    ("git checkout -b safety-backup && git reset --hard origin/main", "situation-normal", "reset", "origin/main", False,
     "A backup branch is created before the hard reset; rule 7 — the statement carries its own backout."),
    ("git tag pre-rebase-backup && git rebase -i HEAD~5", "situation-normal", "rebase", "HEAD~5", False,
     "Tag captures the pre-rewrite state in the text itself; rule 7 backout makes this situation-normal."),
    ("cat deploy.sh | grep 'push --force'", "informative", "grep", "deploy.sh", False,
     "Pipes a script to grep for the flag text; nothing executes the force-push, pure read (rule 8)."),
    ("history | grep 'reset --hard'", "informative", "grep", "shell history", False,
     "Searches shell history for a past command string; reading history does not re-run it (rule 8)."),
    ("git branch -m old-name new-name && git push --force origin new-name", "data-critical", "push", "origin/new-name", False,
     "Rename then force-push publishes an overwrite of new-name on the remote; no backout stated."),
    ("git commit --amend && git push -f origin main; echo done", "data-critical", "push", "origin/main", False,
     "Amend then shorthand force-push to main; semicolon chain still executes the rewrite unconditionally."),
    ("git reset --hard origin/main && rm -rf node_modules && npm install", "data-critical", "reset", "origin/main", False,
     "The reinstall is derived-data noise (rule 10); the reset still discards local commits with no backout, so the chain stays data-critical."),
    ("git filter-repo --force --path old_secrets/ --invert-paths", "data-critical", "filter-repo", "old_secrets/", False,
     "Force flag plus history-rewriting filter-repo; strips a path from all commits with no undo."),
    ("git commit --amend -m 'update' on a commit already shared via the open PR", "data-critical", "commit", "HEAD", False,
     "Text states the commit is shared via an open PR; amending it invalidates reviewers' diffs with no backout."),
    ("git push --force-with-lease origin main right after the teammate's PR was approved", "data-critical", "push", "origin/main", False,
     "Protected branch, explicitly just-approved by a teammate; lease guards concurrency, not the rewrite (rule 12)."),
    ("git fetch && git reset --hard origin/main && npm run build", "data-critical", "reset", "origin/main", False,
     "Rebuild step is unrelated derived output; the preceding hard reset still discards local commits with no stated backout."),
]
for t, l, v, r, c, n in sonnet:
    rows.append({"text": t, "label": l, "verb": v, "resource": r, "contested": c, "note": n, "author": "sonnet-history"})

# ---- dedupe by normalized text, validate word count ----
seen = set()
out = []
for r in rows:
    norm = " ".join(r["text"].lower().split())
    wc = len(r["text"].split())
    if norm in seen:
        print("DUP skipped:", r["text"])
        continue
    if not (1 <= wc <= 40):
        print("BAD WORDCOUNT skipped:", wc, r["text"])
        continue
    seen.add(norm)
    out.append(r)

with open("/home/atobey/src/lfm2d/training/v9/slice2/incoming/history.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")

print(f"wrote {len(out)} rows")
