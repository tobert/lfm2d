#!/usr/bin/env python3
"""Tests for stage4.py — static checks on classifier data.

Run:  python3 lfm2d/hooks/test_stage4.py

Plain script, no pytest (none on this machine). Exits non-zero on failure.
Requires the real `kaish` binary: stage 4 reads the plan's structured
facts, so every case here is planned by the parser the hook actually
shells out to. Hand-built fact dicts would pin a shape kaish may not
produce — the two cases that ARE hand-built are shapes kaish cannot
produce (a missing verb, an unknown redirect kind).

WHY THESE CASES
---------------
Each was checked to fail under a plausible wrong implementation first:

- matching the rendered text instead of the facts dismisses
  `echo restored > /etc/shadow` (case 2), the exact failure
  `_command_facts` exists to prevent;
- treating any `>` in a redirect kind as a write refuses `echo hi 2>&1`
  (case 3), and treating any `&` as harmless dismisses a real
  `2> /tmp/log` (case 4);
- checking only the git subcommand dismisses `git branch -D` (case 7);
- matching a verb by basename dismisses `/bin/cat` and
  `./contrib/install.sh` (cases 17, 18) — a path-named verb is not the
  coreutil it is spelled like;
- forgetting that `sort -o` and `find -delete` write dismisses two verbs
  whose ordinary form is a pure read (cases 13-16);
- ignoring the stage-2 argument dismisses a guard DENIAL (case 19),
  which is the one thing the cascade's ratchet forbids.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from kaish_plan import plan_clauses  # noqa: E402
from stage4 import review  # noqa: E402

FAILURES = []


def check(label, got, expected):
    ok = got == expected
    print(f'{"ok " if ok else "FAIL"}  {label}')
    if not ok:
        print(f'      got:      {got!r}')
        print(f'      expected: {expected!r}')
        FAILURES.append(label)


def facts(cmd, index=0):
    """The planned facts for one clause of `cmd`, or None if kaish did not
    plan it — a parser that cannot read the case makes the case unprovable,
    which is reported, never quietly passed."""
    r = plan_clauses(cmd)
    if not r.get('ok') or index >= len(r['clauses']):
        return None
    return r['clauses'][index]


def dismissed(cmd, index=0, guard='allow'):
    """(dismissible, reason) for one clause of `cmd`."""
    f = facts(cmd, index)
    if f is None:
        return ('unplanned', 'unplanned')
    v = review(f, guard_decision=guard)
    return (v.dismissible, v.reason)


def check_dismissed(cmd, reason, **kw):
    check(f'dismisses {cmd!r}', dismissed(cmd, **kw), (True, reason))


def check_survives(cmd, reason, **kw):
    check(f'survives  {cmd!r}', dismissed(cmd, **kw), (False, reason))


def main():
    # -- 1-4: the canonical case from the cascade design, and the redirect
    # discriminator. `echo restored` and `echo restored > /etc/shadow`
    # differ by a redirect and by nothing else in the text.
    check_dismissed('echo restored', 'read_only_verb:echo')
    check_survives('echo restored > /etc/shadow', 'write_redirect:>')
    check_dismissed('echo hi 2>&1', 'read_only_verb:echo')
    check_survives('echo hi 2> /tmp/log', 'write_redirect:2>')

    # `&>` sends stdout AND stderr to a FILE. Its `&` comes before the `>`,
    # so "any `&` in the kind is an fd-dup" read it as harmless and
    # dismissed a write to /etc/shadow (found 2026-09-21, live in record
    # mode since stage 4 shipped). The fd-dup is `N>&M` and nothing else.
    check_survives('echo restored &> /etc/shadow', 'write_redirect:&>')
    check_dismissed('echo hi 1>&2', 'read_only_verb:echo')

    # An append is a write, and so is a write to a target that happens to
    # be spelled like a sink: /dev/null is not special-cased, because
    # deciding on a target's SPELLING is the prose-reading failure. If this
    # costs real coverage the report will show it as a `write_redirect` row.
    check_survives('echo hi >> /tmp/log', 'write_redirect:>>')
    check_survives('echo hi 2> /dev/null', 'write_redirect:2>')

    # A read redirect and a heredoc move no data onto the filesystem.
    check_dismissed('cat < README.md', 'read_only_verb:cat')

    # -- 5: the five real `rm` winners in the live window. rm is not listed
    # and must never be: this is precisely what should reach the desk.
    check_survives('rm -f tests/sz.rs', 'verb_not_listed:rm')
    check_survives('rm -rf "${SCRATCH}"', 'verb_not_listed:rm')

    # -- 6-11: git, ten of the twenty-five real data-critical winners.
    check_dismissed('git status --short', 'read_subcommand:git status')
    check_dismissed('git log --oneline -3', 'read_subcommand:git log')
    check_dismissed('git branch -a', 'read_subcommand:git branch')
    check_dismissed('git branch -r', 'read_subcommand:git branch')
    check_dismissed('git branch --merged main', 'read_subcommand:git branch')
    # -D deletes a ref. The subcommand is identical to the three above, so
    # a subcommand-only check gets this wrong.
    check_survives('git branch -D drop-per-call-turn-limits',
                   'forbidden_flag:git branch -D')
    # An operand with no listing flag creates a branch. `git branch` is a
    # strict-allowlist subcommand for exactly this reason.
    check_survives('git branch newbranch', 'operand_not_allowed:git branch')
    # push publishes; merge-tree --write-tree writes objects. Neither verb
    # is a read, and neither is listed.
    check_survives('git push -q origin main', 'subcommand_not_listed:git push')
    check_survives('git merge-tree --write-tree --name-only main other',
                   'subcommand_not_listed:git merge-tree')
    # -- 12: git's value-taking global flags come before the subcommand.
    check_dismissed('git -C d1 status', 'read_subcommand:git status')
    # An unknown leading flag means we cannot locate the subcommand. Fail
    # closed rather than guess which word is the verb.
    check_survives('git --frobnicate status', 'subcommand_unreadable:git')
    check_survives('git', 'no_subcommand:git')

    # -- 13-16: two verbs whose ordinary form is a pure read and whose
    # flags make them write. Both appear in live traffic.
    check_dismissed('find codex-rs/core/templates -type f', 'read_only_verb:find')
    check_survives('find . -delete', 'forbidden_flag:find -delete')
    check_dismissed('sort in', 'read_only_verb:sort')
    check_survives('sort -o out in', 'forbidden_flag:sort -o')
    # The joined spelling of the same flag. Exact-match-only misses this.
    check_survives('sort --output=x in', 'forbidden_flag:sort --output')

    # -- 17-18: a path-named verb. `/bin/cat` may well BE cat; the point is
    # that stage 4 cannot know that, and `./contrib/install.sh` is the real
    # live winner that proves the rule earns its keep.
    check_survives('/bin/cat f', 'verb_is_a_path:/bin/cat')
    check_survives('./contrib/install-codex-app-server-systemd.sh',
                   'verb_is_a_path:./contrib/install-codex-app-server-systemd.sh')

    # -- 19: THE RATCHET. Stages 1-4 may only raise concern; stage 4 may
    # dismiss a classifier verdict, never a stage-2 denial. Passing the
    # guard decision is required by the signature so this cannot be
    # forgotten by a caller.
    #
    # The guard's real decision vocabulary is exactly these four (counted
    # over the live log: allow 52508, soft_deny 194, warn 160, deny 119).
    # `soft_deny` is a control and must never be lowered here. `warn` is
    # NOT a control -- the statement proceeds carrying its warning -- so
    # the classifier verdict is dismissible on a warn row, and the warn
    # itself is untouched by that. Reading the vocabulary as two values
    # gets both of these wrong.
    check_dismissed('echo restored', 'read_only_verb:echo', guard='allow')
    check_survives('echo restored', 'guard_denied:deny', guard='deny')
    check_survives('echo restored', 'guard_denied:soft_deny', guard='soft_deny')
    check_dismissed('echo restored', 'read_only_verb:echo', guard='warn')
    # A decision this module has never heard of is not an allow. A new
    # guard verdict shows up here as a countable reason, not as a silent
    # dismissal.
    check_survives('echo restored', 'guard_unknown:maybe', guard='maybe')
    check_survives('echo restored', 'guard_unknown:None', guard=None)

    # -- 20: unknown verbs survive, and the reason NAMES the verb so the
    # report can rank what is worth a ruling. Both are live winners.
    check_survives('pkill -x mistralrs', 'verb_not_listed:pkill')
    check_survives('rustup target add x', 'subcommand_not_listed:rustup target add')
    check_dismissed('rustup target list --installed',
                    'read_subcommand:rustup target list')

    # -- other real live shapes worth pinning
    check_dismissed('cat cosign.out', 'read_only_verb:cat')
    check_dismissed('set -euo pipefail', 'read_only_verb:set')
    check_dismissed('git worktree list', 'read_subcommand:git worktree list')
    check_survives('git worktree remove wt', 'subcommand_not_listed:git worktree remove')
    # sed can edit in place, so it is not a reader at all -- not even
    # `sed -n`, which would need flag grammar we deliberately do not have.
    check_survives('sed -n 5,12p src/main.rs', 'verb_not_listed:sed')
    # A verb that RUNS another verb is named by the wrapper, and the wrapper
    # is never a read.
    check_survives('sudo -n systemctl restart lfm2d', 'verb_not_listed:sudo')
    check_survives('xargs rm', 'verb_not_listed:xargs')
    check_survives('env FOO=1 rm -rf /tmp/x', 'verb_not_listed:env')

    # -- shapes kaish cannot produce, so they are built by hand: a clause
    # with no simple command (a pure assignment or a statement-level
    # fallback) and a redirect kind we have never seen.
    v = review({'name': None, 'args': [], 'redirects': []}, guard_decision='allow')
    check('no verb at all', (v.dismissible, v.reason), (False, 'no_verb'))
    v = review({'name': 'echo', 'args': ['hi'],
                'redirects': [{'kind': '>%', 'target': 'x'}]},
               guard_decision='allow')
    check('unknown redirect kind fails closed',
          (v.dismissible, v.reason), (False, 'write_redirect:>%'))
    # An unknown kind with no `>` in it must fail closed too: the default is
    # "write" and the read kinds are the enumerated exception, not the other
    # way round (kaibo review 2026-09-21).
    v = review({'name': 'echo', 'args': ['hi'],
                'redirects': [{'kind': '%', 'target': 'x'}]},
               guard_decision='allow')
    check('unknown kind without > fails closed',
          (v.dismissible, v.reason), (False, 'write_redirect:%'))
    for kind in ('<', '<<', '<<-', '<<<'):
        v = review({'name': 'cat', 'args': [], 'redirects': [{'kind': kind, 'target': 'x'}]},
                   guard_decision='allow')
        check(f'read kind {kind} is not a write', (v.dismissible, v.reason),
              (True, 'read_only_verb:cat'))
    # "Never raises" is the contract: a redirect that is not a dict is a
    # shape we cannot read, so it is not dismissible.
    v = review({'name': 'echo', 'args': ['hi'], 'redirects': ['>']},
               guard_decision='allow')
    check('malformed redirect fails closed without raising',
          (v.dismissible, v.reason), (False, 'write_redirect:?'))

    # -- every reason is a non-empty stable slug, both ways. A silent
    # dismissal is the failure mode this whole stage must not have.
    for cmd in ['echo hi', 'rm -rf /', 'git branch -a', 'git push', 'find . -delete']:
        f = facts(cmd)
        if f is None:
            check(f'reason present for {cmd!r}', 'unplanned', 'planned')
            continue
        v = review(f, guard_decision='allow')
        check(f'reason is a slug for {cmd!r}',
              (isinstance(v.reason, str), bool(v.reason), ' ' in v.reason.split(':')[0]),
              (True, True, False))

    print()
    if FAILURES:
        print(f'FAILED: {len(FAILURES)} case(s): {FAILURES}')
        return 1
    print('ALL OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
