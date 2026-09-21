#!/usr/bin/env python3
"""Tests for kaish_plan.py — plan-first clause extraction (v10 slice 1).

Run:  python3 lfm2d/hooks/test_kaish_plan.py

Plain script, no pytest (none on this machine). Exits non-zero on failure.
Requires the real `kaish` binary — these are contract tests against the
parser the hook shells out to, and a stub would pin the wrong thing. The
floor baseline build is kaish 0.16.0 b27ea4dd; a behavior change here on
a later kaish is signal, not test rot.

WHY THESE CASES
---------------
The clause text these produce is the v10 training unit (PLAN.md: "the
canonical simple command as kaish's plan renders it"), so what is pinned
here is the training-serving contract: canonical quoting from the plan's
own `plain` values, redirect operators+targets present (an `echo x >
/etc/hosts` is data-critical because of the target), heredoc bodies
absent (they are data, and 68% are interpreter source), pipelines split
into members (the v10 reversal of the clause_split rule, deliberate),
and every failure shape falling back loudly instead of guessing.
"""
import os
import shutil
import stat
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaish_plan  # noqa: E402
from kaish_plan import is_fd_dup, plan_clauses  # noqa: E402


@contextmanager
def fake_kaish(stdout: str, exit_code: int = 0):
    """Swap KAISH_BIN for a shim that prints canned plan JSON — the way to
    feed plan_clauses a plan shape the installed kaish doesn't produce
    (yet). The never-raises contract is about FUTURE kaish output, so it
    can only be tested against shapes we invent."""
    with tempfile.TemporaryDirectory() as d:
        shim = Path(d) / 'kaish'
        shim.write_text('#!/bin/sh\ncat <<\'SHIM_EOF\'\n' + stdout + '\nSHIM_EOF\n'
                        + f'exit {exit_code}\n')
        shim.chmod(shim.stat().st_mode | stat.S_IXUSR)
        old = kaish_plan.KAISH_BIN
        kaish_plan.KAISH_BIN = str(shim)
        try:
            yield
        finally:
            kaish_plan.KAISH_BIN = old

assert shutil.which('kaish'), (
    'these are contract tests against the real kaish binary; '
    'install kaish (floor baseline 0.16.0 b27ea4dd) before running'
)

FAILURES = []


def check(label, got, expected):
    ok = got == expected
    print(f'{"ok " if ok else "FAIL"}  {label}')
    if not ok:
        print(f'      got:      {got!r}')
        print(f'      expected: {expected!r}')
        FAILURES.append(label)


def texts(res):
    return [c['text'] for c in res['clauses']]


def main():
    # -- the unit: one simple command survives verbatim
    r = plan_clauses('cat README.md')
    check('simple command ok', r['ok'], True)
    check('simple command text', texts(r), ['cat README.md'])

    # -- canonical quoting comes from the plan, not from us
    r = plan_clauses('echo "hi there"')
    check('quoting canonicalized', texts(r), ["echo 'hi there'"])

    # -- pipelines split into members: the deliberate v10 reversal of
    # clause_split's no-pipeline rule (see module docstring)
    r = plan_clauses('cat f.txt | grep pattern')
    check('pipeline splits', texts(r), ['cat f.txt', 'grep pattern'])

    # -- and_chain + redirects: operators and targets are shape, kept
    r = plan_clauses('cd /tmp && cargo test 2>&1 | tail -5')
    check('chain with redirects', texts(r), ['cd /tmp', 'cargo test 2>&1', 'tail -5'])

    # -- the redirect target is the severity signal, never dropped
    r = plan_clauses('echo hi > /etc/hosts')
    check('write redirect target kept', texts(r), ['echo hi > /etc/hosts'])

    # -- heredoc: operator+delimiter stay, body never appears, kind tagged
    r = plan_clauses('python3 - <<PYEOF\nprint(1)\nPYEOF')
    check('heredoc body stripped', texts(r), ['python3 - <<PYEOF'])
    check('heredoc tag', r['clauses'][0]['heredoc'],
          {'delimiter': 'PYEOF', 'literal': False, 'kind': 'interpreter'})

    r = plan_clauses("cat > notes.txt <<'EOF'\nsome text\nEOF")
    check('file heredoc kind', r['clauses'][0]['heredoc'],
          {'delimiter': 'EOF', 'literal': True, 'kind': 'file'})

    # -- a substitution's inner command is scored on its own
    r = plan_clauses('echo "a $(hostname) b"')
    check('substitution inner command', texts(r),
          ['echo "a $(hostname) b"', 'hostname'])

    # -- pure assignment: statement rendering, parity with the old path
    r = plan_clauses('X=1')
    check('assignment renders', texts(r), ['X=1'])

    # -- multiple statements flatten in order
    r = plan_clauses('ls; pwd && whoami')
    check('statements flatten', texts(r), ['ls', 'pwd', 'whoami'])
    check('statement count', r['statement_count'], 2)

    # -- the renderer's identity rides on every success: kaish's canonical
    # rendering IS the scored text and it changes between releases
    check('kaish version present', r['kaish_version'].startswith('kaish '), True)

    # -- failure shapes: named, never raised
    r = plan_clauses('echo === step ===')
    check('unquoted adjacency fails closed', r['ok'], False)
    check('parse error named', r['error'], 'parse')

    r = plan_clauses('')
    check('empty command fails closed', (r['ok'], r['error']), (False, 'empty'))

    old = kaish_plan.KAISH_BIN
    kaish_plan.KAISH_BIN = '/nonexistent/kaish'
    try:
        r = plan_clauses('ls')
        check('missing binary named', (r['ok'], r['error']), (False, 'kaish_missing'))
    finally:
        kaish_plan.KAISH_BIN = old

    # -- a redirect to a FILE named `null`: the fd-dup placeholder ("2>&1"
    # carries target "null") must be recognized by the KIND, never by the
    # target's spelling — kaibo's review caught the string-match version
    # dropping this target, and the target is the severity signal
    r = plan_clauses('echo hi > null')
    check('file named null keeps its target', texts(r), ['echo hi > null'])

    # -- the never-raises contract, against plan shapes the INSTALLED
    # kaish doesn't emit (yet): the extraction walks a structure owned by
    # a separately-released binary, and a shape surprise must fall back,
    # never raise — an exception here would crash the hook before it
    # emits the regex decision (kaibo review 2026-08-24, critical)
    with fake_kaish('{"statements":[{"index":0,"plan":{"rendered":"x y",'
                    '"commands":[{"name":"x","args":[{"plain":null}]}]}}]}'):
        r = plan_clauses('x y')
        check('null plain falls back to statement rendering',
              (r['ok'], texts(r), r['clauses'][0]['stmt_fallback']),
              (True, ['x y'], True))

    with fake_kaish('null'):
        r = plan_clauses('ls')
        check('non-dict top-level JSON fails closed',
              (r['ok'], r['error'].startswith('extract:')), (False, True))

    with fake_kaish('{"errors":[42]}'):
        r = plan_clauses('ls')
        check('malformed errors array fails closed', r['ok'], False)

    with fake_kaish('{"statements":[{"index":0,"plan":{"rendered":"r","commands":[42]}}]}'):
        r = plan_clauses('ls')
        check('non-dict command fails closed',
              (r['ok'], r['error'].startswith('extract:')), (False, True))

    # -- structured command facts: what a static check needs so it does not
    # have to re-parse the clause text it was just handed. The escalation
    # cascade (kaish plan -> static checks -> classifier -> static checks on
    # classifier data -> adjudicator) makes its fourth stage out of these:
    # `echo restored` scoring data-critical is dismissible ONLY if the caller
    # can see that the verb is `echo` AND that nothing redirects. Text alone
    # cannot carry that safely -- `echo restored` and `echo restored >
    # /etc/shadow` differ by a redirect, and matching on the rendered string
    # is the prose-reading failure this module exists to avoid.
    r = plan_clauses('cat README.md')
    c = r['clauses'][0]
    check('facts: name', c['name'], 'cat')
    check('facts: args', c['args'], ['README.md'])
    check('facts: no redirects', c['redirects'], [])

    r = plan_clauses('echo hi > /etc/hosts')
    c = r['clauses'][0]
    check('facts: redirect name', c['name'], 'echo')
    check('facts: redirect args', c['args'], ['hi'])
    check('facts: redirect captured structurally',
          c['redirects'], [{'kind': '>', 'target': '/etc/hosts'}])

    # An fd-dup carries its destination INSIDE the kind, and kaish emits a
    # placeholder target for it. Passing that placeholder through would hand
    # stage 4 a redirect that looks like it writes a file named `null` --
    # same discriminator `_render_command` uses, applied to the facts.
    r = plan_clauses('make 2>&1')
    c = r['clauses'][0]
    check('facts: fd-dup target is not a filename',
          [x['target'] for x in c['redirects']], [None])
    check('facts: fd-dup kind kept', [x['kind'] for x in c['redirects']], ['2>&1'])

    # A real file named `null` must NOT be confused with the placeholder.
    r = plan_clauses('echo hi > null')
    c = r['clauses'][0]
    check('facts: real file named null survives',
          c['redirects'], [{'kind': '>', 'target': 'null'}])

    # `&>` has an `&` in its kind and is NOT an fd-dup: it writes stdout
    # and stderr to a file. "Any `&`" dropped its target from both the
    # facts and the rendered clause text.
    r = plan_clauses('make &> build.log')
    c = r['clauses'][0]
    check('facts: &> keeps its file target',
          c['redirects'], [{'kind': '&>', 'target': 'build.log'}])
    check('render: &> keeps its file target', c['text'], 'make &> build.log')
    for kind, dup in (('2>&1', True), ('1>&2', True), ('&>', False), ('>', False),
                      ('2>', False), ('>>', False), ('<', False), ('', False), (None, False)):
        check(f'is_fd_dup({kind!r})', is_fd_dup(kind), dup)

    # Every row carries the keys, including the shapes that have no command
    # to describe -- a consumer must never need a KeyError guard.
    for cmd in ('cat README.md', 'X=1', 'ls | wc -l', 'cat <<EOF\nbody\nEOF'):
        r = plan_clauses(cmd)
        if not r.get('ok'):
            check(f'facts: keys present for {cmd!r} (plan ok)', r.get('ok'), True)
            continue
        missing = [k for row in r['clauses'] for k in ('name', 'args', 'redirects')
                   if k not in row]
        check(f'facts: all keys present for {cmd!r}', missing, [])

    # The pure-assignment row has no command, so its facts are empty rather
    # than absent or guessed.
    r = plan_clauses('X=1')
    if r.get('ok'):
        c = r['clauses'][0]
        check('facts: assignment has no verb', (c['name'], c['args'], c['redirects']),
              (None, [], []))

    print()
    if FAILURES:
        print(f'FAILED: {len(FAILURES)} case(s): {FAILURES}')
        return 1
    print('ALL OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
