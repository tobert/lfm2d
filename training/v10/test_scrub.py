#!/usr/bin/env python3
"""Tests for scrub.py (v10 slice 4: real-text training set, ruling (b)).

Run:  python3 training/v10/test_scrub.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scrub import Numbering, scrub_command, scrub_path, scrub_word  # noqa: E402

FAILURES = []


def check(label, got, expected):
    ok = got == expected
    print(f'{"ok " if ok else "FAIL"}  {label}')
    if not ok:
        print(f'      got:      {got!r}')
        print(f'      expected: {expected!r}')
        FAILURES.append(label)


def cmd(name, *args, redirects=()):
    return {'name': name, 'args': [{'plain': a} for a in args],
            'redirects': [{'kind': k, 'target': {'plain': t}} for k, t in redirects]}


def main():
    n = Numbering()
    # -- paths: identifying components go, form stays
    check('home collapses', scrub_path('/home/atobey/src/kaijutsu/crates/kj-kernel/src/lib.rs', n),
          '~/src/d1/crates/d2/src/lib.rs')
    check('same dir keeps its number', scrub_path('/home/atobey/src/kaijutsu/docs/x.md', n), '~/src/d1/docs/f1.md')
    check('scratchpad collapses', scrub_path(
        '/tmp/claude-1000/-home-atobey-src-kaish/5e12bf1c-4c88-48ee-a280-1eed7f7adf51/scratchpad/rd4/out.log', Numbering()),
        '/tmp/scratch/d1/f1.log')
    check('severity paths stay whole', scrub_path('/dev/sda', Numbering()), '/dev/sda')
    check('etc stays whole', scrub_path('/etc/shadow', Numbering()), '/etc/shadow')
    check('var/log stays whole', scrub_path('/var/log/nginx/access.log', Numbering()), '/var/log/nginx/access.log')
    check('dev null', scrub_path('/dev/null', Numbering()), '/dev/null')
    check('globs are form', scrub_path('crates/kaish-kernel/src/*.rs', Numbering()), 'crates/d1/src/*.rs')
    check('well-known file names survive', scrub_path('crates/foo/Cargo.toml', Numbering()), 'crates/d1/Cargo.toml')
    check('uuid component', scrub_path('/tmp/x/7f3a1b2c-1111-2222-3333-444455556666/run42.log', Numbering()),
          '/tmp/d1/id/f1.log')
    check('trailing slash kept', scrub_path('crates/kaijutsu-app/', Numbering()), 'crates/d1/')
    check('variables inside paths untouched', scrub_path('"${GIX}/src/decode.rs"'.strip('"'), Numbering()),
          '${GIX}/src/f1.rs')
    check('extension kept on scrubbed file', scrub_path('notes/melt-coordination.md', Numbering()), 'notes/f1.md')

    # -- words
    check('flag untouched', scrub_word('--all-targets', Numbering()), '--all-targets')
    check('numeric flag untouched', scrub_word('-1', Numbering()), '-1')
    check('quoted pattern untouched', scrub_word("'^pub enum Action'", Numbering()), "'^pub enum Action'")
    check('quoted path scrubbed inside quotes', scrub_word('"/home/atobey/src/kaish/x.kai"', Numbering()),
          '"~/src/d1/f1.kai"')
    check('--opt=path scrubs the value', scrub_word("--include='*.rs'", Numbering()), "--include='*.rs'")
    check('--opt=path scrubs identifying value', scrub_word('--body-file=/home/atobey/notes/pr.md', Numbering()),
          '--body-file=~/notes/f1.md')
    check('dd-style key=value scrubs value, keeps key', scrub_word('of=/dev/sda', Numbering()), 'of=/dev/sda')
    check('dd-style key=value with identifying value', scrub_word('if=/home/atobey/backup.img', Numbering()),
          'if=~/f1.img')
    check('message value becomes placeholder', scrub_word("'fix: the thing Amy said'", Numbering(), after_message_flag=True),
          "'msg'")
    check('message via = form', scrub_word('--title="Some PR title"', Numbering()), '--title="msg"')
    check('email scrubbed', scrub_word('tobert@gmail.com', Numbering()), 'user@example.com')
    check('tailnet host scrubbed', scrub_word('http://lfm2d-1.taila4abc.ts.net:8088/v1/models', Numbering()),
          'http://host.example:8088/v1/models')
    check('known host kept', scrub_word('https://huggingface.co/api/models/x', Numbering()),
          'https://huggingface.co/api/models/x')
    check('unknown host scrubbed', scrub_word('https://ws-amkfkur9nfa1gu6j.us-east-1.maas.aliyuncs.com/v1', Numbering()),
          'https://host.example/v1')
    check('token redacted', scrub_word('Authorization: Bearer ghp_abcdefghijklmnopqrstuvwxyz1234', Numbering()),
          'Authorization: Bearer <redacted>')
    check('variable untouched', scrub_word('"${TOK}"', Numbering()), '"${TOK}"')
    check('command substitution untouched', scrub_word('$(git rev-parse HEAD)', Numbering()), '$(git rev-parse HEAD)')
    check('hash untouched', scrub_word('b2cdb770', Numbering()), 'b2cdb770')
    check('pr number untouched', scrub_word('370', Numbering()), '370')

    # -- program text: regex-level scrubs only, never component numbering
    check('sed program untouched', scrub_word("'s/:[0-9]*$//'", Numbering()), "'s/:[0-9]*$//'")
    check('sed program with path keeps form, loses home',
          scrub_word("'s#/home/atobey/src/kaish#~/src/kaish#'", Numbering()), "'s#~/src/kaish#~/src/kaish#'")
    check('sed multi-expression untouched', scrub_word("'s/\\bpytool_run\\b/python_mcp_run/g; s/a/b/'", Numbering()),
          "'s/\\bpytool_run\\b/python_mcp_run/g; s/a/b/'")
    check('sh -c body: scratchpad collapsed, rest intact',
          scrub_word("'./target/debug/kaish < /tmp/claude-1000/-home-atobey-src-kaish/633b4f25-bd3c-43ca-821e-49d01ba18c9e/scratchpad/in2.kaish'", Numbering()),
          "'./target/debug/kaish < /tmp/scratch/in2.kaish'")
    check('python payload: home collapsed, username gone',
          scrub_word("'\nimport json\nd=json.load(open(\"/home/atobey/.claude/projects/-home-atobey-src-x/t.json\"))'", Numbering()),
          "'\nimport json\nd=json.load(open(\"~/.claude/projects/-home-user-src-x/t.json\"))'")
    check('bare username in a pattern', scrub_word('atobey', Numbering()), 'user')
    check('long path is not a token', scrub_word('/usr/lib/python3/dist-packages/some/very/long/module/path/name.py', Numbering()),
          '/usr/lib/python3/dist-packages/some/very/long/module/path/name.py')

    # -- whole commands through the serving renderer
    check('render: grep with redirect', scrub_command(cmd(
        'grep', '-rn', "'fn foo'", '"${GIX}/src/lib.rs"', "--include='*.rs'",
        redirects=[('>', '/tmp/claude-1000/-home-atobey-src-x/abc-123/scratchpad/out.log'), ('2>&1', 'null')])),
        'grep -rn \'fn foo\' "${GIX}/src/lib.rs" --include=\'*.rs\' > /tmp/scratch/f1.log 2>&1')
    check('render: commit message placeholder', scrub_command(cmd('git', 'commit', '-q', '-m', "'wip: drop rejection'")),
          "git commit -q -m 'msg'")
    check('render: heredoc marker kept, body absent', scrub_command(
        {'name': 'python3', 'args': [{'plain': '-'}], 'redirects': [{'kind': "<<'PYEOF'", 'target': {'plain': ''}}]}),
        "python3 - <<'PYEOF'")
    check('render: name path scrubbed, basename kept', scrub_command(cmd('/home/atobey/bin/kaijutsu-mcp', '--help')),
          '~/bin/kaijutsu-mcp --help')
    check('render: relative binary kept', scrub_command(cmd('./target/debug/kaish', '--plan', "'ls -la'")),
          "./target/debug/kaish --plan 'ls -la'")
    check('render: distinct files stay distinct', scrub_command(cmd('cp', 'notes/a.md', 'notes/b.md')),
          'cp notes/f1.md notes/f2.md')
    check('render: same stem stays same', scrub_command(cmd('diff', 'foo/x.rs', 'foo/x.rs.bak')),
          'diff d1/f1.rs d1/f2.bak')
    check('render: rm -rf severity path intact', scrub_command(cmd('sudo', 'rm', '-rf', '/var/lib/data')),
          'sudo rm -rf /var/lib/data')
    check('render: unrenderable arg is None', scrub_command({'name': 'x', 'args': [{'nope': 1}]}), None)

    print()
    if FAILURES:
        print(f'FAILED: {len(FAILURES)} case(s): {FAILURES}')
        return 1
    print('ALL OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
