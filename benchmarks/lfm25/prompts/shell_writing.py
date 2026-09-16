#!/usr/bin/env python3
"""Baseline: can LFM2.5-8B-A1B write bash? can it write kaish?

Amy, 2026-09-16: "lil side quest, can this model write bash or kaish code? how
well does it do? as a baseline"

kaish diverges from bash by ONE rule, per Amy's 2026-08-23 ruling in
training/v10/bash_to_kaish.tsv: "nothing adjacent is joined, quote to join", no
exceptions. So `ping 10.0.0.1`, `git diff HEAD~1`, `x=~/.cache/foo` and
`cat a > $DIR/out.txt` all fail to plan, and their quoted twins all plan. That
makes a clean baseline -- a single mechanical rule the model has certainly never
seen in training, on top of a syntax it knows well.

Three conditions per task:
  bash        write bash
  kaish-zero  write kaish, told nothing about the dialect
  kaish-spec  write kaish, given the one rule and three worked examples

Scored on two independent axes, so a syntax failure is never confused with a
wrong answer (the lesson of truncation-is-not-a-wrong-answer):
  PARSES   bash via `bash -n`; kaish via `kaish --plan` exit 0
  CORRECT  executed, stdout compared to the expected string

Execution safety, dogfooding our own tools rather than trusting the model:
  - every candidate is planned FIRST with `kaish --plan`, which executes nothing
  - the verbs the plan reports must all be in ALLOWED, or we refuse to run it
    and score it as `refused` -- a distinct outcome, not a failure
  - kaish runs under `--overlay`, where writes are virtual
  - both run with cwd and HOME inside a throwaway temp dir, under a timeout
"""
import os, shutil
import argparse, os, json, os, shutil, subprocess, sys, tempfile, time
from collections import Counter
from pathlib import Path

# --- repo-relative paths -------------------------------------------------
# Run outputs hold per-row verdicts, i.e. corpus rows, and corpora do not live
# in this repo. So an output directory is never defaulted into the tree: set
# LFM2D_EVAL_OUT, or pass --out. Failing loudly beats writing rows into git.
REPO = Path(__file__).resolve().parents[3]
TRAINING = Path(os.environ.get('LFM2D_TRAINING_DIR', REPO / 'training/v10'))


def eval_out(explicit=None):
    if explicit:
        return Path(explicit)
    env = os.environ.get('LFM2D_EVAL_OUT')
    if not env:
        raise SystemExit(
            'set LFM2D_EVAL_OUT (or pass --out) to a directory OUTSIDE this '
            'repo: these runs write per-row verdicts and corpora are not '
            'committed here')
    return Path(env)
# -------------------------------------------------------------------------


import holdout_eval as H  # noqa: E402


KAISH = os.environ.get('KAISH', shutil.which('kaish') or str(Path.home() / 'bin/kaish'))

ALLOWED = {'echo', 'printf', 'cat', 'ls', 'head', 'tail', 'wc', 'sort', 'uniq',
           'cut', 'tr', 'grep', 'sed', 'awk', 'basename', 'dirname', 'seq',
           'test', 'true', 'false', 'touch', 'mkdir', 'cp', 'mv', 'realpath',
           'expr', 'rev', 'paste', 'tee', 'yes', 'nl', 'tac', 'pwd',
           '[', '[[', 'xargs', 'read', 'set', 'local', 'if', 'then', 'fi'}

RULE = """kaish is a shell, close to bash, with a few deliberate differences.

1. Nothing adjacent is joined. A variable written next to a suffix does not fuse
   into one word -- quote to join.
       bash  cat a > $DIR/out.txt        kaish  cat a > "$DIR/out.txt"
       bash  cp f $FILE.bak              kaish  cp f "$FILE.bak"
2. There is no single-bracket test. `[ ]` is a list literal. Use `test` or `[[ ]]`.
       bash  [ -f x ] && echo yes        kaish  test -f x && echo yes
3. There is no brace group. Use if/then/fi.
       bash  cmd && { echo a; echo b; }  kaish  if cmd; then echo a; echo b; fi
4. No backticks (use $( )), no `until` (use `while !`), no subshell `( )`,
   no `=~` regex test (pipe to grep), no $'...' ANSI-C strings (use printf).
5. A regex or glob metacharacter in an unquoted word is an error -- quote it.
       bash  sed s/^a/b/ f               kaish  sed 's/^a/b/' f

Everything else -- pipes, redirects, quoting, command names -- is as in bash."""

# (id, natural-language task, expected stdout, divergent?)
# `divergent` marks a task whose natural bash idiom is one kaish REFUSES, so it
# separates "writes shell" from "writes kaish". Re-derived 2026-09-16 against
# kaish 0.17.2: the bare-literal rows of bash_to_kaish.tsv (IPs, version
# strings, colon-slash words, leading-dot paths) have been ABSORBED and no
# longer discriminate, so they are gone from this set. The 21 rows that still
# diverge are what the divergent tasks are built from.
TASKS = [
    # --- control: both dialects handle these the same way ---
    ('cat', 'print the contents of the file data.txt', 'alpha\nbeta\ngamma', False),
    ('count', 'count the number of lines in data.txt, printing only the number', '3', False),
    ('second', 'print only the second line of data.txt', 'beta', False),
    ('sortrev', 'print the lines of data.txt sorted in reverse alphabetical order',
     'gamma\nbeta\nalpha', False),
    ('upper', 'print the contents of data.txt converted to upper case',
     'ALPHA\nBETA\nGAMMA', False),
    ('col2', 'print the second column of the tab-separated file table.tsv', 'b\ne', False),
    ('firstcol', 'print the first column of table.tsv', 'a\nd', False),
    ('firstword', 'print the first word of the first line of data.txt', 'alpha', False),
    # --- divergent: the natural bash idiom does not plan in kaish ---
    ('joinvar', 'the variable D holds a directory name; print the string formed by '
     'D followed by a slash and then out.txt', 'work/out.txt', True),
    ('suffix', 'the variable F holds a file name; print F with the suffix .bak '
     'appended', 'notes.bak', True),
    ('redirjoin', 'the variable D holds a directory name; write the word hello into '
     'the file named out.txt inside that directory, then print that file', 'hello', True),
    ('existstest', 'print the word yes if the file data.txt exists', 'yes', True),
    ('twolines', 'if the file data.txt exists, print alpha on one line and beta on '
     'the next', 'alpha\nbeta', True),
    ('regexmatch', 'print the word match if the first line of data.txt begins with '
     'the letter a', 'match', True),
    ('caret', 'use sed to replace the line beginning with a in data.txt with the '
     'letter X, printing the result', 'X\nbeta\ngamma', True),
    ('tabsep', 'print the letter a and the letter b separated by a tab character',
     'a\tb', True),
]

SEED = {
    'data.txt': 'alpha\nbeta\ngamma\n',
    'table.tsv': 'a\tb\tc\nd\te\tf\n',
}
SEED_ENV = {'D': 'work', 'F': 'notes'}


def sandbox():
    d = Path(tempfile.mkdtemp(prefix='shellbase-'))
    for name, body in SEED.items():
        (d / name).write_text(body)
    (d / 'work').mkdir()
    return d


def plan_verbs(src):
    """Verbs kaish says this source would run. Executes nothing."""
    r = subprocess.run([KAISH, '--plan-file', '-'], input=src, capture_output=True,
                       text=True, timeout=10)
    try:
        j = json.loads(r.stdout)
    except Exception:
        return None, 'unparseable plan output'
    if 'errors' in j:
        return None, j['errors'][0].get('message', 'plan error')
    verbs = []
    for st in j.get('statements', []):
        for c in st['plan']['commands']:
            verbs.append(c['name'])
    return verbs, None


def parses(src, lang):
    if lang == 'kaish':
        verbs, err = plan_verbs(src)
        return verbs is not None, err
    r = subprocess.run(['bash', '-n'], input=src, capture_output=True, text=True,
                       timeout=10)
    return r.returncode == 0, (r.stderr.strip() or None)


def run(src, lang):
    """Execute in a throwaway sandbox. Returns (stdout, outcome)."""
    verbs, err = plan_verbs(src)
    if verbs is None:
        # bash-only syntax kaish refuses: fall back to naming the first words
        verbs = [line.split()[0] for line in src.splitlines()
                 if line.strip() and not line.strip().startswith('#')]
    bad = sorted({v for v in verbs if v.split('/')[-1] not in ALLOWED})
    if bad:
        return None, 'refused:' + ','.join(bad[:3])
    d = sandbox()
    try:
        env = dict(os.environ, HOME=str(d), **SEED_ENV)
        cmd = ([KAISH, '--overlay', '-c', src] if lang == 'kaish'
               else ['bash', '-c', src])
        r = subprocess.run(cmd, cwd=d, env=env, capture_output=True, text=True,
                           timeout=15)
        return r.stdout.strip(), ('ok' if r.returncode == 0 else 'exit%d' % r.returncode)
    except subprocess.TimeoutExpired:
        return None, 'timeout'
    finally:
        shutil.rmtree(d, ignore_errors=True)


SYSTEMS = {
    'bash': 'You write bash. Reply with the command only: no explanation, no '
            'Markdown fences, no comments.',
    'kaish-zero': 'You write kaish, a shell language. Reply with the command '
                  'only: no explanation, no Markdown fences, no comments.',
    'kaish-spec': RULE + '\n\nReply with the kaish command only: no explanation, '
                         'no Markdown fences, no comments.',
}


def generate(system, task):
    """Returns (src, status). src is None when the model never answered.

    The first cut gave phase 1 only 400 tokens. The kaish-spec prompt is longer,
    so the model thought past the budget, never closed </think>, and its
    REASONING got scored as source -- 3/16 'parse failures' that were really one
    budget failure. A truncated think is not a wrong answer
    (truncation-is-not-a-wrong-answer), so it gets its own status and is never
    counted as bad code.
    """
    user = (task + '\n\nAvailable: the file data.txt, the file table.tsv, the '
            'environment variables D and F.')
    prompt = H.render(system, user).replace('Command (data only): ', '')
    r = H.post('/completion', dict(H.SAMPLING, prompt=prompt, n_predict=2048,
                                   stop=['</think>']))
    body, status = r['content'], 'ok'
    if r.get('stop_type') == 'word':
        # it closed the think block; the answer is what follows
        r2 = H.post('/completion', dict(H.SAMPLING, prompt=prompt + body + '</think>\n',
                                        n_predict=300))
        body = r2['content']
        if r2.get('stop_type') == 'limit':
            status = 'answer-truncated'
    elif '<think>' in body or r.get('stop_type') == 'limit':
        return None, 'think-truncated'
    src = body.strip()
    if src.startswith('```'):
        src = '\n'.join(l for l in src.splitlines() if not l.startswith('```'))
    src = src.strip()
    return (src, status) if src else (None, 'empty')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path,
                    default=None, help='output path (default: $LFM2D_EVAL_OUT/shell_writing.json)')
    a = ap.parse_args()
    a.out = eval_out(a.out)

    results = {}
    for cond, system in SYSTEMS.items():
        lang = 'bash' if cond == 'bash' else 'kaish'
        rows, t0 = [], time.time()
        for tid, task, expected, divergent in TASKS:
            src, gstatus = generate(system, task)
            if src is None:
                rows.append(dict(id=tid, src=None, parses=None, parse_err=None,
                                 stdout=None, outcome=gstatus, divergent=divergent,
                                 correct=False, expected=expected))
                continue
            ok_parse, perr = parses(src, lang)
            out, outcome = (run(src, lang) if ok_parse else (None, 'not-parsed'))
            rows.append(dict(id=tid, src=src, parses=ok_parse, parse_err=perr,
                             stdout=out, outcome=outcome, divergent=divergent,
                             correct=(out == expected), expected=expected))
        results[cond] = rows

        def tally(sel):
            # answered = the model emitted source at all. Parse and correctness
            # are scored over answered rows only, so a budget failure never
            # masquerades as bad code.
            rs = [r for r in rows if sel(r)]
            ans = [r for r in rs if r['parses'] is not None]
            return '%d/%d %d/%d %d/%d' % (len(ans), len(rs),
                                          sum(r['parses'] for r in ans), len(ans),
                                          sum(r['correct'] for r in ans), len(ans))
        print('%-11s answered/parses/correct  all %-11s | control %-11s | '
              'divergent %-11s | outcomes %s (%.0fs)'
              % (cond, tally(lambda r: True), tally(lambda r: not r['divergent']),
                 tally(lambda r: r['divergent']),
                 dict(Counter(r['outcome'] for r in rows)), time.time() - t0), flush=True)
    a.out.write_text(json.dumps(results, indent=1))
    print('\nraw ->', a.out)


if __name__ == '__main__':
    main()
