"""Is the 9.2% situation-normal recall a model failure or a definition clash?

The prompt defines informative as "read-only or display-only" and
situation-normal as "ordinary recoverable developer changes". Both definitions
appear twice -- once in the system prompt, once in the RUBRIC_THOUGHT prefill.
So if the model calls a writing command informative, it is not following a
different rubric, it is not applying the stated criterion.

This sorts every val_F situation-normal row by what the kaish parse says it
DOES, with no model in the loop, and reports recall per bucket.
"""
import os
import json, sys
from collections import Counter, defaultdict
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


sys.path.insert(0, str(REPO / 'lfm2d/hooks'))
import kaish_plan


RUN = eval_out() / 'run/p0.json'

READ_ONLY = {'ls', 'cat', 'grep', 'rg', 'find', 'head', 'tail', 'wc', 'diff',
             'echo', 'printf', 'stat', 'file', 'which', 'ps', 'df', 'du', 'env',
             'pwd', 'date', 'uname', 'whoami', 'id', 'sort', 'uniq', 'cut', 'tr',
             'awk', 'jq', 'basename', 'dirname', 'realpath', 'readlink', 'seq',
             'xargs', 'sed'}
GIT_READ = {'log', 'status', 'diff', 'show', 'branch', 'remote', 'reflog'}


def shape(text):
    plan = kaish_plan.plan_clauses(text)
    if not (plan.get('ok') and plan['clauses'] and plan['clauses'][0].get('name')):
        return 'unparsed'
    c = plan['clauses'][0]
    name, args = c['name'], c['args']
    reds = c.get('redirects') or []
    # a write redirect makes the clause a writer whatever the verb is
    for r in reds:
        kind, tgt = r.get('kind') or '', r.get('target')
        if '>' in kind and '&' not in kind and tgt != '/dev/null':
            return 'write redirect'
    # unwrap wrappers so `timeout 60 cargo test` is judged as cargo
    while name in ('timeout', 'sudo', 'env', 'nohup', 'nice') and args:
        pos = [a for a in args if not a.startswith('-')]
        if not pos:
            break
        i = args.index(pos[0])
        # timeout's first positional is its duration
        i = args.index(pos[1]) if (name == 'timeout' and len(pos) > 1) else i
        name, args = args[i], args[i + 1:]
    if name == 'git':
        sub = next((a for a in args if not a.startswith('-')), None)
        return 'git read-ish (%s)' % sub if sub in GIT_READ else 'git write (%s)' % sub
    if name in READ_ONLY:
        return 'read-only verb'
    return 'other verb (%s)' % name


def main():
    rows = json.loads(RUN.read_text())
    sn = [r for r in rows if r['label'] == 'situation-normal']
    buckets = defaultdict(lambda: Counter())
    for r in sn:
        s = shape(r['text'])
        key = s.split(' (')[0]
        buckets[key][r['severity']] += 1
        buckets[key]['_n'] += 1

    print('val_F situation-normal rows (n=%d), by what the kaish parse says they do\n' % len(sn))
    print('%-18s %-5s %-8s %s' % ('parse shape', 'n', 'sn recall', 'what the model said instead'))
    for key, c in sorted(buckets.items(), key=lambda kv: -kv[1]['_n']):
        n = c['_n']
        got = c['situation-normal']
        other = ', '.join('%s=%d' % (k[:3], v) for k, v in c.most_common()
                          if k not in ('_n', 'situation-normal'))
        print('%-18s %-5d %-8s %s' % (key, n, '%d/%d' % (got, n), other))

    # how many sn rows are things the prompt's own definition would call informative?
    readish = sum(c['_n'] for k, c in buckets.items()
                  if k in ('read-only verb', 'git read-ish'))
    print('\nsn rows whose parse shows no write at all: %d/%d (%.0f%%)'
          % (readish, len(sn), 100 * readish / len(sn)))


if __name__ == '__main__':
    main()
