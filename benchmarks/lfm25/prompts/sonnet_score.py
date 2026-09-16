#!/usr/bin/env python3
"""Score the bigger-model control against LFM2.5 on the same 150 rows.

Reports three things, in order of how much they decide:

1. the three-way call, and the always-informative floor for this sample
2. `writes` — the structural read LFM2.5 could not do (it answered true 77% of
   the time regardless, including on 73% of read-only rows). The kaish parse is
   the reference here, not a model.
3. data-critical vs situation-normal among the rows the parse says WRITE. That
   is the only axis LFM2.5 contributed anything on: 80% recall at a 14% false
   alarm rate. If a larger model does not beat that, size was never the
   constraint.

LFM2.5's numbers come from the saved p0 run, restricted to these same ids, so
the comparison is on identical rows and not against its val_F aggregate.
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


HERE = Path(__file__).resolve().parent
P0 = eval_out() / 'run/p0.json'
sys.path.insert(0, str(REPO / 'lfm2d/hooks'))
from clause_shape import shape  # parse-only: no model in the loop


LABELS = ['informative', 'situation-normal', 'data-critical']


def load_judges():
    out = {}
    for s in range(3):
        f = HERE / f'out{s}.jsonl'
        if not f.exists():
            sys.exit('missing %s — judge %d has not finished' % (f, s))
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[r['id']] = r
    return out


def report(name, verdict, writes, key, texts):
    n = len(key)
    right = sum(verdict.get(i) == key[i] for i in key)
    conf = defaultdict(Counter)
    for i, lab in key.items():
        conf[lab][str(verdict.get(i))] += 1
    per = ' '.join('%s=%d/%d' % (l[:3], conf[l][l], sum(conf[l].values())) for l in LABELS)
    print('%-12s %3d/%d  %.3f   %s' % (name, right, n, right / n, per))

    if writes:
        agree = sum(1 for i in key if writes.get(i) is not None
                    and writes[i] == (shape(texts[i]).split(' (')[0]
                                      not in ('read-only verb', 'git read-ish')))
        have = sum(1 for i in key if writes.get(i) is not None)
        true_rate = sum(1 for i in key if writes.get(i)) / max(have, 1)
        print('             writes vs parse: %d/%d (%.0f%%)  says-true rate %.0f%%'
              % (agree, have, 100 * agree / max(have, 1), 100 * true_rate))

    writers = [i for i in key if shape(texts[i]).split(' (')[0]
               not in ('read-only verb', 'git read-ish')]
    dc = [i for i in writers if key[i] == 'data-critical']
    sn = [i for i in writers if key[i] == 'situation-normal']
    saydc = lambda ids: sum(1 for i in ids if verdict.get(i) == 'data-critical')
    if dc and sn:
        print('             among %d writers: dc recall %d/%d (%.0f%%)  '
              'sn false alarm %d/%d (%.0f%%)'
              % (len(writers), saydc(dc), len(dc), 100 * saydc(dc) / len(dc),
                 saydc(sn), len(sn), 100 * saydc(sn) / len(sn)))


def main():
    key = json.loads((HERE / 'key.json').read_text())
    texts = {}
    for s in range(3):
        for line in (HERE / f'shard{s}.jsonl').read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                texts[r['id']] = r['command']

    judges = load_judges()
    missing = [i for i in key if i not in judges]
    if missing:
        print('WARNING: %d rows unjudged: %s' % (len(missing), missing[:5]))
    bad = {i: j['severity'] for i, j in judges.items()
           if j.get('severity') not in LABELS + ['undecidable']}
    if bad:
        print('WARNING: %d out-of-vocabulary verdicts: %s' % (len(bad), list(bad.items())[:3]))

    # LFM2.5 on exactly these rows, from the saved p0 run
    p0 = {r['text']: r['severity'] for r in json.loads(P0.read_text())}
    lfm = {i: p0.get(texts[i]) for i in key}
    unseen = sum(1 for i in key if lfm[i] is None)
    if unseen:
        print('note: %d rows absent from the p0 run' % unseen)

    floor = max(Counter(key.values()).values())
    print('sample n=%d  %s' % (len(key), dict(Counter(key.values()))))
    print('always-informative floor: %d/%d (%.3f)\n' % (floor, len(key), floor / len(key)))
    print('%-12s %-9s %-8s %s' % ('judge', 'right', 'acc', 'per-label recall'))
    report('LFM2.5-8B', lfm, None, key, texts)
    report('sonnet', {i: j['severity'] for i, j in judges.items()},
           {i: j.get('writes') for i, j in judges.items()}, key, texts)


if __name__ == '__main__':
    main()
