#!/usr/bin/env python3
"""Read the severity distribution at the SCAFFOLDED position, for ~free.

The 2026-09-16 one-token probe read severity with no fields in front of it and
found that position is a different animal: on 40 rows the full generation got
right, the bare read reproduced data-critical only 17 times. The effect / scope
/ reversibility / reason fields are load-bearing. So the distribution worth
having is the one at the natural position -- after the model has written those
fields and is about to name the label.

Getting it costs almost nothing now that prefix caching hits on this model:
generate the answer exactly as the harness does, then replay the prompt up to
the literal `"severity": "` that the model itself emitted and read one token
with n_probs. The replay is a cache hit, so it is one forward pass.

Two traps, both found the hard way on llama.cpp b9820:
  - `prob` in completion_probabilities is 0 in this build. Only `logprob` is
    populated. A consumer reading `prob` sees silent zeros, not an error.
  - the label-discriminating token depends on what precedes it. After an
    opening quote the bare stems are in/s/data/und; with the quote merged in
    they are different ids entirely. Anchor on the emitted text, never guess.

This module does not touch holdout_eval.py: that file is the scorer that the
09-15 numbers were produced by and must keep reproducing them.
"""
import os
import json, math, re, sys
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


# First token of each bare label, i.e. immediately after the opening quote.
STEM = {268: 'informative', 91: 'situation-normal',
        5911: 'data-critical', 855: 'undecidable'}

ANCHOR = re.compile(r'"severity"\s*:\s*"')


def _read_dist(prompt):
    r = H.post('/completion', dict(H.SAMPLING, prompt=prompt, n_predict=1, n_probs=20))
    top = r['completion_probabilities'][0]['top_logprobs']
    d = {}
    for p in top:
        lbl = STEM.get(p['id'])
        if lbl:
            d[lbl] = d.get(lbl, 0.0) + math.exp(p['logprob'])
    return d


def answer_with_dist(system, cmd, prefill, sampling=None):
    """The harness's facts-nothink answer, plus the distribution behind it.

    Returns the usual fields with `dist` (label -> prob), `margin` (top minus
    runner-up) and `dist_top`. dist is None if the anchor is not found, which is
    loud on purpose: a silently absent distribution would look like confidence.
    """
    sampling = sampling or H.SAMPLING
    base = H.render(system, cmd) + prefill
    base += '\n</think>\n' if not base.endswith('\n') else '</think>\n'
    r2 = H.post('/completion', dict(sampling, prompt=base, n_predict=600,
                                   json_schema=H.ANSWER_SCHEMA))
    text = r2['content']
    try:
        sev = json.loads(text)['severity']
        status = 'ok'
    except Exception:
        sev, status = None, 'answer-unparsed'

    dist = margin = top = None
    m = ANCHOR.search(text)
    if m:
        # replay up to and including the quote the model actually emitted
        dist = _read_dist(base + text[:m.end()])
        if dist:
            order = sorted(dist.items(), key=lambda kv: -kv[1])
            top = order[0][0]
            margin = order[0][1] - (order[1][1] if len(order) > 1 else 0.0)
    elif status == 'ok':
        status = 'anchor-missing'

    return dict(severity=sev, status=status, dist=dist, dist_top=top,
                margin=margin, answer_tokens=r2['tokens_predicted'])


def main():
    """Compare the scaffolded position against the bare one on sampled rows."""
    import argparse
    from collections import Counter
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path,
                    default=Path(__file__).resolve().parent / 'run/p0.json')
    ap.add_argument('--per-group', type=int, default=40)
    a = ap.parse_args()

    rows = json.loads(a.run.read_text())
    cov = Counter()
    groups = {
        'inf correct': lambda r: r['label'] == 'informative' and r['severity'] == 'informative',
        'sn correct': lambda r: r['label'] == 'situation-normal' and r['severity'] == 'situation-normal',
        'sn missed': lambda r: r['label'] == 'situation-normal' and r['severity'] == 'informative',
        'dc correct': lambda r: r['label'] == 'data-critical' and r['severity'] == 'data-critical',
        'dc missed': lambda r: r['label'] == 'data-critical' and r['severity'] != 'data-critical',
    }
    print('%-12s %-4s %-9s %-24s %-8s %s' %
          ('group', 'n', 'agree', 'true-label rank', 'p50 prob', 'p50 margin'))
    for name, pred in groups.items():
        rs = [r for r in rows if pred(r)][:a.per_group]
        if not rs:
            continue
        ranks, agree, probs, margins, bad = Counter(), 0, [], [], 0
        for r in rs:
            facts = H.build_facts(r['text'], cov)
            res = answer_with_dist(H.JSON_SYSTEM, r['text'],
                                   '<think>\n' + H.RUBRIC_THOUGHT + '\n' + facts)
            if not res['dist']:
                bad += 1
                continue
            order = sorted(res['dist'].items(), key=lambda kv: -kv[1])
            names = [l for l, _ in order]
            ranks[names.index(r['label']) + 1 if r['label'] in names else 0] += 1
            probs.append(res['dist'].get(r['label'], 0.0))
            margins.append(res['margin'])
            agree += (res['dist_top'] == res['severity'])
        probs.sort(); margins.sort()
        print('%-12s %-4d %-9s %-24s %-8.3f %.3f%s' %
              (name, len(rs), '%d/%d' % (agree, len(probs)), dict(sorted(ranks.items())),
               probs[len(probs) // 2], margins[len(margins) // 2],
               '  (%d no-dist)' % bad if bad else ''))


if __name__ == '__main__':
    main()
