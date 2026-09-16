#!/usr/bin/env python3
"""What is the shell snippet worth on its own, with no instructions at all?

Amy, 2026-09-16: "what if we put the shell snippets into a pass alone with no
instructions? what do we get in the numbers?"

Ablation: strip the system prompt, the rubric thought and the per-clause facts.
Feed the clause and nothing else, anchor at `{"severity": "` and read the same
four-way distribution the scaffolded harness reads. One forward pass per row, so
the whole of val_F is a couple of minutes.

That makes it directly comparable to p0 (the facts-nothink baseline, 482/733)
and to the trivial always-informative floor (419/733). The gap between bare and
p0 is what the whole prompt apparatus is actually buying.

A rejected approach, recorded so nobody burns an afternoon on it: forcing the
clause through a GBNF literal to harvest per-token logprobs gives numbers that
look like a familiarity measure and are not one. GBNF constrains bytes, so the
model walks a character-level path -- 'git branch' came out as g|it| |b|ranch --
and the logprobs belong to that unnatural path. `ls -la` scored -8.44 mean and
the nonsense `zqx --frobnicate /dev/hyperbole` scored -9.41: no separation,
because both were forced off the canonical tokenization. llama.cpp's
/v1/completions `echo` is also not honoured on b9820 -- it generates instead of
echoing -- so prompt-token logprobs need incremental prefix scoring, not this.
"""
import os
import argparse, os, json, math, sys
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


import holdout_eval as H  # noqa: E402


STEM = {268: 'informative', 91: 'situation-normal',
        5911: 'data-critical', 855: 'undecidable'}
ANCHOR = '{"severity": "'


def dist(prompt):
    r = H.post('/completion', dict(H.SAMPLING, prompt=prompt, n_predict=1, n_probs=40))
    d = {}
    for p in r['completion_probabilities'][0]['top_logprobs']:
        lbl = STEM.get(p['id'])
        if lbl:
            d[lbl] = d.get(lbl, 0.0) + math.exp(p['logprob'])
    return d


def variants(cmd):
    """Increasing amounts of frame, so we can see what each layer buys."""
    return {
        # nothing at all: the snippet, then the anchor
        'bare': cmd + '\n' + ANCHOR,
        # same, but inside the chat template the model was tuned on
        'bare-chat': H.render('', cmd) + ANCHOR,
        # the full scaffolded prompt minus the generated fields, for reference
        'system-only': H.render(H.JSON_SYSTEM, cmd) + ANCHOR,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path,
                    default=TRAINING / 'val_F.jsonl')
    ap.add_argument('--out', type=Path,
                    default=None, help='output path (default: $LFM2D_EVAL_OUT/bare_clause.json)')
    ap.add_argument('--limit', type=int)
    a = ap.parse_args()
    a.out = eval_out(a.out)

    rows = [json.loads(l) for l in a.data.read_text().splitlines() if l.strip()]
    if a.limit:
        rows = rows[:a.limit]

    results = defaultdict(list)
    for r in rows:
        for name, prompt in variants(r['text']).items():
            d = dist(prompt)
            top = max(d.items(), key=lambda kv: kv[1])[0] if d else None
            results[name].append(dict(text=r['text'], label=r['label'], top=top,
                                      dist=d, mass=sum(d.values())))

    a.out.write_text(json.dumps(results, indent=1))
    print('%-13s %-6s %-6s %-26s %s' % ('variant', 'right', 'acc', 'per-label recall', 'p50 label mass'))
    for name, rs in results.items():
        right = sum(x['top'] == x['label'] for x in rs)
        conf = defaultdict(Counter)
        for x in rs:
            conf[x['label']][str(x['top'])] += 1
        per = ' '.join('%s=%d/%d' % (l[:3], conf[l][l], sum(conf[l].values()))
                       for l in ['informative', 'situation-normal', 'data-critical'])
        masses = sorted(x['mass'] for x in rs)
        print('%-13s %-6d %-6.3f %-26s %.3f' % (name, right, right / len(rs), per,
                                                masses[len(masses) // 2]))
        print('    predicted: %s' % dict(Counter(x['top'] for x in rs)))


if __name__ == '__main__':
    main()
