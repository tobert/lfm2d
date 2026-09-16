#!/usr/bin/env python3
"""Look at our prompts as token data rather than as prose.

Amy, 2026-09-16: "are there any representations of these prompts we can look at
from a data perspective? maybe build a viz?"

Two views, both grounded in the measured collision (the phrase
`Command (data only):` raises data-critical 15x at the severity slot because
`data` is that label's first token):

COLLISION MAP. Tokenize each rendered prompt and mark every position holding the
exact token that DISCRIMINATES a severity label at the answer slot. For the real
vocabulary that is in=268, s=91, data=5911, und=855; an arm with alternate label
words has its own four. At the answer slot, whichever of those the model emits
IS the verdict, so every earlier occurrence is that token being activated in
context.

Count TOKENS, not words. A word-level grep of the shipped prompt finds `data`
four times; at token level only two of those are token 5911, because the
tokenizer merges the rest into other units. That distinction matters: ctl-p0 and
collision differ by exactly ONE occurrence of token 5911, and that one token is
worth 6 data-critical rows on val_F.

INFLUENCE. Remove one sentence at a time, re-read the four-way distribution at
the severity slot, and record the shift. One forward pass per sentence, so a
whole prompt costs about as much as one row of an eval. This says which
sentences are actually doing work, as opposed to which ones we believe in.

Writes prompt_anatomy.json for the viz; prints a summary.
"""
import os
import argparse, os, json, sys
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


import ste_experiment as S  # the prompts, frozen as constants


URL = 'http://127.0.0.1:2031'
STEM = {268: 'informative', 91: 'situation-normal',
        5911: 'data-critical', 855: 'undecidable'}
ANCHOR = '{"severity": "'
PROBE_CLAUSE = 'true'   # a no-op: any severity signal is coming from the prompt


def post(path, body):
    import urllib.request
    req = urllib.request.Request(URL + path, json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req))


def tokenize(text):
    return post('/tokenize', {'content': text, 'with_pieces': True})['tokens']


def dist(prompt, stem):
    import math
    r = post('/completion', {'prompt': prompt, 'n_predict': 1, 'n_probs': 200,
                             'temperature': 0, 'cache_prompt': True})
    by_id = {p['id']: math.exp(p['logprob'])
             for p in r['completion_probabilities'][0]['top_logprobs']}
    return {name: by_id.get(tid, 0.0) for tid, name in stem.items()}


def sentences(system):
    """Split on newline first, then on '. ' -- STE writes one idea per line."""
    out = []
    for line in system.split('\n'):
        line = line.strip()
        if not line:
            continue
        parts, buf = [], ''
        for chunk in line.split('. '):
            buf = chunk if not buf else buf
            parts.append(chunk)
            buf = ''
        for p in parts:
            p = p.strip()
            if p:
                out.append(p if p.endswith('.') else p + '.')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path,
                    default=None, help='output path (default: $LFM2D_EVAL_OUT/prompt_anatomy.json)')
    ap.add_argument('--no-influence', action='store_true',
                    help='collision map only; skip the ablation passes')
    a = ap.parse_args()
    a.out = eval_out(a.out)

    result = {}
    for name, (prose, prefix, alt, stated) in S.VARIANTS.items():
        # An arm with alternate label words is discriminated by ITS words, not
        # the real ones. Using the real stems on those arms reports all zeros --
        # which is true and useless, since the model is never choosing between
        # them there.
        surface = [alt[l] for l in S.REAL] if alt else list(S.REAL)
        stem = {}
        for real, word in zip(S.REAL, surface):
            stem[tokenize(word)[0]['id']] = real
        # Build the system prompt exactly as the experiment does: prose plus the
        # arm's own serialised schema. Anatomy of the prose alone would describe
        # a prompt nobody ran -- those 754 schema characters are most of the
        # token budget and carry the enum, which is where the label tokens live.
        system = prose + S.schema_line(S.schema(alt, stated))
        rendered = S.render(system, PROBE_CLAUSE, prefix) + ANCHOR
        toks = tokenize(rendered)
        hits = [{'i': i, 'piece': t['piece'], 'label': stem[t['id']]}
                for i, t in enumerate(toks) if t['id'] in stem]
        entry = {
            'system_chars': len(system),
            'prose_chars': len(prose),
            'n_tokens': len(toks),
            'tokens': [t['piece'] for t in toks],
            'hits': hits,
            'hit_counts': {l: sum(1 for h in hits if h['label'] == l)
                           for l in S.REAL},
            'surface': dict(zip(S.REAL, surface)),
            'baseline': dist(rendered, stem),
            'alt_labels': alt,
            'stated_first': bool(stated),
        }
        if not a.no_influence:
            sents = sentences(system)
            infl = []
            for s in sents:
                reduced = system.replace(s, '', 1)
                d = dist(S.render(reduced, PROBE_CLAUSE, prefix) + ANCHOR, stem)
                infl.append({
                    'sentence': s,
                    'delta': {k: round(d[k] - entry['baseline'][k], 4) for k in d},
                })
            entry['influence'] = infl
        result[name] = entry
        print('%-11s %4d tokens | label-token hits %s | baseline %s'
              % (name, entry['n_tokens'], entry['hit_counts'],
                 {k: round(v, 3) for k, v in entry['baseline'].items()}), flush=True)

    a.out.write_text(json.dumps(result, indent=1))
    print('\nwrote', a.out)


if __name__ == '__main__':
    main()
