#!/usr/bin/env python3
"""How much would excluding a syntactic shape change the firing rate?

Written 2026-08-16 after measuring the heredoc question twice with throwaway
scripts and getting the scope WRONG the first time -- see
`commit-the-scorer`. A metric ships with its scorer.

The question this answers: agents carry programs and prose inside shell
framing (heredocs, `python3 -c '...'`, pipes into an interpreter). The
classifier reads that payload as if it were the command. This measures
(a) how often each shape occurs, (b) whether it is OVER-represented among
data-critical firings, and (c) with --url, how many firings actually clear
when the payload is replaced.

(b) is the one that stops you overclaiming. Heredocs turned out to be
represented at exactly 1.0x their share of traffic -- present in 10.2% of
traffic AND 10.2% of firings -- so a 45% clear-rate WITHIN heredoc firings
buys only ~5% of firings overall.

ALWAYS pass --model-id. The advisory log spans checkpoints (v8 -> v9 cut
over 2026-08-16) and a firing rate mixed across two models describes
neither. Run with no filter and it reports per-model and refuses to total.

Privacy: the advisory log is 0600, local-only, real command text. This
script reads it and emits AGGREGATES ONLY -- same contract as
`backtest_candidate.py`. Do not add per-row text output.

    python3 training/v9/severity_probes/shape_impact.py --model-id kube_ordinal_v9
    python3 training/v9/severity_probes/shape_impact.py --model-id kube_ordinal_v9 \
        --url http://lfm2d-1.taila4abc.ts.net:8088 --sample 40
"""
import argparse
import json
import random
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

DEFAULT_LOG = '~/.cache/claude-hooks/lfm2d-advisory.jsonl'

# A heredoc's opening token. The delimiter may be quoted (literal body) or
# not (shell expands first) -- kaish calls this `literal`, and it decides
# whether the text you classified is the text that runs.
HEREDOC_OPEN = re.compile(r'<<-?\s*(["\']?)(\w+)\1')

# Whole heredoc incl. body, for stripping. Non-greedy body, terminator on
# its own line. Note `re.M` so ^/$ are line anchors, `re.S` so . spans lines.
#
# `[^\n]*\n` after the delimiter is LOAD-BEARING and was missing until
# 2026-08-16. A heredoc body starts after the END OF THE INTRODUCER LINE,
# not after the delimiter token -- everything between them still belongs to
# the command. Without it, `git commit -F - <<'EOF' && git push origin main`
# captured " && git push origin main" as part of the body, so the stripper
# deleted a real command and hid it from the classifier. That is a false
# NEGATIVE in a guard, the one direction we cannot accept, and it survived
# review because every test case put the delimiter at end-of-line.
# Diagnosed by diffing against kaish's parser (kaish-lead, from the
# character counts: 13 chars for `&& echo done`, 24 for `&& git push
# origin main`). Group 1 now spans opener THROUGH that newline.
HEREDOC_FULL = re.compile(r'(<<-?\s*(["\']?)(\w+)\2[^\n]*\n)(.*?)(^\s*\3\s*$)', re.S | re.M)

INTERPRETERS = r'python3?|perl|ruby|node|deno|bun|php|Rscript|osascript'
DASH_C = re.compile(rf'\b({INTERPRETERS})\b[^|;&]*\s-c\s')
PIPE_INTO = re.compile(rf'\|\s*({INTERPRETERS}|sh|bash|zsh)\b')

SHAPES = {
    'heredoc': HEREDOC_OPEN,
    'interp_dash_c': DASH_C,
    'pipe_into_interp': PIPE_INTO,
}

PLACEHOLDER = '\n<BODY>\n'


def shapes_in(cmd: str) -> set:
    """Which payload-carrying shapes does this command use?"""
    return {name for name, rx in SHAPES.items() if rx.search(cmd)}


def strip_heredoc_bodies(cmd: str) -> str:
    """Replace every heredoc BODY with a placeholder, keeping shell framing.

    This is the local stand-in for kaish's `PlannedHeredoc.body_offset ..
    body_offset + body.len()`, which is exact. This regex is NOT exact -- it
    cannot see quoting context and will miss pathological nesting -- so treat
    its output as a LOWER bound on what real span data would achieve.
    """
    return HEREDOC_FULL.sub(lambda m: m.group(1) + PLACEHOLDER + m.group(5), cmd)


def delimiter_stats(cmds):
    """Quoted vs unquoted delimiters, and which words get used."""
    quoted = unquoted = 0
    words = Counter()
    for c in cmds:
        for m in HEREDOC_OPEN.finditer(c):
            if m.group(1):
                quoted += 1
            else:
                unquoted += 1
            words[m.group(2)] += 1
    return quoted, unquoted, words


def load_rows(log_path: Path, model_id=None):
    """Advisory rows that carry a real verdict. Aggregates only downstream."""
    rows = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            lf = r.get('lfm2d') or {}
            if not lf.get('ok') or not lf.get('top') or not r.get('command'):
                continue
            if model_id and lf.get('model_id') != model_id:
                continue
            rows.append(r)
    return rows


def representation(rows):
    """Per shape: share of traffic, share of firings, and the ratio.

    A ratio of 1.0 means the shape is present in firings exactly as often as
    in traffic -- i.e. it is NOT a driver, however plausible it looks.
    """
    n = len(rows)
    fired = [r for r in rows if r['lfm2d']['top'] == 'data-critical']
    out = {}
    for name in SHAPES:
        t = sum(1 for r in rows if name in shapes_in(r['command']))
        f = sum(1 for r in fired if name in shapes_in(r['command']))
        out[name] = {
            'traffic': t, 'traffic_pct': 100 * t / n if n else 0,
            'firings': f, 'firings_pct': 100 * f / len(fired) if fired else 0,
            'ratio': ((f / len(fired)) / (t / n)) if fired and t and n else None,
        }
    return n, len(fired), out


def clear_rate(rows, url, sample, seed=20260816, timeout=30):
    """Of firings carrying a heredoc, how many stop firing without the body?"""
    def classify(text):
        req = urllib.request.Request(
            f'{url}/v1/classify', data=json.dumps({'inputs': text}).encode(),
            headers={'content-type': 'application/json'})
        return json.load(urllib.request.urlopen(req, timeout=timeout))[0]['top']

    cmds, seen = [], set()
    for r in rows:
        c = r['command']
        if 'heredoc' in shapes_in(c) and c not in seen:
            seen.add(c)
            cmds.append(c)
    random.Random(seed).shuffle(cmds)

    tested = cleared = 0
    for c in cmds:
        if tested >= sample:
            break
        if classify(c) != 'data-critical':
            continue          # only firings are informative here
        stripped = strip_heredoc_bodies(c)
        if stripped == c:
            continue          # nothing strippable; not a data point
        tested += 1
        cleared += classify(stripped) != 'data-critical'
    return tested, cleared


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', default=DEFAULT_LOG, type=Path)
    ap.add_argument('--model-id', help='REQUIRED for a total; the log spans checkpoints')
    ap.add_argument('--url', help='also measure the live clear rate (costs one call per row)')
    ap.add_argument('--sample', type=int, default=40)
    args = ap.parse_args()

    log = Path(str(args.log)).expanduser()
    if not args.model_id:
        # Refuse to total across checkpoints; report the split instead.
        counts = Counter(r['lfm2d'].get('model_id') for r in load_rows(log))
        print('No --model-id given. The advisory log spans checkpoints, and a '
              'firing rate mixed across models describes neither.\n')
        for m, c in counts.most_common():
            print(f'  {m}: {c} rows')
        print('\nRe-run with --model-id <one of the above>.')
        return 2

    rows = load_rows(log, args.model_id)
    if not rows:
        print(f'no rows for model_id={args.model_id}', file=sys.stderr)
        return 1

    n, fired, rep = representation(rows)
    print(f'model_id={args.model_id}   rows with a verdict: {n}')
    print(f'data-critical firings: {fired} ({100 * fired / n:.1f}% of traffic)\n')
    print(f'{"shape":20s} {"traffic":>16s} {"firings":>16s} {"ratio":>7s}')
    for name, d in rep.items():
        ratio = f'{d["ratio"]:.2f}x' if d['ratio'] is not None else '   n/a'
        print(f'  {name:18s} {d["traffic"]:6d} {d["traffic_pct"]:6.1f}%'
              f' {d["firings"]:6d} {d["firings_pct"]:6.1f}% {ratio:>7s}')
    print('\n  ratio 1.0 = present in firings exactly as often as in traffic,'
          '\n  i.e. NOT a driver however plausible it looks.')

    q, u, words = delimiter_stats([r['command'] for r in rows])
    if q or u:
        tot = q + u
        print(f'\nheredoc delimiters: {q} quoted / {u} unquoted '
              f'({100 * q / tot:.0f}% literal -- body IS what runs)')
        print('  top words:', dict(words.most_common(6)))

    if args.url:
        tested, cleared = clear_rate(rows, args.url, args.sample)
        if not tested:
            print('\nno heredoc-carrying firings to test')
            return 0
        within = 100 * cleared / tested
        overall = within / 100 * rep['heredoc']['firings_pct']
        print(f'\nbody-strip clear rate: {cleared}/{tested} ({within:.0f}%) of '
              f'heredoc firings stop firing')
        print(f'  => ~{overall:.0f}% of ALL firings for this model'
              f'  (the number to quote; regex strip is a LOWER bound vs real spans)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
