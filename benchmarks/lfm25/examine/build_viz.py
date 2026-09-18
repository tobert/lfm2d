#!/usr/bin/env python3
"""Turn `lfm25-examine` records into one self-contained page.

    build_viz.py --out DIR  NAME=path/to/examination.json [NAME=... ...]

Each record is reduced to what the page draws (router logits dropped, floats
rounded) and embedded in viz_template.html. Nothing about the labels, the token
sets or the layer stack is written here or in the template: the page reads them
out of the records, so it follows whatever vocabulary the examination used.

--ladder N says the first N tokens of the lens set are an ORDERED scale, low to
high, as the caller listed them; any after that are off the scale. The ladder is
then drawn with luminance carrying the order (the top rung is the most salient
in either theme) and hue shifting alongside so crossing lines stay traceable.
Strictly equal-luminance hues were tried first and fail colour-vision checks
(worst pair dE 2-5): luminance is the channel a colour-blind reader has left, so
it is spent on purpose rather than matched away. Without --ladder the page keeps
its validated categorical hues.

The page carries prompt text, so --out is required and never defaulted, the same
rule as the harnesses in ../prompts/.
"""
import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA = 'lfm25-examination-v1'
MARK = '/*__EXAMINATIONS__*/'
TOKENS_MARK = '/*__SERIES_TOKENS__*/'
LADDER_MARK = '/*__LADDER__*/0'


def oklch_hex(L, C, h):
    """OKLCH -> sRGB hex, giving up chroma until the colour is in gamut."""
    while True:
        a, b = C * math.cos(math.radians(h)), C * math.sin(math.radians(h))
        l, m, s = ((L + 0.3963377774 * a + 0.2158037573 * b) ** 3,
                   (L - 0.1055613458 * a - 0.0638541728 * b) ** 3,
                   (L - 0.0894841775 * a - 1.2914855480 * b) ** 3)
        rgb = (4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
               -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
               -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s)
        if all(-1e-4 <= v <= 1 + 1e-4 for v in rgb) or C <= 0:
            enc = lambda v: 12.92 * v if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055
            return '#%02x%02x%02x' % tuple(round(255 * enc(min(1, max(0, v)))) for v in rgb)
        C -= 0.005


def ladder_tokens(n):
    """CSS for an n-rung ladder: same hue per rung in both themes, luminance
    rising toward the salient end of each theme's surface."""
    if not 2 <= n <= 6:
        raise SystemExit('--ladder takes 2..6 rungs')
    def rungs(lo, hi):
        return ' '.join('--series-%d: %s;' % (i + 1, oklch_hex(lo + (hi - lo) * i / (n - 1), 0.15,
                                                                 100 + 185 * i / (n - 1)))
                        for i in range(n))
    light, dark = rungs(0.76, 0.42), rungs(0.50, 0.81)
    return (':root { %s }\n@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { %s } }\n'
            ':root[data-theme="dark"] { %s }' % (light, dark, dark))


def rounded(x, places):
    if isinstance(x, list):
        return [rounded(v, places) for v in x]
    return round(x, places)


def margins(r):
    """Per position, how firmly this layer's router chose: the biased score of
    its last chosen expert minus the best one it left out. None without the
    router logits and selection bias in the record."""
    if 'logits' not in r or 'selection_bias' not in r:
        return None
    k, out = len(r['experts'][0]), []
    for row in r['logits']:
        s = sorted((1 / (1 + math.exp(-v)) + b for v, b in zip(row, r['selection_bias'])), reverse=True)
        out.append(round(s[k - 1] - s[k], 4))
    return out


def reduce(name, record):
    e = record['examination']
    if e['schema'] != SCHEMA:
        raise SystemExit('%s: schema %r, this builder reads %r' % (name, e['schema'], SCHEMA))
    return {
        'name': name,
        'identity': record['identity'],
        'source': record['source'],
        'tokens': e['tokens'],
        'layers': e['layers'],
        'depths': e['depths'],
        'residual_norm': rounded(e['residual_norm'], 2),
        'routing': [{'layer': r['layer'], 'n_experts': r['n_experts'],
                     'experts': r['experts'], 'weights': rounded(r['weights'], 3),
                     'margin': margins(r)}
                    for r in e['routing']],
        'lens': {k: {'tokens': v['tokens'],
                     'words': [x['word'] for x in record['first_token_resolution'] if x['set'] == k],
                     'logprob': rounded(v['logprob'], 3),
                     'mass_logprob': rounded(v['mass_logprob'], 3)}
                 for k, v in e['lens'].items()},
        'top': [{'position': t['position'],
                 'by_depth': [[{'piece': x['piece'], 'logprob': round(x['logprob'], 3)}
                               for x in rank] for rank in t['by_depth']]}
                for t in e['top']],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', type=Path, required=True,
                    help='directory for examine-viz.html, outside this repo')
    ap.add_argument('--ladder', type=int, default=0, metavar='N',
                    help='the first N tokens of the lens set are an ordered scale, low to high')
    ap.add_argument('records', nargs='+', metavar='NAME=PATH')
    a = ap.parse_args()
    exams = []
    for arg in a.records:
        name, sep, path = arg.partition('=')
        if not sep or not name or not path:
            raise SystemExit('expected NAME=PATH, got %r' % arg)
        exams.append(reduce(name, json.loads(Path(path).read_text())))
    template = (HERE / 'viz_template.html').read_text()
    for mark in (MARK, TOKENS_MARK, LADDER_MARK):
        if template.count(mark) != 1:
            raise SystemExit('viz_template.html must hold %s exactly once' % mark)
    template = template.replace(TOKENS_MARK, ladder_tokens(a.ladder) if a.ladder else '')
    template = template.replace(LADDER_MARK, str(a.ladder))
    # `</` cannot appear inside an inline script; pieces like </think> can.
    data = json.dumps(exams, separators=(',', ':')).replace('</', '<\\/')
    a.out.mkdir(parents=True, exist_ok=True)
    target = a.out / 'examine-viz.html'
    target.write_text(template.replace(MARK, data))
    print('%d examinations, %.0f KiB -> %s' % (len(exams), target.stat().st_size / 1024, target))


if __name__ == '__main__':
    main()
