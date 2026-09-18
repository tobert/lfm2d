#!/usr/bin/env python3
"""Embed an expert_contrast.json in contrast_template.html.

    build_contrast_viz.py CONTRAST.json OUT_DIR [KNOCKOUT_REPORT.json]    (OUT_DIR outside this repo)
"""
import json, sys
from pathlib import Path

MARK = '/*__CONTRAST__*/'


def knockout_view(data, report):
    """The causal map as one more view on the same grid. A cell's value is how
    far the expert PUSHES toward `up`: minus the margin shift its knockout
    caused. Cells no probe chose stay empty."""
    layers, n = data['layers'], data['n_experts']
    grid = lambda: [[0.0] * n for _ in layers]
    push, count, flips = grid(), grid(), grid()
    for c in report['cells']:
        i = layers.index(c['layer'])
        push[i][c['expert']], count[i][c['expert']], flips[i][c['expert']] = -c['mean_shift'], c['n'], c['flips']
    ringed = report.get('ringed', [])
    by = {(c['layer'], c['expert']): c for c in report['cells']}
    same = sum(1 for r in ringed if (r['layer'], r['expert']) in by
               and (r['usage_diff'] > 0) == (by[(r['layer'], r['expert'])]['mean_shift'] < 0))
    return {'kind': 'knockout', 'name': '%s vs %s' % (report['up'], report['down']), 'window': 'knockout',
            'n': [sum(report['groups'].values()), 0], 'bar': 0.3, 'diff': push, 'count': count, 'flips': flips,
            'ringed_checked': len(ringed), 'ringed_same_direction': same, 'backend': report['identity']['backend']}


def main():
    data, out = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2])
    if len(sys.argv) > 3:
        data['contrasts'].append(knockout_view(data, json.loads(Path(sys.argv[3]).read_text())))
    template = (Path(__file__).resolve().parent / 'contrast_template.html').read_text()
    if template.count(MARK) != 1:
        raise SystemExit('contrast_template.html must hold %s exactly once' % MARK)
    out.mkdir(parents=True, exist_ok=True)
    target = out / 'expert-contrast.html'
    target.write_text(template.replace(MARK, json.dumps(data, separators=(',', ':')).replace('</', '<\\/')))
    print('%.0f KiB -> %s' % (target.stat().st_size / 1024, target))


if __name__ == '__main__':
    main()
