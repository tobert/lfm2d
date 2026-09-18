#!/usr/bin/env python3
"""Embed one or more verdict_ribbon.py records in ribbon_template.html.

    build_ribbon_viz.py OUT_DIR 'LABEL=RIBBON.json' ['LABEL=RIBBON.json' ...]

OUT_DIR is outside this repo. Each record becomes one choice of the page's Slot
switch, in the order given. The page draws the per-depth lean and set mass of
every row; the residual norms ride along in the record but are not drawn, so
they are dropped here to keep the page small.
"""
import json, sys
from pathlib import Path

MARK = '/*__RIBBON__*/'
DRAWN = ('n', 'group', 'label', 'wrote', 'examiner_top', 'scope', 'undo', 'lean', 'mass_logprob')


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    out, data = Path(sys.argv[1]), []
    for arg in sys.argv[2:]:
        label, _, path = arg.partition('=')
        if not label or not path:
            raise SystemExit('expected LABEL=RIBBON.json but got %r' % arg)
        run = json.loads(Path(path).read_text())
        if run.get('schema') != 'lfm25-verdict-ribbon-v2':
            raise SystemExit('%s is not a v2 ribbon record: schema %r' % (path, run.get('schema')))
        run['label'] = label
        run['rows'] = [{k: r[k] for k in DRAWN} for r in run['rows']]
        data.append(run)
    template = (Path(__file__).resolve().parent / 'ribbon_template.html').read_text()
    if template.count(MARK) != 1:
        raise SystemExit('ribbon_template.html must hold %s exactly once' % MARK)
    out.mkdir(parents=True, exist_ok=True)
    target = out / 'verdict-ribbon.html'
    target.write_text(template.replace(MARK, json.dumps(data, separators=(',', ':')).replace('</', '<\\/')))
    print('%.0f KiB -> %s' % (target.stat().st_size / 1024, target))


if __name__ == '__main__':
    main()
