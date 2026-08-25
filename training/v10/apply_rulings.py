#!/usr/bin/env python3
"""Apply adjudications to the bulk vote file: every shape gets a final
`label` (its ruling if one exists, else its consensus, else null for a
split nobody has ruled on) and ruled shapes carry the `ruling` record.

`score` regenerates bulk_votes.json from the raw votes and drops these
fields; run this again afterwards. Loud on a ruling that names a shape
the sample does not contain -- a silently ignored ruling is a label that
never happened.

    python3 training/v10/apply_rulings.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VOTES = HERE / 'bulk_votes.json'
RULINGS = HERE / 'rulings.json'


class RulingError(ValueError):
    pass


def apply(records, rulings):
    """Pure: returns (records with label/ruling set, unresolved shape keys)."""
    by_shape = {r['shape']: r for r in records}
    missing = [x['shape'] for x in rulings if x['shape'] not in by_shape]
    if missing:
        raise RulingError(f'ruling(s) for shape(s) not in the sample: {missing}')
    for r in records:
        r.pop('ruling', None)
        r['label'] = r['consensus']
    for x in rulings:
        r = by_shape[x['shape']]
        r['ruling'] = {k: v for k, v in x.items() if k != 'shape'}
        r['label'] = x['label']
    unresolved = [r['shape'] for r in records if r['label'] is None]
    return records, unresolved


def main(argv=None):
    votes = json.loads(VOTES.read_text())
    rulings = json.loads(RULINGS.read_text())['rulings']
    try:
        votes['shapes'], unresolved = apply(votes['shapes'], rulings)
    except RulingError as e:
        raise SystemExit(str(e))
    votes['rulings_applied'] = len(rulings)
    votes['unresolved'] = unresolved
    VOTES.write_text(json.dumps(votes, indent=1) + '\n')
    labels = {}
    for r in votes['shapes']:
        labels[r['label']] = labels.get(r['label'], 0) + 1
    print(f'{len(rulings)} ruling(s) applied; final labels {labels}; unresolved {unresolved}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
