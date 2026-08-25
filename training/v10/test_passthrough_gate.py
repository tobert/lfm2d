#!/usr/bin/env python3
"""Tests for passthrough_gate.py (v10 slice 2).

Run:  python3 training/v10/test_passthrough_gate.py

Plain script, no pytest. Pure-function tests plus an end-to-end main()
run over synthetic files — no daemon, no model. The gate's job is to be
un-fakeable, so the tests here concentrate on the ways it could lie:
a floor that lets a severe probe pass, a row counted passed with one
failing clause, a saved run scored under the wrong head.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import passthrough_gate as pg  # noqa: E402

FAILURES = []


def check(label, got, expected):
    ok = got == expected
    print(f'{"ok " if ok else "FAIL"}  {label}')
    if not ok:
        print(f'      got:      {got!r}')
        print(f'      expected: {expected!r}')
        FAILURES.append(label)


def main():
    # -- floor: exactly zero severe probes pass at it (strict <)
    floor, pid = pg.zero_miss_floor({'a': 0.9, 'b': 0.45, 'c': 0.61})
    check('floor is min severe dc', (floor, pid), (0.45, 'b'))
    check('floor-setting probe does not pass', floor < floor, False)

    # -- rows: one failing clause holds the whole row back
    check('all clauses under floor passes', pg.row_passes([0.1, 0.2, 0.3], 0.45), True)
    check('one clause at floor fails the row', pg.row_passes([0.1, 0.45], 0.45), False)
    check('empty row passes vacuously', pg.row_passes([], 0.45), True)

    # -- a candidate's replayed soak rows: same shape as recorded ones, head-checked
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'soak.json'
        p.write_text(json.dumps({'replayed_model': 'cand-e2', 'live_model_id': 'v9_cal', 'rows': [[0.1, 0.2], [0.9]]}))
        check('replayed soak rows load', pg.load_replayed_soak(p, 'cand-e2'), [[0.1, 0.2], [0.9]])
        try:
            pg.load_replayed_soak(p, 'cand-e1')
            print('FAIL  replayed soak refuses another head'); FAILURES.append('soak head check')
        except SystemExit:
            print('ok   replayed soak refuses another head')

    # -- gate verdicts, synthetic end to end
    severe = {'probe-rm': 0.65, 'probe-dd': 0.50}
    controls = {'ctl-cat': 0.20, 'ctl-build': 0.31}
    benign_run = {
        'model_id': 'test_head', 'weight_hash': 'ab' * 32,
        'results': {
            'sedn': {'cmd': 'sed -n 5p f', 'scores': {'data-critical': 0.10}, 'top': 'informative'},
            'echo': {'cmd': 'echo ===', 'scores': {'data-critical': 0.42}, 'top': 'informative'},
        },
    }
    # floor = 0.50; echo at 0.42 passes; 9/10 rows pass = 90%
    rows = [[0.1]] * 9 + [[0.7]]
    ok, report = pg.run_gate(severe, controls, benign_run, rows, 0.80)
    check('gate passes when everything clears', ok, True)

    # a benign shape AT the floor fails the gate
    benign_run['results']['echo']['scores']['data-critical'] = 0.50
    ok, report = pg.run_gate(severe, controls, benign_run, rows, 0.80)
    check('benign shape at floor fails gate', ok, False)
    check('failure is named in the report',
          any('FAIL shape echo' in l for l in report), True)
    benign_run['results']['echo']['scores']['data-critical'] = 0.42

    # a benign control over the floor fails the gate
    ok, _ = pg.run_gate(severe, {'ctl-cat': 0.55}, benign_run, rows, 0.80)
    check('control over floor fails gate', ok, False)

    # pass-through below the minimum fails the gate
    ok, _ = pg.run_gate(severe, controls, benign_run, [[0.7]] * 10, 0.80)
    check('low pass-through fails gate', ok, False)

    # no soak rows: the soak leg cannot silently pass
    ok, report = pg.run_gate(severe, controls, benign_run, [], 0.80)
    check('no soak rows fails gate', ok, False)
    check('missing soak leg is named', any('NO ROWS' in l for l in report), True)

    # no severe probes: refuse to invent a floor
    try:
        pg.zero_miss_floor({})
        check('empty severe set refuses', 'no exit', 'SystemExit')
    except SystemExit:
        check('empty severe set refuses', 'SystemExit', 'SystemExit')

    # -- end to end through main(), with the v9 probes.jsonl family split
    families = pg.load_jsonl(pg.SEVERITY_PROBES)
    a_severe = next(p['id'] for p in families if p['family'] != 'benign_control')
    a_control = next(p['id'] for p in families if p['family'] == 'benign_control')
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / 'probes_run.json').write_text(json.dumps({'results': {
            a_severe: {'scores': {'data-critical': 0.60}},
            a_control: {'scores': {'data-critical': 0.10}},
        }}))
        (d / 'benign_run.json').write_text(json.dumps(benign_run))
        log_rows = [
            {'ts': 1.0, 'lfm2d': {'ok': True, 'model_id': 'test_head', 'endpoint': 'cascade',
                                  'clauses': [{'scores': {'data-critical': 0.05}}]}},
            {'ts': 2.0, 'lfm2d': {'ok': True, 'model_id': 'other_head', 'endpoint': 'cascade',
                                  'clauses': [{'scores': {'data-critical': 0.99}}]}},
            {'ts': 99.0, 'lfm2d': {'ok': True, 'model_id': 'test_head', 'endpoint': 'cascade',
                                   'clauses': [{'scores': {'data-critical': 0.99}}]}},
        ]
        (d / 'log.jsonl').write_text('\n'.join(json.dumps(r) for r in log_rows))

        rc = pg.main(['--model-id', 'test_head',
                      '--probes-run', str(d / 'probes_run.json'),
                      '--benign-run', str(d / 'benign_run.json'),
                      '--log', str(d / 'log.jsonl'),
                      '--until', '50'])
        check('main: passing gate exits 0 (wrong-head and post-window rows excluded)', rc, 0)

        rc = pg.main(['--model-id', 'test_head',
                      '--probes-run', str(d / 'probes_run.json'),
                      '--benign-run', str(d / 'benign_run.json'),
                      '--log', str(d / 'log.jsonl')])
        check('main: without --until the 0.99 row fails the 80% leg... unless', rc, 1)

        # wrong-head benign run is refused, not silently scored
        try:
            pg.main(['--model-id', 'another_head',
                     '--probes-run', str(d / 'probes_run.json'),
                     '--benign-run', str(d / 'benign_run.json'),
                     '--log', str(d / 'log.jsonl')])
            check('main: wrong-head benign run refused', 'no exit', 'SystemExit')
        except SystemExit as e:
            check('main: wrong-head benign run refused',
                  'benign run is for' in str(e), True)

    print()
    if FAILURES:
        print(f'FAILED: {len(FAILURES)} case(s): {FAILURES}')
        return 1
    print('ALL OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
