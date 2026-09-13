#!/usr/bin/env python3
"""Tests for device_agreement.py — does the severity head's verdict survive a device change?

Run:  .venv-train/bin/python training/v10/test_device_agreement.py

Plain script, no pytest, same shape as test_passthrough_gate.py. Pure
functions over synthetic scores — no daemon, no model. (The torch venv is
only needed because the row verdict reuses clause_replay.py's aggregation,
and that module imports torch at load.)

The tool's job is to put a number on device drift that nobody can talk
down, so these tests aim at the ways it could under-count: a flip missed
on a near-tie, a row verdict that ignores a winner change, a soak file
that mixes endpoints, a comparison across two different heads, a device
claim taken on faith instead of read from the daemon's own startup line.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import device_agreement as da  # noqa: E402

FAILURES = []
LABELS = ['informative', 'situation-normal', 'data-critical']


def check(label, got, expected):
    ok = got == expected
    print(f'{"ok " if ok else "FAIL"}  {label}')
    if not ok:
        print(f'      got:      {got!r}')
        print(f'      expected: {expected!r}')
        FAILURES.append(label)


def raises(label, exc, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc:
        check(label, True, True)
        return
    except Exception as e:  # the wrong refusal is still a failure
        check(label, f'raised {type(e).__name__}: {e}', f'raises {exc.__name__}')
        return
    check(label, 'returned normally', f'raises {exc.__name__}')


def _raised(fn, *args, **kwargs):
    """Name of the exception fn raises, or 'returned normally'."""
    try:
        fn(*args, **kwargs)
    except BaseException as e:  # SystemExit is not an Exception
        return type(e).__name__
    return 'returned normally'


def s(inf, sn, dc):
    return {'informative': inf, 'situation-normal': sn, 'data-critical': dc}


def keyed(d):
    return {da.text_key(t): v for t, v in d.items()}


def main():
    # -- keys: text never stored, but the same text must always meet itself
    check('text_key is stable', da.text_key('ls -la'), da.text_key('ls -la'))
    check('text_key separates texts', da.text_key('ls -la') == da.text_key('ls -l'), False)

    # -- top label: exact ties go to the EARLIER label in checkpoint order
    check('top label is argmax', da.top_label(s(0.2, 0.3, 0.5), LABELS), 'data-critical')
    check('exact tie goes to earlier label', da.top_label(s(0.4, 0.4, 0.2), LABELS), 'informative')

    # -- clause comparison: one near-tie flip among three clauses
    a = keyed({'x': s(0.6, 0.3, 0.1), 'y': s(0.2, 0.3, 0.5), 'z': s(0.34, 0.33, 0.33)})
    b = keyed({'x': s(0.6, 0.3, 0.1), 'y': s(0.2, 0.31, 0.49), 'z': s(0.32, 0.33, 0.35)})
    c = da.compare_clauses(a, b, LABELS)
    check('clause count', c['clauses'], 3)
    check('a near-tie flip is counted', c['top_flips'], 1)
    check('flip pair names both sides', c['flip_pairs'], {'informative->data-critical': 1})
    check('max abs diff per label',
          {k: round(v, 6) for k, v in c['max_abs_diff'].items()},
          {'informative': 0.02, 'situation-normal': 0.01, 'data-critical': 0.02})
    check('flipped clause margin on side a', round(c['flip_margin_max'], 6), 0.01)
    # "0 flips" only means something next to how many clauses COULD flip:
    # the closest call's margin, and how many clauses sit near a boundary
    check('closest call margin on side a', round(c['min_margin'], 6), 0.01)
    check('clauses near a boundary, by margin threshold', c['margin_below'],
          {'0.0001': 0, '0.001': 0, '0.01': 0, '0.05': 1})

    same = da.compare_clauses(a, a, LABELS)
    check('identical runs: zero flips', same['top_flips'], 0)
    check('identical runs: zero diff', max(same['max_abs_diff'].values()), 0.0)
    check('identical runs: no flip margin', same['flip_margin_max'], None)

    raises('refuses different clause sets', ValueError,
           da.compare_clauses, a, keyed({'x': s(0.6, 0.3, 0.1)}), LABELS)

    # -- row verdicts: the decisions the hook actually makes
    rows = [
        # cascade: winner moves clause 0 -> 1, both winners data-critical,
        # so fired and top agree while the winner does not
        {'command': 'p; q', 'lfm2d': {'ok': True, 'endpoint': 'cascade', 'clauses': [
            {'clause': 'p'}, {'clause': 'q'}]}},
        # classify: informative on a, data-critical on b -> fired flips
        {'command': 'r', 'lfm2d': {'ok': True, 'endpoint': 'classify'}},
        # classify_batch: no winner by design; fired agrees
        {'command': 'p q', 'lfm2d': {'ok': True, 'endpoint': 'classify_batch', 'clauses': [
            {'clause': 'p'}, {'clause': 'q'}]}},
    ]
    ra = keyed({'p': s(0.05, 0.05, 0.9), 'q': s(0.8, 0.1, 0.1), 'r': s(0.7, 0.2, 0.1)})
    rb = keyed({'p': s(0.1, 0.6, 0.3), 'q': s(0.0, 0.05, 0.95), 'r': s(0.4, 0.1, 0.5)})
    ag = da.row_agreement(rows, ra, rb, LABELS, ['situation-normal', 'data-critical'])
    check('row agreement counts', ag, {
        'rows': 3, 'fired_disagree': 1, 'top_disagree': 1,
        'cascade_rows': 1, 'winner_disagree': 1,
    })
    raises('row agreement refuses an unscored clause', KeyError,
           da.row_agreement, rows, {}, rb, LABELS, ['situation-normal', 'data-critical'])

    # -- soak file: passthrough_gate's format, cascade rows only, one side's dcs
    doc = da.soak_doc(rows, ra, 'kube_ordinal_v10F_candraw-e2', 'kube_ordinal_v9_cal', 1787497523)
    check('soak doc header', (doc['replayed_model'], doc['live_model_id'], doc['until']),
          ('kube_ordinal_v10F_candraw-e2', 'kube_ordinal_v9_cal', 1787497523))
    check('soak doc keeps cascade rows only, dc per clause', doc['rows'], [[0.9, 0.1]])

    # -- device evidence comes from the daemon's own startup line
    line = ('\x1b[2m2026-09-12T21:30:00.1Z\x1b[0m \x1b[32m INFO\x1b[0m lfm2d: lfm2d: startup '
            'observability available_parallelism=32 configured_threads=8 '
            'requested_device="rocm" device_index=0 device_type=gpu backend=rocm '
            'dtype=f32 container_runtime=none cgroup_cpu_max=None')
    check('startup line parsed through ANSI',
          da.parse_startup('noise\n' + line + '\nmore'),
          {'device_type': 'gpu', 'backend': 'rocm', 'dtype': 'f32', 'configured_threads': 8,
           'requested_device': 'rocm'})
    raises('no startup line is a refusal, not a guess', ValueError,
           da.parse_startup, 'lfm2d: loaded model id=x')

    # -- identity: a comparison must be ONE head on two devices
    good = [{'model_id': 'm', 'weight_hash': 'h'}, {'model_id': 'm', 'weight_hash': 'h'}]
    check('identity of a consistent batch', da.batch_identity(good), ('m', 'h'))
    raises('mixed weight hashes in one batch refused', SystemExit,
           da.batch_identity, [{'model_id': 'm', 'weight_hash': 'h'},
                               {'model_id': 'm', 'weight_hash': 'other'}])
    raises('different heads across runs refused', SystemExit,
           da.same_head, {'model_id': 'm', 'weight_hash': 'h'},
           {'model_id': 'm', 'weight_hash': 'other'})

    # -- the live log keeps growing while a two-hour collection runs: the
    #    row window must be frozen, never "whatever the log holds at compare"
    live = [{'ts': 100.0}, {'ts': 199.9}, {'ts': 200.0}, {'ts': 350.0}]
    check('live window keeps rows strictly before until',
          [r['ts'] for r in da.live_window(live, 200.0)], [100.0, 199.9])
    check('live window resolves to the EARLIER recorded read',
          da.resolve_live_until(None, {'log_read_at': 300.0}, {'log_read_at': 250.0}), 250.0)
    check('explicit --live-until wins',
          da.resolve_live_until(120.0, {'log_read_at': 300.0}, {'log_read_at': 250.0}), 120.0)
    check('explicit --live-until after a read is refused',
          _raised(da.resolve_live_until, 400.0, {'log_read_at': 300.0}, {'log_read_at': 250.0}),
          'SystemExit')
    check('no read time and no flag is refused, not guessed',
          _raised(da.resolve_live_until, None, {}, {'log_read_at': 250.0}), 'SystemExit')

    print(f'\n{len(FAILURES)} failure(s)' if FAILURES else '\nall passed')
    return 1 if FAILURES else 0


if __name__ == '__main__':
    sys.exit(main())
