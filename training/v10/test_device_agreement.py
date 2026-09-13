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
    # every label's drift, not just one hard-coded name
    check('per-label abs diff p50',
          {k: round(v, 6) for k, v in c['abs_diff_p50'].items()},
          {'informative': 0.0, 'situation-normal': 0.0, 'data-critical': 0.01})
    check('per-label abs diff p99',
          {k: round(v, 6) for k, v in c['abs_diff_p99'].items()},
          {'informative': 0.02, 'situation-normal': 0.01, 'data-critical': 0.02})

    # -- key sets: compare must not silently narrow what it measured. The
    #    2026-09-12 cpu8 run held 3 clauses rocm0 lacked, and the old note
    #    stayed silent because one set was a strict SUBSET of the other.
    two = keyed({'x': 1, 'y': 2})
    one = keyed({'x': 1})
    check('equal key sets pass with counts', da.shared_keys(two, dict(two), intersect=False)[1],
          {'shared': 2, 'only_a': 0, 'only_b': 0})
    check('a strict subset is refused without --intersect',
          _raised(da.shared_keys, one, two, intersect=False), 'SystemExit')
    check('--intersect records what was dropped, as numbers',
          da.shared_keys(one, two, intersect=True)[1], {'shared': 1, 'only_a': 0, 'only_b': 1})
    check('--intersect keeps only the shared keys',
          da.shared_keys(one, two, intersect=True)[0], set(one))
    check('two runs with one name are refused (their gate files collide)',
          _raised(da.check_names, {'name': 'cpu8'}, {'name': 'cpu8'}), 'SystemExit')

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
    severe, fired = ['situation-normal', 'data-critical'], frozenset({'data-critical'})
    ag = da.row_agreement(rows, ra, rb, LABELS, severe, fired)
    counts = ('rows', 'fired_disagree', 'top_disagree', 'cascade_rows', 'winner_disagree')
    check('row agreement counts', {k: ag[k] for k in counts}, {
        'rows': 3, 'fired_disagree': 1, 'top_disagree': 1,
        'cascade_rows': 1, 'winner_disagree': 1,
    })
    raises('row agreement refuses an unscored clause', KeyError,
           da.row_agreement, rows, {}, rb, LABELS, severe, fired)
    check('the fired set is the caller\'s, not a default',
          da.row_agreement(rows, ra, rb, LABELS, severe, frozenset({'informative', 'data-critical'}))['fired_disagree'], 0)

    # -- winner headroom: the clause-margin bound says nothing about which
    #    CLAUSE wins a cascade, so measure how close the closest winner call is.
    #    Weights sn=1, dc=2: p 1.85, q 0.30, r 0.40, t 0.295.
    cas = lambda *cl: {'command': '; '.join(cl), 'lfm2d': {'ok': True, 'endpoint': 'cascade',
                                                            'clauses': [{'clause': x} for x in cl]}}
    mrows = [cas('p', 'q'),   # winner p by 1.55
             cas('p', 'p'),   # same text twice: both move together, never a swap
             cas('q', 'r'),   # winner r by 0.10
             cas('q', 't'),   # winner q by 0.005, the closest call — both informative
             cas('w', 'v'),   # winner v (dc 1.01) over w (informative 0.98) by 0.03
             {'command': 'r', 'lfm2d': {'ok': True, 'endpoint': 'classify'}}]
    ma = keyed({'p': s(0.05, 0.05, 0.9), 'q': s(0.8, 0.1, 0.1), 'r': s(0.7, 0.2, 0.1),
                't': s(0.705, 0.295, 0.0), 'w': s(0.51, 0.0, 0.49), 'v': s(0.33, 0.33, 0.34)})
    mb = dict(ma, **keyed({'q': s(0.8, 0.12, 0.08)}))  # q: 0.30 -> 0.28
    mg = da.row_agreement(mrows, ma, mb, LABELS, severe, fired)
    check('closest winner call, duplicate clauses excluded', round(mg['min_winner_margin'], 6), 0.005)
    check('cascade rows with a winner margin', mg['winner_margin_rows'], 4)
    check('winner calls near a boundary, by margin threshold', mg['winner_margin_below'],
          {'0.0001': 0, '0.001': 0, '0.01': 1, '0.05': 2})
    check('largest clause severity change', round(mg['max_severity_abs_diff'], 6), 0.02)
    check('that change swaps the closest call', mg['winner_disagree'], 1)
    # a swap between two clauses with the same top label changes which clause
    # is shown, not the top or fired decision; headroom for THAT is separate
    check('closest winner call against a rival with a different top',
          round(mg['min_top_changing_winner_margin'], 6), 0.03)
    check('cascade rows with a top-changing rival', mg['top_changing_winner_margin_rows'], 2)
    check('top-changing winner calls by margin threshold', mg['top_changing_winner_margin_below'],
          {'0.0001': 0, '0.001': 0, '0.01': 0, '0.05': 1})

    # -- soak file: passthrough_gate's format, cascade rows only, one side's dcs
    check('soak label is the single fired label', da.soak_label(fired), 'data-critical')
    check('a multi-label fired set cannot fill a one-score soak file',
          _raised(da.soak_label, frozenset(severe)), 'SystemExit')
    check('a fired label passthrough_gate does not read is refused',
          _raised(da.soak_label, frozenset({'situation-normal'})), 'SystemExit')
    doc = da.soak_doc(rows, ra, 'data-critical', 'kube_ordinal_v10F_candraw-e2',
                      'kube_ordinal_v9_cal', 1787497523)
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
           'requested_device': 'rocm', 'device_index': '0'})
    raises('a restart onto another GPU index is refused', ValueError,
           da.parse_startup, line + '\n' + line.replace('device_index=0', 'device_index=1'))
    raises('no startup line is a refusal, not a guess', ValueError,
           da.parse_startup, 'lfm2d: loaded model id=x')
    # a restart appends a second startup line; the FIRST must not speak for it
    check('a restart on the same device still parses',
          da.parse_startup(line + '\n' + line)['device_type'], 'gpu')
    cpu_line = line.replace('device_type=gpu backend=rocm', 'device_type=cpu backend=cpu')
    raises('startup lines that disagree are refused', ValueError,
           da.parse_startup, line + '\nrestart\n' + cpu_line)

    # -- --url must be the daemon whose log is the evidence
    served = line + '\n\x1b[32m INFO\x1b[0m lfm2d: lfm2d: will serve on tcp addr=0.0.0.0:18141'
    check('url on the logged port passes', _raised(da.check_url_matches_log,
          'http://127.0.0.1:18141', served), 'returned normally')
    check('url on another daemon\'s port is refused',
          _raised(da.check_url_matches_log, 'http://127.0.0.1:18140', served), 'SystemExit')
    check('a log with no tcp listener is refused',
          _raised(da.check_url_matches_log, 'http://127.0.0.1:18141', line), 'SystemExit')

    # -- labels come from the daemon's /v1/models, never from this file
    models = [{'id': 'router', 'kind': 'router', 'weight_hash': 'r', 'hidden_size': 1024},
              {'id': 'm', 'kind': 'classifier', 'weight_hash': 'h', 'labels': LABELS,
               'hidden_size': 1024}]
    check('classifier labels read from /v1/models', da.classifier_labels(models, 'm', 'h'), LABELS)
    check('no model with that id and hash is refused',
          _raised(da.classifier_labels, models, 'm', 'other'), 'SystemExit')
    check('a matching model without labels is refused',
          _raised(da.classifier_labels, models, 'router', 'r'), 'SystemExit')
    check('scores keyed by exactly the labels pass',
          _raised(da.check_score_labels, [{'scores': s(0.1, 0.2, 0.7)}], LABELS), 'returned normally')
    check('scores missing a label are refused',
          _raised(da.check_score_labels, [{'scores': {'informative': 1.0}}], LABELS), 'SystemExit')
    check('recorded labels that agree are used',
          da.resolve_labels(None, {'labels': LABELS}, {'labels': list(LABELS)}), LABELS)
    check('recorded labels that disagree are refused',
          _raised(da.resolve_labels, None, {'labels': LABELS}, {'labels': LABELS[::-1]}), 'SystemExit')
    check('no recorded labels and no flag is refused',
          _raised(da.resolve_labels, None, {}, {'labels': LABELS}), 'SystemExit')
    check('--labels for runs that predate recording', da.resolve_labels(LABELS, {}, {}), LABELS)
    check('--labels contradicting a recording is refused',
          _raised(da.resolve_labels, LABELS[::-1], {'labels': LABELS}, {}), 'SystemExit')

    # -- resume: the device must match, and the corpus read time is the FIRST
    dev = {'device_type': 'gpu', 'backend': 'rocm'}
    prev = {'name': 'rocm0', 'device': dev, 'meta': {'model_id': 'm'}, 'scores': {'k': {}},
            'log_read_at': 5.0}
    check('resume keeps scores, meta and the first read',
          da.resume_state(prev, dev, 'rocm0'), ({'k': {}}, {'model_id': 'm'}, 5.0))
    check('resume on another device is refused',
          _raised(da.resume_state, prev, {"device_type": "cpu", "backend": "cpu"}, 'rocm0'), 'SystemExit')
    check('resume under another name is refused (it would relabel the run)',
          _raised(da.resume_state, prev, dev, 'cpu8'), 'SystemExit')
    legacy = {k: v for k, v in prev.items() if k != 'log_read_at'}
    check('a run that predates read times stays unknown, never "now"',
          da.resume_state(legacy, dev, 'rocm0')[2], None)

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
