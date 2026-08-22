#!/usr/bin/env python3
"""Standalone tests for the clause-faithful shadow scorer's pure core
(no pytest in .venv-train; no torch in the tested surface).

    python3 training/v9/severity_probes/test_shadow_score.py

The field-naming tests exist because the old schema inverted once in
production (2026-08-16: 'v8_verdict' carrying v9's answer after the
cutover). Mislabelled data is data corruption; the self-describing fields
are pinned here so a rename cannot slip through.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shadow_score import shadow_row, is_replayable  # noqa: E402

FAILURES = []


def check(name, fn):
    try:
        fn()
    except AssertionError as e:
        FAILURES.append(name)
        print(f'FAIL {name}: {e}')
    else:
        print(f'ok   {name}')


def advisory_row(endpoint='cascade', top='situation-normal', clauses=None,
                 model='kube_ordinal_v9_cal', command='ls'):
    lf = {'ok': True, 'top': top, 'model_id': model, 'endpoint': endpoint}
    if clauses is not None:
        lf['clauses'] = clauses
    return {'ts': 1.0, 'command': command, 'lfm2d': lf}


def test_schema_fields_are_self_describing():
    row = shadow_row(
        advisory_row(clauses=[{'clause': 'ls', 'top': 'situation-normal'}]),
        replay={'top': 'situation-normal', 'fired': False, 'winner': 0},
        shadow_model='kube_ordinal_v9')
    assert row['live_model'] == 'kube_ordinal_v9_cal'
    assert row['live_verdict'] == 'situation-normal'
    assert row['shadow_model'] == 'kube_ordinal_v9'
    assert row['shadow_verdict'] == 'situation-normal'
    assert row['agree'] is True
    assert row['endpoint'] == 'cascade' and row['clause_count'] == 1


def test_agree_reflects_real_disagreement():
    row = shadow_row(
        advisory_row(top='data-critical',
                     clauses=[{'clause': 'rm x', 'top': 'data-critical'}]),
        replay={'top': 'informative', 'fired': False, 'winner': 0},
        shadow_model='cand')
    assert row['agree'] is False
    assert row['live_verdict'] == 'data-critical'
    assert row['shadow_verdict'] == 'informative'


def test_batch_rows_compare_firing_not_verdict():
    # batch rows carry no winner by design; both sides reduce to fired/not
    live = advisory_row(endpoint='classify_batch', top=None,
                        clauses=[{'clause': 'a', 'top': 'informative'},
                                 {'clause': 'b', 'top': 'data-critical'}])
    row = shadow_row(live, replay={'top': None, 'fired': True, 'winner': None},
                     shadow_model='cand')
    assert row['live_verdict'] is None and row['shadow_verdict'] is None
    assert row['live_fired'] is True and row['shadow_fired'] is True
    assert row['agree'] is True
    row = shadow_row(live, replay={'top': None, 'fired': False, 'winner': None},
                     shadow_model='cand')
    assert row['agree'] is False


def test_batch_live_firing_reads_the_recorded_clauses():
    live = advisory_row(endpoint='classify_batch', top=None,
                        clauses=[{'clause': 'a', 'top': 'informative'},
                                 {'clause': 'b', 'top': 'situation-normal'}])
    row = shadow_row(live, replay={'top': None, 'fired': False, 'winner': None},
                     shadow_model='cand')
    assert row['live_fired'] is False


def test_rows_without_replayable_text_are_skipped_not_guessed():
    assert is_replayable(advisory_row(endpoint='cascade',
                                      clauses=[{'clause': 'x', 'top': 'y'}]))
    # cascade row that somehow lost its clause text: not replayable
    assert not is_replayable(advisory_row(endpoint='cascade'))
    assert not is_replayable(advisory_row(endpoint='classify_batch'))
    # ok:false rows (circuit_open etc.) never reach the scorer
    bad = advisory_row()
    bad['lfm2d']['ok'] = False
    assert not is_replayable(bad)
    # classify rows replay the sent/command text -- always replayable
    assert is_replayable(advisory_row(endpoint='classify'))


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            check(name, fn)
    print()
    if FAILURES:
        print(f'{len(FAILURES)} FAILED')
        sys.exit(1)
    print('all passed')
