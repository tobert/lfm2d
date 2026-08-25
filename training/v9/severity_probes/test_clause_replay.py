#!/usr/bin/env python3
"""Standalone tests for clause_replay's pure core (no pytest in .venv-train).

    python3 training/v9/severity_probes/test_clause_replay.py

The aggregation is the part worth guarding: the replay only means anything
if it reproduces the LIVE verdict path (winner-by-ordinal-severity, then the
winner clause's argmax), not some plausible variant of it. A replayer that
picked the max-data-critical-probability clause instead would read like it
works and disagree with production exactly on the interesting rows.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clause_replay import (  # noqa: E402
    aggregate_row, clause_severity, length_buckets, pick_winner, row_inputs,
    severity_weights,
)

LABELS = ['data-critical', 'informative', 'situation-normal']  # sorted, as exports are
DC, INFO, SN = 0, 1, 2
WEIGHTS = severity_weights(LABELS, ['situation-normal', 'data-critical'])

FAILURES = []


def check(name, fn):
    try:
        fn()
    except AssertionError as e:
        FAILURES.append(name)
        print(f'FAIL {name}: {e}')
    else:
        print(f'ok   {name}')


def probs(dc, info, sn):
    return [dc, info, sn]


def test_weights_are_ascending_rungs():
    # the daemon yaml lists severe labels LEAST severe first; the Nth weighs N
    assert WEIGHTS == {'situation-normal': 1.0, 'data-critical': 2.0,
                        'informative': 0.0}


def test_weights_refuse_severe_label_the_checkpoint_lacks():
    try:
        severity_weights(LABELS, ['situation-normal', 'destructive'])
    except ValueError:
        return
    raise AssertionError('accepted a severe label outside the checkpoint vocabulary')


def test_weights_refuse_empty_and_duplicate_severe_lists():
    for bad in ([], ['data-critical', 'data-critical']):
        try:
            severity_weights(LABELS, bad)
        except ValueError:
            continue
        raise AssertionError(f'accepted severe list {bad!r}')


def test_clause_severity_is_expected_ordinal_rank():
    p = probs(0.3, 0.5, 0.2)
    assert abs(clause_severity(p, LABELS, WEIGHTS) - (0.3 * 2 + 0.2 * 1)) < 1e-9


def test_pick_winner_takes_the_max():
    assert pick_winner([0.2, 0.9, 0.5]) == 1


def test_pick_winner_ties_break_to_the_earlier_clause():
    # stable-descending contract from cascade.rs, pinned by its own test;
    # the replay must reproduce it or winner indices drift on ties
    assert pick_winner([0.5, 0.9, 0.9]) == 1
    assert pick_winner([0.9, 0.5, 0.9]) == 0


def test_classify_row_argmax_is_the_verdict():
    r = aggregate_row('classify', [probs(0.1, 0.2, 0.7)], LABELS, WEIGHTS)
    assert r['top'] == 'situation-normal' and r['fired'] is False
    r = aggregate_row('classify', [probs(0.8, 0.1, 0.1)], LABELS, WEIGHTS)
    assert r['top'] == 'data-critical' and r['fired'] is True


def test_cascade_winner_is_severity_ranked_not_max_dc_prob():
    # clause B has the HIGHER data-critical probability, but clause A has the
    # higher ordinal severity (its situation-normal mass counts as rung 1).
    # Picking by max dc prob would pick B and fire; the live rule does not.
    a = probs(0.4, 0.1, 0.5)   # severity 1.3, top situation-normal
    b = probs(0.5, 0.5, 0.0)   # severity 1.0, top data-critical
    r = aggregate_row('cascade', [a, b], LABELS, WEIGHTS)
    assert r['winner'] == 0, f'winner {r["winner"]} is not the severity leader'
    assert r['top'] == 'situation-normal' and r['fired'] is False


def test_cascade_fires_when_the_winner_clause_is_dc():
    a = probs(0.1, 0.8, 0.1)   # severity 0.1, top informative
    b = probs(0.7, 0.2, 0.1)   # severity 1.5, top data-critical
    r = aggregate_row('cascade', [a, b], LABELS, WEIGHTS)
    assert r['winner'] == 1 and r['top'] == 'data-critical' and r['fired'] is True


def test_cascade_refuses_no_clauses():
    try:
        aggregate_row('cascade', [], LABELS, WEIGHTS)
    except ValueError:
        return
    raise AssertionError('an empty cascade row reported a verdict nobody scored')


def test_batch_row_fires_on_set_membership_not_ranking():
    r = aggregate_row('classify_batch',
                      [probs(0.05, 0.9, 0.05), probs(0.6, 0.2, 0.2)],
                      LABELS, WEIGHTS)
    assert r['fired'] is True and r['top'] is None, 'batch rows carry no winner'
    r = aggregate_row('classify_batch',
                      [probs(0.05, 0.9, 0.05), probs(0.1, 0.2, 0.7)],
                      LABELS, WEIGHTS)
    assert r['fired'] is False


def test_row_inputs_prefers_what_was_actually_scored():
    cascade = {'command': 'a && b', 'lfm2d': {
        'endpoint': 'cascade',
        'clauses': [{'clause': 'a', 'top': 'x'}, {'clause': 'b', 'top': 'y'}]}}
    assert row_inputs(cascade) == ('cascade', ['a', 'b'])
    cleaned = {'command': '# c\nls', 'lfm2d': {'endpoint': 'classify', 'sent': 'ls'}}
    assert row_inputs(cleaned) == ('classify', ['ls'])
    raw = {'command': 'ls', 'lfm2d': {'endpoint': 'classify'}}
    assert row_inputs(raw) == ('classify', ['ls'])


def test_length_buckets_are_uniform_and_cover_every_input():
    # the unmasked conv layers make padding a correctness bug, not an
    # optimisation -- every batch must be same-length so no pad ever appears
    lengths = [3, 1, 3, 7, 1, 3]
    groups = length_buckets(lengths)
    flat = [i for g in groups for i in g]
    assert sorted(flat) == list(range(len(lengths))), 'an input was lost'
    for g in groups:
        assert len({lengths[i] for i in g}) == 1, f'bucket {g} mixes lengths'


def test_length_buckets_keeps_original_order_within_a_bucket():
    groups = length_buckets([2, 5, 2, 5, 2])
    assert groups == [[0, 2, 4], [1, 3]]


def test_soak_rows_are_cascade_rows_only_with_per_clause_dc():
    # the pass-through gate's soak leg is defined over cascade rows; a
    # candidate replay must hand it exactly those, as dc lists, no text
    from clause_replay import soak_rows
    rows = [{'lfm2d': {'endpoint': 'cascade'}}, {'lfm2d': {'endpoint': 'classify'}},
            {'lfm2d': {}}, {'lfm2d': {'endpoint': 'cascade'}}]
    verdicts = [{'dcs': [0.1, 0.9]}, {'dcs': [0.5]}, {'dcs': [0.2]}, {'dcs': [0.3, 0.3, 0.3]}]
    assert soak_rows(rows, verdicts) == [[0.1, 0.9], [0.3, 0.3, 0.3]]


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            check(name, fn)
    print()
    if FAILURES:
        print(f'{len(FAILURES)} FAILED')
        sys.exit(1)
    print('all passed')
