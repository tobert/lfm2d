#!/usr/bin/env python3
"""Standalone tests for calibrate_prior's pure functions (no pytest in .venv-train).

    python3 training/v9/severity_probes/test_calibrate_prior.py
"""
import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_prior import calibrated_bias, class_prior  # noqa: E402

FAILURES = []


def check(name, fn):
    try:
        fn()
    except AssertionError as e:
        FAILURES.append(name)
        print(f'FAIL {name}: {e}')
    else:
        print(f'ok   {name}')


def write(rows):
    f = tempfile.NamedTemporaryFile('w', suffix='.jsonl', delete=False)
    for r in rows:
        f.write(json.dumps(r) + '\n')
    f.close()
    return Path(f.name)


LABELS = ['informative', 'situation-normal', 'data-critical']


def test_prior_is_in_checkpoint_label_order_not_file_order():
    """The bias vector is indexed by config id2label; a prior in Counter order
    would silently shift the correction onto the wrong classes."""
    p = write([{'label': 'data-critical'}] * 6 + [{'label': 'informative'}] * 3
              + [{'label': 'situation-normal'}] * 1)
    prior, _ = class_prior(p, LABELS)
    assert prior == [0.3, 0.1, 0.6], prior


def test_missing_class_raises_rather_than_inventing_a_prior():
    p = write([{'label': 'informative'}, {'label': 'data-critical'}])
    try:
        class_prior(p, LABELS)
    except SystemExit:
        return
    raise AssertionError('a class absent from training data must not get a silent prior')


def test_tau_zero_is_exactly_the_model_as_trained():
    b = [0.1, -0.2, 0.3]
    assert calibrated_bias(b, [0.2, 0.3, 0.5], 0.0) == b


def test_correction_favours_the_RARE_class():
    """The whole point: a 52.6% data-critical prior must be pushed DOWN
    relative to an 18% informative prior."""
    prior = [0.1844, 0.2896, 0.5259]          # v9's real training mix
    out = calibrated_bias([0.0, 0.0, 0.0], prior, 0.5)
    assert out[0] > out[1] > out[2], out
    assert out[2] > 0, 'even the majority class gets a positive offset (-log p > 0)'


def test_it_is_exactly_the_p_over_prior_correction():
    """b' = b - tau*log(prior) must equal dividing the softmax by prior^tau."""
    b = [0.4, -0.1, 0.25]
    prior = [0.2, 0.3, 0.5]
    tau = 0.5
    h = [1.3, -0.7, 0.9]                       # stand-in for h@W.T
    direct = [math.exp(hi + bi) for hi, bi in zip(h, b)]
    z = sum(direct)
    corrected = [(d / z) / p ** tau for d, p in zip(direct, prior)]
    zc = sum(corrected)
    corrected = [c / zc for c in corrected]

    nb = calibrated_bias(b, prior, tau)
    viabias = [math.exp(hi + bi) for hi, bi in zip(h, nb)]
    zb = sum(viabias)
    viabias = [v / zb for v in viabias]

    for a, c in zip(corrected, viabias):
        assert abs(a - c) < 1e-12, f'{corrected} != {viabias}'


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            check(name, fn)
    print()
    if FAILURES:
        print(f'{len(FAILURES)} FAILED')
        sys.exit(1)
    print('all passed')
