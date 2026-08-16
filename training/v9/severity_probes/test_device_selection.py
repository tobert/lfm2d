#!/usr/bin/env python3
"""Standalone tests for resolve_device (no pytest in .venv-train).

    .venv-train/bin/python training/v9/severity_probes/test_device_selection.py

The point of these: an explicit --device cuda that lands on cpu because the
GPU vanished is exactly the silent fallback this project refuses. And the
shadow scorer's cpu default is load-bearing -- see the flag's help text.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_candidate import VALID_DEVICES, resolve_device  # noqa: E402

FAILURES = []


def check(name, fn):
    try:
        fn()
    except AssertionError as e:
        FAILURES.append(f'{name}: {e}')
        print(f'FAIL {name}: {e}')
    else:
        print(f'ok   {name}')


def raises(exc, fn):
    try:
        fn()
    except exc:
        return
    except Exception as e:  # wrong exception type is still a failure
        raise AssertionError(f'expected {exc.__name__}, got {type(e).__name__}: {e}')
    raise AssertionError(f'expected {exc.__name__}, nothing raised')


def test_auto_picks_cuda_when_present():
    assert resolve_device('auto', cuda_available=True) == 'cuda'


def test_auto_falls_back_to_cpu():
    assert resolve_device('auto', cuda_available=False) == 'cpu'


def test_explicit_cpu_ignores_a_present_gpu():
    # the shadow scorer's whole fix: a GPU being there must not pull us onto it
    assert resolve_device('cpu', cuda_available=True) == 'cpu'


def test_explicit_cuda_without_a_gpu_raises_not_falls_back():
    raises(RuntimeError, lambda: resolve_device('cuda', cuda_available=False))


def test_explicit_cuda_with_a_gpu_is_honoured():
    assert resolve_device('cuda', cuda_available=True) == 'cuda'


def test_garbage_device_raises():
    raises(ValueError, lambda: resolve_device('rocm', cuda_available=True))
    raises(ValueError, lambda: resolve_device('', cuda_available=True))


def test_shadow_scorer_defaults_to_cpu():
    """Guards the default itself -- the bug was never in the resolver."""
    import shadow_score
    import argparse
    ap = argparse.ArgumentParser()
    # mirror shadow_score's own parser construction by invoking --help parsing
    src = Path(shadow_score.__file__).read_text()
    assert "'--device', choices=VALID_DEVICES, default='cpu'" in src, \
        'shadow_score --device default is no longer cpu'
    assert 'auto' in VALID_DEVICES and 'cpu' in VALID_DEVICES
    del ap


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            check(name, fn)
    print()
    if FAILURES:
        print(f'{len(FAILURES)} FAILED')
        sys.exit(1)
    print('all passed')
