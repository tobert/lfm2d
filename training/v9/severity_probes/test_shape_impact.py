#!/usr/bin/env python3
"""Standalone tests for shape_impact's pure functions (no pytest in .venv-train).

    python3 training/v9/severity_probes/test_shape_impact.py

The ratio arithmetic is the part worth guarding: it is what stopped the
"heredocs are our dominant false-positive source" overclaim, and a metric
that silently reads 1.0 when it should read 3.0 would have let it stand.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shape_impact import (  # noqa: E402
    PLACEHOLDER, delimiter_stats, representation, shapes_in, strip_heredoc_bodies,
)

FAILURES = []


def check(name, fn):
    try:
        fn()
    except AssertionError as e:
        FAILURES.append(name)
        print(f'FAIL {name}: {e}')
    else:
        print(f'ok   {name}')


def row(cmd, top='informative', model='m'):
    return {'command': cmd, 'lfm2d': {'ok': True, 'top': top, 'model_id': model}}


def test_detects_each_shape():
    assert 'heredoc' in shapes_in("cat <<'EOF'\nhi\nEOF")
    assert 'interp_dash_c' in shapes_in("""python3 -c 'import os'""")
    assert 'pipe_into_interp' in shapes_in('echo hi | python3')
    assert shapes_in('ls -la') == set()


def test_dash_c_does_not_leak_across_command_separators():
    # `-c` belongs to grep here, not to python; a naive regex would claim it
    assert 'interp_dash_c' not in shapes_in('python3 script.py && grep -c foo bar')


def test_strip_replaces_body_but_keeps_framing():
    cmd = "cat <<'EOF'\nrm -rf /\nEOF"
    out = strip_heredoc_bodies(cmd)
    assert 'rm -rf /' not in out, 'body survived the strip'
    assert "<<'EOF'" in out and PLACEHOLDER.strip() in out, 'framing was destroyed'


def test_strip_handles_two_heredocs_in_one_statement():
    cmd = "cat <<'A'\nfirst\nA\ncat <<'B'\nsecond\nB"
    out = strip_heredoc_bodies(cmd)
    assert 'first' not in out and 'second' not in out, 'only one body stripped'


def test_strip_is_a_noop_without_a_heredoc():
    assert strip_heredoc_bodies('ls -la') == 'ls -la'


def test_delimiter_stats_separates_quoted_from_unquoted():
    q, u, words = delimiter_stats(["cat <<'PY'\nx\nPY", 'cat <<EOF\ny\nEOF'])
    assert (q, u) == (1, 1), f'expected 1 quoted / 1 unquoted, got {q}/{u}'
    assert words['PY'] == 1 and words['EOF'] == 1


def test_ratio_is_1_when_shape_is_merely_proportional():
    """The overclaim guard: 50% of traffic, 50% of firings => 1.0, not 'a driver'."""
    rows = [row("cat <<'E'\na\nE", 'data-critical'), row("cat <<'E'\nb\nE"),
            row('ls', 'data-critical'), row('pwd')]
    _, fired, rep = representation(rows)
    assert fired == 2
    assert abs(rep['heredoc']['ratio'] - 1.0) < 1e-9, rep['heredoc']['ratio']


def test_ratio_exceeds_1_when_shape_really_does_drive_firings():
    """And it must be able to say so -- otherwise it is a constant, not a metric."""
    rows = [row("cat <<'E'\na\nE", 'data-critical'), row("cat <<'E'\nb\nE", 'data-critical'),
            row('ls'), row('pwd')]
    _, _, rep = representation(rows)
    assert rep['heredoc']['ratio'] == 2.0, rep['heredoc']['ratio']


def test_ratio_below_1_when_shape_is_under_represented():
    rows = [row("cat <<'E'\na\nE"), row("cat <<'E'\nb\nE"),
            row('ls', 'data-critical'), row('pwd', 'data-critical')]
    _, _, rep = representation(rows)
    assert rep['heredoc']['ratio'] == 0.0, rep['heredoc']['ratio']


def test_no_firings_does_not_divide_by_zero():
    _, fired, rep = representation([row('ls'), row('pwd')])
    assert fired == 0 and rep['heredoc']['ratio'] is None


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            check(name, fn)
    print()
    if FAILURES:
        print(f'{len(FAILURES)} FAILED')
        sys.exit(1)
    print('all passed')
