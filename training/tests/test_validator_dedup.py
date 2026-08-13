#!/usr/bin/env python3
"""Regression tests for the incoming-row validators' duplicate check.

Run with: python3 -m unittest discover training/tests

The bug this locks down (found 2026-08-13 during slice-1 generation): the
validators normalized `text` with `.lower()` before comparing, so

    git branch -d feature-y     situation-normal   (interlock refuses unmerged)
    git branch -D feature-y     data-critical      (forces past the interlock)

collided and the second row was rejected as a duplicate. In shell text case is
SEMANTIC — flags, paths, env vars — and this failure was structurally biased
against exactly the interlock-vs-force pairs that rules 11-13 turn on. It cost
a generator real rows before anyone noticed, because the row it dropped looked
like a legitimate near-duplicate.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# Both validators share the 7-key contract and the dedup logic, but NOT the
# label vocabulary: validate_incoming.py is the older v6 gen2 axis
# (informative/mutating/destructive) and validate_v8.py is the v7 recoverability
# axis (informative/situation-normal/data-critical). Each is exercised with its
# own labels so a vocabulary mismatch cannot masquerade as a dedup failure.
VALIDATORS = [
    (REPO / 'training' / 'coverage_v8' / 'validate_v8.py',
     ('situation-normal', 'data-critical')),
    (REPO / 'training' / 'validate_incoming.py',
     ('mutating', 'destructive')),
]


def row(text, label, **kw):
    r = {'text': text, 'label': label, 'verb': 'git branch', 'resource': 'feature-y',
         'contested': False, 'note': 'test row', 'author': 'unittest'}
    r.update(kw)
    return r


def run(validator, rows):
    with tempfile.NamedTemporaryFile('w', suffix='.jsonl', delete=False) as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
        path = fh.name
    try:
        p = subprocess.run([sys.executable, str(validator), path],
                           capture_output=True, text=True)
        return p.returncode, p.stdout
    finally:
        Path(path).unlink(missing_ok=True)


class TestDedupIsCaseSensitive(unittest.TestCase):
    def test_flag_case_pair_is_not_a_duplicate(self):
        """-d and -D are different commands with opposite labels."""
        for v, (safe, harsh) in VALIDATORS:
            with self.subTest(validator=v.name):
                rows = [row('git branch -d feature-y', safe),
                        row('git branch -D feature-y', harsh)]
                code, out = run(v, rows)
                self.assertNotIn('duplicate normalized text', out)
                self.assertEqual(code, 0, out)

    def test_true_duplicate_is_still_caught(self):
        """The fix must not disable duplicate detection outright."""
        for v, (safe, _) in VALIDATORS:
            with self.subTest(validator=v.name):
                rows = [row('git branch -d feature-y', safe),
                        row('git branch -d feature-y', safe)]
                code, out = run(v, rows)
                self.assertIn('duplicate normalized text', out)
                self.assertNotEqual(code, 0)

    def test_whitespace_still_normalized(self):
        """Whitespace collapsing is the part that SHOULD stay."""
        for v, (safe, _) in VALIDATORS:
            with self.subTest(validator=v.name):
                rows = [row('git branch -d feature-y', safe),
                        row('git   branch  -d   feature-y', safe)]
                code, out = run(v, rows)
                self.assertIn('duplicate normalized text', out)


class TestBuilderNormAgrees(unittest.TestCase):
    """build_v9.py carries its own norm(); keep it in step with the validators."""

    def test_norm_is_case_sensitive(self):
        import importlib.util
        rel = 'training/v9/build_v9.py'
        spec = importlib.util.spec_from_file_location('m', REPO / rel)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        self.assertNotEqual(m.norm('git branch -d x'), m.norm('git branch -D x'))
        self.assertEqual(m.norm('a  b'), m.norm('a b'))


if __name__ == '__main__':
    unittest.main()
