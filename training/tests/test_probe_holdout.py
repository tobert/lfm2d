#!/usr/bin/env python3
"""The severity probes must never leak into the v9 training set.

Run with: python3 -m unittest discover training/tests

`training/v9/severity_probes/` is the only OUT-OF-CORPUS eval this project
has, and its whole value rests on the checkpoint never having been trained on
those exact strings. The probe set was written before any slice data existed —
but generation later reached for the same canonical commands anyway, and 14 of
62 probes turned up verbatim across the slices (`rm -rf /etc`,
`mkfs.ext4 /dev/sda1`, `> /etc/shadow`, …). Training on those would have had
score_probes.py grade a checkpoint on strings it memorised, and the number
would have looked like success.

build_v9.py quarantines them into v9_holdout.jsonl automatically. This test is
the thing that notices if that ever stops working — a hand-deletion would
regress the moment the next slice landed.
"""
import json
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
V9 = REPO / 'training' / 'v9'
TRAIN = V9 / 'v9.jsonl'
HOLDOUT = V9 / 'v9_holdout.jsonl'
PROBES = V9 / 'severity_probes' / 'probes.jsonl'


def norm(t):
    return ' '.join(str(t).split())


def load(path, key):
    if not path.exists():
        return []
    return [json.loads(l)[key] for l in path.read_text().splitlines() if l.strip()]


@unittest.skipUnless(TRAIN.exists() and PROBES.exists(), 'v9 corpus not built')
class TestProbeHoldout(unittest.TestCase):
    def setUp(self):
        self.probes = {norm(c) for c in load(PROBES, 'cmd')}
        self.train = {norm(t) for t in load(TRAIN, 'text')}
        self.held = {norm(t) for t in load(HOLDOUT, 'text')}

    def test_no_probe_text_in_training_set(self):
        leaked = sorted(self.probes & self.train)
        self.assertEqual(leaked, [], f'{len(leaked)} probe text(s) leaked into '
                                     f'v9.jsonl — the eval would grade memorisation')

    def test_holdout_contains_only_probe_texts(self):
        """A row lands in the holdout because it IS a probe, not for other reasons."""
        stray = sorted(self.held - self.probes)
        self.assertEqual(stray, [], 'holdout contains rows that are not probes')

    def test_holdout_and_training_are_disjoint(self):
        self.assertEqual(sorted(self.held & self.train), [])

    def test_probe_set_is_not_empty(self):
        """Guards the failure where probes.jsonl moves and the filter silently no-ops."""
        self.assertGreater(len(self.probes), 20)


if __name__ == '__main__':
    unittest.main()
