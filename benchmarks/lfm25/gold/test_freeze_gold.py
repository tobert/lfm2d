"""The freeze must never produce a verdict by majority: unanimous or ruled, or
it refuses. A dropped row leaves, a ruling on a dropped row is ignored."""
import unittest

import freeze_gold as F

POOL = {'p1': {'command': 'a', 'set': 'challenge', 'family': 'destroy', 'intended': 'ask', 'gen': 'x'},
        'p2': {'command': 'b', 'set': 'challenge', 'family': 'destroy', 'intended': 'allow', 'gen': 'x'},
        'p3': {'command': 'c', 'set': 'pass', 'family': 'pass', 'intended': 'allow', 'gen': 'y'},
        'p4': {'command': 'd', 'set': 'pass', 'family': 'pass', 'intended': 'allow', 'gen': 'y',
               'drop_reason': 'leak'}}
KEY = {'b1': 'p1', 'b2': 'p2', 'b3': 'p3', 'b4': 'p4'}
LABELS = {'a': {'b1': 'ask', 'b2': 'ask', 'b3': 'allow', 'b4': 'allow'},
          'b': {'b1': 'ask', 'b2': 'allow', 'b3': 'allow', 'b4': 'ask'}}


class Freeze(unittest.TestCase):
    def test_split_without_ruling_refuses(self):
        gold, problems, dropped = F.freeze(KEY, POOL, LABELS, {})
        self.assertEqual([g['blind_id'] for g in gold], ['b1', 'b3'])
        self.assertEqual(len(problems), 1)
        self.assertIn('b2', problems[0])
        self.assertEqual(dropped, ['b4'])

    def test_ruling_settles_a_split_and_overrides_unanimity(self):
        gold, problems, _ = F.freeze(KEY, POOL, LABELS, {'b2': 'ask', 'b3': 'ask'})
        self.assertEqual(problems, [])
        by = {g['blind_id']: g for g in gold}
        self.assertEqual((by['b2']['verdict'], by['b2']['source']), ('ask', 'amy'))
        self.assertEqual((by['b3']['verdict'], by['b3']['source']), ('ask', 'amy'))
        self.assertEqual((by['b1']['verdict'], by['b1']['source']), ('ask', 'unanimous'))
        self.assertEqual(by['b2']['votes'], {'a': 'ask', 'b': 'allow'})

    def test_dropped_row_is_out_even_when_ruled(self):
        gold, _, dropped = F.freeze(KEY, POOL, LABELS, {'b4': 'allow', 'b2': 'allow'})
        self.assertNotIn('b4', [g['blind_id'] for g in gold])
        self.assertEqual(dropped, ['b4'])


class Rulings(unittest.TestCase):
    def test_parses_first_word_and_reports_empty_rulings(self):
        text = ('# head\n\n## c001 — x / y\n```bash\nls\n```\nnote\n\nAmy: ask, sudo\n\n'
                '## c002 — x\n```bash\nls\n```\n\nAmy: allow - fine\n\n'
                '## c003 — x\n\nAmy:\n\n## c004 — x\n\nAmy: NOPE. no\n')
        rulings, unruled = F.parse_rulings(text)
        self.assertEqual(rulings, {'c001': 'ask', 'c002': 'allow', 'c004': 'ask'})
        self.assertEqual(unruled, ['c003'])

    def test_a_non_verdict_word_is_refused(self):
        with self.assertRaises(SystemExit):
            F.parse_rulings('## c001 — x\n\nAmy: maybe\n')


if __name__ == '__main__':
    unittest.main()
