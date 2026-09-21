"""The pilot tally must refuse malformed labeler output and put each row in the
right shape bucket -- unanimous-against-intent is the one a human must read."""
import unittest

import pilot_tally as T

POOL = {'p1': {'set': 'challenge', 'family': 'destroy', 'intended': 'ask', 'gen': 'qwen'},
        'p2': {'set': 'challenge', 'family': 'destroy', 'intended': 'allow', 'gen': 'qwen'},
        'p3': {'set': 'pass', 'family': 'pass', 'intended': 'allow', 'gen': 'glm'}}
KEY = {'b1': 'p1', 'b2': 'p2', 'b3': 'p3'}


class Validation(unittest.TestCase):
    def check(self, text):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / 'x.tsv'
            p.write_text(text)
            return T.load_labels(p, set(KEY))

    def test_clean_file_has_no_problems(self):
        got, problems = self.check('b1\task\t1\tx\nb2\tallow\t-\t\nb3\tALLOW\t-\t\n')
        self.assertEqual(problems, [])
        self.assertEqual(got['b3'], 'allow')

    def test_skipped_repeated_unknown_and_bad_verdict_are_all_reported(self):
        _, problems = self.check('b1\task\nb1\tallow\nb9\tallow\nb2\tdeny\n')
        text = ' '.join(problems)
        for needle in ('repeated id b1', "unknown id 'b9'", "verdict 'deny'", 'never labelled'):
            self.assertIn(needle, text)


class Shapes(unittest.TestCase):
    def test_buckets(self):
        labels = {'a': {'b1': 'ask', 'b2': 'ask', 'b3': 'allow'},
                  'b': {'b1': 'ask', 'b2': 'ask', 'b3': 'ask'},
                  'c': {'b1': 'ask', 'b2': 'ask', 'b3': 'unsure'}}
        out = T.tally(KEY, POOL, labels, {'b': 'qwen'})
        self.assertEqual(out['shape'], {'unanimous_with_intended': 1,
                                        'unanimous_against_intended': 1, 'split': 1})
        against = [r for r in out['to_read'] if r['kind'] == 'unanimous_against_intended']
        self.assertEqual(against[0]['blind_id'], 'b2')
        self.assertEqual(against[0]['same_family_labelers'], ['b'])
        self.assertEqual(out['per_labeler']['a']['agrees_with_intended'], [2, 3])


if __name__ == '__main__':
    unittest.main()
