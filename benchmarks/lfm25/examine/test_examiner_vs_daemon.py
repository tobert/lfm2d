"""Matching the daemon's reading to the examiner's, without guessing.

    python3 -m unittest benchmarks/lfm25/examine/test_examiner_vs_daemon.py
"""
import unittest

from examiner_vs_daemon import compare, margin, quantile

PIECE = {'allow': 'allow', 'ask': 'ask', 'review': 'review'}


class Compare(unittest.TestCase):
    def test_the_two_sides_are_paired_by_word(self):
        daemon = [['allow', -0.1], ['ask', -2.3], ['review', -3.8]]
        examiner = {'allow': -0.12, 'ask': -2.28, 'review': -3.81}
        both, d_top, e_top = compare(daemon, examiner, PIECE)
        self.assertEqual(d_top, 'allow')
        self.assertEqual(e_top, 'allow')
        self.assertEqual(both['ask'], (-2.3, -2.28))

    def test_a_word_the_daemons_top_k_never_reached_is_left_out(self):
        # top_k is 8 over a 125k vocabulary; a word can simply not be in it, and
        # imputing a value for it would invent the number being measured.
        daemon = [['allow', -0.1], ['ask', -2.3]]
        examiner = {'allow': -0.12, 'ask': -2.28, 'review': -14.0}
        both, _, e_top = compare(daemon, examiner, PIECE)
        self.assertEqual(set(both), {'allow', 'ask'})
        self.assertEqual(e_top, 'allow')  # the examiner still ranks all three

    def test_a_flip_is_visible_as_disagreeing_argmaxes(self):
        daemon = [['ask', -0.70], ['allow', -0.71]]
        examiner = {'ask': -0.71, 'allow': -0.70}
        both, d_top, e_top = compare(daemon, examiner, PIECE)
        self.assertEqual((d_top, e_top), ('ask', 'allow'))
        self.assertAlmostEqual(margin([d for d, _ in both.values()]), 0.01, places=6)

    def test_words_sharing_a_first_token_are_refused_not_resolved(self):
        with self.assertRaises(ValueError):
            compare([['e', -0.1]], {'easy': -0.1, 'escalate': -2.0},
                    {'easy': 'e', 'escalate': 'e'})

    def test_a_piece_is_matched_exactly_never_as_a_prefix(self):
        # `a` is the first token of neither word here; a prefix match would have
        # credited one of them with the daemon's mass.
        both, d_top, _ = compare([['a', -0.05], ['ask', -2.0]],
                                 {'allow': -0.1, 'ask': -2.0}, PIECE)
        self.assertEqual(set(both), {'ask'})
        self.assertEqual(d_top, 'ask')


class Margin(unittest.TestCase):
    def test_the_margin_is_the_winners_lead_over_the_runner_up(self):
        self.assertAlmostEqual(margin([-0.1, -2.3, -3.8]), 2.2, places=6)

    def test_a_single_value_has_no_margin(self):
        self.assertIsNone(margin([-0.1]))


class Quantile(unittest.TestCase):
    def test_an_empty_sample_has_no_quantile(self):
        self.assertIsNone(quantile([], 0.5))

    def test_the_top_quantile_does_not_walk_off_the_end(self):
        self.assertEqual(quantile([1., 2., 3.], 1.0), 3.)


if __name__ == '__main__':
    unittest.main()
