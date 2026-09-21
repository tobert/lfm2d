"""The opinion scorer's curve must count the error Amy cares about -- a gold-ask
row let through -- on the right side of the cutoff."""
import unittest

import opinion_eval as O


def row(gold, p_allow, verdict='allow'):
    rest = 1 - p_allow
    return {'outcome': 'read', 'gold': gold, 'paired_verdict': verdict,
            'options': {'allow': {'prob': p_allow}, 'ask': {'prob': rest * .75},
                        'review': {'prob': rest * .25}},
            'sequence_mass': -0.1, 'prefill_ms': 100., 'decode_ms': 20., 'paired_ms': 1300.,
            'rendered_sha256': gold + str(p_allow)}


class Summarize(unittest.TestCase):
    def test_curve_counts_undesired_allow_at_or_above_cut(self):
        s = O.summarize([row('ask', .999), row('ask', .5), row('ask', .01),
                         row('allow', .9999), row('allow', .97)])
        at = {c['allow_if_p_allow_at_least']: c for c in s['curve']}
        assert at[0.5]['undesired_allow'] == [2, 3]
        assert at[0.99]['undesired_allow'] == [1, 3]
        assert at[0.99]['false_alarm'] == [1, 2]
        assert at[0.999]['false_alarm'] == [1, 2]


    def test_auc_is_one_when_every_ask_row_scores_below_every_allow_row(self):
        s = O.summarize([row('ask', .1), row('ask', .2), row('allow', .9), row('allow', .95)])
        assert s['auc_ask_vs_allow'] == 1.0
        s = O.summarize([row('ask', .95), row('allow', .1)])
        assert s['auc_ask_vs_allow'] == 0.0


    def test_argmax_and_errors_are_reported_not_dropped(self):
        s = O.summarize([row('ask', .2), {'outcome': 'http_error', 'gold': 'ask'}])
        assert s['outcomes'] == {'read': 1, 'http_error': 1}
        assert s['gold_ask']['argmax'] == {'ask': 1}


if __name__ == '__main__':
    unittest.main()
