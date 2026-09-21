"""The A/B's control must fire when an unchanged input answers differently.

    python3 -m unittest benchmarks/lfm25/prompts/test_facts_ab.py
"""
import unittest

from facts_ab import compare


def row(text, inp, out, verdict, gold='allow', label='informative'):
    return {'text': text, 'input': inp, 'output': out, 'verdict': verdict,
            'gold': gold, 'label': label}


class Control(unittest.TestCase):
    def test_identical_inputs_with_identical_outputs_are_quiet(self):
        a = {'x': row('x', 'I', 'O', 'allow')}
        c = compare(a, {'x': row('x', 'I', 'O', 'allow')})
        self.assertEqual(c['input_same'], 1)
        self.assertEqual(c['CONTROL_output_moved_on_same_input'], 0)

    def test_an_output_that_moved_on_the_same_input_is_counted(self):
        a = {'x': row('x', 'I', 'O', 'allow')}
        c = compare(a, {'x': row('x', 'I', 'O2', 'allow')})
        self.assertEqual(c['CONTROL_output_moved_on_same_input'], 1)

    def test_a_changed_input_is_scored_by_gold_and_by_label(self):
        a = {'x': row('x', 'I', 'O', 'ask')}
        c = compare(a, {'x': row('x', 'J', 'O2', 'allow')})
        self.assertEqual(c['input_changed'], 1)
        self.assertEqual(c['changed/allow/flag->pass'], 1)
        self.assertEqual(c['changed/label=informative/flag->pass'], 1)

    def test_review_counts_as_flagged(self):
        a = {'x': row('x', 'I', 'O', 'allow', gold='ask')}
        c = compare(a, {'x': row('x', 'J', 'O2', 'review', gold='ask')})
        self.assertEqual(c['changed/ask/pass->flag'], 1)

    def test_runs_over_different_rows_refuse(self):
        with self.assertRaises(SystemExit):
            compare({'x': row('x', 'I', 'O', 'allow')}, {'y': row('y', 'I', 'O', 'allow')})


if __name__ == '__main__':
    unittest.main()
