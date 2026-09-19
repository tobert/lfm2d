"""The tags this diagnostic prints have to mean what they say.

    python3 -m unittest benchmarks/lfm25/examine/test_replay_vs_rebuild.py
"""
import unittest

from replay_vs_rebuild import why


class Why(unittest.TestCase):
    def test_identical_prefills_have_nothing_to_report(self):
        text = '{"first": "one", "second": "'
        self.assertIsNone(why(text, text))

    def test_escaping_is_a_byte_difference_under_the_same_parse(self):
        # This is the one a rebuild can never fix, and the one that moves the
        # tokens the examiner stands behind.
        self.assertEqual(why('{"first": "caf\\u00e9", "second": "',
                             '{"first": "café", "second": "'),
                         'same parse, different bytes')

    def test_separator_style_is_also_a_byte_difference(self):
        self.assertEqual(why('{"first":"one","second":"',
                             '{"first": "one", "second": "'),
                         'same parse, different bytes')

    def test_the_sorted_key_bug_reads_as_a_field_order_difference(self):
        self.assertEqual(why('{"effect": "e", "scope": "s", "verdict": "',
                             '{"effect": "e", "reason": "r", "verdict": "'),
                         'different field order')

    def test_a_changed_value_is_its_own_tag(self):
        # What a re-rendered user turn looks like in the prefill's mirror: the
        # fields agree in shape and disagree in content.
        self.assertEqual(why('{"first": "one", "second": "',
                             '{"first": "ONE", "second": "'),
                         'different values')

    def test_unparseable_text_is_reported_not_swallowed(self):
        # A prefill that does not end at an opening quote cannot be closed with a
        # throwaway value, so it never parses. Say so rather than tagging it.
        self.assertEqual(why('{"first": ', '{"first": "'),
                         'one side is not parseable')


if __name__ == '__main__':
    unittest.main()
