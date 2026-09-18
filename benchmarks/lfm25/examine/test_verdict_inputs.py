"""The examiner must stand where the daemon stood.

    python3 -m unittest benchmarks/lfm25/examine/test_verdict_inputs.py
"""
import json, unittest

from verdict_inputs import prefill_for

# What rows.jsonl holds: the parsed report, keys SORTED. What the daemon wrote:
# the spec's `required` order. The two differ whenever a field that sorts early
# is emitted late.
STORED = {'a_written_last': 'because', 'first': 'one "quoted" thing', 'second': 'two', 'third': 'three'}
ORDER = ['first', 'second', 'third', 'a_written_last']


class PrefillOrder(unittest.TestCase):
    def test_a_field_written_after_the_slot_never_appears_before_it(self):
        text = prefill_for(STORED, ORDER, 'third')
        self.assertNotIn('a_written_last', text)
        self.assertNotIn('because', text)

    def test_fields_before_the_slot_come_in_emission_order_and_the_slot_is_left_open(self):
        text = prefill_for(STORED, ORDER, 'third')
        self.assertEqual(text, '{"first": "one \\"quoted\\" thing", "second": "two", "third": "')
        # closing it with a value gives back exactly the fields up to the slot
        self.assertEqual(list(json.loads(text + 'x"}')), ['first', 'second', 'third'])

    def test_the_first_field_has_nothing_before_it(self):
        self.assertEqual(prefill_for(STORED, ORDER, 'first'), '{"first": "')

    def test_a_report_that_is_not_the_specs_fields_is_refused(self):
        with self.assertRaises(ValueError):
            prefill_for({'first': 'x'}, ORDER, 'first')


if __name__ == '__main__':
    unittest.main()
