"""The examiner must stand where the daemon stood.

    python3 -m unittest benchmarks/lfm25/examine/test_verdict_inputs.py

Two ways to build the prefill, and only one of them is a replay:

REPLAY slices the bytes the daemon generated. RECONSTRUCT rebuilds them from the
parsed report, which forks -- it once put `reason` in front of the verdict on 733
rows, and it still cannot reproduce the model's own escaping. Replay is the
default; reconstruct stays for runs recorded before the raw text was saved.
"""
import json, unittest

from verdict_inputs import field_offsets, input_for_row, prefill_for, prefill_from_output

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


class Replay(unittest.TestCase):
    """Slicing the generated text, byte for byte."""

    def test_the_slot_is_left_open_at_the_opening_quote_the_model_wrote(self):
        out = '{"first": "one", "second": "two", "third": "three", "a_written_last": "because"}'
        text = prefill_from_output(out, ORDER, 'third')
        self.assertEqual(text, '{"first": "one", "second": "two", "third": "')
        self.assertTrue(out.startswith(text))

    def test_the_first_field_has_nothing_before_it(self):
        out = '{"first": "one", "second": "two", "third": "three", "a_written_last": "x"}'
        self.assertEqual(prefill_from_output(out, ORDER, 'first'), '{"first": "')

    def test_an_escaped_solidus_survives_where_a_re_render_loses_it(self):
        # THE divergence the output grammar can actually produce. It admits the
        # short escapes, so the model may write `\/` where json.dumps writes a
        # bare `/`. Both parse to the same report, so RECONSTRUCT cannot tell --
        # and the examiner then stands in front of different bytes. Paths are most
        # of what this corpus is about, so this is not a hypothetical shape.
        out = '{"first": "under \\/srv", "second": "two", "third": "three", "a_written_last": "x"}'
        report = json.loads(out)
        replay = prefill_from_output(out, ORDER, 'third')
        rebuilt = prefill_for(report, ORDER, 'third')
        self.assertEqual(report['first'], 'under /srv')
        self.assertTrue(out.startswith(replay))
        self.assertNotEqual(replay, rebuilt)
        self.assertIn('under \\/srv', replay)
        self.assertIn('under /srv', rebuilt)

    def test_a_u_escape_is_replayed_too_though_the_grammar_forbids_it(self):
        # `\uXXXX` is masked out for a schema-bearing prompt (constrain.rs: every
        # character it can express is reachable literally, and admitting it admits
        # lone surrogates). A replay does not depend on that being true.
        out = '{"first": "caf\\u00e9", "second": "two", "third": "three", "a_written_last": "x"}'
        replay = prefill_from_output(out, ORDER, 'third')
        self.assertTrue(out.startswith(replay))
        self.assertNotEqual(replay, prefill_for(json.loads(out), ORDER, 'third'))

    def test_a_field_name_quoted_inside_an_earlier_value_does_not_fool_the_scan(self):
        # The model writes about the fields it just filled; a search for the key
        # would stop inside this value. Note the ESCAPED quotes: this is one
        # string, and the scan has to walk it as one.
        out = ('{"first": "it looks like \\"third\\": \\"gotcha\\" to me", '
               '"second": "two", "third": "three", "a_written_last": "x"}')
        text = prefill_from_output(out, ORDER, 'third')
        self.assertEqual(text.count('"third": "'), 1)
        self.assertTrue(text.endswith('"third": "'))
        self.assertTrue(out.startswith(text))

    def test_a_value_ending_in_an_escaped_backslash_is_walked(self):
        out = '{"first": "ends with \\\\", "second": "two", "third": "three", "a_written_last": "x"}'
        self.assertTrue(prefill_from_output(out, ORDER, 'third').endswith('"third": "'))

    def test_compact_separators_are_replayed_as_written(self):
        out = '{"first":"one","second":"two","third":"three","a_written_last":"x"}'
        self.assertEqual(prefill_from_output(out, ORDER, 'third'),
                         '{"first":"one","second":"two","third":"')

    def test_trailing_text_after_the_object_is_not_a_field(self):
        out = '{"first": "one", "second": "two", "third": "three", "a_written_last": "x"} and then some'
        self.assertEqual(list(field_offsets(out)), ORDER)

    def test_generated_order_that_is_not_the_spec_order_is_refused(self):
        # After the schema-order fix nothing should state a different order, so a
        # run whose bytes disagree with the spec is an error, not a fallback.
        out = '{"second": "two", "first": "one", "third": "three", "a_written_last": "x"}'
        with self.assertRaises(ValueError):
            prefill_from_output(out, ORDER, 'third')

    def test_a_non_string_slot_has_no_opening_quote_to_stand_at(self):
        out = '{"first": "one", "second": true, "third": "three", "a_written_last": "x"}'
        with self.assertRaises(ValueError):
            prefill_from_output(out, ORDER, 'second')

    def test_a_boolean_before_the_slot_is_walked_not_refused(self):
        out = '{"first": "one", "second": false, "third": "three", "a_written_last": "x"}'
        self.assertEqual(prefill_from_output(out, ORDER, 'third'),
                         '{"first": "one", "second": false, "third": "')

    def test_text_that_never_opens_an_object_is_refused(self):
        with self.assertRaises(ValueError):
            field_offsets('I decline to answer.')

    def test_an_unterminated_string_is_refused(self):
        with self.assertRaises(ValueError):
            field_offsets('{"first": "one')


ROW = {'text': 'chmod -R 777 /srv', 'outcome': 'answered',
       'input': 'FACTS\nCommand:\nchmod -R 777 /srv',
       'output': '{"first": "one", "second": "two", "third": "three", "a_written_last": "x"}',
       'report': {'a_written_last': 'x', 'first': 'one', 'second': 'two', 'third': 'three'}}


class RowSource(unittest.TestCase):
    """Which bytes a row's input is built from, and what happens when they are
    not there."""

    def test_replay_uses_the_stored_input_and_output_verbatim(self):
        got = input_for_row(ROW, ORDER, 'third', reconstruct=False)
        self.assertEqual(got['input'], ROW['input'])
        self.assertEqual(got['assistant_prefill'], prefill_from_output(ROW['output'], ORDER, 'third'))

    def test_a_row_without_raw_text_is_refused_rather_than_quietly_rebuilt(self):
        old = {k: v for k, v in ROW.items() if k not in ('input', 'output')}
        with self.assertRaises(ValueError) as e:
            input_for_row(old, ORDER, 'third', reconstruct=False)
        self.assertIn('--reconstruct', str(e.exception))

    def test_a_row_with_no_stored_input_is_refused_even_if_it_has_output(self):
        no_input = {k: v for k, v in ROW.items() if k != 'input'}
        with self.assertRaises(ValueError):
            input_for_row(no_input, ORDER, 'third', reconstruct=False)


if __name__ == '__main__':
    unittest.main()
