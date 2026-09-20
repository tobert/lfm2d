"""The chunk statistic has to name the right quantity.

    python3 -m unittest benchmarks/lfm25/examine/test_verdict_ribbon.py

Its first run reported `slot 1-8 tokens into a chunk`, and that phrasing sent
the F6 investigation at the short convolution, whose window is three tokens and
could never reach eight. The number was right and the word was wrong:
`n_tokens % chunk` is the final chunk's LENGTH, not the slot's offset inside
it. What it had found were the rows whose FINAL prefill chunk was one to eight
tokens long. The one place the old expression was also wrong is a prompt of
exactly one chunk, where it returned 0 and filed a full block with the
shortest -- which is why `test_a_full_chunk_is_not_an_empty_one` exists. `lfm25-chunk-sweep` then reproduced that
boundary on the model, and it is candle's quantized matmul taking its MMVQ
decode kernel at `b_size <= 8` (`quantized/rocm.rs:276`), not the convolution.

So the statistic reports the final chunk's LENGTH, and groups both sides by it
rather than printing a min/max range that a single outlier widens until it
covers everything.
"""
import unittest

from verdict_ribbon import chunk_effect, final_chunk_length


class FinalChunkLength(unittest.TestCase):
    def test_a_full_chunk_is_not_an_empty_one(self):
        self.assertEqual(final_chunk_length(128, 128), 128)
        self.assertEqual(final_chunk_length(256, 128), 128)

    def test_the_slot_is_the_last_token_not_the_next_one(self):
        self.assertEqual(final_chunk_length(129, 128), 1)
        self.assertEqual(final_chunk_length(135, 128), 7)
        self.assertEqual(final_chunk_length(136, 128), 8)
        self.assertEqual(final_chunk_length(137, 128), 9)

    def test_a_prompt_shorter_than_a_chunk_is_its_own_final_chunk(self):
        self.assertEqual(final_chunk_length(9, 128), 9)

    def test_a_slot_at_a_chunk_head_is_a_final_chunk_of_one(self):
        self.assertEqual(final_chunk_length(128 * 3 + 1, 128), 1)


class ChunkEffect(unittest.TestCase):
    def test_rows_are_counted_by_final_chunk_length(self):
        rows = [{'n_tokens': n} for n in (129, 130, 136, 200, 300)]
        values = [1.0, 1.0, 0.5, 0.5, 0.5]
        got = chunk_effect(rows, values, 128)
        self.assertEqual(got['off_line'], 2)
        self.assertEqual(got['on_line'], 3)
        self.assertEqual(got['off_line_by_length'], {'1': 1, '2': 1})
        # 136 -> 8, 200 -> 72, 300 -> 44.
        self.assertEqual(got['on_line_by_length'], {'8': 1, '44': 1, '72': 1})
        self.assertEqual(got['off_line_max_length'], 2)

    def test_an_off_line_row_with_a_long_final_chunk_is_reported_not_hidden(self):
        # The dangerous direction: a statistic that only looked at short chunks
        # would call this explained. It is not.
        # 129 -> 1, 301 -> 45; the modal value needs the majority to be modal.
        rows = [{'n_tokens': n} for n in (129, 301, 200, 300, 400)]
        got = chunk_effect(rows, [1.0, 1.0, 0.5, 0.5, 0.5], 128)
        self.assertEqual(got['off_line_by_length'], {'1': 1, '45': 1})
        self.assertEqual(got['off_line_max_length'], 45)
        self.assertFalse(got['explained_by_short_final_chunk'])

    def test_every_off_line_row_short_is_explained(self):
        rows = [{'n_tokens': n} for n in (129, 136, 200, 300, 400)]
        got = chunk_effect(rows, [1.0, 1.0, 0.5, 0.5, 0.5], 128)
        self.assertEqual(got['off_line_by_length'], {'1': 1, '8': 1})
        self.assertTrue(got['explained_by_short_final_chunk'])

    def test_no_row_off_the_line_is_not_an_effect(self):
        rows = [{'n_tokens': n} for n in (129, 300)]
        self.assertIsNone(chunk_effect(rows, [0.5, 0.5], 128))

    def test_the_shift_is_measured_against_the_modal_value(self):
        rows = [{'n_tokens': n} for n in (129, 200, 300, 400)]
        got = chunk_effect(rows, [1.32, 0.5, 0.5, 0.5], 128)
        self.assertAlmostEqual(got['max_shift'], 0.82, places=3)


if __name__ == '__main__':
    unittest.main()
