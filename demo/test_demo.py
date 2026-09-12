"""Fast contract/algorithm tests; no checkpoints or network required."""
import math
import contextlib
import io
import unittest
from unittest.mock import patch

from lfm2 import Client, chunks, dot, extract, main, normalize, summarize


MODEL = {"id": "test", "kind": "embedder", "weight_hash": "a" * 64, "hidden_size": 2}
HEADERS = {"x-model-id": "test", "x-model-weight-hash": "a" * 64}


class DemoTests(unittest.TestCase):
    def test_crlf_paragraph_boundaries_and_offsets(self):
        source = "First.\r\n\r\nLast.\r\n"
        parts = chunks(source)
        self.assertEqual([p.text for p in parts], ["First.", "Last."])
        for part in parts:
            self.assertEqual(source[part.start:part.end], part.text)

    def test_empty_query_fails_before_any_network_call(self):
        for argv in (["lfm2", "search", ""], ["lfm2", "extract", "file", "--query", ""]):
            with patch("sys.argv", argv), patch("lfm2.Client") as client:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)
                client.assert_not_called()

    def test_malformed_generation_response_is_a_contract_error(self):
        for result in ({}, {"choices": []}, {"choices": [None]},
                       {"choices": [{"message": None}]}, []):
            with patch("lfm2.request", return_value=(result, {})):
                with self.subTest(result=result), self.assertRaises(ValueError):
                    summarize("http://localhost:2031", "test", "Source")

    def test_final_paragraph_with_trailing_newline_is_not_lost(self):
        source = "Initial idea.\n\nFinal decision is still undecided.\n"
        self.assertEqual([p.text for p in chunks(source)],
                         ["Initial idea.", "Final decision is still undecided."])

    def test_chunks_preserve_all_non_whitespace_text_and_offsets(self):
        source = "First paragraph.\n\n" + "日本語。" * 200 + "\n\nFinal decision: keep the backup."
        parts = chunks(source)
        self.assertGreater(len(parts), 3)
        previous = 0
        for part in parts:
            self.assertFalse(source[previous:part.start].strip())
            self.assertEqual(source[part.start:part.end], part.text)
            self.assertLessEqual(len(part.text.encode()), 384)
            previous = part.end
        self.assertFalse(source[previous:].strip())

    def test_normalization_and_invalid_vectors(self):
        self.assertEqual(normalize([3, 4], 2), [0.6, 0.8])
        for vector in ([0, 0], [math.nan, 1], [math.inf, 1], [1], [True, 1]):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                normalize(vector, 2)
        with self.assertRaises(ValueError):
            dot([1], [1, 2])

    def test_client_explicit_roles_normalization_and_identity(self):
        with patch.object(Client, "request", side_effect=[([MODEL], {}), ([[3, 4]], HEADERS)]) as request:
            client = Client("http://localhost:8088")
            self.assertEqual(client.embed(["hello"], "query"), [[0.6, 0.8]])
            self.assertEqual(request.call_args.args, ("/embed", {"inputs": ["hello"], "kind": "query"}))
        for body, headers in (([], HEADERS), ([[1, 2, 3]], HEADERS), ([[1, 0]], {}),
                              ([[1, 0]], dict(HEADERS, **{"x-model-weight-hash": "b" * 64}))):
            with self.subTest(body=body, headers=headers):
                with patch.object(Client, "request", side_effect=[([MODEL], {}), (body, headers)]):
                    with self.assertRaises(ValueError):
                        Client("http://localhost:8088").embed(["text"], "document")

    def test_discovery_and_input_guards(self):
        for models in ([], [MODEL, MODEL], [dict(MODEL, weight_hash="bad")]):
            with patch.object(Client, "request", return_value=(models, {})):
                with self.assertRaises(ValueError):
                    Client("http://localhost:8088")
        with patch.object(Client, "request", return_value=([MODEL], {})) as request:
            client = Client("http://localhost:8088")
            for texts, kind in ((["hello"], "typo"), (["a" * 385], "query"), ([""], "document")):
                with self.assertRaises(ValueError):
                    client.embed(texts, kind)
            self.assertEqual(request.call_count, 1)

    def test_extract_reduces_repetition_and_preserves_source_order(self):
        parts = chunks("First decision.\n\nDuplicate decision.\n\nSecond topic.")
        selected = extract(parts, [[1, 0], [1, 0], [0, 1]], 2)
        self.assertEqual([p.text for p in selected], ["First decision.", "Second topic."])

    def test_generation_refuses_truncated_or_empty_answers(self):
        for reason, content in (("length", "Partial summary"), ("stop", ""), ("tool_calls", None)):
            result = {"choices": [{"finish_reason": reason, "message": {"content": content}}]}
            with patch("lfm2.request", return_value=(result, {})):
                with self.assertRaises(ValueError):
                    summarize("http://localhost:2031", "test", "Source text.")
        result = {"choices": [{"finish_reason": "stop", "message": {"content": "Complete summary."}}]}
        with patch("lfm2.request", return_value=(result, {})):
            self.assertEqual(summarize("http://localhost:2031", "test", "Source text."), result)

    def test_preview_budget_and_word_limit(self):
        result = {"choices": [{"finish_reason": "stop", "message": {"content": "An untested hypothesis."}}]}
        with patch("lfm2.request", return_value=(result, {})) as request:
            summarize("http://localhost:2031", "test", "Source text.", "preview", 1024)
            payload = request.call_args.args[2]
            self.assertEqual(payload["max_tokens"], 1024)
            self.assertIn("Preserve uncertainty", payload["messages"][0]["content"])
        result["choices"][0]["message"]["content"] = "word " * 31
        with patch("lfm2.request", return_value=(result, {})):
            with self.assertRaisesRegex(ValueError, "30-word"):
                summarize("http://localhost:2031", "test", "Source", "preview")


if __name__ == "__main__":
    unittest.main()
