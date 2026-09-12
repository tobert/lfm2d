"""Source-selection and keyphrase tests; synthetic prose and fake embeddings."""
import unittest
from html.parser import HTMLParser

from keyphrases import candidates, keyphrases, prose_mask, render_html, select_block


def block(content, **fields):
    return dict(id="thought", role="model", kind="thinking", status="done",
                content=content, **fields)


class FakeClient:
    def __init__(self):
        self.calls = []

    def embed(self, texts, kind):
        self.calls.append((list(texts), kind))
        # Semantically identical phrases must compete for a single slot.
        return [[2 ** -0.5, 2 ** -0.5] if "cache" in text.lower() and "disk" in text.lower()
                else [1.0, 0.0] if any(word in text.lower() for word in ("cache", "eviction", "replacement", "policy"))
                else [0.0, 1.0]
                for text in texts]


class KeyphraseTests(unittest.TestCase):
    def test_fewer_candidates_return_fewer_phrases_without_inventing_fillers(self):
        result = keyphrases(block("Concurrency."), FakeClient(), count=8)
        self.assertEqual([p["text"] for p in result], ["Concurrency"])

    def test_html_escapes_untrusted_text_and_preserves_unicode_source_ranges(self):
        source = '日本語 🌸 <script>alert("x")</script> Cache eviction & latency.'
        start = source.index("Cache eviction")
        phrase = dict(block_id="thought", start=start, end=start + len("Cache eviction"),
                      text="Cache eviction", score=0.5)
        page = render_html(block(source), [phrase])

        class ViewParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.tags, self.links, self.source, self.mark = [], [], [], []
                self.in_source = self.in_mark = False

            def handle_starttag(self, tag, attrs):
                self.tags.append(tag)
                attrs = dict(attrs)
                if tag == "a":
                    self.links.append(attrs.get("href"))
                if tag == "pre":
                    self.in_source = True
                if tag == "mark":
                    self.in_mark = True
                    self.mark_id = attrs["id"]

            def handle_endtag(self, tag):
                if tag == "pre":
                    self.in_source = False
                if tag == "mark":
                    self.in_mark = False

            def handle_data(self, text):
                if self.in_source:
                    self.source.append(text)
                if self.in_mark:
                    self.mark.append(text)

        parsed = ViewParser()
        parsed.feed(page)
        self.assertNotIn("script", parsed.tags)
        self.assertEqual("".join(parsed.source), source)
        self.assertEqual("".join(parsed.mark), "Cache eviction")
        self.assertEqual(parsed.links, ["#" + parsed.mark_id])
        # Both labels and source excerpts are untrusted text.
        hostile = dict(phrase, start=source.index("<script>"), end=source.index(" Cache"),
                       text=source[source.index("<script>"):source.index(" Cache")])
        self.assertNotIn("<script>", render_html(block(source), [hostile]))

    def test_html_refuses_invalid_provenance_and_overlapping_ranges(self):
        target = block("Cache eviction policy.")
        phrase = dict(block_id="thought", start=0, end=14, text="Cache eviction", score=0.5)
        for changes in ({"block_id": "other"}, {"text": "invented"}, {"start": -1},
                        {"end": 1000}, {"start": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                render_html(target, [dict(phrase, **changes)])
        with self.assertRaises(ValueError):
            render_html(target, [phrase, phrase])

    def test_explicit_block_and_unrelated_prompt_invariance(self):
        target = block("The cache eviction policy needs testing. Disk latency remains uncertain.")
        noisy = [dict(target, id="system", role="system", content="instruction " * 1000),
                 dict(target, id="tool", role="tool", kind="tool_result", content="log " * 1000),
                 dict(target, id="code", content="```python\n" + "code\n" * 500 + "```"), target]
        clean_client, noisy_client = FakeClient(), FakeClient()
        clean = keyphrases(select_block([target], "thought"), clean_client)
        dirty = keyphrases(select_block(noisy, "thought"), noisy_client)
        self.assertEqual(clean, dirty)
        self.assertEqual(clean_client.calls, noisy_client.calls)
        self.assertTrue(clean)
        for item in clean:
            self.assertEqual(item["block_id"], "thought")
            self.assertEqual(target["content"][item["start"]:item["end"]], item["text"])

    def test_noneligible_ambiguous_and_missing_blocks_fail(self):
        target = block("Enough ordinary prose to select from.")
        for changes in ({"role": "system"}, {"role": "tool"}, {"kind": "tool_call"},
                        {"status": "running"}, {"excluded": True}, {"ephemeral": True},
                        {"content_type": "diff"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                select_block([dict(target, **changes)], "thought")
        for blocks, block_id in (([target], ""), ([target], "absent"), ([target, target], "thought")):
            with self.assertRaises(ValueError):
                select_block(blocks, block_id)

    def test_code_mask_preserves_offsets_and_candidate_boundaries(self):
        text = "改善（かいぜん）. Cache eviction.\n```python\npoison token\n```\nDisk latency. `secret code` stays hidden."
        masked = prose_mask(text)
        self.assertEqual(len(masked), len(text))
        self.assertNotIn("poison", masked)
        self.assertNotIn("secret", masked)
        phrases = candidates(text)
        self.assertTrue(any(p.text == "Disk latency" for p in phrases))
        self.assertFalse(any("eviction.\n" in p.text or "poison" in p.text or "secret" in p.text for p in phrases))
        for phrase in phrases:
            self.assertEqual(text[phrase.start:phrase.end], phrase.text)

    def test_budget_covers_late_phrases_and_multiple_lengths(self):
        text = ". ".join(f"topic{i} useful details" for i in range(100))
        phrases = candidates(text, budget=24)
        self.assertLessEqual(len(phrases), 24)
        self.assertEqual(phrases, candidates(text, budget=24))
        self.assertEqual({len(p.text.split()) for p in phrases}, {1, 2, 3})
        self.assertTrue(any("topic99" in p.text for p in phrases))

    def test_fenced_code_cannot_change_prose_inference(self):
        prose = "Cache eviction remains uncertain. Disk latency needs measurement."
        before, after = FakeClient(), FakeClient()
        clean = keyphrases(block(prose), before)
        prefix = "~~~python\n" + "misleading terms\n" * 20 + "~~~\n"
        dirty = keyphrases(block(prefix + prose), after)
        self.assertEqual([x["text"] for x in clean], [x["text"] for x in dirty])
        self.assertEqual(before.calls, after.calls)
        self.assertEqual([x["start"] + len(prefix) for x in clean], [x["start"] for x in dirty])

    def test_diversity_suppresses_duplicate_and_overlapping_phrases(self):
        result = keyphrases(block("Cache eviction policy. Cache replacement policy. Disk latency."), FakeClient(), count=2)
        self.assertEqual(len(result), 2)
        self.assertTrue(any("Disk" in x["text"] or "latency" in x["text"] for x in result))
        for i, a in enumerate(result):
            for b in result[i + 1:]:
                self.assertFalse(a["start"] < b["end"] and b["start"] < a["end"])

    def test_empty_code_only_and_oversized_inputs_are_explicit(self):
        for source in ("", "```\ncode only\n```", "x" * 8001):
            client = FakeClient()
            with self.assertRaises(ValueError):
                keyphrases(block(source), client)
            self.assertFalse(client.calls)


if __name__ == "__main__":
    unittest.main()
