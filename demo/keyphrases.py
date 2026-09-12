#!/usr/bin/env python3
"""Clickable source keyphrases from ONE explicit block; Python stdlib only.

English stopword/1–3 word heuristics, not a parser or generated summary. JSON
input is an array of block snapshots; output offsets are Python character
indices into the original block (not UTF-8 bytes or JavaScript UTF-16 units).
Fenced/inline code is masked. Unmarked logs/code in prose remain a limitation.
At most 48 phrase candidates plus prose chunks are embedded; no generation.
"""
import argparse
from dataclasses import asdict
from html import escape
import json
from pathlib import Path
import re
import time

from lfm2 import Client, Passage, PASSAGE_BYTES, chunks, dot, normalize


STOPWORDS = set("""a an and are as at be been being but by can could did do does
for from had has have how i if in into is it its may might more most my no not of
on or our should so some than that the their them then there these they this those
to too us was we were what when where which while who will with would you your
need needs remains remain also only still very must about after before because
""".split())
WORDS = re.compile(r"\b[^\W\d_][\w]*(?:['’-][^\W_]+)*\b")


def select_block(blocks, block_id):
    """Keep provenance explicit; a full prompt is never a synthesis input."""
    if not block_id or not isinstance(blocks, list) or any(not isinstance(b, dict) for b in blocks):
        raise ValueError("provide a block array and an explicit --block-id")
    matches = [b for b in blocks if b.get("id") == block_id]
    if len(matches) != 1:
        raise ValueError("block id must match exactly one block")
    block = matches[0]
    if (block.get("role") not in ("user", "model")
            or block.get("kind") not in ("text", "thinking")
            or block.get("status") != "done"
            or block.get("excluded", False) is not False
            or block.get("ephemeral", False) is not False
            or block.get("content_type", "plain") not in ("plain", "markdown")
            or not isinstance(block.get("content"), str)):
        raise ValueError("select a completed, included user/model prose block")
    return block


def prose_mask(source):
    """Replace Markdown code by spaces, retaining every original offset."""
    lines = []
    fence = None
    for line in source.splitlines(keepends=True):
        marker = re.match(r"^[ \t]{0,3}(`{3,}|~{3,})(.*)$", line.rstrip("\r\n"))
        if fence:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                fence = None
            lines.append(re.sub(r"[^\r\n]", " ", line))
        elif marker:
            fence = marker[1]
            lines.append(re.sub(r"[^\r\n]", " ", line))
        else:
            lines.append(line)
    masked = "".join(lines)
    return re.sub(r"(`+)(?!`)([^`]|(?!\1)`)*?\1(?!`)",
                  lambda m: re.sub(r"[^\r\n]", " ", m[0]), masked)


def candidates(source, budget=48):
    """Balance phrase lengths and positions before spending on embeddings."""
    if type(budget) is not int or not 3 <= budget <= 96:
        raise ValueError("candidate budget must be 3–96")
    groups = [[], [], []]
    seen = set()
    run = []
    for word in WORDS.finditer(prose_mask(source)):
        if word[0].casefold() in STOPWORDS:
            run = []
            continue
        if run and not re.fullmatch(r"[ \t]+", source[run[-1].end():word.start()]):
            run = []
        run.append(word)
        run = run[-3:]
        for n in range(1, len(run) + 1):
            start, end = run[-n].start(), word.end()
            phrase = source[start:end]
            identity = " ".join(phrase.casefold().split())
            if len(identity) >= 3 and len(phrase.encode()) <= PASSAGE_BYTES and identity not in seen:
                seen.add(identity)
                groups[n - 1].append(Passage(phrase, start, end))
    quotas = [0, 0, 0]
    for _ in range(budget):
        available = [i for i in range(3) if quotas[i] < len(groups[i])]
        if not available:
            break
        chosen = min(available, key=lambda i: (quotas[i], -i))
        quotas[chosen] += 1
    selected = []
    for group, quota in zip(groups, quotas):
        if quota == 1:
            selected.append(group[len(group) // 2])
        elif quota:
            selected.extend(group[i * (len(group) - 1) // (quota - 1)] for i in range(quota))
    return sorted(selected, key=lambda p: (p.start, p.end))


def keyphrases(block, client, count=3):
    """Return up to count non-overlapping source phrases; never invent fillers."""
    block = select_block([block], block.get("id"))
    source = block["content"]
    if not source.strip() or len(source.encode()) > 8000 or type(count) is not int or not 1 <= count <= 8:
        raise ValueError("use 1–8000 UTF-8 bytes and request 1–8 keyphrases")
    phrases = candidates(source)
    if not phrases:
        raise ValueError("block has no eligible prose keyphrases")
    parts = chunks(prose_mask(source))
    documents = client.embed([p.text for p in parts], "document")
    vectors = client.embed([p.text for p in phrases], "document")
    center = normalize([sum(column) for column in zip(*documents)], len(documents[0]))
    relevance = [dot(center, vector) for vector in vectors]
    selected = []
    remaining = set(range(len(phrases)))
    while remaining and len(selected) < count:
        def score(i):
            redundancy = max((dot(vectors[i], vectors[j]) for j in selected), default=0)
            return (0.65 * relevance[i] - 0.35 * redundancy,
                    len(phrases[i].text.split()), -phrases[i].start)
        winner = max(remaining, key=score)
        selected.append(winner)
        remaining = {i for i in remaining if not (
            phrases[i].start < phrases[winner].end and phrases[winner].start < phrases[i].end)}
    return [dict(asdict(phrases[i]), block_id=block["id"], score=relevance[i]) for i in selected]


def render_html(block, phrases):
    """Render source anchors in Python, avoiding browser string-offset conversion."""
    block = select_block([block], block.get("id"))
    source = block["content"]
    for phrase in phrases:
        start, end = phrase.get("start"), phrase.get("end")
        if (phrase.get("block_id") != block["id"] or type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(source) or source[start:end] != phrase.get("text")):
            raise ValueError("keyphrase must identify an exact range in this block")
    links = " ".join(f'<a href="#phrase-{i}">{escape(p["text"])}</a>' for i, p in enumerate(phrases))
    rendered = []
    cursor = 0
    for i, phrase in sorted(enumerate(phrases), key=lambda pair: pair[1]["start"]):
        start, end = phrase["start"], phrase["end"]
        if start < cursor:
            raise ValueError("keyphrase ranges must not overlap")
        rendered.append(escape(source[cursor:start]))
        rendered.append(f'<mark id="phrase-{i}">{escape(source[start:end])}</mark>')
        cursor = end
    rendered.append(escape(source[cursor:]))
    return f'''<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>Keyphrase preview</title>
<style>
body {{ max-width: 54rem; margin: 3rem auto; padding: 0 1rem; font: 18px/1.6 system-ui; color: #172b3a; background: #fafafa; }}
nav {{ display: flex; flex-wrap: wrap; gap: .6rem; margin: 1.5rem 0; }}
a {{ padding: .3rem .8rem; border: 1px solid #426b85; border-radius: 1rem; color: #174d6b; text-decoration: none; }}
a:focus-visible {{ outline: 3px solid #174d6b; }}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; font: inherit; }}
mark {{ color: inherit; background: #e3edf3; scroll-margin: 2rem; }}
mark:target {{ background: #ffe082; outline: 2px solid #947000; }}
</style>
<h1>Block keyphrases</h1>
<p>Choose a phrase to highlight its original source in the full block below.</p>
<nav aria-label="Keyphrases">{links}</nav>
<p>Block: <code>{escape(block["id"])}</code></p>
<pre id="source">{"".join(rendered)}</pre>
</html>
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path, help="JSON array of block snapshots")
    parser.add_argument("--block-id", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8088")
    parser.add_argument("--count", type=int, default=3, help="return up to this many non-overlapping phrases")
    parser.add_argument("--html", type=Path, metavar="OUTPUT", help="write a standalone clickable source view")
    args = parser.parse_args()
    block = select_block(json.loads(args.file.read_text()), args.block_id)
    start = time.monotonic()
    client = Client(args.endpoint)
    result = keyphrases(block, client, args.count)
    if args.html:
        args.html.write_text(render_html(block, result), encoding="utf-8")
    print(json.dumps({"keyphrases": result, "seconds": time.monotonic() - start,
                      "model_id": client.model["id"], "weight_hash": client.model["weight_hash"],
                      "offset_unit": "unicode_codepoints"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
