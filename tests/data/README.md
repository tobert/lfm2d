# tests/data

## semantic_search_eval.json

A hand-authored semantic search evaluation corpus for this crate's LFM2.5-Embedding-350M
encoder, sized for eyeballing retrieval quality on CPU without needing a real vector DB.

**Provenance — read this before trusting the numbers.** The documents and queries were
written by an LLM subagent acting as a human author would: picking topics, drafting
sentences, and manually assigning `relevant` / `hard_negatives` labels by reading the
text and reasoning about it, the same way a person building a test set would. It is
**not** model *output* used as its own ground truth — nothing here was produced by
embedding text with this encoder (or any embedding model) and harvesting nearest
neighbors as labels. That distinction matters: if the labels came from an embedding
model's own similarity judgments, a high score on this set would partly just mean "the
model agrees with itself," which is circular and proves nothing about retrieval
quality. Because the labels here come from independent human-style reading
comprehension, a high score is actual (weak) evidence the encoder's embedding space
lines up with how a person judges relevance.

That said, the corpus is small (105 documents, 35 queries) and was authored in a single
pass by one LLM, not curated by multiple people or sourced from real user queries and
real click/relevance data. Treat it as a **smoke test and regression harness**, not a
benchmark:

- Good use: "did this change to tokenization / pooling / normalization visibly hurt
  ranking on the hard negatives it used to get right?" — run it before/after a change
  and diff the results.
- Good use: sanity-checking that query/document prefixing (`query: ` / `document: `)
  is actually wired up correctly — swap the prefixes and confirm scores get worse.
- Bad use: quoting a score from this file as "our recall@k" in anything customer-facing.
  It's 105 hand-written sentences about software engineering topics, not a
  representative sample of kaijutsu's session history or kaibo's target codebases.
- Bad use: tuning the encoder or its config to maximize this file's score. That's
  overfitting to 35 queries and would make the file useless as a regression check.

### Shape

```json
{
  "description": "...",
  "generated": "2026-08-04",
  "documents": [{"id": "d001", "text": "...", "topic": "rust-ownership"}],
  "queries": [{"id": "q001", "text": "...", "relevant": ["d001"], "hard_negatives": ["d042"]}]
}
```

- **105 documents** across **21 topic values** (15 English topics + 6 Japanese, suffixed
  `-ja`), each 1–3 sentences, chosen to mirror what kaijutsu (agent/CRDT kernel routing
  and doc/session retrieval) and kaibo (semantic search over source, docs, and issue
  text) actually search: Rust ownership/concurrency/error-handling, git workflow,
  testing/TDD, observability, containers/deploy, databases, networking, security
  (including PII/secrets — this project ships a secrets detector), embeddings/ML
  itself, CI/CD, code review process, issue tracking, and config management.
- **35 queries**, phrased the way a developer actually types into a search box —
  lowercase, sometimes a question, sometimes just 2–4 keywords — each with 1–4
  `relevant` document ids and 1–3 `hard_negatives`.
- **Hard negatives share vocabulary with the query but are the wrong answer** — e.g.
  query "how do I lock a mutex in rust" (`q001`) has hard negative `d052` ("SELECT ...
  FOR UPDATE" row locking — same word "lock," wrong domain) and `d003` (borrow checker
  "mutable reference" rules — adjacent Rust concept, not the Mutex API). A corpus whose
  negatives are only topically unrelated (Rust vs. a cake recipe) doesn't test anything
  a bag-of-words match wouldn't also pass.
- **Near-duplicate pairs** test fine discrimination within a topic: `d040`/`d041` are
  identical multi-stage Dockerfile descriptions except `distroless` vs. `alpine`, and
  `d002`/`d105` both describe Rust move semantics but only `d105` adds the C++
  copy-constructor comparison — `q032` and `q033` specifically probe whether the model
  can tell the twins apart.
- **Japanese slice** (`d098`–`d103`, topics `*-ja`; queries `q034`, `q035`) exercises
  the tokenizer and embedding space on non-English input and, since `q034`/`q035` also
  list an English document as `relevant`, cross-lingual retrieval.

### Verification performed at generation time

- `python3 -c "import json; json.load(open('tests/data/semantic_search_eval.json'))"`
  — parses cleanly.
- Every `id` in `documents` is unique; every `id` in `queries` is unique.
- Every id referenced in any `relevant` or `hard_negatives` list exists in `documents`.
- No query has an id appearing in both its `relevant` and `hard_negatives` lists.
- Every query has 1–4 `relevant` ids and 1–3 `hard_negatives` ids.

If you add documents or queries by hand later, re-run the same checks.
`tests/retrieval_quality.rs` reads this file on every `cargo test` and holds the
retrieval numbers in docs/encoders.md, but it does not re-validate the schema.
