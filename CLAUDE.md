# lfm2d

LFM2.5 bidirectional encoders on candle. `README.md` is the what; this file
is the working context.

Workspace of two crates: `lfm2-encoder` (library, repo root) and `lfm2d`
(HTTP daemon). The daemon serves three heads today: a **severity
classifier** fine-tuned here, the **Prompt-Router**, and the
**PII-Detector**.

## Where things are

| thing | path |
|---|---|
| consumer contract (numbered invariants, never renumbered) | `docs/integration.md` |
| daemon API, deploy notes, known problems | `lfm2d/README.md` |
| the advisory hook and its tests | `lfm2d/hooks/` |
| deploy examples | `lfm2d/deploy/` — a generic k8s manifest, a quadlet unit, and a host-specific manifest beside them |
| live training head: numbers and the recipe | `training/v10/README.md` |
| its design, rulings and labeling rubric | `training/v10/PLAN.md`, `training/v10/rubric.md` |
| the training tree's map | `training/README.md` |
| checkpoint fixtures | `tests/fixtures/` — REAL Hub configs (refresh with curl from `https://huggingface.co/LiquidAI/<model>/raw/main/config.json`) plus the parity references `tests/reference/dump_*.py` regenerates |

**Adaptation source**: `candle-transformers/src/models/lfm2.rs` in a candle
checkout (659 lines, the CAUSAL branch — config, blocks and attention are
what we adapt; drop the causal mask and KV cache). Dual MIT/Apache-2.0;
keep attribution in adapted files. Upstreaming is a decision Amy makes
after reading whatever AI policy candle has at the time.

**Python reference**: HF transformers `models/lfm2/modeling_lfm2.py` is the
causal one; the bidirectional variants live in each checkpoint's `auto_map`
custom code on the Hub. READ the checkpoint's own modeling file for head
shapes before implementing a head — never guess.

## Checkpoint facts (fixture-verified — trust these over docs)

- Hybrid stack via `layer_types` (conv-heavy, interleaved full_attention).
  The 230M base has **14** layers, the 350M family 16. hidden 1024,
  16 heads / 8 kv-heads, vocab 65536.
- `"full_attn_idxs"` ships as a present key with a **null** value across
  the family — absent is not the same as null here.
- PII detector: BIOES over 40 entity types, **161 labels**, five of them
  `credential.*` (api_key, jwt, private_key, password, connection_string)
  — it is a secrets detector too.
- Prompt-Router: `Lfm2BidirForSequenceRouting`, `rule_proj_dim` 256 — it
  scores prompts against rule projections (zero-shot-shaped), NOT a fixed
  label softmax.
- rope theta appears BOTH flat (`rope_theta`) and structured
  (`rope_parameters.rope_theta`) across checkpoints;
  `Lfm2EncoderConfig::rope_theta()` resolves precedence, default 1e6.

## Two rules that keep coming back

**Read the label vocabulary at runtime, from `GET /v1/models`.** Never
hard-code a label name, count, or index anywhere — not in a consumer, not
in a test fixture's expectations, not in a doc example. The vocabulary has
changed wholesale between checkpoints and will again. `docs/integration.md`
invariants 1–3 are the contract version of this.

**The shell severity head judges shell.** Its job is operator and agent
safety — making it hard for a curious human or an agent to do something
irreversible by accident — not adversarial defense. Secret detection
belongs to the PII head and the routing suite, and isolating a context so
it only sees what it needs is the consuming system's job, not this
model's. So data-position and exfiltration shapes are out of scope for
this head; do not open a training slice for them.

## Conventions

Amy's global CLAUDE.md applies (TDD, 改善, loud failures). Verify against
fixtures, not documentation — the fixtures have contradicted plausible
assumptions repeatedly, starting on day 0. Training corpora never live in
the repo; the tools print aggregates, never raw rows.
