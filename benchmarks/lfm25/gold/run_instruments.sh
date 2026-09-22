#!/bin/bash
# Run the adjudicator over a frozen gold set, generative path first and then
# each opinion spec paired on the generative run's exact inputs, and score
# every run as the two instruments. Warm ROCm, our stack, like production.
#
#   GOLD=~/.local/share/lfm2-training-data/llm-gold/f9-bulk \
#   OUT=~/exomemory/lfm2d/lfm25-f9-gold-$(date +%F) \
#   benchmarks/lfm25/gold/run_instruments.sh
#
# Needs $GOLD/gold.jsonl and $GOLD/gold-as-val.jsonl (text/label, same order).
set -eu
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
GOLD="${GOLD:?directory holding gold.jsonl and gold-as-val.jsonl}"
OUT="${OUT:?output directory}"
BIN="${BIN:-$REPO/target/release/lfm2d}"
MODEL="${MODEL:-/tank/ml/models/llama.cpp/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q5_K_M.gguf}"
TOK="${TOK:-$REPO/.models/LFM2.5-8B-A1B/tokenizer.json}"
PY="${PY:-$REPO/.venv-train/bin/python}"
GEN_SPEC="${GEN_SPEC:-command-verdict-enum-v1}"
OPINION_SPECS="${OPINION_SPECS:-command-verdict-enum-v1-opinion command-verdict-opinion-v1}"
export PYTHONDONTWRITEBYTECODE=1 RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-8}"
mkdir -p "$OUT"
cp "$GOLD/gold.meta.json" "$OUT/gold.meta.json"
sha256sum "$GOLD/gold.jsonl" "$BIN" > "$OUT/inputs.sha256"
cd "$REPO/benchmarks/lfm25/prompts"
"$PY" verdict_eval.py --binary "$BIN" --model "$MODEL" --tokenizer "$TOK" \
  --prompt "$REPO/lfm2d/prompts/$GEN_SPEC.json" --data "$GOLD/gold-as-val.jsonl" \
  --ask-labels ask --out "$OUT" --name generative
"$PY" "$REPO/benchmarks/lfm25/gold/score_instruments.py" --gold "$GOLD/gold.jsonl" \
  --run "$OUT/generative/rows.jsonl" > "$OUT/generative/instruments.json"
for spec in $OPINION_SPECS; do
  "$PY" opinion_eval.py --binary "$BIN" --model "$MODEL" --tokenizer "$TOK" \
    --prompt "$REPO/lfm2d/prompts/$spec.json" --pair "$OUT/generative/rows.jsonl" \
    --out "$OUT" --name "$spec"
  "$PY" "$REPO/benchmarks/lfm25/gold/score_instruments.py" --gold "$GOLD/gold.jsonl" \
    --run "$OUT/$spec/rows.jsonl" > "$OUT/$spec/instruments.json"
done
echo DONE "$OUT"
