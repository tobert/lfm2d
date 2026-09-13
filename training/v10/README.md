# kube_ordinal v10 — the filter (live 2026-08-25)

`kube_ordinal_v10` (candidate F-e2, weight `e90e0ba8f47e…`) replaced
`kube_ordinal_v9_cal` on 2026-08-25. Read `PLAN.md` for the design and
the rulings; this file is how every number was produced, in order.

| | v9_cal | **v10** |
|---|---|---|
| severity probes (classic / margin ≥ 0.05) | 16/23 / 14/23 | **20/23 / 18/23** |
| benign controls / benign shapes at the floor | 1/7 / 12/42 | **7/7 / 42/42** |
| soak pass-through at the zero-miss floor (11,085 rows) | 1.2% | **98.7%** |
| firing on the 16,719-row replay window | 8.71% (1,456; replay validates 99.99% vs live) | **1.30% (217)** |
| val accuracy (its own split) | — | 93.3% |

## Pipeline (regenerate in this order)

```
# slice 3: the labeled shape sample (real text stays local, 0600)
.venv-train/bin/python training/v10/shape_sample.py --model-id kube_ordinal_v9_cal --top 400 \
    --out ~/.cache/claude-hooks/v10-shape-sample-r3.jsonl
python3 training/v10/bulk_label.py --size 50 --sample ~/.cache/claude-hooks/v10-shape-sample-r3.jsonl chunk
#   ... three families label each chunk via kaibo oneshot (labeler_prompt_v10.txt attached);
#   replies go to bulk/raw/<family>_cNN.txt (shape keys + labels only)
python3 training/v10/bulk_label.py --size 50 --sample ~/.cache/claude-hooks/v10-shape-sample-r3.jsonl score
python3 training/v10/apply_rulings.py                  # rulings.json -> final label per shape

# slice 4: the training set
.venv-train/bin/python training/v10/build_real.py --model-id kube_ordinal_v9_cal --cap 30 \
    --out training/v10/real_cap30.jsonl                # structural scrub (scrub.py), leak gate
python3 training/v10/slice_severe_forms.py             # deterministic severe-forms slice
python3 training/v10/build_v10.py --full --real training/v10/real_cap30.jsonl --without-rm-slice --suffix _F
python3 training/v10/build_v10.py --full --real training/v10/real_cap30.jsonl --without-rm-slice --suffix _F --check

# train (zorak GPU, rocm7.2 wheels) -- v9's candidate recipe minus class weights
.venv-train/bin/python training/finetune_sequence_classifier.py \
    --train training/v10/train_F.jsonl --val training/v10/val_F.jsonl \
    --base .models/LFM2.5-Encoder-350M --out .models/kube_ordinal_v10F_candraw \
    --epochs 3 --batch-size 16 --lr 1e-5 --max-len 128 --seed 20260825 \
    --label-order informative,situation-normal,data-critical --class-weight none --export-every-epoch

# gate each epoch: serve it locally, then probes + benign + soak
target/release/lfm2d --classifier-dir .models/kube_ordinal_v10F_candraw-e2 --bind-addr 127.0.0.1:18132 &
(cd training/v9/severity_probes && python3 score_probes.py --url http://127.0.0.1:18132 --save ../../v10/probes_run_v10F_candraw-e2.json)
.venv-train/bin/python training/v9/severity_probes/clause_replay.py --model .models/kube_ordinal_v10F_candraw-e2 \
    --model-id kube_ordinal_v9_cal --until 1787497523 --device cuda --save-soak ~/.cache/claude-hooks/v10-soak-F-e2.json
python3 training/v10/passthrough_gate.py --url http://127.0.0.1:18132 --model-id kube_ordinal_v10F_candraw-e2 \
    --probes-run training/v10/probes_run_v10F_candraw-e2.json --save-benign training/v10/benign_run_v10F_candraw-e2.json \
    --soak-rows ~/.cache/claude-hooks/v10-soak-F-e2.json
```

Tests: `python3 training/v10/test_*.py` (shape key, bulk labeling,
rulings, scrub, build, pass-through gate) and
`.venv-train/bin/python training/v9/severity_probes/test_clause_replay.py`.

## What the candidates taught (A → F, all in one day)

| cand | data | weights | best epoch | why it did not ship |
|---|---|---|---|---|
| A | v9 + live cap 30 | inverse_freq | e1 86.9% soak | bare `echo`, `cargo check` read as dc: weighting ×6 on dc starves head shapes |
| B | same | none | **e2 PASS 97.5%** | `git branch -D` in the unseen-shape holdout → would pass through |
| C | cap 100 + rm-devfile slice | none | e2 21/23 sev | floor sinks on `rm -r <dir>` (the rm slice bleeds); 6/7 controls |
| D | B + holdout folded in | none | all fail | floor 0.04–0.08: shred / hard-reset forms untaught, the gate measured the draw |
| E | D + severe-forms slice | none | **e2 PASS 99.5%** | `git branch -D` 0/3 on live instances (dc 0.014) — short spelling never separated from `-d` |
| **F** | E + `-D`/`--force` forms | none | **e2 PASS 98.7%** | shipped: gold 10/10 shapes on live instances |

Two gate defects surfaced and were fixed on the way: the zero-miss
floor was being set by situation-normal-truth probes (`probe_truth.json`
now gives every probe a rung, `b26fff1`), and a floor set by a form the
corpus barely holds measures training noise until that form is taught
(`slice_severe_forms.py`; memory `floor-set-by-an-untaught-form-is-noise`).

## Rulings that shaped the data (Amy, 2026-08-24/25)

- real text option **(b)**: scrub structurally via the parser's argv; publish
  the scrubbed set (HF publication itself still needs her review of
  `~/.cache/claude-hooks/v10-scrub-review.txt`).
- GitHub posting via `gh` is **informative** for the classifier; posting
  policy belongs to the judge layer (`rulings.json`, prompt + rubric).
- two-axis head → v11.

## Device agreement — CPU vs ROCm (2026-09-12)

A GPU rollout of this head was held until its verdicts were checked
across devices. The hardware gate (`lfm2d/tests/device_real.rs`) shows
scores agree within 0.005 on fixtures, but that is arithmetic, not
verdicts: on 2026-08-16 torch flipped 2 of 60 verdicts moving CPU → ROCm.
So this scores the real corpus through the daemon itself: same ROCm
release binary, same weights (`e90e0ba8…`), `--device cpu --threads 8`
(production parity) vs `--device rocm`. The clauses are the gate window
(16,719 rows) plus live v10 traffic (20,630 rows) plus both probe files.

| comparison | clauses | top-label flips | max \|Δp\| (any label) | closest call (top-two margin) |
|---|---|---|---|---|
| **CPU@8 vs ROCm** | **82,430** | **0** | 2.33e-5 | 2.28e-4 (13 clauses < 1e-3, 99 < 1e-2) |
| CPU@8 vs CPU@32 (control) | 7,994 | 0 | 0.0, bit-identical | 3.43e-4 |
| ROCm vs ROCm repeat (control) | 7,989 | 0 | 0.0, bit-identical | — |

- **Hook decisions:** 0 disagreements on fired, cascade winner or top label.
  That covers all 16,719 gate-window rows (11,085 cascade) and all 20,630
  live v10 rows (17,469 cascade).
- **Gates:** the committed scorers print byte-identical output on both
  devices. `score_probes.py` passes 20/23 gating constraints.
  `passthrough_gate.py` gives a floor of 0.5427, controls 7/7, benign shapes
  45/49, and soak 98.7% of 11,085.
- **Why both gates say FAIL anyway:** it is v10's standing state, not the
  device. The committed CPU run `probes_run_v10F_candraw-e2.json` fails the
  same 3 constraints. All 4 benign-shape failures are the bare build/test
  probes added for v10.1 (`8a2c732`) after v10 shipped.
- **Why 0 flips is structural, not luck:** a flip needs two labels to cross
  the top-two margin. At the largest per-label difference measured, they can
  close at most ~4.7e-5, under the closest call's 2.28e-4, so there is ~5×
  headroom.
- **Both controls are bit-identical.** candle's CPU path does not depend on
  thread count, and ROCm repeats itself exactly. The CPU/ROCm difference is
  systematic and tiny, not noise.

What this does NOT settle:
- **Other devices and dtypes:** CUDA, Metal, other ROCm GPUs, f16 and bf16
  are all unmeasured.
- **Other checkpoints:** a head whose closest call sits under ~5e-5 could
  flip. Re-run on every new head; `margin_below` shows the exposure.
- **Torch-scored numbers:** anything scored with torch, such as
  `clause_replay.py --device cuda`, is still subject to the 08-16 drift.
- **Whether to roll out at all:** throughput under burst, and sharing the
  iGPU and its unified memory with the host llama-server, are the open
  deployment questions.

```
# three daemons, one binary, one head (a ROCm build runs CPU too)
target/release/lfm2d --classifier-dir .models/kube_ordinal_v10F_candraw-e2 --device cpu  --threads 8  --bind-addr 127.0.0.1:18140 2> cpu8.log &
target/release/lfm2d --classifier-dir .models/kube_ordinal_v10F_candraw-e2 --device rocm --threads 8  --bind-addr 127.0.0.1:18141 2> rocm.log &
target/release/lfm2d --classifier-dir .models/kube_ordinal_v10F_candraw-e2 --device cpu  --threads 32 --bind-addr 127.0.0.1:18142 2> cpu32.log &
# collect (device read from each daemon's startup line; 0600, sha256 keys, resumable)
D=training/v10/device_agreement.py; O=~/.cache/claude-hooks/device-agreement
.venv-train/bin/python $D collect --name cpu8  --url http://127.0.0.1:18140 --daemon-log cpu8.log  --expect-device cpu --out $O/cpu8.json
.venv-train/bin/python $D collect --name rocm0 --url http://127.0.0.1:18141 --daemon-log rocm.log  --expect-device gpu --out $O/rocm0.json
.venv-train/bin/python $D collect --name cpu32-sample --url http://127.0.0.1:18142 --daemon-log cpu32.log --expect-device cpu --out $O/cpu32-sample.json --sample 8000
.venv-train/bin/python $D collect --name rocm0-repeat --url http://127.0.0.1:18141 --daemon-log rocm.log  --expect-device gpu --out $O/rocm0-repeat.json --sample 8000
# compare (live rows frozen before both runs read the log) + the committed gates per device
.venv-train/bin/python $D compare $O/cpu8.json $O/rocm0.json --live-until 1789247897 --gate-dir $O/gate
(cd training/v9/severity_probes && python3 score_probes.py --results $O/gate/probes_run_rocm0.json)
python3 training/v10/passthrough_gate.py --model-id kube_ordinal_v10F_candraw-e2 \
    --probes-run $O/gate/probes_run_rocm0.json --benign-run $O/gate/benign_run_rocm0.json --soak-rows $O/gate/soak_rocm0.json
```

Aggregates: `device_agreement_2026-09-12.json`. Tests:
`.venv-train/bin/python training/v10/test_device_agreement.py` (29 checks).
CPU@8 took ~2.5 h on zorak next to production. Bench startup cost the live
hook a few timeouts, so run it when the node is quiet.

## Open

- kaijutsu's escalation volume before/after on their 10,751-row window
  (sent to kaijutsu-lead 2026-08-25: firing on the replay window 8.71% → 1.30%).
- `probe_truth.json` rungs beyond Amy's three direct rulings await her eye.
- cluster G (two-level subcommand key) kept as bias-up; `git worktree` verbs
  and kubectl verbs still share a key.
- `rm_devfile.jsonl` is built and committed but not in the shipped set.
