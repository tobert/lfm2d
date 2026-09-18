//! Examine one prompt inside the LFM2.5 causal model: what the model did at
//! every token and every depth, not only what it said at the end.
//!
//! One observed prefill over exact token ids yields an [`Examination`]:
//!
//! - **routing** — for each expert layer and each token, which experts the
//!   router chose and the weights their outputs were combined with.
//! - **residual_norm** — the L2 norm of the residual stream at every depth.
//! - **lens** — the logit lens. The residual at every depth and position is
//!   read through the model's own final norm and output projection, and the
//!   log-probability of each token in a caller-named set is recorded. At the
//!   last depth this is the model's actual next-token distribution; earlier
//!   depths show where in the stack that verdict forms.
//! - **top** — at chosen positions, the top-k tokens at every depth.
//!
//! Position `p` predicts token `p + 1`, as always in a causal model: the lens
//! row for the final position is the distribution over what comes next.
//!
//! Two rules carried over from `docs/field-requests.md` decision 5. Values are
//! **log-probabilities** over the FULL vocabulary (GGUF padding rows included),
//! because the interesting end of the scale is saturated in probability space.
//! And a set's `mass_logprob` is its **raw** mass, never renormalised within
//! the set: near-zero mass means the model was not choosing among those tokens
//! at all, and a renormalised read hides exactly that.
//!
//! This is an instrument, not a serving path. It synchronises with the device
//! at every layer and keeps no state; the unobserved forward the daemon serves
//! is untouched and bit-identical (see the candle-side tests).
//!
//! It examines a COLD prefill: chunks of the daemon's size, counted from token
//! 0. A warm daemon request chunks from the end of its cached prefix instead,
//! and chunk geometry is not bit-exact, so expect agreement with a warm request
//! to a tolerance, not to the bit.

use candle_core::{D, IndexOp, Tensor};
use candle_transformers::models::quantized_lfm2_moe::{Model, Observer};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const SCHEMA: &str = "lfm25-examination-v1";
pub const MAX_TOP_K: usize = 64;

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExamineSpec {
    /// Named token-id sets to follow through the lens.
    #[serde(default)]
    pub token_sets: BTreeMap<String, Vec<u32>>,
    /// Tokens to keep per depth at each of `top_k_positions`; 0 for none.
    #[serde(default)]
    pub top_k: usize,
    #[serde(default)]
    pub top_k_positions: Vec<usize>,
    /// Also record the raw router outputs for every expert, before the
    /// sigmoid and the selection bias, and that bias. Large; off unless asked.
    #[serde(default)]
    pub router_logits: bool,
    /// Read the lens only at these positions, strictly increasing. Empty reads
    /// it at every recorded position. The lens is the expensive part: one
    /// full-vocabulary projection per position per depth.
    #[serde(default)]
    pub lens_positions: Vec<usize>,
    /// Record nothing before this position. The whole prompt is still run and
    /// `tokens` still lists it all; routing, norms and the lens start here.
    /// For a batch of prompts that share a prefix, which routes identically
    /// every time in a causal model.
    #[serde(default)]
    pub record_from: usize,
}
impl ExamineSpec {
    pub fn validate(&self, n_tokens: usize, vocab: usize) -> Result<(), String> {
        for (name, ids) in &self.token_sets {
            if name.is_empty() || ids.is_empty() {
                return Err("token sets need a name and at least one token id".into());
            }
            let mut seen = std::collections::BTreeSet::new();
            for &id in ids {
                if id as usize >= vocab {
                    return Err(format!("token set {name}: id {id} is outside the vocabulary"));
                }
                if !seen.insert(id) {
                    return Err(format!("token set {name}: id {id} is listed twice"));
                }
            }
        }
        if self.top_k > MAX_TOP_K {
            return Err(format!("top_k must be 0..={MAX_TOP_K}"));
        }
        if (self.top_k == 0) != self.top_k_positions.is_empty() {
            return Err("top_k and top_k_positions are given together or not at all".into());
        }
        if self.record_from >= n_tokens {
            return Err(format!("record_from {} leaves nothing of {n_tokens} tokens", self.record_from));
        }
        let mut seen = std::collections::BTreeSet::new();
        for &p in &self.top_k_positions {
            if p >= n_tokens {
                return Err(format!("top_k position {p} is outside the {n_tokens} tokens"));
            }
            if !seen.insert(p) {
                return Err(format!("top_k position {p} is listed twice"));
            }
        }
        if self.lens_positions.windows(2).any(|w| w[0] >= w[1]) {
            return Err("lens_positions must be strictly increasing".into());
        }
        for &p in self.lens_positions.iter().chain(&self.top_k_positions) {
            if p >= n_tokens {
                return Err(format!("lens position {p} is outside the {n_tokens} tokens"));
            }
            if p < self.record_from {
                return Err(format!("position {p} is before record_from {}", self.record_from));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TokenRecord {
    pub id: u32,
    pub piece: Option<String>,
}
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LayerRecord {
    pub layer: usize,
    /// `attention` or `conv`.
    pub operator: &'static str,
    /// `moe` or `dense`.
    pub ffn: &'static str,
}
#[derive(Clone, Debug, Serialize)]
pub struct RoutingRecord {
    /// The model's layer index. Its residual is depth `layer + 1`.
    pub layer: usize,
    pub n_experts: usize,
    /// `[position][slot]`, in the router's own order.
    pub experts: Vec<Vec<u32>>,
    /// `[position][slot]`: the normalised, unbiased combine weights.
    pub weights: Vec<Vec<f32>>,
    /// `[position][expert]`, only with [`ExamineSpec::router_logits`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logits: Option<Vec<Vec<f32>>>,
    /// `[expert]`, with `logits`: this router's static selection bias. An
    /// expert is chosen by `sigmoid(logit) + bias` and weighted without the
    /// bias, so the two together give the margin each choice was made by.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selection_bias: Option<Vec<f32>>,
}
#[derive(Clone, Debug, Serialize)]
pub struct SetLens {
    pub tokens: Vec<TokenRecord>,
    /// The absolute token positions the lens was read at. Index `k` below is
    /// `positions[k]`.
    pub positions: Vec<usize>,
    /// `[depth][k][token]`, full-vocabulary log-probabilities.
    pub logprob: Vec<Vec<Vec<f32>>>,
    /// `[depth][k]`: log of the set's raw probability mass.
    pub mass_logprob: Vec<Vec<f32>>,
}
#[derive(Clone, Debug, Serialize)]
pub struct TopToken {
    pub id: u32,
    pub piece: Option<String>,
    pub logprob: f32,
}
#[derive(Clone, Debug, Serialize)]
pub struct TopRecord {
    pub position: usize,
    /// `[depth][rank]`, most probable first; ties broken by token id.
    pub by_depth: Vec<Vec<TopToken>>,
}
#[derive(Clone, Debug, Serialize)]
pub struct Examination {
    pub schema: &'static str,
    pub tokens: Vec<TokenRecord>,
    pub layers: Vec<LayerRecord>,
    /// Names the depth axis: `embedding`, then one entry per layer. Every
    /// `[depth]` index below is an index into this.
    pub depths: Vec<String>,
    /// First recorded position. Every `[position]` index in `residual_norm`
    /// and `routing` is `record_from + index`; `tokens` is always the whole
    /// prompt.
    pub record_from: usize,
    /// `[depth][position]`.
    pub residual_norm: Vec<Vec<f32>>,
    /// One per expert layer, in layer order.
    pub routing: Vec<RoutingRecord>,
    pub lens: BTreeMap<String, SetLens>,
    pub top: Vec<TopRecord>,
}

fn msg(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(e.to_string())
}

/// log(sum(exp(x))) in f64. The raw mass of a token set from its members'
/// log-probabilities.
fn logsumexp(values: &[f32]) -> f32 {
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = values.iter().map(|&v| (v as f64 - max).exp()).sum();
    (max + sum.ln()) as f32
}

fn top_k(row: &[f32], k: usize) -> candle_core::Result<Vec<(u32, f32)>> {
    // total_cmp ranks NaN above everything: it would be crowned, not caught.
    if row.iter().any(|v| v.is_nan()) {
        return Err(msg("NaN log-probability in a top-k row"));
    }
    let mut order: Vec<u32> = (0..row.len() as u32).collect();
    let by = |a: &u32, b: &u32| {
        row[*b as usize]
            .total_cmp(&row[*a as usize])
            .then(a.cmp(b))
    };
    let k = k.min(order.len());
    if k < order.len() {
        order.select_nth_unstable_by(k, by);
        order.truncate(k);
    }
    order.sort_unstable_by(by);
    Ok(order.into_iter().map(|id| (id, row[id as usize])).collect())
}

struct Collector<'a> {
    model: &'a Model,
    spec: &'a ExamineSpec,
    /// Every set's ids, concatenated in `spec.token_sets` order, on device.
    gather: Option<Tensor>,
    /// Absolute position of the current chunk's first token.
    chunk_start: usize,
    residual_norm: Vec<Vec<f32>>,
    /// `[set][depth][position][token]`, sets in `spec.token_sets` order.
    set_logprob: Vec<Vec<Vec<Vec<f32>>>>,
    /// `[top_k position][depth]`.
    top: Vec<Vec<Vec<(u32, f32)>>>,
    routing: BTreeMap<usize, RoutingRecord>,
}
impl Collector<'_> {
    fn read(&mut self, depth: usize, x: &Tensor) -> candle_core::Result<()> {
        let (_, seq, _) = x.dims3()?;
        let skip = self.skip(seq);
        if skip == seq {
            // Validation put every lens and top-k position at or after
            // `record_from`, so an unrecorded chunk has nothing to read.
            return Ok(());
        }
        let norms = x.sqr()?.sum(D::Minus1)?.sqrt()?.i(0)?.to_vec1::<f32>()?;
        if norms.iter().any(|n| !n.is_finite()) {
            return Err(msg(format!("non-finite residual at depth {depth}")));
        }
        self.residual_norm[depth].extend(&norms[skip..]);
        // Chunk-local rows the lens reads: the chosen positions that fall in
        // this chunk, or every recorded row.
        let chunk = self.chunk_start..self.chunk_start + seq;
        let lens_rows: Vec<u32> = if self.gather.is_none() {
            Vec::new()
        } else if self.spec.lens_positions.is_empty() {
            (skip as u32..seq as u32).collect()
        } else {
            self.spec
                .lens_positions
                .iter()
                .filter(|p| chunk.contains(*p))
                .map(|p| (p - self.chunk_start) as u32)
                .collect()
        };
        let wanted: Vec<(usize, usize)> = self
            .spec
            .top_k_positions
            .iter()
            .enumerate()
            .filter(|(_, p)| (self.chunk_start..self.chunk_start + seq).contains(*p))
            .map(|(slot, p)| (slot, p - self.chunk_start))
            .collect();
        if let (Some(ids), false) = (&self.gather, lens_rows.is_empty()) {
            // Project only the rows that are read: the projection is
            // (rows x vocabulary) and dominates the cost of an examination.
            let n = lens_rows.len();
            let pick = Tensor::from_vec(lens_rows, n, x.device())?;
            let logits = self.model.project(&x.index_select(&pick, 1)?)?;
            let logprob = candle_nn::ops::log_softmax(&logits, D::Minus1)?;
            let rows = logprob.i(0)?.index_select(ids, 1)?.to_vec2::<f32>()?;
            let mut at = 0;
            for (set, ids) in self.spec.token_sets.values().enumerate() {
                for row in &rows {
                    let values = row[at..at + ids.len()].to_vec();
                    if values.iter().any(|v| v.is_nan()) {
                        return Err(msg(format!("NaN log-probability at depth {depth}")));
                    }
                    self.set_logprob[set][depth].push(values);
                }
                at += ids.len();
            }
        }
        for (slot, local) in wanted {
            let logits = self.model.project(&x.narrow(1, local, 1)?)?;
            let row = candle_nn::ops::log_softmax(&logits, D::Minus1)?
                .i((0, 0))?
                .to_vec1::<f32>()?;
            self.top[slot][depth] = top_k(&row, self.spec.top_k)?;
        }
        Ok(())
    }

    /// Rows at the head of this chunk that fall before `record_from`.
    fn skip(&self, seq: usize) -> usize {
        self.spec.record_from.saturating_sub(self.chunk_start).min(seq)
    }
}
impl Observer for Collector<'_> {
    fn embedding(&mut self, x: &Tensor) -> candle_core::Result<()> {
        self.read(0, x)
    }
    fn residual(&mut self, layer: usize, x: &Tensor) -> candle_core::Result<()> {
        self.read(layer + 1, x)
    }
    fn routing(
        &mut self,
        layer: usize,
        logits: &Tensor,
        ids: &Tensor,
        weights: &Tensor,
    ) -> candle_core::Result<()> {
        let record = self
            .routing
            .get_mut(&layer)
            .ok_or_else(|| msg(format!("layer {layer} routed but has no experts")))?;
        let skip = self
            .spec
            .record_from
            .saturating_sub(self.chunk_start)
            .min(ids.dims2()?.0);
        if skip == ids.dims2()?.0 {
            return Ok(());
        }
        record.experts.extend(ids.to_vec2::<u32>()?.split_off(skip));
        record.weights.extend(weights.to_vec2::<f32>()?.split_off(skip));
        if let Some(all) = &mut record.logits {
            all.extend(logits.to_vec2::<f32>()?.split_off(skip));
        }
        Ok(())
    }
}

/// Examine `tokens` from a fresh state, prefilling in the daemon's chunk size
/// (see the module docs on cold against warm). `piece` names a token id for the
/// record; the library never needs a tokenizer.
pub fn examine(
    model: &Model,
    tokens: &[u32],
    spec: &ExamineSpec,
    piece: impl Fn(u32) -> Option<String>,
) -> Result<Examination, String> {
    examine_chunked(model, tokens, spec, piece, crate::adjudicator::CHUNK)
}

fn examine_chunked(
    model: &Model,
    tokens: &[u32],
    spec: &ExamineSpec,
    piece: impl Fn(u32) -> Option<String>,
    chunk: usize,
) -> Result<Examination, String> {
    if tokens.is_empty() || tokens.len() > model.context_length() {
        return Err(format!(
            "examine needs 1..={} tokens, got {}",
            model.context_length(),
            tokens.len()
        ));
    }
    spec.validate(tokens.len(), model.vocab_size())?;
    let kinds = model.layers();
    let depths = kinds.len() + 1;
    let all_ids: Vec<u32> = spec.token_sets.values().flatten().copied().collect();
    let mut state = model.new_state();
    let mut collector = Collector {
        model,
        spec,
        gather: if all_ids.is_empty() {
            None
        } else {
            let n = all_ids.len();
            Some(Tensor::from_vec(all_ids, n, model.device()).map_err(|e| e.to_string())?)
        },
        chunk_start: 0,
        residual_norm: vec![Vec::with_capacity(tokens.len()); depths],
        set_logprob: vec![vec![Vec::with_capacity(tokens.len()); depths]; spec.token_sets.len()],
        top: vec![vec![Vec::new(); depths]; spec.top_k_positions.len()],
        routing: BTreeMap::new(),
    };
    for (layer, kind) in kinds.iter().enumerate() {
        let Some((n_experts, _)) = kind.experts else {
            continue;
        };
        let selection_bias = match (spec.router_logits, model.router_bias(layer)) {
            (false, _) => None,
            (true, Some(bias)) => Some(bias.to_vec1::<f32>().map_err(|e| e.to_string())?),
            (true, None) => return Err(format!("expert layer {layer} has no selection bias")),
        };
        collector.routing.insert(
            layer,
            RoutingRecord {
                layer,
                n_experts,
                experts: Vec::new(),
                weights: Vec::new(),
                logits: spec.router_logits.then(Vec::new),
                selection_bias,
            },
        );
    }
    for part in tokens.chunks(chunk) {
        model
            .forward_observed(part, &mut state, &mut collector)
            .map_err(|e| e.to_string())?;
        collector.chunk_start += part.len();
    }
    let Collector {
        residual_norm,
        set_logprob,
        top,
        routing,
        ..
    } = collector;
    let token = |id: u32| TokenRecord {
        id,
        piece: piece(id),
    };
    let lens = spec
        .token_sets
        .iter()
        .zip(set_logprob)
        .map(|((name, ids), logprob)| {
            let mass_logprob = logprob
                .iter()
                .map(|depth| depth.iter().map(|row| logsumexp(row)).collect())
                .collect();
            (
                name.clone(),
                SetLens {
                    tokens: ids.iter().map(|&id| token(id)).collect(),
                    positions: if spec.lens_positions.is_empty() {
                        (spec.record_from..tokens.len()).collect()
                    } else {
                        spec.lens_positions.clone()
                    },
                    logprob,
                    mass_logprob,
                },
            )
        })
        .collect();
    let top = spec
        .top_k_positions
        .iter()
        .zip(top)
        .map(|(&position, by_depth)| TopRecord {
            position,
            by_depth: by_depth
                .into_iter()
                .map(|rank| {
                    rank.into_iter()
                        .map(|(id, logprob)| TopToken {
                            id,
                            piece: piece(id),
                            logprob,
                        })
                        .collect()
                })
                .collect(),
        })
        .collect();
    Ok(Examination {
        schema: SCHEMA,
        tokens: tokens.iter().map(|&id| token(id)).collect(),
        layers: kinds
            .iter()
            .enumerate()
            .map(|(layer, k)| LayerRecord {
                layer,
                operator: if k.attention { "attention" } else { "conv" },
                ffn: if k.experts.is_some() { "moe" } else { "dense" },
            })
            .collect(),
        depths: std::iter::once("embedding".to_string())
            .chain((0..kinds.len()).map(|l| format!("layer {l}")))
            .collect(),
        record_from: spec.record_from,
        residual_norm,
        routing: routing.into_values().collect(),
        lens,
        top,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // conv, attention, conv; layer 0 dense, then 3 experts choosing 2;
    // hidden 8, vocabulary 16, context 32.
    fn tiny() -> Model {
        let mut f = std::io::Cursor::new(include_bytes!("../tests/fixtures/lfm2-moe/tiny.gguf"));
        let ct = candle_core::quantized::gguf_file::Content::read(&mut f).unwrap();
        Model::from_gguf(ct, &mut f, &candle_core::Device::Cpu).unwrap()
    }
    const TOKENS: [u32; 7] = [1, 2, 3, 4, 5, 6, 7];
    fn name(id: u32) -> Option<String> {
        Some(format!("<{id}>"))
    }
    fn spec() -> ExamineSpec {
        ExamineSpec {
            token_sets: BTreeMap::from([
                ("pair".to_string(), vec![3, 9]),
                ("everything".to_string(), (0..16).collect()),
            ]),
            top_k: 4,
            top_k_positions: vec![6, 2],
            router_logits: true,
            ..Default::default()
        }
    }
    /// The model's real next-token log-probabilities after `tokens`, computed
    /// on the host in f64 from the ordinary unobserved forward.
    fn reference_logprobs(model: &Model, tokens: &[u32]) -> Vec<f64> {
        let logits: Vec<f32> = model
            .forward(tokens, &mut model.new_state())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let z = max + logits.iter().map(|&l| (l as f64 - max).exp()).sum::<f64>().ln();
        logits.iter().map(|&l| l as f64 - z).collect()
    }

    #[test]
    fn last_depth_of_the_lens_is_the_models_real_distribution_at_every_position() {
        let model = tiny();
        let e = examine(&model, &TOKENS, &spec(), name).unwrap();
        let last = e.depths.len() - 1;
        for position in 0..TOKENS.len() {
            let want = reference_logprobs(&model, &TOKENS[..=position]);
            let got = &e.lens["everything"].logprob[last][position];
            for (id, g) in got.iter().enumerate() {
                assert!(
                    (*g as f64 - want[id]).abs() < 1e-4,
                    "position {position} token {id}: {g} vs {}",
                    want[id]
                );
            }
            let pair = &e.lens["pair"].logprob[last][position];
            assert!((pair[0] as f64 - want[3]).abs() < 1e-4);
            assert!((pair[1] as f64 - want[9]).abs() < 1e-4);
        }
    }

    #[test]
    fn set_mass_is_raw_not_renormalised() {
        let model = tiny();
        let e = examine(&model, &TOKENS, &spec(), name).unwrap();
        for depth in 0..e.depths.len() {
            for position in 0..TOKENS.len() {
                // The whole vocabulary holds all the mass ...
                let all = e.lens["everything"].mass_logprob[depth][position];
                assert!(all.abs() < 1e-4, "depth {depth} position {position}: {all}");
                // ... and two tokens of sixteen hold what they hold: the sum of
                // their probabilities, well short of all of it.
                let pair = &e.lens["pair"];
                let sum: f64 = pair.logprob[depth][position]
                    .iter()
                    .map(|&l| (l as f64).exp())
                    .sum();
                let mass = pair.mass_logprob[depth][position] as f64;
                assert!((mass.exp() - sum).abs() < 1e-6, "{mass} vs {sum}");
                assert!(mass < -1e-3, "a 2-of-16 set cannot hold all the mass: {mass}");
            }
        }
    }

    #[test]
    fn record_is_shaped_by_the_model_and_names_its_own_axes() {
        let model = tiny();
        let e = examine(&model, &TOKENS, &spec(), name).unwrap();
        assert_eq!(e.schema, SCHEMA);
        assert_eq!(e.tokens.len(), 7);
        assert_eq!(e.tokens[2], TokenRecord { id: 3, piece: Some("<3>".into()) });
        assert_eq!(e.depths, vec!["embedding", "layer 0", "layer 1", "layer 2"]);
        assert_eq!(
            e.layers
                .iter()
                .map(|l| (l.layer, l.operator, l.ffn))
                .collect::<Vec<_>>(),
            vec![(0, "conv", "dense"), (1, "attention", "moe"), (2, "conv", "moe")]
        );
        assert_eq!(e.residual_norm.len(), 4);
        assert!(e.residual_norm.iter().all(|d| d.len() == 7));
        assert!(e.residual_norm.iter().flatten().all(|n| *n > 0.));
        assert_eq!(e.routing.iter().map(|r| r.layer).collect::<Vec<_>>(), vec![1, 2]);
        for r in &e.routing {
            assert_eq!(r.n_experts, 3);
            assert_eq!((r.experts.len(), r.weights.len()), (7, 7));
            assert!(r.experts.iter().all(|row| row.len() == 2 && row.iter().all(|&x| x < 3)));
            let (logits, bias) = (r.logits.as_ref().unwrap(), r.selection_bias.as_ref().unwrap());
            assert!(logits.iter().all(|row| row.len() == 3) && bias.len() == 3);
            // A reader can re-derive every choice and weight from the record.
            for ((l, chosen), w) in logits.iter().zip(&r.experts).zip(&r.weights) {
                let sig = |e: usize| 1. / (1. + (-l[e]).exp());
                let mut order = vec![0usize, 1, 2];
                order.sort_by(|a, b| (sig(*b) + bias[*b]).total_cmp(&(sig(*a) + bias[*a])));
                let mut got: Vec<usize> = chosen.iter().map(|&e| e as usize).collect();
                got.sort();
                order.truncate(2);
                order.sort();
                assert_eq!(got, order, "layer {}", r.layer);
                let total: f32 = chosen.iter().map(|&e| sig(e as usize)).sum::<f32>() + 1e-6;
                for (&e, got) in chosen.iter().zip(w) {
                    assert!((got - sig(e as usize) / total).abs() < 1e-5);
                }
            }
        }
        for set in e.lens.values() {
            assert_eq!(set.logprob.len(), 4);
            assert!(set.logprob.iter().all(|d| d.len() == 7));
            assert!(set.logprob.iter().flatten().all(|row| row.len() == set.tokens.len()));
        }
    }

    #[test]
    fn top_tokens_are_the_models_own_ranking_at_the_asked_positions() {
        let model = tiny();
        let e = examine(&model, &TOKENS, &spec(), name).unwrap();
        assert_eq!(e.top.iter().map(|t| t.position).collect::<Vec<_>>(), vec![6, 2]);
        for record in &e.top {
            assert_eq!(record.by_depth.len(), 4);
            let want = reference_logprobs(&model, &TOKENS[..=record.position]);
            let mut order: Vec<usize> = (0..16).collect();
            order.sort_by(|a, b| want[*b].total_cmp(&want[*a]).then(a.cmp(b)));
            let last = record.by_depth.last().unwrap();
            assert_eq!(
                last.iter().map(|t| t.id as usize).collect::<Vec<_>>(),
                order[..4].to_vec()
            );
            assert_eq!(last[0].piece, Some(format!("<{}>", order[0])));
            for depth in &record.by_depth {
                assert_eq!(depth.len(), 4);
                assert!(depth.windows(2).all(|w| w[0].logprob >= w[1].logprob));
            }
        }
    }

    #[test]
    fn chunk_size_changes_nothing_a_reader_would_see() {
        // The fixture's context is shorter than the daemon's chunk, so drive
        // the chunk boundary directly. Offsets into the record are absolute.
        let model = tiny();
        let whole = examine_chunked(&model, &TOKENS, &spec(), name, 32).unwrap();
        for chunk in [1, 2, 3] {
            let parts = examine_chunked(&model, &TOKENS, &spec(), name, chunk).unwrap();
            for (a, b) in whole.routing.iter().zip(&parts.routing) {
                assert_eq!(a.experts, b.experts, "chunk {chunk}");
            }
            let close = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-4);
            for (a, b) in whole.residual_norm.iter().zip(&parts.residual_norm) {
                assert!(a.len() == b.len() && close(a, b), "chunk {chunk}");
            }
            for (a, b) in whole.lens["pair"].logprob.iter().zip(&parts.lens["pair"].logprob) {
                assert_eq!(a.len(), b.len());
                assert!(a.iter().zip(b).all(|(x, y)| close(x, y)), "chunk {chunk}");
            }
            for (a, b) in whole.lens["pair"].mass_logprob.iter().zip(&parts.lens["pair"].mass_logprob) {
                assert!(a.len() == b.len() && close(a, b), "chunk {chunk}");
            }
            for (a, b) in whole.top.iter().zip(&parts.top) {
                let ids = |t: &TopRecord| -> Vec<Vec<u32>> {
                    t.by_depth.iter().map(|d| d.iter().map(|x| x.id).collect()).collect()
                };
                assert_eq!(ids(a), ids(b), "chunk {chunk} position {}", a.position);
                for (da, db) in a.by_depth.iter().zip(&b.by_depth) {
                    let values = |d: &[TopToken]| d.iter().map(|x| x.logprob).collect::<Vec<_>>();
                    assert!(close(&values(da), &values(db)), "chunk {chunk}");
                }
            }
        }
    }

    fn close(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-4)
    }

    #[test]
    fn a_lens_read_at_chosen_positions_equals_the_full_lens_there() {
        let model = tiny();
        let full = examine(&model, &TOKENS, &spec(), name).unwrap();
        let mut chosen = spec();
        chosen.lens_positions = vec![2, 6];
        // Chunks of 2 put the two positions in different chunks.
        for chunk in [2, 32] {
            let part = examine_chunked(&model, &TOKENS, &chosen, name, chunk).unwrap();
            for set in ["pair", "everything"] {
                assert_eq!(full.lens[set].positions, (0..7).collect::<Vec<_>>());
                assert_eq!(part.lens[set].positions, vec![2, 6]);
                for depth in 0..full.depths.len() {
                    assert_eq!(part.lens[set].logprob[depth].len(), 2);
                    for (k, &p) in [2usize, 6].iter().enumerate() {
                        assert!(close(
                            &full.lens[set].logprob[depth][p],
                            &part.lens[set].logprob[depth][k]
                        ));
                        assert!(close(
                            &[full.lens[set].mass_logprob[depth][p]],
                            &[part.lens[set].mass_logprob[depth][k]]
                        ));
                    }
                }
            }
            // Everything that is not the lens is untouched by the choice.
            assert_eq!(part.residual_norm[3].len(), 7);
            assert_eq!(part.routing[0].experts, full.routing[0].experts);
        }
    }

    #[test]
    fn recording_from_a_position_keeps_exactly_the_tail() {
        let model = tiny();
        let mut whole = spec();
        whole.top_k_positions = vec![6];
        let full = examine(&model, &TOKENS, &whole, name).unwrap();
        let mut tail = whole.clone();
        tail.record_from = 4;
        for chunk in [3, 32] {
            let part = examine_chunked(&model, &TOKENS, &tail, name, chunk).unwrap();
            assert_eq!((full.record_from, part.record_from), (0, 4));
            // The tokens are the whole prompt either way: the axis stays absolute.
            assert_eq!(part.tokens, full.tokens);
            for (a, b) in full.residual_norm.iter().zip(&part.residual_norm) {
                assert!(close(&a[4..], b), "chunk {chunk}");
            }
            for (a, b) in full.routing.iter().zip(&part.routing) {
                assert_eq!(a.experts[4..], b.experts[..], "chunk {chunk}");
                assert_eq!(b.weights.len(), 3);
                assert_eq!(b.logits.as_ref().unwrap().len(), 3);
            }
            assert_eq!(part.lens["pair"].positions, vec![4, 5, 6]);
            for (a, b) in full.lens["pair"].logprob.iter().zip(&part.lens["pair"].logprob) {
                assert!(a[4..].iter().zip(b).all(|(x, y)| close(x, y)), "chunk {chunk}");
            }
            let ids = |t: &TopRecord| t.by_depth[3].iter().map(|x| x.id).collect::<Vec<_>>();
            assert_eq!(ids(&full.top[0]), ids(&part.top[0]));
        }
    }

    #[test]
    fn top_k_ranks_ties_by_id_and_refuses_nan() {
        let row = [-1.0, -0.5, -3.0, -0.5, -2.0];
        assert_eq!(top_k(&row, 3).unwrap(), vec![(1, -0.5), (3, -0.5), (0, -1.0)]);
        assert_eq!(top_k(&row, 9).unwrap().len(), 5);
        assert!(top_k(&[-1.0, f32::NAN, -2.0], 1).is_err());
    }

    #[test]
    fn a_spec_that_asks_for_nothing_still_maps_routing_and_norms() {
        let model = tiny();
        let e = examine(&model, &TOKENS, &ExamineSpec::default(), name).unwrap();
        assert!(e.lens.is_empty() && e.top.is_empty());
        assert_eq!(e.routing.len(), 2);
        assert!(e.routing.iter().all(|r| r.logits.is_none() && r.selection_bias.is_none() && r.experts.len() == 7));
        assert!(!serde_json::to_string(&e).unwrap().contains("\"logits\""));
    }

    #[test]
    fn bad_requests_are_refused_not_repaired() {
        let model = tiny();
        let bad = |edit: &dyn Fn(&mut ExamineSpec)| {
            let mut s = spec();
            edit(&mut s);
            examine(&model, &TOKENS, &s, name).unwrap_err()
        };
        assert!(bad(&|s| s.token_sets.insert("x".into(), vec![16]).map(|_| ()).unwrap_or(()))
            .contains("outside the vocabulary"));
        assert!(bad(&|s| s.token_sets.insert("x".into(), vec![2, 2]).map(|_| ()).unwrap_or(()))
            .contains("twice"));
        assert!(bad(&|s| s.token_sets.insert("x".into(), vec![]).map(|_| ()).unwrap_or(()))
            .contains("at least one"));
        assert!(bad(&|s| s.top_k_positions = vec![7]).contains("outside the 7 tokens"));
        assert!(bad(&|s| s.top_k_positions = vec![1, 1]).contains("twice"));
        assert!(bad(&|s| s.top_k = 0).contains("together"));
        assert!(bad(&|s| s.top_k_positions.clear()).contains("together"));
        assert!(bad(&|s| s.top_k = MAX_TOP_K + 1).contains("top_k must be"));
        assert!(bad(&|s| s.lens_positions = vec![7]).contains("outside the 7 tokens"));
        assert!(bad(&|s| s.lens_positions = vec![3, 3]).contains("strictly increasing"));
        assert!(bad(&|s| s.lens_positions = vec![5, 2]).contains("strictly increasing"));
        assert!(bad(&|s| s.record_from = 7).contains("record_from"));
        // Asking to read a position the record was told to leave out.
        assert!(bad(&|s| s.record_from = 3).contains("before record_from"));
        assert!(bad(&|s| {
            s.record_from = 3;
            s.top_k_positions = vec![6];
            s.lens_positions = vec![1];
        })
        .contains("before record_from"));
        assert!(examine(&model, &[], &spec(), name).is_err());
        assert!(examine(&model, &[1; 33], &spec(), name).is_err());
        assert!(examine(&model, &[16], &spec(), name).is_err());
    }
}
