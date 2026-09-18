//! Which experts carry an answer: a causal map, by knockout.
//!
//! [`crate::examine`] shows which experts a router chose. It cannot show that a
//! choice mattered. This does, for the one token where an answer is decided:
//!
//! 1. Prefill a probe up to, but not including, its last token. That state is
//!    computed once and only ever branched.
//! 2. Run the last token from a branch, unsteered: the baseline distribution and
//!    the experts each router chose for that token.
//! 3. For every chosen expert, run the last token again from a fresh branch with
//!    that one expert knocked out of that one router, and record how far the
//!    followed tokens' log-probabilities moved.
//!
//! Knocking out an expert the router did not choose changes nothing, so only the
//! chosen ones are tried: `expert layers x top-k` single-token forwards per probe
//! rather than `expert layers x experts`. A knockout is a replacement selection
//! bias ([`Steering`]), so the expert that takes the vacated slot is weighted by
//! its own score, exactly as it would be had the router chosen it.
//!
//! What it measures is narrow on purpose. The knockout acts at the last token
//! only; every earlier token, and so every key and value the last token attends
//! to, is untouched. "Does this expert matter at the decision" is the question,
//! not "what if the model never had it".
//!
//! Log-probabilities are over the full vocabulary and never renormalised within
//! the followed tokens, for the reason given in [`crate::examine`].

use candle_core::{D, Tensor};
use candle_transformers::models::quantized_lfm2_moe::{Model, Observer, State, Steering};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const SCHEMA: &str = "lfm25-expert-map-v1";
/// Added to a selection score in `0..=1` plus a bias of order 0.1: decisive, and
/// far inside what an f32 adds exactly enough to keep the ranking of the rest.
const KNOCKOUT: f32 = -1e4;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Probe {
    pub name: String,
    pub tokens: Vec<u32>,
    /// Free-form, carried into the result so a reader can aggregate by it.
    #[serde(default)]
    pub group: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Knockout {
    pub layer: usize,
    pub expert: u32,
    /// The combine weight the expert had before it was removed.
    pub weight: f32,
    /// The expert that took the vacated slot.
    pub replacement: u32,
    /// Per followed token: log-probability with the expert out, minus baseline.
    pub delta: Vec<f32>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProbeResult {
    pub name: String,
    pub group: Option<String>,
    /// Per followed token: baseline log-probability at the last position.
    pub baseline: Vec<f32>,
    pub knockouts: Vec<Knockout>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Cell {
    pub layer: usize,
    pub expert: u32,
    /// Probes in which the router chose this expert, so it could be knocked out.
    pub n: usize,
    /// Per followed token: mean `delta` over those probes.
    pub mean_delta: Vec<f32>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExpertMap {
    pub schema: String,
    /// The token ids followed; every `Vec<f32>` above is indexed like this.
    pub followed: Vec<u32>,
    /// Expert layers, in order, with how many experts each has.
    pub layers: Vec<(usize, usize)>,
    pub probes: Vec<ProbeResult>,
    /// Every (layer, expert) some probe chose, in layer then expert order.
    pub cells: Vec<Cell>,
}

#[derive(Default)]
struct Chosen(BTreeMap<usize, (Vec<u32>, Vec<f32>)>);
impl Observer for Chosen {
    fn routing(&mut self, layer: usize, _: &Tensor, ids: &Tensor, weights: &Tensor) -> candle_core::Result<()> {
        let (ids, weights) = (ids.to_vec2::<u32>()?, weights.to_vec2::<f32>()?);
        match (ids.as_slice(), weights.as_slice()) {
            ([i], [w]) => self.0.insert(layer, (i.clone(), w.clone())),
            _ => candle_core::bail!("an expert map steps one token at a time"),
        };
        Ok(())
    }
}

fn followed_logprobs(logits: &Tensor, followed: &Tensor) -> Result<Vec<f32>, String> {
    let run = || -> candle_core::Result<Vec<f32>> {
        candle_nn::ops::log_softmax(&logits.flatten_all()?, D::Minus1)?
            .index_select(followed, 0)?
            .to_vec1::<f32>()
    };
    let values = run().map_err(|e| e.to_string())?;
    if values.iter().any(|v| v.is_nan()) {
        return Err("NaN log-probability".into());
    }
    Ok(values)
}

/// One step of `last` from a branch of `state`, which is left as it was.
fn step(
    model: &Model,
    state: &State,
    last: u32,
    steering: &Steering,
    followed: &Tensor,
) -> Result<(Vec<f32>, Chosen), String> {
    let mut branch = state.clone();
    let mut chosen = Chosen::default();
    let logits = model
        .forward_steered(&[last], &mut branch, steering, Some(&mut chosen))
        .map_err(|e| e.to_string())?;
    Ok((followed_logprobs(&logits, followed)?, chosen))
}

fn probe(model: &Model, probe: &Probe, followed: &Tensor, chunk: usize) -> Result<ProbeResult, String> {
    let Some((&last, head)) = probe.tokens.split_last().filter(|(_, head)| !head.is_empty()) else {
        return Err(format!("{}: a probe needs at least two tokens", probe.name));
    };
    let mut state = model.new_state();
    for part in head.chunks(chunk) {
        model.forward(part, &mut state).map_err(|e| format!("{}: {e}", probe.name))?;
    }
    let (baseline, chosen) = step(model, &state, last, &Steering::default(), followed)?;
    let mut knockouts = Vec::new();
    for (&layer, (experts, weights)) in &chosen.0 {
        let own = model
            .router_bias(layer)
            .ok_or_else(|| format!("layer {layer} routed but has no selection bias"))?;
        let bias = own.to_vec1::<f32>().map_err(|e| e.to_string())?;
        for (&expert, &weight) in experts.iter().zip(weights) {
            let mut without = bias.clone();
            without[expert as usize] += KNOCKOUT;
            let mut steering = Steering::default();
            steering.set_bias(
                layer,
                Tensor::from_vec(without, bias.len(), own.device()).map_err(|e| e.to_string())?,
            );
            let (logprob, after) = step(model, &state, last, &steering, followed)?;
            let now = &after.0.get(&layer).ok_or("a steered layer did not route")?.0;
            let replacement = *now
                .iter()
                .find(|e| !experts.contains(e))
                .ok_or_else(|| format!("layer {layer}: knocking out expert {expert} changed nothing"))?;
            if now.contains(&expert) {
                return Err(format!("layer {layer}: expert {expert} survived its knockout"));
            }
            knockouts.push(Knockout {
                layer,
                expert,
                weight,
                replacement,
                delta: logprob.iter().zip(&baseline).map(|(a, b)| a - b).collect(),
            });
        }
    }
    Ok(ProbeResult {
        name: probe.name.clone(),
        group: probe.group.clone(),
        baseline,
        knockouts,
    })
}

/// The mean effect of each knockout over the probes in which it could be made.
pub fn aggregate(probes: &[ProbeResult], n_followed: usize) -> Vec<Cell> {
    let mut sums: BTreeMap<(usize, u32), (usize, Vec<f64>)> = BTreeMap::new();
    for k in probes.iter().flat_map(|p| &p.knockouts) {
        let (n, sum) = sums.entry((k.layer, k.expert)).or_insert((0, vec![0.; n_followed]));
        *n += 1;
        for (s, d) in sum.iter_mut().zip(&k.delta) {
            *s += f64::from(*d);
        }
    }
    sums.into_iter()
        .map(|((layer, expert), (n, sum))| Cell {
            layer,
            expert,
            n,
            mean_delta: sum.iter().map(|s| (s / n as f64) as f32).collect(),
        })
        .collect()
}

/// Build the map. `progress` is told each finished probe's name.
pub fn expert_map(
    model: &Model,
    probes: &[Probe],
    followed: &[u32],
    mut progress: impl FnMut(&str),
) -> Result<ExpertMap, String> {
    expert_map_chunked(model, probes, followed, crate::adjudicator::CHUNK, &mut progress)
}

fn expert_map_chunked(
    model: &Model,
    probes: &[Probe],
    followed: &[u32],
    chunk: usize,
    progress: &mut dyn FnMut(&str),
) -> Result<ExpertMap, String> {
    if probes.is_empty() || followed.is_empty() {
        return Err("an expert map needs at least one probe and one followed token".into());
    }
    let mut seen = std::collections::BTreeSet::new();
    for &id in followed {
        if id as usize >= model.vocab_size() || !seen.insert(id) {
            return Err(format!("followed token {id} is outside the vocabulary or listed twice"));
        }
    }
    let ids = Tensor::from_vec(followed.to_vec(), followed.len(), model.device()).map_err(|e| e.to_string())?;
    let mut results = Vec::with_capacity(probes.len());
    for p in probes {
        results.push(probe(model, p, &ids, chunk)?);
        progress(&p.name);
    }
    Ok(ExpertMap {
        schema: SCHEMA.into(),
        followed: followed.to_vec(),
        layers: model
            .layers()
            .iter()
            .enumerate()
            .filter_map(|(l, k)| k.experts.map(|(n, _)| (l, n)))
            .collect(),
        cells: aggregate(&results, followed.len()),
        probes: results,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // conv, attention, conv; layer 0 dense, then 3 experts choosing 2; vocabulary 16.
    fn tiny() -> Model {
        let mut f = std::io::Cursor::new(include_bytes!("../tests/fixtures/lfm2-moe/tiny.gguf"));
        let ct = candle_core::quantized::gguf_file::Content::read(&mut f).unwrap();
        Model::from_gguf(ct, &mut f, &candle_core::Device::Cpu).unwrap()
    }
    fn probes() -> Vec<Probe> {
        vec![
            Probe { name: "a".into(), tokens: vec![1, 2, 3, 4, 5], group: Some("x".into()) },
            Probe { name: "b".into(), tokens: vec![9, 8, 7, 6, 5, 4], group: None },
        ]
    }
    fn map(chunk: usize) -> ExpertMap {
        expert_map_chunked(&tiny(), &probes(), &[3, 9, 12], chunk, &mut |_| {}).unwrap()
    }

    #[test]
    fn the_baseline_is_the_models_own_distribution_at_the_last_token() {
        let model = tiny();
        let m = map(32);
        for (p, r) in probes().iter().zip(&m.probes) {
            let logits: Vec<f32> = model
                .forward(&p.tokens, &mut model.new_state())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
            let z = max + logits.iter().map(|&l| (l as f64 - max).exp()).sum::<f64>().ln();
            for (&id, got) in m.followed.iter().zip(&r.baseline) {
                assert!((*got as f64 - (logits[id as usize] as f64 - z)).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn every_chosen_expert_is_knocked_out_once_and_something_else_takes_its_place() {
        let m = map(32);
        assert_eq!(m.layers, vec![(1, 3), (2, 3)]);
        for r in &m.probes {
            // two expert layers, two chosen in each
            assert_eq!(r.knockouts.len(), 4);
            for layer in [1, 2] {
                let here: Vec<&Knockout> = r.knockouts.iter().filter(|k| k.layer == layer).collect();
                assert_eq!(here.len(), 2);
                assert_ne!(here[0].expert, here[1].expert);
                for k in &here {
                    // With three experts and two chosen, the third is the only replacement.
                    assert!(k.replacement != here[0].expert && k.replacement != here[1].expert);
                    assert!(k.weight > 0. && k.weight < 1.);
                    assert_eq!(k.delta.len(), 3);
                }
            }
        }
        // A knockout that moved nothing anywhere would mean the steering never reached the model.
        assert!(m.probes.iter().flat_map(|r| &r.knockouts).flat_map(|k| &k.delta).any(|d| d.abs() > 1e-6));
    }

    #[test]
    fn a_knockout_is_the_steered_step_it_claims_to_be_and_leaves_the_probe_state_alone() {
        let model = tiny();
        let m = map(32);
        let p = &probes()[0];
        let mut state = model.new_state();
        model.forward(&p.tokens[..4], &mut state).unwrap();
        let followed = Tensor::from_vec(vec![3u32, 9, 12], 3, model.device()).unwrap();
        for k in &m.probes[0].knockouts {
            let mut bias = model.router_bias(k.layer).unwrap().to_vec1::<f32>().unwrap();
            bias[k.expert as usize] += KNOCKOUT;
            let mut steering = Steering::default();
            steering.set_bias(k.layer, Tensor::from_vec(bias, 3, model.device()).unwrap());
            let (by_hand, _) = step(&model, &state, p.tokens[4], &steering, &followed).unwrap();
            for ((got, hand), base) in k.delta.iter().zip(&by_hand).zip(&m.probes[0].baseline) {
                assert_eq!(*got, hand - base);
            }
        }
        // After every branch, an unsteered step from the same state is still the baseline.
        let (again, _) = step(&model, &state, p.tokens[4], &Steering::default(), &followed).unwrap();
        assert_eq!(again, m.probes[0].baseline);
    }

    #[test]
    fn cells_average_each_knockout_over_the_probes_that_could_make_it() {
        let m = map(32);
        for cell in &m.cells {
            let made: Vec<&Knockout> = m
                .probes
                .iter()
                .flat_map(|r| &r.knockouts)
                .filter(|k| (k.layer, k.expert) == (cell.layer, cell.expert))
                .collect();
            assert_eq!(cell.n, made.len());
            for (i, mean) in cell.mean_delta.iter().enumerate() {
                let want = made.iter().map(|k| k.delta[i] as f64).sum::<f64>() / made.len() as f64;
                assert!((*mean as f64 - want).abs() < 1e-6);
            }
        }
        assert_eq!(m.cells.iter().map(|c| c.n).sum::<usize>(), 8);
    }

    #[test]
    fn prefill_chunking_does_not_change_which_experts_are_tried() {
        let (a, b) = (map(32), map(2));
        for (x, y) in a.probes.iter().zip(&b.probes) {
            let cells = |r: &ProbeResult| r.knockouts.iter().map(|k| (k.layer, k.expert)).collect::<Vec<_>>();
            assert_eq!(cells(x), cells(y));
            for (p, q) in x.baseline.iter().zip(&y.baseline) {
                assert!((p - q).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn requests_that_cannot_be_mapped_are_refused() {
        let model = tiny();
        let run = |probes: &[Probe], followed: &[u32]| expert_map(&model, probes, followed, |_| {});
        assert!(run(&[], &[3]).is_err());
        assert!(run(&probes(), &[]).is_err());
        assert!(run(&probes(), &[16]).is_err());
        assert!(run(&probes(), &[3, 3]).is_err());
        let short = [Probe { name: "s".into(), tokens: vec![1], group: None }];
        assert!(run(&short, &[3]).unwrap_err().contains("at least two tokens"));
    }
}
