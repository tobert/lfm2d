//! Does the LENGTH of the final prefill chunk change what the model reads at
//! its last position?
//!
//! The 09-17 ribbon found 41 of 733 examined rows off the common depth-1
//! value, and `verdict_ribbon.py`'s own statistic put every one of them — and
//! no other row — at `n_tokens % 128` in `1..=8`. `n_tokens` is a COUNT, so
//! that is the slot at chunk offset `0..=7`: the rows whose FINAL prefill
//! chunk is one to eight tokens long. The short convolution's `l_cache` is 3
//! (read from the GGUF, not from a doc), so at depth 1 it sees three tokens; a
//! broken history carry could move offsets 0 and 1 and no further. Eight is
//! out of its reach. So this sweep varies the tail length directly rather than
//! assuming the convolution is the mechanism.
//!
//! [`read_tail`] feeds `tokens[..n - m]` as one block and the last `m` tokens
//! as a second block, reading every depth at the final position. `m == n` is
//! the unsplit reference.
//!
//! Varying `m` also varies the prefix length, which is two differences, not
//! one — the confound that cost the 09-19 session a wrong mechanism. So the
//! sweep carries its own control: tails far larger than any convolution window
//! or small-GEMM tile (120..=136, say) move the prefix by the same amounts. If
//! those read identically to the reference and the short tails do not, the
//! prefix length is not what moved them.
//!
//! Neither the daemon nor the examiner ever prefills in one block. This
//! measures INVARIANCE of the forward pass, not either schedule.

use candle_core::{D, IndexOp, Tensor};
use candle_transformers::models::quantized_lfm2_moe::{Model, Observer};
use serde::Serialize;
use std::collections::BTreeMap;

/// Every depth's reading at one token position. Depth 0 is the embedding,
/// depth `d` the residual stream leaving layer `d - 1`.
#[derive(Debug, Clone)]
pub struct Reading {
    pub residual: Vec<Vec<f32>>,
    /// The same depths read through the model's own final norm and projection.
    pub logprob: Vec<Vec<f32>>,
    /// Expert layer -> the experts chosen for this position.
    pub experts: BTreeMap<usize, Vec<u32>>,
    /// Rows in the forward call that was observed — the tail's own length.
    /// Recorded so a split that never happened cannot read as agreement.
    pub observed_rows: usize,
}

impl Reading {
    fn new(depths: usize) -> Self {
        Self {
            residual: vec![Vec::new(); depths],
            logprob: vec![Vec::new(); depths],
            experts: BTreeMap::new(),
            observed_rows: 0,
        }
    }
    /// The model's own top token at each depth.
    pub fn top(&self) -> Vec<u32> {
        self.logprob.iter().map(|row| argmax(row)).collect()
    }
    /// Per depth, how far the top token leads the runner-up, in nats. A delta
    /// smaller than this cannot change what the model writes here, so the two
    /// numbers only mean something side by side.
    pub fn top_margin(&self) -> Vec<f32> {
        self.logprob
            .iter()
            .map(|row| {
                let (mut first, mut second) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
                for &v in row {
                    if v > first {
                        second = first;
                        first = v;
                    } else if v > second {
                        second = v;
                    }
                }
                first - second
            })
            .collect()
    }
}

fn argmax(row: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, v) in row.iter().enumerate() {
        if v > &row[best] {
            best = i;
        }
    }
    best as u32
}

/// An observer that keeps nothing. Used for the prefix so both blocks take the
/// same code path as the reference's single observed call.
struct Silent;
impl Observer for Silent {}

struct LastRow<'a> {
    model: &'a Model,
    reading: Reading,
}
impl LastRow<'_> {
    fn read(&mut self, depth: usize, x: &Tensor) -> candle_core::Result<()> {
        let seq = x.dim(1)?;
        self.reading.observed_rows = seq;
        let row = x.narrow(1, seq - 1, 1)?;
        self.reading.residual[depth] = row.i((0, 0))?.to_vec1::<f32>()?;
        let logits = self.model.project(&row)?;
        self.reading.logprob[depth] = candle_nn::ops::log_softmax(&logits, D::Minus1)?
            .i((0, 0))?
            .to_vec1::<f32>()?;
        Ok(())
    }
}
impl Observer for LastRow<'_> {
    fn embedding(&mut self, x: &Tensor) -> candle_core::Result<()> {
        self.read(0, x)
    }
    fn residual(&mut self, layer: usize, x: &Tensor) -> candle_core::Result<()> {
        self.read(layer + 1, x)
    }
    fn routing(
        &mut self,
        layer: usize,
        _logits: &Tensor,
        ids: &Tensor,
        _weights: &Tensor,
    ) -> candle_core::Result<()> {
        let mut rows = ids.to_vec2::<u32>()?;
        let last = rows
            .pop()
            .ok_or_else(|| candle_core::Error::Msg(format!("layer {layer} routed no rows")))?;
        self.reading.experts.insert(layer, last);
        Ok(())
    }
}

/// Read every depth at the last token, having fed the last `tail` tokens as
/// their own forward call. `tail == tokens.len()` is the unsplit reference.
pub fn read_tail(model: &Model, tokens: &[u32], tail: usize) -> Result<Reading, String> {
    let n = tokens.len();
    if tail == 0 || tail > n {
        return Err(format!("tail must be 1..={n}, got {tail}"));
    }
    let depths = model.layers().len() + 1;
    let mut state = model.new_state();
    if tail < n {
        model
            .forward_observed(&tokens[..n - tail], &mut state, &mut Silent)
            .map_err(|e| e.to_string())?;
    }
    let mut observer = LastRow {
        model,
        reading: Reading::new(depths),
    };
    model
        .forward_observed(&tokens[n - tail..], &mut state, &mut observer)
        .map_err(|e| e.to_string())?;
    if observer.reading.logprob.iter().any(Vec::is_empty) {
        return Err("a depth was never observed".into());
    }
    if observer.reading.observed_rows != tail {
        return Err(format!(
            "asked for a tail of {tail}, observed {} rows",
            observer.reading.observed_rows
        ));
    }
    Ok(observer.reading)
}

/// One tail length measured against the reference.
#[derive(Debug, Clone, Serialize)]
pub struct Delta {
    pub tail: usize,
    /// Per depth, the largest absolute difference in the residual stream.
    pub residual_max_abs: Vec<f32>,
    /// Per depth, the largest absolute difference in log-probability, in nats.
    pub logprob_max_abs: Vec<f32>,
    /// Per depth, the reference's top token and this reading's, when they
    /// differ. A changed top token at the last depth is a changed next token.
    pub top_token_changed: BTreeMap<usize, (u32, u32)>,
    /// Expert layers whose chosen set differs from the reference's.
    pub expert_layers_changed: Vec<usize>,
}

/// Compare one reading against the reference. Shape disagreement is an error:
/// two readings of the same model at the same position have the same shape,
/// and a silently truncated comparison would read as agreement.
pub fn compare(reference: &Reading, other: &Reading, tail: usize) -> Result<Delta, String> {
    if reference.residual.len() != other.residual.len()
        || reference.logprob.len() != other.logprob.len()
    {
        return Err(format!(
            "depth mismatch: reference {}/{}, tail {tail} {}/{}",
            reference.residual.len(),
            reference.logprob.len(),
            other.residual.len(),
            other.logprob.len()
        ));
    }
    let max_abs = |a: &[f32], b: &[f32], what: &str, depth: usize| -> Result<f32, String> {
        if a.len() != b.len() {
            return Err(format!(
                "{what} width mismatch at depth {depth}: {} vs {}",
                a.len(),
                b.len()
            ));
        }
        Ok(a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max))
    };
    let mut residual_max_abs = Vec::with_capacity(reference.residual.len());
    let mut logprob_max_abs = Vec::with_capacity(reference.logprob.len());
    for depth in 0..reference.residual.len() {
        residual_max_abs.push(max_abs(
            &reference.residual[depth],
            &other.residual[depth],
            "residual",
            depth,
        )?);
        logprob_max_abs.push(max_abs(
            &reference.logprob[depth],
            &other.logprob[depth],
            "logprob",
            depth,
        )?);
    }
    let (want, got) = (reference.top(), other.top());
    let top_token_changed = want
        .iter()
        .zip(&got)
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(depth, (a, b))| (depth, (*a, *b)))
        .collect();
    if reference.experts.keys().ne(other.experts.keys()) {
        return Err(format!("tail {tail} routed a different set of layers"));
    }
    let expert_layers_changed = reference
        .experts
        .iter()
        .filter(|(layer, ids)| other.experts.get(layer) != Some(ids))
        .map(|(layer, _)| *layer)
        .collect();
    Ok(Delta {
        tail,
        residual_max_abs,
        logprob_max_abs,
        top_token_changed,
        expert_layers_changed,
    })
}

/// The whole sweep: what the unsplit reference reads, and every tail against it.
#[derive(Debug, Clone, Serialize)]
pub struct Sweep {
    /// The reference's own top token at each depth.
    pub reference_top: Vec<u32>,
    /// The reference's top-to-runner-up margin at each depth, in nats. Read
    /// each delta against the margin at the same depth.
    pub reference_margin: Vec<f32>,
    pub deltas: Vec<Delta>,
}

/// Read the reference once, then every tail against it. Readings are dropped
/// as they are compared: a full-vocabulary reading is ~13 MB per depth-stack.
///
/// `progress` is called before each reading with `(done, total)`. A sweep is
/// seconds on a GPU and hours on a slow backend, and a run that prints nothing
/// until it finishes cannot be told from a run that has hung -- one was left
/// for 73 minutes on that ambiguity.
pub fn sweep(
    model: &Model,
    tokens: &[u32],
    tails: &[usize],
    mut progress: impl FnMut(usize, usize),
) -> Result<Sweep, String> {
    let total = tails.len() + 1;
    progress(0, total);
    let reference = read_tail(model, tokens, tokens.len())?;
    let deltas = tails
        .iter()
        .enumerate()
        .map(|(n, &tail)| {
            progress(n + 1, total);
            let reading = read_tail(model, tokens, tail)?;
            compare(&reference, &reading, tail)
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(Sweep {
        reference_top: reference.top(),
        reference_margin: reference.top_margin(),
        deltas,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> Model {
        let mut f = std::io::Cursor::new(include_bytes!("../tests/fixtures/lfm2-moe/tiny.gguf"));
        let ct = candle_core::quantized::gguf_file::Content::read(&mut f).unwrap();
        Model::from_gguf(ct, &mut f, &candle_core::Device::Cpu).unwrap()
    }
    const TOKENS: [u32; 7] = [1, 2, 3, 4, 5, 6, 7];

    #[test]
    fn every_tail_length_reads_the_same_last_position() {
        let model = tiny();
        let mut seen = Vec::new();
        let swept = sweep(&model, &TOKENS, &[1, 2, 3, 4, 5, 6, 7], |n, total| {
            seen.push((n, total))
        })
        .unwrap();
        // Every reading is announced before it runs, the reference included.
        assert_eq!(seen, (0..=7).map(|n| (n, 8)).collect::<Vec<_>>());
        assert_eq!(swept.deltas.len(), 7);
        assert!(swept.reference_margin.iter().all(|m| *m >= 0.0));
        for d in &swept.deltas {
            let worst = d.logprob_max_abs.iter().copied().fold(0f32, f32::max);
            assert!(
                worst < 1e-4,
                "tail {} moved the lens by {worst} nats: {:?}",
                d.tail,
                d.logprob_max_abs
            );
            assert!(d.top_token_changed.is_empty(), "tail {}", d.tail);
            assert!(d.expert_layers_changed.is_empty(), "tail {}", d.tail);
        }
    }

    /// The dangerous direction: a comparison that always reports agreement.
    /// Two genuinely different positions must come back loud.
    #[test]
    fn the_comparison_sees_a_difference_when_there_is_one() {
        let model = tiny();
        let a = read_tail(&model, &TOKENS, 7).unwrap();
        let b = read_tail(&model, &[7, 6, 5, 4, 3, 2, 1], 7).unwrap();
        let d = compare(&a, &b, 7).unwrap();
        let worst = d.logprob_max_abs.iter().copied().fold(0f32, f32::max);
        assert!(worst > 1e-3, "different inputs read identically: {worst}");
        assert!(
            d.residual_max_abs.iter().copied().fold(0f32, f32::max) > 1e-3,
            "residuals identical for different inputs"
        );
    }

    /// The last depth is what the model actually samples, so it has to equal
    /// the ordinary unobserved forward's logits.
    #[test]
    fn the_last_depth_is_the_models_own_next_token_distribution() {
        let model = tiny();
        let reading = read_tail(&model, &TOKENS, 3).unwrap();
        // f64 on the host, max subtracted, as examine's own reference does.
        let logits: Vec<f32> = model
            .forward(&TOKENS, &mut model.new_state())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let z = max + logits.iter().map(|&l| (l as f64 - max).exp()).sum::<f64>().ln();
        let want: Vec<f64> = logits.iter().map(|&l| l as f64 - z).collect();
        let got = reading.logprob.last().unwrap();
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(&want) {
            assert!((*g as f64 - w).abs() < 1e-4, "lens {g} vs forward {w}");
        }
    }

    #[test]
    fn a_tail_outside_the_sequence_is_refused() {
        let model = tiny();
        assert!(read_tail(&model, &TOKENS, 0).is_err());
        assert!(read_tail(&model, &TOKENS, 8).is_err());
    }
}
