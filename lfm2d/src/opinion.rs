//! System-1 reads over the resident adjudicator: one prefill, no decoding.
//!
//! The generative adjudicator writes a whole report to reach one verdict:
//! ~80 tokens at ~15 ms each, after the prefill. An opinion stands the model at
//! the verdict slot instead — the prompt spec's `opinion.prefill` is written as
//! if the assistant had already begun its answer — and scores every option in
//! `opinion.options` by teacher-forcing its full token sequence, terminator
//! included. Nothing is sampled, so there is nothing for a grammar to mask and
//! nothing to argmax away: the caller gets the distribution and decides.
//!
//! Why full sequences rather than first tokens: an option can span several
//! tokens, and forcing bytes through a grammar walks an off-canonical path
//! (gibberish and `ls -la` once scored one point apart that way). Scoring the
//! tokenizer's own encoding of `prefill + option + close` stays on the path the
//! model would have written.
//!
//! Why the tokenization is split at the longest common prefix of the options'
//! full encodings, not at the end of the prefill: a BPE merge can cross the
//! prefill/option boundary. Splitting where the encodings first diverge means
//! every option is scored on its canonical tokens and the shared part is
//! prefilled once.
use candle_core::{D, Result, Tensor};
use candle_transformers::models::quantized_lfm2_moe::{Model, Observer, State};
use serde::{Deserialize, Serialize};

/// The question a spec asks at its verdict slot.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OpinionSpec {
    /// Text the assistant's turn is continued with before the options, e.g.
    /// `{"verdict": "`. Rendered after the spec's reasoning opening.
    pub prefill: String,
    /// The answer set, in the order the response reports it.
    pub options: Vec<String>,
    /// Written after each option, e.g. the closing quote. It makes an option
    /// that is a prefix of another (`ask` / `asking`) score as a whole word,
    /// and is part of every option's score.
    pub close: String,
}

impl OpinionSpec {
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.options.len() < 2 {
            return Err("opinion.options needs at least two options".into());
        }
        let mut seen = std::collections::BTreeSet::new();
        for option in &self.options {
            if option.is_empty() {
                return Err("opinion.options must not contain an empty option".into());
            }
            if !seen.insert(option) {
                return Err(format!("opinion.options repeats {option:?}"));
            }
        }
        if self.close.is_empty() {
            return Err("opinion.close must not be empty".into());
        }
        for text in [&self.prefill, &self.close].into_iter().chain(&self.options) {
            if text.contains("<|") || text.contains("<think>") || text.contains("</think>") {
                return Err("opinion text must not carry model control tokens".into());
            }
        }
        Ok(())
    }
}

/// One option's standing at the slot.
#[derive(Clone, Debug, Serialize)]
pub struct OptionScore {
    pub option: String,
    /// Sum of the RAW logprobs (full-vocabulary denominators) of the option's
    /// tokens, terminator included. `<= 0`.
    pub logprob: f32,
    /// Renormalized over the options. Read it beside `sequence_mass`: when the
    /// model put little mass on the answer set, this is noise that looks like
    /// an answer.
    pub prob: f32,
    pub tokens: Vec<u32>,
}

/// The whole read. The response never picks a winner; the caller does.
#[derive(Clone, Debug, Serialize)]
pub struct OpinionRead {
    pub options: Vec<OptionScore>,
    /// logsumexp of the options' sequence logprobs: the log probability that
    /// the model writes exactly one of the options and its terminator.
    pub sequence_mass: f32,
    /// Raw log mass on the distinct first tokens of the options' continuations.
    pub first_token_mass: f32,
    /// Tokens prefilled before the options diverge, and tokens teacher-forced
    /// across all options after that.
    pub shared_tokens: usize,
    pub scored_tokens: usize,
    /// sha256 of the rendered text the options continue: prefix, user turn,
    /// reasoning opening and prefill. Results carry it so a replay can prove it
    /// rendered what the daemon ran.
    pub rendered_sha256: String,
}

/// Length of the longest prefix every sequence shares.
pub fn common_prefix_len(seqs: &[Vec<u32>]) -> usize {
    let Some(first) = seqs.first() else { return 0 };
    let mut n = first.len();
    for s in &seqs[1..] {
        n = n.min(first.iter().zip(s).take_while(|(a, b)| a == b).count());
    }
    n
}

fn log_softmax_rows(logits: &Tensor) -> Result<Vec<Vec<f32>>> {
    candle_nn::ops::log_softmax(logits, D::Minus1)?.to_vec2::<f32>()
}

struct LastResidual {
    layer: usize,
    residual: Option<Tensor>,
}
impl Observer for LastResidual {
    fn residual(&mut self, layer: usize, x: &Tensor) -> Result<()> {
        if layer == self.layer {
            self.residual = Some(x.clone());
        }
        Ok(())
    }
}

/// Raw logprob of each continuation, teacher-forced from `state`, whose
/// next-token logits are `logits` (`(1, vocab)`). `state` is not advanced.
///
/// The first token of every continuation is read from `logits`; the rest from
/// one forward over the continuation minus its last token, projected at every
/// position through the model's own final norm and output head.
pub fn score_continuations(
    model: &Model,
    state: &State,
    logits: &Tensor,
    continuations: &[Vec<u32>],
    check: &dyn Fn() -> Result<()>,
) -> Result<Vec<f32>> {
    let first = log_softmax_rows(logits)?;
    let first = first
        .first()
        .ok_or_else(|| candle_core::Error::Msg("empty logits".into()))?;
    let last_layer = model.layers().len() - 1;
    let mut scores = Vec::with_capacity(continuations.len());
    for cont in continuations {
        check()?;
        let (&head, rest) = cont
            .split_first()
            .ok_or_else(|| candle_core::Error::Msg("empty continuation".into()))?;
        let mut total = first[head as usize];
        if !rest.is_empty() {
            let mut branch = state.clone();
            let mut tap = LastResidual {
                layer: last_layer,
                residual: None,
            };
            model.forward_observed(&cont[..cont.len() - 1], &mut branch, &mut tap)?;
            let residual = tap
                .residual
                .ok_or_else(|| candle_core::Error::Msg("last residual not observed".into()))?;
            let rows = log_softmax_rows(&model.project(&residual)?.squeeze(0)?)?;
            for (row, &next) in rows.iter().zip(rest) {
                total += row[next as usize];
            }
        }
        scores.push(total);
    }
    Ok(scores)
}

pub fn logsumexp(values: &[f32]) -> f32 {
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return max;
    }
    max + values.iter().map(|&v| (v - max).exp()).sum::<f32>().ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Model {
        let mut f = std::io::Cursor::new(include_bytes!("../tests/fixtures/lfm2-moe/tiny.gguf"));
        let ct = candle_core::quantized::gguf_file::Content::read(&mut f).unwrap();
        Model::from_gguf(ct, &mut f, &candle_core::Device::Cpu).unwrap()
    }

    fn ok() -> Result<()> {
        Ok(())
    }

    #[test]
    fn common_prefix_stops_at_first_divergence() {
        assert_eq!(
            common_prefix_len(&[vec![1, 2, 3, 4], vec![1, 2, 5], vec![1, 2, 3]]),
            2
        );
        assert_eq!(common_prefix_len(&[vec![1, 2], vec![3]]), 0);
        assert_eq!(common_prefix_len(&[vec![7, 8, 9]]), 3);
    }

    /// The batched read must equal decoding the continuation one token at a
    /// time, which is what the generative path does. A mistake in the
    /// position alignment (reading row i for token i instead of i+1) fails
    /// this by nats, not by rounding.
    #[test]
    fn batched_score_matches_token_by_token_decode() {
        let model = fixture();
        let mut state = model.new_state();
        let logits = model.forward(&[1, 2, 3], &mut state).unwrap();
        let conts = vec![vec![4, 5, 6], vec![7], vec![4, 9]];
        let got = score_continuations(&model, &state, &logits, &conts, &ok).unwrap();
        for (cont, got) in conts.iter().zip(&got) {
            let mut s = state.clone();
            let mut l = logits.clone();
            let mut want = 0f32;
            for &t in cont {
                want += log_softmax_rows(&l).unwrap()[0][t as usize];
                l = model.forward(&[t], &mut s).unwrap();
            }
            assert!(
                (got - want).abs() < 1e-3,
                "cont {cont:?}: batched {got} vs stepwise {want}"
            );
        }
    }

    #[test]
    fn scoring_does_not_advance_the_shared_state() {
        let model = fixture();
        let mut state = model.new_state();
        let logits = model.forward(&[1, 2, 3], &mut state).unwrap();
        score_continuations(&model, &state, &logits, &[vec![4, 5, 6]], &ok).unwrap();
        assert_eq!(state.len(), 3);
    }

    #[test]
    fn spec_refuses_degenerate_answer_sets() {
        let spec = |options: &[&str], close: &str| OpinionSpec {
            prefill: "{\"verdict\": \"".into(),
            options: options.iter().map(|s| s.to_string()).collect(),
            close: close.into(),
        };
        assert!(spec(&["allow", "ask"], "\"").validate().is_ok());
        assert!(spec(&["allow"], "\"").validate().is_err());
        assert!(spec(&["allow", "allow"], "\"").validate().is_err());
        assert!(spec(&["allow", ""], "\"").validate().is_err());
        assert!(spec(&["allow", "ask"], "").validate().is_err());
        assert!(spec(&["allow", "<|im_end|>"], "\"").validate().is_err());
    }

    #[test]
    fn logsumexp_is_stable() {
        let v = logsumexp(&[-1000.0, -1000.0]);
        assert!((v - (-1000.0 + 2f32.ln())).abs() < 1e-3);
        assert_eq!(logsumexp(&[f32::NEG_INFINITY]), f32::NEG_INFINITY);
    }
}
