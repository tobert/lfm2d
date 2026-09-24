//! Zero-shot prompt routing — `LFM2.5-Encoder-350M-Prompt-Router`
//! (`Lfm2BidirForSequenceRouting`): scores a prompt against caller-supplied
//! routing lanes ("routes"), defined as free text, in a single encoder pass.
//! Not a fixed-label classifier — there is no `id2label` and no trained
//! output width; the "labels" are whatever routes the caller passes at call
//! time, via [`crate::config::Lfm2EncoderConfig::rule_proj_dim`]'s learned
//! projection space.
//!
//! # How a score is produced
//!
//! One sequence is tokenized: `prefix + text`, where `prefix` lists every
//! route under a `Categories:` header before the actual `Text:` body (see
//! [`build_prefix`]). ONE trunk forward pass runs over that whole sequence.
//! Then, from that single pass:
//!
//! - a **query span** is mean-pooled into `text_rep` — by default the
//!   whole routed text, but see "Focused pooling" below,
//! - **each route's own span** (the `"- {route}"` line naming it) is
//!   mean-pooled into that route's `category_rep`,
//! - both are projected into a shared 256-dim space (`tok_proj` /
//!   `rule_proj`, both `Linear` WITH bias) and L2-normalized,
//! - `logits[r] = cosine(query, categories[r]) * scale + score_bias`, where
//!   `scale = min(exp(logit_scale), 30.0)`.
//!
//! A **route** that selects zero tokens (an empty or entirely-whitespace
//! route string, or one whose text doesn't survive tokenization as
//! separate tokens) pools to the **zero vector** — not a skipped row, not
//! an error. Because `rule_proj` carries a bias, projecting the zero
//! vector does NOT produce a zero score: the bias alone still defines a
//! point in the projection space, and it still gets normalized and scored
//! like any other route. This is LiquidAI's own behavior and this crate
//! preserves it exactly for routes. The QUERY side is different — see
//! below.
//!
//! # Focused pooling: judge one span, keep the rest as context
//!
//! [`Lfm2SequenceRouter::compute`] (and every unfocused `route_*` method)
//! pools the query over the ENTIRE routed text — that is what LiquidAI's
//! Python `route()` does, and it is the bit-exact-parity default.
//!
//! But nothing in the architecture requires a uniform mean over the whole
//! text: `text_pool` is an arbitrary per-token weight vector, and the
//! Python implementation just happens to fill it uniformly. That means a
//! caller can feed a multi-line CONTEXT through the trunk — so
//! bidirectional attention sees the whole history and every token's hidden
//! state is shaped by it — while pooling the query over only a narrower
//! FOCUS inside that text. [`Lfm2SequenceRouter::compute_focused`] and the
//! `*_focused`/`*_windowed` methods expose this directly.
//!
//! The motivating case: an agent's guard-evasion retry (`rm -rf X` blocked,
//! then `rm -r X`) is only recognizable as evasion in light of the
//! preceding denial, but a uniform mean over a 5-command window gives the
//! short dangerous tail only ~1/5 of the pooling mass.
//! [`Lfm2SequenceRouter::route_scores_windowed`] runs the whole window
//! through the trunk (so attention sees the denial) but pools only the
//! last command's tokens (so the score isn't diluted by the four that
//! came before it).
//!
//! `focus` is a byte range **into `text`** — the same unit
//! [`tokenizers::Encoding::get_offsets`] reports for this crate's
//! tokenizer, verified below — NOT a Python character range, and NOT a
//! range into the full `prefix + text` prompt. [`Lfm2SequenceRouter::compute`]
//! is defined as `compute_focused` with `focus = 0..text.len()`: there is
//! exactly one pooling implementation, and the unfocused, bit-exact-parity
//! path runs through it like everything else.
//!
//! A focus that is empty, out of range, off a UTF-8 boundary, or that
//! overlaps zero tokens is a loud [`Error::InvalidInput`], never a silent
//! zero-vector query — a zero-token focus would otherwise pool to zero and
//! then, same as a zero-token ROUTE, get a plausible-looking non-zero score
//! out of `tok_proj`'s bias. For a route that is fine (it is the
//! documented, faithfully-ported Python behavior); for the thing being
//! JUDGED it is exactly the silent-wrong-answer shape this crate's house
//! rule refuses. Note this makes an entirely-empty `text` (`""`) a hard
//! error even on the default unfocused path (`0..0` is an empty focus) —
//! a narrow, deliberate divergence from Python, which would silently pool
//! that case to zero too; every non-degenerate input takes the identical
//! path Python does.
//!
//! ## What focusing removes, and what it doesn't — measured, not assumed
//!
//! Focused pooling removes exactly one thing: the AVERAGING-WEIGHT
//! dilution of a uniform mean over many tokens. It does NOT, and
//! architecturally cannot, remove REPRESENTATIONAL dilution — the fact
//! that bidirectional attention itself pulls every token's hidden state
//! toward the others' as context grows, before pooling ever runs. Both
//! `whole` and `focus`-pooled queries are computed from hidden states of
//! the SAME trunk pass over the SAME full sequence; focusing changes which
//! rows get averaged, not what those rows already contain.
//!
//! Swept against this checkpoint (2 to 40 lines of filler before one
//! decisive final line, several route pairs and content domains): the
//! resulting query-vector shift was real, reproducible, and always
//! present — but SMALL (cosine between the whole-pooled and focus-pooled
//! query stayed above 0.999 even at 40 filler lines), and its effect on
//! the final softmax was smaller still, sometimes only a few parts in
//! 10⁻⁴ and not even reliably signed in the "focus should win" direction
//! as filler grew. In other words: on THIS checkpoint, for THESE window
//! lengths, representational dilution via attention dominates over
//! averaging-weight dilution, so focusing recovers only a small fraction
//! of what a caller might expect from the "1/N of the mass" framing. The
//! mechanism is correct and never hurts; it is not a guaranteed rescue for
//! a diluted signal on every checkpoint or every window length, and a
//! caller relying on it for a hard safety decision should verify the
//! effect size on their own checkpoint and window sizes rather than assume
//! it from this doc comment. See `tests/router_parity.rs`'s
//! `focused_pooling_removes_the_averaging_weight_dilution_it_targets` for
//! what is and isn't asserted here.
//!
//! # Why both `route_scores` and `route_probs` exist
//!
//! [`Lfm2SequenceRouter::route_probs`] is a softmax over routes — LiquidAI's
//! own `route()` contract, and the right choice for "pick exactly one
//! lane" dispatch, since by construction the probabilities sum to 1 and
//! adding a route steals mass from every other one.
//!
//! [`Lfm2SequenceRouter::route_scores`] (the pre-softmax scaled logits) and
//! [`Lfm2SequenceRouter::route_cosines`] (the raw, unscaled cosine
//! similarities, before `scale`/`score_bias`) exist for the opposite
//! shape of question: "which of these independent specialists should ALL
//! look at this prompt?" A fan-out/multi-label caller doing a severity
//! union across routes needs a score PER route that does not trade off
//! against the others — softmax is actively wrong there, because it will
//! suppress a second real match just for co-occurring with a first one.
//!
//! # The softmax is saturated — threshold on `route_scores`, not `route_probs`
//!
//! Measured against the reference fixture's (prompt, route) pairs: **68.9%
//! sit at `|cosine| > 0.99`**, piled at the ±1 extremes of the space
//! `query`/`categories` are normalized into. That is not specific to any
//! one fixture case — it falls out of the math whenever a match is
//! clear-cut. With `scale = min(exp(logit_scale), 30.0)` (≈1.3715 on this
//! checkpoint) and a query that sits at cosine ≈ +1 against its true route
//! and ≈ −1 against every other one, the top-route softmax probability
//! reduces to
//!
//! ```text
//! p_top(R) = 1 / (1 + (R - 1) * exp(-2 * scale))
//! ```
//!
//! — a function of the ROUTE COUNT `R` alone, not of how confidently the
//! model actually matched. This reproduces the observed top-1 probability
//! almost exactly for R = 1, 2, 3, 4, 8 in the reference fixture. In other
//! words: **`route_probs`'/`route()`'s score is telling you how many
//! routes you passed in, far more than it is telling you how sure the
//! model is.** Two prompts with wildly different match quality (cosine
//! 0.63 vs cosine 0.9954 — the two least-saturated cases this crate's
//! parity fixture contains) can land on nearly the same softmax
//! probability once the winning margin clears the "clear-cut" regime,
//! because softmax only sees the GAP between logits, and scale compresses
//! that gap fast.
//!
//! Consumers that need to know HOW confident a match is — not just which
//! route won against however many alternatives were offered — should
//! threshold on [`Lfm2SequenceRouter::route_scores`] (or
//! [`Lfm2SequenceRouter::route_cosines`], which is bounded and
//! route-count-independent by construction) rather than on `route_probs`.
//! `route_probs`/`route()` remain right for "pick exactly one lane and act
//! on it," where the identity of the winner is all that matters; they are
//! the wrong tool for "is this actually a strong match," which is the
//! question a guard/threshold decision usually is.
//!
//! # Empty routes: a deliberate divergence from Python
//!
//! LiquidAI's Python `route()` accepts an empty `routes` list and silently
//! builds the prefix `"Categories:\n- (none)\n\nText:\n"` — one
//! degenerate category literally named `(none)`, which then gets scored
//! and returned as if it meant something. This crate refuses instead: an
//! empty routes slice is a caller bug (there is no route to have called
//! `route()` for), and a numeric-looking score for `(none)` is a worse
//! failure mode than a loud error, per this crate's house rule against
//! silent fallbacks. See [`Error::InvalidInput`].
//!
//! # Offsets are bytes, not Python characters — verified, not assumed
//!
//! LiquidAI's Python `_category_ranges` computes route spans using
//! `len(route)`, and matches them against `tokenizer(..., return_offsets_mapping=True)`
//! — both **Python string length and HF fast-tokenizer offsets are in
//! Unicode CHARACTERS** in Python. The Rust `tokenizers` crate's
//! `Encoding::get_offsets()` uses a different unit. This was checked
//! empirically against this checkpoint's own `tokenizer.json` before
//! writing this module (encode a Japanese route/text, slice the source
//! string at the reported offsets, and check whether the result equals the
//! token text): the offsets are **byte** indices into the UTF-8 source,
//! not char indices. `tests/router_parity.rs`'s
//! `offsets_are_byte_indices_not_char_indices` is that same check kept as
//! a regression test, and this module's own unit test
//! `category_ranges_span_math_handles_non_ascii_routes` proves the
//! consequence for its span math.
//!
//! Consequently this port uses Rust `str::len()` (which **is** byte
//! length) everywhere the Python source uses `len()`, in [`build_prefix`]
//! and [`category_ranges`]. The two implementations compute different
//! raw NUMBERS for the same route (Python counts chars, Rust counts
//! bytes) — but each computes them in ITS OWN tokenizer's offset unit, so
//! both select the identical SET of tokens for a given route. A naive
//! port that instead called `.chars().count()` here — mimicking Python's
//! unit rather than matching Rust's own tokenizer's unit — would pass
//! every pure-ASCII test (where chars and bytes coincide) and then
//! silently mis-pool any non-ASCII route or prompt: the byte range it
//! computed would land mid-character or on the wrong tokens entirely, with
//! no error, just a quietly wrong score.

use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use tokenizers::Tokenizer;

use crate::config::Lfm2EncoderConfig;
use crate::error::{Error, Result};
use crate::trunk::Lfm2Trunk;

/// Literal header every prompt starts with. ASCII, so its Rust byte length
/// and Python char length happen to agree — this is the one constant in
/// the span math that is NOT where the byte/char trap could bite.
const CATEGORIES_HEADER: &str = "Categories:\n";

/// Literal separator between the route list and the routed text.
const TEXT_HEADER: &str = "\n\nText:\n";

/// Build the exact prompt LiquidAI's `route()` scores: every route listed
/// under `Categories:`, one per line prefixed `"- "`, then the input text
/// under `Text:`.
///
/// Mirrors `Lfm2BidirForSequenceRouting._prefix` byte-for-byte in
/// structure, INCLUDING its degenerate `"- (none)"` behavior for an empty
/// `routes` slice — this function is a faithful, testable port of the
/// Python string-building; the public heads refuse empty `routes` before
/// ever calling it (see the module docs). Kept `pub(crate)` rather than
/// exposed, since a caller has no sound use for the degenerate form.
pub(crate) fn build_prefix<S: AsRef<str>>(routes: &[S]) -> String {
    let body = if routes.is_empty() {
        "- (none)".to_string()
    } else {
        routes
            .iter()
            .map(|r| format!("- {}", r.as_ref()))
            .collect::<Vec<_>>()
            .join("\n")
    };
    format!("{CATEGORIES_HEADER}{body}{TEXT_HEADER}")
}

/// Byte-offset `(start, end)` span of each route's OWN text (the part
/// after `"- "`, not the dash/space/newline around it) inside a prompt
/// built by [`build_prefix`] over the same `routes`.
///
/// Byte offsets, matching Rust's `tokenizers` crate — see the module docs'
/// "Offsets are bytes, not Python characters" section. Ported structurally
/// from `Lfm2BidirForSequenceRouting._category_ranges`; the only
/// difference from the Python is the unit `len()` measures in, which is
/// exactly the point.
///
/// `pub` (not `pub(crate)`) and `#[doc(hidden)]` ONLY so
/// `tests/router_parity.rs` can assert this function's output directly
/// against the reference fixture's `category_ranges_byte` field — the
/// sharpest available cross-check of the byte-vs-char offsets finding,
/// since it compares span NUMBERS rather than end-to-end scores that
/// could coincidentally agree even given a wrong span. Not meant for
/// outside use: a caller has no sound use for a bare span without the
/// tokenizer offsets it exists to be compared against.
#[doc(hidden)]
pub fn category_ranges<S: AsRef<str>>(routes: &[S]) -> Vec<(usize, usize)> {
    let mut ranges = Vec::with_capacity(routes.len());
    let mut pos = CATEGORIES_HEADER.len();
    for r in routes {
        let route = r.as_ref();
        let start = pos + 2; // skip "- "
        let end = start + route.len(); // BYTE length, deliberately (see module docs)
        ranges.push((start, end));
        pos = end + 1; // skip the "\n" this route's line ends with
    }
    ranges
}

/// Token indices whose span overlaps `(start, end)` — used both for a
/// route's own span (against the `Categories:` list) and for a query
/// focus (against the routed text). Overlap, not containment: a token can
/// straddle the boundary and still count. A token with an empty span
/// (`tok_start == tok_end`, e.g. a special token) never counts.
///
/// The unfocused default (`compute`'s `focus = 0..text.len()`, translated
/// to `text_start..full_text.len()`) reduces to exactly the Python
/// `route()`'s plain "ends after text_start" rule: every real token
/// satisfies `tok_start < full_text.len()` by construction, so the overlap
/// test's extra left-bound condition is vacuous there and only starts
/// doing work once `focus` narrows below the whole text.
fn select_overlapping_tokens(offsets: &[(usize, usize)], (start, end): (usize, usize)) -> Vec<usize> {
    offsets
        .iter()
        .enumerate()
        .filter(|&(_, &(tok_start, tok_end))| {
            tok_start < end && tok_end > start && tok_start != tok_end
        })
        .map(|(i, _)| i)
        .collect()
}

/// Uniform mean over the selected rows of `hidden` (shape `(seq, hidden)`).
/// An empty selection pools to the ZERO vector. Callers that must not
/// accept a zero-token selection (the query focus) check that BEFORE
/// calling this; callers that must (a route) call it unconditionally — see
/// the module docs on why those two cases are handled differently.
fn pool_mean(hidden: &Tensor, idxs: &[usize]) -> Result<Tensor> {
    let (_, hidden_size) = hidden.dims2()?;
    if idxs.is_empty() {
        return Ok(Tensor::zeros(hidden_size, hidden.dtype(), hidden.device())?);
    }
    let idx: Vec<u32> = idxs.iter().map(|&i| i as u32).collect();
    let idx_t = Tensor::new(idx.as_slice(), hidden.device())?;
    Ok(hidden.index_select(&idx_t, 0)?.mean(0)?)
}

/// L2-normalize the last dimension. Works on a single vector `(dim,)` or a
/// batch of rows `(n, dim)` — [`candle_core::Tensor::broadcast_div`]
/// handles both against a `sum_all`/`sum_keepdim` norm respectively, so one
/// helper covers `query` (1D) and `categories` (2D).
fn l2_normalize(x: &Tensor) -> Result<Tensor> {
    let norm = match x.dims().len() {
        1 => x.sqr()?.sum_all()?.sqrt()?,
        _ => x.sqr()?.sum_keepdim(x.dims().len() - 1)?.sqrt()?,
    };
    Ok(x.broadcast_div(&norm)?)
}

/// Every value in `v` must be finite. The house rule against silent
/// fallbacks applies to numeric degeneracy too: a route whose text (or
/// every route's text) is empty enough to make a norm exactly zero would
/// otherwise divide out to NaN/Inf and hand the caller a score that looks
/// like a number.
fn require_finite(what: &str, v: &[f32]) -> Result<()> {
    if let Some(bad) = v.iter().find(|x| !x.is_finite()) {
        return Err(Error::InvalidInput(format!(
            "{what} produced a non-finite value ({bad}) — likely a degenerate (zero-norm) \
             pooled representation; refusing to return a score that looks like a number"
        )));
    }
    Ok(())
}

/// `routes` must be non-empty — see the module docs' "Empty routes"
/// section. Factored out so it is unit-testable without a loaded model.
fn require_nonempty_routes<S>(routes: &[S]) -> Result<()> {
    if routes.is_empty() {
        return Err(Error::InvalidInput(
            "route scoring needs at least one route; LiquidAI's Python route() falls back \
             to a degenerate single category named '(none)' for an empty routes slice, \
             which this crate deliberately refuses rather than silently scoring against it \
             — see the routing module docs"
                .to_string(),
        ));
    }
    Ok(())
}

/// `focus` must be a non-empty, in-range, UTF-8-boundary-aligned byte range
/// into `text`. Factored out so it is unit-testable without a loaded
/// model or tokenizer.
fn validate_focus(text: &str, focus: &Range<usize>) -> Result<()> {
    if focus.start >= focus.end {
        return Err(Error::InvalidInput(format!(
            "focus {focus:?} is empty (start >= end); pooling needs at least one byte of text \
             to select tokens from"
        )));
    }
    if focus.end > text.len() {
        return Err(Error::InvalidInput(format!(
            "focus {focus:?} is out of range for a {}-byte text",
            text.len()
        )));
    }
    if !text.is_char_boundary(focus.start) || !text.is_char_boundary(focus.end) {
        return Err(Error::InvalidInput(format!(
            "focus {focus:?} does not fall on UTF-8 character boundaries of the text \
             ({:?}) — offsets here are BYTES, not chars, see the routing module docs",
            text
        )));
    }
    Ok(())
}

/// The token selection for a query `focus` must be non-empty — unlike a
/// route (see the module docs), a zero-token query focus is refused rather
/// than silently pooled to zero. Factored out so it is unit-testable
/// without a loaded model.
fn require_nonempty_selection(idxs: &[usize], focus: &Range<usize>, text_len: usize) -> Result<()> {
    if idxs.is_empty() {
        return Err(Error::InvalidInput(format!(
            "focus {focus:?} into a {text_len}-byte text selects zero tokens; pooling would \
             silently fall back to the zero vector, and tok_proj's bias would turn that into a \
             plausible-looking but meaningless query — refusing instead"
        )));
    }
    Ok(())
}

/// Every intermediate of one [`Lfm2SequenceRouter::compute`] /
/// [`Lfm2SequenceRouter::compute_focused`] call, in pipeline order. Exposed
/// (rather than folding straight to `probs`) so a parity test can localize
/// a divergence to a specific stage — pooling, projection, or the final
/// scale/softmax — instead of only knowing the end-to-end answer disagrees.
#[derive(Debug, Clone)]
pub struct RouteComputation {
    /// Mean-pooled trunk hidden state over the query focus's tokens,
    /// `[hidden_size]`.
    pub text_rep: Vec<f32>,
    /// Mean-pooled trunk hidden state over each route's OWN tokens, one row
    /// per route, `[n_routes][hidden_size]`.
    pub category_rep: Vec<Vec<f32>>,
    /// `l2_normalize(tok_proj(text_rep))`, `[rule_proj_dim]`.
    pub query: Vec<f32>,
    /// `l2_normalize(rule_proj(category_rep))`, one row per route,
    /// `[n_routes][rule_proj_dim]`.
    pub categories: Vec<Vec<f32>>,
    /// `dot(query, categories[r])` — cosine similarity, since both sides
    /// are unit-normalized. Pre-scale, pre-bias: the number a fan-out
    /// caller wants (see the module docs).
    pub cosines: Vec<f32>,
    /// `cosines[r] * scale + score_bias` — pre-softmax. What
    /// [`Lfm2SequenceRouter::route_scores`] returns.
    pub logits: Vec<f32>,
    /// `softmax(logits)` — what [`Lfm2SequenceRouter::route_probs`] and
    /// LiquidAI's own `route()` return.
    pub probs: Vec<f32>,
}

/// One route's probability from [`Lfm2SequenceRouter::route`], mirroring
/// the Python `route()` return shape (`{"route": ..., "score": ...}`,
/// sorted descending, optionally threshold-filtered).
#[derive(Debug, Clone, PartialEq)]
pub struct RouteMatch {
    pub route: String,
    /// A softmax probability (sums to 1.0 across the routes that were
    /// scored — NOT across only the ones that passed a threshold).
    pub score: f32,
}

/// A loaded `Lfm2BidirForSequenceRouting` checkpoint (the Prompt-Router).
#[derive(Debug)]
pub struct Lfm2SequenceRouter {
    trunk: Arc<Lfm2Trunk>,
    tok_proj: Linear,
    rule_proj: Linear,
    /// `min(exp(logit_scale), 30.0)`, precomputed at load — deterministic
    /// given the loaded weight, so there's no reason to redo the exp/clamp
    /// on every call.
    scale: f32,
    score_bias: f32,
    tokenizer: Tokenizer,
}

/// Parse+validate `config.json` and load the tokenizer — the prefix shared
/// by [`Lfm2SequenceRouter::from_dir_with`] (which goes on to load its own
/// trunk) and [`Lfm2SequenceRouter::from_trunk`] (which reuses one instead).
fn load_head_metadata(dir: &Path) -> Result<(Lfm2EncoderConfig, Tokenizer)> {
    let cfg_path = dir.join("config.json");
    let bytes = std::fs::read(&cfg_path).map_err(|e| Error::io(&cfg_path, e))?;
    let cfg = Lfm2EncoderConfig::from_json(&bytes).map_err(|source| Error::ConfigParse {
        path: cfg_path.clone(),
        source,
    })?;
    cfg.validate().map_err(|message| Error::ConfigInvalid {
        path: cfg_path.clone(),
        message,
    })?;

    let tok_path = dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| Error::Tokenizer {
        path: tok_path.clone(),
        message: e.to_string(),
    })?;

    Ok((cfg, tokenizer))
}

/// Load `tok_proj`/`rule_proj`/`logit_scale`/`score_bias` from an already
/// open `VarBuilder` — the router's own weights, shared by both loading
/// paths regardless of whether `vb` was opened at the trunk's dtype/device
/// (via [`Lfm2SequenceRouter::from_trunk`]) or a caller-chosen one (via
/// [`Lfm2SequenceRouter::from_dir_with`]).
fn load_projections(
    cfg: &Lfm2EncoderConfig,
    vb: &VarBuilder,
) -> Result<(Linear, Linear, f32, f32)> {
    let proj_dim = cfg.rule_proj_dim();
    // Both projections carry a bias — unlike every projection inside the
    // trunk, and like the token/sequence-classification heads' `classifier`.
    let tok_proj = candle_nn::linear(cfg.hidden_size, proj_dim, vb.pp("tok_proj"))?;
    let rule_proj = candle_nn::linear(cfg.hidden_size, proj_dim, vb.pp("rule_proj"))?;

    // Two bare scalar parameters (shape `[]` in the checkpoint, not `[1]`)
    // — `VarBuilder::get` accepts `()` as a 0-dim shape.
    let logit_scale: f32 = vb.get((), "logit_scale")?.to_dtype(DType::F32)?.to_scalar()?;
    let score_bias: f32 = vb.get((), "score_bias")?.to_dtype(DType::F32)?.to_scalar()?;
    // Precompute the clamp Python applies at call time
    // (`torch.clamp(self.logit_scale.exp(), max=30.0)`); it depends only on
    // the loaded weight, so there's nothing to redo per call.
    let scale = logit_scale.exp().min(30.0);

    Ok((tok_proj, rule_proj, scale, score_bias))
}

impl Lfm2SequenceRouter {
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        Self::from_dir_with(dir, DType::F32, &Device::Cpu)
    }

    pub fn from_dir_with(dir: impl AsRef<Path>, dtype: DType, device: &Device) -> Result<Self> {
        let dir = dir.as_ref();
        let (cfg, tokenizer) = load_head_metadata(dir)?;

        let weights = dir.join("model.safetensors");
        if !weights.is_file() {
            return Err(Error::io(
                &weights,
                std::io::Error::new(std::io::ErrorKind::NotFound, "no model.safetensors"),
            ));
        }
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], dtype, device)? };
        let trunk = Arc::new(Lfm2Trunk::load(&cfg, vb.clone())?);
        let (tok_proj, rule_proj, scale, score_bias) = load_projections(&cfg, &vb)?;

        Ok(Self {
            trunk,
            tok_proj,
            rule_proj,
            scale,
            score_bias,
            tokenizer,
        })
    }

    /// Build a router over an already-loaded, possibly-shared trunk — loads
    /// ONLY the tokenizer and the `tok_proj`/`rule_proj`/`logit_scale`/
    /// `score_bias` weights from `dir`, at the trunk's own dtype/device.
    /// `dir`'s own trunk weights (under `lfm2.` in the same
    /// `model.safetensors`) are never read; see
    /// [`crate::sequence_classification::Lfm2SequenceClassifier::from_trunk`]
    /// for the fuller rationale.
    ///
    /// Errors loudly, naming every mismatched field, if `dir`'s config
    /// disagrees with the trunk's shape — see [`Lfm2Trunk::check_compatible`].
    pub fn from_trunk(trunk: Arc<Lfm2Trunk>, dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let (cfg, tokenizer) = load_head_metadata(dir)?;

        let cfg_path = dir.join("config.json");
        trunk
            .check_compatible(&cfg)
            .map_err(|message| Error::ConfigInvalid { path: cfg_path, message })?;

        let weights = dir.join("model.safetensors");
        if !weights.is_file() {
            return Err(Error::io(
                &weights,
                std::io::Error::new(std::io::ErrorKind::NotFound, "no model.safetensors"),
            ));
        }
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights], trunk.dtype(), trunk.device())?
        };
        let (tok_proj, rule_proj, scale, score_bias) = load_projections(&cfg, &vb)?;

        Ok(Self {
            trunk,
            tok_proj,
            rule_proj,
            scale,
            score_bias,
            tokenizer,
        })
    }


    /// Dimension of the projection space `query`/`categories` live in
    /// (256 on every shipped checkpoint).
    pub fn proj_dim(&self) -> usize {
        self.tok_proj.weight().dim(0).unwrap_or(0)
    }

    /// Run the full pipeline, pooling the query over the ENTIRE `text` —
    /// LiquidAI's own `route()` behavior, bit-exact. Defined as
    /// [`Self::compute_focused`] with `focus = 0..text.len()`; see the
    /// module docs for the one edge case that differs (an entirely empty
    /// `text`, which this refuses rather than silently pooling to zero).
    pub fn compute<S: AsRef<str>>(&self, text: &str, routes: &[S]) -> Result<RouteComputation> {
        self.compute_focused(text, 0..text.len(), routes)
    }

    /// Run the full pipeline, but pool the query ONLY over tokens
    /// overlapping `focus` — a byte range into `text` (see the module docs'
    /// "Focused pooling" section for the unit and the motivating case).
    /// The rest of `text` still runs through the trunk and still shapes
    /// `focus`'s tokens via bidirectional attention; it just carries no
    /// direct pooling weight in the query.
    ///
    /// This is the ONE pooling implementation — [`Self::compute`] and every
    /// unfocused `route_*` method are thin wrappers around this with
    /// `focus = 0..text.len()`.
    pub fn compute_focused<S: AsRef<str>>(
        &self,
        text: &str,
        focus: Range<usize>,
        routes: &[S],
    ) -> Result<RouteComputation> {
        require_nonempty_routes(routes)?;
        validate_focus(text, &focus)?;

        let prefix = build_prefix(routes);
        let text_start = prefix.len();
        let full_text = format!("{prefix}{text}");

        let encoding = self
            .tokenizer
            .encode(full_text, true)
            .map_err(|e| Error::Encode(e.to_string()))?;
        let ids = encoding.get_ids();
        let offsets = encoding.get_offsets();

        let device = self.trunk.device();
        let input = Tensor::new(ids, device)?.unsqueeze(0)?;
        // One sequence at a time, unpadded — the reproducible path (see the
        // crate docs on why padding is not inert here). `route()` never
        // needs a batch: the caller wants one prompt scored against many
        // routes, which this does in a single forward pass already.
        let hidden = self.trunk.forward(&input, None)?.squeeze(0)?;

        // `focus` is a byte range into `text` alone; translate into
        // `full_text` coordinates by shifting past the prefix.
        let abs_focus = (text_start + focus.start, text_start + focus.end);
        let text_idxs = select_overlapping_tokens(offsets, abs_focus);
        require_nonempty_selection(&text_idxs, &focus, text.len())?;
        let text_rep = pool_mean(&hidden, &text_idxs)?;

        let ranges = category_ranges(routes);
        let mut category_reps = Vec::with_capacity(routes.len());
        for range in &ranges {
            // A route's own zero-token selection is NOT an error — see the
            // module docs; only the query focus enforces non-emptiness.
            let idxs = select_overlapping_tokens(offsets, *range);
            category_reps.push(pool_mean(&hidden, &idxs)?);
        }
        let category_rep = Tensor::stack(&category_reps, 0)?; // (n_routes, hidden)

        // Cast to F32 right after projection, before normalizing — same
        // convention as `ColbertModel::project`: normalization is precision
        // sensitive (a small denominator amplifies error), and the
        // checkpoint's own storage dtype (bf16) is not what CPU matmul
        // supports anyway.
        //
        // `Linear::forward` only special-cases rank-3/4 inputs for its
        // fast matmul path (see candle_nn::Linear); a bare rank-1 tensor
        // falls through to a plain `matmul` against the (in, out) weight,
        // which candle requires matching ranks for. `text_rep` is 1D
        // (pool_mean fully reduces the sequence dim), so it needs an
        // explicit batch dim of 1 going in and stripping coming back out.
        // `category_rep` is already 2D (`n_routes`, hidden) and needs
        // neither.
        let query = l2_normalize(
            &self
                .tok_proj
                .forward(&text_rep.unsqueeze(0)?)?
                .squeeze(0)?
                .to_dtype(DType::F32)?,
        )?;
        let categories =
            l2_normalize(&self.rule_proj.forward(&category_rep)?.to_dtype(DType::F32)?)?;

        // dot(query, categories[r]) per route: broadcast-multiply query
        // against every row, then sum the projection dimension.
        let cosines = categories.broadcast_mul(&query)?.sum(1)?;
        let logits = cosines.affine(self.scale as f64, self.score_bias as f64)?;
        let probs = candle_nn::ops::softmax_last_dim(&logits)?;

        let text_rep = text_rep.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let category_rep = category_rep.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let query = query.to_vec1::<f32>()?;
        let categories = categories.to_vec2::<f32>()?;
        let cosines = cosines.to_vec1::<f32>()?;
        let logits = logits.to_vec1::<f32>()?;
        let probs = probs.to_vec1::<f32>()?;

        require_finite("query", &query)?;
        require_finite(
            "categories",
            categories.iter().flatten().copied().collect::<Vec<_>>().as_slice(),
        )?;
        require_finite("probs", &probs)?;

        Ok(RouteComputation {
            text_rep,
            category_rep,
            query,
            categories,
            cosines,
            logits,
            probs,
        })
    }

    /// Pre-softmax scaled logits: `cosine * scale + score_bias`, one per
    /// route, in `routes` order. Does NOT sum to anything in particular
    /// across routes — see the module docs on why this (or
    /// [`Self::route_cosines`]) is the right score for a fan-out/multi-label
    /// caller, where [`Self::route_probs`]' softmax would actively be wrong.
    pub fn route_scores<S: AsRef<str>>(&self, text: &str, routes: &[S]) -> Result<Vec<f32>> {
        Ok(self.compute(text, routes)?.logits)
    }

    /// [`Self::route_scores`], pooling the query over `focus` only. See the
    /// module docs' "Focused pooling" section.
    pub fn route_scores_focused<S: AsRef<str>>(
        &self,
        text: &str,
        focus: Range<usize>,
        routes: &[S],
    ) -> Result<Vec<f32>> {
        Ok(self.compute_focused(text, focus, routes)?.logits)
    }

    /// Raw cosine similarity `dot(query, categories[r])`, before `scale`
    /// and `score_bias` — the cheapest-to-interpret per-route number this
    /// head produces (bounded to `[-1, 1]` by construction), for the same
    /// fan-out use case as [`Self::route_scores`].
    pub fn route_cosines<S: AsRef<str>>(&self, text: &str, routes: &[S]) -> Result<Vec<f32>> {
        Ok(self.compute(text, routes)?.cosines)
    }

    /// Softmax over [`Self::route_scores`] — LiquidAI's own `route()`
    /// contract. Sums to 1.0 across `routes`; right for "pick exactly one
    /// lane" dispatch, wrong for fan-out (see the module docs).
    pub fn route_probs<S: AsRef<str>>(&self, text: &str, routes: &[S]) -> Result<Vec<f32>> {
        Ok(self.compute(text, routes)?.probs)
    }

    /// [`Self::route_probs`], pooling the query over `focus` only. See the
    /// module docs' "Focused pooling" section.
    pub fn route_probs_focused<S: AsRef<str>>(
        &self,
        text: &str,
        focus: Range<usize>,
        routes: &[S],
    ) -> Result<Vec<f32>> {
        Ok(self.compute_focused(text, focus, routes)?.probs)
    }

    /// Convenience for the guard-evasion-retry case: `history` (oldest
    /// first) is joined with `"\n"` and run through the trunk in full — so
    /// every earlier entry is visible to bidirectional attention and can
    /// shape the last entry's hidden states — but the query is pooled ONLY
    /// over the LAST entry's own tokens. Earlier entries are context only;
    /// they carry no direct pooling weight, so a 5-command window doesn't
    /// dilute the one command actually being judged down to 1/5 mass. See
    /// the module docs' "Focused pooling" section.
    ///
    /// Errors if `history` is empty (nothing to focus on).
    pub fn route_scores_windowed<H: AsRef<str>, R: AsRef<str>>(
        &self,
        history: &[H],
        routes: &[R],
    ) -> Result<Vec<f32>> {
        let (text, focus) = windowed_focus(history)?;
        self.route_scores_focused(&text, focus, routes)
    }

    /// As [`Self::route_scores_windowed`], but softmax probabilities.
    pub fn route_probs_windowed<H: AsRef<str>, R: AsRef<str>>(
        &self,
        history: &[H],
        routes: &[R],
    ) -> Result<Vec<f32>> {
        let (text, focus) = windowed_focus(history)?;
        self.route_probs_focused(&text, focus, routes)
    }

    /// [`Self::route_probs`], paired back up with route names, sorted
    /// descending, and optionally threshold-filtered — mirroring Python
    /// `Lfm2BidirForSequenceRouting.route()`'s return shape. The threshold
    /// filters AFTER softmax, so surviving scores still reflect the full
    /// `routes` set's competition, not a renormalization over the survivors.
    pub fn route<S: AsRef<str>>(
        &self,
        text: &str,
        routes: &[S],
        threshold: Option<f32>,
    ) -> Result<Vec<RouteMatch>> {
        let probs = self.route_probs(text, routes)?;
        let mut matches: Vec<RouteMatch> = routes
            .iter()
            .zip(probs)
            .filter(|(_, score)| threshold.is_none_or(|t| *score >= t))
            .map(|(route, score)| RouteMatch {
                route: route.as_ref().to_string(),
                score,
            })
            .collect();
        matches.sort_by(|a, b| b.score.total_cmp(&a.score));
        Ok(matches)
    }

    /// Route each clause of a decomposed statement separately, then union
    /// the results — the answer to "one pooled vector carries one intent"
    /// (see [`ClauseRouting`] and the module docs' fan-out section).
    ///
    /// One trunk pass per clause: this costs `clauses.len()` forward passes,
    /// not one. That is the price of the second intent; a statement's
    /// clause count is small (the 15 hardest probes of the router study
    /// decomposed to 1–4 clauses each).
    ///
    /// `clauses` must come from a real parser — kaish's
    /// `plan_statement` (`Plan.commands[]`) is what this was built against.
    /// This crate deliberately ships NO splitter: a regex that splits on
    /// `&&` and `|` would disagree with the parser that actually decides
    /// what runs, and disagreeing about clause boundaries is exactly the
    /// silent-wrong-answer shape the whole crate refuses. Feed it real
    /// clauses or feed it the whole statement as one clause.
    ///
    /// Errors on an empty `clauses` (nothing to route — an empty union
    /// would read as "no intent detected", which is the most dangerous
    /// possible thing to say about a statement nobody looked at) and on an
    /// empty `routes`, same as every other method here.
    pub fn route_clauses<C: AsRef<str>, S: AsRef<str>>(
        &self,
        clauses: &[C],
        routes: &[S],
    ) -> Result<ClauseRouting> {
        if clauses.is_empty() {
            return Err(Error::InvalidInput(
                "route_clauses needs at least one clause: an empty clause list has no \
                 union to report, and reporting one would say 'no intent' about a \
                 statement that was never scored"
                    .into(),
            ));
        }
        require_nonempty_routes(routes)?;

        let clause_cosines: Vec<Vec<f32>> = clauses
            .iter()
            .map(|c| self.route_cosines(c.as_ref(), routes))
            .collect::<Result<_>>()?;

        // Max, not mean: a lane fires if ANY clause fires it. Averaging
        // would let three benign clauses dilute one destructive one — the
        // same dilution the windowing experiment already measured, moved
        // one level up.
        let mut union_cosines = vec![f32::NEG_INFINITY; routes.len()];
        for row in &clause_cosines {
            for (lane, &cos) in row.iter().enumerate() {
                union_cosines[lane] = union_cosines[lane].max(cos);
            }
        }

        Ok(ClauseRouting {
            clause_cosines,
            union_cosines,
        })
    }

    /// The underlying trunk, for callers that want raw hidden states.
    pub fn trunk(&self) -> &Lfm2Trunk {
        &self.trunk
    }

    /// This checkpoint's own tokenizer, for a caller that needs to inspect
    /// tokenization directly (e.g. `lfm2d`'s `POST /v1/tokenize`).
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
}

/// One statement's routing, computed clause by clause.
///
/// Whole-statement routing suppresses every intent but one: on
/// `git clone … && npm install` the package lane reads +0.95 while git
/// reads **−0.93**, and no threshold recovers a lane that is confidently
/// negative. The suppression happens in the pooled representation, upstream
/// of both the softmax and the cosine, so the only place to fix it is
/// upstream of the model — route the clauses, union the answers.
///
/// Cosines, never probabilities: this head's softmax is a pure function of
/// route count and carries no confidence at all (module docs, finding 2).
/// Across clauses it would be worse than useless — each clause's softmax
/// renormalizes over the same lanes, so a clause with no real intent still
/// hands its highest lane a large share.
#[derive(Debug, Clone)]
pub struct ClauseRouting {
    /// Raw cosines per clause, `[n_clauses][n_routes]`, both in input
    /// order. Kept rather than folded away: which clause fired a lane is
    /// the thing a gate explains itself with, and a union alone cannot say.
    pub clause_cosines: Vec<Vec<f32>>,
    /// Per-lane max over clauses, `[n_routes]` — a lane fires if any clause
    /// fires it. Not a distribution and does not sum to anything.
    pub union_cosines: Vec<f32>,
}

impl ClauseRouting {
    /// Lane indices whose union cosine is at least `threshold`, in
    /// descending score order. The fan-out analysis found no usable global
    /// threshold on whole statements (best balance still missed 35–40% of
    /// correct lanes); the caller picks one and owns that choice.
    pub fn lanes_above(&self, threshold: f32) -> Vec<usize> {
        let mut lanes: Vec<usize> = (0..self.union_cosines.len())
            .filter(|&i| self.union_cosines[i] >= threshold)
            .collect();
        lanes.sort_by(|&a, &b| self.union_cosines[b].total_cmp(&self.union_cosines[a]));
        lanes
    }

    /// Which clause supplied lane `lane`'s union score, as an index into
    /// [`Self::clause_cosines`]. The provenance a gate quotes when it
    /// explains why it fired.
    pub fn firing_clause(&self, lane: usize) -> Option<usize> {
        (0..self.clause_cosines.len()).max_by(|&a, &b| {
            self.clause_cosines[a][lane].total_cmp(&self.clause_cosines[b][lane])
        })
    }
}

/// Join `history` with `"\n"` and compute the byte focus range of its LAST
/// entry — shared by [`Lfm2SequenceRouter::route_scores_windowed`] and
/// [`Lfm2SequenceRouter::route_probs_windowed`]. The last entry, after a
/// `"\n"`-join, always occupies exactly its own `len()` bytes at the tail
/// of the joined string, regardless of what came before it or of any
/// non-ASCII content — no separate span bookkeeping needed.
fn windowed_focus<H: AsRef<str>>(history: &[H]) -> Result<(String, Range<usize>)> {
    let last = history.last().ok_or_else(|| {
        Error::InvalidInput(
            "route_scores_windowed/route_probs_windowed need at least one history entry to \
             focus on"
                .to_string(),
        )
    })?;
    let text = history.iter().map(|h| h.as_ref()).collect::<Vec<_>>().join("\n");
    let end = text.len();
    let start = end - last.as_ref().len();
    Ok((text, start..end))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_prefix_matches_the_python_string_shape() {
        let routes = ["Coding", "Sales"];
        let prefix = build_prefix(&routes);
        assert_eq!(prefix, "Categories:\n- Coding\n- Sales\n\nText:\n");
    }

    /// The deliberately-unused-in-production degenerate form — kept as a
    /// faithful port of the Python for its own sake (see the doc comment
    /// on [`build_prefix`]); the public heads never reach this because
    /// `compute_focused` refuses empty routes first.
    #[test]
    fn build_prefix_empty_routes_matches_pythons_degenerate_form() {
        let routes: [&str; 0] = [];
        assert_eq!(build_prefix(&routes), "Categories:\n- (none)\n\nText:\n");
    }

    #[test]
    fn category_ranges_span_math_is_correct_for_ascii() {
        let routes = ["Coding", "Sales"];
        let prefix = build_prefix(&routes);
        let ranges = category_ranges(&routes);
        assert_eq!(ranges.len(), 2);
        for (route, &(start, end)) in routes.iter().zip(&ranges) {
            assert_eq!(&prefix[start..end], *route, "route {route:?} span {start}..{end}");
        }
    }

    /// The regression test for the byte-vs-char trap: `category_ranges`
    /// must locate each route by BYTE offset. Slicing the constructed
    /// prefix at the computed span must recover the exact original route
    /// text. If this module used `.chars().count()` instead of `.len()`
    /// (mimicking Python's unit rather than Rust's own), this slice would
    /// either panic (landing mid-character, not a UTF-8 boundary) or
    /// silently return the wrong substring — see the module docs.
    #[test]
    fn category_ranges_span_math_handles_non_ascii_routes() {
        let routes = ["日本語のカテゴリ", "Emoji 🎉 route", "English"];
        let prefix = build_prefix(&routes);
        let ranges = category_ranges(&routes);
        assert_eq!(ranges.len(), routes.len());
        for (route, &(start, end)) in routes.iter().zip(&ranges) {
            assert!(prefix.is_char_boundary(start), "route {route:?}: start {start} not a char boundary");
            assert!(prefix.is_char_boundary(end), "route {route:?}: end {end} not a char boundary");
            assert_eq!(&prefix[start..end], *route, "route {route:?} span {start}..{end}");
        }
    }

    #[test]
    fn select_overlapping_tokens_default_focus_excludes_the_categories_header_and_specials() {
        // A synthetic offset table over some `full_text` where text_start
        // (and thus the default focus's absolute start) is 10: two header
        // tokens fully before it, an empty-span special token straddling
        // it, then two real text tokens. abs_end is the full length, 18.
        let text_start = 10;
        let full_len = 18;
        let offsets = [(0, 4), (4, 10), (10, 10), (10, 13), (13, 18)];
        let idxs = select_overlapping_tokens(&offsets, (text_start, full_len));
        assert_eq!(idxs, vec![3, 4], "only tokens overlapping [text_start, full_len) with a non-empty span");
    }

    #[test]
    fn select_overlapping_tokens_uses_overlap_not_containment() {
        // Range (5, 12). A token that starts before and ends inside (2,7)
        // must count (it overlaps); a token entirely before (0,5) must not
        // (touching, not overlapping); an empty-span token at the boundary
        // (12,12) must not.
        let offsets = [(0, 5), (2, 7), (7, 12), (12, 12), (12, 15)];
        let idxs = select_overlapping_tokens(&offsets, (5, 12));
        assert_eq!(idxs, vec![1, 2], "overlap test: touching-but-not-overlapping tokens excluded");
    }

    #[test]
    fn select_overlapping_tokens_on_a_range_with_no_surviving_tokens_is_empty() {
        let offsets = [(0, 5), (5, 10)];
        let idxs = select_overlapping_tokens(&offsets, (100, 105));
        assert!(idxs.is_empty());
    }

    #[test]
    fn require_nonempty_routes_rejects_empty_and_names_the_reason() {
        let empty: [&str; 0] = [];
        let err = require_nonempty_routes(&empty).expect_err("empty routes must be refused");
        let msg = err.to_string();
        assert!(msg.contains("route"), "{msg}");

        require_nonempty_routes(&["Coding"]).expect("non-empty routes must pass");
    }

    #[test]
    #[allow(clippy::reversed_empty_ranges)] // deliberate: proving an inverted focus is refused
    fn validate_focus_rejects_empty_out_of_range_and_non_boundary() {
        let text = "hello world"; // 11 bytes, all ASCII
        validate_focus(text, &(0..11)).expect("full range is valid");
        validate_focus(text, &(6..11)).expect("a real sub-range is valid");

        validate_focus(text, &(3..3)).expect_err("empty focus must be refused");
        validate_focus(text, &(0..12)).expect_err("out-of-range focus must be refused");
        validate_focus(text, &(5..3)).expect_err("inverted focus must be refused");

        // "日" is 3 bytes; byte 1 lands inside it.
        let ja = "日本語";
        validate_focus(ja, &(0..3)).expect("a real char boundary range is valid");
        validate_focus(ja, &(1..3)).expect_err("a mid-character start must be refused");
        validate_focus(ja, &(0..2)).expect_err("a mid-character end must be refused");
    }

    #[test]
    fn require_nonempty_selection_rejects_empty_idxs_only() {
        require_nonempty_selection(&[], &(0..5), 5).expect_err("empty selection must be refused");
        require_nonempty_selection(&[0, 2], &(0..5), 5).expect("non-empty selection must pass");
    }

    #[test]
    fn windowed_focus_targets_only_the_last_entry_regardless_of_earlier_content() {
        let history = ["kubectl get pods", "rm -rf /data", "rm -r /data"];
        let (text, focus) = windowed_focus(&history).expect("non-empty history");
        assert_eq!(text, "kubectl get pods\nrm -rf /data\nrm -r /data");
        assert_eq!(&text[focus.clone()], "rm -r /data");
        // Not the earlier, superficially similar "rm -rf" entry.
        assert_ne!(&text[focus], "rm -rf /data");
    }

    #[test]
    fn windowed_focus_handles_non_ascii_history_by_bytes() {
        let history = ["日本語のコマンド", "rm -r /data"];
        let (text, focus) = windowed_focus(&history).expect("non-empty history");
        assert_eq!(&text[focus], "rm -r /data");
    }

    #[test]
    fn windowed_focus_rejects_empty_history() {
        let empty: [&str; 0] = [];
        windowed_focus(&empty).expect_err("empty history must be refused");
    }
}
