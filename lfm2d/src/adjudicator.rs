//! Causal adjudication with one resident prompt and isolated evaluation branches.
//!
//! The worker owns the model. Prefix state is immutable and tensor-shared; each
//! request appends into replacement storage. Outputs are model reports, never
//! executed actions. The protocol deliberately exposes truncation and errors.
use crate::{
    config::Cli,
    hash::{sha256_hex_bytes, sha256_hex_file},
    types::{ModelInfo, ModelKind},
    worker::WorkerExit,
};
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use candle_core::{Tensor, quantized::gguf_file};
use candle_nn::sampling::GreedySampler;
use candle_transformers::models::quantized_lfm2_moe::{Model, State as ModelState};
use serde::{Deserialize, Serialize};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};
use tokio::sync::oneshot;

pub(crate) const CHUNK: usize = 128;
/// `<|im_end|>`, which ends a turn. `Checkpoint::load` refuses a tokenizer that
/// puts it anywhere else.
const EOS: u32 = 124900;
const MAX_NEW: usize = 2048;
const MAX_INPUT_BYTES: usize = 65536;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PromptSpec {
    pub system: String,
    #[serde(default)]
    pub tools: Vec<serde_json::Value>,
    #[serde(default)]
    pub output_schema: Option<serde_json::Value>,
    /// Whether the assistant's turn opens with a finished reasoning region.
    /// Policy, so it lives with the prompt rather than in the renderer.
    #[serde(default)]
    pub reasoning: Reasoning,
    /// The System-1 question asked at the verdict slot by `opinion` requests.
    /// Not part of the prefix: it is rendered into each request's suffix.
    #[serde(default)]
    pub opinion: Option<crate::opinion::OpinionSpec>,
}

/// Whether the assistant's turn opens with a *completed* reasoning region.
///
/// The checkpoint's chat template never emits one: `<think>` is a token the
/// model writes, and after `<|im_start|>assistant\n` it writes it at p=1.00.
/// Under an `output_schema` the grammar's first legal byte is the object's, so
/// the model was masked off its own manifold at step 0 — the forced `{"` scored
/// logprob -17.8 to -21.8, and every later token was conditioned on a prefix
/// the model considers impossible.
///
/// `Closed` prefills the region already finished, so generation begins where the
/// report begins. The bytes are `<think>\n\n</think>\n` and they were measured,
/// not derived: the template's own dialect for a completed region
/// (`"<think>" + thinking + "</think>"`, no surrounding newlines) leaves the
/// model wanting a newline at p=0.97, and the grammar forces `{"` from rank 5.
/// Five candidates over three inputs on ROCm, `{"`'s standing at the slot:
///
/// | after `<|im_start|>assistant\n` | `{"` |
/// |---|---|
/// | nothing (v1)          | logprob -17.8 to -21.8, rank 6 or worse |
/// | `<think></think>`     | rank 5 |
/// | `<think></think>\n`   | rank 2-3 |
/// | `<think>\n</think>\n` | rank 2 |
/// | `<think>\n\n</think>\n` | **rank 1, p 0.42-0.71** |
///
/// A second round varied one newline at a time (one more inside, one more
/// after, none after) and every neighbour was worse, so this is a local
/// optimum rather than the best of an arbitrary five.
/// `~/exomemory/lfm2d/lfm25-think-prefill-2026-09-18/`.
///
/// `Open` leaves the turn as the template opens it and the model reasons. That
/// is today's free-generation behaviour and is honest *without* a schema.
/// Combined with one it is refused: the grammar would have to admit a free-text
/// region it cannot bound and then start the object after `</think>`, which is
/// not built. See `crate::constrain`'s first ruling, and the measurement that
/// LFM2.5's reasoning argues severity *down*.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Reasoning {
    #[default]
    Closed,
    Open,
}
impl Reasoning {
    /// What the renderer writes after `<|im_start|>assistant\n`.
    fn opening(self) -> &'static str {
        match self {
            Reasoning::Closed => "<think>\n\n</think>\n",
            Reasoning::Open => "",
        }
    }
}
/// Prompt content carries no model control tokens; the renderer supplies them.
pub fn validate_text(s: &str) -> Result<(), String> {
    if s.trim().is_empty() {
        return Err("input must not be empty".into());
    }
    if s.contains("<|") || s.contains("<think>") || s.contains("</think>") {
        return Err("literal model control tokens are not allowed in prompt content".into());
    }
    Ok(())
}
impl PromptSpec {
    /// The checkpoint's single-system/single-user chat-template subset. Tool
    /// schemas are part of the frozen system message, as in the GGUF template.
    pub fn render_prefix(&self) -> Result<String, String> {
        validate_text(&self.system)?;
        if self.reasoning == Reasoning::Open && self.output_schema.is_some() {
            return Err(
                "reasoning \"open\" under an output_schema is not built: the grammar would have \
                 to admit an unbounded reasoning region and start the object after </think>"
                    .into(),
            );
        }
        let mut system = self.system.clone();
        if let Some(schema) = &self.output_schema {
            if !self.tools.is_empty() {
                return Err("choose output_schema or tools, not both".into());
            }
            validate_schema(schema)?;
            let schema = render_schema(schema)?;
            validate_text(&schema)?;
            system.push_str("\nReturn exactly one JSON object matching this schema: ");
            system.push_str(&schema);
        }
        if !self.tools.is_empty() {
            let mut rendered = Vec::new();
            for tool in &self.tools {
                if tool["type"] != "function" || !tool["function"]["name"].is_string() {
                    return Err("each tool must be a named function schema".into());
                }
                let json = serde_json::to_string(tool).map_err(|e| e.to_string())?;
                validate_text(&json)?;
                rendered.push(json);
            }
            system.push_str("\nList of tools: [");
            system.push_str(&rendered.join(", "));
            system.push(']');
        }
        Ok(format!(
            "<|startoftext|><|im_start|>system\n{system}<|im_end|>\n"
        ))
    }

    /// The single user turn and the opening of the assistant's, appended to
    /// [`PromptSpec::render_prefix`]. One definition, so anything examining a
    /// prompt renders exactly what the daemon runs.
    pub fn render_user_turn(&self, input: &str) -> String {
        let opening = self.reasoning.opening();
        format!("<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n{opening}")
    }

    /// The rendered template is different bytes per [`Reasoning`] mode, so the
    /// mode is part of the version a consumer reads from `/v1/adjudicator`,
    /// and part of the prefix snapshot's identity. v1 was `closed`'s bytes
    /// without the reasoning region.
    pub fn template_version(&self) -> &'static str {
        match self.reasoning {
            Reasoning::Closed => "lfm25-single-user-v2-closed",
            Reasoning::Open => "lfm25-single-user-v2-open",
        }
    }

    /// The user turn continued by text the assistant has already written, for
    /// the examination tools that stand at an answer slot. A prefill carrying
    /// reasoning delimiters would render a second region beside the one a
    /// `closed` spec has already written — `<think>\n\n</think>\n<think>...`
    /// — which is not a prompt anyone meant to examine, so it is refused
    /// rather than quietly rendered. Both delimiters are refused: a bare
    /// `</think>` closes a region that is already closed.
    ///
    /// The remedy is exact text, not `reasoning: "open"`, which is refused
    /// alongside an `output_schema` and so is unavailable for exactly the
    /// schema-bearing prompts these tools mostly examine.
    pub fn render_user_turn_with_prefill(
        &self,
        input: &str,
        prefill: &str,
    ) -> Result<String, String> {
        let reasoning_delimiter = prefill.contains("<think>") || prefill.contains("</think>");
        if self.reasoning == Reasoning::Closed && reasoning_delimiter {
            return Err(
                "this prompt spec already closes a reasoning region, so a prefill cannot carry \
                 reasoning delimiters: supply the whole prompt as exact text instead"
                    .into(),
            );
        }
        Ok(format!("{}{prefill}", self.render_user_turn(input)))
    }
}



#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdjudicateRequest {
    pub input: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    /// Explicit cold baseline for parity/latency measurements, not a fallback.
    #[serde(default = "yes")]
    pub use_cache: bool,
    #[serde(default = "default_timeout")]
    pub timeout_ms: u64,
    /// Per-generated-token distributions (top-k logprobs, raw named-set
    /// mass). Omitted entirely (the default) => `AdjudicateResponse` is
    /// today's shape, byte-identical; see `docs/field-requests.md`
    /// decision 5 and [`crate::types::DistributionRequest`].
    #[serde(default)]
    pub distributions: Option<crate::types::DistributionRequest>,
    /// Read the spec's `opinion` question instead of generating: one prefill,
    /// every option scored, nothing decoded. See [`crate::opinion`].
    #[serde(default)]
    pub opinion: bool,
}
fn default_max_tokens() -> usize {
    2048
}
fn default_timeout() -> u64 {
    30000
}
fn yes() -> bool {
    true
}
impl AdjudicateRequest {
    pub fn validate(&self) -> Result<(), String> {
        validate_text(&self.input)?;
        if self.input.len() > MAX_INPUT_BYTES {
            return Err("input exceeds 65536 bytes".into());
        }
        if self.max_tokens == 0 || self.max_tokens > MAX_NEW {
            return Err("max_tokens must be 1..=2048".into());
        }
        if self.timeout_ms == 0 || self.timeout_ms > 120000 {
            return Err("timeout_ms must be 1..=120000".into());
        }
        if let Some(d) = &self.distributions {
            d.validate()?;
        }
        if self.opinion && self.distributions.is_some() {
            return Err("an opinion read decodes nothing, so distributions do not apply".into());
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct PrefixInfo {
    pub model_id: String,
    pub weight_hash: String,
    pub tokenizer_hash: String,
    pub template_version: String,
    pub snapshot_id: String,
    pub prefix_tokens: usize,
    /// Complete-input checkpoints retained alongside the fixed prefix.
    pub input_cache_capacity: usize,
    pub context_limit: usize,
    pub backend: String,
    pub dtype: String,
    pub sampling: String,
    pub weight_dtypes: Vec<String>,
}
#[derive(Clone, Debug, Serialize)]
pub struct AdjudicateResponse {
    #[serde(flatten)]
    pub prefix: PrefixInfo,
    /// Includes reasoning and native tool-call delimiters. No tool is executed.
    pub output: String,
    pub report: Option<serde_json::Value>,
    pub report_error: Option<String>,
    pub finish_reason: String,
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub completion_tokens: usize,
    pub queue_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    /// One entry per generated token (including a trailing eos, if any),
    /// same count as `completion_tokens`. Present only when the request set
    /// `distributions` — `#[serde(skip_serializing_if)]` keeps the field
    /// entirely absent from the JSON body otherwise, so a request that
    /// never asked for it gets byte-identical output to before this field
    /// existed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub distributions: Option<Vec<crate::types::StepDistribution>>,
    /// Present only on an `opinion` request. `output` is then empty,
    /// `finish_reason` is `opinion`, `prompt_tokens` counts the prefilled
    /// shared prefix (the scored continuations are in `opinion.scored_tokens`),
    /// `completion_tokens` is 0 and `decode_ms` is the time spent scoring the
    /// options.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub opinion: Option<crate::opinion::OpinionRead>,
    /// Present only when the generation resumed from a described state that
    /// `/v1/opinion` left for these exact prompt bytes: how many of
    /// `completion_tokens` were taken from it rather than decoded now. The
    /// report is the one a fresh generation writes (greedy under the grammar
    /// is a pure function of the prompt); only the time differs. Cold
    /// requests and requests asking for `distributions` never resume.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resumed_tokens: Option<usize>,
}

#[derive(Debug)]
pub enum Failure {
    BadRequest(String),
    Internal(String),
    Cancelled,
    Deadline,
}
impl From<candle_core::Error> for Failure {
    fn from(e: candle_core::Error) -> Self {
        Self::Internal(e.to_string())
    }
}
impl IntoResponse for Failure {
    fn into_response(self) -> Response {
        let (status, kind, msg) = match self {
            Self::BadRequest(s) => (StatusCode::BAD_REQUEST, "bad_request", s),
            Self::Internal(s) => (StatusCode::INTERNAL_SERVER_ERROR, "internal_error", s),
            Self::Cancelled => (
                StatusCode::REQUEST_TIMEOUT,
                "cancelled",
                "evaluation cancelled".into(),
            ),
            Self::Deadline => (
                StatusCode::GATEWAY_TIMEOUT,
                "deadline",
                "evaluation deadline exceeded".into(),
            ),
        };
        (
            status,
            Json(serde_json::json!({"error":{"type":kind,"message":msg}})),
        )
            .into_response()
    }
}

pub trait Generator: Send + 'static {
    fn generate(
        &mut self,
        request: &AdjudicateRequest,
        check: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<AdjudicateResponse, Failure>;
    /// Describe-then-read for `/v1/opinion`: `question` was resolved against
    /// the spec's menu by the handler. See [`crate::opinion_api`].
    fn opine(
        &mut self,
        request: &crate::opinion_api::OpinionRequest,
        question: &crate::opinion_api::ResolvedQuestion,
        check: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<crate::opinion_api::OpinionResponse, Failure>;
}

// One exact-input checkpoint in addition to the fixed system prefix. This is
// reusable model computation, never a cached adjudication or mutable sampler.
struct PreparedPrompt {
    token_ids: Vec<u32>,
    state: ModelState,
    logits: Tensor,
}
struct PreparedEvaluation {
    state: ModelState,
    logits: Tensor,
    cached_tokens: usize,
}
struct PromptCache {
    prefix: ModelState,
    prefix_ids: Vec<u32>,
    ready: Option<PreparedPrompt>,
}
impl PromptCache {
    fn prepare(
        &mut self,
        model: &Model,
        full: &[u32],
        use_cache: bool,
        check: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<PreparedEvaluation, Failure> {
        check()?;
        if !model.owns_state(&self.prefix) {
            return Err(Failure::Internal(
                "input checkpoint belongs to another model".into(),
            ));
        }
        if !full.starts_with(&self.prefix_ids) || full.len() <= self.prefix_ids.len() {
            return Err(Failure::BadRequest(
                "rendered input changes cached token prefix or has no suffix".into(),
            ));
        }
        if use_cache
            && let Some(ready) = &self.ready
            && ready.token_ids == full
        {
            return Ok(PreparedEvaluation {
                state: ready.state.clone(),
                logits: ready.logits.clone(),
                cached_tokens: full.len(),
            });
        }
        let (mut state, start) = if use_cache {
            (self.prefix.clone(), self.prefix_ids.len())
        } else {
            (model.new_state(), 0)
        };
        let mut logits = None;
        for chunk in full[start..].chunks(CHUNK) {
            check()?;
            logits = Some(model.forward(chunk, &mut state)?);
        }
        let logits = logits.ok_or_else(|| Failure::Internal("no suffix tokens".into()))?;
        // Publish only complete prefill. A failed/cancelled preparation retains
        // the previous entry; later decode never mutates the saved state/logits.
        logits.device().synchronize()?;
        check()?;
        if use_cache {
            self.ready = Some(PreparedPrompt {
                token_ids: full.to_vec(),
                state: state.clone(),
                logits: logits.clone(),
            });
        }
        Ok(PreparedEvaluation {
            state,
            logits,
            cached_tokens: start,
        })
    }
}

/// A validated LFM2.5 GGUF with its tokenizer: the chat template is the one
/// this renderer was checked against, the tokenizer agrees with the GGUF
/// vocabulary row for row, and both files are hashed. The daemon and the
/// examiner load through here so neither can run an unvalidated pairing.
pub struct Checkpoint {
    pub model: Model,
    pub tokenizer: tokenizers::Tokenizer,
    pub model_id: String,
    pub weight_hash: String,
    pub tokenizer_hash: String,
    pub weight_dtypes: Vec<String>,
    pub execution: crate::device::ExecutionDevice,
}
impl Checkpoint {
    pub fn load(
        path: &std::path::Path,
        tokenizer_path: &std::path::Path,
        device: crate::device::DeviceArg,
        device_index: usize,
    ) -> Result<Self, String> {
        let mut tokenizer =
            tokenizers::Tokenizer::from_file(tokenizer_path).map_err(|e| e.to_string())?;
        tokenizer.with_padding(None);
        tokenizer.with_truncation(None).map_err(|e| e.to_string())?;
        let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
        let ct = gguf_file::Content::read(&mut file).map_err(|e| e.to_string())?;
        let embedded_template = ct
            .metadata
            .get("tokenizer.chat_template")
            .ok_or("GGUF has no chat template")?
            .to_string()
            .map_err(|e| e.to_string())?;
        if sha256_hex_bytes(embedded_template.as_bytes())
            != "6d65c8804847ad74eea912dd7eca3dc1cf7a457b53a77f47d841a14121910963"
        {
            return Err("unsupported LFM2.5 chat template; validate the renderer before using this checkpoint".into());
        }
        let weight_dtypes: Vec<String> = ct
            .tensor_infos
            .values()
            .map(|t| format!("{:?}", t.ggml_dtype))
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        // Match every usable token, including control tokens, to these weights.
        // GGUF pads vocabulary rows beyond the tokenizer's usable vocabulary.
        let vocab = ct
            .metadata
            .get("tokenizer.ggml.tokens")
            .ok_or("GGUF has no tokenizer vocabulary")?
            .to_vec()
            .map_err(|e| e.to_string())?;
        for (text, id) in tokenizer.get_vocab(true) {
            if vocab
                .get(id as usize)
                .and_then(|v| v.to_string().ok())
                .map(String::as_str)
                != Some(text.as_str())
            {
                return Err(format!("tokenizer disagrees with GGUF at token {id}"));
            }
        }
        for (text, id) in [
            ("<|startoftext|>", 124894),
            ("<|im_start|>", 124899),
            ("<|im_end|>", EOS),
        ] {
            if tokenizer.token_to_id(text) != Some(id) {
                return Err(format!("unsupported LFM2.5 control token {text}"));
            }
        }
        let weight_hash = sha256_hex_file(path).map_err(|e| e.to_string())?;
        let tokenizer_hash = sha256_hex_file(tokenizer_path).map_err(|e| e.to_string())?;
        let execution = crate::device::ExecutionDevice::select(device, device_index)?;
        for reason in &execution.selection_reasons {
            eprintln!("lfm2d adjudicator device: {reason}");
        }
        let model =
            Model::from_gguf(ct, &mut file, &execution.device).map_err(|e| e.to_string())?;
        let model_id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or("invalid GGUF filename")?
            .to_string();
        Ok(Self {
            model,
            tokenizer,
            model_id,
            weight_hash,
            tokenizer_hash,
            weight_dtypes,
            execution,
        })
    }
}

/// How many described states each spec keeps ([`DescribedCache`]).
pub const DESCRIBED_CACHE_CAPACITY: usize = 16;

/// The prompt plus its generated description, standing at a question's slot:
/// the fourth cache layer. Greedy decoding under the grammar is a pure
/// function of the rendered prompt on a fixed backend, so the description
/// and the model state after it are reusable computation — never a cached
/// answer; the options are always scored fresh. Keyed by the exact prompt
/// token ids and the question field (a different field stops at a different
/// slot). Least recently used goes first.
struct DescribedEntry {
    prompt_ids: Vec<u32>,
    field: String,
    generated: Vec<u32>,
    text: String,
    state: ModelState,
    logits: Tensor,
}
struct DescribedCache {
    capacity: usize,
    entries: std::collections::VecDeque<DescribedEntry>,
}
impl DescribedCache {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: std::collections::VecDeque::with_capacity(capacity),
        }
    }
    /// A hit is moved to the back (most recent) and its parts cloned:
    /// `State::clone` shares the immutable prefix buffers.
    fn get(&mut self, prompt_ids: &[u32], field: &str) -> Option<(Vec<u32>, String, ModelState, Tensor)> {
        let at = self
            .entries
            .iter()
            .position(|e| e.field == field && e.prompt_ids == prompt_ids)?;
        let entry = self.entries.remove(at).expect("position came from this deque");
        let out = (
            entry.generated.clone(),
            entry.text.clone(),
            entry.state.clone(),
            entry.logits.clone(),
        );
        self.entries.push_back(entry);
        Some(out)
    }
    /// The deepest description held for these prompt bytes, whatever field
    /// it stopped at: the most of the report that a generation can resume
    /// from. Moved to the back like a hit.
    fn get_deepest(&mut self, prompt_ids: &[u32]) -> Option<(Vec<u32>, String, ModelState, Tensor)> {
        let field = self
            .entries
            .iter()
            .filter(|e| e.prompt_ids == prompt_ids)
            .max_by_key(|e| e.generated.len())
            .map(|e| e.field.clone())?;
        self.get(prompt_ids, &field)
    }
    fn insert(&mut self, entry: DescribedEntry) {
        if self.capacity == 0 {
            return;
        }
        if let Some(at) = self
            .entries
            .iter()
            .position(|e| e.field == entry.field && e.prompt_ids == entry.prompt_ids)
        {
            self.entries.remove(at);
        }
        while self.entries.len() >= self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
    }
    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.len()
    }
}

/// One loaded prompt spec: its rendered prefix resident as model state, its
/// grammar compiled once, its identity, and the caches that hang off it.
/// `Adjudicator::specs[0]` is the `--adjudicator-prompt` spec, which
/// `/v1/adjudicate` serves; every loaded spec is on the `/v1/opinion` menu.
struct LoadedSpec {
    name: String,
    /// The spec as loaded: it renders every user turn, so the reasoning mode
    /// and the schema cannot drift apart from what the prefix was built from.
    prompt: PromptSpec,
    prefix_text: String,
    /// `prompt.output_schema` compiled against the tokenizer, once, at load.
    grammar: Option<std::sync::Arc<crate::constrain::Grammar>>,
    cache: PromptCache,
    described: DescribedCache,
    info: PrefixInfo,
    menu: crate::opinion_api::SpecMenuEntry,
}
impl LoadedSpec {
    fn load(name: String, prompt: PromptSpec, checkpoint: &Checkpoint, cli: &Cli) -> Result<Self, String> {
        let Checkpoint {
            model,
            tokenizer,
            model_id,
            weight_hash,
            tokenizer_hash,
            weight_dtypes,
            execution,
        } = checkpoint;
        let prefix_text = prompt.render_prefix()?;
        if let Some(opinion) = &prompt.opinion {
            opinion.validate()?;
            prompt.render_user_turn_with_prefill("probe", &opinion.prefill)?;
        }
        // Compile the output grammar before the prefix prefill: a schema this
        // tokenizer cannot honour refuses to start, and never reaches a request.
        let grammar = prompt
            .output_schema
            .as_ref()
            .map(|schema| {
                crate::constrain::Grammar::compile(schema, tokenizer, model.vocab_size(), EOS)
                    .map_err(|e| format!("output_schema cannot be constrained: {e}"))
            })
            .transpose()?;
        let prefix_ids = tokenizer
            .encode(prefix_text.as_str(), false)
            .map_err(|e| e.to_string())?
            .get_ids()
            .to_vec();
        let boundary_probe = tokenizer
            .encode(format!("{prefix_text}<|im_start|>user\n"), false)
            .map_err(|e| e.to_string())?;
        if !boundary_probe.get_ids().starts_with(&prefix_ids) {
            return Err("tokenizer changes the system prefix at the user-message boundary".into());
        }
        if prefix_ids.is_empty() || prefix_ids.len() + 1 >= cli.adjudicator_context {
            return Err("adjudicator prefix leaves no context for input/output".into());
        }
        // The option split depends on the tokenizer and the spec, not on the
        // input, so a close that cannot separate the options is refused here
        // and never reaches a request.
        if let Some(opinion) = &prompt.opinion {
            let probe = prompt.render_user_turn_with_prefill("probe", &opinion.prefill)?;
            crate::opinion::split_options(tokenizer, &format!("{prefix_text}{probe}"), opinion)
                .map_err(|e| format!("opinion block cannot be read with this tokenizer: {e}"))?;
        }
        let mut prefix = model.new_state();
        for chunk in prefix_ids.chunks(CHUNK) {
            let _ = model
                .forward(chunk, &mut prefix)
                .map_err(|e| e.to_string())?;
        }
        let template_version = prompt.template_version();
        // The repetition penalty shapes every description and report, so a
        // consumer that refits on `snapshot_id` must see it change.
        let mut identity = serde_json::json!([
            template_version,
            weight_hash,
            tokenizer_hash,
            prefix_ids,
            execution.backend.as_str(),
            "f32",
            cli.adjudicator_repeat_penalty
        ]);
        // The opinion question is part of what this daemon answers, so it is
        // part of its identity. Appended only when present: specs without one
        // keep the snapshot ids they had.
        if let Some(opinion) = &prompt.opinion {
            identity
                .as_array_mut()
                .expect("identity is an array")
                .push(serde_json::to_value(opinion).map_err(|e| e.to_string())?);
        }
        let snapshot_id =
            sha256_hex_bytes(&serde_json::to_vec(&identity).map_err(|e| e.to_string())?);
        let info = PrefixInfo {
            model_id: model_id.clone(),
            weight_hash: weight_hash.clone(),
            tokenizer_hash: tokenizer_hash.clone(),
            template_version: template_version.into(),
            snapshot_id,
            prefix_tokens: prefix_ids.len(),
            input_cache_capacity: 1,
            context_limit: cli.adjudicator_context,
            backend: execution.backend.as_str().into(),
            dtype: "f32".into(),
            sampling: format!(
                "greedy; repetition_penalty={}; history=full",
                cli.adjudicator_repeat_penalty
            ),
            weight_dtypes: weight_dtypes.clone(),
        };
        let menu = crate::opinion_api::SpecMenuEntry::from_prompt(
            &name,
            &prompt,
            &info.snapshot_id,
            DESCRIBED_CACHE_CAPACITY,
        )
        .map_err(|e| format!("prompt spec {name:?}: {e}"))?;
        // Every choice field's options must tokenize on their own pretokens
        // after the slot, or the read would score off the model's path. The
        // tokenizer's pre-tokenizer decides this, not the input, so it is
        // checked here on a probe and is an invariant at request time.
        for field in &menu.fields {
            if field.kind != crate::opinion_api::FieldKind::Choice {
                continue;
            }
            for (close, probe) in [("\",", "{\"x\": \"a\", \""), ("\"}", "{\"x\": \"a\", \"")] {
                let slot = format!(
                    "{probe}{}: \"",
                    serde_json::to_string(&field.field).map_err(|e| e.to_string())?
                );
                for option in &field.options {
                    let alone = encode_ids(tokenizer, &format!("{option}{close}"))?;
                    let whole = encode_ids(tokenizer, &format!("{prefix_text}{slot}{option}{close}"))?;
                    if alone.is_empty() || !whole.ends_with(&alone) {
                        return Err(format!(
                            "prompt spec {name:?}: option {option:?} of {:?} does not tokenize on \
                             its own after the slot; it cannot be read",
                            field.field
                        ));
                    }
                }
            }
        }
        Ok(Self {
            name,
            prompt,
            prefix_text,
            grammar,
            cache: PromptCache {
                prefix,
                prefix_ids,
                ready: None,
            },
            described: DescribedCache::new(DESCRIBED_CACHE_CAPACITY),
            info,
            menu,
        })
    }
}

fn encode_ids(tokenizer: &tokenizers::Tokenizer, text: &str) -> Result<Vec<u32>, String> {
    Ok(tokenizer
        .encode(text, false)
        .map_err(|e| e.to_string())?
        .get_ids()
        .to_vec())
}

pub struct Adjudicator {
    model: Model,
    tokenizer: tokenizers::Tokenizer,
    eos: u32,
    repeat_penalty: f32,
    /// `[0]` is the adjudicate spec; the rest come from `--opinion-spec`.
    specs: Vec<LoadedSpec>,
}
impl Adjudicator {
    pub fn load(cli: &Cli) -> Result<Self, String> {
        let path = cli
            .adjudicator_model
            .as_ref()
            .ok_or("missing adjudicator model")?;
        let tokenizer_path = cli
            .adjudicator_tokenizer
            .as_ref()
            .ok_or("missing adjudicator tokenizer")?;
        let prompt_path = cli
            .adjudicator_prompt
            .as_ref()
            .ok_or("missing adjudicator prompt")?;
        let checkpoint = Checkpoint::load(path, tokenizer_path, cli.device, cli.device_index)?;
        if cli.adjudicator_context > checkpoint.model.context_length() {
            return Err("adjudicator context exceeds model context".into());
        }
        let mut specs = Vec::new();
        let mut names = std::collections::BTreeSet::new();
        for spec_path in std::iter::once(prompt_path).chain(&cli.opinion_specs) {
            let name = spec_path
                .file_stem()
                .and_then(|s| s.to_str())
                .ok_or_else(|| format!("prompt spec path {} has no name", spec_path.display()))?
                .to_string();
            if !names.insert(name.clone()) {
                return Err(format!("prompt spec name {name:?} repeats; names come from file stems"));
            }
            let prompt: PromptSpec = serde_json::from_slice(
                &std::fs::read(spec_path).map_err(|e| format!("{}: {e}", spec_path.display()))?,
            )
            .map_err(|e| format!("{}: {e}", spec_path.display()))?;
            specs.push(LoadedSpec::load(name, prompt, &checkpoint, cli)?);
        }
        // Compile/warm device selection kernels before announcing readiness.
        let _ = GreedySampler::new(
            &checkpoint.execution.device,
            checkpoint.model.vocab_size(),
            &specs[0].cache.prefix_ids,
            cli.adjudicator_repeat_penalty,
        )
        .map_err(|e| e.to_string())?;
        checkpoint
            .execution
            .device
            .synchronize()
            .map_err(|e| e.to_string())?;
        let Checkpoint {
            model, tokenizer, ..
        } = checkpoint;
        Ok(Self {
            model,
            tokenizer,
            eos: EOS,
            repeat_penalty: cli.adjudicator_repeat_penalty,
            specs,
        })
    }
    /// The adjudicate spec's identity.
    pub fn info(&self) -> PrefixInfo {
        self.specs[0].info.clone()
    }
    pub fn model_info(&self) -> ModelInfo {
        ModelInfo {
            id: self.specs[0].info.model_id.clone(),
            kind: ModelKind::Adjudicator,
            weight_hash: self.specs[0].info.weight_hash.clone(),
            labels: None,
            hidden_size: self.model.hidden_size(),
        }
    }
    /// Every loaded spec, as `/v1/opinion/specs` lists it.
    pub fn menu(&self) -> Vec<crate::opinion_api::SpecMenuEntry> {
        self.specs.iter().map(|s| s.menu.clone()).collect()
    }
}
impl Generator for Adjudicator {
    fn generate(
        &mut self,
        request: &AdjudicateRequest,
        check: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<AdjudicateResponse, Failure> {
        request.validate().map_err(Failure::BadRequest)?;
        if request.opinion {
            return self.opinion(request, check);
        }
        let spec = &mut self.specs[0];
        if let Some(d) = &request.distributions {
            d.validate_vocab(self.model.vocab_size())
                .map_err(Failure::BadRequest)?;
            if d.constrained && spec.prompt.output_schema.is_none() {
                return Err(Failure::BadRequest(
                    "distributions.constrained needs an output schema; this adjudicator has none"
                        .into(),
                ));
            }
        }
        check()?;
        let suffix = spec.prompt.render_user_turn(&request.input);
        let full = self
            .tokenizer
            .encode(format!("{}{suffix}", spec.prefix_text), false)
            .map_err(|e| Failure::Internal(e.to_string()))?
            .get_ids()
            .to_vec();
        if full
            .len()
            .checked_add(request.max_tokens)
            .is_none_or(|n| n > spec.info.context_limit)
        {
            return Err(Failure::BadRequest(
                "prompt plus max_tokens exceeds adjudicator context".into(),
            ));
        }
        let begin = Instant::now();
        // Escalation from an opinion: if `/v1/opinion` left a described state
        // for these exact prompt bytes, continue from its slot rather than
        // describing again. Greedy under the grammar is a pure function of
        // the prompt, so the report is the one a fresh generation writes. A
        // request that wants every step's distribution, or a cold one, gets
        // the fresh path.
        let resumed = if request.use_cache && request.distributions.is_none() {
            spec.described
                .get_deepest(&full)
                .filter(|(generated, ..)| generated.len() < request.max_tokens)
        } else {
            None
        };
        let (mut state, mut logits, cached_tokens, mut generated, resumed_tokens) = match resumed {
            Some((generated, _, state, logits)) => {
                let n = generated.len();
                (state, logits, full.len(), generated, Some(n))
            }
            None => {
                let PreparedEvaluation {
                    state,
                    logits,
                    cached_tokens,
                } = spec
                    .cache
                    .prepare(&self.model, &full, request.use_cache, check)?;
                (state, logits, cached_tokens, Vec::new(), None)
            }
        };
        let prefill_ms = begin.elapsed().as_secs_f64() * 1000.;
        let decode = Instant::now();
        // Constrained decoding. With an `output_schema` this masks the logits
        // against the JSON grammar compiled at load before every greedy selection, so an
        // invalid report is unreachable rather than merely detected afterwards
        // by `validate_report`. Without a schema it is `GreedySampler` itself.
        // See `crate::constrain` for the grammar's scope and rulings. The mask
        // lives inside the decoder and never touches `logits`; the loop below
        // goes through `Decoder::step`, which records from the raw logits.
        // A resumed description is history for the penalty and walked by the
        // grammar, exactly as if this loop had sampled it.
        let history: Vec<u32> = full.iter().chain(&generated).copied().collect();
        let mut sampler = crate::constrain::Decoder::new(
            spec.grammar.as_ref(),
            logits.device(),
            self.model.vocab_size(),
            &history,
            self.repeat_penalty,
        )?;
        sampler.advance(&generated)?;
        // Allocated only when asked for: the distribution computation below
        // costs nothing on the request path that doesn't request it.
        let mut distributions = request
            .distributions
            .as_ref()
            .map(|_| Vec::with_capacity(request.max_tokens));
        let mut finish_reason = "length";
        for i in generated.len()..request.max_tokens {
            check()?;
            // One call selects AND records, so the record cannot be moved
            // to the wrong side of the grammar mask: see `Decoder::step`.
            let (token, step) = sampler.step(&logits, request.distributions.as_ref(), |id| {
                self.tokenizer.id_to_token(id)
            })?;
            if self.tokenizer.id_to_token(token).is_none() {
                return Err(Failure::Internal(format!(
                    "model selected unused vocabulary row {token}"
                )));
            }
            if let Some(step) = step {
                distributions
                    .as_mut()
                    .expect("allocated above whenever distributions is requested")
                    .push(step);
            }
            generated.push(token);
            if token == self.eos {
                finish_reason = "stop";
                break;
            }
            if i + 1 < request.max_tokens {
                logits = self.model.forward(&[token], &mut state)?;
            }
        }
        check()?;
        // Nothing is stripped but the terminating im_end: tool delimiters, and
        // anything else the model wrote, reach `output` as generated.
        let content = if generated.last() == Some(&self.eos) {
            &generated[..generated.len() - 1]
        } else {
            &generated
        };
        let output = self
            .tokenizer
            .decode(content, false)
            .map_err(|e| Failure::Internal(e.to_string()))?;
        let (report, report_error) = match &spec.prompt.output_schema {
            None => (None, None),
            Some(schema) => match validate_report(&output, schema, finish_reason) {
                Ok(report) => (Some(report), None),
                Err(error) => (None, Some(error)),
            },
        };
        Ok(AdjudicateResponse {
            prefix: spec.info.clone(),
            output,
            report,
            report_error,
            finish_reason: finish_reason.into(),
            prompt_tokens: full.len(),
            cached_tokens,
            completion_tokens: generated.len(),
            queue_ms: 0.,
            prefill_ms,
            decode_ms: decode.elapsed().as_secs_f64() * 1000.,
            distributions,
            opinion: None,
            resumed_tokens,
        })
    }

    fn opine(
        &mut self,
        request: &crate::opinion_api::OpinionRequest,
        question: &crate::opinion_api::ResolvedQuestion,
        check: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<crate::opinion_api::OpinionResponse, Failure> {
        request.validate().map_err(Failure::BadRequest)?;
        self.describe_then_read(request, question, check)
    }
}

impl Adjudicator {
    /// The F8 read primitive: the spec's fixed `opinion` question at its
    /// prefill, on `/v1/adjudicate` with `opinion: true`.
    fn opinion(
        &mut self,
        request: &AdjudicateRequest,
        check: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<AdjudicateResponse, Failure> {
        let spec_slot = &mut self.specs[0];
        let spec = spec_slot.prompt.opinion.clone().ok_or_else(|| {
            Failure::BadRequest("this adjudicator's prompt spec asks no opinion question".into())
        })?;
        check()?;
        let rendered = format!(
            "{}{}",
            spec_slot.prefix_text,
            spec_slot
                .prompt
                .render_user_turn_with_prefill(&request.input, &spec.prefill)
                .map_err(Failure::BadRequest)?
        );
        // Load checked this split on a probe input; failing here means the
        // input changed the tokenization across the assistant boundary, which
        // the added tokens are supposed to make impossible.
        let crate::opinion::SplitOptions {
            shared: shared_ids,
            continuations,
        } = crate::opinion::split_options(&self.tokenizer, &rendered, &spec)
            .map_err(Failure::Internal)?;
        let shared = shared_ids.len();
        let longest = shared + continuations.iter().map(Vec::len).max().unwrap_or(0);
        if longest > spec_slot.info.context_limit {
            return Err(Failure::BadRequest(
                "prompt plus options exceeds adjudicator context".into(),
            ));
        }
        let begin = Instant::now();
        let PreparedEvaluation {
            state,
            logits,
            cached_tokens,
        } = spec_slot
            .cache
            .prepare(&self.model, &shared_ids, request.use_cache, check)?;
        let prefill_ms = begin.elapsed().as_secs_f64() * 1000.;
        let score = Instant::now();
        let candle_check = || check().map_err(|_| candle_core::Error::Msg("cancelled".into()));
        let scores = crate::opinion::score_continuations(
            &self.model,
            &state,
            &logits,
            &continuations,
            &candle_check,
        )
        .map_err(|e| {
            // Recover the precise failure: a cancellation surfaced through candle.
            check().err().unwrap_or(Failure::Internal(e.to_string()))
        })?;
        logits.device().synchronize()?;
        let decode_ms = score.elapsed().as_secs_f64() * 1000.;
        let read = read_options(&spec.options, &scores, &continuations, &logits)?;
        Ok(AdjudicateResponse {
            prefix: spec_slot.info.clone(),
            output: String::new(),
            report: None,
            report_error: None,
            finish_reason: "opinion".into(),
            prompt_tokens: shared,
            cached_tokens,
            completion_tokens: 0,
            queue_ms: 0.,
            prefill_ms,
            decode_ms,
            distributions: None,
            opinion: Some(crate::opinion::OpinionRead {
                options: read.options,
                sequence_mass: read.sequence_mass,
                first_token_mass: read.first_token_mass,
                shared_tokens: shared,
                scored_tokens: continuations.iter().map(Vec::len).sum(),
                rendered_sha256: sha256_hex_bytes(rendered.as_bytes()),
            }),
            resumed_tokens: None,
        })
    }

    /// `/v1/opinion`: generate the fields before the question under the
    /// grammar, stop at the question's value slot, score every option there.
    /// See [`crate::opinion_api`] for why the description comes first and
    /// what the caches are.
    fn describe_then_read(
        &mut self,
        request: &crate::opinion_api::OpinionRequest,
        question: &crate::opinion_api::ResolvedQuestion,
        check: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<crate::opinion_api::OpinionResponse, Failure> {
        use crate::opinion_api::{Answer, CacheOutcome, OpinionResponse};
        let index = self
            .specs
            .iter()
            .position(|s| s.name == request.spec)
            .ok_or_else(|| Failure::BadRequest(format!("no loaded spec {:?}", request.spec)))?;
        let spec = &mut self.specs[index];
        let grammar = spec.grammar.clone().ok_or_else(|| {
            Failure::BadRequest(format!(
                "spec {:?} has no output schema, so there is nothing to describe or ask",
                request.spec
            ))
        })?;
        check()?;
        let prompt_text = format!(
            "{}{}",
            spec.prefix_text,
            spec.prompt.render_user_turn(&request.state.render())
        );
        let prompt_ids = self
            .tokenizer
            .encode(prompt_text.as_str(), false)
            .map_err(|e| Failure::Internal(e.to_string()))?
            .get_ids()
            .to_vec();
        let context_limit = spec.info.context_limit;
        if prompt_ids.len() + 2 >= context_limit {
            return Err(Failure::BadRequest(
                "prompt leaves no adjudicator context to describe the command".into(),
            ));
        }
        let slot_text = question.slot_text();
        let mut cache = CacheOutcome {
            prefix: if request.use_cache { "hit" } else { "bypass" }.into(),
            state: "miss".into(),
            described: "miss".into(),
        };
        let described_hit = if request.use_cache {
            spec.described.get(&prompt_ids, &question.field)
        } else {
            cache.described = "bypass".into();
            cache.state = "bypass".into();
            None
        };
        let (state, logits, generated, text, cached_tokens, prefill_ms, describe_ms) =
            match described_hit {
                Some((generated, text, state, logits)) => {
                    cache.described = "hit".into();
                    cache.state = "skipped".into();
                    (state, logits, generated, text, prompt_ids.len(), 0., 0.)
                }
                None => {
                    let begin = Instant::now();
                    let PreparedEvaluation {
                        mut state,
                        mut logits,
                        cached_tokens,
                    } = spec
                        .cache
                        .prepare(&self.model, &prompt_ids, request.use_cache, check)?;
                    if request.use_cache && cached_tokens == prompt_ids.len() {
                        cache.state = "hit".into();
                    }
                    let prefill_ms = begin.elapsed().as_secs_f64() * 1000.;
                    let describe = Instant::now();
                    let mut sampler = crate::constrain::Decoder::new(
                        Some(&grammar),
                        logits.device(),
                        self.model.vocab_size(),
                        &prompt_ids,
                        self.repeat_penalty,
                    )?;
                    let mut generated = Vec::new();
                    let mut text = String::new();
                    let mut at_slot = false;
                    for _ in prompt_ids.len()..context_limit {
                        check()?;
                        let token = sampler.sample(&logits)?;
                        if self.tokenizer.id_to_token(token).is_none() {
                            return Err(Failure::Internal(format!(
                                "model selected unused vocabulary row {token}"
                            )));
                        }
                        if token == self.eos {
                            break;
                        }
                        generated.push(token);
                        let before = text.len();
                        text = self
                            .tokenizer
                            .decode(&generated, false)
                            .map_err(|e| Failure::Internal(e.to_string()))?;
                        if text.ends_with(&slot_text) {
                            logits = self.model.forward(&[token], &mut state)?;
                            at_slot = true;
                            break;
                        }
                        // The grammar admits any token whose bytes begin the
                        // value, so one token can write the slot's closing
                        // quote AND the first bytes of an answer (`"a`) — the
                        // model leaving the canonical path at the slot. There
                        // is no slot to stand at then, and a read off this
                        // state would score the options on a path the model
                        // did not take. Refuse with the right diagnosis; the
                        // harness counts these as their own outcome.
                        if let Some(at) = text.find(&slot_text)
                            && at + slot_text.len() > before
                            && at + slot_text.len() < text.len()
                        {
                            return Err(Failure::Internal(format!(
                                "model wrote past the {:?} slot in one token ({:?}); no slot to read",
                                question.field,
                                &text[before..]
                            )));
                        }
                        logits = self.model.forward(&[token], &mut state)?;
                    }
                    if !at_slot {
                        // The grammar makes every field reachable and required,
                        // so this is the context running out or the model
                        // ending the turn where the grammar forbids it — loud,
                        // never a read off the wrong slot.
                        return Err(Failure::Internal(format!(
                            "description ended before the {:?} slot after {} tokens: {text:?}",
                            question.field,
                            generated.len()
                        )));
                    }
                    logits.device().synchronize()?;
                    check()?;
                    let describe_ms = describe.elapsed().as_secs_f64() * 1000.;
                    if request.use_cache {
                        spec.described.insert(DescribedEntry {
                            prompt_ids: prompt_ids.clone(),
                            field: question.field.clone(),
                            generated: generated.clone(),
                            text: text.clone(),
                            state: state.clone(),
                            logits: logits.clone(),
                        });
                    }
                    (state, logits, generated, text, cached_tokens, prefill_ms, describe_ms)
                }
            };
        let described = parse_described(&text, &slot_text, &question.describe)?;
        let read = Instant::now();
        // Each option and its close on their own pretokens after the slot:
        // checked at load on a probe, held to here.
        let mut continuations = Vec::with_capacity(question.options.len());
        for option in &question.options {
            let alone = encode_ids(&self.tokenizer, &format!("{option}{}", question.close))
                .map_err(Failure::Internal)?;
            let whole = encode_ids(
                &self.tokenizer,
                &format!("{prompt_text}{text}{option}{}", question.close),
            )
            .map_err(Failure::Internal)?;
            if alone.is_empty() || !whole.ends_with(&alone) {
                return Err(Failure::Internal(format!(
                    "option {option:?} does not tokenize on its own after the slot"
                )));
            }
            continuations.push(alone);
        }
        let candle_check = || check().map_err(|_| candle_core::Error::Msg("cancelled".into()));
        let scores = crate::opinion::score_continuations(
            &self.model,
            &state,
            &logits,
            &continuations,
            &candle_check,
        )
        .map_err(|e| check().err().unwrap_or(Failure::Internal(e.to_string())))?;
        logits.device().synchronize()?;
        let read_ms = read.elapsed().as_secs_f64() * 1000.;
        let read = read_options(&question.options, &scores, &continuations, &logits)?;
        let margin = crate::opinion_api::margin(
            &read.options.iter().map(|o| o.prob).collect::<Vec<_>>(),
        );
        let rendered = format!("{prompt_text}{text}");
        Ok(OpinionResponse {
            prefix: spec.info.clone(),
            spec: request.spec.clone(),
            described,
            answers: vec![Answer {
                field: question.field.clone(),
                read: crate::opinion::OpinionRead {
                    options: read.options,
                    sequence_mass: read.sequence_mass,
                    first_token_mass: read.first_token_mass,
                    shared_tokens: prompt_ids.len() + generated.len(),
                    scored_tokens: continuations.iter().map(Vec::len).sum(),
                    rendered_sha256: sha256_hex_bytes(rendered.as_bytes()),
                },
                margin,
            }],
            rendered: request.rendered.then_some(rendered),
            cache,
            prompt_tokens: prompt_ids.len(),
            cached_tokens,
            described_tokens: generated.len(),
            queue_ms: 0.,
            prefill_ms,
            describe_ms,
            read_ms,
        })
    }
}

/// The scored options and the two masses, from raw sequence scores and the
/// slot's next-token logits.
struct ReadOptions {
    options: Vec<crate::opinion::OptionScore>,
    sequence_mass: f32,
    first_token_mass: f32,
}
fn read_options(
    names: &[String],
    scores: &[f32],
    continuations: &[Vec<u32>],
    logits: &Tensor,
) -> Result<ReadOptions, Failure> {
    let mass = crate::opinion::logsumexp(scores);
    let first = candle_nn::ops::log_softmax(logits, candle_core::D::Minus1)?
        .to_vec2::<f32>()?
        .remove(0);
    let first_ids: std::collections::BTreeSet<u32> =
        continuations.iter().map(|c| c[0]).collect();
    let first_token_mass = crate::opinion::logsumexp(
        &first_ids.iter().map(|&t| first[t as usize]).collect::<Vec<_>>(),
    );
    let options = names
        .iter()
        .zip(scores)
        .zip(continuations)
        .map(|((option, &logprob), tokens)| crate::opinion::OptionScore {
            option: option.clone(),
            logprob,
            first_logprob: first[tokens[0] as usize],
            prob: (logprob - mass).exp(),
            tokens: tokens.clone(),
        })
        .collect();
    Ok(ReadOptions {
        options,
        sequence_mass: mass,
        first_token_mass,
    })
}

/// The fields written before the slot, parsed from the generated text.
/// `text` ends with `slot_text` (the caller stopped there); what precedes it
/// is either the object's opening (first field asked) or complete fields and
/// the grammar's separator.
pub fn parse_described(
    text: &str,
    slot_text: &str,
    fields: &[String],
) -> Result<Vec<crate::opinion_api::DescribedField>, Failure> {
    let body = text.strip_suffix(slot_text).ok_or_else(|| {
        Failure::Internal(format!("generated text does not end at the slot: {text:?}"))
    })?;
    if fields.is_empty() {
        if body == "{" {
            return Ok(vec![]);
        }
        return Err(Failure::Internal(format!(
            "no field precedes the slot, but the model wrote {body:?} before it"
        )));
    }
    let body = body.strip_suffix(", ").ok_or_else(|| {
        Failure::Internal(format!("described fields do not end at a separator: {body:?}"))
    })?;
    let object: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&format!("{body}}}"))
            .map_err(|e| Failure::Internal(format!("described fields are not JSON: {e}: {body:?}")))?;
    if object.len() != fields.len() {
        return Err(Failure::Internal(format!(
            "described {} fields where the spec lists {} before the slot",
            object.len(),
            fields.len()
        )));
    }
    fields
        .iter()
        .map(|field| {
            object
                .get(field)
                .cloned()
                .map(|value| crate::opinion_api::DescribedField {
                    field: field.clone(),
                    value,
                })
                .ok_or_else(|| Failure::Internal(format!("described fields lack {field:?}")))
        })
        .collect()
}

/// Supported output schema: a closed object of required string/boolean fields.
/// Refuse unsupported constraints rather than pretending to enforce JSON Schema.
/// The schema's top-level keys, in the order the system prompt states them.
///
/// The model copies the stated schema's FIRST key into the report's first key
/// slot. Measured on ROCm over three inputs, `effect`'s probability there:
/// 0.36-0.58 under `serde_json`'s sorted order, where `additionalProperties`
/// comes first and won one row of three; 0.62-0.76 with `type` first; 0.86-0.95
/// with `properties` last. So the field list is stated last, nearest the slot
/// that copies it, and `properties` is also the only key whose own order
/// matters. `~/exomemory/lfm2d/lfm25-think-prefill-2026-09-18/toplevel-*`.
const SCHEMA_KEYS: [&str; 6] = [
    "type",
    "additionalProperties",
    "title",
    "description",
    "required",
    "properties",
];

/// Serialize the output schema in [`SCHEMA_KEYS`] order, `properties` in
/// `required` order.
///
/// `serde_json::Value` is backed by a `BTreeMap`, so the default rendering
/// sorts every object's keys, and arrays keep theirs — which is why `required`
/// was the only order that ever reached `crate::constrain`. The grammar emits
/// fields in `required` order, so the sorted rendering stated one order in the
/// system prompt and then masked the model into another: measured on LFM2.5 at
/// the second key slot, the model put 0.98 on `reason` and was forced to
/// `scope`.
///
/// Call it after `validate_schema`, which is what guarantees `required` and
/// `properties` name the same set and that no key falls outside
/// [`SCHEMA_KEYS`]. A schema that slipped past it is refused rather than
/// rendered with a key silently dropped.
fn render_schema(schema: &serde_json::Value) -> Result<String, String> {
    let object = schema
        .as_object()
        .ok_or("output_schema must be an object")?;
    let order = object
        .get("required")
        .and_then(|r| r.as_array())
        .ok_or("missing required field list")?
        .iter()
        .map(|name| name.as_str().ok_or("required field names must be strings"))
        .collect::<Result<Vec<_>, _>>()?;
    let mut out = String::from("{");
    let mut written = 0;
    for key in SCHEMA_KEYS {
        let Some(value) = object.get(key) else {
            continue;
        };
        if written > 0 {
            out.push(',');
        }
        written += 1;
        out.push_str(&serde_json::to_string(key).map_err(|e| e.to_string())?);
        out.push(':');
        if key == "properties" {
            out.push_str(&render_properties(value, &order)?);
        } else {
            out.push_str(&serde_json::to_string(value).map_err(|e| e.to_string())?);
        }
    }
    if written != object.len() {
        return Err("unsupported output_schema keyword".into());
    }
    out.push('}');
    Ok(out)
}
fn render_properties(properties: &serde_json::Value, order: &[&str]) -> Result<String, String> {
    let properties = properties
        .as_object()
        .ok_or("missing schema properties")?;
    if properties.len() != order.len() {
        return Err("every property must be required exactly once".into());
    }
    let mut out = String::from("{");
    for (i, name) in order.iter().enumerate() {
        let spec = properties
            .get(*name)
            .ok_or("required names a property that does not exist")?;
        if i > 0 {
            out.push(',');
        }
        out.push_str(&serde_json::to_string(name).map_err(|e| e.to_string())?);
        out.push(':');
        out.push_str(&serde_json::to_string(spec).map_err(|e| e.to_string())?);
    }
    out.push('}');
    Ok(out)
}
fn validate_schema(schema: &serde_json::Value) -> Result<(), String> {
    let object = schema
        .as_object()
        .ok_or("output_schema must be an object")?;
    if object.keys().any(|k| {
        ![
            "type",
            "properties",
            "required",
            "additionalProperties",
            "description",
            "title",
        ]
        .contains(&k.as_str())
    }) {
        return Err("unsupported output_schema keyword".into());
    }
    if schema["type"] != "object" || schema["additionalProperties"] != false {
        return Err("output_schema must declare object and additionalProperties=false".into());
    }
    let props = schema["properties"]
        .as_object()
        .ok_or("missing schema properties")?;
    let required = schema["required"]
        .as_array()
        .ok_or("missing required field list")?;
    let names: std::collections::BTreeSet<_> = required.iter().filter_map(|v| v.as_str()).collect();
    if props.is_empty()
        || names.len() != required.len()
        || names.len() != props.len()
        || props.keys().any(|k| !names.contains(k.as_str()))
    {
        return Err("every property must be required exactly once".into());
    }
    for spec in props.values() {
        let obj = spec.as_object().ok_or("invalid property schema")?;
        if obj
            .keys()
            .any(|k| !["type", "enum", "description", "title"].contains(&k.as_str()))
        {
            return Err("unsupported property constraint".into());
        }
        let string = spec["type"] == "string";
        if !string && spec["type"] != "boolean" {
            return Err("report properties must be string or boolean".into());
        }
        if let Some(choices) = spec.get("enum") {
            let choices = choices.as_array().ok_or("enum must be an array")?;
            if choices.is_empty()
                || choices.iter().any(|v| {
                    if string {
                        !v.is_string()
                    } else {
                        !v.is_boolean()
                    }
                })
            {
                return Err("invalid enum values".into());
            }
        }
    }
    Ok(())
}

// serde_json::Value normally keeps the last duplicate key. Reports are flat,
// so reject duplicate top-level fields before any schema/semantic validation.
struct ReportObject(serde_json::Map<String, serde_json::Value>);
impl<'de> Deserialize<'de> for ReportObject {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = ReportObject;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("one report object without duplicate fields")
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> Result<Self::Value, A::Error> {
                let mut out = serde_json::Map::new();
                while let Some((key, value)) = map.next_entry::<String, serde_json::Value>()? {
                    if out.insert(key, value).is_some() {
                        return Err(serde::de::Error::custom("duplicate report field"));
                    }
                }
                Ok(ReportObject(out))
            }
        }
        de.deserialize_map(Visitor)
    }
}
/// Validate a completed report. The whole completion is the document: reasoning
/// regions, fences, trailing prose, duplicate fields, truncated output, and
/// coercions are never accepted.
///
/// This used to strip a leading `<think>...</think>`. That branch was
/// unreachable by construction: this function only runs under an
/// `output_schema`, which is the same expression that compiles the grammar, and
/// the grammar's first legal byte is the object's `{`. (`<think>` is also an
/// added token, which `constrain::Vocabulary` masks unconditionally — but that
/// is the weaker argument, since it would still be admissible inside a string
/// value.) The reasoning region
/// the model expects is supplied already closed by the prompt instead
/// ([`Reasoning`]), so a completion that still carried one would be something
/// gone wrong, and is now reported rather than quietly stripped.
pub fn validate_report(
    output: &str,
    schema: &serde_json::Value,
    finish: &str,
) -> Result<serde_json::Value, String> {
    validate_schema(schema)?;
    if finish != "stop" {
        return Err("generation was truncated; no report returned".into());
    }
    let ReportObject(report) =
        serde_json::from_str(output.trim()).map_err(|e| format!("invalid report JSON: {e}"))?;
    let properties = schema["properties"]
        .as_object()
        .ok_or("invalid report schema")?;
    if report.len() != properties.len() || properties.keys().any(|k| !report.contains_key(k)) {
        return Err("report has missing or additional fields".into());
    }
    for (name, spec) in properties {
        let value = &report[name];
        let valid = if spec["type"] == "string" {
            value.as_str().is_some_and(|s| !s.trim().is_empty())
        } else {
            value.is_boolean()
        };
        if !valid {
            return Err(format!(
                "report field {name} has an invalid type or empty value"
            ));
        }
        if let Some(choices) = spec.get("enum").and_then(|v| v.as_array())
            && !choices.contains(value)
        {
            return Err(format!("report field {name} is outside its enum"));
        }
    }
    Ok(serde_json::Value::Object(report))
}

/// One unit of work for the single generation worker: a generation or an
/// opinion read. Both share the queue, the deadline and the cancellation
/// path; only the generator call differs.
enum Job {
    Adjudicate(AdjudicateRequest),
    Opinion(
        crate::opinion_api::OpinionRequest,
        crate::opinion_api::ResolvedQuestion,
    ),
}
impl Job {
    fn timeout_ms(&self) -> u64 {
        match self {
            Job::Adjudicate(r) => r.timeout_ms,
            Job::Opinion(r, _) => r.timeout_ms,
        }
    }
    fn operation(&self) -> &'static str {
        match self {
            Job::Adjudicate(_) => "adjudicate",
            Job::Opinion(..) => "opinion",
        }
    }
}
enum Reply {
    Adjudicate(AdjudicateResponse),
    Opinion(crate::opinion_api::OpinionResponse),
}
struct Work {
    job: Job,
    reply: oneshot::Sender<Result<Reply, Failure>>,
    enqueued: Instant,
    span: tracing::Span,
}
#[derive(Clone)]
pub struct Handle {
    tx: mpsc::SyncSender<Work>,
    info: PrefixInfo,
    /// What `/v1/opinion` may be asked: read off the loaded specs, so a
    /// question is refused at the handler and never queued.
    menu: Arc<Vec<crate::opinion_api::SpecMenuEntry>>,
    exit: WorkerExit,
    stopping: Arc<AtomicBool>,
}
impl Handle {
    pub fn spawn<G: Generator>(mut generator: G, info: PrefixInfo) -> Self {
        let (tx, rx) = mpsc::sync_channel::<Work>(8);
        let exit = WorkerExit::default();
        let finished = exit.clone();
        let stopping = Arc::new(AtomicBool::new(false));
        let worker_stopping = stopping.clone();
        let worker = std::thread::Builder::new()
            .name("lfm2d-adjudicator".into())
            .spawn(move || {
                while let Ok(work) = rx.recv() {
                    let deadline = work.enqueued + Duration::from_millis(work.job.timeout_ms());
                    let check = || {
                        if work.reply.is_closed() || worker_stopping.load(Ordering::SeqCst) {
                            Err(Failure::Cancelled)
                        } else if Instant::now() >= deadline {
                            Err(Failure::Deadline)
                        } else {
                            Ok(())
                        }
                    };
                    let _span = work.span.enter();
                    let queue_ms = work.enqueued.elapsed().as_secs_f64() * 1000.;
                    let operation = work.job.operation();
                    let start = Instant::now();
                    let result = match &work.job {
                        Job::Adjudicate(request) => check()
                            .and_then(|()| generator.generate(request, &check))
                            .map(|mut r| {
                                r.queue_ms = queue_ms;
                                tracing::info!(
                                    cached_tokens = r.cached_tokens,
                                    prompt_tokens = r.prompt_tokens,
                                    completion_tokens = r.completion_tokens,
                                    prefill_ms = r.prefill_ms,
                                    decode_ms = r.decode_ms,
                                    "adjudication complete"
                                );
                                Reply::Adjudicate(r)
                            }),
                        Job::Opinion(request, question) => check()
                            .and_then(|()| generator.opine(request, question, &check))
                            .map(|mut r| {
                                r.queue_ms = queue_ms;
                                tracing::info!(
                                    spec = %r.spec,
                                    field = %question.field,
                                    cached_tokens = r.cached_tokens,
                                    prompt_tokens = r.prompt_tokens,
                                    described_tokens = r.described_tokens,
                                    described_cache = %r.cache.described,
                                    prefill_ms = r.prefill_ms,
                                    describe_ms = r.describe_ms,
                                    read_ms = r.read_ms,
                                    "opinion read complete"
                                );
                                Reply::Opinion(r)
                            }),
                    };
                    if let Err(e) = &result {
                        let kind = match e {
                            Failure::BadRequest(_) => "bad_request",
                            Failure::Internal(_) => "internal",
                            Failure::Cancelled => "cancelled",
                            Failure::Deadline => "deadline",
                        };
                        tracing::warn!(error_kind = kind, operation, "adjudicator work failed");
                    }
                    crate::telemetry::record_inference_duration(operation, start.elapsed());
                    let _ = work.reply.send(result);
                }
                drop(generator);
                finished.mark();
            })
            .expect("spawn adjudicator worker");
        std::thread::spawn(move || {
            if worker.join().is_err() {
                eprintln!("lfm2d: adjudicator worker panicked");
                std::process::exit(1);
            }
        });
        Self {
            tx,
            info,
            menu: Arc::new(Vec::new()),
            exit,
            stopping,
        }
    }
    /// The specs `/v1/opinion` serves. Without this, every opinion request
    /// is refused as naming an unknown spec.
    pub fn with_menu(mut self, menu: Vec<crate::opinion_api::SpecMenuEntry>) -> Self {
        self.menu = Arc::new(menu);
        self
    }
    pub fn menu(&self) -> Vec<crate::opinion_api::SpecMenuEntry> {
        self.menu.as_ref().clone()
    }
    pub fn stop_signal(&self) -> Arc<AtomicBool> {
        self.stopping.clone()
    }
    pub fn exit_signal(&self) -> WorkerExit {
        self.exit.clone()
    }
    // The Err is an axum `Response`, the shape a handler hands back; boxing it
    // to please the size heuristic would cost a heap allocation per refusal.
    #[allow(clippy::result_large_err)]
    async fn submit(&self, job: Job, span: &'static str) -> Result<Reply, Response> {
        if self.stopping.load(Ordering::SeqCst) {
            return Err(Failure::Cancelled.into_response());
        }
        let (reply, rx) = oneshot::channel();
        let timeout = Duration::from_millis(job.timeout_ms());
        let work = Work {
            job,
            reply,
            enqueued: Instant::now(),
            span: tracing::info_span!("adjudicator", operation = span),
        };
        self.tx.try_send(work).map_err(|e|match e {
            mpsc::TrySendError::Full(_)=>(StatusCode::SERVICE_UNAVAILABLE,Json(serde_json::json!({"error":{"type":"overloaded","message":"adjudicator queue is full"}}))).into_response(),
            mpsc::TrySendError::Disconnected(_)=>Failure::Internal("adjudicator worker unavailable".into()).into_response(),
        })?;
        match tokio::time::timeout(timeout, rx).await {
            Err(_) => Err(Failure::Deadline.into_response()),
            Ok(Err(_)) => {
                Err(Failure::Internal("adjudicator worker dropped response".into()).into_response())
            }
            Ok(Ok(result)) => result.map_err(IntoResponse::into_response),
        }
    }
    #[allow(clippy::result_large_err)]
    pub async fn evaluate(
        &self,
        request: AdjudicateRequest,
    ) -> Result<AdjudicateResponse, Response> {
        request
            .validate()
            .map_err(|e| Failure::BadRequest(e).into_response())?;
        match self.submit(Job::Adjudicate(request), "adjudicate").await? {
            Reply::Adjudicate(r) => Ok(r),
            Reply::Opinion(_) => {
                Err(Failure::Internal("worker answered a generation with an opinion".into())
                    .into_response())
            }
        }
    }
    /// Validate against the menu here, so a question the daemon cannot ask
    /// is a 400 that never occupies the worker.
    #[allow(clippy::result_large_err)]
    pub async fn opine(
        &self,
        request: crate::opinion_api::OpinionRequest,
    ) -> Result<crate::opinion_api::OpinionResponse, Response> {
        request
            .validate()
            .map_err(|e| Failure::BadRequest(e).into_response())?;
        let entry = self
            .menu
            .iter()
            .find(|e| e.spec == request.spec)
            .ok_or_else(|| {
                Failure::BadRequest(format!(
                    "no loaded spec {:?}; GET /v1/opinion/specs lists them",
                    request.spec
                ))
                .into_response()
            })?;
        let question = entry
            .resolve(&request.questions[0])
            .map_err(|e| Failure::BadRequest(e).into_response())?;
        match self.submit(Job::Opinion(request, question), "opinion").await? {
            Reply::Opinion(r) => Ok(r),
            Reply::Adjudicate(_) => {
                Err(Failure::Internal("worker answered an opinion with a generation".into())
                    .into_response())
            }
        }
    }
}
pub fn router(handle: Handle) -> Router {
    Router::new()
        .route("/v1/adjudicator", get(info))
        .route("/v1/adjudicate", post(adjudicate))
        .route("/v1/opinion", post(opine))
        .route("/v1/opinion/specs", get(specs))
        .with_state(handle)
        .layer(axum::middleware::from_fn(
            crate::server::telemetry_middleware,
        ))
}
async fn info(State(h): State<Handle>) -> Json<PrefixInfo> {
    Json(h.info)
}
async fn specs(State(h): State<Handle>) -> Json<Vec<crate::opinion_api::SpecMenuEntry>> {
    Json(h.menu())
}
#[allow(clippy::result_large_err)] // an axum handler's Err is a Response by design
async fn adjudicate(
    State(h): State<Handle>,
    crate::server::ValidJson(request): crate::server::ValidJson<AdjudicateRequest>,
) -> Result<Json<AdjudicateResponse>, Response> {
    h.evaluate(request).await.map(Json)
}
#[allow(clippy::result_large_err)]
async fn opine(
    State(h): State<Handle>,
    crate::server::ValidJson(request): crate::server::ValidJson<crate::opinion_api::OpinionRequest>,
) -> Result<Json<crate::opinion_api::OpinionResponse>, Response> {
    h.opine(request).await.map(Json)
}

#[cfg(test)]
mod prompt_cache_tests {
    use super::*;
    use std::cell::Cell;
    fn fixture() -> (Model, PromptCache) {
        let mut f = std::io::Cursor::new(include_bytes!("../tests/fixtures/lfm2-moe/tiny.gguf"));
        let ct = candle_core::quantized::gguf_file::Content::read(&mut f).unwrap();
        let model = Model::from_gguf(ct, &mut f, &candle_core::Device::Cpu).unwrap();
        let mut prefix = model.new_state();
        model.forward(&[1, 2, 3], &mut prefix).unwrap();
        (
            model,
            PromptCache {
                prefix,
                prefix_ids: vec![1, 2, 3],
                ready: None,
            },
        )
    }
    fn values(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1().unwrap()
    }
    #[test]
    fn prepared_prompt_keeps_logits_and_hybrid_state_isolated() {
        let (model, mut cache) = fixture();
        let tokens = [1, 2, 3, 4, 5];
        let mut first = cache.prepare(&model, &tokens, true, &|| Ok(())).unwrap();
        assert_eq!(first.cached_tokens, 3);
        let expected = values(&first.logits);
        let expected_next = values(&model.forward(&[6], &mut first.state).unwrap());
        model.forward(&[7, 8], &mut first.state).unwrap();
        let mut second = cache.prepare(&model, &tokens, true, &|| Ok(())).unwrap();
        assert_eq!(second.cached_tokens, tokens.len());
        assert_eq!(second.state.len(), tokens.len());
        assert_eq!(values(&second.logits), expected);
        assert_eq!(
            values(&model.forward(&[6], &mut second.state).unwrap()),
            expected_next
        );
        assert_eq!(cache.prefix.len(), 3);
    }
    #[test]
    fn ready_input_cannot_cross_model_instances() {
        let (model, mut cache) = fixture();
        let (other, _) = fixture();
        cache
            .prepare(&model, &[1, 2, 3, 4], true, &|| Ok(()))
            .unwrap();
        assert!(
            cache
                .prepare(&other, &[1, 2, 3, 4], true, &|| Ok(()))
                .is_err()
        );
    }

    #[test]
    fn cache_is_exact_bounded_and_cold_requests_bypass_it() {
        let (model, mut cache) = fixture();
        let a = [1, 2, 3, 4];
        let b = [1, 2, 3, 5];
        cache.prepare(&model, &a, true, &|| Ok(())).unwrap();
        assert_eq!(
            cache
                .prepare(&model, &b, false, &|| Ok(()))
                .unwrap()
                .cached_tokens,
            0
        );
        assert_eq!(
            cache
                .prepare(&model, &a, true, &|| Ok(()))
                .unwrap()
                .cached_tokens,
            a.len()
        );
        assert_eq!(
            cache
                .prepare(&model, &b, true, &|| Ok(()))
                .unwrap()
                .cached_tokens,
            3
        );
        assert_eq!(
            cache
                .prepare(&model, &a, true, &|| Ok(()))
                .unwrap()
                .cached_tokens,
            3
        );
    }
    #[test]
    fn step_distribution_over_real_model_logits_is_deterministic_and_internally_consistent() {
        // Exercises crate::types::step_distribution against an ACTUAL Tensor
        // -> Vec<f32> extraction from the real (tiny) model, not just
        // hand-built float arrays — the same `flatten_all().to_vec1()` path
        // the decode-loop hunk in `generate` uses. Two independent forward
        // passes from a fresh state on identical tokens must agree exactly:
        // this stack has been bit-deterministic elsewhere (see
        // `prepared_prompt_keeps_logits_and_hybrid_state_isolated` above),
        // and if it were NOT deterministic here, that would be a finding to
        // report, not paper over.
        let (model, _cache) = fixture();
        let mut state_a = model.new_state();
        let raw_a = values(&model.forward(&[1, 2, 3, 4], &mut state_a).unwrap());
        let mut state_b = model.new_state();
        let raw_b = values(&model.forward(&[1, 2, 3, 4], &mut state_b).unwrap());
        assert_eq!(
            raw_a, raw_b,
            "identical prompt tokens from a fresh state must produce bit-identical logits"
        );

        let text = |id: u32| Some(format!("<{id}>"));
        let mut sets = std::collections::BTreeMap::new();
        sets.insert("probe".to_string(), vec![0u32, 1]);
        let step_a = crate::types::step_distribution(&raw_a, 0, 5, &sets, text).unwrap();
        let step_b = crate::types::step_distribution(&raw_b, 0, 5, &sets, text).unwrap();
        assert_eq!(step_a.logprob, step_b.logprob);
        assert_eq!(
            step_a
                .top_logprobs
                .iter()
                .map(|t| (t.token, t.logprob))
                .collect::<Vec<_>>(),
            step_b
                .top_logprobs
                .iter()
                .map(|t| (t.token, t.logprob))
                .collect::<Vec<_>>()
        );
        assert_eq!(step_a.set_mass["probe"].logprob, step_b.set_mass["probe"].logprob);

        assert!(step_a.logprob <= 0.0);
        for w in step_a.top_logprobs.windows(2) {
            assert!(w[0].logprob >= w[1].logprob, "{:?} not descending", step_a.top_logprobs);
        }
        let total: f32 = raw_a
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(total.is_finite(), "tiny fixture must produce finite logits");
    }

    #[test]
    fn described_cache_is_keyed_by_prompt_and_field_and_evicts_the_oldest() {
        let (model, _) = fixture();
        let entry = |prompt: &[u32], field: &str, generated: &[u32]| {
            let mut state = model.new_state();
            let logits = model.forward(prompt, &mut state).unwrap();
            DescribedEntry {
                prompt_ids: prompt.to_vec(),
                field: field.into(),
                generated: generated.to_vec(),
                text: format!("{field}:{}", generated.len()),
                state,
                logits,
            }
        };
        let mut cache = DescribedCache::new(2);
        cache.insert(entry(&[1, 2, 3], "verdict", &[4, 5]));
        cache.insert(entry(&[1, 2, 3], "scope", &[4]));
        // Same prompt, different field: a different slot, so a different entry.
        assert_eq!(cache.len(), 2);
        let (generated, text, state, _) = cache.get(&[1, 2, 3], "verdict").unwrap();
        assert_eq!(generated, [4, 5]);
        assert_eq!(text, "verdict:2");
        assert_eq!(state.len(), 3);
        assert!(cache.get(&[1, 2, 3], "undo").is_none());
        assert!(cache.get(&[1, 2, 4], "verdict").is_none());
        // The hit above made `verdict` most recent, so a third entry evicts `scope`.
        cache.insert(entry(&[9, 9, 9], "verdict", &[1]));
        assert_eq!(cache.len(), 2);
        assert!(cache.get(&[1, 2, 3], "scope").is_none());
        assert!(cache.get(&[1, 2, 3], "verdict").is_some());
        // Re-inserting a key replaces rather than duplicates.
        cache.insert(entry(&[9, 9, 9], "verdict", &[1, 2, 3]));
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get(&[9, 9, 9], "verdict").unwrap().0, [1, 2, 3]);
        let mut none = DescribedCache::new(0);
        none.insert(entry(&[1], "verdict", &[2]));
        assert_eq!(none.len(), 0);
    }

    /// The resume path rebuilds the sampler with `prompt + resumed` as
    /// history instead of sampling the resumed tokens. Candle's history is a
    /// seen-set that every selection marks, so the two must agree; this pins
    /// it with logits where the penalty decides the pick.
    #[test]
    fn a_sampler_built_from_resumed_history_selects_as_the_incremental_one_would() {
        let device = candle_core::Device::Cpu;
        let vocab = 8usize;
        // `top` leads by less than the penalty takes off; `runner` is never
        // in any history here.
        let logits = |top: usize, runner: usize| {
            let mut v = vec![0.5f32; vocab];
            v[top] = 3.0;
            v[runner] = 2.9;
            Tensor::new(v.as_slice(), &device).unwrap().unsqueeze(0).unwrap()
        };
        let prompt = [1u32, 2];
        // Incremental: sample 5 then 6 (each becomes history).
        let mut incremental =
            crate::constrain::Decoder::new(None, &device, vocab, &prompt, 1.05).unwrap();
        let mut resumed_tokens = Vec::new();
        for top in [5usize, 6] {
            let t = incremental.sample(&logits(top, 7)).unwrap();
            assert_eq!(t as usize, top);
            resumed_tokens.push(t);
        }
        // Rebuilt: the same tokens as constructor history, walked by `advance`.
        let history: Vec<u32> = prompt.iter().chain(&resumed_tokens).copied().collect();
        let mut rebuilt =
            crate::constrain::Decoder::new(None, &device, vocab, &history, 1.05).unwrap();
        rebuilt.advance(&resumed_tokens).unwrap();
        // 5 is the raw argmax but penalised in both, so 7 wins; a sampler
        // that forgot the resumed tokens would pick 5.
        let l = logits(5, 7);
        let a = incremental.sample(&l).unwrap();
        let b = rebuilt.sample(&l).unwrap();
        assert_eq!(a, b);
        let mut forgetful =
            crate::constrain::Decoder::new(None, &device, vocab, &prompt, 1.05).unwrap();
        assert_eq!(forgetful.sample(&l).unwrap(), 5);
        assert_ne!(a, 5, "the penalty must have moved the pick off the resumed token");
    }

    #[test]
    fn get_deepest_returns_the_longest_description_for_the_prompt() {
        let (model, _) = fixture();
        let entry = |field: &str, generated: &[u32]| {
            let mut state = model.new_state();
            let logits = model.forward(&[1, 2, 3], &mut state).unwrap();
            DescribedEntry {
                prompt_ids: vec![1, 2, 3],
                field: field.into(),
                generated: generated.to_vec(),
                text: String::new(),
                state,
                logits,
            }
        };
        let mut cache = DescribedCache::new(4);
        cache.insert(entry("scope", &[4]));
        cache.insert(entry("verdict", &[4, 5, 6]));
        cache.insert(entry("undo", &[4, 5]));
        assert_eq!(cache.get_deepest(&[1, 2, 3]).unwrap().0, [4, 5, 6]);
        assert!(cache.get_deepest(&[1, 2, 4]).is_none());
    }

    #[test]
    fn a_cached_description_reads_the_same_state_it_was_generated_from() {
        // The cache hands back a clone of the state at the slot; scoring off it
        // must equal scoring off the state that produced it, and must not
        // advance what the cache holds.
        let (model, _) = fixture();
        let mut state = model.new_state();
        model.forward(&[1, 2, 3], &mut state).unwrap();
        let logits = model.forward(&[4, 5], &mut state).unwrap();
        let direct = crate::opinion::score_continuations(&model, &state, &logits, &[vec![6, 7], vec![8]], &|| Ok(())).unwrap();
        let mut cache = DescribedCache::new(1);
        cache.insert(DescribedEntry {
            prompt_ids: vec![1, 2, 3],
            field: "verdict".into(),
            generated: vec![4, 5],
            text: String::new(),
            state,
            logits,
        });
        for _ in 0..2 {
            let (_, _, hit_state, hit_logits) = cache.get(&[1, 2, 3], "verdict").unwrap();
            let cached = crate::opinion::score_continuations(&model, &hit_state, &hit_logits, &[vec![6, 7], vec![8]], &|| Ok(())).unwrap();
            assert_eq!(direct, cached);
            assert_eq!(hit_state.len(), 5);
        }
    }

    #[test]
    fn parse_described_reads_the_fields_before_the_slot_in_order() {
        let fields = |names: &[&str]| names.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let out = parse_described(
            "{\"effect\": \"Removes it.\", \"scope\": \"project\", \"undo\": \"easy\", \"verdict\": \"",
            "\"verdict\": \"",
            &fields(&["effect", "scope", "undo"]),
        )
        .unwrap();
        let got: Vec<(String, serde_json::Value)> = out.into_iter().map(|d| (d.field, d.value)).collect();
        assert_eq!(
            got,
            [
                ("effect".to_string(), serde_json::json!("Removes it.")),
                ("scope".to_string(), serde_json::json!("project")),
                ("undo".to_string(), serde_json::json!("easy")),
            ]
        );
        // The first field asked: nothing described.
        assert!(parse_described("{\"verdict\": \"", "\"verdict\": \"", &[]).unwrap().is_empty());
        // Booleans arrive unquoted and stay booleans.
        let out = parse_described("{\"writes\": true, \"verdict\": \"", "\"verdict\": \"", &fields(&["writes"])).unwrap();
        assert_eq!(out[0].value, serde_json::json!(true));
        // Loud on every mismatch between the spec's field list and the text.
        assert!(parse_described("{\"effect\": \"x\", \"verdict\": \"", "\"verdict\": \"", &[]).is_err());
        assert!(parse_described("{\"effect\": \"x\", \"verdict\": \"", "\"verdict\": \"", &fields(&["effect", "scope"])).is_err());
        assert!(parse_described("{\"effect\": \"x\", \"verdict\": \"", "\"verdict\": \"", &fields(&["scope"])).is_err());
        assert!(parse_described("{\"effect\": \"x\"\"verdict\": \"", "\"verdict\": \"", &fields(&["effect"])).is_err());
        assert!(parse_described("{\"effect\": \"x\", ", "\"verdict\": \"", &fields(&["effect"])).is_err());
    }

    #[test]
    fn failed_or_cancelled_prefill_does_not_replace_ready_input() {
        let (model, mut cache) = fixture();
        let a = [1, 2, 3, 4];
        cache.prepare(&model, &a, true, &|| Ok(())).unwrap();
        assert!(
            cache
                .prepare(&model, &[1, 2, 3, 16], true, &|| Ok(()))
                .is_err()
        );
        assert!(
            cache
                .prepare(&model, &[1, 2, 9, 4], true, &|| Ok(()))
                .is_err()
        );
        let calls = Cell::new(0);
        let check = || {
            calls.set(calls.get() + 1);
            if calls.get() > 2 {
                Err(Failure::Cancelled)
            } else {
                Ok(())
            }
        };
        assert!(matches!(
            cache.prepare(&model, &[1, 2, 3, 5], true, &check),
            Err(Failure::Cancelled)
        ));
        assert_eq!(
            cache
                .prepare(&model, &a, true, &|| Ok(()))
                .unwrap()
                .cached_tokens,
            a.len()
        );
        assert!(matches!(
            cache.prepare(&model, &a, true, &|| Err(Failure::Deadline)),
            Err(Failure::Deadline)
        ));
    }
}
