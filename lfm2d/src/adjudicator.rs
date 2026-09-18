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
const MAX_NEW: usize = 2048;
const MAX_INPUT_BYTES: usize = 65536;
const TEMPLATE_VERSION: &str = "lfm25-single-user-v1";

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PromptSpec {
    pub system: String,
    #[serde(default)]
    pub tools: Vec<serde_json::Value>,
    #[serde(default)]
    pub output_schema: Option<serde_json::Value>,
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
        let mut system = self.system.clone();
        if let Some(schema) = &self.output_schema {
            if !self.tools.is_empty() {
                return Err("choose output_schema or tools, not both".into());
            }
            validate_schema(schema)?;
            let schema = serde_json::to_string(schema).map_err(|e| e.to_string())?;
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
}

/// The single user turn and the opening of the assistant's, appended to
/// [`PromptSpec::render_prefix`]. One definition, so anything examining a
/// prompt renders exactly what the daemon runs.
pub fn render_user_turn(input: &str) -> String {
    format!("<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n")
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
        if use_cache {
            if let Some(ready) = &self.ready {
                if ready.token_ids == full {
                    return Ok(PreparedEvaluation {
                        state: ready.state.clone(),
                        logits: ready.logits.clone(),
                        cached_tokens: full.len(),
                    });
                }
            }
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
            ("<|im_end|>", 124900),
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

pub struct Adjudicator {
    model: Model,
    tokenizer: tokenizers::Tokenizer,
    cache: PromptCache,
    prefix_text: String,
    eos: u32,
    info: PrefixInfo,
    output_schema: Option<serde_json::Value>,
    repeat_penalty: f32,
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
        let prompt: PromptSpec =
            serde_json::from_slice(&std::fs::read(prompt_path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        let prefix_text = prompt.render_prefix()?;
        let Checkpoint {
            model,
            tokenizer,
            model_id,
            weight_hash,
            tokenizer_hash,
            weight_dtypes,
            execution,
        } = Checkpoint::load(path, tokenizer_path, cli.device, cli.device_index)?;
        if cli.adjudicator_context > model.context_length() {
            return Err("adjudicator context exceeds model context".into());
        }
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
        let mut prefix = model.new_state();
        for chunk in prefix_ids.chunks(CHUNK) {
            let _ = model
                .forward(chunk, &mut prefix)
                .map_err(|e| e.to_string())?;
        }
        // Compile/warm device selection kernels before announcing readiness.
        let _ = GreedySampler::new(
            &execution.device,
            model.vocab_size(),
            &prefix_ids,
            cli.adjudicator_repeat_penalty,
        )
        .map_err(|e| e.to_string())?;
        execution.device.synchronize().map_err(|e| e.to_string())?;
        let identity = serde_json::json!([
            TEMPLATE_VERSION,
            weight_hash,
            tokenizer_hash,
            prefix_ids,
            execution.backend.as_str(),
            "f32"
        ]);
        let snapshot_id =
            sha256_hex_bytes(&serde_json::to_vec(&identity).map_err(|e| e.to_string())?);
        let info = PrefixInfo {
            model_id,
            weight_hash,
            tokenizer_hash,
            template_version: TEMPLATE_VERSION.into(),
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
            weight_dtypes,
        };
        Ok(Self {
            model,
            tokenizer,
            cache: PromptCache {
                prefix,
                prefix_ids,
                ready: None,
            },
            prefix_text,
            eos: 124900,
            info,
            output_schema: prompt.output_schema,
            repeat_penalty: cli.adjudicator_repeat_penalty,
        })
    }
    pub fn info(&self) -> PrefixInfo {
        self.info.clone()
    }
    pub fn model_info(&self) -> ModelInfo {
        ModelInfo {
            id: self.info.model_id.clone(),
            kind: ModelKind::Adjudicator,
            weight_hash: self.info.weight_hash.clone(),
            labels: None,
            hidden_size: self.model.hidden_size(),
        }
    }
}
impl Generator for Adjudicator {
    fn generate(
        &mut self,
        request: &AdjudicateRequest,
        check: &dyn Fn() -> Result<(), Failure>,
    ) -> Result<AdjudicateResponse, Failure> {
        request.validate().map_err(Failure::BadRequest)?;
        if let Some(d) = &request.distributions {
            d.validate_vocab(self.model.vocab_size())
                .map_err(Failure::BadRequest)?;
            if d.constrained && self.output_schema.is_none() {
                return Err(Failure::BadRequest(
                    "distributions.constrained needs an output schema; this adjudicator has none"
                        .into(),
                ));
            }
        }
        check()?;
        let suffix = render_user_turn(&request.input);
        let full = self
            .tokenizer
            .encode(format!("{}{suffix}", self.prefix_text), false)
            .map_err(|e| Failure::Internal(e.to_string()))?
            .get_ids()
            .to_vec();
        if full
            .len()
            .checked_add(request.max_tokens)
            .is_none_or(|n| n > self.info.context_limit)
        {
            return Err(Failure::BadRequest(
                "prompt plus max_tokens exceeds adjudicator context".into(),
            ));
        }
        let begin = Instant::now();
        let PreparedEvaluation {
            mut state,
            mut logits,
            cached_tokens,
        } = self
            .cache
            .prepare(&self.model, &full, request.use_cache, check)?;
        let prefill_ms = begin.elapsed().as_secs_f64() * 1000.;
        let decode = Instant::now();
        // Constrained decoding. With an `output_schema` this masks the logits
        // against a compiled JSON grammar before every greedy selection, so an
        // invalid report is unreachable rather than merely detected afterwards
        // by `validate_report`. Without a schema it is `GreedySampler` itself.
        // See `crate::constrain` for the grammar's scope and rulings. The mask
        // lives inside the decoder and never touches `logits`; the loop below
        // goes through `Decoder::step`, which records from the raw logits.
        let mut sampler = crate::constrain::Decoder::new(
            self.output_schema.as_ref(),
            &self.tokenizer,
            logits.device(),
            self.model.vocab_size(),
            self.eos,
            &full,
            self.repeat_penalty,
        )?;
        let mut generated = Vec::new();
        // Allocated only when asked for: the distribution computation below
        // costs nothing on the request path that doesn't request it.
        let mut distributions = request
            .distributions
            .as_ref()
            .map(|_| Vec::with_capacity(request.max_tokens));
        let mut finish_reason = "length";
        for i in 0..request.max_tokens {
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
        // Keep reasoning/tool delimiters; remove only the terminating im_end.
        let content = if generated.last() == Some(&self.eos) {
            &generated[..generated.len() - 1]
        } else {
            &generated
        };
        let output = self
            .tokenizer
            .decode(content, false)
            .map_err(|e| Failure::Internal(e.to_string()))?;
        let (report, report_error) = match &self.output_schema {
            None => (None, None),
            Some(schema) => match validate_report(&output, schema, finish_reason) {
                Ok(report) => (Some(report), None),
                Err(error) => (None, Some(error)),
            },
        };
        Ok(AdjudicateResponse {
            prefix: self.info.clone(),
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
        })
    }
}

/// Supported output schema: a closed object of required string/boolean fields.
/// Refuse unsupported constraints rather than pretending to enforce JSON Schema.
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
/// Validate a completed report. Reasoning may precede the JSON; fences, trailing
/// prose, duplicate fields, truncated output, and coercions are never accepted.
pub fn validate_report(
    output: &str,
    schema: &serde_json::Value,
    finish: &str,
) -> Result<serde_json::Value, String> {
    validate_schema(schema)?;
    if finish != "stop" {
        return Err("generation was truncated; no report returned".into());
    }
    let body = if let Some(thinking) = output.strip_prefix("<think>") {
        thinking
            .split_once("</think>")
            .ok_or("unterminated reasoning")?
            .1
    } else {
        output
    };
    let ReportObject(report) =
        serde_json::from_str(body.trim()).map_err(|e| format!("invalid report JSON: {e}"))?;
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

struct Work {
    request: AdjudicateRequest,
    reply: oneshot::Sender<Result<AdjudicateResponse, Failure>>,
    enqueued: Instant,
    span: tracing::Span,
}
#[derive(Clone)]
pub struct Handle {
    tx: mpsc::SyncSender<Work>,
    info: PrefixInfo,
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
                    let deadline = work.enqueued + Duration::from_millis(work.request.timeout_ms);
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
                    let queue = work.enqueued.elapsed();
                    let start = Instant::now();
                    let result = check()
                        .and_then(|()| generator.generate(&work.request, &check))
                        .map(|mut r| {
                            r.queue_ms = queue.as_secs_f64() * 1000.;
                            r
                        });
                    if let Ok(r) = &result {
                        tracing::info!(
                            cached_tokens = r.cached_tokens,
                            prompt_tokens = r.prompt_tokens,
                            completion_tokens = r.completion_tokens,
                            prefill_ms = r.prefill_ms,
                            decode_ms = r.decode_ms,
                            "adjudication complete"
                        );
                    }
                    if let Err(e) = &result {
                        let kind = match e {
                            Failure::BadRequest(_) => "bad_request",
                            Failure::Internal(_) => "internal",
                            Failure::Cancelled => "cancelled",
                            Failure::Deadline => "deadline",
                        };
                        tracing::warn!(error_kind = kind, "adjudication failed");
                    }
                    crate::telemetry::record_inference_duration("adjudicate", start.elapsed());
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
            exit,
            stopping,
        }
    }
    pub fn stop_signal(&self) -> Arc<AtomicBool> {
        self.stopping.clone()
    }
    pub fn exit_signal(&self) -> WorkerExit {
        self.exit.clone()
    }
    pub async fn evaluate(
        &self,
        request: AdjudicateRequest,
    ) -> Result<AdjudicateResponse, Response> {
        request
            .validate()
            .map_err(|e| Failure::BadRequest(e).into_response())?;
        if self.stopping.load(Ordering::SeqCst) {
            return Err(Failure::Cancelled.into_response());
        }
        let (reply, rx) = oneshot::channel();
        let timeout = Duration::from_millis(request.timeout_ms);
        let work = Work {
            request,
            reply,
            enqueued: Instant::now(),
            span: tracing::info_span!("adjudicate"),
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
}
pub fn router(handle: Handle) -> Router {
    Router::new()
        .route("/v1/adjudicator", get(info))
        .route("/v1/adjudicate", post(adjudicate))
        .with_state(handle)
        .layer(axum::middleware::from_fn(
            crate::server::telemetry_middleware,
        ))
}
async fn info(State(h): State<Handle>) -> Json<PrefixInfo> {
    Json(h.info)
}
async fn adjudicate(
    State(h): State<Handle>,
    crate::server::ValidJson(request): crate::server::ValidJson<AdjudicateRequest>,
) -> Result<Json<AdjudicateResponse>, Response> {
    h.evaluate(request).await.map(Json)
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
