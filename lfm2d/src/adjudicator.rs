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

const CHUNK: usize = 128;
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
fn validate_text(s: &str) -> Result<(), String> {
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

pub struct Adjudicator {
    model: Model,
    tokenizer: tokenizers::Tokenizer,
    prefix: ModelState,
    prefix_text: String,
    prefix_ids: Vec<u32>,
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
        let execution = crate::device::ExecutionDevice::select(cli.device, cli.device_index)?;
        for reason in &execution.selection_reasons {
            eprintln!("lfm2d adjudicator device: {reason}");
        }
        let model =
            Model::from_gguf(ct, &mut file, &execution.device).map_err(|e| e.to_string())?;
        if cli.adjudicator_context > model.context_length() {
            return Err("adjudicator context exceeds model context".into());
        }
        let prefix_ids = tokenizer
            .encode(prefix_text.as_str(), false)
            .map_err(|e| e.to_string())?
            .get_ids()
            .to_vec();
        if prefix_ids.is_empty() || prefix_ids.len() + 1 >= cli.adjudicator_context {
            return Err("adjudicator prefix leaves no context for input/output".into());
        }
        let mut prefix = model.new_state();
        for chunk in prefix_ids.chunks(CHUNK) {
            let _ = model
                .forward(chunk, &mut prefix)
                .map_err(|e| e.to_string())?;
        }
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
        let model_id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or("invalid GGUF filename")?
            .to_string();
        let info = PrefixInfo {
            model_id,
            weight_hash,
            tokenizer_hash,
            template_version: TEMPLATE_VERSION.into(),
            snapshot_id,
            prefix_tokens: prefix_ids.len(),
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
            prefix,
            prefix_text,
            prefix_ids,
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
        check()?;
        let suffix = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            request.input
        );
        let full = self
            .tokenizer
            .encode(format!("{}{suffix}", self.prefix_text), false)
            .map_err(|e| Failure::Internal(e.to_string()))?
            .get_ids()
            .to_vec();
        if !full.starts_with(&self.prefix_ids) {
            return Err(Failure::BadRequest(
                "rendered input changes cached token prefix".into(),
            ));
        }
        if full
            .len()
            .checked_add(request.max_tokens)
            .is_none_or(|n| n > self.info.context_limit)
        {
            return Err(Failure::BadRequest(
                "prompt plus max_tokens exceeds adjudicator context".into(),
            ));
        }
        let (mut state, start) = if request.use_cache {
            (self.prefix.clone(), self.prefix_ids.len())
        } else {
            (self.model.new_state(), 0)
        };
        let begin = Instant::now();
        let mut logits: Option<Tensor> = None;
        for chunk in full[start..].chunks(CHUNK) {
            check()?;
            logits = Some(self.model.forward(chunk, &mut state)?);
        }
        let mut logits = logits.ok_or_else(|| Failure::Internal("no suffix tokens".into()))?;
        logits.device().synchronize()?;
        let prefill_ms = begin.elapsed().as_secs_f64() * 1000.;
        let decode = Instant::now();
        let mut generated = Vec::new();
        let mut history = full.clone();
        let mut finish_reason = "length";
        for i in 0..request.max_tokens {
            check()?;
            let values = logits.flatten_all()?.to_vec1::<f32>()?;
            let token = greedy_token(&values, &history, self.repeat_penalty)?;
            if self.tokenizer.id_to_token(token).is_none() {
                return Err(Failure::Internal(format!(
                    "model selected unused vocabulary row {token}"
                )));
            }
            generated.push(token);
            history.push(token);
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
            cached_tokens: start,
            completion_tokens: generated.len(),
            queue_ms: 0.,
            prefill_ms,
            decode_ms: decode.elapsed().as_secs_f64() * 1000.,
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

/// Transformers-style sign-aware penalty, once for each token appearing in the
/// full prompt or this branch's continuation. History never leaks across calls.
fn greedy_token(values: &[f32], history: &[u32], penalty: f32) -> Result<u32, Failure> {
    if values.is_empty() || values.iter().any(|v| !v.is_finite()) {
        return Err(Failure::Internal("empty or nonfinite model logits".into()));
    }
    let mut values = values.to_vec();
    let mut seen = std::collections::HashSet::new();
    for &token in history {
        if seen.insert(token) {
            let v = values
                .get_mut(token as usize)
                .ok_or_else(|| Failure::Internal("history token outside vocabulary".into()))?;
            *v = if *v < 0. { *v * penalty } else { *v / penalty };
        }
    }
    let mut token = 0;
    for i in 1..values.len() {
        if values[i] > values[token] {
            token = i;
        }
    }
    Ok(token as u32)
}

#[cfg(test)]
mod sampling_tests {
    use super::*;
    #[test]
    fn repetition_penalty_changes_positive_and_negative_choices_once_per_token() {
        assert_eq!(greedy_token(&[1.02, 1.0], &[0, 0], 1.05).unwrap(), 1);
        assert_eq!(greedy_token(&[-1.0, -1.02], &[0, 0], 1.05).unwrap(), 1);
        assert_eq!(greedy_token(&[1.08, 1.0], &[0, 0], 1.05).unwrap(), 0);
        assert_eq!(greedy_token(&[1.02, 1.0], &[0], 1.0).unwrap(), 0);
        assert_eq!(greedy_token(&[1.0, 1.0], &[], 1.05).unwrap(), 0);
        assert!(greedy_token(&[f32::NAN], &[], 1.05).is_err());
        assert!(greedy_token(&[1.0], &[2], 1.05).is_err());
    }
}
