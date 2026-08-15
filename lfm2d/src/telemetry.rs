//! Tracing + OTLP export wiring — metrics, traces, and logs over ONE
//! pipeline (Amy's explicit preference over bare Prometheus). Human-readable
//! stderr logging (`tracing-subscriber`'s `fmt` layer) is ALWAYS on; the
//! OTLP layers are added on top only when `OTEL_EXPORTER_OTLP_ENDPOINT` is
//! set. A missing/unreachable collector must never crash this daemon or
//! slow down inference — every exporter here is a BATCH exporter (bounded
//! queue, drops under backpressure rather than blocking), and building the
//! providers never itself contacts the collector (the gRPC channel connects
//! lazily on first export).
//!
//! Called exactly once from `main.rs`, AFTER every configured model has
//! loaded — the resource attributes below include each loaded model's
//! `weight_hash`, which isn't known any earlier. See [`init`].

use std::time::Duration;

use opentelemetry::trace::TracerProvider as _;
use opentelemetry::{KeyValue, global};
use opentelemetry_appender_tracing::layer::OpenTelemetryTracingBridge;
use opentelemetry_otlp::{LogExporter, MetricExporter, SpanExporter};
use opentelemetry_sdk::logs::SdkLoggerProvider;
use opentelemetry_sdk::metrics::{PeriodicReader, SdkMeterProvider};
use opentelemetry_sdk::trace::SdkTracerProvider;
use opentelemetry_sdk::Resource;
use tracing_subscriber::layer::SubscriberExt as _;
use tracing_subscriber::util::SubscriberInitExt as _;

use crate::types::{ModelInfo, ModelKind};

/// The env var this crate treats as "is an OTLP collector configured at
/// all" — the standard OTEL var, checked directly (rather than only
/// relying on the exporter builders' own env parsing) so we can log the one
/// loud enabled/disabled line the task requires BEFORE attempting to build
/// any exporter.
const OTLP_ENDPOINT_VAR: &str = "OTEL_EXPORTER_OTLP_ENDPOINT";

/// Holds every OTLP provider alive for the process lifetime — dropping this
/// (at the end of `main`, after `serve()` resolves) calls `shutdown()` on
/// each, which flushes the batch exporters' buffered data one last time
/// before the process exits. `None` fields mean OTLP export was disabled;
/// dropping a disabled guard is a no-op.
pub struct TelemetryGuard {
    tracer_provider: Option<SdkTracerProvider>,
    meter_provider: Option<SdkMeterProvider>,
    logger_provider: Option<SdkLoggerProvider>,
}

impl Drop for TelemetryGuard {
    fn drop(&mut self) {
        if let Some(p) = &self.tracer_provider
            && let Err(e) = p.shutdown()
        {
            eprintln!("lfm2d: telemetry: tracer provider shutdown reported an error (already-flushed data is unaffected): {e}");
        }
        if let Some(p) = &self.meter_provider
            && let Err(e) = p.shutdown()
        {
            eprintln!("lfm2d: telemetry: meter provider shutdown reported an error (already-flushed data is unaffected): {e}");
        }
        if let Some(p) = &self.logger_provider
            && let Err(e) = p.shutdown()
        {
            eprintln!("lfm2d: telemetry: logger provider shutdown reported an error (already-flushed data is unaffected): {e}");
        }
    }
}

fn model_kind_attr(kind: ModelKind) -> &'static str {
    match kind {
        ModelKind::Embedder => "embedder",
        ModelKind::Classifier => "classifier",
        ModelKind::Router => "router",
        ModelKind::TokenClassifier => "token_classifier",
    }
}

fn resource(service_name: &str, models: &[ModelInfo]) -> Resource {
    let mut builder = Resource::builder()
        .with_service_name(service_name.to_string())
        .with_attribute(KeyValue::new("service.version", env!("CARGO_PKG_VERSION")));
    for model in models {
        builder = builder.with_attribute(KeyValue::new(
            format!("lfm2d.model.{}_hash", model_kind_attr(model.kind)),
            model.weight_hash.clone(),
        ));
    }
    builder.build()
}

/// Build the fmt (always-on stderr) layer plus, if `OTEL_EXPORTER_OTLP_ENDPOINT`
/// is set, the OTLP trace/log layers and the global meter provider; install
/// the combined subscriber as the process-wide default (`tracing_subscriber`
/// only allows this once, which is why this whole crate goes through
/// [`tracing`] macros rather than any bare `eprintln!` from this point on).
///
/// `service_name` should already reflect `OTEL_SERVICE_NAME` (default
/// `"lfm2d"`) — resolved by the caller so this function stays a pure
/// "given a name and the loaded models, wire everything up" step.
pub fn init(service_name: &str, models: &[ModelInfo]) -> TelemetryGuard {
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let fmt_layer = tracing_subscriber::fmt::layer().with_target(true);

    let endpoint = std::env::var(OTLP_ENDPOINT_VAR).ok().filter(|s| !s.is_empty());

    let Some(endpoint) = endpoint else {
        tracing_subscriber::registry().with(env_filter).with(fmt_layer).init();
        tracing::info!(
            "telemetry: OTLP export disabled ({OTLP_ENDPOINT_VAR} unset) — logging to stderr only"
        );
        return TelemetryGuard { tracer_provider: None, meter_provider: None, logger_provider: None };
    };

    let resource = resource(service_name, models);

    // Each exporter reads OTEL_EXPORTER_OTLP_ENDPOINT (and its
    // signal-specific overrides, e.g. OTEL_EXPORTER_OTLP_TRACES_ENDPOINT)
    // itself via `.with_tonic().build()` — the standard env config the task
    // asks for comes for free from the exporter builders, not from us
    // re-threading the value by hand. gRPC/tonic + rustls (webpki roots),
    // never openssl — this is the transport the capture server this was
    // verified against actually accepts (see lfm2d/README.md).
    let build_failure = |signal: &str, e: opentelemetry_otlp::ExporterBuildError| {
        tracing::error!(
            signal,
            error = %e,
            "telemetry: failed to build an OTLP exporter — continuing with stderr logging only for this signal"
        );
    };

    let tracer_provider = match SpanExporter::builder().with_tonic().build() {
        Ok(exporter) => Some(
            SdkTracerProvider::builder()
                .with_batch_exporter(exporter)
                .with_resource(resource.clone())
                .build(),
        ),
        Err(e) => {
            build_failure("traces", e);
            None
        }
    };

    let meter_provider = match MetricExporter::builder().with_tonic().build() {
        Ok(exporter) => {
            let reader = PeriodicReader::builder(exporter).build();
            Some(SdkMeterProvider::builder().with_reader(reader).with_resource(resource.clone()).build())
        }
        Err(e) => {
            build_failure("metrics", e);
            None
        }
    };

    let logger_provider = match LogExporter::builder().with_tonic().build() {
        Ok(exporter) => {
            Some(SdkLoggerProvider::builder().with_batch_exporter(exporter).with_resource(resource).build())
        }
        Err(e) => {
            build_failure("logs", e);
            None
        }
    };

    // Register the subscriber ONCE with every layer that successfully
    // built. Order doesn't matter for correctness here (each layer only
    // observes, none of them short-circuit the others).
    let otel_trace_layer =
        tracer_provider.as_ref().map(|p| tracing_opentelemetry::layer().with_tracer(p.tracer("lfm2d")));
    let otel_log_layer = logger_provider.as_ref().map(OpenTelemetryTracingBridge::new);

    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer)
        .with(otel_trace_layer)
        .with(otel_log_layer)
        .init();

    if let Some(p) = &meter_provider {
        global::set_meter_provider(p.clone());
    }

    tracing::info!(endpoint = %endpoint, "telemetry: OTLP export enabled (gRPC/tonic, rustls)");

    TelemetryGuard { tracer_provider, meter_provider, logger_provider }
}

// ------------------------------------------------------------- instruments

/// `lfm2d`'s [`opentelemetry::metrics::Meter`] — obtained fresh each call
/// (cheap: the global meter provider caches by instrumentation name), never
/// stored in a `static`, so tests that never call [`init`] still get a
/// harmless no-op meter rather than needing special-case wiring.
pub fn meter() -> opentelemetry::metrics::Meter {
    global::meter("lfm2d")
}

/// Duration instrument names. **The unit is NOT part of the name** — it is
/// carried by `.with_unit("ms")`, and exporters append it themselves.
///
/// These used to be `lfm2d.request.duration_ms`, which the Prometheus
/// exporter turned into `lfm2d_request_duration_ms_milliseconds_bucket` —
/// the unit twice, once in our spelling and once in the exporter's. Verified
/// live in VictoriaMetrics 2026-08-12 before the rename, all 6 histogram
/// series carrying the doubled suffix.
///
/// **This rename BREAKS any query using the old names.** Taken deliberately
/// and early: nothing had been built on them yet (checked — 8 lfm2d series,
/// no dashboards), and the cost of this only ever grows. If you are reading
/// this because a query broke, the mapping is
/// `lfm2d_*_duration_ms_milliseconds_*` → `lfm2d_*_duration_milliseconds_*`.
const REQUEST_DURATION: &str = "lfm2d.request.duration";
const INFERENCE_DURATION: &str = "lfm2d.inference.duration";

/// Request counter + duration histogram, recorded once per HTTP response by
/// `server`'s telemetry middleware.
pub fn record_request(route: &str, method: &str, status: u16, elapsed: Duration) {
    let attrs = [
        KeyValue::new("route", route.to_string()),
        KeyValue::new("method", method.to_string()),
        KeyValue::new("status", i64::from(status)),
    ];
    meter().u64_counter("lfm2d.requests").with_description("HTTP requests served").build().add(1, &attrs);
    meter()
        .f64_histogram(REQUEST_DURATION)
        .with_description("HTTP request duration")
        .with_unit("ms")
        .build()
        .record(elapsed.as_secs_f64() * 1000.0, &attrs);
}

/// Inference duration histogram, recorded once per worker-thread command by
/// `worker`'s command loop — `operation` is `embed`/`predict`/`classify`/
/// `route`/`cascade`/`list_models`.
pub fn record_inference_duration(operation: &'static str, elapsed: Duration) {
    let attrs = [KeyValue::new("operation", operation)];
    meter()
        .f64_histogram(INFERENCE_DURATION)
        .with_description("Inference duration by operation kind")
        .with_unit("ms")
        .build()
        .record(elapsed.as_secs_f64() * 1000.0, &attrs);
}

/// Shadow-classifier agreement counter, recorded once per `classify`/
/// `cascade` call when `--candidate-classifier-dir` is configured
/// (`engine_real.rs`). Labels are the two model ids plus BOTH verdicts —
/// each is one of the classifier's own small label set (3 in this
/// checkpoint family), so cardinality is bounded at `ids × labels²`, not
/// per-request. Deliberately never carries the input text or a per-request
/// identifier — same reasoning `--log-input-hash`'s doc comment gives for
/// keeping unbounded values out of metric labels entirely.
pub fn record_candidate_agreement(primary_id: &str, candidate_id: &str, primary_top: &str, candidate_top: &str) {
    let attrs = [
        KeyValue::new("primary_model", primary_id.to_string()),
        KeyValue::new("candidate_model", candidate_id.to_string()),
        KeyValue::new("primary_top", primary_top.to_string()),
        KeyValue::new("candidate_top", candidate_top.to_string()),
        KeyValue::new("agree", primary_top == candidate_top),
    ];
    meter()
        .u64_counter("lfm2d.candidate.agreement")
        .with_description("Shadow-classifier verdict agreement, by primary/candidate top label")
        .build()
        .add(1, &attrs);
}

/// Register the queue-depth observable gauge over `queue_depth` — called
/// once from `main.rs` right after [`crate::worker::WorkerHandle::spawn_crash_on_panic`]
/// returns a handle (the [`std::sync::Arc<std::sync::atomic::AtomicUsize>`]
/// the handle increments on send / the worker decrements on pickup — see
/// `worker.rs`). The returned [`opentelemetry::metrics::ObservableGauge`]
/// must be kept alive for the process lifetime (dropping it deregisters the
/// callback), so the caller holds it inside the same `_telemetry` binding
/// as [`TelemetryGuard`].
pub fn register_queue_depth_gauge(
    queue_depth: std::sync::Arc<std::sync::atomic::AtomicUsize>,
) -> opentelemetry::metrics::ObservableGauge<u64> {
    meter()
        .u64_observable_gauge("lfm2d.worker.queue_depth")
        .with_description("Commands queued for the serial inference worker, not yet picked up")
        .with_callback(move |observer| {
            observer.observe(queue_depth.load(std::sync::atomic::Ordering::SeqCst) as u64, &[]);
        })
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The unit belongs in `.with_unit()`, never in the instrument name —
    /// exporters append it, so spelling it twice produces
    /// `..._duration_ms_milliseconds_bucket`. That shipped and reached
    /// VictoriaMetrics before anyone noticed, because nothing in the build
    /// looks at a metric name.
    ///
    /// Deliberately checks SUFFIXES that encode a unit rather than the two
    /// current names: a new instrument added next year gets this check for
    /// free, which an equality assertion would not give.
    #[test]
    fn instrument_names_do_not_repeat_their_unit() {
        const UNIT_SUFFIXES: [&str; 8] =
            ["_ms", "_us", "_ns", "_s", "_bytes", "_seconds", "_milliseconds", "_count"];

        for name in [REQUEST_DURATION, INFERENCE_DURATION] {
            for suffix in UNIT_SUFFIXES {
                assert!(
                    !name.ends_with(suffix),
                    "instrument {name:?} ends with unit suffix {suffix:?} — the unit belongs \
                     in .with_unit(), and exporters append it, so this becomes a doubled \
                     suffix like lfm2d_request_duration_ms_milliseconds_bucket"
                );
            }
        }
    }

    /// Names are a wire contract for every dashboard and alert. Renaming one
    /// silently breaks queries with no build error and no runtime error, so
    /// pin them: this test failing means someone must go update the queries.
    #[test]
    fn instrument_names_are_pinned() {
        assert_eq!(REQUEST_DURATION, "lfm2d.request.duration");
        assert_eq!(INFERENCE_DURATION, "lfm2d.inference.duration");
    }
}
