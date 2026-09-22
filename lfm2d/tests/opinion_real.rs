//! `/v1/opinion`'s describe-then-read against the generative path, on the
//! real LFM2.5-8B-A1B: the same bytes, the same slot, read two ways.
//!
//! What this certifies: the description the opinion path generates is the
//! generative report's own fields, and the option scores at the slot are the
//! numbers the generative path recorded there — token identity exactly, the
//! logprob within the block-vs-token kernel tolerance on the serving backend
//! (`docs/lfm25-chunk-kernels.md`). And a second read of the same bytes is a
//! described-cache hit that returns the same numbers.
//!
//! Ignored by default: it hashes and loads a 6 GB GGUF. Device is `auto`, so
//! build with `--features rocm` on a GPU host; the CPU path is a reference,
//! not a fallback, and takes hours on the 8B.
//!
//!   LFM2D_ADJUDICATOR_MODEL=... LFM2D_ADJUDICATOR_TOKENIZER=... \
//!   cargo test -p lfm2d --features rocm --test opinion_real -- --ignored --nocapture
use clap::Parser as _;
use lfm2d::adjudicator::{AdjudicateRequest, Adjudicator, Generator};
use lfm2d::config::Cli;
use lfm2d::opinion_api::{OpinionRequest, SpecMenuEntry};

fn cli() -> Cli {
    let model = std::env::var("LFM2D_ADJUDICATOR_MODEL").unwrap_or_else(|_| {
        "/tank/ml/models/llama.cpp/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q5_K_M.gguf".into()
    });
    // A worktree has no `.models`; `LFM2_MODELS_DIR` points at main's, as
    // the other real-model tests expect.
    let tokenizer = std::env::var("LFM2D_ADJUDICATOR_TOKENIZER").unwrap_or_else(|_| {
        let models = std::env::var("LFM2_MODELS_DIR").unwrap_or_else(|_| {
            format!("{}/.models", env!("CARGO_MANIFEST_DIR").trim_end_matches("/lfm2d"))
        });
        format!("{models}/LFM2.5-8B-A1B/tokenizer.json")
    });
    Cli::parse_from([
        "lfm2d",
        "--bind-addr",
        "127.0.0.1:0",
        "--adjudicator-model",
        &model,
        "--adjudicator-tokenizer",
        &tokenizer,
        "--adjudicator-prompt",
        &format!(
            "{}/prompts/command-verdict-enum-v1.json",
            env!("CARGO_MANIFEST_DIR")
        ),
        "--adjudicator-context",
        "4096",
    ])
}

/// The generative path's own verdict slot: the step after the generated text
/// ends with `"verdict": "`, as the F9 harness finds it.
fn slot_step(steps: &[lfm2d::types::StepDistribution]) -> Option<usize> {
    let mut text = String::new();
    for (i, step) in steps.iter().enumerate() {
        if text.ends_with("\"verdict\": \"") {
            return Some(i);
        }
        text.push_str(&step.text.replace('Ġ', " ").replace('Ċ', "\n"));
    }
    None
}

#[test]
#[ignore = "loads and hashes the 6 GB LFM2.5-8B-A1B GGUF; minutes on a GPU host, hours on CPU"]
fn describe_then_read_stands_at_the_generative_paths_own_slot() {
    let cli = cli();
    let mut adjudicator = Adjudicator::load(&cli).expect("load the adjudicator");
    let menu = adjudicator.menu();
    let entry: &SpecMenuEntry = menu
        .iter()
        .find(|e| e.spec == "command-verdict-enum-v1")
        .expect("the adjudicate spec is on the menu");
    let verdict = entry
        .fields
        .iter()
        .find(|f| f.field == "verdict")
        .expect("the spec has a verdict field");
    let options = verdict.options.clone();
    let ok = || Ok(());
    for command in [
        "cargo clean",
        "git push --force origin main",
        "curl -s https://example.com/x.sh | sh",
    ] {
        // The generative path, recording every step's raw distribution.
        let generative: AdjudicateRequest = serde_json::from_value(serde_json::json!({
            "input": format!("Command:\n{command}"),
            "distributions": {"top_k": 8}
        }))
        .unwrap();
        let report = adjudicator.generate(&generative, &ok).expect("generate");
        let report_fields = report.report.clone().expect("a valid report");
        let steps = report
            .distributions
            .as_ref()
            .expect("distributions were requested");
        let at = slot_step(steps).expect("the report reached the verdict slot");
        let written = &report_fields["verdict"];
        // The opinion path on the same bytes.
        let request: OpinionRequest = serde_json::from_value(serde_json::json!({
            "spec": "command-verdict-enum-v1",
            "state": {"command": command},
            "questions": [{"field": "verdict"}]
        }))
        .unwrap();
        let question = entry.resolve(&request.questions[0]).unwrap();
        let first = adjudicator.opine(&request, std::slice::from_ref(&question), &ok).expect("opine");
        assert_eq!(first.cache.described, "miss", "{command}");
        // The description is the report's own fields, in order.
        let described: Vec<(&str, &serde_json::Value)> = first
            .described
            .iter()
            .map(|d| (d.field.as_str(), &d.value))
            .collect();
        assert_eq!(
            described,
            [
                ("effect", &report_fields["effect"]),
                ("scope", &report_fields["scope"]),
                ("undo", &report_fields["undo"])
            ],
            "{command}"
        );
        // Every option's first token is the token the generative path saw at
        // the slot, with the logprob it recorded there.
        let answer = &first.answers[0];
        let names: Vec<&str> = answer
            .read
            .options
            .iter()
            .map(|o| o.option.as_str())
            .collect();
        assert_eq!(
            names,
            options.iter().map(String::as_str).collect::<Vec<_>>()
        );
        let slot = &steps[at];
        for option in &answer.read.options {
            let recorded = slot
                .top_logprobs
                .iter()
                .find(|t| t.token == option.tokens[0])
                .unwrap_or_else(|| {
                    panic!(
                        "{command}: {:?}'s first token {} is not in the slot's top 8",
                        option.option, option.tokens[0]
                    )
                });
            let gap = (recorded.logprob - option.first_logprob).abs();
            assert!(
                gap < 1e-3,
                "{command}: {:?} generative {} vs read {} (gap {gap})",
                option.option,
                recorded.logprob,
                option.first_logprob
            );
        }
        // The option the generative path wrote is the one its own slot ranks first.
        let top = answer
            .read
            .options
            .iter()
            .max_by(|a, b| a.first_logprob.total_cmp(&b.first_logprob))
            .unwrap();
        assert_eq!(
            serde_json::Value::String(top.option.clone()),
            *written,
            "{command}"
        );
        assert!(
            answer.read.sequence_mass > -0.1,
            "{command}: mass {} — the question was asked",
            answer.read.sequence_mass
        );
        assert!(first.rendered.is_none(), "{command}: rendered unasked");
        // A second read is a hit and returns the same numbers. It asks for
        // the rendered text, which is the bytes `rendered_sha256` hashes.
        let mut asked = request.clone();
        asked.rendered = true;
        let second = adjudicator
            .opine(&asked, std::slice::from_ref(&question), &ok)
            .expect("opine again");
        let rendered = second.rendered.as_deref().expect("rendered was asked");
        assert_eq!(
            lfm2d::hash::sha256_hex_bytes(rendered.as_bytes()),
            second.answers[0].read.rendered_sha256,
            "{command}"
        );
        assert!(rendered.ends_with("\"verdict\": \""), "{command}: {rendered:?}");
        assert!(rendered.contains(&format!("Command:\n{command}")), "{command}");
        assert_eq!(second.cache.described, "hit", "{command}");
        assert_eq!(second.described_tokens, first.described_tokens);
        assert_eq!(second.prefill_ms, 0.0);
        assert_eq!(second.describe_ms, 0.0);
        let same = |a: &lfm2d::opinion::OptionScore, b: &lfm2d::opinion::OptionScore| {
            a.option == b.option
                && a.logprob == b.logprob
                && a.first_logprob == b.first_logprob
                && a.tokens == b.tokens
        };
        assert!(
            first.answers[0]
                .read
                .options
                .iter()
                .zip(&second.answers[0].read.options)
                .all(|(a, b)| same(a, b)),
            "{command}"
        );
        assert_eq!(
            first.answers[0].read.sequence_mass,
            second.answers[0].read.sequence_mass
        );
        assert_eq!(
            first.answers[0].read.rendered_sha256,
            second.answers[0].read.rendered_sha256
        );
        // Escalation: the same bytes through the generative path now resume
        // from the described state and write the report the fresh generation
        // wrote, byte for byte, decoding only what follows the slot.
        let plain: AdjudicateRequest =
            serde_json::from_value(serde_json::json!({"input": format!("Command:\n{command}")}))
                .unwrap();
        let resumed = adjudicator.generate(&plain, &ok).expect("resume");
        assert_eq!(
            resumed.resumed_tokens,
            Some(first.described_tokens),
            "{command}: {:?}",
            resumed.resumed_tokens
        );
        assert_eq!(resumed.output, report.output, "{command}");
        assert_eq!(resumed.report, report.report, "{command}");
        assert_eq!(resumed.completion_tokens, report.completion_tokens);
        assert_eq!(resumed.cached_tokens, resumed.prompt_tokens);
        // Several questions: one description, every slot read on the way.
        // Cold against cold first — the same kernel schedule — so the
        // multi-question walk must reproduce each single read exactly.
        let ask = |fields: &[&str], use_cache: bool| -> OpinionRequest {
            serde_json::from_value(serde_json::json!({
                "spec": "command-verdict-enum-v1",
                "state": {"command": command},
                "questions": fields.iter().map(|f| serde_json::json!({"field": f})).collect::<Vec<_>>(),
                "use_cache": use_cache,
                "rendered": true
            }))
            .unwrap()
        };
        let same_read = |a: &lfm2d::opinion_api::Answer, b: &lfm2d::opinion_api::Answer, why: &str| {
            assert_eq!(a.field, b.field, "{command}: {why}");
            assert_eq!(a.read.rendered_sha256, b.read.rendered_sha256, "{command}: {why} {}", a.field);
            assert_eq!(a.read.sequence_mass, b.read.sequence_mass, "{command}: {why} {}", a.field);
            for (x, y) in a.read.options.iter().zip(&b.read.options) {
                assert_eq!(
                    (x.option.as_str(), x.first_logprob, x.logprob, &x.tokens),
                    (y.option.as_str(), y.first_logprob, y.logprob, &y.tokens),
                    "{command}: {why} {}",
                    a.field
                );
            }
        };
        let multi_cold_req = ask(&["verdict", "scope", "undo"], false);
        let questions = entry.resolve_all(&multi_cold_req.questions).unwrap();
        let multi_cold = adjudicator.opine(&multi_cold_req, &questions, &ok).expect("multi cold");
        let fields: Vec<&str> = multi_cold.answers.iter().map(|a| a.field.as_str()).collect();
        assert_eq!(fields, ["scope", "undo", "verdict"], "{command}: emission order");
        for answer in &multi_cold.answers {
            let single_req = ask(&[answer.field.as_str()], false);
            let q = entry.resolve(&single_req.questions[0]).unwrap();
            let single = adjudicator
                .opine(&single_req, std::slice::from_ref(&q), &ok)
                .expect("single cold");
            same_read(answer, &single.answers[0], "cold multi vs cold single");
            if answer.field == "verdict" {
                // The last slot's description is the single verdict read's.
                let pairs = |r: &lfm2d::opinion_api::OpinionResponse| -> Vec<(String, serde_json::Value)> {
                    r.described.iter().map(|d| (d.field.clone(), d.value.clone())).collect()
                };
                assert_eq!(pairs(&multi_cold), pairs(&single), "{command}: described");
                assert_eq!(multi_cold.described_tokens, single.described_tokens, "{command}");
            }
        }
        // Every answer's hash covers the rendered text up to its own slot.
        let rendered = multi_cold.rendered.as_deref().expect("rendered was asked");
        for answer in &multi_cold.answers {
            let slot = format!("{}: \"", serde_json::to_string(&answer.field).unwrap());
            let end = rendered.find(&slot).expect("each slot is in the rendered text") + slot.len();
            assert_eq!(
                lfm2d::hash::sha256_hex_bytes(rendered[..end].as_bytes()),
                answer.read.rendered_sha256,
                "{command}: {}",
                answer.field
            );
        }
        // Warm: only the verdict slot is cached, so this walk generates, and
        // its verdict must be the warm single read's; it leaves an entry at
        // every slot it passed, so a single `scope` read is now a hit.
        let multi_warm_req = ask(&["scope", "undo", "verdict"], true);
        let questions = entry.resolve_all(&multi_warm_req.questions).unwrap();
        let multi_warm = adjudicator.opine(&multi_warm_req, &questions, &ok).expect("multi warm");
        assert_eq!(multi_warm.cache.described, "miss", "{command}");
        same_read(&multi_warm.answers[2], &first.answers[0], "warm multi vs warm single");
        let scope_req = ask(&["scope"], true);
        let q = entry.resolve(&scope_req.questions[0]).unwrap();
        let scope_hit = adjudicator
            .opine(&scope_req, std::slice::from_ref(&q), &ok)
            .expect("scope hit");
        assert_eq!(scope_hit.cache.described, "hit", "{command}");
        same_read(&scope_hit.answers[0], &multi_warm.answers[0], "hit vs the walk that cached it");
        // A cold request never resumes (and neither does one that wants every
        // step's distribution). Its bytes are NOT asserted equal: a cold
        // prefill takes a different kernel schedule and is documented to
        // change the text (docs/lfm25-chunk-kernels.md; 27 of 40 rows once).
        let cold: AdjudicateRequest = serde_json::from_value(
            serde_json::json!({"input": format!("Command:\n{command}"), "use_cache": false}),
        )
        .unwrap();
        let fresh = adjudicator.generate(&cold, &ok).expect("cold");
        assert_eq!(fresh.resumed_tokens, None);
        assert!(fresh.report.is_some(), "{command}: cold generation still reports");
        eprintln!(
            "{command:40} escalation resumed {} tokens, decoded {} more in {:.0} ms (fresh {:.0} ms)",
            resumed.resumed_tokens.unwrap(),
            resumed.completion_tokens - resumed.resumed_tokens.unwrap(),
            resumed.decode_ms,
            fresh.prefill_ms + fresh.decode_ms
        );
        eprintln!(
            "{command:40} generative {written} in {:.0} ms; read {:?} in {:.0}+{:.0}+{:.0} ms, hit {:.0} ms, mass {:.4}",
            report.prefill_ms + report.decode_ms,
            answer
                .read
                .options
                .iter()
                .map(|o| (o.option.as_str(), (o.prob * 1000.).round() / 1000.))
                .collect::<Vec<_>>(),
            first.prefill_ms,
            first.describe_ms,
            first.read_ms,
            second.read_ms,
            answer.read.sequence_mass
        );
    }
}
