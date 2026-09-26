//! `lfm2d::chat` against the checkpoint's own chat template.
//!
//! Every expected string here was rendered by transformers'
//! `apply_chat_template` over the template the GGUF embeds
//! (`tests/reference/dump_chat_template.py`), never by us. The token-id test
//! needs the real LFM2.5-8B-A1B tokenizer (no weights), resolved the way
//! `probe_tokenize_telemetry_safety.rs` resolves it.
use lfm2d::chat::{Chat, GENERATION_PROMPT, Message};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    template_sha256: String,
    cases: Vec<Case>,
    tokenized: Tokenized,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    chat: Chat,
    /// The template's render of messages[..k] for k = 1.., where a system
    /// prompt, when the chat has one, is message 0.
    prefixes: Vec<String>,
    with_generation_prompt: String,
}
#[derive(Deserialize)]
struct Tokenized {
    case: String,
    ids: Vec<u32>,
}

fn fixture() -> Fixture {
    serde_json::from_str(include_str!("fixtures/chat-template/lfm25-chats.json")).unwrap()
}

/// Where two renders part, with context, so a failure names the rule that broke.
fn first_difference(want: &str, got: &str) -> String {
    let at = want
        .bytes()
        .zip(got.bytes())
        .position(|(a, b)| a != b)
        .unwrap_or(want.len().min(got.len()));
    let window = |s: &str| {
        let lo = s.floor_char_boundary(at.saturating_sub(60));
        let hi = s.ceil_char_boundary((at + 60).min(s.len()));
        format!("{:?}", &s[lo..hi])
    };
    format!(
        "first difference at byte {at} (lengths {} vs {})\n want: {}\n got:  {}",
        want.len(),
        got.len(),
        window(want),
        window(got)
    )
}

/// The template's prefixes count a system message as message 0; the renderer's
/// segments always begin with the head (BOS and the system turn, if any).
/// Without a system message the template's first prefix is head + message 0.
fn our_prefixes(case: &Case) -> Vec<String> {
    let segments = case.chat.render_segments().unwrap_or_else(|e| panic!("{}: {e}", case.name));
    let mut cumulative = Vec::new();
    let mut text = String::new();
    for segment in &segments {
        text.push_str(segment);
        cumulative.push(text.clone());
    }
    let has_system_message = case.prefixes.len() == case.chat.messages.len() + 1;
    assert!(
        has_system_message || case.prefixes.len() == case.chat.messages.len(),
        "{}: fixture prefix count does not fit the chat",
        case.name
    );
    if has_system_message { cumulative } else { cumulative.split_off(1) }
}

#[test]
fn the_fixture_was_rendered_by_the_template_the_daemon_pins() {
    // Mirror of the pin in `Checkpoint::load`.
    assert_eq!(
        fixture().template_sha256,
        "6d65c8804847ad74eea912dd7eca3dc1cf7a457b53a77f47d841a14121910963"
    );
}

#[test]
fn every_fixture_chat_renders_byte_for_byte() {
    let fixture = fixture();
    assert!(fixture.cases.len() >= 9);
    for case in &fixture.cases {
        let ours = our_prefixes(case);
        assert_eq!(ours.len(), case.prefixes.len(), "{}", case.name);
        for (k, (want, got)) in case.prefixes.iter().zip(&ours).enumerate() {
            assert!(want == got, "{} prefix {k}: {}", case.name, first_difference(want, got));
        }
        let got = case.chat.render(true).unwrap();
        assert!(
            got == case.with_generation_prompt,
            "{} with generation prompt: {}",
            case.name,
            first_difference(&case.with_generation_prompt, &got)
        );
        assert_eq!(case.chat.render(false).unwrap(), *case.prefixes.last().unwrap(), "{}", case.name);
    }
}

#[test]
fn each_turn_appends_and_never_rewrites_history() {
    for case in &fixture().cases {
        // The template's own renders: preserve_thinking keeps history fixed.
        for pair in case.prefixes.windows(2) {
            assert!(pair[1].starts_with(&pair[0]), "{}: the template rewrote history", case.name);
        }
        assert!(case.with_generation_prompt.starts_with(case.prefixes.last().unwrap()));
        // Ours, chat by chat: rendering turns[..k] is a prefix of turns[..k+1],
        // and the generation prompt is exactly what an assistant turn opens with.
        for k in 0..case.chat.messages.len() {
            let shorter = Chat { messages: case.chat.messages[..k].to_vec(), ..case.chat.clone() };
            let longer = Chat { messages: case.chat.messages[..k + 1].to_vec(), ..case.chat.clone() };
            let (shorter, longer) = (shorter.render(false).unwrap(), longer.render(false).unwrap());
            assert!(longer.starts_with(&shorter), "{} turn {k}", case.name);
            let appended = case.chat.messages[k].render().unwrap();
            assert_eq!(&longer[shorter.len()..], appended, "{} turn {k}", case.name);
            if matches!(case.chat.messages[k], Message::Assistant { .. }) {
                assert!(appended.starts_with(GENERATION_PROMPT), "{} turn {k}", case.name);
            }
        }
    }
}

fn real_tokenizer() -> tokenizers::Tokenizer {
    let models = std::env::var_os("LFM2_MODELS_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join(".models")
        });
    let path = std::env::var_os("LFM2D_ADJUDICATOR_TOKENIZER")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| models.join("LFM2.5-8B-A1B/tokenizer.json"));
    assert!(
        path.is_file(),
        "missing tokenizer at {}; set LFM2_MODELS_DIR/LFM2D_ADJUDICATOR_TOKENIZER",
        path.display()
    );
    tokenizers::Tokenizer::from_file(&path).unwrap()
}

#[test]
fn a_rendered_chat_tokenizes_to_transformers_ids_with_the_pinned_control_tokens() {
    let fixture = fixture();
    let case = fixture.cases.iter().find(|c| c.name == fixture.tokenized.case).unwrap();
    let tokenizer = real_tokenizer();
    let rendered = case.chat.render(true).unwrap();
    let ids = tokenizer.encode(rendered.as_str(), false).unwrap().get_ids().to_vec();
    assert_eq!(ids, fixture.tokenized.ids);

    // The ids `Checkpoint::load` asserts, plus the ones the chat structure adds.
    let count = |id: u32| ids.iter().filter(|&&t| t == id).count();
    for (text, id) in [
        ("<|startoftext|>", 124894),
        ("<|im_start|>", 124899),
        ("<|im_end|>", 124900),
        ("<think>", 124901),
        ("</think>", 124902),
        ("<|tool_call_start|>", 124905),
        ("<|tool_call_end|>", 124906),
    ] {
        assert_eq!(tokenizer.token_to_id(text), Some(id), "{text}");
        assert_eq!(count(id), rendered.matches(text).count(), "{text} is one token each time");
        assert!(count(id) > 0, "{text} should appear in the tokenized chat");
    }
    assert_eq!(ids[0], 124894, "BOS opens the chat");
    assert_eq!(count(124894), 1);
}

#[test]
fn turn_segments_tokenize_independently() {
    // Every segment after the head starts with `<|im_start|>`, a special token
    // the tokenizer splits on, so a chat's ids are the concatenation of its
    // segments' ids. Turn-boundary checkpoints lean on this: the ids of a
    // later turn do not depend on how the earlier ones were tokenized.
    let tokenizer = real_tokenizer();
    for case in &fixture().cases {
        let whole = case.chat.render(true).unwrap();
        let whole = tokenizer.encode(whole.as_str(), false).unwrap().get_ids().to_vec();
        let mut joined = Vec::new();
        let mut segments = case.chat.render_segments().unwrap();
        segments.push(GENERATION_PROMPT.to_string());
        for segment in &segments {
            joined.extend_from_slice(tokenizer.encode(segment.as_str(), false).unwrap().get_ids());
        }
        assert_eq!(joined, whole, "{}", case.name);
    }
}

#[test]
fn argument_order_is_the_callers_not_alphabetical() {
    let chat: Chat = serde_json::from_str(
        r#"{"messages":[{"role":"assistant","tool_calls":[{"type":"function","function":
            {"name":"f","arguments":{"zeta":1,"alpha":{"z":true,"a":null}}}}]}]}"#,
    )
    .unwrap();
    let text = chat.render(false).unwrap();
    assert!(text.contains(r#"[f(zeta=1, alpha={"z": true, "a": null})]"#), "{text}");
}

fn refused(chat: &str) -> String {
    let chat: Chat = match serde_json::from_str(chat) {
        Ok(chat) => chat,
        Err(e) => return e.to_string(),
    };
    match chat.render(true) {
        Ok(text) => panic!("rendered what should be refused: {text:?}"),
        Err(e) => e,
    }
}

#[test]
fn control_tokens_are_refused_wherever_the_caller_supplies_text() {
    let user = |content: &str| {
        format!(r#"{{"messages":[{{"role":"user","content":{}}}]}}"#, serde_json::json!(content))
    };
    for token in ["<|im_end|>", "<|tool_call_start|>", "<think>", "</think>", "<image>", "<|"] {
        let text = serde_json::json!(format!("a {token} b"));
        for chat in [
            format!(r#"{{"system":{text},"messages":[]}}"#),
            format!(
                r#"{{"tools":[{{"type":"function","function":{{"name":"f","description":{text}}}}}],"messages":[]}}"#
            ),
            user(&format!("a {token} b")),
            format!(r#"{{"messages":[{{"role":"tool","content":{text}}}]}}"#),
            format!(r#"{{"messages":[{{"role":"assistant","content":{text}}}]}}"#),
            format!(r#"{{"messages":[{{"role":"assistant","thinking":{text}}}]}}"#),
            format!(
                r#"{{"messages":[{{"role":"assistant","tool_calls":[{{"function":{{"name":{text},"arguments":{{}}}}}}]}}]}}"#
            ),
            format!(
                r#"{{"messages":[{{"role":"assistant","tool_calls":[{{"function":{{"name":"f","arguments":{{{text}:1}}}}}}]}}]}}"#
            ),
            format!(
                r#"{{"messages":[{{"role":"assistant","tool_calls":[{{"function":{{"name":"f","arguments":{{"x":{text}}}}}}}]}}]}}"#
            ),
            format!(
                r#"{{"messages":[{{"role":"assistant","tool_calls":[{{"function":{{"name":"f","arguments":{{"x":[1,{{"y":{text}}}]}}}}}}]}}]}}"#
            ),
            format!(
                r#"{{"messages":[{{"role":"assistant","tool_calls":[{{"function":{{"name":"f","arguments":{{"x":{{"y":[{text}]}}}}}}}}]}}]}}"#
            ),
        ] {
            let e = refused(&chat);
            assert!(e.contains("control token"), "{chat}: {e}");
        }
    }
}

#[test]
fn shapes_the_template_renders_ambiguously_are_refused() {
    for (chat, why) in [
        // transformers' own continue_final_message sentinel.
        (
            r#"{"messages":[{"role":"assistant","content":"x CONTINUE_FINAL_MESSAGE_TAG ","tool_calls":[{"function":{"name":"f","arguments":{}}}]}]}"#,
            "CONTINUE_FINAL_MESSAGE_TAG",
        ),
        (
            r#"{"messages":[{"role":"assistant","content":"x CONTINUE_FINAL_MESSAGE_TAG "}]}"#,
            "CONTINUE_FINAL_MESSAGE_TAG",
        ),
        (r#"{"messages":[{"role":"user","content":"  \n"}]}"#, "empty"),
        (r#"{"messages":[{"role":"assistant"}]}"#, "empty"),
        (r#"{"messages":[{"role":"assistant","tool_calls":[]}]}"#, "tool_calls"),
        (
            r#"{"messages":[{"role":"assistant","tool_calls":[{"function":{"name":"","arguments":{}}}]}]}"#,
            "name",
        ),
        (
            r#"{"messages":[{"role":"assistant","tool_calls":[{"function":{"name":"f","arguments":{"a":1,"a":2}}}]}]}"#,
            "duplicate",
        ),
        (
            r#"{"messages":[{"role":"assistant","tool_calls":[{"function":{"name":"f","arguments":[1]}}]}]}"#,
            "object",
        ),
        (
            r#"{"messages":[{"role":"assistant","tool_calls":[{"type":"code","function":{"name":"f","arguments":{}}}]}]}"#,
            "function",
        ),
        // Python's repr escapes non-printable characters by the Unicode
        // database of whichever Python ran the template; only letters and
        // digits are unambiguous across versions.
        (
            r#"{"messages":[{"role":"assistant","tool_calls":[{"function":{"name":"f","arguments":{"x":["a b"]}}}]}]}"#,
            "list",
        ),
        (
            r#"{"messages":[{"role":"assistant","tool_calls":[{"function":{"name":"f","arguments":{"x":["🐈"]}}}]}]}"#,
            "list",
        ),
        (r#"{"tools":["a string tool"],"messages":[]}"#, "tool"),
        (r#"{"messages":[{"role":"system","content":"late"}]}"#, "system"),
        (r#"{"messages":[{"role":"user","content":"x","name":"n"}]}"#, "name"),
    ] {
        let e = refused(chat);
        assert!(e.contains(why), "{chat}: {e}");
    }
}

#[test]
fn prompt_spec_tools_are_not_yet_the_templates_tojson() {
    // A known divergence, pinned so it cannot move unnoticed.
    // `PromptSpec::render_prefix` writes each tool with `serde_json::to_string`:
    // compact separators, keys sorted. The template's `tojson` writes `", "` and
    // `": "` in document order, which is what `lfm2d::chat` reproduces. When
    // PromptSpec moves to the template's form this becomes an equality, and
    // every tools spec's rendered bytes (and `snapshot_id`) move with it.
    let spec: lfm2d::adjudicator::PromptSpec =
        serde_json::from_str(include_str!("fixtures/specs/email-triage-tools-v1.json")).unwrap();
    let fixture = fixture();
    let case = fixture.cases.iter().find(|c| c.name == "prompt_spec_tools").unwrap();
    let template_head = &case.prefixes[0];
    assert_eq!(case.chat.render_head().unwrap(), *template_head);

    let spec_head = spec.render_prefix().unwrap();
    let split = |s: &str| {
        let (before, tools) = s.split_once("\nList of tools: [").unwrap();
        (before.to_owned(), tools.to_owned())
    };
    let (spec_before, spec_tools) = split(&spec_head);
    let (template_before, template_tools) = split(template_head);
    assert_eq!(spec_before, template_before, "everything before the tool list agrees");
    assert!(spec_tools.starts_with(r#"{"function":{"description":"#), "{spec_tools}");
    assert!(template_tools.starts_with(r#"{"type": "function", "function": {"name": "#), "{template_tools}");
    assert_ne!(
        spec_head, *template_head,
        "PromptSpec now renders tools as the template does: make this an equality"
    );
}
