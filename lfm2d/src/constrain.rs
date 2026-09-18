//! Token-level constrained JSON decoding for the causal adjudicator.
//!
//! # Why
//!
//! `adjudicator::validate_report` grades a report *after* generation. Measured
//! on 733 rows through a reference implementation, unconstrained generation
//! makes the model echo the schema back as its answer; under a grammar the
//! format-failure rate went to zero (`docs/field-requests.md`). This module
//! makes an invalid document unreachable instead of merely detectable: every
//! decode step masks the logits so only tokens that can continue a document
//! satisfying the schema survive the greedy selection.
//!
//! # Scope, stated as rulings rather than left implicit
//!
//! - **The whole completion is the JSON document.** No `<think>` preamble, no
//!   fences, no trailing prose; `validate_report` rejects all three. A grammar
//!   that admits a free-text region admits one we cannot bound, and the one
//!   measurement we have says LFM2.5's reasoning argues severity *down*.
//!   So: object first byte, EOS
//!   immediately after the closing brace. The reasoning region the model
//!   expects is supplied *already closed* by the prompt instead
//!   (`adjudicator::Reasoning`), which is what keeps the object's first byte
//!   from being a token the model scores 17 to 22 nats below its own choice.
//! - **Key order is the schema's `required` order** and is load-bearing
//!   (`docs/field-requests.md` decision 2: reading `severity` with no fields
//!   in front of it reproduced 17/40 severe rows; after the scaffold fields,
//!   40/40). `properties` map order is ignored, and the system prompt states
//!   the schema in that same `required` order — `adjudicator::render_schema`,
//!   because stating one order and masking into another cost a forced step.
//! - **Spaced separators, fixed.** `{"a": "x", "b": true}` — one space after
//!   `:` and after `,`, nowhere else, matching the model's own canonical JSON
//!   style. Measured against LFM2.5-8B under the earlier compact ruling
//!   (`{"a":"x","b":true}`): after `"effect":` the model put probability 1.00
//!   on the token ` "` (space, quote), which compact grammar forbade. The
//!   best-scoring *legal* token was `":` — an opening quote followed by a
//!   colon — so its quote opened the string and its colon became the
//!   string's first CONTENT byte. Every free-text field came out as the
//!   literal value `": `. Forcing the model off its own tokenization made it
//!   misread a structural quote as content; spacing the separators to match
//!   its canonical style fixed it outright. The whitespace stays fixed
//!   rather than optional for the same reason as before: every byte we admit
//!   is a byte the model can spend instead of answering.
//! - **No `\uXXXX` escapes.** Every character they can express is reachable
//!   literally as UTF-8, and admitting them admits lone surrogate escapes,
//!   which `serde_json` rejects — i.e. a path from a "valid" grammar walk to
//!   an invalid document. The short escapes (`\" \\ \/ \b \f \n \r \t`) are
//!   admitted.
//! - **Truncation stays visible.** A completion that runs out of `max_tokens`
//!   mid-document still reports `finish_reason: "length"` and still fails
//!   `validate_report`. Inventing filler so the object closes would be
//!   fabricating a field value; a loud short read is the better failure.
//!
//! # Where the cost is paid
//!
//! [`Grammar`] is built once, at load, and shared by every generation: the
//! compiled [`Program`], the [`Vocabulary`]'s byte expansions, and a mask plan
//! per cursor, memoised the first time any report visits it. A plan depends
//! only on the cursor, so it is never wrong for a later report. [`Masker`] is
//! what one generation owns — a cursor and a scratch buffer. Measured on the
//! real 125,024-row vocabulary (CPU device, the severity schema, a 48-token
//! report; `constraint_overhead_is_cold_plan_build_plus_a_fixed_per_step_cost`):
//! compile 97 ms, the first report 3.2 ms/token while it builds 36 plans, every
//! later report 0.19 ms/token. Before the grammar was shared every request paid
//! the compile and the cold plans again: roughly 240 ms per report.
//!
//! # What "cannot be constrained" means
//!
//! `adjudicator::validate_schema` accepts a slightly larger language than this
//! grammar can honour, because `validate_report` imposes conditions the schema
//! keywords do not: string values must be non-empty after `trim()`. A schema
//! whose `enum` contains a blank string is therefore accepted by
//! `validate_schema` and satisfiable by *no* document. [`Program::compile`]
//! refuses it loudly rather than falling through to free generation.
//! Because the daemon compiles its grammar in `Adjudicator::load`, that refusal
//! stops the daemon from starting; it never reaches a request.

use candle_core::{Device, Tensor};
use candle_nn::sampling::GreedySampler;
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

/// Additive penalty for a disallowed token. Deliberately finite: both
/// `GreedySampler` paths (host and the ROCm kernel) reject non-finite logits
/// outright, and `-1e30 * 1.05` (the repetition penalty applied afterwards)
/// is still finite and still astronomically below any real logit.
const BLOCK: f32 = -1e30;

// ---------------------------------------------------------------------------
// Vocabulary: token id -> the exact bytes the decoder will emit for it
// ---------------------------------------------------------------------------

/// The GPT-2 byte-level alphabet, inverted: unicode scalar -> byte.
///
/// `tokenizers`' `ByteLevel` decoder concatenates the byte expansion of every
/// token and runs one `String::from_utf8_lossy` over the whole run, so the
/// decoded text is exactly this concatenation — which is why byte-level
/// bookkeeping is the right granularity and why a multi-byte character split
/// across two tokens is fine.
fn char_to_byte() -> Vec<i16> {
    let mut table = vec![-1i16; 0x200];
    let mut printable: Vec<u8> = Vec::new();
    printable.extend(b'!'..=b'~');
    printable.extend(0xA1u8..=0xAC);
    printable.extend(0xAEu8..=0xFF);
    for &b in &printable {
        table[b as usize] = b as i16;
    }
    let mut n = 0u32;
    for b in 0..=255u8 {
        if !printable.contains(&b) {
            table[(256 + n) as usize] = b as i16;
            n += 1;
        }
    }
    table
}

/// Byte expansions for every row of the model's output vocabulary.
///
/// Rows that must never be sampled inside a JSON document — added/control
/// tokens, and the padding rows a GGUF carries beyond the tokenizer's usable
/// vocabulary — carry `None` and are masked unconditionally. The one exception
/// is end-of-text, which the grammar admits exactly once, after the closing
/// brace.
pub struct Vocabulary {
    /// Every usable expansion, concatenated in token-id order. Building the
    /// per-cursor mask touches most of the vocabulary, so one contiguous blob
    /// beats 125k separate allocations: measured 142 ms -> 78 ms on the cold
    /// plan build for a five-field schema, by
    /// `constraint_overhead_is_cold_plan_build_plus_a_fixed_per_step_cost`.
    blob: Vec<u8>,
    /// `(offset, len)` into `blob`; `len == 0` marks a row that must never be
    /// sampled inside a JSON document.
    span: Vec<(u32, u32)>,
    by_first: Vec<Vec<u32>>,
    eos: u32,
}

impl Vocabulary {
    /// Build from a live tokenizer. `vocab` is the model's output width, which
    /// is >= the tokenizer's usable vocabulary (GGUF pads the rows).
    pub fn from_tokenizer(
        tokenizer: &tokenizers::Tokenizer,
        vocab: usize,
        eos: u32,
    ) -> Result<Self, String> {
        if vocab == 0 || vocab > u32::MAX as usize {
            return Err(format!("implausible vocabulary width {vocab}"));
        }
        if eos as usize >= vocab {
            return Err(format!("end-of-text id {eos} outside vocabulary {vocab}"));
        }
        let table = char_to_byte();
        let mut pairs: Vec<(u32, Vec<u8>)> = Vec::with_capacity(vocab);
        // `get_vocab(false)` is the base BPE vocabulary: added/control tokens
        // are excluded by construction, so they stay `None` below.
        for (text, id) in tokenizer.get_vocab(false) {
            if id as usize >= vocab {
                continue;
            }
            let mut expansion = Vec::with_capacity(text.len());
            let mut usable = true;
            for c in text.chars() {
                match table.get(c as usize).copied().unwrap_or(-1) {
                    -1 => {
                        usable = false;
                        break;
                    }
                    b => expansion.push(b as u8),
                }
            }
            // A zero-byte token cannot advance the document and would let the
            // decoder spin without progress; treat it as unusable.
            if usable && !expansion.is_empty() {
                pairs.push((id, expansion));
            }
        }
        if pairs.is_empty() {
            return Err("tokenizer exposes no byte-level tokens".into());
        }
        Ok(Self::assemble(pairs, vocab, eos))
    }

    /// Explicit construction, for tests and for callers that already hold byte
    /// expansions. Panics are avoided; a row outside `vocab` is an error.
    pub fn from_pairs(
        pairs: impl IntoIterator<Item = (u32, Vec<u8>)>,
        vocab: usize,
        eos: u32,
    ) -> Result<Self, String> {
        let pairs: Vec<(u32, Vec<u8>)> = pairs.into_iter().collect();
        if let Some((id, _)) = pairs.iter().find(|(id, _)| *id as usize >= vocab) {
            return Err(format!("token id {id} outside vocabulary {vocab}"));
        }
        if let Some((id, _)) = pairs.iter().find(|(_, b)| b.is_empty()) {
            return Err(format!("token id {id} has an empty byte expansion"));
        }
        if eos as usize >= vocab {
            return Err(format!("end-of-text id {eos} outside vocabulary {vocab}"));
        }
        Ok(Self::assemble(pairs, vocab, eos))
    }

    fn assemble(mut pairs: Vec<(u32, Vec<u8>)>, vocab: usize, eos: u32) -> Self {
        pairs.sort_unstable_by_key(|(id, _)| *id);
        let mut blob = Vec::with_capacity(pairs.iter().map(|(_, b)| b.len()).sum());
        let mut span = vec![(0u32, 0u32); vocab];
        let mut by_first = vec![Vec::new(); 256];
        for (id, expansion) in pairs {
            by_first[expansion[0] as usize].push(id);
            span[id as usize] = (blob.len() as u32, expansion.len() as u32);
            blob.extend_from_slice(&expansion);
        }
        Self {
            blob,
            span,
            by_first,
            eos,
        }
    }

    pub fn len(&self) -> usize {
        self.span.len()
    }
    pub fn is_empty(&self) -> bool {
        self.span.is_empty()
    }
    pub fn expansion(&self, id: u32) -> Option<&[u8]> {
        let (offset, len) = *self.span.get(id as usize)?;
        if len == 0 {
            return None;
        }
        Some(&self.blob[offset as usize..(offset + len) as usize])
    }

    /// Bytes with no single-byte token.
    ///
    /// This is the whole dead-end argument. Every node of a compiled
    /// [`Program`] lies on a path to `Done` — literal chains and tries move
    /// forward, and a string body always admits a content byte and then its
    /// closing quote — so every legally reached cursor admits at least one
    /// *byte*. If every byte also has a single-byte token, then every legally
    /// reached cursor admits at least one *token*, and a zero-token mask is
    /// impossible by construction rather than by hope. A byte-level BPE
    /// vocabulary covers all 256 by definition; one that does not cannot carry
    /// that guarantee, and we say so rather than discovering it mid-report.
    pub fn uncovered_bytes(&self) -> Vec<u8> {
        let mut missing = Vec::new();
        for byte in 0..=255u8 {
            if !self.by_first[byte as usize]
                .iter()
                .any(|id| self.span[*id as usize].1 == 1)
            {
                missing.push(byte);
            }
        }
        missing
    }
}

// ---------------------------------------------------------------------------
// The grammar: a byte-level automaton compiled from one output schema
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum Node {
    /// Branch on the next byte. One arm is a literal; several are a trie over
    /// the permitted `enum` / boolean spellings.
    Bytes(Vec<(u8, u32)>),
    /// Free-form JSON string content. `close` is the node reached by the
    /// closing quote, which is only admitted once the content is non-blank.
    StringBody { close: u32 },
    /// The document is complete. Only end-of-text may follow.
    Done,
}

/// Where the decoder is inside the document. Copy, small, and hashable so the
/// per-step token mask can be memoised on it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Cursor {
    node: u32,
    /// String content so far contains a non-whitespace character.
    nonws: bool,
    /// UTF-8 continuation bytes still expected (0..=3).
    cont: u8,
    /// Total length of the UTF-8 sequence in flight (0 when `cont == 0`).
    width: u8,
    /// 0 = normal, 1 = the previous byte was a backslash.
    escape: u8,
    /// Partially assembled scalar value; normalised to 0 whenever `cont == 0`
    /// so that otherwise-identical cursors compare equal.
    scalar: u32,
}

impl Cursor {
    fn at(node: u32) -> Self {
        Self {
            node,
            nonws: false,
            cont: 0,
            width: 0,
            escape: 0,
            scalar: 0,
        }
    }
}

/// A compiled schema: the byte-level automaton for exactly one JSON document.
#[derive(Debug)]
pub struct Program {
    nodes: Vec<Node>,
    start: u32,
    /// Field names in emission order, kept for diagnostics and tests.
    order: Vec<String>,
}

enum Segment {
    Literal(Vec<u8>),
    FreeString,
    /// Full JSON spellings, e.g. `"informative"` or `true`.
    Choice(Vec<Vec<u8>>),
}

impl Program {
    /// Compile the subset `adjudicator::validate_schema` accepts.
    ///
    /// Refuses, loudly, anything that validator accepts but this grammar
    /// cannot honour. Today that is exactly one shape: an `enum` string whose
    /// value is blank, which `validate_report` would reject however it was
    /// produced.
    pub fn compile(schema: &serde_json::Value) -> Result<Self, String> {
        let properties = schema["properties"]
            .as_object()
            .ok_or("output_schema has no properties object")?;
        let required = schema["required"]
            .as_array()
            .ok_or("output_schema has no required list")?;
        if required.is_empty() {
            return Err("output_schema requires no fields; no document is constrainable".into());
        }
        // `validate_schema` enforces these too, but this function is public and
        // must not depend on having been called after it: a property that is
        // not in `required` would be missing from every document we emit, and a
        // duplicate `required` entry would emit a key twice, both of which
        // `validate_report` rejects. Refuse rather than generate the invalid
        // document.
        if properties.len() != required.len() {
            return Err(format!(
                "output_schema has {} properties but requires {}; every property must be \
                 required exactly once",
                properties.len(),
                required.len()
            ));
        }
        let mut seen = std::collections::BTreeSet::new();
        let mut segments = Vec::new();
        let mut order = Vec::new();
        for (index, name) in required.iter().enumerate() {
            let name = name
                .as_str()
                .ok_or("required entries must be property names")?;
            if !seen.insert(name) {
                return Err(format!("required field {name} is listed more than once"));
            }
            let spec = properties
                .get(name)
                .ok_or_else(|| format!("required field {name} has no property schema"))?;
            let encoded = serde_json::to_string(name).map_err(|e| e.to_string())?;
            segments.push(Segment::Literal(
                format!("{}{encoded}: ", if index == 0 { "{" } else { ", " }).into_bytes(),
            ));
            segments.push(value_segment(name, spec)?);
            order.push(name.to_string());
        }
        segments.push(Segment::Literal(b"}".to_vec()));

        let mut nodes = vec![Node::Done];
        let mut next = 0u32;
        for segment in segments.into_iter().rev() {
            next = match segment {
                Segment::Literal(bytes) => {
                    let mut cursor = next;
                    for byte in bytes.into_iter().rev() {
                        nodes.push(Node::Bytes(vec![(byte, cursor)]));
                        cursor = nodes.len() as u32 - 1;
                    }
                    cursor
                }
                Segment::FreeString => {
                    nodes.push(Node::StringBody { close: next });
                    let body = nodes.len() as u32 - 1;
                    nodes.push(Node::Bytes(vec![(b'"', body)]));
                    nodes.len() as u32 - 1
                }
                Segment::Choice(values) => trie(&mut nodes, &values, next),
            };
        }
        Ok(Self {
            nodes,
            start: next,
            order,
        })
    }

    /// Field names in the order the grammar will emit them.
    pub fn order(&self) -> &[String] {
        &self.order
    }

    pub fn start(&self) -> Cursor {
        Cursor::at(self.start)
    }

    /// The document is complete and only end-of-text may follow.
    pub fn is_done(&self, cursor: Cursor) -> bool {
        matches!(self.nodes[cursor.node as usize], Node::Done)
    }

    /// Consume one byte. `None` means the byte cannot continue any document
    /// this schema admits.
    pub fn step(&self, mut cursor: Cursor, byte: u8) -> Option<Cursor> {
        match &self.nodes[cursor.node as usize] {
            Node::Done => None,
            Node::Bytes(arms) => arms
                .iter()
                .find(|(b, _)| *b == byte)
                .map(|(_, next)| Cursor::at(*next)),
            Node::StringBody { close } => {
                if cursor.cont > 0 {
                    if !(0x80..=0xBF).contains(&byte) {
                        return None;
                    }
                    cursor.scalar = (cursor.scalar << 6) | u32::from(byte & 0x3F);
                    cursor.cont -= 1;
                    if !reachable(cursor.scalar, cursor.cont, cursor.width) {
                        return None;
                    }
                    if cursor.cont == 0 {
                        let scalar = cursor.scalar;
                        cursor.scalar = 0;
                        cursor.width = 0;
                        let c = char::from_u32(scalar)?;
                        cursor.nonws |= !c.is_whitespace();
                    }
                    return Some(cursor);
                }
                if cursor.escape == 1 {
                    let produced = match byte {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        // `\uXXXX` is deliberately not admitted; see the module
                        // docs. Everything else is not a JSON escape at all.
                        _ => return None,
                    };
                    cursor.escape = 0;
                    cursor.nonws |= !produced.is_whitespace();
                    return Some(cursor);
                }
                match byte {
                    b'"' if cursor.nonws => Some(Cursor::at(*close)),
                    // A blank string value fails `validate_report`, so the
                    // closing quote is simply not reachable yet.
                    b'"' => None,
                    b'\\' => {
                        cursor.escape = 1;
                        Some(cursor)
                    }
                    0x00..=0x1F => None,
                    0x20..=0x7F => {
                        cursor.nonws |= !(byte as char).is_whitespace();
                        Some(cursor)
                    }
                    0xC2..=0xF4 => {
                        let (mask, cont) = match byte {
                            0xC2..=0xDF => (0x1F, 1),
                            0xE0..=0xEF => (0x0F, 2),
                            _ => (0x07, 3),
                        };
                        cursor.scalar = u32::from(byte & mask);
                        cursor.cont = cont;
                        cursor.width = cont + 1;
                        // Overlong forms, surrogates and scalars past U+10FFFF
                        // are all pruned here, at the lead byte, so no token
                        // can steer the decoder into a continuation state that
                        // has no legal continuation. That is what keeps the
                        // "every reachable cursor admits a byte" argument in
                        // `Vocabulary::uncovered_bytes` true.
                        reachable(cursor.scalar, cursor.cont, cursor.width).then_some(cursor)
                    }
                    // 0x80..=0xC1 is a stray continuation byte or an overlong
                    // two-byte form; 0xF5.. is out of Unicode range.
                    _ => None,
                }
            }
        }
    }

    /// Consume a token's whole byte expansion.
    pub fn walk(&self, cursor: Cursor, bytes: &[u8]) -> Option<Cursor> {
        bytes.iter().try_fold(cursor, |c, &b| self.step(c, b))
    }
}

/// Can the partially assembled scalar still become a legal character?
///
/// `scalar` holds the bits decoded so far, `cont` continuation bytes remain,
/// and `width` is the total sequence length. The reachable final scalars are
/// the contiguous range `[scalar << 6*cont, .. | 6*cont ones]`; the sequence is
/// legal only if that range meets a scalar which is in range, not a surrogate,
/// and not an overlong encoding of a shorter sequence.
fn reachable(scalar: u32, cont: u8, width: u8) -> bool {
    let shift = 6 * u32::from(cont);
    let low = scalar << shift;
    let high = low | ((1u32 << shift) - 1);
    let floor = match width {
        2 => 0x80,
        3 => 0x800,
        _ => 0x10000,
    };
    let first = low.max(floor);
    let last = high.min(0x10FFFF);
    first <= last && !(first >= 0xD800 && last <= 0xDFFF)
}

fn value_segment(name: &str, spec: &serde_json::Value) -> Result<Segment, String> {
    let string = spec["type"] == "string";
    if !string && spec["type"] != "boolean" {
        return Err(format!("field {name} is neither a string nor a boolean"));
    }
    let Some(choices) = spec.get("enum") else {
        return Ok(if string {
            Segment::FreeString
        } else {
            Segment::Choice(vec![b"true".to_vec(), b"false".to_vec()])
        });
    };
    let choices = choices
        .as_array()
        .ok_or_else(|| format!("field {name} has a non-array enum"))?;
    let mut spellings = Vec::new();
    for choice in choices {
        if string {
            let text = choice
                .as_str()
                .ok_or_else(|| format!("field {name} has a non-string enum value"))?;
            // `validate_schema` accepts this; `validate_report` rejects every
            // document containing it. Refusing is the only honest answer —
            // constraining to it would guarantee an invalid report, and
            // dropping it silently would narrow the caller's schema behind
            // their back.
            if text.trim().is_empty() {
                return Err(format!(
                    "field {name} has a blank enum value, which validate_report rejects; \
                     no document satisfying this schema exists"
                ));
            }
        } else if !choice.is_boolean() {
            return Err(format!("field {name} has a non-boolean enum value"));
        }
        let spelling = serde_json::to_string(choice)
            .map_err(|e| e.to_string())?
            .into_bytes();
        if !spellings.contains(&spelling) {
            spellings.push(spelling);
        }
    }
    if spellings.is_empty() {
        return Err(format!("field {name} has an empty enum"));
    }
    Ok(Segment::Choice(spellings))
}

/// Build a trie over `values` whose accepting edges all land on `exit`.
fn trie(nodes: &mut Vec<Node>, values: &[Vec<u8>], exit: u32) -> u32 {
    let mut index: BTreeMap<Vec<u8>, u32> = BTreeMap::new();
    for value in values {
        for len in 0..value.len() {
            let prefix = value[..len].to_vec();
            index.entry(prefix).or_insert_with(|| {
                nodes.push(Node::Bytes(Vec::new()));
                nodes.len() as u32 - 1
            });
        }
    }
    for value in values {
        for len in 0..value.len() {
            let here = index[&value[..len]];
            let target = if len + 1 == value.len() {
                exit
            } else {
                index[&value[..len + 1]]
            };
            if let Node::Bytes(arms) = &mut nodes[here as usize]
                && !arms.iter().any(|(b, _)| *b == value[len])
            {
                arms.push((value[len], target));
            }
        }
    }
    index[&Vec::new()]
}

// ---------------------------------------------------------------------------
// Masking
// ---------------------------------------------------------------------------

/// A per-cursor mask, stored as whichever side is smaller.
#[derive(Debug)]
struct Plan {
    fill: f32,
    exception: f32,
    exceptions: Vec<u32>,
    allowed: usize,
}

/// A compiled schema, the vocabulary it masks, and every mask built so far.
/// Built ONCE, at load: a schema this vocabulary cannot honour stops the daemon
/// there instead of failing the first request, and no request pays to compile
/// it again. Shared by every generation; a mask depends only on the cursor, so
/// the scaffold states one report paid for are free for every report after it.
pub struct Grammar {
    program: Program,
    vocabulary: Arc<Vocabulary>,
    plans: Mutex<HashMap<Cursor, Arc<Plan>>>,
}

impl Grammar {
    /// Compile `schema` against a real tokenizer. Refuses a vocabulary with no
    /// single-byte token for some byte value: without full coverage a legal
    /// continuation is not guaranteed to exist at every step.
    pub fn compile(
        schema: &serde_json::Value,
        tokenizer: &tokenizers::Tokenizer,
        vocab: usize,
        eos: u32,
    ) -> Result<Arc<Self>, String> {
        let program = Program::compile(schema)?;
        let vocabulary = Arc::new(Vocabulary::from_tokenizer(tokenizer, vocab, eos)?);
        let missing = vocabulary.uncovered_bytes();
        if !missing.is_empty() {
            return Err(format!(
                "tokenizer has no single-byte token for {} byte value(s) (first: {:?}); \
                 constrained decoding cannot guarantee a legal continuation exists at \
                 every step with this vocabulary",
                missing.len(),
                &missing[..missing.len().min(8)]
            ));
        }
        Self::from_parts(program, vocabulary)
    }

    /// A grammar over an explicit vocabulary. Proves a legal first token
    /// exists before anything spends a forward pass on a grammar that cannot
    /// start.
    pub fn from_parts(program: Program, vocabulary: Arc<Vocabulary>) -> Result<Arc<Self>, String> {
        let grammar = Self::unproven(program, vocabulary);
        grammar.plan(grammar.program.start())?;
        Ok(grammar)
    }

    fn unproven(program: Program, vocabulary: Arc<Vocabulary>) -> Arc<Self> {
        Arc::new(Self { program, vocabulary, plans: Mutex::new(HashMap::new()) })
    }

    pub fn program(&self) -> &Program {
        &self.program
    }
    pub fn vocabulary(&self) -> &Arc<Vocabulary> {
        &self.vocabulary
    }
    /// How many cursors have a built mask — diagnostics and tests.
    pub fn cached_plans(&self) -> usize {
        self.plans.lock().expect("plan cache poisoned").len()
    }

    /// Token ids the grammar admits at `cursor`, end-of-text included.
    pub fn allowed(&self, cursor: Cursor) -> Vec<u32> {
        let mut allowed = Vec::new();
        if self.program.is_done(cursor) {
            allowed.push(self.vocabulary.eos);
            return allowed;
        }
        for first in 0..=255u8 {
            let Some(after) = self.program.step(cursor, first) else {
                continue;
            };
            for &id in &self.vocabulary.by_first[first as usize] {
                let (offset, len) = self.vocabulary.span[id as usize];
                let rest = &self.vocabulary.blob
                    [(offset + 1) as usize..(offset + len) as usize];
                if self.program.walk(after, rest).is_some() {
                    allowed.push(id);
                }
            }
        }
        allowed
    }

    fn plan(&self, cursor: Cursor) -> Result<Arc<Plan>, String> {
        if let Some(plan) = self.plans.lock().expect("plan cache poisoned").get(&cursor) {
            return Ok(plan.clone());
        }
        let allowed = self.allowed(cursor);
        if allowed.is_empty() {
            return Err(format!(
                "constrained decoding left no legal token at {cursor:?}; \
                 the grammar cannot be satisfied and generation must not continue"
            ));
        }
        let width = self.vocabulary.len();
        let plan = if allowed.len() * 2 > width {
            let mut permitted = vec![false; width];
            for id in &allowed {
                permitted[*id as usize] = true;
            }
            Arc::new(Plan {
                fill: 0.,
                exception: BLOCK,
                exceptions: (0..width as u32)
                    .filter(|id| !permitted[*id as usize])
                    .collect(),
                allowed: allowed.len(),
            })
        } else {
            Arc::new(Plan {
                fill: BLOCK,
                exception: 0.,
                allowed: allowed.len(),
                exceptions: allowed,
            })
        };
        self.plans.lock().expect("plan cache poisoned").insert(cursor, plan.clone());
        Ok(plan)
    }
}

/// One generation's position in a [`Grammar`].
pub struct Masker {
    grammar: Arc<Grammar>,
    cursor: Cursor,
    buffer: Vec<f32>,
}

impl Masker {
    /// A generation over a shared, already-proven grammar.
    pub fn over(grammar: Arc<Grammar>) -> Self {
        let cursor = grammar.program.start();
        let buffer = vec![BLOCK; grammar.vocabulary.len()];
        Self { grammar, cursor, buffer }
    }

    /// A generation over a private grammar, unproven: the first
    /// [`Self::mask`] or [`Self::admissible`] reports a grammar that cannot
    /// start.
    pub fn new(program: Program, vocabulary: Arc<Vocabulary>) -> Self {
        Self::over(Grammar::unproven(program, vocabulary))
    }

    pub fn cursor(&self) -> Cursor {
        self.cursor
    }
    pub fn program(&self) -> &Program {
        &self.grammar.program
    }
    pub fn is_done(&self) -> bool {
        self.grammar.program.is_done(self.cursor)
    }

    /// Token ids the grammar admits at `cursor`, end-of-text included.
    pub fn allowed(&self, cursor: Cursor) -> Vec<u32> {
        self.grammar.allowed(cursor)
    }

    /// Additive logit mask for the current cursor: `0.` for admissible tokens,
    /// [`BLOCK`] for the rest.
    pub fn mask(&mut self, device: &Device) -> candle_core::Result<Tensor> {
        let cursor = self.cursor;
        let plan = self.grammar.plan(cursor).map_err(candle_core::Error::msg)?;
        self.buffer.fill(plan.fill);
        for &id in &plan.exceptions {
            self.buffer[id as usize] = plan.exception;
        }
        Tensor::new(self.buffer.as_slice(), device)
    }

    /// Advance over a selected token. An inadmissible token is an error, not a
    /// recoverable state: it means the mask and the automaton disagree.
    pub fn accept(&mut self, token: u32) -> Result<(), String> {
        if token == self.grammar.vocabulary.eos {
            return if self.grammar.program.is_done(self.cursor) {
                Ok(())
            } else {
                Err(format!(
                    "constrained decoding selected end-of-text at {:?}, before the report closed",
                    self.cursor
                ))
            };
        }
        let bytes = self
            .grammar
            .vocabulary
            .expansion(token)
            .ok_or_else(|| format!("constrained decoding selected unusable token {token}"))?
            .to_vec();
        self.cursor = self.grammar.program.walk(self.cursor, &bytes).ok_or_else(|| {
            format!(
                "constrained decoding selected token {token} that cannot continue the report at {:?}",
                self.cursor
            )
        })?;
        Ok(())
    }

    /// One flag per vocabulary row: does the current cursor admit it. The
    /// host-side twin of [`Self::mask`], for recording what the grammar allowed.
    pub fn permitted(&mut self) -> Result<Vec<bool>, String> {
        let cursor = self.cursor;
        let plan = self.grammar.plan(cursor)?;
        let listed_are_legal = plan.exception == 0.;
        let mut permitted = vec![!listed_are_legal; self.grammar.vocabulary.len()];
        for &id in &plan.exceptions {
            permitted[id as usize] = listed_are_legal;
        }
        Ok(permitted)
    }

    /// How many tokens the current cursor admits — diagnostics and tests.
    pub fn admissible(&mut self) -> Result<usize, String> {
        let cursor = self.cursor;
        Ok(self.grammar.plan(cursor)?.allowed)
    }
}

// ---------------------------------------------------------------------------
// The sampler seam
// ---------------------------------------------------------------------------

/// Drop-in replacement for [`GreedySampler`] that masks the logits against a
/// compiled output schema before every selection. Without a schema it *is* a
/// `GreedySampler`, byte for byte.
pub enum Decoder {
    Free(GreedySampler),
    Json {
        sampler: GreedySampler,
        masker: Box<Masker>,
    },
}

impl Decoder {
    /// `schema` is the adjudicator's `output_schema`. `None` keeps the previous
    /// free-generation behaviour; `Some` compiles a grammar and fails loudly if
    /// it cannot.
    pub fn new(
        grammar: Option<&Arc<Grammar>>,
        device: &Device,
        vocab: usize,
        history: &[u32],
        penalty: f32,
    ) -> candle_core::Result<Self> {
        let sampler = GreedySampler::new(device, vocab, history, penalty)?;
        let Some(grammar) = grammar else {
            return Ok(Self::Free(sampler));
        };
        if grammar.vocabulary.len() != vocab {
            return Err(candle_core::Error::msg(format!(
                "grammar covers {} vocabulary rows, the model has {vocab}",
                grammar.vocabulary.len()
            )));
        }
        Ok(Self::Json { sampler, masker: Box::new(Masker::over(grammar.clone())) })
    }

    /// Build a decoder over an explicit vocabulary. Used by tests and by
    /// callers that already own byte expansions.
    pub fn with_vocabulary(
        schema: &serde_json::Value,
        vocabulary: Arc<Vocabulary>,
        device: &Device,
        history: &[u32],
        penalty: f32,
    ) -> candle_core::Result<Self> {
        let program = Program::compile(schema).map_err(candle_core::Error::msg)?;
        let width = vocabulary.len();
        let grammar = Grammar::from_parts(program, vocabulary).map_err(candle_core::Error::msg)?;
        Self::new(Some(&grammar), device, width, history, penalty)
    }

    /// Select one token. Same signature and same contract as
    /// `GreedySampler::sample`, so the decode loop does not change.
    ///
    /// **`logits` is left untouched.** The mask is combined into a new tensor
    /// and only that is sampled, so a caller holding `logits` still holds the
    /// raw model distribution — which is what per-token logprobs and
    /// "mass inside a named set" have to be read from. Reading them after an
    /// in-place mask would make mass-in-set 1.0 by construction and quietly
    /// destroy the "low mass means *unasked*" signal.
    pub fn sample(&mut self, logits: &Tensor) -> candle_core::Result<u32> {
        match self {
            Self::Free(sampler) => sampler.sample(logits),
            Self::Json { sampler, masker } => {
                let mask = masker.mask(logits.device())?;
                let masked = logits.broadcast_add(&mask)?.contiguous()?;
                let token = sampler.sample(&masked)?;
                masker.accept(token).map_err(candle_core::Error::msg)?;
                Ok(token)
            }
        }
    }

    /// Select one token and, when asked, record the distribution it was
    /// selected from. The decode loop calls this and nothing else, so the
    /// record is read from the RAW `logits` by construction: the mask exists
    /// only inside [`Self::sample`], and reaches the record only as the set of
    /// rows it admitted, captured before the cursor advances.
    pub fn step(
        &mut self,
        logits: &Tensor,
        spec: Option<&crate::types::DistributionRequest>,
        token_text: impl Fn(u32) -> Option<String>,
    ) -> candle_core::Result<(u32, Option<crate::types::StepDistribution>)> {
        let permitted = match (spec, &mut *self) {
            (Some(spec), Self::Json { masker, .. }) if spec.constrained => {
                Some(masker.permitted().map_err(candle_core::Error::msg)?)
            }
            (Some(spec), Self::Free(_)) if spec.constrained => {
                return Err(candle_core::Error::msg(
                    "distributions.constrained needs an output grammar; this decoder has none",
                ));
            }
            _ => None,
        };
        let token = self.sample(logits)?;
        let Some(spec) = spec else {
            return Ok((token, None));
        };
        let raw: Vec<f32> = logits.flatten_all()?.to_vec1()?;
        let step = crate::types::step_distribution_under(
            &raw,
            token,
            spec.top_k,
            &spec.token_sets,
            permitted.as_deref(),
            token_text,
        )
        .map_err(candle_core::Error::msg)?;
        Ok((token, Some(step)))
    }

    pub fn masker(&self) -> Option<&Masker> {
        match self {
            Self::Free(_) => None,
            Self::Json { masker, .. } => Some(masker),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn schema() -> serde_json::Value {
        json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                // deliberately NOT in `required` order
                "reason": {"type": "string"},
                "severity": {"type": "string", "enum": ["informative", "data-critical"]},
                "writes": {"type": "boolean"}
            },
            "required": ["severity", "writes", "reason"]
        })
    }

    fn accepts(program: &Program, text: &str) -> bool {
        program
            .walk(program.start(), text.as_bytes())
            .is_some_and(|c| program.is_done(c))
    }

    #[test]
    fn byte_alphabet_is_a_bijection_over_all_256_bytes() {
        let table = char_to_byte();
        let mut seen = vec![false; 256];
        for entry in &table {
            if *entry >= 0 {
                assert!(!seen[*entry as usize], "byte {entry} mapped twice");
                seen[*entry as usize] = true;
            }
        }
        assert!(seen.iter().all(|s| *s), "byte-level alphabet is not total");
        // Spot-check the three anchors of the GPT-2 table.
        assert_eq!(table['!' as usize], b'!' as i16);
        assert_eq!(table['Ġ' as usize], b' ' as i16);
        assert_eq!(table['Ċ' as usize], b'\n' as i16);
    }

    #[test]
    fn emits_required_order_and_refuses_properties_order() {
        let program = Program::compile(&schema()).unwrap();
        assert_eq!(program.order(), ["severity", "writes", "reason"]);
        assert!(accepts(
            &program,
            r#"{"severity": "informative", "writes": false, "reason": "read-only"}"#
        ));
        assert!(!accepts(
            &program,
            r#"{"reason": "read-only", "severity": "informative", "writes": false}"#
        ));
    }

    #[test]
    fn enum_field_admits_only_listed_values() {
        let program = Program::compile(&schema()).unwrap();
        for value in ["informative", "data-critical"] {
            assert!(
                program
                    .walk(program.start(), format!(r#"{{"severity": "{value}""#).as_bytes())
                    .is_some(),
                "{value} should be admissible"
            );
        }
        for value in ["catastrophic", "informativ", "INFORMATIVE", ""] {
            assert!(
                program
                    .walk(program.start(), format!(r#"{{"severity": "{value}""#).as_bytes())
                    .is_none(),
                "{value} must not be admissible"
            );
        }
    }

    #[test]
    fn boolean_field_admits_only_true_and_false() {
        let program = Program::compile(&schema()).unwrap();
        let head = r#"{"severity": "informative", "writes": "#;
        assert!(program.walk(program.start(), format!("{head}true").as_bytes()).is_some());
        assert!(program.walk(program.start(), format!("{head}false").as_bytes()).is_some());
        for bad in ["True", "1", "\"true\"", "null", "yes"] {
            assert!(
                program
                    .walk(program.start(), format!("{head}{bad}").as_bytes())
                    .is_none(),
                "{bad} must not be admissible for a boolean"
            );
        }
    }

    #[test]
    fn free_string_cannot_close_while_blank() {
        let program = Program::compile(&schema()).unwrap();
        let head = r#"{"severity": "informative", "writes": true, "reason": ""#;
        let open = program.walk(program.start(), head.as_bytes()).unwrap();
        // Empty, ASCII-blank, and a Unicode blank (U+00A0) all fail `trim()`.
        assert!(program.step(open, b'"').is_none());
        let spaces = program.walk(open, b"   ").unwrap();
        assert!(program.step(spaces, b'"').is_none());
        // A raw tab is a JSON control character; only the escape is admitted.
        assert!(program.step(spaces, b'\t').is_none());
        let escaped = program.walk(spaces, br"\t\r").unwrap();
        assert!(program.step(escaped, b'"').is_none());
        let nbsp = program.walk(open, "\u{a0}\u{2003}".as_bytes()).unwrap();
        assert!(program.step(nbsp, b'"').is_none());
        let real = program.walk(open, "  \u{a0}x".as_bytes()).unwrap();
        assert!(program.step(real, b'"').is_some());
    }

    #[test]
    fn whitespace_produced_by_an_escape_does_not_count_as_content() {
        let program = Program::compile(&schema()).unwrap();
        let head = r#"{"severity": "informative", "writes": true, "reason": ""#;
        let open = program.walk(program.start(), head.as_bytes()).unwrap();
        let newline = program.walk(open, br"\n\t").unwrap();
        assert!(program.step(newline, b'"').is_none());
        // \b is a control character, not whitespace, so it is content.
        let backspace = program.walk(open, br"\b").unwrap();
        assert!(program.step(backspace, b'"').is_some());
    }

    #[test]
    fn string_rejects_raw_controls_bad_utf8_and_u_escapes() {
        let program = Program::compile(&schema()).unwrap();
        let head = r#"{"severity": "informative", "writes": true, "reason": "x"#;
        let open = program.walk(program.start(), head.as_bytes()).unwrap();
        assert!(program.step(open, 0x0A).is_none(), "raw newline");
        assert!(program.step(open, 0x00).is_none(), "raw NUL");
        assert!(program.walk(open, br"\u0041").is_none(), "\\u escape");
        assert!(program.walk(open, br"\q").is_none(), "bogus escape");
        assert!(program.walk(open, &[0x80]).is_none(), "stray continuation");
        assert!(program.walk(open, &[0xC0, 0x80]).is_none(), "overlong");
        assert!(program.walk(open, &[0xED, 0xA0, 0x80]).is_none(), "surrogate");
        // A real multi-byte character, split the way a tokenizer might.
        let half = program.walk(open, &[0xE6, 0x97]).unwrap();
        assert!(program.step(half, b'"').is_none(), "closed mid-character");
        assert!(program.step(half, 0xA5).is_some());
    }

    #[test]
    fn utf8_leads_that_cannot_finish_are_pruned_at_the_lead_byte() {
        // Found by the random-logit property test: admitting a lead byte whose
        // continuation range cannot land on a legal scalar strands the decoder
        // in a state with no legal next byte at all.
        let program = Program::compile(&schema()).unwrap();
        let head = r#"{"severity": "informative", "writes": true, "reason": "x"#;
        let open = program.walk(program.start(), head.as_bytes()).unwrap();
        // Every admissible partial sequence must still have a legal next byte.
        let mut frontier = vec![open];
        for _ in 0..4 {
            let mut next = Vec::new();
            for cursor in frontier.drain(..) {
                let legal: Vec<u8> = (0..=255u8)
                    .filter(|b| program.step(cursor, *b).is_some())
                    .collect();
                assert!(!legal.is_empty(), "stranded at {cursor:?}");
                for byte in legal {
                    if let Some(after) = program.step(cursor, byte)
                        && after.cont > 0
                    {
                        next.push(after);
                    }
                }
            }
            frontier = next;
        }
        // Specific shapes, spelled out.
        assert!(program.walk(open, &[0xF4, 0x90]).is_none(), "past U+10FFFF");
        assert!(program.walk(open, &[0xF4, 0x8F, 0xBF, 0xBF]).is_some(), "U+10FFFF");
        assert!(program.walk(open, &[0xE0, 0x80]).is_none(), "overlong three-byte");
        assert!(program.walk(open, &[0xED, 0xA0]).is_none(), "surrogate lead");
        assert!(program.walk(open, &[0xED, 0x9F, 0xBF]).is_some(), "U+D7FF");
    }

    #[test]
    fn end_of_text_is_reachable_only_after_the_closing_brace() {
        let program = Program::compile(&schema()).unwrap();
        let doc = r#"{"severity": "data-critical", "writes": true, "reason": "rm -rf"}"#;
        let end = program.walk(program.start(), doc.as_bytes()).unwrap();
        assert!(program.is_done(end));
        assert!(program.step(end, b' ').is_none(), "nothing follows the report");
        let short = program
            .walk(program.start(), &doc.as_bytes()[..doc.len() - 1])
            .unwrap();
        assert!(!program.is_done(short));
    }

    #[test]
    fn compile_stands_on_its_own_and_does_not_assume_validate_schema_ran() {
        // A property not in `required` would be absent from every document we
        // emit; a duplicated `required` entry would emit a key twice. Both are
        // rejected by `validate_report`, so both are refused here.
        let extra = json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
            "required": ["a"]
        });
        assert!(Program::compile(&extra).unwrap_err().contains("required exactly once"));
        let twice = json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
            "required": ["a", "a"]
        });
        assert!(Program::compile(&twice).unwrap_err().contains("more than once"));
        let missing = json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {"a": {"type": "string"}},
            "required": ["z"]
        });
        assert!(Program::compile(&missing).unwrap_err().contains("no property schema"));
    }

    #[test]
    fn blank_enum_value_is_a_loud_error_not_free_generation() {
        // `adjudicator::validate_schema` accepts this; `validate_report`
        // rejects every document it can produce.
        let bad = json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {"severity": {"type": "string", "enum": ["  "]}},
            "required": ["severity"]
        });
        assert!(crate::adjudicator::validate_report("{}", &bad, "stop").is_err());
        let error = Program::compile(&bad).unwrap_err();
        assert!(error.contains("blank enum value"), "{error}");
    }

    // --- masking -----------------------------------------------------------

    /// A toy byte-level vocabulary: multi-byte pieces on purpose, including
    /// ones that straddle grammar segment boundaries.
    fn toy() -> (Arc<Vocabulary>, Vec<&'static str>) {
        let pieces = vec![
            "{", "}", "\"", ":", ",", "severity", "writes", "reason",
            "informative", "data-critical", "true", "false",
            // boundary-straddling pieces, spaced: one space after `:` and
            // after `,`, never before `}`. `"\":"` and `": \""` both start
            // from the position right before a key's closing quote; `": "`
            // is the boolean-value shape (no opening value quote to fold
            // in); `" \""` is the model's own token for opening a string
            // value, and the one the compact-separator bug rejected.
            "{\"", "\":", "\", \"", "\"}", "\": \"", "\": ", ": \"", " \"", ", \"",
            // content and traps
            "rm", " -rf", "x", " ", "null", "informativ",
        ];
        let pairs = pieces
            .iter()
            .enumerate()
            .map(|(i, p)| (i as u32, p.as_bytes().to_vec()));
        // one row past the pieces is end-of-text
        let eos = pieces.len() as u32;
        (
            Arc::new(Vocabulary::from_pairs(pairs, pieces.len() + 1, eos).unwrap()),
            pieces,
        )
    }

    #[test]
    fn a_colon_cannot_smuggle_in_as_the_strings_opening_byte() {
        // Regression for the bug that ended the compact-separator ruling: at
        // the cursor right after a key's closing quote and colon, the
        // model's own top token was ` "` (space, quote). Under compact
        // separators that position expected a quote immediately, so `":`
        // (quote, colon) was legal there too — its quote opened the string
        // and its colon became the string's first CONTENT byte, so every
        // free-text field came out as the literal value `": `. With the
        // fixed space in place, only the space can follow the colon; `":`
        // must be rejected outright, not merely disfavoured.
        let program = Program::compile(&schema()).unwrap();
        let (vocabulary, pieces) = toy();
        let masker = Masker::new(Program::compile(&schema()).unwrap(), vocabulary);
        // Walk to just past `"reason":` — the key and its colon are emitted,
        // the separating space is not.
        let prefix = r#"{"severity": "informative", "writes": true, "reason":"#;
        let cursor = program.walk(program.start(), prefix.as_bytes()).unwrap();
        let space_quote = pieces.iter().position(|p| *p == " \"").unwrap() as u32;
        let colon_quote = pieces.iter().position(|p| *p == "\":").unwrap() as u32;
        let allowed = masker.allowed(cursor);
        assert!(
            allowed.contains(&space_quote),
            "the model's own token ` \"` must be legal here"
        );
        assert!(
            !allowed.contains(&colon_quote),
            "`\":` must not be legal here; its colon would land as the \
             string's first content byte"
        );
    }

    #[test]
    fn mask_admits_exactly_the_tokens_that_continue_the_document() {
        let (vocabulary, pieces) = toy();
        let mut masker = Masker::new(Program::compile(&schema()).unwrap(), vocabulary);
        let names = |ids: Vec<u32>| {
            let mut out: Vec<&str> = ids
                .iter()
                .map(|id| pieces.get(*id as usize).copied().unwrap_or("<eos>"))
                .collect();
            out.sort_unstable();
            out
        };
        // At the start only `{` and the `{"` piece can begin the document.
        assert_eq!(names(masker.allowed(masker.cursor())), ["{", "{\""]);
        masker.accept(0).unwrap(); // "{"
        // `":` cannot follow `{` even though it starts with a quote: the key
        // name has to come first. This is the boundary tokens straddle.
        assert_eq!(names(masker.allowed(masker.cursor())), ["\""]);
        masker.accept(2).unwrap(); // "\""
        // Wrong key is unreachable even though `writes` is a vocabulary entry.
        assert_eq!(names(masker.allowed(masker.cursor())), ["severity"]);
    }

    #[test]
    fn a_legal_document_always_has_a_legal_next_token() {
        let (vocabulary, pieces) = toy();
        let eos = pieces.len() as u32;
        let mut masker = Masker::new(Program::compile(&schema()).unwrap(), vocabulary);
        let id = |p: &str| pieces.iter().position(|q| *q == p).unwrap() as u32;
        // One tokenization of a complete report, chosen so that several pieces
        // straddle a grammar segment boundary (`{"`, `": "`, `", "`, `"}`).
        let plan = [
            "{\"", "severity", "\": \"", "data-critical", "\", \"", "writes", "\": ",
            "true", ", \"", "reason", "\": \"", "rm", " -rf", "\"}",
        ];
        for piece in plan {
            let admissible = masker.allowed(masker.cursor());
            assert!(!admissible.is_empty(), "no legal token before {piece:?}");
            assert!(
                admissible.contains(&id(piece)),
                "{piece:?} was masked out at {:?}",
                masker.cursor()
            );
            assert!(!masker.is_done(), "document closed early before {piece:?}");
            masker.accept(id(piece)).unwrap();
        }
        assert!(masker.is_done());
        assert_eq!(masker.allowed(masker.cursor()), vec![eos]);
        masker.accept(eos).unwrap();
    }

    #[test]
    fn accept_rejects_a_token_the_mask_would_have_blocked() {
        let (vocabulary, pieces) = toy();
        let mut masker = Masker::new(Program::compile(&schema()).unwrap(), vocabulary);
        let writes = pieces.iter().position(|p| *p == "writes").unwrap() as u32;
        masker.accept(0).unwrap(); // "{"
        masker.accept(2).unwrap(); // "\""
        let error = masker.accept(writes).unwrap_err();
        assert!(error.contains("cannot continue the report"), "{error}");
    }

    #[test]
    fn end_of_text_before_the_report_closes_is_an_error() {
        let (vocabulary, pieces) = toy();
        let eos = pieces.len() as u32;
        let mut masker = Masker::new(Program::compile(&schema()).unwrap(), vocabulary);
        let error = masker.accept(eos).unwrap_err();
        assert!(error.contains("before the report closed"), "{error}");
    }

    #[test]
    fn a_vocabulary_that_cannot_start_the_document_fails_loudly() {
        // No token begins with `{`, so the very first mask is empty.
        let vocabulary = Arc::new(
            Vocabulary::from_pairs([(0u32, b"a".to_vec()), (1, b"b".to_vec())], 3, 2).unwrap(),
        );
        let mut masker = Masker::new(Program::compile(&schema()).unwrap(), vocabulary);
        let error = masker.admissible().unwrap_err();
        assert!(error.contains("no legal token"), "{error}");
    }

    #[test]
    fn a_grammar_that_cannot_start_is_refused_when_it_is_built() {
        // The daemon builds its grammar once, at load. A schema this
        // vocabulary cannot even open must stop the daemon there, not turn
        // into a 500 on the first request.
        let vocabulary = Arc::new(
            Vocabulary::from_pairs([(0u32, b"a".to_vec()), (1, b"b".to_vec())], 3, 2).unwrap(),
        );
        let error = Grammar::from_parts(Program::compile(&schema()).unwrap(), vocabulary)
            .err()
            .expect("no token begins with `{`");
        assert!(error.contains("no legal token"), "{error}");
    }

    #[test]
    fn plans_built_by_one_generation_are_reused_by_the_next() {
        let (vocabulary, pieces) = toy();
        let grammar =
            Grammar::from_parts(Program::compile(&schema()).unwrap(), vocabulary).unwrap();
        let document = [
            "{\"", "severity", "\": \"", "informative", "\", \"", "writes", "\": ", "true",
            ", \"", "reason", "\": \"", "x", "\"}",
        ];
        let walk = |grammar: &Arc<Grammar>| -> Vec<Vec<f32>> {
            let mut masker = Masker::over(grammar.clone());
            let mut masks = Vec::new();
            for piece in document {
                masks.push(masker.mask(&Device::Cpu).unwrap().to_vec1().unwrap());
                let id = pieces.iter().position(|p| *p == piece).unwrap() as u32;
                masker.accept(id).unwrap();
            }
            masks
        };
        let first = walk(&grammar);
        let built = grammar.cached_plans();
        assert!(built >= document.len() / 2, "the first walk must have built plans: {built}");
        let second = walk(&grammar);
        assert_eq!(grammar.cached_plans(), built, "the second generation built nothing new");
        assert_eq!(first, second, "a cached plan is the same mask");
        // Two generations in flight share the cache and not the cursor.
        let (mut a, b) = (Masker::over(grammar.clone()), Masker::over(grammar.clone()));
        a.accept(pieces.iter().position(|p| *p == "{\"").unwrap() as u32).unwrap();
        assert_ne!(a.cursor(), b.cursor());
        assert_eq!(b.cursor(), grammar.program().start());
    }

    #[test]
    fn mask_tensor_blocks_everything_the_grammar_rejects() {
        let (vocabulary, pieces) = toy();
        let width = pieces.len() + 1;
        let mut masker = Masker::new(Program::compile(&schema()).unwrap(), vocabulary);
        let allowed = masker.allowed(masker.cursor());
        let mask = masker.mask(&Device::Cpu).unwrap();
        let values: Vec<f32> = mask.to_vec1().unwrap();
        assert_eq!(values.len(), width);
        for (id, value) in values.iter().enumerate() {
            let permitted = allowed.contains(&(id as u32));
            assert_eq!(
                *value == 0.,
                permitted,
                "token {id} ({:?}) mask {value}",
                pieces.get(id)
            );
            assert!(value.is_finite(), "mask must stay finite for GreedySampler");
        }
    }

    #[test]
    fn permitted_is_the_mask_in_both_plan_encodings_and_at_the_end() {
        // The record's legal set and the sampler's mask must be one predicate.
        // Walk a whole document so both encodings occur: few legal rows (a key,
        // an enum) and most rows legal (inside a free string), then `Done`.
        let (vocabulary, pieces) = toy();
        let eos = pieces.len() as u32;
        let mut masker = Masker::new(Program::compile(&schema()).unwrap(), vocabulary.clone());
        let document = [
            "{\"", "severity", "\": \"", "informative", "\", \"", "writes", "\": ", "true",
            ", \"", "reason", "\": \"", "x", "\"}",
        ];
        let (mut sparse, mut dense) = (0, 0);
        let mut check = |masker: &mut Masker| {
            let values: Vec<f32> = masker.mask(&Device::Cpu).unwrap().to_vec1().unwrap();
            let permitted = masker.permitted().unwrap();
            let legal = permitted.iter().filter(|p| **p).count();
            assert_eq!(legal, masker.admissible().unwrap());
            for (id, value) in values.iter().enumerate() {
                assert_eq!(*value == 0., permitted[id], "row {id} at {:?}", masker.cursor());
            }
            if legal * 2 > permitted.len() { dense += 1 } else { sparse += 1 }
            permitted
        };
        for piece in document {
            check(&mut masker);
            let id = pieces.iter().position(|p| *p == piece).unwrap() as u32;
            masker.accept(id).unwrap();
        }
        let at_end = check(&mut masker);
        assert!(at_end[eos as usize] && at_end.iter().filter(|p| **p).count() == 1);
        assert!(sparse > 0 && dense > 0, "both encodings must occur: {sparse} sparse, {dense} dense");
    }

    #[test]
    fn sampling_under_a_mask_leaves_the_caller_s_logits_untouched() {
        // MERGE CONTRACT, and the reason this test exists.
        //
        // `adjudicator.rs` reads the full distribution out of `logits` straight
        // after `Decoder::sample(&logits)`, to report per-token logprobs and the
        // raw mass held by a named token set. That mass is only meaningful
        // against the FULL-vocabulary denominator: decision 5 of
        // docs/field-requests.md treats near-zero in-set mass as "the model was
        // never asked this", which is what distinguishes a real answer from four
        // renormalised near-zero tails.
        //
        // If sampling ever masks in place, that reader silently sees the
        // post-mask distribution instead. Mass-in-set then reads ~1.0 by
        // construction and the signal dies with no error anywhere. Each branch
        // passed its own tests; only the merge can express this defect, so only
        // an integrated test can catch it.
        let (vocabulary, pieces) = toy();
        let width = pieces.len() + 1;
        let mut masker = Masker::new(Program::compile(&schema()).unwrap(), vocabulary);
        let allowed = masker.allowed(masker.cursor());
        let blocked = (0..width as u32)
            .find(|id| !allowed.contains(id))
            .expect("the toy vocabulary must contain a token the grammar rejects here");

        // Give the blocked token the winning logit, so a mask that mutated in
        // place would be unmistakable in the values afterwards.
        let mut raw = vec![0.5f32; width];
        raw[blocked as usize] = 99.0;
        let logits = Tensor::new(raw.as_slice(), &Device::Cpu).unwrap();
        let before: Vec<f32> = logits.to_vec1().unwrap();

        let mask = masker.mask(&Device::Cpu).unwrap();
        let chosen = {
            let masked = logits.broadcast_add(&mask).unwrap();
            let values: Vec<f32> = masked.to_vec1().unwrap();
            let mut best = 0usize;
            for (i, v) in values.iter().enumerate() {
                if v > &values[best] {
                    best = i;
                }
            }
            best as u32
        };
        assert_ne!(chosen, blocked, "the mask must have excluded the blocked token");

        let after: Vec<f32> = logits.to_vec1().unwrap();
        assert_eq!(
            before, after,
            "masking mutated the caller's logits; the distribution reader in \
             adjudicator.rs would report post-mask mass"
        );
        assert_eq!(
            after[blocked as usize], 99.0,
            "the blocked token must keep its raw score for the distribution reader"
        );
    }

    #[test]
    fn decoder_without_a_schema_is_an_unmodified_greedy_sampler() {
        let logits = Tensor::new(&[0.1f32, 0.9, 0.3], &Device::Cpu).unwrap();
        let mut decoder = Decoder::new(None, &Device::Cpu, 3, &[], 1.0).unwrap();
        assert!(decoder.masker().is_none());
        assert_eq!(decoder.sample(&logits).unwrap(), 1);
    }
}
