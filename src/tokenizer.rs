//! The Qwen byte-level BPE, read straight from the checkpoint's `tokenizer.json`.
//!
//! Nothing here is model-specific beyond the files it loads: the split pattern,
//! the merge table and the added tokens all come from the checkpoint, so a
//! different Qwen release needs no code change. The pretokenizer pattern is used
//! verbatim - it contains a lookahead (`\s+(?!\S)`), which is why this uses
//! `fancy-regex` rather than `regex`.

use std::collections::HashMap;
use std::path::Path;

use fancy_regex::Regex;
use serde_json::Value;
use unicode_normalization::UnicodeNormalization;

/// GPT-2's byte-to-unicode table: the printable Latin-1 ranges map to themselves
/// and every other byte maps to a code point above 256, so a byte string becomes
/// a `str` the merge table can talk about without escaping.
fn byte_table() -> ([char; 256], HashMap<char, u8>) {
    let mut cs = [0u32; 256];
    let mut direct = [false; 256];
    for b in 0..256u32 {
        let ok = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        direct[b as usize] = ok;
        if ok {
            cs[b as usize] = b;
        }
    }
    let mut n = 256u32;
    for b in 0..256 {
        if !direct[b] {
            cs[b] = n;
            n += 1;
        }
    }
    let mut enc = ['\0'; 256];
    let mut dec = HashMap::new();
    for b in 0..256 {
        let c = char::from_u32(cs[b]).unwrap();
        enc[b] = c;
        dec.insert(c, b as u8);
    }
    (enc, dec)
}

/// The pattern `ByteLevel(use_regex = true)` splits with, applied after the
/// checkpoint's own Split pretokenizer - HF runs both, in that order.
const BYTE_LEVEL: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    tokens: Vec<String>,
    pair_rank: HashMap<(u32, u32), u32>,
    split: Regex,
    byte_split: Option<Regex>,
    nfc: bool,
    added: Vec<(String, u32)>,
    added_re: Option<Regex>,
    enc: [char; 256],
    dec: HashMap<char, u8>,
    pub eos: Vec<u32>,
    pub pad: u32,
}

fn escape(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        if "\\.^$|?*+()[]{}".contains(c) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

impl Tokenizer {
    pub fn from_dir(dir: &str) -> Result<Tokenizer, String> {
        let text = std::fs::read_to_string(Path::new(dir).join("tokenizer.json"))
            .map_err(|e| format!("tokenizer.json: {e}"))?;
        let v: Value = serde_json::from_str(&text).map_err(|e| format!("tokenizer.json: {e}"))?;
        let model = &v["model"];
        if model["type"].as_str() != Some("BPE") {
            return Err(format!("expected a BPE tokenizer, got {:?}", model["type"]));
        }

        let mut vocab = HashMap::new();
        let mut tokens: Vec<String> = Vec::new();
        for (k, id) in model["vocab"].as_object().ok_or("vocab is not an object")? {
            let id = id.as_u64().ok_or("vocab id is not a number")? as u32;
            vocab.insert(k.clone(), id);
            if tokens.len() <= id as usize {
                tokens.resize(id as usize + 1, String::new());
            }
            tokens[id as usize] = k.clone();
        }

        // Merges arrive either as "a b" strings or as ["a","b"] pairs depending
        // on which tokenizer version wrote the file.
        let mut pair_rank = HashMap::new();
        for (rank, m) in model["merges"].as_array().ok_or("merges is not an array")?.iter().enumerate()
        {
            let (a, b) = match m {
                Value::String(s) => match s.split_once(' ') {
                    Some((a, b)) => (a.to_string(), b.to_string()),
                    None => continue,
                },
                Value::Array(p) if p.len() == 2 => (
                    p[0].as_str().unwrap_or("").to_string(),
                    p[1].as_str().unwrap_or("").to_string(),
                ),
                _ => continue,
            };
            if let (Some(&x), Some(&y)) = (vocab.get(&a), vocab.get(&b)) {
                pair_rank.insert((x, y), rank as u32);
            }
        }

        let mut added: Vec<(String, u32)> = Vec::new();
        for a in v["added_tokens"].as_array().cloned().unwrap_or_default() {
            if let (Some(c), Some(id)) = (a["content"].as_str(), a["id"].as_u64()) {
                let id = id as u32;
                if vocab.get(c).copied() != Some(id) {
                    vocab.insert(c.to_string(), id);
                }
                if tokens.len() <= id as usize {
                    tokens.resize(id as usize + 1, String::new());
                }
                tokens[id as usize] = c.to_string();
                added.push((c.to_string(), id));
            }
        }
        // Longest first: an alternation is leftmost-first, so a shorter special
        // token would otherwise shadow a longer one that starts with it.
        added.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        let added_re = if added.is_empty() {
            None
        } else {
            let pat = added.iter().map(|(s, _)| escape(s)).collect::<Vec<_>>().join("|");
            Some(Regex::new(&format!("({pat})")).map_err(|e| format!("added-token pattern: {e}"))?)
        };

        // The checkpoint's Split pattern, used verbatim. ByteLevel only splits
        // again when it says use_regex: this checkpoint sets it false, and
        // re-splitting with the GPT-2 pattern breaks `(a` into `(` + `a`.
        let mut pattern = BYTE_LEVEL.to_string();
        let mut byte_split = None;
        let mut nfc = false;
        if let Some(seq) = v["pre_tokenizer"]["pretokenizers"].as_array() {
            for p in seq {
                match p["type"].as_str() {
                    Some("Split") => {
                        if let Some(r) = p["pattern"]["Regex"].as_str() {
                            pattern = r.to_string();
                        }
                    }
                    Some("ByteLevel") if p["use_regex"].as_bool() == Some(true) => {
                        byte_split = Some(
                            Regex::new(BYTE_LEVEL).map_err(|e| format!("byte pattern: {e}"))?,
                        );
                    }
                    _ => {}
                }
            }
        }
        if v["pre_tokenizer"]["type"].as_str() == Some("ByteLevel")
            && v["pre_tokenizer"]["use_regex"].as_bool() == Some(true)
        {
            byte_split = Some(Regex::new(BYTE_LEVEL).map_err(|e| format!("byte pattern: {e}"))?);
        }
        match v["normalizer"]["type"].as_str() {
            Some("NFC") => nfc = true,
            Some("Sequence") => {
                nfc = v["normalizer"]["normalizers"]
                    .as_array()
                    .map(|a| a.iter().any(|n| n["type"].as_str() == Some("NFC")))
                    .unwrap_or(false);
            }
            _ => {}
        }
        let split = Regex::new(&pattern).map_err(|e| format!("split pattern: {e}"))?;

        // generation_config.json is authoritative for the stop ids: this
        // checkpoint stops on both <|im_end|> and <|endoftext|>, while
        // tokenizer_config.json names only the first.
        let mut eos: Vec<u32> = Vec::new();
        if let Ok(text) = std::fs::read_to_string(Path::new(dir).join("generation_config.json")) {
            if let Ok(g) = serde_json::from_str::<Value>(&text) {
                match &g["eos_token_id"] {
                    Value::Array(a) => eos.extend(a.iter().filter_map(|x| x.as_u64().map(|i| i as u32))),
                    Value::Number(n) => eos.extend(n.as_u64().map(|i| i as u32)),
                    _ => {}
                }
            }
        }
        let mut pad = 0u32;
        if let Ok(text) = std::fs::read_to_string(Path::new(dir).join("tokenizer_config.json")) {
            if let Ok(c) = serde_json::from_str::<Value>(&text) {
                let id_of = |v: &Value| -> Option<u32> {
                    let s = v.as_str().or_else(|| v["content"].as_str())?;
                    vocab.get(s).copied()
                };
                if eos.is_empty() {
                    match &c["eos_token"] {
                        Value::Array(a) => eos.extend(a.iter().filter_map(id_of)),
                        other => eos.extend(id_of(other)),
                    }
                }
                pad = id_of(&c["pad_token"]).or_else(|| id_of(&c["eos_token"])).unwrap_or(0);
            }
        }
        if eos.is_empty() {
            if let Some(&id) = vocab.get("<|endoftext|>") {
                eos.push(id);
            }
        }

        let (enc, dec) = byte_table();
        Ok(Tokenizer {
            vocab, tokens, pair_rank, split, byte_split, nfc, added, added_re, enc, dec, eos, pad,
        })
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn id_of(&self, s: &str) -> Option<u32> {
        self.vocab.get(s).copied()
    }

    pub fn token(&self, id: u32) -> &str {
        self.tokens.get(id as usize).map(|s| s.as_str()).unwrap_or("")
    }

    pub fn is_special(&self, id: u32) -> bool {
        self.added.iter().any(|(_, i)| *i == id)
    }

    fn byte_encode(&self, s: &str) -> String {
        s.as_bytes().iter().map(|&b| self.enc[b as usize]).collect()
    }

    /// One pretokenized piece through the merge table. Symbols are vocab ids
    /// throughout, so a merge is a pair lookup and a concatenation, not a string
    /// search over the whole word.
    fn bpe(&self, piece: &str) -> Vec<u32> {
        let mut word: Vec<u32> = Vec::with_capacity(piece.len());
        for c in piece.chars() {
            match self.vocab.get(&c.to_string()) {
                Some(&id) => word.push(id),
                // Every byte-level char is in the vocab by construction; a miss
                // means the checkpoint and its tokenizer.json disagree.
                None => return Vec::new(),
            }
        }
        if word.len() < 2 {
            return word;
        }
        loop {
            let mut best: Option<(u32, usize)> = None;
            for i in 0..word.len() - 1 {
                if let Some(&r) = self.pair_rank.get(&(word[i], word[i + 1])) {
                    if best.map(|(br, _)| r < br).unwrap_or(true) {
                        best = Some((r, i));
                    }
                }
            }
            let Some((_, at)) = best else { break };
            let (a, b) = (word[at], word[at + 1]);
            let merged = format!("{}{}", self.tokens[a as usize], self.tokens[b as usize]);
            let Some(&mid) = self.vocab.get(&merged) else { break };
            let mut out = Vec::with_capacity(word.len());
            let mut i = 0;
            while i < word.len() {
                if i + 1 < word.len() && word[i] == a && word[i + 1] == b {
                    out.push(mid);
                    i += 2;
                } else {
                    out.push(word[i]);
                    i += 1;
                }
            }
            word = out;
            if word.len() < 2 {
                break;
            }
        }
        word
    }

    fn pieces(&self, text: &str, out: &mut Vec<u32>) {
        let mut at = 0usize;
        while at < text.len() {
            let rest = &text[at..];
            let m = self.split.find(rest).ok().flatten();
            let Some(m) = m else {
                // The pattern covers every byte, so a miss means an empty tail.
                break;
            };
            if m.start() > 0 {
                // Isolated behaviour keeps the unmatched gap as its own piece.
                self.byte_pieces(&rest[..m.start()], out);
            }
            self.byte_pieces(m.as_str(), out);
            at += m.end();
            if m.end() == 0 {
                break;
            }
        }
    }

    fn byte_pieces(&self, piece: &str, out: &mut Vec<u32>) {
        let Some(re) = &self.byte_split else {
            out.extend(self.bpe(&self.byte_encode(piece)));
            return;
        };
        let mut at = 0usize;
        while at < piece.len() {
            let rest = &piece[at..];
            let Some(m) = re.find(rest).ok().flatten() else { break };
            if m.start() > 0 {
                out.extend(self.bpe(&self.byte_encode(&rest[..m.start()])));
            }
            out.extend(self.bpe(&self.byte_encode(m.as_str())));
            if m.end() == 0 {
                break;
            }
            at += m.end();
        }
    }

    /// Special tokens are matched literally before anything else, so a
    /// `<|im_start|>` in the text becomes one id rather than eleven bytes.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let normalized = if self.nfc { text.nfc().collect::<String>() } else { text.to_string() };
        let text = normalized.as_str();
        let mut out = Vec::new();
        let Some(re) = &self.added_re else {
            self.pieces(text, &mut out);
            return out;
        };
        let mut at = 0usize;
        while at < text.len() {
            let rest = &text[at..];
            let Some(m) = re.find(rest).ok().flatten() else {
                self.pieces(rest, &mut out);
                break;
            };
            if m.start() > 0 {
                self.pieces(&rest[..m.start()], &mut out);
            }
            if let Some(&id) = self.vocab.get(m.as_str()) {
                out.push(id);
            }
            if m.end() == 0 {
                break;
            }
            at += m.end();
        }
        out
    }

    /// The raw bytes behind `ids`, before any UTF-8 interpretation. Streaming
    /// callers need this: one token can end mid-character, and decoding it alone
    /// would turn the split sequence into replacement characters permanently.
    pub fn decode_bytes(&self, ids: &[u32], skip_special: bool) -> Vec<u8> {
        let mut s = String::new();
        for &id in ids {
            if skip_special && self.is_special(id) {
                continue;
            }
            s.push_str(self.token(id));
        }
        let mut bytes = Vec::with_capacity(s.len());
        for c in s.chars() {
            match self.dec.get(&c) {
                Some(&b) => bytes.push(b),
                None => bytes.extend_from_slice(c.encode_utf8(&mut [0u8; 4]).as_bytes()),
            }
        }
        bytes
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids, skip_special)).into_owned()
    }
}

/// The Qwen chat template. `tokenizer_config.json` carries no `chat_template`
/// for this checkpoint - only a `response_template` describing the assistant
/// turn - so the markup is built here from the same anchors it names.
pub fn chat_prompt(messages: &[(String, String)], enable_thinking: bool) -> String {
    let mut s = String::new();
    for (role, content) in messages {
        s.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
    }
    s.push_str("<|im_start|>assistant\n");
    if !enable_thinking {
        // An empty think block is how the Qwen template suppresses reasoning
        // without changing the tokenizer or the stop tokens.
        s.push_str("\n\n</think>\n\n");
    }
    s
}