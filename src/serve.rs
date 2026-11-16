//! The HTTP surface every Fennec consumer already speaks: OpenAI-compatible
//! `/v1/chat/completions` (JSON or SSE), `/v1/models`, `/health`, `/metrics`,
//! `/experts`, `/profile`, plus fox's `/api/generate`.
//!
//! `std::net` only, one thread per connection, one model behind a mutex. The
//! engine is a single sequential decoder - two requests cannot share the GPU -
//! so serialising them is the honest behaviour rather than a queue that
//! pretends otherwise.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde_json::{json, Value};

use crate::config::TextConfig;
use crate::model::Model;
use crate::safetensors::Shards;
use crate::tokenizer::{chat_prompt, Tokenizer};

/// Prompt tokens per prefill chunk. A chunk is the batch the dense trunk's
/// weight rows are reused across, so bigger is cheaper until the chunk's own
/// activations stop fitting a core's L2: at the shipped model's 2048-wide
/// hidden state the slab is 128 KiB at 16 tokens and 256 KiB at 32, against
/// 1-2 MB of L2. Long prompts simply run the loop again, and the buffers stay
/// one chunk wide either way.
///
/// 32 beat 16 in every pairing of a four-run session (TTFT 1.875 s twice,
/// against 1.906 s then 2.015 s at 16) and left TPOT to the session drift, as
/// it should: the experts a token routes to are per-token work no chunk size
/// amortises, so this buys back the dense trunk passes and nothing else.
const PREFILL_CHUNK: usize = 32;

/// Where the token-by-token loop starts. The chunked prefill leaves the prompt
/// in the layer states up to the last token, so the loop resumes there; a
/// device tier has no batched path, and then the loop owns every prompt token.
/// Getting this wrong is not slow, it is silent: the model answers the last
/// prompt token as if it were the whole conversation.
fn first_forward(prompt_len: usize, batched: bool) -> usize {
    if batched {
        prompt_len.saturating_sub(1)
    } else {
        0
    }
}

pub struct Stats {
    pub requests: AtomicU64,
    pub prompt_tokens: AtomicU64,
    pub generated: AtomicU64,
    pub busy_ns: AtomicU64,
}

pub struct Server {
    pub model: Arc<Mutex<Model>>,
    pub shards: Arc<Shards>,
    pub cfg: Arc<TextConfig>,
    pub tok: Tokenizer,
    pub model_id: String,
    pub api_key: String,
    pub stats: Arc<Stats>,
    pub started: Instant,
    pub max_ctx: usize,
}

/// Sampling knobs, defaulted from the checkpoint's `generation_config.json`.
#[derive(Clone, Copy)]
pub struct Sample {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: u64,
}

impl Default for Sample {
    fn default() -> Sample {
        Sample { temperature: 1.0, top_k: 20, top_p: 0.95, seed: 0x5EED }
    }
}

impl Sample {
    fn from_body(v: &Value) -> Sample {
        let mut s = Sample::default();
        if let Some(t) = v["temperature"].as_f64() {
            s.temperature = t as f32;
        }
        if let Some(k) = v["top_k"].as_u64() {
            s.top_k = k as usize;
        }
        if let Some(p) = v["top_p"].as_f64() {
            s.top_p = p as f32;
        }
        if let Some(seed) = v["seed"].as_u64() {
            s.seed = seed;
        }
        s
    }

    /// Greedy at temperature 0, otherwise top-k then nucleus then a draw. The
    /// RNG is a splitmix so a seed reproduces a completion exactly.
    fn pick(&self, logits: &[f32], rng: &mut u64) -> usize {
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, &x) in logits.iter().enumerate() {
            if x > bv {
                bv = x;
                best = i;
            }
        }
        if self.temperature <= 0.0 {
            return best;
        }
        let inv = 1.0 / self.temperature;
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        // A partial selection is enough: only the top-k can survive, and the
        // nucleus is a prefix of that, so sorting 248320 logits is wasted work.
        let k = self.top_k.min(idx.len()).max(1);
        idx.select_nth_unstable_by(k - 1, |&a, &b| {
            logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal)
        });
        idx.truncate(k);
        idx.sort_unstable_by(|&a, &b| {
            logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal)
        });
        let max = logits[idx[0]];
        let mut probs: Vec<f32> = idx.iter().map(|&i| ((logits[i] - max) * inv).exp()).collect();
        let sum: f32 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= sum;
        }
        let mut keep = probs.len();
        if self.top_p < 1.0 {
            let mut acc = 0f32;
            for (i, &p) in probs.iter().enumerate() {
                acc += p;
                if acc >= self.top_p {
                    keep = i + 1;
                    break;
                }
            }
        }
        *rng = rng.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = *rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        let draw = (z >> 11) as f32 / 9007199254740992.0;
        let total: f32 = probs[..keep].iter().sum();
        let mut acc = 0f32;
        for i in 0..keep {
            acc += probs[i] / total;
            if draw < acc {
                return idx[i];
            }
        }
        idx[keep - 1]
    }
}

impl Server {
    pub fn new(
        model: Arc<Mutex<Model>>,
        shards: Arc<Shards>,
        cfg: Arc<TextConfig>,
        tok: Tokenizer,
        model_id: String,
        api_key: String,
        max_ctx: usize,
    ) -> Server {
        Server {
            model,
            shards,
            cfg,
            tok,
            model_id,
            api_key,
            stats: Arc::new(Stats {
                requests: AtomicU64::new(0),
                prompt_tokens: AtomicU64::new(0),
                generated: AtomicU64::new(0),
                busy_ns: AtomicU64::new(0),
            }),
            started: Instant::now(),
            max_ctx,
        }
    }

    fn authorized(&self, head: &str) -> bool {
        bearer_ok(head, &self.api_key)
    }

    /// One completion, token by token. `emit` is called with each decoded piece
    /// and whether it is inside a think block; returning false stops the turn,
    /// which is how a disconnected client cancels generation.
    pub fn generate<F>(
        &self,
        prompt_ids: &[u32],
        max_new: usize,
        s: Sample,
        mut emit: F,
    ) -> Result<(usize, String), String>
    where
        F: FnMut(&str, bool) -> bool,
    {
        let model = self.model.lock().map_err(|_| "model mutex poisoned")?;
        // The rope tables are sized to max_ctx at load, so a prompt at or past
        // that limit would panic mid-forward and poison the mutex for every
        // later request. Refuse it, and clamp the continuation to what fits.
        if prompt_ids.len() >= self.max_ctx {
            return Err(format!("prompt of {} tokens exceeds the {max_ctx}-token context", prompt_ids.len(), max_ctx = self.max_ctx));
        }
        let max_new = max_new.min(self.max_ctx - prompt_ids.len() - 1);
        if max_new == 0 {
            return Ok((0, "length".into()));
        }
        let h = self.cfg.hidden_size;
        let eps = self.cfg.rms_norm_eps as f32;
        let mut st = model.states(prompt_ids.len() + max_new + 1);
        let mut logits = vec![0f32; self.cfg.vocab_size];
        let mut hn = vec![0f32; h];
        let mut rng = s.seed;
        let mut ids = prompt_ids.to_vec();
        let mut think = false;
        let mut pending = String::new();
        let mut raw: Vec<u8> = Vec::new();
        let mut produced = 0usize;
        let mut finish = "length";

        // Prefill takes the prompt in chunks, one pass over the dense trunk per
        // chunk instead of one per token, and stops one token short so the
        // logits path below stays exactly where it was. Generation stays token
        // by token: a new token is the argument to the next forward. Chunking
        // the prompt is host-tier only - the gate below says why. Pulling the
        // prompt off the device and running it through the host kernels measured
        // 2166 ms against 1750 ms on the VRAM path, so the device tier keeps its
        // own device path for the prompt even though it runs it one token at a
        // time.
        let prompt_len = ids.len();
        let batched = prompt_len > 1 && model.prefill_ready();
        let first = first_forward(prompt_len, batched);
        if batched {
            let mut xs = vec![0f32; PREFILL_CHUNK.min(first) * h];
            let nrow = xs.len() / h;
            let mut c0 = 0usize;
            while c0 < first {
                let n = nrow.min(first - c0);
                for (i, &id) in ids[c0..c0 + n].iter().enumerate() {
            let row = model.embed_row(&self.shards, id as usize)?;
                    xs[i * h..(i + 1) * h].copy_from_slice(&row);
                }
                model.prefill(&mut xs[..n * h], &mut st, c0);
                c0 += n;
            }
            crate::prof::split(first);
        }
        let end = prompt_len + max_new;
        for t in first..end {
            let mut x = model.embed_row(&self.shards, ids[t] as usize)?;
            model.forward(&mut x, &mut st, t);
            // Only the chunked arm has a phase boundary of its own: a prompt run
            // token by token would otherwise leave every prompt stage in the
            // decode counters, and `prompt` would mean two different things in
            // the two profiles being compared. One token short, same as
            // `split(first)` above.
            if !batched && t + 2 == prompt_len {
                crate::prof::split(t + 1);
            }
            if t + 1 < ids.len() {
                continue;
            }
            crate::kernel::rmsnorm(&x, &model.norm, eps, &mut hn);
            model.logits(&self.shards, &hn, &mut logits)?;
            let next = s.pick(&logits, &mut rng) as u32;
            if self.tok.eos.contains(&next) {
                finish = "stop";
                break;
            }
            produced += 1;
            ids.push(next);
            // A token can end mid-character. Hold the incomplete tail back as
            // bytes rather than letting a lossy decode make it permanent.
            raw.extend_from_slice(&self.tok.decode_bytes(&[next], false));
            let valid = match std::str::from_utf8(&raw) {
                Ok(s) => s.len(),
                Err(e) => e.valid_up_to(),
            };
            let piece = String::from_utf8(raw[..valid].to_vec()).map_err(|e| e.to_string())?;
            raw.drain(..valid);
            // Route think-block text to `reasoning_content` without leaking the
            // markers into either field. A marker can straddle two tokens, so
            // hold back a tail until it is unambiguous.
            pending.push_str(&piece);
            loop {
                let (open, close) = ("<think>", "</think>");
                let marker = if think { close } else { open };
                if let Some(at) = pending.find(marker) {
                    let before = pending[..at].to_string();
                    if !before.is_empty() && !emit(&before, think) {
                        return Ok((produced, "cancelled".into()));
                    }
                    think = !think;
                    pending = pending[at + marker.len()..].to_string();
                    continue;
                }
                // Keep any suffix that could still become the marker.
                let hold = (1..marker.len())
                    .rev()
                    .find(|&n| pending.ends_with(&marker[..n]))
                    .unwrap_or(0);
                let cut = pending.len() - hold;
                if cut > 0 {
                    let out = pending[..cut].to_string();
                    pending = pending[cut..].to_string();
                    if !emit(&out, think) {
                        return Ok((produced, "cancelled".into()));
                    }
                }
                break;
            }
            if produced >= max_new {
                break;
            }
        }
        if !pending.is_empty() {
            emit(&pending, think);
        }
        Ok((produced, finish.into()))
    }

    fn messages(v: &Value) -> Result<Vec<(String, String)>, String> {
        let arr = v["messages"].as_array().ok_or("messages must be an array")?;
        let mut out = Vec::with_capacity(arr.len());
        for m in arr {
            let role = m["role"].as_str().ok_or("each message needs a role")?;
            let content = match &m["content"] {
                Value::String(s) => s.clone(),
                Value::Array(parts) => {
                    let mut text = String::new();
                    for part in parts {
                        match part["text"].as_str() {
                            Some(t) => text.push_str(t),
                            // A part without readable text used to be filtered out, so an
                            // image-bearing request was answered from its caption alone and
                            // every client showed that as an analyzed image. This checkpoint
                            // has no vision tower: refuse instead of quietly losing input.
                            None => return Err(format!(
                                "content part `{}` is not text: this engine analyzes no images",
                                part["type"].as_str().unwrap_or("unknown")
                            )),
                        }
                    }
                    text
                }
                Value::Null => String::new(),
                _ => return Err("content must be a string or an array of text parts".into()),
            };
            out.push((role.to_string(), content));
        }
        if out.is_empty() {
            return Err("messages is empty".into());
        }
        Ok(out)
    }

    fn chat(&self, v: &Value, stream: bool, sock: &mut TcpStream) -> Result<(), String> {
        let msgs = Self::messages(v)?;
        // Colibri's qwen renderer defaults thinking off unless a reasoning effort
        // asks for it (openai_server.py: `body.get("enable_thinking",
        // reasoning_effort not in (None, "none"))`). Defaulting it on instead made
        // a plain request spend its whole budget in `reasoning_content` and return
        // an empty `content` to every client that reads the answer field.
        let thinking = match v["enable_thinking"].as_bool() {
            Some(on) => on,
            None => !matches!(v["reasoning_effort"].as_str(), None | Some("none")),
        };
        let prompt = chat_prompt(&msgs, thinking);
        let ids = self.tok.encode(&prompt);
        let max_new = v["max_tokens"].as_u64().unwrap_or(2048).min(8192) as usize;
        let s = Sample::from_body(v);
        let n_prompt = ids.len();
        let id = format!("chatcmpl-{:x}", self.stats.requests.load(Relaxed));
        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        if stream {
            let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
                        Cache-Control: no-cache\r\nConnection: close\r\n\
                        Access-Control-Allow-Origin: *\r\n\r\n";
            sock.write_all(head.as_bytes()).map_err(|e| e.to_string())?;
            let mut content = String::new();
            let mut reasoning = String::new();
            let mut send = |delta: Value, finish: Value| -> bool {
                let chunk = json!({
                    "id": id, "object": "chat.completion.chunk", "created": created,
                    "model": self.model_id,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                });
                let line = format!("data: {chunk}\n\n");
                sock.write_all(line.as_bytes()).is_ok()
            };
            // The role delta opens the stream, as OpenAI clients expect.
            if !send(json!({"role": "assistant"}), Value::Null) {
                return Ok(());
            }
            let t0 = Instant::now();
            crate::prof::reset();
            let (n, finish) = self.generate(&ids, max_new, s, |piece, think| {
                if think {
                    reasoning.push_str(piece);
                    send(json!({"reasoning_content": piece}), Value::Null)
                } else {
                    content.push_str(piece);
                    send(json!({"content": piece}), Value::Null)
                }
            })?;
            self.stats.prompt_tokens.fetch_add(n_prompt as u64, Relaxed);
            self.stats.generated.fetch_add(n as u64, Relaxed);
            self.stats.busy_ns.fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
            profile_dump(t0.elapsed().as_secs_f64() * 1e3);
            let usage = json!({"prompt_tokens": n_prompt, "completion_tokens": n,
                               "total_tokens": n_prompt + n});
            let last = json!({
                "id": id, "object": "chat.completion.chunk", "created": created,
                "model": self.model_id,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                "usage": usage,
            });
            let _ = sock.write_all(format!("data: {last}\n\ndata: [DONE]\n\n").as_bytes());
            return Ok(());
        }

        let t0 = Instant::now();
        crate::prof::reset();
        let mut content = String::new();
        let mut reasoning = String::new();
        let (n, finish) = self.generate(&ids, max_new, s, |piece, think| {
            if think {
                reasoning.push_str(piece);
            } else {
                content.push_str(piece);
            }
            true
        })?;
        self.stats.prompt_tokens.fetch_add(n_prompt as u64, Relaxed);
        self.stats.generated.fetch_add(n as u64, Relaxed);
        self.stats.busy_ns.fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
        profile_dump(t0.elapsed().as_secs_f64() * 1e3);
        let mut message = json!({"role": "assistant", "content": content});
        if !reasoning.is_empty() {
            message["reasoning_content"] = json!(reasoning);
        }
        let body = json!({
            "id": id, "object": "chat.completion", "created": created, "model": self.model_id,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": n_prompt, "completion_tokens": n,
                      "total_tokens": n_prompt + n},
        });
        write_json(sock, 200, &body)
    }

    fn route(&self, method: &str, path: &str, head: &str, body: &Value, sock: &mut TcpStream) -> Result<(), String> {
        // A browser preflight carries no credentials, so gating it on the key
        // would make every cross-origin request fail before it starts.
        if method != "OPTIONS" && !self.authorized(head) {
            return write_json(sock, 401, &json!({"error": {"message": "invalid api key", "type": "auth"}}));
        }
        self.stats.requests.fetch_add(1, Relaxed);
        match (method, path) {
            ("GET", "/health") => write_json(sock, 200, &json!({
                "status": "ok", "engine": "fennec", "version": crate::VERSION,
                "vision": false,
                "model": self.model_id, "uptime_s": self.started.elapsed().as_secs(),
            })),
            ("GET", "/v1/models") => write_json(sock, 200, &json!({
                "object": "list",
                "data": [{"id": self.model_id, "object": "model", "owned_by": "fennec",
                          "created": 0, "root": self.model_id,
                          "max_model_len": self.max_ctx}],
            })),
            ("GET", "/metrics") => {
                let g = self.stats.generated.load(Relaxed);
                let busy = self.stats.busy_ns.load(Relaxed) as f64 / 1e9;
                write_json(sock, 200, &json!({
                    "requests": self.stats.requests.load(Relaxed),
                    "prompt_tokens": self.stats.prompt_tokens.load(Relaxed),
                    "generated_tokens": g,
                    "busy_seconds": busy,
                    "tokens_per_second": if busy > 0.0 { g as f64 / busy } else { 0.0 },
                    "uptime_s": self.started.elapsed().as_secs(),
                }))
            }
            ("GET", "/experts") => {
                let m = self.model.lock().map_err(|_| "model mutex poisoned")?;
                let first = m.first;
                let layers: Vec<Value> = m.layers.iter().enumerate().map(|(i, l)| {
                    // A dense checkpoint routes nothing, so its layers report
                    // zero hits rather than an error: the shape stays the same.
                    let moe = l.moe.as_ref();
                    json!({"layer": first + i,
                           "device_hits": moe.map_or(0, |m| m.dev_hits.load(Relaxed)),
                           "host_hits": moe.map_or(0, |m| m.host_hits.load(Relaxed)),
                           "disk_hits": moe.map_or(0, |m| m.disk_hits.load(Relaxed))})
                }).collect();
                write_json(sock, 200, &json!({"layers": layers, "num_experts": self.cfg.num_experts,
                                              "top_k": self.cfg.num_experts_per_tok}))
            }
            ("GET", "/profile") => {
                let m = self.model.lock().map_err(|_| "model mutex poisoned")?;
                write_json(sock, 200, &json!({
                    "layers_loaded": m.layers.len(),
                    "layers_total": self.cfg.num_hidden_layers,
                    "first_layer": m.first,
                    "hidden_size": self.cfg.hidden_size,
                    "num_experts": self.cfg.num_experts,
                    "moe_intermediate_size": self.cfg.moe_intermediate_size,
                    "is_moe": self.cfg.is_moe(),
                }))
            }
            ("GET", "/diagnostics") => {
                // Comprehensive diagnostics for desktop app and IDE plugins.
                // Combines health, metrics, system info, and device status.
                let m = self.model.lock().map_err(|_| "model mutex poisoned")?;
                let g = self.stats.generated.load(Relaxed);
                let busy = self.stats.busy_ns.load(Relaxed) as f64 / 1e9;
                let expert_stats: Vec<Value> = m.layers.iter().enumerate().map(|(i, l)| {
                    let moe = l.moe.as_ref();
                    json!({
                        "layer": m.first + i,
                        "device_hits": moe.map_or(0, |m| m.dev_hits.load(Relaxed)),
                        "host_hits": moe.map_or(0, |m| m.host_hits.load(Relaxed)),
                        "disk_hits": moe.map_or(0, |m| m.disk_hits.load(Relaxed)),
                    })
                }).collect();
                write_json(sock, 200, &json!({
                    "engine": {
                        "name": "fennec",
                        "version": crate::VERSION,
                        "status": "ok",
                        "uptime_s": self.started.elapsed().as_secs(),
                    },
                    "model": {
                        "id": self.model_id,
                        "layers_loaded": m.layers.len(),
                        "layers_total": self.cfg.num_hidden_layers,
                        "first_layer": m.first,
                        "hidden_size": self.cfg.hidden_size,
                        "num_experts": self.cfg.num_experts,
                        "moe_intermediate_size": self.cfg.moe_intermediate_size,
                        "is_moe": self.cfg.is_moe(),
                        "max_context": self.max_ctx,
                        "vision": false,
                    },
                    "performance": {
                        "requests": self.stats.requests.load(Relaxed),
                        "prompt_tokens": self.stats.prompt_tokens.load(Relaxed),
                        "generated_tokens": g,
                        "busy_seconds": busy,
                        "tokens_per_second": if busy > 0.0 { g as f64 / busy } else { 0.0 },
                    },
                    "experts": expert_stats,
                    "system": {
                        "threads": crate::kernel::get_threads(),
                        "physical_cores": crate::kernel::physical_cores(),
                        "dense_int4": crate::model::dense_int4(),
                        "dense_i8": crate::model::dense_i8(),
                    },
                }))
            }
            ("POST", "/v1/chat/completions") => {
                let stream = body["stream"].as_bool().unwrap_or(false);
                self.chat(body, stream, sock)
            }
            ("POST", "/api/generate") => {
                let prompt = body["prompt"].as_str().ok_or("prompt must be a string")?;
                let ids = self.tok.encode(prompt);
                let max_new = body["options"]["num_predict"].as_u64()
                    .or_else(|| body["max_tokens"].as_u64()).unwrap_or(256).min(8192) as usize;
                let s = Sample::from_body(body);
                let stream = body["stream"].as_bool().unwrap_or(false);
                let n_prompt = ids.len();
                if stream {
                    let head = "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\n\
                                Connection: close\r\n\r\n";
                    sock.write_all(head.as_bytes()).map_err(|e| e.to_string())?;
                    let mut text = String::new();
                    let (n, finish) = self.generate(&ids, max_new, s, |piece, _| {
                        text.push_str(piece);
                        let line = json!({"model": self.model_id, "response": piece, "done": false});
                        sock.write_all(format!("{line}\n").as_bytes()).is_ok()
                    })?;
                    let last = json!({"model": self.model_id, "response": "", "done": true,
                                      "done_reason": finish, "prompt_eval_count": n_prompt,
                                      "eval_count": n});
                    let _ = sock.write_all(format!("{last}\n").as_bytes());
                    return Ok(());
                }
                let mut text = String::new();
                let (n, finish) = self.generate(&ids, max_new, s, |piece, _| {
                    text.push_str(piece);
                    true
                })?;
                write_json(sock, 200, &json!({
                    "model": self.model_id, "response": text, "done": true,
                    "done_reason": finish, "prompt_eval_count": n_prompt, "eval_count": n,
                }))
            }
            ("OPTIONS", _) => {
                let head = "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\n\
                            Access-Control-Allow-Headers: Content-Type, Authorization\r\n\
                            Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
                            Content-Length: 0\r\nConnection: close\r\n\r\n";
                sock.write_all(head.as_bytes()).map_err(|e| e.to_string())
            }
            _ => write_json(sock, 404, &json!({"error": {"message": format!("no route {method} {path}")}})),
        }
    }
}

/// Header names and the auth scheme are case-insensitive; the credential is
/// not. An empty key means the engine is open, which is the loopback default.
fn bearer_ok(head: &str, key: &str) -> bool {
    if key.is_empty() {
        return true;
    }
    for line in head.lines() {
        let Some((name, value)) = line.split_once(':') else { continue };
        if !name.trim().eq_ignore_ascii_case("authorization") {
            continue;
        }
        let value = value.trim();
        let token = match value.get(..7) {
            Some(p) if p.eq_ignore_ascii_case("bearer ") => &value[7..],
            _ => value,
        };
        return token.trim() == key;
    }
    false
}

fn write_json(sock: &mut TcpStream, code: u16, v: &Value) -> Result<(), String> {
    let body = serde_json::to_vec(v).map_err(|e| e.to_string())?;
    let reason = match code {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        413 => "Payload Too Large",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        _ => "Error",
    };
    let head = format!(
        "HTTP/1.1 {code} {reason}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
        body.len()
    );
    sock.write_all(head.as_bytes()).map_err(|e| e.to_string())?;
    sock.write_all(&body).map_err(|e| e.to_string())
}

/// The stage split for one request, when FENNEC_PROFILE is on. `serve` is what
/// the benchmark harness drives, and it warms up first: a breakdown from `run`
/// or `decode` instead would be measuring cold page faults, not steady state.
fn profile_dump(wall_ms: f64) {
    if crate::prof::enabled() {
        eprint!("{}", crate::prof::report(wall_ms));
    }
}

/// Read one request. Bodies are capped: this is a loopback engine, not a proxy,
/// and an unbounded read is a trivial way to exhaust the host pool's RAM.
const MAX_BODY: usize = 32 << 20;

fn handle(srv: &Server, mut sock: TcpStream) -> Result<(), String> {
    let peer = sock.peer_addr().ok();
    sock.set_nodelay(true).ok();
    let mut reader = BufReader::new(sock.try_clone().map_err(|e| e.to_string())?);
    let mut head = String::new();
    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line).map_err(|e| e.to_string())?;
        if n == 0 {
            return Ok(());
        }
        head.push_str(&line);
        if head.ends_with("\r\n\r\n") || head.ends_with("\n\n") {
            break;
        }
        if head.len() > 64 << 10 {
            return write_json(&mut sock, 431, &json!({"error": {"message": "headers too large"}}));
        }
    }
    let mut first = head.lines().next().unwrap_or("").split_whitespace();
    let method = first.next().unwrap_or("").to_string();
    let target = first.next().unwrap_or("/").to_string();
    let path = target.split('?').next().unwrap_or("/").to_string();

    let len = head
        .lines()
        .find_map(|l| {
            let (k, v) = l.split_once(':')?;
            if k.trim().eq_ignore_ascii_case("content-length") {
                v.trim().parse::<usize>().ok()
            } else {
                None
            }
        })
        .unwrap_or(0);
    if len > MAX_BODY {
        return write_json(&mut sock, 413, &json!({"error": {"message": "body too large"}}));
    }
    let mut raw = vec![0u8; len];
    if len > 0 {
        reader.read_exact(&mut raw).map_err(|e| e.to_string())?;
    }
    let body: Value = if len == 0 {
        Value::Null
    } else {
        match serde_json::from_slice(&raw) {
            Ok(v) => v,
            Err(e) => {
                return write_json(&mut sock, 400, &json!({"error": {"message": format!("bad json: {e}")}}))
            }
        }
    };
    let started = Instant::now();
    let result = srv.route(&method, &path, &head, &body, &mut sock);
    if let Err(e) = result {
        let _ = write_json(&mut sock, 500, &json!({"error": {"message": e}}));
    }
    if let Some(p) = peer {
        let _ = std::io::stderr().write_all(
            format!("[fennec] {} {} {} {:.0} ms\n", p.ip(), method, path, started.elapsed().as_secs_f64() * 1e3)
                .as_bytes(),
        );
    }
    let _ = sock.flush();
    Ok(())
}

pub fn serve(srv: Arc<Server>, host: &str, port: u16) -> Result<(), String> {
    let listener = TcpListener::bind((host, port)).map_err(|e| format!("bind {host}:{port}: {e}"))?;
    eprintln!("[fennec] serving {} on http://{host}:{port}", srv.model_id);
    for conn in listener.incoming() {
        let Ok(sock) = conn else { continue };
        let srv = srv.clone();
        std::thread::spawn(move || {
            if let Err(e) = handle(&srv, sock) {
                let _ = std::io::stderr().write_all(format!("[fennec] connection: {e}\n").as_bytes());
            }
        });
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bearer_survives_header_and_scheme_case() {
        // The whole header block used to be lowercased before this check, which
        // folded "Bearer" into "bearer" and rejected every real client.
        let head = "GET /health HTTP/1.1\r\nHost: 127.0.0.1:8184\r\n\
                    Authorization:   Bearer local\r\nAccept: */*\r\n\r\n";
        assert!(bearer_ok(head, "local"));
        assert!(bearer_ok(&head.to_lowercase(), "local"));
        assert!(bearer_ok("authorization: bearer local\r\n", "local"));
        assert!(bearer_ok("AUTHORIZATION: Bearer local\r\n", "local"));
        assert!(bearer_ok("Authorization: local\r\n", "local"));
    }

    #[test]
    fn the_credential_itself_is_case_sensitive() {
        assert!(bearer_ok("Authorization: Bearer LoCaL\r\n", "LoCaL"));
        assert!(!bearer_ok("Authorization: Bearer local\r\n", "LoCaL"));
        assert!(!bearer_ok("Authorization: Bearer wrong\r\n", "local"));
        assert!(!bearer_ok("GET /health HTTP/1.1\r\nHost: x\r\n\r\n", "local"));
        assert!(!bearer_ok("X-Authorization: Bearer local\r\n", "local"));
    }

    #[test]
    fn an_empty_key_means_the_engine_is_open() {
        assert!(bearer_ok("GET /health HTTP/1.1\r\n\r\n", ""));
    }

    #[test]
    fn greedy_takes_the_argmax() {
        let logits = [0.1, 9.0, -3.0, 2.0];
        let s = Sample { temperature: 0.0, ..Sample::default() };
        assert_eq!(s.pick(&logits, &mut 1), 1);
    }

    #[test]
    fn top_k_never_reaches_past_the_k_th_logit() {
        let logits: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let s = Sample { temperature: 1.0, top_k: 3, top_p: 1.0, seed: 0 };
        let mut rng = 0x5EEDu64;
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..400 {
            let i = s.pick(&logits, &mut rng);
            assert!(i >= 7, "top_k=3 returned index {i}");
            seen.insert(i);
        }
        assert!(seen.len() > 1, "sampling never varied: {seen:?}");
    }

    #[test]
    fn a_dominant_logit_survives_any_nucleus() {
        let mut logits = vec![0.0f32; 8];
        logits[3] = 100.0;
        let s = Sample { temperature: 1.0, top_k: 8, top_p: 0.5, seed: 0 };
        let mut rng = 7u64;
        for _ in 0..100 {
            assert_eq!(s.pick(&logits, &mut rng), 3);
        }
    }

    #[test]
    fn the_same_seed_replays_the_same_tokens() {
        let logits: Vec<f32> = (0..16).map(|i| (i as f32 * 0.37).sin()).collect();
        let s = Sample { temperature: 1.0, top_k: 8, top_p: 0.95, seed: 0 };
        let run = |seed: u64| -> Vec<usize> {
            let mut rng = seed;
            (0..32).map(|_| s.pick(&logits, &mut rng)).collect()
        };
        assert_eq!(run(0x5EED), run(0x5EED));
        assert_ne!(run(0x5EED), run(0x5EEE));
    }

    #[test]
    fn text_parts_join_and_an_image_part_refuses_to_be_ignored() {
        let plain = json!({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "look at "}, {"type": "input_text", "text": "this"}]}]});
        assert_eq!(Server::messages(&plain).unwrap(),
                   vec![("user".to_string(), "look at this".to_string())]);

        for image in [
            json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}),
            json!({"type": "input_image", "url": "data:image/png;base64,AAAA"}),
        ] {
            let body = json!({"messages": [{"role": "user", "content": [
                {"type": "text", "text": "what is in this?"}, image]}]});
            let error = Server::messages(&body).unwrap_err();
            assert!(error.contains("analyzes no images"), "{error}");
        }
    }

    #[test]
    fn a_batched_prefill_hands_the_loop_the_last_prompt_token_only() {
        // The chunked prefill covers the prompt up to the token whose logits
        // come first, so the token loop resumes there. `saturating_sub` because
        // a one-token prompt has nothing to prefill.
        assert_eq!(first_forward(30, true), 29);
        assert_eq!(first_forward(1, true), 0);
        assert_eq!(first_forward(0, true), 0);
    }

    #[test]
    fn a_plan_without_a_batched_path_walks_the_whole_prompt() {
        // This is the device tier: `prefill_ready` is false because the dense
        // projections live in VRAM, so starting the loop at the last prompt
        // token would drop everything before it. The output stays fluent and
        // the TTFT looks wonderful, which is why the bound is a named function.
        for prompt_len in [0usize, 1, 2, 30, 4096] {
            assert_eq!(first_forward(prompt_len, false), 0, "{prompt_len} tokens");
        }
    }
}
