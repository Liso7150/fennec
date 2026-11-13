//! Model configuration: the checkpoint's `config.json`, including the nested
//! `text_config` wrapper that `qwen3_5_moe` ships with.

use serde_json::Value;

pub type Error = String;
pub type Result<T> = std::result::Result<T, Error>;

/// The tensor prefix a `qwen3_5_moe` checkpoint stores its text stack under.
/// Dense Qwen3 exports `model.layers.N.*` instead, so this is only the default:
/// `model::bind_prefix` reads it back off the shard index at open time.
pub const DEFAULT_PREFIX: &str = "model.language_model";

#[derive(Debug, Clone)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    /// How the checkpoint stores RMSNorm weights: `true` as the delta `w` that
    /// the kernel applies as `(1 + w)` (Qwen3.5/3.6, Qwen3-Next), `false` as the
    /// full gain (plain `qwen3` / `qwen3_moe`, whose `Qwen3RMSNorm` multiplies
    /// `w` directly). The 1 comes off at load so the forward pass, and every
    /// device kernel with it, keeps one convention.
    pub norm_zero_centered: bool,
    pub layer_types: Vec<LayerKind>,
    // attention
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub partial_rotary_factor: f64,
    pub rope_theta: f64,
    pub attn_output_gate: bool,
    // dense MLP: the FFN of a checkpoint with no routed experts
    pub intermediate_size: usize,
    pub tie_word_embeddings: bool,
    // linear attention (gated delta net) - all zero when the checkpoint has none
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    // moe - all zero when the checkpoint is dense (`is_moe`)
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    /// The common prefix of every text-stack tensor name in this checkpoint.
    pub prefix: String,
    pub name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    Linear,
    Full,
}

impl TextConfig {
    pub fn rotary_dim(&self) -> usize {
        // transformers: dim = head_dim * partial_rotary_factor, inv_freq over arange(0, dim, 2)
        (self.head_dim as f64 * self.partial_rotary_factor) as usize
    }

    /// Routed experts at all: a dense checkpoint has no `num_experts` key, and
    /// every MoE/GDN number in this struct is then zero and must stay unused.
    pub fn is_moe(&self) -> bool {
        self.num_experts > 0
    }

    pub fn has_gdn(&self) -> bool {
        self.linear_num_value_heads > 0
    }

    /// One dense SiLU MLP: gate and up are `[I, H]`, down is `[H, I]`.
    pub fn dense_mlp_params(&self) -> usize {
        3 * self.intermediate_size * self.hidden_size
    }

    pub fn key_dim(&self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim
    }

    pub fn value_dim(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    pub fn conv_dim(&self) -> usize {
        self.key_dim() * 2 + self.value_dim()
    }

    pub fn n_linear_layers(&self) -> usize {
        self.layer_types.iter().filter(|k| **k == LayerKind::Linear).count()
    }

    /// The dense attention block of one layer: query (with its output gate
    /// when the checkpoint has one), the KV projections and o_proj.
    fn dense_attn_params(&self) -> usize {
        let h = self.hidden_size;
        let (nh, nkv, hd) = (self.num_attention_heads, self.num_key_value_heads, self.head_dim);
        let q = if self.attn_output_gate { 2 } else { 1 } * nh * hd * h;
        q + 2 * nkv * hd * h + nh * hd * h
    }

    /// Embedding plus lm_head; a tied checkpoint stores the head only as a
    /// second view of the embedding table, so it is read once, not twice.
    fn head_params(&self) -> usize {
        let copies = if self.tie_word_embeddings { 1 } else { 2 };
        copies * self.hidden_size * self.vocab_size
    }

    /// Params that a single decode token actually touches, and the bytes that
    /// implies at a given bits-per-weight. This is the number that decides
    /// whether a 35B model runs at 2 tok/s or 20.
    pub fn active_params(&self) -> usize {
        if !self.is_moe() {
            return self.num_hidden_layers * (self.dense_attn_params() + self.dense_mlp_params())
                + self.head_params();
        }
        let h = self.hidden_size;
        let experts = self.num_experts_per_tok * (2 * self.moe_intermediate_size * h + h * self.moe_intermediate_size);
        let shared = 3 * self.shared_expert_intermediate_size * h + 1 * h;
        let router = self.num_experts * h;
        let dense = experts + shared + router;
        let linear = self.linear_num_key_heads * 0
            + self.conv_dim() * h
            + self.value_dim() * h
            + 2 * self.linear_num_value_heads * h
            + h * self.value_dim()
            + self.conv_dim() * self.linear_conv_kernel_dim;
        let full = {
            let q = 2 * self.num_attention_heads * self.head_dim * h; // query + gate
            let kv = 2 * self.num_key_value_heads * self.head_dim * h;
            let o = self.num_attention_heads * self.head_dim * h;
            q + kv + o
        };
        let mut total = 0usize;
        for k in &self.layer_types {
            total += match k {
                LayerKind::Linear => dense + linear,
                LayerKind::Full => dense + full,
            };
        }
        total + self.head_params()
    }

    pub fn total_params(&self) -> usize {
        if !self.is_moe() {
            return self.num_hidden_layers * (self.dense_attn_params() + self.dense_mlp_params())
                + self.head_params();
        }
        let h = self.hidden_size;
        let experts = self.num_experts * (2 * self.moe_intermediate_size * h + h * self.moe_intermediate_size);
        let shared = 3 * self.shared_expert_intermediate_size * h + h;
        let router = self.num_experts * h;
        let dense = experts + shared + router;
        let linear = self.conv_dim() * h
            + self.value_dim() * h
            + 2 * self.linear_num_value_heads * h
            + h * self.value_dim()
            + self.conv_dim() * self.linear_conv_kernel_dim;
        let full = {
            let q = 2 * self.num_attention_heads * self.head_dim * h;
            let kv = 2 * self.num_key_value_heads * self.head_dim * h;
            let o = self.num_attention_heads * self.head_dim * h;
            q + kv + o
        };
        let mut total = 0usize;
        for k in &self.layer_types {
            total += match k {
                LayerKind::Linear => dense + linear,
                LayerKind::Full => dense + full,
            };
        }
        total + self.head_params()
    }

    pub fn from_json(raw: &Value) -> Result<TextConfig> {
        let tc = raw.get("text_config").unwrap_or(raw);
        let g = |k: &str| -> Result<f64> {
            tc.get(k)
                .and_then(Value::as_f64)
                .ok_or_else(|| format!("config: missing numeric key `{k}`"))
        };
        let gi = |k: &str| -> Result<usize> {
            g(k).map(|v| v as usize)
        };
        // Keys that only a routed-expert or gated-delta-net checkpoint carries.
        // Absent reads as zero, which is what makes `is_moe` and `has_gdn` the
        // only place the dense lane has to ask about them.
        let go = |k: &str| -> usize { tc.get(k).and_then(Value::as_f64).unwrap_or(0.0) as usize };
        // A dense Qwen3 ships no `layer_types` at all in some exports and an
        // all-full list in others; both mean plain attention in every layer.
        let n_layers = gi("num_hidden_layers")?;
        let layer_types = match tc.get("layer_types").and_then(Value::as_array) {
            Some(list) => list
                .iter()
                .map(|v| match v.as_str() {
                    Some("linear_attention") => Ok(LayerKind::Linear),
                    Some("full_attention") => Ok(LayerKind::Full),
                    other => Err(format!("config: unknown layer type {other:?}")),
                })
                .collect::<Result<Vec<_>>>()?,
            None => vec![LayerKind::Full; n_layers],
        };
        let rope = tc
            .get("rope_parameters")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));
        // transformers reads `rope_theta` from the top level for Qwen3 and only
        // from `rope_parameters` for the newer families. Guessing 10000 when the
        // checkpoint says 1000000 does not fail: it mis-rotates every attention
        // pair and produces fluent nonsense, so the default is the last resort.
        let mut rope_theta = tc
            .get("rope_theta")
            .and_then(Value::as_f64)
            .unwrap_or(10_000.0);
        let mut partial = tc
            .get("partial_rotary_factor")
            .and_then(Value::as_f64)
            .unwrap_or(1.0);
        if let Some(o) = rope.as_object() {
            if let Some(v) = o.get("rope_theta").and_then(Value::as_f64) {
                rope_theta = v;
            }
            if let Some(v) = o.get("partial_rotary_factor").and_then(Value::as_f64) {
                partial = v;
            }
        }
        // this checkpoint's head_dim (256) is not hidden/heads (128), so the
        // explicit key wins and the quotient is only a fallback
        let head_dim = match tc.get("head_dim").and_then(Value::as_f64) {
            Some(v) => v as usize,
            None => gi("hidden_size")? / gi("num_attention_heads")?,
        };
        // The RMSNorm dialect is a property of the model class, and `model_type`
        // is the only place that class is named: `Qwen3RMSNorm` multiplies the
        // stored gain, `Qwen3_5MoeRMSNorm` adds one to it first. Guessing wrong
        // is not a rounding error - every normalised activation comes out
        // doubled, which is the same fluent nonsense a wrong rope base makes.
        let mtype = tc
            .get("model_type")
            .and_then(Value::as_str)
            .or_else(|| raw.get("model_type").and_then(Value::as_str))
            .unwrap_or("unknown")
            .to_string();
        let zero_centered = !matches!(mtype.as_str(), "qwen3" | "qwen3_moe");
        Ok(TextConfig {
            hidden_size: gi("hidden_size")?,
            num_hidden_layers: n_layers,
            vocab_size: gi("vocab_size")?,
            rms_norm_eps: g("rms_norm_eps").unwrap_or(1e-6),
            norm_zero_centered: zero_centered,
            layer_types,
            num_attention_heads: gi("num_attention_heads")?,
            num_key_value_heads: gi("num_key_value_heads")?,
            head_dim,
            partial_rotary_factor: partial,
            rope_theta,
            attn_output_gate: tc
                .get("attn_output_gate")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            intermediate_size: go("intermediate_size"),
            tie_word_embeddings: tc
                .get("tie_word_embeddings")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            linear_num_key_heads: go("linear_num_key_heads"),
            linear_num_value_heads: go("linear_num_value_heads"),
            linear_key_head_dim: go("linear_key_head_dim"),
            linear_value_head_dim: go("linear_value_head_dim"),
            linear_conv_kernel_dim: go("linear_conv_kernel_dim"),
            num_experts: go("num_experts"),
            num_experts_per_tok: go("num_experts_per_tok"),
            moe_intermediate_size: go("moe_intermediate_size"),
            shared_expert_intermediate_size: go("shared_expert_intermediate_size"),
            prefix: DEFAULT_PREFIX.to_string(),
            name: mtype,
        })
    }

    pub fn from_file(path: &str) -> Result<TextConfig> {
        let body = std::fs::read_to_string(path).map_err(|e| format!("config: {path}: {e}"))?;
        let raw: Value = serde_json::from_str(&body).map_err(|e| format!("config: {path}: {e}"))?;
        TextConfig::from_json(&raw)
    }
}
