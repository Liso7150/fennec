//! Model configuration: the checkpoint's `config.json`, including the nested
//! `text_config` wrapper that `qwen3_5_moe` ships with.

use serde_json::Value;

pub type Error = String;
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub layer_types: Vec<LayerKind>,
    // attention
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub partial_rotary_factor: f64,
    pub rope_theta: f64,
    pub attn_output_gate: bool,
    // linear attention (gated delta net)
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    // moe
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
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

    /// Params that a single decode token actually touches, and the bytes that
    /// implies at a given bits-per-weight. This is the number that decides
    /// whether a 35B model runs at 2 tok/s or 20.
    pub fn active_params(&self) -> usize {
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
        total + h * self.vocab_size * 2 // embed + lm_head, untied
    }

    pub fn total_params(&self) -> usize {
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
        total + h * self.vocab_size * 2
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
        let layer_types = tc
            .get("layer_types")
            .and_then(Value::as_array)
            .ok_or("config: missing layer_types")?
            .iter()
            .map(|v| match v.as_str() {
                Some("linear_attention") => Ok(LayerKind::Linear),
                Some("full_attention") => Ok(LayerKind::Full),
                other => Err(format!("config: unknown layer type {other:?}")),
            })
            .collect::<Result<Vec<_>>>()?;
        let rope = tc
            .get("rope_parameters")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));
        let mut rope_theta = 10_000.0;
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
        Ok(TextConfig {
            hidden_size: gi("hidden_size")?,
            num_hidden_layers: gi("num_hidden_layers")?,
            vocab_size: gi("vocab_size")?,
            rms_norm_eps: g("rms_norm_eps").unwrap_or(1e-6),
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
            linear_num_key_heads: gi("linear_num_key_heads")?,
            linear_num_value_heads: gi("linear_num_value_heads")?,
            linear_key_head_dim: gi("linear_key_head_dim")?,
            linear_value_head_dim: gi("linear_value_head_dim")?,
            linear_conv_kernel_dim: gi("linear_conv_kernel_dim")?,
            num_experts: gi("num_experts")?,
            num_experts_per_tok: gi("num_experts_per_tok")?,
            moe_intermediate_size: gi("moe_intermediate_size")?,
            shared_expert_intermediate_size: gi("shared_expert_intermediate_size")?,
            name: tc
                .get("model_type")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string(),
        })
    }

    pub fn from_file(path: &str) -> Result<TextConfig> {
        let body = std::fs::read_to_string(path).map_err(|e| format!("config: {path}: {e}"))?;
        let raw: Value = serde_json::from_str(&body).map_err(|e| format!("config: {path}: {e}"))?;
        TextConfig::from_json(&raw)
    }
}
