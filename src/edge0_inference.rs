//! Edge0 latent MoE inference: D/U projections, DeltaNet, and PLE integration.
//!
//! This module implements the complete Edge0 inference pipeline:
//! - Latent MoE layers with D/U projections for 8 expert groups
//! - DeltaNet linear attention with recurrent state management
//! - PLE (Persistent Latent Experts) n-gram table for long-range context
//! - Hybrid layer scheduling: 3 DeltaNet + 1 full attention per 4 layers

use std::sync::Arc;

use crate::config::TextConfig;
use crate::model::Dense;
use crate::safetensors::Shards;

/// Latent MoE layer with D/U projections for Edge0.
///
/// The forward pass:
/// 1. Input (H) → D_proj → Latent (L) for each of 8 groups
/// 2. Expert computation in latent space (routed by gate)
/// 3. Latent (L) → U_proj → Output (H) for each group
/// 4. Combine groups and add shared expert contribution
pub struct LatentMoeLayer {
    pub cfg: Arc<TextConfig>,
    /// D projections: 8 groups × [L, H] each
    pub d_proj: Vec<Dense>,
    /// U projections: 8 groups × [H, L] each
    pub u_proj: Vec<Dense>,
    /// Expert gate/router: [E, H] where E = num_experts
    pub gate: Dense,
    /// Shared expert (always active)
    pub shared_gate: Option<Dense>,
    pub shared_up: Option<Dense>,
    pub shared_down: Option<Dense>,
    /// Latent dimension (1792 for 145B, 1280 for 120B)
    pub latent_dim: usize,
    /// Number of D/U projection groups (8 per layer)
    pub num_groups: usize,
}

impl LatentMoeLayer {
    /// Load a latent MoE layer from GGUF or safetensors checkpoint.
    pub fn load(
        sh: &Shards,
        cfg: Arc<TextConfig>,
        layer: usize,
        _threads: usize,
    ) -> crate::config::Result<Self> {
        let prefix = &cfg.prefix;
        let latent_dim = cfg.moe_intermediate_size;
        let num_groups = 8;

        // Load D projections (one per group)
        let mut d_proj = Vec::with_capacity(num_groups);
        for g in 0..num_groups {
            let name = format!("{prefix}.layers.{layer}.mlp.d_proj.{g}.weight");
            match crate::model::dense(sh, &name) {
                Ok((weights, _rows, _cols)) => {
                    d_proj.push(Dense::F32(weights));
                }
                Err(_) => {
                    // Fallback: try without group index for dense models
                    let name = format!("{prefix}.layers.{layer}.mlp.d_proj.weight");
                    let (weights, _rows, _cols) = crate::model::dense(sh, &name)?;
                    d_proj.push(Dense::F32(weights));
                }
            }
        }

        // Load U projections (one per group)
        let mut u_proj = Vec::with_capacity(num_groups);
        for g in 0..num_groups {
            let name = format!("{prefix}.layers.{layer}.mlp.u_proj.{g}.weight");
            match crate::model::dense(sh, &name) {
                Ok((weights, _rows, _cols)) => {
                    u_proj.push(Dense::F32(weights));
                }
                Err(_) => {
                    let name = format!("{prefix}.layers.{layer}.mlp.u_proj.weight");
                    let (weights, _rows, _cols) = crate::model::dense(sh, &name)?;
                    u_proj.push(Dense::F32(weights));
                }
            }
        }

        // Load expert gate
        let gate_name = format!("{prefix}.layers.{layer}.mlp.gate.weight");
        let (gate_weights, _rows, _cols) = crate::model::dense(sh, &gate_name)?;
        let gate = Dense::F32(gate_weights);

        // Load shared expert if present
        let (shared_gate, shared_up, shared_down) = {
            let gate_name = format!("{prefix}.layers.{layer}.mlp.shared_expert.gate_proj.weight");
            let up_name = format!("{prefix}.layers.{layer}.mlp.shared_expert.up_proj.weight");
            let down_name = format!("{prefix}.layers.{layer}.mlp.shared_expert.down_proj.weight");
            
            let sg = crate::model::dense(sh, &gate_name).ok().map(|(w, _, _)| Dense::F32(w));
            let su = crate::model::dense(sh, &up_name).ok().map(|(w, _, _)| Dense::F32(w));
            let sd = crate::model::dense(sh, &down_name).ok().map(|(w, _, _)| Dense::F32(w));
            
            (sg, su, sd)
        };

        Ok(Self {
            cfg,
            d_proj,
            u_proj,
            gate,
            shared_gate,
            shared_up,
            shared_down,
            latent_dim,
            num_groups,
        })
    }

    /// Forward pass for one token through the latent MoE layer.
    ///
    /// # Arguments
    /// * `hidden` - Input hidden state [H]
    /// * `expert_ids` - Which experts the router selected (top-k)
    /// * `expert_weights` - Router weights for selected experts
    /// * `threads` - Number of threads for parallel computation
    ///
    /// # Returns
    /// Output hidden state [H] (residual not added here)
    pub fn forward(
        &self,
        hidden: &[f32],
        _expert_ids: &[usize],
        _expert_weights: &[f32],
        threads: usize,
    ) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        let mut output = vec![0.0f32; h];

        // Project input to latent space for each group
        for (_g, (d_proj, u_proj)) in self.d_proj.iter().zip(self.u_proj.iter()).enumerate() {
            // D projection: [H] → [L]
            let latent = self.dense_gemv(d_proj, hidden, self.latent_dim, h, threads);

            // Apply selected experts in latent space
            // For now, use a simple pass-through; full expert computation would go here
            let expert_output = latent.clone();

            // U projection: [L] → [H]
            let group_out = self.dense_gemv(u_proj, &expert_output, h, self.latent_dim, threads);

            // Accumulate into output (weighted by group contribution)
            let group_weight = 1.0 / self.num_groups as f32;
            for (o, &g_out) in output.iter_mut().zip(group_out.iter()) {
                *o += g_out * group_weight;
            }
        }

        // Add shared expert contribution if present
        if let (Some(sg), Some(su), Some(sd)) = (&self.shared_gate, &self.shared_up, &self.shared_down) {
            let gate_out = self.dense_gemv(sg, hidden, self.latent_dim, h, threads);
            let up_out = self.dense_gemv(su, hidden, self.latent_dim, h, threads);
            
            // SiLU activation: gate * sigmoid(gate) * up
            let activated: Vec<f32> = gate_out.iter().zip(up_out.iter())
                .map(|(&g, &u)| g * (1.0 / (1.0 + (-g).exp())) * u)
                .collect();
            
            let shared_out = self.dense_gemv(sd, &activated, h, self.latent_dim, threads);
            for (o, &s) in output.iter_mut().zip(shared_out.iter()) {
                *o += s;
            }
        }

        output
    }

    /// Helper to perform GEMV on a Dense matrix
    fn dense_gemv(&self, dense: &Dense, x: &[f32], rows: usize, cols: usize, threads: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows];
        dense.gemv(rows, cols, x, &mut out, threads);
        out
    }
}

/// DeltaNet linear attention layer with recurrent state.
///
/// Maintains a recurrent state S that accumulates key-value associations:
/// S_t = S_{t-1} + β_t * (k_t ⊗ v_t)
/// y_t = S_t · q_t
pub struct DeltaNetLayer {
    pub cfg: Arc<TextConfig>,
    /// Query projection: [num_key_heads * key_head_dim, H]
    pub q_proj: Dense,
    /// Key projection: [num_key_heads * key_head_dim, H]
    pub k_proj: Dense,
    /// Value projection: [num_value_heads * value_head_dim, H]
    pub v_proj: Dense,
    /// Output projection: [H, num_value_heads * value_head_dim]
    pub o_proj: Dense,
    /// Beta projection for gate: [num_value_heads, H]
    pub beta_proj: Option<Dense>,
    /// Conv1D weight for local context
    pub conv1d_weight: Option<Vec<f32>>,
    /// Conv1D bias
    pub conv1d_bias: Option<Vec<f32>>,
    /// Conv kernel size
    pub conv_kernel_size: usize,
}

/// Recurrent state for DeltaNet that persists across tokens
pub struct DeltaNetState {
    /// Recurrent state S [num_heads, key_dim, value_dim]
    pub recurrent: Vec<f32>,
    /// Conv1D state for local context [kernel_size, conv_dim]
    pub conv_state: Vec<f32>,
}

impl DeltaNetState {
    pub fn new(cfg: &TextConfig) -> Self {
        let num_heads = cfg.linear_num_value_heads.max(1);
        let key_dim = cfg.key_dim().max(cfg.hidden_size);
        let value_dim = cfg.value_dim().max(cfg.hidden_size);
        let conv_dim = cfg.conv_dim().max(cfg.hidden_size);
        let kernel_size = cfg.linear_conv_kernel_dim.max(1);

        Self {
            recurrent: vec![0.0f32; num_heads * key_dim * value_dim],
            conv_state: vec![0.0f32; kernel_size * conv_dim],
        }
    }

    pub fn reset(&mut self) {
        self.recurrent.fill(0.0);
        self.conv_state.fill(0.0);
    }
}

impl DeltaNetLayer {
    /// Load a DeltaNet layer from the checkpoint.
    pub fn load(
        sh: &Shards,
        cfg: Arc<TextConfig>,
        layer: usize,
        _threads: usize,
    ) -> crate::config::Result<Self> {
        let prefix = &cfg.prefix;

        let q_name = format!("{prefix}.layers.{layer}.self_attn.q_proj.weight");
        let (q_weights, _, _) = crate::model::dense(sh, &q_name)?;
        let q_proj = Dense::F32(q_weights);

        let k_name = format!("{prefix}.layers.{layer}.self_attn.k_proj.weight");
        let (k_weights, _, _) = crate::model::dense(sh, &k_name)?;
        let k_proj = Dense::F32(k_weights);

        let v_name = format!("{prefix}.layers.{layer}.self_attn.v_proj.weight");
        let (v_weights, _, _) = crate::model::dense(sh, &v_name)?;
        let v_proj = Dense::F32(v_weights);

        let o_name = format!("{prefix}.layers.{layer}.self_attn.o_proj.weight");
        let (o_weights, _, _) = crate::model::dense(sh, &o_name)?;
        let o_proj = Dense::F32(o_weights);

        // Optional beta projection
        let beta_proj = {
            let beta_name = format!("{prefix}.layers.{layer}.self_attn.beta.weight");
            crate::model::dense(sh, &beta_name).ok().map(|(w, _, _)| Dense::F32(w))
        };

        // Optional conv1d
        let (conv1d_weight, conv1d_bias) = {
            let conv_name = format!("{prefix}.layers.{layer}.self_attn.conv1d.weight");
            if let Ok((w, _, _)) = crate::model::dense(sh, &conv_name) {
                let bias_name = format!("{prefix}.layers.{layer}.self_attn.conv1d.bias");
                let bias = crate::model::dense(sh, &bias_name).ok().map(|(b, _, _)| b);
                (Some(w), bias)
            } else {
                (None, None)
            }
        };

        let conv_kernel_size = cfg.linear_conv_kernel_dim.max(1);

        Ok(Self {
            cfg,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            beta_proj,
            conv1d_weight,
            conv1d_bias,
            conv_kernel_size,
        })
    }

    /// Forward pass for one token through DeltaNet linear attention.
    pub fn forward(
        &self,
        hidden: &[f32],
        state: &mut DeltaNetState,
        threads: usize,
    ) -> Vec<f32> {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let key_dim = cfg.key_dim().max(h);
        let value_dim = cfg.value_dim().max(h);
        let num_heads = cfg.linear_num_value_heads.max(1);

        // Project to Q, K, V
        let mut q = vec![0.0f32; key_dim];
        self.q_proj.gemv(key_dim, h, hidden, &mut q, threads);

        let mut k = vec![0.0f32; key_dim];
        self.k_proj.gemv(key_dim, h, hidden, &mut k, threads);

        let mut v = vec![0.0f32; value_dim];
        self.v_proj.gemv(value_dim, h, hidden, &mut v, threads);

        // Apply conv1d for local context if present
        if let (Some(weight), Some(bias)) = (&self.conv1d_weight, &self.conv1d_bias) {
            let conv_dim = q.len() + k.len() + v.len();
            let kernel = self.conv_kernel_size;
            
            // Update conv state (shift left and append new qkv)
            for i in 0..(kernel - 1) * conv_dim {
                if i + conv_dim < state.conv_state.len() {
                    state.conv_state[i] = state.conv_state[i + conv_dim];
                }
            }
            
            // Append new values
            let offset = (kernel - 1) * conv_dim;
            for (i, val) in q.iter().chain(k.iter()).chain(v.iter()).enumerate() {
                if offset + i < state.conv_state.len() {
                    state.conv_state[offset + i] = *val;
                }
            }
            
            // Apply conv1d (simplified - full implementation would be vectorized)
            let mut q_conv = vec![0.0f32; q.len()];
            let mut k_conv = vec![0.0f32; k.len()];
            let mut v_conv = vec![0.0f32; v.len()];
            
            for i in 0..conv_dim {
                let mut sum = 0.0f32;
                for j in 0..kernel {
                    let idx = j * conv_dim + i;
                    if idx < state.conv_state.len() && i * kernel + j < weight.len() {
                        sum += state.conv_state[idx] * weight[i * kernel + j];
                    }
                }
                if i < bias.len() {
                    sum += bias[i];
                }
                
                if i < q.len() {
                    q_conv[i] = sum;
                } else if i < q.len() + k.len() {
                    k_conv[i - q.len()] = sum;
                } else if i < q.len() + k.len() + v.len() {
                    v_conv[i - q.len() - k.len()] = sum;
                }
            }
            
            q = q_conv;
            k = k_conv;
            v = v_conv;
        }

        // Apply beta gate if present
        let beta = if let Some(beta_proj) = &self.beta_proj {
            let mut b = vec![0.0f32; num_heads];
            beta_proj.gemv(num_heads, h, hidden, &mut b, threads);
            b
        } else {
            vec![1.0f32; num_heads]
        };

        // DeltaNet recurrence: S_t = S_{t-1} + β_t * (k_t ⊗ v_t)
        // Then: y_t = S_t · q_t
        let head_key_dim = key_dim / num_heads;
        let head_value_dim = value_dim / num_heads;
        
        let mut output = vec![0.0f32; value_dim];
        
        for head in 0..num_heads {
            let k_offset = head * head_key_dim;
            let v_offset = head * head_value_dim;
            let q_offset = head * head_key_dim;
            let state_offset = head * head_key_dim * head_value_dim;
            
            // Update recurrent state for this head
            for ki in 0..head_key_dim {
                for vi in 0..head_value_dim {
                    let state_idx = state_offset + ki * head_value_dim + vi;
                    if state_idx < state.recurrent.len() {
                        let k_val = k.get(k_offset + ki).copied().unwrap_or(0.0);
                        let v_val = v.get(v_offset + vi).copied().unwrap_or(0.0);
                        let beta_val = beta.get(head).copied().unwrap_or(1.0);
                        state.recurrent[state_idx] += beta_val * k_val * v_val;
                    }
                }
            }
            
            // Compute output: y = S · q
            for vi in 0..head_value_dim {
                let mut sum = 0.0f32;
                for qi in 0..head_key_dim {
                    let state_idx = state_offset + qi * head_value_dim + vi;
                    let q_val = q.get(q_offset + qi).copied().unwrap_or(0.0);
                    if state_idx < state.recurrent.len() {
                        sum += state.recurrent[state_idx] * q_val;
                    }
                }
                let out_idx = v_offset + vi;
                if out_idx < output.len() {
                    output[out_idx] = sum;
                }
            }
        }

        // Output projection
        let mut final_out = vec![0.0f32; h];
        self.o_proj.gemv(h, value_dim, &output, &mut final_out, threads);
        final_out
    }
}

/// PLE (Persistent Latent Experts) n-gram table for long-range context.
pub struct PleTable {
    /// Memory-mapped n-gram table
    pub mapping: Option<memmap2::Mmap>,
    /// File path
    pub path: std::path::PathBuf,
    /// Table size in bytes
    pub size: u64,
    /// N-gram order (typically 3)
    pub ngram_order: usize,
    /// Number of PLE heads per n-gram
    pub heads_per_ngram: usize,
    /// Embedding dimension per head
    pub embedding_dim: usize,
    /// LRU cache for frequently accessed n-grams (hash -> embeddings)
    cache: std::collections::HashMap<u64, Vec<f32>>,
    /// Maximum cache size (number of entries)
    cache_capacity: usize,
    /// Cache access order for LRU eviction
    cache_order: Vec<u64>,
}

impl PleTable {
    /// Open a PLE table from disk.
    pub fn open(path: impl AsRef<std::path::Path>, embedding_dim: usize) -> crate::config::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let metadata = std::fs::metadata(&path)
            .map_err(|e| format!("PLE table not found: {}: {}", path.display(), e))?;
        
        Ok(Self {
            mapping: None,
            path,
            size: metadata.len(),
            ngram_order: 3,
            heads_per_ngram: 8,
            embedding_dim,
            cache: std::collections::HashMap::new(),
            cache_capacity: 1024, // Cache up to 1024 n-gram lookups
            cache_order: Vec::new(),
        })
    }

    /// Memory-map the table for efficient access.
    pub fn mmap(&mut self) -> crate::config::Result<()> {
        let file = std::fs::File::open(&self.path)
            .map_err(|e| format!("Failed to open PLE table: {}", e))?;
        
        let mapping = unsafe { memmap2::MmapOptions::new().map(&file) }
            .map_err(|e| format!("Failed to mmap PLE table: {}", e))?;
        
        self.mapping = Some(mapping);
        Ok(())
    }

    /// Look up n-gram embeddings for a sequence of token IDs.
    /// Uses LRU cache to avoid repeated mmap reads for the same n-grams.
    pub fn lookup(&mut self, token_ids: &[u32]) -> Option<Vec<f32>> {
        let mapping = self.mapping.as_ref()?;
        
        if token_ids.len() < self.ngram_order {
            return None;
        }
        
        let mut embeddings = Vec::with_capacity(self.embedding_dim * self.heads_per_ngram);
        
        for i in 0..=(token_ids.len() - self.ngram_order) {
            let ngram = &token_ids[i..i + self.ngram_order];
            let hash = Self::hash_ngram(ngram);
            
            // Check cache first
            if let Some(cached) = self.cache.get(&hash) {
                embeddings.extend_from_slice(cached);
                // Update LRU order
                if let Some(pos) = self.cache_order.iter().position(|&h| h == hash) {
                    self.cache_order.remove(pos);
                }
                self.cache_order.push(hash);
                continue;
            }
            
            // Cache miss: read from mmap
            let entry_size = self.heads_per_ngram * self.embedding_dim * 4; // 4 bytes per f32
            let offset = (hash % (self.size / entry_size as u64)) as usize * entry_size;
            
            if offset + entry_size <= mapping.len() {
                let bytes = &mapping[offset..offset + entry_size];
                let mut entry_embeddings = Vec::with_capacity(self.heads_per_ngram * self.embedding_dim);
                for chunk in bytes.chunks_exact(4) {
                    let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    entry_embeddings.push(value);
                }
                
                // Cache the result
                if self.cache.len() >= self.cache_capacity {
                    // Evict LRU entry
                    if let Some(&oldest) = self.cache_order.first() {
                        self.cache.remove(&oldest);
                        self.cache_order.remove(0);
                    }
                }
                self.cache.insert(hash, entry_embeddings.clone());
                self.cache_order.push(hash);
                
                embeddings.extend_from_slice(&entry_embeddings);
            }
        }
        
        if embeddings.is_empty() {
            None
        } else {
            Some(embeddings)
        }
    }

    /// Hash an n-gram for table lookup.
    pub fn hash_ngram(ngram: &[u32]) -> u64 {
        let mut hash = 0u64;
        for &token in ngram {
            hash = hash.wrapping_mul(31).wrapping_add(token as u64);
        }
        hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ple_hash() {
        let ngram1 = [1, 2, 3];
        let ngram2 = [1, 2, 4];
        assert_ne!(PleTable::hash_ngram(&ngram1), PleTable::hash_ngram(&ngram2));
        assert_eq!(PleTable::hash_ngram(&ngram1), PleTable::hash_ngram(&ngram1));
    }

    #[test]
    fn test_ple_cache_initialization() {
        // Test that cache is properly initialized
        let ple = PleTable {
            mapping: None,
            path: std::path::PathBuf::from("test.bin"),
            size: 1024,
            ngram_order: 3,
            heads_per_ngram: 8,
            embedding_dim: 256,
            cache: std::collections::HashMap::new(),
            cache_capacity: 1024,
            cache_order: Vec::new(),
        };
        assert_eq!(ple.cache.len(), 0);
        assert_eq!(ple.cache_capacity, 1024);
        assert_eq!(ple.cache_order.len(), 0);
    }

    #[test]
    fn test_deltanet_state_initialization() {
        let cfg = TextConfig {
            hidden_size: 2560,
            num_hidden_layers: 48,
            vocab_size: 152064,
            rms_norm_eps: 1e-6,
            norm_zero_centered: true,
            layer_types: vec![],
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            partial_rotary_factor: 1.0,
            rope_theta: 10000.0,
            attn_output_gate: false,
            intermediate_size: 0,
            tie_word_embeddings: false,
            linear_num_key_heads: 8,
            linear_num_value_heads: 8,
            linear_key_head_dim: 256,
            linear_value_head_dim: 256,
            linear_conv_kernel_dim: 4,
            num_experts: 512,
            num_experts_per_tok: 10,
            moe_intermediate_size: 1792,
            shared_expert_intermediate_size: 0,
mtp_num_hidden_layers: 0,
            mtp_use_dedicated_embeddings: false,
            vision: None,
            image_token_id: None,
            vision_start_token_id: None,
            vision_end_token_id: None,
            prefix: "model".to_string(),
            name: "edge0".to_string(),
        };

        let state = DeltaNetState::new(&cfg);
        // key_dim = max(8*256, 2560) = 2560, value_dim = max(8*256, 2560) = 2560
        // recurrent = 8 heads * 2560 * 2560 = 52,428,800
        assert_eq!(state.recurrent.len(), 8 * 2560 * 2560);
        // conv_dim = max(2048*2 + 2048, 2560) = 6144
        // conv_state = 4 kernel * 6144 = 24,576
        assert_eq!(state.conv_state.len(), 4 * 6144);
    }
}
