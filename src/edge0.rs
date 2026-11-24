//! Edge0 model support documentation and detection.
//!
//! Edge0-145B and Edge0-120B are latent MoE models based on Qwen3.8-Flash-Next.
//! They use D/U projections to narrow the expert width, reducing memory while
//! maintaining quality.
//!
//! # Architecture
//!
//! - Base: Qwen3.8-Flash-Next (125B + 51B PLE table)
//! - Latent MoE: experts narrowed from 2560 to 1792 (145B) or 1280 (120B)
//! - D/U projections: 8 groups per layer, shared across experts
//! - PLE: 51.2B n-gram embedding table (stays on SSD)
//! - Active params per token: 6.6B (145B) or 6.1B (120B)
//!
//! # File Format
//!
//! Ships as GGUF files (Strata engine format):
//! - `model-skeleton.gguf`: attention, routers, shared experts, embeddings
//! - `experts.bin`: routed expert pool (quantized)
//! - `latent_dense_q8.bin`: D/U projections
//! - `ple-fp8.gguf`: n-gram embedding table
//! - `mtp/`: multi-token prediction draft layer
//!
//! # Implementation Status
//!
//! This module provides:
//! - GGUF format loader (`gguf.rs`)
//! - Model detection from metadata
//! - Configuration templates
//!
//! Full Edge0 inference requires additional work in `model.rs`:
//! - Latent MoE layer with D/U projections
//! - PLE n-gram table streaming from SSD
//! - Hybrid attention (DeltaNet + standard)
//!
//! # References
//!
//! - https://huggingface.co/Edge0/Edge0-145B-NVFP4
//! - https://huggingface.co/Edge0/Edge0-120B-Q3_K
//! - https://github.com/Niko1221/Strata

use crate::config::{LayerKind, TextConfig};
use crate::gguf::GgufValue;
use std::collections::HashMap;

/// Detect Edge0 model from GGUF metadata and return appropriate config.
///
/// Returns None if the metadata doesn't indicate an Edge0 model.
pub fn detect_edge0_model(metadata: &HashMap<String, GgufValue>) -> Option<TextConfig> {
    let model_name = match metadata.get("general.name") {
        Some(GgufValue::String(s)) => s.as_str(),
        _ => return None,
    };
    
    if !model_name.contains("edge0") && !model_name.contains("Edge0") {
        return None;
    }
    
    // Check latent width to distinguish 145B from 120B
    let latent_width = match metadata.get("edge0.latent_width") {
        Some(GgufValue::Uint32(w)) => Some(*w as usize),
        Some(GgufValue::Uint64(w)) => Some(*w as usize),
        _ => None,
    };
    
    let config = match latent_width {
        Some(1792) => {
            eprintln!("Detected Edge0-145B (latent width 1792)");
            create_edge0_145b_config()
        }
        Some(1280) => {
            eprintln!("Detected Edge0-120B (latent width 1280)");
            create_edge0_120b_config()
        }
        _ => {
            eprintln!("Detected Edge0 model, defaulting to 145B configuration");
            create_edge0_145b_config()
        }
    };
    
    Some(config)
}

/// Create Edge0-145B configuration.
///
/// Specs:
/// - 145B total parameters
/// - 6.6B active per token
/// - 48 layers, 512 experts, 10 active
/// - Latent width: 1792 (70% of base 2560)
/// - 8 D/U projection groups per layer
/// - 256K context window
/// - Hybrid: 3 DeltaNet + 1 attention per 4 layers
fn create_edge0_145b_config() -> TextConfig {
    TextConfig {
        hidden_size: 2560,
        num_hidden_layers: 48,
        vocab_size: 320_001_536,
        rms_norm_eps: 1e-5,
        norm_zero_centered: true,
        
        num_attention_heads: 32,
        num_key_value_heads: 8,
        head_dim: 128,
        partial_rotary_factor: 1.0,
        rope_theta: 1_000_000.0,
        attn_output_gate: true,
        
        intermediate_size: 0,
        tie_word_embeddings: false,
        
        linear_num_key_heads: 0,
        linear_num_value_heads: 0,
        linear_key_head_dim: 0,
        linear_value_head_dim: 0,
        linear_conv_kernel_dim: 0,
        
        num_experts: 512,
        num_experts_per_tok: 10,
        moe_intermediate_size: 640,
        shared_expert_intermediate_size: 640,
mtp_num_hidden_layers: 0,
        mtp_use_dedicated_embeddings: false,
        vision: None,
        image_token_id: None,
        vision_start_token_id: None,
        vision_end_token_id: None,
        
        layer_types: generate_hybrid_layer_types(48),
        
        prefix: "model.".to_string(),
        name: "edge0-145b".to_string(),
    }
}

/// Create Edge0-120B configuration.
///
/// Same as 145B but with narrower latent width (1280 instead of 1792).
fn create_edge0_120b_config() -> TextConfig {
    TextConfig {
        hidden_size: 2560,
        num_hidden_layers: 48,
        vocab_size: 320_001_536,
        rms_norm_eps: 1e-5,
        norm_zero_centered: true,
        
        num_attention_heads: 32,
        num_key_value_heads: 8,
        head_dim: 128,
        partial_rotary_factor: 1.0,
        rope_theta: 1_000_000.0,
        attn_output_gate: true,
        
        intermediate_size: 0,
        tie_word_embeddings: false,
        
        linear_num_key_heads: 0,
        linear_num_value_heads: 0,
        linear_key_head_dim: 0,
        linear_value_head_dim: 0,
        linear_conv_kernel_dim: 0,
        
        num_experts: 512,
        num_experts_per_tok: 10,
        moe_intermediate_size: 640,
        shared_expert_intermediate_size: 640,
mtp_num_hidden_layers: 0,
        mtp_use_dedicated_embeddings: false,
        vision: None,
        image_token_id: None,
        vision_start_token_id: None,
        vision_end_token_id: None,
        
        layer_types: generate_hybrid_layer_types(48),
        
        prefix: "model.".to_string(),
        name: "edge0-120b".to_string(),
    }
}

/// Generate hybrid layer types: 3 DeltaNet (Linear) + 1 attention (Full) per 4 layers.
fn generate_hybrid_layer_types(num_layers: usize) -> Vec<LayerKind> {
    let mut types = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        if i % 4 == 3 {
            types.push(LayerKind::Full);
        } else {
            types.push(LayerKind::Linear);
        }
    }
    types
}

/// Edge0-specific parameters not in TextConfig.
///
/// These document what would be needed for full Edge0 support.
#[derive(Debug, Clone)]
pub struct Edge0Params {
    pub latent_width: usize,
    pub latent_groups: usize,
    pub ple_ngram_size: usize,
    pub ple_heads_per_ngram: usize,
    pub ple_embedding_per_layer: usize,
    pub quantization: String,
}

impl Edge0Params {
    pub fn edge0_145b() -> Self {
        Self {
            latent_width: 1792,
            latent_groups: 8,
            ple_ngram_size: 3,
            ple_heads_per_ngram: 8,
            ple_embedding_per_layer: 160,
            quantization: "nvfp4".to_string(),
        }
    }
    
    pub fn edge0_120b() -> Self {
        Self {
            latent_width: 1280,
            latent_groups: 8,
            ple_ngram_size: 3,
            ple_heads_per_ngram: 8,
            ple_embedding_per_layer: 160,
            quantization: "q3_k".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge0_145b_config() {
        let cfg = create_edge0_145b_config();
        assert_eq!(cfg.hidden_size, 2560);
        assert_eq!(cfg.num_experts, 512);
        assert_eq!(cfg.num_experts_per_tok, 10);
        assert_eq!(cfg.name, "edge0-145b");
    }

    #[test]
    fn test_edge0_120b_config() {
        let cfg = create_edge0_120b_config();
        assert_eq!(cfg.hidden_size, 2560);
        assert_eq!(cfg.num_experts, 512);
        assert_eq!(cfg.name, "edge0-120b");
    }

    #[test]
    fn test_hybrid_layer_types() {
        let types = generate_hybrid_layer_types(8);
        assert_eq!(types[0], LayerKind::Linear);
        assert_eq!(types[1], LayerKind::Linear);
        assert_eq!(types[2], LayerKind::Linear);
        assert_eq!(types[3], LayerKind::Full);
    }

    #[test]
    fn test_edge0_params() {
        let p145 = Edge0Params::edge0_145b();
        assert_eq!(p145.latent_width, 1792);
        
        let p120 = Edge0Params::edge0_120b();
        assert_eq!(p120.latent_width, 1280);
    }
}
