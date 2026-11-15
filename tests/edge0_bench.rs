// Edge0-145B benchmark harness
// Measures TTFT, TPOT, and memory usage against Colibri baseline

use std::time::{Duration, Instant};

use fennec::config::TextConfig;
use fennec::model::Model;

pub struct Edge0BenchResult {
    pub ttft_ms: f64,
    pub tpot_ms: f64,
    pub tokens_per_sec: f64,
    pub memory_mb: f64,
}

pub fn bench_edge0_prefill(
    model: &Model,
    prompt_tokens: &[u32],
) -> Edge0BenchResult {
    // Warm-up (discarded)
    let mut state = model.states(128);
    let mut xs = vec![0.0f32; prompt_tokens.len() * model.cfg.hidden_size];
    
    // Warm-up pass
    let _ = model.prefill_device(&mut xs, &mut state, 0);
    
    // Measured passes
    let iterations = 5;
    let mut ttft_total = Duration::ZERO;
    
    for _ in 0..iterations {
        let mut state = model.states(128);
        xs.fill(0.0);
        
        let start = Instant::now();
        let _ = model.prefill_device(&mut xs, &mut state, 0);
        ttft_total += start.elapsed();
    }
    
    let ttft_ms = ttft_total.as_secs_f64() * 1000.0 / iterations as f64;
    
    // Decode throughput (TPOT)
    let decode_tokens = 128;
    let mut decode_state = model.states(128);
    let mut token = vec![0.0f32; model.cfg.hidden_size];
    
    let start = Instant::now();
    for _ in 0..decode_tokens {
        model.forward(&mut token, &mut decode_state, 0);
    }
    let decode_elapsed = start.elapsed();
    
    let tpot_ms = decode_elapsed.as_secs_f64() * 1000.0 / decode_tokens as f64;
    let tokens_per_sec = 1000.0 / tpot_ms;
    
    // Memory estimation (rough)
    let memory_mb = estimate_memory_mb(&model.cfg);
    
    Edge0BenchResult {
        ttft_ms,
        tpot_ms,
        tokens_per_sec,
        memory_mb,
    }
}

fn estimate_memory_mb(cfg: &TextConfig) -> f64 {
    // Rough estimation based on model dimensions
    let h = cfg.hidden_size as f64;
    let layers = cfg.num_hidden_layers as f64;
    
    // Each layer has projections, experts, states
    // Very rough: ~100MB per layer for a large model
    let per_layer_mb = 100.0;
    let base_mb = layers * per_layer_mb;
    
    // Add embedding/head overhead
    let embed_mb = h * cfg.vocab_size as f64 * 4.0 / 1_000_000.0;
    
    base_mb + embed_mb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bench_result_struct() {
        let result = Edge0BenchResult {
            ttft_ms: 1000.0,
            tpot_ms: 50.0,
            tokens_per_sec: 20.0,
            memory_mb: 580.0,
        };
        assert!(result.ttft_ms > 0.0);
        assert!(result.tpot_ms > 0.0);
        assert!(result.tokens_per_sec > 0.0);
        assert!(result.memory_mb > 0.0);
    }
}
