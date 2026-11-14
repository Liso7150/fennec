# Edge0 Model Support

Fennec now supports the Edge0 family of latent MoE models, including Edge0-145B and Edge0-120B.

## What is Edge0?

Edge0 is a latent MoE (Mixture of Experts) architecture based on Qwen3.8-Flash-Next. Key characteristics:

- **Edge0-145B**: 145B total parameters, 6.6B active per token
- **Edge0-120B**: 120B total parameters, similar active count
- **512 experts per layer** with 10 active per token (vs typical 8/2)
- **Hybrid attention**: 3 DeltaNet (linear) layers + 1 standard attention layer per 4-layer block
- **Latent MoE**: D/U projections narrow expert width from 2560 to 1792 (145B) or 1280 (120B)
- **PLE n-gram table**: 51.2 GiB persistent latent expert table stored on SSD

## GGUF Format Support

Edge0 models are distributed in GGUF format (GPT-Generated Unified Format), the binary format used by llama.cpp and compatible engines.

Fennec's GGUF reader (`src/gguf.rs`) supports:

- **Tensor types**: F32, F16, BF16, Q8_0, Q4_0, Q3_K (partial)
- **Metadata**: Full GGUF key-value metadata parsing
- **Dequantization**: On-the-fly conversion from quantized formats to f32
- **Alignment**: 32-byte alignment per GGUF specification

### Usage

```rust
use fennec::gguf::GgufReader;

let mut reader = GgufReader::open("path/to/model.gguf")?;
println!("Model: {:?}", reader.model_name());
println!("Tensors: {}", reader.tensors.len());

// Load a tensor and dequantize to f32
let weights = reader.tensor_f32("model.layers.0.self_attn.q_proj.weight")?;
```

## Edge0 Detection

The `edge0` module detects Edge0 models from GGUF metadata and constructs the appropriate `TextConfig`:

```rust
use fennec::gguf::GgufReader;
use fennec::edge0::detect_edge0_model;

let reader = GgufReader::open("edge0-145b.gguf")?;
if let Some(config) = detect_edge0_model(&reader.metadata) {
    println!("Detected: {}", config.name);
    println!("Layers: {}", config.num_hidden_layers);
    println!("Experts: {} per layer, {} active", 
             config.num_experts, config.num_experts_per_tok);
}
```

## Architecture Details

### Latent MoE Layer

Edge0 uses a latent MoE structure with D/U projections:

```
Input (2560) → D_proj → Latent (1792) → 8 groups → Experts → U_proj → Output (2560)
```

Each layer has 8 D/U projection groups that narrow the expert computation width, reducing active parameters while maintaining capacity.

### Hybrid Attention

The model alternates between two layer types:

- **Linear (DeltaNet)**: 3 out of every 4 layers use gated delta net linear attention
- **Full**: 1 out of every 4 layers uses standard softmax attention

This hybrid approach balances efficiency (linear attention) with expressiveness (full attention).

### PLE (Persistent Latent Experts)

Edge0 maintains a 51.2 GiB n-gram table on SSD that stores persistent latent expert states. This allows the model to:

- Cache frequently-used expert computations
- Reduce redundant computation across tokens
- Maintain long-range context through expert activation patterns

## Configuration

Edge0-145B configuration:

```rust
TextConfig {
    hidden_size: 2560,
    num_hidden_layers: 48,
    vocab_size: 152064,
    num_attention_heads: 32,
    num_key_value_heads: 8,
    head_dim: 128,
    num_experts: 512,
    num_experts_per_tok: 10,
    moe_intermediate_size: 1792,
    layer_types: [Linear, Linear, Linear, Full, ...], // 3:1 ratio
    norm_zero_centered: true,
    // ...
}
```

## Implementation Status

### Completed

- ✅ GGUF format reader with dequantization
- ✅ Edge0 model detection from metadata
- ✅ Configuration generation for Edge0-145B and Edge0-120B
- ✅ Hybrid layer type support (Linear + Full attention)
- ✅ **Latent MoE layer with D/U projections**
  - 8 expert groups per layer
  - D projection: [H=2560] → [L=1792]
  - U projection: [L=1792] → [H=2560]
  - Shared expert path with SiLU activation
  - Group-weighted output combination
- ✅ **DeltaNet linear attention**
  - Q/K/V/O projections with optional beta gating
  - Conv1D for local context (kernel_size=4)
  - Recurrent state management across tokens
  - DeltaNet recurrence: S_t = S_{t-1} + β_t * (k_t ⊗ v_t)
- ✅ **PLE n-gram table**
  - Memory-mapped 51.2 GiB table
  - Trigram hashing for long-range context
  - 8 heads per n-gram
  - Lazy mmap loading
- ✅ **Full integration with model.rs**
  - Weight loading from GGUF/safetensors
  - Fallback paths for dense models
  - State management infrastructure

### Future Work

- ⏳ Optimized int4 quantization for latent experts
- ⏳ GPU kernels for D/U projection groups (Vulkan/Metal/CUDA)
- ⏳ Batched prefill for latent MoE
- ⏳ DeltaNet kernel optimization (vectorized recurrence)
- ⏳ PLE cache coherence and prefetching strategies
- ⏳ End-to-end inference benchmarks vs Colibri

## Performance Expectations

Edge0-145B activates only 6.6B parameters per token despite having 145B total. This should yield:

- **Decode speed**: Similar to a 7B model (active parameter count)
- **Memory**: ~30 GiB at Q4 (145B × 0.25 bytes/param + overhead)
- **Context**: Extended through PLE caching

Actual benchmarks pending full implementation.

## References

- Edge0 announcement: https://youtu.be/1PJGtLpYGOY
- GGUF format: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
- DeltaNet: https://arxiv.org/abs/2406.06484
- Latent MoE: https://arxiv.org/abs/2404.05217

## Files

- `src/gguf.rs` - GGUF format reader
- `src/edge0.rs` - Edge0 model detection and configuration
- `src/edge0_inference.rs` - Edge0 inference architecture (latent MoE, DeltaNet, PLE)
- `src/config.rs` - TextConfig structure (extended for Edge0)
- `src/lib.rs` - Module exports
