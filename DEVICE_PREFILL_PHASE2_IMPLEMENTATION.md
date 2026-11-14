# Device Prefill Batching Phase 2: Implementation Plan

**Status:** Ready for implementation  
**Priority:** High (expected ~620ms TTFT improvement)  
**Risk:** Medium-High (requires careful GPU synchronization)  
**Estimated Effort:** 11-16 hours (multi-day task)

---

## Executive Summary

Phase 1 infrastructure is complete and tested. The `prefill_device()` method exists at `src/model.rs:2734` but currently falls back to token-by-token forward passes. Phase 2 will replace this with actual batched GPU operations.

**Expected Impact:**
- TTFT improvement: ~620ms (1.644s → ~1.0s)
- Decode throughput: unchanged (±2%)
- Memory usage: unchanged or reduced

**Current Baseline (2026-11-14):**
- Colibri Vulkan: 20.160 tok/s
- Fennec CUDA: 17.529 tok/s (0.87x)
- Fennec Vulkan: 13.485 tok/s (0.67x)

---

## Implementation Strategy

### What Needs to Be Batched

The device forward pass currently processes tokens one at a time:
```rust
// Current: token-by-token (src/model.rs:2739-2743)
for i in 0..n {
    let mut x = xs[i * h..(i + 1) * h].to_vec();
    self.forward(&mut x, st, pos0 + i);
    xs[i * h..(i + 1) * h].copy_from_slice(&x);
}
```

Phase 2 will batch two categories of operations:
1. **Dense projections** (GdnLayer, AttnLayer): Batch GEMV operations over n tokens
2. **Expert routing** (MoeLayer): Group tokens by expert ID, process each expert once

### Why Host Batch Doesn't Work

Previous attempt measured host batch for device prompt: 2166ms vs device token-by-token: 1750ms.

**Root cause:** Host batch reads weights from system RAM, competing with host expert pool for bandwidth. Device tier needs device-side batching where weights are already in VRAM.

---

## Detailed Implementation Steps

### Step 1: Dense GEMV Batching (Medium Risk)

**Location:** `src/model.rs`, `GdnLayer` and `AttnLayer` implementations

**Current state:** Each layer has 5 input projections (in_qkv, in_z, in_b, in_a, out_proj) that are called per-token.

**Target state:** Batch these projections over n tokens using a single GPU command buffer.

#### 1.1 Add batched GEMV method to Dense

The `Dense::gemv_batch` method already exists at `src/model.rs:141`. Verify it works for device tensors:

```rust
// src/model.rs:141 (already exists)
pub fn gemv_batch(&self, rows: usize, cols: usize, xs: &[f32], outs: &mut [f32], threads: usize) {
    let n = xs.len() / cols;
    kernel::par_index(n, threads, |i0, cnt| {
        for i in i0..i0 + cnt {
            self.gemv(rows, cols, &xs[i * cols..], &mut outs[i * rows..], threads);
        }
    });
}
```

**Action:** Verify this works for device tensors. If not, add a device-specific batched path.

#### 1.2 Add device batch method to GdnLayer

Create `GdnLayer::step_batch_device`:

```rust
impl GdnLayer {
    pub fn step_batch_device(&self, xs: &[f32], states: &mut [LayerState], pos0: usize) {
        let h = self.cfg.hidden_size;
        let n = xs.len() / h;
        
        // Allocate output buffer for all tokens
        let mut out = vec![0f32; n * h];
        
        // Batch the 5 input projections
        // Each projection: (rows, cols, xs, out)
        self.in_qkv.gemv_batch_device(cd, h, xs, &mut qkv_out, n);
        self.in_z.gemv_batch_device(vd, h, xs, &mut z_out, n);
        self.in_b.gemv_batch_device(nv, h, xs, &mut b_out, n);
        self.in_a.gemv_batch_device(nv, h, xs, &mut a_out, n);
        
        // gdn_rec is recurrent, must be per-token
        for i in 0..n {
            self.gdn_rec(
                &qkv_out[i * cd..],
                &z_out[i * vd..],
                &b_out[i * nv..],
                &a_out[i * nv..],
                &mut states[i],
                &mut out[i * h..],
            );
        }
        
        // Batch the output projection
        self.out_proj.gemv_batch_device(h, vd, &out, &mut final_out, n);
        
        // Copy results back
        xs.copy_from_slice(&final_out);
    }
}
```

**Key insight:** The recurrent part (`gdn_rec`) cannot be batched because it depends on the previous token's state. Only the dense projections can be batched.

#### 1.3 Implement GPU-side batched GEMV

Add a method to `VkBatch` that records n GEMV operations in a single command buffer:

```rust
impl<'a> VkBatch<'a> {
    pub fn gemv_half_batch(&mut self, w: usize, rows: usize, cols: usize, 
                           x_offs: &[usize], y_offs: &[usize]) -> Result<&mut VkBatch<'a>> {
        assert_eq!(x_offs.len(), y_offs.len());
        let n = x_offs.len();
        let s = self.gpu.scratch;
        let op = if self.gpu.bufs[w].f16 { Op::GemvF16 } else { Op::GemvBf16 };
        let groups = (rows as u32 + 3) / 4;
        
        // Record all n GEMV operations without barriers between them
        for i in 0..n {
            self.gpu.record_nb(op, [w, s, s, w], 
                &[rows as u32, cols as u32, x_offs[i] as u32, y_offs[i] as u32],
                groups, groups, true)?;
        }
        
        // Single barrier after all operations
        self.gpu.barrier(PIPELINE_STAGE_COMPUTE_SHADER, PIPELINE_STAGE_COMPUTE_SHADER,
            ACCESS_SHADER_WRITE, ACCESS_SHADER_READ);
        
        self.ops += n;
        Ok(self)
    }
}
```

**Risk:** GPU synchronization. If barriers are missing or misplaced, results will be incorrect.

**Mitigation:** Test with small batch sizes first (n=2, 4, 8) and verify bit-identical results to token-by-token.

### Step 2: Expert Grouping (High Risk)

**Location:** `src/model.rs`, `MoeLayer::forward`

**Current state:** Each token is routed independently, and experts are loaded per-token.

**Target state:** Group all tokens by expert ID, load each expert once, process all its tokens.

#### 2.1 Route all tokens and collect picks

```rust
impl MoeLayer {
    pub fn forward_batch_device(&self, xs: &[f32], states: &mut [LayerState], pos0: usize) {
        let h = self.cfg.hidden_size;
        let n = xs.len() / h;
        
        // Route all tokens
        let mut routings = Vec::with_capacity(n);
        for i in 0..n {
            let x = &xs[i * h..(i + 1) * h];
            let routing = self.route(x);
            routings.push(routing);
        }
        
        // Group tokens by expert
        let mut expert_tokens: HashMap<usize, Vec<usize>> = HashMap::new();
        for (token_idx, routing) in routings.iter().enumerate() {
            for (pick_idx, &expert_id) in routing.ids.iter().enumerate() {
                expert_tokens.entry(expert_id)
                    .or_insert_with(Vec::new)
                    .push(token_idx);
            }
        }
        
        // Process each expert once
        let mut outputs = vec![vec![0f32; h]; n];
        for (expert_id, token_indices) in expert_tokens {
            // Load expert weights (once per expert)
            let expert = self.load_expert(expert_id);
            
            // Process all tokens for this expert
            for &token_idx in &token_indices {
                let x = &xs[token_idx * h..(token_idx + 1) * h];
                let out = expert.forward(x);
                outputs[token_idx] = out;
            }
        }
        
        // Reduce and write back
        for i in 0..n {
            xs[i * h..(i + 1) * h].copy_from_slice(&outputs[i]);
        }
    }
}
```

**Risk:** Memory pressure. Loading all expert weights simultaneously may exceed VRAM.

**Mitigation:** Process experts sequentially, not in parallel. The grouping is the optimization, not concurrent execution.

#### 2.2 Optimize expert loading

The key insight from Colibri's `moe()` is that each unique expert is loaded once. Currently, if 8 tokens all pick expert 42, it's loaded 8 times. With grouping, it's loaded once and processes 8 tokens.

**Implementation:** Use the existing `MoeLayer::prefill` logic (line 1050 in model.rs) as a reference. It already implements batch-union on the host.

### Step 3: Integration (Medium Risk)

#### 3.1 Update `prefill_device` to use batched operations

```rust
pub fn prefill_device(&self, xs: &mut [f32], st: &mut [LayerState], pos0: usize) {
    let h = self.cfg.hidden_size;
    let n = xs.len() / h;
    
    for (l, s) in self.layers.iter().zip(st.iter_mut()) {
        match (&l.kind, s) {
            (LayerKind::Linear, LayerState::Linear(g)) => {
                l.gdn.as_ref().unwrap().step_batch_device(xs, &mut [g.clone()], pos0);
            }
            (LayerKind::Full, LayerState::Full(a)) => {
                l.attn.as_ref().unwrap().step_batch_device(xs, &mut [a.clone()], pos0);
            }
            _ => panic!("layer state mismatch"),
        }
        
        // MoE layer
        if let Some(moe) = &l.moe {
            moe.forward_batch_device(xs, st, pos0);
        }
    }
}
```

#### 3.2 Add feature flag for safe rollout

```rust
// src/main.rs or src/serve.rs
let use_device_batch = std::env::var("FENNEC_DEVICE_BATCH").is_ok();

if batched && model.prefill_device_ready() {
    if use_device_batch {
        model.prefill_device_batched(&mut xs, &mut st, 0);
    } else {
        model.prefill_device(&mut xs, &mut st, 0);  // fallback
    }
}
```

---

## Testing Strategy

### Unit Tests

1. **Bit-identical output:** `prefill_device_batched` must produce identical results to `prefill_device` (token-by-token)
2. **Dense batching:** Verify `gemv_half_batch` matches n calls to `gemv_half`
3. **Expert grouping:** Verify grouped expert processing matches per-token processing

### Integration Tests

1. **Golden tests:** All 14 existing unit tests must pass
2. **GPU tests:** All 9 GPU tests must pass
3. **End-to-end:** Generate text with batched prefill, verify output matches baseline

### Benchmark Protocol

**Steady-state measurement (from engine-bench-hygiene.md):**
1. Run 2 warm-up iterations (discard)
2. Run 5 measured iterations
3. Report `steady_tokens_per_s` from the driver
4. Order-swapped A/B to detect drift (Colibri-Fennec-Colibri vs Fennec-Colibri-Fennec)

**Target metrics:**
- TTFT: 1.644s → ~1.0s (620ms improvement)
- Decode throughput: unchanged (±2%)
- Memory: unchanged or reduced

---

## Risk Mitigation

### What Could Go Wrong

1. **Synchronization bugs:** Missing GPU barriers between operations
2. **Memory pressure:** Batching may require more VRAM
3. **Correctness:** Batched operations may produce different results due to floating-point ordering
4. **Performance regression:** Batching overhead may outweigh benefits

### Safeguards

1. **Feature flag:** `FENNEC_DEVICE_BATCH` environment variable, off by default
2. **Fallback:** If batching fails or regresses, disable the flag
3. **Testing:** Run full test suite with batching enabled before merging
4. **Benchmarking:** Measure before/after with steady-state protocol

### Rollback Plan

If batching causes regressions:
1. Disable `FENNEC_DEVICE_BATCH` flag
2. Investigate root cause
3. Fix or revert the implementation
4. Re-benchmark to verify fix

---

## Success Criteria

### Must Have

- [ ] All 30 existing tests pass
- [ ] TTFT improves by at least 400ms (target: 620ms)
- [ ] Decode throughput unchanged (±2%)
- [ ] No new compiler warnings
- [ ] Bit-identical outputs to token-by-token (within floating-point tolerance)

### Nice to Have

- [ ] TTFT improves by 620ms+ (target achieved)
- [ ] Memory usage unchanged or reduced
- [ ] Code is well-documented with inline comments
- [ ] Benchmark results published in README

---

## Implementation Timeline

| Phase | Task | Estimated Time | Risk |
|-------|------|----------------|------|
| 1 | Dense GEMV batching | 3-4 hours | Medium |
| 2 | Expert grouping | 4-6 hours | High |
| 3 | Integration + feature flag | 2-3 hours | Medium |
| 4 | Testing + benchmarking | 2-3 hours | Low |
| **Total** | | **11-16 hours** | |

**Note:** This is a multi-day task. Do not attempt in a single session.

---

## References

- **Phase 1 infrastructure:** `src/model.rs:2727-2750`
- **Host batch example:** `MlpLayer::prefill` at `src/model.rs:1476`
- **Colibri reference:** `c/colibri.c:6334` (batch-union expert loading)
- **GPU API:** `src/gpu.rs:1314-1400` (VkBatch methods)
- **Benchmark protocol:** `project-engine-bench-hygiene.md`
- **Defect register:** `project-engine-defect-register.md`

---

## Next Steps

1. **Review this plan** with the team
2. **Create a feature branch:** `git checkout -b feature/device-prefill-phase-2`
3. **Implement Step 1** (dense batching) first, test thoroughly
4. **Benchmark Step 1** to verify improvement
5. **Proceed to Step 2** (expert grouping) only if Step 1 succeeds
6. **Integration test** the full batched prefill
7. **Update README** with new benchmark results

**Current Status:** Plan complete, ready for implementation.

---

## Appendix: Code Locations

| Component | File | Line | Description |
|-----------|------|------|-------------|
| `prefill_device` | src/model.rs | 2734 | Current token-by-token fallback |
| `prefill_device_ready` | src/model.rs | 2748 | GPU presence check |
| `Dense::gemv_batch` | src/model.rs | 141 | Host batched GEMV (exists) |
| `MlpLayer::prefill` | src/model.rs | 1476 | Host batched prefill example |
| `MoeLayer::prefill` | src/model.rs | 1050 | Host batch-union example |
| `VkBatch::gemv_half` | src/gpu.rs | 1323 | Device GEMV operation |
| `VkBatch::gemv_half_nb` | src/gpu.rs | 1367 | No-barrier device GEMV |
| `Colibri moe()` | c/colibri.c | 6334 | Reference batch-union implementation |

---

**Document Version:** 1.0  
**Last Updated:** 2026-11-14  
**Author:** Fennec optimization team  
**Status:** Ready for review and implementation
