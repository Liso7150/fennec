# Device Prefill Batching - Phase 1 Complete

**Date:** 2026-11-14  
**Status:** ✅ Phase 1 (Infrastructure) Complete  
**Tests:** 100/100 passing  
**Behavior:** No change (safe fallback to token-by-token)

---

## What Was Implemented

### 1. Model Methods (model.rs)

**Added `prefill_device` method:**
```rust
pub fn prefill_device(&self, xs: &mut [f32], st: &mut [LayerState], pos0: usize) {
    let h = self.cfg.hidden_size;
    let n = xs.len() / h;
    // Phase 1: fall back to token-by-token forward (no behavior change)
    for i in 0..n {
        let mut x = xs[i * h..(i + 1) * h].to_vec();
        self.forward(&mut x, st, pos0 + i);
        xs[i * h..(i + 1) * h].copy_from_slice(&x);
    }
}
```

**Added `prefill_device_ready` method:**
```rust
pub fn prefill_device_ready(&self) -> bool {
    self.gpu.is_some()
}
```

### 2. Serve Integration (serve.rs)

**Updated prefill logic:**
```rust
let batched = prompt_len > 1 && (model.prefill_ready() || model.prefill_device_ready());
let first = first_forward(prompt_len, batched);
if batched {
    // ... chunk loop ...
    if model.prefill_device_ready() {
        model.prefill_device(&mut xs[..n * h], &mut st, c0);
    } else {
        model.prefill(&mut xs[..n * h], &mut st, c0);
    }
}
```

---

## Current Behavior

### Host Tier (CPU-only)
- `prefill_ready()` returns `true`
- Uses `model.prefill()` - batched host kernels
- Processes n tokens per layer in one pass
- **Status:** Unchanged, working as before

### Device Tier (GPU)
- `prefill_device_ready()` returns `true`
- Uses `model.prefill_device()` - Phase 1 fallback
- Currently loops through `forward()` token-by-token
- **Behavior:** Identical to previous implementation
- **Performance:** No change (same token-by-token path)

---

## Phase 2: Actual Device Batching

### Goal
Replace the token-by-token loop in `prefill_device` with actual GPU-batched operations:
1. Batch dense GEMV operations (n rows per command buffer submit)
2. Group experts by unique ID (batch-union)
3. Keep recurrent parts (gdn_rec) per-token

### Expected Impact
- **TTFT:** ~620ms improvement (1.644s → ~1.0s)
- **Throughput:** Unchanged (decode is already token-by-token)

### Implementation Strategy

#### Option A: Layer-by-Layer Batching (Like Host)
```rust
pub fn prefill_device(&self, xs: &mut [f32], st: &mut [LayerState], pos0: usize) {
    for (l, s) in self.layers.iter().zip(st.iter_mut()) {
        l.prefill_device(xs, s, pos0);  // New layer method
    }
}
```

**Pros:** Mirrors host architecture, easier to reason about  
**Cons:** Requires adding `prefill_device` to every layer type

#### Option B: Token-by-Token with Batched Submissions
```rust
pub fn prefill_device(&self, xs: &mut [f32], st: &mut [LayerState], pos0: usize) {
    let h = self.cfg.hidden_size;
    let n = xs.len() / h;
    
    // Batch all tokens through all layers in one command buffer
    let mut bt = self.gpu.as_ref().unwrap().batch().unwrap();
    for i in 0..n {
        // Submit all n tokens' work to one command buffer
        self.forward_device_batched(&mut xs[i * h..], st, pos0 + i, &mut bt);
    }
    bt.submit()?;
    self.gpu.as_ref().unwrap().wait()?;
}
```

**Pros:** Single command buffer for all tokens  
**Cons:** Complex state management, harder to debug

### Recommended: Option A (Layer-by-Layer)

**Rationale:**
1. Mirrors the proven host architecture
2. Easier to test and debug
3. Can optimize each layer independently
4. Lower risk of synchronization bugs

---

## Phase 2 Implementation Steps

### Step 1: Add Layer::prefill_device
```rust
impl Layer {
    pub fn prefill_device(&self, xs: &mut [f32], st: &mut LayerState, pos0: usize) {
        // Batch dense projections on device
        // Keep recurrent parts per-token
        match (&self.kind, st) {
            (LayerKind::Linear, LayerState::Linear(g)) => {
                self.gdn.as_ref().unwrap().prefill_device(xs, g, pos0);
            }
            (LayerKind::Full, LayerState::Full(a)) => {
                self.attn.as_ref().unwrap().prefill_device(xs, pos0, a);
            }
            _ => panic!("layer state mismatch"),
        }
    }
}
```

### Step 2: Add GdnLayer::prefill_device
```rust
impl GdnLayer {
    pub fn prefill_device(&self, xs: &mut [f32], st: &mut GdnState, pos0: usize) {
        let h = xs.len() / self.cfg.hidden_size;
        let n = xs.len() / h;
        
        // Batch the 5 input projections on device
        // (in_qkv, in_z, in_b, in_a, out_proj)
        
        // Keep gdn_rec per-token (recurrent, can't batch)
        for i in 0..n {
            // Process token i through recurrent part
        }
    }
}
```

### Step 3: Add AttnLayer::prefill_device
Similar to GdnLayer, but for attention layers.

### Step 4: Implement Device Batched GEMV
```rust
impl Dense {
    pub fn gemv_device_batched(&self, rows: usize, cols: usize, xs: &[f32], outs: &mut [f32], gpu: &Gpu) {
        let n = xs.len() / cols;
        // Submit n GEMV operations in one command buffer
        let mut bt = gpu.batch().unwrap();
        for i in 0..n {
            // Bind pipeline, push constants, dispatch for token i
        }
        bt.submit()?;
        gpu.wait()?;
    }
}
```

### Step 5: Implement Expert Grouping (Batch-Union)
```rust
impl MoeLayer {
    pub fn prefill_device(&self, xs: &mut [f32], st: &mut MoeState, pos0: usize) {
        let n = xs.len() / self.hidden_size;
        
        // Route all tokens, collect picks
        let mut expert_tokens = HashMap::new();
        for i in 0..n {
            let picks = self.route(&xs[i * h..]);
            for expert in picks {
                expert_tokens.entry(expert).or_insert(vec![]).push(i);
            }
        }
        
        // Process each expert once for all its tokens
        for (expert, tokens) in expert_tokens {
            // Load expert weights to device
            // Process all tokens for this expert
        }
    }
}
```

---

## Testing Strategy

### Unit Tests
- [ ] `prefill_device` produces bit-identical results to Phase 1 fallback
- [ ] Dense batching matches non-batched version
- [ ] Expert grouping produces correct outputs
- [ ] All existing golden tests pass

### Integration Tests
- [ ] Device tier uses `prefill_device` path
- [ ] Host tier still uses `prefill` path
- [ ] No regressions in decode throughput
- [ ] TTFT improves by ~620ms

### Benchmarks
- [ ] Measure TTFT before/after (target: 1.644s → ~1.0s)
- [ ] Measure decode throughput (should be unchanged)
- [ ] Compare against Colibri baseline
- [ ] Test with different prompt lengths (16, 32, 64, 128 tokens)

---

## Risk Mitigation

### What Could Go Wrong
1. **Synchronization bugs** - GPU command buffer ordering issues
2. **Memory pressure** - Batching may require more VRAM
3. **Correctness** - Batched operations may produce different results
4. **Performance regression** - Batching overhead may outweigh benefits

### Safeguards
1. **Feature flag** - Add `--device-prefill-batch` flag, off by default
2. **Fallback** - If batching fails, fall back to Phase 1 (token-by-token)
3. **Testing** - Run full test suite with batching enabled
4. **Benchmarking** - Measure before/after with steady-state protocol

### Rollback Plan
If Phase 2 causes regressions:
1. Disable feature flag
2. Revert to Phase 1 (current implementation)
3. Investigate issue
4. Fix or abandon approach

---

## Current Status

### ✅ Phase 1: Infrastructure (Complete)
- [x] Added `prefill_device` method to Model
- [x] Added `prefill_device_ready` method to Model
- [x] Updated serve.rs to use device prefill path
- [x] All tests passing (100/100)
- [x] No behavior change (safe fallback)

### ⏳ Phase 2: Actual Batching (Next)
- [ ] Add `prefill_device` to Layer
- [ ] Add `prefill_device` to GdnLayer
- [ ] Add `prefill_device` to AttnLayer
- [ ] Implement device batched GEMV
- [ ] Implement expert grouping (batch-union)
- [ ] Test correctness (golden tests)
- [ ] Benchmark performance (TTFT improvement)

### 📋 Phase 3: Optimization (Future)
- [ ] Profile and identify bottlenecks
- [ ] Optimize command buffer submission
- [ ] Tune batch sizes
- [ ] Consider submission pipelining (#102)

---

## Files Modified

1. **D:\Fennec\src\model.rs**
   - Added `prefill_device` method (lines 2727-2744)
   - Added `prefill_device_ready` method (lines 2746-2750)

2. **D:\Fennec\src\serve.rs**
   - Updated prefill logic to use device path (lines 231-256)

---

## Next Steps

1. **Review Phase 1 implementation** - Ensure infrastructure is correct
2. **Benchmark Phase 1** - Verify no regression (should be identical to before)
3. **Proceed to Phase 2** - Implement actual device batching
4. **Test thoroughly** - Golden tests + performance benchmarks
5. **Ship if successful** - Update README with new benchmarks

---

**Phase 1 Status:** ✅ Complete and tested  
**Phase 2 Status:** ⏳ Ready to implement  
**Risk Level:** Medium (Phase 2 requires careful GPU work)  
**Expected Impact:** ~620ms TTFT improvement (if Phase 2 succeeds)
