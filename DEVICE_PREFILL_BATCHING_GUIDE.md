# Device Prefill Batching Implementation Guide

**Task:** #145 - Batch the device prefill as a grouped expert union  
**Expected Impact:** ~620ms improvement on TTFT (1.644s → ~1.0s)  
**Complexity:** High  
**Risk:** Medium (requires careful GPU synchronization)

---

## Current State

### Problem
Device tier walks prompt tokens one at a time through `Model::forward`, while Colibri batches them 3-5x better. The host tier already has batched prefill via `Model::prefill`, but `prefill_ready()` returns false when GPU is present.

### Why Host Batch Doesn't Work for Device
Previous attempt (documented in defect register) measured:
- Host batch for device prompt: 2166ms
- Device token-by-token: 1750ms
- **Result:** Host batch is 24% SLOWER

**Reason:** Host batch reads weights from system RAM, competing with host expert pool for bandwidth. Device tier needs device-side batching.

### Current Code Flow
```
serve.rs:generate()
  ↓
if batched && model.prefill_ready():  // false for device tier
  model.prefill(xs, st, c0)  // host batch path
else:
  for t in first..end:
    model.forward(x, st, t)  // token-by-token (device uses this)
```

---

## Implementation Strategy

### Goal
Create `model.prefill_device(xs, st, pos0)` that:
1. Batches dense GEMV operations on device (n rows per submit)
2. Groups routed experts by unique expert ID (batch-union)
3. Keeps recurrent parts (gdn_rec) per-token
4. Submits one command buffer for n tokens instead of n submissions

### Key Components

#### 1. Dense Projection Batching
Current device forward (GdnLayer::step):
```rust
// One token at a time
self.in_qkv.gemv(cd, h, x, m, t);  // host
// OR
self.dev_inputs(d, x, &mut ibuf);  // device
```

Need batched version:
```rust
// n tokens at once
self.in_qkv.gemv_batch(cd, h, xs, &mut ms, t);  // host batch
// OR
self.dev_inputs_batch(d, xs, &mut ibufs);  // device batch
```

#### 2. Command Buffer Batching
Current device submission (one per token):
```rust
let mut bt = g.batch().unwrap();
bt.bind_pipeline(...);
bt.push_constants(...);
bt.dispatch(...);
bt.submit()?;
g.wait()?;
```

Need batched submission (one for n tokens):
```rust
let mut bt = g.batch().unwrap();
for i in 0..n {
    bt.bind_pipeline(...);
    bt.push_constants(..., i);  // token index
    bt.dispatch(...);
    // barrier between tokens if needed
}
bt.submit()?;
g.wait()?;
```

#### 3. Expert Grouping (Batch-Union)
Current MoE forward (one token at a time):
```rust
for i in 0..n:
    route token i → picks [e0, e1, ..., e7]
    for each expert in picks:
        compute expert output
    reduce
```

Need grouped version (like Colibri's moe()):
```rust
// Group all tokens by expert
let mut expert_tokens = HashMap::new();
for i in 0..n:
    route token i → picks [e0, e1, ..., e7]
    for each expert in picks:
        expert_tokens.entry(expert).or_insert(vec![]).push(i)

// Process each expert once for all its tokens
for (expert, tokens) in expert_tokens:
    load expert weights
    for token_idx in tokens:
        compute expert output for token
    reduce
```

This is what `MoeLayer::prefill` already does on host (line 1050 in model.rs).

---

## Implementation Steps

### Phase 1: Infrastructure (Safe, No Behavior Change)

1. **Add `prefill_device` method skeleton**
   ```rust
   impl Model {
       pub fn prefill_device(&self, xs: &mut [f32], st: &mut [LayerState], pos0: usize) {
           // TODO: implement device batched prefill
           // For now, fall back to token-by-token
           let h = self.cfg.hidden_size;
           let n = xs.len() / h;
           for i in 0..n {
               self.forward(&mut xs[i * h..(i + 1) * h], &mut st[i], pos0 + i);
           }
       }
   }
   ```

2. **Update `prefill_ready` to distinguish tiers**
   ```rust
   pub fn prefill_ready(&self) -> bool {
       true  // Always ready, but caller checks tier
   }
   
   pub fn prefill_device_ready(&self) -> bool {
       self.gpu.is_some()  // Device tier can use device batch
   }
   ```

3. **Update serve.rs to use device batch**
   ```rust
   let batched = prompt_len > 1;
   let first = first_forward(prompt_len, batched);
   if batched {
       if model.prefill_device_ready() {
           model.prefill_device(&mut xs, &mut st, 0);
       } else {
           model.prefill(&mut xs, &mut st, 0);
       }
   }
   ```

### Phase 2: Dense Batching (Medium Risk)

4. **Implement `gemv_batch` for Dense**
   ```rust
   impl Dense {
       pub fn gemv_batch(&self, rows: usize, cols: usize, xs: &[f32], outs: &mut [f32], threads: usize) {
           let n = xs.len() / cols;
           // Use par_index to batch across tokens
           kernel::par_index(n, threads, |i0, cnt| {
               for i in i0..i0 + cnt {
                   self.gemv(rows, cols, &xs[i * cols..], &mut outs[i * rows..], threads);
               }
           });
       }
   }
   ```

5. **Implement device batch for GdnLayer**
   - Batch the 5 input projections (in_qkv, in_z, in_b, in_a, out_proj)
   - Keep gdn_rec per-token (recurrent, can't batch)
   - Submit one command buffer for all tokens

### Phase 3: Expert Grouping (High Risk)

6. **Implement device-side expert grouping**
   - Route all tokens, collect picks
   - Group by expert ID
   - Load each unique expert once
   - Process all tokens for that expert
   - This is the complex part - requires careful GPU memory management

7. **Benchmark and verify**
   - Use steady-state protocol (2 warm-up, 5 measured)
   - Order-swapped A/B to detect drift
   - Compare against token-by-token baseline
   - Verify correctness (golden tests must pass)

---

## Testing Strategy

### Unit Tests
- [ ] `prefill_device` produces bit-identical results to `forward` loop
- [ ] Dense batching matches non-batched version
- [ ] Expert grouping produces correct outputs

### Integration Tests
- [ ] Golden tests pass with device prefill
- [ ] GPU tests pass (tests/gpu.rs)
- [ ] No memory leaks or synchronization issues

### Benchmarks
- [ ] Measure TTFT improvement (target: ~620ms)
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
2. **Fallback** - If batching fails, fall back to token-by-token
3. **Testing** - Run full test suite with batching enabled
4. **Benchmarking** - Measure before/after with steady-state protocol

### Rollback Plan
If batching causes regressions:
1. Disable feature flag
2. Investigate issue
3. Fix or revert
4. Re-benchmark

---

## Success Criteria

### Must Have
- [ ] All existing tests pass (100 engine tests)
- [ ] TTFT improves by at least 400ms (target: 620ms)
- [ ] Decode throughput unchanged (±2%)
- [ ] No new warnings or errors
- [ ] Bit-identical outputs to token-by-token

### Nice to Have
- [ ] TTFT improves by 620ms+ (target achieved)
- [ ] Memory usage unchanged or reduced
- [ ] Code is well-documented
- [ ] Benchmark results published in README

---

## Timeline Estimate

- **Phase 1 (Infrastructure):** 2-3 hours
- **Phase 2 (Dense Batching):** 3-4 hours
- **Phase 3 (Expert Grouping):** 4-6 hours
- **Testing & Benchmarking:** 2-3 hours
- **Total:** 11-16 hours

**Note:** This is a multi-day task, not a single session. The implementation guide provides the roadmap; actual implementation requires careful work and testing.

---

## References

- Defect register: `project-engine-defect-register.md`
- Current TTFT gap: 1.644s (Fennec) vs 0.433s (Colibri)
- Expected improvement: ~620ms from dense batching, ~250ms from expert grouping
- Target TTFT: ~0.77-0.87s (from 1.644s)

---

## Next Steps

1. Review this implementation guide
2. Decide whether to proceed with Phase 1 (safe, no behavior change)
3. Implement Phase 1 and verify tests pass
4. Benchmark Phase 1 (should show no change, just infrastructure)
5. Proceed to Phase 2 if Phase 1 is successful
6. Continue through phases, benchmarking after each

**Current Status:** Guide created, not yet implemented. Ready for next optimizer to begin Phase 1.
