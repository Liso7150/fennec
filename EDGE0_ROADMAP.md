# Edge0 Implementation Roadmap

**Status:** Core architecture complete, GPU kernels pending  
**Date:** 2026-11-14  
**Version:** 1.0.0

---

## Completed ✅

### Core Architecture
- [x] Edge0-145B model detection from GGUF metadata
- [x] Edge0-120B model detection
- [x] Hybrid layer configuration (3 Linear + 1 Full per 4 layers)
- [x] GGUF reader with dequantization support (F32/F16/BF16/Q8_0/Q4_0/Q3_K)
- [x] Plugin system architecture (Cordis-inspired)
- [x] Device prefill batching Phase 1 infrastructure

### Inference Pipeline (CPU)
- [x] Latent MoE layer with D/U projections (8 groups)
- [x] Shared expert with SiLU gating
- [x] DeltaNet linear attention layer
  - [x] Q/K/V/O projections
  - [x] Beta gating mechanism
  - [x] Conv1D local context
  - [x] Recurrent state management
- [x] PLE table with mmap-based trigram hashing
- [x] Group-weighted output computation

### Optimizations
- [x] **PLE Lookup Cache** (Priority 5 - COMPLETED)
  - LRU cache for frequently accessed n-grams (1024 entries)
  - Reduces repeated mmap reads
  - Expected ~50-100ms improvement on PLE-heavy workloads
  - Test coverage: `test_ple_cache_initialization`

### Testing
- [x] 114/114 tests passing (added PLE cache test)
- [x] Edge0 config tests (120B, 145B, params, hybrid layers)
- [x] DeltaNet state initialization tests
- [x] PLE hash tests
- [x] PLE cache initialization tests
- [x] Golden tests (14 passing)
- [x] GPU tests (9 passing)
- [x] Zero compiler warnings

---

## Future Work 🚧

### Priority 1: GPU Kernels for D/U Projections

**Goal:** Move latent MoE D/U projections to GPU for device tiers

**Current State:** CPU-only implementation in `edge0_inference.rs`

**Requirements:**
- [ ] Vulkan compute shader for D projection (hidden→latent, 2560→1792)
- [ ] Vulkan compute shader for U projection (latent→hidden, 1792→2560)
- [ ] CUDA kernel variants for NVIDIA devices
- [ ] Metal shader variants for Apple devices
- [ ] Batched GEMV operations over prompt tokens
- [ ] Integration with existing `VkBatch` API

**Expected Impact:** ~200-300ms improvement on device tier TTFT

**Complexity:** High  
**Estimated Effort:** 8-12 hours

---

### Priority 2: Int4 Quantization for Latent Experts

**Goal:** Reduce memory footprint of latent expert weights

**Current State:** F16/BF16 only

**Requirements:**
- [ ] Int4 quantization kernel for D/U projections
- [ ] Dequantization on-the-fly during forward pass
- [ ] Group size optimization (64 or 128)
- [ ] Accuracy validation against F16 baseline
- [ ] Memory savings measurement

**Expected Impact:** ~50% reduction in latent expert memory (1.2 GiB → 0.6 GiB per layer)

**Complexity:** Medium  
**Estimated Effort:** 4-6 hours

---

### Priority 3: Batched Prefill Optimization

**Goal:** Implement Phase 2 device prefill batching for Edge0

**Current State:** Phase 1 infrastructure in place, token-by-token fallback

**Requirements:**
- [ ] Dense GEMV batching on device (see DEVICE_PREFILL_PHASE2_IMPLEMENTATION.md)
- [ ] Expert grouping (batch-union like Colibri's moe())
- [ ] Feature flag for safe rollout
- [ ] Benchmark verification (~620ms TTFT improvement target)

**Expected Impact:** ~620ms improvement on TTFT (1.644s → ~1.0s)

**Complexity:** High  
**Estimated Effort:** 11-16 hours

---

### Priority 4: DeltaNet Kernel Vectorization

**Goal:** Vectorize DeltaNet recurrent computation

**Current State:** Scalar implementation in `DeltaNetLayer::forward`

**Requirements:**
- [ ] AVX2/AVX-512 vectorization for CPU tier
- [ ] Vulkan compute shader for device tier
- [ ] Recurrent state update optimization
- [ ] Conv1D state management optimization
- [ ] Numerical accuracy validation

**Expected Impact:** ~2-3x speedup on DeltaNet layers

**Complexity:** High  
**Estimated Effort:** 6-8 hours

---

### Priority 5: PLE Prefetching Strategies ✅ COMPLETED

**Goal:** Reduce PLE table lookup latency

**Current State:** ✅ LRU cache implemented and tested

**Completed:**
- [x] LRU cache for frequently accessed n-grams (1024 entries)
- [x] Cache invalidation with LRU eviction policy
- [x] Integration with existing `PleTable::lookup` method
- [x] Test coverage: `test_ple_cache_initialization`

**Expected Impact:** ~50-100ms reduction in PLE lookup latency for repeated n-grams

**Complexity:** Medium  
**Actual Effort:** 2 hours (completed 2026-11-14)

**Future Enhancements (Not Implemented):**
- [ ] Predictive prefetching based on n-gram context
- [ ] Async I/O for prefetch operations
- [ ] Memory-mapped page fault optimization
- [ ] Cache hit rate measurement

---

### Priority 6: End-to-End Benchmarks vs Colibri

**Goal:** Measure Edge0-145B performance against Colibri baseline

**Current State:** No Edge0 benchmarks yet

**Requirements:**
- [ ] Benchmark harness for Edge0-145B model
- [ ] TTFT measurement (prefill latency)
- [ ] TPOT measurement (decode throughput)
- [ ] Memory usage profiling
- [ ] Comparison with Colibri on same hardware
- [ ] Results documentation in README

**Expected Impact:** Establish performance baseline, identify optimization opportunities

**Complexity:** Low  
**Estimated Effort:** 2-3 hours

---

## Implementation Order

**Completed sequence:**

1. ✅ **PLE Prefetching (LRU Cache)** (Priority 5) - COMPLETED
   - Reduces repeated mmap reads
   - ~50-100ms improvement for PLE-heavy workloads
   - 2 hours actual effort

**Remaining sequence (recommended):**

2. **Batched Prefill Optimization** (Priority 3)
   - Highest remaining impact (~620ms TTFT)
   - Builds on existing Phase 1 infrastructure
   - Well-documented implementation plan
   - 11-16 hours estimated

3. **GPU Kernels for D/U Projections** (Priority 1)
   - Second highest impact (~200-300ms)
   - Required for device tier competitiveness
   - Can reuse existing GPU infrastructure
   - 8-12 hours estimated

4. **End-to-End Benchmarks** (Priority 6)
   - Establish baseline measurements
   - Validate improvements from above optimizations
   - Low effort, high value
   - 2-3 hours estimated

5. **DeltaNet Kernel Vectorization** (Priority 4)
   - Significant speedup for recurrent layers
   - Complex but well-scoped
   - 6-8 hours estimated

6. **Int4 Quantization** (Priority 2)
   - Memory savings, not speed
   - Lower priority unless memory-constrained
   - 4-6 hours estimated

**Completed Effort:** 2 hours (PLE cache)  
**Remaining Estimated Effort:** 33-47 hours (5-6 working days)

---

## Success Criteria

### Must Have
- [ ] All 113 existing tests continue to pass
- [ ] No regressions in decode throughput
- [ ] Edge0-145B TTFT within 1.2x of Colibri on Vulkan
- [ ] Memory usage within 1.1x of Colibri
- [ ] Zero compiler warnings

### Nice to Have
- [ ] Edge0-145B TTFT within 1.0x of Colibri (beat or match)
- [ ] Memory usage within 0.9x of Colibri (better)
- [ ] Comprehensive benchmark suite with automated regression detection
- [ ] Published benchmark results in README

---

## Technical Debt

### Known Issues
- None currently (all tests passing, zero warnings)

### Future Considerations
- Edge0-200B support when model is released
- Multi-GPU distribution for large models
- Flash attention integration for full attention layers
- Sparse expert execution optimization

---

## References

- **Phase 2 Implementation Plan:** `DEVICE_PREFILL_PHASE2_IMPLEMENTATION.md`
- **Optimization Status:** `OPTIMIZATION_STATUS.md`
- **Verification Report:** `VERIFICATION_REPORT.md`
- **Edge0 Support Docs:** `EDGE0_SUPPORT.md`
- **Engine Defect Register:** `project-engine-defect-register.md` (memory)

---

**Document Version:** 1.1  
**Last Updated:** 2026-11-14  
**Author:** Fennec optimization team  
**Status:** PLE cache complete, GPU kernels and batched prefill pending
