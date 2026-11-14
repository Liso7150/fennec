# Fennec Engine Optimization Status Report

**Date:** 2026-11-14  
**Version:** 1.0.0  
**Status:** Phase 1 complete, Phase 2 ready for implementation

---

## Executive Summary

The Fennec engine optimization effort has completed Phase 1 infrastructure for device prefill batching and fixed all compilation issues. The codebase is now in a clean, tested state with 113 passing tests and zero warnings.

**Key Achievements:**
- ✅ Memory pressure fix saves ~1.8 GiB on device tiers
- ✅ Device prefill batching Phase 1 infrastructure in place
- ✅ Plugin system doctest fixed and passing
- ✅ All 113 tests passing (76 unit + 2 binary + 4 dense lane + 14 golden + 9 GPU + 6 tokenizer + 1 doctest + 1 device prefill)
- ✅ Clean build with zero warnings
- ✅ Comprehensive Phase 2 implementation plan created

**Current Performance (Baseline):**
- Colibri Vulkan: 20.160 tok/s
- Fennec CUDA: 17.529 tok/s (0.87x)
- Fennec Vulkan: 13.485 tok/s (0.67x)
- Gap: Prefill amortization (Colibri batches 3-5x better)

**Expected Improvement (Phase 2):**
- TTFT: 1.644s → ~1.0s (~620ms improvement)
- Target: Close gap to Colibri on Vulkan tier

---

## Completed Work

### 1. Memory Pressure Fix (#82) ✅

**Problem:** Device tiers were paying for ~1.8 GiB of int8 quantization work that `attach_dense` would immediately throw away.

**Solution:** Added `quantize_i8: bool` parameter to `GdnLayer::load`, `AttnLayer::load`, and `MlpLayer::load`. The parameter is set based on `lp.dense_on_device` from the plan, skipping quantization when the dense layer will be uploaded to device.

**Files Modified:**
- `src/model.rs`: Updated load functions and call sites
- `tests/golden.rs`: Updated test calls to pass `true` for quantize_i8

**Impact:**
- Saves ~1.8 GiB of memory commit on device tiers
- Reduces startup time by skipping unnecessary quantization work
- No impact on decode throughput

**Testing:** All tests pass

### 2. Device Prefill Batching Phase 1: Infrastructure (#145) ✅

**Problem:** Device tier had no batched prefill path; infrastructure was needed to support future batching.

**Solution:** Added `prefill_device()` and `prefill_device_ready()` methods to Model. Updated serve.rs to use the device prefill path when GPU is present. Phase 1 is a safe fallback that loops through forward token-by-token (no behavior change).

**Files Modified:**
- `src/model.rs`: Added `prefill_device` (lines 2727-2744) and `prefill_device_ready` (lines 2746-2750)
- `src/serve.rs`: Updated prefill logic to use device path (lines 231-256)

**Impact:**
- Zero behavior change (Phase 1 is infrastructure only)
- Zero performance change (same token-by-token path as before)
- Enables Phase 2: actual device batching

**Testing:** All tests pass

### 3. Plugin System Doctest Fix ✅

**Problem:** The plugin module's doctest had incorrect signatures and closure syntax.

**Solution:** Fixed the example to use correct `PluginError` return type, proper closure syntax, and added missing `MyService` definition.

**Files Modified:**
- `src/plugin.rs`: Fixed doctest example (lines 16-44)

**Impact:**
- Doctests now compile and pass
- Example code is accurate and runnable

**Testing:** All tests pass including the fixed doctest

### 4. Dead Code Warning Fix ✅

**Problem:** `GgufReader.path` field was stored but never read, causing a compiler warning.

**Solution:** Added a `path()` getter method to expose the field.

**Files Modified:**
- `src/gguf.rs`: Added `path()` method (line 212)

**Impact:**
- Clean build with zero warnings
- Field is now accessible for debugging/error messages

**Testing:** All tests pass

---

## Current State

### Code Quality

| Metric | Status |
|--------|--------|
| Build | ✅ Clean (zero warnings) |
| Tests | ✅ 113/113 passing |
| Documentation | ✅ Comprehensive (IMPROVEMENTS.md, guides) |
| Phase 1 Infrastructure | ✅ Complete |
| Phase 2 Implementation | ⏳ Ready (plan created) |

### Test Breakdown

- **Unit tests:** 76 passed
- **Binary tests:** 2 passed
- **Dense lane tests:** 4 passed
- **Golden tests:** 14 passed
- **GPU tests:** 9 passed
- **Tokenizer tests:** 6 passed
- **Doctests:** 1 passed
- **Device prefill tests:** 1 passed
- **Total:** 113 passed, 0 failed

### Benchmark Standing

| Engine | Backend | tok/s | vs Colibri |
|--------|---------|-------|------------|
| Colibri | Vulkan | 20.160 | baseline |
| Fennec | CUDA | 17.529 | 0.87x |
| Fennec | Vulkan | 13.485 | 0.67x |
| Colibri | CPU | 9.399 | baseline |
| Fennec | CPU | 8.048 | 0.86x |

**Gap Analysis:**
- Vulkan: 1.55x behind (1.20s prefill + 1.59s decode)
- CPU: 1.17x behind (1.19s prefill + 1.12s decode)
- Main issue: Prefill amortization (Colibri batches 3-5x better)

---

## Remaining Work

### Priority 1: Device Prefill Batching Phase 2 (#145)

**Status:** Plan complete, ready for implementation  
**Expected Impact:** ~620ms improvement on TTFT (1.644s → ~1.0s)  
**Complexity:** High  
**Risk:** Medium-High (requires careful GPU synchronization)  
**Estimated Effort:** 11-16 hours (multi-day task)

**Implementation Plan:** See `DEVICE_PREFILL_PHASE2_IMPLEMENTATION.md`

**Key Components:**
1. Dense GEMV batching on device (3-4 hours, Medium risk)
2. Expert grouping (batch-union, like Colibri's moe()) (4-6 hours, High risk)
3. Integration + feature flag (2-3 hours, Medium risk)
4. Testing + benchmarking (2-3 hours, Low risk)

**Success Criteria:**
- [ ] All 113 tests pass
- [ ] TTFT improves by at least 400ms (target: 620ms)
- [ ] Decode throughput unchanged (±2%)
- [ ] No new compiler warnings
- [ ] Bit-identical outputs to token-by-token

### Priority 2: Submission Pipelining (#102)

**Status:** Not started  
**Expected Impact:** ~2-3 ms/token on CUDA tier  
**Complexity:** Medium-High  
**Risk:** Medium-High (requires deep understanding of Vulkan/CUDA async semantics)

**Problem:** CUDA MoE record+launch overhead is 13.19 ms/token.

**Approach:**
- Pipeline command buffer recording and submission
- Overlap CPU work with GPU execution

**Estimated Effort:** 8-12 hours

### Priority 3: Host Work onto Idle GPU (#76)

**Status:** In progress (partial - linear-attention mixer host half already on GPU)  
**Expected Impact:** ~5-10 ms/token on device tiers  
**Complexity:** High  
**Risk:** High (requires careful synchronization and memory management)

**Problem:** Host experts run on CPU while GPU idles.

**Approach:**
- Move host expert computation to GPU when device tier is active
- Overlap device and host work on the same token

**Estimated Effort:** 12-16 hours

---

## Architecture Decisions

### What NOT to Do (Measured Regressions)

1. **Int4 trunk on device tier** - Measured -9% slower (#147)
2. **4-row Vulkan int4 gemv** - Measured -8% slower (#127)
3. **Fuse dense input projections** - Measured -14.4% slower (#86)
4. **Host prefill path** - Measured +0.53s TTFT worse (#142)
5. **gemv_q4 divide-free index** - Noise level improvement, not worth complexity

### What Works

1. **Bigger prefill chunks (32 vs 16)** - Small TTFT win
2. **Batch-union host experts** - Like Colibri's moe()
3. **SSE transport** - 6.8ms vs WebView2 IPC 73.1ms
4. **Native Rust desktop (egui)** - 4.4MB binary, no Electron overhead
5. **Performance core pinning** - Removes straggler waits on hybrid CPUs

---

## Documentation

### Created Documents

1. **IMPROVEMENTS.md** - This report, tracking all optimization work
2. **DEVICE_PREFILL_BATCHING_GUIDE.md** - Phase 1 completion summary + Phase 2 roadmap
3. **DEVICE_PREFILL_PHASE2_IMPLEMENTATION.md** - Detailed Phase 2 implementation plan
4. **CHANGELOG.md** - Version history and changes
5. **EDGE0_SUPPORT.md** - Edge0-145B model support documentation

### Code Documentation

- All public APIs have doc comments
- Complex algorithms have inline comments explaining "why"
- Test names are descriptive and explain the behavior being tested

---

## Testing Strategy

Every optimization must:
1. Pass all existing tests (113 engine tests)
2. Benchmark against warm Colibri baseline (steady state, not mid-ramp)
3. Measure in one session with order swapping to detect drift
4. Quote `steady_tokens_per_s` from the driver, not raw medians
5. Document the measurement protocol in the README

---

## Roadmap

### Week 1: Quick Wins ✅ (Complete)
- [x] Fix memory pressure (#82)
- [x] Device prefill batching Phase 1: Infrastructure (#145)
- [x] Fix plugin doctest
- [x] Fix dead code warning
- [x] Verify all tests pass (113/113)
- [x] Create Phase 2 implementation plan

### Week 2: Prefill Batching Phase 2
- [ ] Implement dense GEMV batching on device
- [ ] Implement expert grouping (batch-union)
- [ ] Add feature flag for safe rollout
- [ ] Benchmark and verify ~620ms TTFT improvement
- [ ] Update README with new numbers

### Week 3: Submission Pipelining
- [ ] Implement CUDA/Vulkan submission pipelining (#102)
- [ ] Benchmark and verify ~2-3 ms/token improvement
- [ ] Update README with new numbers

### Week 4: Host-GPU Overlap
- [ ] Move host expert work to GPU (#76)
- [ ] Benchmark and verify ~5-10 ms/token improvement
- [ ] Final README update with all new numbers

---

## Success Metrics

**Goal:** Beat Colibri on at least one shared tier (CPU or Vulkan)

**Current Status:**
- CPU: 0.86x (need 1.0x+)
- Vulkan: 0.67x (need 1.0x+)

**Path to Victory:**
- Prefill batching Phase 2: ~1.5x improvement on TTFT
- Submission pipelining: ~1.1x improvement on decode
- Combined: Could reach 1.0x on Vulkan, possibly >1.0x on CPU

---

## Risk Assessment

### High-Risk Items

1. **Phase 2 GPU batching** - Requires careful synchronization, could introduce correctness bugs
   - Mitigation: Feature flag, fallback path, extensive testing
   
2. **Submission pipelining** - Deep understanding of Vulkan/CUDA async semantics required
   - Mitigation: Start with simple cases, measure incrementally
   
3. **Host-GPU overlap** - Complex memory management, potential for race conditions
   - Mitigation: Formal verification of synchronization, stress testing

### Low-Risk Items

1. **Benchmarking** - Well-established protocol, low implementation risk
2. **Documentation** - No code changes, purely additive
3. **Feature flags** - Safe rollout mechanism, easy to disable

---

## Conclusion

The Fennec engine is now in a solid foundation state with all Phase 1 infrastructure complete, all tests passing, and zero warnings. The codebase is ready for Phase 2 implementation, which targets a ~620ms improvement in TTFT through device-side batching of dense GEMV operations and expert grouping.

**Next Action:** Begin Phase 2 implementation following the detailed plan in `DEVICE_PREFILL_PHASE2_IMPLEMENTATION.md`. Start with dense GEMV batching (Step 1), test thoroughly, then proceed to expert grouping (Step 2) only if Step 1 succeeds.

**Estimated Time to Victory:** 3-4 weeks of focused implementation and benchmarking.

---

**Engine Version:** 1.0.0  
**Build:** Clean (zero warnings)  
**Tests:** 113/113 passing  
**Phase 1 Status:** ✅ Complete  
**Phase 2 Status:** ⏳ Ready (plan created)  
**Ready for:** Phase 2 implementation and benchmarking
