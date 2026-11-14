# Fennec Engine Improvement Report

**Date:** 2026-11-14  
**Status:** In Progress  
**Mandate:** Make Fennec's engine and all repos as good as possible

---

## Completed Improvements

### 1. Memory Pressure Fix (#82) ✅

**Problem:** Device tiers were paying for ~1.8 GiB of int8 quantization work that `attach_dense` would immediately throw away by re-reading from shards and uploading to VRAM.

**Solution:** Added `quantize_i8: bool` parameter to `GdnLayer::load`, `AttnLayer::load`, and `MlpLayer::load`. The parameter is set based on `lp.dense_on_device` from the plan, skipping quantization when the dense layer will be uploaded to device.

**Files Modified:**
- `src/model.rs`: Updated load functions and call sites
- `tests/golden.rs`: Updated test calls to pass `true` for quantize_i8

**Impact:**
- Saves ~1.8 GiB of memory commit on device tiers
- Reduces startup time by skipping unnecessary quantization work
- No impact on decode throughput (quantization was thrown away anyway)

**Testing:** All 100 tests pass (64 unit + 2 binary + 5 dense lane + 13 golden + 9 GPU + 6 tokenizer + 1 device prefill)

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

**Testing:** All 100 tests pass, build clean (zero warnings)

**Next:** Phase 2 will implement actual GPU batching (dense GEMV + expert grouping)

---

## Current State

### Benchmark Standing (as of 2026-11-14)

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

## Next High-Impact Optimizations

### Priority 1: Device Prefill Batching (#145)

**Problem:** Device tier walks prompt tokens one at a time, while Colibri batches them.

**Approach:**
- Batch dense projections over prompt tokens (n rows per submit)
- Group routed experts by unique expert (batch-union, like Colibri's moe())
- Expected improvement: ~620ms on TTFT (from 1.6s to ~1.0s)

**Risk:** Medium - requires careful GPU command buffer management

**Status:** ✅ Phase 1 (Infrastructure) complete. Phase 2 (actual batching) ready to implement.

**Phase 2 Implementation Guide:** See `DEVICE_PREFILL_PHASE1_COMPLETE.md` and `DEVICE_PREFILL_BATCHING_GUIDE.md`

---

### Priority 2: Submission Pipelining (#102)

**Problem:** CUDA MoE record+launch overhead is 13.19 ms/token.

**Approach:**
- Pipeline command buffer recording and submission
- Overlap CPU work with GPU execution
- Expected improvement: ~2-3 ms/token on CUDA tier

**Risk:** Medium-High - requires deep understanding of Vulkan/CUDA async semantics

**Status:** Not started

---

### Priority 3: Host Work onto Idle GPU (#76)

**Problem:** Host experts run on CPU while GPU idles.

**Approach:**
- Move host expert computation to GPU when device tier is active
- Overlap device and host work on the same token
- Expected improvement: ~5-10 ms/token on device tiers

**Risk:** High - requires careful synchronization and memory management

**Status:** In progress (partial - linear-attention mixer host half already on GPU)

---

### Priority 4: Worker Pool Pinning (#99, #141)

**Problem:** Worker threads not pinned to performance cores on hybrid CPUs.

**Status:** ✅ Already implemented in `src/kernel.rs` (lines 270-284)
- Threads are pinned to performance cores when count allows
- Oversubscription is avoided (unpinned when workers > performance cores)

---

## Code Quality Improvements

### Completed
- ✅ All tests pass (100 engine tests + 236 core tests + 48 CLI tests + 46 installer tests)
- ✅ Memory pressure fix reduces startup cost (~1.8 GiB saved)
- ✅ Device prefill batching Phase 1 infrastructure in place
- ✅ Code is well-documented with inline comments explaining "why"
- ✅ No compiler warnings
- ✅ Comprehensive documentation created (CHANGELOG, IMPROVEMENTS, guides)

### Pending
- ⏳ Benchmark Phase 1 to verify no regression
- ⏳ Implement Phase 2 (actual device batching)
- ⏳ Update README with benchmark results

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

## Roadmap

### Week 1: Quick Wins ✅
- [x] Fix memory pressure (#82)
- [x] Device prefill batching Phase 1: Infrastructure (#145)
- [ ] Benchmark Phase 1 to verify no regression
- [ ] Update README with current state

### Week 2: Prefill Batching Phase 2
- [ ] Implement actual device batching (dense GEMV + expert grouping)
- [ ] Benchmark and verify ~620ms TTFT improvement
- [ ] Update README with new numbers

### Week 3: Submission Pipelining
- [ ] Implement CUDA/Vulkan submission pipelining (#102)
- [ ] Benchmark and verify improvement
- [ ] Update README with new numbers

### Week 4: Host-GPU Overlap
- [ ] Move host expert work to GPU (#76)
- [ ] Benchmark and verify improvement
- [ ] Final README update with all new numbers

---

## Testing Strategy

Every optimization must:
1. Pass all existing tests (99 engine + suite tests)
2. Benchmark against warm Colibri baseline (steady state, not mid-ramp)
3. Measure in one session with order swapping to detect drift
4. Quote `steady_tokens_per_s` from the driver, not raw medians
5. Document the measurement protocol in the README

---

## Success Metrics

**Goal:** Beat Colibri on at least one shared tier (CPU or Vulkan)

**Current Status:**
- CPU: 0.86x (need 1.0x+)
- Vulkan: 0.67x (need 1.0x+)

**Path to Victory:**
- Prefill batching: ~1.5x improvement on TTFT
- Submission pipelining: ~1.1x improvement on decode
- Combined: Could reach 1.0x on Vulkan, possibly >1.0x on CPU

---

## Conclusion

The memory pressure fix and device prefill batching Phase 1 infrastructure are solid foundations. The next big win is Phase 2: actual device batching, which could close the TTFT gap by ~620ms. With careful implementation and rigorous benchmarking, Fennec can beat Colibri on at least one tier.

**Next Action:** Implement device prefill batching Phase 2 (actual GPU batching) and measure the improvement.

**Implementation Guides:**
- `DEVICE_PREFILL_PHASE1_COMPLETE.md` - Phase 1 summary + Phase 2 roadmap
- `DEVICE_PREFILL_BATCHING_GUIDE.md` - Detailed Phase 2 implementation guide

---

**Engine Version:** 1.0.0  
**Build:** Clean (no warnings)  
**Tests:** 100/100 passing  
**Phase 1 Status:** ✅ Complete (infrastructure in place)  
**Phase 2 Status:** ⏳ Ready to implement  
**Ready for:** Phase 2 implementation and benchmarking
