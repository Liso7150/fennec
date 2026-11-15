# Changelog

All notable changes to the Fennec inference engine.

This repo publishes nothing: `git tag -l` is empty and
`gh api repos/Liso7150/fennec/releases` returns an empty list, both checked
2026-11-15. The published Fennec ladder - `v0.1.0` … `v0.9.0`, newest
2026-11-11T19:29Z - lives in the nine suite repos. `Cargo.toml` here says
1.0.0, and nothing at 1.0.0 has been published.

The headings below therefore carry the date the work happened, not a tag date.
This repo's first commit is 2026-11-13 (35 commits through 2026-11-15), so the
0.5.0-0.8.0 milestones describe engine work that predates this history. They do
not line up with the suite repos' published tags of the same number either: the
published `v0.9.0` in `fennec-core` is 2026-11-11, while the 0.9.0 milestone
here - the ORT integration - is 2026-11-13. The 2.x numbering an earlier
revision of this file used was retracted; no 2.x was ever published anywhere.

## [Unreleased] — 1.0.0 in preparation

### Fixed
- **GPU serve deadlock on the first request.** `Model::prefill_device` held the
  GPU mutex across the layer loop while the MoE device-expert fallback re-locked
  the same non-reentrant mutex inside `forward`, hanging every Vulkan/CUDA serve
  request. The batched device-prefill scratch also overlapped the decode MoE
  arena. The device tier now prefills the prompt token by token through the
  proven `forward` path; decode throughput is unchanged. Verified: Vulkan serve
  returns correct output, all tests pass.
- **int8 dense trunk was wrong on every non-AVX2 build** (`kernel.rs`
  `scalar_i8`). The activation offset-binary bias was removed twice — once inside
  the scalar dot and again by `row_dot_i8`'s `acc - 64 * wsum` correction — so an
  ARM (macOS) or AVX2-less x86 build decoded a `-64 * wsum * scale` error on the
  whole trunk. The AVX2 `maddubs` path was correct, which hid it on Windows and
  Linux x86; only the new macOS CI leg exposed it. The scalar fallback now sums
  the raw products like the SIMD path.
- **macOS/Ubuntu CI legs green.** Beyond the scalar int8 bug, two assertions
  encoded an x86-only truth: `ort::tests::the_provider_follows_the_device`
  expected DirectML on every OS although `for_device` returns it only under
  `cfg!(windows)`, and the router-logits golden band (1e-6) plus the half-gemv
  bit-exact test both assume AVX2's lane-and-tree reduction, which the serial
  scalar fallback misses by f32 rounding (~1.02e-6). Both now branch on the same
  feature guard the kernels dispatch on. Windows, macOS and Ubuntu all pass.

### Added
- Host-pool budget is capped to available system RAM (`mem::available_bytes`,
  Windows `GlobalMemoryStatusEx` / Linux `MemAvailable`) less headroom, so a busy
  box streams cold experts on purpose instead of faulting a too-large "RAM" pool
  every token. Prevents the thrashing seen when several engine instances exhaust
  physical memory. Its two `mem::tests` cases take the suite from 115 to 117.

### Changed
- **Host-expert int4 dot: four independent FMA accumulators** (`quant.rs`
  `dot_int`). A single accumulator chained every 64-column block on the ~4-cycle
  FMA latency, stalling a 2048-column row ~128 cycles on the accumulate alone.
  Four chains keep both FMA ports busy; the sum is the same products in a
  different order, inside the routed-expert golden band. Host experts measured
  108 → 41 ms/token on the CPU lane.
- **CPU-lane lm_head quantised to int8** when `--dense-i8` is on (the default),
  matching Colibri's head precision. The mmap half path read 2 bytes per weight
  every token; int8 reads 1. lm_head measured 25 → 10 ms/token. The GPU and ORT
  tiers keep their own head paths; `--no-dense-i8` restores the exact half dot.
- **MoE device batch skips the redundant fence wait on CUDA** (`read_syncs`):
  the blocking DtoH read already synchronises the context, so the explicit wait
  paid a second WDDM flush per layer. Vulkan's mapped read still waits.
- **Vulkan int4 gemv: eight lanes per row, eight rows per workgroup**
  (`gemv_q4.comp`). A row used to own a whole 64-thread workgroup, so its
  reduction cost six barrier steps and the per-row time was flat in bytes rather
  than in bandwidth. Eight lanes now split a row (three steps) and a workgroup
  covers eight rows, so the nibble LUT and its barrier are paid once for eight
  rows instead of once per row, and the activation is read as `vec4`. The decode
  MoE measured 1006.5 → 264.0 us per layer (3.81x), 40.26 → 10.56 ms per token
  over 40 layers. Re-run on 2026-11-15 the same probe reads 1179.1 → 280.6 us per
  layer (4.20x) and 47.16 → 11.22 ms per token, so the ratio is the claim and the
  levels are one run each: `tests/gpu.rs moe_kernel_shapes` is an `#[ignore]`d
  device-timing probe that prints instead of asserting, so it runs under
  `cargo test --release --offline --test gpu -- --ignored`, not inside the
  default 117. Eight partials folded by a three-step tree is a different
  association of the same products, and the same probe checks it: the new kernel
  stays within 1e-5 relative RMS of the one-row reference over a whole layer and
  at two quant-group widths (32 and 64) - also only under `--ignored`. Two other
  shapes were priced and rejected —
  concatenating the k expert blocks into one dispatch (the kernel is not
  dispatch-bound) and four rows per 256-thread workgroup (slower than both).
- **Vulkan half-precision gemv: hardware convert + wide activation loads**
  (`gemv_f16.comp`, `gemv_bf16.comp`). f16→f32 now uses `unpackHalf2x16` (the
  hardware `cvt.f32.f16` the CUDA kernel uses) instead of a software conversion
  that spent ~7 integer ops plus `findMSB` per weight; the activation is read as
  `vec4` through a fifth descriptor binding (X viewed as vec4) instead of eight
  scalars, matching CUDA's `ld.global.v4.f32`. Vulkan serve 73.2 → 61.7 ms/token.
  A shared-memory activation-staging variant was tried and reverted: it measured
  slower (lm_head 9.2 → 11.1 ms), confirming the head is weight-bandwidth-bound,
  not activation-bound — the same plateau the defect register records.

### Measured
- Paired ABBA against Colibri (qwen36-35b, 64-token greedy budget, 32768-token
  context, 8 threads, two discarded full-length warm-ups, one request per visit,
  cold server launch per visit, RTX 4060 Laptop + Core Ultra 7 155H, 21-22 GB
  free at session start), session `2026-11-15T09:28Z`:
  - CPU: Fennec 115.6 vs Colibri 137.4 ms/token — **Fennec faster in 4 of 4
    rounds** (per-round deltas -23.3 / -21.8 / -14.9 / -13.4 ms, both orders
    agree).
  - Vulkan: Fennec 68.7 vs Colibri 66.2 ms/token — Colibri faster in 4 of 4
    rounds (+7.7 / +4.7 / +4.7 / +2.5 ms, both orders agree).
  - CUDA: Fennec 56.5 ms/token, block-ordered. Colibri has no CUDA tier on this
    box (its `qwen36.exe` carries no `coli_cuda.dll`), so that cell is recorded
    unavailable with the evidence rather than as a comparator.
- Vulkan re-run after the `gemv_q4.comp` swap, session `2026-11-15T10:47Z`:
  Fennec 57.3 vs Colibri 57.5 ms/token, Fennec faster in 3 of 4 rounds, median
  delta -0.24 ms/token. **The two orders disagree** (+3.2 in `ab`, -1.5 in `ba`),
  so no winner may be named from it; what it does say is that the +4.7 ms/token
  deficit is gone and the lane is at parity. Both engines read ~12 ms/token
  faster than in the 09:28Z session, so the levels of the two sessions are not
  comparable — only the per-round deltas are.
- **Two headline claims are retracted.** An earlier revision of this section
  read "CPU: Fennec 92.0 vs Colibri 111.6 — Fennec wins ~18%" and "CUDA 43.4 vs
  Colibri Vulkan 45.1 — Fennec wins ~4%". No archived matrix session produces
  that pair: each number came from a different session, and one CPU cell had been
  merged into a table built from another.
- Earlier revisions of this section led with the block-ordered matrix of session
  `2026-11-15T08:23Z` (CPU 114.3 vs 97.7, Vulkan 66.0 vs 60.3). It is superseded
  by the paired sessions above: paired ABBA replaced block ordering because the
  same unchanged Fennec binary read 85.1 ms/token in one block session and 121.3
  in another, so a block-ordered CPU "loss" was never a comparison of engines.
- The matrix driver now refuses to start when another engine or another copy of
  itself is live, and records free RAM per cell. Two concurrent drivers on
  2026-11-15 put the CPU lane at 191 ms/token against 92-114 on a quiet box.
- AVX-VNNI `vpdpbusd` was tried for the host dot and reverted: this hybrid CPU
  reports `avxvnni` in CPUID but traps the 256-bit instruction (SIGILL), so the
  maddubs path stays. Colibri's `IDOT_KERNEL` is also `avx2` on this box.
- CPU `gemv_i8` work decomposition, from the ignored probe
  `tests/dense_i8.rs gemv_i8_throughput_at_the_real_projection_shapes` (synthetic
  weights, 8 workers left unpinned over 6 performance cores, run with
  `cargo test --release --offline --test dense_i8 -- --ignored`): one dispatch of
  each real projection shape reads 84.9-105.9 GiB/s (gdn in_qkv 98.2, in_z 84.9,
  out_proj 96.9, attn q_proj 105.9, o_proj 80.4; k_proj 47.5) and the lm_head
  49.1. The same 24 MiB as one call against split in two reads 0.330 ms against
  0.308 ms, so dispatch granularity is not what costs the lane. All 40 layers'
  projections in token order read 33.59 ms over 1.18 GiB = 35.2 GiB/s, against the
  served profiler's 38.57 ms at 33 GiB/s: the shortfall is the interleaving of 40
  different matrices inside one token, not the kernel and not the call count, so
  only the model loop can move it.

## 1.0.0 milestone - work of 2026-11-14 (unpublished)

### Added
- Edge0-145B model support: GGUF reader, latent MoE detection, hybrid attention config
- Edge0 inference pipeline: latent MoE, DeltaNet, PLE integration
- PLE (Partial Expert Loading) lookup cache for Edge0 inference optimization
- Edge0 benchmark harness for TTFT/TPOT/memory measurement
- Cordis-inspired plugin architecture with device prefill Phase 1
- Dense Qwen3 lane: optional MoE/GDN keys, dense SiLU FFN, configurable prefix, tied head
- Typed event envelope system for diagnostics
- Comprehensive diagnostics surface for JetBrains IDE sync
- CI build verification expanded to macOS and Ubuntu

### Changed
- Phase-batch the device MoE and mixer for improved throughput
- Eliminated per-token allocations and use optimized kernels
- Add opt-in `--dense-i8` host GEMV for int8 activation dot on CPU lane
- Worker pool pinned to performance cores (physical, not logical)
- Half-precision lm_head GEMV vectorized
- Host int4 GEMV made shuffle-free (permute x, not weights)
- CUDA activation arena pinned + device-mapped as documented

### Fixed
- Systematic bug hunt: multiple fixes and diagnostics added
- `Opts::has()` ignoring valued CLI options (`--device` silently dropped)
- lm_head VRAM demotion near card capacity
- CUDA context not current on serve's request threads
- Plugin doctest and eliminated warnings
- All 115 tests passing

## 1.0.0 milestone - work of 2026-11-13 (unpublished; this section was headed `2.5.0`, a version that was never published)

### Added
- Three-tier Rust MoE engine: device (VRAM) / host (RAM) / cold (NVMe)
- int4-dense residency lever (`--dense-int4`)
- Self-hosted GPU bench workflow (build + sign + goldens + matrix vs Colibri)
- `--planar-dot` host-expert lever for A/B benchmarking

### Changed
- README refreshed: Vulkan precision matrix (int4/bf16 measured; int8/int16/fp4/fp8 unavailable)
- Colibri upstream tracked at `68cfbb5d`

## [0.9.0] - 2026-11-13

### Added
- ONNX Runtime integration as portability wrapper (`ort` module)
- DirectML execution provider for AMD/Intel GPU
- OpenVINO execution provider for Intel AI Boost NPU
- CoreML execution provider for Apple Silicon
- Minimal ONNX protobuf writer for single-MatMul graphs
- F16/BF16 weight widening to F32 for ORT sessions
- Device-based EP selection (`OrtEp::for_device`)
- Runtime DLL loading via `load-dynamic` feature (no build-time dependency)

### Changed
- Dense projections (lm_head, GDN, attention, shared expert) accelerated via ORT MatMul
- Int4 routed experts kept on Fennec's own host kernels (no ONNX export)
- ORT is an optional dependency behind `ort` feature flag

## [0.8.0] - 2026-11-12

### Added
- Batched GPU prefill infrastructure: 5 batch shaders
- Op enum with 14 variants for batched operations
- GDN (Gated Delta Net) batching on device
- Attention batching on device
- MoE batching with grouped expert union
- Device fallback for batched operations

### Changed
- Grouped expert union: each unique prompt expert loaded once per prefill
- Phase-batched device MoE and mixer

## [0.7.0] - 2026-11-11

### Added
- Vulkan compute tier with concurrent CPU/GPU decode
- CUDA backend via Driver-API/PTX (no runtime dependency)
- Metal backend for Apple Silicon (implemented, unmeasured)
- Int4 quantization with grouped kernel (group 64)
- Three-tier expert routing: VRAM hot / host pool / SSD-streamed cold
- mmap FQ4 sidecar cache for cold load within 180 s boot window
- F16 device path for dense + lm_head reaching GPU
- BPE tokenizer and sampling
- OpenAI-compatible HTTP serve on `127.0.0.1:8184`

### Changed
- Linear-attention mixer host half moved to GPU
- Compacted agent payload measured on live engine
- Engine-vs-AirLLM benchmark added

## [0.6.0] - 2026-11-10

### Added
- NPU backend support (Intel AI Boost via OpenVINO)
- AMD backend support (via DirectML)
- GPU backend auto-detection
- Host expert pool defaulting to 4 GB on backend auto
- Vulkan tier requested when backend auto finds a loader

### Changed
- Default shared config roots to user home instead of D:
- Model roots made drive-agnostic

## [0.5.0] - 2026-11-09

### Added
- Initial Fennec engine commit
- Rust-based LLM inference engine
- Replaces three former engines (fox, Colibri, rabbit)
- OpenAI-compatible HTTP API
- Qwen3.5-MoE model architecture support
