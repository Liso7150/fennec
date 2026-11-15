# Changelog

All notable changes to the Fennec inference engine.

## [Unreleased]

### Fixed
- **GPU serve deadlock on the first request.** `Model::prefill_device` held the
  GPU mutex across the layer loop while the MoE device-expert fallback re-locked
  the same non-reentrant mutex inside `forward`, hanging every Vulkan/CUDA serve
  request. The batched device-prefill scratch also overlapped the decode MoE
  arena. The device tier now prefills the prompt token by token through the
  proven `forward` path; decode throughput is unchanged. Verified: Vulkan serve
  returns correct output, all 115 tests pass.

### Added
- Host-pool budget is capped to available system RAM (`mem::available_bytes`,
  Windows `GlobalMemoryStatusEx` / Linux `MemAvailable`) less headroom, so a busy
  box streams cold experts on purpose instead of faulting a too-large "RAM" pool
  every token. Prevents the thrashing seen when several engine instances exhaust
  physical memory.

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
- **Vulkan f16 gemv: lazy subnormal conversion** (`gemv_f16.comp`). Normal
  weights (e != 0, the vast majority of a trained head) return early without
  `findMSB`; only subnormals pay for the normalisation. Same arithmetic, lm_head
  ~11.6 → 10.2 ms/token.

### Measured
- Steady-state backend matrix (qwen36-35b, 64 tokens, 3 pairs, 2 discarded
  full-length warm-ups, RTX 4060 Laptop + Core Ultra 7 155H). Same-session
  decode TPOT, lower is better:
  - **CPU: Fennec 92.0 ms/token vs Colibri 111.6 — Fennec wins ~18%.**
  - **CUDA: Fennec 43.4 ms/token vs Colibri Vulkan 45.1 — Fennec wins ~4%.**
  - Vulkan: Fennec 71.2 ms/token vs Colibri 56.1 (same session) — Fennec still
    trails ~27%, down from ~51% before the host-expert work. The Vulkan critical
    path is GPU-shader bound (mixer device + lm_head + fence waits); the same
    engine on CUDA beats Colibri, so the gap is the Vulkan SPIR-V kernels, a
    measured plateau recorded in the defect register. On NVIDIA, CUDA is the
    recommended device tier.
  - Before these changes Fennec CPU was 154.5 ms/token (Colibri 102.4 in that
    session); the host-expert and lm_head work closed the CPU gap.
- AVX-VNNI `vpdpbusd` was tried for the host dot and reverted: this hybrid CPU
  reports `avxvnni` in CPUID but traps the 256-bit instruction (SIGILL), so the
  maddubs path stays. Colibri's `IDOT_KERNEL` is also `avx2` on this box.

## [1.0.0] - 2026-11-14

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

## [2.5.0] - 2026-11-13

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
