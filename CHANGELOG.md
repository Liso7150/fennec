# Changelog

All notable changes to fennec, the native Rust LLM inference engine that runs
Qwen3.5/3.6 MoE, dense Qwen3 and Edge0-family latent-MoE checkpoints over
Vulkan, CUDA, Metal, ONNX Runtime and CPU backends.

Version 1.0.0 is published as `v1.0.0`. Earlier engine history was source-only; the `v0.1.0` … `v0.9.0` ladder belongs to the suite products. Dates below are commit dates unless a release date is shown. A commit subject carrying `2.5.0` is quoted as a literal historical message, not as a release.

## [1.0.0] - 2026-11-24
Version 1.0.0 is the current release.
Measured on the working tree: **234 passed, 0 failed** — the results hub in the
suite checkout (`benchmarks/artifacts/hub/RESULTS.json`, `updated_at`
2026-11-24T13:27:28Z) lists this repo's lane as status `measured`, 234 passed,
0 failed. The README records the same run as 234 passed, 0 failed, 6 ignored
across 12 suites for `cargo test --release --offline`. `Cargo.toml`
says `version = "1.0.0"`, edition 2024, `rust-version = 1.85`.

### Added
- Three-tier expert routing as the shipping shape: device (VRAM, FQ4 over
  SPIR-V or PTX), host (system RAM, zero-copy borrow) and disk (streamed from
  NVMe), planned by `src/sched.rs` and bucketed per pick
  (`a772431`, three-tier commit)
- SSD prefetching for cold experts: `src/prefetch.rs` (LRU cache plus a
  background prefetcher thread) wired to real shard loading (`fe7bf5f`) and into
  every cold expert path (`1689f25`), with hit/miss counters surfaced in
  `main.rs` output and golden coverage in `tests/golden.rs`
- `--max-memory-gb`: a strict limit for dynamic tier placement, so a capped pool
  is enforced absolutely instead of a "resident" tier degrading into page faults
  (`2692c4d`)
- A plugin set under `src/plugins/`: `cache_clock_plugin`,
  `effort_optimizer_plugin`, `handoff_plugin`, `image_preprocess_plugin`,
  `input_router_plugin`, `model_router_plugin`, `research_router_plugin`, and an
  expanded `tokenizer_plugin`, registered through `src/plugin.rs` (`61dbebd`,
  `34767ca`)
- Tag-triggered Windows/macOS/Linux release packaging publishes updater-compatible
  `fennec-<os>-<arch>` binaries and basename-only `SHA256SUMS`; Windows assets are
  explicitly unsigned because SAC remains off
- `src/updater.rs`: stable binary installation now requires the exact checksums
  asset, one exact basename entry, a 64-hex digest and a matching hash before the
  current executable is replaced; the experimental channel follows `master`
- Profiler stage splits: prefill stages broken out so chunk cost is attributable
  (`5e919ae`), and `MIXER_DEV` split into upload / compute / readback (`893dd75`);
  the stage profiler is billed on decode steps rather than prompt passes
  (`abb8111`)
- CUDA graph replay for the device mixer, so one submit replaces the per-stage
  dispatches (`4667b33`), plus cache soundness work in the same pass
- MTP speculative decoding driven from the `mtp.*` block the checkpoint already
  carries, behind `--mtp N`: greedy only, served on the host tier and declined
  on the device tiers where measurement did not support it (`e1f1614`, round
  tests in `6f83dbb`)
- A vision lane in code: `src/vision.rs` loads a checkpoint's `vision_config`
  tower, `Model::vision_features` embeds an image, `src/serve.rs:700` wraps it in
  the `<|fim_prefix|>` block and `/health` reports `vision` from whether the tower
  loaded (`e1f1614`). This is wiring, not a verified feature: no test or smoke
  route covers the image path, and the checkpoint served on this box answers
  `vision: false` (`/health`, 2026-11-24, `wsp-qwen35-0.8b`), so no served image
  round-trip is measured here
- A serve surface that answers while busy: health is answered from state kept
  outside the model lock, `/api/generate` bills its tokens, and the reasoning vs
  content channel decision moved into a unit-testable `ThinkSplit` (`e1f1614`)
- Dense Qwen3 lane: optional MoE/GDN keys, dense SiLU FFN, configurable prefix,
  tied head (`cbdc7ed`)
- Edge0-family support: GGUF reader, latent-MoE detection, hybrid attention
  config (`7436227`), the inference pipeline over latent MoE / DeltaNet / PLE
  (`fb3cc40`, `f49f8d4`), a PLE lookup cache (`6f02704`) and a TTFT/TPOT/memory
  benchmark harness (`ccca066`)
- Cross-platform CI legs: engine build verification on Windows, macOS (ARM64)
  and Ubuntu x86_64 (`05e19a6`), plus a self-hosted GPU bench workflow
  (`8e77212`)

### Changed
- Crate moved to Rust edition 2024 across `dx.rs`, `gpu.rs`, `kernel.rs`,
  `quant.rs`, `vk.rs` (`d1c045f`)
- `COLI_*` knob aliases and the MEGAsync model roots retired (`a929a20`) — the
  engine reads `FENNEC_*` only
- Device prefill made real, then gated on measurement: pick-level expert union
  (`7de054c`), one expert read per chunk (`282847d`), served on the tier that won
  for the chunk (`ba83cb7`). Chunking beat the per-token device loop by
  8.1–8.4 ms per prompt token at 34, 126 and 531 tokens on CUDA (both orders);
  the identical paired run on Vulkan was 27.5 ms per prompt token slower at the
  served length, so that tier keeps the per-token path
  (`ab-2026-11-21-prefill-201-vulkan/picklevel.json`)
- Profiling made honest before it was trusted: the device mixer billed once
  (`fa8d2aa`), billed on the device clock with `o_proj` no longer dropped
  (`a4fb100`), a measured host window treated as a lane span rather than a cost
  per byte (`aec80a6`), and the host expert tier measured directly instead of
  echoed off the stage that uses it (`a7f8d2e`)
- Decode-path costs closed: per-token allocations removed with optimized kernels
  (`45e3f2e`), the CPU decode gap versus Colibri closed alongside a GPU serve
  deadlock fix (`ca8c963`), phase-batched device MoE and mixer with an opt-in
  `--dense-i8` host gemv (`e3ac2e0`), decode thread count handed back to layers
  built on load lanes (`8bbc4ee`)
- Kernel work priced against alternatives rather than assumed: Vulkan int4 gemv
  at eight lanes per row and eight rows per workgroup (`b4715b4`), Vulkan
  half-precision gemv with hardware convert and vec4 activation loads
  (`cfc9135`), and dispatch granularity tested against interleaving for the CPU
  int8 gemv (`c25c4f3`)
- No hardcoded default model: the model argument defaults to empty and the run
  fails with an explicit "no model directory" error naming the remedies that actually
  work (`--model <dir>`, `FENNEC_MODEL`). The text had also promised a `model` key in
  `~/.fennec/config.json`, which no engine code path reads: every `config.json` the
  engine opens is the checkpoint's own, inside the model directory
- Documentation regenerated rather than retyped: the README matrix rows are
  produced from the suite's results hub (`22b4628`), and the test-count line is
  rewritten by `python benchmarks/results.py sync`

### Fixed
- Safetensors headers now prove that each declared shape and dtype exactly span
  its `data_offsets` byte range, with checked element/byte arithmetic; malformed,
  overflowing and truncated tensors are rejected before loading
- GPU serve deadlock on the first request: `prefill_device` held the GPU mutex
  across the layer loop while the device-expert fallback re-locked it inside
  `forward`; the device tier now prefills token by token (`ca8c963`)
- int8 dense trunk wrong on every non-AVX2 build: the offset-binary bias was
  removed twice in `scalar_i8`, so ARM and AVX2-less x86 decoded a
  `-64 * wsum * scale` error; the scalar fallback now matches the SIMD path
- macOS/Ubuntu CI legs made green (`afabbd5`): two assertions encoded an
  x86-only truth — the ORT provider test expected DirectML on every OS, and the
  router-logits golden band plus the half-gemv bit-exact test assumed AVX2's
  reduction order (`1a08cb0` guards the band)
- int4 checkpoint signed-flag serialization and PCIe bandwidth detection
  (`7c63703`)
- Matrix chart generation repointed at the relocated artifacts tree (`db70674`)
- `--help` and `-h` reached the command dispatch and were rejected as unknown
  commands, while `--version` had already been special-cased ahead of it; both
  spellings now print the usage block, and a bare `version` joins `--version`,
  `-V` and `-v`. A new `tests/cli.rs` lane spawns the binary to lock every
  spelling, the unknown-command rejection, and that the no-model error names
  only the remedies the engine actually reads

### Known limits
- **Expert routing has three tiers, not four.** `src/sched.rs:19` defines
  `enum Tier { Device, Host, Disk }`. An earlier commit added an NPU tier to the
  scheduler (`34767ca`, Device→NPU→Host→Disk); that arm does not ship. There is
  no NPU expert lane: DirectML refuses the AI Boost (0x887a0004) and the
  OpenVINO provider reaches only the dense ONNX projections, so warm experts
  stay in the host tier.
- **Edge0 loads from the converted int4 container**, via the `switch_mlp`
  separate gate/up layout in `src/model.rs:2353`. Raw Edge0 NVFP4 GGUF is not a
  loadable path, and no Edge0-145B native-run figure is claimed here.
- **Served time to first token trails Colibri on the Vulkan tier**: ABBA session
  `2026-11-21T20:22:21Z`, Fennec 1.992 s against Colibri 0.797 s, 4 of 4 paired
  rounds to Colibri — the per-token prompt loop pays a full expert read per
  prompt token. Why the batched chunk costs time on SPIR-V and buys it through
  the CUDA driver is not measured.
- **Vulkan decode sits behind Colibri on the same tier**: ABBA session
  `2026-11-21T20:22:21Z`, 67.6 against 58.4 ms/token, 4 of 4 rounds, a 15.7%
  median gap against a 5.0% run-to-run spread. Expert coverage does not carry
  this gap: four device budgets on one binary moved 367 slots for 0.254 ms/token
  with the orders disagreeing (`ab-2026-11-21-devbudget-vulkan/`).
- **CPU decode leads Colibri**: ABBA session `2026-11-21T01:39:40Z`,
  143.6 against 169.4 ms/token, 4 of 4 rounds, both orders agreeing.
- **CUDA decode trails llama.cpp as a level difference, not a verdict**: both
  rows in session `2026-11-21T01:39:40Z` are block medians (Fennec 65.2 against
  llama.cpp 47.9 ms/token), and the two engines read different bytes — llama.cpp
  a Q4_K_M GGUF, Fennec the suite's int4 container.
- **Planner optimism**: bytes-only pricing does not model DRAM-resident expert
  compute; the CUDA path lacks a phase-batched `moe_reduce`; Apple Metal is
  compile-verified but not performance-measured.

Every ms/token and TTFT figure above is an order-counterbalanced (ABBA) or
block-median pairing taken inside one named session from the suite's results hub
and the artifacts under `benchmarks/artifacts/`. Cross-session absolute values
are not comparable and are never ranked against each other here.
