//! Fennec — one engine that takes the good half of three others.
//!
//! * from `ferrumox/rabbit`: grouped int4 (FQ4) experts and a usage-ranked
//!   resident-expert pool, rebuilt on portable `mmap` instead of `io_uring`
//!   so it compiles on Windows;
//! * from `JustVugg/colibri`: the device/host/disk tier scheduler, fed by the
//!   real router histogram, and its telemetry line format;
//! * from `ferrumox/fox`: reading the public checkpoint straight from
//!   safetensors instead of needing a converted GGUF, and the same
//!   `/api/generate` + `/v1/chat/completions` surface, in `serve`.
//!
//! The hot path is `std` + `memmap2` + `libloading`: no CUDA toolkit, no MSVC,
//! no ash. `serde_json` speaks the HTTP and sidecar formats, `fancy-regex`
//! carries the tokenizer's lookahead split, `unicode-normalization` its NFC
//! step. `x86_64-pc-windows-gnu` is the build we test.
//!
//! The device tier picks its backend by vendor (`backend`): CUDA on NVIDIA
//! through the driver library that ships with the driver, Metal on Apple
//! through MoltenVK, Vulkan everywhere else. No backend needs an SDK at build
//! time - the SPIR-V and the PTX are both checked in.

pub mod backend;
pub mod config;
pub mod cuda;
pub mod cudagpu;
pub mod dx;
pub mod edge0;
pub mod gguf;
pub mod gpu;
pub mod kernel;
pub mod mem;
pub mod model;
pub mod npy;
pub mod ort;
pub mod plugin;
pub mod plugins;
pub mod pool;
pub mod prof;
pub mod quant;
pub mod safetensors;
pub mod sched;
pub mod serve;
pub mod tokenizer;
pub mod vk;

pub use config::{LayerKind, TextConfig};
pub use quant::Q4;
pub use safetensors::Shards;
pub use sched::{Bandwidth, Plan, Usage};

/// Version string carried by the FQ4 sidecar header and the CLI.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
