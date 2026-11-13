//! The qwen3_5_moe block math, transcribed from the transformers reference and
//! driven by Fennec's own tensors. Every function here has a golden test.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};

use crate::config::{LayerKind, Result, TextConfig};
use crate::gpu::Gpu;
use crate::kernel;
use crate::pool::Stream;
use crate::prof;
use crate::quant::{DotPrep, Q4};
use crate::safetensors::{DType, Shards};
use crate::sched::Plan;

/// The prefix a routed `qwen3_5_moe` checkpoint stores its text stack under.
/// Dense Qwen3 exports use `model.` instead, and no key in config.json records
/// the difference, so `bind_prefix` reads it back off the shard index and every
/// name below is built from `cfg.prefix`.
pub use crate::config::DEFAULT_PREFIX as PREFIX;

/// Point the config at the prefix this checkpoint actually uses.
pub fn bind_prefix(sh: &Shards, cfg: &mut TextConfig) {
    cfg.prefix = detect_prefix(sh);
}

/// The prefix a shard index stores the text stack under. The embedding tensor
/// is the one name every architecture here has to have, so its parent is the
/// prefix; a set without one keeps the default and fails later with the usual
/// "missing tensor" name in the error.
pub fn detect_prefix(sh: &Shards) -> String {
    sh.names()
        .iter()
        .find_map(|n| n.strip_suffix(".embed_tokens.weight"))
        .map(str::to_string)
        .unwrap_or_else(|| PREFIX.to_string())
}

pub fn dense(sh: &Shards, name: &str) -> Result<(Vec<f32>, usize, usize)> {
    let i = sh
        .info(name)
        .ok_or_else(|| format!("missing tensor {name}"))?;
    let cols = *i.shape.last().ok_or_else(|| format!("{name}: no last dim"))?;
    let rows = i.elems() / cols;
    Ok((sh.tensor_f32(name)?, rows, cols))
}

/// A dense projection kept in the checkpoint's own precision.
///
/// Widening bf16/f16 to f32 at load doubled the bytes every decode token reads
/// through these matrices, and the mixer is bandwidth-bound: 72 ms of a 289 ms
/// token on the CPU tier. The half kernels widen inside the vector loop instead,
/// eight lanes per F16C op, so the same arithmetic reads half the memory and the
/// load copies half the bytes. An f32 checkpoint keeps the f32 path.
/// Opt-in: store the attention projections in VRAM as int4 instead of f16,
/// freeing device bytes for more resident routed experts (the host expert dot
/// is the decode bottleneck). Default off keeps the tight f16 golden bands.
static DENSE_INT4: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub fn set_dense_int4(v: bool) {
    DENSE_INT4.store(v, std::sync::atomic::Ordering::Relaxed);
    crate::sched::DENSE_DEV_BPP.store(if v { 1 } else { 2 }, std::sync::atomic::Ordering::Relaxed);
}
pub fn dense_int4() -> bool { DENSE_INT4.load(std::sync::atomic::Ordering::Relaxed) }

/// Opt-in: hold the mixer's dense projections at one byte per weight on the
/// host instead of two. On the CPU tier those projections are the largest block
/// of the token (~2.3 GiB of bf16 per token) and they are bandwidth-bound, so
/// halving the weight traffic halves the block. The 8-bit round trip costs about
/// a tenth of a percent on the dot - far less than the int4 routed experts
/// already carry - but it does move the last digits, so the default stays the
/// exact bf16/f16 path and the golden bands stay as they are. A/B with
/// `--dense-i8`.
static DENSE_I8: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub fn set_dense_i8(v: bool) { DENSE_I8.store(v, std::sync::atomic::Ordering::Relaxed); }
pub fn dense_i8() -> bool { DENSE_I8.load(std::sync::atomic::Ordering::Relaxed) }

pub enum Dense {
    F32(Vec<f32>),
    Half(Vec<u8>, kernel::Half),
    /// One byte per weight plus a scale per row, for the host dot (`--dense-i8`).
    I8(I8Mat),
    /// The projection parked in an ONNX Runtime session (the `ort` portability
    /// backend). Building it consumes the CPU weights; `gemv` then runs the
    /// MatMul graph on the device's provider instead of the host kernel. The
    /// int4 routed experts never come here - they have no ONNX export.
    #[cfg(feature = "ort")]
    Ort(crate::ort::OrtDense),
}

impl Dense {
    fn load(sh: &Shards, name: &str) -> Result<Dense> {
        let i = sh.info(name).ok_or_else(|| format!("missing tensor {name}"))?;
        let fmt = match i.dtype {
            DType::BF16 => Some(kernel::Half::Bf16),
            DType::F16 => Some(kernel::Half::F16),
            _ => None,
        };
        match fmt {
            Some(f) => {
                let bytes = &sh.data(i.file)[i.off..i.off + i.elems() * 2];
                Ok(Dense::Half(bytes.to_vec(), f))
            }
            None => Ok(Dense::F32(sh.tensor_f32(name)?)),
        }
    }

    /// One byte per weight plus a row scale, when `--dense-i8` is on.
    pub fn quantize_i8(self, rows: usize, cols: usize) -> Dense {
        match I8Mat::quantize(&self, rows, cols) {
            Some(m) => Dense::I8(m),
            None => self,
        }
    }

    pub fn gemv(&self, rows: usize, cols: usize, x: &[f32], out: &mut [f32], threads: usize) {
        match self {
            Dense::F32(w) => kernel::gemv(w, rows, cols, x, out, threads),
            Dense::Half(w, f) => kernel::gemv_half(w, rows, cols, x, out, threads, *f),
            Dense::I8(m) => kernel::gemv_i8(&m.w, &m.wsum, &m.scale, rows, cols, x, out, threads),
            // ORT manages its own thread pool and already knows its shape; the
            // host `rows`/`cols`/`threads` are the CPU kernel's contract only.
            #[cfg(feature = "ort")]
            Dense::Ort(d) => d.gemv(x, out).expect("ort dense gemv"),
        }
    }

    fn clear(&mut self) {
        *self = Dense::F32(Vec::new());
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Dense::F32(w) => w.is_empty(),
            Dense::Half(w, _) => w.is_empty(),
            Dense::I8(m) => m.w.is_empty(),
            #[cfg(feature = "ort")]
            Dense::Ort(_) => false,
        }
    }

    /// Replace the CPU weights with an ONNX Runtime MatMul session. `rows` and
    /// `cols` are the projection's shape - the same pair `gemv` is called with.
    /// The host copy is dropped: the session owns the widened weight from here.
    #[cfg(feature = "ort")]
    pub fn into_ort(self, ort: &crate::ort::Ort, rows: usize, cols: usize) -> Result<Dense> {
        let d = match self {
            Dense::Half(w, f) => ort.dense(&w, rows, cols, matches!(f, kernel::Half::F16))?,
            Dense::F32(w) => ort.dense_f32(&w, rows, cols)?,
            Dense::I8(m) => ort.dense_f32(&m.to_f32(), rows, cols)?,
            Dense::Ort(d) => d,
        };
        Ok(Dense::Ort(d))
    }
}

/// A projection held at one byte per weight, with one scale per row, so the host
/// dot reads half the bytes of the bf16 original.
pub struct I8Mat {
    pub w: Vec<i8>,
    /// `sum(w)` per row, which is what the offset-binary activations owe back.
    pub wsum: Vec<i32>,
    pub scale: Vec<f32>,
}

impl I8Mat {
    /// Quantise `rows x cols` out of a dense projection. A shape mismatch is a
    /// caller bug rather than something to paper over, so it returns `None` and
    /// the caller keeps the weights it had.
    fn quantize(src: &Dense, rows: usize, cols: usize) -> Option<I8Mat> {
        if rows == 0 || cols == 0 {
            return None;
        }
        let mut w = vec![0i8; rows * cols];
        let mut wsum = vec![0i32; rows];
        let mut scale = vec![0f32; rows];
        let mut put = |r: usize, c: usize, v: f32, inv: f32| {
            let q = (v * inv).round().clamp(-127.0, 127.0) as i8;
            w[r * cols + c] = q;
            wsum[r] += q as i32;
        };
        match src {
            Dense::Half(bytes, fmt) => {
                if bytes.len() != rows * cols * 2 {
                    return None;
                }
                // Decode each row once into a scratch row: the fold needs a max
                // before the quantise pass, and decoding bf16 twice per element
                // added ~7 s to the load on its own.
                let mut tmp = vec![0f32; cols];
                for r in 0..rows {
                    let row = &bytes[r * cols * 2..(r + 1) * cols * 2];
                    for (c, t) in tmp.iter_mut().enumerate() {
                        *t = fmt.decode(u16::from_le_bytes([row[c * 2], row[c * 2 + 1]]));
                    }
                    let amax = tmp.iter().fold(0f32, |m, x| m.max(x.abs()));
                    scale[r] = if amax > 0.0 { amax / 127.0 } else { 0.0 };
                    let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
                    for c in 0..cols {
                        put(r, c, tmp[c], inv);
                    }
                }
            }
            Dense::F32(v) => {
                if v.len() != rows * cols {
                    return None;
                }
                for r in 0..rows {
                    let row = &v[r * cols..(r + 1) * cols];
                    let amax = row.iter().fold(0f32, |m, x| m.max(x.abs()));
                    scale[r] = if amax > 0.0 { amax / 127.0 } else { 0.0 };
                    let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
                    for c in 0..cols {
                        put(r, c, row[c], inv);
                    }
                }
            }
            #[cfg(feature = "ort")]
            Dense::Ort(_) => return None,
            Dense::I8(_) => return None,
        }
        Some(I8Mat { w, wsum, scale })
    }

    /// The weights back as f32, for the ONNX Runtime path only.
    #[cfg(feature = "ort")]
    fn to_f32(&self) -> Vec<f32> {
        let rows = self.scale.len().max(1);
        let cols = self.w.len() / rows;
        let mut out = Vec::with_capacity(self.w.len());
        for (r, s) in self.scale.iter().enumerate() {
            for c in 0..cols {
                out.push(self.w[r * cols + c] as f32 * s);
            }
        }
        out
    }
}

pub fn dense_half(sh: &Shards, name: &str) -> Result<(Dense, usize, usize)> {
    let i = sh.info(name).ok_or_else(|| format!("missing tensor {name}"))?;
    let cols = *i.shape.last().ok_or_else(|| format!("{name}: no last dim"))?;
    let rows = i.elems() / cols;
    Ok((Dense::load(sh, name)?, rows, cols))
}

/// An RMSNorm weight, in the form `kernel::rmsnorm` expects.
///
/// The kernel applies `(1 + w)` because the Qwen3.5/3.6 checkpoints store the
/// delta. A plain Qwen3 stores the full gain, so the 1 comes off at load: the
/// forward pass, the device batches and every shader keep exactly one
/// convention, and the dialect stays a property of the weights rather than of
/// the code that reads them. The gated delta net's norm is a full gain in both
/// dialects and does not come through here.
fn norm_w(sh: &Shards, name: &str, zero_centered: bool) -> Result<Vec<f32>> {
    let mut w = dense(sh, name)?.0;
    if !zero_centered {
        for v in w.iter_mut() {
            *v -= 1.0;
        }
    }
    Ok(w)
}

// --- the ORT portability backend: dense projections as MatMul sessions ------
//
// With `--backend ort` there is no arena `Gpu` tier, so every dense projection
// would otherwise run on the host kernel. `Model::enable_ort` walks the same
// projections `attach_dense` would park in VRAM and replaces each with an ONNX
// Runtime session on the device's provider (DirectML / OpenVINO / CoreML). The
// int4 routed experts stay on Fennec's host kernels - no ONNX export, no tiered
// residency. Each shape below is the exact pair its `gemv` call site uses.

#[cfg(feature = "ort")]
fn replace_with_ort(d: &mut Dense, ort: &crate::ort::Ort, rows: usize, cols: usize) -> Result<()> {
    *d = std::mem::replace(d, Dense::F32(Vec::new())).into_ort(ort, rows, cols)?;
    Ok(())
}

#[cfg(feature = "ort")]
impl MoeLayer {
    fn dense_to_ort(&mut self, ort: &crate::ort::Ort) -> Result<()> {
        let h = self.cfg.hidden_size;
        let e = self.cfg.num_experts;
        let i = self.cfg.shared_expert_intermediate_size;
        replace_with_ort(&mut self.gate, ort, e, h)?;
        replace_with_ort(&mut self.s_gate, ort, i, h)?;
        replace_with_ort(&mut self.s_up, ort, i, h)?;
        replace_with_ort(&mut self.s_down, ort, h, i)?;
        Ok(())
    }
}

#[cfg(feature = "ort")]
impl GdnLayer {
    fn dense_to_ort(&mut self, ort: &crate::ort::Ort) -> Result<()> {
        let cfg = &self.cfg;
        let (h, cd, vd, nv) = (cfg.hidden_size, cfg.conv_dim(), cfg.value_dim(), cfg.linear_num_value_heads);
        replace_with_ort(&mut self.in_qkv, ort, cd, h)?;
        replace_with_ort(&mut self.in_z, ort, vd, h)?;
        replace_with_ort(&mut self.in_b, ort, nv, h)?;
        replace_with_ort(&mut self.in_a, ort, nv, h)?;
        replace_with_ort(&mut self.out_proj, ort, h, vd)?;
        Ok(())
    }
}

#[cfg(feature = "ort")]
impl AttnLayer {
    fn dense_to_ort(&mut self, ort: &crate::ort::Ort) -> Result<()> {
        let cfg = &self.cfg;
        let (h, nh, nkv, hd) = (cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim);
        let q_out = if cfg.attn_output_gate { nh * hd * 2 } else { nh * hd };
        replace_with_ort(&mut self.q, ort, q_out, h)?;
        replace_with_ort(&mut self.k, ort, nkv * hd, h)?;
        replace_with_ort(&mut self.v, ort, nkv * hd, h)?;
        replace_with_ort(&mut self.o, ort, h, nh * hd)?;
        Ok(())
    }
}

#[cfg(feature = "ort")]
impl Layer {
    fn dense_to_ort(&mut self, ort: &crate::ort::Ort) -> Result<()> {
        if let Some(m) = &mut self.moe {
            m.dense_to_ort(ort)?;
        }
        if let Some(g) = &mut self.gdn {
            g.dense_to_ort(ort)?;
        }
        if let Some(a) = &mut self.attn {
            a.dense_to_ort(ort)?;
        }
        Ok(())
    }
}

/// The device tier of one layer: experts whose FQ4 weights live in VRAM and
/// whose whole gate_up -> silu -> down chain runs there. The `Gpu` is shared
/// with the rest of the stack and behind a mutex because a command buffer and
/// its fence are one resource, not one per layer.
pub struct DevExperts {
    pub gpu: Arc<Mutex<Gpu>>,
    /// expert id -> device slot, or -1
    pub slot: Vec<i32>,
    /// per slot: gate_up nibbles, gate_up scales, down nibbles, down scales
    pub bufs: Vec<[usize; 4]>,
    pub group: usize,
    /// The shared expert's gate/up/down halves, when the checkpoint stores them
    /// as f16 or bf16. It is active on every token, so leaving it on the host
    /// put 503 MiB of widened f32 weights on the critical path *after* the
    /// device wait — 26 ms/token measured, while the device sat idle.
    pub shared: Option<[usize; 3]>,
}

/// Arena layout for one MoE batch, in floats: x at 0, the device accumulator
/// next, then a disjoint gate_up / hidden / out block per routed pick so all k
/// experts can be in flight in one command buffer. Returns (acc, base, stride).
fn moe_arena(cfg: &TextConfig) -> (usize, usize, usize) {
    let (h, i) = (cfg.hidden_size, cfg.moe_intermediate_size);
    (h, 2 * h, 4 * i + h)
}

/// The lm_head parked in VRAM: its bf16 buffer, where the logits land in the
/// activation arena, and the row count that buffer was uploaded with.
pub struct DevHead {
    pub buf: usize,
    pub off: usize,
    pub rows: usize,
}

/// Dense activations live above the MoE arena and the logits block. Attention
/// and the delta net run on the CPU between two device gemv batches, so their
/// inputs and outputs need a region neither of the other two ever touches.
const DENSE_OFF: usize = 1 << 19;
const DENSE_FLOATS: usize = 1 << 15;

fn f32_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn bytes_into_f32(b: &[u8], out: &mut [f32]) {
    assert_eq!(b.len(), out.len() * 4);
    unsafe {
        std::ptr::copy_nonoverlapping(b.as_ptr(), out.as_mut_ptr() as *mut u8, b.len());
    }
}

/// The raw two-byte-per-weight bytes of a projection, with its shape and which
/// half format they are in. None when the checkpoint stores it as something
/// wider: converting f32 weights down to a half to get them onto the device
/// would throw away mantissa bits the CPU path keeps. f16 and bf16 are both
/// accepted because both are lossless here — each keeps its own shader.
fn half_of<'a>(sh: &'a Shards, name: &str) -> Option<(&'a [u8], usize, usize, bool)> {
    let i = sh.info(name)?;
    let f16 = match i.dtype {
        DType::BF16 => false,
        DType::F16 => true,
        _ => return None,
    };
    let cols = *i.shape.last()?;
    Some((&sh.data(i.file)[i.off..i.off + i.elems() * 2], i.elems() / cols, cols, f16))
}

/// The dense projections of a gated delta net layer, parked in VRAM as the
/// checkpoint's own half format. With `mixer` present the conv, the delta
/// recursion and the gated norm run there too, so a layer is one round trip.
pub struct GdnDev {
    pub gpu: Arc<Mutex<Gpu>>,
    pub qkv: usize,
    pub z: usize,
    pub b: usize,
    pub a: usize,
    pub out: usize,
    pub mixer: Option<GdnMixer>,
}

/// Buffers for the fused mixer. The two state buffers are separate so neither
/// kernel needs a state-offset push constant: every pipeline shares one layout
/// of four storage bindings and eight u32 push slots, and `gdn_rec` uses all
/// eight already.
pub struct GdnMixer {
    pub conv_w: usize,
    /// a_log[nv] | dt_bias[nv] | norm_w[dv], packed into one binding.
    pub params: usize,
    pub conv_state: usize,
    pub rec_state: usize,
}

/// The same for a full-attention layer: q/k/v in, o out.
pub struct AttnDev {
    pub gpu: Arc<Mutex<Gpu>>,
    pub q: usize,
    pub k: usize,
    pub v: usize,
    pub o: usize,
    /// int4 dense: scale-buffer ids per projection and the quant group.
    pub q4: bool,
    pub qs: [usize; 4],
    pub group: usize,
}

/// Upload a whole projection set, or nothing. A layer with a tensor wider than
/// two bytes keeps the CPU's f32 copy rather than losing mantissa bits on half
/// of them, and spends no device memory finding that out.
fn put_set(g: &mut Gpu, sh: &Shards, names: &[String]) -> Result<Option<Vec<usize>>> {
    let mut src = Vec::with_capacity(names.len());
    for n in names {
        match half_of(sh, n) {
            Some((bytes, _, _, f16)) => src.push((bytes, f16)),
            None => return Ok(None),
        }
    }
    let mut bufs = Vec::with_capacity(src.len());
    for (bytes, f16) in src {
        bufs.push(g.upload_half(bytes, f16)?);
    }
    Ok(Some(bufs))
}

/// One layer's MoE FFN: router and shared expert in the checkpoint's precision,
/// int4 grouped experts.
pub struct MoeLayer {
    pub cfg: Arc<TextConfig>,
    pub gate: Dense,
    pub s_gate: Dense,
    pub s_up: Dense,
    pub s_down: Dense,
    pub s_egate: Vec<f32>,
    /// rows = ids.len() * 2 * I, cols = H  (slot s: rows [2I*s, 2I*s+I) gate, then up)
    pub gu: Q4,
    /// rows = ids.len() * H, cols = I
    pub dn: Q4,
    /// The same resident set borrowed from the checkpoint's mapping, one pair per
    /// slot, when the container is already int4. `gu`/`dn` are then empty: the
    /// nibbles are clean file pages, not 13 GiB of anonymous memory the OS has to
    /// keep resident or compress.
    pub mapped: Option<Vec<(Q4, Q4)>>,
    /// The expert ids the two matrices hold, in slot order.
    pub ids: Vec<usize>,
    /// expert id -> slot, or -1 for an expert that lives in `stream` instead.
    /// A table rather than a map because the router hands back ids and this runs
    /// eight times a token.
    pub slot: Vec<i32>,
    pub stream: Option<Mutex<Stream>>,
    pub dev: Option<DevExperts>,
    pub threads: usize,
    /// Routed picks answered by each tier, and the CPU milliseconds spent on the
    /// host and disk tiers while a device batch was in flight. The overlap is
    /// the claim the split exists for, so it gets counted rather than asserted.
    pub dev_hits: AtomicU64,
    pub host_hits: AtomicU64,
    pub disk_hits: AtomicU64,
    pub overlap_ns: AtomicU64,
}

#[derive(Debug, Clone, Default)]
pub struct Routing {
    pub ids: Vec<usize>,
    pub weights: Vec<f32>,
}

impl MoeLayer {
    /// Every expert resident, straight from the checkpoint. The shape a machine
    /// with enough RAM for the whole layer uses; `load_set` is the general case.
    pub fn load(sh: &Shards, cfg: Arc<TextConfig>, layer: usize, group: usize, threads: usize) -> Result<MoeLayer> {
        let all: Vec<usize> = (0..cfg.num_experts).collect();
        let t0 = std::time::Instant::now();
        let (gu, dn) = gather_experts(sh, &cfg, layer, &all, group, threads)?;
        eprintln!(
            "[fennec] layer {layer}: {} experts read and quantised in {:.1} s, {:.2} GiB int4",
            all.len(),
            t0.elapsed().as_secs_f64(),
            (gu.bytes_total() + dn.bytes_total()) as f64 / 2f64.powi(30)
        );
        Self::assemble(sh, cfg, layer, all, gu, dn, None, None, threads)
    }

    /// Exactly these experts in RAM, the rest answered from `stream` when the
    /// router picks one. `ids` arrives in the order the plan ranks them, so the
    /// hottest expert is slot 0 and a truncated list is a truncated cache.
    ///
    /// A pre-quantised container is borrowed rather than copied: `map_experts`
    /// hands back one mapped pair per slot and the gathered matrices stay empty.
    pub fn load_set(
        sh: &Arc<Shards>,
        cfg: Arc<TextConfig>,
        layer: usize,
        ids: &[usize],
        stream: Option<Stream>,
        group: usize,
        threads: usize,
    ) -> Result<MoeLayer> {
        if let Some(mapped) = map_experts(sh, &cfg, layer, ids, group)? {
            let gu = Q4::empty(cfg.hidden_size, group);
            let dn = Q4::empty(cfg.moe_intermediate_size, group);
            return Self::assemble(sh, cfg, layer, ids.to_vec(), gu, dn, Some(mapped), stream, threads);
        }
        let (gu, dn) = gather_experts(sh, &cfg, layer, ids, group, threads)?;
        Self::assemble(sh, cfg, layer, ids.to_vec(), gu, dn, None, stream, threads)
    }

    fn assemble(
        sh: &Shards,
        cfg: Arc<TextConfig>,
        layer: usize,
        ids: Vec<usize>,
        gu: Q4,
        dn: Q4,
        mapped: Option<Vec<(Q4, Q4)>>,
        stream: Option<Stream>,
        threads: usize,
    ) -> Result<MoeLayer> {
        let p = format!("{}.layers.{layer}.mlp", cfg.prefix);
        let (gate, _, _) = dense_half(sh, &format!("{p}.gate.weight"))?;
        let (s_gate, _, _) = dense_half(sh, &format!("{p}.shared_expert.gate_proj.weight"))?;
        let (s_up, _, _) = dense_half(sh, &format!("{p}.shared_expert.up_proj.weight"))?;
        let (s_down, _, _) = dense_half(sh, &format!("{p}.shared_expert.down_proj.weight"))?;
        let (s_egate, _, _) = dense(sh, &format!("{p}.shared_expert_gate.weight"))?;
        let mut slot = vec![-1i32; cfg.num_experts];
        for (s, &e) in ids.iter().enumerate() {
            slot[e] = s as i32;
        }
        Ok(MoeLayer {
            cfg,
            gate,
            s_gate,
            s_up,
            s_down,
            s_egate,
            gu,
            dn,
            mapped,
            ids,
            slot,
            stream: stream.map(Mutex::new),
            dev: None,
            threads: threads.max(1),
            dev_hits: AtomicU64::new(0),
            host_hits: AtomicU64::new(0),
            disk_hits: AtomicU64::new(0),
            overlap_ns: AtomicU64::new(0),
        })
    }

    pub fn from_parts(cfg: Arc<TextConfig>, parts: MoeParts, gu: Q4, dn: Q4, threads: usize) -> MoeLayer {
        // the matrices carry the resident set: gu.rows / 2I experts, ids 0..n
        let ids: Vec<usize> = (0..gu.rows / (2 * cfg.moe_intermediate_size)).collect();
        let mut slot = vec![-1i32; cfg.num_experts];
        for (s, &e) in ids.iter().enumerate() {
            slot[e] = s as i32;
        }
        MoeLayer {
            cfg,
            gate: parts.gate,
            s_gate: parts.s_gate,
            s_up: parts.s_up,
            s_down: parts.s_down,
            s_egate: parts.s_egate,
            gu,
            dn,
            mapped: None,
            ids,
            slot,
            stream: None,
            dev: None,
            threads: threads.max(1),
            dev_hits: AtomicU64::new(0),
            host_hits: AtomicU64::new(0),
            disk_hits: AtomicU64::new(0),
            overlap_ns: AtomicU64::new(0),
        }
    }

    pub fn route(&self, x: &[f32]) -> Routing {
        let h = self.cfg.hidden_size;
        let e = self.cfg.num_experts;
        let k = self.cfg.num_experts_per_tok;
        let mut logits = vec![0f32; e];
        self.gate.gemv(e, h, x, &mut logits, self.threads);
        let mut weights = vec![0f32; k];
        let mut ids = vec![0usize; k];
        kernel::softmax_topk(&logits, k, &mut weights, &mut ids);
        Routing { ids, weights }
    }

    /// y = expert_e(x) into `y` (len H). Gate/up are the two halves of the row
    /// block that holds e; `expert` is the id->slot lookup on top of it.
    pub fn expert(&self, e: usize, x: &[f32], hh: &mut [f32], y: &mut [f32]) {
        let (gu, dn, s) = self.expert_pair(e);
        let (i, h) = (self.cfg.moe_intermediate_size, self.cfg.hidden_size);
        expert_in(gu, dn, i, h, s, x, hh, y);
    }

    /// The two matrices backing expert `e`, and the row block they start at, so
    /// a caller can address single rows without going through `expert_in`.
    fn expert_pair(&self, e: usize) -> (&Q4, &Q4, usize) {
        let s = self.slot[e];
        assert!(s >= 0, "expert {e} is not resident in this layer");
        match &self.mapped {
            // A borrowed pair is one expert, so its row block starts at 0.
            Some(v) => {
                let (gu, dn) = &v[s as usize];
                (gu, dn, 0)
            }
            None => (&self.gu, &self.dn, s as usize),
        }
    }

    /// Bytes the host tier holds, whether it owns them or borrows them.
    pub fn host_bytes(&self) -> usize {
        self.host_gu_bytes() + self.host_dn_bytes()
    }

    pub fn host_gu_bytes(&self) -> usize {
        match &self.mapped {
            Some(v) => v.iter().map(|(g, _)| g.bytes_total()).sum(),
            None => self.gu.bytes_total(),
        }
    }

    pub fn host_dn_bytes(&self) -> usize {
        match &self.mapped {
            Some(v) => v.iter().map(|(_, d)| d.bytes_total()).sum(),
            None => self.dn.bytes_total(),
        }
    }

    /// The same, counting allocator capacity rather than matrix size.
    pub fn host_bytes_held(&self) -> usize {
        match &self.mapped {
            Some(v) => v.iter().map(|(g, d)| g.bytes_held() + d.bytes_held()).sum(),
            None => self.gu.bytes_held() + self.dn.bytes_held(),
        }
    }

    /// True when the host tier is borrowed from the checkpoint's mapping.
    pub fn host_is_mapped(&self) -> bool {
        self.mapped.is_some()
    }

    /// Row `r` of the host gate_up matrix in slot-block coordinates, so a caller
    /// can dequantise a probe row without knowing which storage backs the tier.
    pub fn host_gu_row(&self, r: usize) -> Vec<f32> {
        let per = 2 * self.cfg.moe_intermediate_size;
        match &self.mapped {
            Some(v) => v[r / per].0.dequant_row(r % per),
            None => self.gu.dequant_row(r),
        }
    }

    /// Row `r` of the host down matrix, same addressing.
    pub fn host_dn_row(&self, r: usize) -> Vec<f32> {
        let per = self.cfg.hidden_size;
        match &self.mapped {
            Some(v) => v[r / per].1.dequant_row(r % per),
            None => self.dn.dequant_row(r),
        }
    }

    pub fn shared(&self, x: &[f32], out: &mut [f32]) {
        let h = self.cfg.hidden_size;
        let i = self.cfg.shared_expert_intermediate_size;
        let mut hh = vec![0f32; i];
        let mut g = vec![0f32; i];
        let mut u = vec![0f32; i];
        self.s_gate.gemv(i, h, x, &mut g, self.threads);
        self.s_up.gemv(i, h, x, &mut u, self.threads);
        for j in 0..i {
            hh[j] = kernel::silu(g[j]) * u[j];
        }
        self.s_down.gemv(h, i, &hh, out, self.threads);
    }

    /// Quantise these experts and park them in VRAM. They are *not* in the host
    /// resident set, so a routed pick is either a device dispatch or a CPU gemv,
    /// never both, and the two can run at the same time.
    pub fn attach_gpu(
        &mut self,
        sh: &Arc<Shards>,
        layer: usize,
        gpu: Arc<Mutex<Gpu>>,
        ids: &[usize],
        group: usize,
    ) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let (i, h) = (self.cfg.moe_intermediate_size, self.cfg.hidden_size);
        let t0 = std::time::Instant::now();
        let (gu, dn) = gather_experts(sh, &self.cfg, layer, ids, group, self.threads)?;
        let mut g = gpu.lock().unwrap();
        let (_, base, stride) = moe_arena(&self.cfg);
        // One slot past the routed picks: the shared expert rides in the same
        // command buffer rather than running serially after the wait. The tail
        // 16 floats hold the reduce kernel's per-pick weights.
        g.reserve(base + (self.cfg.num_experts_per_tok + 1) * stride + 16)?;
        let mut bufs = Vec::with_capacity(ids.len());
        for s in 0..ids.len() {
            let (gw, gs) = g.upload_q4(&gu.slice_rows(2 * i * s, 2 * i * (s + 1)))?;
            let (dw, ds) = g.upload_q4(&dn.slice_rows(h * s, h * (s + 1)))?;
            bufs.push([gw, gs, dw, ds]);
        }
        let p = format!("{}.layers.{layer}.mlp.shared_expert", self.cfg.prefix);
        let shared = match (
            half_of(sh, &format!("{p}.gate_proj.weight")),
            half_of(sh, &format!("{p}.up_proj.weight")),
            half_of(sh, &format!("{p}.down_proj.weight")),
        ) {
            (Some((gw, _, _, gf)), Some((uw, _, _, uf)), Some((dw, _, _, df))) => {
                Some([g.upload_half(gw, gf)?, g.upload_half(uw, uf)?, g.upload_half(dw, df)?])
            }
            // An f32 checkpoint would lose mantissa bits on the way down, and
            // the shared expert is on every token, so it stays on the host.
            _ => None,
        };
        let bytes = g.bytes_held();
        drop(g);
        let mut slot = vec![-1i32; self.cfg.num_experts];
        for (s, &e) in ids.iter().enumerate() {
            slot[e] = s as i32;
        }
        eprintln!(
            "[fennec] layer {layer}: {} experts on {} in {:.1} s, {:.2} GiB device total",
            ids.len(),
            gpu.lock().unwrap().name(),
            t0.elapsed().as_secs_f64(),
            bytes as f64 / 2f64.powi(30)
        );
        self.dev = Some(DevExperts { gpu, slot, bufs, group, shared });
        Ok(())
    }

    /// The host and disk tiers, on the CPU. Called while a device batch is in
    /// flight, which is the whole point of splitting the tiers this way.
    fn cpu_picks(&self, x: &[f32], r: &Routing, host: &[usize], cold: &[usize], parts: &mut [Vec<f32>]) {
        let (h, i) = (self.cfg.hidden_size, self.cfg.moe_intermediate_size);
        if !cold.is_empty() {
            let mut pool = self
                .stream
                .as_ref()
                .unwrap_or_else(|| panic!("a non-resident expert needs a pool"))
                .lock()
                .unwrap();
            for &n in cold {
                let (id, w) = (r.ids[n], r.weights[n]);
                let mut hh = vec![0f32; i];
                let mut y = vec![0f32; h];
                let mut part = vec![0f32; h];
                pool.with(id, |gu, dn| expert_in(gu, dn, i, h, 0, x, &mut hh, &mut y))
                    .unwrap_or_else(|e| panic!("streamed expert {id}: {e}"));
                kernel::axpy(&mut part, &y, w);
                parts[n] = part;
            }
        }
        if self.threads == 1 || host.len() < 2 {
            let mut hh = vec![0f32; i];
            let mut y = vec![0f32; h];
            for &n in host {
                self.expert(r.ids[n], x, &mut hh, &mut y);
                if parts[n].is_empty() {
                    parts[n] = vec![0f32; h];
                }
                kernel::axpy(&mut parts[n], &y, r.weights[n]);
            }
        } else {
            // Rows split across every worker, in two phases. One expert per
            // worker left ~2.4 of 8 idle: a layer averages 5.6 host picks, so
            // the picks alone cannot fill the pool. The old objection to
            // row-splitting was the second barrier's condvar round-trip
            // (+0.81 ms/layer); the pool now spins before it parks, which is
            // what makes this pay. Splitting rows does not change the arithmetic
            // each row does, so it is bit-identical to the serial path.
            let np = host.len();
            for &n in host {
                if parts[n].is_empty() {
                    parts[n] = vec![0f32; h];
                }
            }
            let mut hh = vec![0f32; np * i];
            let hhp = kernel::SharedPtr::new(hh.as_mut_ptr());
            // One activation order for the whole phase: every gate and up row
            // reads the same x, so the permutation is paid once, not per row.
            // Every expert in a layer comes from the same pack, so the first
            // one decides whether the integer dot is available to all of them.
            let int = host.first().is_some_and(|&n| self.expert_pair(r.ids[n]).0.wants_int());
            let mut px = DotPrep::with_int(h, int);
            px.fill(x);
            kernel::par_index(np * i, self.threads, |start, len| {
                for g in start..start + len {
                    let p = g / i;
                    let j = g - p * i;
                    let (gu, _, s) = self.expert_pair(r.ids[host[p]]);
                    let base = s * 2 * i;
                    let a = gu.dot_prepped(base + j, &px);
                    let b = gu.dot_prepped(base + i + j, &px);
                    unsafe { *hhp.at(p * i + j) = kernel::silu(a) * b };
                }
            });
            let pp = kernel::SharedPtr::new(parts.as_mut_ptr());
            kernel::par_index(np * h, self.threads, |start, len| {
                // Each worker owns a contiguous run of rows, so it re-permutes
                // only when the run crosses into the next expert's hidden state.
                let mut ph = DotPrep::with_int(i, int);
                let mut cur = usize::MAX;
                for g in start..start + len {
                    let p = g / h;
                    let j = g - p * h;
                    let n = host[p];
                    let (_, dn, s) = self.expert_pair(r.ids[n]);
                    if p != cur {
                        ph.fill(unsafe { std::slice::from_raw_parts(hhp.at(p * i), i) });
                        cur = p;
                    }
                    let v = dn.dot_prepped(s * h + j, &ph);
                    unsafe { (&mut *pp.at(n))[j] += r.weights[n] * v };
                }
            });
        }
    }

    /// The whole block: routed experts plus the sigmoid-gated shared expert.
    ///
    /// Picks are split three ways by where their weights live. The device picks
    /// are recorded into one command buffer and submitted first, then this same
    /// thread works the host and disk tiers while the device is still running,
    /// then it waits and reads one accumulator back. That is the CPU and the GPU
    /// doing the same token at the same time, not one after the other.
    ///
    /// On the CPU side each host pick is one worker's whole expert, gate/up then
    /// down, so no barrier sits inside the expert and the reduction tree inside
    /// each dot product is untouched.
    pub fn forward(&self, x: &[f32]) -> (Vec<f32>, Routing) {
        let (h, i) = (self.cfg.hidden_size, self.cfg.moe_intermediate_size);
        let r = {
            let _t = prof::start(prof::ROUTE);
            self.route(x)
        };
        let k = r.ids.len();
        let mut parts: Vec<Vec<f32>> = vec![Vec::new(); k];
        let (mut dev_picks, mut host_picks, mut cold) = (Vec::new(), Vec::new(), Vec::new());
        for n in 0..k {
            let id = r.ids[n];
            if self.dev.as_ref().is_some_and(|d| d.slot[id] >= 0) {
                dev_picks.push(n);
            } else if self.slot[id] >= 0 {
                host_picks.push(n);
            } else {
                cold.push(n);
            }
        }
        let mut dev_out = vec![0f32; h];
        self.dev_hits.fetch_add(dev_picks.len() as u64, Relaxed);
        self.host_hits.fetch_add(host_picks.len() as u64, Relaxed);
        self.disk_hits.fetch_add(cold.len() as u64, Relaxed);
        let dev_shared = self.dev.as_ref().and_then(|d| d.shared);
        // The sigmoid gate is 2048 f32 MACs on the host; computing it here lets
        // the shared expert's axpy ride in the same batch as the routed picks.
        let shared_gate = kernel::sigmoid(kernel::dot(&self.s_egate, x));
        if dev_picks.is_empty() && dev_shared.is_none() {
            let _t = prof::start(prof::MOE_CPU);
            self.cpu_picks(x, &r, &host_picks, &cold, &mut parts);
        } else {
            let _submit = prof::start(prof::MOE_UPLOAD);
            let d = self.dev.as_ref().unwrap();
            let (acc, base, stride) = moe_arena(&self.cfg);
            let w_off = base + (self.cfg.num_experts_per_tok + 1) * stride;
            let mut weights: Vec<f32> = dev_picks.iter().map(|&n| r.weights[n]).collect();
            if dev_shared.is_some() {
                weights.push(shared_gate);
            }
            let mut g = d.gpu.lock().unwrap();
            g.write(0, x);
            g.write(w_off, &weights);
            drop(_submit);
            {
                let _record = prof::start(prof::MOE_RECORD);
                let mut b = g.batch().unwrap();
                match &mut b {
                    crate::gpu::Batch::Vk(v) => {
                        // Phase-batched: every expert's gate_up gemv is recorded
                        // without a barrier, so independent experts overlap on
                        // the device; one barrier separates the phases and one
                        // reduce replaces the per-expert axpy chain. The old
                        // barrier-after-every-dispatch serialised the whole layer.
                        for (j, &n) in dev_picks.iter().enumerate() {
                            let [gw, gs, _, _] = d.bufs[d.slot[r.ids[n]] as usize];
                            v.gemv_q4_nb(gw, gs, 2 * i, h, d.group, 0, base + j * stride).unwrap();
                        }
                        if let Some([sg, su, _]) = dev_shared {
                            let gu_off = base + dev_picks.len() * stride;
                            v.gemv_half_nb(sg, i, h, 0, gu_off).unwrap();
                            v.gemv_half_nb(su, i, h, 0, gu_off + i).unwrap();
                        }
                        v.barrier().unwrap();
                        for j in 0..weights.len() {
                            let gu_off = base + j * stride;
                            v.silu_mul_nb(i, gu_off, gu_off + i, gu_off + 2 * i).unwrap();
                        }
                        v.barrier().unwrap();
                        for (j, &n) in dev_picks.iter().enumerate() {
                            let [_, _, dw, ds] = d.bufs[d.slot[r.ids[n]] as usize];
                            let gu_off = base + j * stride;
                            v.gemv_q4_nb(dw, ds, h, i, d.group, gu_off + 2 * i, gu_off + 4 * i).unwrap();
                        }
                        if let Some([_, _, sd]) = dev_shared {
                            let gu_off = base + dev_picks.len() * stride;
                            v.gemv_half_nb(sd, h, i, gu_off + 2 * i, gu_off + 4 * i).unwrap();
                        }
                        v.barrier().unwrap();
                        v.moe_reduce(weights.len(), h, stride, base, 4 * i, acc, w_off).unwrap();
                    }
                    crate::gpu::Batch::Cuda(c) => {
                        for (j, &n) in dev_picks.iter().enumerate() {
                            let [gw, gs, dw, ds] = d.bufs[d.slot[r.ids[n]] as usize];
                            let gu_off = base + j * stride;
                            let hh_off = gu_off + 2 * i;
                            let out_off = gu_off + 4 * i;
                            c.gemv_q4(gw, gs, 2 * i, h, d.group, 0, gu_off).unwrap();
                            c.silu_mul(i, gu_off, gu_off + i, hh_off).unwrap();
                            c.gemv_q4(dw, ds, h, i, d.group, hh_off, out_off).unwrap();
                            c.axpy(h, out_off, acc, r.weights[n], j == 0).unwrap();
                        }
                        if let Some([sg, su, sd]) = dev_shared {
                            let gu_off = base + dev_picks.len() * stride;
                            let hh_off = gu_off + 2 * i;
                            let out_off = gu_off + 4 * i;
                            c.gemv_half(sg, i, h, 0, gu_off).unwrap();
                            c.gemv_half(su, i, h, 0, gu_off + i).unwrap();
                            c.silu_mul(i, gu_off, gu_off + i, hh_off).unwrap();
                            c.gemv_half(sd, h, i, hh_off, out_off).unwrap();
                            c.axpy(h, out_off, acc, shared_gate, dev_picks.is_empty()).unwrap();
                        }
                    }
                }
                b.submit().unwrap();
            }
            let _cpu = prof::start(prof::MOE_CPU);
            let t = std::time::Instant::now();
            self.cpu_picks(x, &r, &host_picks, &cold, &mut parts);
            self.overlap_ns.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
            drop(_cpu);
            {
                let _sync = prof::start(prof::MOE_SYNC);
                g.wait().unwrap();
            }
            let _read = prof::start(prof::MOE_READ);
            g.read_into(acc, &mut dev_out);
        }
        let _shared = prof::start(prof::MOE_SHARED);
        let mut out = vec![0f32; h];
        kernel::axpy(&mut out, &dev_out, 1.0);
        for part in &parts {
            if !part.is_empty() {
                kernel::axpy(&mut out, part, 1.0);
            }
        }
        if dev_shared.is_none() {
            let mut y = vec![0f32; h];
            self.shared(x, &mut y);
            kernel::axpy(&mut out, &y, shared_gate);
        }
        (out, r)
    }
}

/// One expert's forward pass over its own two matrices: `s` picks the row block
/// in `gu` (2I rows: gate, then up) and in `dn` (H rows). Split out of
/// `MoeLayer` because a streamed expert has the same shape and no layer.
pub fn expert_in(gu: &Q4, dn: &Q4, i: usize, h: usize, s: usize, x: &[f32], hh: &mut [f32], y: &mut [f32]) {
    let base = s * 2 * i;
    let px = gu.prepare(x);
    for j in 0..i {
        let g = gu.dot_prepped(base + j, &px);
        let u = gu.dot_prepped(base + i + j, &px);
        hh[j] = kernel::silu(g) * u;
    }
    let ph = dn.prepare(hh);
    for j in 0..h {
        y[j] = dn.dot_prepped(s * h + j, &ph);
    }
}

/// The group width of a pre-quantised container, derived from its scale count.
///
/// Colibri's int4 checkpoints store one expert as `merged_weight` (U8 nibbles,
/// gate|up|down) beside `.qs` (f32 scales). Nothing in config.json records the
/// group width, and decoding 64-column scales as 32 would stretch every scale
/// over twice its columns — wrong logits, no error. So it is read back from the
/// bytes: 3*i*h weights over N scales means a group of 3*i*h/N.
pub fn packed_group(sh: &Shards, prefix: &str, h: usize, i: usize) -> Option<usize> {
    let p = format!("{prefix}.layers.0.mlp.experts.0");
    let w = sh.info(&format!("{p}.merged_weight"))?;
    let s = sh.info(&format!("{p}.qs"))?;
    let elems = 3 * i * h;
    if w.bytes() != elems / 2 || s.elems() == 0 || elems % s.elems() != 0 {
        return None;
    }
    let g = elems / s.elems();
    (g >= 8 && g % 8 == 0 && h % g == 0 && i % g == 0).then_some(g)
}

/// One expert out of a pre-quantised container. The three blocks are the same
/// size (i*h/2 bytes and i*(h/group) scales each), so gate|up is the first two
/// thirds and down is the last — the same row order FQ4 gathers into.
///
/// The nibble convention is not the same, though: `c/qwen36.c` and
/// `convert_qwen36.py::pack_int4` store a *signed* 4-bit value in [-8,7], while
/// FQ4 stores it biased by +8. Signed decodes as `(n ^ 8) - 8` and biased as
/// `n - 8`, so the two differ by exactly a flip of bit 3 in each nibble: one XOR
/// per byte adopts the container without an unpack/repack pass. Getting this
/// wrong is silent — every weight lands eight scale-steps off and the model
/// still emits tokens, just not ones that mean anything.
pub fn packed_expert(
    sh: &Shards,
    prefix: &str,
    layer: usize,
    id: usize,
    h: usize,
    i: usize,
    group: usize,
) -> Result<(Q4, Q4)> {
    let p = format!("{prefix}.layers.{layer}.mlp.experts.{id}");
    let block_bytes = i * h / 2;
    let block_scales = i * (h / group);
    let mut w = sh.slice_bytes(&format!("{p}.merged_weight"), 0, 3 * block_bytes)?;
    let s = sh.slice_f32(&format!("{p}.qs"), 0, 3 * block_scales)?;
    for b in w.iter_mut() {
        *b ^= 0x88;
    }
    let cut = 2 * block_bytes;
    let scut = 2 * block_scales;
    Ok((
        Q4::from_qs(&w[..cut], &s[..scut], 2 * i, h, group),
        Q4::from_qs(&w[cut..], &s[scut..], h, i, group),
    ))
}

/// One expert straight out of an attached FQ4 cache, when the cache covers this
/// layer at the same group size. `None` means "not cached, quantise it".
pub fn cached_expert(
    sh: &Shards,
    prefix: &str,
    layer: usize,
    id: usize,
    h: usize,
    i: usize,
    group: usize,
) -> Option<Result<(Q4, Q4)>> {
    if let Some(sc) = sh.fq4() {
        if let (Some(gi), Some(di)) = (
            sc.item(&format!("blk.{layer}.ffn.gate_up_exps")),
            sc.item(&format!("blk.{layer}.ffn.down_exps")),
        ) {
            if gi.group == group && di.group == group && gi.cols == h && di.cols == i {
                return Some(|| -> Result<(Q4, Q4)> {
                    Ok((
                        sc.rows(&gi, id * 2 * i, (id + 1) * 2 * i)?,
                        sc.rows(&di, id * h, (id + 1) * h)?,
                    ))
                }());
            }
        }
    }
    // An already-int4 checkpoint needs no cache: it is its own.
    let name = format!("{prefix}.layers.{layer}.mlp.experts.{id}.merged_weight");
    sh.contains(&name)
        .then(|| packed_expert(sh, prefix, layer, id, h, i, group))
}

/// The host tier without a copy: one borrowed pair of matrices per expert,
/// straight out of the pre-quantised container's mapping.
///
/// `None` means this checkpoint is not a packed container, so the caller has to
/// quantise into owned memory. Only the scales are read out — a ninth of the
/// bytes — and the nibbles stay clean file pages. That is the difference between
/// a host tier the OS can keep resident and one it has to compress: the owned
/// version of this same set committed 13 GiB of anonymous memory and measured
/// 48 to 85 ms/token on identical work depending on how much of it Windows let
/// stay in RAM.
///
/// The container's nibbles are two's complement where FQ4's are offset by 8, so
/// the packs are marked `signed` and the kernel flips bit 3 in the vector
/// instead of the loader flipping it in 12 GiB of bytes.
pub fn map_experts(
    sh: &Shards,
    cfg: &TextConfig,
    layer: usize,
    ids: &[usize],
    group: usize,
) -> Result<Option<Vec<(Q4, Q4)>>> {
    let (h, i) = (cfg.hidden_size, cfg.moe_intermediate_size);
    let p = format!("{}.layers.{layer}.mlp.experts", cfg.prefix);
    if !sh.contains(&format!("{p}.0.merged_weight")) || h % group != 0 {
        return Ok(None);
    }
    let block_bytes = i * h / 2;
    let block_scales = i * (h / group);
    let mut out = Vec::with_capacity(ids.len());
    for &e in ids {
        let ep = format!("{p}.{e}");
        let Some((map, off)) = sh.mapped(&format!("{ep}.merged_weight")) else {
            return Ok(None);
        };
        let s = sh.slice_f32(&format!("{ep}.qs"), 0, 3 * block_scales)?;
        let cut = 2 * block_scales;
        out.push((
            Q4::mapped(map.clone(), off, 2 * i, h, group, s[..cut].to_vec(), true),
            Q4::mapped(map, off + 2 * block_bytes, h, i, group, s[cut..].to_vec(), true),
        ));
    }
    Ok(Some(out))
}

/// The two FQ4 matrices for exactly `ids`, in that order, one expert block at a
/// time. Reading per block rather than `tensor_f32` of the whole tensor is the
/// difference between 16 MiB of fp32 staging and 1.5 GiB: the gathered set can
/// be a tenth of the layer, and the block is contiguous either way.
pub fn gather_experts(
    sh: &Shards,
    cfg: &TextConfig,
    layer: usize,
    ids: &[usize],
    group: usize,
    threads: usize,
) -> Result<(Q4, Q4)> {
    let (h, i) = (cfg.hidden_size, cfg.moe_intermediate_size);
    let p = format!("{}.layers.{layer}.mlp.experts", cfg.prefix);
    let (gu_name, dn_name) = (format!("{p}.gate_up_proj"), format!("{p}.down_proj"));
    let mut gu = Q4::with_capacity(ids.len() * 2 * i, h, group);
    let mut dn = Q4::with_capacity(ids.len() * h, i, group);
    // Hoist the cache lookup: with it, a resident set is a memcpy per expert
    // instead of a bf16 read plus a quantisation pass over the same rows.
    let cached = sh.fq4().and_then(|sc| {
        let gi = sc.item(&format!("blk.{layer}.ffn.gate_up_exps"))?;
        let di = sc.item(&format!("blk.{layer}.ffn.down_exps"))?;
        (gi.group == group && di.group == group && gi.cols == h && di.cols == i)
            .then_some((sc, gi, di))
    });
    if let Some((sc, gi, di)) = cached {
        for &e in ids {
            gu.push(&sc.rows(&gi, e * 2 * i, (e + 1) * 2 * i)?);
            dn.push(&sc.rows(&di, e * h, (e + 1) * h)?);
        }
        return Ok((gu, dn));
    }
    if sh.contains(&format!("{p}.0.merged_weight")) {
        // One expert per worker. Sequentially this ran at ~210 MB/s against an
        // NVMe part, which is a queue-depth-1 number: the drive can only fetch
        // the next 1.5 MiB block once the previous one has been copied and
        // XORed. The reads are independent, so issuing them together is what
        // the controller is built for.
        let mut blocks: Vec<Result<(Q4, Q4)>> = (0..ids.len())
            .map(|_| Ok((Q4::empty(h, group), Q4::empty(i, group))))
            .collect();
        kernel::par_rows(&mut blocks, threads, |slice, start| {
            for (k, slot) in slice.iter_mut().enumerate() {
                *slot = packed_expert(sh, &cfg.prefix, layer, ids[start + k], h, i, group);
            }
        });
        for b in blocks {
            let (g, d) = b?;
            gu.push(&g);
            dn.push(&d);
        }
        return Ok((gu, dn));
    }
    for &e in ids {
        let g = sh.slice_f32(&gu_name, e * 2 * i * h, (e + 1) * 2 * i * h)?;
        gu.push(&Q4::quantize(&g, 2 * i, h, group, threads));
        let d = sh.slice_f32(&dn_name, e * h * i, (e + 1) * h * i)?;
        dn.push(&Q4::quantize(&d, h, i, group, threads));
    }
    Ok((gu, dn))
}

pub struct MoeParts {
    pub gate: Dense,
    pub s_gate: Dense,
    pub s_up: Dense,
    pub s_down: Dense,
    pub s_egate: Vec<f32>,
}

impl MoeParts {
    pub fn load(sh: &Shards, prefix: &str, layer: usize) -> Result<MoeParts> {
        let p = format!("{prefix}.layers.{layer}.mlp");
        Ok(MoeParts {
            gate: dense_half(sh, &format!("{p}.gate.weight"))?.0,
            s_gate: dense_half(sh, &format!("{p}.shared_expert.gate_proj.weight"))?.0,
            s_up: dense_half(sh, &format!("{p}.shared_expert.up_proj.weight"))?.0,
            s_down: dense_half(sh, &format!("{p}.shared_expert.down_proj.weight"))?.0,
            s_egate: dense(sh, &format!("{p}.shared_expert_gate.weight"))?.0,
        })
    }
}

/// The FFN of a dense checkpoint: one SiLU MLP per layer, on every token.
///
/// `MoeLayer::shared` is the same gate/up/down chain with different tensor
/// names, so this reuses `Dense` and its kernels rather than transcribing the
/// math a second time. `intermediate_size` is the width the MoE lane calls
/// `moe_intermediate_size`.
pub struct MlpLayer {
    pub cfg: Arc<TextConfig>,
    pub gate: Dense,
    pub up: Dense,
    pub down: Dense,
    pub threads: usize,
}

impl MlpLayer {
    pub fn load(sh: &Shards, cfg: Arc<TextConfig>, layer: usize, threads: usize) -> Result<MlpLayer> {
        let p = format!("{}.layers.{layer}.mlp", cfg.prefix);
        let mut l = MlpLayer {
            gate: dense_half(sh, &format!("{p}.gate_proj.weight"))?.0,
            up: dense_half(sh, &format!("{p}.up_proj.weight"))?.0,
            down: dense_half(sh, &format!("{p}.down_proj.weight"))?.0,
            cfg,
            threads: threads.max(1),
        };
        if dense_i8() {
            let (i, h) = (l.cfg.intermediate_size, l.cfg.hidden_size);
            l.gate = l.gate.quantize_i8(i, h);
            l.up = l.up.quantize_i8(i, h);
            l.down = l.down.quantize_i8(h, i);
        }
        Ok(l)
    }

    /// out (len H) = down @ (silu(gate @ x) * (up @ x)). `u` carries the
    /// activation once `g` is spent, so the block holds three H/I buffers and
    /// not four.
    pub fn forward(&self, x: &[f32], out: &mut [f32]) {
        let (h, i) = (self.cfg.hidden_size, self.cfg.intermediate_size);
        let t = self.threads;
        let _p = prof::start(prof::MIXER_PROJ);
        let mut g = vec![0f32; i];
        let mut u = vec![0f32; i];
        self.gate.gemv(i, h, x, &mut g, t);
        self.up.gemv(i, h, x, &mut u, t);
        for j in 0..i {
            u[j] = kernel::silu(g[j]) * u[j];
        }
        self.down.gemv(h, i, &u, out, t);
    }
}

/// Per-layer gated-delta-net cache: conv window plus one [k_dim, v_dim] matrix
/// per value head. 32 heads x 128 x 128 x 4 B = 2 MiB per layer, per sequence.
pub struct GdnState {
    /// Rolling pre-conv inputs, time-major: `conv[i * channels + c]` holds the
    /// input `i` steps ago for channel `c`, oldest first. `conv_depthwise` reads
    /// it that way, and a whole time slice is one contiguous `channels`-long
    /// row, so rotating the window is a row copy instead of a strided shuffle.
    pub conv: Vec<f32>,
    pub rec: Vec<f32>,
    pub channels: usize,
    pub hist: usize,
    pub n_v: usize,
    pub dk: usize,
    pub dv: usize,
    pub filled: usize,
    /// True once `conv`/`rec` have been handed to a device mixer, which then
    /// owns them. Not pointer identity: `serve` builds a fresh state vec per
    /// request and a freed allocation's address gets reused, so identity alone
    /// would skip the upload and decode against the previous request's state.
    pub mirrored: bool,
}

impl GdnState {
    pub fn new(cfg: &TextConfig) -> GdnState {
        let c = cfg.conv_dim();
        let k = cfg.linear_conv_kernel_dim - 1;
        GdnState {
            conv: vec![0f32; c * k],
            rec: vec![0f32; cfg.linear_num_value_heads * cfg.linear_key_head_dim * cfg.linear_value_head_dim],
            channels: c,
            hist: k,
            n_v: cfg.linear_num_value_heads,
            dk: cfg.linear_key_head_dim,
            dv: cfg.linear_value_head_dim,
            filled: 0,
            mirrored: false,
        }
    }
}

pub struct GdnLayer {
    pub cfg: Arc<TextConfig>,
    pub in_qkv: Dense,
    pub in_z: Dense,
    pub in_b: Dense,
    pub in_a: Dense,
    pub conv_w: Vec<f32>,
    pub a_log: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub norm_w: Vec<f32>,
    pub out_proj: Dense,
    pub threads: usize,
    pub dev: Option<GdnDev>,
}

impl GdnLayer {
    pub fn load(sh: &Shards, cfg: Arc<TextConfig>, layer: usize, threads: usize) -> Result<GdnLayer> {
        let p = format!("{}.layers.{layer}.linear_attn", cfg.prefix);
        let (h, cd, vd, nv) = (
            cfg.hidden_size,
            cfg.conv_dim(),
            cfg.value_dim(),
            cfg.linear_num_value_heads,
        );
        let mut l = GdnLayer {
            cfg,
            in_qkv: dense_half(sh, &format!("{p}.in_proj_qkv.weight"))?.0,
            in_z: dense_half(sh, &format!("{p}.in_proj_z.weight"))?.0,
            in_b: dense_half(sh, &format!("{p}.in_proj_b.weight"))?.0,
            in_a: dense_half(sh, &format!("{p}.in_proj_a.weight"))?.0,
            conv_w: dense(sh, &format!("{p}.conv1d.weight"))?.0,
            a_log: dense(sh, &format!("{p}.A_log"))?.0,
            dt_bias: dense(sh, &format!("{p}.dt_bias"))?.0,
            norm_w: dense(sh, &format!("{p}.norm.weight"))?.0,
            out_proj: dense_half(sh, &format!("{p}.out_proj.weight"))?.0,
            threads: threads.max(1),
            dev: None,
        };
        if dense_i8() {
            // The five projections are this tier's dense read; at one byte per
            // weight they cost half the bytes.
            l.in_qkv = l.in_qkv.quantize_i8(cd, h);
            l.in_z = l.in_z.quantize_i8(vd, h);
            l.in_b = l.in_b.quantize_i8(nv, h);
            l.in_a = l.in_a.quantize_i8(nv, h);
            l.out_proj = l.out_proj.quantize_i8(h, vd);
        }
        Ok(l)
    }

    /// Move the five projections to VRAM and drop the host's copies, and
    /// with them the conv weights, A_log, dt_bias and the gated norm plus two
    /// zeroed state buffers, so the whole mixer can run on the device.
    pub fn attach_dense(&mut self, sh: &Shards, layer: usize, gpu: Arc<Mutex<Gpu>>) -> Result<()> {
        let p = format!("{}.layers.{layer}.linear_attn", self.cfg.prefix);
        let names = [
            format!("{p}.in_proj_qkv.weight"),
            format!("{p}.in_proj_z.weight"),
            format!("{p}.in_proj_b.weight"),
            format!("{p}.in_proj_a.weight"),
            format!("{p}.out_proj.weight"),
        ];
        let mut g = gpu.lock().unwrap();
        g.reserve(DENSE_OFF + DENSE_FLOATS)?;
        let Some(b) = put_set(&mut g, sh, &names)? else {
            return Ok(());
        };
        let mixer = self.upload_mixer(&mut g)?;
        drop(g);
        self.dev = Some(GdnDev { gpu, qkv: b[0], z: b[1], b: b[2], a: b[3], out: b[4], mixer });
        self.in_qkv.clear();
        self.in_z.clear();
        self.in_b.clear();
        self.in_a.clear();
        self.out_proj.clear();
        Ok(())
    }

    /// `gdn_rec` keeps q, k and the read-out in fixed-size shared arrays and
    /// covers a head with 512 threads as `512/dv` row groups, four rows at a
    /// time. A shape that does not divide that way stays on the split path.
    fn upload_mixer(&self, g: &mut Gpu) -> Result<Option<GdnMixer>> {
        let cfg = &self.cfg;
        let (nv, nk) = (cfg.linear_num_value_heads, cfg.linear_num_key_heads);
        let (dk, dv) = (cfg.linear_key_head_dim, cfg.linear_value_head_dim);
        if dk == 0 || dv == 0 || dk > 256 || dv > 256 || nv == 0 || nk == 0 || nv % nk != 0 || 512 % dv != 0 {
            return Ok(None);
        }
        let rows = dk / (512 / dv);
        if rows == 0 || dk % (512 / dv) != 0 || rows % 4 != 0 {
            return Ok(None);
        }
        let hist = cfg.linear_conv_kernel_dim.saturating_sub(1);
        let cd = cfg.conv_dim();
        let mut params = Vec::with_capacity(2 * nv + dv);
        params.extend_from_slice(&self.a_log);
        params.extend_from_slice(&self.dt_bias);
        params.extend_from_slice(&self.norm_w);
        if self.a_log.len() != nv || self.dt_bias.len() != nv || self.norm_w.len() != dv {
            return Ok(None);
        }
        if self.conv_w.len() != cd * cfg.linear_conv_kernel_dim {
            return Ok(None);
        }
        let conv_w = g.upload(&f32_bytes(&self.conv_w))?;
        let params = g.upload(&f32_bytes(&params))?;
        let conv_state = g.upload(&f32_bytes(&vec![0f32; hist * cd]))?;
        let rec_state = g.upload(&f32_bytes(&vec![0f32; nv * dk * dv]))?;
        Ok(Some(GdnMixer { conv_w, params, conv_state, rec_state }))
    }

    /// Pull the device's conv window and recurrent state back into `st`, so a
    /// caller that inspects or forks the state sees what the mixer actually
    /// holds. Also clears `mirrored`, so the next step re-uploads.
    pub fn sync_state(&self, st: &mut GdnState) -> Result<()> {
        let Some(d) = &self.dev else { return Ok(()) };
        let Some(m) = &d.mixer else { return Ok(()) };
        if !st.mirrored {
            return Ok(());
        }
        let mut g = d.gpu.lock().unwrap();
        let mut cb = vec![0u8; st.conv.len() * 4];
        g.read_buf(m.conv_state, &mut cb)?;
        let mut rb = vec![0u8; st.rec.len() * 4];
        g.read_buf(m.rec_state, &mut rb)?;
        bytes_into_f32(&cb, &mut st.conv);
        bytes_into_f32(&rb, &mut st.rec);
        st.mirrored = false;
        Ok(())
    }

    /// The whole layer in one command buffer: the input projections, the conv
    /// and its window rotation, the delta recursion with the gated norm, and
    /// out_proj. One H-float write and one H-float read per token, where the
    /// split path crossed PCIe twice with 32 KiB and 16 KiB in between.
    fn step_fused(&self, d: &GdnDev, m: &GdnMixer, x: &[f32], st: &mut GdnState, out: &mut [f32]) {
        let cfg = &self.cfg;
        let (h, cd, kd, vd) = (cfg.hidden_size, cfg.conv_dim(), cfg.key_dim(), cfg.value_dim());
        let (nv, nk) = (cfg.linear_num_value_heads, cfg.linear_num_key_heads);
        let (dk, dv) = (cfg.linear_key_head_dim, cfg.linear_value_head_dim);
        let mut g = d.gpu.lock().unwrap();
        if !st.mirrored {
            g.write_buf(m.conv_state, 0, &f32_bytes(&st.conv)).unwrap();
            g.write_buf(m.rec_state, 0, &f32_bytes(&st.rec)).unwrap();
            st.mirrored = true;
        }
        let ib = DENSE_OFF + h;
        // CUDA's half gemv reads x as v4.f32, so the offset must be 16-byte
        // aligned; Vulkan reads scalars and would not notice an odd nv.
        let normed = (ib + cd + vd + 2 * nv + 3) & !3;
        g.write(DENSE_OFF, x);
        let read_syncs = g.read_syncs();
        {
            let mut bt = g.batch().unwrap();
            // The four input projections read the same x and write disjoint
            // slices, so they are independent. Recorded barrier-free they fill
            // the device together; one barrier then covers everything conv,
            // the recursion and out_proj read back.
            bt.gemv_half_indep(d.qkv, cd, h, DENSE_OFF, ib).unwrap();
            bt.gemv_half_indep(d.z, vd, h, DENSE_OFF, ib + cd).unwrap();
            bt.gemv_half_indep(d.b, nv, h, DENSE_OFF, ib + cd + vd).unwrap();
            bt.gemv_half_indep(d.a, nv, h, DENSE_OFF, ib + cd + vd + nv).unwrap();
            bt.barrier().unwrap();
            bt.gdn_conv(m.conv_w, m.conv_state, ib, cd, cfg.linear_conv_kernel_dim, st.hist).unwrap();
            bt.gdn_rec(m.params, m.rec_state, ib, normed, kd, nv, dk, dv, nv / nk, cfg.rms_norm_eps as f32).unwrap();
            bt.gemv_half(d.out, h, vd, normed, DENSE_OFF).unwrap();
            // The read below is a blocking DtoH on CUDA, so waiting here too
            // pays a second WDDM flush per layer for nothing. Vulkan's read is
            // a mapped copy and still needs the wait.
            if read_syncs { bt.submit().unwrap(); } else { bt.run().unwrap(); }
        }
        g.read_into(DENSE_OFF, out);
        st.filled = (st.filled + 1).min(st.hist);
    }

    /// The four input projections in one command buffer and one read. They all
    /// consume the same x, so crossing PCIe four times would buy nothing.
    fn dev_inputs(&self, d: &GdnDev, x: &[f32], buf: &mut [f32]) {
        let cfg = &self.cfg;
        let (h, cd, vd, nv) = (
            cfg.hidden_size,
            cfg.conv_dim(),
            cfg.value_dim(),
            cfg.linear_num_value_heads,
        );
        let mut g = d.gpu.lock().unwrap();
        g.write(DENSE_OFF, x);
        {
            let mut bt = g.batch().unwrap();
            bt.gemv_half_indep(d.qkv, cd, h, DENSE_OFF, DENSE_OFF + h).unwrap();
            bt.gemv_half_indep(d.z, vd, h, DENSE_OFF, DENSE_OFF + h + cd).unwrap();
            bt.gemv_half_indep(d.b, nv, h, DENSE_OFF, DENSE_OFF + h + cd + vd).unwrap();
            bt.gemv_half_indep(d.a, nv, h, DENSE_OFF, DENSE_OFF + h + cd + vd + nv).unwrap();
            bt.barrier().unwrap();
            bt.run().unwrap();
        }
        g.read_into(DENSE_OFF + h, buf);
    }

    /// One decode token. `x` is the post-conv input row of length H.
    pub fn step(&self, x: &[f32], st: &mut GdnState, out: &mut [f32]) {
        if let Some(d) = &self.dev {
            if let Some(m) = &d.mixer {
                let _t = prof::start(prof::MIXER_DEV);
                self.step_fused(d, m, x, st, out);
                return;
            }
        }
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let cd = cfg.conv_dim();
        let kd = cfg.key_dim();
        let vd = cfg.value_dim();
        let nv = cfg.linear_num_value_heads;
        let nk = cfg.linear_num_key_heads;
        let dk = cfg.linear_key_head_dim;
        let dv = cfg.linear_value_head_dim;
        let kernel_dim = cfg.linear_conv_kernel_dim;
        let t = self.threads;

        // one buffer, one layout, whichever tier fills it: qkv | z | b | a
        let mut ibuf = vec![0f32; cd + vd + 2 * nv];
        match &self.dev {
            Some(d) => {
                let _t = prof::start(prof::MIXER_DEV);
                self.dev_inputs(d, x, &mut ibuf);
            }
            None => {
                // The four input projections are the mixer's own 25 MiB of dense
                // weights per layer and they used to run between two timers, so
                // their cost showed up only as "unaccounted" stack time.
                let _p = prof::start(prof::MIXER_PROJ);
                let (m, r) = ibuf.split_at_mut(cd);
                let (zz, r) = r.split_at_mut(vd);
                let (bb, aa) = r.split_at_mut(nv);
                self.in_qkv.gemv(cd, h, x, m, t);
                self.in_z.gemv(vd, h, x, zz, t);
                self.in_b.gemv(nv, h, x, bb, t);
                self.in_a.gemv(nv, h, x, aa, t);
            }
        }
        let _cpu = prof::start(prof::MIXER_CPU);
        let (mixed, r) = ibuf.split_at(cd);
        let (z, r) = r.split_at(vd);
        let (b, a) = r.split_at(nv);

        // depthwise causal conv with the rolling window, then SiLU
        let mut conv_out = vec![0f32; cd];
        {
            let rows = [mixed];
            kernel::conv_depthwise(&rows, &self.conv_w, cd, kernel_dim, &st.conv, &mut conv_out);
            // rotate the window: [hist][channels], oldest first
            for i in 0..st.hist.saturating_sub(1) {
                st.conv.copy_within((i + 1) * cd..(i + 2) * cd, i * cd);
            }
            if st.hist > 0 {
                st.conv[(st.hist - 1) * cd..st.hist * cd].copy_from_slice(mixed);
            }
        }
        for v in conv_out.iter_mut() {
            *v = kernel::silu(*v);
        }
        let (q_all, rest) = conv_out.split_at(kd);
        let (k_all, v_all) = rest.split_at(kd);

        let rep = nv / nk;
        let mut o = vec![0f32; vd];
        // Each head owns a disjoint [dk, dv] block of the recurrent state and a
        // disjoint run of `o`, so the heads are independent. They are also the
        // bulk of the layer: 32 heads x 128 x 128 over three passes is millions
        // of flops, and serially that measured as milliseconds per token.
        let recp = kernel::SharedPtr::new(st.rec.as_mut_ptr());
        let op = kernel::SharedPtr::new(o.as_mut_ptr());
        kernel::par_count(nv, t, |h0, hn| {
            // One buffer set per worker, not per head: 32 heads x 30 layers was
            // ~3800 small allocations a token.
            let mut qs = vec![0f32; dk];
            let mut ks = vec![0f32; dk];
            let mut kv_mem = vec![0f32; dv];
            let mut delta = vec![0f32; dv];
            for hv in h0..h0 + hn {
                let hk = hv / rep;
                let q = &q_all[hk * dk..(hk + 1) * dk];
                let k = &k_all[hk * dk..(hk + 1) * dk];
                let v = &v_all[hv * dv..(hv + 1) * dv];
                let beta = kernel::sigmoid(b[hv]);
                let g = -(self.a_log[hv].exp()) * kernel::softplus(a[hv] + self.dt_bias[hv]);
                let decay = g.exp();
                qs.copy_from_slice(q);
                ks.copy_from_slice(k);
                kernel::l2norm(&mut qs, 1e-6);
                kernel::l2norm(&mut ks, 1e-6);
                let scale = 1.0 / (dk as f32).sqrt();
                for x in qs.iter_mut() {
                    *x *= scale;
                }
                let s = unsafe { std::slice::from_raw_parts_mut(recp.at(hv * dk * dv), dk * dv) };
                let out = unsafe { std::slice::from_raw_parts_mut(op.at(hv * dv), dv) };
                // S = exp(g) * (I - beta k k^T / ...) style delta update, in place
                // kv_mem[j] = sum_i S[i][j] * k[i]
                kv_mem.fill(0.0);
                for i in 0..dk {
                    let d = s[i * dv..(i + 1) * dv]
                        .iter_mut()
                        .map(|x| {
                            *x *= decay;
                            *x
                        });
                    for (j, sv) in d.enumerate() {
                        kv_mem[j] += sv * ks[i];
                    }
                }
                // delta depends only on j, so it is hoisted out of the i loop
                // rather than recomputed dk times per element.
                for j in 0..dv {
                    delta[j] = (v[j] - kv_mem[j]) * beta;
                    out[j] = 0.0;
                }
                // The update and the read-out share one row-major pass. Split
                // apart, the read-out walked the state with a dv-float stride
                // and re-read all of it; fusing keeps the i-ascending
                // accumulation order, so the result is bit-identical.
                for i in 0..dk {
                    let row = &mut s[i * dv..(i + 1) * dv];
                    let (ki, qi) = (ks[i], qs[i]);
                    for j in 0..dv {
                        row[j] += ki * delta[j];
                        out[j] += row[j] * qi;
                    }
                }
            }
        });
        // gated RMSNorm per value head, then out_proj
        let mut normed = vec![0f32; vd];
        for hv in 0..nv {
            let (x, g) = (
                &o[hv * dv..(hv + 1) * dv],
                &z[hv * dv..(hv + 1) * dv],
            );
            kernel::rmsnorm_gated(x, g, &self.norm_w, cfg.rms_norm_eps as f32, &mut normed[hv * dv..(hv + 1) * dv]);
        }
        // out_proj is a projection, not part of the recursion: close the mixer
        // timer here so the two stages stay disjoint (a nested timer would be
        // counted twice, which showed up as negative unaccounted stack time).
        drop(_cpu);
        match &self.dev {
            Some(d) => {
                let mut g = d.gpu.lock().unwrap();
                g.write(DENSE_OFF, &normed);
                {
                    let mut bt = g.batch().unwrap();
                    bt.gemv_half(d.out, h, vd, DENSE_OFF, DENSE_OFF + vd).unwrap();
                    bt.run().unwrap();
                }
                g.read_into(DENSE_OFF + vd, out);
            }
            None => {
                let _p = prof::start(prof::MIXER_PROJ);
                self.out_proj.gemv(h, vd, &normed, out, t)
            }
        }
        st.filled = (st.filled + 1).min(st.hist);
    }
}

pub struct AttnState {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub len: usize,
    pub kv: usize,
    pub hd: usize,
}

impl AttnState {
    pub fn new(cfg: &TextConfig, cap: usize) -> AttnState {
        let kv = cfg.num_key_value_heads * cfg.head_dim;
        AttnState {
            k: vec![0f32; cap * kv],
            v: vec![0f32; cap * kv],
            len: 0,
            kv,
            hd: cfg.head_dim,
        }
    }
}

pub struct AttnLayer {
    pub cfg: Arc<TextConfig>,
    pub q: Dense,
    pub k: Dense,
    pub v: Dense,
    pub o: Dense,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub threads: usize,
    pub dev: Option<AttnDev>,
}

impl AttnLayer {
    pub fn load(sh: &Shards, cfg: Arc<TextConfig>, layer: usize, max_pos: usize, threads: usize) -> Result<AttnLayer> {
        let p = format!("{}.layers.{layer}.self_attn", cfg.prefix);
        let mut cos = Vec::new();
        let mut sin = Vec::new();
        kernel::rope_tables(cfg.rope_theta, cfg.rotary_dim(), max_pos, &mut cos, &mut sin);
        let (h, nh, nkv, hd) = (
            cfg.hidden_size,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
        );
        let q_out = if cfg.attn_output_gate { nh * hd * 2 } else { nh * hd };
        let zc = cfg.norm_zero_centered;
        let mut l = AttnLayer {
            cfg,
            q: dense_half(sh, &format!("{p}.q_proj.weight"))?.0,
            k: dense_half(sh, &format!("{p}.k_proj.weight"))?.0,
            v: dense_half(sh, &format!("{p}.v_proj.weight"))?.0,
            o: dense_half(sh, &format!("{p}.o_proj.weight"))?.0,
            q_norm: norm_w(sh, &format!("{p}.q_norm.weight"), zc)?,
            k_norm: norm_w(sh, &format!("{p}.k_norm.weight"), zc)?,
            cos,
            sin,
            threads: threads.max(1),
            dev: None,
        };
        if dense_i8() {
            l.q = l.q.quantize_i8(q_out, h);
            l.k = l.k.quantize_i8(nkv * hd, h);
            l.v = l.v.quantize_i8(nkv * hd, h);
            l.o = l.o.quantize_i8(h, nh * hd);
        }
        Ok(l)
    }

    /// q/k/v/o to VRAM. The per-head norms, rope, the KV append and the softmax
    /// stay on the CPU: they read the cache, which is host memory that grows
    /// with the sequence and has no business being re-uploaded every token.
    pub fn attach_dense(&mut self, sh: &Shards, layer: usize, gpu: Arc<Mutex<Gpu>>) -> Result<()> {
        let p = format!("{}.layers.{layer}.self_attn", self.cfg.prefix);
        let names = [
            format!("{p}.q_proj.weight"),
            format!("{p}.k_proj.weight"),
            format!("{p}.v_proj.weight"),
            format!("{p}.o_proj.weight"),
        ];
        let mut g = gpu.lock().unwrap();
        g.reserve(DENSE_OFF + DENSE_FLOATS)?;
        if dense_int4() {
            let mut ws = [0usize; 4];
            let mut ss = [0usize; 4];
            for (i, n) in names.iter().enumerate() {
                let f32w = sh.tensor_f32(n)?;
                let info = sh.info(n).ok_or_else(|| format!("missing tensor {n}"))?;
                let cols = *info.shape.last().ok_or_else(|| format!("{n}: no last dim"))?;
                let rows = info.elems() / cols;
                let q = Q4::quantize(&f32w, rows, cols, 64, 8);
                let (w, sc) = g.upload_q4(&q)?;
                ws[i] = w;
                ss[i] = sc;
            }
            drop(g);
            self.dev = Some(AttnDev { gpu, q: ws[0], k: ws[1], v: ws[2], o: ws[3], q4: true, qs: ss, group: 64 });
        } else {
            let Some(b) = put_set(&mut g, sh, &names)? else {
                return Ok(());
            };
            drop(g);
            self.dev = Some(AttnDev { gpu, q: b[0], k: b[1], v: b[2], o: b[3], q4: false, qs: [0; 4], group: 0 });
        }
        self.q.clear();
        self.k.clear();
        self.v.clear();
        self.o.clear();
        Ok(())
    }

    /// q, k and v in one batch and one contiguous read, same as the delta net.
    fn dev_inputs(&self, d: &AttnDev, x: &[f32], buf: &mut [f32]) {
        let cfg = &self.cfg;
        let (h, hd, nkv) = (cfg.hidden_size, cfg.head_dim, cfg.num_key_value_heads);
        let q_out = if cfg.attn_output_gate { cfg.num_attention_heads * hd * 2 } else { cfg.num_attention_heads * hd };
        let mut g = d.gpu.lock().unwrap();
        g.write(DENSE_OFF, x);
        {
            let mut bt = g.batch().unwrap();
            if d.q4 {
                bt.gemv_q4_indep(d.q, d.qs[0], q_out, h, d.group, DENSE_OFF, DENSE_OFF + h).unwrap();
                bt.gemv_q4_indep(d.k, d.qs[1], nkv * hd, h, d.group, DENSE_OFF, DENSE_OFF + h + q_out).unwrap();
                bt.gemv_q4_indep(d.v, d.qs[2], nkv * hd, h, d.group, DENSE_OFF, DENSE_OFF + h + q_out + nkv * hd).unwrap();
            } else {
                bt.gemv_half_indep(d.q, q_out, h, DENSE_OFF, DENSE_OFF + h).unwrap();
                bt.gemv_half_indep(d.k, nkv * hd, h, DENSE_OFF, DENSE_OFF + h + q_out).unwrap();
                bt.gemv_half_indep(d.v, nkv * hd, h, DENSE_OFF, DENSE_OFF + h + q_out + nkv * hd).unwrap();
            }
            // q, k and v are independent projections of the same x: one
            // barrier before the host reads them back, not three.
            bt.barrier().unwrap();
            bt.run().unwrap();
        }
        g.read_into(DENSE_OFF + h, buf);
    }

    /// One decode token against the KV cache in `st`.
    pub fn step(&self, x: &[f32], pos: usize, st: &mut AttnState, out: &mut [f32]) {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let rot = cfg.rotary_dim();
        let t = self.threads;
        let eps = cfg.rms_norm_eps as f32;
        let q_out = if cfg.attn_output_gate { nh * hd * 2 } else { nh * hd };

        // q_proj is [nh, head_dim*2] per head when the checkpoint gates the
        // output: query first, then the gate. A dense Qwen3 has no gate, so the
        // heads are packed one head_dim apart and reading them at 2*hd would
        // stride every other head into the following row.
        let stride = if cfg.attn_output_gate { hd * 2 } else { hd };
        let mut ibuf = vec![0f32; q_out + 2 * nkv * hd];
        match &self.dev {
            Some(d) => {
                let _t = prof::start(prof::MIXER_DEV);
                self.dev_inputs(d, x, &mut ibuf);
            }
            None => {
                let _p = prof::start(prof::MIXER_PROJ);
                let (q, r) = ibuf.split_at_mut(q_out);
                let (k, v) = r.split_at_mut(nkv * hd);
                self.q.gemv(q_out, h, x, q, t);
                self.k.gemv(nkv * hd, h, x, k, t);
                self.v.gemv(nkv * hd, h, x, v, t);
            }
        }
        let _cpu = prof::start(prof::MIXER_CPU);
        let (qraw, r) = ibuf.split_at(q_out);
        let (kraw, vv) = r.split_at(nkv * hd);

        let (c, s) = (
            &self.cos[pos * rot..(pos + 1) * rot],
            &self.sin[pos * rot..(pos + 1) * rot],
        );
        let mut qq = vec![0f32; nh * hd];
        for i in 0..nh {
            let base = i * stride;
            kernel::rmsnorm(&qraw[base..base + hd], &self.q_norm, eps, &mut qq[i * hd..(i + 1) * hd]);
            kernel::rope_apply(&mut qq[i * hd..(i + 1) * hd], c, s, rot);
        }
        let mut kk = vec![0f32; nkv * hd];
        for i in 0..nkv {
            let base = i * hd;
            kernel::rmsnorm(&kraw[base..base + hd], &self.k_norm, eps, &mut kk[base..base + hd]);
            kernel::rope_apply(&mut kk[base..base + hd], c, s, rot);
        }
        let row = st.len * st.kv;
        st.k[row..row + st.kv].copy_from_slice(&kk);
        st.v[row..row + st.kv].copy_from_slice(vv);
        st.len += 1;

        let mut attn = vec![0f32; nh * hd];
        let rep = nh / nkv;
        let scale = 1.0f32 / (hd as f32).sqrt();
        // One scores buffer reused across heads; st.len is fixed for this step.
        let mut scores = Vec::with_capacity(st.len);
        for i in 0..nh {
            let q = &qq[i * hd..(i + 1) * hd];
            let hkv = i / rep;
            let mut best = f32::NEG_INFINITY;
            scores.clear();
            for j in 0..st.len {
                let kr = j * st.kv + hkv * hd;
                scores.push(kernel::dot(q, &st.k[kr..kr + hd]) * scale);
                best = best.max(*scores.last().unwrap());
            }
            let mut sum = 0f32;
            for s in scores.iter_mut() {
                *s = (*s - best).exp();
                sum += *s;
            }
            let o = &mut attn[i * hd..(i + 1) * hd];
            for v in o.iter_mut() {
                *v = 0.0;
            }
            for (j, p) in scores.iter().enumerate() {
                let w = p / sum;
                let vr = j * st.kv + hkv * hd;
                kernel::axpy(o, &st.v[vr..vr + hd], w);
            }
        }
        if cfg.attn_output_gate {
            for i in 0..nh {
                let gbase = i * stride + hd;
                for j in 0..hd {
                    attn[i * hd + j] *= kernel::sigmoid(qraw[gbase + j]);
                }
            }
        }
        drop(_cpu);
        match &self.dev {
            Some(d) => {
                let mut g = d.gpu.lock().unwrap();
                g.write(DENSE_OFF, &attn);
                {
                    let mut bt = g.batch().unwrap();
                    if d.q4 {
                        bt.gemv_q4(d.o, d.qs[3], h, nh * hd, d.group, DENSE_OFF, DENSE_OFF + nh * hd).unwrap();
                    } else {
                        bt.gemv_half(d.o, h, nh * hd, DENSE_OFF, DENSE_OFF + nh * hd).unwrap();
                    }
                    bt.run().unwrap();
                }
                g.read_into(DENSE_OFF + nh * hd, out);
            }
            None => {
                let _p = prof::start(prof::MIXER_PROJ);
                self.o.gemv(h, nh * hd, &attn, out, t)
            }
        }
    }
}

pub struct Layer {
    pub cfg: Arc<TextConfig>,
    pub kind: LayerKind,
    pub norm_in: Vec<f32>,
    pub norm_post: Vec<f32>,
    /// Routed-expert FFN: `Some` only when the checkpoint has experts.
    pub moe: Option<MoeLayer>,
    /// Dense SiLU FFN: `Some` exactly when `moe` is `None`.
    pub mlp: Option<MlpLayer>,
    pub gdn: Option<GdnLayer>,
    pub attn: Option<AttnLayer>,
}

pub enum LayerState {
    Linear(GdnState),
    Full(AttnState),
}

impl Layer {
    /// Every expert of the layer resident. `with_experts` is the general case.
    pub fn load(
        sh: &Arc<Shards>,
        cfg: Arc<TextConfig>,
        layer: usize,
        group: usize,
        threads: usize,
        kv_cap: usize,
    ) -> Result<Layer> {
        let all: Vec<usize> = (0..cfg.num_experts).collect();
        Layer::with_experts(sh, cfg, layer, &all, None, group, threads, kv_cap)
    }

    /// `ids` resident in slot order, the rest answered from `stream`. A plan
    /// that lists its experts hottest-first makes slot 0 the cheapest miss.
    pub fn with_experts(
        sh: &Arc<Shards>,
        cfg: Arc<TextConfig>,
        layer: usize,
        ids: &[usize],
        stream: Option<Stream>,
        group: usize,
        threads: usize,
        kv_cap: usize,
    ) -> Result<Layer> {
        let kind = cfg.layer_types[layer];
        let p = format!("{}.layers.{layer}", cfg.prefix);
        let zc = cfg.norm_zero_centered;
        let (moe, mlp) = if cfg.is_moe() {
            (
                Some(MoeLayer::load_set(sh, cfg.clone(), layer, ids, stream, group, threads)?),
                None,
            )
        } else {
            (None, Some(MlpLayer::load(sh, cfg.clone(), layer, threads)?))
        };
        let gdn = match kind {
            LayerKind::Linear => Some(GdnLayer::load(sh, cfg.clone(), layer, threads)?),
            LayerKind::Full => None,
        };
        let attn = match kind {
            LayerKind::Full => Some(AttnLayer::load(sh, cfg.clone(), layer, kv_cap, threads)?),
            LayerKind::Linear => None,
        };
        Ok(Layer {
            cfg,
            kind,
            norm_in: norm_w(sh, &format!("{p}.input_layernorm.weight"), zc)?,
            norm_post: norm_w(sh, &format!("{p}.post_attention_layernorm.weight"), zc)?,
            moe,
            mlp,
            gdn,
            attn,
        })
    }

    pub fn state(&self, cfg: &TextConfig, kv_cap: usize) -> LayerState {
        match self.kind {
            LayerKind::Linear => LayerState::Linear(GdnState::new(cfg)),
            LayerKind::Full => LayerState::Full(AttnState::new(cfg, kv_cap)),
        }
    }

    /// Park this layer's dense projections in VRAM. Which block that is depends
    /// on the layer kind, and a checkpoint that is not bf16 keeps the CPU copy.
    pub fn attach_dense(&mut self, sh: &Shards, layer: usize, gpu: Arc<Mutex<Gpu>>) -> Result<()> {
        match self.kind {
            LayerKind::Linear => self.gdn.as_mut().unwrap().attach_dense(sh, layer, gpu),
            LayerKind::Full => self.attn.as_mut().unwrap().attach_dense(sh, layer, gpu),
        }
    }

    pub fn forward(&self, x: &mut [f32], st: &mut LayerState, pos: usize) {
        let h = x.len();
        let eps = self.cfg.rms_norm_eps as f32;
        let mut buf = vec![0f32; h];
        let mut res = vec![0f32; h];
        {
            let _t = prof::start(prof::NORM);
            kernel::rmsnorm(x, &self.norm_in, eps, &mut buf);
        }
        match (&self.kind, st) {
            (LayerKind::Linear, LayerState::Linear(g)) => self.gdn.as_ref().unwrap().step(&buf, g, &mut res),
            (LayerKind::Full, LayerState::Full(a)) => self.attn.as_ref().unwrap().step(&buf, pos, a, &mut res),
            _ => panic!("layer state mismatch"),
        }
        {
            let _t = prof::start(prof::RESIDUAL);
            kernel::axpy(x, &res, 1.0);
        }
        {
            let _t = prof::start(prof::NORM);
            kernel::rmsnorm(x, &self.norm_post, eps, &mut buf);
        }
        let _t = prof::start(prof::RESIDUAL);
        match (&self.moe, &self.mlp) {
            (Some(moe), _) => {
                let (moe_out, _) = moe.forward(&buf);
                for i in 0..h {
                    x[i] += moe_out[i];
                }
            }
            (None, mlp) => {
                mlp.as_ref().unwrap().forward(&buf, &mut res);
                for i in 0..h {
                    x[i] += res[i];
                }
            }
        }
    }
}

/// The text stack: embedding gather, a run of decoder layers, the trailing norm
/// and the lm_head. `layers` holds `count` layers starting at `first`, which is
/// how a small run of the graph can be diffed against transformers without
/// holding all 20 GiB of FQ4 experts in RAM.
pub struct Model {
    pub cfg: Arc<TextConfig>,
    pub first: usize,
    pub layers: Vec<Layer>,
    pub norm: Vec<f32>,
    pub embed: String,
    pub head: String,
    pub threads: usize,
    /// The compute device, shared with every layer's `DevExperts`. None means
    /// the whole stack runs on the CPU and the plan's device tier merges into
    /// the host tier.
    pub gpu: Option<Arc<Mutex<Gpu>>>,
    pub dev_head: Option<DevHead>,
    /// The lm_head parked in an ONNX Runtime session when `--backend ort` is
    /// selected. `None` keeps the device or blocked-CPU head path.
    #[cfg(feature = "ort")]
    pub ort_head: Option<crate::ort::OrtDense>,
}

impl Model {
    /// Every expert of every loaded layer resident: `count` * 256 * 19.2 MiB of
    /// FQ4, which is why `load_plan` exists.
    pub fn load(
        sh: &Arc<Shards>,
        cfg: Arc<TextConfig>,
        first: usize,
        count: usize,
        group: usize,
        threads: usize,
        kv_cap: usize,
    ) -> Result<Model> {
        let mut layers = Vec::with_capacity(count);
        for l in first..first + count {
            layers.push(Layer::load(sh, cfg.clone(), l, group, threads, kv_cap)?);
        }
        Model::finish(sh, cfg, first, layers, threads)
    }

    /// The stack exactly as `sched::Plan` laid it out: the experts the plan
    /// puts in RAM (device tier then host tier, hottest first) are the only
    /// ones quantised at load, and a pick from the disk tier costs a shard
    /// read, a quantisation pass and a ring slot.
    ///
    /// With a `gpu`, the plan's device tier goes to VRAM and its host tier stays
    /// in system RAM, so one routed pick is answered by exactly one of the two
    /// and they can work the same token at once. Without one, both tiers are
    /// system RAM and merge into the resident set, which is the CPU-only plan.
    pub fn load_plan(
        sh: &Arc<Shards>,
        cfg: Arc<TextConfig>,
        plan: &Plan,
        group: usize,
        threads: usize,
        kv_cap: usize,
        stream_slots: usize,
        gpu: Option<Arc<Mutex<Gpu>>>,
    ) -> Result<Model> {
        let mut layers = Vec::with_capacity(plan.layers.len());
        let t_host = std::time::Instant::now();
        for lp in &plan.layers {
            let ids: Vec<usize> = if gpu.is_some() {
                lp.pooled.clone()
            } else {
                lp.resident.iter().chain(&lp.pooled).copied().collect()
            };
            let stream = if lp.streamed.is_empty() || stream_slots == 0 {
                None
            } else {
                Some(Stream::new(sh.clone(), &cfg, lp.layer, group, threads, stream_slots))
            };
            layers.push(Layer::with_experts(
                sh,
                cfg.clone(),
                lp.layer,
                &ids,
                stream,
                group,
                threads,
                kv_cap,
            )?);
        }
        let host_ms = t_host.elapsed().as_secs_f64() * 1e3;
        crate::mem::report("host pool");
        let first = plan.layers.first().map(|lp| lp.layer).unwrap_or(0);
        let t_finish = std::time::Instant::now();
        let mut model = Model::finish(sh, cfg, first, layers, threads)?;
        let finish_ms = t_finish.elapsed().as_secs_f64() * 1e3;
        crate::mem::report("dense+embed");
        let (mut head_ms, mut dev_ms) = (0.0, 0.0);
        if let Some(g) = gpu {
            // The head goes up first, before the expert pool. WDDM places an
            // allocation that exceeds the process budget in system memory, and
            // the loser is whichever buffer was allocated last. The head is the
            // worst possible loser: 0.95 GiB read once per token, against the
            // experts' ~224 MiB of picks. Uploaded last on a near-full 8 GiB
            // card it was demoted and cost 45.9 ms/token instead of 5.1 ms.
            model.gpu = Some(g.clone());
            let t = std::time::Instant::now();
            model.attach_head(sh)?;
            head_ms = t.elapsed().as_secs_f64() * 1e3;
            crate::mem::report("lm_head");
            let t = std::time::Instant::now();
            for (l, lp) in model.layers.iter_mut().zip(&plan.layers) {
                if let Some(m) = &mut l.moe {
                    m.attach_gpu(sh, lp.layer, g.clone(), &lp.resident, group)?;
                }
                if lp.dense_on_device {
                    l.attach_dense(sh, lp.layer, g.clone())?;
                }
            }
            dev_ms = t.elapsed().as_secs_f64() * 1e3;
            crate::mem::report("device upload");
        }
        let (used, held) = model.layers.iter().fold((0usize, 0usize), |(u, h), l| {
            match &l.moe {
                Some(m) => (u + m.host_bytes(), h + m.host_bytes_held()),
                None => (u, h),
            }
        });
        let mapped = model
            .layers
            .iter()
            .filter(|l| l.moe.as_ref().is_some_and(|m| m.host_is_mapped()))
            .count();
        eprintln!(
            "[fennec] load phases: host pool {:.1} s | dense+embed {:.1} s | lm_head {:.1} s | device upload {:.1} s",
            host_ms / 1e3, finish_ms / 1e3, head_ms / 1e3, dev_ms / 1e3
        );
        if model.cfg.is_moe() {
            eprintln!(
                "[fennec] host experts: {:.2} GiB of matrix, {:.2} GiB of allocator capacity, {mapped}/{} layers borrowed from the mapping",
                used as f64 / 2f64.powi(30),
                held as f64 / 2f64.powi(30),
                model.layers.len()
            );
        }
        Ok(model)
    }

    /// Upload the lm_head and reserve the arena region its logits land in. A
    /// head wider than two bytes keeps the blocked CPU path: the device shaders
    /// read pairs of bytes, and converting an f32 head down to a half to get it
    /// onto the device would throw away mantissa bits the CPU path does not.
    pub fn attach_head(&mut self, sh: &Shards) -> Result<()> {
        let g = match &self.gpu {
            Some(g) => g.clone(),
            None => return Ok(()),
        };
        let cols = self.cfg.hidden_size;
        let i = sh
            .info(&self.head)
            .ok_or_else(|| format!("missing tensor {}", self.head))?;
        let f16 = match i.dtype {
            DType::BF16 => false,
            DType::F16 => true,
            _ => return Ok(()),
        };
        if i.elems() % cols != 0 {
            return Ok(());
        }
        let rows = i.elems() / cols;
        let bytes = &sh.data(i.file)[i.off..i.off + i.elems() * 2];
        let (_, base, stride) = moe_arena(&self.cfg);
        let off = base + self.cfg.num_experts_per_tok * stride;
        let t0 = std::time::Instant::now();
        let mut gpu = g.lock().unwrap();
        gpu.reserve(off + rows)?;
        let buf = gpu.upload_half(bytes, f16)?;
        let held = gpu.bytes_held();
        let name = gpu.name().to_string();
        drop(gpu);
        eprintln!(
            "[fennec] lm_head {rows}x{cols} {} on {name} in {:.1} s, {:.2} GiB device total",
            if f16 { "f16" } else { "bf16" },
            t0.elapsed().as_secs_f64(),
            held as f64 / 2f64.powi(30)
        );
        self.dev_head = Some(DevHead { buf, off, rows });
        Ok(())
    }

    fn finish(
        sh: &Shards,
        cfg: Arc<TextConfig>,
        first: usize,
        layers: Vec<Layer>,
        threads: usize,
    ) -> Result<Model> {
        // A tied checkpoint may still ship a copy of the head; when it does not,
        // the embedding table *is* the head and the same bf16 rows read through
        // the same gemv, so the only difference is the tensor name.
        let head = match (cfg.tie_word_embeddings, sh.contains("lm_head.weight")) {
            (true, false) => format!("{}.embed_tokens.weight", cfg.prefix),
            _ => "lm_head.weight".to_string(),
        };
        Ok(Model {
            embed: format!("{}.embed_tokens.weight", cfg.prefix),
            head,
            norm: norm_w(sh, &format!("{}.norm.weight", cfg.prefix), cfg.norm_zero_centered)?,
            cfg,
            first,
            layers,
            threads: threads.max(1),
            gpu: None,
            dev_head: None,
            #[cfg(feature = "ort")]
            ort_head: None,
        })
    }

    pub fn states(&self, kv_cap: usize) -> Vec<LayerState> {
        self.layers.iter().map(|l| l.state(&self.cfg, kv_cap)).collect()
    }

    pub fn embed_row(&self, sh: &Shards, id: usize) -> Result<Vec<f32>> {
        sh.row_f32(&self.embed, id)
    }

    /// One token through the loaded layers, in place on `x`.
    pub fn forward(&self, x: &mut [f32], st: &mut [LayerState], pos: usize) {
        for (l, s) in self.layers.iter().zip(st.iter_mut()) {
            l.forward(x, s, pos);
        }
    }

    /// logits = lm_head @ x. The head is never materialised: for a two-byte
    /// format the dot reads the mmap directly (`gemv_bf16` / `gemv_f16`),
    /// because an f32 copy of a 1 GiB tensor pays 2 GiB of writes and 2 GiB of
    /// reads for work that is 1 GiB of bytes wide. Anything else keeps the
    /// blocked conversion, one row block at a time so the read still streams.
    pub fn logits(&self, sh: &Shards, x: &[f32], out: &mut [f32]) -> Result<()> {
        let cols = self.cfg.hidden_size;
        #[cfg(feature = "ort")]
        if let Some(d) = &self.ort_head {
            if out.len() == d.rows() {
                d.gemv(x, out)?;
                return Ok(());
            }
        }
        if let (Some(g), Some(dh)) = (&self.gpu, &self.dev_head) {
            if out.len() == dh.rows {
                let mut gpu = g.lock().unwrap();
                gpu.write(0, x);
                {
                    let mut b = gpu.batch()?;
                    b.gemv_half(dh.buf, dh.rows, cols, 0, dh.off)?;
                    b.run()?;
                }
                gpu.read_into(dh.off, out);
                return Ok(());
            }
        }
        let i = sh
            .info(&self.head)
            .ok_or_else(|| format!("missing tensor {}", self.head))?;
        let (file, off, dtype, elems) = (i.file, i.off, i.dtype, i.elems());
        if elems == out.len() * cols {
            let bytes = &sh.data(file)[off..off + elems * 2];
            match dtype {
                DType::BF16 => {
                    kernel::gemv_bf16(bytes, out.len(), cols, x, out, self.threads);
                    return Ok(());
                }
                DType::F16 => {
                    kernel::gemv_f16(bytes, out.len(), cols, x, out, self.threads);
                    return Ok(());
                }
                _ => {}
            }
        }
        let block = (1024).min(out.len());        let mut buf = vec![0f32; block * cols];
        for r0 in (0..out.len()).step_by(block) {
            let n = block.min(out.len() - r0);
            let raw = sh.slice_f32(&self.head, r0 * cols, (r0 + n) * cols)?;
            buf[..n * cols].copy_from_slice(&raw);
            kernel::gemv(&buf[..n * cols], n, cols, x, &mut out[r0..r0 + n], self.threads);
        }
        Ok(())
    }
}

#[cfg(feature = "ort")]
impl Model {
    /// Move every dense projection and the lm_head onto ONNX Runtime. Called
    /// after `load_plan` with no `Gpu` when `--backend ort` is selected: the
    /// routed experts stay on the host tier, only the dense F16 matmuls move to
    /// the device's provider. An f32 head keeps the blocked CPU path, exactly as
    /// `attach_head` declines a head wider than two bytes.
    pub fn enable_ort(&mut self, sh: &Shards, ort: &crate::ort::Ort) -> Result<()> {
        for l in &mut self.layers {
            l.dense_to_ort(ort)?;
        }
        let cols = self.cfg.hidden_size;
        if let Some(i) = sh.info(&self.head) {
            let f16 = match i.dtype {
                DType::BF16 => Some(false),
                DType::F16 => Some(true),
                _ => None,
            };
            if let Some(f16) = f16 {
                if i.elems() % cols == 0 {
                    let rows = i.elems() / cols;
                    let bytes = &sh.data(i.file)[i.off..i.off + i.elems() * 2];
                    self.ort_head = Some(ort.dense(bytes, rows, cols, f16)?);
                }
            }
        }
        eprintln!(
            "[fennec] dense projections{} on ONNX Runtime {} EP",
            if self.ort_head.is_some() { " + lm_head" } else { "" },
            ort.ep.name()
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors::testutil::shard;
    use std::path::PathBuf;

    /// The qwen36 container's own packing, per `convert_qwen36.py::pack_int4`:
    /// one f32 scale per `group` columns, low nibble = even column, each nibble
    /// a *signed* 4-bit value in [-8,7]. Not FQ4's +8 bias — that difference is
    /// the whole point of the adoption test below.
    fn pack(w: &[f32], rows: usize, cols: usize, group: usize) -> (Vec<u8>, Vec<f32>) {
        let mut data = vec![0u8; rows * cols / 2];
        let mut scale = Vec::new();
        for r in 0..rows {
            for g in 0..cols / group {
                let base = r * cols + g * group;
                let amax = w[base..base + group].iter().fold(0f32, |a, v| a.max(v.abs()));
                let s = (amax / 7.0).max(1e-8);
                scale.push(s);
                for c in 0..group {
                    let col = g * group + c;
                    let q = ((w[base + c] / s).round() as i32).clamp(-8, 7);
                    let n = (q & 0xf) as u8;
                    let pos = r * (cols / 2) + col / 2;
                    if col % 2 == 0 {
                        data[pos] = (data[pos] & 0xf0) | n;
                    } else {
                        data[pos] = (data[pos] & 0x0f) | (n << 4);
                    }
                }
            }
        }
        (data, scale)
    }

    /// y = W x straight off the container's signed nibbles, the way
    /// `unpack_int4_to_int8` in c/qwen36.c reads them. An independent decode, so
    /// a convention mix-up cannot pass by being consistent with itself.
    fn signed_dot(
        data: &[u8],
        scale: &[f32],
        cols: usize,
        group: usize,
        r: usize,
        x: &[f32],
    ) -> f32 {
        let ng = cols / group;
        let mut s = 0f32;
        for c in 0..cols {
            let byte = data[r * (cols / 2) + c / 2];
            let n = if c % 2 == 0 { byte & 0xf } else { byte >> 4 };
            let v = if n < 8 { n as i32 } else { n as i32 - 16 };
            s += v as f32 * scale[r * ng + c / group] * x[c];
        }
        s
    }

    fn weights(rows: usize, cols: usize, seed: usize) -> Vec<f32> {
        (0..rows * cols)
            .map(|k| (((k * 7 + seed * 13) % 23) as f32) / 11.0 - 1.0)
            .collect()
    }

    /// Write one expert as gate|up|down nibbles plus its `.qs` scale row.
    fn container(tag: &str, h: usize, i: usize, group: usize) -> (PathBuf, Vec<u8>, Vec<f32>) {
        let d = std::env::temp_dir().join(format!("fennec-{}-{tag}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let (gd, gs) = pack(&weights(i, h, 1), i, h, group);
        let (ud, us) = pack(&weights(i, h, 2), i, h, group);
        let (dd, ds) = pack(&weights(h, i, 3), h, i, group);
        let mut w = gd.clone();
        w.extend_from_slice(&ud);
        w.extend_from_slice(&dd);
        let mut s = gs.clone();
        s.extend_from_slice(&us);
        s.extend_from_slice(&ds);
        let sb: Vec<u8> = s.iter().flat_map(|v| v.to_le_bytes()).collect();
        let p = format!("{PREFIX}.layers.0.mlp.experts.0");
        shard(&d.join("model-00000.safetensors"), &[
            (&format!("{p}.merged_weight"), "U8", vec![w.len()], w.clone()),
            (&format!("{p}.qs"), "F32", vec![s.len()], sb),
        ]);
        (d, w, s)
    }

    #[test]
    fn a_prequantised_expert_is_adopted_byte_for_byte() {
        let (h, i, group) = (64usize, 32usize, 16usize);
        let (d, w, s) = container("packed", h, i, group);
        let sh = Shards::open(&d).unwrap();
        assert_eq!(packed_group(&sh, PREFIX, h, i), Some(group));

        let (gu, dn) = packed_expert(&sh, PREFIX, 0, 0, h, i, group).unwrap();
        // gate|up is the first two thirds, down the last: no repack, no requantise.
        let cut = 2 * (i * h / 2);
        let scut = 2 * (i * (h / group));
        assert_eq!((gu.rows, gu.cols, gu.group), (2 * i, h, group));
        assert_eq!((dn.rows, dn.cols, dn.group), (h, i, group));
        // Signed nibbles become biased ones by flipping bit 3 of each nibble.
        let biased: Vec<u8> = w.iter().map(|b| b ^ 0x88).collect();
        assert_eq!(gu.data, biased[..cut]);
        assert_eq!(dn.data, biased[cut..]);
        assert_eq!(gu.scale, s[..scut]);
        assert_eq!(dn.scale, s[scut..]);

        // And the adopted matrices must dot to the container's own numbers.
        let x: Vec<f32> = (0..h).map(|k| ((k * 5) % 13) as f32 / 6.0 - 1.0).collect();
        for r in [0usize, 1, i, 2 * i - 1] {
            let want = signed_dot(&w[..cut], &s[..scut], h, group, r, &x);
            let got = gu.dot_row(r, &x);
            assert!((got - want).abs() <= 1e-4 * want.abs().max(1.0), "gate_up row {r}: {got} vs {want}");
        }
        let xd: Vec<f32> = (0..i).map(|k| ((k * 3) % 7) as f32 / 3.0 - 1.0).collect();
        for r in [0usize, 1, h / 2, h - 1] {
            let want = signed_dot(&w[cut..], &s[scut..], i, group, r, &xd);
            let got = dn.dot_row(r, &xd);
            assert!((got - want).abs() <= 1e-4 * want.abs().max(1.0), "down row {r}: {got} vs {want}");
        }
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn the_group_width_is_read_from_the_scale_count() {
        let (h, i) = (64usize, 32usize);
        for group in [8usize, 16, 32] {
            let (d, _, _) = container(&format!("g{group}"), h, i, group);
            let sh = Shards::open(&d).unwrap();
            assert_eq!(packed_group(&sh, PREFIX, h, i), Some(group), "group {group}");
            std::fs::remove_dir_all(&d).ok();
        }
    }

    #[test]
    fn a_bf16_checkpoint_has_no_packed_group() {
        let d = std::env::temp_dir().join(format!("fennec-{}-fused", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let p = format!("{PREFIX}.layers.0.mlp.experts");
        shard(&d.join("model-00000.safetensors"), &[
            (&format!("{p}.gate_up_proj"), "BF16", vec![4, 8], vec![0u8; 64]),
        ]);
        let sh = Shards::open(&d).unwrap();
        assert_eq!(packed_group(&sh, PREFIX, 8, 2), None);
        assert!(!sh.contains(&format!("{p}.0.merged_weight")));
        std::fs::remove_dir_all(&d).ok();
    }
}
