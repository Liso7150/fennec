//! The qwen3_5_moe block math, transcribed from the transformers reference and
//! driven by Fennec's own tensors. Every function here has a golden test.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};
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

/// Whether the host holds its dense projections at one byte per weight instead
/// of two. On the CPU tier those projections are the largest block of the token
/// (~2.3 GiB of bf16 per token) and they are bandwidth-bound, so halving the
/// weight traffic halves the block: 10.16 s median over five requests of the
/// matrix prompt against 12.75 s with the flag off - and the off block ran
/// second, after the on block had warmed the page cache, so the order bias
/// works against the win, not for it. (Both are pre-prefill-batching.)
///
/// It is not free, and the earlier claim here that it cost "a tenth of a
/// percent" was wrong. Per-row int8 puts about one part in `sqrt(cols)` of the
/// dot into the result, because each weight agrees to 0.4% while the sum
/// cancels: measured 12.4% rms on the final logits of the 0.3B dense stack,
/// 20.7% worst over the eight fixture positions, and one of those eight greedy
/// tokens changes. `tests/dense_i8.rs` pins that band and the golden token;
/// the other golden suites keep the exact bf16/f16 dot, because this static
/// defaults to off and only the CLI turns it on. The trade is still worth
/// making by default - Colibri's dense trunk has been int8 all along - but the
/// way back to the exact answer is `--no-dense-i8`.
static DENSE_I8: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub fn set_dense_i8(v: bool) { DENSE_I8.store(v, std::sync::atomic::Ordering::Relaxed); }
pub fn dense_i8() -> bool { DENSE_I8.load(std::sync::atomic::Ordering::Relaxed) }

static HOST_INT2: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub fn set_host_int2(v: bool) { HOST_INT2.store(v, std::sync::atomic::Ordering::Relaxed); }
pub fn host_int2() -> bool { HOST_INT2.load(std::sync::atomic::Ordering::Relaxed) }

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

    /// `n = xs.len() / cols` tokens through this projection in one weight pass,
    /// `out` token-major (`out[t * rows + r]`). Bit-identical to `n` calls of
    /// [`Dense::gemv`]; see [`kernel::gemv_batch`]. A session-hosted ORT
    /// projection has no batch entry point, so it takes the same call per token.
    pub fn gemv_batch(&self, rows: usize, cols: usize, xs: &[f32], out: &mut [f32], threads: usize) {
        match self {
            Dense::F32(w) => kernel::gemv_batch(w, rows, cols, xs, out, threads),
            Dense::Half(w, f) => kernel::gemv_half_batch(w, rows, cols, xs, out, threads, *f),
            Dense::I8(m) => kernel::gemv_i8_batch(&m.w, &m.wsum, &m.scale, rows, cols, xs, out, threads),
            #[cfg(feature = "ort")]
            Dense::Ort(d) => {
                for t in 0..xs.len() / cols {
                    d.gemv(&xs[t * cols..(t + 1) * cols], &mut out[t * rows..(t + 1) * rows])
                        .expect("ort dense gemv");
                }
            }
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

impl Layer {
    /// Hand every sub-layer the decode worker count.
    ///
    /// `Model::load_plan` builds layers on lanes with `threads = 1`, because the
    /// worker pool has one barrier counter for the whole process and two lanes
    /// queueing at once would release each other early. That is right for the
    /// build and wrong to keep: a layer built with 1 thread computes every token
    /// on one core forever, which measured as the whole CPU decode lane running at
    /// 1.28 of its `--threads 8` workers (r21/r22). Once the lanes are joined the
    /// pool has a single caller again and decode may use all of it.
    pub fn set_threads(&mut self, threads: usize) {
        let t = threads.max(1);
        if let Some(m) = &mut self.moe {
            m.threads = t;
        }
        if let Some(f) = &mut self.mlp {
            f.threads = t;
        }
        if let Some(g) = &mut self.gdn {
            g.threads = t;
        }
        if let Some(a) = &mut self.attn {
            a.threads = t;
        }
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

/// A batched prefill chunk's own region of the activation arena. Decode claims
/// everything below `DENSE_OFF` - the MoE block at `moe_arena`'s offsets and
/// the logits - and a chunk that picks a VRAM expert still runs that per-token
/// chain, which writes its `x` at offset 0. So the batch lives above both.
const PREFILL_OFF: usize = 1 << 20;

/// Floats a mixer needs per token beyond its input row: attention's q, k, v
/// and the read-out that feeds `o_proj`, or the delta net's four projections
/// and its gated norm. One region serves every layer, so the chunk reserves the
/// wider of the two.
fn mixer_aux(cfg: &TextConfig) -> usize {
    let (nh, nkv, hd) = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim);
    let q_out = if cfg.attn_output_gate { 2 * nh * hd } else { nh * hd };
    let full = q_out + 2 * nkv * hd + nh * hd;
    let linear = cfg.conv_dim() + 2 * cfg.value_dim() + 2 * cfg.linear_num_value_heads;
    full.max(linear)
}

/// Where one chunk of `n` prompt tokens puts its rows, in arena floats. `xs`
/// holds the tokens and is the only thing the model reads back; `buf` is the
/// normalised input to each block, `res` the block's output before the residual
/// add, `mix` the mixer's intermediates. `w_in` and `w_post` hold the current
/// layer's two RMSNorm gains, which the batched norm kernel reads out of the
/// arena rather than from a weight buffer.
#[derive(Clone, Copy)]
pub struct Chunk {
    pub n: usize,
    pub h: usize,
    pub xs: usize,
    pub buf: usize,
    pub res: usize,
    pub mix: usize,
    pub w_in: usize,
    pub w_post: usize,
}

impl Chunk {
    /// Floats touched, from the base to one past the last element.
    pub fn span(cfg: &TextConfig, n: usize) -> usize {
        let h = cfg.hidden_size;
        3 * n * h + n * mixer_aux(cfg) + 2 * h
    }

    pub fn new(cfg: &TextConfig, n: usize) -> Chunk {
        let (h, aux) = (cfg.hidden_size, mixer_aux(cfg));
        let mix = PREFILL_OFF + 3 * n * h;
        Chunk {
            n,
            h,
            xs: PREFILL_OFF,
            buf: PREFILL_OFF + n * h,
            res: PREFILL_OFF + 2 * n * h,
            mix,
            w_in: mix + n * aux,
            w_post: mix + n * aux + h,
        }
    }
}

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
    /// Layer index in the model, used for prefetcher cache keys.
    pub layer: usize,
    pub gate: Dense,
    pub s_gate: Dense,
    pub s_up: Dense,
    pub s_down: Dense,
    pub s_egate: Vec<f32>,
    /// rows = ids.len() * 2 * I, cols = H  (slot s: rows [2I*s, 2I*s+I) gate, then up)
    pub gu: Q4,
    /// rows = ids.len() * H, cols = I
    pub dn: Q4,
    /// Int2 host experts: when RAM is the constraint, host-resident experts use
    /// 2-bit quantization (50% memory reduction vs int4). Device experts stay Q4.
    pub host_gu: Option<crate::quant::Q2>,
    pub host_dn: Option<crate::quant::Q2>,
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
    /// Shared prefetcher cache for cold experts. When present, the disk path
    /// checks here before loading from shards, and the forward pass hints it
    /// with the next layer's cold experts.
    pub prefetcher: Option<Arc<crate::prefetch::Prefetcher>>,
    /// The next layer's hottest streamed experts, in that layer's own plan order.
    /// Handed to the prefetcher while *this* layer computes, which is the only
    /// window in which a disk read can still arrive in time for this token.
    pub ahead: Vec<usize>,
    /// The layer `ahead` names experts in, kept separate because the last layer
    /// prefetches for layer zero of the next token.
    pub ahead_layer: usize,
    pub dev: Option<DevExperts>,
    pub threads: usize,
    /// Routed picks answered by each tier, and the CPU milliseconds spent on the
    /// host and disk tiers while a device batch was in flight. The overlap is
    /// the claim the split exists for, so it gets counted rather than asserted.
    pub dev_hits: AtomicU64,
    pub host_hits: AtomicU64,
    pub disk_hits: AtomicU64,
    pub overlap_ns: AtomicU64,
    /// Nanoseconds this layer spent computing picks out of the RAM tier, over
    /// every token. `host_hits` x the pack size over this is the tier's decode
    /// throughput - what the lane achieved under load, which is why it is measured
    /// on the real work instead of on a copy loop. It is *not* a cost per byte:
    /// the span is the lane, and the streamed tier is on it too (#473, r23), so
    /// `sched::Bandwidth::apply` reports this and prices with the profile constant.
    pub host_ns: AtomicU64,
    /// Picks the batched-prefill union answered, and the nanoseconds its window
    /// ran. Deliberately separate from `host_hits`/`host_ns`: the union reads one
    /// pack per *expert* and reuses it across every pick of the chunk, so its
    /// bytes-per-second is a different quantity from the decode tier's, and the
    /// plan prices decode tokens. Folding the two together is what made the rate
    /// file claim 21.7 GB/s on a tier the profiler timed at 4.3 (see
    /// `host_traffic`), and `Measured::merge` keeps the warmest sample, so the
    /// blend was always quoted as the price.
    pub union_picks: AtomicU64,
    pub union_ns: AtomicU64,
    /// Matrix bytes the batched-prefill union read. A pick is a pick in
    /// `union_picks`, but the union touches each expert once for the whole chunk,
    /// so the traffic on that path is not `union_picks` x pack and adding the two
    /// together would make the tier look faster than it moved.
    pub host_union_bytes: AtomicU64,
    /// Selections per expert id since this process started. The plan that hands
    /// out the RAM slots is built from a histogram of exactly these numbers, so
    /// a checkpoint that has never been run has no ranking to place itself by
    /// and its hottest experts land on the disk tier by index order.
    pub picks: Vec<AtomicU32>,
}

#[derive(Debug, Clone, Default)]
pub struct Routing {
    pub ids: Vec<usize>,
    pub weights: Vec<f32>,
}

/// Nanoseconds the current `load_plan` has spent inside `MoeLayer::load_set`
/// (`EXPERT_SET_NS`) and inside its `host_storage` call (`EXPERT_READ_NS`), so
/// the difference is the assemble pass. Zeroed when the host pool starts.
static EXPERT_READ_NS: AtomicU64 = AtomicU64::new(0);
static EXPERT_SET_NS: AtomicU64 = AtomicU64::new(0);
/// Nanoseconds the current `load_plan` has spent on the dense half of each
/// layer: the MLP or mixer, the attention block and the two norms.
static DENSE_LOAD_NS: AtomicU64 = AtomicU64::new(0);

impl MoeLayer {
    /// The host tier for `ids`: borrowed from the checkpoint's own mapping when the
    /// container is already int4, read and quantised into owned memory when it is
    /// not. The third element carries the borrowed pairs, and `gu`/`dn` come back
    /// empty alongside them because nothing owns those bytes.
    fn host_storage(
        sh: &Shards,
        cfg: &TextConfig,
        layer: usize,
        ids: &[usize],
        group: usize,
        threads: usize,
    ) -> Result<(Q4, Q4, Option<Vec<(Q4, Q4)>>)> {
        if let Some(m) = map_experts(sh, cfg, layer, ids, group)? {
            return Ok((
                Q4::empty(cfg.hidden_size, group),
                Q4::empty(cfg.moe_intermediate_size, group),
                Some(m),
            ));
        }
        let (gu, dn) = gather_experts(sh, cfg, layer, ids, group, threads)?;
        Ok((gu, dn, None))
    }

    /// Bytes the resident host set holds, whichever of the two storages backs it.
    fn resident_bytes(gu: &Q4, dn: &Q4, mapped: &Option<Vec<(Q4, Q4)>>) -> usize {
        match mapped {
            Some(v) => v.iter().map(|(g, d)| g.bytes_total() + d.bytes_total()).sum(),
            None => gu.bytes_total() + dn.bytes_total(),
        }
    }

    /// Every expert resident, straight from the checkpoint. The shape a machine
    /// with enough RAM for the whole layer uses; `load_set` is the general case.
    pub fn load(sh: &Shards, cfg: Arc<TextConfig>, layer: usize, group: usize, threads: usize) -> Result<MoeLayer> {
        let all: Vec<usize> = (0..cfg.num_experts).collect();
        let t0 = std::time::Instant::now();
        let (gu, dn, mapped) = Self::host_storage(sh, &cfg, layer, &all, group, threads)?;
        eprintln!(
            "[fennec] layer {layer}: {} experts {} in {:.1} s, {:.2} GiB int4",
            all.len(),
            if mapped.is_some() { "borrowed from the mapping" } else { "read and quantised" },
            t0.elapsed().as_secs_f64(),
            Self::resident_bytes(&gu, &dn, &mapped) as f64 / 2f64.powi(30)
        );
        Self::assemble(sh, cfg, layer, all, gu, dn, mapped, None, threads)
    }

    /// Exactly these experts in RAM, the rest answered from `stream` when the
    /// router picks one. `ids` arrives in the order the plan ranks them, so the
    /// hottest expert is slot 0 and a truncated list is a truncated cache.
    ///
    /// A pre-quantised container is borrowed rather than copied: `map_experts`
    /// hands back one mapped pair per slot and the gathered matrices stay empty.
    /// The nibbles stay clean file pages, which is what keeps the resident set out
    /// of the commit charge Windows has to find room for under pressure.
    pub fn load_set(
        sh: &Arc<Shards>,
        cfg: Arc<TextConfig>,
        layer: usize,
        ids: &[usize],
        stream: Option<Stream>,
        group: usize,
        threads: usize,
    ) -> Result<MoeLayer> {
        let t0 = std::time::Instant::now();
        let (gu, dn, mapped) = Self::host_storage(sh, &cfg, layer, ids, group, threads)?;
        EXPERT_READ_NS.fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
        let moe = Self::assemble(sh, cfg, layer, ids.to_vec(), gu, dn, mapped, stream, threads);
        EXPERT_SET_NS.fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
        moe
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
        let picks: Vec<AtomicU32> = (0..cfg.num_experts).map(|_| AtomicU32::new(0)).collect();
        for (s, &e) in ids.iter().enumerate() {
            slot[e] = s as i32;
        }
        
        // Convert Q4 host experts to Q2 when --host-int2 is set
        let (host_gu, host_dn, gu, dn) = if host_int2() && mapped.is_none() {
            let t0 = std::time::Instant::now();
            let q4_bytes = gu.bytes_total() + dn.bytes_total();
            let group = gu.group;
            let q2_gu = gu.to_q2(threads);
            let q2_dn = dn.to_q2(threads);
            let q2_bytes = q2_gu.bytes_total() + q2_dn.bytes_total();
            eprintln!(
                "[fennec] layer {layer}: converted {} host experts to int2 in {:.1} s ({:.2} -> {:.2} GiB, saved {:.1}%)",
                ids.len(),
                t0.elapsed().as_secs_f64(),
                q4_bytes as f64 / 2f64.powi(30),
                q2_bytes as f64 / 2f64.powi(30),
                100.0 * (1.0 - q2_bytes as f64 / q4_bytes as f64)
            );
            // Free the Q4 matrices: replace with empty shells so only Q2 is held
            let gu_empty = Q4::empty(cfg.hidden_size, group);
            let dn_empty = Q4::empty(cfg.moe_intermediate_size, group);
            (Some(q2_gu), Some(q2_dn), gu_empty, dn_empty)
        } else {
            (None, None, gu, dn)
        };
        
        Ok(MoeLayer {
            cfg,
            layer,
            gate,
            s_gate,
            s_up,
            s_down,
            s_egate,
            gu,
            dn,
            host_gu,
            host_dn,
            mapped,
            ids,
            slot,
            stream: stream.map(Mutex::new),
            prefetcher: None,
            ahead: Vec::new(),
            ahead_layer: 0,
            dev: None,
            threads: threads.max(1),
            dev_hits: AtomicU64::new(0),
            host_hits: AtomicU64::new(0),
            disk_hits: AtomicU64::new(0),
            overlap_ns: AtomicU64::new(0),
            host_ns: AtomicU64::new(0),
            host_union_bytes: AtomicU64::new(0),
            union_picks: AtomicU64::new(0),
            union_ns: AtomicU64::new(0),
            picks,
        })
    }

    pub fn from_parts(cfg: Arc<TextConfig>, layer: usize, parts: MoeParts, gu: Q4, dn: Q4, threads: usize) -> MoeLayer {
        // the matrices carry the resident set: gu.rows / 2I experts, ids 0..n
        let ids: Vec<usize> = (0..gu.rows / (2 * cfg.moe_intermediate_size)).collect();
        let mut slot = vec![-1i32; cfg.num_experts];
        let picks: Vec<AtomicU32> = (0..cfg.num_experts).map(|_| AtomicU32::new(0)).collect();
        for (s, &e) in ids.iter().enumerate() {
            slot[e] = s as i32;
        }
        
        // Convert Q4 host experts to Q2 when --host-int2 is set
        let (host_gu, host_dn, gu, dn) = if host_int2() {
            let group = gu.group;
            let q2_gu = gu.to_q2(threads);
            let q2_dn = dn.to_q2(threads);
            let gu_empty = Q4::empty(cfg.hidden_size, group);
            let dn_empty = Q4::empty(cfg.moe_intermediate_size, group);
            (Some(q2_gu), Some(q2_dn), gu_empty, dn_empty)
        } else {
            (None, None, gu, dn)
        };
        
        MoeLayer {
            cfg,
            layer,
            gate: parts.gate,
            s_gate: parts.s_gate,
            s_up: parts.s_up,
            s_down: parts.s_down,
            s_egate: parts.s_egate,
            gu,
            dn,
            host_gu,
            host_dn,
            mapped: None,
            ids,
            slot,
            stream: None,
            prefetcher: None,
            ahead: Vec::new(),
            ahead_layer: 0,
            dev: None,
            threads: threads.max(1),
            dev_hits: AtomicU64::new(0),
            host_hits: AtomicU64::new(0),
            disk_hits: AtomicU64::new(0),
            overlap_ns: AtomicU64::new(0),
            host_ns: AtomicU64::new(0),
            host_union_bytes: AtomicU64::new(0),
            union_picks: AtomicU64::new(0),
            union_ns: AtomicU64::new(0),
            picks,
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

    /// Batch routing: route all n tokens at once, returning n Routing structs.
    /// More efficient than calling route() n times because it batches the gate.gemv.
    pub fn route_batch(&self, xs: &[f32], n: usize) -> Vec<Routing> {
        let h = self.cfg.hidden_size;
        let e = self.cfg.num_experts;
        let k = self.cfg.num_experts_per_tok;
        let t = self.threads;
        let mut routings = Vec::with_capacity(n);
        
        // Batch the gate projection: n tokens × H → n tokens × E
        let mut all_logits = vec![0f32; n * e];
        self.gate.gemv_batch(e, h, xs, &mut all_logits, t);
        
        // Softmax top-k for each token
        for i in 0..n {
            let logits = &all_logits[i * e..(i + 1) * e];
            let mut weights = vec![0f32; k];
            let mut ids = vec![0usize; k];
            kernel::softmax_topk(logits, k, &mut weights, &mut ids);
            routings.push(Routing { ids, weights });
        }
        
        routings
    }

    /// y = expert_e(x) into `y` (len H). Gate/up are the two halves of the row
    /// block that holds e; `expert` is the id->slot lookup on top of it.
    pub fn expert(&self, e: usize, x: &[f32], hh: &mut [f32], y: &mut [f32]) {
        let (i, h) = (self.cfg.moe_intermediate_size, self.cfg.hidden_size);
        let s = self.slot[e];
        assert!(s >= 0, "expert {e} is not resident in this layer");
        
        // Use Q2 path when host experts are int2-quantized
        if let (Some(gu_q2), Some(dn_q2)) = (&self.host_gu, &self.host_dn) {
            expert_in_q2(gu_q2, dn_q2, i, h, s as usize, x, hh, y);
        } else {
            let (gu, dn, s) = self.expert_pair(e);
            expert_in(gu, dn, i, h, s, x, hh, y);
        }
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

    /// Matrix bytes one pick from the RAM tier reads, from the tier's own
    /// accounting rather than from a formula, so a borrowed pack and an int2
    /// pack are priced at what they actually are. Zero when the layer has no
    /// RAM tier, which is also when it should record no host traffic.
    pub fn host_pick_bytes(&self) -> usize {
        let n = self.slot.iter().filter(|&&s| s >= 0).count();
        if n == 0 {
            0
        } else {
            self.host_bytes() / n
        }
    }

    /// Bytes and nanoseconds the per-token host path moved: one pack per pick,
    /// over the window the picks ran. `host_pick_bytes` is the tier's own pack, so
    /// a borrowed pair and an int2 pair are priced at what they actually are.
    pub fn decode_traffic(&self) -> (u64, u64) {
        (
            self.host_hits
                .load(Relaxed)
                .saturating_mul(self.host_pick_bytes() as u64),
            self.host_ns.load(Relaxed),
        )
    }

    /// Bytes and nanoseconds the batched-prefill union moved: one pack per expert
    /// the chunk touched, summed where the read happened. Kept out of
    /// `decode_traffic` because a union pick costs a *fraction* of a pack, and the
    /// two folded together priced decode at 21.7 GB/s on a lane timed at 4.3.
    pub fn union_traffic(&self) -> (u64, u64) {
        (
            self.host_union_bytes.load(Relaxed),
            self.union_ns.load(Relaxed),
        )
    }

    pub fn host_gu_bytes(&self) -> usize {
        match &self.mapped {
            Some(v) => v.iter().map(|(g, _)| g.bytes_total()).sum(),
            None => {
                // Q2 takes precedence when active
                if let Some(q2) = &self.host_gu {
                    q2.bytes_total()
                } else {
                    self.gu.bytes_total()
                }
            }
        }
    }

    pub fn host_dn_bytes(&self) -> usize {
        match &self.mapped {
            Some(v) => v.iter().map(|(_, d)| d.bytes_total()).sum(),
            None => {
                // Q2 takes precedence when active
                if let Some(q2) = &self.host_dn {
                    q2.bytes_total()
                } else {
                    self.dn.bytes_total()
                }
            }
        }
    }

    /// The same, counting allocator capacity rather than matrix size.
    pub fn host_bytes_held(&self) -> usize {
        match &self.mapped {
            Some(v) => v.iter().map(|(g, d)| g.bytes_held() + d.bytes_held()).sum(),
            None => {
                // Q2 takes precedence when active
                let gu_bytes = if let Some(q2) = &self.host_gu {
                    q2.bytes_held()
                } else {
                    self.gu.bytes_held()
                };
                let dn_bytes = if let Some(q2) = &self.host_dn {
                    q2.bytes_held()
                } else {
                    self.dn.bytes_held()
                };
                gu_bytes + dn_bytes
            }
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

    /// The prefetcher's copy of one of this layer's streamed experts, waiting out
    /// a read the worker already started rather than issuing a second copy of the
    /// same bytes. `None` means nobody is fetching it and the pool must.
    fn pick_cached(&self, id: usize) -> Option<Arc<crate::prefetch::CachedExpert>> {
        let pf = self.prefetcher.as_ref()?;
        pf.get_cached(self.layer, id)
            .or_else(|| pf.wait_cached(self.layer, id, crate::prefetch::AHEAD_WAIT))
    }

    /// The host and disk tiers, on the CPU. Called while a device batch is in
    /// flight, which is the whole point of splitting the tiers this way.
    fn cpu_picks(&self, x: &[f32], r: &Routing, host: &[usize], cold: &[usize], parts: &mut [Vec<f32>]) {
        let (h, i) = (self.cfg.hidden_size, self.cfg.moe_intermediate_size);
        // Prime the next layer before touching this one's cold experts: a hint
        // for the layer asking right now cannot land before the loop below reads
        // the same bytes, so only the layer after this one is worth queueing.
        if let Some(pf) = &self.prefetcher {
            pf.prefetch_ahead(self.ahead_layer, &self.ahead);
        }
        if !cold.is_empty() {
            let mut pool = self
                .stream
                .as_ref()
                .unwrap_or_else(|| panic!("a non-resident expert needs a pool"))
                .lock()
                .unwrap();
            // Read every cold expert of this layer before computing any of them.
            // The router named them all above, so this is the same traffic the
            // loop below would issue, just deep instead of serial.
            let want: Vec<usize> = cold
                .iter()
                .map(|&n| r.ids[n])
                .filter(|&id| !self.prefetcher.as_ref().is_some_and(|pf| pf.covers(self.layer, id)))
                .collect();
            pool.warm(&want);
            for &n in cold {
                let (id, w) = (r.ids[n], r.weights[n]);
                let mut hh = vec![0f32; i];
                let mut y = vec![0f32; h];
                let mut part = vec![0f32; h];
                // The prefetch cache first: a hit costs a reference count.
                let loaded = if let Some(c) = self.pick_cached(id) {
                    expert_in(&c.gate_up, &c.down, i, h, 0, x, &mut hh, &mut y);
                    true
                } else {
                    false
                };
                if !loaded {
                    pool.with(id, |gu, dn| expert_in(gu, dn, i, h, 0, x, &mut hh, &mut y))
                        .unwrap_or_else(|e| panic!("streamed expert {id}: {e}"));
                }
                kernel::axpy(&mut part, &y, w);
                parts[n] = part;
            }
        }
        // Timed, because the plan's host price was being taken from a copy loop
        // that did not represent the tier: `fennec bench` runs 1.5-2.4x faster than
        // a decode (13.3-19.4 GB/s against 8.1, r13), and a placement priced on it
        // thinks RAM picks are nearly free. The activation buffers this section
        // also touches are under 1% of the pack each pick reads, so picks x pack
        // bytes over this span is the tier's decode throughput on this box - which
        // #473 turned out not to be a cost per byte either, because the same shape
        // read 4.2 GB/s cold and 15.6 warm with the streamed window moving in
        // step. So this number is reported as contention and the plan keeps its
        // profile constant; see `Bandwidth::apply`.
        let t_host = (!host.is_empty()).then(std::time::Instant::now);
        if self.threads == 1 || host.len() < 2 || self.host_gu.is_some() {
            // Serial path: one expert at a time. Also used when int2 host experts
            // are active, since the batched DotPrep path is Q4-specific.
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
        if let Some(t) = t_host {
            self.host_ns.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
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
        let r = {
            let _t = prof::start(prof::ROUTE);
            self.route(x)
        };
        let out = match &self.dev {
            Some(d) => {
                let mut g = d.gpu.lock().unwrap();
                self.forward_with(Some(&mut g), x, &r)
            }
            None => self.forward_with(None, x, &r),
        };
        (out, r)
    }

    /// Fold the router's picks into the history the next plan is ranked by. One
    /// relaxed increment per pick on the thread that did the routing, which is
    /// why it can sit in the hot path: the ids are already in hand.
    fn note_picks(&self, ids: &[usize]) {
        for &id in ids {
            if let Some(c) = self.picks.get(id) {
                c.fetch_add(1, Relaxed);
            }
        }
    }

    /// One token whose routing is already known, under the guard its caller
    /// holds. The device tier shares one `Mutex<Gpu>` across the whole stack
    /// and `std::sync::Mutex` is not reentrant, so a prompt chunk that is
    /// already inside that lock must come through here rather than back
    /// through `forward`. A host-only plan has no arena to name and takes only
    /// the branch below that never touches a device pick.
    fn forward_with(&self, gpu: Option<&mut Gpu>, x: &[f32], r: &Routing) -> Vec<f32> {
        let (h, i) = (self.cfg.hidden_size, self.cfg.moe_intermediate_size);
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
        self.note_picks(&r.ids);
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
            let g = gpu.expect("a device pick needs the guard its chunk runs under");
            let read_syncs = g.read_syncs();
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
            // CUDA's read_into is a blocking DtoH that synchronises the context,
            // so an explicit wait here pays a second WDDM flush per layer for
            // nothing (the same trade step_fused makes). Vulkan's mapped read
            // still needs the fence wait first.
            if !read_syncs {
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
        out
    }

    /// A chunk of prompt tokens through the routed experts, grouped by expert.
    ///
    /// Colibri's `moe()` does this union whenever its batch width `S > 1`: every
    /// unique expert of the batch is read once and multiplied against all the
    /// positions that picked it. A chunk's `n * k` picks land on fewer distinct
    /// experts than there are picks, so walking (expert, weight row, positions
    /// that use it) keeps each weight row in L1 across those positions instead of
    /// pulling it from DRAM once per pick. Every dot is still the same product
    /// against the same activation, non-resident picks still run through the
    /// streaming pool, and each row still sums its picks in pick order with the
    /// shared expert last, so the chunk is bit-identical to `n` calls of
    /// [`MoeLayer::forward`].
    ///
    /// Returns false when there is nothing to group (`n < 2`), so a caller can
    /// fall back to the per-token path — which is also the only path that
    /// overlaps host work with a device batch.
    pub fn prefill(&self, xs: &[f32], out: &mut [f32]) -> bool {
        let (h, i) = (self.cfg.hidden_size, self.cfg.moe_intermediate_size);
        let (e, k) = (self.cfg.num_experts, self.cfg.num_experts_per_tok);
        let n = xs.len() / h;
        if n < 2 {
            return false;
        }
        let mut logits = vec![0f32; n * e];
        {
            let _t = prof::start(prof::ROUTE);
            self.gate.gemv_batch(e, h, xs, &mut logits, self.threads);
        }
        let mut ids = vec![0usize; n * k];
        let mut wts = vec![0f32; n * k];
        for r in 0..n {
            kernel::softmax_topk(
                &logits[r * e..(r + 1) * e],
                k,
                &mut wts[r * k..(r + 1) * k],
                &mut ids[r * k..(r + 1) * k],
            );
        }
        self.note_picks(&ids);
        // The union of the chunk's picks, in first-seen order, each expert
        // carrying the flat pick indices that use it.
        let mut at = vec![usize::MAX; e];
        let mut union: Vec<(usize, Vec<u32>)> = Vec::new();
        let mut cold: Vec<usize> = Vec::new();
        for (q, &id) in ids.iter().enumerate() {
            if self.slot[id] < 0 {
                cold.push(q);
                continue;
            }
            let u = at[id];
            at[id] = if u == usize::MAX {
                union.push((id, Vec::new()));
                union.len() - 1
            } else {
                u
            };
            union[at[id]].1.push(q as u32);
        }
        if union.is_empty() {
            return false;
        }
        let mut parts = vec![0f32; n * k * h];
        {
            let _cpu = prof::start(prof::MOE_CPU);
            self.union_picks
                .fetch_add((n * k - cold.len()) as u64, Relaxed);
            // Prime the next layer while this one still has host work left to
            // do; this layer's own misses are too late to overlap anything.
            if let Some(pf) = &self.prefetcher {
                pf.prefetch_ahead(self.ahead_layer, &self.ahead);
            }
            if !cold.is_empty() {
                let mut pool = self
                    .stream
                    .as_ref()
                    .unwrap_or_else(|| panic!("a non-resident expert needs a pool"))
                    .lock()
                    .unwrap();
                let want: Vec<usize> = cold
                    .iter()
                    .map(|&q| ids[q])
                    .filter(|&id| !self.prefetcher.as_ref().is_some_and(|pf| pf.covers(self.layer, id)))
                    .collect();
                pool.warm(&want);
                let mut hh = vec![0f32; i];
                let mut y = vec![0f32; h];
                for &q in &cold {
                    let (id, w, r) = (ids[q], wts[q], q / k);
                    let loaded = if let Some(c) = self.pick_cached(id) {
                        expert_in(&c.gate_up, &c.down, i, h, 0, &xs[r * h..(r + 1) * h], &mut hh, &mut y);
                        true
                    } else {
                        false
                    };
                    if !loaded {
                        pool.with(id, |gu, dn| {
                            expert_in(gu, dn, i, h, 0, &xs[r * h..(r + 1) * h], &mut hh, &mut y)
                        })
                        .unwrap_or_else(|er| panic!("streamed expert {id}: {er}"));
                    }
                    kernel::axpy(&mut parts[q * h..(q + 1) * h], &y, w);
                }
            }
            // The RAM tier's bandwidth is priced on this work too, so the union's
            // own read is timed and counted: one pack per union member, not one
            // per pick.
            let t_host = std::time::Instant::now();
            // Within a layer every expert comes from the same pack, so the first
            // resident one decides whether the integer dot is available to all of
            // them — the assumption `cpu_picks` already makes.
            let (gu0, _, _) = self.expert_pair(union[0].0);
            let int = gu0.wants_int();
            let mut px: Vec<DotPrep> = (0..n).map(|_| DotPrep::with_int(h, int)).collect();
            for r in 0..n {
                px[r].fill(&xs[r * h..(r + 1) * h]);
            }
            let nu = union.len();
            let mut hh = vec![0f32; n * k * i];
            {
                let hhp = kernel::SharedPtr::new(hh.as_mut_ptr());
                kernel::par_index(nu * i, self.threads, |start, len| {
                    for g in start..start + len {
                        let u = g / i;
                        let j = g - u * i;
                        let (gu, _, s) = self.expert_pair(union[u].0);
                        let base = s * 2 * i;
                        for &q in &union[u].1 {
                            let p = &px[q as usize / k];
                            let a = gu.dot_prepped(base + j, p);
                            let b = gu.dot_prepped(base + i + j, p);
                            unsafe { *hhp.at(q as usize * i + j) = kernel::silu(a) * b };
                        }
                    }
                });
            }
            let mut ph: Vec<DotPrep> = (0..n * k).map(|_| DotPrep::with_int(i, int)).collect();
            for q in 0..n * k {
                ph[q].fill(&hh[q * i..(q + 1) * i]);
            }
            let pp = kernel::SharedPtr::new(parts.as_mut_ptr());
            kernel::par_index(nu * h, self.threads, |start, len| {
                for g in start..start + len {
                    let u = g / h;
                    let j = g - u * h;
                    let (_, dn, s) = self.expert_pair(union[u].0);
                    let row = s * h + j;
                    for &q in &union[u].1 {
                        let v = dn.dot_prepped(row, &ph[q as usize]);
                        unsafe { *pp.at(q as usize * h + j) = wts[q as usize] * v };
                    }
                }
            });
            self.host_union_bytes
                .fetch_add(nu as u64 * self.host_pick_bytes() as u64, Relaxed);
            self.union_ns
                .fetch_add(t_host.elapsed().as_nanos() as u64, Relaxed);
        }
        let si = self.cfg.shared_expert_intermediate_size;
        let mut shared_out = vec![0f32; n * h];
        {
            let _t = prof::start(prof::MOE_SHARED);
            let mut g = vec![0f32; n * si];
            let mut u = vec![0f32; n * si];
            self.s_gate.gemv_batch(si, h, xs, &mut g, self.threads);
            self.s_up.gemv_batch(si, h, xs, &mut u, self.threads);
            for q in 0..n * si {
                u[q] = kernel::silu(g[q]) * u[q];
            }
            self.s_down.gemv_batch(h, si, &u, &mut shared_out, self.threads);
        }
        // Same reduction `forward` does, row by row: pick order, shared expert
        // last, each step a `+= 1.0 *` so the floating-point order matches.
        let _res = prof::start(prof::RESIDUAL);
        for r in 0..n {
            let gate = kernel::sigmoid(kernel::dot(&self.s_egate, &xs[r * h..(r + 1) * h]));
            let o = &mut out[r * h..(r + 1) * h];
            o.fill(0.0);
            for q in r * k..(r + 1) * k {
                kernel::axpy(o, &parts[q * h..(q + 1) * h], 1.0);
            }
            kernel::axpy(o, &shared_out[r * h..(r + 1) * h], gate);
        }
        true
    }

    /// Device-batched grouped expert union: processes all n tokens through the MoE
    /// layer. Uses grouped expert processing for HOST experts (like Colibri's batch-union moe()).
    /// 
    /// Tokens whose routing touches VRAM run the device chain one at a time;
    /// the rest share one grouped pass over the host-resident and cold experts.
    /// 
    /// Returns the output buffer (n × h floats) ready to write back to GPU.
    pub fn prefill_device_grouped(&self, gpu: &mut Gpu, xs: &[f32], n: usize) -> Vec<f32> {
        let (h, i) = (self.cfg.hidden_size, self.cfg.moe_intermediate_size);
        let (e, k) = (self.cfg.num_experts, self.cfg.num_experts_per_tok);
        
        let routings = self.route_batch(xs, n);
        let on_device = |id: usize| self.dev.as_ref().is_some_and(|d| d.slot[id] >= 0);
        // A token whose picks are all in RAM joins the union; one that touches
        // VRAM runs the device chain on its own, because the union has no
        // batched q4 kernel yet. Splitting per token rather than per chunk is
        // the point: in a device plan most picks are in VRAM, so the old
        // all-or-nothing test sent the whole chunk down the per-token path and
        // the union never ran on the tier it was written for.
        let vram: Vec<usize> = (0..n).filter(|&t| routings[t].ids.iter().copied().any(on_device)).collect();
        let grouped: Vec<usize> = (0..n).filter(|t| !vram.contains(t)).collect();
        
        // No device experts - use batched grouped processing for host experts
        let mut at = vec![usize::MAX; e];
        let mut union: Vec<(usize, Vec<(usize, usize, f32)>)> = Vec::new();
        let mut cold: Vec<(usize, usize, usize, f32)> = Vec::new();
        
        for &token_idx in &grouped {
            let routing = &routings[token_idx];
            for pick_idx in 0..k {
                let id = routing.ids[pick_idx];
                let w = routing.weights[pick_idx];
                
                if self.slot[id] >= 0 {
                    // Host resident expert
                    let u = at[id];
                    let union_idx = if u == usize::MAX {
                        at[id] = union.len();
                        union.push((id, Vec::new()));
                        union.len() - 1
                    } else {
                        u
                    };
                    union[union_idx].1.push((token_idx, pick_idx, w));
                } else {
                    // Cold (disk) expert
                    cold.push((token_idx, pick_idx, id, w));
                }
            }
        }
        
        let mut parts = vec![0f32; n * k * h];
        
        // Process cold experts (disk-streamed)
        if let Some(pf) = &self.prefetcher {
            pf.prefetch_ahead(self.ahead_layer, &self.ahead);
        }
        if !cold.is_empty() {
            let mut pool = self.stream.as_ref().unwrap().lock().unwrap();
            let want: Vec<usize> = cold
                .iter()
                .map(|&(_, _, id, _)| id)
                .filter(|&id| !self.prefetcher.as_ref().is_some_and(|pf| pf.covers(self.layer, id)))
                .collect();
            pool.warm(&want);
            let mut hh = vec![0f32; i];
            let mut y = vec![0f32; h];
            for &(token_idx, pick_idx, id, w) in &cold {
                let x = &xs[token_idx * h..(token_idx + 1) * h];
                let loaded = if let Some(c) = self.pick_cached(id) {
                    expert_in(&c.gate_up, &c.down, i, h, 0, x, &mut hh, &mut y);
                    true
                } else {
                    false
                };
                if !loaded {
                    pool.with(id, |gu, dn| expert_in(gu, dn, i, h, 0, x, &mut hh, &mut y))
                        .unwrap_or_else(|er| panic!("streamed expert {id}: {er}"));
                }
                let out = &mut parts[(token_idx * k + pick_idx) * h..(token_idx * k + pick_idx + 1) * h];
                kernel::axpy(out, &y, w);
            }
        }
        
        // Process host expert groups
        if !union.is_empty() {
            let (gu0, _, _) = self.expert_pair(union[0].0);
            let int = gu0.wants_int();
            let px: Vec<DotPrep> = (0..n).map(|r| {
                let mut prep = DotPrep::with_int(h, int);
                prep.fill(&xs[r * h..(r + 1) * h]);
                prep
            }).collect();
            
            let mut hh = vec![0f32; n * k * i];
            for (_u_idx, (expert_id, picks)) in union.iter().enumerate() {
                let (gu, _, s) = self.expert_pair(*expert_id);
                let base = s * 2 * i;
                for &(token_idx, pick_idx, _) in picks {
                    let p = &px[token_idx];
                    let hh_off = (token_idx * k + pick_idx) * i;
                    for j in 0..i {
                        let g = gu.dot_prepped(base + j, p);
                        let u = gu.dot_prepped(base + i + j, p);
                        hh[hh_off + j] = kernel::silu(g) * u;
                    }
                }
            }
            
            let ph: Vec<DotPrep> = (0..n * k).map(|q| {
                let mut prep = DotPrep::with_int(i, int);
                prep.fill(&hh[q * i..(q + 1) * i]);
                prep
            }).collect();
            
            for (_u_idx, (expert_id, picks)) in union.iter().enumerate() {
                let (_, dn, s) = self.expert_pair(*expert_id);
                let row_base = s * h;
                for &(token_idx, pick_idx, w) in picks {
                    let q = token_idx * k + pick_idx;
                    let p = &ph[q];
                    let out = &mut parts[q * h..(q + 1) * h];
                    for j in 0..h {
                        let v = dn.dot_prepped(row_base + j, p);
                        out[j] = w * v;
                    }
                }
            }
        }
        
        // Add shared expert contribution for the grouped tokens, on the host:
        // attaching the device tier uploads this expert but keeps the host
        // copy, so the union multiplies the same weights `forward` would. The
        // VRAM tokens below get it through the device chain instead.
        let si = self.cfg.shared_expert_intermediate_size;
        let mut shared_out = vec![0f32; n * h];
        {
            let mut g = vec![0f32; n * si];
            let mut u = vec![0f32; n * si];
            self.s_gate.gemv_batch(si, h, xs, &mut g, self.threads);
            self.s_up.gemv_batch(si, h, xs, &mut u, self.threads);
            for q in 0..n * si {
                u[q] = kernel::silu(g[q]) * u[q];
            }
            self.s_down.gemv_batch(h, si, &u, &mut shared_out, self.threads);
        }
        
        // Sum all contributions per token
        let mut out = vec![0f32; n * h];
        for &r in &grouped {
            let gate = kernel::sigmoid(kernel::dot(&self.s_egate, &xs[r * h..(r + 1) * h]));
            let o = &mut out[r * h..(r + 1) * h];
            o.fill(0.0);
            for q in r * k..(r + 1) * k {
                kernel::axpy(o, &parts[q * h..(q + 1) * h], 1.0);
            }
            kernel::axpy(o, &shared_out[r * h..(r + 1) * h], gate);
        }
        for &t in &vram {
            let x = &xs[t * h..(t + 1) * h];
            let moe_out = self.forward_with(Some(gpu), x, &routings[t]);
            out[t * h..(t + 1) * h].copy_from_slice(&moe_out);
        }
        
        out
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

/// Int2 variant of expert_in: uses Q2 dot product for host-resident experts when
/// --host-int2 is set. Slower per-dot but halves RAM usage.
pub fn expert_in_q2(gu: &crate::quant::Q2, dn: &crate::quant::Q2, i: usize, h: usize, s: usize, x: &[f32], hh: &mut [f32], y: &mut [f32]) {
    let base = s * 2 * i;
    for j in 0..i {
        let g = gu.dot_row(base + j, x);
        let u = gu.dot_row(base + i + j, x);
        hh[j] = kernel::silu(g) * u;
    }
    for j in 0..h {
        y[j] = dn.dot_row(s * h + j, hh);
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

/// One streamed expert without moving its bytes: whichever pre-quantised form
/// this checkpoint has, the FQ4 sidecar's rows or the packed container's blocks,
/// both left in the file mapping they came from.
///
/// `None` means the tier has to own them - a bf16 checkpoint that is quantised on
/// the way in - so the caller pays the allocation and the encode pass. Where a
/// pre-quantised source does exist, copying it into a ring slot made the streamed
/// lane touch every expert twice: once to fault its pages in, once to `memcpy` the
/// same 1.7 MiB, in an anonymous page the OS cannot share or reclaim quietly.
pub fn borrow_expert(
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
                return Some((|| -> Result<(Q4, Q4)> {
                    Ok((
                        sc.rows_borrowed(&gi, id * 2 * i, (id + 1) * 2 * i)?,
                        sc.rows_borrowed(&di, id * h, (id + 1) * h)?,
                    ))
                })());
            }
        }
    }
    map_expert(sh, prefix, layer, id, h, i, group)
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
    let mut out = Vec::with_capacity(ids.len());
    for &e in ids {
        match map_expert(sh, &cfg.prefix, layer, e, h, i, group) {
            Some(r) => out.push(r?),
            None => return Ok(None),
        }
    }
    Ok(Some(out))
}

/// One expert of a packed int4 container, still sitting in the shard mapping.
///
/// The nibbles are never read here, only located, so the cost of the pair is its
/// scales: a ninth of the bytes. `None` means "not a packed container", and an
/// `Err` is the tensor read failing, which the caller reports rather than
/// replays through the copying path.
fn map_expert(
    sh: &Shards,
    prefix: &str,
    layer: usize,
    e: usize,
    h: usize,
    i: usize,
    group: usize,
) -> Option<Result<(Q4, Q4)>> {
    if h % group != 0 {
        return None;
    }
    let ep = format!("{prefix}.layers.{layer}.mlp.experts.{e}");
    let (map, off) = sh.mapped(&format!("{ep}.merged_weight"))?;
    let block_bytes = i * h / 2;
    let block_scales = i * (h / group);
    let s = match sh.slice_f32(&format!("{ep}.qs"), 0, 3 * block_scales) {
        Ok(s) => s,
        Err(err) => return Some(Err(err)),
    };
    let cut = 2 * block_scales;
    Some(Ok((
        Q4::mapped(map.clone(), off, 2 * i, h, group, s[..cut].to_vec(), true),
        Q4::mapped(map, off + 2 * block_bytes, h, i, group, s[cut..].to_vec(), true),
    )))
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
        gu.shrink_to_fit();
        dn.shrink_to_fit();
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
        gu.shrink_to_fit();
        dn.shrink_to_fit();
        return Ok((gu, dn));
    }
    // Detect which expert tensor format the checkpoint uses. Three layouts exist:
    //   A) Fused `{p}.gate_up_proj` (Qwen3-MoE safetensors)
    //   B) Separate `{p}.gate_proj` + `{p}.up_proj` (some HF exports)
    //   C) Edge0 `{p2}.gate_proj.weight` + `{p2}.up_proj.weight` under `switch_mlp`
    let has_fused = sh.contains(&gu_name);
    let gate_name = format!("{p}.gate_proj");
    let up_name = format!("{p}.up_proj");
    let gate_w = format!("{gate_name}.weight");
    let up_w = format!("{up_name}.weight");
    let p2 = format!("{}.layers.{layer}.mlp.switch_mlp", cfg.prefix);
    let edge_gate = format!("{p2}.gate_proj.weight");
    let edge_up = format!("{p2}.up_proj.weight");
    let edge_dn = format!("{p2}.down_proj.weight");
    let has_separate = sh.contains(&gate_name) || sh.contains(&gate_w);
    let has_edge0 = sh.contains(&edge_gate);
    for &e in ids {
        if has_fused {
            let g = sh.slice_f32(&gu_name, e * 2 * i * h, (e + 1) * 2 * i * h)?;
            gu.push(&Q4::quantize(&g, 2 * i, h, group, threads));
        } else if has_edge0 {
            // Edge0: separate gate/up under switch_mlp, each [num_experts*I, H] with .weight
            let gate_rows = sh.slice_f32(&edge_gate, e * i * h, (e + 1) * i * h)?;
            let up_rows = sh.slice_f32(&edge_up, e * i * h, (e + 1) * i * h)?;
            let mut fused = gate_rows;
            fused.extend_from_slice(&up_rows);
            gu.push(&Q4::quantize(&fused, 2 * i, h, group, threads));
        } else if has_separate {
            let gn = if sh.contains(&gate_w) { &gate_w } else { &gate_name };
            let un = if sh.contains(&up_w) { &up_w } else { &up_name };
            let gate_rows = sh.slice_f32(gn, e * i * h, (e + 1) * i * h)?;
            let up_rows = sh.slice_f32(un, e * i * h, (e + 1) * i * h)?;
            let mut fused = gate_rows;
            fused.extend_from_slice(&up_rows);
            gu.push(&Q4::quantize(&fused, 2 * i, h, group, threads));
        } else {
            let g = sh.slice_f32(&gu_name, e * 2 * i * h, (e + 1) * 2 * i * h)?;
            gu.push(&Q4::quantize(&g, 2 * i, h, group, threads));
        }
        let dn_slice = if has_edge0 {
            sh.slice_f32(&edge_dn, e * h * i, (e + 1) * h * i)?
        } else {
            sh.slice_f32(&dn_name, e * h * i, (e + 1) * h * i)?
        };
        dn.push(&Q4::quantize(&dn_slice, h, i, group, threads));
    }
    // Shrink to fit: release memory wasted by Vec's doubling growth strategy.
    // This is automatic memory management so the user never needs to clean up RAM by hand.
    gu.shrink_to_fit();
    dn.shrink_to_fit();
    Ok((gu, dn))
}

/// Load one expert's gate-up and down as raw f32, handling all three checkpoint
/// formats (fused, separate, Edge0 switch_mlp). Returns `(gate_up, down)` where
/// `gate_up` has `2*i*h` elements and `down` has `h*i` elements.
pub fn load_expert_pair(
    sh: &Shards,
    prefix: &str,
    layer: usize,
    id: usize,
    h: usize,
    i: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let p = format!("{prefix}.layers.{layer}.mlp.experts");
    let gu_name = format!("{p}.gate_up_proj");
    let dn_name = format!("{p}.down_proj");
    if sh.contains(&gu_name) {
        let gu = sh.slice_f32(&gu_name, id * 2 * i * h, (id + 1) * 2 * i * h)?;
        let dn = sh.slice_f32(&dn_name, id * h * i, (id + 1) * h * i)?;
        return Ok((gu, dn));
    }
    let p2 = format!("{prefix}.layers.{layer}.mlp.switch_mlp");
    let edge_gate = format!("{p2}.gate_proj.weight");
    let edge_up = format!("{p2}.up_proj.weight");
    let edge_dn = format!("{p2}.down_proj.weight");
    if sh.contains(&edge_gate) {
        let gate = sh.slice_f32(&edge_gate, id * i * h, (id + 1) * i * h)?;
        let up = sh.slice_f32(&edge_up, id * i * h, (id + 1) * i * h)?;
        let dn = sh.slice_f32(&edge_dn, id * h * i, (id + 1) * h * i)?;
        let mut gu = gate;
        gu.extend_from_slice(&up);
        return Ok((gu, dn));
    }
    let gate_name = format!("{p}.gate_proj");
    let up_name = format!("{p}.up_proj");
    let gate_w = format!("{gate_name}.weight");
    let up_w = format!("{up_name}.weight");
    let gn = if sh.contains(&gate_w) { &gate_w } else { &gate_name };
    let un = if sh.contains(&up_w) { &up_w } else { &up_name };
    let gate = sh.slice_f32(gn, id * i * h, (id + 1) * i * h)?;
    let up = sh.slice_f32(un, id * i * h, (id + 1) * i * h)?;
    let dn = sh.slice_f32(&dn_name, id * h * i, (id + 1) * h * i)?;
    let mut gu = gate;
    gu.extend_from_slice(&up);
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
    pub fn load(sh: &Shards, cfg: Arc<TextConfig>, layer: usize, threads: usize, quantize_i8: bool) -> Result<MlpLayer> {
        let p = format!("{}.layers.{layer}.mlp", cfg.prefix);
        let mut l = MlpLayer {
            gate: dense_half(sh, &format!("{p}.gate_proj.weight"))?.0,
            up: dense_half(sh, &format!("{p}.up_proj.weight"))?.0,
            down: dense_half(sh, &format!("{p}.down_proj.weight"))?.0,
            cfg,
            threads: threads.max(1),
        };
        if quantize_i8 && dense_i8() {
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

    /// A chunk of prompt tokens through the same three projections, each read
    /// once for the whole chunk. Same products in the same order as `n` calls
    /// of [`MlpLayer::forward`], token-major in and out.
    pub fn prefill(&self, xs: &[f32], out: &mut [f32]) {
        let (h, i) = (self.cfg.hidden_size, self.cfg.intermediate_size);
        let t = self.threads;
        let n = xs.len() / h;
        let _p = prof::start(prof::MIXER_PROJ);
        let mut g = vec![0f32; n * i];
        let mut u = vec![0f32; n * i];
        self.gate.gemv_batch(i, h, xs, &mut g, t);
        self.up.gemv_batch(i, h, xs, &mut u, t);
        for j in 0..u.len() {
            u[j] = kernel::silu(g[j]) * u[j];
        }
        self.down.gemv_batch(h, i, &u, out, t);
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
    pub fn load(sh: &Shards, cfg: Arc<TextConfig>, layer: usize, threads: usize, quantize_i8: bool) -> Result<GdnLayer> {
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
        if quantize_i8 && dense_i8() {
            // The five projections are this tier's dense read; at one byte per
            // weight they cost half the bytes. Skip when attach_dense will
            // re-read from shards and upload to device (#82 memory pressure).
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
        debug_assert!(nk > 0, "linear_num_key_heads must be > 0 for GDN layers");
        let mut g = d.gpu.lock().unwrap();
        if !st.mirrored {
            let _u = prof::start(prof::MIXER_DEV_UPLOAD);
            g.write_buf(m.conv_state, 0, &f32_bytes(&st.conv)).unwrap();
            g.write_buf(m.rec_state, 0, &f32_bytes(&st.rec)).unwrap();
            st.mirrored = true;
        }
        let ib = DENSE_OFF + h;
        // CUDA's half gemv reads x as v4.f32, so the offset must be 16-byte
        // aligned; Vulkan reads scalars and would not notice an odd nv.
        let normed = (ib + cd + vd + 2 * nv + 3) & !3;
        {
            let _u = prof::start(prof::MIXER_DEV_UPLOAD);
            g.write(DENSE_OFF, x);
        }
        let read_syncs = g.read_syncs();
        {
            let _c = prof::start(prof::MIXER_DEV_COMPUTE);
            // Also bill this batch on the device's own clock, which separates
            // GPU execution from the host's wait (no-op on Vulkan).
            g.time_mixer_batch();
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
            //
            // Every argument in this queue - the seven weight and state buffers,
            // the arena offsets, the shapes - belongs to the layer and not to the
            // token, so CUDA replays it as one captured graph instead of paying
            // seven cuLaunchKernel calls per layer per token.
            if read_syncs { bt.submit_replay().unwrap(); } else { bt.run().unwrap(); }
        }
        {
            let _r = prof::start(prof::MIXER_DEV_READBACK);
            g.read_into(DENSE_OFF, out);
        }
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
        {
            let _u = prof::start(prof::MIXER_DEV_UPLOAD);
            g.write(DENSE_OFF, x);
        }
        {
            let _c = prof::start(prof::MIXER_DEV_COMPUTE);
            // Also bill this batch on the device's own clock, which separates
            // GPU execution from the host's wait (no-op on Vulkan).
            g.time_mixer_batch();
            let mut bt = g.batch().unwrap();
            bt.gemv_half_indep(d.qkv, cd, h, DENSE_OFF, DENSE_OFF + h).unwrap();
            bt.gemv_half_indep(d.z, vd, h, DENSE_OFF, DENSE_OFF + h + cd).unwrap();
            bt.gemv_half_indep(d.b, nv, h, DENSE_OFF, DENSE_OFF + h + cd + vd).unwrap();
            bt.gemv_half_indep(d.a, nv, h, DENSE_OFF, DENSE_OFF + h + cd + vd + nv).unwrap();
            bt.barrier().unwrap();
            bt.run().unwrap();
        }
        {
            let _r = prof::start(prof::MIXER_DEV_READBACK);
            g.read_into(DENSE_OFF + h, buf);
        }
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
        let vd = cfg.value_dim();
        let nv = cfg.linear_num_value_heads;
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
        let mut normed = vec![0f32; vd];
        self.mix(mixed, z, b, a, st, &mut normed);
        drop(_cpu);
        // out_proj is a projection, not part of the recursion: the mixer timer
        // closed above so the two stages stay disjoint (a nested timer would be
        // counted twice, which showed up as negative unaccounted stack time).
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
    }

    /// A chunk of prompt tokens through the mixer: the four input projections
    /// and `out_proj` each read their weights once for the whole chunk, and the
    /// recursion runs token by token because the conv window and the delta state
    /// are a chain. Host tier only - with a device the projections live in VRAM
    /// and `step` already batches them into one command buffer.
    fn prefill(&self, xs: &[f32], st: &mut GdnState, out: &mut [f32]) {
        debug_assert!(self.dev.is_none());
        let cfg = &self.cfg;
        let (h, cd, vd, nv) = (
            cfg.hidden_size,
            cfg.conv_dim(),
            cfg.value_dim(),
            cfg.linear_num_value_heads,
        );
        let t = self.threads;
        let n = xs.len() / h;
        let mut qkv = vec![0f32; n * cd];
        let mut zb = vec![0f32; n * vd];
        let mut bb = vec![0f32; n * nv];
        let mut ab = vec![0f32; n * nv];
        {
            let _p = prof::start(prof::MIXER_PROJ);
            self.in_qkv.gemv_batch(cd, h, xs, &mut qkv, t);
            self.in_z.gemv_batch(vd, h, xs, &mut zb, t);
            self.in_b.gemv_batch(nv, h, xs, &mut bb, t);
            self.in_a.gemv_batch(nv, h, xs, &mut ab, t);
        }
        let _cpu = prof::start(prof::MIXER_CPU);
        let mut normed = vec![0f32; n * vd];
        for i in 0..n {
            self.mix(
                &qkv[i * cd..(i + 1) * cd],
                &zb[i * vd..(i + 1) * vd],
                &bb[i * nv..(i + 1) * nv],
                &ab[i * nv..(i + 1) * nv],
                st,
                &mut normed[i * vd..(i + 1) * vd],
            );
        }
        drop(_cpu);
        let _p = prof::start(prof::MIXER_PROJ);
        self.out_proj.gemv_batch(h, vd, &normed, out, t);
    }

    /// Device-batched prefill: batch the 5 dense projections on GPU, loop the
    /// stateful conv+rec per token. This is the device tier's answer to
    /// [`GdnLayer::prefill`], which batches on CPU. The projections read their
    /// weights once for all n tokens from VRAM, amortizing the VRAM bandwidth.
    ///
    /// Buffer layout in scratch (offsets in floats):
    /// - xs_off..xs_off+n*h: input tokens
    /// - ib_off..ib_off+n*(cd+vd+2*nv): intermediate projections (qkv|z|b|a)
    /// - normed_off..normed_off+n*vd: post-conv/rec normed output
    /// - out_off..out_off+n*h: final output
    /// A chunk of `n` tokens through the delta net: the four input projections
    /// go to the device as one batch each, the conv and the recurrence stay per
    /// token because they carry state, and the gated norm's read-out goes back
    /// for a batched `o_proj`. `aux` is the chunk's mixer region; the span was
    /// reserved by `Model::prefill_device`, so nothing here reserves again.
    pub fn prefill_device(&self, gpu: &mut Gpu, xs_off: usize, aux: usize, st: &mut GdnState, out_off: usize, n: usize) -> Result<()> {
        let Some(d) = &self.dev else {
            // No device tier, fall back to CPU prefill
            let mut xs = vec![0f32; n * self.cfg.hidden_size];
            gpu.read_into(xs_off, &mut xs);
            let mut out = vec![0f32; n * self.cfg.hidden_size];
            self.prefill(&xs, st, &mut out);
            gpu.write(out_off, &out);
            return Ok(());
        };

        let cfg = &self.cfg;
        let (h, cd, vd, nv) = (cfg.hidden_size, cfg.conv_dim(), cfg.value_dim(), cfg.linear_num_value_heads);
        let ib_off = aux;
        let normed_off = ib_off + n * (cd + vd + 2 * nv);

        // The chunk's recursion runs on the host copy of the state, so anything
        // the decode left in VRAM has to come back before the first token - and
        // the host copy owns the result afterwards, which is what `mirrored =
        // false` tells the next decode step. `sync_state` reads the same two
        // buffers but takes the guard this call is already running under.
        if let Some(m) = &d.mixer {
            if st.mirrored {
                let mut cb = vec![0u8; st.conv.len() * 4];
                gpu.read_buf(m.conv_state, &mut cb)?;
                let mut rb = vec![0u8; st.rec.len() * 4];
                gpu.read_buf(m.rec_state, &mut rb)?;
                bytes_into_f32(&cb, &mut st.conv);
                bytes_into_f32(&rb, &mut st.rec);
                st.mirrored = false;
            }
        }

        // Batch the 4 input projections: xs -> [qkv|z|b|a]
        {
            let _p = prof::start(prof::MIXER_PROJ);
            let mut bt = gpu.batch()?;
            bt.gemv_half_batch(d.qkv, cd, h, n, xs_off, ib_off)?;
            bt.gemv_half_batch(d.z, vd, h, n, xs_off, ib_off + n * cd)?;
            bt.gemv_half_batch(d.b, nv, h, n, xs_off, ib_off + n * (cd + vd))?;
            bt.gemv_half_batch(d.a, nv, h, n, xs_off, ib_off + n * (cd + vd + nv))?;
            bt.barrier()?;
            bt.run()?;
        }

        // Read the projections back. Each is its own token-major block in the
        // arena - the batch kernel writes `y[t * rows + r]`, not one row per
        // token holding all four - so they come out as four buffers, the same
        // shape the host `prefill` keeps them in.
        let mut qkv = vec![0f32; n * cd];
        let mut zb = vec![0f32; n * vd];
        let mut bb = vec![0f32; n * nv];
        let mut ab = vec![0f32; n * nv];
        gpu.read_into(ib_off, &mut qkv);
        gpu.read_into(ib_off + n * cd, &mut zb);
        gpu.read_into(ib_off + n * (cd + vd), &mut bb);
        gpu.read_into(ib_off + n * (cd + vd + nv), &mut ab);

        // Per-token conv + rec (stateful, cannot batch)
        let mut normed = vec![0f32; n * vd];
        {
            let _cpu = prof::start(prof::MIXER_CPU);
            for i in 0..n {
                self.mix(
                    &qkv[i * cd..(i + 1) * cd],
                    &zb[i * vd..(i + 1) * vd],
                    &bb[i * nv..(i + 1) * nv],
                    &ab[i * nv..(i + 1) * nv],
                    st,
                    &mut normed[i * vd..(i + 1) * vd],
                );
            }
        }

        // Upload normed for batched out_proj
        {
            gpu.write(normed_off, &normed);
            let mut bt = gpu.batch()?;
            bt.gemv_half_batch(d.out, h, vd, n, normed_off, out_off)?;
            bt.run()?;
        }

        Ok(())
    }

    /// One token's recursion: the depthwise conv and its rolling window, the
    /// delta update over the heads, and the gated norm whose read-out the
    /// caller's `out_proj` consumes. Everything that sits between the input
    /// projections and `out_proj`, so a prefill chunk can project all its tokens
    /// at once and then walk this one token at a time.
    fn mix(&self, mixed: &[f32], z: &[f32], b: &[f32], a: &[f32], st: &mut GdnState, normed: &mut [f32]) {
        let cfg = &self.cfg;
        let cd = cfg.conv_dim();
        let kd = cfg.key_dim();
        let vd = cfg.value_dim();
        let (nv, nk) = (cfg.linear_num_value_heads, cfg.linear_num_key_heads);
        let (dk, dv) = (cfg.linear_key_head_dim, cfg.linear_value_head_dim);
        let kernel_dim = cfg.linear_conv_kernel_dim;
        let t = self.threads;

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
        // gated RMSNorm per value head
        for hv in 0..nv {
            let (x, g) = (
                &o[hv * dv..(hv + 1) * dv],
                &z[hv * dv..(hv + 1) * dv],
            );
            kernel::rmsnorm_gated(x, g, &self.norm_w, cfg.rms_norm_eps as f32, &mut normed[hv * dv..(hv + 1) * dv]);
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
    pub fn load(sh: &Shards, cfg: Arc<TextConfig>, layer: usize, max_pos: usize, threads: usize, quantize_i8: bool) -> Result<AttnLayer> {
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
        if quantize_i8 && dense_i8() {
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
        {
            let _u = prof::start(prof::MIXER_DEV_UPLOAD);
            g.write(DENSE_OFF, x);
        }
        {
            let _c = prof::start(prof::MIXER_DEV_COMPUTE);
            // Also bill this batch on the device's own clock, which separates
            // GPU execution from the host's wait (no-op on Vulkan).
            g.time_mixer_batch();
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
        {
            let _r = prof::start(prof::MIXER_DEV_READBACK);
            g.read_into(DENSE_OFF + h, buf);
        }
    }

    /// One decode token against the KV cache in `st`.
    pub fn step(&self, x: &[f32], pos: usize, st: &mut AttnState, out: &mut [f32]) {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let t = self.threads;
        let q_out = if cfg.attn_output_gate { nh * hd * 2 } else { nh * hd };

        // q_proj is [nh, head_dim*2] per head when the checkpoint gates the
        // output: query first, then the gate. A dense Qwen3 has no gate, so the
        // heads are packed one head_dim apart and reading them at 2*hd would
        // stride every other head into the following row.
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
        let mut attn = vec![0f32; nh * hd];
        self.mix(qraw, kraw, vv, pos, st, &mut attn);
        drop(_cpu);
        match &self.dev {
            Some(d) => {
                // o_proj's three rows need this parent too. `total` drops them on
                // the grounds that `mixer device` already bills them, so a device
                // block outside it was billed by neither: on the served CUDA cell
                // the parts summed to 11.85 ms/token over a 10.85 whole.
                let _t = prof::start(prof::MIXER_DEV);
                let mut g = d.gpu.lock().unwrap();
                {
                    let _u = prof::start(prof::MIXER_DEV_UPLOAD);
                    g.write(DENSE_OFF, &attn);
                }
                {
                    let _c = prof::start(prof::MIXER_DEV_COMPUTE);
                    g.time_mixer_batch();
                    let mut bt = g.batch().unwrap();
                    if d.q4 {
                        bt.gemv_q4(d.o, d.qs[3], h, nh * hd, d.group, DENSE_OFF, DENSE_OFF + nh * hd).unwrap();
                    } else {
                        bt.gemv_half(d.o, h, nh * hd, DENSE_OFF, DENSE_OFF + nh * hd).unwrap();
                    }
                    bt.run().unwrap();
                }
                {
                    let _r = prof::start(prof::MIXER_DEV_READBACK);
                    g.read_into(DENSE_OFF + nh * hd, out);
                }
            }
            None => {
                let _p = prof::start(prof::MIXER_PROJ);
                self.o.gemv(h, nh * hd, &attn, out, t)
            }
        }
    }

    /// A chunk of prompt tokens through the mixer. q, k, v and o_proj each read
    /// their weights once for the whole chunk; the cache writes and the scores
    /// stay token by token because a causal row can only see the rows before it.
    /// Host tier only - with a device the projections live in VRAM.
    fn prefill(&self, xs: &[f32], pos0: usize, st: &mut AttnState, out: &mut [f32]) {
        debug_assert!(self.dev.is_none());
        let cfg = &self.cfg;
        let (h, nh, nkv, hd) = (
            cfg.hidden_size,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
        );
        let q_out = if cfg.attn_output_gate { nh * hd * 2 } else { nh * hd };
        let t = self.threads;
        let n = xs.len() / h;
        let mut qb = vec![0f32; n * q_out];
        let mut kb = vec![0f32; n * nkv * hd];
        let mut vb = vec![0f32; n * nkv * hd];
        {
            let _p = prof::start(prof::MIXER_PROJ);
            self.q.gemv_batch(q_out, h, xs, &mut qb, t);
            self.k.gemv_batch(nkv * hd, h, xs, &mut kb, t);
            self.v.gemv_batch(nkv * hd, h, xs, &mut vb, t);
        }
        let _cpu = prof::start(prof::MIXER_CPU);
        let mut attn = vec![0f32; n * nh * hd];
        for i in 0..n {
            self.mix(
                &qb[i * q_out..(i + 1) * q_out],
                &kb[i * nkv * hd..(i + 1) * nkv * hd],
                &vb[i * nkv * hd..(i + 1) * nkv * hd],
                pos0 + i,
                st,
                &mut attn[i * nh * hd..(i + 1) * nh * hd],
            );
        }
        drop(_cpu);
        let _p = prof::start(prof::MIXER_PROJ);
        self.o.gemv_batch(h, nh * hd, &attn, out, t);
    }

    /// Device-batched prefill: batch q/k/v/o projections on GPU, loop attention
    /// mix per token (KV cache dependencies). Similar pattern to GdnLayer::prefill_device.
    /// A chunk of `n` tokens through attention: q/k/v projected in one batch,
    /// the mix per token because each row depends on the KV cache it just
    /// wrote, then `o_proj` batched over the read-out. `aux` is the chunk's
    /// mixer region, reserved by `Model::prefill_device`.
    pub fn prefill_device(&self, gpu: &mut Gpu, xs_off: usize, aux: usize, pos0: usize, st: &mut AttnState, out_off: usize, n: usize) -> Result<()> {
        let Some(d) = &self.dev else {
            // No device tier, fall back to CPU prefill
            let mut xs = vec![0f32; n * self.cfg.hidden_size];
            gpu.read_into(xs_off, &mut xs);
            let mut out = vec![0f32; n * self.cfg.hidden_size];
            self.prefill(&xs, pos0, st, &mut out);
            gpu.write(out_off, &out);
            return Ok(());
        };

        let cfg = &self.cfg;
        let (h, nh, nkv, hd) = (cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim);
        let q_out = if cfg.attn_output_gate { nh * hd * 2 } else { nh * hd };

        // Buffer layout within the chunk's mixer region: q | k | v | attn
        let q_off = aux;
        let k_off = q_off + n * q_out;
        let v_off = k_off + n * nkv * hd;
        let attn_off = v_off + n * nkv * hd;

        {
            let _p = prof::start(prof::MIXER_PROJ);
            let mut bt = gpu.batch()?;
            bt.gemv_half_batch(d.q, q_out, h, n, xs_off, q_off)?;
            bt.gemv_half_batch(d.k, nkv * hd, h, n, xs_off, k_off)?;
            bt.gemv_half_batch(d.v, nkv * hd, h, n, xs_off, v_off)?;
            bt.barrier()?;
            bt.run()?;
        }

        let mut qb = vec![0f32; n * q_out];
        let mut kb = vec![0f32; n * nkv * hd];
        let mut vb = vec![0f32; n * nkv * hd];
        gpu.read_into(q_off, &mut qb);
        gpu.read_into(k_off, &mut kb);
        gpu.read_into(v_off, &mut vb);

        // Per-token attention mix (KV cache dependencies)
        let mut attn = vec![0f32; n * nh * hd];
        {
            let _cpu = prof::start(prof::MIXER_CPU);
            for i in 0..n {
                self.mix(
                    &qb[i * q_out..(i + 1) * q_out],
                    &kb[i * nkv * hd..(i + 1) * nkv * hd],
                    &vb[i * nkv * hd..(i + 1) * nkv * hd],
                    pos0 + i,
                    st,
                    &mut attn[i * nh * hd..(i + 1) * nh * hd],
                );
            }
        }

        {
            gpu.write(attn_off, &attn);
            let mut bt = gpu.batch()?;
            bt.gemv_half_batch(d.o, h, nh * hd, n, attn_off, out_off)?;
            bt.run()?;
        }

        Ok(())
    }

    /// One token's attention against the KV cache: rope, the cache row, the
    /// causal scores and the output gate - everything between the q/k/v
    /// projections and `o_proj`, so a prefill chunk can project all its tokens
    /// at once and then walk this one token at a time.
    fn mix(
        &self,
        qraw: &[f32],
        kraw: &[f32],
        vv: &[f32],
        pos: usize,
        st: &mut AttnState,
        attn: &mut [f32],
    ) {
        let cfg = &self.cfg;
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        debug_assert!(nkv > 0, "num_key_value_heads must be > 0 for attention layers");
        debug_assert!(nh >= nkv, "num_attention_heads must be >= num_key_value_heads");
        let rot = cfg.rotary_dim();
        let eps = cfg.rms_norm_eps as f32;
        let stride = if cfg.attn_output_gate { hd * 2 } else { hd };

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
        Layer::with_experts(sh, cfg, layer, &all, None, group, threads, kv_cap, false)
    }

    /// `ids` resident in slot order, the rest answered from `stream`. A plan
    /// that lists its experts hottest-first makes slot 0 the cheapest miss.
    /// `dense_on_device` skips the int8 quantisation when the dense projections
    /// will be re-read from shards and uploaded to VRAM by attach_dense (#82).
    pub fn with_experts(
        sh: &Arc<Shards>,
        cfg: Arc<TextConfig>,
        layer: usize,
        ids: &[usize],
        stream: Option<Stream>,
        group: usize,
        threads: usize,
        kv_cap: usize,
        dense_on_device: bool,
    ) -> Result<Layer> {
        let kind = cfg.layer_types[layer];
        let p = format!("{}.layers.{layer}", cfg.prefix);
        let zc = cfg.norm_zero_centered;
        let quantize_i8 = !dense_on_device;
        let (moe, mlp) = if cfg.is_moe() {
            (
                Some(MoeLayer::load_set(sh, cfg.clone(), layer, ids, stream, group, threads)?),
                None,
            )
        } else {
            let t = std::time::Instant::now();
            let mlp = MlpLayer::load(sh, cfg.clone(), layer, threads, quantize_i8)?;
            DENSE_LOAD_NS.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
            (None, Some(mlp))
        };
        let t_dense = std::time::Instant::now();
        let gdn = match kind {
            LayerKind::Linear => Some(GdnLayer::load(sh, cfg.clone(), layer, threads, quantize_i8)?),
            LayerKind::Full => None,
        };
        let attn = match kind {
            LayerKind::Full => Some(AttnLayer::load(sh, cfg.clone(), layer, kv_cap, threads, quantize_i8)?),
            LayerKind::Linear => None,
        };
        let norm_in = norm_w(sh, &format!("{p}.input_layernorm.weight"), zc)?;
        let norm_post = norm_w(sh, &format!("{p}.post_attention_layernorm.weight"), zc)?;
        DENSE_LOAD_NS.fetch_add(t_dense.elapsed().as_nanos() as u64, Relaxed);
        Ok(Layer { cfg, kind, norm_in, norm_post, moe, mlp, gdn, attn })
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
        match (&self.moe, &self.mlp) {
            (Some(moe), _) => {
                let (moe_out, _) = moe.forward(&buf);
                let _t = prof::start(prof::RESIDUAL);
                kernel::axpy(x, &moe_out, 1.0);
            }
            (None, mlp) => {
                mlp.as_ref().unwrap().forward(&buf, &mut res);
                let _t = prof::start(prof::RESIDUAL);
                kernel::axpy(x, &res, 1.0);
            }
        }
    }

    /// One chunk of `n` prompt tokens through this layer: batched RMSNorm and
    /// residual add around the mixer, and the routed experts grouped over the
    /// chunk. Every offset comes from `c`, the chunk's own region of the arena,
    /// reserved by `Model::prefill_device` before the layer loop.
    pub fn prefill_device(&self, gpu: &mut Gpu, c: Chunk, st: &mut LayerState, pos0: usize) -> Result<()> {
        let (h, n, eps) = (c.h, c.n, self.cfg.rms_norm_eps as f32);
        // The batched norm kernel reads its gains out of the arena, so copy
        // them in: two 8 KiB memcpys per layer per chunk against the dispatches
        // they feed. `upload` was wrong here twice over - it returns a buffer id
        // where the kernel wants a float offset, and `new_buf` never reuses a
        // slot, so this leaked two device buffers per layer per chunk.
        gpu.write(c.w_in, &self.norm_in);
        gpu.write(c.w_post, &self.norm_post);

        {
            let _t = prof::start(prof::NORM);
            let mut bt = gpu.batch()?;
            bt.rmsnorm_batch(h, n, c.xs, c.w_in, c.buf, eps)?;
            bt.run()?;
        }

        match (&self.kind, st) {
            (LayerKind::Linear, LayerState::Linear(g)) => {
                self.gdn.as_ref().unwrap().prefill_device(gpu, c.buf, c.mix, g, c.res, n)?;
            }
            (LayerKind::Full, LayerState::Full(a)) => {
                self.attn.as_ref().unwrap().prefill_device(gpu, c.buf, c.mix, pos0, a, c.res, n)?;
            }
            _ => panic!("layer state mismatch"),
        }

        {
            let _t = prof::start(prof::RESIDUAL);
            let mut bt = gpu.batch()?;
            bt.axpy_batch(h, n, c.res, c.xs, 1.0, false)?;
            bt.run()?;
        }

        {
            let _t = prof::start(prof::NORM);
            let mut bt = gpu.batch()?;
            bt.rmsnorm_batch(h, n, c.xs, c.w_post, c.buf, eps)?;
            bt.run()?;
        }

        match (&self.moe, &self.mlp) {
            (Some(moe), _) => {
                let mut all = vec![0f32; n * h];
                gpu.read_into(c.buf, &mut all);
                let moe_out = moe.prefill_device_grouped(gpu, &all, n);
                gpu.write(c.res, &moe_out);
                let mut bt = gpu.batch()?;
                bt.axpy_batch(h, n, c.res, c.xs, 1.0, false)?;
                bt.run()?;
            }
            (None, mlp) => {
                let mlp = mlp.as_ref().unwrap();
                let mut rows = vec![0f32; n * h];
                gpu.read_into(c.buf, &mut rows);
                let mut res = vec![0f32; n * h];
                for i in 0..n {
                    mlp.forward(&rows[i * h..(i + 1) * h], &mut res[i * h..(i + 1) * h]);
                }
                gpu.write(c.res, &res);
                let mut bt = gpu.batch()?;
                bt.axpy_batch(h, n, c.res, c.xs, 1.0, false)?;
                bt.run()?;
            }
        }

        Ok(())
    }

    /// A chunk of prompt tokens through this layer: `xs` holds `n` token-major
    /// rows of `h` floats and is advanced in place. The mixer and a dense FFN
    /// read their weights once for the whole chunk; the routed experts read each
    /// expert they need once too, but only for the experts this chunk actually
    /// picks — a token's eight of 256 are not the next token's eight.
    /// Bit-identical to `n` calls of [`Layer::forward`] at consecutive positions.
    pub fn prefill(&self, xs: &mut [f32], st: &mut LayerState, pos0: usize) {
        let h = self.cfg.hidden_size;
        let n = xs.len() / h;
        let eps = self.cfg.rms_norm_eps as f32;
        let mut buf = vec![0f32; n * h];
        let mut res = vec![0f32; n * h];
        {
            let _t = prof::start(prof::NORM);
            for i in 0..n {
                kernel::rmsnorm(&xs[i * h..(i + 1) * h], &self.norm_in, eps, &mut buf[i * h..(i + 1) * h]);
            }
        }
        match (&self.kind, st) {
            (LayerKind::Linear, LayerState::Linear(g)) => self.gdn.as_ref().unwrap().prefill(&buf, g, &mut res),
            (LayerKind::Full, LayerState::Full(a)) => self.attn.as_ref().unwrap().prefill(&buf, pos0, a, &mut res),
            _ => panic!("layer state mismatch"),
        }
        {
            let _t = prof::start(prof::RESIDUAL);
            kernel::axpy(xs, &res, 1.0);
        }
        {
            let _t = prof::start(prof::NORM);
            for i in 0..n {
                kernel::rmsnorm(&xs[i * h..(i + 1) * h], &self.norm_post, eps, &mut buf[i * h..(i + 1) * h]);
            }
        }
        match (&self.moe, &self.mlp) {
            (Some(moe), _) => {
                // Grouped by expert when the chunk holds more than one token;
                // otherwise, or when a pick cannot be grouped, one token at a
                // time. `res` is reused as the expert output: every row of it is
                // written before it is read back.
                if moe.prefill(&buf, &mut res) {
                    let _t = prof::start(prof::RESIDUAL);
                    kernel::axpy(xs, &res, 1.0);
                } else {
                    for i in 0..n {
                        let (moe_out, _) = moe.forward(&buf[i * h..(i + 1) * h]);
                        let _t = prof::start(prof::RESIDUAL);
                        kernel::axpy(&mut xs[i * h..(i + 1) * h], &moe_out, 1.0);
                    }
                }
            }
            (None, mlp) => {
                mlp.as_ref().unwrap().prefill(&buf, &mut res);
                let _t = prof::start(prof::RESIDUAL);
                kernel::axpy(xs, &res, 1.0);
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
    /// The lm_head quantised to int8 for the CPU lane when `--dense-i8` is on
    /// (the default). Colibri ships its head int8 too; reading the checkpoint's
    /// two-byte rows instead costs twice the bandwidth per token. `None` keeps
    /// the mmap half-precision path (GPU tier, ORT, or `--no-dense-i8`).
    pub head_i8: Option<Dense>,
    /// The lm_head parked in an ONNX Runtime session when `--backend ort` is
    /// selected. `None` keeps the device or blocked-CPU head path.
    #[cfg(feature = "ort")]
    pub ort_head: Option<crate::ort::OrtDense>,
    /// Background prefetcher for cold experts. When present, layers check its
    /// cache before hitting the disk pool, and hint it with the next layer's
    /// cold experts after each forward pass.
    pub prefetcher: Option<Arc<crate::prefetch::Prefetcher>>,
    /// Where a run folds the router picks it observed. The plan that hands out
    /// the RAM slots is ranked by a histogram of exactly those picks, so a model
    /// that has never been run has no ranking and places itself blind. `None`
    /// means this process records nothing: a dense stack, or a caller that asked
    /// for no history.
    pub usage_store: Option<String>,
    /// Where a run writes the rates it measured while running, beside the usage
    /// file and under the same rule: Fennec's own, never anything Colibri reads.
    pub rates_store: Option<String>,
    /// Whether the streamed tier borrowed its packs for this plan, which is what
    /// decides whether the recorded disk rate carries an encode pass with it.
    pub rates_borrows: bool,
    /// The lane traffic the last rates sample was taken from, so that a sample is
    /// a window rather than the process lifetime re-read. `serve` saves after
    /// every turn and the lane counters run for the whole process, so without
    /// this mark the samples it merges are nested windows of one run - turn 1
    /// inside turn 2 inside turn 3 - and `Measured::merge`, which keeps the
    /// coldest disk sample, mechanically chooses the earliest one. r16's narrow
    /// arms show it: six saves gave disk 4.0, then 3.7, then 3.7 four times over,
    /// while the same lane's cumulative read at end of run was 6.5 GB/s. Turn 1,
    /// whose window is nothing but first-touch page faults, was pricing every
    /// turn after it. Zero for a run that saves once, which is every decode run.
    rates_seen: Mutex<LaneWindow>,
}

/// Traffic the timed lanes have moved since the process started, which is what
/// they report cumulatively: see `Model::rates_seen` for why a sample has to be
/// the difference between two readings of it.
#[derive(Debug, Clone, Copy, Default)]
struct LaneWindow {
    read_bytes: u64,
    read_ms: f64,
    quant_bytes: u64,
    quant_ms: f64,
    host_bytes: u64,
    host_ns: u64,
    union_bytes: u64,
    union_ns: u64,
}

impl LaneWindow {
    /// What moved between an earlier reading of the counters and this one.
    fn since(&self, prev: &LaneWindow) -> LaneWindow {
        LaneWindow {
            read_bytes: self.read_bytes.saturating_sub(prev.read_bytes),
            read_ms: self.read_ms - prev.read_ms,
            quant_bytes: self.quant_bytes.saturating_sub(prev.quant_bytes),
            quant_ms: self.quant_ms - prev.quant_ms,
            host_bytes: self.host_bytes.saturating_sub(prev.host_bytes),
            host_ns: self.host_ns.saturating_sub(prev.host_ns),
            union_bytes: self.union_bytes.saturating_sub(prev.union_bytes),
            union_ns: self.union_ns.saturating_sub(prev.union_ns),
        }
    }
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
        EXPERT_READ_NS.store(0, Relaxed);
        EXPERT_SET_NS.store(0, Relaxed);
        DENSE_LOAD_NS.store(0, Relaxed);
        // One layer's host half is a burst of independent file reads plus its own
        // quantise pass, and the worker pool is not what either of them waits on: the
        // dense pass summed to 7.5 s with the pool at 16 threads and 7.7 s with it at
        // 1. The layers themselves are independent, so building them on lanes of
        // their own does overlap them — and the shared pool stays out of it, since
        // `par_rows` runs inline at threads <= 1 and a lane must not queue work into
        // the pool another lane may already be holding.
        //
        // Only for the no-device plan, though, and that half is measurement rather
        // than mechanism: paired on one binary, the device plan took host pool
        // 2.6 -> 1.3 s from 2 lanes but 6.2 -> 9.6 s from the device upload that
        // follows, so connect went 9.5 -> 11.7 s (3, 4 and 6 lanes all land at
        // 11.4-12.3 s; the pair repeated to within 0.2 s). The two phases do not
        // even read the same expert sets, so nothing here explains why overlapping
        // the first starves the second — it only says not to. Without an upload
        // there is nothing to starve and the lanes are a clean win: connect
        // 10.9 -> 5.2 s, repeated as 10.8 -> 5.0 s.
        //
        // `FENNEC_LOAD_LANES=1` is the A/B control for both halves of that — it
        // selects the serial loop on either plan, same binary, same checkpoint.
        let cap = std::env::var("FENNEC_LOAD_LANES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|n| *n > 0)
            .unwrap_or(6);
        let has_gpu = gpu.is_some();
        let lanes = if has_gpu || plan.layers.len() < 2 {
            1
        } else {
            // Each in-flight layer is another working set, and RAM is the scarce
            // resource here rather than threads, so no more lanes than the box has
            // usable cores.
            threads.clamp(1, cap).min(plan.layers.len())
        };
        let inner = if lanes > 1 { 1 } else { threads };
        let build = |i: usize| -> Result<Layer> {
            let lp = &plan.layers[i];
            let ids: Vec<usize> = if has_gpu {
                // GPU path: the host pool is the RAM tier; the resident set is in VRAM
                lp.pooled.clone()
            } else {
                // CPU-only path: the device tier has no VRAM to live in, so the
                // host reads it out of RAM like the pool
                lp.resident.iter().chain(&lp.pooled).copied().collect()
            };
            let stream = if lp.streamed.is_empty() || stream_slots == 0 {
                None
            } else {
                // `threads`, not `inner`: this one is what the ring refiller reads
                // with at decode time, and the lane is long gone by then.
                Some(Stream::new(sh.clone(), &cfg, lp.layer, group, threads, stream_slots))
            };
            Layer::with_experts(
                sh,
                cfg.clone(),
                lp.layer,
                &ids,
                stream,
                group,
                inner,
                kv_cap,
                lp.dense_on_device,
            )
        };
        let t_host = std::time::Instant::now();
        if lanes == 1 {
            for i in 0..plan.layers.len() {
                layers.push(build(i)?);
            }
        } else {
            // `Model::finish` and the prefetcher's next-layer handoff both index by
            // position in the plan, so the lanes fill slots and the order is
            // restored here rather than by whatever lane finished first.
            let per = plan.layers.len().div_ceil(lanes);
            let mut slots: Vec<Option<Result<Layer>>> =
                (0..plan.layers.len()).map(|_| None).collect();
            let build = &build;
            std::thread::scope(|s| {
                for (c, chunk) in slots.chunks_mut(per).enumerate() {
                    let base = c * per;
                    s.spawn(move || {
                        for (j, slot) in chunk.iter_mut().enumerate() {
                            *slot = Some(build(base + j));
                        }
                    });
                }
            });
            layers = slots
                .into_iter()
                .map(|s| s.unwrap())
                .collect::<Result<Vec<Layer>>>()?;
        }
        // The build lanes are joined, so the worker pool has one caller again and the
        // layers may use all of it. Leaving them at `inner` decodes every token on one
        // core, which is what cost the served CPU lane most of its throughput (#463).
        for layer in layers.iter_mut() {
            layer.set_threads(threads);
        }
        let host_ms = t_host.elapsed().as_secs_f64() * 1e3;
        crate::mem::report("host pool");
        let first = plan.layers.first().map(|lp| lp.layer).unwrap_or(0);
        let t_finish = std::time::Instant::now();
        let mut model = Model::finish(sh, cfg.clone(), first, layers, threads)?;
        // Start the prefetcher when any layer has streamed (cold) experts. Its job
        // is to have the *next* layer's picks in RAM while this one computes, so
        // each layer is handed the prefix of the next layer's streamed list — a
        // list that already arrives ranked hottest-first from the usage histogram.
        let has_streamed = plan.layers.iter().any(|lp| !lp.streamed.is_empty());
        if has_streamed && stream_slots > 0 {
            let mb = std::env::var("FENNEC_PREFETCH_MB")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(256)
                .max(16);
            let cache_bytes = mb * 1024 * 1024;
            // Speculation has one competitor: the OS file cache for the very same
            // bytes. Measured on a box with room to cache the whole expert matrix
            // (MATRIX.md, r5 sweep), an engine-side cache cost 2 % of decode at
            // depth 4 and 18.5 % at depth 16 — it displaced file pages that were
            // already serving the streamed tier, and the histogram guessed which
            // experts to hoist right only 3.6-13.4 % of the time. So speculate
            // only when the streamed matrix cannot stay cached, which is where a
            // read is genuinely cold. `FENNEC_PREFETCH_AHEAD` forces either way.
            let streamed: usize = plan
                .layers
                .iter()
                .map(|lp| lp.streamed.len() * plan.expert_bytes)
                .sum();
            let ram = crate::mem::available_bytes() as usize;
            let fits = streamed <= ram;
            let depth = std::env::var("FENNEC_PREFETCH_AHEAD")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(if fits { 0 } else { cfg.num_experts_per_tok });
            if depth == 0 {
                eprintln!(
                    "[fennec] prefetch: off — {:.1} GiB of streamed experts fits the {:.1} GiB \
                     the OS can cache, and a speculated copy of a cached page is pure overhead",
                    streamed as f64 / 2f64.powi(30),
                    ram as f64 / 2f64.powi(30)
                );
            } else {
                let pf = Arc::new(crate::prefetch::Prefetcher::start(
                    cache_bytes,
                    sh.clone(),
                    cfg.clone(),
                    group,
                    threads,
                ));
                model.prefetcher = Some(pf.clone());
                // Share the prefetcher with every layer's MoE and hand each one the
                // layer computed after it. `plan.layers` is built in layer order, so
                // the next entry is the next layer; the last wraps to the first
                // because that is where the following token starts.
                let n = plan.layers.len();
                for (idx, layer) in model.layers.iter_mut().enumerate() {
                    if let Some(moe) = &mut layer.moe {
                        moe.prefetcher = Some(pf.clone());
                        if let Some(np) = plan.layers.get((idx + 1) % n) {
                            moe.ahead_layer = np.layer;
                            moe.ahead = np.streamed.iter().take(depth).copied().collect();
                        }
                    }
                }
                eprintln!(
                    "[fennec] prefetcher: {} MiB cache, {depth} experts of the next layer per hint",
                    cache_bytes / 1024 / 1024
                );
            }
        }
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
        } else if dense_i8() {
            // CPU lane: quantise the head to int8 like the rest of the dense
            // trunk (and like Colibri). The mmap half path reads 2 bytes per
            // weight every token; int8 reads 1, which is the difference between
            // a bandwidth-bound head and a worse one. The quality trade is the
            // same one `--dense-i8` already makes for the projections.
            let t = std::time::Instant::now();
            let rows = model.cfg.vocab_size;
            let cols = model.cfg.hidden_size;
            model.head_i8 = Some(Dense::load(sh, &model.head)?.quantize_i8(rows, cols));
            head_ms = t.elapsed().as_secs_f64() * 1e3;
            crate::mem::report("lm_head int8");
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
            let read = EXPERT_READ_NS.load(Relaxed);
            let set = EXPERT_SET_NS.load(Relaxed);
            let dense = DENSE_LOAD_NS.load(Relaxed);
            eprintln!(
                // The three components are work summed over the build lanes, so
                // only their total is comparable to the wall-clock host pool number.
                "[fennec] host pool inside ({lanes} lanes, work summed): expert read {:.1} s | assemble {:.1} s | dense {:.1} s | plan+stream {:.1} s",
                read as f64 / 1e9,
                set.saturating_sub(read) as f64 / 1e9,
                dense as f64 / 1e9,
                ((host_ms * 1e6) as u64).saturating_sub(set + dense) as f64 / 1e9
            );
        }
        if model.cfg.is_moe() {
            eprintln!(
                "[fennec] host experts: {:.2} GiB of matrix, {:.2} GiB held (file pages, not commit, where borrowed), {mapped}/{} layers borrowed from the mapping",
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
            head_i8: None,
            #[cfg(feature = "ort")]
            ort_head: None,
            prefetcher: None,
            usage_store: None,
            rates_store: None,
            rates_borrows: false,
            rates_seen: Mutex::new(LaneWindow::default()),
        })
    }

    /// Summed over every layer's ring: bytes and milliseconds the streamed tier
    /// really moved during this process's life. The plan's `disk_gbs` and
    /// `quant_gbs` are guesses until something like this says otherwise, and this
    /// costs no load time because it is the production path itself being measured.
    pub fn stream_totals(&self) -> crate::pool::StreamTotals {
        let mut t = crate::pool::StreamTotals::default();
        for l in &self.layers {
            if let Some(pool) = l.moe.as_ref().and_then(|m| m.stream.as_ref()) {
                let st = pool.lock().unwrap();
                t.hits += st.hits;
                t.misses += st.misses;
                t.warmed += st.warmed;
                t.borrowed += st.borrowed;
                t.ring_bytes += st.bytes();
                t.ring_bytes_full += st.slots() * st.pack_bytes();

                t.read_ms += st.read_ms;
                t.quant_ms += st.quant_ms;
                t.read_bytes += st.read_bytes;
                t.quant_bytes += st.quant_bytes;
                t.rings += 1;
            }
        }
        t
    }

    /// Bytes the RAM tier read on the decode path and the nanoseconds it spent
    /// reading them, over this process's life. One pack per pick, so bytes/ns is
    /// the rate a per-token placement decision is priced with.
    ///
    /// The batched-prefill union used to be folded in here, and that is why the
    /// rates file held 21.7 GB/s for a tier the served profiler timed at 4.3: the
    /// union's picks were counted in `host_hits` *and* its experts in
    /// `host_union_bytes`, so a 26-token chunk billed (picks + experts) x pack for
    /// the traffic of experts x pack, and `Measured::merge` keeps the warmest
    /// sample, which is always the chunk with the most reuse. Two paths that do
    /// different work per byte cannot share one price; the union reports on its own
    /// through `union_traffic`.
    pub fn host_traffic(&self) -> (u64, u64) {
        let (mut bytes, mut ns) = (0u64, 0u64);
        for l in &self.layers {
            if let Some(m) = &l.moe {
                let (b, n) = m.decode_traffic();
                bytes += b;
                ns += n;
            }
        }
        (bytes, ns)
    }

    /// The same pair for the batched-prefill union: one pack per expert the chunk
    /// touched, over the window that read it. Printed, not priced - the plan ranks
    /// tokens, not prompt chunks - and it is the number that says whether a cold
    /// prompt is being served out of the tier the prefiller expected.
    pub fn union_traffic(&self) -> (u64, u64) {
        let (mut bytes, mut ns) = (0u64, 0u64);
        for l in &self.layers {
            if let Some(m) = &l.moe {
                bytes += m.host_union_bytes.load(Relaxed);
                ns += m.union_ns.load(Relaxed);
            }
        }
        (bytes, ns)
    }

    /// What the RAM tier *was* when it moved those bytes: the workers one pick was
    /// split over, and the matrix bytes a pick reads. Both are rate-bearing, both
    /// have changed under a still-valid rates file, and neither can be recovered
    /// afterwards - #463 found the tier computing with `threads = 1` for the life
    /// of the process, and #225/#464 changed the pack from 1.77 MiB to 0.88 and
    /// back. `Measured::merge` drops a sample whose shape differs from the file's.
    ///
    /// The widest layer answers: every MoE layer gets the same count from
    /// `Layer::set_threads`, and a layer with no RAM tier contributes zero, so a
    /// max is how you read the one real value without assuming the stack is uniform.
    fn host_shape(&self) -> (Option<u32>, Option<u64>) {
        let (mut threads, mut pick) = (0usize, 0usize);
        for l in &self.layers {
            if let Some(m) = &l.moe {
                threads = threads.max(m.threads);
                pick = pick.max(m.host_pick_bytes());
            }
        }
        (
            (threads > 0).then(|| threads as u32),
            (pick > 0).then(|| pick as u64),
        )
    }

    /// Matrix bytes the RAM tier presents per byte it commits: around 9 when
    /// every layer borrows its experts out of the file mapping, 1.0 when none
    /// does. Printed, not priced - r13 measured that a wider tier costs both
    /// more commit and more time, so the budget stays in matrix bytes.
    fn host_held_ratio(&self) -> Option<f64> {
        let (mut matrix, mut held) = (0f64, 0f64);
        for l in &self.layers {
            if let Some(m) = &l.moe {
                matrix += m.host_bytes() as f64;
                held += m.host_bytes_held() as f64;
            }
        }
        (matrix > 0.0 && held > 0.0).then(|| matrix / held)
    }

    /// Bytes moved in a timed window, as decimal GB/s, when and only when the
    /// window was long enough to be a rate at all.
    ///
    /// The floor sits between two populations r15 measured on the served path with
    /// one binary. A placement whose RAM tier covered nearly every non-resident
    /// expert left the read timer holding 5.5-9.1 ms of cumulative traffic: five
    /// ring refills, 8.85 MB, across a 20 s run. The 0.81-1.44 GB/s `save_rates`
    /// persisted from those arms is the same 8.85 MB over 6.1-10.9 ms. Because
    /// `Measured::merge` keeps the coldest disk sample it is given, one of those
    /// first-fault readings would have priced this box's streamed tier for as long
    /// as the rates file lived. The arms that actually carried the tier read
    /// 3.38 GiB in 449-599 ms, for 6.1-8.1 GB/s. Twenty milliseconds is above
    /// every poison window and an order of magnitude below every genuine one,
    /// which at these rates is about 125 host picks or 84 refills - a seat between
    /// the populations, not a tuned constant.
    fn sustained_gbs(bytes: u64, ms: f64) -> Option<f64> {
        const OBSERVED_MS: f64 = 20.0;
        (ms >= OBSERVED_MS && bytes > 0).then(|| bytes as f64 / (ms / 1e3) / 1e9)
    }

    /// Write down what this run measured about the rates the plan prices with, so
    /// the next start plans with this box's numbers instead of the constants
    /// `laptop_4060()` guessed at it. A lane that did not run long enough to be
    /// timed contributes nothing, which is what keeps a turn that streamed nothing
    /// from blanking the disk rate a run that streamed a tier had earned.
    ///
    /// It goes to Fennec's own rates file beside `.fennec_usage`, never to
    /// anything Colibri reads. Returns what was written and the window this
    /// sample was taken over, or None when this run measured nothing worth
    /// persisting.
    ///
    /// The window is part of the answer because the number alone was not
    /// traceable: `save_rates` runs after every serve turn against counters that
    /// are cumulative for the process, and a merged rate gives no clue which
    /// bytes it came from. Sampling the difference between two readings - rather
    /// than the whole life so far - is also what makes the merge mean anything:
    /// nested cumulative samples are one window repeated, so coldest-wins kept
    /// turn 1, first page faults and all, and rejected every turn after it.
    pub fn save_rates(&self, path: &str, borrows: bool) -> Option<(crate::sched::Measured, String)> {
        let st = self.stream_totals();
        let (bytes, ns) = self.host_traffic();
        let (ub, uns) = self.union_traffic();
        let (threads, pick_bytes) = self.host_shape();
        let now = LaneWindow {
            read_bytes: st.read_bytes,
            read_ms: st.read_ms,
            quant_bytes: st.quant_bytes,
            quant_ms: st.quant_ms,
            host_bytes: bytes,
            host_ns: ns,
            union_bytes: ub,
            union_ns: uns,
        };
        let turn = {
            let mut seen = self.rates_seen.lock().unwrap();
            let turn = now.since(&seen);
            *seen = now;
            turn
        };
        let fresh = crate::sched::Measured {
            disk_gbs: Self::sustained_gbs(turn.read_bytes, turn.read_ms),
            quant_gbs: Self::sustained_gbs(turn.quant_bytes, turn.quant_ms),
            host_gbs: Self::sustained_gbs(turn.host_bytes, turn.host_ns as f64 / 1e6),
            held_ratio: (bytes > 0).then(|| self.host_held_ratio()).flatten(),
            threads,
            pick_bytes,
            borrows,
        };
        if fresh.is_empty() {
            return None;
        }
        let window = format!(
            "streamed {} MB in {:.0} ms, host {} MB in {:.0} ms, union {} MB in {:.0} ms",
            turn.read_bytes / 1_048_576,
            turn.read_ms,
            turn.host_bytes / 1_048_576,
            turn.host_ns as f64 / 1e6,
            turn.union_bytes / 1_048_576,
            turn.union_ns as f64 / 1e6
        );
        let mut merged = crate::sched::Measured::load(path).unwrap_or_default();
        merged.merge(&fresh);
        match merged.save(path) {
            Ok(()) => Some((merged, window)),
            Err(e) => {
                eprintln!("[fennec] rates: {e}");
                None
            }
        }
    }

    /// The router picks counted since the last call, as a histogram over the
    /// whole model: a row for every layer, filled only for the layers it holds.
    ///
    /// Take, not read - each counter goes back to zero as it is collected. The
    /// reason is `serve`, which folds after every turn: a cumulative read would
    /// write turn one again on turn two, and the ranking would drift toward
    /// whatever the process happened to run first. A `fetch_swap` is exact, and
    /// an increment landing between two layers' swaps simply arrives one flush
    /// later, which is how the numbers are counted anyway.
    fn take_picks(&self) -> crate::sched::Usage {
        let (n, e) = (self.cfg.num_hidden_layers, self.cfg.num_experts);
        let mut counts = vec![vec![0u64; e]; n];
        for l in &self.layers {
            if let Some(m) = &l.moe {
                // `m.layer` is the index in the checkpoint, not in this process,
                // which is what lets a `--layers` run write its rows where the
                // full model expects them instead of at the top of the file.
                if m.layer < n && m.picks.len() == e {
                    for (c, got) in counts[m.layer].iter_mut().zip(&m.picks) {
                        *c += got.swap(0, Relaxed) as u64;
                    }
                }
            }
        }
        crate::sched::Usage { layers: n, experts: e, counts }
    }

    /// Fold the picks counted since the last call into the histogram the next
    /// plan is ranked by, so a model's placement improves with use instead of
    /// staying blind forever. Returns how many were written, or None when there
    /// was nothing to say or the file could not be touched.
    ///
    /// It goes to Fennec's own file and never to `.coli_usage`: Colibri reads
    /// that one at start to decide what to AUTOPIN, so what Fennec counts must
    /// not be able to quietly change what Colibri does. Merging onto the file on
    /// disk rather than overwriting it is what keeps a three-layer debug run from
    /// writing zeros over forty layers of history.
    pub fn save_picks(&self, path: &str) -> Option<u64> {
        let mine = self.take_picks();
        let added = mine.total();
        if added == 0 {
            return None;
        }
        let mut merged = match crate::sched::Usage::load(path) {
            Ok(u) if u.layers == mine.layers && u.experts == mine.experts => u,
            Ok(u) => {
                eprintln!(
                    "[fennec] usage: {path} is {}x{}, not this model's {}x{} — left alone",
                    u.layers, u.experts, mine.layers, mine.experts
                );
                return None;
            }
            Err(_) => crate::sched::Usage {
                layers: mine.layers,
                experts: mine.experts,
                counts: vec![vec![0u64; mine.experts]; mine.layers],
            },
        };
        for (row, ours) in merged.counts.iter_mut().zip(&mine.counts) {
            for (c, &add) in row.iter_mut().zip(ours) {
                *c += add;
            }
        }
        match merged.save(path) {
            Ok(()) => Some(added),
            Err(e) => {
                eprintln!("[fennec] usage: {e}");
                None
            }
        }
    }

    pub fn states(&self, kv_cap: usize) -> Vec<LayerState> {
        self.layers.iter().map(|l| l.state(&self.cfg, kv_cap)).collect()
    }

    pub fn embed_row(&self, sh: &Shards, id: usize) -> Result<Vec<f32>> {
        sh.row_f32(&self.embed, id)
    }

    /// One token through the loaded layers, in place on `x`. This is also the
    /// unit the profiler divides by, so it counts itself.
    pub fn forward(&self, x: &mut [f32], st: &mut [LayerState], pos: usize) {
        prof::tick_decode_step();
        for (l, s) in self.layers.iter().zip(st.iter_mut()) {
            l.forward(x, s, pos);
        }
    }

    /// A chunk of prompt tokens through the loaded layers, in place on `xs`
    /// (`n` token-major rows of `hidden_size`). Every host dense projection is
    /// read once for the chunk instead of once per token, which is the whole
    /// difference between this and `n` calls of [`Model::forward`] - and the
    /// results are bit-identical to those calls.
    pub fn prefill(&self, xs: &mut [f32], st: &mut [LayerState], pos0: usize) {
        for (l, s) in self.layers.iter().zip(st.iter_mut()) {
            l.prefill(xs, s, pos0);
        }
    }

    /// Whether [`Model::prefill`] can take the prompt: its batch kernels are
    /// host kernels, so a plan that put the dense tier in VRAM has to keep
    /// walking [`Model::forward`] token by token. Routing a device prompt
    /// through these kernels instead was measured: 26 prompt tokens in 2166 ms
    /// against 1750 ms on the VRAM path, because the host buys its one weight
    /// pass per chunk (379 ms for the mixer projections) while the GPU idles and
    /// the dot competes with the host expert pool for the same bandwidth. The
    /// bytes are cheaper on the host, the token is not.
    pub fn prefill_ready(&self) -> bool {
        self.gpu.is_none()
    }

    /// A chunk of prompt tokens through the loaded layers on the device tier,
    /// the device answer to [`Model::prefill`], which is made of host kernels
    /// and so only serves a host tier.
    ///
    /// The rows get their own region of the activation arena, above everything
    /// decode claims, because a token that picks a VRAM expert still runs the
    /// per-token device chain inside the chunk and that chain writes its `x` at
    /// offset 0.
    pub fn prefill_device(&self, xs: &mut [f32], st: &mut [LayerState], pos0: usize) -> Result<()> {
        let h = self.cfg.hidden_size;
        let n = xs.len() / h;
        let Some(arc) = &self.gpu else {
            return Err("prefill_device requires a device tier".into());
        };
        let mut g = arc.lock().unwrap();
        let c = Chunk::new(&self.cfg, n);
        g.reserve(PREFILL_OFF + Chunk::span(&self.cfg, n))?;
        g.write(c.xs, xs);
        for (l, s) in self.layers.iter().zip(st.iter_mut()) {
            l.prefill_device(&mut *g, c, s, pos0)?;
        }
        g.read_into(c.xs, xs);
        Ok(())
    }

    /// Whether [`Model::prefill_device`] can take the prompt: requires a GPU
    /// to be present (device tier). Returns false for host-only tiers.
    pub fn prefill_device_ready(&self) -> bool {
        self.gpu.is_some()
    }

    /// logits = lm_head @ x. The head is never materialised: for a two-byte
    /// format the dot reads the mmap directly (`gemv_bf16` / `gemv_f16`),
    /// because an f32 copy of a 1 GiB tensor pays 2 GiB of writes and 2 GiB of
    /// reads for work that is 1 GiB of bytes wide. Anything else keeps the
    /// blocked conversion, one row block at a time so the read still streams.
    pub fn logits(&self, sh: &Shards, x: &[f32], out: &mut [f32]) -> Result<()> {
        // Named here rather than at each caller so every path that reads the
        // lm_head — device, ORT, host, `bench`, `serve` — lands in one stage.
        let _h = prof::start(prof::HEAD);
        let cols = self.cfg.hidden_size;
        if let Some(d) = &self.head_i8 {
            if out.len() == self.cfg.vocab_size {
                d.gemv(self.cfg.vocab_size, cols, x, out, self.threads);
                return Ok(());
            }
        }
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

    /// r15's two populations, as the golden inputs the gate actually saw.
    #[test]
    fn a_lane_touched_only_in_passing_reports_no_rate() {
        // The poison: five ring refills, 8.85 MB, timed over a whole serve run at
        // 0.81-0.97 GB/s - 9.1, 9.3 and 10.9 ms for these bytes. Coldest-wins then
        // keeps whichever was worst and prices every later start on this box.
        let five_refills = 8_847_360u64;
        assert_eq!(Model::sustained_gbs(five_refills, 10.94), None);
        assert_eq!(Model::sustained_gbs(five_refills, 0.0), None);
        assert_eq!(Model::sustained_gbs(0, 10_000.0), None);
        // The genuine tier: 3.38 GiB in 448.6 ms, the narrow arms' lifetime disk
        // traffic. Kept however poor the number is, because a lane that stayed busy
        // for half a second has a bandwidth.
        let carried = Model::sustained_gbs(3_634_495_488, 448.6367).unwrap();
        assert!((carried - 8.10).abs() < 0.01, "{carried}");
        // The floor is inclusive, and it gates windows, never numbers: the 2.910
        // GB/s this box's rates file carried through r14 and r15, held against
        // warmer samples, is still earnable from a window this wide.
        let cold = Model::sustained_gbs(58_205_582, 20.0).unwrap();
        assert!((cold - 2.910).abs() < 0.001, "{cold}");
    }

    /// Serve takes a rates sample after every turn off counters that are
    /// cumulative for the process, so a sample has to be the difference between
    /// two readings. When it was not, r16's arms saved 4.0 GB/s on turn 1 and 3.7
    /// on every turn after: each nested reading still contained turn 1, whose
    /// window is first-touch page faults, and coldest-wins chose it every time.
    /// The windows below are arithmetic, not a transcript.
    #[test]
    fn each_serve_turn_is_priced_on_its_own_traffic() {
        let after_turn1 = LaneWindow {
            read_bytes: 400_000_000,
            read_ms: 107.0,
            host_bytes: 400_000_000,
            host_ns: 40_000_000,
            ..Default::default()
        };
        let after_turn2 = LaneWindow {
            read_bytes: 900_000_000,
            read_ms: 232.0,
            host_bytes: 800_000_000,
            host_ns: 65_000_000,
            ..Default::default()
        };
        let turn2 = after_turn2.since(&after_turn1);
        assert_eq!(turn2.read_bytes, 500_000_000);
        assert_eq!(turn2.host_bytes, 400_000_000);
        assert!((turn2.read_ms - 125.0).abs() < 1e-9);
        // Turn 2's own 500 MB in 125 ms is 4.0 GB/s. Read cumulatively - the shape
        // r16 merged - the same two turns are 900 MB in 232 ms, 3.88, because turn
        // 1's faults ride along, and that colder number is the one that wins.
        assert!(
            (Model::sustained_gbs(turn2.read_bytes, turn2.read_ms).unwrap() - 4.0).abs() < 1e-9
        );
        assert!(
            (Model::sustained_gbs(after_turn2.read_bytes, after_turn2.read_ms).unwrap() - 3.879)
                .abs()
                < 0.001
        );
        // A turn that touched neither lane adds no sample, rather than a repeat of
        // the lifetime total it already contributed.
        let idle = after_turn2.since(&after_turn2);
        assert_eq!(Model::sustained_gbs(idle.read_bytes, idle.read_ms), None);
        assert_eq!(Model::sustained_gbs(idle.host_bytes, idle.host_ns as f64 / 1e6), None);
    }

    /// The union and the per-token picks read the same packs, not the same number
    /// of them: a 26-token chunk that routes 8 picks a token may reach three of
    /// the tier's resident experts. Folding the union's picks into `host_hits`
    /// while also summing its experts into `host_union_bytes` billed 29 packs for
    /// 3 reads, and because `Measured::merge` keeps the warmest sample the most
    /// reuse-heavy chunk of every run set the price - the rates file's 21.7 GB/s
    /// against the 3.1-8.0 GB/s the served profiler timed on the same tier.
    #[test]
    fn a_prefill_union_pick_bills_the_expert_not_the_token() {
        let cfg = Arc::new(TextConfig {
            hidden_size: 128,
            num_hidden_layers: 1,
            vocab_size: 256,
            rms_norm_eps: 1e-6,
            norm_zero_centered: true,
            layer_types: vec![LayerKind::Full],
            num_attention_heads: 2,
            num_key_value_heads: 2,
            head_dim: 64,
            partial_rotary_factor: 1.0,
            rope_theta: 1e6,
            attn_output_gate: false,
            intermediate_size: 0,
            tie_word_embeddings: false,
            linear_num_key_heads: 0,
            linear_num_value_heads: 0,
            linear_key_head_dim: 0,
            linear_value_head_dim: 0,
            linear_conv_kernel_dim: 0,
            num_experts: 8,
            num_experts_per_tok: 2,
            moe_intermediate_size: 64,
            shared_expert_intermediate_size: 64,
            prefix: "model".into(),
            name: "tiny-host-tier".into(),
        });
        let (h, i, group) = (cfg.hidden_size, cfg.moe_intermediate_size, 64);
        // Two experts resident: gate|up is 2*I rows per expert, down is H rows.
        let gu = Q4::quantize(&weights(2 * i * 2, h, 1), 2 * i * 2, h, group, 2);
        let dn = Q4::quantize(&weights(h * 2, i, 2), h * 2, i, group, 2);
        let parts = MoeParts {
            gate: Dense::F32(vec![]),
            s_gate: Dense::F32(vec![]),
            s_up: Dense::F32(vec![]),
            s_down: Dense::F32(vec![]),
            s_egate: vec![],
        };
        let m = MoeLayer::from_parts(cfg, 0, parts, gu, dn, 4);
        let pack = m.host_pick_bytes() as u64;
        assert!(pack > 0, "a tier with a resident expert has a pack to price");

        // Five decode picks in 50 us, and one chunk that reached three experts in
        // 9 us of union time.
        m.host_hits.store(5, Relaxed);
        m.host_ns.store(50_000, Relaxed);
        m.union_picks.store(26, Relaxed);
        m.host_union_bytes.store(3 * pack, Relaxed);
        m.union_ns.store(9_000, Relaxed);

        assert_eq!(m.decode_traffic(), (5 * pack, 50_000));
        assert_eq!(m.union_traffic(), (3 * pack, 9_000));
        // The arithmetic the folded counter produced for the same traffic.
        let folded = m.union_picks.load(Relaxed) * pack + m.host_union_bytes.load(Relaxed);
        assert_eq!(folded, 29 * pack);
    }
}
