//! The CUDA device tier: same arena, same offsets, same six kernels as Vulkan.
//!
//! The driver API is loaded at runtime from `nvcuda.dll` / `libcuda.so.1`, which
//! ships with the display driver, and the PTX in `kernels/fennec.ptx` is JIT'd by
//! `cuModuleLoadData`. There is no nvcc, no ptxas and no CUDA toolkit in the
//! build.
//!
//! The activation arena is **device-local** (`cuMemAlloc`), and `write`/`read_into`
//! are explicit HtoD/DtoH copies. This is deliberately not Vulkan's host-visible
//! coherent arena: CUDA zero-copy memory makes every kernel read its activations
//! over PCIe, and the gemv re-reads the same x once per row. Measured on this
//! card, pinned host memory cost lm_head 6.7 ms against 4.7 device-local and the
//! mixer 25.5 ms against 14.6, so the copies are the cheaper side of that trade.
//! Every offset the model computes still means the same thing on either backend.
//!
//! A batch is a list of launches rather than a recorded command buffer: the
//! default stream is in-order, so consecutive kernels see each other's writes
//! without the explicit barriers Vulkan needs.
//!
//! **CUDA graphs**: repeated dispatch patterns (e.g., one decode step) can be
//! captured into a graph and replayed with minimal launch overhead. The graph
//! cache stores instantiated executables keyed by their launch sequence hash.

use crate::config::Result;
use crate::cuda::{version_string, Cuda, DevPtr, Handle};
use crate::gpu::Op;
use crate::quant::Q4;
use crate::vk::Device;
use std::collections::HashMap;
use std::ffi::c_void;

const PTX: &str = include_str!("kernels/fennec.ptx");

/// One kernel per `Op`, in the same order the Vulkan pipelines are built.
const NAMES: [&str; 8] = ["gemv_bf16", "gemv_q4", "rmsnorm", "silu_mul", "axpy", "gemv_f16", "gdn_conv", "gdn_rec"];

struct CBuf {
    ptr: DevPtr,
    size: u64,
    f16: bool,
}

/// A recorded launch. Arguments are held by value so the parameter block can
/// point at them when the batch is submitted.
#[derive(Clone, Copy)]
enum Launch {
    Axpy { n: u32, x_off: u32, y_off: u32, w: f32, set: bool },
    SiluMul { n: u32, a_off: u32, b_off: u32, y_off: u32 },
    RmsNorm { n: u32, x_off: u32, w_off: u32, y_off: u32, eps: f32 },
    GemvQ4 { w: DevPtr, s: DevPtr, rows: u32, cols: u32, group: u32, x_off: u32, y_off: u32 },
    GemvHalf { w: DevPtr, f16: bool, rows: u32, cols: u32, x_off: u32, y_off: u32 },
    GdnConv { w: DevPtr, state: DevPtr, base: u32, cd: u32, kernel: u32, hist: u32 },
    GdnRec { w: DevPtr, state: DevPtr, base: u32, out_off: u32, kd: u32, nv: u32, dk: u32, dv: u32, rep: u32, eps: f32 },
}

pub struct CudaGpu {
    cu: Cuda,
    pub info: Device,
    bufs: Vec<CBuf>,
    funcs: [Handle; 8],
    arena_dev: DevPtr,
    arena_floats: usize,
    pub submitted: u64,
    pub dispatches: u64,
    pub sync_ms: f64,
    pub syncs: u64,
    /// Stream for graph capture and async operations
    stream: Handle,
    /// Cache of instantiated graph executables, keyed by launch sequence hash
    graph_cache: HashMap<u64, Handle>,
    /// Number of graph cache hits (for profiling)
    pub graph_hits: u64,
    /// Number of graph captures (for profiling)
    pub graph_captures: u64,
    /// The pair of stream markers bracketing one mixer batch. CUDA events are
    /// queue positions, not host stamps: `ev_a` fires when the GPU reaches it,
    /// which after a drained stream is immediately, and `ev_b` when the batch
    /// it follows is done. Their difference is what the silicon actually spent,
    /// as opposed to how long the host spent blocked in the readback.
    ev_a: Handle,
    ev_b: Handle,
    /// Set by `time_mixer_batch`, consumed by the next submit.
    time_next: bool,
    /// A bracket is in the queue, waiting for its end event to fire.
    pending: bool,
    /// What the GPU reported for the mixer batches, and how many brackets were
    /// harvested. `sync_ms` beside it is the host's side of the same waits.
    pub device_ms: f64,
    pub device_samples: u64,
}

// The context is current on the owning thread only; the CPU workers that run
// alongside a submitted batch never touch it.
unsafe impl Send for CudaGpu {}

fn arg<T>(v: &mut T) -> *mut c_void {
    v as *mut T as *mut c_void
}

fn div_up(n: u32, d: u32) -> u32 {
    n.div_ceil(d)
}

impl CudaGpu {
    pub fn open(index: usize) -> Result<CudaGpu> {
        let mut cu = Cuda::load().map_err(|e| format!("no CUDA driver: {e}"))?;
        cu.open(index, &[PTX]).map_err(|e| format!("CUDA device {index}: {e}"))?;
        let mut funcs: [Handle; 8] = [std::ptr::null_mut(); 8];
        for (i, n) in NAMES.iter().enumerate() {
            funcs[i] = cu.function(n).map_err(|e| format!("CUDA kernel {n}: {e}"))?;
        }
        let d = cu.devices[index].clone();
        let bytes = (d.total_gib * 2f64.powi(30)) as u64;
        let info = Device {
            index,
            name: d.name.clone(),
            vendor_id: 0x10DE,
            device_id: 0,
            class: 2,
            api: format!("CUDA {}.{}", d.compute.0, d.compute.1),
            driver: version_string(cu.driver_version),
            device_local_gib: d.total_gib,
            free_gib: d.free_gib,
            dedicated_gib: d.total_gib,
            host_visible_gib: 0.0,
            heaps: vec![(bytes, 0x1, false)],
            extensions: Vec::new(),
            vendor: "NVIDIA",
            rank: 0,
        };
        let mut g = CudaGpu {
            cu,
            info,
            bufs: Vec::new(),
            funcs,
            arena_dev: 0,
            arena_floats: 0,
            submitted: 0,
            dispatches: 0,
            sync_ms: 0.0,
            syncs: 0,
            stream: std::ptr::null_mut(),
            graph_cache: HashMap::new(),
            graph_hits: 0,
            graph_captures: 0,
            ev_a: std::ptr::null_mut(),
            ev_b: std::ptr::null_mut(),
            time_next: false,
            pending: false,
            device_ms: 0.0,
            device_samples: 0,
        };
        // Create a stream for graph capture and async operations
        g.stream = g.cu.create_stream().unwrap_or(std::ptr::null_mut());
        // The timing bracket. Null on failure, which turns the instrument off
        // rather than failing the tier: it measures, it does not compute.
        g.ev_a = g.cu.create_event().unwrap_or(std::ptr::null_mut());
        g.ev_b = g.cu.create_event().unwrap_or(std::ptr::null_mut());
        // The same starting arena the Vulkan tier opens with, so a caller that
        // writes activations without naming a size behaves identically on both.
        g.reserve(crate::gpu::SCRATCH_FLOATS)?;
        Ok(g)
    }

    pub fn name(&self) -> &str {
        &self.info.name
    }

    /// Refresh the free VRAM reading. Call after large allocations or frees to
    /// get an accurate current figure rather than the one from device open.
    pub fn refresh_free_vram(&mut self) -> Result<()> {
        let free_bytes = self.cu.query_free_mem()?;
        self.info.free_gib = free_bytes as f64 / 2f64.powi(30);
        Ok(())
    }

    /// Current free VRAM in GiB, refreshed on demand.
    pub fn free_vram_gib(&mut self) -> Result<f64> {
        self.refresh_free_vram()?;
        Ok(self.info.free_gib)
    }

    /// Grow the device-local activation arena. Like the Vulkan tier's `reserve`,
    /// this discards the contents: callers write their inputs after reserving.
    pub fn reserve(&mut self, floats: usize) -> Result<()> {
        if floats <= self.arena_floats {
            return Ok(());
        }
        let dev = self
            .cu
            .alloc(floats * 4)
            .map_err(|e| format!("CUDA arena of {} floats: {e}", floats))?;
        self.cu.free(self.arena_dev);
        self.arena_dev = dev;
        self.arena_floats = floats;
        // A captured graph carries the arena base in every kernel parameter, so
        // moving the arena invalidates all of them. Without this a queue that
        // repeats after a grow would replay against the freed buffer.
        for (_, exec) in self.graph_cache.drain() {
            self.cu.destroy_graph_exec(exec);
        }
        Ok(())
    }

    pub fn arena(&self) -> usize {
        self.arena_floats
    }

    pub fn upload(&mut self, bytes: &[u8]) -> Result<usize> {
        self.upload_half(bytes, false)
    }

    pub fn upload_half(&mut self, bytes: &[u8], f16: bool) -> Result<usize> {
        let ptr = self.cu.alloc(bytes.len()).map_err(|e| format!("CUDA alloc {} bytes: {e}", bytes.len()))?;
        if let Err(e) = self.cu.write(ptr, bytes) {
            self.cu.free(ptr);
            return Err(format!("CUDA upload: {e}"));
        }
        self.bufs.push(CBuf { ptr, size: bytes.len() as u64, f16 });
        Ok(self.bufs.len() - 1)
    }

    pub fn upload_q4(&mut self, q: &Q4) -> Result<(usize, usize)> {
        if q.cols % 8 != 0 || q.group % 8 != 0 {
            return Err(format!("the q4 kernel needs cols and group divisible by 8, got {} / {}", q.cols, q.group));
        }
        if q.signed {
            return Err("the q4 kernel reads offset-binary nibbles; a borrowed pack must be flipped first".into());
        }
        if q.cols % q.group != 0 {
            return Err(format!("the q4 kernel needs cols divisible by group, got {} / {}", q.cols, q.group));
        }
        let w = self.upload(&q.data)?;
        let mut sb = Vec::with_capacity(q.scale.len() * 4);
        for s in &q.scale {
            sb.extend_from_slice(&s.to_le_bytes());
        }
        let s = self.upload(&sb)?;
        Ok((w, s))
    }

    pub fn alloc_half(&mut self, bytes: usize, f16: bool) -> Result<usize> {
        let ptr = self.cu.alloc(bytes).map_err(|e| format!("CUDA alloc {bytes} bytes: {e}"))?;
        self.bufs.push(CBuf { ptr, size: bytes as u64, f16 });
        Ok(self.bufs.len() - 1)
    }

    pub fn write_buf(&mut self, id: usize, at: usize, bytes: &[u8]) -> Result<()> {
        if at + bytes.len() > self.bufs[id].size as usize {
            return Err(format!("write_buf: {} bytes at {at} into a {}-byte buffer", bytes.len(), self.bufs[id].size));
        }
        if bytes.is_empty() {
            return Ok(());
        }
        self.cu.write(self.bufs[id].ptr + at as u64, bytes).map_err(|e| format!("CUDA write_buf {id}: {e}"))
    }

    pub fn read_buf(&mut self, id: usize, out: &mut [u8]) -> Result<()> {
        if out.len() > self.bufs[id].size as usize {
            return Err(format!("read_buf: {} bytes out of a {}-byte buffer", out.len(), self.bufs[id].size));
        }
        if out.is_empty() {
            return Ok(());
        }
        self.cu.read(out, self.bufs[id].ptr).map_err(|e| format!("CUDA read_buf {id}: {e}"))
    }

    pub fn write(&mut self, off: usize, v: &[f32]) {
        assert!(off + v.len() <= self.arena_floats, "activation arena too small");
        let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        self.cu
            .write(self.arena_dev + (off * 4) as u64, bytes)
            .unwrap_or_else(|e| panic!("CUDA arena write at {off}: {e}"));
    }

    pub fn read(&self, off: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0f32; n];
        self.read_into(off, &mut out);
        out
    }

    pub fn read_into(&self, off: usize, out: &mut [f32]) {
        assert!(off + out.len() <= self.arena_floats, "activation arena too small");
        let bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out.len() * 4) };
        self.cu
            .read(bytes, self.arena_dev + (off * 4) as u64)
            .unwrap_or_else(|e| panic!("CUDA arena read at {off}: {e}"));
    }

    pub fn batch(&mut self) -> Result<CudaBatch<'_>> {
        Ok(CudaBatch { gpu: self, queue: Vec::new(), ops: 0, live: true })
    }

    fn exec(&mut self, l: Launch) -> Result<()> {
        let arena = self.arena_dev;
        match l {
            Launch::Axpy { n, x_off, y_off, w, set } => {
                let (mut a, mut n, mut x, mut y, mut w, mut s) =
                    (arena, n, x_off, y_off, w, set as u32);
                let mut p = [arg(&mut a), arg(&mut n), arg(&mut x), arg(&mut y), arg(&mut w), arg(&mut s)];
                self.cu.launch(self.funcs[Op::Axpy as usize], div_up(n, 256), 256, 0, &mut p)
            }
            Launch::SiluMul { n, a_off, b_off, y_off } => {
                let (mut a, mut n, mut x, mut b, mut y) = (arena, n, a_off, b_off, y_off);
                let mut p = [arg(&mut a), arg(&mut n), arg(&mut x), arg(&mut b), arg(&mut y)];
                self.cu.launch(self.funcs[Op::SiluMul as usize], div_up(n, 256), 256, 0, &mut p)
            }
            Launch::RmsNorm { n, x_off, w_off, y_off, eps } => {
                let (mut a, mut n, mut x, mut w, mut y, mut e) = (arena, n, x_off, w_off, y_off, eps);
                let mut p = [arg(&mut a), arg(&mut n), arg(&mut x), arg(&mut w), arg(&mut y), arg(&mut e)];
                self.cu.launch(self.funcs[Op::RmsNorm as usize], 1, 256, 0, &mut p)
            }
            Launch::GemvQ4 { w, s, rows, cols, group, x_off, y_off } => {
                // grid.x is a signed 32-bit count, so the 65535-workgroup split
                // the Vulkan path needs for a tall matrix does not apply here.
                let (mut w, mut s, mut a, mut r, mut c, mut g, mut x, mut y, mut base) =
                    (w, s, arena, rows, cols, group, x_off, y_off, 0u32);
                let mut p = [
                    arg(&mut w), arg(&mut s), arg(&mut a), arg(&mut r), arg(&mut c),
                    arg(&mut g), arg(&mut x), arg(&mut y), arg(&mut base),
                ];
                self.cu.launch(self.funcs[Op::GemvQ4 as usize], rows, 64, 0, &mut p)
            }
            Launch::GemvHalf { w, f16, rows, cols, x_off, y_off } => {
                let (mut w, mut a, mut r, mut c, mut x, mut y, mut base) =
                    (w, arena, rows, cols, x_off, y_off, 0u32);
                let mut p = [
                    arg(&mut w), arg(&mut a), arg(&mut r), arg(&mut c),
                    arg(&mut x), arg(&mut y), arg(&mut base),
                ];
                let op = if f16 { Op::GemvF16 } else { Op::GemvBf16 };
                self.cu.launch(self.funcs[op as usize], div_up(rows, 4), 256, 0, &mut p)
            }
            Launch::GdnConv { w, state, base, cd, kernel, hist } => {
                let (mut a, mut wp, mut sp, mut b, mut c, mut k, mut h) = (arena, w, state, base, cd, kernel, hist);
                let mut p = [arg(&mut a), arg(&mut wp), arg(&mut sp), arg(&mut b), arg(&mut c), arg(&mut k), arg(&mut h)];
                self.cu.launch(self.funcs[Op::GdnConv as usize], div_up(cd, 256), 256, 0, &mut p)
            }
            Launch::GdnRec { w, state, base, out_off, kd, nv, dk, dv, rep, eps } => {
                let (mut a, mut wp, mut sp, mut b, mut o, mut k, mut n, mut dkk, mut dvv, mut r, mut e) =
                    (arena, w, state, base, out_off, kd, nv, dk, dv, rep, eps);
                let mut p = [
                    arg(&mut a), arg(&mut wp), arg(&mut sp), arg(&mut b), arg(&mut o), arg(&mut k),
                    arg(&mut n), arg(&mut dkk), arg(&mut dvv), arg(&mut r), arg(&mut e),
                ];
                self.cu.launch(self.funcs[Op::GdnRec as usize], nv, 512, 0, &mut p)
            }
        }
    }

    /// Execute a launch on a specific stream (for graph capture).
    fn exec_on_stream(&mut self, l: Launch, stream: Handle) -> Result<()> {
        let arena = self.arena_dev;
        match l {
            Launch::Axpy { n, x_off, y_off, w, set } => {
                let (mut a, mut n, mut x, mut y, mut w, mut s) =
                    (arena, n, x_off, y_off, w, set as u32);
                let mut p = [arg(&mut a), arg(&mut n), arg(&mut x), arg(&mut y), arg(&mut w), arg(&mut s)];
                self.cu.launch_on_stream(self.funcs[Op::Axpy as usize], stream, div_up(n, 256), 256, 0, &mut p)
            }
            Launch::SiluMul { n, a_off, b_off, y_off } => {
                let (mut a, mut n, mut x, mut b, mut y) = (arena, n, a_off, b_off, y_off);
                let mut p = [arg(&mut a), arg(&mut n), arg(&mut x), arg(&mut b), arg(&mut y)];
                self.cu.launch_on_stream(self.funcs[Op::SiluMul as usize], stream, div_up(n, 256), 256, 0, &mut p)
            }
            Launch::RmsNorm { n, x_off, w_off, y_off, eps } => {
                let (mut a, mut n, mut x, mut w, mut y, mut e) = (arena, n, x_off, w_off, y_off, eps);
                let mut p = [arg(&mut a), arg(&mut n), arg(&mut x), arg(&mut w), arg(&mut y), arg(&mut e)];
                self.cu.launch_on_stream(self.funcs[Op::RmsNorm as usize], stream, 1, 256, 0, &mut p)
            }
            Launch::GemvQ4 { w, s, rows, cols, group, x_off, y_off } => {
                let (mut w, mut s, mut a, mut r, mut c, mut g, mut x, mut y, mut base) =
                    (w, s, arena, rows, cols, group, x_off, y_off, 0u32);
                let mut p = [
                    arg(&mut w), arg(&mut s), arg(&mut a), arg(&mut r), arg(&mut c),
                    arg(&mut g), arg(&mut x), arg(&mut y), arg(&mut base),
                ];
                self.cu.launch_on_stream(self.funcs[Op::GemvQ4 as usize], stream, rows, 64, 0, &mut p)
            }
            Launch::GemvHalf { w, f16, rows, cols, x_off, y_off } => {
                let (mut w, mut a, mut r, mut c, mut x, mut y, mut base) =
                    (w, arena, rows, cols, x_off, y_off, 0u32);
                let mut p = [
                    arg(&mut w), arg(&mut a), arg(&mut r), arg(&mut c),
                    arg(&mut x), arg(&mut y), arg(&mut base),
                ];
                let op = if f16 { Op::GemvF16 } else { Op::GemvBf16 };
                self.cu.launch_on_stream(self.funcs[op as usize], stream, div_up(rows, 4), 256, 0, &mut p)
            }
            Launch::GdnConv { w, state, base, cd, kernel, hist } => {
                let (mut a, mut wp, mut sp, mut b, mut c, mut k, mut h) = (arena, w, state, base, cd, kernel, hist);
                let mut p = [arg(&mut a), arg(&mut wp), arg(&mut sp), arg(&mut b), arg(&mut c), arg(&mut k), arg(&mut h)];
                self.cu.launch_on_stream(self.funcs[Op::GdnConv as usize], stream, div_up(cd, 256), 256, 0, &mut p)
            }
            Launch::GdnRec { w, state, base, out_off, kd, nv, dk, dv, rep, eps } => {
                let (mut a, mut wp, mut sp, mut b, mut o, mut k, mut n, mut dkk, mut dvv, mut r, mut e) =
                    (arena, w, state, base, out_off, kd, nv, dk, dv, rep, eps);
                let mut p = [
                    arg(&mut a), arg(&mut wp), arg(&mut sp), arg(&mut b), arg(&mut o), arg(&mut k),
                    arg(&mut n), arg(&mut dkk), arg(&mut dvv), arg(&mut r), arg(&mut e),
                ];
                self.cu.launch_on_stream(self.funcs[Op::GdnRec as usize], stream, nv, 512, 0, &mut p)
            }
        }
    }

    /// Ask for the *next* batch submit to be bracketed by the timing events.
    /// Only the mixer calls this: its host-side rows are the ones whose GPU
    /// share is unknown, and the harvest lands in the row that splits them.
    pub fn time_mixer_batch(&mut self) {
        self.harvest();
        self.time_next = !self.ev_a.is_null() && !self.ev_b.is_null();
    }

    /// Bill a finished bracket. Called from `time_mixer_batch`, which runs after
    /// the previous batch's blocking read, so both markers have fired and this
    /// never waits. A bracket still pending is left for the next call, and the
    /// last one of a request is dropped: one sample in tens of thousands.
    fn harvest(&mut self) {
        if !self.pending {
            return;
        }
        if let Some(ms) = self.cu.event_elapsed(self.ev_a, self.ev_b) {
            self.pending = false;
            self.device_ms += ms;
            self.device_samples += 1;
            crate::prof::add(crate::prof::MIXER_DEV_DEVICE, ms);
        }
    }

    /// Open a bracket on the stream the batch is about to be launched on. The
    /// markers are queue positions, so recording one on a stream other than the
    /// work's measures the gap between two unrelated queues.
    fn open_timing(&mut self, stream: Handle) -> bool {
        if !std::mem::take(&mut self.time_next) {
            return false;
        }
        self.cu.record_event(self.ev_a, stream).is_ok()
    }

    fn close_timing(&mut self, stream: Handle, opened: bool) {
        if opened && self.cu.record_event(self.ev_b, stream).is_ok() {
            self.pending = true;
        }
    }

    pub fn wait(&mut self) -> Result<()> {
        let t0 = std::time::Instant::now();
        let r = self.cu.sync();
        self.sync_ms += t0.elapsed().as_secs_f64() * 1e3;
        self.syncs += 1;
        r
    }

    pub fn wait_idle(&mut self) -> Result<()> {
        self.cu.sync()
    }

    pub fn bytes_held(&self) -> u64 {
        self.bufs.iter().map(|b| b.size).sum::<u64>() + (self.arena_floats * 4) as u64
    }

    pub fn reset_counters(&mut self) {
        self.submitted = 0;
        self.dispatches = 0;
        self.sync_ms = 0.0;
        self.syncs = 0;
        self.graph_hits = 0;
        self.graph_captures = 0;
        self.device_ms = 0.0;
        self.device_samples = 0;
        // An in-flight bracket belongs to the window that opened it.
        self.pending = false;
        self.time_next = false;
    }
}

impl Drop for CudaGpu {
    fn drop(&mut self) {
        let _ = self.cu.sync();
        for b in &self.bufs {
            self.cu.free(b.ptr);
        }
        self.cu.free(self.arena_dev);
        // Clean up cached graph executables
        for (_, exec) in self.graph_cache.drain() {
            self.cu.destroy_graph_exec(exec);
        }
        // Clean up the stream and the timing events
        if !self.stream.is_null() {
            self.cu.destroy_stream(self.stream);
        }
        self.cu.destroy_event(self.ev_a);
        self.cu.destroy_event(self.ev_b);
    }
}

pub struct CudaBatch<'a> {
    gpu: &'a mut CudaGpu,
    queue: Vec<Launch>,
    pub ops: usize,
    live: bool,
}

impl<'a> CudaBatch<'a> {
    fn push(&mut self, l: Launch) -> Result<&mut CudaBatch<'a>> {
        self.queue.push(l);
        self.ops += 1;
        Ok(self)
    }

    pub fn gemv_half(&mut self, w: usize, rows: usize, cols: usize, x_off: usize, y_off: usize) -> Result<&mut CudaBatch<'a>> {
        assert!(cols % 2 == 0, "half pairs need an even column count");
        // The PTX takes its wide path on `cols % 8 == 0` alone, and there its
        // scalar fallback sits behind the same branch. `ld.global.v4.f32`
        // requires a 16-byte-aligned address, so an `x_off` that is not a whole
        // float4 faults the context instead of reading the row at a stride. The
        // Vulkan shaders gained an `x_off & 3` guard; this precompiled PTX has no
        // source or toolchain here, so the layout has to answer for it: prefill's
        // arena keeps every gemv input on a float4 boundary (see `Chunk::new`).
        assert!(cols % 8 != 0 || x_off % 4 == 0, "gemv_half at cols {cols} loads x as float4, so x_off {x_off} must be a multiple of 4");
        let f16 = self.gpu.bufs[w].f16;
        self.push(Launch::GemvHalf { w: self.gpu.bufs[w].ptr, f16, rows: rows as u32, cols: cols as u32, x_off: x_off as u32, y_off: y_off as u32 })
    }

    pub fn gemv_q4(&mut self, w: usize, sc: usize, rows: usize, cols: usize, group: usize, x_off: usize, y_off: usize) -> Result<&mut CudaBatch<'a>> {
        let (wp, sp) = (self.gpu.bufs[w].ptr, self.gpu.bufs[sc].ptr);
        self.push(Launch::GemvQ4 { w: wp, s: sp, rows: rows as u32, cols: cols as u32, group: group as u32, x_off: x_off as u32, y_off: y_off as u32 })
    }

    pub fn rmsnorm(&mut self, n: usize, x_off: usize, w_off: usize, y_off: usize, eps: f32) -> Result<&mut CudaBatch<'a>> {
        self.push(Launch::RmsNorm { n: n as u32, x_off: x_off as u32, w_off: w_off as u32, y_off: y_off as u32, eps })
    }

    pub fn silu_mul(&mut self, n: usize, a_off: usize, b_off: usize, y_off: usize) -> Result<&mut CudaBatch<'a>> {
        self.push(Launch::SiluMul { n: n as u32, a_off: a_off as u32, b_off: b_off as u32, y_off: y_off as u32 })
    }

    pub fn axpy(&mut self, n: usize, x_off: usize, y_off: usize, w: f32, set: bool) -> Result<&mut CudaBatch<'a>> {
        self.push(Launch::Axpy { n: n as u32, x_off: x_off as u32, y_off: y_off as u32, w, set })
    }

    pub fn gdn_conv(&mut self, w: usize, state: usize, base: usize, cd: usize, kernel: usize, hist: usize) -> Result<&mut CudaBatch<'a>> {
        let (wp, sp) = (self.gpu.bufs[w].ptr, self.gpu.bufs[state].ptr);
        self.push(Launch::GdnConv { w: wp, state: sp, base: base as u32, cd: cd as u32, kernel: kernel as u32, hist: hist as u32 })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gdn_rec(&mut self, w: usize, state: usize, base: usize, out_off: usize, kd: usize, nv: usize, dk: usize, dv: usize, rep: usize, eps: f32) -> Result<&mut CudaBatch<'a>> {
        let (wp, sp) = (self.gpu.bufs[w].ptr, self.gpu.bufs[state].ptr);
        self.push(Launch::GdnRec {
            w: wp, state: sp, base: base as u32, out_off: out_off as u32, kd: kd as u32,
            nv: nv as u32, dk: dk as u32, dv: dv as u32, rep: rep as u32, eps,
        })
    }

    pub fn submit(&mut self) -> Result<()> {
        if !self.live {
            return Err("batch already submitted".into());
        }
        self.live = false;
        CudaBatch::run_plain(self.gpu, std::mem::take(&mut self.queue))
    }

    pub fn wait(&mut self) -> Result<()> {
        self.gpu.wait()
    }

    pub fn run(&mut self) -> Result<()> {
        self.submit()?;
        self.gpu.wait()
    }

    /// Every value that reaches the driver has to reach the hash, including the
    /// device pointers and arena offsets. A queue keyed on shape alone is the
    /// same for all forty MoE and mixer layers, so the second layer would replay
    /// the first layer's weights and nothing would report an error.
    fn queue_hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.queue.len().hash(&mut hasher);
        for l in &self.queue {
            match *l {
                Launch::Axpy { n, x_off, y_off, w, set } => {
                    (0u8, n, x_off, y_off, w.to_bits(), set as u8).hash(&mut hasher)
                }
                Launch::SiluMul { n, a_off, b_off, y_off } => {
                    (1u8, n, a_off, b_off, y_off).hash(&mut hasher)
                }
                Launch::RmsNorm { n, x_off, w_off, y_off, eps } => {
                    (2u8, n, x_off, w_off, y_off, eps.to_bits()).hash(&mut hasher)
                }
                Launch::GemvQ4 { w, s, rows, cols, group, x_off, y_off } => {
                    (3u8, w, s, rows, cols, group, x_off, y_off).hash(&mut hasher)
                }
                Launch::GemvHalf { w, f16, rows, cols, x_off, y_off } => {
                    (4u8, w, f16 as u8, rows, cols, x_off, y_off).hash(&mut hasher)
                }
                Launch::GdnConv { w, state, base, cd, kernel, hist } => {
                    (5u8, w, state, base, cd, kernel, hist).hash(&mut hasher)
                }
                Launch::GdnRec { w, state, base, out_off, kd, nv, dk, dv, rep, eps } => {
                    (6u8, w, state, base, out_off, kd, nv, dk, dv, rep, eps.to_bits()).hash(&mut hasher)
                }
            }
        }
        hasher.finish()
    }

    /// Plain per-launch submit of an already-detached queue.
    fn run_plain(gpu: &mut CudaGpu, q: Vec<Launch>) -> Result<()> {
        // `exec` launches on the legacy stream, so the markers go there too.
        let timed = gpu.open_timing(std::ptr::null_mut());
        for l in q {
            gpu.exec(l)?;
            gpu.dispatches += 1;
        }
        gpu.close_timing(std::ptr::null_mut(), timed);
        gpu.submitted += 1;
        Ok(())
    }

    /// Submit the batch using a cached CUDA graph if available, or capture and
    /// cache it on first use. Graph replay avoids per-launch driver overhead.
    pub fn submit_graph(&mut self) -> Result<()> {
        if !self.live {
            return Err("batch already submitted".into());
        }
        self.live = false;

        // Check if we have a cached graph for this launch pattern
        let hash = self.queue_hash();
        if let Some(&exec) = self.gpu.graph_cache.get(&hash) {
            // Replay the cached graph
            let stream = self.gpu.stream;
            let timed = self.gpu.open_timing(stream);
            self.gpu.cu.launch_graph(exec, stream)?;
            self.gpu.close_timing(stream, timed);
            self.gpu.graph_hits += 1;
            self.gpu.submitted += 1;
            self.gpu.dispatches += self.queue.len() as u64;
            self.queue.clear();
            return Ok(());
        }

        // One capture per reusable queue, so a couple of hundred distinct queues
        // is well past any model's layer count. Beyond that the callers repeat
        // nothing and capture costs more than the launches it saves.
        const MAX_GRAPHS: usize = 256;
        let q = std::mem::take(&mut self.queue);
        let stream = self.gpu.stream;
        if stream.is_null() || self.gpu.graph_cache.len() >= MAX_GRAPHS {
            return CudaBatch::run_plain(self.gpu, q);
        }

        // A launch that fails partway through capture would leave the stream in
        // capture mode, where every later call on it fails too, so the capture is
        // always closed and anything not usable comes out as plain launches.
        let mut graph = None;
        if self.gpu.cu.begin_capture(stream).is_ok() {
            let mut recorded = true;
            for l in &q {
                if self.gpu.exec_on_stream(*l, stream).is_err() {
                    recorded = false;
                    break;
                }
            }
            match (recorded, self.gpu.cu.end_capture(stream)) {
                (true, Ok(g)) => graph = Some(g),
                (_, Ok(g)) => self.gpu.cu.destroy_graph(g),
                _ => {}
            }
        }
        let Some(captured) = graph else {
            return CudaBatch::run_plain(self.gpu, q);
        };
        // Instantiate the graph into an executable
        let exec = match self.gpu.cu.instantiate_graph(captured) {
            Ok(e) => e,
            Err(_) => {
                self.gpu.cu.destroy_graph(captured);
                return CudaBatch::run_plain(self.gpu, q);
            }
        };

        // Cache the executable (we keep the graph alive via the exec)
        self.gpu.graph_cache.insert(hash, exec);
        self.gpu.graph_captures += 1;
        self.gpu.submitted += 1;
        self.gpu.dispatches += q.len() as u64;

        // Clean up the graph (the exec holds a reference)
        self.gpu.cu.destroy_graph(captured);

        // Now launch the graph for real
        self.gpu.cu.launch_graph(exec, stream)
    }
}

impl Drop for CudaBatch<'_> {
    fn drop(&mut self) {
        // Match the Vulkan batch: recorded work that was never submitted is
        // discarded, not run. The queue drops with it.
        self.queue.clear();
    }
}