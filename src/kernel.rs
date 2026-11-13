//! CPU kernels. Everything here is either a direct transcription of the
//! transformers reference (verified in tests against the goldens) or an AVX2
//! version of it that must match the scalar path to within float tolerance.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

#[inline]
pub fn chunk(n: usize, threads: usize) -> Vec<(usize, usize)> {
    let t = threads.clamp(1, n.max(1));
    let base = n / t;
    let rem = n % t;
    let mut out = Vec::with_capacity(t);
    let mut s = 0usize;
    for i in 0..t {
        let len = base + if i < rem { 1 } else { 0 };
        if len > 0 {
            out.push((s, s + len));
        }
        s += len;
    }
    out
}

/// One piece of a `par_rows` call with its lifetime erased. Sound because the
/// submitting call blocks until every worker has run its task, so the slice and
/// the closure outlive every pointer handed out here. The worker owns the box
/// and frees it by running it.
struct Task {
    ptr: *mut (),
    call: unsafe fn(*mut ()),
}
unsafe impl Send for Task {}

struct RowJob<T, F> {
    part: *mut T,
    len: usize,
    start: usize,
    f: *const F,
}

impl Task {
    fn new<T: Send, F: Fn(&mut [T], usize) + Sync>(part: &mut [T], start: usize, f: &F) -> Task {
        unsafe fn call<T, F: Fn(&mut [T], usize)>(p: *mut ()) {
            let j = Box::from_raw(p as *mut RowJob<T, F>);
            let part = std::slice::from_raw_parts_mut(j.part, j.len);
            (*j.f)(part, j.start);
        }
        let job = Box::new(RowJob {
            part: part.as_mut_ptr(),
            len: part.len(),
            start,
            f: f as *const F,
        });
        Task { ptr: Box::into_raw(job) as *mut (), call: call::<T, F> }
    }

    fn index<F: Fn(usize, usize) + Sync>(start: usize, len: usize, f: &F) -> Task {
        unsafe fn call<F: Fn(usize, usize)>(p: *mut ()) {
            let j = Box::from_raw(p as *mut IndexJob<F>);
            (*j.f)(j.start, j.len);
        }
        let job = Box::new(IndexJob { start, len, f: f as *const F });
        Task { ptr: Box::into_raw(job) as *mut (), call: call::<F> }
    }
}

/// An index-range job: no slice to borrow, so a caller can parallelise over a
/// computed index space without materialising a token array to chunk.
struct IndexJob<F> {
    start: usize,
    len: usize,
    f: *const F,
}

struct Pool {
    q: Mutex<Vec<Task>>,
    wake: Condvar,
    left: AtomicUsize,
    done: Mutex<()>,
    done_cv: Condvar,
}

/// One submitter at a time: `left` counts a single call's tasks, and two
/// concurrent calls would otherwise share the counter. Workers never take it.
static SUBMIT: Mutex<()> = Mutex::new(());
static POOL: OnceLock<Arc<Pool>> = OnceLock::new();
static POOL_SIZE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Size the persistent worker pool before any parallel kernel runs.
///
/// `available_parallelism` reports 22 logical CPUs on this hybrid part, and a
/// pool that large is worse than useless: every dispatch notifies the queue,
/// all 22 workers contend for one mutex to pop 8 tasks, and the 14 that lose go
/// back to sleep. The OS is also free to place the 8 real tasks on E-cores.
/// Sizing the pool to the caller's thread count removes both effects.
pub fn set_threads(n: usize) {
    POOL_SIZE.store(n.max(1), std::sync::atomic::Ordering::Relaxed);
}

/// Which logical processors belong to the fast cores, and pinning to them.
///
/// A hybrid part schedules a freshly spawned thread wherever it likes. An
/// efficiency core cracks every 256-bit op into two 128-bit halves, so one
/// worker placed there runs at a fraction of its siblings' speed and the whole
/// `par_index` barrier waits for it. Asking Windows for the efficiency class of
/// each physical core and pinning one worker per fast core removes that
/// straggler; it is what Colibri's `omp_tune` does by sizing to P-cores only.
///
/// Everything here fails open: an unfamiliar topology, a non-Windows build or a
/// refused affinity call leaves the pool exactly as it was, unpinned.
#[cfg(windows)]
mod placement {
    use std::ffi::c_void;

    const RELATION_PROCESSOR_CORE: u32 = 0;

    #[link(name = "kernel32")]
    extern "system" {
        fn GetLogicalProcessorInformationEx(rel: u32, buf: *mut u8, len: *mut u32) -> i32;
        fn GetCurrentThread() -> *mut c_void;
        fn SetThreadAffinityMask(thread: *mut c_void, mask: usize) -> usize;
    }

    /// Affinity masks of the highest-efficiency-class cores, one per physical
    /// core. Empty when the topology cannot be read or has no classes to split.
    pub fn performance_masks() -> Vec<u64> {
        let mut len = 0u32;
        // The first call is a size query and is expected to fail.
        unsafe { GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, std::ptr::null_mut(), &mut len) };
        if len == 0 {
            return Vec::new();
        }
        // u64 storage so the buffer meets GROUP_AFFINITY's pointer alignment.
        let mut buf = vec![0u64; (len as usize).div_ceil(8)];
        if unsafe { GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, buf.as_mut_ptr() as *mut u8, &mut len) } == 0 {
            return Vec::new();
        }
        let base = buf.as_ptr() as *const u8;
        let mut cores: Vec<(u8, u64)> = Vec::new();
        let mut at = 0usize;
        while at + 48 <= len as usize {
            // SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX: Relationship u32 @0,
            // Size u32 @4, then PROCESSOR_RELATIONSHIP @8 with Flags u8 @8,
            // EfficiencyClass u8 @9, GroupCount u16 @30, GROUP_AFFINITY @32.
            let size = unsafe { *(base.add(at + 4) as *const u32) } as usize;
            if size < 48 || at + size > len as usize {
                break;
            }
            if unsafe { *(base.add(at) as *const u32) } == RELATION_PROCESSOR_CORE
                && unsafe { *(base.add(at + 30) as *const u16) } == 1
            {
                // More than one processor group means more than 64 logical CPUs,
                // where a mask alone does not identify a core. Leave those
                // machines unpinned rather than pin to the wrong group.
                cores.push((unsafe { *base.add(at + 9) }, unsafe { *(base.add(at + 32) as *const u64) }));
            }
            at += size;
        }
        let best = match cores.iter().map(|&(e, _)| e).max() {
            Some(b) => b,
            None => return Vec::new(),
        };
        // A homogeneous part reports one class for every core. Pinning there
        // buys nothing and takes the scheduler's freedom away, so leave it be.
        if cores.iter().all(|&(e, _)| e == best) {
            return Vec::new();
        }
        cores.into_iter().filter(|&(e, _)| e == best).map(|(_, m)| m).collect()
    }

    pub fn pin(mask: u64) -> bool {
        unsafe { SetThreadAffinityMask(GetCurrentThread(), mask as usize) != 0 }
    }
}

#[cfg(not(windows))]
mod placement {
    /// No topology query here yet: Linux and macOS keep the unpinned pool.
    pub fn performance_masks() -> Vec<u64> {
        Vec::new()
    }

    pub fn pin(_mask: u64) -> bool {
        false
    }
}

fn pool() -> Arc<Pool> {
    POOL.get_or_init(|| {
        let p = Arc::new(Pool {
            q: Mutex::new(Vec::new()),
            wake: Condvar::new(),
            left: AtomicUsize::new(0),
            done: Mutex::new(()),
            done_cv: Condvar::new(),
        });
        let n = match POOL_SIZE.load(std::sync::atomic::Ordering::Relaxed) {
            0 => std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1),
            v => v,
        };
        // Pinning only pays when every worker gets a physical core to itself.
        // Eight workers on six performance cores double up on two cores'
        // hyperthread siblings, which share one set of vector ports, and the
        // barrier then waits for the doubled-up pair: measured 1.71 tok/s
        // against 2.83 unpinned on the CPU tier. Oversubscribed, leave the pool
        // to the scheduler, which can use the efficiency cores as extra cores.
        let masks = placement::performance_masks();
        let pin = masks.len() >= n;
        if pin {
            eprintln!("[threads] {n} workers pinned one per performance-core mask ({} available)", masks.len());
        } else if masks.is_empty() {
            eprintln!("[threads] {n} workers, no performance-core mask available; left unpinned");
        } else {
            eprintln!("[threads] {n} workers over {} performance cores: oversubscribed, left unpinned", masks.len());
        }
        for i in 0..n {
            let p = p.clone();
            let mask = if pin { Some(masks[i]) } else { None };
            std::thread::spawn(move || {
                if let Some(m) = mask {
                    placement::pin(m);
                }
                loop {
                    let t = {
                        let mut q = p.q.lock().unwrap();
                        loop {
                            if let Some(t) = q.pop() {
                                break t;
                            }
                            q = p.wake.wait(q).unwrap();
                        }
                    };
                    unsafe { (t.call)(t.ptr) };
                    if p.left.fetch_sub(1, Ordering::AcqRel) == 1 {
                        let _g = p.done.lock().unwrap();
                        p.done_cv.notify_one();
                    }
                }
            });
        }
        p
    })
    .clone()
}

/// Parallel loop over disjoint pieces of a mutable slice, with each piece's
/// start index.
///
/// The workers persist for the process. Spawning an OS thread per call cost
/// more than the rows it computed: a decode layer issues dozens of small gemvs,
/// and on Windows each spawn is tens of microseconds of kernel time, which
/// measured as milliseconds per layer of pure overhead.
pub fn par_rows<T: Send, F>(rows: &mut [T], threads: usize, f: F)
where
    F: Fn(&mut [T], usize) + Sync,
{
    let n = rows.len();
    if threads <= 1 || n < 2 {
        f(rows, 0);
        return;
    }
    let parts = chunk(n, threads);
    if parts.len() < 2 {
        f(rows, 0);
        return;
    }
    let mut tasks = Vec::with_capacity(parts.len());
    let mut rest = rows;
    for (a, b) in &parts {
        let (head, tail) = rest.split_at_mut(b - a);
        tasks.push(Task::new(head, *a, &f));
        rest = tail;
    }
    dispatch(tasks);
}

/// Hand one call's tasks to the pool and wait for every one of them.
///
/// The wait spins before it parks. A decode layer's tasks finish in tens of
/// microseconds, and a Windows condvar round-trip costs more than the work
/// still outstanding; parking remains the fallback so a long call does not
/// burn a core.
fn dispatch(tasks: Vec<Task>) {
    let p = pool();
    let _serial = SUBMIT.lock().unwrap();
    p.left.store(tasks.len(), Ordering::Release);
    {
        let mut q = p.q.lock().unwrap();
        let k = tasks.len();
        q.extend(tasks);
        for _ in 0..k {
            p.wake.notify_one();
        }
    }
    let mut spins = 0u32;
    while p.left.load(Ordering::Acquire) > 0 {
        if spins < 8192 {
            spins += 1;
            std::hint::spin_loop();
            continue;
        }
        let mut g = p.done.lock().unwrap();
        while p.left.load(Ordering::Acquire) > 0 {
            g = p.done_cv.wait(g).unwrap();
        }
        break;
    }
}

/// Parallel loop over `n` computed indices in disjoint runs. Same pool and
/// barrier as `par_rows`, but nothing is allocated to chunk: the index space is
/// the work, so a caller can split a matrix's rows across every worker without
/// owning a slice of them.
pub fn par_index<F>(n: usize, threads: usize, f: F)
where
    F: Fn(usize, usize) + Sync,
{
    if threads <= 1 || n < 2 {
        f(0, n);
        return;
    }
    let parts = chunk(n, threads);
    if parts.len() < 2 {
        f(0, n);
        return;
    }
    dispatch(parts.iter().map(|&(a, b)| Task::index(a, b - a, &f)).collect());
}

/// A pointer that may be shared across the workers of one `par_count` call.
/// Sound only because the caller guarantees the runs are disjoint and the
/// pointee outlives the call, which blocks until every worker is done.
///
/// The field is private on purpose: a closure that read `p.0` would capture the
/// bare pointer (edition 2021 captures fields, not whole structs) and lose the
/// `Sync` impl below. Going through `at` captures the wrapper instead.
pub struct SharedPtr<T>(*mut T);
// Manual impls: deriving would add a `T: Copy` bound, but the pointer is
// copyable whatever it points at, and callers share pointers to Vec and slices.
impl<T> Clone for SharedPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SharedPtr<T> {}
unsafe impl<T> Send for SharedPtr<T> {}
unsafe impl<T> Sync for SharedPtr<T> {}

impl<T> SharedPtr<T> {
    pub fn new(p: *mut T) -> SharedPtr<T> {
        SharedPtr(p)
    }

    pub fn at(self, i: usize) -> *mut T {
        unsafe { self.0.add(i) }
    }
}

/// Parallel loop over `n` indices in disjoint runs, for work that is not a
/// slice of its own: the delta net's per-head recurrent state, for instance.
pub fn par_count<F>(n: usize, threads: usize, f: F)
where
    F: Fn(usize, usize) + Sync,
{
    if threads <= 1 || n < 2 {
        f(0, n);
        return;
    }
    // A token slice carries the chunking so head boundaries stay whole:
    // chunking the state array itself could split one head across two workers.
    let mut tokens = vec![0u8; n];
    par_rows(&mut tokens, threads, |part, start| f(start, part.len()));
}

#[inline]
pub fn dot(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        return unsafe { avx::dot(x, y) };
    }
    let mut s = 0f32;
    for i in 0..x.len() {
        s += x[i] * y[i];
    }
    s
}

/// out[i] += w * x[i]
#[inline]
pub fn axpy(out: &mut [f32], x: &[f32], w: f32) {
    debug_assert_eq!(out.len(), x.len());
    for i in 0..out.len() {
        out[i] += w * x[i];
    }
}

/// Threads worth spawning for a `rows` x `cols` gemv. A scoped thread costs
/// tens of microseconds and a decode step calls this a dozen times, so the
/// router and the shared expert - half a million to one million multiply-adds
/// - go slower with 22 workers than with two. The floor is per thread, not
/// per call: below ~256K multiply-adds a worker does less work than it costs
/// to start one.
fn budget(threads: usize, rows: usize, cols: usize) -> usize {
    threads.min(rows).min((rows * cols) >> 18).max(1)
}

/// y = W x for a row-major W of `rows` x `cols`, written into `out` (len rows).
pub fn gemv(w: &[f32], rows: usize, cols: usize, x: &[f32], out: &mut [f32], threads: usize) {
    debug_assert_eq!(w.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(out.len(), rows);
    let t = budget(threads, rows, cols);
    par_rows(out, t, |part, start| {
        for (i, o) in part.iter_mut().enumerate() {
            let r = start + i;
            *o = dot(&w[r * cols..(r + 1) * cols], x);
        }
    });
}

/// Which two-byte format a mmap'd matrix is in. The vector kernels differ:
/// bf16 is the top half of an f32 and widens with a shift, f16 needs F16C.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Half {
    Bf16,
    F16,
}

impl Half {
    #[inline]
    pub fn decode(self, bits: u16) -> f32 {
        match self {
            Half::Bf16 => f32::from_bits((bits as u32) << 16),
            Half::F16 => crate::safetensors::f16_to_f32(bits),
        }
    }
}

/// y = W x with W left as bf16 in the mmap. Materialising an f32 copy of the
/// lm_head costs a 2 GiB write and a 2 GiB read for a tensor that is only
/// 1 GiB wide, so the dot converts each pair of bytes as it arrives instead.
pub fn gemv_bf16(w: &[u8], rows: usize, cols: usize, x: &[f32], out: &mut [f32], threads: usize) {
    gemv_half(w, rows, cols, x, out, threads, Half::Bf16);
}

/// The same for an f16 checkpoint. The two formats are both two bytes per
/// weight but decode differently, and converting f16 down to bf16 to reuse one
/// kernel would drop three mantissa bits the checkpoint still has.
pub fn gemv_f16(w: &[u8], rows: usize, cols: usize, x: &[f32], out: &mut [f32], threads: usize) {
    gemv_half(w, rows, cols, x, out, threads, Half::F16);
}

pub fn gemv_half(
    w: &[u8],
    rows: usize,
    cols: usize,
    x: &[f32],
    out: &mut [f32],
    threads: usize,
    fmt: Half,
) {
    debug_assert_eq!(w.len(), rows * cols * 2);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(out.len(), rows);
    let t = budget(threads, rows, cols);
    // The lm_head is 248320x2048 two-byte weights read straight from the mmap
    // once per token. Decoding each pair in scalar code - two byte loads, a
    // software f16 conversion with subnormal handling, one dependent add -
    // cost more than the whole rest of the decoder stack put together.
    #[cfg(target_arch = "x86_64")]
    let vec = cols >= 8
        && std::is_x86_feature_detected!("avx2")
        && (fmt == Half::Bf16 || std::is_x86_feature_detected!("f16c"));
    #[cfg(not(target_arch = "x86_64"))]
    let vec = false;
    par_rows(out, t, |part, start| {
        for (i, o) in part.iter_mut().enumerate() {
            let r = start + i;
            let row = &w[r * cols * 2..(r + 1) * cols * 2];
            *o = row_dot(row, x, fmt, vec);
        }
    });
}

/// `out[r] = scale[r] * sum_c w[r*cols+c] * x[c]`, with the weights at one byte
/// each and the activations quantised per token at 8 bits.
///
/// The byte count is the point: on the CPU tier the dense mixer projections read
/// ~2.3 GiB of bf16 per token and are bandwidth-bound, so halving the weight
/// traffic halves the block. The arithmetic has to get *cheaper* at the same
/// time or the win evaporates - an int8 weight with f32 activations needs a
/// sign-extend plus an int-to-float per 8 weights, and measured 1.9x slower than
/// the bf16 it replaced. `maddubs` does 32 weights per instruction, so both
/// operands are int8 here.
///
/// `wsum[r]` is `sum(w)` for the row, fixed at load: `maddubs` wants an unsigned
/// operand, so the token's activations go in at offset binary (`x_i8 + 128`) and
/// that costs exactly `128 * sum(w)` per row, which the kernel subtracts back out
/// in integer arithmetic.
pub fn gemv_i8(
    w: &[i8],
    wsum: &[i32],
    scale: &[f32],
    rows: usize,
    cols: usize,
    x: &[f32],
    out: &mut [f32],
    threads: usize,
) {
    debug_assert_eq!(w.len(), rows * cols);
    debug_assert_eq!(wsum.len(), rows);
    debug_assert_eq!(scale.len(), rows);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(out.len(), rows);
    let t = budget(threads, rows, cols);
    // One scale for the whole activation vector, quantised once per call and
    // shared by every row: the prep is 2048 floats against a 25 MiB matrix.
    //
    // The grid is 6 bits wide, not 8, because `maddubs` sums two products into
    // an i16 and saturates: with 7-bit weights (|w| <= 127) the activations can
    // only carry 63, otherwise 2*127*127 overflows and the row silently clamps.
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let sx = if amax > 0.0 { amax / 63.0 } else { 1.0 };
    let inv = 1.0 / sx;
    let mut xu = vec![0u8; cols];
    for (i, &v) in x.iter().enumerate() {
        // offset binary in [1, 127]: `maddubs` needs its first operand unsigned
        let q = (v * inv).round().clamp(-63.0, 63.0) as i32;
        xu[i] = (q + 64) as u8;
    }
    #[cfg(target_arch = "x86_64")]
    let vec = cols >= 32 && std::is_x86_feature_detected!("avx2");
    #[cfg(not(target_arch = "x86_64"))]
    let vec = false;
    par_rows(out, t, |part, start| {
        for (i, o) in part.iter_mut().enumerate() {
            let r = start + i;
            let row = &w[r * cols..(r + 1) * cols];
            *o = row_dot_i8(row, &xu, wsum[r], scale[r] * sx, vec);
        }
    });
}

#[inline]
fn row_dot_i8(row: &[i8], xu: &[u8], wsum: i32, scale: f32, vec: bool) -> f32 {
    #[cfg(target_arch = "x86_64")]
    let acc: i32 = if vec {
        unsafe { i8_avx::row(row.as_ptr(), xu.as_ptr(), xu.len()) }
    } else {
        scalar_i8(row, xu)
    };
    #[cfg(not(target_arch = "x86_64"))]
    let acc: i32 = scalar_i8(row, xu);
    let _ = vec;
    // undo the offset binary of the activations, in integers
    (acc - 64 * wsum) as f32 * scale
}

#[inline]
fn scalar_i8(row: &[i8], xu: &[u8]) -> i32 {
    let mut s = 0i32;
    for c in 0..row.len() {
        s += (xu[c] as i32 - 64) * row[c] as i32;
    }
    s
}

#[cfg(target_arch = "x86_64")]
mod i8_avx {
    use std::arch::x86_64::*;

    /// 32 weights per `maddubs`, 64 per iteration into two integer accumulators.
    /// Products are 8-bit x 8-bit so a pair sums to at most 2*127*127 = 32258,
    /// inside i16; each i32 lane then accumulates at most 64*4 of them.
    #[target_feature(enable = "avx2")]
    pub unsafe fn row(w: *const i8, xu: *const u8, cols: usize) -> i32 {
        let one = _mm256_set1_epi16(1);
        let mut a0 = _mm256_setzero_si256();
        let mut a1 = _mm256_setzero_si256();
        let mut k = 0usize;
        while k + 64 <= cols {
            let p0 = _mm256_maddubs_epi16(
                _mm256_loadu_si256(xu.add(k) as *const __m256i),
                _mm256_loadu_si256(w.add(k) as *const __m256i),
            );
            let p1 = _mm256_maddubs_epi16(
                _mm256_loadu_si256(xu.add(k + 32) as *const __m256i),
                _mm256_loadu_si256(w.add(k + 32) as *const __m256i),
            );
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(p0, one));
            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(p1, one));
            k += 64;
        }
        while k + 32 <= cols {
            let p = _mm256_maddubs_epi16(
                _mm256_loadu_si256(xu.add(k) as *const __m256i),
                _mm256_loadu_si256(w.add(k) as *const __m256i),
            );
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(p, one));
            k += 32;
        }
        let v = _mm256_add_epi32(a0, a1);
        let s = _mm_add_epi32(
            _mm256_castsi256_si128(v),
            _mm256_extracti128_si256::<1>(v),
        );
        let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b0100_1110>(s));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b1011_0001>(s));
        let mut total = _mm_cvtsi128_si32(s);
        while k < cols {
            total += (*xu.add(k) as i32 - 64) * *w.add(k) as i32;
            k += 1;
        }
        total
    }
}

#[cfg(test)]
mod i8_tests {
    use super::*;

    fn quantise(w: &[f32], rows: usize, cols: usize) -> (Vec<i8>, Vec<i32>, Vec<f32>) {
        let mut q = vec![0i8; rows * cols];
        let mut wsum = vec![0i32; rows];
        let mut scale = vec![0f32; rows];
        for r in 0..rows {
            let row = &w[r * cols..(r + 1) * cols];
            let amax = row.iter().fold(0f32, |m, v| m.max(v.abs()));
            let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            scale[r] = s;
            let mut sum = 0i32;
            for c in 0..cols {
                let v = (row[c] / s).round().clamp(-127.0, 127.0) as i8;
                q[r * cols + c] = v;
                sum += v as i32;
            }
            wsum[r] = sum;
        }
        (q, wsum, scale)
    }

    /// Two separate claims: the integer kernel must reproduce the dequantised
    /// matrix to f32 rounding, and the 8-bit round trip itself must stay inside
    /// its documented budget (far tighter than the int4 experts already are).
    #[test]
    fn an_int8_gemv_tracks_the_exact_dot() {
        let (rows, cols) = (9usize, 256usize);
        let mut seed = 0x9e37_79b9u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed as f64 / u64::MAX as f64) as f32 - 0.5
        };
        let w: Vec<f32> = (0..rows * cols).map(|_| next() * 0.08).collect();
        let x: Vec<f32> = (0..cols).map(|_| next() * 2.0).collect();
        let (q, wsum, scale) = quantise(&w, rows, cols);
        let mut got = vec![0f32; rows];
        gemv_i8(&q, &wsum, &scale, rows, cols, &x, &mut got, 4);
        let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
        let sx = amax / 63.0;
        for r in 0..rows {
            let dequant: f32 = (0..cols).map(|c| q[r * cols + c] as f32 * scale[r] * x[c]).sum();
            let want: f32 = (0..cols).map(|c| w[r * cols + c] * x[c]).sum();
            let kern = (got[r] - dequant).abs() / dequant.abs().max(1e-6);
            let quant = (got[r] - want).abs() / want.abs().max(1e-6);
            // The kernel quantises the activations too, so its drift against the
            // unquantised dot is the 6-bit grid, not f32 rounding.
            assert!(kern < 2e-2, "row {r}: kernel drift {kern:.2e} (activation grid {sx:.2e})");
            assert!(quant < 3e-2, "row {r}: 8-bit round trip {quant:.2e} of {want}");
        }
    }

    /// The offset-binary correction is the part that can silently go wrong: a
    /// row whose weights sum to zero must still match, and a constant row must
    /// come out exact.
    #[test]
    fn the_offset_correction_survives_a_constant_row() {
        let cols = 64usize;
        let w = vec![0.5f32; cols];
        let x: Vec<f32> = (0..cols).map(|c| (c as f32) * 0.01 - 0.3).collect();
        let (q, wsum, scale) = quantise(&w, 1, cols);
        let mut got = vec![0f32; 1];
        gemv_i8(&q, &wsum, &scale, 1, cols, &x, &mut got, 1);
        let want: f32 = (0..cols).map(|c| w[c] * x[c]).sum();
        assert!((got[0] - want).abs() / want.abs().max(1e-6) < 3e-2, "{got:?} vs {want}");
    }
}

#[inline]
fn row_dot(row: &[u8], x: &[f32], fmt: Half, vec: bool) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if vec {
        return unsafe { half_avx::row(row.as_ptr(), x.len(), x.as_ptr(), fmt) };
    }
    let _ = vec;
    // four accumulators: the add chain is the latency here, not the loads
    let (mut s0, mut s1, mut s2, mut s3) = (0f32, 0f32, 0f32, 0f32);
    let mut k = 0usize;
    while k + 3 < x.len() {
        s0 += fmt.decode(u16::from_le_bytes([row[k * 2], row[k * 2 + 1]])) * x[k];
        s1 += fmt.decode(u16::from_le_bytes([row[k * 2 + 2], row[k * 2 + 3]])) * x[k + 1];
        s2 += fmt.decode(u16::from_le_bytes([row[k * 2 + 4], row[k * 2 + 5]])) * x[k + 2];
        s3 += fmt.decode(u16::from_le_bytes([row[k * 2 + 6], row[k * 2 + 7]])) * x[k + 3];
        k += 4;
    }
    let mut s = (s0 + s1) + (s2 + s3);
    while k < x.len() {
        s += fmt.decode(u16::from_le_bytes([row[k * 2], row[k * 2 + 1]])) * x[k];
        k += 1;
    }
    s
}

/// Eight two-byte weights widened to eight f32 lanes, dot against x, summed.
///
/// bf16 needs no conversion instruction at all: it is the high half of an f32,
/// so `vpmovzxwd` + `vpslld 16` leaves the exact bit pattern in the register.
/// f16 uses F16C's `vcvtph2ps`, which is one op for eight lanes and handles
/// subnormals in hardware where the scalar decoder branches on them.
#[cfg(target_arch = "x86_64")]
mod half_avx {
    use super::Half;
    use std::arch::x86_64::*;

    /// # Safety
    /// Requires AVX2, plus F16C for `Half::F16`. `w` holds `cols` byte pairs,
    /// `x` holds `cols` floats, and `cols >= 8`.
    #[target_feature(enable = "avx2,f16c")]
    pub unsafe fn row(w: *const u8, cols: usize, x: *const f32, fmt: Half) -> f32 {
        if fmt == Half::Bf16 {
            row_fmt::<true>(w, cols, x)
        } else {
            row_fmt::<false>(w, cols, x)
        }
    }

    #[target_feature(enable = "avx2,f16c")]
    #[inline]
    unsafe fn row_fmt<const BF16: bool>(w: *const u8, cols: usize, x: *const f32) -> f32 {
        let (mut a0, mut a1, mut a2, mut a3) =
            (_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps());
        let mut k = 0usize;
        // Four independent chains: one accumulator would serialise on the
        // four-cycle add latency and leave the load ports idle.
        //
        // mul then add, not fmadd, and the same tree as `avx::dot`: bf16 and f16
        // widen losslessly, so this row must be bit-identical to the f32 row of
        // the same weights. FMA rounds once where the f32 path rounds twice,
        // which is more accurate but drifted 1.1e-6 relative rms past the golden
        // band. The loop is bandwidth-bound, so the extra multiply is free.
        while k + 32 <= cols {
            a0 = _mm256_add_ps(a0, _mm256_mul_ps(load8::<BF16>(w.add(k * 2)), _mm256_loadu_ps(x.add(k))));
            a1 = _mm256_add_ps(a1, _mm256_mul_ps(load8::<BF16>(w.add(k * 2 + 16)), _mm256_loadu_ps(x.add(k + 8))));
            a2 = _mm256_add_ps(a2, _mm256_mul_ps(load8::<BF16>(w.add(k * 2 + 32)), _mm256_loadu_ps(x.add(k + 16))));
            a3 = _mm256_add_ps(a3, _mm256_mul_ps(load8::<BF16>(w.add(k * 2 + 48)), _mm256_loadu_ps(x.add(k + 24))));
            k += 32;
        }
        while k + 8 <= cols {
            a0 = _mm256_add_ps(a0, _mm256_mul_ps(load8::<BF16>(w.add(k * 2)), _mm256_loadu_ps(x.add(k))));
            k += 8;
        }
        let s = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
        let mut lane = [0f32; 8];
        _mm256_storeu_ps(lane.as_mut_ptr(), s);
        let mut acc = ((lane[0] + lane[1]) + (lane[2] + lane[3]))
            + ((lane[4] + lane[5]) + (lane[6] + lane[7]));
        while k < cols {
            let bits = u16::from_le_bytes([*w.add(k * 2), *w.add(k * 2 + 1)]);
            let v = if BF16 { f32::from_bits((bits as u32) << 16) } else { crate::safetensors::f16_to_f32(bits) };
            acc += v * *x.add(k);
            k += 1;
        }
        acc
    }

    #[target_feature(enable = "avx2,f16c")]
    #[inline]
    unsafe fn load8<const BF16: bool>(p: *const u8) -> __m256 {
        let v = _mm_loadu_si128(p as *const __m128i);
        if BF16 {
            _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(v)))
        } else {
            _mm256_cvtph_ps(v)
        }
    }
}

/// Qwen3_5MoeRMSNorm is zero-centred: the stored weight is a delta, use (1 + w).
pub fn rmsnorm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    let mut ss = 0f64;
    for v in x {
        ss += (*v as f64) * (*v as f64);
    }
    let inv = 1.0 / (ss as f32 / n as f32 + eps).sqrt();
    for i in 0..n {
        out[i] = x[i] * inv * (1.0 + weight[i]);
    }
}

/// Qwen3_5MoeRMSNormGated: norm first, then multiply by silu(gate). The weight
/// here is a plain gain (init ones), not a delta.
pub fn rmsnorm_gated(x: &[f32], gate: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    let mut ss = 0f64;
    for v in x {
        ss += (*v as f64) * (*v as f64);
    }
    let inv = 1.0 / (ss as f32 / n as f32 + eps).sqrt();
    for i in 0..n {
        let g = gate[i];
        out[i] = weight[i] * (x[i] * inv) * silu(g);
    }
}

#[inline]
pub fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

#[inline]
pub fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

#[inline]
pub fn softplus(v: f32) -> f32 {
    if v > 20.0 {
        v
    } else {
        (1.0 + v.exp()).ln()
    }
}

pub fn l2norm(v: &mut [f32], eps: f32) {
    let mut ss = 0f32;
    for x in v.iter() {
        ss += x * x;
    }
    let inv = 1.0 / (ss + eps).sqrt();
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// Full softmax over `logits`, top-k, renormalise the k weights.
/// Returns the k weights (descending) and their ids, as the reference does.
pub fn softmax_topk(logits: &[f32], k: usize, weights: &mut [f32], ids: &mut [usize]) {
    let m = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum = 0f32;
    let mut p: Vec<f32> = logits.iter().map(|&v| (v - m).exp()).collect();
    for &v in &p {
        sum += v;
    }
    for v in p.iter_mut() {
        *v /= sum;
    }
    let mut order: Vec<usize> = (0..p.len()).collect();
    order.sort_by(|&a, &b| {
        p[b]
            .partial_cmp(&p[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let mut acc = 0f32;
    for i in 0..k {
        acc += p[order[i]];
    }
    for i in 0..k {
        weights[i] = p[order[i]] / acc;
        ids[i] = order[i];
    }
}

/// Partial NeoX-style rope over the first `rotary` dims: pairs (i, i+rotary/2).
/// The trailing `head_dim - rotary` dimensions pass through untouched.
pub fn rope_apply(v: &mut [f32], cos: &[f32], sin: &[f32], rotary: usize) {
    let half = rotary / 2;
    for i in 0..half {
        let a = v[i];
        let b = v[i + half];
        v[i] = a * cos[i] - b * sin[i];
        v[i + half] = b * cos[i + half] + a * sin[i + half];
    }
}

/// cos/sin exactly as Qwen3_5MoeTextRotaryEmbedding produces them for text:
/// inv_freq over arange(0, rotary, 2) with base `theta`, then cat(f, f) so the
/// table length is `rotary` and the second half repeats the first.
pub fn rope_tables(theta: f64, rotary: usize, max_pos: usize, cos: &mut Vec<f32>, sin: &mut Vec<f32>) {
    let nf = rotary / 2;
    let inv: Vec<f64> = (0..nf)
        .map(|i| (-((2 * i) as f64) / (rotary as f64)) * theta.ln())
        .map(|e| e.exp())
        .collect();
    cos.clear();
    sin.clear();
    cos.reserve(max_pos * rotary);
    sin.reserve(max_pos * rotary);
    for p in 0..max_pos {
        let mut row_c = Vec::with_capacity(nf);
        let mut row_s = Vec::with_capacity(nf);
        for i in 0..nf {
            let f = p as f64 * inv[i];
            row_c.push(f.cos() as f32);
            row_s.push(f.sin() as f32);
        }
        for _ in 0..2 {
            for i in 0..nf {
                cos.push(row_c[i]);
                sin.push(row_s[i]);
            }
        }
    }
}

/// Causal depthwise conv1d: `x[t]` is one row of `channels`, `w` is [C][K],
/// `past` is the K-1 rows before the window, oldest first (empty for prefill).
/// Output is [seq][C] *before* the activation, matching F.conv1d(..., padding=K-1)[:, :, :S].
pub fn conv_depthwise(
    x: &[&[f32]],
    w: &[f32],
    channels: usize,
    kernel: usize,
    past: &[f32],
    out: &mut [f32],
) {
    let hist = kernel - 1;
    for (t, o) in out.chunks_mut(channels).enumerate().take(x.len()) {
        for c in 0..channels {
            let mut acc = 0f32;
            for k in 0..kernel {
                let src = t as i64 + k as i64 - hist as i64;
                let v = if src >= 0 {
                    x[src as usize][c]
                } else {
                    let p = hist as i64 + src; // 0..=hist-1, oldest first
                    if (p as usize + 1) * channels <= past.len() {
                        past[p as usize * channels + c]
                    } else {
                        0.0
                    }
                };
                acc += v * w[c * kernel + k];
            }
            o[c] = acc;
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2")]
    pub unsafe fn dot(x: &[f32], y: &[f32]) -> f32 {
        let n = x.len();
        let mut i = 0usize;
        // One accumulator serialises on the four-cycle add latency: a 2048-wide
        // row ran 256 dependent adds back to back while the load ports idled.
        // Four chains overlap them, and the tree at the end is both shorter and
        // better conditioned than the left-to-right lane sum it replaces.
        let (mut a0, mut a1, mut a2, mut a3) =
            (_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps());
        while i + 32 <= n {
            let p = x.as_ptr();
            let q = y.as_ptr();
            a0 = _mm256_add_ps(a0, _mm256_mul_ps(_mm256_loadu_ps(p.add(i)), _mm256_loadu_ps(q.add(i))));
            a1 = _mm256_add_ps(a1, _mm256_mul_ps(_mm256_loadu_ps(p.add(i + 8)), _mm256_loadu_ps(q.add(i + 8))));
            a2 = _mm256_add_ps(a2, _mm256_mul_ps(_mm256_loadu_ps(p.add(i + 16)), _mm256_loadu_ps(q.add(i + 16))));
            a3 = _mm256_add_ps(a3, _mm256_mul_ps(_mm256_loadu_ps(p.add(i + 24)), _mm256_loadu_ps(q.add(i + 24))));
            i += 32;
        }
        while i + 8 <= n {
            a0 = _mm256_add_ps(a0, _mm256_mul_ps(_mm256_loadu_ps(x.as_ptr().add(i)), _mm256_loadu_ps(y.as_ptr().add(i))));
            i += 8;
        }
        let acc = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
        let mut t = [0f32; 8];
        _mm256_storeu_ps(t.as_mut_ptr(), acc);
        let mut s: f32 = ((t[0] + t[1]) + (t[2] + t[3])) + ((t[4] + t[5]) + (t[6] + t[7]));
        while i < n {
            s += x[i] * y[i];
            i += 1;
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// f16 patterns whose f32 values are exact, spanning zero, the subnormal
    /// range and the largest finite half. The subnormals are the point: an
    /// unpack that only rebases the exponent field silently flushes them.
    const F16: [(u16, f32); 8] = [
        (0x0000, 0.0),
        (0x8000, -0.0),
        (0x3C00, 1.0),
        (0xBC00, -1.0),
        (0x0400, 6.103515625e-5),
        (0x03FF, 6.097555160522461e-5),
        (0x0001, 5.960464477539063e-8),
        (0x7BFF, 65504.0),
    ];

    #[test]
    fn the_placement_query_returns_usable_masks_or_nothing() {
        // Runs on every build host, so it cannot assert a hybrid topology: a
        // homogeneous desktop and a non-Windows build both legitimately return
        // empty. What it can assert is that whatever comes back is a real mask.
        let masks = placement::performance_masks();
        let mut seen = std::collections::HashSet::new();
        for &m in &masks {
            assert_ne!(m, 0, "an empty affinity mask would pin a worker nowhere");
            assert!(seen.insert(m), "duplicate core mask {m:#x}");
        }
        println!("  {} performance-core masks: {:?}", masks.len(),
            masks.iter().map(|m| format!("{m:#x}")).collect::<Vec<_>>());
    }

    #[test]
    fn a_wide_half_gemv_matches_an_f64_dot_in_both_formats() {
        // 2051 columns exercises the 32-wide loop, the 8-wide remainder and the
        // scalar tail. The checkpoint's 2048 is a whole number of every chunk,
        // so an off-by-one in the vector bounds would never show up there.
        let (rows, cols) = (3usize, 2051usize);
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.13).sin() * 0.75).collect();
        for fmt in [Half::Bf16, Half::F16] {
            let mut w = vec![0u8; rows * cols * 2];
            let mut exact = vec![0f32; rows * cols];
            for k in 0..rows * cols {
                let (bits, val) = F16[k % F16.len()];
                // bf16 has eight mantissa bits to f16's eleven, so truncating
                // an f16 pattern loses precision. The oracle is what the format
                // actually stores, not what was offered to it.
                let stored = match fmt {
                    Half::Bf16 => (val.to_bits() >> 16) as u16,
                    Half::F16 => bits,
                };
                exact[k] = fmt.decode(stored);
                w[k * 2..k * 2 + 2].copy_from_slice(&stored.to_le_bytes());
            }
            for threads in [1usize, 4] {
                let mut got = vec![0f32; rows];
                match fmt {
                    Half::Bf16 => gemv_bf16(&w, rows, cols, &x, &mut got, threads),
                    Half::F16 => gemv_f16(&w, rows, cols, &x, &mut got, threads),
                }
                for r in 0..rows {
                    let row = &exact[r * cols..(r + 1) * cols];
                    let want: f64 = row.iter().zip(&x).map(|(a, b)| *a as f64 * *b as f64).sum();
                    let terms: f64 = row.iter().zip(&x).map(|(a, b)| (*a as f64 * *b as f64).abs()).sum();
                    let err = (got[r] as f64 - want).abs();
                    assert!(err <= 1e-6 * terms,
                        "{fmt:?} threads {threads} row {r}: abs {err:.3e} over terms {terms:.3e}");
                }
            }
        }
    }

    #[test]
    fn an_f16_gemv_is_the_f32_gemv_of_the_same_weights() {
        let cols = F16.len();
        let rows = 5;
        let (mut w, mut f) = (Vec::new(), Vec::new());
        for r in 0..rows {
            for c in 0..cols {
                let (bits, val) = F16[(c + r) % cols];
                w.extend_from_slice(&bits.to_le_bytes());
                f.push(val);
            }
        }
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 - 3.0) * 0.25).collect();
        for threads in [1, 4] {
            let mut got = vec![0f32; rows];
            gemv_f16(&w, rows, cols, &x, &mut got, threads);
            for r in 0..rows {
                let row = &f[r * cols..(r + 1) * cols];
                let want: f32 = row.iter().zip(&x).map(|(a, b)| a * b).sum();
                let scale: f32 = row.iter().zip(&x).map(|(a, b)| (a * b).abs()).sum();
                assert!(
                    (got[r] - want).abs() <= 1e-5 * scale.max(1e-30),
                    "threads {threads} row {r}: {} vs {want}",
                    got[r]
                );
            }
        }
    }

    #[test]
    fn a_half_gemv_is_bit_identical_to_the_f32_gemv_of_the_widened_weights() {
        // The dense projections are stored in the checkpoint's own precision and
        // widened inside the vector loop instead of at load. That is only a
        // bandwidth optimisation if it changes no arithmetic: bf16 and f16 widen
        // losslessly, so the same weights must give the same bits, not merely a
        // close answer. 2051 columns covers the 32-wide loop, the 8-wide
        // remainder and the scalar tail; 2048 is the shipped hidden size.
        let rows = 3usize;
        for cols in [2048usize, 2051] {
            let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.13).sin() * 0.75).collect();
            for fmt in [Half::Bf16, Half::F16] {
                let mut w = vec![0u8; rows * cols * 2];
                let mut exact = vec![0f32; rows * cols];
                for k in 0..rows * cols {
                    let (bits, val) = F16[k % F16.len()];
                    // The oracle is what the format stores: bf16 has eight
                    // mantissa bits to f16's eleven, so some of these patterns
                    // truncate.
                    let stored = match fmt {
                        Half::Bf16 => (val.to_bits() >> 16) as u16,
                        Half::F16 => bits,
                    };
                    exact[k] = fmt.decode(stored);
                    w[k * 2..k * 2 + 2].copy_from_slice(&stored.to_le_bytes());
                }
                for threads in [1usize, 4] {
                    let mut got = vec![0f32; rows];
                    let mut want = vec![0f32; rows];
                    gemv_half(&w, rows, cols, &x, &mut got, threads, fmt);
                    gemv(&exact, rows, cols, &x, &mut want, threads);
                    for r in 0..rows {
                        assert_eq!(
                            got[r].to_bits(), want[r].to_bits(),
                            "{fmt:?} cols {cols} threads {threads} row {r}: {} vs {}",
                            got[r], want[r]
                        );
                    }
                }
            }
        }
    }
}
