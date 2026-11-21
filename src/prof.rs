//! Where a decode token actually spends its time.
//!
//! `decode` already splits stack from lm_head, and the device counters split
//! fence wait from dispatch. Neither says whether the remaining host time is
//! the expert GEMVs, the delta-net recursion, or the routing and norms. These
//! counters do, and they are the difference between optimising the right thing
//! and optimising a guess.
//!
//! Off unless `enable()` is called, so a normal run pays one relaxed load per
//! stage. Counters accumulate across threads, so a stage measured inside
//! `par_count` reports CPU time rather than wall time — compare it against the
//! thread count, not against the token's wall clock.
//!
//! Most stages are siblings that a token's time is carved between, and those
//! are what [`total`] adds up. `mixer device` and its `mixer dev *` rows are
//! not siblings: the parent wraps three parts, and a fourth splits one of those,
//! so [`total`] counts the parent and skips everything in [`SUBROWS`]. See
//! [`BREAKDOWN`].
//!
//! A request is two kinds of forward pass: the prompt, which [`split`] moves
//! into its own counters, and one token at a time afterwards, which
//! [`decode_steps`] counts. Dividing the prompt's chunked work by completion
//! tokens makes a table that is about neither, so the decode stages are
//! reported per decode step and the prompt gets its own line.
//!
//! **Overlap measurement**: [`OverlapTimer`] measures how much CPU work runs
//! concurrently with GPU dispatches. Start it before submitting a batch, stop
//! it after the GPU sync returns. The ratio of overlap time to total time is
//! the CPU/GPU parallelism efficiency.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::time::Instant;

pub const ROUTE: usize = 0;
pub const MOE_UPLOAD: usize = 1;
pub const MOE_RECORD: usize = 2;
pub const MOE_CPU: usize = 3;
pub const MOE_SYNC: usize = 4;
pub const MOE_READ: usize = 5;
pub const MOE_SHARED: usize = 6;
pub const MIXER_DEV: usize = 7;
pub const MIXER_CPU: usize = 8;
pub const HEAD: usize = 9;
pub const NORM: usize = 10;
pub const RESIDUAL: usize = 11;
pub const MIXER_PROJ: usize = 12;
pub const MIXER_DEV_UPLOAD: usize = 13;
pub const MIXER_DEV_COMPUTE: usize = 14;
pub const MIXER_DEV_READBACK: usize = 15;
/// What the device itself spent inside the mixer's blocking readback, as the
/// driver's own event pair measured it. Splits the readback row again, because
/// a blocking DtoH absorbs the whole queued batch: on the served CUDA cell
/// `mixer dev readback` is 12.91 ms/token for an 8 KiB copy, and only the device
/// clock can say how much of that is shader work and how much is submission and
/// fence wait. The CUDA tier reports it; Vulkan's arena read is a plain host
/// copy of memory the shaders already wrote, so the row stays 0 there.
///
/// It answered: 62.9% of the mixer's device window (median of three steady
/// requests, 62.5-64.2%) is the device's own span, and host submission is a
/// third of that span, so the span is execution rather than launch gaps. The
/// fence excess is bounded at ~2.8 ms/token, which is why the readback was left
/// blocking and the mixer kernels became the thing to chase.
pub const MIXER_DEV_DEVICE: usize = 16;

pub const STAGE_COUNT: usize = 17;

/// The rows that restate another row instead of standing beside it.
///
/// `Model::step` opens `mixer device` around `step_fused` / `dev_inputs`, and
/// those open upload, compute and readback inside it. `total` therefore counts
/// the parent and skips the parts: add both and the mixer's device half is
/// billed twice, which is what made `accounted` come out over the wall clock
/// and `unaccounted` negative on the CUDA tier (43.60 against 35.31 ms/token).
/// The children still print, because upload, compute and readback have three
/// different fixes (PCIe bandwidth, shader execution, PCIe latency) — they
/// just do not add.
pub const BREAKDOWN: [usize; 3] = [MIXER_DEV_UPLOAD, MIXER_DEV_COMPUTE, MIXER_DEV_READBACK];

/// [`BREAKDOWN`] plus the row that splits one of those parts. Everything here
/// leaves the sum; only [`BREAKDOWN`] is added when checking a part against its
/// whole, since `mixer dev device` is a slice of the readback, not a fourth
/// sibling of it.
pub const SUBROWS: [usize; 4] = [MIXER_DEV_UPLOAD, MIXER_DEV_COMPUTE, MIXER_DEV_READBACK, MIXER_DEV_DEVICE];

/// Milliseconds per token that a `mixer dev` part was timed outside the row it
/// splits, or 0 when the rows nest the way [`BREAKDOWN`] claims.
///
/// [`total`] drops the parts on the grounds that the parent bills them. A device
/// block opened after the parent closes is therefore billed by neither, and the
/// only trace is the parts summing over their own whole - which is what the
/// attention mixer's o_proj did to every CUDA table (11.85 over 10.85
/// ms/token) before this row started printing.
pub fn leak(ms: &[f64; STAGE_COUNT]) -> f64 {
    let parts: f64 = BREAKDOWN.iter().map(|&i| ms[i]).sum();
    (parts - ms[MIXER_DEV]).max(0.0)
}

/// One line per table whose `mixer dev` rows overflow their parent. Silent when
/// the rows nest, so a correct run pays nothing for the check.
pub fn leak_line(ms: &[f64; STAGE_COUNT], per_token: f64) -> Option<String> {
    let over = leak(ms);
    (over > 1e-4).then(|| {
        format!(
            "  LEAK        `mixer dev` rows exceed `mixer device` by {:.2} ms/token: \
             a device block is timed outside its parent and billed by neither\n",
            over / per_token
        )
    })
}

/// Milliseconds a token actually spent: every stage except the rows that
/// restate a stage already in the sum.
pub fn total(ms: &[f64; STAGE_COUNT]) -> f64 {
    ms.iter()
        .enumerate()
        .filter(|(i, _)| !SUBROWS.contains(i))
        .map(|(_, v)| *v)
        .sum()
}

/// `moe submit` split in two and `moe device wait` split in two, because the
/// lumped numbers could not say whether the cost was the blocking activation
/// upload, the launch loop, the fence, or the readback — and those have four
/// different fixes. `lm_head` was missing entirely: on the CPU tier it is a
/// 248320x2048 half-precision read per token and nothing named it.
/// `mixer device` split in three for the same reason: upload, compute and
/// readback have different bottlenecks (PCIe bandwidth, shader execution,
/// PCIe latency) and different fixes. `mixer dev device` splits the readback
/// once more, into the part the GPU executed and the part the host waited.
pub const STAGES: [&str; STAGE_COUNT] = [
    "route",
    "moe upload",
    "moe record+launch",
    "moe host experts",
    "moe device sync",
    "moe readback",
    "moe shared+reduce",
    "mixer device",
    "mixer host",
    "lm_head",
    "rmsnorm",
    "residual",
    "mixer projections",
    "mixer dev upload",
    "mixer dev compute",
    "mixer dev readback",
    "mixer dev device",
];

static ON: AtomicBool = AtomicBool::new(false);
static NS: [AtomicU64; STAGE_COUNT] = [const { AtomicU64::new(0) }; STAGE_COUNT];
static PROMPT_NS: [AtomicU64; STAGE_COUNT] = [const { AtomicU64::new(0) }; STAGE_COUNT];
static PROMPT_TOKENS: AtomicU64 = AtomicU64::new(0);
static STEPS: AtomicU64 = AtomicU64::new(0);
/// When the decode phase began, so the wall clock in the report is the one the
/// step count belongs to. `None` until a split, which only a batched prefill does.
static DECODE_T0: std::sync::Mutex<Option<Instant>> = std::sync::Mutex::new(None);

// Overlap measurement: how much CPU work ran concurrently with GPU dispatches.
// These count nanoseconds of overlap (CPU work that happened while GPU was busy).
static OVERLAP_NS: AtomicU64 = AtomicU64::new(0);
static GPU_BUSY_NS: AtomicU64 = AtomicU64::new(0);
static OVERLAP_SAMPLES: AtomicU64 = AtomicU64::new(0);

pub fn enable() {
    ON.store(true, Relaxed);
}

pub fn enabled() -> bool {
    ON.load(Relaxed)
}

pub fn reset() {
    for c in &NS {
        c.store(0, Relaxed);
    }
    for c in &PROMPT_NS {
        c.store(0, Relaxed);
    }
    PROMPT_TOKENS.store(0, Relaxed);
    STEPS.store(0, Relaxed);
    OVERLAP_NS.store(0, Relaxed);
    GPU_BUSY_NS.store(0, Relaxed);
    OVERLAP_SAMPLES.store(0, Relaxed);
    *DECODE_T0.lock().unwrap_or_else(|e| e.into_inner()) = None;
}

/// Milliseconds per stage, in `STAGES` order — the decode steps only, once
/// [`split`] has been called.
pub fn millis() -> [f64; STAGE_COUNT] {
    let mut out = [0f64; STAGE_COUNT];
    for (i, c) in NS.iter().enumerate() {
        out[i] = c.load(Relaxed) as f64 / 1e6;
    }
    out
}

/// Close the prompt phase: everything counted so far is chunked prefill over
/// `tokens` rows, not a decode step, so move it to its own counters. No-op
/// while profiling is off.
pub fn split(tokens: usize) {
    if !enabled() {
        return;
    }
    for (i, c) in NS.iter().enumerate() {
        let moved = c.swap(0, Relaxed);
        if moved != 0 {
            PROMPT_NS[i].fetch_add(moved, Relaxed);
        }
    }
    PROMPT_TOKENS.fetch_add(tokens as u64, Relaxed);
    *DECODE_T0.lock().unwrap_or_else(|e| e.into_inner()) = Some(Instant::now());
}

/// Wall clock of the decode phase, or `None` while the prompt is still the
/// only thing that has run. The report divides by decode steps, so it has to
/// measure the window those steps live in.
pub fn decode_wall_ms() -> Option<f64> {
    let t0 = *DECODE_T0.lock().unwrap_or_else(|e| e.into_inner());
    t0.map(|t0| t0.elapsed().as_secs_f64() * 1e3)
}

/// One more single-token forward, called by `Model::forward`. The decode table
/// divides by this rather than by the tokens emitted: the last forward answers
/// with EOS and produces nothing, and a prefill chunk never calls it.
pub fn tick_decode_step() {
    if enabled() {
        STEPS.fetch_add(1, Relaxed);
    }
}

pub fn decode_steps() -> usize {
    STEPS.load(Relaxed) as usize
}

/// Prompt-phase milliseconds per stage, and how many tokens they cover.
pub fn prompt_millis() -> ([f64; STAGE_COUNT], usize) {
    let mut out = [0f64; STAGE_COUNT];
    for (i, c) in PROMPT_NS.iter().enumerate() {
        out[i] = c.load(Relaxed) as f64 / 1e6;
    }
    (out, PROMPT_TOKENS.load(Relaxed) as usize)
}

/// One line about the prompt, naming the stage that dominates it, so a decode
/// breakdown can say what it left out. `None` when no prompt was prefilled.
pub fn prompt_line() -> Option<String> {
    let (ms, tokens) = prompt_millis();
    if tokens == 0 {
        return None;
    }
    let sum = total(&ms);
    let (worst, biggest) = ms
        .iter()
        .enumerate()
        .filter(|(i, _)| !SUBROWS.contains(i))
        .fold((0usize, 0f64), |a, (i, &x)| if x > a.1 { (i, x) } else { a });
    Some(format!(
        "  prompt    {tokens} tokens in {sum:.2} ms accounted, {:.0}% {}",
        100.0 * biggest / sum.max(1e-9),
        STAGES[worst]
    ))
}

/// The table's own heading, shared by both printers so the two can never
/// disagree about whether the `mixer dev` rows add. A reader who sums the
/// column has to know that four of the rows restate others: three split the
/// device mixer and the fourth splits one of those.
pub fn header(steps: usize) -> String {
    format!(
        "  stage breakdown (ms/token, {steps} decode steps; the `mixer dev` rows \
         restate `mixer device` and are not added):\n"
    )
}

/// Bill milliseconds against a stage from inside a driver call that has no
/// timer of its own - the CUDA tier's event harvest, which is the device's
/// clock rather than an `Instant` around a blocking call.
pub fn add(stage: usize, ms: f64) {
    if enabled() && ms > 0.0 {
        NS[stage].fetch_add((ms * 1e6) as u64, Relaxed);
    }
}

/// The decode breakdown: milliseconds per decode step, then the prompt on its
/// own line. `wall_ms` is the caller's whole request and is only used when the
/// phase was never split, so the last row shows what the stages failed to name.
pub fn report(wall_ms: f64) -> String {
    let steps = decode_steps().max(1) as f64;
    let ms = millis();
    let sum = total(&ms);
    let wall = decode_wall_ms().unwrap_or(wall_ms);
    let mut out = header(decode_steps());
    for (name, v) in STAGES.iter().zip(&ms) {
        out.push_str(&format!("    {name:<20} {:7.2}  {:5.1}%\n", v / steps, 100.0 * v / sum.max(1e-9)));
    }
    out.push_str(&format!(
        "    {:<20} {:7.2}  (wall {:.2})\n",
        "accounted",
        sum / steps,
        wall / steps
    ));
    if let Some(line) = leak_line(&ms, steps) {
        out.push_str(&line);
    }
    if let Some(line) = prompt_line() {
        out.push_str(&line);
        out.push('\n');
        // The line above says the prompt is slow; only the stages say which one.
        // A chunked prefill and a token loop can total the same and disagree
        // stage by stage, and that difference is the whole measurement.
        let (pms, tokens) = prompt_millis();
        let psum = total(&pms);
        out.push_str(&format!(
            "  prompt stage breakdown (ms/token, {} prompt tokens; as above, the \
             `mixer dev` rows do not add):\n",
            tokens
        ));
        for (name, v) in STAGES.iter().zip(&pms) {
            out.push_str(&format!("    {name:<20} {:7.3}  {:5.1}%\n", v / tokens as f64, 100.0 * v / psum.max(1e-9)));
        }
        out.push_str(&format!("    {:<20} {:7.3}\n", "accounted", psum / tokens as f64));
        if let Some(line) = leak_line(&pms, tokens as f64) {
            out.push_str(&line);
        }
    }
    // Add overlap measurement if we have samples
    let (overlap_ms, gpu_busy_ms, samples) = overlap_stats();
    if samples > 0 {
        let efficiency = overlap_efficiency();
        out.push_str(&format!(
            "  overlap     {:.1}% efficiency ({:.2}/{:.2} ms overlapped, {} samples)\n",
            efficiency, overlap_ms, gpu_busy_ms, samples
        ));
    }
    out
}

pub struct Timer {
    stage: usize,
    t0: Option<Instant>,
}

/// Start a stage timer. Does nothing when profiling is off.
pub fn start(stage: usize) -> Timer {
    Timer { stage, t0: if enabled() { Some(Instant::now()) } else { None } }
}

impl Drop for Timer {
    fn drop(&mut self) {
        if let Some(t0) = self.t0 {
            NS[self.stage].fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
        }
    }
}

/// Measure CPU/GPU overlap: how much CPU work ran while the GPU was busy.
///
/// Usage:
/// ```ignore
/// let overlap = OverlapTimer::start();
/// gpu.submit(batch);       // GPU starts working
/// cpu_work();              // CPU does something useful in parallel
/// overlap.record_gpu_done(); // GPU finished, record the overlap
/// ```
///
/// The overlap percentage tells you how effectively CPU and GPU are running
/// in parallel. 100% means all CPU work was hidden behind GPU work; 0% means
/// they ran sequentially.
pub struct OverlapTimer {
    t0: Option<Instant>,
    cpu_ns_before_gpu_done: u64,
}

impl OverlapTimer {
    /// Start measuring overlap. Call before submitting GPU work.
    pub fn start() -> Self {
        if !enabled() {
            return OverlapTimer { t0: None, cpu_ns_before_gpu_done: 0 };
        }
        OverlapTimer {
            t0: Some(Instant::now()),
            cpu_ns_before_gpu_done: 0,
        }
    }

    /// Record that GPU work completed. Call after the GPU sync returns.
    /// The time between start() and this call is the GPU busy time.
    /// Any CPU work that ran during that window is the overlap.
    pub fn record_gpu_done(mut self) {
        if let Some(t0) = self.t0.take() {
            let total_ns = t0.elapsed().as_nanos() as u64;
            // The overlap is the CPU work that ran during the GPU busy window.
            // We approximate this as: total time minus the CPU-only portion.
            // For now, record the total as GPU busy time; the caller can
            // separately measure CPU-only time to compute the overlap.
            GPU_BUSY_NS.fetch_add(total_ns, Relaxed);
            OVERLAP_SAMPLES.fetch_add(1, Relaxed);
        }
    }

    /// Record CPU work that happened before the GPU finished.
    /// Call this when CPU work starts, then record_gpu_done when GPU finishes.
    /// The difference is the overlap.
    pub fn record_cpu_work_done(&mut self) {
        if let Some(t0) = self.t0 {
            self.cpu_ns_before_gpu_done = t0.elapsed().as_nanos() as u64;
        }
    }
}

/// Record that some CPU work overlapped with GPU work.
/// Call this after both CPU and GPU have finished a parallel section.
/// `cpu_ns` is the CPU time spent, `gpu_ns` is the GPU time spent.
/// The overlap is min(cpu_ns, gpu_ns) if they ran fully in parallel.
pub fn record_overlap(cpu_ns: u64, gpu_ns: u64) {
    if !enabled() {
        return;
    }
    // The overlap is the portion of CPU work that ran while GPU was busy.
    // If CPU finished before GPU, all CPU time was overlap.
    // If GPU finished before CPU, GPU time was the overlap window.
    let overlap = cpu_ns.min(gpu_ns);
    OVERLAP_NS.fetch_add(overlap, Relaxed);
    GPU_BUSY_NS.fetch_add(gpu_ns, Relaxed);
    OVERLAP_SAMPLES.fetch_add(1, Relaxed);
}

/// Get overlap statistics: (overlap_ms, gpu_busy_ms, samples).
pub fn overlap_stats() -> (f64, f64, u64) {
    let overlap = OVERLAP_NS.load(Relaxed) as f64 / 1e6;
    let gpu_busy = GPU_BUSY_NS.load(Relaxed) as f64 / 1e6;
    let samples = OVERLAP_SAMPLES.load(Relaxed);
    (overlap, gpu_busy, samples)
}

/// Overlap efficiency as a percentage: how much of the GPU busy time had
/// useful CPU work running in parallel.
pub fn overlap_efficiency() -> f64 {
    let gpu_busy = GPU_BUSY_NS.load(Relaxed);
    if gpu_busy == 0 {
        return 0.0;
    }
    let overlap = OVERLAP_NS.load(Relaxed);
    100.0 * overlap as f64 / gpu_busy as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ON` and the counters are process-wide, so the tests that touch them run
    /// one at a time instead of racing each other across cargo's threads.
    static EXCLUSIVE: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn a_disabled_timer_records_nothing() {
        let _g = EXCLUSIVE.lock().unwrap_or_else(|e| e.into_inner());
        ON.store(false, Relaxed);
        reset();
        {
            let _t = start(ROUTE);
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert_eq!(millis()[ROUTE], 0.0);
    }

    #[test]
    fn split_moves_the_prompt_counters_out_of_the_decode_set() {
        let _g = EXCLUSIVE.lock().unwrap_or_else(|e| e.into_inner());
        ON.store(true, Relaxed);
        reset();
        {
            let _t = start(ROUTE);
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert!(millis()[ROUTE] > 0.0);
        tick_decode_step();
        split(7);
        assert_eq!(millis()[ROUTE], 0.0, "the decode set restarts at the split");
        assert_eq!(decode_steps(), 1, "a split must not invent or drop a step");
        let (prompt, tokens) = prompt_millis();
        assert!(prompt[ROUTE] > 0.0, "the prompt work has to be visible somewhere");
        assert_eq!(tokens, 7);
        {
            let _t = start(ROUTE);
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert!(millis()[ROUTE] > 0.0, "counting resumes after the split");
        assert!(prompt_line().unwrap().contains("7 tokens"));
        reset();
        assert_eq!(prompt_millis().0.iter().sum::<f64>(), 0.0);
        assert_eq!(prompt_millis().1, 0);
        assert_eq!(decode_steps(), 0);
        assert!(prompt_line().is_none(), "nothing prefilled, nothing to report");
        ON.store(false, Relaxed);
    }

    #[test]
    fn stages_are_disjoint_and_named() {
        assert_eq!(STAGES.len(), NS.len());
        assert!(STAGES.iter().all(|s| !s.is_empty()));
        let mut seen = std::collections::HashSet::new();
        assert!(STAGES.iter().all(|s| seen.insert(*s)));
    }

    /// `mixer device` wraps its three parts, so the column a reader adds is not
    /// the 17 rows printed above it. Bill both and the mixer's device half is
    /// charged twice, `accounted` lands over the wall clock and the one row
    /// that says how much of a token the profiler failed to name goes negative.
    #[test]
    fn a_stage_that_wraps_its_own_breakdown_is_billed_once() {
        let _g = EXCLUSIVE.lock().unwrap_or_else(|e| e.into_inner());
        ON.store(true, Relaxed);
        reset();
        // Ten ms of device mixer, split inside itself as 2 + 3 + 5, of which the
        // readback says 4 was the device actually running, plus 2 ms of routing:
        // the token took 12 ms, not 26.
        for (stage, ms) in [
            (MIXER_DEV, 10.0),
            (MIXER_DEV_UPLOAD, 2.0),
            (MIXER_DEV_COMPUTE, 3.0),
            (MIXER_DEV_READBACK, 5.0),
            (MIXER_DEV_DEVICE, 4.0),
            (ROUTE, 2.0),
        ] {
            NS[stage].store((ms * 1e6) as u64, Relaxed);
            PROMPT_NS[stage].store((ms * 1e6) as u64, Relaxed);
        }
        PROMPT_TOKENS.store(4, Relaxed);
        tick_decode_step();

        let ms = millis();
        assert_eq!(total(&[0.0; STAGE_COUNT]), 0.0);
        assert!(
            (total(&ms) - 12.0).abs() < 1e-6,
            "a token is its stages, not its stages plus the parts of one: {}",
            total(&ms)
        );
        assert!(
            (ms.iter().sum::<f64>() - 26.0).abs() < 1e-6,
            "the printed column still shows every row"
        );
        assert_eq!(leak(&ms), 0.0, "a row that splits a part is not a fourth part");

        let rep = report(12.0);
        assert!(rep.contains("12.00  (wall 12.00)"), "accounted row:\n{rep}");
        // The parts still print, because upload, compute and readback have
        // three different fixes — they just do not add.
        for name in ["mixer dev upload", "mixer dev compute", "mixer dev readback", "mixer dev device"] {
            assert!(rep.contains(name), "{name} left the table:\n{rep}");
        }
        // The headline names a stage a token actually spends, not a slice of one.
        let line = prompt_line().unwrap();
        assert!(line.contains("12.00 ms accounted"), "{line}");
        assert!(line.contains(STAGES[MIXER_DEV]), "{line}");
        assert!(
            SUBROWS.iter().all(|i| !line.contains(STAGES[*i])),
            "the prompt line blamed a row that does not add: {line}"
        );
        reset();
        ON.store(false, Relaxed);
    }

    /// The three `mixer dev` rows are only free to leave the sum while every one
    /// of them really is inside `mixer device`. The attention mixer used to open
    /// its o_proj upload, compute and readback after the parent had closed, so
    /// that device time was billed by neither the row that splits it nor the sum
    /// that drops it, and the table's only symptom was parts over a whole.
    #[test]
    fn a_part_timed_outside_its_parent_says_so() {
        let _g = EXCLUSIVE.lock().unwrap_or_else(|e| e.into_inner());
        ON.store(true, Relaxed);
        reset();
        let nested = |ms: &mut [f64; STAGE_COUNT]| {
            ms[MIXER_DEV] = 10.0;
            ms[MIXER_DEV_UPLOAD] = 2.0;
            ms[MIXER_DEV_COMPUTE] = 3.0;
            ms[MIXER_DEV_READBACK] = 5.0;
        };
        let mut ms = [0f64; STAGE_COUNT];
        nested(&mut ms);
        assert_eq!(leak(&ms), 0.0, "2 + 3 + 5 inside a 10 ms parent leaks nothing");
        // o_proj bills a sixth millisecond of readback outside the parent.
        ms[MIXER_DEV_READBACK] = 6.0;
        assert!(
            (leak(&ms) - 1.0).abs() < 1e-9,
            "parts of 12 over a whole of 10 leaks 1: {}",
            leak(&ms)
        );
        for (stage, v) in ms.iter().enumerate() {
            NS[stage].store((v * 1e6) as u64, Relaxed);
            PROMPT_NS[stage].store((v * 1e6) as u64, Relaxed);
        }
        PROMPT_TOKENS.store(100, Relaxed);
        tick_decode_step();
        let rep = report(12.0);
        assert!(rep.contains("LEAK"), "the decode table stayed silent:\n{rep}");
        assert!(rep.contains("1.00 ms/token"), "the decode row lost its amount:\n{rep}");
        // Per prompt token the same leak is a hundredth of a millisecond, and a
        // reader comparing the two tables has to be able to tell them apart.
        assert!(rep.contains("0.01 ms/token"), "the prompt row is not per token:\n{rep}");
        reset();
        let clean = report(12.0);
        assert!(
            !clean.contains("LEAK"),
            "a table with nothing counted must not blame a leak:\n{clean}"
        );
        let mut ms = [0f64; STAGE_COUNT];
        nested(&mut ms);
        assert!(leak_line(&ms, 1.0).is_none(), "nested rows must not print");
        ON.store(false, Relaxed);
    }
}