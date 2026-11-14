//! Where a decode token actually spends its time.
//!
//! `decode` already splits stack from lm_head, and the device counters split
//! fence wait from dispatch. Neither says whether the remaining host time is
//! the expert GEMVs, the delta-net recursion, or the routing and norms. These
//! counters do, and they are the difference between optimising the right thing
//! and optimising a guess.
//!
//! Off unless `enable()` is called, so a normal run pays one relaxed load per
//! stage. Stages are disjoint: a timer never nests inside another. Counters
//! accumulate across threads, so a stage measured inside `par_count` reports
//! CPU time rather than wall time — compare it against the thread count, not
//! against the token's wall clock.
//!
//! A request is two kinds of forward pass: the prompt, which [`split`] moves
//! into its own counters, and one token at a time afterwards, which
//! [`decode_steps`] counts. Dividing the prompt's chunked work by completion
//! tokens makes a table that is about neither, so the decode stages are
//! reported per decode step and the prompt gets its own line.

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

/// `moe submit` split in two and `moe device wait` split in two, because the
/// lumped numbers could not say whether the cost was the blocking activation
/// upload, the launch loop, the fence, or the readback — and those have four
/// different fixes. `lm_head` was missing entirely: on the CPU tier it is a
/// 248320x2048 half-precision read per token and nothing named it.
pub const STAGES: [&str; 13] = [
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
];

static ON: AtomicBool = AtomicBool::new(false);
static NS: [AtomicU64; 13] = [const { AtomicU64::new(0) }; 13];
static PROMPT_NS: [AtomicU64; 13] = [const { AtomicU64::new(0) }; 13];
static PROMPT_TOKENS: AtomicU64 = AtomicU64::new(0);
static STEPS: AtomicU64 = AtomicU64::new(0);
/// When the decode phase began, so the wall clock in the report is the one the
/// step count belongs to. `None` until a split, which only a batched prefill does.
static DECODE_T0: std::sync::Mutex<Option<Instant>> = std::sync::Mutex::new(None);

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
    *DECODE_T0.lock().unwrap_or_else(|e| e.into_inner()) = None;
}

/// Milliseconds per stage, in `STAGES` order — the decode steps only, once
/// [`split`] has been called.
pub fn millis() -> [f64; 13] {
    let mut out = [0f64; 13];
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
pub fn prompt_millis() -> ([f64; 13], usize) {
    let mut out = [0f64; 13];
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
    let sum: f64 = ms.iter().sum();
    let (worst, biggest) = ms
        .iter()
        .enumerate()
        .fold((0usize, 0f64), |a, (i, &x)| if x > a.1 { (i, x) } else { a });
    Some(format!(
        "  prompt    {tokens} tokens in {sum:.2} ms accounted, {:.0}% {}",
        100.0 * biggest / sum.max(1e-9),
        STAGES[worst]
    ))
}

/// The decode breakdown: milliseconds per decode step, then the prompt on its
/// own line. `wall_ms` is the caller's whole request and is only used when the
/// phase was never split, so the last row shows what the stages failed to name.
pub fn report(wall_ms: f64) -> String {
    let steps = decode_steps().max(1) as f64;
    let ms = millis();
    let sum: f64 = ms.iter().sum();
    let wall = decode_wall_ms().unwrap_or(wall_ms);
    let mut out = format!("  stage breakdown (ms/token, {} decode steps):\n", decode_steps());
    for (name, v) in STAGES.iter().zip(&ms) {
        out.push_str(&format!("    {name:<20} {:7.2}  {:5.1}%\n", v / steps, 100.0 * v / sum.max(1e-9)));
    }
    out.push_str(&format!(
        "    {:<20} {:7.2}  (wall {:.2})\n",
        "accounted",
        sum / steps,
        wall / steps
    ));
    if let Some(line) = prompt_line() {
        out.push_str(&line);
        out.push('\n');
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
}