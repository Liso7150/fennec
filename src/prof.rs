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

/// `moe submit` split in two and `moe device wait` split in two, because the
/// lumped numbers could not say whether the cost was the blocking activation
/// upload, the launch loop, the fence, or the readback — and those have four
/// different fixes. `lm_head` was missing entirely: on the CPU tier it is a
/// 248320x2048 half-precision read per token and nothing named it.
pub const STAGES: [&str; 12] = [
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
];

static ON: AtomicBool = AtomicBool::new(false);
static NS: [AtomicU64; 12] = [const { AtomicU64::new(0) }; 12];

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
}

/// Milliseconds per stage, in `STAGES` order.
pub fn millis() -> [f64; 12] {
    let mut out = [0f64; 12];
    for (i, c) in NS.iter().enumerate() {
        out[i] = c.load(Relaxed) as f64 / 1e6;
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

    #[test]
    fn a_disabled_timer_records_nothing() {
        reset();
        ON.store(false, Relaxed);
        {
            let _t = start(ROUTE);
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert_eq!(millis()[ROUTE], 0.0);
    }

    #[test]
    fn stages_are_disjoint_and_named() {
        assert_eq!(STAGES.len(), NS.len());
        assert!(STAGES.iter().all(|s| !s.is_empty()));
        let mut seen = std::collections::HashSet::new();
        assert!(STAGES.iter().all(|s| seen.insert(*s)));
    }
}