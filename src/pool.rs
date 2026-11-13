//! The expert pool: the experts a plan does not put in RAM come off the shard.
//!
//! `sched::Plan` decides how many experts per layer are allowed to live in
//! memory. The rest are read, quantised and dropped as the router asks for
//! them, so a 40-layer model fits in a machine that cannot hold its 20 GiB of
//! FQ4 experts. One slot is a whole expert - both matrices, `sched::expert_bytes`
//! of it - so `cap` slots per layer is the second memory knob after the plan.
//!
//! The ring is behind a mutex because `MoeLayer::forward` runs the resident
//! experts on several threads at once. Streaming is the slow tier precisely
//! because it costs a disk read and a quantisation pass, so serialising it
//! gives up nothing that the tier did not already cost.

use std::sync::Arc;
use std::time::Instant;

use crate::config::{Result, TextConfig};
use crate::quant::Q4;
use crate::safetensors::Shards;

struct Slot {
    id: usize,
    /// Logical clock at last use: the smallest one is the eviction candidate.
    seq: u64,
    gu: Q4,
    dn: Q4,
}

pub struct Stream {
    sh: Arc<Shards>,
    gu_name: String,
    dn_name: String,
    layer: usize,
    h: usize,
    i: usize,
    group: usize,
    threads: usize,
    cap: usize,
    ring: Vec<Slot>,
    clock: u64,
    pub hits: u64,
    pub misses: u64,
    pub read_ms: f64,
    pub quant_ms: f64,
}

impl Stream {
    pub fn new(
        sh: Arc<Shards>,
        cfg: &TextConfig,
        layer: usize,
        group: usize,
        threads: usize,
        cap: usize,
    ) -> Stream {
        let p = format!("{}.layers.{layer}.mlp.experts", crate::model::PREFIX);
        Stream {
            gu_name: format!("{p}.gate_up_proj"),
            dn_name: format!("{p}.down_proj"),
            layer,
            h: cfg.hidden_size,
            i: cfg.moe_intermediate_size,
            sh,
            group,
            threads: threads.max(1),
            cap: cap.max(1),
            ring: Vec::with_capacity(cap.max(1)),
            clock: 0,
            hits: 0,
            misses: 0,
            read_ms: 0.0,
            quant_ms: 0.0,
        }
    }

    /// Bytes the ring is holding right now, for the memory report.
    pub fn bytes(&self) -> usize {
        self.ring.iter().map(|s| s.gu.bytes_total() + s.dn.bytes_total()).sum()
    }

    /// Experts currently in the ring, never above `cap`.
    pub fn slots(&self) -> usize {
        self.ring.len()
    }

    /// Bytes one streamed expert costs: the same `sched::expert_bytes` figure,
    /// measured from a slot that actually exists.
    pub fn slot_bytes(&self) -> usize {
        self.ring
            .first()
            .map(|s| s.gu.bytes_total() + s.dn.bytes_total())
            .unwrap_or(0)
    }

    /// Hand `f` the two matrices of expert `id`, loading it first on a miss.
    pub fn with<R>(&mut self, id: usize, f: impl FnOnce(&Q4, &Q4) -> R) -> Result<R> {
        // destructure so the counters stay writable while a slot is borrowed
        let Self {
            sh, gu_name, dn_name, layer, h, i, group, threads, cap, ring, clock,
            hits, misses, read_ms, quant_ms, ..
        } = self;
        *clock += 1;
        let seq = *clock;
        if let Some(k) = ring.iter().position(|s| s.id == id) {
            *hits += 1;
            ring[k].seq = seq;
            return Ok(f(&ring[k].gu, &ring[k].dn));
        }
        *misses += 1;
        let rows_gu = 2 * *i;
        let t0 = Instant::now();
        // The cache turns the slow tier's quantisation pass into a memcpy; the
        // disk read is still there, but it now reads int4 instead of bf16.
        let (gu, dn, t_read) = match crate::model::cached_expert(sh, *layer, id, *h, *i, *group) {
            Some(r) => {
                let (gu, dn) = r?;
                (gu, dn, t0.elapsed())
            }
            None => {
                let gsrc = sh.slice_f32(gu_name, id * rows_gu * *h, (id + 1) * rows_gu * *h)?;
                let dsrc = sh.slice_f32(dn_name, id * *h * *i, (id + 1) * *h * *i)?;
                let t_read = t0.elapsed();
                let gu = Q4::quantize(&gsrc, rows_gu, *h, *group, *threads);
                let dn = Q4::quantize(&dsrc, *h, *i, *group, *threads);
                (gu, dn, t_read)
            }
        };
        *read_ms += t_read.as_secs_f64() * 1e3;
        *quant_ms += (t0.elapsed() - t_read).as_secs_f64() * 1e3;
        let slot = Slot { id, seq, gu, dn };
        let k = if ring.len() < *cap {
            ring.push(slot);
            ring.len() - 1
        } else {
            let coldest = ring
                .iter()
                .enumerate()
                .min_by_key(|(_, s)| s.seq)
                .map(|(n, _)| n)
                .unwrap_or(0);
            ring[coldest] = slot;
            coldest
        };
        Ok(f(&ring[k].gu, &ring[k].dn))
    }
}
