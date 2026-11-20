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
use crate::kernel;
use crate::quant::Q4;
use crate::safetensors::Shards;

/// What the streamed tier moved in total, over every layer's ring. This is the
/// only honest source for the plan's `disk_gbs` and `quant_gbs`: they are
/// guesses until a run says what this box actually did.
#[derive(Debug, Default, Clone, Copy)]
pub struct StreamTotals {
    /// Picks that found the expert already in the ring.
    pub hits: u64,
    /// Picks that had to wait for it - the only misses there are, since a batch
    /// that read an expert the caller never reached is `warmed`, not a miss.
    pub misses: u64,
    /// Experts a `warm` batch put in the ring, whether or not a pick then used
    /// one. This is the traffic the read-ahead issues beyond the picks it
    /// satisfies, so a ring too small for a layer's cold set shows up here
    /// rather than hiding inside the miss count.
    pub warmed: u64,
    /// Refills that came back as a view into the mapping instead of a copy, over
    /// every layer's ring. Zero means the tier is still copying, whatever the
    /// wall clock looks like.
    pub borrowed: u64,
    pub rings: usize,
    pub ring_bytes: usize,
    /// What those slots would hold if none of them were borrowed, so a report can
    /// say what the zero-copy path saved rather than only what it costs.
    pub ring_bytes_full: usize,
    pub read_ms: f64,
    pub quant_ms: f64,
    pub read_bytes: u64,
    pub quant_bytes: u64,
}

fn gbs(bytes: u64, ms: f64) -> f64 {
    if ms > 0.0 {
        bytes as f64 / (ms / 1e3) / 1e9
    } else {
        0.0
    }
}

impl StreamTotals {
    /// Decimal GB/s, the unit `sched::Bandwidth` is written in.
    pub fn read_gbs(&self) -> f64 {
        gbs(self.read_bytes, self.read_ms)
    }

    pub fn quant_gbs(&self) -> f64 {
        gbs(self.quant_bytes, self.quant_ms)
    }
}

struct Slot {
    id: usize,
    /// Logical clock at last use: the smallest one is the eviction candidate.
    seq: u64,
    gu: Q4,
    dn: Q4,
}

/// No expert lives here right now. `usize::MAX` is past the last expert id, so a
/// vacant slot can never be mistaken for a hit, and its zeroed matrices are
/// never read.
const VACANT: usize = usize::MAX;

/// What one refill thread managed to do with an expert.
enum Fetched {
    /// The pair points into the file mapping: no bytes were copied and the ring
    /// holds nothing but the scales, which is a ninth of the pack.
    Borrowed { gu: Q4, dn: Q4 },
    /// This checkpoint has no pre-quantised form to borrow - a bf16 shard that
    /// has to be quantised, or a container whose tensors are not mapped - so
    /// here is an owned pair. `read` is what its source cost to bring in.
    Owned { gu: Q4, dn: Q4, read: u64 },
    /// Leave it out of the ring; `Stream::with` will do the real thing (and
    /// report the real error) for this one.
    Skip,
}

/// Where a refill's result goes: the ring slot it takes over, or `VACANT` when
/// the ring still has room to grow into.
struct Dest {
    at: usize,
    got: Fetched,
}

/// One streamed expert, borrowed from the mapping when this checkpoint has a
/// pre-quantised form and quantised into owned memory when it does not.
///
/// The last branch matters more than it looks: a bf16 checkpoint has neither a
/// sidecar nor a packed container, so without it a batch would fetch nothing and
/// the tier would quietly stay serial. Quantising here is not extra work - it is
/// the same pass `with` pays, just issued while the other experts of this layer
/// are still in flight.
fn fetch(
    sh: &Shards,
    prefix: &str,
    layer: usize,
    h: usize,
    i: usize,
    group: usize,
    threads: usize,
    id: usize,
) -> Fetched {
    match crate::model::borrow_expert(sh, prefix, layer, id, h, i, group) {
        Some(Ok((gu, dn))) => Fetched::Borrowed { gu, dn },
        Some(Err(_)) => Fetched::Skip,
        None => match crate::model::cached_expert(sh, prefix, layer, id, h, i, group) {
            Some(Ok(p)) => {
                let read = (p.0.bytes_total() + p.1.bytes_total()) as u64;
                Fetched::Owned {
                    gu: p.0,
                    dn: p.1,
                    read,
                }
            }
            _ => match crate::model::load_expert_pair(sh, prefix, layer, id, h, i) {
                Ok((gs, ds)) => {
                    let read = ((gs.len() + ds.len()) * std::mem::size_of::<f32>()) as u64;
                    Fetched::Owned {
                        gu: Q4::quantize(&gs, 2 * i, h, group, threads),
                        dn: Q4::quantize(&ds, h, i, group, threads),
                        read,
                    }
                }
                Err(_) => Fetched::Skip,
            },
        },
    }
}

pub struct Stream {
    sh: Arc<Shards>,
    prefix: String,
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
    /// Experts loaded by `warm`, counted separately from `misses` so
    /// `hits + misses` stays the number of picks the caller blocked on.
    pub warmed: u64,
    /// Refills that ended as a view into the file mapping rather than a copy.
    /// The control for the zero-copy path: if this stays 0 the tier is still
    /// paying the `memcpy`, whatever the timing says.
    pub borrowed: u64,
    pub read_ms: f64,
    pub quant_ms: f64,
    /// Bytes actually moved by the two timings above, so the planner's guessed
    /// `disk_gbs` and `quant_gbs` can be compared with what this box really did
    /// on a real workload instead of a probe. A borrowed pack counts the pages
    /// the lane has to fault in; the quantising fallback counts the f32 source it
    /// encodes, and is the only one that pays a quantisation pass.
    pub read_bytes: u64,
    pub quant_bytes: u64,
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
        Stream {
            layer,
            h: cfg.hidden_size,
            i: cfg.moe_intermediate_size,
            sh,
            prefix: cfg.prefix.clone(),
            group,
            threads: threads.max(1),
            cap: cap.max(1),
            ring: Vec::with_capacity(cap.max(1)),
            clock: 0,
            hits: 0,
            misses: 0,
            warmed: 0,
            borrowed: 0,
            read_ms: 0.0,
            quant_ms: 0.0,
            read_bytes: 0,
            quant_bytes: 0,
        }
    }

    /// Bytes the ring's slots hold right now, for the memory report. A borrowed
    /// expert contributes only its scales: its nibbles are the file's page cache,
    /// which the OS can hand to anything else without writing, and which a second
    /// ring asking for the same block would share rather than copy.
    ///
    /// Vacant slots count too. A slot is only vacant because a fetch failed after
    /// its previous expert was dropped, and the buffers it held stay allocated
    /// until something refills the hole - RAM the tier is paying for even though
    /// no expert lives there.
    pub fn bytes(&self) -> usize {
        self.ring
            .iter()
            .map(|s| s.gu.bytes_held() + s.dn.bytes_held())
            .sum()
    }

    /// Experts currently in the ring, never above `cap`.
    pub fn slots(&self) -> usize {
        self.ring.iter().filter(|s| s.id != VACANT).count()
    }

    /// What one expert costs if the ring has to own it - the same
    /// `sched::expert_bytes` figure, measured from a slot that actually exists.
    /// A borrowed slot holds less than this by design, so `bytes()` is bounded by
    /// `slots() * pack_bytes()` rather than equal to it.
    pub fn pack_bytes(&self) -> usize {
        self.ring
            .iter()
            .find(|s| s.id != VACANT)
            .map(|s| s.gu.bytes_total() + s.dn.bytes_total())
            .unwrap_or(0)
    }

    /// Load these experts into the ring at once, before any of them is computed.
    ///
    /// The router has already named every one of them for this layer, so this is
    /// not the histogram prefetcher guessing at the future: it is the same read
    /// the caller is about to block on, issued several deep instead of one at a
    /// time. Decoding Edge0 spent 70% of its wall clock inside the serial
    /// version at 0.9 GB/s - under both what the disk and what the memory bus can
    /// do - and every one of those milliseconds left the compute workers idle.
    ///
    /// Best effort on purpose: an expert this cannot fetch is simply absent from
    /// the ring, and `with` does for it exactly what it did before.
    pub fn warm(&mut self, ids: &[usize]) {
        let mut want: Vec<usize> = Vec::with_capacity(ids.len());
        for &id in ids {
            if !want.contains(&id) && !self.ring.iter().any(|s| s.id == id) {
                want.push(id);
            }
        }
        // Only `cap` of them can be resident at once, and loading more would
        // evict what the caller is about to read.
        want.truncate(self.cap);
        // Destinations, cheapest first: a slot an earlier failure left vacant,
        // then the ring's unused room, then the coldest live expert.
        let mut at: Vec<usize> = self
            .ring
            .iter()
            .enumerate()
            .filter(|(_, s)| s.id == VACANT)
            .map(|(k, _)| k)
            .take(want.len())
            .collect();
        let grow = (self.cap - self.ring.len()).min(want.len() - at.len());
        let evict = want.len() - at.len() - grow;
        if evict > 0 {
            let mut cold: Vec<(u64, usize)> = self
                .ring
                .iter()
                .enumerate()
                .filter(|(_, s)| s.id != VACANT)
                .map(|(k, s)| (s.seq, k))
                .collect();
            cold.sort_unstable();
            at.extend(cold.iter().take(evict).map(|&(_, k)| k));
        }
        // Vacate the destinations first: an expert a worker fails to fetch stays
        // out of the ring instead of leaving a slot claiming the evicted one.
        let mut dests: Vec<Dest> = at
            .iter()
            .map(|&k| {
                self.ring[k].id = VACANT;
                self.ring[k].seq = 0;
                Dest { at: k, got: Fetched::Skip }
            })
            .chain((0..grow).map(|_| Dest { at: VACANT, got: Fetched::Skip }))
            .collect();
        if dests.is_empty() {
            return;
        }

        let t0 = Instant::now();
        {
            let Self { sh, prefix, layer, h, i, group, threads, .. } = &*self;
            let (sh, prefix) = (&**sh, prefix.as_str());
            kernel::par_rows(&mut dests, *threads, |part, first| {
                for (n, d) in part.iter_mut().enumerate() {
                    d.got = fetch(sh, prefix, *layer, *h, *i, *group, *threads, want[first + n]);
                }
            });
        }

        let wall = t0.elapsed().as_secs_f64() * 1e3;
        let mut loaded = 0u64;
        let mut bytes = 0u64;
        for (n, d) in dests.into_iter().enumerate() {
            let id = want[n];
            let (gu, dn, moved) = match d.got {
                Fetched::Borrowed { gu, dn } => {
                    // The packs cost no copy, but the lane still has to fault
                    // every page of them in on first touch, which is the traffic
                    // the plan prices. Counted the same way as the copied path so
                    // the two are comparable.
                    self.borrowed += 1;
                    let b = (gu.bytes_total() + dn.bytes_total()) as u64;
                    (gu, dn, b)
                }
                Fetched::Owned { gu, dn, read } => (gu, dn, read),
                Fetched::Skip => continue,
            };
            bytes += moved;
            loaded += 1;
            self.clock += 1;
            let seq = self.clock;
            if d.at == VACANT {
                self.ring.push(Slot { id, seq, gu, dn });
            } else {
                let s = &mut self.ring[d.at];
                s.id = id;
                s.seq = seq;
                s.gu = gu;
                s.dn = dn;
            }
        }
        if loaded > 0 {
            // Not `misses`: a pick this batch served then reads as a hit, and one
            // it could not hold still reads as the serial miss it is. The count
            // that stays true is `hits + misses == picks`.
            self.warmed += loaded;
            self.read_bytes += bytes;
            // Wall time of the batch, not the sum over threads: the rate is meant
            // to describe the lane, which is the thing being widened. A batch that
            // had to quantise is timed the same way - `quant_ms` stays the serial
            // measurement, the one the planner compares a single worker against.
            self.read_ms += wall;
        }
    }

    /// Hand `f` the two matrices of expert `id`, loading it first on a miss.
    pub fn with<R>(&mut self, id: usize, f: impl FnOnce(&Q4, &Q4) -> R) -> Result<R> {
        // destructure so the counters stay writable while a slot is borrowed
        let Self {
            sh, prefix, layer, h, i, group, threads, cap, ring, clock,
            hits, misses, borrowed, read_ms, quant_ms, read_bytes, quant_bytes, ..
        } = self;
        *clock += 1;
        let seq = *clock;
        if let Some(k) = ring.iter().position(|s| s.id == id) {
            *hits += 1;
            ring[k].seq = seq;
            return Ok(f(&ring[k].gu, &ring[k].dn));
        }
        *misses += 1;
        let t0 = Instant::now();
        let el = |t0: Instant| t0.elapsed().as_secs_f64() * 1e3;
        // The same three sources `fetch` works through, but with the errors kept:
        // a batch leaves an expert it cannot read out of the ring, whereas the
        // caller here is about to compute on it and needs to hear why it can't.
        let (gu, dn) = match crate::model::borrow_expert(sh, prefix.as_str(), *layer, id, *h, *i, *group) {
            Some(r) => {
                let (gu, dn) = r?;
                *borrowed += 1;
                *read_bytes += (gu.bytes_total() + dn.bytes_total()) as u64;
                *read_ms += el(t0);
                (gu, dn)
            }
            // The cache turns the copied tier's quantisation pass into a memcpy;
            // the disk read is still there, but it now reads int4 instead of bf16.
            None => match crate::model::cached_expert(sh, prefix.as_str(), *layer, id, *h, *i, *group) {
                Some(r) => {
                    let (gu, dn) = r?;
                    *read_bytes += (gu.bytes_total() + dn.bytes_total()) as u64;
                    *read_ms += el(t0);
                    (gu, dn)
                }
                None => {
                    let (gsrc, dsrc) =
                        crate::model::load_expert_pair(sh, prefix.as_str(), *layer, id, *h, *i)?;
                    let src = ((gsrc.len() + dsrc.len()) * std::mem::size_of::<f32>()) as u64;
                    let read = el(t0);
                    *read_bytes += src;
                    *quant_bytes += src;
                    let gu = Q4::quantize(&gsrc, 2 * *i, *h, *group, *threads);
                    let dn = Q4::quantize(&dsrc, *h, *i, *group, *threads);
                    *read_ms += read;
                    *quant_ms += el(t0) - read;
                    (gu, dn)
                }
            },
        };
        commit(ring, *cap, id, seq, gu, dn, f)
    }
}

/// Put one fetched expert in the ring and hand `f` its matrices.
///
/// A slot an earlier `warm` vacated because its fetch failed is filled before
/// the ring is grown, so a failed read leaves a hole the tier reuses instead of
/// pushing `ring.len()` past `cap` and evicting a live expert to make room.
fn commit<R>(
    ring: &mut Vec<Slot>,
    cap: usize,
    id: usize,
    seq: u64,
    gu: Q4,
    dn: Q4,
    f: impl FnOnce(&Q4, &Q4) -> R,
) -> Result<R> {
    let slot = Slot { id, seq, gu, dn };
    let k = match ring.iter().position(|s| s.id == VACANT) {
        Some(k) => k,
        None if ring.len() < cap => {
            ring.push(slot);
            let s = &ring[ring.len() - 1];
            return Ok(f(&s.gu, &s.dn));
        }
        None => ring
            .iter()
            .enumerate()
            .min_by_key(|(_, s)| s.seq)
            .map(|(n, _)| n)
            .unwrap_or(0),
    };
    ring[k] = slot;
    Ok(f(&ring[k].gu, &ring[k].dn))
}
