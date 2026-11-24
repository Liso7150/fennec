//! The tier scheduler: what colibri does well, made explicit.
//!
//! Given a device budget, a host-RAM pool and a histogram of which experts the
//! model actually routes to, decide per layer how many experts sit on the GPU,
//! how many stay in system RAM, and how many are streamed from disk. Then price
//! a decode token in bytes-per-tier against measured bandwidth, so two plans can
//! be compared before anything is built.
//!
//! Three tiers, because three is what this crate can name a reader for. An NPU
//! tier would need a kernel: `MoeLayer` has no way to run an expert anywhere but
//! its own threads and the device batch, so experts placed on an accelerator
//! Fennec cannot compute on are host experts wearing a different name - and the
//! price the plan would then read off that tier is twice the rate of the core
//! that does the work.

use crate::config::{Result, TextConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Device,  // GPU VRAM - fastest
    Host,    // System RAM - medium
    Disk,    // SSD - slowest
}

/// One row of colibri's `.coli_usage`: `layer expert count`. The two header rows
/// use negative layer indices and are skipped.
pub struct Usage {
    pub layers: usize,
    pub experts: usize,
    pub counts: Vec<Vec<u64>>,
}

impl Usage {
    pub fn load(path: &str) -> Result<Usage> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let mut layers = 0usize;
        let mut experts = 0usize;
        let mut rows: Vec<(usize, usize, u64)> = Vec::new();
        for line in text.lines() {
            let mut it = line.split_whitespace();
            let l: i64 = match it.next().and_then(|s| s.parse().ok()) {
                Some(v) => v,
                None => continue,
            };
            let e: i64 = match it.next().and_then(|s| s.parse().ok()) {
                Some(v) => v,
                None => continue,
            };
            let c: u64 = it.next().and_then(|s| s.parse().ok()).unwrap_or(0);
            if l == -1 {
                // the shape row: `-1 <layers> <experts>`
                layers = e.max(0) as usize;
                experts = c.max(1) as usize;
                continue;
            }
            if l < 0 || e < 0 {
                // `-2 <n> <total selections>` and any other header
                continue;
            }
            rows.push((l as usize, e as usize, c));
        }
        if layers == 0 {
            layers = rows.iter().map(|r| r.0 + 1).max().unwrap_or(0);
        }
        if experts == 0 {
            experts = rows.iter().map(|r| r.1 + 1).max().unwrap_or(0);
        }
        let mut counts = vec![vec![0u64; experts]; layers];
        for (l, e, c) in rows {
            if l < layers && e < experts {
                counts[l][e] = c;
            }
        }
        Ok(Usage { layers, experts, counts })
    }

    pub fn uniform(layers: usize, experts: usize) -> Usage {
        Usage {
            layers,
            experts,
            counts: vec![vec![1u64; experts]; layers],
        }
    }

    /// Share of all recorded selections that a perfectly packed set of `n`
    /// experts per layer would answer from: the headroom residency buys.
    pub fn coverage(&self, n: usize) -> f64 {
        let mut best = 0u64;
        let mut tot = 0u64;
        for row in &self.counts {
            let mut v: Vec<u64> = row.clone();
            v.sort_unstable_by(|a, b| b.cmp(a));
            best += v.iter().take(n).sum::<u64>();
            tot += v.iter().sum::<u64>();
        }
        best as f64 / tot.max(1) as f64
    }

    /// Every selection the file records, summed.
    pub fn total(&self) -> u64 {
        self.counts.iter().map(|r| r.iter().sum::<u64>()).sum()
    }

    /// Publish a histogram as colibri's headerless legacy triples: the `-1
    /// <layers> <experts>` shape row plus one `layer expert count` per non-zero
    /// cell. The `-2 <version> <engine id>` form is deliberately not written -
    /// that record exists so colibri's reader can reject another engine's
    /// history, and Fennec has no name in its table. The triples are accepted by
    /// anybody once the shape row matches the model being loaded, which is the
    /// check that matters: a ranking for another model's dimensions is the only
    /// way this file could be misread.
    pub fn save(&self, path: &str) -> Result<()> {
        let mut text = format!("-1 {} {}\n", self.layers, self.experts);
        for (l, row) in self.counts.iter().enumerate() {
            for (e, c) in row.iter().enumerate() {
                if *c > 0 {
                    text.push_str(&format!("{l} {e} {c}\n"));
                }
            }
        }
        // Temp file then rename, because a run killed mid-write must leave the
        // previous ranking intact rather than a half file the next start refuses.
        let tmp = format!("{path}.tmp");
        std::fs::write(&tmp, text.as_bytes()).map_err(|e| format!("{tmp}: {e}"))?;
        std::fs::rename(&tmp, path).map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            format!("{path}: {e}")
        })
    }
}

/// Bytes moved per second per tier, for the box this runs on, plus the FQ4
/// encode rate: a streamed expert that has no pre-quantised form to borrow is
/// read *and* quantised, and the encode was the half the original model left
/// out, which is why the disk tier measured three times slower than the plan
/// said it would be. A tier that borrows pays no encode, so it has no business
/// being charged one — see `stream_price_ms`.
///
/// Every field is a seed. `Measured` is what a real run of this model on this
/// box replaced them with.
#[derive(Debug, Clone, Copy)]
pub struct Bandwidth {
    pub device_gbs: f64,
    pub host_gbs: f64,
    pub pcie_gbs: f64,
    pub disk_gbs: f64,
    pub quant_gbs: f64,
}

impl Bandwidth {
    /// RTX 4060 Laptop (8 GiB GDDR6, PCIe 4.0 x8) with DDR5-5600 in two channels.
    ///
    /// `host_gbs` is 17, not the 45 the spec sheet claims. That 17 came from
    /// `fennec bench`'s copy loop (18.5 GB/s over the MoE block, 15.4-17 over
    /// lm_head), and #464 finally measured what the tier really does: the int4
    /// expert GEMV holds 31-37 GB/s on private pages from 8 to 16 workers against
    /// a 63.6 GB/s read ceiling, so the kernel - not the RAM - is the limit, and
    /// it is already issue-limited at the shipped worker count. The 17 is left
    /// against the 31-37 on purpose: this one field prices three byte classes (the
    /// int4 experts, the int8 dense trunk, the int8 lm_head) and 15.4-18.5 is what
    /// two of the three were timed at, so raising it to the expert kernel's rate
    /// would price the trunk and the head too cheaply to pay for the experts.
    /// Splitting the classes is #436's structural job; editing this number is not
    /// that fix, and a plan rebased on 34 would sit quietly 2x off every
    /// prediction recorded against 17. Two consequences for this constant, both of
    /// which it now honors:
    ///
    /// - This is a *cost of bytes*, which is the only thing the plan divides by,
    ///   and it is deliberately not a served-window number. The windows the engine
    ///   logs are not comparable to it: two runs of one shape on one box put the
    ///   CPU lane at 4.2 GB/s and 15.6 90 s apart, because the span timed is the
    ///   lane the host picks share with the streamed refills they wait on.
    ///   `Measured::host_gbs` records that achievement and `apply` refuses to price
    ///   with it; see the comment there for the 149 ms/token machine that predicted
    ///   1802.5 when it did. Until #473 the windows were worse than noisy: they
    ///   blended the decode tier with the prefill union and double-billed the
    ///   union's bytes, the r20 CPU arm reporting 759 MB/token against the 542 MiB
    ///   a token can read even when every one of its 320 picks comes from RAM.
    /// - It must not be read as a ceiling to buy back with more threads. The
    ///   marginal byte past ~8 workers is already free of bandwidth pressure
    ///   (efficiency falls 85-91 % to 57-62 % while GB/s stays flat), so a plan
    ///   that expects a tier to get proportionally faster per worker is wrong
    ///   twice over. The planner has no worker axis to get it wrong on, which is
    ///   the one part of #473 that needed no code.
    ///
    /// The CPU reads its own RAM at that rate whether or not a GPU is attached, so
    /// both profiles share it; only `device_gbs` differs.
    ///
    /// Two findings from #464 that would otherwise get re-bought: reading the same
    /// packs as *file-mapped* pages costs 79-101 % of the private-buffer time
    /// (freeing 12 GiB of commit for a few percent, which is the trade the brief
    /// asks for), and randomising the pick order changes the time by
    /// -3.8 %...+3.6 % - the router's hot list is already effectively random, so no
    /// placement wins by reordering it.
    pub fn laptop_4060() -> Bandwidth {
        Bandwidth {
            device_gbs: 272.0,
            host_gbs: 17.0,
            pcie_gbs: 13.5,
            disk_gbs: 3.2,
            quant_gbs: 1.9,
        }
    }

    /// CPU-only decode: everything is read from system RAM, so the device tier
    /// is a second name for the host tier and every bandwidth is the host's. Same
    /// 17 GB/s and the same provenance. Taking the device batch out does not turn
    /// the served window into a byte cost either: r23 ran this profile with no
    /// device in sight and logged 4.2 GB/s cold and 15.6 warm for one shape 90 s
    /// apart, with the streamed window of the same pair moving 3.8 to 23.6 in step -
    /// which is what says the host number times a lane rather than the time one
    /// pack's first byte takes to reach its last.
    pub fn cpu_only() -> Bandwidth {
        Bandwidth {
            device_gbs: 17.0,
            host_gbs: 17.0,
            pcie_gbs: 17.0,
            disk_gbs: 3.2,
            quant_gbs: 1.9,
        }
    }

    /// Override whatever a previous run of this model measured, leaving the rest
    /// of the profile as the seed. Only rates that were actually earned get in:
    /// a run that streamed nothing measured nothing.
    ///
    /// `device_gbs` and `pcie_gbs` are absent on purpose. Nothing times the
    /// device's own reads or the load-time upload per byte, so the only honest
    /// source for those two is the profile they came from.
    ///
    /// `host_gbs` is absent for a different reason, and it is the reason #473
    /// stayed open after the union's double-count was fixed: the measured host
    /// number is not a cost per byte, so there is no price for it to replace.
    /// Two runs of one shape on one box, `decode --cpu --window 4` on edge0-35b,
    /// 16 workers, 1.69 MiB packs, 90 s apart: the first logged 573 MB in 144 ms
    /// (4.2 GB/s) and the second 889 MB in 60 ms (15.6). Nothing about the tier
    /// changed between them but the OS page cache - the streamed window in the
    /// same pair went 3.8 to 23.6 GB/s - and a rate that swings 3.7x on cache
    /// state cannot be what a byte costs. Pricing the second run with the first
    /// one's 4.2 made a machine that served 149 ms/token predict 1802.5. The two
    /// windows move together, which is what says the host span is a lane the
    /// streamed tier is also using rather than the time between one pack's first
    /// byte and its last: the numerator bills host picks only, the denominator is
    /// the lane. It stays in the rates file as what the box was doing under load,
    /// which is worth reading and not worth dividing by.
    pub fn apply(&mut self, m: &Measured) {
        if let Some(v) = m.disk_gbs {
            self.disk_gbs = v;
        }
        if let Some(v) = m.quant_gbs {
            self.quant_gbs = v;
        }
    }
}

/// What a finished run on this box learned about the rates the plan prices with.
///
/// This is the difference between a plan and a guess. `laptop_4060()` is one
/// machine's numbers quoted to every machine, and the streamed rate in particular
/// is not a property of the SSD: r11 measured 13-20 GB/s when the tier borrowed
/// pages the OS was already caching and 1.3-1.9 GB/s when it copied through a
/// private buffer, on the same disk, in the same session. Neither number belongs
/// in a constant, so neither is used as one once a run has said otherwise -
/// except the host rate, which #473 stopped applying because a lane span is not a
/// cost per byte at all. See `Bandwidth::apply`.
///
/// It lives beside `.fennec_usage` because both are learned the same way - the
/// first start assumes, every start after it knows - and it is written by Fennec
/// about Fennec, never into anything Colibri reads.
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct Measured {
    /// Decimal GB/s the streamed tier moved per pick, as the ring timed it. The
    /// coldest sample wins: see `merge`.
    ///
    /// The same page-cache swing that took the host window from 4.2 to 15.6 took
    /// this one from 3.8 to 23.6 across the same two runs, so it is as much a
    /// property of what the OS still holds as of the SSD. `merge` keeps the coldest
    /// of the two, which is the pessimistic side of that band, and it is the rate
    /// that would move a placement if a placement were ever sensitive to it (r23
    /// found none: three arms, 4.2/15.6/23.6, all placed 2427 slots).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disk_gbs: Option<f64>,
    /// Decimal GB/s of the FQ4 encode pass. Absent when the tier borrowed and
    /// paid no encode, which is the case the plan must stop pricing. The coldest
    /// sample wins, as with `disk_gbs`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quant_gbs: Option<f64>,
    /// Decimal GB/s the host tier sustained while computing decode picks.
    /// `Model::host_traffic` times it on the decode path; the batched-prefill
    /// union, which reads one pack per expert instead of one per pick, is timed on
    /// its own and never mixed in (#473). The warmest sample of the same shape
    /// wins: see `merge`.
    ///
    /// Reported, never priced: `Bandwidth::apply` leaves `host_gbs` alone. This is
    /// a served-path *achievement* over a lane span, so its denominator contains
    /// the streamed refills the host picks wait on (and, on a GPU plan, the device
    /// batch they overlap), and it moves 3.7x on page-cache state for one shape
    /// (4.2 then 15.6 GB/s, 90 s apart). What a host byte costs is the profile
    /// constant, which #464 bracketed at 31-37 GB/s for the expert kernel and
    /// `fennec bench` at 15.4-18.5 for the copy loop.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host_gbs: Option<f64>,
    /// Matrix bytes held per byte committed by the RAM tier: 9.0-ish when every
    /// layer borrows its experts out of the mapping, 1.0 when none does. Printed,
    /// not priced: r13 measured that spending it buys nothing, so the host
    /// budget stays in matrix bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub held_ratio: Option<f64>,
    /// The tier was still borrowing when this was written, so a reader can tell
    /// a borrowed `disk_gbs` from a copied one instead of guessing from the box.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub borrows: bool,
    /// Worker threads one decode pick was split over, and the matrix bytes such a
    /// pick reads: the two facts that decide whether two samples of `host_gbs` or
    /// `disk_gbs` describe the same machine's work. Both have changed under a
    /// still-valid rates file - #463 had the tier computing at `threads = 1` for
    /// the life of the process, #419 changed the fill, #225 and #464 changed the
    /// pack from 1.69 MiB to 0.85 and back - and a rate cannot outlive its shape.
    /// Absent in a file written before the shape was recorded, which `merge`
    /// treats as a mismatch rather than as a wildcard.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threads: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pick_bytes: Option<u64>,
}

impl Measured {
    pub fn load(path: &str) -> Result<Measured> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))
    }

    /// Temp file then rename, so a start that died mid-write leaves the last
    /// honest rates intact rather than a truncated JSON the next start ignores.
    pub fn save(&self, path: &str) -> Result<()> {
        let text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        let tmp = format!("{path}.tmp");
        std::fs::write(&tmp, text.as_bytes()).map_err(|e| format!("{tmp}: {e}"))?;
        std::fs::rename(&tmp, path).map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            format!("{path}: {e}")
        })
    }

    /// Nothing a run could have measured yet.
    pub fn is_empty(&self) -> bool {
        self.disk_gbs.is_none()
            && self.quant_gbs.is_none()
            && self.host_gbs.is_none()
            && self.held_ratio.is_none()
    }

    /// Fold what a run just measured onto what earlier runs measured.
    ///
    /// The fields do not share a rule, because the two rates are distorted by
    /// opposite conditions. The streamed tier is distorted by warmth in the
    /// direction that flatters it: the same pack read a second time comes out of
    /// the OS cache at over 20 GB/s, which is a fact about the cache and not about
    /// the drive the streamed lane has to reach for on a cold page, and the only
    /// sample that prices the streamed tier is the one that had to go to disk. So
    /// disk and quantise keep their worst. Measured on one box, four runs apart, on
    /// the same 8 GiB borrowed tier, disk read 4.4 GB/s cold and 22.9 warm.
    ///
    /// The host rate keeps its best sample for the mirror reason, and now for that
    /// reason only - it no longer prices anything (`Bandwidth::apply`), so which
    /// sample survives is a display choice about what the box was capable of, not a
    /// cost. It used to be load-bearing, and the note here bounded the choice:
    /// #464 measured the same packs read as file-mapped pages costing 79-101 % of
    /// what they cost in private buffers, which caps warmth at 1/0.79 = 1.27x. r23
    /// broke that bound in the served lane - one shape, 90 s apart, 4.2 GB/s then
    /// 15.6, a 3.7x swing - which is the second reason the field left the price: a
    /// sample whose own spread is 3.7x is not a cost per byte, and a cap derived
    /// from an isolated copy loop does not describe a window other tiers share.
    ///
    /// (The 5x gap that this file used to show between its stored rate and its own
    /// logged windows came from blending the two paths, not from warmth - see
    /// `Model::host_traffic`.)
    ///
    /// A change in the tier's shape resets the traffic rates. A tier that memcpys
    /// packs out of a mapping and one that reads a whole source and encodes it are
    /// different machines as far as either number means anything; so are 1 worker
    /// and 16, and 1.69 MiB per pick and 0.85 (#463, #419, #225). The newest shape
    /// is the one the next plan is built for, so it is the only shape whose
    /// samples survive into it - which is also what stops `warmest` from being a
    /// ratchet: a rate that can only ever go up, and that survived every rebuild
    /// until now, held 21.7 GB/s on a box whose served windows never read above 8.
    pub fn merge(&mut self, fresh: &Measured) {
        if !(self.borrows == fresh.borrows
            && self.threads == fresh.threads
            && self.pick_bytes == fresh.pick_bytes)
        {
            *self = Measured {
                borrows: fresh.borrows,
                threads: fresh.threads,
                pick_bytes: fresh.pick_bytes,
                ..Default::default()
            };
        }
        self.disk_gbs = coldest(self.disk_gbs, fresh.disk_gbs);
        self.quant_gbs = coldest(self.quant_gbs, fresh.quant_gbs);
        self.host_gbs = warmest(self.host_gbs, fresh.host_gbs);
        // Structural rather than thermal: matrix bytes per byte committed, read
        // off the tier's own accounting. The newest run describes the newest build.
        if fresh.held_ratio.is_some() {
            self.held_ratio = fresh.held_ratio;
        }
    }

    /// The conditions the traffic rates were earned under, for the lines that
    /// quote them. "host 21.7 GB/s" was read as a property of the box for three
    /// campaigns; it is a property of one shape, so the shape travels with it.
    pub fn shape_note(&self) -> String {
        match (self.threads, self.pick_bytes) {
            (Some(t), Some(b)) => format!(
                "{t} workers per pick, {:.2} MiB each",
                b as f64 / 1_048_576.0
            ),
            (Some(t), None) => format!("{t} workers per pick, pack size unrecorded"),
            (None, _) => "shape unrecorded".into(),
        }
    }
}

/// The slower of two samples, either of which may be absent.
fn coldest(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.min(y)),
        (Some(x), None) => Some(x),
        (None, y) => y,
    }
}

/// The faster of two samples, either of which may be absent.
fn warmest(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.max(y)),
        (Some(x), None) => Some(x),
        (None, y) => y,
    }
}

#[derive(Debug, Clone)]
pub struct LayerPlan {
    pub layer: usize,
    pub resident: Vec<usize>,      // GPU VRAM - hottest experts
    pub pooled: Vec<usize>,        // System RAM - medium frequency
    pub streamed: Vec<usize>,      // SSD - cold experts
    pub bytes_expert: usize,
    pub dense_bytes: usize,
    pub dense_on_device: bool,
}

#[derive(Debug, Clone)]
pub struct Plan {
    pub layers: Vec<LayerPlan>,
    pub device_bytes_used: usize,
    pub device_bytes_budget: usize,
    pub host_bytes_used: usize,
    /// The pool's ceiling as the machine pays it: the expert budget the caller
    /// named (`host_gb` in main.rs) plus `host_dense_bytes`, the trunk a plan with
    /// no device tier has no choice about. Kept as the sum so
    /// `host_bytes_used <= host_bytes_budget` stays true and the report can print
    /// one used/total pair without qualifying it.
    pub host_bytes_budget: usize,
    /// The dense bytes the host tier holds because nothing else can (#463).
    pub host_dense_bytes: usize,
    pub device_hit_rate: f64,
    pub host_hit_rate: f64,
    pub disk_rate: f64,
    pub expert_bytes: usize,
    pub dense_bytes_layer: usize,
    pub ms_per_token: f64,
    pub lm_head_ms: f64,
    /// The MoE work of one layer priced onto the device lane and onto the CPU
    /// lane separately, before the two are combined. `ms_per_token` bills the
    /// longer of the two per layer (see the pricing block in `Plan::build`), so
    /// these two are what the billed number came from.
    pub moe_device_ms: f64,
    pub moe_host_ms: f64,
    /// Milliseconds per token the concurrent-lane pricing refused to bill:
    /// `(moe_device_ms + moe_host_ms) - what the plan charges`. Zero for a CPU
    /// plan, where there is no second lane, and for a plan whose device tier is
    /// empty or whose host tier is. `decode` prints the same quantity measured
    /// (`cpu_overlap_ms`), which is how this one stays honest.
    pub overlap_credit_ms: f64,
}

fn rank_order(counts: &[u64]) -> Vec<usize> {
    let mut v: Vec<usize> = (0..counts.len()).collect();
    v.sort_by(|&a, &b| counts[b].cmp(&counts[a]).then(a.cmp(&b)));
    v
}

/// int4 bytes for one expert, scales included: two matrices of gate/up plus one
/// of down, at 4 bits, plus one f32 scale per group of columns.
pub fn expert_bytes(cfg: &TextConfig, group: usize) -> usize {
    let h = cfg.hidden_size;
    let i = cfg.moe_intermediate_size;
    let params = 2 * i * h + h * i;
    // gate_up is 2I rows of H, down is H rows of I: one f32 scale per group of
    // columns in each
    let groups = h.div_ceil(group) * 2 * i + i.div_ceil(group) * h;
    params / 2 + groups * 4
}

/// bf16 bytes one streamed expert costs to read out of the shard: the FQ4
/// figure is what it occupies afterwards, three and a half times smaller.
pub fn expert_source_bytes(cfg: &TextConfig) -> usize {
    (2 * cfg.moe_intermediate_size * cfg.hidden_size
        + cfg.hidden_size * cfg.moe_intermediate_size)
        * 2
}

/// Milliseconds one layer's streamed picks cost per token.
///
/// The tier has two shapes an order of magnitude apart, and pricing the cheap
/// one as the expensive one is how the plan came to promise 63.88 ms/token while
/// the same tier served 112:
///
/// * **borrowed** - the checkpoint has a pre-quantised form (an FQ4 sidecar or a
///   packed container), so a pick touches `pack_bytes` of file-mapped pages and
///   pays no encode. r11 measured 13-20 GB/s on this box.
/// * **copied** - a bf16 shard has to be read whole (`source_bytes`) and encoded,
///   which is the only case where `quant_gbs` belongs in the sum. r11 measured
///   1.3-1.9 GB/s for the same bytes on the same disk.
///
/// `disk_rate` is the share of picks the plan leaves to this tier, so a plan that
/// streams nothing costs nothing here whatever the checkpoint's shape.
///
/// Both shapes then pay for the compute on top of the fetch. A pick is a pick:
/// the kernel that applies a pack does the same work whether the pack came from
/// the RAM tier or out of a ring, and pricing only the read is how the plan came
/// to treat the streamed tier as a discount on RAM. The compute is charged at
/// `host_gbs`, the profile's cost of a host byte (#464 bracketed the expert kernel
/// at 31-37 GB/s, `fennec bench` the copy loop at 15.4-18.5, and the constant sits
/// at 17 for the reasons on `laptop_4060`), so a disk pick costs its refill plus
/// one host pick and nothing else invented.
fn stream_price_ms(
    borrows: bool,
    pack_bytes: usize,
    source_bytes: usize,
    k: usize,
    disk_rate: f64,
    bw: &Bandwidth,
) -> f64 {
    let (src, rate) = if borrows {
        (pack_bytes as f64, 1.0 / bw.disk_gbs)
    } else {
        (
            source_bytes as f64,
            1.0 / bw.disk_gbs + 1.0 / bw.quant_gbs,
        )
    };
    let fetch = src * rate;
    let compute = pack_bytes as f64 / bw.host_gbs;
    k as f64 * disk_rate * (fetch + compute) / 1e6
}

/// Bytes for the non-expert tensors of one layer at `bpp` bytes per parameter:
/// 2 for a bf16 copy parked on the GPU, 4 for the f32 copy the CPU kernels
/// read. Same weights, twice the traffic per token, which is why the CPU plan
/// has to say so.
pub fn dense_bytes_layer(cfg: &TextConfig, linear: bool, bpp: usize) -> usize {
    let h = cfg.hidden_size;
    let si = cfg.shared_expert_intermediate_size;
    let e = cfg.num_experts;
    // The FFN bytes of a routed checkpoint are its shared expert plus the
    // router; the FFN bytes of a dense one are the single MLP every token
    // reads. Those terms are zero when there are no experts, which would let
    // the plan price a dense layer as nothing but its mixer.
    let ffn = if cfg.is_moe() { 3 * si * h + h + e * h } else { cfg.dense_mlp_params() };
    let attn = if linear {
        cfg.conv_dim() * h
            + cfg.value_dim() * h
            + 2 * cfg.linear_num_value_heads * h
            + h * cfg.value_dim()
            + cfg.conv_dim() * cfg.linear_conv_kernel_dim
    } else {
        2 * cfg.num_attention_heads * cfg.head_dim * h
            + 2 * cfg.num_key_value_heads * cfg.head_dim * h
            + cfg.num_attention_heads * cfg.head_dim * h
    };
    (ffn + attn) * bpp
}

/// bf16 in VRAM, f32 in the host copy the CPU kernels read: the same weights at
/// twice the traffic per token, so a layer has to say which copy it holds.
const DEV_BPP: usize = 2;
const HOST_BPP: usize = 4;

/// Bytes/weight for dense projections in VRAM. 2 = f16 (default); 1 when the
/// int4-dense lever is on, so the planner frees device room for more experts.
pub static DENSE_DEV_BPP: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(2);

impl Plan {
    /// Dense projections get the device first — at batch 1 they are read every
    /// token and they are essentially the whole cost of a layer — then experts
    /// take what is left, ranked by how often the router actually picks them.
    ///
    /// `gpu` says whether a compute device is attached. Without one the device
    /// tier is a second name for system RAM, every dense block is the CPU's f32
    /// copy, and the lm_head is priced at host bandwidth.
    ///
    /// `stream_borrows` says whether this checkpoint has a pre-quantised form the
    /// streamed tier can read straight out of the file mapping. It changes both
    /// how many bytes a streamed pick moves and whether it pays an encode pass,
    /// so the caller asks `model::borrow_expert` rather than letting the plan
    /// assume the expensive shape.
    pub fn build(
        cfg: &TextConfig,
        usage: &Usage,
        device_budget: usize,
        host_budget: usize,
        reserve: usize,
        bw: &Bandwidth,
        group: usize,
        gpu: bool,
        stream_borrows: bool,
    ) -> Plan {
        let n_layers = cfg.num_hidden_layers.min(usage.layers);
        let eb = expert_bytes(cfg, group);
        let dbpp = DENSE_DEV_BPP.load(std::sync::atomic::Ordering::Relaxed);
        // The shared expert of every layer rides in the device batch with its
        // routed picks, so it comes out of the budget before the experts are
        // sized. Without this the lm_head upload at the end is the allocation
        // that hits the wall, 240 MiB after the plan said everything fit.
        let shared_dev = if gpu { n_layers * shared_bytes_of(cfg, DEV_BPP) } else { 0 };
        // int4 dense adds per-group f32 scale buffers and alignment padding the
        // bytes-per-weight estimate misses; keep headroom so the last upload
        // does not hit the wall (vkAllocateMemory out of device memory).
        let int4_headroom = if gpu && dbpp == 1 { 1 << 30 } else { 0 };
        let budget = device_budget.saturating_sub(reserve).saturating_sub(shared_dev).saturating_sub(int4_headroom);

        // pass 1: bf16 dense into VRAM, smallest block first. Every dense byte
        // is read once per token, so value and weight are the same number and
        // the greedy that fills the budget most completely wins the most
        // traffic. A layer that does not fit keeps the CPU's f32 copy.
        let mut dense_on_device = vec![false; n_layers];
        let mut order: Vec<usize> = (0..n_layers).collect();
        order.sort_by_key(|&l| dense_bytes_of(cfg, l, dbpp));
        let mut used = 0usize;
        if gpu {
            for &l in &order {
                let d = dense_bytes_of(cfg, l, dbpp);
                if used + d <= budget {
                    used += d;
                    dense_on_device[l] = true;
                }
            }
        }
        let mut device_bytes_used = used;
        let off_dense: usize = (0..n_layers)
            .filter(|&l| !dense_on_device[l])
            .map(|l| dense_bytes_of(cfg, l, HOST_BPP))
            .sum();
        // The trunk is accounted here but *not* against `host_budget`. A plan with
        // no device tier pays those bytes whether anyone asks for them or not -
        // there is nowhere else to put them - while the budget prices the one
        // discretionary thing in the pool, which is how many experts get placed
        // (`host_gb` in main.rs and `solve_ram` below both call the number matrix
        // bytes placed in the tier). Charging the trunk to it made the same flag
        // mean two different things on the two plan shapes, and that is #463: the
        // default came down 8.0 -> 4.0 GiB on the strength of r18, which ran
        // `--auto-tier`, where the trunk rides in VRAM and every budget GiB became
        // one GiB of experts (r18's own ladder prints "host-gb 2.0 -> placed 2.00
        // GiB"). On a `--cpu` plan the trunk of this model is 5.33 GiB, so a 4.0
        // GiB budget placed *zero* experts - `fennec plan --cpu --host-gb 0` and
        // `--host-gb 4` emitted byte-identical JSON - every routed pick went to the
        // streamed lane, and the lane served 2.926 tok/s in the 2026-11-20 matrix
        // against the 8.77 an ABBA-established run measured on 2026-11-15 with 5689
        // experts in RAM.
        let mut host_bytes_used = off_dense;

        // pass 2: split the remaining device room evenly over the layers, then fill
        // system RAM with the next-hot experts per layer. In CPU mode (gpu=false)
        // the device tier does not exist, so every expert is host or disk.
        let free_device = if gpu { budget.saturating_sub(device_bytes_used) } else { 0 };
        let mut slots_dev = vec![0usize; n_layers];
        let mut rem = free_device / eb.max(1);
        for r in 0..n_layers {
            let take = rem / (n_layers - r);
            slots_dev[r] = take;
            rem -= take;
        }

        let free_host = host_budget;
        let mut slots_host = vec![0usize; n_layers];
        let mut rem = free_host / eb.max(1);
        for r in 0..n_layers {
            let take = rem / (n_layers - r);
            slots_host[r] = take;
            rem -= take;
        }

        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let order = rank_order(&usage.counts[l]);
            let e = cfg.num_experts;
            let nd = slots_dev[l].min(e);
            let nh = slots_host[l].min(e.saturating_sub(nd));
            device_bytes_used += nd * eb;
            host_bytes_used += nh * eb;
            layers.push(LayerPlan {
                layer: l,
                resident: order[..nd].to_vec(),
                pooled: order[nd..nd + nh].to_vec(),
                streamed: order[nd + nh..].to_vec(),
                bytes_expert: eb,
                dense_bytes: dense_bytes_of(cfg, l, if dense_on_device[l] { dbpp } else { HOST_BPP }),
                dense_on_device: dense_on_device[l],
            });
        }

        // price one token
        let k = cfg.num_experts_per_tok;
        let mut hit = [0u64; 3];  // [Device, Host, Disk]
        let mut tot = 0u64;
        for (l, lp) in layers.iter().enumerate() {
            let mut set = vec![Tier::Disk; cfg.num_experts];
            for &x in &lp.resident {
                set[x] = Tier::Device;
            }
            for &x in &lp.pooled {
                set[x] = Tier::Host;
            }
            for (x, &c) in usage.counts[l].iter().enumerate() {
                let i = match set[x] {
                    Tier::Device => 0,
                    Tier::Host => 1,
                    Tier::Disk => 2,
                };
                hit[i] += c;
                tot += c;
            }
        }
        let t = tot.max(1) as f64;
        let (device_hit_rate, host_hit_rate, disk_rate) =
            (hit[0] as f64 / t, hit[1] as f64 / t, hit[2] as f64 / t);
        let per_token_bytes = (k * eb) as f64;
        // bytes / (GB/s) is 1e9 times too big to be seconds, so one 1e6 divisor
        // lands directly on milliseconds.
        let mut ms = 0.0f64;
        let stream_ms = stream_price_ms(
            stream_borrows,
            eb,
            expert_source_bytes(cfg),
            k,
            disk_rate,
            bw,
        );
        // The two MoE lanes of one layer. They are not two costs, they are two
        // workers: `decode_step` submits the device batch, runs the host and
        // streamed picks while it is in flight (`cpu_picks`, timed as
        // `MoeLayer::overlap_ns`) and only then waits for the batch. So the layer
        // pays the longer lane, not the sum of both - which is what #430 and #433
        // built and this sum ignored, billing the streamed tier for time the
        // machine spends on the GPU and then fleeing from it into RAM.
        //
        // The consequence the planner needs is the kink: host and stream work is
        // priced 1:1 only until the CPU lane reaches the device lane's time, so
        // the budget worth holding is the one that puts the two lanes level, and
        // every GiB above that buys nothing.
        //
        // A CPU plan is unaffected: pass 2 leaves the device tier empty, so its
        // device lane is zero and the max is the same number the sum used to be.
        let dev_lane = per_token_bytes * device_hit_rate / bw.device_gbs / 1e6;
        // the host tier is read by the CPU out of its own RAM: no PCIe crossing,
        // so it is priced at the profile's cost of a host byte, not the link's.
        // The rate a served run reports about this lane never enters here (#473).
        let host_lane = per_token_bytes * host_hit_rate / bw.host_gbs / 1e6 + stream_ms;
        let mut credit = 0.0f64;
        for lp in &layers {
            let dsrc = if lp.dense_on_device {
                bw.device_gbs
            } else {
                bw.host_gbs
            };
            // The trunk is not inside the overlap window: it runs before the
            // router names the picks and so before there is a batch to hide
            // under. It stays serial.
            ms += lp.dense_bytes as f64 / dsrc / 1e6;
            ms += dev_lane.max(host_lane);
            credit += dev_lane + host_lane - dev_lane.max(host_lane);
        }
        let (moe_device_ms, moe_host_ms) =
            (dev_lane * n_layers as f64, host_lane * n_layers as f64);
        let head_gbs = if gpu { bw.device_gbs } else { bw.host_gbs };
        let lm_head_ms = (cfg.vocab_size * cfg.hidden_size * 2) as f64 / head_gbs / 1e6;
        ms += lm_head_ms;
        let dense_mean = layers.iter().map(|l| l.dense_bytes).sum::<usize>() / n_layers.max(1);
        Plan {
            layers,
            device_bytes_used,
            device_bytes_budget: device_budget,
            host_bytes_used,
            host_bytes_budget: host_budget + off_dense,
            host_dense_bytes: off_dense,
            device_hit_rate,
            host_hit_rate,
            disk_rate,
            expert_bytes: eb,
            dense_bytes_layer: dense_mean,
            ms_per_token: ms,
            lm_head_ms,
            moe_device_ms,
            moe_host_ms,
            overlap_credit_ms: credit,
        }
    }

    pub fn gib(b: usize) -> f64 {
        b as f64 / 2f64.powi(30)
    }

    pub fn tok_s(&self) -> f64 {
        1000.0 / self.ms_per_token.max(1e-9)
    }

    /// Printed in the shape colibri's telemetry uses, so the two are comparable
    /// line for line.
    pub fn report(&self, cfg: &TextConfig) -> String {
        let resident: usize = self.layers.iter().map(|l| l.resident.len()).sum();
        let pooled: usize = self.layers.iter().map(|l| l.pooled.len()).sum();
        let streamed: usize = self.layers.iter().map(|l| l.streamed.len()).sum();
        let slots = self.layers.len() * cfg.num_experts;
        format!(
            "[VK] tier fennec: device {} of {} expert slots ({:.1}%) | host pool {} | disk {} | \
             turn: {:.1}% of routed experts on device, {:.1}% host, {:.1}% disk | \
             device {:.2}/{:.2} GiB, host {:.2}/{:.2} GiB ({:.2} of it the dense trunk) | \
             {:.2} ms/token = {:.2} tok/s | \
             lanes: device {:.2} ms, cpu {:.2} ms, {:.2} ms of it concurrent",
            resident,
            slots,
            100.0 * resident as f64 / slots.max(1) as f64,
            pooled,
            streamed,
            100.0 * self.device_hit_rate,
            100.0 * self.host_hit_rate,
            100.0 * self.disk_rate,
            Plan::gib(self.device_bytes_used),
            Plan::gib(self.device_bytes_budget),
            Plan::gib(self.host_bytes_used),
            Plan::gib(self.host_bytes_budget),
            Plan::gib(self.host_dense_bytes),
            self.ms_per_token,
            self.tok_s(),
            self.moe_device_ms,
            self.moe_host_ms,
            self.overlap_credit_ms,
        )
    }
}

fn dense_bytes_of(cfg: &TextConfig, l: usize, bpp: usize) -> usize {
    match cfg.layer_types[l] {
        crate::config::LayerKind::Linear => dense_bytes_layer(cfg, true, bpp),
        crate::config::LayerKind::Full => dense_bytes_layer(cfg, false, bpp),
    }
}

/// gate + up + down of the always-on shared expert. Priced at half precision
/// because that is what the checkpoint stores and what `half_of` uploads
/// losslessly; an f32 checkpoint keeps it on the host instead.
fn shared_bytes_of(cfg: &TextConfig, bpp: usize) -> usize {
    3 * cfg.shared_expert_intermediate_size * cfg.hidden_size * bpp
}

#[inline]
pub fn frac_device(plan: &Plan) -> f64 {
    plan.device_hit_rate
}

#[cfg(test)]
mod tests {
    use super::{
        Bandwidth, Measured, Plan, TextConfig, Usage, expert_bytes, expert_source_bytes,
        stream_price_ms,
    };
    use crate::config::LayerKind;

    fn scratch(tag: &str) -> String {
        let mut p = std::env::temp_dir();
        p.push(format!("fennec-usage-{}-{}.txt", tag, std::process::id()));
        p.to_string_lossy().into_owned()
    }

    #[test]
    fn a_saved_histogram_reads_back_as_the_same_histogram() {
        // The file is the plan's only memory, so the write format and the read
        // format have to agree exactly - including that a zero cell is absent
        // rather than present, which is what keeps 12k cells from becoming a
        // 12k-line file on a model that routed to nine of them.
        let path = scratch("roundtrip");
        let u = Usage {
            layers: 3,
            experts: 4,
            counts: vec![vec![7, 0, 0, 2], vec![0, 0, 0, 0], vec![1, 1, 0, 0]],
        };
        u.save(&path).unwrap();
        let back = Usage::load(&path).unwrap();
        assert_eq!((back.layers, back.experts), (3, 4));
        assert_eq!(back.counts, u.counts);
        assert_eq!(back.total(), 11);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_second_run_adds_to_the_history_rather_than_replacing_it() {
        // What lets a `--layers` debug run or a three-token request contribute
        // instead of writing zeros over every layer it did not touch.
        let path = scratch("merge");
        Usage { layers: 2, experts: 3, counts: vec![vec![5, 0, 0], vec![0, 0, 9]] }
            .save(&path)
            .unwrap();
        let mut merged = Usage::load(&path).unwrap();
        for (row, ours) in merged.counts.iter_mut().zip([vec![0u64, 3, 0], vec![1, 0, 0]]) {
            for (c, add) in row.iter_mut().zip(ours) {
                *c += add;
            }
        }
        merged.save(&path).unwrap();
        let back = Usage::load(&path).unwrap();
        assert_eq!(back.counts, vec![vec![5, 3, 0], vec![1, 0, 9]]);
        assert_eq!(back.total(), 18);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn measured_rates_round_trip_and_leave_absent_fields_alone() {
        let path = scratch("rates");
        let m = Measured {
            disk_gbs: Some(17.4),
            quant_gbs: None,
            host_gbs: Some(12.1),
            held_ratio: Some(8.9),
            borrows: true,
            threads: Some(8),
            pick_bytes: Some(1_769_472),
        };
        m.save(&path).unwrap();
        let back = Measured::load(&path).unwrap();
        assert_eq!(back.disk_gbs, Some(17.4));
        assert_eq!(back.quant_gbs, None);
        assert!(back.borrows);
        assert_eq!((back.threads, back.pick_bytes), (Some(8), Some(1_769_472)));
        // A rate the run never measured must not overwrite the seed profile:
        // `quant_gbs: None` means "this tier borrowed", not "encode is free".
        let mut bw = Bandwidth::laptop_4060();
        bw.apply(&back);
        assert_eq!(bw.disk_gbs, 17.4);
        assert_eq!(bw.quant_gbs, 1.9);
        assert_eq!(bw.pcie_gbs, 13.5);
        assert_eq!(bw.device_gbs, 272.0);
        // And a later turn of the same shape that saw no streaming must leave the
        // earned disk rate standing rather than writing its own silence over it.
        let (th, pb) = (back.threads, back.pick_bytes);
        let mut kept = back;
        kept.merge(&Measured {
            host_gbs: Some(9.0),
            borrows: true,
            threads: th,
            pick_bytes: pb,
            ..Default::default()
        });
        assert_eq!(kept.disk_gbs, Some(17.4));
        assert_eq!(
            kept.host_gbs,
            Some(12.1),
            "the reported lane achievement is the box's best pass, not one that paid the page faults"
        );
        // The disk lane is priced the other way round: the same pack read twice
        // comes out of the OS cache, and that number is about the cache, not the
        // drive the streamed tier has to reach for on a cold page.
        let mut slow = back;
        slow.merge(&Measured {
            disk_gbs: Some(4.4),
            borrows: true,
            threads: th,
            pick_bytes: pb,
            ..Default::default()
        });
        assert_eq!(slow.disk_gbs, Some(4.4));
        // A different fill shape is a different machine, so its samples do not
        // mix with the ones they would contradict.
        let mut flipped = back;
        flipped.merge(&Measured {
            host_gbs: Some(1.6),
            ..Default::default()
        });
        assert_eq!(flipped.disk_gbs, None, "a copied tier never measured the borrowed lane");
        assert_eq!(flipped.host_gbs, Some(1.6));
        assert!(!flipped.borrows, "the shape flag follows the newest run");
        std::fs::remove_file(&path).ok();
    }

    /// #473, second half: what a run reports about its host lane must not reach
    /// the price of a host byte. r23 ran one shape twice on one box, 90 s apart,
    /// and got 4.2 GB/s then 15.6 - the OS page cache, not the memory - while the
    /// streamed window in the same pair moved 3.8 to 23.6. Pricing the second run
    /// with the first one's sample predicted 1802.5 ms/token for a machine that
    /// served 149.1. Both samples now live in the rates file as contention
    /// evidence; neither is allowed to change the plan.
    #[test]
    fn a_measured_host_window_never_prices_a_host_byte() {
        let seed = Bandwidth::laptop_4060();
        for (cold, warm) in [(4.1668, 15.6), (2.5, 21.7), (31.0, 37.0)] {
            for rate in [cold, warm] {
                let mut bw = Bandwidth::laptop_4060();
                bw.apply(&Measured {
                    host_gbs: Some(rate),
                    borrows: true,
                    threads: Some(16),
                    pick_bytes: Some(1_769_472),
                    ..Default::default()
                });
                assert_eq!(
                    bw.host_gbs,
                    seed.host_gbs,
                    "{rate} GB/s of served lane is reported, not divided by"
                );
            }
        }
        // The rates that do price are the ones measured against the medium they
        // price, and they still override.
        let mut bw = Bandwidth::laptop_4060();
        bw.apply(&Measured {
            host_gbs: Some(4.1668),
            disk_gbs: Some(9.4),
            quant_gbs: Some(2.2),
            borrows: true,
            threads: Some(16),
            pick_bytes: Some(1_769_472),
            ..Default::default()
        });
        assert_eq!((bw.disk_gbs, bw.quant_gbs), (9.4, 2.2));
        assert_eq!(bw.host_gbs, seed.host_gbs);
    }

    /// #473: a rate is a property of a shape, not of a box. The worker count and
    /// the pack size have both changed under a rates file that stayed valid -
    /// #463 left the tier computing at one thread, #225 and #464 moved the pack
    /// between 1.69 and 0.85 MiB - and `warmest` carried the old sample forward as
    /// the price of the new tier. That is how 21.7 GB/s came to price a tier whose
    /// logged windows read 3-8.
    #[test]
    fn a_rate_does_not_survive_the_shape_that_earned_it() {
        let mut m = Measured {
            host_gbs: Some(21.7),
            disk_gbs: Some(3.4),
            borrows: true,
            threads: Some(1),
            pick_bytes: Some(1_769_472),
            ..Default::default()
        };
        m.merge(&Measured {
            host_gbs: Some(4.3),
            borrows: true,
            threads: Some(16),
            pick_bytes: Some(1_769_472),
            ..Default::default()
        });
        assert_eq!(m.host_gbs, Some(4.3), "warmest cannot ratchet across worker counts");
        assert_eq!(m.disk_gbs, None, "the reset is the file, not the field that moved");
        assert_eq!(m.threads, Some(16), "the newest shape is what the next plan is built for");
        // An unlabelled file - every shape field absent, which is what every rates
        // file on disk looked like before this - is not a wildcard either.
        let mut legacy = Measured {
            host_gbs: Some(21.7),
            borrows: true,
            ..Default::default()
        };
        legacy.merge(&Measured {
            host_gbs: Some(4.3),
            borrows: true,
            threads: Some(16),
            pick_bytes: Some(1_769_472),
            ..Default::default()
        });
        assert_eq!(legacy.host_gbs, Some(4.3));
        // Same shape both sides, and the rule is still the warmest pass: #464
        // bounded what warmth can be worth (1.27x) rather than making it impossible.
        let mut same = Measured {
            host_gbs: Some(4.3),
            threads: Some(16),
            pick_bytes: Some(1_769_472),
            ..Default::default()
        };
        same.merge(&Measured {
            host_gbs: Some(4.9),
            threads: Some(16),
            pick_bytes: Some(1_769_472),
            ..Default::default()
        });
        assert_eq!(same.host_gbs, Some(4.9));
        assert_eq!(same.shape_note(), "16 workers per pick, 1.69 MiB each");
        assert_eq!(
            Measured { threads: Some(8), ..Default::default() }.shape_note(),
            "8 workers per pick, pack size unrecorded"
        );
        assert_eq!(Measured::default().shape_note(), "shape unrecorded");
    }

    #[test]
    fn empty_measured_set_changes_nothing() {
        let mut bw = Bandwidth::laptop_4060();
        let seed = bw;
        bw.apply(&Measured::default());
        assert!(Measured::default().is_empty());
        assert_eq!((bw.device_gbs, bw.host_gbs, bw.disk_gbs, bw.quant_gbs),
            (seed.device_gbs, seed.host_gbs, seed.disk_gbs, seed.quant_gbs));
    }

    /// The borrowed shape moves a third of the bytes and skips the encode pass,
    /// so on the same box it prices several times cheaper. If a future edit makes
    /// the two agree again, the plan is back to charging a zero-copy tier for an
    /// FQ4 encode it never runs.
    #[test]
    fn borrowed_stream_prices_below_copied() {
        let bw = Bandwidth::laptop_4060();
        let (pack, source, k) = (1_780_000usize, 6_291_456usize, 8usize);
        let borrowed = stream_price_ms(true, pack, source, k, 0.02, &bw);
        let copied = stream_price_ms(false, pack, source, k, 0.02, &bw);
        assert!(borrowed * 4.0 < copied, "{borrowed} vs {copied}");
        // Nothing streamed means nothing priced, whatever the shape.
        assert_eq!(stream_price_ms(false, pack, source, k, 0.0, &bw), 0.0);
        // And the compute half survives an infinitely fast drive: a pick read for
        // free still pays the kernel that applies it, at the same price the same
        // pick costs in the RAM tier. Without that term the streamed tier reads as
        // a discount on RAM and the solver cuts the RAM budget for nothing.
        let free = Bandwidth {
            disk_gbs: f64::INFINITY,
            quant_gbs: f64::INFINITY,
            ..bw
        };
        let share = 0.02f64;
        let streamed = stream_price_ms(true, pack, source, k, share, &free);
        let resident = (k * pack) as f64 * share / bw.host_gbs / 1e6;
        assert!(
            (streamed - resident).abs() < 1e-12,
            "a free refill must leave exactly the pick's own compute: {streamed} vs {resident}"
        );
    }

    /// A four-layer, eight-expert MoE small enough to price in a test and shaped
    /// like the real thing: GDN layers interleaved with full attention, a shared
    /// expert, and a router that picks two of eight.
    fn tiny_moe() -> TextConfig {
        TextConfig {
            hidden_size: 512,
            num_hidden_layers: 4,
            vocab_size: 1024,
            rms_norm_eps: 1e-6,
            norm_zero_centered: true,
            layer_types: vec![LayerKind::Linear, LayerKind::Full, LayerKind::Linear, LayerKind::Full],
            num_attention_heads: 4,
            num_key_value_heads: 4,
            head_dim: 128,
            partial_rotary_factor: 1.0,
            rope_theta: 1e6,
            attn_output_gate: false,
            intermediate_size: 0,
            tie_word_embeddings: false,
            linear_num_key_heads: 4,
            linear_num_value_heads: 4,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            num_experts: 8,
            num_experts_per_tok: 2,
            moe_intermediate_size: 256,
            shared_expert_intermediate_size: 256,
            mtp_num_hidden_layers: 0,
            mtp_use_dedicated_embeddings: false,
            vision: None,
            image_token_id: None,
            vision_start_token_id: None,
            vision_end_token_id: None,
            prefix: "model".into(),
            name: "tiny-moe-test".into(),
        }
    }

    /// One layer's routing histogram: expert 0 twice as hot as 1, and so on down
    /// to a tail the streamed tier exists to serve. Every layer the same, so a
    /// tier's hit rate is just the sum of the counts it holds.
    fn tiny_usage() -> Usage {
        let row = vec![64u64, 32, 16, 8, 4, 2, 2, 1];
        Usage {
            layers: 4,
            experts: 8,
            counts: vec![row; 4],
        }
    }

    /// A router that spends almost every pick on two experts, which is the shape
    /// that puts the CPU lane under the device lane and so the only shape the
    /// level-point claim can be tested on. `tiny_usage`'s flatter tail keeps the
    /// drive busy no matter how large the RAM tier gets, and a tier that is still
    /// being paid for has no free lane to hide in.
    fn dominant_usage() -> Usage {
        let row = vec![200u64, 4, 2, 1, 0, 0, 0, 0];
        Usage {
            layers: 4,
            experts: 8,
            counts: vec![row; 4],
        }
    }

    /// The serial half of the price, read off the plan's own rows: the dense block
    /// of every layer (its mixer, its router and its shared expert) at the
    /// bandwidth of the copy it ended up in. `l.dense_bytes` already carries the
    /// bytes-per-weight of that copy, so the row says both how much and from where.
    fn dense_ms(plan: &Plan, bw: &Bandwidth) -> f64 {
        plan.layers
            .iter()
            .map(|l| {
                let src = if l.dense_on_device {
                    bw.device_gbs
                } else {
                    bw.host_gbs
                };
                l.dense_bytes as f64 / src / 1e6
            })
            .sum::<f64>()
    }

    /// Expert slots per layer: the dense blocks plus the shared experts come off
    /// the device budget first, so a tier is sized relative to that, not in the
    /// abstract. 76 is what this configuration spends before it owns one expert.
    const TINY_BASE_EB: usize = 76;

    #[test]
    fn a_layer_bills_the_longer_lane_not_both_lanes_added() {
        let cfg = tiny_moe();
        let (usage, bw, group) = (tiny_usage(), Bandwidth::laptop_4060(), 64);
        let eb = expert_bytes(&cfg, group);
        // Three hot experts per layer in VRAM, two more in RAM, the tail streamed.
        let plan = Plan::build(&cfg, &usage, (TINY_BASE_EB + 9) * eb, 8 * eb, 0, &bw, group, true, true);
        let (res, pool, stream) = (
            plan.layers.iter().map(|l| l.resident.len()).sum::<usize>(),
            plan.layers.iter().map(|l| l.pooled.len()).sum::<usize>(),
            plan.layers.iter().map(|l| l.streamed.len()).sum::<usize>(),
        );
        assert!(
            res > 0 && pool > 0 && stream > 0,
            "a test that only filled one tier would prove nothing about two lanes: \
             device {res} host {pool} disk {stream}"
        );

        let n = plan.layers.len() as f64;
        let per_token = (cfg.num_experts_per_tok * eb) as f64;
        assert!(
            (per_token * plan.device_hit_rate / bw.device_gbs / 1e6 * n - plan.moe_device_ms).abs()
                < 1e-12,
            "moe_device_ms is the device lane of every layer: {:?}",
            plan.moe_device_ms
        );
        let stream = stream_price_ms(
            true,
            eb,
            expert_source_bytes(&cfg),
            cfg.num_experts_per_tok,
            plan.disk_rate,
            &bw,
        );
        let host_lane = per_token * plan.host_hit_rate / bw.host_gbs / 1e6 + stream;
        assert!(
            (host_lane * n - plan.moe_host_ms).abs() < 1e-12,
            "moe_host_ms is the CPU lane (its own reads plus the streamed refill) of every layer"
        );
        // The MoE of a layer is billed once, at its longer lane, so what the
        // concurrency earns back is exactly the shorter lane times the layers.
        assert!(
            plan.overlap_credit_ms > 0.0,
            "both tiers hold experts, so one lane sits idle while the other works"
        );
        assert!(
            (plan.moe_device_ms + plan.moe_host_ms - plan.overlap_credit_ms
                - plan.moe_device_ms.max(plan.moe_host_ms))
                .abs()
                < 1e-9,
            "the credit is the difference between the sum of the lanes and the longer one"
        );
        // And the whole token is the serial parts plus one billed lane per layer -
        // no term of the price is unaccounted for.
        let want = dense_ms(&plan, &bw)
            + plan.moe_device_ms.max(plan.moe_host_ms)
            + plan.lm_head_ms;
        assert!(
            (want - plan.ms_per_token).abs() < 1e-9,
            "reconstructed {want} != billed {}",
            plan.ms_per_token
        );
    }

    #[test]
    fn a_cpu_plan_is_billed_exactly_as_the_serial_sum_was() {
        // The change has to be invisible on the lane that has only one worker. If
        // it moves this plan at all, the CPU goldens and every comparison built on
        // them (#101, #187) move with it.
        let cfg = tiny_moe();
        let (usage, bw, group) = (tiny_usage(), Bandwidth::cpu_only(), 64);
        let eb = expert_bytes(&cfg, group);
        let plan = Plan::build(&cfg, &usage, (TINY_BASE_EB + 9) * eb, 8 * eb, 0, &bw, group, false, true);
        assert_eq!(plan.device_hit_rate, 0.0, "no device, so no device tier");
        assert_eq!(plan.moe_device_ms, 0.0);
        assert_eq!(
            plan.overlap_credit_ms, 0.0,
            "there is nothing to hide under a lane that does not exist"
        );
        let n = plan.layers.len() as f64;
        let per_token = (cfg.num_experts_per_tok * eb) as f64;
        let stream = stream_price_ms(
            true,
            eb,
            expert_source_bytes(&cfg),
            cfg.num_experts_per_tok,
            plan.disk_rate,
            &bw,
        );
        let serial = per_token * plan.host_hit_rate / bw.host_gbs / 1e6 + stream;
        let want = dense_ms(&plan, &bw) + n * serial + plan.lm_head_ms;
        assert!(
            (want - plan.ms_per_token).abs() < 1e-9,
            "the serial price of the old planner: {want} vs billed {}",
            plan.ms_per_token
        );
    }

    #[test]
    fn a_plan_with_one_tier_full_has_nothing_to_credit() {
        let cfg = tiny_moe();
        let (usage, bw, group) = (tiny_usage(), Bandwidth::laptop_4060(), 64);
        let eb = expert_bytes(&cfg, group);
        // Every expert resident: the CPU lane goes empty, so max and sum agree.
        let plan = Plan::build(&cfg, &usage, (TINY_BASE_EB + 64) * eb, 0, 0, &bw, group, true, true);
        assert_eq!(plan.host_hit_rate, 0.0);
        assert_eq!(plan.disk_rate, 0.0);
        assert_eq!(plan.moe_host_ms, 0.0);
        assert_eq!(plan.overlap_credit_ms, 0.0);
        assert!(plan.ms_per_token > 0.0);
    }

    #[test]
    fn ram_beyond_the_level_point_buys_nothing() {
        // The kink the brief turns on: while the CPU lane is the shorter of the
        // two, moving picks off the drive costs the token nothing, so every GiB of
        // RAM above that point is RAM bought for nothing. Under the serial price
        // each of these plans was billed for the refill it was already hiding.
        let cfg = tiny_moe();
        let (usage, bw, group) = (dominant_usage(), Bandwidth::laptop_4060(), 64);
        let eb = expert_bytes(&cfg, group);
        // Two slots per layer in VRAM: the hot pair, which is 98.6% of the picks,
        // so the device lane is the long one.
        let device = (TINY_BASE_EB + 9) * eb;
        let narrow = Plan::build(&cfg, &usage, device, 4 * eb, 0, &bw, group, true, true);
        let wide = Plan::build(&cfg, &usage, device, 8 * eb, 0, &bw, group, true, true);
        assert!(
            narrow.moe_host_ms < narrow.moe_device_ms && wide.moe_host_ms < wide.moe_device_ms,
            "both plans must have the CPU lane inside the device lane: \
             narrow {:.6}/{:.6} wide {:.6}/{:.6}",
            narrow.moe_host_ms,
            narrow.moe_device_ms,
            wide.moe_host_ms,
            wide.moe_device_ms
        );
        assert!(
            wide.host_hit_rate > narrow.host_hit_rate && wide.disk_rate < narrow.disk_rate,
            "the wider tier has to have actually moved picks off the drive, or the two plans \
             are the same plan: host {:.4}->{:.4} disk {:.4}->{:.4}",
            narrow.host_hit_rate,
            wide.host_hit_rate,
            narrow.disk_rate,
            wide.disk_rate
        );
        assert_eq!(
            wide.ms_per_token, narrow.ms_per_token,
            "a placement that hides all its CPU work under the device lane decodes at the \
             device lane's speed whatever it moved off the drive"
        );
        // What the old price would have said, stated rather than assumed: the sum
        // of the lanes does fall as RAM widens, so a planner billing it would have
        // bought this coverage and reported it as speed.
        assert!(
            narrow.moe_device_ms + narrow.moe_host_ms
                > wide.moe_device_ms + wide.moe_host_ms,
            "the serial sum would have ranked the wider tier faster"
        );
    }
}
