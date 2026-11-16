//! The tier scheduler: what colibri does well, made explicit.
//!
//! Given a device budget, a host-RAM pool and a histogram of which experts the
//! model actually routes to, decide per layer how many experts sit on the GPU,
//! how many stay in system RAM, and how many are streamed from disk. Then price
//! a decode token in bytes-per-tier against measured bandwidth, so two plans can
//! be compared before anything is built.

use crate::config::{Result, TextConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Device,
    Host,
    Disk,
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
}

/// Bytes moved per second per tier, for the box this runs on, plus the FQ4
/// encode rate: a streamed expert is read *and* quantised, and the encode was
/// the half the original model left out, which is why the disk tier measured
/// three times slower than the plan said it would be.
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
    /// `host_gbs` is 17, not the 45 the spec sheet claims: `fennec bench` streams
    /// the MoE block at 18.5 GB/s and the lm_head at 15.4-17 GB/s on this box,
    /// and a plan priced at the marketing number predicts three times the tokens
    /// per second the stopwatch then refuses to produce. The CPU reads its own
    /// RAM at that rate whether or not a GPU is attached, so both profiles share
    /// it; only `device_gbs` differs.
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
    /// is a second name for the host tier and every bandwidth is the host's.
    pub fn cpu_only() -> Bandwidth {
        Bandwidth {
            device_gbs: 17.0,
            host_gbs: 17.0,
            pcie_gbs: 17.0,
            disk_gbs: 3.2,
            quant_gbs: 1.9,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LayerPlan {
    pub layer: usize,
    pub resident: Vec<usize>,
    pub pooled: Vec<usize>,
    pub streamed: Vec<usize>,
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
    pub host_bytes_budget: usize,
    pub device_hit_rate: f64,
    pub host_hit_rate: f64,
    pub disk_rate: f64,
    pub expert_bytes: usize,
    pub dense_bytes_layer: usize,
    pub ms_per_token: f64,
    pub lm_head_ms: f64,
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
    pub fn build(
        cfg: &TextConfig,
        usage: &Usage,
        device_budget: usize,
        host_budget: usize,
        reserve: usize,
        bw: &Bandwidth,
        group: usize,
        gpu: bool,
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
        let mut host_bytes_used = off_dense.min(host_budget);

        // pass 2: split the remaining device room evenly over the layers, then fill
        // system RAM with the next-hot experts per layer. In CPU mode (gpu=false),
        // the device tier doesn't exist, so all experts go to host or disk.
        let free_device = if gpu { budget.saturating_sub(device_bytes_used) } else { 0 };
        let mut slots_dev = vec![0usize; n_layers];
        let mut rem = free_device / eb.max(1);
        for r in 0..n_layers {
            let take = rem / (n_layers - r);
            slots_dev[r] = take;
            rem -= take;
        }
        let free_host = host_budget.saturating_sub(host_bytes_used);
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
            let nh = slots_host[l].min(e - nd);
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
        let mut hit = [0u64; 3];
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
        for lp in &layers {
            let dsrc = if lp.dense_on_device {
                bw.device_gbs
            } else {
                bw.host_gbs
            };
            ms += lp.dense_bytes as f64 / dsrc / 1e6;
            ms += per_token_bytes * device_hit_rate / bw.device_gbs / 1e6;
            // the host tier is read by the CPU out of its own RAM: no PCIe
            // crossing, so it is priced at host bandwidth, not the link's
            ms += per_token_bytes * host_hit_rate / bw.host_gbs / 1e6;
            let src = expert_source_bytes(cfg) as f64;
            ms += k as f64 * disk_rate * src * (1.0 / bw.disk_gbs + 1.0 / bw.quant_gbs) / 1e6;
        }
        let head_gbs = if gpu { bw.device_gbs } else { bw.host_gbs };
        let lm_head_ms = (cfg.vocab_size * cfg.hidden_size * 2) as f64 / head_gbs / 1e6;
        ms += lm_head_ms;
        let dense_mean = layers.iter().map(|l| l.dense_bytes).sum::<usize>() / n_layers.max(1);
        Plan {
            layers,
            device_bytes_used,
            device_bytes_budget: device_budget,
            host_bytes_used,
            host_bytes_budget: host_budget,
            device_hit_rate,
            host_hit_rate,
            disk_rate,
            expert_bytes: eb,
            dense_bytes_layer: dense_mean,
            ms_per_token: ms,
            lm_head_ms,
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
             device {:.2}/{:.2} GiB, host {:.2}/{:.2} GiB | {:.2} ms/token = {:.2} tok/s",
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
            self.ms_per_token,
            self.tok_s(),
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
