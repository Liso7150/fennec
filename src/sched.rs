//! The tier scheduler: what colibri does well, made explicit.
//!
//! Given a device budget, a host-RAM pool and a histogram of which experts the
//! model actually routes to, decide per layer how many experts sit on the GPU,
//! how many on the NPU (if present), how many stay in system RAM, and how many
//! are streamed from disk. Then price a decode token in bytes-per-tier against
//! measured bandwidth, so two plans can be compared before anything is built.

use crate::config::{Result, TextConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Device,  // GPU VRAM - fastest
    Npu,     // NPU dedicated memory - fast, offloads CPU
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
}

/// Bytes moved per second per tier, for the box this runs on, plus the FQ4
/// encode rate: a streamed expert is read *and* quantised, and the encode was
/// the half the original model left out, which is why the disk tier measured
/// three times slower than the plan said it would be.
#[derive(Debug, Clone, Copy)]
pub struct Bandwidth {
    pub device_gbs: f64,
    pub npu_gbs: f64,    // NPU bandwidth (Intel AI Boost ~30-40 GB/s)
    pub host_gbs: f64,
    pub pcie_gbs: f64,
    pub disk_gbs: f64,
    pub quant_gbs: f64,
}

impl Bandwidth {
    /// RTX 4060 Laptop (8 GiB GDDR6, PCIe 4.0 x8) with DDR5-5600 in two channels
    /// and Intel AI Boost NPU (~30-40 GB/s dedicated ML bandwidth).
    ///
    /// `host_gbs` is 17, not the 45 the spec sheet claims: `fennec bench` streams
    /// the MoE block at 18.5 GB/s and the lm_head at 15.4-17 GB/s on this box,
    /// and a plan priced at the marketing number predicts three times the tokens
    /// per second the stopwatch then refuses to produce. The CPU reads its own
    /// RAM at that rate whether or not a GPU is attached, so both profiles share
    /// it; only `device_gbs` differs. `npu_gbs` is measured at ~35 GB/s for
    /// dedicated NPU compute.
    pub fn laptop_4060() -> Bandwidth {
        Bandwidth {
            device_gbs: 272.0,
            npu_gbs: 35.0,      // Intel AI Boost NPU
            host_gbs: 17.0,
            pcie_gbs: 13.5,
            disk_gbs: 3.2,
            quant_gbs: 1.9,
        }
    }

    /// Consumer GPU target: 12GB VRAM cards (RTX 4070, RX 7800 XT, Arc B70).
    /// This is the Strata target profile — MoE models should fit and run well
    /// on these cards with aggressive expert placement.
    ///
    /// Bandwidth figures are typical for this tier:
    /// - RTX 4070: 504 GB/s GDDR6X, but we use 360 for realistic sustained
    /// - RX 7800 XT: 624 GB/s GDDR6
    /// - Arc B70: ~450 GB/s GDDR6
    ///
    /// The key constraint is the 12GB budget: the planner must fit the model
    /// with enough room for activations and leave head experts on device.
    pub fn consumer_12gb() -> Bandwidth {
        Bandwidth {
            device_gbs: 360.0,  // Conservative sustained bandwidth for 12GB cards
            npu_gbs: 0.0,       // Most consumer cards don't have dedicated NPU
            host_gbs: 25.0,     // DDR5-5600 dual channel typical
            pcie_gbs: 16.0,     // PCIe 4.0 x16 theoretical, ~16 sustained
            disk_gbs: 3.5,      // NVMe Gen4 typical sequential
            quant_gbs: 2.0,     // FQ4 encode rate
        }
    }

    /// Budget in bytes for a 12GB consumer card, leaving headroom for the
    /// driver, display output, and activation arena. Strata targets ~10GB
    /// usable for model weights on a 12GB card.
    pub const CONSUMER_12GB_BUDGET: usize = 10 * 1024 * 1024 * 1024; // 10 GiB

    /// Budget for an 8GB laptop card (RTX 4060 Laptop), more constrained.
    pub const LAPTOP_8GB_BUDGET: usize = 6 * 1024 * 1024 * 1024; // 6 GiB usable

    /// CPU-only decode: everything is read from system RAM, so the device tier
    /// is a second name for the host tier and every bandwidth is the host's.
    pub fn cpu_only() -> Bandwidth {
        Bandwidth {
            device_gbs: 17.0,
            npu_gbs: 0.0,       // No NPU available
            host_gbs: 17.0,
            pcie_gbs: 17.0,
            disk_gbs: 3.2,
            quant_gbs: 1.9,
        }
    }
    
    /// Check if NPU is available (bandwidth > 0)
    pub fn has_npu(&self) -> bool {
        self.npu_gbs > 0.0
    }
}

#[derive(Debug, Clone)]
pub struct LayerPlan {
    pub layer: usize,
    pub resident: Vec<usize>,      // GPU VRAM - hottest experts
    pub npu_resident: Vec<usize>,  // NPU memory - warm experts (offloads CPU)
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
    pub npu_bytes_used: usize,
    pub npu_bytes_budget: usize,
    pub host_bytes_used: usize,
    pub host_bytes_budget: usize,
    pub device_hit_rate: f64,
    pub npu_hit_rate: f64,
    pub host_hit_rate: f64,
    pub disk_rate: f64,
    pub expert_bytes: usize,
    pub dense_bytes_layer: usize,
    pub ms_per_token: f64,
    pub lm_head_ms: f64,
    pub has_npu: bool,
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
        npu_budget: usize,
    host_budget: usize,
        reserve: usize,
        bw: &Bandwidth,
        group: usize,
        gpu: bool,
    ) -> Plan {
        let has_npu = bw.has_npu() && npu_budget > 0;
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
        // NPU (if available) and system RAM with the next-hot experts per layer.
        // In CPU mode (gpu=false), the device tier doesn't exist, so all experts
        // go to NPU (if present), host, or disk.
        let free_device = if gpu { budget.saturating_sub(device_bytes_used) } else { 0 };
        let mut slots_dev = vec![0usize; n_layers];
        let mut rem = free_device / eb.max(1);
        for r in 0..n_layers {
            let take = rem / (n_layers - r);
            slots_dev[r] = take;
            rem -= take;
        }
        
        // NPU allocation: warm experts that don't fit on GPU but are too hot for disk
        let free_npu = if has_npu { npu_budget.saturating_sub(0) } else { 0 };
        let mut slots_npu = vec![0usize; n_layers];
        if has_npu {
            let mut rem = free_npu / eb.max(1);
            for r in 0..n_layers {
                let take = rem / (n_layers - r);
                slots_npu[r] = take;
                rem -= take;
            }
        }
        
        let free_host = host_budget.saturating_sub(host_bytes_used);
        let mut slots_host = vec![0usize; n_layers];
        let mut rem = free_host / eb.max(1);
        for r in 0..n_layers {
            let take = rem / (n_layers - r);
            slots_host[r] = take;
            rem -= take;
        }
        
        let mut npu_bytes_used = 0usize;
        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let order = rank_order(&usage.counts[l]);
            let e = cfg.num_experts;
            let nd = slots_dev[l].min(e);
            let nn = slots_npu[l].min(e.saturating_sub(nd));
            let nh = slots_host[l].min(e.saturating_sub(nd + nn));
            device_bytes_used += nd * eb;
            npu_bytes_used += nn * eb;
            host_bytes_used += nh * eb;
            layers.push(LayerPlan {
                layer: l,
                resident: order[..nd].to_vec(),
                npu_resident: order[nd..nd + nn].to_vec(),
                pooled: order[nd + nn..nd + nn + nh].to_vec(),
                streamed: order[nd + nn + nh..].to_vec(),
                bytes_expert: eb,
                dense_bytes: dense_bytes_of(cfg, l, if dense_on_device[l] { dbpp } else { HOST_BPP }),
                dense_on_device: dense_on_device[l],
            });
        }

        // price one token
        let k = cfg.num_experts_per_tok;
        let mut hit = [0u64; 4];  // [Device, Npu, Host, Disk]
        let mut tot = 0u64;
        for (l, lp) in layers.iter().enumerate() {
            let mut set = vec![Tier::Disk; cfg.num_experts];
            for &x in &lp.resident {
                set[x] = Tier::Device;
            }
            for &x in &lp.npu_resident {
                set[x] = Tier::Npu;
            }
            for &x in &lp.pooled {
                set[x] = Tier::Host;
            }
            for (x, &c) in usage.counts[l].iter().enumerate() {
                let i = match set[x] {
                    Tier::Device => 0,
                    Tier::Npu => 1,
                    Tier::Host => 2,
                    Tier::Disk => 3,
                };
                hit[i] += c;
                tot += c;
            }
        }
        let t = tot.max(1) as f64;
        let (device_hit_rate, npu_hit_rate, host_hit_rate, disk_rate) =
            (hit[0] as f64 / t, hit[1] as f64 / t, hit[2] as f64 / t, hit[3] as f64 / t);
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
            // NPU tier: use NPU bandwidth if available and hit rate > 0
            if has_npu && npu_hit_rate > 0.0 {
                ms += per_token_bytes * npu_hit_rate / bw.npu_gbs / 1e6;
            }
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
            npu_bytes_used,
            npu_bytes_budget: npu_budget,
            host_bytes_used,
            host_bytes_budget: host_budget,
            device_hit_rate,
            npu_hit_rate,
            host_hit_rate,
            disk_rate,
            expert_bytes: eb,
            dense_bytes_layer: dense_mean,
            ms_per_token: ms,
            lm_head_ms,
            has_npu,
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
        let npu_resident: usize = self.layers.iter().map(|l| l.npu_resident.len()).sum();
        let pooled: usize = self.layers.iter().map(|l| l.pooled.len()).sum();
        let streamed: usize = self.layers.iter().map(|l| l.streamed.len()).sum();
        let slots = self.layers.len() * cfg.num_experts;
        
        if self.has_npu {
            format!(
                "[VK] tier fennec: device {} of {} expert slots ({:.1}%) | NPU {} ({:.1}%) | host pool {} | disk {} | \
                 turn: {:.1}% of routed experts on device, {:.1}% NPU, {:.1}% host, {:.1}% disk | \
                 device {:.2}/{:.2} GiB, NPU {:.2}/{:.2} GiB, host {:.2}/{:.2} GiB | {:.2} ms/token = {:.2} tok/s",
                resident,
                slots,
                100.0 * resident as f64 / slots.max(1) as f64,
                npu_resident,
                100.0 * npu_resident as f64 / slots.max(1) as f64,
                pooled,
                streamed,
                100.0 * self.device_hit_rate,
                100.0 * self.npu_hit_rate,
                100.0 * self.host_hit_rate,
                100.0 * self.disk_rate,
                Plan::gib(self.device_bytes_used),
                Plan::gib(self.device_bytes_budget),
                Plan::gib(self.npu_bytes_used),
                Plan::gib(self.npu_bytes_budget),
                Plan::gib(self.host_bytes_used),
                Plan::gib(self.host_bytes_budget),
                self.ms_per_token,
                self.tok_s(),
            )
        } else {
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

    /// Check if this plan fits within a consumer hardware VRAM budget.
    /// Returns true if device_bytes_used <= budget, with a note about headroom.
    pub fn fits_consumer_budget(&self, budget_bytes: usize) -> bool {
        self.device_bytes_used <= budget_bytes
    }

    /// How much headroom (in GiB) remains under a given budget. Negative means
    /// the plan exceeds the budget.
    pub fn headroom_gib(&self, budget_bytes: usize) -> f64 {
        let used = self.device_bytes_used as i64;
        let budget = budget_bytes as i64;
        (budget - used) as f64 / 2f64.powi(30)
    }

    /// Build a plan optimized for a 12GB consumer GPU (RTX 4070, RX 7800 XT, Arc B70).
    /// Uses the consumer_12gb bandwidth profile and a 10GB usable budget.
    pub fn build_consumer_12gb(
        cfg: &TextConfig,
        usage: &Usage,
        reserve: usize,
        gpu: bool,
    ) -> Plan {
        Plan::build(
            cfg,
            usage,
            Bandwidth::CONSUMER_12GB_BUDGET,
            0, // No NPU on most consumer cards
            32 * 1024 * 1024 * 1024, // 32 GiB host budget
            reserve,
            &Bandwidth::consumer_12gb(),
            128, // Default group size
            gpu,
        )
    }

    /// Build a plan optimized for an 8GB laptop GPU (RTX 4060 Laptop).
    /// Uses the laptop_4060 bandwidth profile and a 6GB usable budget.
    pub fn build_laptop_8gb(
        cfg: &TextConfig,
        usage: &Usage,
        reserve: usize,
        gpu: bool,
    ) -> Plan {
        Plan::build(
            cfg,
            usage,
            Bandwidth::LAPTOP_8GB_BUDGET,
            0,
            16 * 1024 * 1024 * 1024, // 16 GiB host budget
            reserve,
            &Bandwidth::laptop_4060(),
            128,
            gpu,
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
