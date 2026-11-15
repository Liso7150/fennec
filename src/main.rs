//! fennec — the CLI that answers the three questions this engine exists for:
//! can it read the checkpoint, where do the experts have to live, and how fast
//! is the int4 CPU path on real weights.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::io::Write;
use std::time::Instant;

use fennec::backend::{self, Backend};
use fennec::config::TextConfig;
use fennec::gpu::Gpu;
use fennec::kernel;
use fennec::model::{Model, MoeLayer};
use fennec::quant::Q4;
use fennec::safetensors::Shards;
use fennec::sched::{Bandwidth, Plan, Usage};
use fennec::vk::Vulkan;

struct Opts {
    pos: Vec<String>,
    kv: BTreeMap<String, String>,
    flags: Vec<String>,
}

fn parse(args: &[String]) -> Opts {
    let mut o = Opts {
        pos: Vec::new(),
        kv: BTreeMap::new(),
        flags: Vec::new(),
    };
    let mut it = args.iter().peekable();
    while let Some(a) = it.next() {
        let Some(rest) = a.strip_prefix("--") else {
            o.pos.push(a.clone());
            continue;
        };
        if let Some((k, v)) = rest.split_once('=') {
            o.kv.insert(k.to_string(), v.to_string());
            continue;
        }
        match it.peek() {
            Some(n) if !n.starts_with('-') => {
                o.kv.insert(rest.to_string(), it.next().unwrap().clone());
            }
            _ => o.flags.push(rest.to_string()),
        }
    }
    o
}

impl Opts {
    fn num(&self, key: &str, default: f64) -> f64 {
        match self.kv.get(key) {
            Some(v) => v
                .parse::<f64>()
                .unwrap_or_else(|_| panic!("--{key} is not a number: {v}")),
            None => default,
        }
        .max(0.0)
    }
    fn usize_at(&self, key: &str, default: usize) -> usize {
        self.num(key, default as f64) as usize
    }
    fn str(&self, key: &str) -> Option<String> {
        self.kv.get(key).cloned()
    }
    /// Whether the caller supplied the key at all, in either spelling. `--gpu`
    /// with no value lands in `flags`, `--device 8` lands in `kv`; a caller that
    /// only asked `flags` silently ignored every valued option.
    fn has(&self, key: &str) -> bool {
        self.kv.contains_key(key) || self.flags.iter().any(|f| f == key)
    }
    /// Whether a shipped-by-default flag is on. `--no-{key}` turns it back off,
    /// `{key}` forces it on, and anything else is the default.
    fn on(&self, key: &str, default: bool) -> bool {
        if self.has(&format!("no-{key}")) {
            return false;
        }
        if self.has(key) {
            return true;
        }
        default
    }
    fn dir(&self) -> String {
        self.str("model")
            .or_else(|| self.pos.first().cloned())
            .or_else(|| env_name(&["FENNEC_MODEL", "COLI_MODEL"]))
            .unwrap_or_else(|| DEFAULT_MODEL.to_string())
    }
}

const DEFAULT_MODEL: &str = "D:/MEGAsync-sync/colibri-models/Qwen3.6-35B-A3B-bf16";

/// Eight token ids from the golden fixtures - the same sequence the
/// transformers reference stepped through, so a decode here and a golden test
/// there are answering the same question.
const DEFAULT_PROMPT: &[usize] = &[15059, 35755, 67084, 23774, 57035, 49208, 58227, 92453];

fn gib(b: usize) -> f64 {
    b as f64 / 2f64.powi(30)
}

fn mib(b: usize) -> f64 {
    b as f64 / 2f64.powi(20)
}

fn main() {
    let all: Vec<String> = std::env::args().skip(1).collect();
    let cmd = all.first().cloned().unwrap_or_else(|| "help".to_string());
    
    // Handle version flags before command parsing
    if cmd == "--version" || cmd == "-V" || cmd == "-v" {
        println!("fennec {}", fennec::VERSION);
        return;
    }
    
    let opts = parse(&all[1.min(all.len())..]);
    let cores = kernel::default_threads();
    let threads = opts.usize_at("threads", cores);
    kernel::set_threads(threads);
    fennec::model::set_dense_int4(opts.has("dense-int4"));
    fennec::model::set_dense_i8(opts.on("dense-i8", true));
    fennec::quant::set_planar_dot(opts.has("planar-dot"));
    let r = match cmd.as_str() {
        "info" => info(&opts),
        "doctor" => doctor(&opts),
        "plan" => plan(&opts),
        "probe" => {
            probe();
            Ok(())
        }
        "quant" => quant(&opts, threads),
        "bench" => bench(&opts, threads),
        "decode" => decode(&opts, threads),
        "serve" => serve_cmd(&opts, threads),
        "run" => run_cmd(&opts, threads),
        "help" => {
            usage();
            Ok(())
        }
        other => Err(format!("unknown command `{other}`")),
    };
    if let Err(e) = r {
        eprintln!("fennec {cmd}: {e}");
        std::process::exit(1);
    }
}

fn usage() {
    println!(
        "fennec {} — one engine built from fox + colibri + rabbit\n\n\
         usage: fennec <command> [dir] [options]\n\
         \x20       fennec --version\n\n\
         commands\n\
         \x20 info              checkpoint config, shards and tensor inventory\n\
         \x20 doctor            named health checks as JSON, for the desktop panel\n\
         \x20 plan              device/host/disk expert placement and a tok/s model\n\
         \x20 probe             every Vulkan device, DXGI adapter and DXCore NPU\n\
         \x20 quant             write FQ4 int4 sidecars for --layers\n\
         \x20 bench             int4 expert throughput on real weights\n\
         \x20 decode            run the plan: greedy tokens, measured against predicted ms\n\
         \x20 run               one prompt, streamed to the terminal\n\
         \x20 serve             OpenAI-compatible HTTP on loopback\n\
         \x20 help              show this help message\n\n\
         options\n\
         \x20 --group 32        quant group width (32 is the AVX2 path)\n\
         \x20 --threads N       default: one worker per physical core\n\
         \x20 --device GiB      GPU budget (default: the whole card)  --reserve 0.5 stays for KV and drivers\n\
         \x20 --host 18.0       system-RAM pool in GiB\n\
         \x20 --cpu             price the plan with no GPU at all\n\
         \x20 --gpu [N]         device tier in VRAM: dense, experts + lm_head on ranked device N (default 0)\n\
         \x20 --backend NAME    cuda | vulkan | metal | cpu | ort (default: NVIDIA=cuda, Apple=metal, else vulkan; ort=ONNX Runtime dense)\n\
         \x20 --usage PATH      .coli_usage router histogram (default: found next to the model)\n\
         \x20 --layers 0,3,39   which layers to quantise, or `all`\n\
         \x20 --out PATH        sidecar path for `quant`\n\
         \x20 --fq4 PATH        FQ4 expert cache to load instead of re-quantising (or FENNEC_FQ4)\n\
         \x20 --curve           print the residency-coverage curve used by the graphs\n\
         \x20 --first 0         first layer of the `decode` window\n\
         \x20 --window 4        how many layers `decode` holds in RAM\n\
         \x20 --slots 16        streamed experts each layer keeps warm\n\
         \x20 --tokens 32       new tokens `decode` samples after the prompt\n\
         \x20 --prompt IDS      comma-separated token ids (default: the fixture trace)\n\
         \x20 --verbose         per-layer placement table\n\
         \x20 --json            one machine-readable line as well\n\
         \x20 --host-gb 18.0    system-RAM pool for `run`/`serve` (`--host` there is the bind address)\n\
         \x20 --allow-remote    permit a non-loopback bind (no TLS, model in RAM)\n\
         \x20 --ctx 32768       context window `run`/`serve` allocates state for\n\
         \x20 --ngen 4096       generation ceiling per request\n\
         \x20 --auto-tier       turn the device tier on without spelling out --gpu\n\
         \x20 --dense-i8        one byte per weight for the host dense projections; on by default, --no-dense-i8 restores the exact bf16/f16 dot\n\
         \x20 --host-gb GiB     system-RAM host pool, default 100 (the plan trims it to what fits)\n\n\
         default model dir: {DEFAULT_MODEL}",
        fennec::VERSION
    );
}

/// Attach an FQ4 expert cache if one is named or sitting next to the shards.
/// Shared by `open` and `load_engine` so `run`/`serve` get the same fast load as
/// `bench`/`decode` — the suite only ever spawns `serve`.
fn attach_cache(opts: &Opts, dir: &str, sh: &mut Shards) -> Result<(), String> {
    let asked = opts.str("fq4").or_else(|| std::env::var("FENNEC_FQ4").ok());
    let fq4 = asked.clone().unwrap_or_else(|| format!("{dir}/fennec.fq4"));
    if std::path::Path::new(&fq4).is_file() {
        sh.attach_fq4(&fq4)?;
        eprintln!("[FQ4] expert cache attached: {fq4}");
    } else if asked.is_some() {
        return Err(format!("FQ4 cache {fq4} does not exist; build it with `fennec quant --layers all`"));
    }
    Ok(())
}

fn open(opts: &Opts) -> Result<(Arc<TextConfig>, Arc<Shards>), String> {
    let dir = opts.dir();
    let mut cfg = TextConfig::from_file(&format!("{dir}/config.json"))?;
    let mut sh = Shards::open(&dir)?;
    // Nothing in config.json records the tensor prefix, so read it back off
    // the shard index before any name is built from it.
    fennec::model::bind_prefix(&sh, &mut cfg);
    attach_cache(opts, &dir, &mut sh)?;
    Ok((Arc::new(cfg), Arc::new(sh)))
}

/// The group width belongs to the checkpoint when its experts are already int4;
/// `--group` only picks a width for weights Fennec has to quantise itself.
fn group_for(opts: &Opts, sh: &Shards, cfg: &TextConfig) -> usize {
    fennec::model::packed_group(sh, &cfg.prefix, cfg.hidden_size, cfg.moe_intermediate_size)
        .unwrap_or_else(|| opts.usize_at("group", 32))
}

fn info(opts: &Opts) -> Result<(), String> {
    let (cfg, sh) = open(opts)?;
    let group = group_for(opts, &sh, &cfg);
    let mut by_dtype: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut expert_tensors = 0usize;
    let mut expert_bytes = 0usize;
    for name in sh.names() {
        let i = sh.info(name).ok_or_else(|| format!("missing tensor info for {name}"))?;
        let e = by_dtype.entry(format!("{:?}", i.dtype)).or_insert((0, 0));
        e.0 += 1;
        e.1 += i.bytes();
        if name.contains(".experts.") {
            expert_tensors += 1;
            expert_bytes += i.bytes();
        }
    }
    println!("model            {}", cfg.name);
    println!(
        "layers           {} ({} linear-attn, {} full-attn)",
        cfg.num_hidden_layers,
        cfg.n_linear_layers(),
        cfg.num_hidden_layers - cfg.n_linear_layers()
    );
    println!("hidden / vocab   {} / {}", cfg.hidden_size, cfg.vocab_size);
    if cfg.is_moe() {
        println!(
            "experts          {} routed, {} per token, moe_i {}, shared_i {}",
            cfg.num_experts, cfg.num_experts_per_tok, cfg.moe_intermediate_size, cfg.shared_expert_intermediate_size
        );
    } else {
        println!(
            "ffn              dense SiLU, intermediate {}, tied head {}",
            cfg.intermediate_size, cfg.tie_word_embeddings
        );
    }
    println!("tensors under    {}", cfg.prefix);
    println!(
        "attn             {} q heads x {}, {} kv heads x {}, rotary {}, output gate {}",
        cfg.num_attention_heads,
        cfg.head_dim,
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.rotary_dim(),
        cfg.attn_output_gate
    );
    if cfg.has_gdn() {
        println!(
            "linear attn      {} key heads x {}, {} value heads x {}, conv {}",
            cfg.linear_num_key_heads,
            cfg.linear_key_head_dim,
            cfg.linear_num_value_heads,
            cfg.linear_value_head_dim,
            cfg.linear_conv_kernel_dim
        );
    }
    println!(
        "params           {:.2} B total, {:.2} B active per token",
        cfg.total_params() as f64 / 1e9,
        cfg.active_params() as f64 / 1e9
    );
    println!("shards           {}", sh.n_shards());
    println!("tensors          {}", sh.names().len());
    for (dt, (n, b)) in &by_dtype {
        println!("  {dt:<5} {n:>5} tensors  {:>9.2} GiB", gib(*b));
    }
    let eb = fennec::sched::expert_bytes(&cfg, group);
    if cfg.is_moe() {
        println!(
            "checkpoint       {:.2} GiB, of which {:.2} GiB ({:.1}%) is expert weights in {expert_tensors} tensors",
            gib(sh.total_bytes as usize),
            gib(expert_bytes),
            100.0 * expert_bytes as f64 / sh.total_bytes.max(1) as f64
        );
        println!(
            "FQ4 expert       {:.1} MiB each -> {:.2} GiB per layer, {:.2} GiB for all {} (from {:.2} GiB bf16)",
            mib(eb),
            gib(eb * cfg.num_experts),
            gib(eb * cfg.num_experts * cfg.num_hidden_layers),
            cfg.num_hidden_layers,
            gib(expert_bytes / cfg.num_hidden_layers)
        );
    } else {
        // Nothing is routed here, so an expert share and an FQ4 sizing would
        // both read as zeros: every parameter the checkpoint holds is read by
        // every token.
        println!("checkpoint       {:.2} GiB", gib(sh.total_bytes as usize));
    }
    if opts.has("json") {
        println!(
            "{}",
            serde_json::json!({
                "cmd": "info", "model": cfg.name, "layers": cfg.num_hidden_layers,
                "linear_layers": cfg.n_linear_layers(), "tensors": sh.names().len(),
                "bytes": sh.total_bytes, "expert_bytes": expert_bytes,
                "fq4_expert_bytes": eb, "group": group,
            })
        );
    }
    Ok(())
}

fn find_usage(opts: &Opts, dir: &str) -> Option<String> {
    if let Some(p) = opts.str("usage") {
        return Some(p);
    }
    let cand = format!("{dir}/.coli_usage");
    Path::new(&cand).is_file().then_some(cand)
}

/// The RAM pool budget, resolved the same way for every command. `serve` spells
/// it `--host-gb` because `--host` there is the bind address. An explicit flag
/// wins over `*_VK_TIER_HOST_GB`, so `plan` prices the pool `serve` will use.
fn host_gb(opts: &Opts) -> f64 {
    if let Some(v) = opts.str("host-gb").and_then(|v| v.parse::<f64>().ok()) {
        return v;
    }
    if opts.kv.contains_key("host") {
        return opts.num("host", 18.0);
    }
    env_name(&["FENNEC_VK_TIER_HOST_GB", "COLI_VK_TIER_HOST_GB"])
        .and_then(|v| v.parse().ok())
        .unwrap_or(18.0)
}

/// `--gpu` turns the device tier into real VRAM; `--gpu N` picks the ranked
/// device `probe` printed. `--cpu` wins over both: it is the no-device plan.
/// One definition of "put the device tier to work", shared by every command.
/// `serve` used to accept `--auto-tier` and `*_VULKAN` while `plan`/`decode`
/// did not, so the plan a user read was not the plan `serve` ran.
fn gpu_wanted(opts: &Opts) -> bool {
    !opts.has("cpu")
        && (opts.has("auto-tier")
            || opts.has("gpu")
            || env_on(&["FENNEC_VULKAN", "COLI_VULKAN"])
            || env_on(&["FENNEC_CUDA", "COLI_CUDA"])
            || env_on(&["FENNEC_METAL", "COLI_METAL"]))
}

fn open_gpu(opts: &Opts) -> Result<Option<Arc<Mutex<Gpu>>>, String> {
    if !gpu_wanted(opts) {
        return Ok(None);
    }
    let index = opts.usize_at("gpu", 0);
    let requested = opts.str("backend");
    let env = |k: &str| std::env::var(k).ok();
    // `--backend cpu` is a no-device request even next to `--gpu`, the same way
    // `--cpu` is. Resolving with vendor 0 only reads the explicit choice and the
    // environment here; the vendor policy below is what picks a real API.
    let resolved = backend::resolve(requested.as_deref(), &env, 0, false)?;
    if resolved == Backend::Cpu {
        return Ok(None);
    }
    if resolved == Backend::Ort {
        // ORT is not an arena `Gpu`: it accelerates the dense projections through
        // ONNX Runtime sessions, opened by `open_ort`. The plan stays host-only
        // (no VRAM tier); `Model::enable_ort` moves the dense matmuls after load.
        #[cfg(feature = "ort")]
        return Ok(None);
        #[cfg(not(feature = "ort"))]
        return Err("backend 'ort' needs a build with --features ort".to_string());
    }
    // Rank through the same enumeration `probe` prints, so --gpu N names the
    // same device on both commands, and let the vendor pick the API: NVIDIA
    // goes to CUDA, AMD and Intel stay on Vulkan, Apple is Vulkan/MoltenVK.
    let devs = Vulkan::load().devices();
    let g = match devs.get(index) {
        Some(d) => {
            let b = backend::resolve(requested.as_deref(), &env, d.vendor_id, d.portability())?;
            Gpu::open_backend(index, b).map_err(|e| format!("--gpu {index} ({}): {e}", b.label()))?
        }
        // No Vulkan loader to ask for a vendor id. A machine in that state with
        // a usable GPU is a CUDA-only one, so try the NVIDIA driver first and
        // keep Vulkan as the second attempt rather than guessing silently.
        None => {
            let b = backend::resolve(requested.as_deref(), &env, 0x10DE, false)?;
            match Gpu::open_backend(index, b) {
                Ok(g) => g,
                Err(e) => Gpu::open_backend(index, Backend::Vulkan)
                    .map_err(|e2| format!("--gpu {index}: {} {e}; Vulkan {e2}", b.label()))?,
            }
        }
    };
    eprintln!("[fennec] compute device: {} on {}", g.name(), g.backend().label());
    Ok(Some(Arc::new(Mutex::new(g))))
}

/// The ONNX Runtime portability backend. When `--backend ort` is selected this
/// opens the signed runtime and picks the ranked device's provider (DirectML on
/// AMD/Intel GPU, OpenVINO on the Intel NPU, CoreML on Apple); the dense
/// projections move onto it through `Model::enable_ort` after the host-only plan
/// loads. `None` for every other backend, so the CUDA/Vulkan path is untouched.
#[cfg(feature = "ort")]
fn open_ort(opts: &Opts) -> Result<Option<Arc<fennec::ort::Ort>>, String> {
    let requested = opts.str("backend");
    let env = |k: &str| std::env::var(k).ok();
    if backend::resolve(requested.as_deref(), &env, 0, false)? != Backend::Ort {
        return Ok(None);
    }
    let index = opts.usize_at("gpu", 0);
    let devs = Vulkan::load().devices();
    let ort = fennec::ort::Ort::open(None, devs.get(index))?;
    Ok(Some(Arc::new(ort)))
}

/// The budgets, the histogram and the placement both `plan` and `decode` need.
/// Kept in one place so the two commands cannot disagree about what the plan was.
struct Setup {
    plan: Plan,
    bw: Bandwidth,
    usage: Usage,
}

/// The GPU budget when `--device` is not given: the whole card, read off the
/// device rather than assumed. A flat 8.0 GiB both over-committed an 8 GiB
/// laptop card and under-used anything larger.
///
/// Over-committing is not free even when the head is protected. WDDM puts the
/// allocation that exceeds the process budget in system memory, and whichever
/// buffer went up last loses; `load_plan` uploads the head first so an expert
/// buffer absorbs it, but every pick of a demoted expert still crosses PCIe.
/// Measured at 8.0 against 7.77 GiB on the same card: 25.8 ms/token on CUDA,
/// 11.1 on Vulkan.
fn default_device_gib(opts: &Opts) -> f64 {
    if !gpu_wanted(opts) {
        return 8.0;
    }
    let devs = Vulkan::load().devices();
    match devs.get(opts.usize_at("gpu", 0)) {
        Some(d) if d.usable_gib() > 0.0 => d.usable_gib(),
        // No loader to ask. Keeping the old flat budget is the honest fallback:
        // guessing a smaller one would shrink the tier on a card we cannot see.
        _ => 8.0,
    }
}

fn setup(opts: &Opts, cfg: &Arc<TextConfig>, group: usize) -> Result<Setup, String> {
    let gpu = gpu_wanted(opts);
    let device_gib = if opts.has("device") { opts.num("device", 8.0) } else { default_device_gib(opts) };
    let device = (device_gib * 2f64.powi(30)) as usize;
    let mut host = (host_gb(opts) * 2f64.powi(30)) as usize;
    // A host pool larger than the RAM the box can actually keep resident is not
    // a memory tier, it is a disk tier with extra page faults: every "resident"
    // pick faults a trimmed page back from the file. Cap the budget to available
    // physical memory less headroom for the OS, the dense trunk, the KV cache and
    // the device staging, so a busy box streams cold experts on purpose instead
    // of thrashing. `0` means the platform does not expose it; leave the budget.
    let avail = fennec::mem::available_bytes();
    if avail > 0 {
        let headroom = 3u64 << 30;
        host = host.min(avail.saturating_sub(headroom) as usize);
    }
    let mut reserve = (opts.num("reserve", 0.5) * 2f64.powi(30)) as usize;
    if gpu {
        // the lm_head is parked in VRAM whole, and the upload path needs its
        // staging block plus the activation arena on top of it
        let head = cfg.vocab_size * cfg.hidden_size * 2;
        reserve = reserve.max(head + (128 << 20));
    }
    let bw = if opts.has("cpu") {
        Bandwidth::cpu_only()
    } else {
        Bandwidth::laptop_4060()
    };
    let usage = if !cfg.is_moe() {
        // A dense checkpoint routes nothing, so there is no histogram to look
        // for and nothing to warn about when none is found: the plan's expert
        // tiers simply stay empty.
        Usage::uniform(cfg.num_hidden_layers, 0)
    } else {
        match find_usage(opts, &opts.dir()) {
            Some(p) => {
                eprintln!("[fennec] router histogram: {p}");
                let u = Usage::load(&p)?;
                // A histogram for a different model would silently drop layers or
                // index past the expert table, so refuse it rather than guess.
                if u.layers != cfg.num_hidden_layers || u.experts != cfg.num_experts {
                    return Err(format!(
                        "router histogram is {}x{} but the model is {}x{}",
                        u.layers, u.experts, cfg.num_hidden_layers, cfg.num_experts
                    ));
                }
                u
            }
            None => {
                eprintln!("[fennec] no histogram; assuming a flat router");
                Usage::uniform(cfg.num_hidden_layers, cfg.num_experts)
            }
        }
    };
    let plan = Plan::build(cfg, &usage, device, host, reserve, &bw, group, gpu);
    Ok(Setup { plan, bw, usage })
}

fn plan(opts: &Opts) -> Result<(), String> {
    let (cfg, sh) = open(opts)?;
    let group = group_for(opts, &sh, &cfg);
    let s = setup(opts, &cfg, group)?;
    let (plan, bw, usage) = (s.plan, s.bw, s.usage);
    let host = (host_gb(opts) * 2f64.powi(30)) as usize;
    println!("{}", plan.report(&cfg));
    println!(
        "  dense {:.1} MiB/layer | FQ4 expert {:.1} MiB | lm_head {:.2} ms | tiers device {:.1} / PCIe {:.1} / disk {:.1} GB/s",
        mib(plan.dense_bytes_layer),
        mib(plan.expert_bytes),
        plan.lm_head_ms,
        bw.device_gbs,
        bw.pcie_gbs,
        bw.disk_gbs
    );
    let ideal = Plan::build(&cfg, &usage, usize::MAX / 8, host, 0, &bw, group, gpu_wanted(opts));
    println!(
        "  upper bound, every expert resident: {:.2} tok/s | histogram: {} selections over {} layers x {}",
        ideal.tok_s(),
        usage.counts.iter().map(|r| r.iter().sum::<u64>()).sum::<u64>(),
        usage.layers,
        usage.experts
    );
    if opts.has("curve") {
        println!("# n coverage");
        for n in [0, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256] {
            println!("{} {:.6}", n, usage.coverage(n.min(cfg.num_experts)));
        }
    }
    if opts.has("verbose") {
        println!("# layer device host disk dense_bytes dense_on_device");
        for lp in &plan.layers {
            println!(
                "{} {} {} {} {} {}",
                lp.layer,
                lp.resident.len(),
                lp.pooled.len(),
                lp.streamed.len(),
                lp.dense_bytes,
                lp.dense_on_device
            );
        }
    }
    if opts.has("json") {
        println!(
            "{}",
            serde_json::json!({
                "cmd": "plan", "device_gib": gib(plan.device_bytes_budget), "host_gib": host_gb(opts),
                "cpu": opts.has("cpu"), "group": group, "ms_per_token": plan.ms_per_token,
                "tok_s": plan.tok_s(), "device_hit": plan.device_hit_rate,
                "host_hit": plan.host_hit_rate, "disk_hit": plan.disk_rate,
                "resident_slots": plan.layers.iter().map(|l| l.resident.len()).sum::<usize>(),
                "pooled_slots": plan.layers.iter().map(|l| l.pooled.len()).sum::<usize>(),
            })
        );
    }
    Ok(())
}

fn probe() {
    let dx = fennec::dx::probe();
    for a in &dx.adapters {
        println!(
            "[DXGI] {} | {} | vendor {:#06x} device {:#06x} | {:.2} GiB dedicated, {:.2} GiB shared{}",
            a.name,
            a.vendor,
            a.vendor_id,
            a.device_id,
            a.dedicated_gib,
            a.shared_gib,
            if a.software { " | software renderer" } else { "" },
        );
    }
    for n in &dx.npus {
        let a = &n.adapter;
        println!(
            "[DXCore] {} | {} | vendor {:#06x} | graphics={} generic_ml={} core_compute={} | \
             {:.2} GiB dedicated, {:.2} GiB shared | {}",
            a.name,
            a.vendor,
            a.vendor_id,
            a.graphics,
            a.generic_ml,
            a.core_compute,
            a.dedicated_gib,
            a.shared_gib,
            n.verdict()
        );
    }
    println!(
        "[DX] NVIDIA {}, AMD {}, NPU {} ({} usable), DirectML {} | Vulkan cannot enumerate an NPU: \
         the compute accelerator is a DXCore adapter, not a graphics adapter",
        dx.nvidia().len(),
        dx.amd().len(),
        dx.npus.len(),
        dx.npus.iter().filter(|n| n.usable()).count(),
        if dx.directml_loadable { "loads" } else { "not found" },
    );
    for note in &dx.notes {
        println!("[DX] note: {note}");
    }
    let mut vk = Vulkan::load();
    let devs = vk.devices();
    if devs.is_empty() {
        println!("[VK] no devices: {}", vk.error);
        return;
    }
    for d in &devs {
        println!(
            "[VK] {} | {} | {} | Vulkan {} | driver {} | {:.2} GiB device-local ({:.2} GiB of it dedicated), {:.2} GiB host-visible | compute rank {}",
            d.name,
            fennec::vk::class_name(d.class),
            d.vendor,
            d.api,
            d.driver,
            d.device_local_gib,
            d.dedicated_gib,
            d.host_visible_gib,
            d.rank,
        );
        for (k, (size, flags, host)) in d.heaps.iter().enumerate() {
            println!(
                "[VK]   heap {k}: {:>8.2} GiB flags {flags:#06x}{}",
                gib(*size as usize),
                if *host { " host-visible" } else { "" }
            );
        }
    }
    println!(
        "[VK] compute goes to the top-ranked device: {} (rank {}, {:.2} GiB usable, backend {}).",
        devs[0].name,
        devs[0].rank,
        devs[0].usable_gib(),
        devs[0].preferred_backend().label()
    );
    if let Ok(cfg) = TextConfig::from_file(&format!("{DEFAULT_MODEL}/config.json")) {
        // The group belongs to the checkpoint, not to this probe: the shipped
        // container is group 64, and assuming 32 overstates an expert by 11.1%.
        let group = Shards::open(DEFAULT_MODEL)
            .ok()
            .and_then(|sh| {
                fennec::model::packed_group(&sh, &cfg.prefix, cfg.hidden_size, cfg.moe_intermediate_size)
            })
            .unwrap_or(32);
        let eb = fennec::sched::expert_bytes(&cfg, group);
        for d in &devs {
            // Apple's unified memory has no dedicated heap, so sizing room from
            // dedicated_gib alone would report zero experts fit on a Mac.
            let room = (d.usable_gib().max(0.0) * 0.85 * 2f64.powi(30)) as usize;
            let dense = cfg.num_hidden_layers * fennec::sched::dense_bytes_layer(&cfg, false, 2);
            let spare = room.saturating_sub(dense.min(room));
            println!(
                "[VK] {}: {:.2} GiB VRAM, {:.2} GiB usable at 85% - {:.2} GiB of bf16 dense layers \
                 leaves {:.2} GiB for {:.0} FQ4 experts at group {group}, which is {:.1} of {} layers fully resident",
                d.name,
                d.usable_gib(),
                gib(room),
                gib(dense),
                gib(spare),
                spare as f64 / eb as f64,
                spare as f64 / (eb * cfg.num_experts) as f64,
                cfg.num_hidden_layers,
            );
        }
    }
    println!(
        "[VK] fox/llama.cpp caveat: its Vulkan gated_delta_net only runs on the GPU when every \
         input and the output are F32 and S_v is one of 16, 32, 64, 128 - otherwise the delta net \
         silently stays on the CPU. Fennec has no such branch: the delta net is one kernel."
    );
}

/// The desktop's Doctor panel. Colibri's `doctor` emitted this shape and the UI
/// still reads `checks[].{id,status,detail}`, so Fennec speaks it too: `ok`,
/// `warn`, `skip`, anything else renders as a failure.
fn doctor(opts: &Opts) -> Result<(), String> {
    let dir = opts.dir();
    let mut checks: Vec<serde_json::Value> = Vec::new();
    fn add(checks: &mut Vec<serde_json::Value>, id: &str, status: &str, detail: String) {
        checks.push(serde_json::json!({"id": id, "status": status, "detail": detail}));
    }

    let cfg = match TextConfig::from_file(&format!("{dir}/config.json")) {
        Ok(c) => {
            add(&mut checks, "config", "ok", if c.is_moe() {
                format!("{} layers, {} experts, top-{}, hidden {}",
                    c.num_hidden_layers, c.num_experts, c.num_experts_per_tok, c.hidden_size)
            } else {
                format!("{} layers, dense FFN i {}, hidden {}",
                    c.num_hidden_layers, c.intermediate_size, c.hidden_size)
            });
            Some(Arc::new(c))
        }
        Err(e) => {
            add(&mut checks, "config", "bad", format!("{dir}/config.json: {e}"));
            None
        }
    };

    let mut shards = match Shards::open(&dir) {
        Ok(s) => {
            add(&mut checks, "weights", "ok",
                format!("{} shards, {:.2} GiB of tensors", s.n_shards(), gib(s.total_bytes as usize)));
            Some(s)
        }
        Err(e) => {
            add(&mut checks, "weights", "bad", e);
            None
        }
    };

    match fennec::tokenizer::Tokenizer::from_dir(&dir) {
        Ok(t) => add(&mut checks, "tokenizer", "ok", format!("{} tokens, {} stop ids", t.len(), t.eos.len())),
        Err(e) => add(&mut checks, "tokenizer", "bad", e),
    }

    let packed = match (cfg.as_ref(), shards.as_ref()) {
        (Some(c), Some(sh)) => {
            let prefix = fennec::model::detect_prefix(sh);
            fennec::model::packed_group(sh, &prefix, c.hidden_size, c.moe_intermediate_size)
        }
        _ => None,
    };
    // An expert cache is a thing only a routed checkpoint can have. On a dense
    // one every branch below would either stay silent or print advice about
    // re-quantising experts that are not in the model.
    if cfg.as_ref().is_some_and(|c| c.is_moe()) {
        if let Some(sh) = shards.as_mut() {
            match attach_cache(opts, &dir, sh) {
                Ok(()) if sh.fq4().is_some() => add(
                    &mut checks,
                    "fq4-cache",
                    "ok",
                    "expert cache attached: cold load copies int4".into(),
                ),
                Ok(()) if packed.is_some() => add(&mut checks, "fq4-cache", "ok",
                    format!("checkpoint is already int4 at group {}: cold load copies it, no cache needed", packed.unwrap())),
                Ok(()) => add(&mut checks, "fq4-cache", "warn",
                    "no FQ4 cache: every start re-quantises the experts from bf16. \
                     Build one with `fennec quant --layers all --out <path>` and pass --fq4.".into()),
                Err(e) => add(&mut checks, "fq4-cache", "bad", e),
            }
        }
    }

    let mut vk = Vulkan::load();
    let devs = vk.devices();
    if devs.is_empty() {
        add(&mut checks, "gpu", "warn", format!("no Vulkan device ({}): CPU and RAM tiers only", vk.error));
    } else {
        add(&mut checks, "gpu", "ok",
            format!("{} on {} ({:.2} GiB dedicated), Vulkan {}",
                devs[0].name, devs[0].vendor, devs[0].dedicated_gib, devs[0].api));
    }

    let dx = fennec::dx::probe();
    let nv = dx.nvidia().len();
    let amd = dx.amd().len();
    add(&mut checks, "vendors", if nv + amd > 0 { "ok" } else { "warn" },
        format!("{nv} NVIDIA, {amd} AMD adapter(s); both take the same vendor-neutral Vulkan path"));
    match dx.npus.first() {
        None => add(&mut checks, "npu", "skip", "no DXCore compute accelerator on this machine".into()),
        Some(n) if n.usable() => add(&mut checks, "npu", "ok", format!("{}: {}", n.adapter.name, n.verdict())),
        Some(n) => add(&mut checks, "npu", "warn", format!("{}: {}", n.adapter.name, n.verdict())),
    }

    if let Some(c) = cfg.as_ref() {
        // A planner failure is a check result, not a reason to swallow the report.
        match setup(opts, c, packed.unwrap_or_else(|| opts.usize_at("group", 32))) {
            Ok(s) => {
                let dev: usize = s.plan.layers.iter().map(|l| l.resident.len()).sum();
                let host: usize = s.plan.layers.iter().map(|l| l.pooled.len()).sum();
                add(&mut checks, "plan", "ok",
                    format!("{dev} experts on device, {host} in host RAM, {:.2} GiB device budget, \
                             {:.1}% predicted device hit",
                        gib(s.plan.device_bytes_budget), s.plan.device_hit_rate * 100.0));
            }
            Err(e) => add(&mut checks, "plan", "bad", e),
        }
    }

    let ok = !checks.iter().any(|c| c["status"] == "bad");
    println!("{}", serde_json::json!({
        "cmd": "doctor", "ok": ok, "engine": "fennec", "version": fennec::VERSION,
        "policy": opts.str("policy").unwrap_or_else(|| "quality".into()),
        "checks": checks,
    }));
    Ok(())
}

fn layers_of(opts: &Opts, n: usize) -> Result<Vec<usize>, String> {
    match opts.str("layers") {
        None => Ok(vec![0]),
        Some(s) if s == "all" => Ok((0..n).collect()),
        Some(s) => s
            .split(',')
            .map(|p| {
                let v = p
                    .trim()
                    .parse::<usize>()
                    .map_err(|_| format!("bad layer `{}`", p.trim()))?;
                if v >= n {
                    Err(format!("layer {v} out of range ({n})"))
                } else {
                    Ok(v)
                }
            })
            .collect(),
    }
}

fn quant(opts: &Opts, threads: usize) -> Result<(), String> {
    let (cfg, sh) = open(opts)?;
    let group = group_for(opts, &sh, &cfg);
    let out = opts.str("out").unwrap_or_else(|| "fennec.fq4".to_string());
    if !cfg.is_moe() {
        return Err(format!("{} is a dense checkpoint: it has no routed experts to quantise.", cfg.name));
    }
    let e = cfg.num_experts;
    let h = cfg.hidden_size;
    let i = cfg.moe_intermediate_size;
    let mut w = SidecarWriter::new(&out)?;
    let mut total = 0usize;
    for l in layers_of(opts, cfg.num_hidden_layers)? {
        let p = format!("{}.layers.{l}.mlp", cfg.prefix);
        let gu_src = sh.tensor_f32(&format!("{p}.experts.gate_up_proj"))?;
        let dn_src = sh.tensor_f32(&format!("{p}.experts.down_proj"))?;
        let t0 = Instant::now();
        let gu = Q4::quantize(&gu_src, e * 2 * i, h, group, threads);
        let dn = Q4::quantize(&dn_src, e * h, i, group, threads);
        let s = t0.elapsed().as_secs_f64();
        let src_bytes = gu_src.len() * 4 + dn_src.len() * 4;
        println!(
            "layer {l}: {:.2} GiB f32 -> {:.2} GiB FQ4 in {s:.1}s ({:.2} GiB/s, {threads} threads) | rel err gate_up {:.5} down {:.5}",
            gib(src_bytes),
            gib(gu.bytes_total() + dn.bytes_total()),
            gib(src_bytes) / s,
            gu.rel_error(&gu_src),
            dn.rel_error(&dn_src)
        );
        total += gu.bytes_total() + dn.bytes_total();
        w.push(&format!("blk.{l}.ffn.gate_up_exps"), &gu)?;
        w.push(&format!("blk.{l}.ffn.down_exps"), &dn)?;
    }
    let bytes = w.finish(&out, (sh.total_bytes, sh.names().len()))?;
    println!(
        "wrote {out}: {:.2} GiB on disk, {:.2} GiB of matrices | attach it with --fq4 {out}",
        gib(bytes),
        gib(total)
    );
    Ok(())
}

/// One FNN1 file holding several matrices, written without ever holding the
/// whole thing in RAM: payloads stream to a temp file while only the offset
/// table is kept, then the header is prepended. A 40-layer expert cache is
/// ~20 GiB, so buffering it would not fit alongside the model.
struct SidecarWriter {
    tmp: std::fs::File,
    tmp_path: std::path::PathBuf,
    table: serde_json::Map<String, serde_json::Value>,
    len: usize,
}

impl SidecarWriter {
    fn new(path: &str) -> Result<SidecarWriter, String> {
        let tmp_path = std::path::PathBuf::from(format!("{path}.payload"));
        let tmp = std::fs::File::create(&tmp_path).map_err(|e| format!("{}: {e}", tmp_path.display()))?;
        Ok(SidecarWriter { tmp, tmp_path, table: serde_json::Map::new(), len: 0 })
    }

    fn push(&mut self, name: &str, q: &Q4) -> Result<(), String> {
        use std::io::Write;
        let one = q.to_bytes(name);
        let hl = u64::from_le_bytes(one[8..16].try_into().map_err(|_| "short header")?) as usize;
        let head: serde_json::Value =
            serde_json::from_slice(&one[16..16 + hl]).map_err(|e| format!("header: {e}"))?;
        let mut item = head["items"][name]
            .as_object()
            .ok_or_else(|| format!("{name}: missing item"))?
            .clone();
        let at = |k: &str| -> Result<(usize, usize), String> {
            let a = item[k].as_array().ok_or_else(|| format!("{name}: {k} is not a pair"))?;
            if a.len() < 2 {
                return Err(format!("{name}: {k} array has {} elements, expected 2", a.len()));
            }
            let v0 = a[0].as_u64().ok_or_else(|| format!("{name}: {k}[0] is not a u64"))?;
            let v1 = a[1].as_u64().ok_or_else(|| format!("{name}: {k}[1] is not a u64"))?;
            Ok((v0 as usize, v1 as usize))
        };
        let off = self.len;
        let (d0, d1) = at("data")?;
        let (s0, s1) = at("scale")?;
        item["data"] = serde_json::json!([off + d0, off + d1]);
        item["scale"] = serde_json::json!([off + s0, off + s1]);
        let payload = &one[16 + hl..];
        self.tmp.write_all(payload).map_err(|e| format!("{}: {e}", self.tmp_path.display()))?;
        self.len += payload.len();
        self.table.insert(name.to_string(), serde_json::Value::Object(item));
        Ok(())
    }

    fn finish(mut self, path: &str, source: (u64, usize)) -> Result<usize, String> {
        use std::io::{Read, Write};
        self.tmp.flush().map_err(|e| e.to_string())?;
        drop(self.tmp);
        let json = serde_json::to_vec(&serde_json::json!({
            "items": self.table,
            "source": { "total_bytes": source.0, "tensors": source.1 },
        }))
        .map_err(|e| e.to_string())?;
        let out = std::path::Path::new(path);
        let mut file = std::fs::File::create(out).map_err(|e| format!("{path}: {e}"))?;
        file.write_all(fennec::quant::MAGIC).map_err(|e| e.to_string())?;
        file.write_all(&(json.len() as u64).to_le_bytes()).map_err(|e| e.to_string())?;
        file.write_all(&json).map_err(|e| e.to_string())?;
        let mut tmp = std::fs::File::open(&self.tmp_path).map_err(|e| e.to_string())?;
        let mut buf = vec![0u8; 8 << 20];
        loop {
            let n = tmp.read(&mut buf).map_err(|e| e.to_string())?;
            if n == 0 {
                break;
            }
            file.write_all(&buf[..n]).map_err(|e| e.to_string())?;
        }
        drop(tmp);
        let n = 16 + json.len() + self.len;
        std::fs::remove_file(&self.tmp_path).map_err(|e| format!("{}: {e}", self.tmp_path.display()))?;
        Ok(n)
    }
}

fn bench(opts: &Opts, threads: usize) -> Result<(), String> {
    let (cfg, sh) = open(opts)?;
    let group = group_for(opts, &sh, &cfg);
    let l = layers_of(opts, cfg.num_hidden_layers)?[0];
    if !cfg.is_moe() {
        return Err(format!("bench times the routed expert block; {} is dense - use `decode` for the whole model.", cfg.name));
    }
    let reps = opts.usize_at("reps", 16).max(1);
    let e = cfg.num_experts;
    let h = cfg.hidden_size;
    let i = cfg.moe_intermediate_size;
    let k = cfg.num_experts_per_tok;

    let t0 = Instant::now();
    let mut moe = MoeLayer::load(&sh, cfg.clone(), l, group, threads)?;
    println!(
        "layer {l}: resident in {:.1}s | FQ4 gate_up {:.2} GiB + down {:.2} GiB = {:.2} GiB for all {e} experts (bf16 would be {:.2} GiB)",
        t0.elapsed().as_secs_f64(),
        gib(moe.host_gu_bytes()),
        gib(moe.host_dn_bytes()),
        gib(moe.host_bytes()),
        gib((e * 3 * i * h) * 2)
    );

    let mut bits = 0x2545f491_4f6cdd1du64;
    let mut frnd = move || {
        bits ^= bits << 13;
        bits ^= bits >> 7;
        bits ^= bits << 17;
        (bits as f64 / u64::MAX as f64) as f32 - 0.5
    };
    let x: Vec<f32> = (0..h).map(|_| frnd()).collect();

    let per_expert_bytes = moe.host_bytes() / e;
    let routed_bytes = k * per_expert_bytes;
    let dense_bytes = (3 * cfg.shared_expert_intermediate_size * h + e * h) * 4;
    let macs = (k * 3 * i * h + 3 * cfg.shared_expert_intermediate_size * h + e * h) as f64;

    let mut seen: Vec<usize> = Vec::new();
    for t in [1, 2, 4, 8, 16, threads] {
        let t = t.clamp(1, cores_max());
        if !seen.contains(&t) {
            seen.push(t);
        }
    }
    println!("# threads ms tok_s_bound mtops gib_s routed_gib_s");
    let mut rows = Vec::new();
    for &t in &seen {
        moe.threads = t;
        let (probe_out, _) = moe.forward(&x);
        let checksum = probe_out.iter().sum::<f32>();
        let t1 = Instant::now();
        let mut acc = 0.0f32;
        for _ in 0..reps {
            let (o, _r) = moe.forward(&x);
            acc += o[0].max(o[1]);
        }
        let ms = t1.elapsed().as_secs_f64() * 1000.0 / reps as f64;
        let total_bytes = routed_bytes + dense_bytes;
        let gib_s = gib(total_bytes) / (ms / 1000.0);
        let routed_gib_s = gib(routed_bytes) / (ms / 1000.0);
        // a whole-token ceiling: the MoE block of every layer, nothing else
        let bound = 1000.0 / (ms * cfg.num_hidden_layers as f64);
        println!(
            "{t} {:.3} {:.2} {:.1} {:.2} {:.2}",
            ms,
            bound,
            macs / (ms * 1e3),
            gib_s,
            routed_gib_s
        );
        rows.push(serde_json::json!({"threads": t, "ms": ms, "tok_s_bound": bound,
            "mtops": macs / (ms * 1e3), "gib_s": gib_s, "routed_gib_s": routed_gib_s,
            "checksum": checksum, "acc": acc}));
    }
    if opts.has("json") {
        println!("{}", serde_json::json!({"cmd": "bench", "layer": l, "group": group, "rows": rows}));
    }
    Ok(())
}

/// The window of the graph this box can hold, placed by the whole-model plan.
/// The point is the last line: a token's measured milliseconds against the
/// milliseconds `plan` promised for the same placement. Predictions that never
/// meet a stopwatch are not a scheduler, they are a wish.
fn decode(opts: &Opts, threads: usize) -> Result<(), String> {
    let (cfg, sh) = open(opts)?;
    let group = group_for(opts, &sh, &cfg);
    let s = setup(opts, &cfg, group)?;
    let first = opts.usize_at("first", 0);
    if first >= cfg.num_hidden_layers {
        return Err(format!("--first {first} is past the last layer"));
    }
    let count = opts
        .usize_at("window", 4)
        .min(cfg.num_hidden_layers - first)
        .max(1);
    let slots = opts.usize_at("slots", 16);
    let n_new = opts.usize_at("tokens", 32).max(1);
    let mut toks: Vec<usize> = match opts.str("prompt") {
        None => DEFAULT_PROMPT.to_vec(),
        Some(p) => p
            .split(',')
            .map(|v| {
                v.trim()
                    .parse::<usize>()
                    .map_err(|_| format!("bad token id `{}`", v.trim()))
            })
            .collect::<Result<_, _>>()?,
    };
    for &t in &toks {
        if t >= cfg.vocab_size {
            return Err(format!("token id {t} is past the vocab ({})", cfg.vocab_size));
        }
    }

    let mut sub = s.plan.clone();
    sub.layers = sub.layers[first..first + count].to_vec();
    let resident: usize = sub.layers.iter().map(|l| l.resident.len()).sum();
    let pooled: usize = sub.layers.iter().map(|l| l.pooled.len()).sum();
    let cold: usize = sub.layers.iter().map(|l| l.streamed.len()).sum();

    let t0 = Instant::now();
    let gpu = open_gpu(opts)?;
    let m = Model::load_plan(&sh, cfg.clone(), &sub, group, threads, toks.len() + n_new + 1, slots, gpu.clone())?;
    #[cfg(feature = "ort")]
    let m = {
        let mut m = m;
        if let Some(ort) = open_ort(opts)? {
            m.enable_ort(&sh, &ort)?;
        }
        m
    };
    let fq4: usize = m
        .layers
        .iter()
        .map(|l| l.moe.as_ref().map_or(0, |m| m.host_bytes()))
        .sum();
    let on_dev: usize = m
        .layers
        .iter()
        .map(|l| l.moe.as_ref().map_or(0, |x| x.dev.as_ref().map_or(0, |d| d.bufs.len())))
        .sum();
    if cfg.is_moe() {
        println!(
            "[fennec] layers {first}..{} of {}: {on_dev} experts in VRAM, {pooled} in RAM, \
             {cold} streamable, {} in the router's top {} \
             loaded in {:.1} s ({:.2} GiB FQ4 host)",
            first + count,
            cfg.num_hidden_layers,
            cfg.num_experts_per_tok,
            cfg.num_experts,
            t0.elapsed().as_secs_f64(),
            gib(fq4),
        );
    } else {
        println!(
            "[fennec] dense layers {first}..{} of {} (FFN i {}), loaded in {:.1} s",
            first + count,
            cfg.num_hidden_layers,
            cfg.intermediate_size,
            t0.elapsed().as_secs_f64()
        );
    }

    let h = cfg.hidden_size;
    let eps = cfg.rms_norm_eps as f32;
    if let Some(g) = &gpu {
        g.lock().unwrap().reset_counters();
    }
    if std::env::var("FENNEC_PROFILE").is_ok_and(|v| !v.is_empty() && v != "0") {
        fennec::prof::enable();
    }
    fennec::prof::reset();
    let mut st = m.states(toks.len() + n_new + 1);
    let mut lg = vec![0f32; cfg.vocab_size];
    let mut hn = vec![0f32; h];
    let (mut stack_ms, mut head_ms) = (Vec::new(), Vec::new());
    let prompt_len = toks.len();
    let total = toks.len() + n_new;
    for t in 0..total {
        let mut x = m.embed_row(&sh, toks[t])?;
        let a = Instant::now();
        m.forward(&mut x, &mut st, t);
        kernel::rmsnorm(&x, &m.norm, eps, &mut hn);
        let stack = a.elapsed();
        let b = Instant::now();
        m.logits(&sh, &hn, &mut lg)?;
        stack_ms.push(stack.as_secs_f64() * 1e3);
        head_ms.push(b.elapsed().as_secs_f64() * 1e3);
        let (best, _) = lg
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |c, (i, &v)| if v > c.1 { (i, v) } else { c });
        // Only the continuation is fed back. Pushing during the prompt phase
        // would make the first generated token the prediction from position 0
        // instead of from the whole prompt, so the timed tail would not be the
        // greedy sequence `serve` produces.
        if t + 1 >= prompt_len {
            toks.push(best);
        }
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    // the first token is cold pages everywhere, so the rate quotes the tail
    let tail = |v: &[f64]| mean(&v[1.min(v.len() - 1)..]);
    let (stack, head) = (tail(&stack_ms), tail(&head_ms));
    let per_layer = stack / count as f64;
    let full = per_layer * cfg.num_hidden_layers as f64 + head;
    println!(
        "  decode {} tokens greedy: stack {stack:.1} ms + lm_head {head:.1} ms = \
         {:.1} ms/token over {count} layers",
        total,
        stack + head,
    );
    println!(
        "  projected to all {} layers: {full:.1} ms/token = {:.2} tok/s | \
         plan said {:.2} ms/token = {:.2} tok/s ({:+.1}%)",
        cfg.num_hidden_layers,
        1000.0 / full,
        s.plan.ms_per_token,
        s.plan.tok_s(),
        100.0 * (full / s.plan.ms_per_token - 1.0),
    );

    // End-to-end latency. The prompt is walked one position at a time, so the
    // step at index prompt_len-1 is the one that emits the first output token:
    // everything up to and including it is TTFT. TPOT is the steady-state cost
    // of the tokens after it.
    let step_ms: Vec<f64> = stack_ms.iter().zip(&head_ms).map(|(a, b)| a + b).collect();
    let wall_ms: f64 = step_ms.iter().sum();
    let ttft_ms: f64 = step_ms[..prompt_len].iter().sum();
    let after = &step_ms[prompt_len..];
    let tpot_ms = if after.is_empty() { 0.0 } else { mean(after) };
    let total_latency_ms = ttft_ms + tpot_ms * n_new as f64;
    println!(
        "  latency: TTFT {ttft_ms:.1} ms over {prompt_len} prompt tokens | TPOT {tpot_ms:.2} ms/token \
         | total {total_latency_ms:.1} ms = TTFT + TPOT x {n_new} output tokens | measured wall {wall_ms:.1} ms"
    );

    let (mut hits, mut misses, mut ring_bytes, mut read_ms, mut quant_ms) = (0u64, 0u64, 0usize, 0f64, 0f64);
    for l in &m.layers {
        if let Some(pool) = l.moe.as_ref().and_then(|m| m.stream.as_ref()) {
            let st = pool.lock().unwrap();
            hits += st.hits;
            misses += st.misses;
            ring_bytes += st.bytes();
            read_ms += st.read_ms;
            quant_ms += st.quant_ms;
        }
    }
    let picks = (hits + misses) as f64;
    println!(
        "  streamed picks {hits} of {picks:.0} ({:.1}%) | {:.2} GiB in {} rings, \
         {:.0} ms shard reads {:.0} ms quantise",
        100.0 * hits as f64 / picks.max(1.0),
        gib(ring_bytes),
        m.layers.iter().filter(|l| l.moe.as_ref().and_then(|m| m.stream.as_ref()).is_some()).count(),
        read_ms,
        quant_ms,
    );
    let (dev_hits, host_hits, disk_hits, overlap_ns) = m.layers.iter().fold(
        (0u64, 0u64, 0u64, 0u64),
        |(a, b, c, d), l| {
            (
                a + l.moe.as_ref().map_or(0, |m| m.dev_hits.load(std::sync::atomic::Ordering::Relaxed)),
                b + l.moe.as_ref().map_or(0, |m| m.host_hits.load(std::sync::atomic::Ordering::Relaxed)),
                c + l.moe.as_ref().map_or(0, |m| m.disk_hits.load(std::sync::atomic::Ordering::Relaxed)),
                d + l.moe.as_ref().map_or(0, |m| m.overlap_ns.load(std::sync::atomic::Ordering::Relaxed)),
            )
        },
    );
    let routed = (dev_hits + host_hits + disk_hits).max(1);
    if let Some(g) = &gpu {
        let g = g.lock().unwrap();
        println!(
            "  routed picks: {dev_hits} device ({:.1}%), {host_hits} host ({:.1}%), {disk_hits} disk ({:.1}%) | \
             {} dispatches in {} submits | CPU worked {:.1} ms while the device was in flight",
            100.0 * dev_hits as f64 / routed as f64,
            100.0 * host_hits as f64 / routed as f64,
            100.0 * disk_hits as f64 / routed as f64,
            g.dispatches(),
            g.submitted(),
            overlap_ns as f64 / 1e6,
        );
        let steps = step_ms.len().max(1) as f64;
        println!(
            "  device sync: {} waits, {:.1} ms total ({:.3} ms/wait) | {:.1} ms/token of the \
             {:.1} ms/token stack is fence time",
            g.syncs(),
            g.sync_ms(),
            g.sync_ms() / g.syncs().max(1) as f64,
            g.sync_ms() / steps,
            stack_ms.iter().sum::<f64>() / steps,
        );
    }
    if fennec::prof::enabled() {
        let steps = step_ms.len().max(1) as f64;
        let ms = fennec::prof::millis();
        let sum: f64 = ms.iter().sum();
        println!("  stage breakdown (ms/token, {} steps):", steps as usize);
        for (name, v) in fennec::prof::STAGES.iter().zip(&ms) {
            println!("    {name:<20} {:7.2}  {:5.1}%", v / steps, 100.0 * v / sum.max(1e-9));
        }
        // Every stage timer wraps its parallel region from the outside, so these
        // are wall time and add up to the stack. The remainder is stack time no
        // stage claims, which is the number to chase rather than a rounding error.
        let stack_mean = stack_ms.iter().sum::<f64>() / steps;
        println!(
            "    {:<20} {:7.2}  (wall time; stack {:.2}, unaccounted {:.2})",
            "accounted", sum / steps, stack_mean, stack_mean - sum / steps
        );
    }
    if opts.has("json") {
        println!(
            "{}",
            serde_json::json!({
                "cmd": "decode", "first": first, "window": count, "slots": slots,
                "group": group, "threads": threads, "tokens": total,
                "stack_ms": stack, "head_ms": head, "ms_per_token": stack + head,
                "projected_ms_full": full, "projected_tok_s_full": 1000.0 / full,
                "planned_ms_per_token": s.plan.ms_per_token,
                "resident": resident + pooled, "stream_hits": hits, "stream_misses": misses,
                "fq4_gib": gib(fq4), "ring_gib": gib(ring_bytes),
                "gpu": gpu.is_some(), "device_experts": on_dev,
                "dev_hits": dev_hits, "host_hits": host_hits, "disk_hits": disk_hits,
                "cpu_overlap_ms": overlap_ns as f64 / 1e6,
                "prompt_tokens": prompt_len, "output_tokens": n_new,
                "ttft_ms": ttft_ms, "tpot_ms": tpot_ms,
                "total_latency_ms": total_latency_ms, "wall_ms": wall_ms,
                "device_syncs": gpu.as_ref().map_or(0, |g| g.lock().unwrap().syncs()),
                "device_sync_ms": gpu.as_ref().map_or(0.0, |g| g.lock().unwrap().sync_ms()),
            })
        );
    }
    Ok(())
}

fn cores_max() -> usize {
    kernel::default_threads()
}

/// Everything `serve` and `run` need: the whole stack loaded once, the
/// tokenizer, and the identity the HTTP surface advertises.
struct Engine {
    model: Arc<Mutex<Model>>,
    sh: Arc<Shards>,
    cfg: Arc<TextConfig>,
    tok: fennec::tokenizer::Tokenizer,
    id: String,
    key: String,
    ctx: usize,
    ngen: usize,
}

/// The first of these environment names that is set. Every Fennec knob is
/// spelled `FENNEC_*`; the `COLI_*` twin stays readable so launchers written
/// against Colibri keep working until they move.
fn env_name(names: &[&str]) -> Option<String> {
    names.iter().find_map(|name| std::env::var(*name).ok())
}

/// The same lookup for the flags that switch the device tier on.
fn env_on(names: &[&str]) -> bool {
    env_name(names).is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

fn load_engine(opts: &Opts, threads: usize) -> Result<Engine, String> {
    if std::env::var("FENNEC_PROFILE").is_ok_and(|v| !v.is_empty() && v != "0") {
        fennec::prof::enable();
    }
    let dir = opts.dir();
    let cfg = Arc::new(TextConfig::from_file(&format!("{dir}/config.json"))?);
    let mut shards = Shards::open(&dir)?;
    attach_cache(opts, &dir, &mut shards)?;
    let sh = Arc::new(shards);
    let tok = fennec::tokenizer::Tokenizer::from_dir(&dir)?;
    let group = group_for(opts, &sh, &cfg);

    let want_gpu = gpu_wanted(opts);

    // `serve --host` is the bind address, exactly as coli takes it, so the RAM
    // pool budget cannot also live under that name: resolve it here and hand
    // `setup` a numeric `host` it can read back.
    let pool = opts.str("host-gb").and_then(|v| v.parse::<f64>().ok())
        .or_else(|| {
            env_name(&["FENNEC_VK_TIER_HOST_GB", "COLI_VK_TIER_HOST_GB"])
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(18.0);
    let mut po = Opts { pos: opts.pos.clone(), kv: opts.kv.clone(), flags: opts.flags.clone() };
    po.kv.insert("host".into(), pool.to_string());
    po.kv.remove("ctx");
    if want_gpu && !po.flags.iter().any(|f| f == "gpu") {
        po.flags.push("gpu".into());
    }
    let s = setup(&po, &cfg, group)?;
    // The suite passes `--ctx 0` to mean "pick for me"; clamping that to 512
    // would silently give an agent a half-kilobyte context window.
    let ctx = match opts.usize_at("ctx", 32768) {
        0 => 32768,
        n => n.max(512),
    };
    let ngen = opts.usize_at("ngen", 4096).max(1);
    let slots = opts.usize_at("slots", 16);
    if let Some(p) = opts.str("policy") {
        eprintln!("[fennec] policy {p}: placement is usage-ranked device/host/disk");
    }

    let t0 = Instant::now();
    let gpu = open_gpu(&po)?;
    let m = Model::load_plan(&sh, cfg.clone(), &s.plan, group, threads, ctx, slots, gpu)?;
    #[cfg(feature = "ort")]
    let m = {
        let mut m = m;
        if let Some(ort) = open_ort(&po)? {
            m.enable_ort(&sh, &ort)?;
        }
        m
    };
    let on_dev: usize = m
        .layers
        .iter()
        .map(|l| l.moe.as_ref().map_or(0, |x| x.dev.as_ref().map_or(0, |d| d.bufs.len())))
        .sum();
    let pooled: usize = s.plan.layers.iter().map(|l| l.pooled.len()).sum();
    // A dense stack routes nothing, so the expert counts would only ever read
    // zero; say what the layer actually holds instead.
    if m.cfg.is_moe() {
        eprintln!(
            "[fennec] {} layers, {on_dev} experts in VRAM, {pooled} in RAM, loaded in {:.1} s",
            m.layers.len(), t0.elapsed().as_secs_f64()
        );
    } else {
        eprintln!(
            "[fennec] {} dense layers (FFN i {}), loaded in {:.1} s",
            m.layers.len(), m.cfg.intermediate_size, t0.elapsed().as_secs_f64()
        );
    }
    let id = Path::new(&dir).file_name().map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "fennec".into());
    // coli's serve took --api-key, so launchers that still pass it keep working.
    // The env is the preferred channel: argv is world-readable in the process
    // list, and FENNEC_API_KEY is the spelling; COLI_API_KEY is its alias.
    let key = opts.str("api-key")
        .or_else(|| env_name(&["FENNEC_API_KEY", "COLI_API_KEY"]))
        .unwrap_or_else(|| "local".into());
    Ok(Engine { model: Arc::new(Mutex::new(m)), sh, cfg, tok, id, key, ctx, ngen })
}

fn serve_cmd(opts: &Opts, threads: usize) -> Result<(), String> {
    let e = load_engine(opts, threads)?;
    let bind = opts.str("host").unwrap_or_else(|| "127.0.0.1".into());
    let port = opts.usize_at("port", 8184) as u16;
    if bind != "127.0.0.1" && bind != "localhost" && bind != "::1" && !opts.has("allow-remote") {
        return Err(format!(
            "refusing to bind {bind}: the engine has no TLS and holds the model in RAM. \
             Pass --allow-remote if you really mean to expose it."
        ));
    }
    let srv = Arc::new(fennec::serve::Server::new(
        e.model, e.sh, e.cfg, e.tok, e.id, e.key, e.ctx,
    ));
    fennec::serve::serve(srv, &bind, port)
}

fn run_cmd(opts: &Opts, threads: usize) -> Result<(), String> {
    let e = load_engine(opts, threads)?;
    let prompt = if opts.pos.is_empty() {
        return Err("run needs a prompt: fennec run \"your text\"".into());
    } else {
        opts.pos.join(" ")
    };
    let system = opts.str("system");
    let mut msgs = Vec::new();
    if let Some(s) = system {
        msgs.push(("system".to_string(), s));
    }
    msgs.push(("user".to_string(), prompt));
    let thinking = !opts.has("no-think");
    let text = fennec::tokenizer::chat_prompt(&msgs, thinking);
    let ids = e.tok.encode(&text);
    let max_new = opts.usize_at("tokens", e.ngen).min(e.ctx.saturating_sub(ids.len() + 1));
    let mut s = fennec::serve::Sample::default();
    if let Some(t) = opts.str("temperature").and_then(|v| v.parse::<f32>().ok()) {
        s.temperature = t;
    }
    if opts.has("greedy") {
        s.temperature = 0.0;
    }
    let srv = fennec::serve::Server::new(
        e.model, e.sh, e.cfg, e.tok, e.id, e.key, e.ctx,
    );
    let mut out = std::io::stdout();
    fennec::prof::reset();
    let t0 = Instant::now();
    let (n, finish) = srv.generate(&ids, max_new, s, |piece, think| {
        let prefix = if think { "\x1b[2m" } else { "" };
        let suffix = if think { "\x1b[0m" } else { "" };
        let _ = out.write_all(format!("{prefix}{piece}{suffix}").as_bytes());
        let _ = out.flush();
        true
    })?;
    let dt = t0.elapsed().as_secs_f64();
    eprintln!("\n[fennec] {n} tokens in {dt:.2} s = {:.2} tok/s ({finish})", n as f64 / dt);
    if fennec::prof::enabled() {
        eprint!("{}", fennec::prof::report(dt * 1e3));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(args: &[&str]) -> Opts {
        parse(&args.iter().map(|s| s.to_string()).collect::<Vec<_>>())
    }

    #[test]
    fn a_valued_option_is_not_invisible_to_has() {
        // `--device 8` used to land in kv while has() only read flags, so the
        // budget gate in setup() never saw it and every run used the default.
        for args in [vec!["--device", "8"], vec!["--device=8"]] {
            let o = opts(&args);
            assert!(o.has("device"), "{args:?}");
            assert_eq!(o.num("device", 0.0), 8.0);
        }
        let o = opts(&["--gpu"]);
        assert!(!o.has("device"));
        assert_eq!(o.num("device", 7.5), 7.5);
    }

    #[test]
    fn a_flag_and_its_indexed_form_both_select_the_gpu() {
        assert!(gpu_wanted(&opts(&["--gpu"])));
        assert!(gpu_wanted(&opts(&["--gpu", "1"])));
        assert_eq!(opts(&["--gpu", "1"]).usize_at("gpu", 0), 1);
        assert!(!gpu_wanted(&opts(&["--gpu", "--cpu"])));
        assert!(!gpu_wanted(&opts(&["--gpu=1", "--cpu"])));
    }
}
