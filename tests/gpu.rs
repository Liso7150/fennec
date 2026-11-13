//! The device compute tier, diffed against the CPU kernels it replaces.
//!
//! A GPU gemv reduces in a different order than the CPU one, so these tests
//! compare to float tolerance, not bit for bit - but the tolerance is tight
//! (1e-5 relative), because a wrong nibble order or a wrong scale index shows
//! up as a percent, not as an ulp.
//!
//! Set `FENNEC_GPU` to a ranked device index to run the same suite on another
//! vendor: 0 is the discrete NVIDIA here, 1 the Intel Arc. Set `FENNEC_BACKEND`
//! to `cuda` to run every one of these tests through the driver-API/PTX tier
//! instead of SPIR-V; the two must agree, since they claim the same kernels.

use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, Mutex};

use fennec::backend::{self, Backend};
use fennec::config::{LayerKind, TextConfig};
use fennec::gpu::Gpu;
use fennec::kernel;
use fennec::model::{LayerState, Model};
use fennec::quant::Q4;
use fennec::safetensors::Shards;
use fennec::sched::{LayerPlan, Plan};

const MODEL: &str = r"D:\MEGAsync-sync\colibri-models\Qwen3.6-35B-A3B-bf16";

/// The bf16 export by default. `FENNEC_TEST_MODEL` points these tests at the
/// shipped int4 export instead, whose lm_head is f16 — a different kernel, and
/// the one decode actually runs.
fn model() -> String {
    std::env::var("FENNEC_TEST_MODEL").unwrap_or_else(|_| MODEL.to_string())
}

fn gpu() -> Option<Gpu> {
    let idx: usize = std::env::var("FENNEC_GPU").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let env = |k: &str| std::env::var(k).ok();
    let explicit = env("FENNEC_BACKEND").is_some();
    // An unparseable FENNEC_BACKEND must fail the test rather than quietly
    // running Vulkan and reporting a CUDA pass.
    let b = match backend::resolve(None, &env, 0, false) {
        Ok(b) => b,
        Err(e) => panic!("FENNEC_BACKEND: {e}"),
    };
    let b = if b == Backend::Cpu { Backend::Vulkan } else { b };
    match Gpu::open_backend(idx, b) {
        Ok(g) => {
            println!("  device {idx}: {} ({}) on {}", g.name(), g.info().vendor, g.backend().label());
            Some(g)
        }
        // A machine with no GPU skips these tests. A backend that was asked for
        // by name and failed is a defect, and skipping it would report a pass.
        Err(e) if explicit => panic!("FENNEC_BACKEND={} device {idx}: {e}", b.name()),
        Err(e) => {
            println!("  no compute device {idx} on {}: {e}", b.label());
            None
        }
    }
}

/// Relative RMS difference, the same metric the golden suite uses.
fn rel(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut w) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let e = *x as f64 - *y as f64;
        d += e * e;
        w += (*x as f64) * (*x as f64);
    }
    (d / w.max(1e-30)).sqrt()
}

fn close(label: &str, want: &[f32], got: &[f32], tol: f64) {
    let e = rel(want, got);
    assert!(e < tol, "{label}: relative RMS {e:.3e} over tolerance {tol:.1e}");
    println!("  {label}: relative RMS {e:.3e}");
}

/// Truncate to bf16 and hand back the exact bytes, so the CPU reference and the
/// device read the same weights rather than two roundings of them.
fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for x in v {
        out.extend_from_slice(&((x.to_bits() >> 16) as u16).to_le_bytes());
    }
    out
}

/// Truncate to f16 and hand back the exact bytes, for the same reason as
/// `bf16_bytes`. Truncation rather than rounding is deliberate: it lands some
/// weights in the f16 subnormal range, which is where a naive unpack breaks.
fn f16_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for x in v {
        out.extend_from_slice(&f32_to_f16(*x).to_le_bytes());
    }
    out
}

fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        return sign | (((man | 0x80_0000) >> (14 - e)) as u16);
    }
    sign | ((e as u16) << 10) | ((man >> 13) as u16)
}

fn lcg(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / 2147483648.0 - 1.0) * scale
        })
        .collect()
}

#[test]
fn a_device_gemv_is_the_cpu_gemv() {
    let Some(mut g) = gpu() else { return };
    let threads = 8;

    for (rows, cols) in [(300usize, 2048usize), (512, 512), (64, 2048)] {
        let w = bf16_bytes(&lcg(rows * cols, 7, 0.05));
        let x = lcg(cols, 11, 1.0);
        let mut want = vec![0f32; rows];
        kernel::gemv_bf16(&w, rows, cols, &x, &mut want, threads);
        let wb = g.upload(&w).unwrap();
        g.write(0, &x);
        g.batch().unwrap().gemv_half(wb, rows, cols, 0, cols).unwrap().run().unwrap();
        close(&format!("bf16 {rows}x{cols}"), &want, &g.read(cols, rows), 2e-6);
    }

    // The same shapes in f16, the format the text-only int4 export stores its
    // dense weights and lm_head in. cols % 8 != 0 takes the scalar fallback.
    for (rows, cols) in [(300usize, 2048usize), (512, 512), (64, 2048), (17, 12)] {
        let w = f16_bytes(&lcg(rows * cols, 29, 0.05));
        let x = lcg(cols, 31, 1.0);
        let mut want = vec![0f32; rows];
        kernel::gemv_f16(&w, rows, cols, &x, &mut want, threads);
        let wb = g.upload_half(&w, true).unwrap();
        g.write(0, &x);
        g.batch().unwrap().gemv_half(wb, rows, cols, 0, cols).unwrap().run().unwrap();
        close(&format!("f16 {rows}x{cols}"), &want, &g.read(cols, rows), 2e-6);
    }

    // Taller than one dispatch: the spec only guarantees 65535 workgroups, so
    // this is the row_base path. 70000 x 64 bf16 is 8.96 MB, cheap to upload.
    let (rows, cols) = (70000usize, 64usize);
    let w = bf16_bytes(&lcg(rows * cols, 13, 0.05));
    let x = lcg(cols, 17, 1.0);
    let mut want = vec![0f32; rows];
    kernel::gemv_bf16(&w, rows, cols, &x, &mut want, threads);
    let wb = g.upload(&w).unwrap();
    g.write(0, &x);
    g.batch().unwrap().gemv_half(wb, rows, cols, 0, cols).unwrap().run().unwrap();
    close("bf16 70000x64 (multi-dispatch)", &want, &g.read(cols, rows), 2e-6);

    for (rows, cols) in [(1024usize, 2048usize), (2048, 512), (256, 2048)] {
        let q = Q4::quantize(&lcg(rows * cols, 19, 0.04), rows, cols, 32, threads);
        let x = lcg(cols, 23, 1.0);
        let want: Vec<f32> = (0..rows).map(|r| q.dot_row(r, &x)).collect();
        let (wb, sb) = g.upload_q4(&q).unwrap();
        g.write(0, &x);
        g.batch().unwrap().gemv_q4(wb, sb, rows, cols, 32, 0, cols).unwrap().run().unwrap();
        close(&format!("q4 {rows}x{cols}"), &want, &g.read(cols, rows), 2e-5);
    }

    // group 64 is what rabbit and the Colibri int4 directory use.
    let (rows, cols) = (512usize, 2048usize);
    let q = Q4::quantize(&lcg(rows * cols, 29, 0.04), rows, cols, 64, threads);
    let x = lcg(cols, 31, 1.0);
    let want: Vec<f32> = (0..rows).map(|r| q.dot_row(r, &x)).collect();
    let (wb, sb) = g.upload_q4(&q).unwrap();
    g.write(0, &x);
    g.batch().unwrap().gemv_q4(wb, sb, rows, cols, 64, 0, cols).unwrap().run().unwrap();
    close("q4 group 64", &want, &g.read(cols, rows), 2e-5);
}

/// Subnormal f16 weights are |w| < 6.1e-5, which real checkpoints do contain.
/// A decode that flushes them reads zero for most of a small-magnitude matrix
/// and still looks plausible in a logits diff, so this pins the whole range:
/// at this scale roughly 60% of the weights are subnormal.
#[test]
fn an_f16_gemv_keeps_subnormal_weights() {
    let Some(mut g) = gpu() else { return };
    let (rows, cols) = (64usize, 2048usize);
    let w = f16_bytes(&lcg(rows * cols, 43, 1e-4));
    let x = lcg(cols, 47, 1.0);
    let mut want = vec![0f32; rows];
    kernel::gemv_f16(&w, rows, cols, &x, &mut want, 8);
    let wb = g.upload_half(&w, true).unwrap();
    g.write(0, &x);
    g.batch().unwrap().gemv_half(wb, rows, cols, 0, cols).unwrap().run().unwrap();
    close("f16 subnormal 64x2048", &want, &g.read(cols, rows), 2e-6);
}

#[test]
fn the_elementwise_ops_agree() {
    let Some(mut g) = gpu() else { return };
    let n = 2048usize;
    let x = lcg(n, 37, 2.0);
    let w = lcg(n, 41, 0.1);
    let eps = 1e-6f32;

    let mut want = vec![0f32; n];
    kernel::rmsnorm(&x, &w, eps, &mut want);
    g.write(0, &x);
    g.write(n, &w);
    g.batch().unwrap().rmsnorm(n, 0, n, 2 * n, eps).unwrap().run().unwrap();
    close("rmsnorm (zero-centred)", &want, &g.read(2 * n, n), 2e-6);

    let a = lcg(n, 43, 3.0);
    let b = lcg(n, 47, 1.5);
    let want: Vec<f32> = (0..n).map(|i| kernel::silu(a[i]) * b[i]).collect();
    g.write(0, &a);
    g.write(n, &b);
    g.batch().unwrap().silu_mul(n, 0, n, 2 * n).unwrap().run().unwrap();
    close("silu_mul", &want, &g.read(2 * n, n), 2e-6);

    let seed = lcg(n, 53, 1.0);
    let mut want = seed.clone();
    kernel::axpy(&mut want, &x, 0.375);
    g.write(0, &x);
    g.write(2 * n, &seed);
    g.batch().unwrap().axpy(n, 0, 2 * n, 0.375, false).unwrap().run().unwrap();
    close("axpy", &want, &g.read(2 * n, n), 2e-6);

    // `set` is the first accumulation of a token: the MoE block no longer
    // zero-fills its accumulator, so this must overwrite whatever is there
    // rather than add to it. The seed is deliberately non-zero and large.
    let want: Vec<f32> = x.iter().map(|v| v * 0.375).collect();
    g.write(2 * n, &lcg(n, 61, 9.0));
    g.batch().unwrap().axpy(n, 0, 2 * n, 0.375, true).unwrap().run().unwrap();
    close("axpy (set)", &want, &g.read(2 * n, n), 2e-6);

    // Chained in one submit, reading what the previous op wrote: norm, then
    // scale the result by a gate. No host round trip between them.
    let mut want = vec![0f32; n];
    kernel::rmsnorm(&x, &w, eps, &mut want);
    // in-place axpy: out and x are the same region, which is safe for a 1:1
    // elementwise op and is exactly what the residual accumulation does.
    for v in want.iter_mut() {
        *v *= 0.75;
    }
    g.write(0, &x);
    g.write(n, &w);
    {
        let mut b = g.batch().unwrap();
        b.rmsnorm(n, 0, n, 2 * n, eps).unwrap();
        b.axpy(n, 2 * n, 2 * n, -0.25, false).unwrap();
        b.run().unwrap();
    }
    close("rmsnorm -> axpy chained", &want, &g.read(2 * n, n), 2e-6);
}

#[test]
fn a_whole_expert_runs_on_the_device() {
    let Some(mut g) = gpu() else { return };
    let (h, i, k) = (2048usize, 512usize, 8usize);
    let x = lcg(h, 59, 1.0);
    g.write(0, &x);

    // Arena layout: x at 0, then per expert a gate_up block, a hidden block and
    // an output block, all disjoint so eight experts fit in one command buffer.
    let stride = 8192usize;
    let base = 4096usize;
    let mut packs = Vec::new();
    let mut want = vec![0f32; k * h];
    for e in 0..k {
        let gu = Q4::quantize(&lcg(2 * i * h, 100 + e as u64, 0.04), 2 * i, h, 32, 8);
        let dn = Q4::quantize(&lcg(h * i, 200 + e as u64, 0.04), h, i, 32, 8);
        let mut hh = vec![0f32; i];
        for r in 0..i {
            hh[r] = kernel::silu(gu.dot_row(r, &x)) * gu.dot_row(i + r, &x);
        }
        for r in 0..h {
            want[e * h + r] = dn.dot_row(r, &hh);
        }
        packs.push((g.upload_q4(&gu).unwrap(), g.upload_q4(&dn).unwrap()));
    }

    let before = g.submitted();
    {
        let mut b = g.batch().unwrap();
        for (e, &((gw, gs), (dw, ds))) in packs.iter().enumerate() {
            let gu_off = base + e * stride;
            let hh_off = gu_off + 2 * i;
            let out_off = gu_off + 4 * i;
            b.gemv_q4(gw, gs, 2 * i, h, 32, 0, gu_off).unwrap();
            b.silu_mul(i, gu_off, gu_off + i, hh_off).unwrap();
            b.gemv_q4(dw, ds, h, i, 32, hh_off, out_off).unwrap();
        }
        b.run().unwrap();
    }
    assert_eq!(g.submitted() - before, 1, "eight experts cost more than one submit");
    for e in 0..k {
        let out_off = base + e * stride + 4 * i;
        close(
            &format!("expert {e} gate_up -> silu -> down"),
            &want[e * h..(e + 1) * h],
            &g.read(out_off, h),
            2e-5,
        );
    }
    println!(
        "  {} ops, {} dispatches, one submit, {:.2} GiB of weights held",
        k * 3,
        g.dispatches(),
        g.bytes_held() as f64 / 2f64.powi(30)
    );
}

#[test]
fn the_lm_head_on_the_device() {
    let Some(mut g) = gpu() else { return };
    let model = model();
    if !std::path::Path::new(&model).is_dir() {
        println!("  no checkpoint at {model}");
        return;
    }
    let sh = Shards::open(&model).expect("shards");
    let head = "model.language_model.lm_head.weight";
    let name = if sh.contains(head) { head } else { "lm_head.weight" };
    let info = sh.info(name).expect("lm_head");
    let cols = 2048usize;
    let rows = info.elems() / cols;
    let bytes = &sh.data(info.file)[info.off..info.off + info.elems() * 2];
    // The bf16 export and the text-only int4 export store the head in different
    // half formats; reading one with the other's kernel is silently wrong.
    let f16 = info.dtype == fennec::safetensors::DType::F16;
    let cpu_gemv = if f16 { kernel::gemv_f16 } else { kernel::gemv_bf16 };
    let fmt = if f16 { "f16" } else { "bf16" };

    let x = lcg(cols, 61, 1.0);
    let mut want = vec![0f32; rows];
    // Warm the mmap first: a cold 1 GiB read is disk, not compute, and quoting
    // it against a warm device number would be measuring the page cache.
    // Eight threads, not all twenty-two: this runs while the machine is in use.
    cpu_gemv(bytes, rows, cols, &x, &mut want, 8);
    let t = std::time::Instant::now();
    for _ in 0..3 {
        cpu_gemv(bytes, rows, cols, &x, &mut want, 8);
    }
    let cpu = t.elapsed() / 3;

    let t = std::time::Instant::now();
    let wb = g.upload_half(bytes, f16).unwrap();
    let upload = t.elapsed();
    g.reserve(rows + cols).unwrap();
    g.write(0, &x);
    let t = std::time::Instant::now();
    g.batch().unwrap().gemv_half(wb, rows, cols, 0, cols).unwrap().run().unwrap();
    let cold = t.elapsed();
    let mut got = vec![0f32; rows];
    let t = std::time::Instant::now();
    for _ in 0..3 {
        g.batch().unwrap().gemv_half(wb, rows, cols, 0, cols).unwrap().run().unwrap();
    }
    let warm = t.elapsed() / 3;
    // Decode pays for the round trip, not just the kernel: a host-visible arena
    // reads back with a memcpy, a device-local one crosses PCIe. Time both legs
    // so the end-to-end head number can be attributed.
    let t = std::time::Instant::now();
    for _ in 0..3 {
        g.read_into(cols, &mut got);
    }
    let readback = t.elapsed() / 3;
    let t = std::time::Instant::now();
    for _ in 0..3 {
        g.write(0, &x);
    }
    let upload_x = t.elapsed() / 3;

    close(&format!("lm_head {rows}x{cols} {fmt}"), &want, &got, 2e-5);
    let argmax = |v: &[f32]| {
        v.iter().enumerate().fold((0, f32::NEG_INFINITY), |c, (i, &x)| if x > c.1 { (i, x) } else { c }).0
    };
    assert_eq!(argmax(&want), argmax(&got), "the device picked a different token");
    println!(
        "  lm_head {fmt} {:.2} GiB: upload {:.0} ms | cpu {:.1} ms | device cold {:.1} ms warm {:.1} ms ({:.1}x) | {:.0} GB/s | x up {:.2} ms, logits down {:.2} ms",
        bytes.len() as f64 / 2f64.powi(30),
        upload.as_secs_f64() * 1e3,
        cpu.as_secs_f64() * 1e3,
        cold.as_secs_f64() * 1e3,
        warm.as_secs_f64() * 1e3,
        cpu.as_secs_f64() / warm.as_secs_f64(),
        bytes.len() as f64 / warm.as_secs_f64() / 1e9,
        upload_x.as_secs_f64() * 1e3,
        readback.as_secs_f64() * 1e3,
    );
}

/// The shipped path: `load_plan` with a device against the same plan without
/// one. Half of layer 0's routed picks go to VRAM and half stay in RAM, so one
/// token is answered by both at once, and the logits must still be the CPU's.
#[test]
fn a_planned_stack_splits_one_token_across_cpu_and_device() {
    let Some(g) = gpu() else { return };
    let model = model();
    if !std::path::Path::new(&model).is_dir() {
        println!("  no checkpoint at {model}");
        return;
    }
    let sh = Arc::new(Shards::open(&model).expect("shards"));
    let cfg = Arc::new(TextConfig::from_file(&format!("{model}/config.json")).expect("config"));
    let (group, threads, h) = (32usize, 8usize, cfg.hidden_size);
    let all: Vec<usize> = (0..cfg.num_experts).collect();
    let plan_of = |resident: Vec<usize>, pooled: Vec<usize>| Plan {
        layers: vec![LayerPlan {
            layer: 0,
            resident,
            pooled,
            streamed: Vec::new(),
            bytes_expert: 0,
            dense_bytes: 0,
            dense_on_device: false,
        }],
        device_bytes_used: 0,
        device_bytes_budget: 0,
        host_bytes_used: 0,
        host_bytes_budget: 0,
        device_hit_rate: 0.0,
        host_hit_rate: 0.0,
        disk_rate: 0.0,
        expert_bytes: 0,
        dense_bytes_layer: 0,
        ms_per_token: 0.0,
        lm_head_ms: 0.0,
    };

    let cpu = Model::load_plan(
        &sh, cfg.clone(), &plan_of(Vec::new(), all.clone()), group, threads, 8, 0, None,
    )
    .expect("cpu stack");
    // Ask the router where this token actually goes, then hand the device half
    // of those picks: a split that is guaranteed to fire, not one that hopes.
    // The MoE is fed the post-attention norm, so walk layer 0's prefix to get
    // the vector the router really sees.
    let x0 = cpu.embed_row(&sh, 15059).expect("embed");
    let eps = cfg.rms_norm_eps as f32;
    let l = &cpu.layers[0];
    let (mut buf, mut res) = (vec![0f32; h], vec![0f32; h]);
    kernel::rmsnorm(&x0, &l.norm_in, eps, &mut buf);
    let mut st0 = l.state(&cfg, 8);
    match (&l.kind, &mut st0) {
        (LayerKind::Linear, LayerState::Linear(g)) => l.gdn.as_ref().unwrap().step(&buf, g, &mut res),
        (LayerKind::Full, LayerState::Full(a)) => l.attn.as_ref().unwrap().step(&buf, 0, a, &mut res),
        _ => panic!("layer state mismatch"),
    }
    let mut pre = x0.clone();
    kernel::axpy(&mut pre, &res, 1.0);
    kernel::rmsnorm(&pre, &l.norm_post, eps, &mut buf);
    let r = l.moe.route(&buf);
    let dev: Vec<usize> = r.ids[..r.ids.len() / 2].to_vec();
    let host: Vec<usize> = all.iter().copied().filter(|e| !dev.contains(e)).collect();

    let gpu = Arc::new(Mutex::new(g));
    let split = Model::load_plan(
        &sh, cfg.clone(), &plan_of(dev.clone(), host), group, threads, 8, 0, Some(gpu.clone()),
    )
    .expect("split stack");

    let (mut a, mut b) = (x0.clone(), x0);
    let (mut sa, mut sb) = (cpu.states(8), split.states(8));
    cpu.forward(&mut a, &mut sa, 0);
    split.forward(&mut b, &mut sb, 0);
    close("layer 0 hidden state", &a, &b, 2e-5);

    let (mut na, mut nb) = (vec![0f32; h], vec![0f32; h]);
    kernel::rmsnorm(&a, &cpu.norm, eps, &mut na);
    kernel::rmsnorm(&b, &split.norm, eps, &mut nb);
    let (mut la, mut lb) = (vec![0f32; cfg.vocab_size], vec![0f32; cfg.vocab_size]);
    cpu.logits(&sh, &na, &mut la).expect("cpu logits");
    split.logits(&sh, &nb, &mut lb).expect("device logits");
    close("lm_head logits", &la, &lb, 2e-5);
    let argmax = |v: &[f32]| {
        v.iter().enumerate().fold((0, f32::NEG_INFINITY), |c, (i, &x)| if x > c.1 { (i, x) } else { c }).0
    };
    assert_eq!(argmax(&la), argmax(&lb), "the split stack picked a different token");

    let m = &split.layers[0].moe;
    let (d, hh, c) = (m.dev_hits.load(Relaxed), m.host_hits.load(Relaxed), m.disk_hits.load(Relaxed));
    assert_eq!(d, dev.len() as u64, "the device tier did not take its picks");
    assert_eq!(hh, (r.ids.len() - dev.len()) as u64, "the host tier did not take its picks");
    assert_eq!(c, 0, "nothing should have streamed");
    let overlap = m.overlap_ns.load(Relaxed);
    assert!(overlap > 0, "no CPU work happened while the device batch was in flight");
    let g = gpu.lock().unwrap();
    println!(
        "  one token: {d} picks on {}, {hh} on the CPU, {:.2} ms of CPU work inside the device batch | {} dispatches, {} submits",
        g.name(),
        overlap as f64 / 1e6,
        g.dispatches(),
        g.submitted(),
    );
}

/// The dense half of the device tier: the same stack with its attention and
/// delta-net projections in VRAM as bf16 against the CPU's f32 copy of the same
/// bf16 weights. Layer 0 is a gated delta net and layer 3 is full attention, so
/// both blocks are covered, and four tokens walk the conv window and the KV
/// cache through the state they actually run in.
#[test]
fn dense_projections_on_the_device_match_the_cpu_stack() {
    let Some(g) = gpu() else { return };
    let model = model();
    if !std::path::Path::new(&model).is_dir() {
        println!("  no checkpoint at {model}");
        return;
    }
    let sh = Arc::new(Shards::open(&model).expect("shards"));
    let cfg = Arc::new(TextConfig::from_file(&format!("{model}/config.json")).expect("config"));
    let (group, threads, h) = (32usize, 8usize, cfg.hidden_size);
    let all: Vec<usize> = (0..cfg.num_experts).collect();
    // Every expert stays in RAM on both sides, so the only difference between
    // the two stacks is where the dense projections live.
    let plan_of = |dense_on_device: bool| Plan {
        layers: [0usize, 3]
            .iter()
            .map(|&l| LayerPlan {
                layer: l,
                resident: Vec::new(),
                pooled: all.clone(),
                streamed: Vec::new(),
                bytes_expert: 0,
                dense_bytes: 0,
                dense_on_device,
            })
            .collect(),
        device_bytes_used: 0,
        device_bytes_budget: 0,
        host_bytes_used: 0,
        host_bytes_budget: 0,
        device_hit_rate: 0.0,
        host_hit_rate: 0.0,
        disk_rate: 0.0,
        expert_bytes: 0,
        dense_bytes_layer: 0,
        ms_per_token: 0.0,
        lm_head_ms: 0.0,
    };

    let cpu = Model::load_plan(&sh, cfg.clone(), &plan_of(false), group, threads, 8, 0, None)
        .expect("cpu stack");
    let gpu = Arc::new(Mutex::new(g));
    let dev = Model::load_plan(&sh, cfg.clone(), &plan_of(true), group, threads, 8, 0, Some(gpu.clone()))
        .expect("device stack");

    // A silent fallback to the CPU would pass the numeric comparison and prove
    // nothing, so assert the move happened and the host copies were released.
    let gdn = dev.layers[0].gdn.as_ref().unwrap();
    let attn = dev.layers[1].attn.as_ref().unwrap();
    assert!(gdn.dev.is_some(), "the delta net did not move to the device");
    assert!(attn.dev.is_some(), "attention did not move to the device");
    assert!(gdn.in_qkv.is_empty() && gdn.out_proj.is_empty(), "the delta net host copy was not released");
    assert!(attn.q.is_empty() && attn.o.is_empty(), "the attention host copy was not released");

    let toks = [15059usize, 9948, 271, 1024];
    let (mut sa, mut sb) = (cpu.states(8), dev.states(8));
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for (pos, &t) in toks.iter().enumerate() {
        a = cpu.embed_row(&sh, t).expect("embed");
        b = dev.embed_row(&sh, t).expect("embed");
        cpu.forward(&mut a, &mut sa, pos);
        dev.forward(&mut b, &mut sb, pos);
        close(&format!("token {pos} hidden state"), &a, &b, 2e-5);
    }

    let eps = cfg.rms_norm_eps as f32;
    let (mut na, mut nb) = (vec![0f32; h], vec![0f32; h]);
    kernel::rmsnorm(&a, &cpu.norm, eps, &mut na);
    kernel::rmsnorm(&b, &dev.norm, eps, &mut nb);
    let (mut la, mut lb) = (vec![0f32; cfg.vocab_size], vec![0f32; cfg.vocab_size]);
    cpu.logits(&sh, &na, &mut la).expect("cpu logits");
    dev.logits(&sh, &nb, &mut lb).expect("device logits");
    close("lm_head logits", &la, &lb, 2e-5);
    let argmax = |v: &[f32]| {
        v.iter().enumerate().fold((0, f32::NEG_INFINITY), |c, (i, &x)| if x > c.1 { (i, x) } else { c }).0
    };
    assert_eq!(argmax(&la), argmax(&lb), "the dense-device stack picked a different token");
    let g = gpu.lock().unwrap();
    println!(
        "  2 layers x {} tokens, dense bf16 on {} | {:.2} GiB device total, {} dispatches in {} submits",
        toks.len(),
        g.name(),
        g.bytes_held() as f64 / 2f64.powi(30),
        g.dispatches(),
        g.submitted(),
    );
}

/// The fused mixer: conv, window rotation, the delta recursion and the gated
/// norm all on the device, against the same layer run on the CPU. The state is
/// compared too, not just the output — a recurrence can produce a plausible
/// first token from a wrong state and then diverge, and the conv window only
/// shows up once it has been rotated past its depth.
#[test]
fn fused_mixer_matches_the_cpu_mixer_and_its_state() {
    let Some(g) = gpu() else { return };
    let model = model();
    if !std::path::Path::new(&model).is_dir() {
        println!("  no checkpoint at {model}");
        return;
    }
    let sh = Arc::new(Shards::open(&model).expect("shards"));
    let cfg = Arc::new(TextConfig::from_file(&format!("{model}/config.json")).expect("config"));
    let (group, threads) = (32usize, 8usize);
    let all: Vec<usize> = (0..cfg.num_experts).collect();
    let plan_of = |dense_on_device: bool| Plan {
        layers: vec![LayerPlan {
            layer: 0,
            resident: Vec::new(),
            pooled: all.clone(),
            streamed: Vec::new(),
            bytes_expert: 0,
            dense_bytes: 0,
            dense_on_device,
        }],
        device_bytes_used: 0,
        device_bytes_budget: 0,
        host_bytes_used: 0,
        host_bytes_budget: 0,
        device_hit_rate: 0.0,
        host_hit_rate: 0.0,
        disk_rate: 0.0,
        expert_bytes: 0,
        dense_bytes_layer: 0,
        ms_per_token: 0.0,
        lm_head_ms: 0.0,
    };
    let cpu = Model::load_plan(&sh, cfg.clone(), &plan_of(false), group, threads, 8, 0, None).expect("cpu stack");
    let gpu = Arc::new(Mutex::new(g));
    let dev = Model::load_plan(&sh, cfg.clone(), &plan_of(true), group, threads, 8, 0, Some(gpu.clone())).expect("device stack");

    let a = cpu.layers[0].gdn.as_ref().unwrap();
    let b = dev.layers[0].gdn.as_ref().unwrap();
    if b.dev.as_ref().and_then(|d| d.mixer.as_ref()).is_none() {
        println!("  head dims outside the fused kernel's shared memory; split path only");
        return;
    }
    assert!(a.dev.is_none(), "the reference stack must keep its mixer on the CPU");
    // Eight tokens: the conv window is four deep, so this rotates it twice over.
    let toks = [15059usize, 9948, 271, 1024, 311, 4096, 88, 12];
    let (mut sa, mut sb) = (cpu.states(16), dev.states(16));
    for (pos, &t) in toks.iter().enumerate() {
        let mut x = cpu.embed_row(&sh, t).expect("embed");
        let mut y = x.clone();
        cpu.forward(&mut x, &mut sa, pos);
        dev.forward(&mut y, &mut sb, pos);
        close(&format!("token {pos} hidden state"), &x, &y, 2e-5);
    }
    let (ga, gb) = match (&mut sa[0], &mut sb[0]) {
        (LayerState::Linear(p), LayerState::Linear(q)) => (p, q),
        _ => panic!("layer 0 is not a gated delta net"),
    };
    assert!(gb.mirrored, "the fused path never took ownership of the state");
    b.sync_state(gb).expect("read the device state back");
    close("conv window", &ga.conv, &gb.conv, 2e-5);
    close("recurrent state", &ga.rec, &gb.rec, 2e-5);

    // Ownership handback: with `mirrored` cleared the next step must re-upload
    // the host state and still agree, which is what a fresh request relies on.
    let mut x = cpu.embed_row(&sh, 512).expect("embed");
    let mut y = x.clone();
    cpu.forward(&mut x, &mut sa, toks.len());
    dev.forward(&mut y, &mut sb, toks.len());
    close("token after state handback", &x, &y, 2e-5);
}

fn gemv_once(g: &mut Gpu, wb: usize, x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    g.write(0, x);
    g.batch().unwrap().gemv_half(wb, rows, cols, 0, cols).unwrap().run().unwrap();
    g.read(cols, rows)
}

/// `serve` loads the model on one thread and answers each request on another.
/// A CUDA context is current per thread, so the device has to keep working after
/// the handle moves. Before this was pinned, the first request died with
/// "cuMemcpyHtoD: invalid device context" while every single-threaded test —
/// including all of `decode` — stayed green.
#[test]
fn the_device_works_after_the_handle_moves_to_another_thread() {
    let Some(mut g) = gpu() else { return };
    let (rows, cols, threads) = (256usize, 2048usize, 8usize);
    let w = f16_bytes(&lcg(rows * cols, 41, 0.05));
    let x = lcg(cols, 43, 1.0);
    let mut want = vec![0f32; rows];
    kernel::gemv_f16(&w, rows, cols, &x, &mut want, threads);
    let wb = g.upload_half(&w, true).unwrap();

    close("on the opening thread", &want, &gemv_once(&mut g, wb, &x, rows, cols), 2e-6);

    let xt = x.clone();
    let (got, mut g) = std::thread::spawn(move || {
        let y = gemv_once(&mut g, wb, &xt, rows, cols);
        (y, g)
    })
    .join()
    .expect("the request thread panicked");
    close("on a request thread", &want, &got, 2e-6);

    close("back on the opening thread", &want, &gemv_once(&mut g, wb, &x, rows, cols), 2e-6);
}