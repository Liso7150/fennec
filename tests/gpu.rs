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

const MODEL: &str = r"D:\fennec-models\Qwen3.6-35B-A3B-bf16";

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

/// The device clock, proven on hardware.
///
/// `MIXER_DEV_DEVICE` is the row that says how much of a mixer batch the
/// silicon executed as opposed to how long the host sat blocked in the readback,
/// and it is built from CUDA events, which are positions in a stream rather than
/// host stamps. That distinction is exactly what a unit test cannot fake, so this
/// drives one real batch through arm → submit → drain → harvest and demands a
/// sample back. Vulkan stays a deliberate no-op: its arena is host-visible, so
/// its upload/compute/readback split already accounts for the whole batch.
#[test]
fn a_timed_mixer_batch_reports_the_device_clock() {
    let Some(mut g) = gpu() else { return };
    if g.backend() != Backend::Cuda {
        println!("  {} has no device clock to bracket", g.backend().label());
        return;
    }
    let (rows, cols) = (2048usize, 2048usize);
    let w = f16_bytes(&lcg(rows * cols, 71, 5e-2));
    let x = lcg(cols, 73, 1.0);
    let wb = g.upload_half(&w, true).unwrap();
    g.write(0, &x);

    // Arming is one-shot: a batch nobody asked about must not be billed, or the
    // row would count submissions the mixer never attributed to itself.
    g.batch().unwrap().gemv_half(wb, rows, cols, 0, cols).unwrap().submit().unwrap();
    g.wait().unwrap();
    assert_eq!(g.device_samples(), 0, "an un-armed submit must not be timed");

    g.time_mixer_batch();
    g.batch().unwrap().gemv_half(wb, rows, cols, 0, cols).unwrap().submit().unwrap();
    // Draining is what fires both markers; the harvest itself never waits.
    g.wait().unwrap();
    g.time_mixer_batch();
    assert_eq!(g.device_samples(), 1, "one armed batch is exactly one sample");
    let ms = g.device_ms();
    assert!(ms > 0.0, "the device reported {ms} ms for a {rows}x{cols} f16 gemv");
    println!("  device clock: {ms:.3} ms for one timed {rows}x{cols} gemv");

    // And the bracket must be a pure observer: what ran is still what the CPU
    // kernels get.
    let mut want = vec![0f32; rows];
    kernel::gemv_f16(&w, rows, cols, &x, &mut want, 8);
    close("timed gemv", &want, &g.read(cols, rows), 2e-6);
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
        host_dense_bytes: 0,
        device_hit_rate: 0.0,
        host_hit_rate: 0.0,
        disk_rate: 0.0,
        expert_bytes: 0,
        dense_bytes_layer: 0,
        ms_per_token: 0.0,
        lm_head_ms: 0.0,
        moe_device_ms: 0.0,
        moe_host_ms: 0.0,
        overlap_credit_ms: 0.0,
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
    let r = l.moe.as_ref().unwrap().route(&buf);
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

    let m = split.layers[0].moe.as_ref().unwrap();
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
        host_dense_bytes: 0,
        device_hit_rate: 0.0,
        host_hit_rate: 0.0,
        disk_rate: 0.0,
        expert_bytes: 0,
        dense_bytes_layer: 0,
        ms_per_token: 0.0,
        lm_head_ms: 0.0,
        moe_device_ms: 0.0,
        moe_host_ms: 0.0,
        overlap_credit_ms: 0.0,
    };

    let cpu = Model::load_plan(&sh, cfg.clone(), &plan_of(false), group, threads, 8, 0, None)
        .expect("cpu stack");
    let gpu = Arc::new(Mutex::new(g));
    let dev = Model::load_plan(&sh, cfg.clone(), &plan_of(true), group, threads, 8, 0, Some(gpu.clone()))
        .expect("device stack");

    // A silent fallback to the CPU would pass the numeric comparison and prove
    // nothing, so assert the move happened and the host copies were released.
    // The batch prompt path a device plan has is the device one: `prefill_ready`
    // stays false because its kernels are host kernels the plan dropped.
    let gdn = dev.layers[0].gdn.as_ref().unwrap();
    let attn = dev.layers[1].attn.as_ref().unwrap();
    assert!(gdn.dev.is_some(), "the delta net did not move to the device");
    assert!(attn.dev.is_some(), "attention did not move to the device");
    assert!(gdn.in_qkv.is_empty() && gdn.out_proj.is_empty(), "the delta net host copy was not released");
    assert!(attn.q.is_empty() && attn.o.is_empty(), "the attention host copy was not released");
    assert!(!dev.prefill_ready(), "a device plan claims a batch prompt path it dropped");
    assert!(cpu.prefill_ready(), "the host stack keeps its batch prompt path");

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
        host_dense_bytes: 0,
        device_hit_rate: 0.0,
        host_hit_rate: 0.0,
        disk_rate: 0.0,
        expert_bytes: 0,
        dense_bytes_layer: 0,
        ms_per_token: 0.0,
        lm_head_ms: 0.0,
        moe_device_ms: 0.0,
        moe_host_ms: 0.0,
        overlap_credit_ms: 0.0,
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
    // The queue belongs to the layer, not to the token, so CUDA captures it once
    // and replays it for the rest. Falling back to per-launch submits keeps every
    // number above identical, so the counters are the only proof the tier is
    // actually graphed and not quietly degrading to seven launches a token.
    let (backend, (captures, hits)) = {
        let g = gpu.lock().unwrap();
        (g.backend(), g.graphs())
    };
    if backend == Backend::Cuda {
        assert_eq!(captures, 1, "one capture for the one layer's mixer queue");
        assert_eq!(hits, toks.len() as u64 - 1, "every later token replays it");
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

/// The GDN mixer stage, kernel by kernel, because `MIXER_DEV` cannot be split
/// in place.
///
/// `Gdn::step_fused` (src/model.rs) records the whole layer — four input
/// projections, a barrier, the conv, the delta recursion, out_proj — into ONE
/// command buffer with ONE submit. A timer around any of those `bt.` calls
/// measures the CPU that recorded the kernel, never the device that ran it, so
/// the profiled stage stays one number however it is instrumented. This probe
/// rebuilds the same shapes on synthetic buffers of the same byte counts and
/// times them apart, which is the precondition the defect register sets before
/// another mixer rewrite is proposed, after the fusion and int4-trunk variants
/// that regressed (#86, #127, #142, #147).
///
/// Printed, not asserted: a shape probe on one laptop is evidence for the next
/// decision, not a regression gate. Run it on the tier being tuned:
///
/// ```text
/// FENNEC_BACKEND=cuda cargo test --release --offline --test gpu -- \
///     --ignored --nocapture
/// ```
///
/// # What it said on CUDA, and what that decided
///
/// The stage is weight bandwidth (`logs/mixer-bandwidth.log`, task #470). One
/// GDN layer streams 67371008 bytes = 64.25 MiB of f16 weights, and the fused
/// submit moves them at 193 GB/s against the 230 GB/s the *same kernel* gets from
/// a 64 MiB buffer: 84% of what the bus allows at that moment. `fused x 30
/// layers` = 10.5 ms/token, which is where the served `mixer dev device` row
/// sits (10.14-11.58), so one layer's cost times the layer count is the whole
/// stage and no launch, record or fence arithmetic is missing from it.
///
/// Two readings the probe exists to prevent:
///
/// * **The part rows are not a target.** Each replays one buffer 300 times, so
///   its weights stay in the on-die cache and it prints 443-574 GB/s on a card
///   whose DRAM ceiling is 230. Adding the parts (205 us) and calling the fused
///   layer (350 us) a regression measures cache, not code.
/// * **The ceiling has to be measured under sustained load.** This card idles its
///   memory bus at 810 MHz of an 8001 MHz maximum, and a burst taken before it
///   ramps reads 160-205 GB/s. Three seconds of continuous reads first is what
///   makes the ceiling line mean anything.
///
/// The ratio is the portable quantity, not the level: a cold-clock session and a
/// ramped one gave 138/160 and 193/230 GB/s, i.e. 86% and 84% of their own
/// ceilings, while the fused layer's absolute microseconds differed by 40%.
#[test]
#[ignore = "device timing probe, prints instead of asserting"]
fn mixer_kernel_shapes() {
    let Some(mut g) = gpu() else { return };
    const ITERS: usize = 300;

    // qwen36-35b's GDN layer, read off its config.json: hidden 2048, 16 key
    // heads x 128, 32 value heads x 128, conv window 4. Same arithmetic as
    // TextConfig::key_dim / value_dim / conv_dim, spelled out so the probe does
    // not need a checkpoint on disk.
    let (h, nk, nv, dk, dv) = (2048usize, 16usize, 32usize, 128usize, 128usize);
    let (kd, vd) = (nk * dk, nv * dv);
    let (cd, kdim) = (2 * kd + vd, 4usize);
    let (hist, rep) = (kdim - 1, nv / nk);
    // The arena layout step_fused uses: DENSE_OFF = 1 << 19 and DENSE_FLOATS =
    // 1 << 15 are private to src/model.rs, and normed is that function's
    // rounding of the input block's end up to a 16-byte boundary.
    let dense_off = 1usize << 19;
    let ib = dense_off + h;
    let normed = (ib + cd + vd + 2 * nv + 3) & !3;

    /// f16 weights of one projection, zeroed. Values do not matter: these gemvs
    /// are weight-bandwidth-bound, and zeroes keep f16 subnormals — whose
    /// conversion cost is a separate question — out of the run.
    fn half_weights(g: &mut Gpu, rows: usize, cols: usize) -> usize {
        let id = g.alloc_half(rows * cols * 2, true).unwrap();
        g.write_buf(id, 0, &vec![0u8; rows * cols * 2]).unwrap();
        id
    }

    /// A zeroed f32 buffer, for the conv weights and the recurrent state.
    fn floats(g: &mut Gpu, n: usize) -> usize {
        let id = g.alloc_half(n * 4, false).unwrap();
        g.write_buf(id, 0, &vec![0u8; n * 4]).unwrap();
        id
    }

    /// Mean microseconds per call, after a warm-up long enough to settle the
    /// driver's command-buffer pool and the clock source.
    fn timed(g: &mut Gpu, f: impl Fn(&mut Gpu)) -> f64 {
        for _ in 0..50 {
            f(g);
        }
        let t = std::time::Instant::now();
        for _ in 0..ITERS {
            f(g);
        }
        t.elapsed().as_secs_f64() * 1e6 / ITERS as f64
    }

    fn one(g: &mut Gpu, parts: &mut Vec<(String, usize, f64)>, label: &str,
           bytes: usize, f: impl Fn(&mut Gpu)) {
        let us = timed(g, f);
        parts.push((label.to_string(), bytes, us));
    }

    g.reserve(dense_off + (1 << 15)).unwrap();
    let qkv = half_weights(&mut g, cd, h);
    let z = half_weights(&mut g, vd, h);
    let bproj = half_weights(&mut g, nv, h);
    let a = half_weights(&mut g, nv, h);
    let out = half_weights(&mut g, h, vd);
    let conv_w = floats(&mut g, cd * kdim);
    let conv_state = floats(&mut g, hist * cd);
    let params = floats(&mut g, 2 * nv + dv);
    let rec_state = floats(&mut g, nv * dk * dv);
    g.write(dense_off, &lcg(h, 7, 1.0));

    // The floor every other line carries: one command buffer, nothing recorded
    // into it, submitted and waited on.
    let floor = timed(&mut g, move |g| {
        g.batch().unwrap().run().unwrap();
    });

    let mut parts: Vec<(String, usize, f64)> = Vec::new();
    one(&mut g, &mut parts, "gemv qkv", cd * h * 2, move |g| {
        g.batch().unwrap().gemv_half_indep(qkv, cd, h, dense_off, ib).unwrap().run().unwrap();
    });
    one(&mut g, &mut parts, "gemv z", vd * h * 2, move |g| {
        g.batch().unwrap().gemv_half_indep(z, vd, h, dense_off, ib + cd).unwrap().run().unwrap();
    });
    one(&mut g, &mut parts, "gemv b", nv * h * 2, move |g| {
        g.batch().unwrap().gemv_half_indep(bproj, nv, h, dense_off, ib + cd + vd)
            .unwrap().run().unwrap();
    });
    one(&mut g, &mut parts, "gemv a", nv * h * 2, move |g| {
        g.batch().unwrap().gemv_half_indep(a, nv, h, dense_off, ib + cd + vd + nv)
            .unwrap().run().unwrap();
    });
    one(&mut g, &mut parts, "gdn_conv", cd * kdim * 4 + hist * cd * 4, move |g| {
        g.batch().unwrap().gdn_conv(conv_w, conv_state, ib, cd, kdim, hist)
            .unwrap().run().unwrap();
    });
    one(&mut g, &mut parts, "gdn_rec", nv * dk * dv * 4, move |g| {
        g.batch().unwrap()
            .gdn_rec(params, rec_state, ib, normed, kd, nv, dk, dv, rep, 1e-6)
            .unwrap().run().unwrap();
    });
    one(&mut g, &mut parts, "gemv out", h * vd * 2, move |g| {
        g.batch().unwrap().gemv_half(out, h, vd, normed, dense_off).unwrap().run().unwrap();
    });

    let (dev, backend_label) = (g.name().to_string(), g.backend().label().to_string());

    // The card's own read bandwidth, so the figures above and the layer below
    // have a ceiling to be read against. 16384x2048 f16 is 64 MiB, which no
    // on-die cache holds, and it is measured twice: once over pseudo-random
    // weights and once over zeroes. If the two agree, the zeroed buffers every
    // other row uses are not making anything look cheaper than it is.
    let (big_rows, big_cols) = (16384usize, 2048usize);
    let big_bytes = big_rows * big_cols * 2;
    let big = half_weights(&mut g, big_rows, big_cols);
    let mut rnd = vec![0u8; 1 << 20];
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    for off in (0..big_bytes).step_by(1 << 20) {
        for b in rnd.iter_mut() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *b = (s >> 24) as u8;
        }
        g.write_buf(big, off, &rnd).unwrap();
    }
    let big_gemv = |g: &mut Gpu| {
        timed(g, move |g| {
            g.batch().unwrap()
                .gemv_half_indep(big, big_rows, big_cols, dense_off, ib)
                .unwrap()
                .run()
                .unwrap();
        })
    };
    let big_once = move |g: &mut Gpu| {
        g.batch().unwrap()
            .gemv_half_indep(big, big_rows, big_cols, dense_off, ib)
            .unwrap()
            .run()
            .unwrap();
    };
    let rand_us = big_gemv(&mut g);
    let zeroes = vec![0u8; 1 << 20];
    for off in (0..big_bytes).step_by(1 << 20) {
        g.write_buf(big, off, &zeroes).unwrap();
    }
    let zero_us = big_gemv(&mut g);
    // A burst measures whatever the memory bus happens to be doing: this card
    // idles at 810 MHz against an 8001 MHz maximum, so the two bursts above may
    // be reading a downclocked bus. Three seconds of continuous load, then the
    // same burst again, and the two are printed side by side.
    let t = std::time::Instant::now();
    let mut laps = 0usize;
    while t.elapsed().as_secs_f64() < 3.0 {
        big_once(&mut g);
        laps += 1;
    }
    let secs = t.elapsed().as_secs_f64();
    let hot_us = big_gemv(&mut g);
    let busy_gbs = |bytes: usize, us: f64| bytes as f64 / ((us - floor).max(1e-6) * 1e-6) / 1e9;

    // The layer as decode runs it, and two variations that say which of the
    // stage's costs are real: dropping the barrier tests whether the four input
    // kernels were ever actually serialized, and skipping the submit leaves the
    // recording alone, which is what an in-place timer would have reported.
    let mut layer = move |barrier: bool, submit: bool| -> f64 {
        timed(&mut g, move |g| {
            let mut bt = g.batch().unwrap();
            bt.gemv_half_indep(qkv, cd, h, dense_off, ib).unwrap();
            bt.gemv_half_indep(z, vd, h, dense_off, ib + cd).unwrap();
            bt.gemv_half_indep(bproj, nv, h, dense_off, ib + cd + vd).unwrap();
            bt.gemv_half_indep(a, nv, h, dense_off, ib + cd + vd + nv).unwrap();
            if barrier {
                bt.barrier().unwrap();
            }
            bt.gdn_conv(conv_w, conv_state, ib, cd, kdim, hist).unwrap();
            bt.gdn_rec(params, rec_state, ib, normed, kd, nv, dk, dv, rep, 1e-6).unwrap();
            bt.gemv_half(out, h, vd, normed, dense_off).unwrap();
            if submit {
                bt.run().unwrap();
            }
        })
    };
    let fused = layer(true, true);
    let fused_nb = layer(false, true);
    let recorded = layer(true, false);

    let sum: f64 = parts.iter().map(|(_, _, us)| *us).sum();
    let linear_layers = 30; // 40 layers, the fourth of every four is full attention
    println!("\n  mixer shapes on {dev} ({backend_label})");
    println!("  h={h} kd={kd} vd={vd} cd={cd} nv={nv} nk={nk} dk={dk} dv={dv} conv={kdim}");
    println!("  submit+wait floor : {floor:.1} us");
    println!("  record-only layer : {recorded:.1} us");
    for (label, bytes, us) in &parts {
        let busy = (us - floor).max(0.0);
        let gbs = if busy > 0.0 { *bytes as f64 / (busy * 1e-6) / 1e9 } else { 0.0 };
        println!("  {label:<9}: {us:6.1} us  busy {busy:6.1} us  {gbs:5.0} GB/s");
    }
    println!(
        "  (the conv and rec figures count state traffic, not weights; the gemv \
         GB/s is an L2 ceiling - each part replays one buffer {ITERS} times)"
    );
    println!(
        "  DRAM reference    : 64 MiB f16 gemv reads at {:.0} GB/s over random \
         weights, {:.0} GB/s over zeroes",
        busy_gbs(big_bytes, rand_us),
        busy_gbs(big_bytes, zero_us)
    );
    println!(
        "                    : sustained {laps} laps in {secs:.2} s = {:.0} GB/s, \
         burst after it {:.0} GB/s",
        big_bytes as f64 * laps as f64 / secs / 1e9,
        busy_gbs(big_bytes, hot_us)
    );
    println!("  sum of parts      : {sum:.1} us, plus the floor {:.1}", sum + floor);
    println!("  fused, one submit : {fused:.1} us   barrier-free {fused_nb:.1} us");
    let weight_bytes = (cd * h + vd * h + 2 * nv * h + h * vd) * 2;
    // Compared against the *best* ceiling measured above, so "bandwidth-bound"
    // is the conservative reading, not the flattering one.
    let ceiling = busy_gbs(big_bytes, rand_us.min(hot_us));
    println!(
        "  one layer reads {weight_bytes} bytes ({:.0} MiB) of weights in {:.1} us busy \
         = {:.0} GB/s, which is {:.0}% of the best measured ceiling",
        weight_bytes as f64 / (1 << 20) as f64,
        fused - floor,
        busy_gbs(weight_bytes, fused),
        100.0 * busy_gbs(weight_bytes, fused) / ceiling
    );
    println!(
        "  overlap saving    : {:.1} us ({:.0}% of the parts)",
        sum + floor - fused,
        100.0 * (sum + floor - fused) / (sum + floor)
    );
    println!(
        "  per token over {linear_layers} GDN layers: fused {:.2} ms, parts {:.2} ms",
        fused * linear_layers as f64 / 1000.0,
        (sum + floor) * linear_layers as f64 / 1000.0
    );
    println!("  (compare with prof::MIXER_DEV on a served decode: if the two differ\n   by more than the floor, the stage is not one layer's cost.)");
}
/// The MoE stage's kernel shapes, priced against the bandwidth they should get.
///
/// Decode records one `gemv_q4` per picked expert per phase, eight dispatches for
/// each of the two matrix products, in one submit. Two ideas have been priced
/// here, and they went opposite ways:
///
/// * **Collapsing the dispatches lost.** Concatenating the k experts' weight
///   blocks and issuing one gemv is slower than the eight dispatches, at every
///   dispatch count in the sweep - the cost is per output row, not per dispatch.
/// * **Re-decomposing the work inside one dispatch won.** Giving a row eight
///   lanes instead of sixty-four cuts the reduction tree from six barrier steps
///   to three and covers eight rows per workgroup, which is 3.46x on the layer as
///   decode records it (908.7 -> 262.7 us). That kernel is what `gemv_q4` is
///   now; the one-row shape it replaced stays compiled in as
///   `gemv_q4_1row_nb` so this loop can still re-run the comparison that retired
///   it, and nothing in the engine records it.
///
/// Two things had to be sorted out before any of that meant anything:
///
/// * **Timing has to interleave.** The first version of this probe ran each
///   variant to completion, one after another, and the fused variant came out
///   *slower* than the eight dispatches. On a laptop card that is drift, not a
///   result: the clock falls as the GPU heats, so a later variant loses. Every
///   number here comes from one round-robin loop instead, in which each variant
///   is exposed to the same thermal state as the one it is compared with.
/// * **These shapes are small.** One expert's gate_up is 1.18 MB of int4, which
///   at the card's bandwidth is ~5 us. A dispatch that costs tens of us is not
///   bandwidth-bound, so the sweep over the dispatch count is what separates a
///   per-dispatch ramp from per-byte traffic, and an f16 gemv over the same
///   rows is what says how much of the cost is the int4 unpacking itself.
///
/// `gate_up` shares one activation across the k picks, so it *can* collapse into
/// one dispatch - but only with a phase-block arena, because a fused gemv writes
/// `y[y_off + r]` contiguously while the current slot layout strides `4i + h`
/// between experts. `down` cannot collapse with this kernel at all: each pick has
/// its own hidden, and one gemv has one `x`. The `down collapsed` figure is
/// therefore a labelled proxy (same bytes, one dispatch, wrong activations for
/// k-1 of the rows). The gate_up, silu, down and reduce re-layouts are checked
/// for bit-exactness, and the two int4 kernels against each other at two group
/// widths: a lane striding eight words changes scale group every step only when
/// the group is 64 columns, so a single width cannot see a desync.
#[test]
#[ignore = "device timing probe, prints instead of asserting"]
fn moe_kernel_shapes() {
    let Some(mut g) = gpu() else { return };
    const ITERS: usize = 60;

    macro_rules! vk {
        ($bt:expr) => {
            match &mut $bt {
                fennec::gpu::Batch::Vk(v) => v,
                _ => panic!("the barrier-free expert calls are Vulkan-only; run with FENNEC_BACKEND=vulkan"),
            }
        };
    }

    // qwen36-35b's routed expert, read off its config.json: hidden 2048,
    // moe_intermediate 512, top-8 of 256, and the shipped int4 checkpoint groups
    // scales every 64 columns.
    let (h, i, k, group) = (2048usize, 512usize, 8usize, 64usize);
    let (gu, dn) = (2 * i, h);
    let (gu_bytes, gu_scale) = (gu * h / 2, gu * h / group * 4);
    let (dn_bytes, dn_scale) = (dn * i / 2, dn * i / group * 4);
    // Two arenas side by side: the one decode uses today, and the phase-block one
    // a collapse would move to.
    let s0 = (h + 15) & !15;
    let stride = 4 * i + h;
    let n0 = s0 + k * stride;
    let (gate_b, up_b, hh_b, out_b) = (n0, n0 + k * i, n0 + 2 * k * i, n0 + 3 * k * i);
    let w_off = n0 + k * (3 * i + h);
    let acc = (w_off + 16 + 15) & !15;
    g.reserve(acc + h + 16).unwrap();
    g.write(0, &lcg(h, 11, 0.05));
    let wts: Vec<f32> = (0..k).map(|j| 0.1 + 0.01 * j as f32).collect();
    g.write(w_off, &wts);

    // Every phase block of every expert gets its own random bytes and its own
    // random per-group scales. A constant fill makes a wrong column stride or a
    // wrong group index agree with the reference by accident, which is the exact
    // class of bug the two re-decompositions below are most likely to carry.
    fn seed(mat: usize, rows: usize, cols: usize, j: usize, phase: usize) -> u64 {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        for k in [mat, rows, cols, j, phase] {
            s = s.wrapping_mul((k as u64) + 0x1000_0000_0000_01b3).rotate_left(27);
        }
        s | 1
    }
    fn rbytes(n: usize, s: u64) -> Vec<u8> {
        let mut x = s;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 33) as u8
            })
            .collect()
    }
    /// Positive scales spread over a decade: a scale read from the neighbouring
    /// group changes the row by ~30%, not by a rounding unit.
    fn fbytes(n: usize, s: u64) -> Vec<u8> {
        let mut x = s;
        let mut out = Vec::with_capacity(n * 4);
        for _ in 0..n {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let v = 0.25 + 1.5 * ((x >> 40) as f32) / ((1u64 << 24) as f32);
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }
    /// One expert's int4 matrix as `blocks` phase blocks of `rows x cols`, laid
    /// out end to end: gate_up ships two (gate then up), down ships one.
    fn q4m(g: &mut Gpu, mat: usize, blocks: usize, rows: usize, cols: usize, group: usize, j: usize) -> [usize; 2] {
        let (rb, rs) = (rows * cols / 2, rows * cols / group * 4);
        let (w, sc) = (g.alloc_half(rb * blocks, false).unwrap(), g.alloc_half(rs * blocks, false).unwrap());
        for p in 0..blocks {
            let s = seed(mat, rows, cols, j, p);
            g.write_buf(w, p * rb, &rbytes(rb, s)).unwrap();
            g.write_buf(sc, p * rs, &fbytes(rs / 4, s ^ 0x8000_0000_0000_0001)).unwrap();
        }
        [w, sc]
    }
    /// The collapsed gate_up buffer: all k experts' gate rows, then all their up
    /// rows - the row order the phase-block arena writes in, byte-identical to
    /// `q4m(.., 2, i, h, ..)` wherever the two layouts name the same block.
    fn q4_halves(g: &mut Gpu, i: usize, h: usize, group: usize, k: usize) -> [usize; 2] {
        let (rb, rs) = (i * h / 2, i * h / group * 4);
        let (w, sc) = (g.alloc_half(2 * rb * k, false).unwrap(), g.alloc_half(2 * rs * k, false).unwrap());
        for j in 0..k {
            for p in 0..2 {
                let s = seed(0, i, h, j, p);
                g.write_buf(w, (p * k + j) * rb, &rbytes(rb, s)).unwrap();
                g.write_buf(sc, (p * k + j) * rs, &fbytes(rs / 4, s ^ 0x8000_0000_0000_0001)).unwrap();
            }
        }
        [w, sc]
    }
    /// A plain concatenation of k blocks, for the down proxy.
    fn q4_cat(g: &mut Gpu, rows: usize, cols: usize, group: usize, k: usize) -> [usize; 2] {
        let (rb, rs) = (rows * cols / 2, rows * cols / group * 4);
        let (w, sc) = (g.alloc_half(rb * k, false).unwrap(), g.alloc_half(rs * k, false).unwrap());
        for j in 0..k {
            let s = seed(1, rows, cols, j, 0);
            g.write_buf(w, j * rb, &rbytes(rb, s)).unwrap();
            g.write_buf(sc, j * rs, &fbytes(rs / 4, s ^ 0x8000_0000_0000_0001)).unwrap();
        }
        [w, sc]
    }
    /// Zeroed f16 weights over the same rows: no unpacking and one scale per row,
    /// so this is the shape's cost with the int4 decoding taken out of it.
    fn halfw(g: &mut Gpu, rows: usize, cols: usize) -> usize {
        let id = g.alloc_half(rows * cols * 2, true).unwrap();
        g.write_buf(id, 0, &vec![0u8; rows * cols * 2]).unwrap();
        id
    }
    let sep: Vec<[usize; 2]> = (0..k).map(|j| q4m(&mut g, 0, 2, i, h, group, j)).collect();
    let sep_dn: Vec<[usize; 2]> = (0..k).map(|j| q4m(&mut g, 1, 1, dn, i, group, j)).collect();
    let fgu = q4_halves(&mut g, i, h, group, k);
    let fdn = q4_cat(&mut g, dn, i, group, k);
    let hgu = halfw(&mut g, k * gu, h);
    let hdn = halfw(&mut g, k * dn, i);

    // One round-robin loop, so that the variants being compared share a thermal
    // state. A variant run after another one has warmed the card reads slower
    // for reasons that have nothing to do with its shape.
    type V = Vec<(String, Box<dyn FnMut(&mut Gpu)>)>;
    fn rr(g: &mut Gpu, vs: &mut V) -> Vec<(String, f64)> {
        for _ in 0..30 {
            for (_, f) in vs.iter_mut() {
                f(g);
            }
        }
        let mut acc = vec![0f64; vs.len()];
        for _ in 0..ITERS {
            for (x, (_, f)) in vs.iter_mut().enumerate() {
                let t = std::time::Instant::now();
                f(g);
                acc[x] += t.elapsed().as_secs_f64() * 1e6;
            }
        }
        vs.iter().enumerate().map(|(x, (l, _))| (l.clone(), acc[x] / ITERS as f64)).collect()
    }
    let mut vs: V = Vec::new();
    fn add(vs: &mut V, label: &str, f: impl FnMut(&mut Gpu) + 'static) {
        vs.push((label.to_string(), Box::new(f)));
    }

    add(&mut vs, "floor", |g: &mut Gpu| {
        g.batch().unwrap().run().unwrap();
    });
    // The dispatch sweep: m expert gate_ups in one submit, so the only thing
    // changing between the lines is the number of dispatches recorded.
    let (sepc, dnc) = (sep.clone(), sep_dn.clone());
    for m in [1usize, 2, 4, 8] {
        let sep = sepc.clone();
        let tag = format!("gu x{m}");
        add(&mut vs, &tag, move |g: &mut Gpu| {
            let mut bt = g.batch().unwrap();
            for j in 0..m {
                let b = sep[j % k];
                vk!(bt).gemv_q4_nb(b[0], b[1], gu, h, group, 0, s0 + (j % k) * stride).unwrap();
            }
            bt.run().unwrap();
        });
        let sep_dn = dnc.clone();
        let tag = format!("dn x{m}");
        add(&mut vs, &tag, move |g: &mut Gpu| {
            let mut bt = g.batch().unwrap();
            for j in 0..m {
                let b = sep_dn[j % k];
                let s = s0 + (j % k) * stride;
                vk!(bt).gemv_q4_nb(b[0], b[1], dn, i, group, s + 2 * i, s + 4 * i).unwrap();
            }
            bt.run().unwrap();
        });
    }
    // Eight gate_ups in eight command-buffer round trips: what a barrier after
    // every expert used to cost.
    let sep = sepc.clone();
    add(&mut vs, "gu x8 submits", move |g: &mut Gpu| {
        for j in 0..k {
            let b = sep[j];
            let mut bt = g.batch().unwrap();
            vk!(bt).gemv_q4(b[0], b[1], gu, h, group, 0, s0 + j * stride).unwrap();
            bt.run().unwrap();
        }
    });
    // The collapses.
    add(&mut vs, "gu fused", move |g: &mut Gpu| {
        let mut bt = g.batch().unwrap();
        vk!(bt).gemv_q4_nb(fgu[0], fgu[1], k * gu, h, group, 0, gate_b).unwrap();
        bt.run().unwrap();
    });
    add(&mut vs, "dn fused", move |g: &mut Gpu| {
        let mut bt = g.batch().unwrap();
        vk!(bt).gemv_q4_nb(fdn[0], fdn[1], k * dn, i, group, hh_b, out_b).unwrap();
        bt.run().unwrap();
    });
    add(&mut vs, "silu x8", move |g: &mut Gpu| {
        let mut bt = g.batch().unwrap();
        for j in 0..k {
            let s = s0 + j * stride;
            vk!(bt).silu_mul_nb(i, s, s + i, s + 2 * i).unwrap();
        }
        bt.run().unwrap();
    });
    add(&mut vs, "silu fused", move |g: &mut Gpu| {
        let mut bt = g.batch().unwrap();
        vk!(bt).silu_mul_nb(k * i, gate_b, up_b, hh_b).unwrap();
        bt.run().unwrap();
    });
    add(&mut vs, "reduce old", move |g: &mut Gpu| {
        let mut bt = g.batch().unwrap();
        vk!(bt).moe_reduce(k, h, stride, s0, 4 * i, acc, w_off).unwrap();
        bt.run().unwrap();
    });
    add(&mut vs, "reduce new", move |g: &mut Gpu| {
        let mut bt = g.batch().unwrap();
        vk!(bt).moe_reduce(k, h, h, out_b, 0, acc, w_off).unwrap();
        bt.run().unwrap();
    });
    // The same rows in f16: 4x the bytes of the int4 pass, and none of its
    // unpacking. int4 at f16's efficiency would be this figure / 4.
    add(&mut vs, "f16 gu 8", move |g: &mut Gpu| {
        g.batch().unwrap().gemv_half(hgu, k * gu, h, 0, out_b).unwrap().run().unwrap();
    });
    add(&mut vs, "f16 dn 8", move |g: &mut Gpu| {
        g.batch().unwrap().gemv_half(hdn, k * dn, i, hh_b, out_b).unwrap().run().unwrap();
    });
    // The retired shape: one row per workgroup, sixty-four lanes deep, six barrier
    // steps per row. This is the kernel the eight-lane one replaced, kept in the
    // tree only so the 3.46x that justified the swap stays re-measurable here.
    for m in [1usize, 2, 4, 8] {
        let sep = sepc.clone();
        let tag = format!("gu 1r x{m}");
        add(&mut vs, &tag, move |g: &mut Gpu| {
            let mut bt = g.batch().unwrap();
            for j in 0..m {
                let b = sep[j % k];
                vk!(bt).gemv_q4_1row_nb(b[0], b[1], gu, h, group, 0, s0 + (j % k) * stride).unwrap();
            }
            bt.run().unwrap();
        });
        let sep_dn = dnc.clone();
        let tag = format!("dn 1r x{m}");
        add(&mut vs, &tag, move |g: &mut Gpu| {
            let mut bt = g.batch().unwrap();
            for j in 0..m {
                let b = sep_dn[j % k];
                let s = s0 + (j % k) * stride;
                vk!(bt).gemv_q4_1row_nb(b[0], b[1], dn, i, group, s + 2 * i, s + 4 * i).unwrap();
            }
            bt.run().unwrap();
        });
    }
    add(&mut vs, "gu 1r fused", move |g: &mut Gpu| {
        let mut bt = g.batch().unwrap();
        vk!(bt).gemv_q4_1row_nb(fgu[0], fgu[1], k * gu, h, group, 0, gate_b).unwrap();
        bt.run().unwrap();
    });
    add(&mut vs, "dn 1r fused", move |g: &mut Gpu| {
        let mut bt = g.batch().unwrap();
        vk!(bt).gemv_q4_1row_nb(fdn[0], fdn[1], k * dn, i, group, hh_b, out_b).unwrap();
        bt.run().unwrap();
    });
    // The layer, several ways. Modes 0..3 are one kernel: 0 is what decode does,
    // 3 says what the three in-layer barriers cost, and 1 and 2 are the collapse
    // candidates (`ideal` uses the down proxy). Adding 4 to a mode runs it through
    // the retired one-row kernel, so every shape is priced against the same arena
    // and the same submit.
    fn q4v(
        mut bt: &mut fennec::gpu::Batch,
        kern: u8,
        w: usize,
        sc: usize,
        rows: usize,
        cols: usize,
        group: usize,
        x_off: usize,
        y_off: usize,
    ) {
        match kern {
            0 => vk!(bt).gemv_q4_nb(w, sc, rows, cols, group, x_off, y_off).unwrap(),
            _ => vk!(bt).gemv_q4_1row_nb(w, sc, rows, cols, group, x_off, y_off).unwrap(),
        };
    }
    let layer: std::rc::Rc<dyn Fn(u8, &mut Gpu)> = std::rc::Rc::new(move |mode: u8, g: &mut Gpu| {
        let (m, kern) = (mode & 3, (mode >> 2) & 3);
        let mut bt = g.batch().unwrap();
        if m == 0 || m == 3 {
            for j in 0..k {
                let b = sepc[j];
                q4v(&mut bt, kern, b[0], b[1], gu, h, group, 0, s0 + j * stride);
            }
            if m == 0 {
                vk!(bt).barrier().unwrap();
            }
            for j in 0..k {
                let s = s0 + j * stride;
                vk!(bt).silu_mul_nb(i, s, s + i, s + 2 * i).unwrap();
            }
            if m == 0 {
                vk!(bt).barrier().unwrap();
            }
            for j in 0..k {
                let b = dnc[j];
                let s = s0 + j * stride;
                q4v(&mut bt, kern, b[0], b[1], dn, i, group, s + 2 * i, s + 4 * i);
            }
            if m == 0 {
                vk!(bt).barrier().unwrap();
            }
            vk!(bt).moe_reduce(k, h, stride, s0, 4 * i, acc, w_off).unwrap();
        } else {
            q4v(&mut bt, kern, fgu[0], fgu[1], k * gu, h, group, 0, gate_b);
            vk!(bt).barrier().unwrap();
            vk!(bt).silu_mul_nb(k * i, gate_b, up_b, hh_b).unwrap();
            vk!(bt).barrier().unwrap();
            if m == 1 {
                for j in 0..k {
                    let b = dnc[j];
                    q4v(&mut bt, kern, b[0], b[1], dn, i, group, hh_b + j * i, out_b + j * h);
                }
            } else {
                q4v(&mut bt, kern, fdn[0], fdn[1], k * dn, i, group, hh_b, out_b);
            }
            vk!(bt).barrier().unwrap();
            vk!(bt).moe_reduce(k, h, h, out_b, 0, acc, w_off).unwrap();
        }
        bt.run().unwrap();
    });
    for (tag, mode) in [
        ("layer now", 0u8),
        ("layer nobar", 3),
        ("layer merged", 1),
        ("layer ideal", 2),
        ("layer 1r", 4),
    ] {
        let layer = layer.clone();
        add(&mut vs, tag, move |g: &mut Gpu| layer(mode, g));
    }

    let out = rr(&mut g, &mut vs);
    let us = |label: &str| -> f64 {
        out.iter().find(|(l, _)| l == label).map(|(_, x)| *x).unwrap_or(f64::NAN)
    };
    let floor = us("floor");
    let gbs = |bytes: usize, t: f64| -> f64 {
        // The floor is itself a submit-and-wait, so it carries tens of us of
        // noise; a variant that measures under it has no busy time to divide by
        // and the quotient is meaningless rather than large.
        let busy = t - floor;
        if busy < 1.0 {
            f64::NAN
        } else {
            bytes as f64 / (busy * 1e-6) / 1e9
        }
    };

    // Exactness of the re-layout, after the timing: the merged path has to
    // reproduce today's numbers row for row, expert for expert, in the arena the
    // collapse would move to. Old slot j keeps its phases at 0, i, 2i, 4i; the
    // new arena puts expert j's gate at gate_b + j*i, its up at up_b + j*i, its
    // hidden at hh_b + j*i and its output at out_b + j*h.
    layer(0, &mut g);
    let (mut old, mut old_dn) = (vec![0f32; k * stride], Vec::new());
    g.read_into(s0, &mut old);
    for j in 0..k {
        let mut v = vec![0f32; h];
        g.read_into(s0 + j * stride + 4 * i, &mut v);
        old_dn.push(v);
    }
    let mut acc_old = vec![0f32; h];
    g.read_into(acc, &mut acc_old);
    layer(1, &mut g);
    let mut neu = vec![0f32; 3 * k * i];
    g.read_into(gate_b, &mut neu);
    let mut acc_new = vec![0f32; h];
    g.read_into(acc, &mut acc_new);
    for j in 0..k {
        let s = j * stride;
        let (gj, uj, hj) = (j * i, k * i + j * i, 2 * k * i + j * i);
        assert_eq!(&old[s..s + i], &neu[gj..gj + i], "expert {j} gate rows moved");
        assert_eq!(&old[s + i..s + gu], &neu[uj..uj + i], "expert {j} up rows moved");
        assert_eq!(&old[s + 2 * i..s + 3 * i], &neu[hj..hj + i], "expert {j} hidden moved");
        let mut dn_new = vec![0f32; h];
        g.read_into(out_b + j * h, &mut dn_new);
        assert_eq!(&old_dn[j], &dn_new, "expert {j} down output moved");
    }
    assert_eq!(acc_old, acc_new, "the phase-block reduce must weight the same k outputs");
    // The shipped kernel and the one it replaced fold the same row differently, so
    // they are compared to tolerance - the same 1e-5 the device tier uses
    // everywhere else in this file - not bit for bit.
    layer(4, &mut g);
    let mut one_row_acc = vec![0f32; h];
    g.read_into(acc, &mut one_row_acc);
    let drift = rel(&acc_old, &one_row_acc);
    assert!(drift < 1e-5, "the one-row int4 kernel drifted {drift:e} from the shipped gemv_q4");
    // Row for row, at two group widths. A lane that strides eight words per step
    // meets a new scale group every step only when the group is 64 columns wide;
    // at any other width its group index desyncs, and one width cannot show it.
    let chk = acc + h + 16;
    g.reserve(chk + gu).unwrap();
    for grp in [32usize, group] {
        let b = q4m(&mut g, 2, 1, gu, h, grp, 3);
        let run = |g: &mut Gpu, kern: u8| {
            let mut bt = g.batch().unwrap();
            q4v(&mut bt, kern, b[0], b[1], gu, h, grp, 0, chk);
            bt.run().unwrap();
        };
        run(&mut g, 1);
        let mut want = vec![0f32; gu];
        g.read_into(chk, &mut want);
        run(&mut g, 0);
        let mut got = vec![0f32; gu];
        g.read_into(chk, &mut got);
        let d = rel(&want, &got);
        assert!(d < 1e-5, "gemv_q4 at group {grp} drifted {d:e} from the one-row reference over {gu} rows");
    }

    let (dev, be) = (g.name().to_string(), g.backend().label().to_string());
    println!("\n  moe shapes on {dev} ({be})");
    println!(
        "  h={h} i={i} k={k} group={group}: gate_up {gu_bytes} B + {gu_scale} B scales per expert, down {dn_bytes} B + {dn_scale} B"
    );
    println!("  floor: one submit, nothing recorded: {floor:.1} us");
    println!("  --- gate_up: {k} x ({gu} rows x {h} cols)");
    for m in [1usize, 2, 4, 8] {
        let t = us(&format!("gu x{m}"));
        let marginal = if m == 1 { t - floor } else { (t - us("gu x1")) / (m - 1) as f64 };
        println!(
            "    {m:>1} dispatch{} {:7.1} us  {:5.0} GB/s  marginal {marginal:6.1} us/dispatch",
            if m == 1 { "" } else { "es" },
            t,
            gbs(m * (gu_bytes + gu_scale), t)
        );
    }
    println!(
        "    1 dispatch   {:7.1} us  {:5.0} GB/s   ({} rows, exact)",
        us("gu fused"),
        gbs(k * (gu_bytes + gu_scale), us("gu fused")),
        k * gu
    );
    println!(
        "    8 submits    {:7.1} us  {:5.0} GB/s",
        us("gu x8 submits"),
        gbs(k * (gu_bytes + gu_scale), us("gu x8 submits"))
    );
    println!(
        "    f16 same rows {:6.1} us  {:5.0} GB/s   ({} MB, no unpacking: int4 at this rate is {:.1} us)",
        us("f16 gu 8"),
        gbs(k * gu * h * 2, us("f16 gu 8")),
        k * gu * h * 2 / 1_000_000,
        us("f16 gu 8") / 4.0
    );
    println!("  --- down: {k} x ({dn} rows x {i} cols)");
    for m in [1usize, 2, 4, 8] {
        let t = us(&format!("dn x{m}"));
        let marginal = if m == 1 { t - floor } else { (t - us("dn x1")) / (m - 1) as f64 };
        println!(
            "    {m:>1} dispatch{} {:7.1} us  {:5.0} GB/s  marginal {marginal:6.1} us/dispatch",
            if m == 1 { "" } else { "es" },
            t,
            gbs(m * (dn_bytes + dn_scale), t)
        );
    }
    println!(
        "    1 dispatch   {:7.1} us  {:5.0} GB/s   (PROXY: one shared x)",
        us("dn fused"),
        gbs(k * (dn_bytes + dn_scale), us("dn fused"))
    );
    println!(
        "    f16 same rows {:6.1} us  {:5.0} GB/s   ({} MB, no unpacking: int4 at this rate is {:.1} us)",
        us("f16 dn 8"),
        gbs(k * dn * i * 2, us("f16 dn 8")),
        k * dn * i * 2 / 1_000_000,
        us("f16 dn 8") / 4.0
    );
    println!(
        "  --- silu {:.1} -> {:.1} us, reduce {:.1} -> {:.1} us (both exact)",
        us("silu x8"), us("silu fused"), us("reduce old"), us("reduce new")
    );
    println!("  --- retired shape: one row per workgroup, sixty-four lanes, six tree steps");
    for m in [1usize, 2, 4, 8] {
        let (a, b) = (us(&format!("gu 1r x{m}")), us(&format!("gu x{m}")));
        let (c, d) = (us(&format!("dn 1r x{m}")), us(&format!("dn x{m}")));
        println!(
            "    {m} dispatch{}  gate_up {a:7.1} -> {b:7.1} us ({:4.2}x)  {:5.0} GB/s   down {c:7.1} -> {d:7.1} us ({:4.2}x) {:5.0} GB/s",
            if m == 1 { "" } else { "es" },
            a / b,
            gbs(m * (gu_bytes + gu_scale), b),
            c / d,
            gbs(m * (dn_bytes + dn_scale), d)
        );
    }
    println!(
        "    fused       gate_up {:7.1} -> {:7.1} us   down (proxy) {:7.1} -> {:7.1} us",
        us("gu 1r fused"), us("gu fused"), us("dn 1r fused"), us("dn fused")
    );
    let layers = 40;
    println!("  --- layer, as decode has it  : {:7.1} us", us("layer now"));
    println!(
        "      without its 3 barriers : {:7.1} us   {:.1} us of barriers",
        us("layer nobar"),
        us("layer now") - us("layer nobar")
    );
    println!(
        "      gate_up + silu merged  : {:7.1} us   {:+7.1} us/layer (exact)",
        us("layer merged"),
        us("layer merged") - us("layer now")
    );
    println!(
        "      everything merged      : {:7.1} us   {:+7.1} us/layer (down proxy)"
        , us("layer ideal"), us("layer ideal") - us("layer now")
    );
    println!(
        "      one row per workgroup  : {:7.1} us   the kernel gemv_q4 replaced, {:4.2}x slower",
        us("layer 1r"),
        us("layer 1r") / us("layer now")
    );
    println!(
        "    per token, {layers} layers: {:.2} ms with one row per workgroup -> {:.2} ms as shipped",
        us("layer 1r") * layers as f64 / 1000.0,
        us("layer now") * layers as f64 / 1000.0
    );
    println!(
        "      the collapses, on top  : {:.2} (merged) -> {:.2} ms (down proxy)",
        us("layer merged") * layers as f64 / 1000.0,
        us("layer ideal") * layers as f64 / 1000.0
    );
    println!("  (compare with MOE_CPU + MOE_SYNC + MOE_READ on a served decode)");
}

/// The batch kernels the device prefill runs, diffed against the single-token
/// kernels decode runs (and the host reference for the same shape). Nothing had
/// ever dispatched these before: `serve` never reached the device prefill, and
/// the one caller that did sat behind `#[ignore]`, so the first real run of
/// `gemv_half_batch` was the stack test below.
#[test]
fn the_batch_kernels_are_the_single_ones() {
    let Some(mut g) = gpu() else { return };
    let threads = 8;

    // rows x cols, over n tokens. 260 000 rows is 65 000 workgroups, past the
    // 65 535 the spec guarantees, so the last shape also walks the row_base split.
    for (rows, cols, n) in [
        (300usize, 2048usize, 4usize),
        (8192, 2048, 6),
        (64, 512, 2),
        (260_000, 64, 3),
    ] {
        let (yb, xs) = (n * cols, lcg(n * cols, 11, 1.0));
        for f16 in [false, true] {
            let w = if f16 { f16_bytes(&lcg(rows * cols, 7, 0.05)) } else { bf16_bytes(&lcg(rows * cols, 7, 0.05)) };
            let mut want = vec![0f32; n * rows];
            let fmt = if f16 { kernel::Half::F16 } else { kernel::Half::Bf16 };
            kernel::gemv_half_batch(&w, rows, cols, &xs, &mut want, threads, fmt);
            let wb = if f16 { g.upload_half(&w, true).unwrap() } else { g.upload(&w).unwrap() };
            g.reserve(yb + n * rows).unwrap();
            g.write(0, &xs);
            g.batch()
                .unwrap()
                .gemv_half_batch(wb, rows, cols, n, 0, yb)
                .unwrap()
                .run()
                .unwrap();
            let got = g.read(yb, n * rows);
            close(&format!("{} batch {rows}x{cols} over {n} tokens", if f16 { "f16" } else { "bf16" }), &want, &got, 2e-6);
        }
    }

    // Batched RMSNorm: one workgroup per token, gains read out of the arena the
    // way the prefill stages them.
    let (h, n, eps) = (2048usize, 5usize, 1e-6f32);
    let xs = lcg(n * h, 17, 1.0);
    let wt = lcg(h, 19, 0.1);
    let mut want = vec![0f32; n * h];
    for t in 0..n {
        kernel::rmsnorm(&xs[t * h..(t + 1) * h], &wt, eps, &mut want[t * h..(t + 1) * h]);
    }
    g.reserve(3 * n * h + h).unwrap();
    g.write(0, &xs);
    g.write(3 * n * h, &wt);
    g.batch().unwrap().rmsnorm_batch(h, n, 0, 3 * n * h, n * h, eps).unwrap().run().unwrap();
    close("rmsnorm batch", &want, &g.read(n * h, n * h), 2e-6);

    // Batched axpy over the whole chunk, the residual add the layer runs twice.
    let add = lcg(n * h, 23, 0.5);
    let mut want = xs.clone();
    for t in 0..n {
        kernel::axpy(&mut want[t * h..(t + 1) * h], &add[t * h..(t + 1) * h], 0.75);
    }
    g.write(0, &xs);
    g.write(2 * n * h, &add);
    g.batch().unwrap().axpy_batch(h, n, 2 * n * h, 0, 0.75, false).unwrap().run().unwrap();
    close("axpy batch", &want, &g.read(0, n * h), 2e-6);
}
/// The served prompt path on the device tier, diffed against the same stack
/// walked one token at a time - the loop `serve` used before this existed. The
/// two read each dense projection a different number of times, so this is what
/// says the batching costs nothing in answers. Each block is also checked on
/// its own, and the mixer states are compared next to the hidden states: a
/// chunk that mis-handles the conv window or the KV row can still hand back a
/// plausible last token and only go wrong on the first decode.
#[test]
fn the_batched_device_prefill_is_the_token_loop() {
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
    let gpu = Arc::new(Mutex::new(g));

    let toks = [15059usize, 9948, 271, 1024, 311, 4096];
    // Layer 0 is a gated delta net, layer 3 a full attention block.
    for layers in [&[0usize][..], &[3][..], &[0, 3][..]] {
        let plan = Plan {
            layers: layers
                .iter()
                .map(|&l| LayerPlan {
                    layer: l,
                    resident: Vec::new(),
                            pooled: all.clone(),
                    streamed: Vec::new(),
                    bytes_expert: 0,
                    dense_bytes: 0,
                    dense_on_device: true,
                })
                .collect(),
            device_bytes_used: 0,
            device_bytes_budget: 0,
            host_bytes_used: 0,
            host_bytes_budget: 0,
            host_dense_bytes: 0,
            device_hit_rate: 0.0,
                host_hit_rate: 0.0,
            disk_rate: 0.0,
            expert_bytes: 0,
            dense_bytes_layer: 0,
            ms_per_token: 0.0,
            lm_head_ms: 0.0,
            moe_device_ms: 0.0,
            moe_host_ms: 0.0,
            overlap_credit_ms: 0.0,
            };
        let dev = Model::load_plan(&sh, cfg.clone(), &plan, group, threads, 8, 0, Some(gpu.clone()))
            .expect("device stack");
        assert!(dev.prefill_device_ready(), "a device plan can batch its prompt on the device");
        assert!(!dev.prefill_ready(), "a device plan has no batched host prompt path to fall back on");

        let mut emb = vec![0f32; toks.len() * h];
        for (t, &id) in toks.iter().enumerate() {
            let x = dev.embed_row(&sh, id).expect("embed");
            emb[t * h..(t + 1) * h].copy_from_slice(&x);
        }
        for n in [1usize, 2, 6] {
            let want = format!("layers {layers:?} chunk {n}");
            let mut chunk = emb[..n * h].to_vec();
            // Batched first: the device holds one mixer state buffer per layer,
            // and the token loop reuses it, so the chunk's own state has to be
            // read out before the reference run overwrites it.
            let mut st_b = dev.states(n);
            dev.prefill_device(&mut chunk, &mut st_b, 0).expect("device prefill");
            let kept: Vec<Vec<f32>> = st_b
                .iter()
                .map(|s| match s {
                    LayerState::Linear(p) => [p.conv.as_slice(), p.rec.as_slice()].concat(),
                    LayerState::Full(a) => [a.k.as_slice(), a.v.as_slice()].concat(),
                })
                .collect();

            let mut stepped = vec![0f32; n * h];
            let mut st_s = dev.states(n);
            for pos in 0..n {
                let mut x = emb[pos * h..(pos + 1) * h].to_vec();
                dev.forward(&mut x, &mut st_s, pos);
                stepped[pos * h..(pos + 1) * h].copy_from_slice(&x);
            }
            // The token loop leaves the delta-net state in VRAM; the chunk keeps
            // it on the host. Pull the reference back before diffing the two.
            for (l, s) in dev.layers.iter().zip(st_s.iter_mut()) {
                if let LayerState::Linear(p) = s {
                    l.gdn.as_ref().unwrap().sync_state(p).expect("read the device state back");
                }
            }
            for t in 0..n {
                close(
                    &format!("{want} token {t} hidden state"),
                    &stepped[t * h..(t + 1) * h],
                    &chunk[t * h..(t + 1) * h],
                    2e-5,
                );
            }
            for (l, (want_, st)) in kept.iter().zip(st_s.iter()).enumerate() {
                let now: Vec<f32> = match st {
                    LayerState::Linear(p) => [p.conv.as_slice(), p.rec.as_slice()].concat(),
                    LayerState::Full(a) => [a.k.as_slice(), a.v.as_slice()].concat(),
                };
                close(&format!("{want} layer {l} mixer state"), want_, &now, 2e-5);
            }
        }
    }

    let g = gpu.lock().unwrap();
    println!(
        "  {} dispatches in {} submits",
        g.dispatches(),
        g.submitted(),
    );
}
