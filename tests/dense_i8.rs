//! The shipped CPU default for the dense trunk: one int8 byte per weight
//! instead of the checkpoint's two. That halves the bytes the mixer
//! projections read per token, which is the difference between a
//! bandwidth-bound decode and a worse one - but it is a *quality trade*, so
//! this file measures the trade instead of asserting it away.
//!
//! It is its own test binary on purpose: `set_dense_i8` is process-wide, and
//! `cargo test` runs the tests inside one binary on parallel threads. Here
//! there is one test, and it sets the flag itself before each load.

use std::env;
use std::path::PathBuf;
use std::sync::Arc;

use fennec::config::TextConfig;
use fennec::kernel;
use fennec::model::{self, Model};
use fennec::npy::Npy;
use fennec::safetensors::Shards;

const DEFAULT: &str = r"D:\Fennec-Suite\downloads\models\MNN-Pocket-0.3B";

fn dir() -> String {
    env::var("FENNEC_DENSE_MODEL").unwrap_or_else(|_| DEFAULT.to_string())
}

fn fx(name: &str) -> Npy {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(format!("{name}.npy"));
    Npy::read(&p.to_string_lossy()).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// Relative RMS difference, the same band the other goldens use: peak-relative
/// numbers are meaningless at these magnitudes.
fn rel(want: &[f64], got: &[f32]) -> f64 {
    let (mut ws, mut es) = (0f64, 0f64);
    for (w, g) in want.iter().zip(got) {
        let e = w - *g as f64;
        ws += w * w;
        es += e * e;
    }
    es.sqrt() / ws.sqrt().max(1e-30)
}

/// Every token through the whole dense stack under one setting of the flag,
/// with the lm_head logits of every position kept: the last one is the one the
/// goldens diff against transformers, the earlier ones say how many decisions
/// the quantisation actually reaches.
fn walk(sh: &Arc<Shards>, c: &Arc<TextConfig>, i8: bool) -> Vec<f32> {
    model::set_dense_i8(i8);
    let ids = fx("q3.ids");
    let (n, h) = (ids.data.len(), c.hidden_size);
    let m = Model::load(sh, c.clone(), 0, c.num_hidden_layers, 32, 4, n).expect("load dense stack");
    let mut st = m.states(n);
    let mut x = vec![0f32; h];
    let mut hn = vec![0f32; h];
    let mut lg = vec![0f32; c.vocab_size];
    let mut out = Vec::with_capacity(n * c.vocab_size);
    for t in 0..n {
        x = m.embed_row(sh, ids.data[t] as usize).expect("embed row");
        m.forward(&mut x, &mut st, t);
        kernel::rmsnorm(&x, &m.norm, c.rms_norm_eps as f32, &mut hn);
        m.logits(sh, &hn, &mut lg).expect("dense logits");
        out.extend_from_slice(&lg);
    }
    out
}

#[test]
fn the_int8_dense_trunk_keeps_the_next_token_and_stays_in_band() {
    let d = dir();
    if !std::path::Path::new(&d).is_dir() {
        println!("  {d} is not on this machine: skipped");
        return;
    }
    let mut cfg = TextConfig::from_file(&format!("{d}/config.json")).expect("config");
    let sh = Arc::new(Shards::open(&d).expect("checkpoint"));
    model::bind_prefix(&sh, &mut cfg);
    let c = Arc::new(cfg);
    let vocab = c.vocab_size;

    let exact = walk(&sh, &c, false);
    let quant = walk(&sh, &c, true);
    model::set_dense_i8(false);
    let n = quant.len() / vocab;

    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |a, (i, &x)| if x > a.1 { (i, x) } else { a })
            .0
    };
    let lg = fx("q3.logits");
    let want = &lg.data[(lg.data.len() - vocab)..];
    let ref_idx = fx("q3.logits_idx").data[0] as usize;
    let ex64: Vec<f64> = exact.iter().map(|v| *v as f64).collect();
    let same: Vec<bool> = (0..n)
        .map(|t| {
            argmax(&exact[t * vocab..(t + 1) * vocab]) == argmax(&quant[t * vocab..(t + 1) * vocab])
        })
        .collect();
    let kept = same.iter().filter(|s| **s).count();
    let worst = (0..n)
        .map(|t| rel(&ex64[t * vocab..(t + 1) * vocab], &quant[t * vocab..(t + 1) * vocab]))
        .fold(0f64, f64::max);
    println!(
        "  int8 dense trunk over {n} positions: {kept}/{n} greedy tokens unchanged, \
         worst {:.2}% rms against bf16 | against the transformers reference int8 {:.2}%, \
         bf16 {:.2}% | last greedy bf16 {} int8 {}, reference {ref_idx}",
        100.0 * worst,
        100.0 * rel(want, &quant[(n - 1) * vocab..n * vocab].to_vec()),
        100.0 * rel(want, &exact[(n - 1) * vocab..n * vocab].to_vec()),
        argmax(&exact[(n - 1) * vocab..n * vocab]),
        argmax(&quant[(n - 1) * vocab..n * vocab]),
    );
    assert_eq!(
        argmax(&exact[(n - 1) * vocab..n * vocab]), ref_idx,
        "the exact path stopped matching transformers"
    );
    assert_eq!(
        argmax(&quant[(n - 1) * vocab..n * vocab]), ref_idx,
        "the int8 default changed the token the golden scores"
    );
    // The band is a measured one, not a wish. Per-row int8 costs about one
    // part in sqrt(cols) of the dot, because the weights agree to 0.4% each
    // while the sum cancels: that is the 12% above. It is also the reason this
    // test reports the greedy flips instead of asserting none - the same flag
    // on the shipped 35B moves the greedy text within 64 tokens of the matrix
    // prompt. The default is a deliberate trade against a ~19% decode win, and
    // Colibri ships the same one, so what is pinned here is the size of the
    // trade rather than the claim that it is free.
    assert!(
        rel(want, &quant[(n - 1) * vocab..n * vocab].to_vec()) < 0.15,
        "int8 moved the last logits past the measured band"
    );
}

/// The CPU lane's `FENNEC_PROFILE` breakdown says the mixer projections read
/// 1.28 GB of int8 in 38.6 ms — 33 GB/s — while the *same kernel* on the
/// lm_head reads 508 MB in 9.7 ms, 52 GB/s. One of those two numbers is not a
/// property of `gemv_i8`, so this probe separates the two remaining causes
/// without loading the 35B checkpoint: dispatch granularity (a GDN layer calls
/// the kernel three times, so each worker gets ~1.5 MB and then waits at a
/// barrier) against interleaving (40 different matrices in a token versus one
/// continuous stream). Same bytes, one call vs three, is the deciding row; if
/// it is a tie, the shortfall is the interleaving and only the model loop can
/// fix it. `#86` reverted a projection fuse at -14.4%, which is why this
/// measures the shape before anything touches `GdnLayer::step`.
#[test]
#[ignore = "throughput probe: needs a quiet box and ~1.5 GB of synthetic weights"]
fn gemv_i8_throughput_at_the_real_projection_shapes() {
    const H: usize = 2048;
    const CD: usize = 8192;
    const VD: usize = 4096;
    // The pool is a `OnceLock`, so one process measures one pool size: the
    // sweep is one run per width. 6 is the width that gets the pool pinned
    // one worker per performance core; 8 is what the CPU lane runs by
    // default and stays unpinned on this box.
    let workers: usize = env::var("PROBE_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    kernel::set_threads(workers);

    // Deterministic pseudo-weights with the magnitudes a trained row has: the
    // kernel's cost is bytes, so the values only have to keep `wsum` honest.
    fn mat(rows: usize, cols: usize) -> (Vec<i8>, Vec<i32>, Vec<f32>) {
        let mut st = 0x9E3779B97F4A7C15u64 ^ (rows * 2654435761) as u64;
        let mut w = Vec::with_capacity(rows * cols);
        for _ in 0..rows * cols {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            w.push(((st >> 33) as u8) as i8);
        }
        let wsum = (0..rows)
            .map(|r| w[r * cols..(r + 1) * cols].iter().map(|&v| v as i32).sum())
            .collect();
        let scale = vec![0.008f32; rows];
        (w, wsum, scale)
    }

    let run = |m: &(Vec<i8>, Vec<i32>, Vec<f32>), rows: usize, cols: usize, x: &[f32], out: &mut [f32]| {
        kernel::gemv_i8(&m.0, &m.1, &m.2, rows, cols, x, out, workers);
    };

    // ~50 ms of one shape, so a Windows timer tick is not the measurement.
    let timed = |m: &(Vec<i8>, Vec<i32>, Vec<f32>), rows: usize, cols: usize, x: &[f32], out: &mut [f32]| -> f64 {
        let reps = (2_500_000_000usize / (rows * cols)).clamp(4, 400);
        run(m, rows, cols, x, out);
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            run(m, rows, cols, x, out);
        }
        t0.elapsed().as_secs_f64() * 1e3 / reps as f64
    };

    const GIB: f64 = (1u64 << 30) as f64;
    const MIB: f64 = (1u64 << 20) as f64;
    let gib = |bytes: f64, ms: f64| bytes / GIB / (ms / 1e3);
    let x: Vec<f32> = (0..H).map(|i| ((i as f32 * 0.37).sin()) * 0.8).collect();
    let xv: Vec<f32> = (0..VD).map(|i| ((i as f32 * 0.21).sin()) * 0.8).collect();

    // The real shapes, alone.
    let shapes: [(&str, usize, usize); 6] = [
        ("gdn in_qkv", CD, H),
        ("gdn in_z", VD, H),
        ("gdn out_proj", H, VD),
        ("attn q_proj", CD, H),
        ("attn k_proj", 512, H),
        ("attn o_proj", H, VD),
    ];
    println!("  gemv_i8 alone, one dispatch of each shape, {workers} workers:");
    for (name, rows, cols) in shapes {
        let m = mat(rows, cols);
        let mut out = vec![0f32; rows];
        let ms = timed(&m, rows, cols, if cols == H { &x } else { &xv }, &mut out);
        println!(
            "    {name:<14} {rows:>6}x{cols:<6} {ms:7.3} ms  {:5.1} GiB/s",
            gib((rows * cols) as f64, ms)
        );
    }

    // lm_head, for the rate the profiler credits at 52 GB/s.
    let head = mat(248320, H);
    let mut hout = vec![0f32; 248320];
    let hms = timed(&head, 248320, H, &x, &mut hout);
    println!(
        "    {:<14} {:>6}x{:<6} {hms:7.3} ms  {:5.1} GiB/s",
        "lm_head",
        248320,
        H,
        gib(248320.0 * H as f64, hms)
    );

    // Same bytes, one dispatch vs two: the fuse question in isolation.
    let big = mat(CD + VD, H);
    let a = mat(CD, H);
    let b = mat(VD, H);
    let mut o12 = vec![0f32; CD + VD];
    let mut oa = vec![0f32; CD];
    let mut ob = vec![0f32; VD];
    run(&big, CD + VD, H, &x, &mut o12);
    let t0 = std::time::Instant::now();
    for _ in 0..200 {
        run(&big, CD + VD, H, &x, &mut o12);
    }
    let one = t0.elapsed().as_secs_f64() * 1e3 / 200.0;
    run(&a, CD, H, &x, &mut oa);
    let t0 = std::time::Instant::now();
    for _ in 0..200 {
        run(&a, CD, H, &x, &mut oa);
        run(&b, VD, H, &x, &mut ob);
    }
    let two = t0.elapsed().as_secs_f64() * 1e3 / 200.0;
    let bytes = (CD + VD) as f64 * H as f64;
    println!(
        "  same {:.1} MiB, one call vs two: {one:.3} ms ({:.1} GiB/s) | {two:.3} ms ({:.1} GiB/s)",
        bytes / MIB,
        gib(bytes, one),
        gib(bytes, two)
    );

    // The whole token's worth of projections, in layer order, with every
    // layer's own matrices: this is what the engine actually does.
    let gdn: Vec<_> = (0..30).map(|_| (mat(CD, H), mat(VD, H), mat(H, VD))).collect();
    let attn: Vec<_> = (0..10).map(|_| (mat(CD, H), mat(512, H), mat(H, VD))).collect();
    let tok_bytes = 30.0 * ((CD + VD) as f64 * H as f64 + (H * VD) as f64)
        + 10.0 * ((CD + 512) as f64 * H as f64 + (H * VD) as f64);
    let mut q = vec![0f32; CD];
    let mut z = vec![0f32; VD];
    let mut sm = vec![0f32; H];
    let mut k = vec![0f32; 512];
    let mut seq_ms = 0f64;
    for _ in 0..8 {
        let t0 = std::time::Instant::now();
        for (in_qkv, in_z, out_proj) in &gdn {
            run(in_qkv, CD, H, &x, &mut q);
            run(in_z, VD, H, &x, &mut z);
            run(out_proj, H, VD, &xv, &mut sm);
        }
        for (qo, ko, oo) in &attn {
            run(qo, CD, H, &x, &mut q);
            run(ko, 512, H, &x, &mut k);
            run(oo, H, VD, &xv, &mut sm);
        }
        seq_ms += t0.elapsed().as_secs_f64() * 1e3;
    }
    seq_ms /= 8.0;
    println!(
        "  all 40 layers' projections in order: {seq_ms:.2} ms/token over {:.2} GiB = {:.1} GiB/s (profiler: 38.57 ms at 33 GiB/s)",
        tok_bytes / GIB,
        gib(tok_bytes, seq_ms)
    );
    assert!(seq_ms > 0.0 && one > 0.0 && hms > 0.0, "the probe measured nothing");
}
