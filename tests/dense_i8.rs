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
