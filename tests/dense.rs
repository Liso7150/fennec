//! Goldens for the dense Qwen3 lane: the same diff-as-debugging method as
//! `golden.rs`, against fixtures that HuggingFace transformers' own `qwen3`
//! modules produced in fp32 from the real MNN-Pocket-0.3B weights
//! (tools/make_goldens_qwen3.py, seed 20261113).
//!
//! The engine reads this checkpoint's dense projections as raw bf16 and upcasts
//! per dot, and the reference upcasts the same bytes to fp32, so the weights
//! agree bit for bit: what is left between them is accumulation order, which
//! belongs in the 1e-6 band rather than a quantisation band.

use std::env;
use std::path::PathBuf;
use std::sync::Arc;

use fennec::config::{LayerKind, TextConfig};
use fennec::kernel;
use fennec::model::{self, LayerState, Model};
use fennec::npy::Npy;
use fennec::safetensors::Shards;

const DEFAULT: &str = r"D:\Fennec-Suite\downloads\models\MNN-Pocket-0.3B";

fn dir() -> String {
    env::var("FENNEC_DENSE_MODEL").unwrap_or_else(|_| DEFAULT.to_string())
}

fn fxpath(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(format!("{name}.npy"))
}

fn fx(name: &str) -> Npy {
    let p = fxpath(name);
    Npy::read(&p.to_string_lossy()).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn row(a: &[f64], r: usize, cols: usize) -> Vec<f64> {
    a[r * cols..(r + 1) * cols].to_vec()
}

/// Relative RMS difference: peak-relative numbers are meaningless at these
/// magnitudes, and one outlier element would dominate a max/max ratio.
fn check(label: &str, want: &[f64], got: &[f32], tol: f64) {
    let (mut ws, mut es, mut d, mut at) = (0f64, 0f64, 0f64, 0usize);
    for (i, (w, g)) in want.iter().zip(got).enumerate() {
        let e = w - *g as f64;
        ws += w * w;
        es += e * e;
        if e.abs() >= d {
            d = e.abs();
            at = i;
        }
    }
    let rel = es.sqrt() / ws.sqrt().max(1e-30);
    println!(
        "  {label:<30} n={:<8} rel={:.2e}  max|d|={:.2e}@{at}",
        got.len(),
        rel,
        d,
    );
    assert!(
        rel <= tol,
        "{label}: relative rms error {rel:.3e} exceeds {tol:.1e} (max|d| {d:.3e} at {at}: \
         ref {} vs got {})",
        want[at],
        got[at],
    );
}

/// The export under test, with its own config: the point of this lane is that
/// nothing about it is assumed from the MoE checkpoint.
fn ckpt() -> Option<(Arc<Shards>, Arc<TextConfig>)> {
    let d = dir();
    if !std::path::Path::new(&d).is_dir() {
        println!("  {d} is not on this machine: skipped");
        return None;
    }
    let mut cfg = TextConfig::from_file(&format!("{d}/config.json"))
        .inspect_err(|e| println!("  config unreadable: {e}"))
        .ok()?;
    let sh = match Shards::open(&d) {
        Ok(s) => Arc::new(s),
        Err(e) => {
            println!("  checkpoint unreadable: {e}");
            return None;
        }
    };
    model::bind_prefix(&sh, &mut cfg);
    Some((sh, Arc::new(cfg)))
}

#[test]
fn a_dense_checkpoint_is_read_as_a_dense_checkpoint() {
    let Some((sh, c)) = ckpt() else { return };
    assert!(!c.is_moe(), "num_experts {} should be 0", c.num_experts);
    assert!(!c.has_gdn(), "no gated-delta-net heads in a qwen3 export");
    assert_eq!(c.prefix, "model", "this export stores the stack under `model.`");
    assert!(c.tie_word_embeddings);
    assert_eq!(c.intermediate_size, 2560);
    assert_eq!(c.partial_rotary_factor, 1.0, "qwen3 rotates the full head");
    assert_eq!(c.rope_theta, 1e6, "theta is a top-level key in this export");
    assert!(
        c.layer_types.iter().all(|k| matches!(k, LayerKind::Full)),
        "every qwen3 layer is full attention"
    );
    let m = Model::load(&sh, c.clone(), 0, c.num_hidden_layers, 32, 4, 8).expect("load dense stack");
    assert!(m.layers[0].mlp.is_some() && m.layers[0].moe.is_none());
    assert!(m.layers[0].gdn.is_none() && m.layers[0].attn.is_some());
    // the params figure is what a reader uses to size the model, so it has to
    // account for every byte the export stores: a tied checkpoint still writes
    // the head twice, and the norm weights are the only tensors `total_params`
    // leaves out
    let want = c.total_params();
    let got: usize = sh.names().iter().map(|n| sh.info(n).unwrap().elems()).sum();
    let dup = if c.tie_word_embeddings {
        c.hidden_size * c.vocab_size
    } else {
        0
    };
    let norms = c.num_hidden_layers * (2 * c.head_dim + 2 * c.hidden_size) + c.hidden_size;
    assert_eq!(
        want + dup + norms,
        got,
        "total_params, plus the second copy of a tied head and the norm weights, is the file"
    );
}

/// Step 0 of a full-attention block is the mixer with its algebra removed: one
/// key, so softmax puts all the weight on that row and the context is exactly
/// `repeat_kv` of the value projection. Every stage of the block then has both
/// a fixture and a Rust counterpart, and a diff that fails names the stage
/// rather than the block.
#[test]
fn layer_zero_step_zero_walks_stage_by_stage() {
    let Some((sh, c)) = ckpt() else { return };
    let eps = c.rms_norm_eps as f32;
    let (h, nh, nkv, hd, inter) = (
        c.hidden_size,
        c.num_attention_heads,
        c.num_key_value_heads,
        c.head_dim,
        c.intermediate_size,
    );
    let rep = nh / nkv;
    let m = Model::load(&sh, c.clone(), 0, 1, 32, 4, 1).expect("load one dense layer");
    let lay = &m.layers[0];
    let a = lay.attn.as_ref().expect("full attention");
    let ff = lay.mlp.as_ref().expect("dense mlp");
    let one = |key: &str, cols: usize| row(&fx(&format!("q3.{key}.0")).data, 0, cols);

    let ids = fx("q3.ids");
    let x = m.embed_row(&sh, ids.data[0] as usize).expect("embed row");
    let mut buf = vec![0f32; h];
    kernel::rmsnorm(&x, &lay.norm_in, eps, &mut buf);
    check("input_layernorm", &one("n0", h), &buf, 1e-5);

    let (mut qr, mut kr, mut vr) = (vec![0f32; nh * hd], vec![0f32; nkv * hd], vec![0f32; nkv * hd]);
    a.q.gemv(nh * hd, h, &buf, &mut qr, 1);
    a.k.gemv(nkv * hd, h, &buf, &mut kr, 1);
    a.v.gemv(nkv * hd, h, &buf, &mut vr, 1);
    check("q_proj", &one("q0", nh * hd), &qr, 1e-5);
    check("k_proj", &one("k0", nkv * hd), &kr, 1e-5);
    check("v_proj", &one("v0", nkv * hd), &vr, 1e-5);

    // the per-head norms are the Qwen3 addition over the older families, and
    // they are the only stage that could pass while the block still fails: at
    // position 0 rope is cos=1, sin=0, so it is the identity and cannot hide
    // a wrong rotation here
    let mut qn = vec![0f32; nh * hd];
    for i in 0..nh {
        kernel::rmsnorm(&qr[i * hd..(i + 1) * hd], &a.q_norm, eps, &mut qn[i * hd..(i + 1) * hd]);
    }
    check("q_norm", &one("qn0", nh * hd), &qn, 1e-5);
    let mut kn = vec![0f32; nkv * hd];
    for i in 0..nkv {
        kernel::rmsnorm(&kr[i * hd..(i + 1) * hd], &a.k_norm, eps, &mut kn[i * hd..(i + 1) * hd]);
    }
    check("k_norm", &one("kn0", nkv * hd), &kn, 1e-5);

    let mut ctx = vec![0f32; nh * hd];
    for i in 0..nh {
        let kv = i / rep;
        ctx[i * hd..(i + 1) * hd].copy_from_slice(&vr[kv * hd..(kv + 1) * hd]);
    }
    check("context = repeat_kv(v)", &one("ctx0", nh * hd), &ctx, 1e-5);
    let mut res = vec![0f32; h];
    a.o.gemv(h, nh * hd, &ctx, &mut res, 1);
    check("o_proj", &one("a0", h), &res, 1e-5);

    // the same block through the code that actually runs: a stage that agrees
    // above but not here means step() reads tensors other than these
    let mut step_out = vec![0f32; h];
    let mut states = m.states(1);
    match &mut states[0] {
        LayerState::Full(s) => a.step(&buf, 0, s, &mut step_out),
        _ => unreachable!(),
    }
    check("AttnLayer::step", &one("a0", h), &step_out, 1e-5);

    let mut post = x.clone();
    kernel::axpy(&mut post, &res, 1.0);
    kernel::rmsnorm(&post, &lay.norm_post, eps, &mut buf);
    check("post_attention_layernorm", &one("np0", h), &buf, 1e-5);

    let (mut g, mut u) = (vec![0f32; inter], vec![0f32; inter]);
    ff.gate.gemv(inter, h, &buf, &mut g, 1);
    ff.up.gemv(inter, h, &buf, &mut u, 1);
    check("gate_proj", &one("g0", inter), &g, 1e-5);
    check("up_proj", &one("u0", inter), &u, 1e-5);
    for j in 0..inter {
        u[j] = kernel::silu(g[j]) * u[j];
    }
    check("silu(gate) * up", &one("act0", inter), &u, 1e-5);
    let mut mo = vec![0f32; h];
    ff.down.gemv(h, inter, &u, &mut mo, 1);
    check("down_proj", &one("m0", h), &mo, 1e-5);
}

#[test]
fn eight_tokens_walk_the_dense_stack_to_the_lm_head() {
    let Some((sh, c)) = ckpt() else { return };
    let n_layers = c.num_hidden_layers;
    let eps = c.rms_norm_eps as f32;
    let ids = fx("q3.ids");
    let (n, h) = (ids.data.len(), c.hidden_size);
    let m = Model::load(&sh, c.clone(), 0, n_layers, 32, 4, n).expect("load dense stack");
    let mut states = m.states(n);

    let e = fx("q3.embed");
    for t in 0..n {
        let got = m.embed_row(&sh, ids.data[t] as usize).expect("embed row");
        check(&format!("embed tok {t}"), &row(&e.data, t, h), &got, 1e-6);
    }

    let mut end = vec![0f32; h];
    for t in 0..n {
        let mut x = m.embed_row(&sh, ids.data[t] as usize).expect("embed row");
        for l in 0..n_layers {
            let lay = &m.layers[l];
            if t > 0 {
                lay.forward(&mut x, &mut states[l], t);
            } else {
                // layer 0's blocks each on their own, fed the reference's own
                // input, so a failure names the block rather than the layer it
                // drifted out of; the cache slot they append is the one the
                // production walk would have taken, so the later steps stay valid
                let (mut buf, mut res) = (vec![0f32; h], vec![0f32; h]);
                kernel::rmsnorm(&x, &lay.norm_in, eps, &mut buf);
                match (&lay.kind, &mut states[l]) {
                    (LayerKind::Full, LayerState::Full(a)) => {
                        lay.attn.as_ref().unwrap().step(&buf, t, a, &mut res)
                    }
                    _ => unreachable!(),
                }
                let a0 = fx(&format!("q3.a0.{l}"));
                check(&format!("layer {l} mixer"), &row(&a0.data, 0, h), &res, 1e-5);
                kernel::axpy(&mut x, &res, 1.0);
                kernel::rmsnorm(&x, &lay.norm_post, eps, &mut buf);
                lay.mlp.as_ref().unwrap().forward(&buf, &mut res);
                let m0 = fx(&format!("q3.m0.{l}"));
                check(&format!("layer {l} mlp"), &row(&m0.data, 0, h), &res, 1e-5);
                kernel::axpy(&mut x, &res, 1.0);
            }
            let want = fx(&format!("q3.qs{t}.{l}"));
            check(
                &format!("stream tok {t} layer {l}"),
                &row(&want.data, 0, h),
                &x,
                1e-5,
            );
        }
        end = x;
    }

    let mut normed = vec![0f32; h];
    kernel::rmsnorm(&end, &m.norm, eps, &mut normed);
    let lg = fx("q3.logits");
    let mut got = vec![0f32; c.vocab_size];
    m.logits(&sh, &normed, &mut got).expect("dense logits");
    check(
        "lm_head logits",
        &row(&lg.data, n - 1, c.vocab_size),
        &got,
        1e-4,
    );
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .fold((0, f32::NEG_INFINITY), |a, (i, &x)| if x > a.1 { (i, x) } else { a })
            .0
    };
    let idx = fx("q3.logits_idx");
    assert_eq!(
        argmax(&got) as f64,
        idx.data[0],
        "the dense stack picked a different next token than transformers"
    );
}

/// Prefill reads every dense projection once per chunk of prompt tokens rather
/// than once per token, which only pays if the answer is the answer the token
/// loop would have produced. So this asserts exact bits, not a tolerance: the
/// goldens above measure against transformers, which accumulates in its own
/// order anyway, and a band that wide would hide a real divergence in the
/// shipped prompt path.
///
/// Chunk sizes of the whole prompt, of one, and of three (so the last chunk is
/// a remainder) cover what `serve` can hand it, including its bookkeeping.
#[test]
fn a_batched_prefill_is_bit_identical_to_the_token_loop() {
    let Some((sh, c)) = ckpt() else { return };
    let n_layers = c.num_hidden_layers;
    let ids = fx("q3.ids");
    let (n, h) = (ids.data.len(), c.hidden_size);
    let m = Model::load(&sh, c.clone(), 0, n_layers, 32, 4, n).expect("load dense stack");
    assert!(m.prefill_ready(), "a host-only stack can batch its projections");

    // Both paths get the same input bytes: Fennec's own gather, not the
    // reference float64 rows rounded a second time.
    let emb: Vec<Vec<f32>> = (0..n)
        .map(|t| m.embed_row(&sh, ids.data[t] as usize).expect("embed row"))
        .collect();

    let mut stepped = vec![0f32; n * h];
    {
        let mut st = m.states(n);
        for t in 0..n {
            let mut x = emb[t].clone();
            m.forward(&mut x, &mut st, t);
            stepped[t * h..(t + 1) * h].copy_from_slice(&x);
        }
    }

    for chunk in [n, 1, 3] {
        let mut st = m.states(n);
        let mut xs = vec![0f32; h * chunk.min(n)];
        let mut seen = 0usize;
        while seen < n {
            let k = chunk.min(n - seen);
            for i in 0..k {
                xs[i * h..(i + 1) * h].copy_from_slice(&emb[seen + i]);
            }
            m.prefill(&mut xs[..k * h], &mut st, seen);
            for i in 0..k {
                let a = &stepped[(seen + i) * h..(seen + i + 1) * h];
                let b = &xs[i * h..(i + 1) * h];
                let differ = a.iter().zip(b).filter(|(x, y)| x != y).count();
                assert_eq!(
                    differ, 0,
                    "chunks of {chunk}, token {}: {differ} of {h} outputs differ from the loop",
                    seen + i
                );
            }
            seen += k;
        }
        println!("  {n} tokens in chunks of {chunk}: every layer bit-identical to the token loop");
    }
}
