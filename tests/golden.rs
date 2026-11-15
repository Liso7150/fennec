//! Golden tests: every kernel in the crate is diffed against a fixture that
//! HuggingFace transformers 5.x produced from the real Qwen3.6-35B-A3B weights
//! in fp32 (tools/make_goldens.py, seed 20261112). If a Rust kernel disagrees
//! with a fixture, the Rust is wrong - the reference is the model definition.
//!
//! Tolerances are split deliberately: a test that compares an int4 path against
//! a torch fp32 path is measuring the *format*, so it gets a wide band plus a
//! second assertion that pins the kernel to its own dequantised weights.

use std::path::PathBuf;
use std::sync::Arc;

use fennec::config::{LayerKind, TextConfig};
use fennec::kernel;
use fennec::model::{
    AttnLayer, AttnState, Dense, GdnLayer, GdnState, LayerState, MoeLayer, MoeParts, Model,
};
use fennec::npy::Npy;
use fennec::pool::Stream;
use fennec::quant::Q4;
use fennec::safetensors::Shards;

const MODEL: &str = r"D:\fennec-models\Qwen3.6-35B-A3B-bf16";

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

fn f32s(a: &[f64]) -> Vec<f32> {
    a.iter().map(|v| *v as f32).collect()
}

fn f64s(a: &[f32]) -> Vec<f64> {
    a.iter().map(|v| *v as f64).collect()
}

fn row(a: &[f64], r: usize, cols: usize) -> Vec<f64> {
    a[r * cols..(r + 1) * cols].to_vec()
}

fn row32(a: &[f64], r: usize, cols: usize) -> Vec<f32> {
    f32s(&row(a, r, cols))
}

fn flat(want: &[f64], got: &[f32]) -> f64 {
    want.iter()
        .zip(got)
        .fold(0f64, |m, (x, y)| m.max((x - *y as f64).abs()))
}

/// Relative RMS difference between two same-length references.
fn rel(a: &[f64], b: &[f64]) -> f64 {
    let (mut d, mut w) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        d += (x - y) * (x - y);
        w += x * x;
    }
    d.sqrt() / w.sqrt().max(1e-30)
}

fn scale_of(a: &[f64]) -> f64 {
    a.iter().fold(0f64, |m, v| m.max(v.abs()))
}

// Set by `checkpoint()` when the export under test is not the one the fixtures
// were generated from. Thread-local because the tests run in parallel and the
// synthetic-fixture tests must keep their tight bands either way.
thread_local! {
    static OTHER_EXPORT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The fixtures are fp32 transformers run on the bf16 export's own weights, so
/// that export reproduces them to accumulation order (13/13 at 1e-6, routing 8/8
/// identical, same greedy token). Another export rounds the same weights
/// differently and what is left is the format's noise floor, not a kernel error.
/// These are the paths that only carry f16 dense rounding: measured over the int4
/// export, embeddings 5.0e-8, mixer 1.2e-2, shared expert 1.7e-2, attention and
/// gated-delta-net 6.0e-3. 3e-2 is that floor with headroom.
fn band(tol: f64) -> f64 {
    if OTHER_EXPORT.get() { tol.max(3e-2) } else { tol }
}

/// Anything that touches a routed expert carries the container's int4 rounding
/// instead, which is a different quantisation grid from the FQ4 the fixtures were
/// generated with. Measured over the whole int4 stack: weight probe rows 1.0e-1
/// (all-zero rows exact), expert activations 1.5e-1, routed sum 2.1e-1, hidden
/// stream 1.1e-1, logits 3.1e-2. The error is unbiased - rms(got) tracks rms(ref)
/// - and silu's negative tail amplifies it, which is why
/// one expert's down row reaches 1.5e-1 while its weights sit at 1.0e-1.
/// Requantising the reference does not reproduce the container, so there is no
/// tighter oracle. 2.5e-1 covers the measured 2.1e-1 and still separates a
/// correctly rounded result from the wrong expert row (rel ~1.4) or a zeroed read
/// (rel 1.0). The greedy-token assertion below is band-independent.
fn qband(tol: f64) -> f64 {
    if OTHER_EXPORT.get() { tol.max(2.5e-1) } else { tol }
}

fn check(label: &str, want: &[f64], got: &[f32], tol: f64) {
    check_at(label, want, got, band(tol));
}

fn check_q(label: &str, want: &[f64], got: &[f32], tol: f64) {
    check_at(label, want, got, qband(tol));
}

/// Error metric: RMS of the difference over RMS of the reference. Peak-relative
/// numbers are meaningless at these magnitudes - the activations here are
/// 1e-4-ish and one outlier element would dominate a max/max ratio.
fn check_at(label: &str, want: &[f64], got: &[f32], tol: f64) {
    let d = flat(want, got);
    let at = want
        .iter()
        .zip(got)
        .position(|(x, y)| (x - *y as f64).abs() >= d)
        .unwrap_or(0);
    let (mut ws, mut es) = (0f64, 0f64);
    for (w, g) in want.iter().zip(got) {
        let e = w - *g as f64;
        ws += w * w;
        es += e * e;
    }
    let rel = es.sqrt() / ws.sqrt().max(1e-30);
    println!(
        "  {label:<32} n={:<8} rel={:.2e}  max|d|={:.2e}@{at}  rms(ref)={:.2e} rms(got)={:.2e}",
        got.len(),
        rel,
        d,
        (ws / want.len() as f64).sqrt(),
        (got.iter().map(|g| (*g as f64) * (*g as f64)).sum::<f64>() / got.len() as f64).sqrt(),
    );
    assert!(
        rel <= tol,
        "{label}: relative rms error {rel:.3e} exceeds {tol:.1e} (max|d| {d:.3e} at {at}: \
         ref {} vs got {})",
        want[at],
        got[at],
    );
}

/// The checkpoint's text config, written out so the cheap tests never need the
/// model directory. Values match config.json exactly.
fn cfg() -> Arc<TextConfig> {
    let types: Vec<&str> = (0..40)
        .map(|l| if l % 4 == 3 { "full_attention" } else { "linear_attention" })
        .collect();
    let raw = serde_json::json!({
        "model_type": "qwen3_5_moe_text",
        "hidden_size": 2048,
        "num_hidden_layers": 40,
        "vocab_size": 248320,
        "rms_norm_eps": 1e-6,
        "layer_types": types,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "partial_rotary_factor": 0.25,
        "attn_output_gate": true,
        "rope_parameters": {"rope_theta": 1e7, "partial_rotary_factor": 0.25},
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
    });
    Arc::new(TextConfig::from_json(&raw).expect("synthetic config"))
}

/// The export under test. Both the bf16 one and the pre-quantised int4 container
/// are the same model, so the same fixtures must hold for either; the int4 one
/// just gets the wider band its format earns.
fn model_dir() -> String {
    std::env::var("FENNEC_GOLDEN_MODEL").unwrap_or_else(|_| MODEL.to_string())
}

fn checkpoint() -> Option<(Arc<Shards>, Arc<TextConfig>)> {
    let dir = model_dir();
    OTHER_EXPORT.set(dir != MODEL);
    if !std::path::Path::new(&dir).is_dir() {
        println!("  {dir} is not on this machine: skipped");
        return None;
    }
    let sh = Shards::open(&dir)
        .inspect_err(|e| println!("  checkpoint unreadable: {e}"))
        .ok()?;
    Some((Arc::new(sh), cfg()))
}

/// The group width belongs to the checkpoint when its experts are already int4.
fn group_of(sh: &Shards, c: &TextConfig) -> usize {
    let prefix = fennec::model::detect_prefix(sh);
    fennec::model::packed_group(sh, &prefix, c.hidden_size, c.moe_intermediate_size).unwrap_or(32)
}

#[test]
fn int4_packing_is_byte_identical_to_rabbit() {
    let src_n = fx("int4.src");
    let (rows, cols) = src_n.as2();
    let gs = fx("int4.group_size").data[0] as usize;
    let q = Q4::quantize(&f32s(&src_n.data), rows, cols, gs, 4);

    let packed = fx("int4.packed");
    let bad: Vec<usize> = (0..packed.data.len())
        .filter(|i| q.data[*i] != packed.data[*i] as u8)
        .take(8)
        .collect();
    println!("  int4.packed                      bytes={} mismatched={}", packed.data.len(), bad.len());
    assert!(bad.is_empty(), "nibbles differ at bytes {bad:?}");

    check("int4.scale", &fx("int4.scale").data, &q.scale, 1e-6);
    let deq = fx("int4.dequant");
    let mut got = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        got.extend(q.dequant_row(r));
    }
    check("int4.dequant", &deq.data, &got, 1e-6);
}

#[test]
fn rmsnorm_is_zero_centred() {
    let (x, w, want) = (fx("rmsnorm.x"), fx("rmsnorm.weight"), fx("rmsnorm.out"));
    let (_, h) = x.as2();
    let weight = f32s(&w.data);
    let mut got = vec![0f32; x.data.len()];
    for r in 0..x.data.len() / h {
        kernel::rmsnorm(&row32(&x.data, r, h), &weight, 1e-6, &mut got[r * h..(r + 1) * h]);
    }
    check("rmsnorm (1 + w)", &want.data, &got, 1e-6);

    // the plain-gain reading of the same weight must NOT match, or the fixture
    // is not the zero-centred module this checkpoint uses
    let plain: Vec<f32> = got
        .iter()
        .enumerate()
        .map(|(i, v)| v / (1.0 + w.data[i % h] as f32) * w.data[i % h] as f32)
        .collect();
    assert!(
        flat(&want.data, &plain) > 0.1 * scale_of(&want.data),
        "centred and uncentred weights give the same answer: fixture is suspect"
    );
}

#[test]
fn rmsnorm_gated_norms_then_silus_the_gate() {
    let (x, g) = (fx("rmsnorm_gated.x"), fx("rmsnorm_gated.gate"));
    let (w, want) = (fx("rmsnorm_gated.weight"), fx("rmsnorm_gated.out"));
    let (n, hv) = x.as2();
    let mut got = vec![0f32; n * hv];
    for r in 0..n {
        kernel::rmsnorm_gated(
            &row32(&x.data, r, hv),
            &row32(&g.data, r, hv),
            &f32s(&w.data),
            1e-6,
            &mut got[r * hv..(r + 1) * hv],
        );
    }
    check("rmsnorm_gated", &want.data, &got, 1e-6);
}

#[test]
fn rope_tables_match_the_text_position_grid() {
    let c = cfg();
    assert_eq!(c.rotary_dim(), 64, "partial_rotary_factor 0.25 of head_dim 256");
    let (cos, sin) = (fx("rope.cos"), fx("rope.sin"));
    let (_, rot) = cos.as2();
    let n = cos.data.len() / rot;
    let (mut got_c, mut got_s) = (Vec::new(), Vec::new());
    kernel::rope_tables(c.rope_theta, rot, n, &mut got_c, &mut got_s);
    check("rope.cos", &cos.data, &got_c, 1e-6);
    check("rope.sin", &sin.data, &got_s, 1e-6);
}

#[test]
fn router_picks_the_same_experts_in_the_same_order() {
    let c = cfg();
    let (x, w) = (fx("router.x"), fx("router.weight"));
    let (n, h) = x.as2();
    assert_eq!(h, c.hidden_size);
    let e = c.num_experts;
    let weight = f32s(&w.data);
    let mut got = vec![0f32; n * e];
    for r in 0..n {
        kernel::gemv(&weight, e, h, &row32(&x.data, r, h), &mut got[r * e..(r + 1) * e], 1);
    }
    // `kernel::dot` reduces in 8 AVX2 lanes + a tree on x86 but serially on the
    // scalar fallback. The transformers reference accumulates in its own order,
    // so at this 2048-term dot the two paths land a hair apart: AVX2 reproduces
    // the logits to 1e-6, the scalar loop (ARM, or x86 without AVX2) to ~1.02e-6
    // - unbiased, rms(got) tracks rms(ref), so it is f32 reduction order, not a
    // kernel error. Relax only on the path that sums serially, and the band is
    // still five orders below the nearest wrong result (a mis-routed row is ~1.4);
    // the top-k id/order assertions below are the real point of this test.
    #[cfg(target_arch = "x86_64")]
    let logits_band = if std::is_x86_feature_detected!("avx2") { 1e-6 } else { 1e-5 };
    #[cfg(not(target_arch = "x86_64"))]
    let logits_band = 1e-5f64;
    check("router logits", &fx("router.logits").data, &got, logits_band);

    let (wid, iid) = (fx("router.weights"), fx("router.indices"));
    let k = c.num_experts_per_tok;
    let mut weights = vec![0f32; k];
    let mut ids = vec![0usize; k];
    for r in 0..n {
        kernel::softmax_topk(&got[r * e..(r + 1) * e], k, &mut weights, &mut ids);
        check(&format!("router weights row {r}"), &row(&wid.data, r, k), &weights, 1e-5);
        let want: Vec<usize> = iid.data[r * k..(r + 1) * k].iter().map(|v| *v as usize).collect();
        assert_eq!(ids, want, "top-k ids, row {r}");
    }
}

/// Layer 0's MoE block assembled entirely out of fixtures: the router and shared
/// expert stay f32 so the reference is exact, one int4 routed expert standing in
/// for the 256. The real-export tests below load through `dense_half` and cover
/// the half kernel.
fn moe_from_fixtures(c: Arc<TextConfig>) -> MoeLayer {
    let parts = MoeParts {
        gate: Dense::F32(f32s(&fx("moe.w.gate.weight").data)),
        s_gate: Dense::F32(f32s(&fx("moe.w.shared_expert.gate_proj.weight").data)),
        s_up: Dense::F32(f32s(&fx("moe.w.shared_expert.up_proj.weight").data)),
        s_down: Dense::F32(f32s(&fx("moe.w.shared_expert.down_proj.weight").data)),
        s_egate: f32s(&fx("moe.w.shared_expert_gate.weight").data),
    };
    let (gu_n, dn_n) = (fx("moe.expert.gate_up_proj"), fx("moe.expert.down_proj"));
    let (gu_rows, gu_cols) = gu_n.as2();
    let (dn_rows, dn_cols) = dn_n.as2();
    let gu = Q4::quantize(&f32s(&gu_n.data), gu_rows, gu_cols, 32, 4);
    let dn = Q4::quantize(&f32s(&dn_n.data), dn_rows, dn_cols, 32, 4);
    MoeLayer::from_parts(c, parts, gu, dn, 1)
}

#[test]
fn one_int4_expert_is_exact_given_its_own_dequantisation() {
    let c = cfg();
    let m = moe_from_fixtures(c.clone());
    let id = fx("moe.expert.id").data[0] as usize;
    let x = fx("moe.x");
    let (h, i) = (c.hidden_size, c.moe_intermediate_size);
    let xr = row32(&x.data, 0, h);
    let (mut hh, mut y) = (vec![0f32; i], vec![0f32; h]);
    // the fixture carries one routed expert, by its real id; here it is the only
    // expert in the stack, so its row blocks start at slot 0
    println!("  reference expert {id}, quantised to FQ4 group 32");
    m.expert(0, &xr, &mut hh, &mut y);

    // (a) the int4 dot against fp32 arithmetic on the very numbers it reads out
    //     of the nibbles: any difference here is a kernel bug, not loss
    let deq_gu: Vec<Vec<f32>> = (0..2 * i).map(|r| m.gu.dequant_row(r)).collect();
    let deq_dn: Vec<Vec<f32>> = (0..h).map(|r| m.dn.dequant_row(r)).collect();
    let mut want_hh = vec![0f32; i];
    for j in 0..i {
        let g = kernel::dot(&deq_gu[j], &xr);
        let u = kernel::dot(&deq_gu[i + j], &xr);
        want_hh[j] = kernel::silu(g) * u;
    }
    let want_y: Vec<f32> = (0..h).map(|r| kernel::dot(&deq_dn[r], &want_hh)).collect();
    check("hh vs own weights", &f64s(&want_hh), &hh, 1e-4);
    check("y vs own weights", &f64s(&want_y), &y, 1e-4);

    // (b) the format against torch in fp32. The band is not a guess: it is the
    //     weight matrices' own Frobenius quantisation loss. For a linear map
    //     with independent per-row error, ||Wq x - W x|| / ||W x|| lands on the
    //     same ratio as ||Wq - W|| / ||W||, so anything much above it means the
    //     kernel lost precision the format did not have to. The factor of two
    //     covers the gate, which multiplies two lossy terms, and y, which
    //     inherits hh's error on top of down_proj's.
    let gu_err = m.gu.rel_error(&f32s(&fx("moe.expert.gate_up_proj").data)) as f64;
    let dn_err = m.dn.rel_error(&f32s(&fx("moe.expert.down_proj").data)) as f64;
    println!("  FQ4 loss: gate_up {gu_err:.3} down {dn_err:.3} (group 32, symmetric)");
    check("moe.expert.hh vs torch", &fx("moe.expert.hh").data, &hh, gu_err * 2.0);
    check("moe.expert.y vs torch", &fx("moe.expert.y").data, &y, (gu_err * gu_err + dn_err * dn_err).sqrt() * 2.0);
}

#[test]
fn shared_expert_and_its_gate_match_transformers() {
    let c = cfg();
    let m = moe_from_fixtures(c.clone());
    let x = fx("moe.x");
    let h = c.hidden_size;
    let xr = row32(&x.data, 0, h);
    let mut y = vec![0f32; h];
    m.shared(&xr, &mut y);
    check("moe.shared.y", &fx("moe.shared.y").data, &y, 1e-5);

    let logit = kernel::dot(m.s_egate.as_slice(), &xr);
    check("moe.shared.gate_logit", &fx("moe.shared.gate_logit").data, &[logit], 1e-5);
}

#[test]
fn routing_of_the_moe_block_matches_transformers() {
    let c = cfg();
    let m = moe_from_fixtures(c.clone());
    let (x, wid) = (fx("moe.x"), fx("moe.routing_weights"));
    let iid = fx("moe.routing_indices");
    let (h, k) = (c.hidden_size, c.num_experts_per_tok);
    for r in 0..x.data.len() / h {
        let rg = m.route(&row32(&x.data, r, h));
        check(&format!("routing_weights row {r}"), &row(&wid.data, r, k), &rg.weights, 1e-5);
        let want: Vec<usize> = iid.data[r * k..(r + 1) * k].iter().map(|v| *v as usize).collect();
        assert_eq!(rg.ids, want, "routed ids, row {r}");
    }
}

#[test]
fn moe_block_forward_tracks_the_fp32_block() {
    let Some((sh, c)) = checkpoint() else { return };
    println!("  quantising layer 0's 256 experts to FQ4");
    let m = MoeLayer::load(&sh, c.clone(), 0, group_of(&sh, &c), 8).expect("load layer 0 mlp");
    let (x, out) = (fx("moe.x"), fx("moe.out"));
    let h = c.hidden_size;
    let (got, _) = m.forward(&row32(&x.data, 0, h));
    check("moe.out[0,0] FQ4 vs fp32", &row(&out.data, 0, h), &got, 0.15);
    assert_eq!(m.forward(&row32(&x.data, 0, h)).1.ids.len(), c.num_experts_per_tok);
}

/// The whole point of grouping a chunk's picks by expert is that the weight read
/// is shared, so the arithmetic has to stay exactly the token loop's: same dots,
/// same operands, same reduction order. Exact bits, not a band — the same reason
/// `eight_tokens_walk_the_decoder_stack_to_the_lm_head` compares `!=` instead of
/// calling `check`.
#[test]
fn grouped_expert_prefill_is_bit_identical_to_the_token_loop() {
    let Some((sh, c)) = checkpoint() else { return };
    let m = MoeLayer::load(&sh, c.clone(), 0, group_of(&sh, &c), 8).expect("load layer 0 mlp");
    let (h, k) = (c.hidden_size, c.num_experts_per_tok);
    let x = fx("moe.x");
    let buf = f32s(&x.data);
    let n = buf.len() / h;
    let mut grouped = vec![0f32; n * h];
    assert!(m.prefill(&buf, &mut grouped), "a {n}-row chunk groups its expert picks");
    let mut differ = 0usize;
    let mut ids = Vec::with_capacity(n * k);
    for r in 0..n {
        let (want, routing) = m.forward(&buf[r * h..(r + 1) * h]);
        ids.extend(routing.ids);
        for j in 0..h {
            if want[j] != grouped[r * h + j] {
                differ += 1;
            }
        }
    }
    ids.sort_unstable();
    ids.dedup();
    println!(
        "  grouped prefill of {n} rows: {differ} of {} floats differ from the token loop; \
         {} picks over {} unique experts, so {} weight reads are shared",
        n * h,
        n * k,
        ids.len(),
        n * k - ids.len()
    );
    assert_eq!(differ, 0, "row outputs that differ from `n` calls of forward");
    let mut one = vec![0f32; h];
    assert!(!m.prefill(&buf[..h], &mut one), "a single row has no union to group");
}

#[test]
fn gated_delta_net_matches_the_reference_recursion() {
    let Some((sh, c)) = checkpoint() else { return };
    let g = GdnLayer::load(&sh, c.clone(), 0, 4, true).expect("load layer 0 linear_attn");
    let mut st = GdnState::new(&c);
    let (x, step, pre, states) = (fx("gdn.x"), fx("gdn.step_out"), fx("gdn.out_prefill"), fx("gdn.step_state"));
    let (n, h) = x.as2();
    let per = states.data.len() / n;
    for t in 0..n {
        let mut o = vec![0f32; h];
        g.step(&row32(&x.data, t, h), &mut st, &mut o);
        // the reference fed one token at a time walks the same recursion Fennec
        // does, so this is fp32-vs-fp32 and should be tight
        check(
            &format!("gdn state {t}"),
            &row(&states.data, t, per),
            &st.rec,
            1e-5,
        );
        check(&format!("gdn step {t} / recurrent ref"), &row(&step.data, t, h), &o, 1e-5);
        // the chunked prefill path is algebraically equal but inverts a
        // triangular system: same answer, different rounding
        check(&format!("gdn step {t} / chunked ref"), &row(&pre.data, t, h), &o, 1e-2);
    }
    check("gdn state / recurrent ref", &row(&states.data, n - 1, per), &st.rec, 1e-5);
    // the reference cache is [channels, kernel] holding the last 4 pre-conv
    // inputs, oldest first; Fennec keeps the kernel-1 it actually needs,
    // time-major, so line the two up slot by slot instead of flat
    let cs = fx("gdn.conv_state");
    let (nc, slots) = cs.as2();
    let mut want: Vec<f64> = Vec::with_capacity(nc * st.hist);
    for i in 0..st.hist {
        for c in 0..nc {
            want.push(cs.data[c * slots + (slots - st.hist + i)]);
        }
    }
    check("gdn.conv_state", &want, &st.conv, 1e-6);

    let xd = fx("gdn.x_decode");
    let mut o = vec![0f32; h];
    g.step(&row32(&xd.data, 0, h), &mut st, &mut o);
    check("gdn.out_decode", &fx("gdn.out_decode").data, &o, 1e-4);
    check("gdn.recurrent_state2", &fx("gdn.recurrent_state2").data, &st.rec, 1e-4);
}

#[test]
fn attention_decodes_one_token_at_a_time_like_the_reference() {
    let Some((sh, c)) = checkpoint() else { return };
    let a = AttnLayer::load(&sh, c.clone(), 3, 8, 4, true).expect("load layer 3 self_attn");
    let mut st = AttnState::new(&c, 16);
    let (x, want) = (fx("attn.x"), fx("attn.out"));
    let (n, h) = x.as2();
    for t in 0..n {
        let mut o = vec![0f32; h];
        a.step(&row32(&x.data, t, h), t, &mut st, &mut o);
        check(&format!("attn.out row {t}"), &row(&want.data, t, h), &o, 1e-4);
    }
    assert_eq!(st.len, n);
}

/// A streamed expert is the same expert: nothing resident, a small ring, and
/// the output has to be bit-identical to the layer that holds all 256 in RAM.
/// This is the plan-driven pool's correctness contract - the tier only changes
/// where the bytes come from, never what the math sees.
#[test]
fn a_streamed_expert_is_the_same_expert() {
    let Some((sh, c)) = checkpoint() else { return };
    let h = c.hidden_size;
    let g = group_of(&sh, &c);
    let hot = MoeLayer::load(&sh, c.clone(), 0, g, 8).expect("resident layer 0");
    let cold = MoeLayer::load_set(
        &sh,
        c.clone(),
        0,
        &[],
        Some(Stream::new(sh.clone(), &c, 0, g, 8, 3)),
        g,
        8,
    )
    .expect("pooled layer 0");
    assert_eq!(cold.ids.len(), 0);
    assert!(cold.slot.iter().all(|&s| s < 0), "nothing may be resident");

    let e = fx("fwd.embed");
    let n = e.data.len() / h;
    let want: Vec<(Vec<f32>, Vec<usize>)> = (0..n)
        .map(|t| {
            let (a, r) = hot.forward(&row32(&e.data, t, h));
            (a, r.ids)
        })
        .collect();

    // Correctness: a streamed expert has to be the same expert, byte for byte.
    let t0 = std::time::Instant::now();
    for t in 0..n {
        let v = row32(&e.data, t, h);
        let (b, rb) = cold.forward(&v);
        assert_eq!(rb.ids, want[t].1, "token {t}: the router picked differently");
        assert_eq!(b, want[t].0, "token {t}: a streamed expert is not the same expert");
    }
    let st = cold.stream.as_ref().unwrap().lock().unwrap();
    println!(
        "  pool: {} tokens x {} experts through a ring of {} slots: {} misses, \
         {:.2} GiB held, {:.0} ms read {:.0} ms quantise in {:.1} s",
        n,
        c.num_experts_per_tok,
        st.slots(),
        st.misses,
        st.bytes() as f64 / 2f64.powi(30),
        st.read_ms,
        st.quant_ms,
        t0.elapsed().as_secs_f64(),
    );
    assert!(st.misses > 0, "the cold layer never streamed anything");
    assert!(st.slots() <= 3, "the ring outgrew its cap");
    assert_eq!(st.bytes(), st.slots() * st.slot_bytes(), "slots differ in size");

    // Cost: the ring is the knob, and every slot it keeps is a disk read and a
    // quantise pass it does not pay again. LRU can only get better with room.
    let mut prev = 0u64;
    let mut curve = Vec::new();
    for cap in [3usize, 8, 24, 64] {
        let pool = MoeLayer::load_set(
            &sh,
            c.clone(),
            0,
            &[],
            Some(Stream::new(sh.clone(), &c, 0, g, 8, cap)),
            g,
            8,
        )
        .expect("pooled layer 0");
        for t in 0..n {
            pool.forward(&row32(&e.data, t, h));
        }
        let st = pool.stream.as_ref().unwrap().lock().unwrap();
        let sel = (n * c.num_experts_per_tok) as u64;
        assert_eq!(st.hits + st.misses, sel, "a pick missed the ring and missed the count");
        assert!(st.hits >= prev, "cap {cap}: a bigger ring found fewer reuse");
        assert!(st.slots() <= cap, "cap {cap}: the ring held {}", st.slots());
        prev = st.hits;
        curve.push((cap, st.hits, st.misses, st.bytes() as f64 / 2f64.powi(30)));
        println!(
            "  ring cap {cap:>3}: {}/{} picks served from RAM ({:.1}%), {:.2} GiB held",
            st.hits,
            sel,
            100.0 * st.hits as f64 / sel as f64,
            st.bytes() as f64 / 2f64.powi(30),
        );
    }
    assert!(prev > 0, "no ring size ever reused an expert");
}

/// Eight real token ids through embedding, the decoder stack, the final norm
/// and the lm_head. The reference steps one token at a time with a cache, the
/// same algorithm Fennec runs, so anything left outside fp32 rounding is a
/// graph difference rather than a solver difference - and because the stream is
/// also captured before and after each sub-block, a failure names the block.
#[test]
fn eight_tokens_walk_the_decoder_stack_to_the_lm_head() {
    let Some((sh, c)) = checkpoint() else { return };
    let n_layers = fx("fwd.n_layers").data[0] as usize;
    let ids = fx("fwd.ids");
    let (n, h) = (ids.data.len(), c.hidden_size);
    let t0 = std::time::Instant::now();
    let m = Model::load(&sh, c.clone(), 0, n_layers, group_of(&sh, &c), 8, n).expect("load model");
    println!(
        "  {n_layers} layers resident in {:.1} s ({:.2} GiB of FQ4 experts)",
        t0.elapsed().as_secs_f64(),
        m.layers
            .iter()
            .map(|l| l.moe.as_ref().unwrap().host_bytes() as f64)
            .sum::<f64>()
            / 2f64.powi(30)
    );

    let e = fx("fwd.embed");
    for t in 0..n {
        let got = m.embed_row(&sh, ids.data[t] as usize).expect("embed row");
        check(&format!("embed tok {t}"), &row(&e.data, t, h), &got, 1e-6);
    }

    // Each sub-block on its own, fed the reference's own input so nothing leaks
    // from the block before it.
    let eps = c.rms_norm_eps as f32;
    for l in 0..n_layers {
        let lay = &m.layers[l];
        let mut st = lay.state(&c, n);
        let prev = if l == 0 {
            fx("fwd.embed")
        } else {
            fx(&format!("fwd.qs{}", l - 1))
        };
        let (mix, moe_ref) = (fx(&format!("fwd.qs{l}.attn")), fx(&format!("fwd.qs{l}.moe")));
        let (shared, experts) = (fx(&format!("fwd.qs{l}.shared")), fx(&format!("fwd.qs{l}.experts")));
        let (rid, rw) = (fx(&format!("fwd.qs{l}.route_id")), fx(&format!("fwd.qs{l}.route_w")));
        let (mut same_ids, mut worst_w) = (0usize, 0f64);
        for t in 0..n {
            let x_in = row32(&prev.data, t, h);
            let mut buf = vec![0f32; h];
            kernel::rmsnorm(&x_in, &lay.norm_in, eps, &mut buf);
            let mut res = vec![0f32; h];
            match (&lay.kind, &mut st) {
                (LayerKind::Linear, LayerState::Linear(g)) => {
                    lay.gdn.as_ref().unwrap().step(&buf, g, &mut res)
                }
                (LayerKind::Full, LayerState::Full(a)) => {
                    lay.attn.as_ref().unwrap().step(&buf, t, a, &mut res)
                }
                _ => unreachable!(),
            }
            check(
                &format!("layer {l} mixer tok {t}"),
                &row(&mix.data, t, h),
                &res,
                1e-6,
            );
            let mut x_mid = x_in.clone();
            kernel::axpy(&mut x_mid, &res, 1.0);
            kernel::rmsnorm(&x_mid, &lay.norm_post, eps, &mut buf);
            let moe = lay.moe.as_ref().unwrap();
            let (out, r) = moe.forward(&buf);
            let want: Vec<usize> = row(&rid.data, t, r.ids.len())
                .iter()
                .map(|v| *v as usize)
                .collect();
            same_ids += usize::from(want == r.ids);
            worst_w = worst_w.max(
                row(&rw.data, t, r.ids.len())
                    .iter()
                    .zip(&r.weights)
                    .fold(0f64, |m, (x, y)| m.max((x - *y as f64).abs())),
            );
            let mut sh = vec![0f32; h];
            moe.shared(&buf, &mut sh);
            // Unlike the standalone shared check above, `buf` here is Fennec's own
            // rmsnorm of (x_in + mixer), so it already carries the mixer's f32
            // ordering error - measured up to 6.9e-7 across the four layers - on
            // top of this dot's own. Measured worst 1.01e-6 over 4 layers x 8
            // tokens; the four-chain tree in `avx::dot` sums in a different order
            // from the transformers fixture, and 1e-6 sat just under the old
            // left-to-right lane sum. 2e-6 is that floor with headroom and still
            // separates a wrong expert row (rel ~1) by five orders of magnitude.
            check(
                &format!("layer {l} shared tok {t}"),
                &row(&shared.data, t, h),
                &sh,
                2e-6,
            );
            // the routed sum, taken back out of the block output with the gate the
            // reference applies to the shared expert
            let g = kernel::sigmoid(kernel::dot(&moe.s_egate, &buf));
            if l == 0 && t == 0 {
                let i = c.moe_intermediate_size;
                let (hh_ref, y_ref) = (fx("fwd.qs0.exp_hh"), fx("fwd.qs0.exp_y"));
                let (rows, e0) = (fx("fwd.qs0.probe_rows"), fx("fwd.qs0.probe_id").data[0] as usize);
                let (gu_ref, dn_ref) = (fx("fwd.qs0.gu_probe"), fx("fwd.qs0.dn_probe"));
                for (k, &r) in rows.data.iter().enumerate() {
                    let r = r as usize;
                    check_q(
                        &format!("gu row {e0}*2I+{r}"),
                        &row(&gu_ref.data, k, h),
                        &moe.host_gu_row(e0 * 2 * i + r),
                        1e-6,
                    );
                    check_q(
                        &format!("dn row {e0}*H+{}", r % c.hidden_size),
                        &row(&dn_ref.data, k, i),
                        &moe.host_dn_row(e0 * h + r % c.hidden_size),
                        1e-6,
                    );
                }
                // The fixture's rows are in the reference's router order. Two
                // near-tied experts can swap under another export's rounding, so
                // pair by expert id rather than by position: the routed sum is
                // order-independent, this row lookup is not.
                let want_ids = row(&rid.data, t, r.ids.len());
                for &id in r.ids.iter() {
                    let e = want_ids
                        .iter()
                        .position(|v| *v as usize == id)
                        .unwrap_or_else(|| panic!("expert {id} is not in the reference route"));
                    let mut hh = vec![0f32; i];
                    let mut y = vec![0f32; h];
                    moe.expert(id, &buf, &mut hh, &mut y);
                    check_q(
                        &format!("expert {id} gate_up"),
                        &row(&hh_ref.data, e, i),
                        &hh,
                        1e-6,
                    );
                    check_q(
                        &format!("expert {id} down"),
                        &row(&y_ref.data, e, h),
                        &y,
                        1e-6,
                    );
                }
            }
            let routed: Vec<f32> = out
                .iter()
                .zip(&sh)
                .map(|(o, s)| *o - g * *s)
                .collect();
            check_q(
                &format!("layer {l} routed tok {t}"),
                &row(&experts.data, t, h),
                &routed,
                1e-6,
            );
            check_q(
                &format!("layer {l} moe tok {t}"),
                &row(&moe_ref.data, t, h),
                &out,
                1e-6,
            );
        }
        println!(
            "  layer {l} routing: {same_ids}/{n} tokens picked the same {} experts, \
             worst weight delta {worst_w:.2e}",
            c.num_experts_per_tok
        );
    }

    // layer by layer, from a clean cache each time, so a drift in layer 2 is not
    // blamed on layer 1. `qs` is the reference walking Fennec's own algorithm -
    // one token at a time with a cache - so the band left over is pure fp32
    // ordering. `qh` is the same FQ4 weights through torch's chunked prefill
    // solver and `h` is that solver at fp32; the gap between the three is what
    // the format and the choice of solver each cost.
    for l in 1..=n_layers {
        let want = fx(&format!("fwd.qs{}", l - 1));
        let pre = fx(&format!("fwd.qh{}", l - 1));
        let fp32 = fx(&format!("fwd.h{}", l - 1));
        println!(
            "  layer {}: FQ4 moved the stream {:.2}% rms, chunked-vs-stepped solver {:.2}%",
            l - 1,
            100.0 * rel(&row(&fp32.data, n - 1, h), &row(&pre.data, n - 1, h)),
            100.0 * rel(&row(&pre.data, n - 1, h), &row(&want.data, n - 1, h)),
        );
        let mut st = m.states(n);
        for t in 0..n {
            let mut x = row32(&e.data, t, h);
            for (lay, s) in m.layers[..l].iter().zip(st[..l].iter_mut()) {
                lay.forward(&mut x, s, t);
            }
            check_q(
                &format!("stack thru layer {} tok {}", l - 1, t),
                &row(&want.data, t, h),
                &x,
                1e-5,
            );
        }
    }

    let mut st = m.states(n);
    let mut x = vec![0f32; h];
    let mut emb = vec![0f32; n * h];
    let mut stepped = vec![0f32; n * h];
    for t in 0..n {
        x = m.embed_row(&sh, ids.data[t] as usize).expect("embed row");
        emb[t * h..(t + 1) * h].copy_from_slice(&x);
        m.forward(&mut x, &mut st, t);
        stepped[t * h..(t + 1) * h].copy_from_slice(&x);
    }
    let mut hn = vec![0f32; h];
    kernel::rmsnorm(&x, &m.norm, c.rms_norm_eps as f32, &mut hn);
    check_q("fwd.norm last tok", &row(&fx("fwd.norm").data, n - 1, h), &hn, 1e-5);

    // Prefill reads each dense projection once for the whole prompt instead of
    // once per token, so it has to be the token loop's own arithmetic: exact
    // bits, because the 1e-5 band the checks above use would swallow a real
    // divergence in the shipped prompt path. This is the copy that covers the
    // gated-delta-net mixer - the dense lane's own version cannot see it.
    assert!(m.prefill_ready(), "an all-host plan can batch its projections");
    let mut px = emb.clone();
    let mut st2 = m.states(n);
    m.prefill(&mut px, &mut st2, 0);
    let differ: Vec<usize> = (0..n)
        .map(|t| {
            let a = &stepped[t * h..(t + 1) * h];
            let b = &px[t * h..(t + 1) * h];
            a.iter().zip(b).filter(|(u, v)| u != v).count()
        })
        .collect();
    println!(
        "  batched prefill of {n} tokens: {} outputs differ from the token loop across {n_layers} layers",
        differ.iter().sum::<usize>()
    );
    assert_eq!(differ.iter().sum::<usize>(), 0, "tokens that differ: {differ:?}");

    let t0 = std::time::Instant::now();
    let mut lg = vec![0f32; c.vocab_size];
    m.logits(&sh, &hn, &mut lg).expect("logits");
    let idx = fx("fwd.logits_idx");
    let top = fx("fwd.logits_top");
    let ours: Vec<usize> = (0..idx.data.len()).map(|i| idx.data[i] as usize).collect();
    // our own top-8, then how many of the reference's made it in
    let mut order: Vec<usize> = (0..lg.len()).collect();
    order.sort_by(|&a, &b| lg[b].partial_cmp(&lg[a]).unwrap());
    let mine = &order[..ours.len()];
    let shared = ours.iter().filter(|i| mine.contains(i)).count();
    println!(
        "  logits over {} rows in {:.1} s | {shared}/{} of the reference top-8 in ours, \
         greedy ref {} vs ours {}",
        c.vocab_size,
        t0.elapsed().as_secs_f64(),
        ours.len(),
        ours[0],
        mine[0]
    );
    let top32 = fx("fwd.logits_top32");
    println!(
        "  FQ4 moved those 8 logits by {:.2}% rms",
        100.0 * rel(&top32.data, &top.data)
    );
    for (i, id) in ours.iter().enumerate() {
        check_q(&format!("logit[{}] ref id {}", i, id), &[top.data[i]], &[lg[*id]], 1e-5);
    }
    assert_eq!(ours[0], mine[0], "the greedy token changed");
}
