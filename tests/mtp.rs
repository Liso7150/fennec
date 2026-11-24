//! What the speculative drafter must hold to (#482).
//!
//! The cheap tests are the invariants the round driver is built out of: which
//! index wins a tie, what a rolled-back cache looks like, and that a saved state
//! comes back whole. They need no checkpoint.
//!
//! The one test that needs weights is `speculation_leaves_the_answer_unchanged`,
//! and it is the reason the feature is allowed to exist: a round may only commit
//! a token the trunk itself would have produced at that position. It runs the
//! shipped binary twice on one prompt, once per decode path, because that is the
//! only pairing that covers the whole turn - the prefill, the sampler, the think
//! routing and both head paths - rather than the arithmetic in between.
//!
//! Like the rest of the weight-backed lanes it is `#[ignore]`d: it costs two model
//! loads and a few minutes of a 31 GiB box. Run it with
//! `cargo test --release --test mtp -- --ignored --nocapture`.

use std::sync::Arc;

use fennec::model::{AttnState, GdnState, LayerState, StateSave};
use fennec::mtp::{argmax, rewind, MtpState};
use fennec::config::TextConfig;

/// The checkpoint the drafter is real on: the served one, whose config advertises
/// `mtp_num_hidden_layers: 1`. The bf16 export and the pre-quantised container are
/// the same model, so either answers; `FENNEC_GOLDEN_MODEL` names the former.
fn model_dir() -> String {
    std::env::var("FENNEC_MTP_MODEL")
        .or_else(|_| std::env::var("FENNEC_GOLDEN_MODEL"))
        .unwrap_or_else(|_| r"D:\fennec-models\qwen36-35b".to_string())
}

/// The trunk's text config, read from the checkpoint rather than restated here:
/// the drafter's shape - width, experts, norms - is the trunk's, and a synthetic
/// copy would let a config change pass a test it breaks in the field.
fn trunk_cfg() -> Option<Arc<TextConfig>> {
    let raw: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{}/config.json", model_dir())).ok()?).ok()?;
    Some(Arc::new(TextConfig::from_json(&raw).expect("checkpoint config")))
}

// ---------------------------------------------------------------- no weights --

#[test]
fn argmax_takes_the_first_maximum() {
    // `Sample::pick` at temperature 0 scans the same way, and the drafter and the
    // verifier have to break a tie identically or the acceptance test is not the
    // sequential loop's rule.
    assert_eq!(argmax(&[0.1, 9.0, -3.0, 2.0]), 1);
    assert_eq!(argmax(&[1.0, 7.0, 7.0, 7.0]), 1, "a tie must go to the earliest index");
    assert_eq!(argmax(&[4.0]), 0);
    assert_eq!(argmax(&[2.0, 2.0]), 0);
}

#[test]
fn a_saved_state_comes_back_whole_for_both_cache_shapes() {
    // The rollback is the only thing that makes a rejected block free. If it misses
    // a field, a rejected draft leaves its mark on the answer - which is exactly
    // the failure this module's contract forbids.
    let raw = serde_json::json!({
        "model_type": "qwen3_5_moe_text",
        "hidden_size": 2048,
        "num_hidden_layers": 4,
        "vocab_size": 1024,
        "rms_norm_eps": 1e-6,
        "layer_types": ["linear_attention", "full_attention"],
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
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
    });
    let c = Arc::new(TextConfig::from_json(&raw).expect("synthetic config"));
    let mut st = vec![
        LayerState::Linear(GdnState::new(&c)),
        LayerState::Full(AttnState::new(&c, 16)),
    ];
    let slots = StateSave::all(&st);

    // A pass that goes too far: the recurrent window, its running matrix, the row
    // counter, and the mirror flag a device tier would have set.
    let g = match &mut st[0] {
        LayerState::Linear(g) => g,
        _ => panic!("layer 0 is the mixer"),
    };
    for (i, v) in g.conv.iter_mut().enumerate() {
        *v = i as f32;
    }
    for (i, v) in g.rec.iter_mut().enumerate() {
        *v = -(i as f32);
    }
    g.filled = g.hist;
    g.mirrored = true;
    if let LayerState::Full(a) = &mut st[1] {
        a.len = 9;
        a.k[0] = 3.5;
    }

    StateSave::restore_all(&mut st, &slots);
    let g = match &st[0] {
        LayerState::Linear(g) => g,
        _ => panic!("layer 0 is the mixer"),
    };
    assert!(g.conv.iter().all(|&v| v == 0.0), "the conv window came back dirty");
    assert!(g.rec.iter().all(|&v| v == 0.0), "the running matrix came back dirty");
    assert_eq!(g.filled, 0, "the mixer thinks it has seen tokens it did not");
    assert!(!g.mirrored, "the mirror flag survived the rollback");
    let a = match &st[1] {
        LayerState::Full(a) => a,
        _ => panic!("layer 1 is attention"),
    };
    assert_eq!(a.len, 0, "attention forgot to forget its rows");
    // The KV bytes deliberately survive. Attention forgets by row count, so the
    // round's rollback copies nothing and the next write overwrites the dead row;
    // only the gated-delta-net state, which has no row store to count, is kept by
    // value. This reads 3.5, and a slot that copied the rows would read 0.0 here
    // at the price of every layer's cache, once per round.
    assert_eq!(a.k[0], 3.5, "the attention rollback is a row count, not a byte copy");

    // And a second grab reuses the allocations, which is what keeps the 63 MiB of
    // state per request out of the round's cost.
    let mut again = slots;
    StateSave::refresh(&mut again, &st);
    assert_eq!(again.len(), 2);
}

#[test]
fn rewind_moves_the_rows_and_the_positions_together() {
    // The drafter's rope positions come from its own row count, so a rewind that
    // moves one and not the other leaves the next step attending to a row it is
    // about to overwrite: a wrong draft, which costs speed and not correctness -
    // except that it also silently caps the acceptance rate, which is the number
    // the whole feature is priced with.
    let c = trunk_cfg().unwrap_or_else(|| {
        let raw = serde_json::json!({
            "model_type": "qwen3_5_moe_text", "hidden_size": 2048, "num_hidden_layers": 4,
            "vocab_size": 1024, "rms_norm_eps": 1e-6,
            "layer_types": ["full_attention"], "num_attention_heads": 16, "num_key_value_heads": 2,
            "head_dim": 256, "partial_rotary_factor": 0.25, "attn_output_gate": true,
            "rope_parameters": {"rope_theta": 1e7}, "linear_num_key_heads": 16,
            "linear_num_value_heads": 32, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4, "num_experts": 8, "num_experts_per_tok": 2,
            "moe_intermediate_size": 512, "shared_expert_intermediate_size": 512,
        });
        Arc::new(TextConfig::from_json(&raw).unwrap())
    });
    let mut st = MtpState { st: vec![LayerState::Full(AttnState::new(&c, 16))], plen: 7 };
    if let LayerState::Full(a) = &mut st.st[0] {
        a.len = 12;
    }
    rewind(&mut st, 4);
    let a = match &st.st[0] {
        LayerState::Full(a) => a,
        _ => panic!("the drafter is attention-only"),
    };
    assert_eq!((st.plen, a.len), (4, 4), "rows and positions moved apart");

    // Asking to keep more than exists clamps rather than runs the counter past the
    // cache: the driver's `base + m.min(k)` cannot exceed what was appended, but a
    // round at the end of the window can.
    rewind(&mut st, 40);
    assert_eq!(st.plen, 40);
    assert_eq!(match &st.st[0] { LayerState::Full(a) => a.len, _ => unreachable!() }, 4,
        "a cache cannot have more rows than it was given");
}

// ------------------------------------------------------------------ weights --

/// `--mtp 2` has to answer with the same tokens as `--mtp 0`, byte for byte, and
/// it has to actually have drafted to prove it. Both halves matter: the first is
/// the feature's whole claim, and without the second the test passes the day
/// speculation quietly turns itself off.
#[test]
#[ignore = "needs the checkpoint: two model loads, a few minutes"]
fn speculation_leaves_the_answer_unchanged() {
    if !std::path::Path::new(&model_dir()).is_dir() {
        println!("  {} is not on this machine: skipped", model_dir());
        return;
    }
    let exe = env!("CARGO_BIN_EXE_fennec");
    let prompt = "Explain in two sentences why the sky is blue.";
    let run = |k: usize| -> (String, String) {
        let out = std::process::Command::new(exe)
            .args([
                "run",
                prompt,
                "--model",
                &model_dir(),
                "--cpu",
                "--greedy",
                "--tokens",
                "64",
                "--mtp",
                &k.to_string(),
            ])
            .output()
            .expect("fennec run");
        assert!(out.status.success(), "fennec run --mtp {k} failed: {}", String::from_utf8_lossy(&out.stderr));
        (
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    };

    let (want, _) = run(0);
    let (got, err) = run(2);
    // "[fennec] mtp: 24 rounds, 39/47 drafts accepted, 2.62 tokens/round". A turn
    // that never speculated prints nothing here, which is the failure this guard
    // exists for: the equality below would have passed.
    let line = err
        .lines()
        .find(|l| l.contains("drafts accepted"))
        .unwrap_or("no drafter report: the turn never speculated")
        .trim()
        .to_string();
    println!("  {line}");
    let frac = line
        .split_whitespace()
        .find(|w| w.matches('/').count() == 1)
        .unwrap_or("0/0");
    let (accepted, proposed): (usize, usize) = frac
        .split_once('/')
        .map(|(a, p)| (a.parse().unwrap(), p.parse().unwrap()))
        .unwrap();
    assert!(proposed > 0, "the round driver proposed nothing");
    assert!(accepted > 0, "the drafter proposed {proposed} tokens and the trunk kept none: check embed_first and pos_off");
    assert_eq!(
        want, got,
        "speculation changed the answer ({line}):\n--- mtp 0 ---\n{want}\n--- mtp 2 ---\n{got}"
    );
}
