//! The command-line surface: every spelling the usage text advertises must
//! actually reach the code it claims to, without loading a model.

fn run(args: &[&str]) -> std::process::Output {
    std::process::Command::new(env!("CARGO_BIN_EXE_fennec"))
        .args(args)
        .env_remove("FENNEC_MODEL")
        .output()
        .expect("failed to spawn the fennec binary")
}

fn text(out: &std::process::Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

#[test]
fn help_reaches_usage_under_every_spelling() {
    for cmd in ["help", "--help", "-h"] {
        let out = run(&[cmd]);
        assert!(out.status.success(), "fennec {cmd} exited {}", out.status);
        let body = text(&out);
        assert!(body.contains("usage: fennec <command>"), "fennec {cmd} printed no usage: {body}");
    }
}

#[test]
fn version_reports_the_library_version_under_every_spelling() {
    let want = format!("fennec {}", fennec::VERSION);
    for cmd in ["version", "--version", "-V", "-v"] {
        let out = run(&[cmd]);
        assert!(out.status.success(), "fennec {cmd} exited {}", out.status);
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), want, "fennec {cmd}");
    }
}

#[test]
fn a_run_without_a_model_names_only_the_remedies_that_work() {
    let out = run(&["run"]);
    assert!(!out.status.success(), "fennec run succeeded without a model directory");
    let body = text(&out);
    assert!(body.contains("no model directory"), "{body}");
    assert!(body.contains("--model <dir>") && body.contains("FENNEC_MODEL"), "{body}");
    assert!(!body.contains("config.json"), "the error advertises a key no engine code reads: {body}");
}

#[test]
fn an_unknown_command_is_rejected_not_run() {
    let out = run(&["frobnicate"]);
    assert!(!out.status.success(), "fennec frobnicate exited {}", out.status);
    assert!(text(&out).contains("unknown command"), "{}", text(&out));
}
