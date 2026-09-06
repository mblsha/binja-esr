// PY_SOURCE: pce500/run_pce500.py
//! Public synthetic CLI control smoke, not real-ROM or hardware evidence.
#![cfg(feature = "cli")]

use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::thread::sleep;
use std::time::{Duration, Instant};

fn run_cli(args: &[&str]) -> Output {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../web/emulator-wasm/testdata/pf1_demo_rom_window.rom");
    let mut child = Command::new(env!("CARGO_BIN_EXE_sc62015-lcd"))
        .arg("--rom")
        .arg(fixture)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    // Never let a native scheduling regression hang the whole test suite.
    // Only the tiny start/end diagnostics use a pipe; LCD output is discarded.
    let started = Instant::now();
    while child.try_wait().unwrap().is_none() {
        if started.elapsed() > Duration::from_secs(5) {
            let _ = child.kill();
            let output = child.wait_with_output().unwrap();
            panic!(
                "native CLI did not finish: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        sleep(Duration::from_millis(5));
    }
    child.wait_with_output().unwrap()
}

#[test]
fn native_modes_preserve_explicit_completion_and_do_not_sleep_after_exhaustion() {
    for model in ["pc-e500", "iq-7000"] {
        let mut reference = None;
        for mode in ["interactive", "turbo", "deterministic"] {
            let output = run_cli(&[
                "--model",
                model,
                "--mode",
                mode,
                "--steps",
                "5000",
                "--iq7000-rtc",
                "202609060000",
            ]);
            let log = String::from_utf8(output.stderr).unwrap();
            assert!(output.status.success(), "{log}");
            assert!(log.contains("not hardware calibrated"));
            let final_state = log
                .lines()
                .find(|line| line.contains("finished boundaries=5000 "))
                .unwrap()
                .split(" dropped_host_ns=")
                .next()
                .unwrap();
            if let Some(expected) = &reference {
                assert_eq!(final_state, expected);
            } else {
                reference = Some(final_state.to_string());
            }
        }
    }
    let output = run_cli(&["--mode", "turbo", "--steps", "1", "--sleep-ms", "60000"]);
    assert!(output.status.success());
}

#[test]
fn native_deterministic_mode_rejects_missing_budget_and_host_rtc_seed() {
    for args in [
        vec!["--mode", "deterministic"],
        vec![
            "--model",
            "iq-7000",
            "--mode",
            "deterministic",
            "--steps",
            "100",
        ],
    ] {
        let output = run_cli(&args);
        assert!(!output.status.success());
        let log = String::from_utf8(output.stderr).unwrap();
        assert!(log.contains("deterministic"));
        assert!(!log.contains("finished boundaries="));
    }
}
