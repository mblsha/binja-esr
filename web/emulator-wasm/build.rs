use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn git_output(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8(out.stdout).ok().map(|s| s.trim().to_string())
}

fn main() {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs().to_string())
        .unwrap_or_else(|_| "0".to_string());
    println!("cargo:rustc-env=BUILD_TIMESTAMP={timestamp}");

    let git_commit = git_output(&["rev-parse", "--short=12", "HEAD"]).unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=GIT_COMMIT={git_commit}");

    // Ensure `BUILD_TIMESTAMP`/`GIT_COMMIT` are refreshed when HEAD changes, even if
    // the Rust sources did not.
    if let Some(head_path) = git_output(&["rev-parse", "--git-path", "HEAD"]) {
        println!("cargo:rerun-if-changed={head_path}");
    }
    if let Some(index_path) = git_output(&["rev-parse", "--git-path", "index"]) {
        println!("cargo:rerun-if-changed={index_path}");
    }
    println!("cargo:rerun-if-changed=build.rs");
}
