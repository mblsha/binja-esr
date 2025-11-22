use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set"));
    let generated = manifest_dir.join("generated");
    let handlers = generated.join("handlers.rs");
    let opcode_index = generated.join("opcode_index.rs");
    let types = generated.join("types.rs");

    // If the generated payload exists, set rerun hints and return.
    if handlers.exists() && opcode_index.exists() && types.exists() {
        println!("cargo:rerun-if-changed={}", handlers.display());
        println!("cargo:rerun-if-changed={}", opcode_index.display());
        println!("cargo:rerun-if-changed={}", types.display());
        return;
    }

    // Attempt to generate the payload via the Python codegen script.
    let repo_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let script = repo_root.join("tools").join("scil_codegen_rust.py");
    let python = env::var("PYTHON").unwrap_or_else(|_| "python3".to_string());
    let status = Command::new(python)
        .arg(script)
        .arg("--out-dir")
        .arg(&generated)
        .status()
        .expect("failed to spawn scil codegen");

    if !status.success() {
        panic!(
            "SCIL payload generation failed; ensure binja-test-mocks is on PYTHONPATH and rerun \
             `python tools/scil_codegen_rust.py --out-dir sc62015/rustcore/generated`"
        );
    }

    println!("cargo:rerun-if-changed={}", generated.display());
}
