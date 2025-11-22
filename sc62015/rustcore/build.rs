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

    // Ensure the repo root is on PYTHONPATH so `sc62015` is importable when running codegen.
    let mut cmd = Command::new(python);
    let mut py_path = env::var("PYTHONPATH").unwrap_or_default();
    if !py_path.is_empty() {
        py_path.push(':');
    }
    py_path.push_str(repo_root.to_string_lossy().as_ref());
    cmd.env("PYTHONPATH", py_path);

    // Ensure output dir exists.
    std::fs::create_dir_all(&generated).expect("create generated dir");

    let status = cmd
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

    // Verify outputs exist; bail early if generation didn't produce them.
    for path in [&handlers, &opcode_index, &types] {
        if !path.exists() {
            panic!(
                "SCIL payload missing expected artifact: {}; rerun codegen locally",
                path.display()
            );
        }
    }

    println!("cargo:rerun-if-changed={}", generated.display());
}
