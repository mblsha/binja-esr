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

    // Allow skipping SCIL codegen when running LLAMA-only builds/tests.
    if env::var_os("LLAMA_SKIP_SCIL").is_some() {
        println!("cargo:warning=LLAMA_SKIP_SCIL set; skipping SCIL payload generation");
        std::fs::create_dir_all(&generated).expect("create generated dir");
        std::fs::write(
            &handlers,
            "// dummy handlers for LLAMA_SKIP_SCIL\npub const PAYLOAD: &str = \"[]\";\n",
        )
        .expect("write handlers");
        std::fs::write(
            &opcode_index,
            "// dummy opcode index for LLAMA_SKIP_SCIL\n\
            #[derive(Copy, Clone, Debug)]\n\
            pub struct PreKey { pub first: &'static str, pub second: &'static str }\n\
            #[derive(Copy, Clone, Debug)]\n\
            pub struct OpcodeIndexEntry { pub opcode: u8, pub pre: Option<PreKey>, pub manifest_index: usize }\n\
            pub static OPCODE_INDEX: &[OpcodeIndexEntry] = &[];\n",
        )
        .expect("write opcode_index");
        std::fs::write(
            &types,
            "// dummy types for LLAMA_SKIP_SCIL\n\
            use serde::{Serialize, Deserialize};\n\
            use std::collections::HashMap;\n\
            #[derive(Clone, Debug, Serialize, Deserialize, Default)]\n\
            pub struct PreInfo { pub first: String, pub second: String }\n\
            #[derive(Clone, Debug, Serialize, Deserialize, Default)]\n\
            pub struct LayoutEntry { pub key: String, pub kind: String, pub meta: HashMap<String, serde_json::Value> }\n\
            #[derive(Clone, Debug, Serialize, Deserialize, Default)]\n\
            pub struct ManifestEntry {\n\
                pub id: usize,\n\
                pub opcode: u32,\n\
                pub mnemonic: String,\n\
                pub family: Option<String>,\n\
                pub length: u8,\n\
                pub pre: Option<PreInfo>,\n\
                pub instr: serde_json::Value,\n\
                pub binder: serde_json::Map<String, serde_json::Value>,\n\
                pub bound_repr: serde_json::Value,\n\
                pub layout: Vec<LayoutEntry>,\n\
            }\n\
            #[derive(Clone, Debug, Serialize, Deserialize, Default)]\n\
            pub struct BoundInstrRepr {\n\
                pub opcode: u32,\n\
                pub mnemonic: String,\n\
                pub family: Option<String>,\n\
                pub length: u8,\n\
                pub pre: Option<PreInfo>,\n\
                pub operands: HashMap<String, serde_json::Value>,\n\
            }\n",
        )
        .expect("write types");
        return;
    }

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
