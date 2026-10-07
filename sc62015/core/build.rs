// PY_SOURCE: pce500/run_pce500.py
// PY_SOURCE: pce500/oz9600/input.py
//! Compile the shared browser/native physical key source into static Rust data.
use std::{collections::BTreeMap, env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=data/physical_keys.json");
    let source = fs::read_to_string("data/physical_keys.json").expect("read physical keys");
    let maps: BTreeMap<String, BTreeMap<String, u8>> =
        serde_json::from_str(&source).expect("valid physical key source");
    assert_eq!(maps.len(), 3, "unexpected physical key model");
    let mut output = String::new();
    for (model, symbol) in [
        ("pc-e500", "PC_E500_KEYS"),
        ("iq-7000", "IQ_7000_KEYS"),
        ("oz-9600", "OZ_9600_KEYS"),
    ] {
        let keys = maps.get(model).expect("required model");
        output.push_str(&format!("const {symbol}: &[(&str, u8)] = &[\n"));
        for (name, code) in keys {
            assert!(!name.is_empty() && *code < 128, "invalid physical key");
            output.push_str(&format!("({name:?}, {code}),\n"));
        }
        output.push_str("];\n");
    }
    let destination = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo output directory"));
    fs::write(destination.join("physical_keys.rs"), output).expect("write physical key tables");
}
