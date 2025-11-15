// @generated; shared types for generated handlers
use serde::Deserialize;
use serde_json::{Map, Value};
use std::collections::HashMap;

#[derive(Clone, Debug, Deserialize)]
pub struct ManifestEntry {
    pub id: u32,
    pub opcode: u32,
    pub mnemonic: String,
    pub family: Option<String>,
    pub length: u8,
    pub pre: Option<PreInfo>,
    pub instr: Value,
    pub binder: Map<String, Value>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Hash)]
pub struct PreInfo {
    pub first: String,
    pub second: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct BoundInstrRepr {
    pub opcode: u32,
    pub mnemonic: String,
    pub family: Option<String>,
    pub length: u8,
    pub pre: Option<PreInfo>,
    pub operands: HashMap<String, Value>,
}
