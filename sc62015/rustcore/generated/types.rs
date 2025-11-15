// @generated; shared types for generated handlers
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct BoundInstrPayload {
    pub id: u32,
    pub opcode: u8,
    pub mnemonic: String,
    pub family: Option<String>,
    pub length: u8,
    pub pre: Option<PreInfo>,
    pub operands: serde_json::Value,
}

#[derive(Clone, Debug, Deserialize)]
pub struct PreInfo {
    pub first: String,
    pub second: String,
}

impl PreInfo {
    pub fn as_tuple(&self) -> (&str, &str) {
        (&self.first, &self.second)
    }
}
