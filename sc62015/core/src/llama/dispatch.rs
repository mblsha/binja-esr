// PY_SOURCE: sc62015/pysc62015/instr/opcode_table.py
//! Dispatch helpers for LLAMA.
//!
//! This is a thin wrapper over the static opcode table to allow lookups by
//! opcode (and eventually PRE/mode variants).

use super::opcodes::{OpcodeEntry, OPCODES};

pub fn lookup(opcode: u8) -> Option<&'static OpcodeEntry> {
    OPCODES.iter().find(|entry| entry.opcode == opcode)
}
