//! Operand helpers for LLAMA (placeholder).
//!
//! These types will evolve to mirror addressing/width semantics from the Python
//! emulator, and to drive decoding/evaluation without LLIL/SCIL.

#[derive(Debug, Clone, Copy)]
pub enum AddressMode {
    Internal,
    External,
    Code,
}

#[derive(Debug, Clone, Copy)]
pub enum OperandValue {
    Immediate(u32, u8), // value, bits
    Register(&'static str, u8),
    Memory(AddressMode, u32, u8),
    MemoryWithOffset(AddressMode, u32, u8, i32), // base, width, signed offset
}

/// Estimate encoded length contribution for an operand (bytes).
pub fn operand_len_bytes(kind: &super::opcodes::OperandKind) -> u8 {
    use super::opcodes::OperandKind::*;
    match kind {
        Imm(bits) => (*bits + 7) / 8,
        IMem(bits) => (*bits + 7) / 8,
        // External memory address/imem selectors are encoded in bytes (1/2/3).
        EMemAddrWidth(bytes) | EMemAddrWidthOp(bytes) => *bytes,
        EMemReg(bits) | EMemIMem(bits) => (*bits + 7) / 8,
        EMemRegWidth(bytes) | EMemRegWidthMode(bytes) | EMemIMemWidth(bytes) => *bytes,
        EMemImemOffsetDestIntMem | EMemImemOffsetDestExtMem => 2, // heuristic for offset+addr
        RegIMemOffset(_) => 1,                                    // opcode stream encodes mode
        EMemRegModePostPre => 1,                                  // mode encoded in operand byte
        RegPair(size) => *size as u8,                             // heuristic
        _ => 0,
    }
}
