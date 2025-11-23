//! Project LLAMA: direct-execution Rust core mirroring the Python LLIL lifter.
//!
//! LLAMA is now the sole Rust CPU core; the SCIL/manifest interpreter path has
//! been removed. These modules hold the opcode table, evaluator, and supporting
//! state/parity helpers used by the PyO3 wrapper.

#![allow(dead_code)]

pub mod dispatch;
pub mod eval;
pub mod opcodes;
pub mod parity;
pub mod state;

/// Placeholder entry point for future LLAMA runtime glue.
pub struct LlamaRuntime;

impl LlamaRuntime {
    pub fn new() -> Self {
        Self
    }
}

impl Default for LlamaRuntime {
    fn default() -> Self {
        Self::new()
    }
}
