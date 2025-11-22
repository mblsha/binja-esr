//! Project LLAMA: direct-execution Rust core mirroring the Python LLIL lifter.
//!
//! This module is intentionally not wired into the build yet; it serves as a
//! staging area for the LLIL-less evaluator, keeping types and tables close to
//! the existing core. The goal is drop-in swapability with the current SCIL
//! path once the evaluator is implemented.

#![allow(dead_code)]

pub mod dispatch;
pub mod eval;
pub mod opcodes;
pub mod operands;
pub mod parity;
pub mod state;

/// Placeholder entry point for future LLAMA runtime glue.
pub struct LlamaRuntime;

impl LlamaRuntime {
    pub fn new() -> Self {
        Self
    }
}
