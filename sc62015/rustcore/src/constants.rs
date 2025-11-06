//! Core architectural constants shared between the Rust and Python SC62015 implementations.

/// Number of bytes in the internal RAM region.
pub const INTERNAL_MEMORY_LENGTH: u32 = 0x100;

/// Total addressable space: 1 MB of external memory + 256 bytes internal.
pub const ADDRESS_SPACE_SIZE: u32 = 0x100000 + INTERNAL_MEMORY_LENGTH;

/// Base address of the internal RAM window within the unified address space.
pub const INTERNAL_MEMORY_START: u32 = ADDRESS_SPACE_SIZE - INTERNAL_MEMORY_LENGTH;

/// Program counter uses only the lower 20 bits of its 24-bit storage.
pub const PC_MASK: u32 = 0x0F_FFFF;

/// Number of scratch (TEMP) registers supported by the architecture.
pub const NUM_TEMP_REGISTERS: usize = 14;
