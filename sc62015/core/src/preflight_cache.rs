// PY_SOURCE: sc62015/pysc62015/emulator.py
//! Memo of successful instruction preflights for stable upper-ROM code.
//!
//! `CoreRuntime` silently validates every instruction and prepares its timing
//! before advancing any device. For an instruction whose bytes are plain upper
//! ROM (no overlay, host range or device mapping), that outcome is a pure
//! function of the instruction bytes: decode rejections, PRE canonicality and
//! the timing selector read only those bytes, and the base-register reads made
//! while computing operand addresses cannot reject. The cache therefore stores
//! the bytes it validated and reuses the result only after re-reading the live
//! bytes and finding them identical, so it never needs explicit invalidation.
//!
//! IR (0xFE) and RESET (0xFF) are never cached because their preflight also
//! validates vector bytes and the vector destination.

use crate::llama::timing::PreparedInstructionTiming;
use crate::memory::MemoryImage;

/// Longest instruction (including two PRE bytes) the cache stores.
pub(crate) const MAX_CACHED_LEN: usize = 8;
const ENTRY_COUNT: usize = 1 << 13;

#[derive(Clone, Copy)]
struct Entry {
    /// Instruction address, or `u32::MAX` for an empty slot.
    pc: u32,
    len: u8,
    bytes: [u8; MAX_CACHED_LEN],
    timing: PreparedInstructionTiming,
}

pub(crate) struct PreflightCache {
    entries: Box<[Option<Entry>]>,
}

impl Default for PreflightCache {
    fn default() -> Self {
        Self {
            entries: vec![None; ENTRY_COUNT].into_boxed_slice(),
        }
    }
}

impl PreflightCache {
    #[inline]
    fn slot(pc: u32) -> usize {
        (pc as usize) & (ENTRY_COUNT - 1)
    }

    /// The cached `(opcode, timing)` for `pc` when its live bytes still match
    /// the bytes that were validated. The caller must have established that no
    /// device maps the upper ROM window.
    #[inline]
    pub(crate) fn lookup(
        &self,
        pc: u32,
        memory: &MemoryImage,
    ) -> Option<(u8, PreparedInstructionTiming)> {
        let entry = self.entries[Self::slot(pc)].as_ref()?;
        if entry.pc != pc {
            return None;
        }
        let len = usize::from(entry.len);
        let live = memory.plain_upper_rom_span(pc, len)?;
        (live == &entry.bytes[..len]).then_some((entry.bytes[0], entry.timing))
    }

    /// Record a successful preflight of the instruction at `pc`.
    pub(crate) fn insert(
        &mut self,
        pc: u32,
        len: u8,
        timing: PreparedInstructionTiming,
        memory: &MemoryImage,
    ) {
        let len_usize = usize::from(len);
        if len_usize == 0 || len_usize > MAX_CACHED_LEN {
            return;
        }
        let Some(live) = memory.plain_upper_rom_span(pc, len_usize) else {
            return;
        };
        if matches!(live[0], 0xFE | 0xFF) || matches!(timing.resolved_opcode(), 0xFE | 0xFF) {
            return;
        }
        let mut bytes = [0u8; MAX_CACHED_LEN];
        bytes[..len_usize].copy_from_slice(live);
        self.entries[Self::slot(pc)] = Some(Entry {
            pc,
            len,
            bytes,
            timing,
        });
    }
}
