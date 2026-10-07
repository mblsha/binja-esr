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
//! validates vector bytes and the vector destination. Byte ADD/SUB are also
//! excluded because selector validity depends on the explicit source policy.

use crate::llama::eval::DecodeMemo;
use crate::llama::timing::PreparedInstructionTiming;
use crate::memory::MemoryImage;

/// Longest instruction (including two PRE bytes) the cache stores.
pub(crate) const MAX_CACHED_LEN: usize = 8;
const ENTRY_COUNT: usize = 1 << 13;

/// Hot per-PC preflight result; `pc == u32::MAX` marks an empty slot.
#[derive(Clone, Copy)]
struct Entry {
    pc: u32,
    len: u8,
    /// The validated bytes, little-endian and zero-padded.
    word: u64,
    timing: PreparedInstructionTiming,
}

/// Compact lookup table plus a parallel table of execution decode memos
/// (kept apart so the per-boundary lookup touches only small entries).
pub(crate) struct PreflightCache {
    entries: Box<[Entry]>,
    memos: Box<[DecodeMemo]>,
}

impl Default for PreflightCache {
    fn default() -> Self {
        let empty = Entry {
            pc: u32::MAX,
            len: 0,
            word: 0,
            timing: PreparedInstructionTiming::default(),
        };
        Self {
            entries: vec![empty; ENTRY_COUNT].into_boxed_slice(),
            memos: vec![DecodeMemo::default(); ENTRY_COUNT].into_boxed_slice(),
        }
    }
}

impl PreflightCache {
    #[inline]
    fn slot(pc: u32) -> usize {
        (pc as usize) & (ENTRY_COUNT - 1)
    }

    /// The cached `(opcode, timing, slot)` for `pc` when its live bytes still
    /// match the bytes that were validated. The caller must have established
    /// that no device maps the upper ROM window.
    #[inline(always)]
    pub(crate) fn lookup(
        &self,
        pc: u32,
        memory: &MemoryImage,
    ) -> Option<(u8, PreparedInstructionTiming, usize)> {
        let slot = Self::slot(pc);
        let entry = &self.entries[slot];
        if entry.pc != pc {
            return None;
        }
        let live = memory.plain_upper_rom_word(pc, usize::from(entry.len))?;
        (live == entry.word).then_some((entry.word as u8, entry.timing, slot))
    }

    /// The decode memo of a slot returned by `lookup`/`insert`. Valid only
    /// while that entry is not replaced.
    #[inline]
    pub(crate) fn memo_ptr(&mut self, slot: usize) -> *mut DecodeMemo {
        &mut self.memos[slot] as *mut DecodeMemo
    }

    /// Record a successful preflight of the instruction at `pc`, returning
    /// its slot when cached.
    pub(crate) fn insert(
        &mut self,
        pc: u32,
        len: u8,
        timing: PreparedInstructionTiming,
        memory: &MemoryImage,
    ) -> Option<usize> {
        let len_usize = usize::from(len);
        if len_usize == 0 || len_usize > MAX_CACHED_LEN || pc == u32::MAX {
            return None;
        }
        let word = memory.plain_upper_rom_word(pc, len_usize)?;
        if matches!(word as u8, 0x46 | 0x4E | 0xFE | 0xFF)
            || matches!(timing.resolved_opcode(), 0x46 | 0x4E | 0xFE | 0xFF)
        {
            return None;
        }
        let slot = Self::slot(pc);
        self.entries[slot] = Entry {
            pc,
            len,
            word,
            timing,
        };
        self.memos[slot] = DecodeMemo::default();
        Some(slot)
    }
}
