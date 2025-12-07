// PY_SOURCE: pce500/memory.py:PCE500Memory

use crate::{CoreError, Result};
use std::cell::Cell;
use std::env;

pub const INTERNAL_MEMORY_START: u32 = 0x100000;
pub const ADDRESS_MASK: u32 = 0x00FF_FFFF;
pub const INTERNAL_ADDR_MASK: u32 = 0xFF;
pub const EXTERNAL_SPACE: usize = 0x100000;
pub const INTERNAL_SPACE: usize = 0x100;
pub const INTERNAL_RAM_START: usize = 0xB8000;
pub const INTERNAL_RAM_SIZE: usize = 0x8000;
pub const IMEM_KOL_OFFSET: u32 = 0xF0;
pub const IMEM_KOH_OFFSET: u32 = 0xF1;
pub const IMEM_KIL_OFFSET: u32 = 0xF2;
pub const IMEM_BP_OFFSET: u32 = 0xEC;
pub const IMEM_PX_OFFSET: u32 = 0xED;
pub const IMEM_PY_OFFSET: u32 = 0xEE;
pub const IMEM_UCR_OFFSET: u32 = 0xF7;
pub const IMEM_USR_OFFSET: u32 = 0xF8;
pub const IMEM_IMR_OFFSET: u32 = 0xFB;
pub const IMEM_ISR_OFFSET: u32 = 0xFC;
pub const IMEM_SCR_OFFSET: u32 = 0xFD;
pub const IMEM_LCC_OFFSET: u32 = 0xFE;
pub const IMEM_SSR_OFFSET: u32 = 0xFF;

fn canonical_address(address: u32) -> u32 {
    address & ADDRESS_MASK
}

thread_local! {
    static IMR_READ_SUPPRESS: Cell<bool> = Cell::new(false);
}

/// Run `f` with IMR read tracing/logging suppressed. Used for perfetto sampling paths that
/// should not emit IMR_Read events.
pub fn with_imr_read_suppressed<F, T>(f: F) -> T
where
    F: FnOnce() -> T,
{
    IMR_READ_SUPPRESS.with(|flag| {
        let prev = flag.replace(true);
        let res = f();
        flag.set(prev);
        res
    })
}

pub fn imr_read_suppressed() -> bool {
    IMR_READ_SUPPRESS.with(|flag| flag.get())
}

#[derive(Clone)]
pub struct MemoryImage {
    external: Vec<u8>,
    dirty: Vec<(u32, u8)>,
    dirty_internal: Vec<(u32, u8)>,
    python_ranges: Vec<(u32, u32)>,
    readonly_ranges: Vec<(u32, u32)>,
    internal: [u8; INTERNAL_SPACE],
    keyboard_bridge: bool,
    memory_reads: Cell<u64>,
    memory_writes: Cell<u64>,
}

impl Default for MemoryImage {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryImage {
    fn default_internal() -> [u8; INTERNAL_SPACE] {
        [0u8; INTERNAL_SPACE]
    }

    pub fn new() -> Self {
        let mut internal = Self::default_internal();
        // Default IMR cleared to mirror Python power-on/reset state; ROM bootstrap sets IMR later.
        internal[0xFB] = 0x00;
        Self {
            external: vec![0; EXTERNAL_SPACE],
            dirty: Vec::new(),
            dirty_internal: Vec::new(),
            python_ranges: Vec::new(),
            readonly_ranges: Vec::new(),
            internal,
            keyboard_bridge: false,
            memory_reads: Cell::new(0),
            memory_writes: Cell::new(0),
        }
    }

    pub fn load_external(&mut self, blob: &[u8]) {
        let limit = self.external.len().min(blob.len());
        self.external[..limit].copy_from_slice(&blob[..limit]);
        self.dirty.clear();
        if env::var("RUST_ROM_DEBUG").is_ok() {
            let addr = 0x0F2BD0usize.min(self.external.len().saturating_sub(8));
            let window = &self.external[addr..addr + 8];
            let hex = window
                .iter()
                .map(|byte| format!("{byte:02X}"))
                .collect::<Vec<_>>()
                .join(" ");
            eprintln!("[rom-debug-load] addr=0x{addr:06X} bytes={hex}");
        }
    }

    pub fn load_internal(&mut self, blob: &[u8]) {
        let limit = self.internal.len().min(blob.len());
        self.internal[..limit].copy_from_slice(&blob[..limit]);
    }

    pub fn set_python_ranges(&mut self, ranges: Vec<(u32, u32)>) {
        self.python_ranges = ranges;
    }

    pub fn python_ranges(&self) -> &[(u32, u32)] {
        &self.python_ranges
    }

    pub fn set_readonly_ranges(&mut self, ranges: Vec<(u32, u32)>) {
        self.readonly_ranges = ranges;
    }

    pub fn readonly_ranges(&self) -> &[(u32, u32)] {
        &self.readonly_ranges
    }

    pub fn set_keyboard_bridge(&mut self, enabled: bool) {
        self.keyboard_bridge = enabled;
    }

    pub fn keyboard_bridge(&self) -> bool {
        self.keyboard_bridge
    }

    pub fn requires_python(&self, address: u32) -> bool {
        let address = canonical_address(address);
        if Self::is_internal(address) {
            let offset = (address - INTERNAL_MEMORY_START) & INTERNAL_ADDR_MASK;
            // Keyboard registers (KOL/KOH/KIL) are local when the bridge is enabled.
            if matches!(offset, 0xF0..=0xF2) && self.keyboard_bridge {
                return false;
            }
            // Keyboard matrix (when not bridged) and E-port input registers require
            // host-side handlers that emulate dynamic hardware state.
            if matches!(offset, 0xF0..=0xF2 | 0xF5 | 0xF6) {
                return true;
            }
            // LCD controller overlay addresses (internal remap used by Python)
            if matches!(offset, 0x00..=0x0F) {
                return false;
            }
            return false;
        }
        let mut in_python_range = false;
        for (start, end) in &self.python_ranges {
            if address >= *start && address <= *end {
                in_python_range = true;
                break;
            }
        }
        if env::var("RUST_ROM_DEBUG").is_ok() && (0x0F2BD0..=0x0F2BD8).contains(&address) {
            eprintln!(
                "[rom-debug-route] addr=0x{addr:06X} python_range={range}",
                addr = address,
                range = in_python_range
            );
        }
        in_python_range || address >= EXTERNAL_SPACE as u32
    }

    pub fn read_byte(&self, address: u32) -> Option<u8> {
        self.memory_reads
            .set(self.memory_reads.get().saturating_add(1));
        let address = canonical_address(address);
        if let Some(index) = Self::internal_index(address) {
            // Optional bridge: allow external-memory writes to mirror into internal for diagnostics.
            return Some(self.internal[index]);
        }
        // External memory fallback with wrap
        let idx = (address as usize) & (EXTERNAL_SPACE - 1);
        self.external.get(idx).copied()
    }

    pub fn load(&self, address: u32, bits: u8) -> Option<u32> {
        self.memory_reads
            .set(self.memory_reads.get().saturating_add(1));
        let address = canonical_address(address);
        if let Some(value) = self.load_internal_value(address, bits) {
            return Some(value);
        }
        let bytes = (bits / 8).max(1) as usize;
        let mut value = 0u32;
        for offset in 0..bytes {
            let idx = (address as usize + offset) & (EXTERNAL_SPACE - 1);
            value |= (self.external[idx] as u32) << (offset * 8);
        }
        if env::var("RUST_ROM_DEBUG").is_ok() && (0x0F2BD0..=0x0F2BD8).contains(&address) {
            eprintln!(
                "[rom-debug-read] addr=0x{addr:06X} bits={bits} value=0x{val:06X}",
                addr = address,
                bits = bits,
                val = value & mask_bits(bits),
            );
        }
        if env::var("RUST_ROM_TRACE").is_ok() && (0x000F_2000..=0x000F_3000).contains(&address) {
            let mask = mask_bits(bits);
            println!(
                "[rom-trace] addr=0x{addr:06X} bits={bits} value=0x{val:06X}",
                addr = address & ADDRESS_MASK,
                bits = bits,
                val = value & mask,
            );
        }
        Some(value)
    }

    pub fn store(&mut self, address: u32, bits: u8, value: u32) -> Option<()> {
        self.memory_writes
            .set(self.memory_writes.get().saturating_add(1));
        let address = canonical_address(address);
        if self.store_internal_value(address, bits, value).is_some() {
            return Some(());
        }
        let bytes = (bits / 8).max(1) as usize;
        if self.is_read_only_range(address, bytes as u32) {
            return Some(());
        }
        for offset in 0..bytes {
            let byte = ((value >> (offset * 8)) & 0xFF) as u8;
            let slot = &mut self.external[(address as usize + offset) & (EXTERNAL_SPACE - 1)];
            if *slot != byte {
                *slot = byte;
                self.dirty.push((address + offset as u32, byte));
            }
        }
        Some(())
    }

    pub fn drain_dirty(&mut self) -> Vec<(u32, u8)> {
        std::mem::take(&mut self.dirty)
    }

    /// Apply a host-driven write (e.g., overlay/bridge) and optionally tag it with a manual-clock cycle for Perfetto.
    pub fn apply_host_write_with_cycle(&mut self, address: u32, value: u8, cycle: Option<u64>) {
        self.memory_writes
            .set(self.memory_writes.get().saturating_add(1));
        let address = canonical_address(address);
        if let Some(index) = Self::internal_index(address) {
            self.internal[index] = value;
            self.dirty_internal.push((address, value));
            if let Ok(mut guard) = crate::PERFETTO_TRACER.lock() {
                if let Some(tracer) = guard.as_mut() {
                    if let Some(cyc) = cycle {
                        tracer.record_mem_write_at_cycle(
                            cyc,
                            Some(crate::llama::eval::perfetto_last_pc()),
                            address,
                            value as u32,
                            "internal",
                            8,
                        );
                    } else {
                        // Use last completed instruction index to avoid emitting unmatched op_index values.
                        let (seq, pc) = crate::llama::eval::perfetto_instr_context()
                            .unwrap_or_else(|| {
                                (
                                    crate::llama::eval::perfetto_last_instr_index(),
                                    crate::llama::eval::perfetto_last_pc(),
                                )
                            });
                        tracer.record_mem_write(seq, pc, address, value as u32, "internal", 8);
                    }
                }
            }
            return;
        }
        if self.is_read_only_range(address, 1) {
            return;
        }
        let addr = (address as usize) & (EXTERNAL_SPACE - 1);
        if let Some(slot) = self.external.get_mut(addr) {
            if *slot != value {
                *slot = value;
                self.dirty.push((address, value));
            }
        }
    }

    pub fn apply_host_write(&mut self, address: u32, value: u8) {
        self.apply_host_write_with_cycle(address, value, None);
    }

    pub fn write_external_byte(&mut self, address: u32, value: u8) {
        self.memory_writes
            .set(self.memory_writes.get().saturating_add(1));
        let address = canonical_address(address);
        let idx = (address as usize) & (EXTERNAL_SPACE - 1);
        if self.external[idx] != value {
            self.external[idx] = value;
            self.dirty.push((address, value));
        }
    }

    pub fn write_external_slice(&mut self, start: usize, data: &[u8]) {
        if start >= self.external.len() {
            return;
        }
        let end = (start + data.len()).min(self.external.len());
        if end > start {
            self.external[start..end].copy_from_slice(&data[..(end - start)]);
        }
    }

    pub fn is_internal(address: u32) -> bool {
        let address = canonical_address(address);
        address >= INTERNAL_MEMORY_START && address < INTERNAL_MEMORY_START + INTERNAL_SPACE as u32
    }

    pub fn internal_index(address: u32) -> Option<usize> {
        let address = canonical_address(address);
        if address >= INTERNAL_MEMORY_START
            && address < INTERNAL_MEMORY_START + INTERNAL_SPACE as u32
        {
            return Some((address - INTERNAL_MEMORY_START) as usize);
        }
        None
    }

    pub fn internal_offset(address: u32) -> Option<u32> {
        Self::internal_index(address).map(|idx| idx as u32)
    }

    pub fn is_keyboard_offset(offset: u32) -> bool {
        matches!(offset, 0xF0..=0xF2)
    }

    pub fn load_internal_value(&self, address: u32, bits: u8) -> Option<u32> {
        let bytes = (bits / 8).max(1) as usize;
        let index = Self::internal_index(address)?;
        if index + bytes > self.internal.len() {
            return None;
        }
        let mut value = 0u32;
        for offset in 0..bytes {
            value |= (self.internal[index + offset] as u32) << (offset * 8);
        }
        // Optional debug for IMR reads to diagnose IMR/IRM coherence.
        if address == INTERNAL_MEMORY_START + 0xFB && !imr_read_suppressed() {
            if let Ok(env) = std::env::var("IMR_READ_DEBUG") {
                if env == "1" {
                    eprintln!(
                        "[imr-read] addr=0x{address:06X} bits={bits} value=0x{val:02X}",
                        val = value & mask_bits(bits)
                    );
                }
            }
            if let Ok(mut guard) = crate::PERFETTO_TRACER.lock() {
                if let Some(tracer) = guard.as_mut() {
                    let ctx = crate::llama::eval::perfetto_instr_context();
                    let (op_idx, pc) = ctx.unwrap_or((
                        crate::llama::eval::perfetto_last_instr_index(),
                        crate::llama::eval::perfetto_last_pc(),
                    ));
                    tracer.record_imr_read(
                        if op_idx == u64::MAX { None } else { Some(pc) },
                        value as u8,
                        if op_idx == u64::MAX {
                            None
                        } else {
                            Some(op_idx)
                        },
                    );
                }
            }
        }
        Some(value)
    }

    pub fn write_internal_byte(&mut self, offset: u32, value: u8) {
        if offset < INTERNAL_SPACE as u32 {
            let index = offset as usize;
            self.memory_writes
                .set(self.memory_writes.get().saturating_add(1));
            if self.internal[index] != value {
                self.internal[index] = value;
                self.dirty_internal
                    .push((INTERNAL_MEMORY_START + offset, value));
                if let Ok(mut guard) = crate::PERFETTO_TRACER.lock() {
                    if let Some(tracer) = guard.as_mut() {
                        let (seq, pc) = crate::llama::eval::perfetto_instr_context()
                            .unwrap_or_else(|| {
                                (crate::llama::eval::perfetto_last_instr_index(), 0)
                            });
                        tracer.record_mem_write(
                            seq,
                            pc,
                            INTERNAL_MEMORY_START + offset,
                            value as u32,
                            "internal",
                            8,
                        );
                        // Diagnostic: emit KEYI_Set via perfetto when ISR is written with KEYI set.
                        if offset == 0xFC && (value & 0x04) != 0 {
                            tracer.record_keyi_set(
                                INTERNAL_MEMORY_START + offset,
                                value,
                                Some(seq),
                                Some(pc),
                            );
                        }
                    }
                }
            }
        }
    }

    pub fn read_internal_byte(&self, offset: u32) -> Option<u8> {
        if offset < INTERNAL_SPACE as u32 {
            self.memory_reads
                .set(self.memory_reads.get().saturating_add(1));
            let val = self.internal[offset as usize];
            // Optional debug hook: enable IMR read logging with IMR_READ_DEBUG=1.
            if offset == 0xFB && !imr_read_suppressed() {
                if let Ok(env) = std::env::var("IMR_READ_DEBUG") {
                    if env == "1" {
                        eprintln!("[imr-read] offset=0x{offset:02X} val=0x{val:02X}");
                    }
                }
                if let Ok(mut guard) = crate::PERFETTO_TRACER.try_lock() {
                    if let Some(tracer) = guard.as_mut() {
                        let ctx = crate::llama::eval::perfetto_instr_context();
                        let (op_idx, pc) = ctx.unwrap_or((
                            crate::llama::eval::perfetto_last_instr_index(),
                            crate::llama::eval::perfetto_last_pc(),
                        ));
                        tracer.record_imr_read(
                            if op_idx == u64::MAX { None } else { Some(pc) },
                            val,
                            if op_idx == u64::MAX {
                                None
                            } else {
                                Some(op_idx)
                            },
                        );
                    }
                }
            }
            if (0xF0..=0xF2).contains(&offset) {
                self.log_kio_read(offset, val);
            }
            Some(val)
        } else {
            None
        }
    }

    /// Read an internal byte without emitting perfetto diagnostics. Intended for
    /// tracing-only snapshots (e.g., IMR/ISR sampling) to avoid creating extra
    /// IMR_Read events.
    pub fn read_internal_byte_silent(&self, offset: u32) -> Option<u8> {
        if offset < INTERNAL_SPACE as u32 {
            Some(self.internal[offset as usize])
        } else {
            None
        }
    }

    pub fn bump_read_count(&self) {
        self.memory_reads
            .set(self.memory_reads.get().saturating_add(1));
    }

    /// Perfetto/logging helper for KIO (KOL/KOH/KIL) reads, preserving instruction context.
    pub fn log_kio_read(&self, offset: u32, value: u8) {
        if let Ok(mut guard) = crate::PERFETTO_TRACER.try_lock() {
            if let Some(tracer) = guard.as_mut() {
                let ctx = crate::llama::eval::perfetto_instr_context();
                let (op_idx, pc) = ctx.unwrap_or((
                    crate::llama::eval::perfetto_last_instr_index(),
                    crate::llama::eval::perfetto_last_pc(),
                ));
                tracer.record_kio_read(
                    if op_idx == u64::MAX { None } else { Some(pc) },
                    offset as u8,
                    value,
                    if op_idx == u64::MAX {
                        None
                    } else {
                        Some(op_idx)
                    },
                );
            }
        }
        if offset == 0xF2 {
            if let Ok(env) = std::env::var("KIL_READ_DEBUG") {
                if env == "1" {
                    eprintln!("[kil-read-rust] offset=0x{offset:02X} val=0x{value:02X}");
                }
            }
        }
    }

    pub fn is_read_only_range(&self, start: u32, len: u32) -> bool {
        if len == 0 {
            return false;
        }
        let start = canonical_address(start);
        let end = start.saturating_add(len.saturating_sub(1));
        for (range_start, range_end) in &self.readonly_ranges {
            if start <= *range_end && end >= *range_start {
                return true;
            }
        }
        false
    }

    pub fn store_internal_value(&mut self, address: u32, bits: u8, value: u32) -> Option<()> {
        let bytes = (bits / 8).max(1) as usize;
        let index = Self::internal_index(address)?;
        if index + bytes > self.internal.len() {
            return None;
        }
        for offset in 0..bytes {
            let byte = ((value >> (offset * 8)) & 0xFF) as u8;
            let slot = &mut self.internal[index + offset];
            if *slot != byte {
                *slot = byte;
                self.dirty_internal
                    .push((INTERNAL_MEMORY_START + offset as u32, byte));
            }
        }
        Some(())
    }

    pub fn drain_dirty_internal(&mut self) -> Vec<(u32, u8)> {
        std::mem::take(&mut self.dirty_internal)
    }

    pub fn clear_dirty(&mut self) {
        self.dirty.clear();
        self.dirty_internal.clear();
    }

    pub fn external_len(&self) -> usize {
        self.external.len()
    }

    pub fn external_slice(&self) -> &[u8] {
        &self.external
    }

    pub fn internal_slice(&self) -> &[u8] {
        &self.internal
    }

    pub fn internal_ram_slice(&self) -> &[u8] {
        let end = (INTERNAL_RAM_START + INTERNAL_RAM_SIZE).min(self.external.len());
        if INTERNAL_RAM_START >= end {
            &[]
        } else {
            &self.external[INTERNAL_RAM_START..end]
        }
    }

    pub fn memory_read_count(&self) -> u64 {
        self.memory_reads.get()
    }

    pub fn memory_write_count(&self) -> u64 {
        self.memory_writes.get()
    }

    pub fn set_memory_counts(&self, reads: u64, writes: u64) {
        self.memory_reads.set(reads);
        self.memory_writes.set(writes);
    }

    pub fn copy_external_from(&mut self, data: &[u8]) -> Result<()> {
        if data.len() != self.external.len() {
            return Err(CoreError::InvalidSnapshot(format!(
                "external_ram.bin size mismatch (expected {}, got {})",
                self.external.len(),
                data.len()
            )));
        }
        self.external.copy_from_slice(data);
        Ok(())
    }

    pub fn write_internal_ram(&mut self, start: u32, payload: &[u8]) {
        let start = start as usize;
        if start >= self.external.len() {
            return;
        }
        let end = (start + payload.len()).min(self.external.len());
        let span = end.saturating_sub(start);
        if span > 0 {
            self.external[start..end].copy_from_slice(&payload[..span]);
        }
    }

    pub fn write_imem(&mut self, payload: &[u8]) {
        let limit = self.internal.len().min(payload.len());
        self.internal[..limit].copy_from_slice(&payload[..limit]);
    }

    pub fn external_segment(&self, start: usize, length: usize) -> Option<&[u8]> {
        if start >= self.external.len() {
            return None;
        }
        let end = (start + length).min(self.external.len());
        Some(&self.external[start..end])
    }
}

fn mask_bits(bits: u8) -> u32 {
    if bits >= 32 {
        u32::MAX
    } else if bits == 0 {
        0
    } else {
        (1u32 << bits) - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn external_access_wraps_24bit() {
        let mut mem = MemoryImage::new();
        let addr = (EXTERNAL_SPACE as u32) + 0x10;
        assert_eq!(mem.store(addr, 8, 0xAA), Some(()));
        assert_eq!(mem.load(addr & ADDRESS_MASK, 8), Some(0xAA));
        assert_eq!(mem.load(addr, 8), Some(0xAA));
    }

    #[test]
    fn internal_index_masks_address() {
        let base = INTERNAL_MEMORY_START;
        assert_eq!(MemoryImage::internal_index(base), Some(0));
        assert_eq!(MemoryImage::internal_index(base | 0xFF00_0000), Some(0));
    }

    #[test]
    fn requires_python_masks_address() {
        let mut mem = MemoryImage::new();
        mem.set_python_ranges(vec![(0x1000, 0x1FFF)]);
        assert!(mem.requires_python(0x0010_0100));
        assert!(mem.requires_python(0x1010_0100));
    }

    #[test]
    fn external_reset_vector_is_not_aliased() {
        let mut mem = MemoryImage::new();
        // Populate the ROM reset vector region (0x0FFFFA-0x0FFFFC) in external space.
        mem.write_external_byte(0x0FFFFA, 0x11);
        mem.write_external_byte(0x0FFFFB, 0x22);
        mem.write_external_byte(0x0FFFFC, 0x33);

        assert!(!MemoryImage::is_internal(0x0FFFFA));
        assert_eq!(MemoryImage::internal_index(0x0FFFFA), None);
        assert_eq!(mem.load(0x0FFFFA, 8), Some(0x11));
        assert_eq!(mem.load(0x0FFFFB, 8), Some(0x22));
        assert_eq!(mem.load(0x0FFFFC, 8), Some(0x33));
    }

    #[test]
    fn default_imr_matches_python_power_on() {
        let mem = MemoryImage::new();
        let imr = mem
            .load(INTERNAL_MEMORY_START + IMEM_IMR_OFFSET, 8)
            .unwrap_or(0xFF);
        assert_eq!(imr, 0x00, "IMR should start cleared like Python reset()");
    }

    #[test]
    fn memory_read_write_counters_increment() {
        let mut mem = MemoryImage::new();
        // External write and read.
        let _ = mem.store(0x0000, 8, 0xAA);
        let _ = mem.load(0x0000, 8);
        // Direct internal read should also bump counters.
        let _ = mem.read_internal_byte(IMEM_IMR_OFFSET);
        assert_eq!(mem.memory_write_count(), 1);
        assert_eq!(mem.memory_read_count(), 2);
    }

    #[test]
    fn apply_host_write_marks_external_dirty() {
        let mut mem = MemoryImage::new();
        mem.apply_host_write_with_cycle(0x0010, 0xBE, Some(0));
        let dirty = mem.drain_dirty();
        assert_eq!(dirty, vec![(0x0010, 0xBE)]);
        // Ensure subsequent apply with same value does not duplicate entries.
        mem.apply_host_write_with_cycle(0x0010, 0xBE, Some(1));
        let dirty_after = mem.drain_dirty();
        assert!(dirty_after.is_empty());
    }
}
