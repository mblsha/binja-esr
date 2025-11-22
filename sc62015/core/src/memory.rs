use crate::{CoreError, Result};
use std::env;

pub const INTERNAL_MEMORY_START: u32 = 0x100000;
pub const ADDRESS_MASK: u32 = 0x00FF_FFFF;
pub const INTERNAL_ADDR_MASK: u32 = 0xFF;
pub const EXTERNAL_SPACE: usize = 0x100000;
pub const INTERNAL_SPACE: usize = 0x100;
pub const INTERNAL_RAM_START: usize = 0xB8000;
pub const INTERNAL_RAM_SIZE: usize = 0x8000;

#[derive(Clone)]
pub struct MemoryImage {
    external: Vec<u8>,
    dirty: Vec<(u32, u8)>,
    dirty_internal: Vec<(u32, u8)>,
    python_ranges: Vec<(u32, u32)>,
    readonly_ranges: Vec<(u32, u32)>,
    internal: [u8; INTERNAL_SPACE],
    keyboard_bridge: bool,
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
        Self {
            external: vec![0; EXTERNAL_SPACE],
            dirty: Vec::new(),
            dirty_internal: Vec::new(),
            python_ranges: Vec::new(),
            readonly_ranges: Vec::new(),
            internal: Self::default_internal(),
            keyboard_bridge: false,
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
        if Self::is_internal(address) {
            let offset = (address - INTERNAL_MEMORY_START) & INTERNAL_ADDR_MASK;
            // Keyboard registers (KOL/KOH/KIL) are local when the bridge is enabled.
            if matches!(offset, 0xF0 | 0xF1 | 0xF2) && self.keyboard_bridge {
                return false;
            }
            // Keyboard matrix (when not bridged) and E-port input registers require
            // host-side handlers that emulate dynamic hardware state.
            if matches!(offset, 0xF0 | 0xF1 | 0xF2 | 0xF5 | 0xF6) {
                return true;
            }
            return false;
        }
        if address >= EXTERNAL_SPACE as u32 {
            return true;
        }
        let mut in_python_range = false;
        for (start, end) in &self.python_ranges {
            if address >= *start && address <= *end {
                in_python_range = true;
                break;
            }
        }
        if env::var("RUST_ROM_DEBUG").is_ok()
            && (0x0F2BD0..=0x0F2BD8).contains(&address)
        {
            eprintln!(
                "[rom-debug-route] addr=0x{addr:06X} python_range={range}",
                addr = address,
                range = in_python_range
            );
        }
        in_python_range
    }

    pub fn read_byte(&self, address: u32) -> Option<u8> {
        if let Some(index) = Self::internal_index(address) {
            return Some(self.internal[index]);
        }
        self.external.get(address as usize).copied()
    }

    pub fn load(&self, address: u32, bits: u8) -> Option<u32> {
        if let Some(value) = self.load_internal_value(address, bits) {
            return Some(value);
        }
        let bytes = (bits / 8).max(1) as usize;
        let end = address as usize + bytes;
        if end > self.external.len() {
            return None;
        }
        let mut value = 0u32;
        for offset in 0..bytes {
            value |= (self.external[address as usize + offset] as u32) << (offset * 8);
        }
        if env::var("RUST_ROM_DEBUG").is_ok()
            && (0x0F2BD0..=0x0F2BD8).contains(&address)
        {
            eprintln!(
                "[rom-debug-read] addr=0x{addr:06X} bits={bits} value=0x{val:06X}",
                addr = address,
                bits = bits,
                val = value & mask_bits(bits),
            );
        }
        if env::var("RUST_ROM_TRACE").is_ok()
            && (0x00F2_000..=0x00F3_000).contains(&address)
        {
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
        if self.store_internal_value(address, bits, value).is_some() {
            return Some(());
        }
        let bytes = (bits / 8).max(1) as usize;
        let end = address as usize + bytes;
        if end > self.external.len() {
            return None;
        }
        if self.is_read_only_range(address, bytes as u32) {
            return Some(());
        }
        for offset in 0..bytes {
            let byte = ((value >> (offset * 8)) & 0xFF) as u8;
            let slot = &mut self.external[address as usize + offset];
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

    pub fn apply_host_write(&mut self, address: u32, value: u8) {
        if let Some(index) = Self::internal_index(address) {
            self.internal[index] = value;
            self.dirty_internal.push((address, value));
            return;
        }
        if self.is_read_only_range(address, 1) {
            return;
        }
        let addr = (address as usize) & (EXTERNAL_SPACE - 1);
        if let Some(slot) = self.external.get_mut(addr) {
            *slot = value;
        }
    }

    pub fn write_external_byte(&mut self, address: u32, value: u8) {
        if address >= EXTERNAL_SPACE as u32 {
            return;
        }
        let idx = address as usize;
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
        address >= INTERNAL_MEMORY_START
            && address < INTERNAL_MEMORY_START + INTERNAL_SPACE as u32
    }

    pub fn internal_index(address: u32) -> Option<usize> {
        if Self::is_internal(address) {
            Some((address - INTERNAL_MEMORY_START) as usize)
        } else {
            None
        }
    }

    pub fn internal_offset(address: u32) -> Option<u32> {
        Self::internal_index(address).map(|idx| idx as u32)
    }

    pub fn is_keyboard_offset(offset: u32) -> bool {
        matches!(offset, 0xF0 | 0xF1 | 0xF2)
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
        Some(value)
    }

    pub fn write_internal_byte(&mut self, offset: u32, value: u8) {
        if offset < INTERNAL_SPACE as u32 {
            let index = offset as usize;
            if self.internal[index] != value {
                self.internal[index] = value;
                self.dirty_internal
                    .push((INTERNAL_MEMORY_START + offset, value));
            }
        }
    }

    pub fn read_internal_byte(&self, offset: u32) -> Option<u8> {
        if offset < INTERNAL_SPACE as u32 {
            Some(self.internal[offset as usize])
        } else {
            None
        }
    }

    pub fn is_read_only_range(&self, start: u32, len: u32) -> bool {
        if len == 0 {
            return false;
        }
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
                self.dirty_internal.push((address + offset as u32, byte));
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
        let end =
            (INTERNAL_RAM_START + INTERNAL_RAM_SIZE).min(self.external.len());
        if INTERNAL_RAM_START >= end {
            &[]
        } else {
            &self.external[INTERNAL_RAM_START..end]
        }
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
