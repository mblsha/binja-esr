// PY_SOURCE: pce500/memory.py:PCE500Memory

use crate::{llama::eval::perfetto_last_pc, CoreError, Result};
use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::rc::Rc;

type ImrIsrHook = Rc<dyn Fn(u32, u8, u8)>;

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
const OVERLAY_LOG_LIMIT: usize = 256;
const MEMORY_CARD_RANGES: &[(usize, u32, u32, &str)] = &[
    // (size bytes, start, end, perfetto thread)
    (8192, 0x040000, 0x041FFF, "Memory_Card"),
    (16384, 0x040000, 0x043FFF, "Memory_Card"),
    (32768, 0x040000, 0x047FFF, "Memory_Card"),
    (65536, 0x040000, 0x04FFFF, "Memory_Card"),
];

fn canonical_address(address: u32) -> u32 {
    address & ADDRESS_MASK
}

thread_local! {
    static IMR_READ_SUPPRESS: Cell<bool> = Cell::new(false);
}

fn perfetto_guard() -> crate::PerfettoGuard<'static> {
    crate::PERFETTO_TRACER.enter()
}

fn perfetto_context_or_last() -> (u64, u32) {
    if let Some((op_idx, pc)) = crate::llama::eval::perfetto_instr_context() {
        (op_idx, pc)
    } else {
        (
            crate::llama::eval::perfetto_last_instr_index(),
            crate::llama::eval::perfetto_last_pc(),
        )
    }
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AccessKind {
    Read,
    Write,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryAccessLog {
    pub kind: AccessKind,
    pub address: u32,
    pub value: u8,
    pub overlay: String,
    pub pc: Option<u32>,
    pub previous: Option<u8>,
}

type OverlayReadHandler = Box<dyn Fn(u32, Option<u32>) -> Option<u8>>;
type OverlayWriteHandler = Box<dyn Fn(u32, u8, Option<u32>) -> bool>;

pub struct MemoryOverlay {
    pub start: u32,
    pub end: u32,
    pub name: String,
    pub data: Option<Vec<u8>>,
    pub read_only: bool,
    pub read_handler: Option<OverlayReadHandler>,
    pub write_handler: Option<OverlayWriteHandler>,
    pub perfetto_thread: Option<String>,
}

impl MemoryOverlay {
    pub fn contains(&self, address: u32) -> bool {
        let addr = canonical_address(address);
        addr >= self.start && addr <= self.end
    }

    fn offset(&self, address: u32) -> Option<usize> {
        let addr = canonical_address(address);
        addr.checked_sub(self.start).and_then(|off| usize::try_from(off).ok())
    }

    fn read(&self, address: u32, pc: Option<u32>) -> Option<u8> {
        if let Some(handler) = self.read_handler.as_ref() {
            if let Some(val) = handler(address, pc) {
                return Some(val & 0xFF);
            }
        }
        if let (Some(data), Some(offset)) = (self.data.as_ref(), self.offset(address)) {
            return data.get(offset).copied();
        }
        None
    }

    fn write(&mut self, address: u32, value: u8, pc: Option<u32>) -> (bool, Option<u8>) {
        if let Some(handler) = self.write_handler.as_mut() {
            let handled = handler(address, value, pc);
            return (handled, None);
        }
        if let Some(offset) = self.offset(address) {
            if let Some(data) = self.data.as_mut() {
                if offset < data.len() {
                    let previous = data[offset];
                    if !self.read_only {
                        data[offset] = value;
                    }
                    return (true, Some(previous));
                }
            }
            if self.read_only {
                return (true, None);
            }
        }
        (false, None)
    }
}

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
    imr_isr_hook: Option<ImrIsrHook>,
    overlays: Vec<MemoryOverlay>,
    read_log: RefCell<VecDeque<MemoryAccessLog>>,
    write_log: RefCell<VecDeque<MemoryAccessLog>>,
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
            imr_isr_hook: None,
            overlays: Vec::new(),
            read_log: RefCell::new(VecDeque::with_capacity(OVERLAY_LOG_LIMIT)),
            write_log: RefCell::new(VecDeque::with_capacity(OVERLAY_LOG_LIMIT)),
        }
    }

    /// Install a callback invoked whenever IMR/ISR are written. Used to keep IRQ bit-watch,
    /// mirror fields, and Perfetto diagnostics in sync with Python.
    pub fn set_imr_isr_hook<F>(&mut self, hook: Option<F>)
    where
        F: Fn(u32, u8, u8) + 'static,
    {
        self.imr_isr_hook = hook.map(|h| Rc::new(h) as ImrIsrHook);
    }

    fn invoke_imr_isr_hook(&self, offset: u32, prev: u8, new: u8) {
        if offset != IMEM_IMR_OFFSET && offset != IMEM_ISR_OFFSET {
            return;
        }
        if let Some(hook) = self.imr_isr_hook.as_ref() {
            hook.as_ref()(offset, prev, new);
        }
    }

    pub fn load_external(&mut self, blob: &[u8]) {
        let limit = self.external.len().min(blob.len());
        self.external[..limit].copy_from_slice(&blob[..limit]);
        self.dirty.clear();
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

    pub fn add_overlay(&mut self, overlay: MemoryOverlay) {
        self.overlays.push(overlay);
        self.overlays
            .sort_by(|a, b| (a.start, a.end, &a.name).cmp(&(b.start, b.end, &b.name)));
    }

    pub fn remove_overlay(&mut self, name: &str) {
        self.overlays.retain(|ov| ov.name != name);
    }

    pub fn overlays(&self) -> &[MemoryOverlay] {
        &self.overlays
    }

    pub fn add_ram_overlay(&mut self, start: u32, size: usize, name: &str) {
        if size == 0 {
            return;
        }
        let end = start.saturating_add(size.saturating_sub(1) as u32);
        self.remove_overlay(name);
        self.add_overlay(MemoryOverlay {
            start,
            end,
            name: name.to_string(),
            data: Some(vec![0u8; size]),
            read_only: false,
            read_handler: None,
            write_handler: None,
            perfetto_thread: Some("Memory_RAM".to_string()),
        });
    }

    pub fn add_rom_overlay(&mut self, start: u32, data: &[u8], name: &str) {
        if data.is_empty() {
            return;
        }
        let end = start.saturating_add(data.len().saturating_sub(1) as u32);
        self.remove_overlay(name);
        self.add_overlay(MemoryOverlay {
            start,
            end,
            name: name.to_string(),
            data: Some(data.to_vec()),
            read_only: true,
            read_handler: None,
            write_handler: None,
            perfetto_thread: Some("Memory_ROM".to_string()),
        });
    }

    pub fn load_memory_card(&mut self, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Err(CoreError::Other("memory card data is empty".to_string()));
        }
        let size = data.len();
        let Some((_, start, end, thread)) =
            MEMORY_CARD_RANGES.iter().find(|(len, _, _, _)| *len == size)
        else {
            return Err(CoreError::Other(format!(
                "unsupported memory card size: {size} bytes"
            )));
        };
        self.remove_overlay("memory_card");
        self.add_overlay(MemoryOverlay {
            start: *start,
            end: *end,
            name: "memory_card".to_string(),
            data: Some(data.to_vec()),
            read_only: true,
            read_handler: None,
            write_handler: None,
            perfetto_thread: Some(thread.to_string()),
        });
        Ok(())
    }

    pub fn clear_overlay_logs(&self) {
        self.read_log.borrow_mut().clear();
        self.write_log.borrow_mut().clear();
    }

    pub fn overlay_read_log(&self) -> Vec<MemoryAccessLog> {
        self.read_log.borrow().iter().cloned().collect()
    }

    pub fn overlay_write_log(&self) -> Vec<MemoryAccessLog> {
        self.write_log.borrow().iter().cloned().collect()
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
        self.load_with_pc(address, bits, None)
    }

    pub fn load_with_pc(&self, address: u32, bits: u8, pc: Option<u32>) -> Option<u32> {
        self.memory_reads
            .set(self.memory_reads.get().saturating_add(1));
        let address = canonical_address(address);
        if let Some(value) = self.load_internal_value(address, bits) {
            return Some(value);
        }
        if let Some(value) = self.load_overlay_value(address, bits, pc) {
            return Some(value);
        }
        let bytes = (bits / 8).max(1) as usize;
        let mut value = 0u32;
        for offset in 0..bytes {
            let idx = (address as usize + offset) & (EXTERNAL_SPACE - 1);
            value |= (self.external[idx] as u32) << (offset * 8);
        }
        Some(value)
    }

    pub fn store(&mut self, address: u32, bits: u8, value: u32) -> Option<()> {
        self.store_with_pc(address, bits, value, None)
    }

    pub fn store_with_pc(
        &mut self,
        address: u32,
        bits: u8,
        value: u32,
        pc: Option<u32>,
    ) -> Option<()> {
        self.memory_writes
            .set(self.memory_writes.get().saturating_add(1));
        let address = canonical_address(address);
        if self.store_internal_value(address, bits, value).is_some() {
            return Some(());
        }
        if self.store_overlay_value(address, bits, value, pc).is_some() {
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
    pub fn apply_host_write_with_cycle(
        &mut self,
        address: u32,
        value: u8,
        cycle: Option<u64>,
        pc: Option<u32>,
    ) {
        self.memory_writes
            .set(self.memory_writes.get().saturating_add(1));
        let address = canonical_address(address);
        if let Some(index) = Self::internal_index(address) {
            let offset = address - INTERNAL_MEMORY_START;
            let prev = self.internal[index];
            self.internal[index] = value;
            self.dirty_internal.push((address, value));
            self.invoke_imr_isr_hook(offset, prev, value);
            let mut guard = perfetto_guard();
            if let Some(tracer) = guard.as_mut() {
                match (cycle, crate::llama::eval::perfetto_instr_context()) {
                    (Some(cyc), _) => {
                        tracer.record_mem_write_at_cycle(
                            cyc,
                            pc,
                            address,
                            value as u32,
                            "internal",
                            8,
                        );
                    }
                    (None, Some((op_idx, pc_ctx))) => {
                        tracer.record_mem_write(
                            op_idx,
                            pc_ctx,
                            address,
                            value as u32,
                            "internal",
                            8,
                        );
                    }
                    _ => {
                        let pc = pc.or_else(|| Some(perfetto_last_pc()));
                        tracer.record_mem_write_at_cycle(
                            0,
                            pc,
                            address,
                            value as u32,
                            "host_async",
                            8,
                        );
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
            let mut guard = perfetto_guard();
            if let Some(tracer) = guard.as_mut() {
                match (cycle, crate::llama::eval::perfetto_instr_context()) {
                    (Some(cyc), _) => {
                        tracer.record_mem_write_at_cycle(
                            cyc,
                            pc,
                            address,
                            value as u32,
                            "external",
                            8,
                        );
                    }
                    (None, Some((op_idx, pc_ctx))) => {
                        tracer.record_mem_write(
                            op_idx,
                            pc_ctx,
                            address,
                            value as u32,
                            "external",
                            8,
                        );
                    }
                    _ => {
                        let pc = pc.or_else(|| Some(perfetto_last_pc()));
                        tracer.record_mem_write_at_cycle(
                            0,
                            pc,
                            address,
                            value as u32,
                            "host_async",
                            8,
                        );
                    }
                }
            }
        }
    }

    fn load_overlay_value(&self, address: u32, bits: u8, pc: Option<u32>) -> Option<u32> {
        let bytes = (bits / 8).max(1) as usize;
        let mut value = 0u32;
        for offset in 0..bytes {
            let addr = canonical_address(address + offset as u32);
            let mut handled = false;
            for overlay in &self.overlays {
                if !overlay.contains(addr) {
                    continue;
                }
                if let Some(byte) = overlay.read(addr, pc) {
                    self.push_overlay_log(AccessKind::Read, addr, byte, pc, &overlay.name, None);
                    value |= (byte as u32) << (offset * 8);
                    handled = true;
                    break;
                }
            }
            if !handled {
                return None;
            }
        }
        Some(value)
    }

    fn store_overlay_value(
        &mut self,
        address: u32,
        bits: u8,
        value: u32,
        pc: Option<u32>,
    ) -> Option<()> {
        let bytes = (bits / 8).max(1) as usize;
        for offset in 0..bytes {
            let addr = canonical_address(address + offset as u32);
            let byte = ((value >> (offset * 8)) & 0xFF) as u8;
            let mut handled = false;
            for idx in 0..self.overlays.len() {
                if !self.overlays[idx].contains(addr) {
                    continue;
                }
                let name = self.overlays[idx].name.clone();
                let (ok, previous) = {
                    let overlay = &mut self.overlays[idx];
                    overlay.write(addr, byte, pc)
                };
                if ok {
                    self.push_overlay_log(AccessKind::Write, addr, byte, pc, &name, previous);
                    let mut guard = perfetto_guard();
                    if let Some(tracer) = guard.as_mut() {
                        if let Some((op_idx, pc_ctx)) = crate::llama::eval::perfetto_instr_context()
                        {
                            tracer.record_mem_write(
                                op_idx,
                                pc_ctx,
                                addr,
                                byte as u32,
                                &name,
                                8,
                            );
                        } else {
                            tracer.record_mem_write_at_cycle(0, pc, addr, byte as u32, &name, 8);
                        }
                    }
                    handled = true;
                    break;
                }
            }
            if !handled {
                return None;
            }
        }
        Some(())
    }

    fn push_overlay_log(
        &self,
        kind: AccessKind,
        address: u32,
        value: u8,
        pc: Option<u32>,
        overlay: &str,
        previous: Option<u8>,
    ) {
        let log = MemoryAccessLog {
            kind,
            address,
            value,
            overlay: overlay.to_string(),
            pc,
            previous,
        };
        match log.kind {
            AccessKind::Read => {
                let mut guard = self.read_log.borrow_mut();
                if guard.len() == OVERLAY_LOG_LIMIT {
                    guard.pop_front();
                }
                guard.push_back(log);
            }
            AccessKind::Write => {
                let mut guard = self.write_log.borrow_mut();
                if guard.len() == OVERLAY_LOG_LIMIT {
                    guard.pop_front();
                }
                guard.push_back(log);
            }
        }
    }

    pub fn apply_host_write(&mut self, address: u32, value: u8) {
        self.apply_host_write_with_cycle(address, value, None, None);
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
        if address == INTERNAL_MEMORY_START + 0xFB && !imr_read_suppressed() {
            let mut guard = perfetto_guard();
            if let Some(tracer) = guard.as_mut() {
                let ctx = crate::llama::eval::perfetto_instr_context();
                let (op_idx, pc) = ctx.unwrap_or((
                    crate::llama::eval::perfetto_last_instr_index(),
                    crate::llama::eval::perfetto_last_pc(),
                ));
                tracer.record_imr_read(
                    if op_idx == u64::MAX { None } else { Some(pc) },
                    value as u8,
                    if op_idx == u64::MAX { None } else { Some(op_idx) },
                );
            }
        }
        Some(value)
    }

    pub fn write_internal_byte(&mut self, offset: u32, value: u8) {
        if offset < INTERNAL_SPACE as u32 {
            let index = offset as usize;
            self.memory_writes
                .set(self.memory_writes.get().saturating_add(1));
            let prev = self.internal[index];
            self.internal[index] = value;
            self.dirty_internal
                .push((INTERNAL_MEMORY_START + offset, value));
            self.invoke_imr_isr_hook(offset, prev, value);
            let mut guard = perfetto_guard();
            if let Some(tracer) = guard.as_mut() {
                let (seq, pc) = perfetto_context_or_last();
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

    pub fn read_internal_byte(&self, offset: u32) -> Option<u8> {
        if offset < INTERNAL_SPACE as u32 {
            self.memory_reads
                .set(self.memory_reads.get().saturating_add(1));
            let val = self.internal[offset as usize];
            if offset == 0xFB && !imr_read_suppressed() {
                let mut guard = perfetto_guard();
                if let Some(tracer) = guard.as_mut() {
                    let (op_idx, pc) = perfetto_context_or_last();
                    tracer.record_imr_read(
                        if op_idx == u64::MAX { None } else { Some(pc) },
                        val,
                        if op_idx == u64::MAX { None } else { Some(op_idx) },
                    );
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

    pub fn bump_write_count(&self) {
        self.memory_writes
            .set(self.memory_writes.get().saturating_add(1));
    }

    /// Perfetto/logging helper for KIO (KOL/KOH/KIL) reads, preserving instruction context.
    pub fn log_kio_read(&self, offset: u32, value: u8) {
        let mut guard = perfetto_guard();
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
                if op_idx == u64::MAX { None } else { Some(op_idx) },
            );
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
        let imem_offset = address - INTERNAL_MEMORY_START;
        let prev = self.internal[index];
        for byte_offset in 0..bytes {
            let byte = ((value >> (byte_offset * 8)) & 0xFF) as u8;
            let slot = &mut self.internal[index + byte_offset];
            if *slot != byte {
                *slot = byte;
                self.dirty_internal
                    .push((address + byte_offset as u32, byte));
            }
        }
        self.invoke_imr_isr_hook(imem_offset, prev, value as u8);
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
    fn internal_dirty_records_exact_address() {
        let mut mem = MemoryImage::new();
        let addr = INTERNAL_MEMORY_START + IMEM_KIL_OFFSET;
        let _ = mem.store(addr, 8, 0xAB);
        let dirty = mem.drain_dirty_internal();
        assert_eq!(dirty, vec![(addr, 0xAB)]);
    }

    #[test]
    fn internal_dirty_tracks_multi_byte_writes() {
        let mut mem = MemoryImage::new();
        let base = INTERNAL_MEMORY_START + 0x10;
        let _ = mem.store(base, 16, 0xBEEF);
        let mut dirty = mem.drain_dirty_internal();
        dirty.sort_by_key(|(addr, _)| *addr);
        assert_eq!(
            dirty,
            vec![
                (base, 0xEF),
                (base + 1, 0xBE),
            ]
        );
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
        mem.apply_host_write_with_cycle(0x0010, 0xBE, Some(0), None);
        let dirty = mem.drain_dirty();
        assert_eq!(dirty, vec![(0x0010, 0xBE)]);
        // Ensure subsequent apply with same value does not duplicate entries.
        mem.apply_host_write_with_cycle(0x0010, 0xBE, Some(1), None);
        let dirty_after = mem.drain_dirty();
        assert!(dirty_after.is_empty());
    }

    #[test]
    fn host_write_without_context_emits_perfetto() {
        use std::fs;
        let tmp = std::env::temp_dir().join("perfetto_host_async.perfetto-trace");
        let _ = fs::remove_file(&tmp);
        {
            let mut guard = crate::PERFETTO_TRACER.enter();
            *guard = Some(crate::PerfettoTracer::new(tmp.clone()));
        }

        let mut mem = MemoryImage::new();
        mem.apply_host_write_with_cycle(0x0020, 0xAA, None, None);

        if let Some(tracer) = std::mem::take(&mut *crate::PERFETTO_TRACER.enter()) {
            let _ = tracer.finish();
        }
        let buf = fs::read(&tmp).expect("read perfetto trace");
        let text = String::from_utf8_lossy(&buf).to_ascii_lowercase();
        assert!(
            text.contains("host_async"),
            "perfetto trace should include host_async write marker"
        );
        assert!(
            text.contains("0x000020"),
            "perfetto trace should include address annotation"
        );
        let _ = fs::remove_file(&tmp);
    }

    #[test]
    fn perfetto_context_falls_back_to_last_pc() {
        // Establish a last-PC hint by executing a simple instruction.
        crate::llama::eval::reset_perf_counters();
        struct NullBus;
        impl crate::llama::eval::LlamaBus for NullBus {}
        let mut exec = crate::llama::eval::LlamaExecutor::new();
        let mut state = crate::llama::state::LlamaState::new();
        state.set_pc(0x123);
        let mut bus = NullBus;
        let _ = exec.execute(0x00, &mut state, &mut bus); // NOP

        let (seq, pc) = super::perfetto_context_or_last();
        assert_eq!(
            pc,
            state.pc() & crate::llama::state::mask_for(crate::llama::opcodes::RegName::PC)
        );
        assert_ne!(seq, u64::MAX, "last instr index should be usable as fallback");
    }

    #[test]
    fn overlay_read_uses_handler_and_logs() {
        let mut mem = MemoryImage::new();
        mem.add_overlay(MemoryOverlay {
            start: 0x2000,
            end: 0x2000,
            name: "test_overlay".to_string(),
            data: None,
            read_only: true,
            read_handler: Some(Box::new(|_addr, _pc| Some(0xAB))),
            write_handler: None,
            perfetto_thread: None,
        });
        let value = mem
            .load_with_pc(0x2000, 8, Some(0x0100))
            .expect("overlay read");
        assert_eq!(value, 0xAB);
        let log = mem.overlay_read_log();
        assert_eq!(log.len(), 1);
        assert_eq!(log[0].overlay, "test_overlay");
        assert_eq!(log[0].value, 0xAB);
        assert_eq!(log[0].pc, Some(0x0100));
    }

    #[test]
    fn overlay_write_updates_data_and_logs() {
        let mut mem = MemoryImage::new();
        mem.add_overlay(MemoryOverlay {
            start: 0x4000,
            end: 0x4003,
            name: "data_overlay".to_string(),
            data: Some(vec![0u8; 4]),
            read_only: false,
            read_handler: None,
            write_handler: None,
            perfetto_thread: None,
        });
        let _ = mem.store_with_pc(0x4000, 16, 0xBEEF, Some(0x0200));
        let log = mem.overlay_write_log();
        assert_eq!(log.len(), 2);
        assert_eq!(log[0].overlay, "data_overlay");
        assert_eq!(log[0].value, 0xEF);
        assert_eq!(log[1].value, 0xBE);
        assert_eq!(mem.overlays[0].data.as_ref().unwrap()[0], 0xEF);
        assert_eq!(mem.overlays[0].data.as_ref().unwrap()[1], 0xBE);
        assert_eq!(log[0].pc, Some(0x0200));
    }

    #[test]
    fn overlay_falls_back_when_unhandled() {
        let mut mem = MemoryImage::new();
        mem.write_external_byte(0x5000, 0x55);
        mem.add_overlay(MemoryOverlay {
            start: 0x5000,
            end: 0x5000,
            name: "noop_overlay".to_string(),
            data: None,
            read_only: false,
            read_handler: Some(Box::new(|_, _| None)),
            write_handler: None,
            perfetto_thread: None,
        });
        let value = mem.load_with_pc(0x5000, 8, Some(0x0300));
        assert_eq!(value, Some(0x55));
        assert!(mem.overlay_read_log().is_empty());
    }

    #[test]
    fn add_ram_overlay_initializes_and_orders() {
        let mut mem = MemoryImage::new();
        mem.add_ram_overlay(0x6000, 4, "ram1");
        assert_eq!(mem.overlays.len(), 1);
        assert_eq!(mem.overlays[0].name, "ram1");
        assert_eq!(mem.overlays[0].data.as_ref().unwrap().len(), 4);
        // Verify overlay read returns zeroed content and logs.
        let val = mem.load_with_pc(0x6000, 8, Some(0x0400)).unwrap();
        assert_eq!(val, 0x00);
        assert_eq!(mem.overlay_read_log().len(), 1);
    }

    #[test]
    fn add_rom_overlay_installs_readonly_data() {
        let mut mem = MemoryImage::new();
        mem.add_rom_overlay(0x7000, &[0x12, 0x34], "rom1");
        assert_eq!(mem.overlays.len(), 1);
        let val = mem.load_with_pc(0x7001, 8, None);
        assert_eq!(val, Some(0x34));
        // Write should be handled (read_only) but not mutate data.
        let _ = mem.store_with_pc(0x7000, 8, 0xFF, Some(0x0500));
        assert_eq!(mem.overlays[0].data.as_ref().unwrap()[0], 0x12);
    }

    #[test]
    fn load_memory_card_maps_sizes() {
        let mut mem = MemoryImage::new();
        let data = vec![0xAA; 8192];
        mem.load_memory_card(&data).expect("load 8KB card");
        let card = mem
            .overlays
            .iter()
            .find(|ov| ov.name == "memory_card")
            .expect("memory card overlay");
        assert_eq!(card.start, 0x040000);
        assert_eq!(card.end, 0x041FFF);
        assert_eq!(card.data.as_ref().unwrap().len(), 8192);
        let val = mem.load_with_pc(0x040000, 8, None);
        assert_eq!(val, Some(0xAA));
    }

    #[test]
    fn load_memory_card_rejects_bad_sizes() {
        let mut mem = MemoryImage::new();
        let err = mem.load_memory_card(&[0xFF; 1024]);
        assert!(err.is_err());
    }
}
