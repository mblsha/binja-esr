// PY_SOURCE: pce500/memory.py:PCE500Memory

use crate::{llama::eval::perfetto_last_pc, CoreError, Result};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
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
const INTERNAL_RAM_MIRROR_START: u32 = 0x80000;
const INTERNAL_RAM_MIRROR_END: u32 = 0xBFFFF;
pub const IMEM_KOL_OFFSET: u32 = 0xF0;
pub const IMEM_KOH_OFFSET: u32 = 0xF1;
pub const IMEM_KIL_OFFSET: u32 = 0xF2;
pub const IMEM_EOL_OFFSET: u32 = 0xF3;
pub const IMEM_EOH_OFFSET: u32 = 0xF4;
pub const IMEM_EIL_OFFSET: u32 = 0xF5;
pub const IMEM_EIH_OFFSET: u32 = 0xF6;
pub const IMEM_BP_OFFSET: u32 = 0xEC;
pub const IMEM_PX_OFFSET: u32 = 0xED;
pub const IMEM_PY_OFFSET: u32 = 0xEE;
pub const IMEM_UCR_OFFSET: u32 = 0xF7;
pub const IMEM_USR_OFFSET: u32 = 0xF8;
pub const IMEM_RXD_OFFSET: u32 = 0xF9;
pub const IMEM_TXD_OFFSET: u32 = 0xFA;
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
const MEMORY_CARD_SLOT_START: u32 = 0x040000;
const MEMORY_CARD_SLOT_END: u32 = 0x04FFFF;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryCardMode {
    Absent,
    Present,
}

/// Exact, process-independent memory-card state for snapshot v4 and later.
///
/// The invariants are intentionally validated by [`MemoryImage`] rather than
/// by trusting an overlay name supplied by a caller. Present and absent cards
/// both retain one hardware-supported capacity, writability policy, and an
/// exact payload; the absent form preserves latent media for later reinsertion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryCardSnapshot {
    pub mode: MemoryCardMode,
    pub capacity: usize,
    pub writable: bool,
    pub payload: Vec<u8>,
}

/// Opaque, validated card restore plan.
///
/// Construction is restricted to [`MemoryImage::prepare_memory_card_restore`]
/// so malformed metadata cannot reach the commit path. The overlay epoch also
/// prevents a plan from being committed after an intervening overlay change.
#[derive(Debug)]
pub struct MemoryCardRestoreCandidate {
    snapshot: Option<MemoryCardSnapshot>,
    overlay_epoch: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RetainedMemoryCard {
    capacity: usize,
    writable: bool,
    payload: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MemoryCardState {
    Unconfigured,
    Invalidated,
    Absent {
        overlay_index: usize,
    },
    PresentExternal,
    PresentOverlay {
        overlay_index: usize,
        capacity: usize,
        writable: bool,
    },
}

fn canonical_address(address: u32) -> u32 {
    address & ADDRESS_MASK
}

thread_local! {
    static IMR_READ_SUPPRESS: Cell<bool> = const { Cell::new(false) };
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
    /// Side-effect-free counterpart to `read_handler` for decode/vector
    /// preflight. Handler-backed overlays without this callback fail closed.
    pub preflight_read_handler: Option<OverlayReadHandler>,
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
        addr.checked_sub(self.start)
            .and_then(|off| usize::try_from(off).ok())
    }

    fn read(&self, address: u32, pc: Option<u32>) -> Option<u8> {
        if let Some(handler) = self.read_handler.as_ref() {
            if let Some(val) = handler(address, pc) {
                return Some(val);
            }
        }
        if let (Some(data), Some(offset)) = (self.data.as_ref(), self.offset(address)) {
            return data.get(offset).copied();
        }
        None
    }

    fn read_for_preflight(&self, address: u32, pc: Option<u32>) -> Option<u8> {
        if let Some(handler) = self.preflight_read_handler.as_ref() {
            if let Some(value) = handler(address, pc) {
                return Some(value);
            }
        } else if self.read_handler.is_some() {
            // Calling an ordinary device handler would make rejection itself
            // observable, while falling through to backing data could validate
            // different bytes than architectural execution.
            return None;
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
    internal_ram_mirror: bool,
    memory_reads: Cell<u64>,
    memory_writes: Cell<u64>,
    imr_isr_hook: Option<ImrIsrHook>,
    overlays: Vec<MemoryOverlay>,
    memory_card_state: MemoryCardState,
    retained_memory_card: Option<RetainedMemoryCard>,
    overlay_epoch: u64,
    read_log: RefCell<VecDeque<MemoryAccessLog>>,
    write_log: RefCell<VecDeque<MemoryAccessLog>>,
    write_capture: Option<HashMap<u32, u8>>,
    rollback_capture: Option<MemoryRollbackCapture>,
}

/// Sparse undo log for a speculative bridge operation.  Cloning the complete
/// 1 MiB external image for every instruction is prohibitively expensive, so
/// only slots actually changed while the capture is active are retained.
struct MemoryRollbackCapture {
    external_previous: HashMap<usize, u8>,
    internal_previous: HashMap<usize, u8>,
    dirty_before: Vec<(u32, u8)>,
    dirty_internal_before: Vec<(u32, u8)>,
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
            internal_ram_mirror: false,
            memory_reads: Cell::new(0),
            memory_writes: Cell::new(0),
            imr_isr_hook: None,
            overlays: Vec::new(),
            memory_card_state: MemoryCardState::Unconfigured,
            retained_memory_card: None,
            overlay_epoch: 0,
            read_log: RefCell::new(VecDeque::with_capacity(OVERLAY_LOG_LIMIT)),
            write_log: RefCell::new(VecDeque::with_capacity(OVERLAY_LOG_LIMIT)),
            write_capture: None,
            rollback_capture: None,
        }
    }

    /// Begin a sparse rollback capture for speculative native execution.
    ///
    /// The Python bridge calls this before RESET or an instruction that may
    /// mutate the local mirror before all host callbacks have succeeded.
    pub fn begin_rollback_capture(&mut self) {
        debug_assert!(self.rollback_capture.is_none());
        self.rollback_capture = Some(MemoryRollbackCapture {
            external_previous: HashMap::new(),
            internal_previous: HashMap::new(),
            dirty_before: self.dirty.clone(),
            dirty_internal_before: self.dirty_internal.clone(),
        });
    }

    /// Commit all writes made since `begin_rollback_capture`.
    pub fn commit_rollback_capture(&mut self) {
        self.rollback_capture = None;
    }

    /// Restore mirror bytes and pending host-write queues to their state at
    /// `begin_rollback_capture`.
    pub fn rollback_capture(&mut self) {
        let Some(capture) = self.rollback_capture.take() else {
            return;
        };
        for (index, previous) in capture.external_previous {
            self.external[index] = previous;
        }
        for (index, previous) in capture.internal_previous {
            self.internal[index] = previous;
        }
        self.dirty = capture.dirty_before;
        self.dirty_internal = capture.dirty_internal_before;
    }

    fn capture_external_previous(&mut self, index: usize) {
        let previous = self.external[index];
        if let Some(capture) = self.rollback_capture.as_mut() {
            capture.external_previous.entry(index).or_insert(previous);
        }
    }

    fn capture_internal_previous(&mut self, index: usize) {
        let previous = self.internal[index];
        if let Some(capture) = self.rollback_capture.as_mut() {
            capture.internal_previous.entry(index).or_insert(previous);
        }
    }

    pub fn begin_write_capture(&mut self) {
        self.write_capture = Some(HashMap::new());
    }

    pub fn take_write_capture(&mut self) -> Vec<(u32, u8)> {
        let Some(map) = self.write_capture.take() else {
            return Vec::new();
        };
        let mut out: Vec<(u32, u8)> = map.into_iter().collect();
        out.sort_by_key(|(addr, _)| *addr);
        out
    }

    fn record_write_capture(&mut self, address: u32, value: u8) {
        if let Some(map) = self.write_capture.as_mut() {
            map.insert(canonical_address(address), value);
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
        if self.rollback_capture.is_some() {
            for (index, &value) in blob[..limit].iter().enumerate() {
                if self.external[index] != value {
                    self.capture_external_previous(index);
                }
            }
        }
        self.external[..limit].copy_from_slice(&blob[..limit]);
        self.dirty.clear();
    }

    pub fn load_internal(&mut self, blob: &[u8]) {
        let limit = self.internal.len().min(blob.len());
        if self.rollback_capture.is_some() {
            for (index, &value) in blob[..limit].iter().enumerate() {
                if self.internal[index] != value {
                    self.capture_internal_previous(index);
                }
            }
        }
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

    fn advance_overlay_epoch(&mut self) {
        self.overlay_epoch = self
            .overlay_epoch
            .checked_add(1)
            .expect("memory overlay epoch exhausted");
    }

    fn invalidate_memory_card_attestation(&mut self) {
        self.advance_overlay_epoch();
        self.memory_card_state = MemoryCardState::Invalidated;
    }

    fn add_overlay_internal(&mut self, overlay: MemoryOverlay) {
        self.overlays.push(overlay);
        self.overlays
            .sort_by(|a, b| (a.start, a.end, &a.name).cmp(&(b.start, b.end, &b.name)));
    }

    fn remove_overlay_internal(&mut self, name: &str) -> bool {
        let previous_len = self.overlays.len();
        self.overlays.retain(|ov| ov.name != name);
        self.overlays.len() != previous_len
    }

    pub fn add_overlay(&mut self, overlay: MemoryOverlay) {
        self.invalidate_memory_card_attestation();
        self.add_overlay_internal(overlay);
    }

    pub fn remove_overlay(&mut self, name: &str) {
        if self.remove_overlay_internal(name) {
            self.invalidate_memory_card_attestation();
        }
    }

    pub fn overlays(&self) -> &[MemoryOverlay] {
        &self.overlays
    }

    /// Stable address-space identity and mapping epoch used to scope opaque
    /// vector-transfer proofs. Ordinary byte writes do not change the mapping;
    /// overlay/card reconfiguration does.
    pub fn vector_transfer_provenance(&self) -> (usize, u64) {
        (self as *const Self as usize, self.overlay_epoch)
    }

    /// Whether a scheduled instruction byte has one static, callback-free
    /// view across a host timer/device tick.
    pub fn instruction_byte_is_stable(&self, address: u32) -> bool {
        let address = canonical_address(address);
        if self.requires_python(address) {
            return false;
        }
        for overlay in &self.overlays {
            if !overlay.contains(address) {
                continue;
            }
            if overlay.read_handler.is_some() || overlay.preflight_read_handler.is_some() {
                return false;
            }
            if overlay
                .data
                .as_ref()
                .and_then(|data| overlay.offset(address).and_then(|offset| data.get(offset)))
                .is_some()
            {
                return true;
            }
        }
        true
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
            preflight_read_handler: None,
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
            preflight_read_handler: None,
            write_handler: None,
            perfetto_thread: Some("Memory_ROM".to_string()),
        });
    }

    pub fn load_memory_card(&mut self, data: &[u8]) -> Result<()> {
        self.load_memory_card_with_writable(data, true)
    }

    pub fn load_memory_card_with_writable(&mut self, data: &[u8], writable: bool) -> Result<()> {
        if data.is_empty() {
            return Err(CoreError::Other("memory card data is empty".to_string()));
        }
        let size = data.len();
        let Some((_, start, end, thread)) = MEMORY_CARD_RANGES
            .iter()
            .find(|(len, _, _, _)| *len == size)
        else {
            return Err(CoreError::Other(format!(
                "unsupported memory card size: {size} bytes"
            )));
        };
        self.advance_overlay_epoch();
        self.retained_memory_card = None;
        self.remove_overlay_internal("memory_card");
        self.add_overlay_internal(MemoryOverlay {
            start: *start,
            end: *end,
            name: "memory_card".to_string(),
            data: Some(data.to_vec()),
            read_only: !writable,
            read_handler: None,
            preflight_read_handler: None,
            write_handler: None,
            perfetto_thread: Some(thread.to_string()),
        });
        self.attest_present_card(size, writable);
        Ok(())
    }

    pub fn set_memory_card_slot_present(&mut self, present: bool) {
        let retained_before_removal = (!present)
            .then(|| self.capture_active_or_retained_memory_card())
            .flatten();
        self.advance_overlay_epoch();
        self.remove_overlay_internal("memory_card_slot");
        if present {
            if let Some(retained) = self.retained_memory_card.take() {
                self.install_present_card_without_epoch(retained.payload, retained.writable);
            } else {
                self.attest_present_external_or_existing_card();
            }
            return;
        }
        self.retained_memory_card =
            Some(
                retained_before_removal.unwrap_or_else(|| RetainedMemoryCard {
                    capacity: 65_536,
                    writable: true,
                    payload: vec![0; 65_536],
                }),
            );
        self.remove_overlay_internal("memory_card");
        // Absent card: reads return 0 and writes are ignored but considered handled.
        self.add_overlay_internal(MemoryOverlay {
            start: MEMORY_CARD_SLOT_START,
            end: MEMORY_CARD_SLOT_END,
            name: "memory_card_slot".to_string(),
            data: None,
            read_only: false,
            read_handler: Some(Box::new(|_addr, _pc| Some(0x00))),
            preflight_read_handler: Some(Box::new(|_addr, _pc| Some(0x00))),
            write_handler: Some(Box::new(|_addr, _value, _pc| true)),
            perfetto_thread: Some("Memory_Card".to_string()),
        });
        self.attest_absent_card();
    }

    /// Capture exact built-in card state without inferring trust from a public
    /// overlay name. Generic overlay mutation invalidates this attestation.
    /// A generic core which has never configured a card returns `None`.
    pub fn memory_card_snapshot(&self) -> Result<Option<MemoryCardSnapshot>> {
        self.validate_memory_card_attestation()?;
        match self.memory_card_state {
            MemoryCardState::Unconfigured => Ok(None),
            MemoryCardState::Absent { .. } => {
                let retained = self
                    .retained_memory_card
                    .as_ref()
                    .expect("validated absent-card retained medium");
                let snapshot = MemoryCardSnapshot {
                    mode: MemoryCardMode::Absent,
                    capacity: retained.capacity,
                    writable: retained.writable,
                    payload: retained.payload.clone(),
                };
                Ok(Some(snapshot))
            }
            MemoryCardState::PresentExternal => {
                let start = MEMORY_CARD_SLOT_START as usize;
                let end = MEMORY_CARD_SLOT_END as usize + 1;
                Ok(Some(MemoryCardSnapshot {
                    mode: MemoryCardMode::Present,
                    capacity: end - start,
                    writable: true,
                    payload: self.external[start..end].to_vec(),
                }))
            }
            MemoryCardState::PresentOverlay {
                overlay_index,
                capacity,
                writable,
            } => {
                let payload = self.overlays[overlay_index]
                    .data
                    .as_ref()
                    .expect("validated card overlay payload")
                    .clone();
                Ok(Some(MemoryCardSnapshot {
                    mode: MemoryCardMode::Present,
                    capacity,
                    writable,
                    payload,
                }))
            }
            MemoryCardState::Invalidated => {
                unreachable!("card attestation validation rejected this state")
            }
        }
    }

    /// Validate the complete overlay contract used by exact snapshots.
    ///
    /// Only the internally attested built-in card overlay may be present. This
    /// separately rejects unrelated generic overlays which do not intersect
    /// the card slot and therefore are outside `memory_card_snapshot()`'s
    /// narrower responsibility.
    pub fn validate_snapshot_overlay_contract(&self) -> Result<()> {
        self.validate_memory_card_attestation()?;
        let attested_index = match self.memory_card_state {
            MemoryCardState::Absent { overlay_index }
            | MemoryCardState::PresentOverlay { overlay_index, .. } => Some(overlay_index),
            MemoryCardState::Unconfigured | MemoryCardState::PresentExternal => None,
            MemoryCardState::Invalidated => {
                unreachable!("card attestation validation rejected this state")
            }
        };
        if self
            .overlays
            .iter()
            .enumerate()
            .any(|(index, _)| Some(index) != attested_index)
        {
            return Err(CoreError::InvalidSnapshot(
                "snapshot cannot represent generic memory-overlay definitions or payloads"
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// Validate an archive-derived card image without changing live memory.
    pub fn prepare_memory_card_restore(
        &self,
        snapshot: Option<MemoryCardSnapshot>,
    ) -> Result<MemoryCardRestoreCandidate> {
        if let Some(snapshot) = snapshot.as_ref() {
            Self::validate_memory_card_snapshot(snapshot)?;
        }
        self.validate_memory_card_attestation()?;
        Ok(MemoryCardRestoreCandidate {
            snapshot,
            overlay_epoch: self.overlay_epoch,
        })
    }

    /// Commit a previously validated card candidate.
    ///
    /// The epoch check occurs before any mutation, so a stale candidate fails
    /// atomically. Once it passes, installation uses only infallible operations.
    pub fn commit_memory_card_restore(
        &mut self,
        candidate: MemoryCardRestoreCandidate,
    ) -> Result<()> {
        if self.overlay_epoch != candidate.overlay_epoch {
            return Err(CoreError::InvalidSnapshot(
                "memory overlays changed after card snapshot validation".to_string(),
            ));
        }
        self.advance_overlay_epoch();
        match candidate.snapshot {
            None => self.install_unconfigured_card_without_epoch(),
            Some(snapshot) => match snapshot.mode {
                MemoryCardMode::Absent => self.install_absent_card_without_epoch(snapshot),
                MemoryCardMode::Present => {
                    self.install_present_card_without_epoch(snapshot.payload, snapshot.writable)
                }
            },
        }
        Ok(())
    }

    fn validate_memory_card_snapshot(snapshot: &MemoryCardSnapshot) -> Result<()> {
        let supported_capacity = MEMORY_CARD_RANGES
            .iter()
            .any(|(capacity, _, _, _)| *capacity == snapshot.capacity);
        match snapshot.mode {
            MemoryCardMode::Absent | MemoryCardMode::Present => {
                if !supported_capacity {
                    return Err(CoreError::InvalidSnapshot(format!(
                        "unsupported snapshot memory-card capacity: {} bytes",
                        snapshot.capacity
                    )));
                }
                if snapshot.payload.len() != snapshot.capacity {
                    return Err(CoreError::InvalidSnapshot(format!(
                        "memory-card payload length mismatch (capacity {}, payload {})",
                        snapshot.capacity,
                        snapshot.payload.len()
                    )));
                }
            }
        }
        Ok(())
    }

    fn capture_active_or_retained_memory_card(&self) -> Option<RetainedMemoryCard> {
        if self.validate_memory_card_attestation().is_err() {
            return None;
        }
        match self.memory_card_state {
            MemoryCardState::Absent { .. } => self.retained_memory_card.clone(),
            MemoryCardState::PresentExternal => {
                let start = MEMORY_CARD_SLOT_START as usize;
                let end = MEMORY_CARD_SLOT_END as usize + 1;
                Some(RetainedMemoryCard {
                    capacity: end - start,
                    writable: true,
                    payload: self.external[start..end].to_vec(),
                })
            }
            MemoryCardState::PresentOverlay {
                overlay_index,
                capacity,
                writable,
            } => Some(RetainedMemoryCard {
                capacity,
                writable,
                payload: self.overlays[overlay_index]
                    .data
                    .as_ref()
                    .expect("validated card overlay payload")
                    .clone(),
            }),
            MemoryCardState::Unconfigured | MemoryCardState::Invalidated => None,
        }
    }

    fn overlay_intersects_memory_card_slot(overlay: &MemoryOverlay) -> bool {
        overlay.start <= MEMORY_CARD_SLOT_END && overlay.end >= MEMORY_CARD_SLOT_START
    }

    fn validate_memory_card_attestation(&self) -> Result<()> {
        let valid = match self.memory_card_state {
            MemoryCardState::Unconfigured => {
                self.retained_memory_card.is_none()
                    && !self
                        .overlays
                        .iter()
                        .any(Self::overlay_intersects_memory_card_slot)
            }
            MemoryCardState::Invalidated => false,
            MemoryCardState::Absent { overlay_index } => {
                self.overlays
                    .get(overlay_index)
                    .is_some_and(Self::is_builtin_absent_card_overlay)
                    && self.only_attested_card_overlay_intersects(overlay_index)
                    && self.retained_memory_card.as_ref().is_some_and(|retained| {
                        Self::validate_memory_card_snapshot(&MemoryCardSnapshot {
                            mode: MemoryCardMode::Absent,
                            capacity: retained.capacity,
                            writable: retained.writable,
                            payload: retained.payload.clone(),
                        })
                        .is_ok()
                    })
            }
            MemoryCardState::PresentExternal => {
                self.retained_memory_card.is_none()
                    && !self
                        .overlays
                        .iter()
                        .any(Self::overlay_intersects_memory_card_slot)
            }
            MemoryCardState::PresentOverlay {
                overlay_index,
                capacity,
                writable,
            } => {
                self.retained_memory_card.is_none()
                    && self.overlays.get(overlay_index).is_some_and(|overlay| {
                        Self::is_builtin_present_card_overlay(overlay, capacity, writable)
                    })
                    && self.only_attested_card_overlay_intersects(overlay_index)
            }
        };
        if valid {
            Ok(())
        } else {
            Err(CoreError::InvalidSnapshot(match self.memory_card_state {
                MemoryCardState::Unconfigured => {
                    "unattested overlay intersects an unconfigured memory-card slot".to_string()
                }
                MemoryCardState::Invalidated => {
                    "memory-card attestation was invalidated by generic overlay mutation"
                        .to_string()
                }
                _ => "memory-card overlay no longer matches its built-in attestation".to_string(),
            }))
        }
    }

    fn only_attested_card_overlay_intersects(&self, attested_index: usize) -> bool {
        self.overlays.iter().enumerate().all(|(index, overlay)| {
            index == attested_index || !Self::overlay_intersects_memory_card_slot(overlay)
        })
    }

    fn is_builtin_absent_card_overlay(overlay: &MemoryOverlay) -> bool {
        overlay.start == MEMORY_CARD_SLOT_START
            && overlay.end == MEMORY_CARD_SLOT_END
            && overlay.name == "memory_card_slot"
            && overlay.data.is_none()
            && !overlay.read_only
            && overlay.read_handler.is_some()
            && overlay.preflight_read_handler.is_some()
            && overlay.write_handler.is_some()
            && overlay.perfetto_thread.as_deref() == Some("Memory_Card")
    }

    fn is_builtin_present_card_overlay(
        overlay: &MemoryOverlay,
        capacity: usize,
        writable: bool,
    ) -> bool {
        let Some((_, start, end, thread)) = MEMORY_CARD_RANGES
            .iter()
            .find(|(candidate, _, _, _)| *candidate == capacity)
        else {
            return false;
        };
        overlay.start == *start
            && overlay.end == *end
            && overlay.name == "memory_card"
            && overlay
                .data
                .as_ref()
                .is_some_and(|data| data.len() == capacity)
            && overlay.read_only != writable
            && overlay.read_handler.is_none()
            && overlay.preflight_read_handler.is_none()
            && overlay.write_handler.is_none()
            && overlay.perfetto_thread.as_deref() == Some(*thread)
    }

    fn attest_absent_card(&mut self) {
        let Some(index) = self
            .overlays
            .iter()
            .position(Self::is_builtin_absent_card_overlay)
        else {
            self.memory_card_state = MemoryCardState::Invalidated;
            return;
        };
        self.memory_card_state = MemoryCardState::Absent {
            overlay_index: index,
        };
        if self.validate_memory_card_attestation().is_err() {
            self.memory_card_state = MemoryCardState::Invalidated;
        }
    }

    fn attest_present_card(&mut self, capacity: usize, writable: bool) {
        let Some(index) = self
            .overlays
            .iter()
            .position(|overlay| Self::is_builtin_present_card_overlay(overlay, capacity, writable))
        else {
            self.memory_card_state = MemoryCardState::Invalidated;
            return;
        };
        self.memory_card_state = MemoryCardState::PresentOverlay {
            overlay_index: index,
            capacity,
            writable,
        };
        if self.validate_memory_card_attestation().is_err() {
            self.memory_card_state = MemoryCardState::Invalidated;
        }
    }

    fn attest_present_external_or_existing_card(&mut self) {
        match self.memory_card_state {
            MemoryCardState::PresentOverlay {
                capacity, writable, ..
            } => self.attest_present_card(capacity, writable),
            _ if self
                .overlays
                .iter()
                .any(Self::overlay_intersects_memory_card_slot) =>
            {
                self.memory_card_state = MemoryCardState::Invalidated;
            }
            _ => self.memory_card_state = MemoryCardState::PresentExternal,
        }
    }

    fn install_unconfigured_card_without_epoch(&mut self) {
        self.remove_overlay_internal("memory_card");
        self.remove_overlay_internal("memory_card_slot");
        self.retained_memory_card = None;
        self.memory_card_state = MemoryCardState::Unconfigured;
    }

    fn install_absent_card_without_epoch(&mut self, snapshot: MemoryCardSnapshot) {
        self.remove_overlay_internal("memory_card");
        self.remove_overlay_internal("memory_card_slot");
        self.retained_memory_card = (snapshot.capacity != 0).then_some(RetainedMemoryCard {
            capacity: snapshot.capacity,
            writable: snapshot.writable,
            payload: snapshot.payload,
        });
        self.add_overlay_internal(MemoryOverlay {
            start: MEMORY_CARD_SLOT_START,
            end: MEMORY_CARD_SLOT_END,
            name: "memory_card_slot".to_string(),
            data: None,
            read_only: false,
            read_handler: Some(Box::new(|_addr, _pc| Some(0x00))),
            preflight_read_handler: Some(Box::new(|_addr, _pc| Some(0x00))),
            write_handler: Some(Box::new(|_addr, _value, _pc| true)),
            perfetto_thread: Some("Memory_Card".to_string()),
        });
        self.attest_absent_card();
    }

    fn install_present_card_without_epoch(&mut self, payload: Vec<u8>, writable: bool) {
        let capacity = payload.len();
        let (_, start, end, thread) = MEMORY_CARD_RANGES
            .iter()
            .find(|(candidate, _, _, _)| *candidate == capacity)
            .expect("validated memory-card capacity");
        self.remove_overlay_internal("memory_card");
        self.remove_overlay_internal("memory_card_slot");
        self.retained_memory_card = None;
        self.add_overlay_internal(MemoryOverlay {
            start: *start,
            end: *end,
            name: "memory_card".to_string(),
            data: Some(payload),
            read_only: !writable,
            read_handler: None,
            preflight_read_handler: None,
            write_handler: None,
            perfetto_thread: Some(thread.to_string()),
        });
        self.attest_present_card(capacity, writable);
    }

    pub fn clear_overlay_logs(&self) {
        self.read_log.borrow_mut().clear();
        self.write_log.borrow_mut().clear();
    }

    pub fn set_internal_ram_mirror(&mut self, enabled: bool) {
        self.internal_ram_mirror = enabled;
    }

    fn mirror_internal_ram_address(&self, address: u32) -> u32 {
        if !self.internal_ram_mirror {
            return address;
        }
        if (INTERNAL_RAM_MIRROR_START..=INTERNAL_RAM_MIRROR_END).contains(&address) {
            let mask = (INTERNAL_RAM_SIZE as u32).saturating_sub(1);
            return (INTERNAL_RAM_START as u32).wrapping_add(address & mask);
        }
        address
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
            // Keyboard matrix (when not bridged) requires host-side handlers that emulate dynamic
            // hardware state. E-port inputs (EIL/EIH: 0xF5/0xF6) are modeled locally so LLAMA can
            // run without Python overlays.
            if matches!(offset, 0xF0..=0xF2) {
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
        let external_addr = self.mirror_internal_ram_address(address);
        let idx = (external_addr as usize) & (EXTERNAL_SPACE - 1);
        self.external.get(idx).copied()
    }

    /// Read the authoritative backing image without overlays, callbacks,
    /// tracing, dirty queues, or memory-access counters. Instruction preflight
    /// uses this path so rejection itself is not an observable bus operation.
    pub fn read_byte_silent(&self, address: u32) -> Option<u8> {
        let address = canonical_address(address);
        if let Some(index) = Self::internal_index(address) {
            return Some(self.internal[index]);
        }
        let external_addr = self.mirror_internal_ram_address(address);
        let idx = (external_addr as usize) & (EXTERNAL_SPACE - 1);
        self.external.get(idx).copied()
    }

    /// Read the same static byte image that architectural `load_with_pc`
    /// would see, without counters, logs, or ordinary device callbacks.
    /// Handler-backed overlays must provide an explicit safe counterpart;
    /// otherwise preflight fails instead of validating stale backing bytes.
    pub fn read_byte_for_preflight(&self, address: u32, pc: Option<u32>) -> Option<u8> {
        let address = canonical_address(address);
        if let Some(index) = Self::internal_index(address) {
            return Some(self.internal[index]);
        }
        for overlay in &self.overlays {
            if !overlay.contains(address) {
                continue;
            }
            if overlay.read_handler.is_some() && overlay.preflight_read_handler.is_none() {
                return None;
            }
            if let Some(value) = overlay.read_for_preflight(address, pc) {
                return Some(value);
            }
        }
        let external_addr = self.mirror_internal_ram_address(address);
        let idx = (external_addr as usize) & (EXTERNAL_SPACE - 1);
        self.external.get(idx).copied()
    }

    pub fn load_for_preflight(&self, address: u32, bits: u8, pc: Option<u32>) -> Option<u32> {
        let bytes = bits.div_ceil(8).max(1);
        let mut value = 0u32;
        for offset in 0..bytes {
            value |= u32::from(
                self.read_byte_for_preflight(address.wrapping_add(u32::from(offset)), pc)?,
            ) << (8 * offset);
        }
        if bits >= 32 {
            Some(value)
        } else {
            Some(value & ((1u32 << bits) - 1))
        }
    }

    pub fn load_silent(&self, address: u32, bits: u8) -> Option<u32> {
        let bytes = bits.div_ceil(8).max(1);
        let mut value = 0u32;
        for offset in 0..bytes {
            value |= u32::from(self.read_byte_silent(address.wrapping_add(u32::from(offset)))?)
                << (8 * offset);
        }
        if bits >= 32 {
            Some(value)
        } else {
            Some(value & ((1u32 << bits) - 1))
        }
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
        let address = self.mirror_internal_ram_address(address);
        let bytes = bits.div_ceil(8).max(1) as usize;
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
        let bytes = bits.div_ceil(8).max(1) as usize;
        let external_addr = self.mirror_internal_ram_address(address);
        if self.is_read_only_range(external_addr, bytes as u32) {
            return Some(());
        }
        for offset in 0..bytes {
            let byte = ((value >> (offset * 8)) & 0xFF) as u8;
            let logical_addr = address + offset as u32;
            let phys_addr = external_addr.wrapping_add(offset as u32);
            self.record_write_capture(logical_addr, byte);
            let index = (phys_addr as usize) & (EXTERNAL_SPACE - 1);
            if self.external[index] != byte {
                self.capture_external_previous(index);
                self.external[index] = byte;
                self.dirty.push((logical_addr, byte));
            }
        }
        Some(())
    }

    pub fn drain_dirty(&mut self) -> Vec<(u32, u8)> {
        std::mem::take(&mut self.dirty)
    }

    pub fn prepend_dirty(&mut self, mut entries: Vec<(u32, u8)>) {
        entries.append(&mut self.dirty);
        self.dirty = entries;
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
        let record_perfetto = |space: &str| {
            let mut guard = perfetto_guard();
            guard.with_some(|tracer| {
                if let Some(cyc) = cycle {
                    tracer.record_mem_write_at_cycle(cyc, pc, address, value as u32, space, 8);
                } else if let Some((op_idx, pc_ctx)) = crate::llama::eval::perfetto_instr_context()
                {
                    let substep = crate::llama::eval::perfetto_next_substep();
                    tracer.record_mem_write_with_substep(
                        op_idx,
                        pc_ctx,
                        address,
                        value as u32,
                        space,
                        8,
                        substep,
                    );
                } else {
                    let pc_val = pc.or_else(|| Some(perfetto_last_pc()));
                    tracer.record_mem_write_at_cycle(
                        crate::llama::eval::perfetto_last_instr_index(),
                        pc_val,
                        address,
                        value as u32,
                        space,
                        8,
                    );
                }
            });
        };
        if let Some(index) = Self::internal_index(address) {
            let offset = address - INTERNAL_MEMORY_START;
            let prev = self.internal[index];
            self.capture_internal_previous(index);
            self.internal[index] = value;
            self.dirty_internal.push((address, value));
            self.invoke_imr_isr_hook(offset, prev, value);
            record_perfetto("internal");
            return;
        }
        if self.is_read_only_range(address, 1) {
            return;
        }
        let physical = self.mirror_internal_ram_address(address);
        let addr = (physical as usize) & (EXTERNAL_SPACE - 1);
        if self.external[addr] != value {
            self.capture_external_previous(addr);
            self.external[addr] = value;
            self.dirty.push((address, value));
        }
        record_perfetto("external");
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
                    self.record_write_capture(addr, byte);
                    self.push_overlay_log(AccessKind::Write, addr, byte, pc, &name, previous);
                    let mut guard = perfetto_guard();
                    guard.with_some(|tracer| {
                        if let Some((op_idx, pc_ctx)) = crate::llama::eval::perfetto_instr_context()
                        {
                            let substep = crate::llama::eval::perfetto_next_substep();
                            tracer.record_mem_write_with_substep(
                                op_idx,
                                pc_ctx,
                                addr,
                                byte as u32,
                                &name,
                                8,
                                substep,
                            );
                        } else {
                            tracer.record_mem_write_at_cycle(0, pc, addr, byte as u32, &name, 8);
                        }
                    });
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

    /// Update the local mirror after a byte write has already succeeded on the
    /// host. Unlike `store`, this must not enqueue the same write for a second
    /// host callback. Any older queued value for this address is superseded by
    /// the committed host value; unrelated dirty entries remain queued.
    pub fn sync_committed_host_write(&mut self, address: u32, value: u8) {
        let address = canonical_address(address);
        if let Some(index) = Self::internal_index(address) {
            self.capture_internal_previous(index);
            self.internal[index] = value;
            self.dirty_internal
                .retain(|(dirty_addr, _)| canonical_address(*dirty_addr) != address);
            return;
        }

        let physical = self.mirror_internal_ram_address(address);
        let index = (physical as usize) & (EXTERNAL_SPACE - 1);
        self.capture_external_previous(index);
        self.external[index] = value;
        self.dirty
            .retain(|(dirty_addr, _)| canonical_address(*dirty_addr) != address);
    }

    pub fn write_external_byte(&mut self, address: u32, value: u8) {
        self.memory_writes
            .set(self.memory_writes.get().saturating_add(1));
        let address = canonical_address(address);
        let physical = self.mirror_internal_ram_address(address);
        self.record_write_capture(address, value);
        let idx = (physical as usize) & (EXTERNAL_SPACE - 1);
        if self.external[idx] != value {
            self.capture_external_previous(idx);
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
            if self.rollback_capture.is_some() {
                for (index, &value) in data[..(end - start)].iter().enumerate() {
                    let external_index = start + index;
                    if self.external[external_index] != value {
                        self.capture_external_previous(external_index);
                    }
                }
            }
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
        let bytes = bits.div_ceil(8).max(1) as usize;
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
            guard.with_some(|tracer| {
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
            });
        }
        Some(value)
    }

    pub fn write_internal_byte(&mut self, offset: u32, value: u8) {
        if offset < INTERNAL_SPACE as u32 {
            let index = offset as usize;
            self.memory_writes
                .set(self.memory_writes.get().saturating_add(1));
            let prev = self.internal[index];
            self.capture_internal_previous(index);
            self.internal[index] = value;
            self.record_write_capture(INTERNAL_MEMORY_START + offset, value);
            self.dirty_internal
                .push((INTERNAL_MEMORY_START + offset, value));
            self.invoke_imr_isr_hook(offset, prev, value);
            let mut guard = perfetto_guard();
            guard.with_some(|tracer| {
                let (seq, pc) = perfetto_context_or_last();
                let substep = crate::llama::eval::perfetto_next_substep();
                tracer.record_mem_write_with_substep(
                    seq,
                    pc,
                    INTERNAL_MEMORY_START + offset,
                    value as u32,
                    "internal",
                    8,
                    substep,
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
            });
        }
    }

    pub fn read_internal_byte(&self, offset: u32) -> Option<u8> {
        if offset < INTERNAL_SPACE as u32 {
            self.memory_reads
                .set(self.memory_reads.get().saturating_add(1));
            let val = self.internal[offset as usize];
            if offset == 0xFB && !imr_read_suppressed() {
                let mut guard = perfetto_guard();
                guard.with_some(|tracer| {
                    let (op_idx, pc) = perfetto_context_or_last();
                    tracer.record_imr_read(
                        if op_idx == u64::MAX { None } else { Some(pc) },
                        val,
                        if op_idx == u64::MAX {
                            None
                        } else {
                            Some(op_idx)
                        },
                    );
                });
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
        guard.with_some(|tracer| {
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
        });
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
        let bytes = bits.div_ceil(8).max(1) as usize;
        let index = Self::internal_index(address)?;
        if index + bytes > self.internal.len() {
            return None;
        }
        let imem_offset = address - INTERNAL_MEMORY_START;
        for byte_offset in 0..bytes {
            let byte = ((value >> (byte_offset * 8)) & 0xFF) as u8;
            self.record_write_capture(address + byte_offset as u32, byte);
            let internal_index = index + byte_offset;
            let prev = self.internal[internal_index];
            if self.internal[internal_index] != byte {
                self.capture_internal_previous(internal_index);
                self.internal[internal_index] = byte;
                self.dirty_internal
                    .push((address + byte_offset as u32, byte));
            }
            self.invoke_imr_isr_hook(imem_offset + byte_offset as u32, prev, byte);
        }
        Some(())
    }

    pub fn drain_dirty_internal(&mut self) -> Vec<(u32, u8)> {
        std::mem::take(&mut self.dirty_internal)
    }

    pub fn prepend_dirty_internal(&mut self, mut entries: Vec<(u32, u8)>) {
        entries.append(&mut self.dirty_internal);
        self.dirty_internal = entries;
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
    fn internal_ram_mirror_routes_0x80000_window() {
        let mut mem = MemoryImage::new();
        mem.set_internal_ram_mirror(true);
        let addr = 0x088000;
        let value = 0xBEEF;
        assert_eq!(mem.store(addr, 16, value), Some(()));
        assert_eq!(mem.load(addr, 16), Some(value));
        assert_eq!(mem.load(INTERNAL_RAM_START as u32, 16), Some(value));
        let slice = mem.internal_ram_slice();
        assert_eq!(slice.first(), Some(&0xEF));
        assert_eq!(slice.get(1), Some(&0xBE));
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
    fn e_port_offsets_do_not_require_python_overlay() {
        let mem = MemoryImage::new();
        for offset in [0xF5, 0xF6] {
            let addr = INTERNAL_MEMORY_START + offset;
            assert!(
                !mem.requires_python(addr),
                "offset 0x{offset:02X} should not require Python overlay"
            );
        }
    }

    #[test]
    fn external_reset_vector_is_not_aliased() {
        let mut mem = MemoryImage::new();
        // Populate the interrupt-vector region (0x0FFFFA-0x0FFFFC) in external space.
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
        assert_eq!(dirty, vec![(base, 0xEF), (base + 1, 0xBE),]);
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
        let _lock = crate::perfetto::perfetto_test_guard();
        let tmp = std::env::temp_dir().join("perfetto_host_async.perfetto-trace");
        let _ = fs::remove_file(&tmp);
        let mut guard = crate::PERFETTO_TRACER.enter();
        guard.replace(Some(crate::PerfettoTracer::new(tmp.clone())));

        let mut mem = MemoryImage::new();
        mem.apply_host_write_with_cycle(0x0020, 0xAA, None, None);
        assert_eq!(mem.memory_write_count(), 1);

        if let Some(tracer) = guard.take() {
            let _ = tracer.finish();
        }
        let _ = fs::remove_file(&tmp);
    }

    #[test]
    fn host_write_uses_last_pc_fallback_when_no_context() {
        use crate::llama::eval::{reset_perf_counters, LlamaBus, LlamaExecutor};
        use crate::llama::state::LlamaState;
        use std::fs;

        let _lock = crate::perfetto::perfetto_test_guard();
        let mut guard = perfetto_guard();
        // Execute a NOP at PC 0x123 to seed perfetto_last_pc without relying on a live context.
        reset_perf_counters();
        let mut exec = LlamaExecutor::new();
        let mut state = LlamaState::new();
        state.set_pc(0x0123);
        struct NullBus;
        impl LlamaBus for NullBus {
            fn load(&mut self, _addr: u32, _bits: u8) -> u32 {
                0
            }
            fn store(&mut self, _addr: u32, _bits: u8, _value: u32) {}
            fn peek_byte_silent(&mut self, _addr: u32) -> Option<u8> {
                Some(0)
            }
        }
        let mut bus = NullBus;
        let _ = exec.execute(0x00, &mut state, &mut bus);

        let tmp = std::env::temp_dir().join("perfetto_host_pc_fallback.perfetto-trace");
        let _ = fs::remove_file(&tmp);
        guard.replace(Some(crate::PerfettoTracer::new(tmp.clone())));

        let mut mem = MemoryImage::new();
        mem.apply_host_write_with_cycle(0x0020, 0xAA, None, None);

        if let Some(tracer) = guard.take() {
            let pcs = tracer.test_mem_write_pcs.borrow().clone();
            assert_eq!(
                pcs.last().copied().flatten(),
                Some(0x0123),
                "host write should use last executed PC when no live context"
            );
            let _ = tracer.finish();
        }
        let _ = fs::remove_file(&tmp);
    }

    #[test]
    fn perfetto_context_falls_back_to_last_pc() {
        let _lock = crate::perfetto::perfetto_test_guard();
        let _guard = perfetto_guard();
        // Establish a last-PC hint by executing a simple instruction.
        crate::llama::eval::reset_perf_counters();
        struct NullBus;
        impl crate::llama::eval::LlamaBus for NullBus {
            fn load(&mut self, _addr: u32, _bits: u8) -> u32 {
                0
            }
            fn store(&mut self, _addr: u32, _bits: u8, _value: u32) {}
        }
        let mut exec = crate::llama::eval::LlamaExecutor::new();
        let mut state = crate::llama::state::LlamaState::new();
        state.set_pc(0x123);
        let mut bus = NullBus;
        let _ = exec.execute(0x00, &mut state, &mut bus); // NOP

        let (seq, pc) = super::perfetto_context_or_last();
        assert_eq!(
            pc,
            0x123 & crate::llama::state::mask_for(crate::llama::opcodes::RegName::PC)
        );
        assert_ne!(
            seq,
            u64::MAX,
            "last instr index should be usable as fallback"
        );
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
            preflight_read_handler: Some(Box::new(|_addr, _pc| Some(0xAB))),
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
            preflight_read_handler: None,
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
            preflight_read_handler: None,
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

    #[test]
    fn memory_card_slot_absent_reads_zero_and_ignores_writes() {
        let mut mem = MemoryImage::new();
        mem.set_memory_card_slot_present(false);
        assert_eq!(mem.load_with_pc(0x040005, 8, None), Some(0));
        mem.store_with_pc(0x040005, 8, 0xFF, None);
        assert_eq!(mem.load_with_pc(0x040005, 8, None), Some(0));
    }

    #[test]
    fn memory_card_slot_present_writes_to_external() {
        let mut mem = MemoryImage::new();
        mem.set_memory_card_slot_present(true);
        mem.store_with_pc(0x040000, 8, 0x9F, None);
        assert_eq!(mem.external_slice()[0x040000], 0x9F);
    }

    #[test]
    fn memory_card_snapshot_distinguishes_unconfigured_absent_and_present() {
        let mut mem = MemoryImage::new();
        assert_eq!(mem.memory_card_snapshot().unwrap(), None);

        mem.set_memory_card_slot_present(false);
        assert_eq!(
            mem.memory_card_snapshot().unwrap(),
            Some(MemoryCardSnapshot {
                mode: MemoryCardMode::Absent,
                capacity: 65_536,
                writable: true,
                payload: vec![0; 65_536],
            })
        );

        mem.set_memory_card_slot_present(true);
        let snapshot = mem
            .memory_card_snapshot()
            .unwrap()
            .expect("present card snapshot");
        assert_eq!(snapshot.mode, MemoryCardMode::Present);
        assert_eq!(snapshot.capacity, 65_536);
        assert!(snapshot.writable);
        assert_eq!(snapshot.payload.len(), 65_536);
    }

    #[test]
    fn memory_card_snapshot_roundtrips_every_capacity_and_write_mode() {
        for (index, capacity) in [8192, 16384, 32768, 65536].into_iter().enumerate() {
            let writable = index % 2 == 0;
            let mut source = MemoryImage::new();
            source.set_memory_card_slot_present(true);
            source
                .load_memory_card_with_writable(&vec![index as u8; capacity], writable)
                .expect("install supported card");
            if writable {
                source.store_with_pc(MEMORY_CARD_SLOT_START, 8, 0xA5, None);
            } else {
                source.store_with_pc(MEMORY_CARD_SLOT_START, 8, 0xFF, None);
            }

            let snapshot = source
                .memory_card_snapshot()
                .unwrap()
                .expect("configured card snapshot");
            assert_eq!(snapshot.mode, MemoryCardMode::Present);
            assert_eq!(snapshot.capacity, capacity);
            assert_eq!(snapshot.writable, writable);
            assert_eq!(
                snapshot.payload[0],
                if writable { 0xA5 } else { index as u8 }
            );

            let mut restored = MemoryImage::new();
            let candidate = restored
                .prepare_memory_card_restore(Some(snapshot.clone()))
                .expect("prepare card restore");
            restored
                .commit_memory_card_restore(candidate)
                .expect("commit card restore");
            assert_eq!(restored.memory_card_snapshot().unwrap(), Some(snapshot));
        }
    }

    #[test]
    fn absent_card_restore_retains_media_for_reinsertion() {
        let retained = MemoryCardSnapshot {
            mode: MemoryCardMode::Absent,
            capacity: 8192,
            writable: false,
            payload: vec![0xA5; 8192],
        };
        let mut mem = MemoryImage::new();
        let candidate = mem
            .prepare_memory_card_restore(Some(retained.clone()))
            .expect("prepare absent card");
        mem.commit_memory_card_restore(candidate)
            .expect("commit absent card");
        assert_eq!(mem.load_with_pc(MEMORY_CARD_SLOT_START, 8, None), Some(0));
        assert_eq!(mem.memory_card_snapshot().unwrap(), Some(retained));

        mem.set_memory_card_slot_present(true);
        assert_eq!(
            mem.load_with_pc(MEMORY_CARD_SLOT_START, 8, None),
            Some(0xA5)
        );
        mem.store_with_pc(MEMORY_CARD_SLOT_START, 8, 0x5A, None);
        assert_eq!(
            mem.load_with_pc(MEMORY_CARD_SLOT_START, 8, None),
            Some(0xA5),
            "retained read-only configuration must survive reinsertion"
        );
        let present = mem.memory_card_snapshot().unwrap().unwrap();
        assert_eq!(present.mode, MemoryCardMode::Present);
        assert_eq!(present.capacity, 8192);
        assert!(!present.writable);
        assert_eq!(present.payload, vec![0xA5; 8192]);
    }

    #[test]
    fn generic_overlay_mutation_invalidates_card_attestation() {
        let mut mem = MemoryImage::new();
        mem.set_memory_card_slot_present(false);
        assert!(mem.memory_card_snapshot().is_ok());

        mem.add_overlay(MemoryOverlay {
            start: 0x7000,
            end: 0x7000,
            name: "unrelated_but_unattested".to_string(),
            data: Some(vec![0]),
            read_only: false,
            read_handler: None,
            preflight_read_handler: None,
            write_handler: None,
            perfetto_thread: None,
        });
        let error = mem
            .memory_card_snapshot()
            .expect_err("generic mutation must invalidate card provenance");
        assert!(error.to_string().contains("invalidated"));

        mem.set_memory_card_slot_present(false);
        mem.remove_overlay("memory_card_slot");
        mem.add_overlay(MemoryOverlay {
            start: MEMORY_CARD_SLOT_START,
            end: MEMORY_CARD_SLOT_END,
            name: "memory_card_slot".to_string(),
            data: None,
            read_only: false,
            read_handler: Some(Box::new(|_, _| Some(0))),
            preflight_read_handler: Some(Box::new(|_, _| Some(0))),
            write_handler: Some(Box::new(|_, _, _| true)),
            perfetto_thread: Some("Memory_Card".to_string()),
        });
        let error = mem
            .memory_card_snapshot()
            .expect_err("same-name handler spoof must not restore provenance");
        assert!(error.to_string().contains("invalidated"));
    }

    #[test]
    fn snapshot_overlay_contract_rejects_preexisting_generic_overlays() {
        let mut mem = MemoryImage::new();
        mem.add_ram_overlay(0x7000, 1, "generic_runtime_device");
        mem.set_memory_card_slot_present(false);

        assert!(
            mem.memory_card_snapshot().is_ok(),
            "the card itself was installed by the trusted built-in API"
        );
        let error = mem
            .validate_snapshot_overlay_contract()
            .expect_err("generic overlay must remain outside the exact snapshot contract");
        assert!(error.to_string().contains("generic memory-overlay"));

        let mut card_only = MemoryImage::new();
        card_only.set_memory_card_slot_present(false);
        card_only
            .validate_snapshot_overlay_contract()
            .expect("attested built-in card overlay is allowed");
    }

    #[test]
    fn stale_card_restore_candidate_fails_before_mutation() {
        let snapshot = MemoryCardSnapshot {
            mode: MemoryCardMode::Present,
            capacity: 8192,
            writable: true,
            payload: vec![0x5A; 8192],
        };
        let mut mem = MemoryImage::new();
        let candidate = mem
            .prepare_memory_card_restore(Some(snapshot))
            .expect("prepare card restore");
        mem.add_ram_overlay(0x7000, 1, "intervening_overlay");
        let overlay_count = mem.overlays().len();

        let error = mem
            .commit_memory_card_restore(candidate)
            .expect_err("stale candidate must fail");
        assert!(error.to_string().contains("changed after"));
        assert_eq!(mem.overlays().len(), overlay_count);
        assert!(mem
            .overlays()
            .iter()
            .all(|overlay| overlay.name != "memory_card"));
    }

    #[test]
    fn malformed_card_restore_is_rejected_without_mutation() {
        let mem = MemoryImage::new();
        for snapshot in [
            MemoryCardSnapshot {
                mode: MemoryCardMode::Absent,
                capacity: 0,
                writable: false,
                payload: Vec::new(),
            },
            MemoryCardSnapshot {
                mode: MemoryCardMode::Absent,
                capacity: 8192,
                writable: true,
                payload: vec![0; 1],
            },
            MemoryCardSnapshot {
                mode: MemoryCardMode::Present,
                capacity: 1024,
                writable: true,
                payload: vec![0; 1024],
            },
        ] {
            assert!(mem.prepare_memory_card_restore(Some(snapshot)).is_err());
            assert_eq!(mem.memory_card_snapshot().unwrap(), None);
            assert!(mem.overlays().is_empty());
        }
    }

    #[test]
    fn failed_flush_entries_can_be_requeued_ahead_of_new_dirt() {
        let mut mem = MemoryImage::new();
        mem.prepend_dirty_internal(vec![(INTERNAL_MEMORY_START + 1, 0x11)]);
        mem.write_internal_byte(2, 0x22);
        assert_eq!(
            mem.drain_dirty_internal(),
            vec![
                (INTERNAL_MEMORY_START + 1, 0x11),
                (INTERNAL_MEMORY_START + 2, 0x22),
            ]
        );

        mem.prepend_dirty(vec![(0x10, 0x33)]);
        mem.write_external_byte(0x20, 0x44);
        assert_eq!(mem.drain_dirty(), vec![(0x10, 0x33), (0x20, 0x44)]);
    }

    #[test]
    fn committed_host_write_updates_mirror_without_requeueing_it() {
        let mut mem = MemoryImage::new();
        let internal = INTERNAL_MEMORY_START + 0x20;
        let unrelated = INTERNAL_MEMORY_START + 0x21;
        let _ = mem.store(internal, 8, 0x11);
        let _ = mem.store(unrelated, 8, 0x22);

        mem.sync_committed_host_write(internal, 0x33);

        assert_eq!(mem.load(internal, 8), Some(0x33));
        assert_eq!(mem.drain_dirty_internal(), vec![(unrelated, 0x22)]);
    }

    #[test]
    fn wide_internal_store_invokes_irq_hooks_for_each_covered_register() {
        let mut mem = MemoryImage::new();
        let transitions = Rc::new(RefCell::new(Vec::new()));
        let captured = Rc::clone(&transitions);
        mem.set_imr_isr_hook(Some(move |offset, previous, new| {
            captured.borrow_mut().push((offset, previous, new));
        }));

        // TXD, IMR, and ISR are adjacent. A 24-bit store beginning at TXD
        // must still notify the IMR and ISR hooks for the second and third
        // bytes, in architectural byte order.
        mem.store(INTERNAL_MEMORY_START + IMEM_TXD_OFFSET, 24, 0x33_22_11)
            .expect("wide internal store");

        assert_eq!(
            transitions.borrow().as_slice(),
            &[(IMEM_IMR_OFFSET, 0x00, 0x22), (IMEM_ISR_OFFSET, 0x00, 0x33),]
        );
        assert_eq!(mem.read_internal_byte_silent(IMEM_TXD_OFFSET), Some(0x11));
        assert_eq!(mem.read_internal_byte_silent(IMEM_IMR_OFFSET), Some(0x22));
        assert_eq!(mem.read_internal_byte_silent(IMEM_ISR_OFFSET), Some(0x33));
    }

    #[test]
    fn rollback_capture_restores_sparse_bytes_and_dirty_queues() {
        let mut mem = MemoryImage::new();
        let internal = INTERNAL_MEMORY_START + 0x20;
        let preexisting_internal = INTERNAL_MEMORY_START + 0x21;

        mem.sync_committed_host_write(0x10, 0x11);
        mem.sync_committed_host_write(internal, 0x22);
        mem.write_external_byte(0x30, 0x33);
        mem.write_internal_byte(0x21, 0x44);

        mem.begin_rollback_capture();
        mem.write_external_byte(0x10, 0xAA);
        mem.write_internal_byte(0x20, 0xBB);
        // A committed host write can remove an older queued entry; rollback
        // must restore that queue as well as the bytes themselves.
        mem.sync_committed_host_write(0x30, 0xCC);
        mem.sync_committed_host_write(preexisting_internal, 0xDD);
        mem.rollback_capture();

        assert_eq!(mem.load(0x10, 8), Some(0x11));
        assert_eq!(mem.load(internal, 8), Some(0x22));
        assert_eq!(mem.load(0x30, 8), Some(0x33));
        assert_eq!(mem.load(preexisting_internal, 8), Some(0x44));
        assert_eq!(mem.drain_dirty(), vec![(0x30, 0x33)]);
        assert_eq!(
            mem.drain_dirty_internal(),
            vec![(preexisting_internal, 0x44)]
        );
    }
}
