//! Lightweight evaluator scaffold for LLAMA.
//!
//! Supports a small subset of opcodes today (imm/reg arithmetic/logic/moves on
//! `A`) to exercise the typed opcode table. The intent is to grow coverage
//! incrementally while keeping masking/aliasing consistent with the Python
//! emulator.
// PY_SOURCE: sc62015/pysc62015/emulator.py:Emulator.execute_instruction
// PY_SOURCE: sc62015/pysc62015/instr/__init__.py:decode

use super::{
    dispatch,
    opcodes::{InstrKind, OpcodeEntry, OperandKind, RegName},
    state::{mask_for, LlamaState, PowerState},
};
use crate::{
    memory::{
        with_imr_read_suppressed, IMEM_BP_OFFSET, IMEM_IMR_OFFSET, IMEM_ISR_OFFSET,
        IMEM_LCC_OFFSET, IMEM_PX_OFFSET, IMEM_PY_OFFSET, IMEM_SCR_OFFSET, IMEM_SSR_OFFSET,
        IMEM_UCR_OFFSET, IMEM_USR_OFFSET, INTERNAL_MEMORY_START,
    },
    perfetto::AnnotationValue,
    PERFETTO_TRACER,
};
use serde::{Deserialize, Serialize};
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

static PERF_INSTR_COUNTER: AtomicU64 = AtomicU64::new(0);

pub const PERFETTO_CALL_STACK_MAX_FRAMES: usize = 8;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerfettoCallStack {
    pub len: u8,
    pub frames: [u32; PERFETTO_CALL_STACK_MAX_FRAMES],
}

thread_local! {
    // Current execution context belongs to one machine on one executor thread.
    // Keeping it process-global allowed unrelated parallel tests/machines to
    // publish a false context into each other's memory and trace callbacks.
    static PERF_CURRENT_PC: Cell<u32> = const { Cell::new(u32::MAX) };
    static PERF_CURRENT_OP: Cell<u64> = const { Cell::new(u64::MAX) };
    static PERF_SUBSTEP: Cell<u32> = const { Cell::new(0) };
    static PERF_LAST_PC: Cell<u32> = const { Cell::new(0) };
    static PERF_LAST_CALL_STACK: Cell<PerfettoCallStack> = const { Cell::new(PerfettoCallStack { len: 0, frames: [0; PERFETTO_CALL_STACK_MAX_FRAMES] }) };
    static PREFLIGHT_DEPTH: Cell<u32> = const { Cell::new(0) };
}

struct PreflightGuard;

impl PreflightGuard {
    fn enter() -> Self {
        PREFLIGHT_DEPTH.with(|depth| depth.set(depth.get().saturating_add(1)));
        Self
    }
}

impl Drop for PreflightGuard {
    fn drop(&mut self) {
        PREFLIGHT_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

struct PerfettoContextGuard;
impl Drop for PerfettoContextGuard {
    fn drop(&mut self) {
        PERF_CURRENT_OP.with(|value| value.set(u64::MAX));
        PERF_CURRENT_PC.with(|value| value.set(u32::MAX));
        PERF_SUBSTEP.with(|value| value.set(0));
    }
}

/// Expose current instruction context for Perfetto correlation outside the executor.
pub fn perfetto_instr_context() -> Option<(u64, u32)> {
    let op = PERF_CURRENT_OP.with(|value| value.get());
    let pc = PERF_CURRENT_PC.with(|value| value.get());
    if op == u64::MAX || pc == u32::MAX {
        None
    } else {
        Some((op, pc))
    }
}

/// Last-seen instruction index for host-side events that occur outside executor context.
pub fn perfetto_last_instr_index() -> u64 {
    PERF_INSTR_COUNTER.load(Ordering::Relaxed)
}

/// Last-seen PC (masked) even outside executor context; useful for host-side tracing.
pub fn perfetto_last_pc() -> u32 {
    PERF_LAST_PC.with(|value| value.get())
}

/// Last-seen call stack (truncated) even outside executor context; useful for host-side tracing.
pub fn perfetto_last_call_stack() -> PerfettoCallStack {
    PERF_LAST_CALL_STACK.with(|value| value.get())
}

pub fn reset_perf_counters() {
    let _guard = PERFETTO_TRACER.enter();
    PERF_INSTR_COUNTER.store(0, Ordering::Relaxed);
    PERF_CURRENT_PC.with(|value| value.set(u32::MAX));
    PERF_CURRENT_OP.with(|value| value.set(u64::MAX));
    PERF_LAST_PC.with(|value| value.set(0));
    PERF_LAST_CALL_STACK.with(|value| value.set(PerfettoCallStack::default()));
    PERF_SUBSTEP.with(|value| value.set(0));
}

/// Set the global instruction index used for Perfetto `op_index` annotations.
///
/// This is used by snapshot-driven runners so traces remain aligned to the absolute
/// instruction_count stored in the snapshot metadata.
pub fn set_perf_instr_counter(value: u64) {
    PERF_INSTR_COUNTER.store(value, Ordering::Relaxed);
}

/// Next per-instruction substep for Perfetto manual clock parity.
pub fn perfetto_next_substep() -> u64 {
    PERF_SUBSTEP.with(|value| {
        let next = value.get().wrapping_add(1);
        value.set(next);
        next as u64
    })
}

fn perfetto_reset_substep() {
    PERF_SUBSTEP.with(|value| value.set(0));
}

fn reject_unknown() -> Result<u8, &'static str> {
    Err("invalid or reserved opcode")
}

pub const TCL_UNIMPLEMENTED_ERROR: &str = "TCL requires a timer-phase-clear bus";
pub const ENCODED_20BIT_UPPER_NIBBLE_ERROR: &str =
    "encoded 20-bit operand has reserved upper-nibble bits";
pub const SILENT_PEEK_UNAVAILABLE_ERROR: &str =
    "side-effect-free instruction preflight memory is unavailable";
pub const VECTOR_UPPER_NIBBLE_ERROR: &str =
    "SC62015 vector upper-nibble behavior requires real-hardware tracing";
pub const VECTOR_CHANGED_DURING_PREFLIGHT_ERROR: &str =
    "SC62015 vector changed between silent preflight and architectural fetch";
pub const PREPARED_VECTOR_MISMATCH_ERROR: &str =
    "prepared SC62015 vector transfer does not match the current instruction";

/// Opaque proof that one fixed vector was silently validated.  The proof also
/// records whether its one architectural fetch has already happened.
///
/// Private fields prevent wrappers from manufacturing a target scalar and
/// bypassing the destination checks. A machine may hold this value across its
/// pre-execution timer/device tick, then consume it exactly once when the
/// corresponding IR/RESET operation commits.
#[derive(Debug, PartialEq, Eq)]
pub struct ValidatedVectorTransfer {
    vector_addr: u32,
    source_pc: u32,
    target: u32,
    target_len: u8,
    provenance: (usize, u64),
    architectural_fetch_validated: bool,
}

impl ValidatedVectorTransfer {
    pub fn target(&self) -> u32 {
        self.target
    }

    pub fn vector_address(&self) -> u32 {
        self.vector_addr
    }

    pub fn source_pc(&self) -> u32 {
        self.source_pc
    }

    pub fn target_len(&self) -> u8 {
        self.target_len
    }

    fn matches<B: LlamaBus>(&self, vector_addr: u32, state: &LlamaState, bus: &B) -> bool {
        self.vector_addr == vector_addr
            && self.source_pc == state.pc() & mask_for(RegName::PC)
            && self.provenance == bus.vector_transfer_provenance()
    }

    pub fn architectural_fetch_validated(&self) -> bool {
        self.architectural_fetch_validated
    }

    /// Consume this proof at the architectural vector-read boundary.
    ///
    /// IR/IRQ callers use a silent-only proof and call this after the complete
    /// five-byte frame has been written. RESET callers use an already-fetched
    /// proof so reset remains fail-closed before its first mutation.
    pub fn consume_after_architectural_fetch<B: LlamaBus>(
        self,
        vector_addr: u32,
        state: &LlamaState,
        bus: &mut B,
    ) -> Result<u32, &'static str> {
        if !self.matches(vector_addr, state, bus) {
            return Err(PREPARED_VECTOR_MISMATCH_ERROR);
        }
        if self.architectural_fetch_validated {
            return Ok(self.target);
        }

        let fetched_vector = bus.load(vector_addr, 8)
            | (bus.load(vector_addr.wrapping_add(1), 8) << 8)
            | (bus.load(vector_addr.wrapping_add(2), 8) << 16);
        if bus.vector_transfer_provenance() != self.provenance {
            return Err(PREPARED_VECTOR_MISMATCH_ERROR);
        }
        if fetched_vector == self.target {
            return Ok(fetched_vector);
        }
        if fetched_vector & 0xF0_0000 != 0 {
            return Err(VECTOR_UPPER_NIBBLE_ERROR);
        }

        // A volatile but canonical vector still needs the same destination
        // validation as the silently observed value.
        let opcode = bus
            .peek_byte_silent_at(fetched_vector, fetched_vector)
            .ok_or(SILENT_PEEK_UNAVAILABLE_ERROR)?;
        let mut target_state = state.clone();
        target_state.set_pc(fetched_vector);
        LlamaExecutor.validate_before_scheduling_with_options(
            opcode,
            &target_state,
            bus,
            false,
            false,
        )?;
        Err(VECTOR_CHANGED_DURING_PREFLIGHT_ERROR)
    }
}

fn effective_i_count(state: &LlamaState) -> u32 {
    let count = state.get_reg(RegName::I) & mask_for(RegName::I);
    if count == 0 {
        // PC-E500 HW-002 measured a 16-bit do-while countdown for every
        // counted family: I=0 means 65,536 iterations.
        mask_for(RegName::I) + 1
    } else {
        count
    }
}

fn wait_cycle_count(state: &LlamaState) -> u32 {
    let initial_i = state.get_reg(RegName::I) & mask_for(RegName::I);
    if initial_i == 0 {
        // PC-E500 HW-002 observed the complete 16-bit do-while countdown:
        // 65,536 idle bus cycles, then I wrapped back to zero and execution
        // continued. These are scheduler cycles, not architectural reads.
        mask_for(RegName::I) + 1
    } else {
        initial_i
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AddressingMode {
    N,
    BpN,
    PxN,
    PyN,
    BpPx,
    BpPy,
}

#[derive(Debug, Clone, Copy)]
struct PreModes {
    first: AddressingMode,
    second: AddressingMode,
}

const PRE_MODES: &[(u8, AddressingMode, AddressingMode)] = &[
    (0x32, AddressingMode::N, AddressingMode::N),
    (0x30, AddressingMode::N, AddressingMode::BpN),
    (0x33, AddressingMode::N, AddressingMode::PyN),
    (0x31, AddressingMode::N, AddressingMode::BpPy),
    (0x22, AddressingMode::BpN, AddressingMode::N),
    (0x23, AddressingMode::BpN, AddressingMode::PyN),
    (0x21, AddressingMode::BpN, AddressingMode::BpPy),
    (0x36, AddressingMode::PxN, AddressingMode::N),
    (0x34, AddressingMode::PxN, AddressingMode::BpN),
    (0x37, AddressingMode::PxN, AddressingMode::PyN),
    (0x35, AddressingMode::PxN, AddressingMode::BpPy),
    (0x26, AddressingMode::BpPx, AddressingMode::N),
    (0x24, AddressingMode::BpPx, AddressingMode::BpN),
    (0x27, AddressingMode::BpPx, AddressingMode::PyN),
    (0x25, AddressingMode::BpPx, AddressingMode::BpPy),
];

const SINGLE_ADDRESSABLE_OPCODES: &[u8] = &[
    0x10, 0x41, 0x42, 0x43, 0x47, 0x49, 0x4A, 0x4B, 0x51, 0x52, 0x53, 0x55, 0x57, 0x59, 0x5A, 0x5B,
    0x5D, 0x61, 0x62, 0x63, 0x65, 0x66, 0x67, 0x69, 0x6A, 0x6B, 0x6D, 0x6F, 0x71, 0x72, 0x73, 0x77,
    0x79, 0x7A, 0x7B, 0x7D, 0x7F, 0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A,
    0x8B, 0x8C, 0x8D, 0x8E, 0x8F, 0x98, 0x99, 0x9A, 0x9B, 0x9C, 0x9D, 0x9E, 0xA0, 0xA1, 0xA2, 0xA3,
    0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xAB, 0xAC, 0xAD, 0xAE, 0xAF, 0xB8, 0xB9, 0xBA, 0xBB,
    0xBC, 0xBD, 0xBE, 0xC5, 0xCC, 0xCD, 0xD5, 0xD6, 0xD7, 0xDC, 0xE3, 0xE5, 0xE7, 0xEB, 0xEC, 0xF5,
    0xF7, 0xFC,
];

const INTERRUPT_VECTOR_ADDR: u32 = 0xFFFFA;
// Reset vector is stored in the top three bytes of the address space.
const ROM_RESET_VECTOR_ADDR: u32 = 0xFFFFD;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimerTrace {
    pub mti_ticks: u64,
    pub sti_ticks: u64,
}

pub trait LlamaBus {
    fn load(&mut self, addr: u32, bits: u8) -> u32;
    fn store(&mut self, addr: u32, bits: u8, value: u32);
    /// Read one byte for validation without advancing devices, consuming host
    /// callbacks, or publishing trace/memory-read accounting. Production buses
    /// with observable reads must override this method.
    fn peek_byte_silent(&mut self, _addr: u32) -> Option<u8> {
        None
    }
    /// Context-aware safe peek for PC-sensitive overlays. The context is the
    /// instruction whose bytes/operands are being validated (the vector
    /// destination while decoding a handler), not necessarily the caller PC.
    fn peek_byte_silent_at(&mut self, addr: u32, _context_pc: u32) -> Option<u8> {
        self.peek_byte_silent(addr)
    }
    /// Stable identity plus mapping epoch for opaque vector-transfer proofs.
    /// Wrappers recreated around the same address space must override this and
    /// forward the underlying memory identity/epoch.
    fn vector_transfer_provenance(&self) -> (usize, u64) {
        (self as *const Self as *const () as usize, 0)
    }
    /// Whether normal and preflight instruction fetch have one static view
    /// for this byte across the wrapper's pre-execution timer/device tick.
    fn instruction_byte_is_stable(&self, _addr: u32) -> bool {
        false
    }
    fn resolve_emem(&mut self, base: u32) -> u32 {
        base
    }
    fn peek_imem(&mut self, offset: u32) -> u8 {
        let addr = INTERNAL_MEMORY_START + offset;
        (self.load(addr, 8) & 0xFF) as u8
    }
    /// Peek IMEM without emitting tracing side-effects (IMR/ISR sampling).
    fn peek_imem_silent(&mut self, offset: u32) -> u8 {
        self.peek_byte_silent(INTERNAL_MEMORY_START + offset)
            .unwrap_or(0)
    }
    /// Optional hook for WAIT to spin timers/keyboard for `cycles` iterations (unused for Python parity WAIT).
    fn wait_cycles(&mut self, _cycles: u32) {}
    fn supports_wait_cycles(&self) -> bool {
        false
    }
    /// Whether TCL can restart the main/sub timer phases atomically.
    fn supports_timer_phase_clear(&self) -> bool {
        false
    }
    /// Restart the selected timer phases at the bus's current cycle.
    fn clear_timer_phases(&mut self, _clear_sti: bool, _clear_mti: bool) {}
    /// Optional timer snapshot for perfetto tracing (ticks since last MTI/STI fire).
    fn timer_trace(&mut self) -> Option<TimerTrace> {
        None
    }
    /// Optional hook to surface the current cycle count for tracing.
    fn cycle_count(&mut self) -> Option<u64> {
        None
    }
}

/// Adapter used by instruction validation. Its `load` path is assembled from
/// side-effect-free byte peeks so a rejected instruction cannot consume an I/O
/// read, invoke a host callback, or alter trace/accounting state.
struct SilentPreflightBus<'a, B: LlamaBus> {
    inner: &'a mut B,
    unavailable: bool,
    context_pc: u32,
}

impl<B: LlamaBus> LlamaBus for SilentPreflightBus<'_, B> {
    fn load(&mut self, addr: u32, bits: u8) -> u32 {
        let bytes = bits.div_ceil(8).max(1);
        let mut value = 0u32;
        for index in 0..bytes {
            let byte = self
                .inner
                .peek_byte_silent_at(addr.wrapping_add(u32::from(index)), self.context_pc);
            match byte {
                Some(byte) => value |= u32::from(byte) << (8 * index),
                None => self.unavailable = true,
            }
        }
        if bits >= 32 {
            value
        } else {
            value & ((1u32 << bits) - 1)
        }
    }

    fn store(&mut self, _addr: u32, _bits: u8, _value: u32) {
        unreachable!("instruction preflight must never write memory")
    }

    fn peek_byte_silent(&mut self, addr: u32) -> Option<u8> {
        let value = self.inner.peek_byte_silent_at(addr, self.context_pc);
        if value.is_none() {
            self.unavailable = true;
        }
        value
    }

    fn vector_transfer_provenance(&self) -> (usize, u64) {
        self.inner.vector_transfer_provenance()
    }

    fn instruction_byte_is_stable(&self, addr: u32) -> bool {
        self.inner.instruction_byte_is_stable(addr)
    }

    fn supports_timer_phase_clear(&self) -> bool {
        self.inner.supports_timer_phase_clear()
    }

    fn peek_byte_silent_at(&mut self, addr: u32, context_pc: u32) -> Option<u8> {
        let value = self.inner.peek_byte_silent_at(addr, context_pc);
        if value.is_none() {
            self.unavailable = true;
        }
        value
    }

    fn resolve_emem(&mut self, base: u32) -> u32 {
        self.inner.resolve_emem(base)
    }

    fn peek_imem_silent(&mut self, offset: u32) -> u8 {
        self.peek_byte_silent(INTERNAL_MEMORY_START + offset)
            .unwrap_or(0)
    }

    fn supports_wait_cycles(&self) -> bool {
        self.inner.supports_wait_cycles()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExtRegMode {
    Simple,
    PostInc,
    PreDec,
    Offset,
}

#[derive(Debug, Clone, Copy)]
struct MemOperand {
    addr: u32,
    bits: u8,
    side_effect: Option<(RegName, u32)>, // register, new value
}

#[derive(Debug, Clone, Copy)]
struct EmemImemTransfer {
    dst_addr: u32,
    src_addr: u32,
    bits: u8,
    dst_is_internal: bool,
    side_effect: Option<(RegName, u32)>,
}

#[derive(Debug, Clone, Copy, Default)]
struct DecodedOperands {
    mem: Option<MemOperand>,
    mem2: Option<MemOperand>,
    imm: Option<(u32, u8)>, // value, bits
    len: u8,
    transfer: Option<EmemImemTransfer>,
    reg3: Option<RegName>,
    reg_pair: Option<(RegName, RegName, u8)>, // (dst, src, bits)
}

fn read_imem_byte<B: LlamaBus>(bus: &mut B, offset: u32) -> u8 {
    bus.load(INTERNAL_MEMORY_START + offset, 8) as u8
}

fn write_imem_byte<B: LlamaBus>(bus: &mut B, offset: u32, value: u8) {
    LlamaExecutor::store_traced(bus, INTERNAL_MEMORY_START + offset, 8, value as u32);
}

fn pre_modes_for(opcode: u8) -> Option<PreModes> {
    PRE_MODES
        .iter()
        .find(|(pre, _, _)| *pre == opcode)
        .map(|(_, first, second)| PreModes {
            first: *first,
            second: *second,
        })
}

fn mode_for_operand(pre: Option<&PreModes>, operand_index: usize) -> AddressingMode {
    match pre {
        Some(modes) => {
            if operand_index == 0 {
                modes.first
            } else {
                modes.second
            }
        }
        None => AddressingMode::BpN,
    }
}

fn operand_uses_pre_mode(op: &OperandKind) -> bool {
    matches!(
        op,
        OperandKind::IMem(_) | OperandKind::IMemWidth(_) | OperandKind::EMemIMemWidth(_)
    )
}

fn pre_selector_count(entry: &OpcodeEntry) -> usize {
    entry
        .operands
        .iter()
        .map(|operand| match operand {
            OperandKind::IMem(_)
            | OperandKind::IMemWidth(_)
            | OperandKind::EMemIMemWidth(_)
            | OperandKind::RegIMemOffset(_) => 1,
            OperandKind::EMemImemOffsetDestIntMem | OperandKind::EMemImemOffsetDestExtMem => 2,
            _ => 0,
        })
        .sum()
}

fn validate_canonical_pre(
    prefix_opcode: u8,
    modes: &PreModes,
    entry: &OpcodeEntry,
) -> Result<(), &'static str> {
    match pre_selector_count(entry) {
        0 => Err("PRE prefix has no addressable internal-memory operand"),
        1 => {
            let proven = match modes.first {
                AddressingMode::BpN => matches!(prefix_opcode, 0x22),
                AddressingMode::N => matches!(prefix_opcode, 0x30..=0x33),
                AddressingMode::PxN => matches!(prefix_opcode, 0x34 | 0x36),
                AddressingMode::BpPx => matches!(prefix_opcode, 0x24 | 0x26),
                AddressingMode::PyN | AddressingMode::BpPy => false,
            };
            if proven {
                Ok(())
            } else {
                Err("noncanonical PRE prefix for one internal-memory operand")
            }
        }
        2 => Ok(()),
        _ => Err("unsupported PRE operand shape"),
    }
}

fn validate_imem_selector(mode: AddressingMode, raw: u8) -> Result<(), &'static str> {
    if mode == AddressingMode::BpPy && raw != 0 {
        Err("nonzero selector byte is invalid for unverified BP+PY addressing")
    } else {
        Ok(())
    }
}

fn imem_offset_for_mode<B: LlamaBus>(bus: &mut B, mode: AddressingMode, raw: u8) -> u32 {
    let bp = read_imem_byte(bus, IMEM_BP_OFFSET) as u32;
    let px = read_imem_byte(bus, IMEM_PX_OFFSET) as u32;
    let py = read_imem_byte(bus, IMEM_PY_OFFSET) as u32;
    let base = match mode {
        AddressingMode::N => raw as u32,
        AddressingMode::BpN => bp.wrapping_add(raw as u32),
        AddressingMode::PxN => px.wrapping_add(raw as u32),
        AddressingMode::PyN => py.wrapping_add(raw as u32),
        AddressingMode::BpPx => bp.wrapping_add(px),
        AddressingMode::BpPy => bp.wrapping_add(py),
    };
    trace_imem_addr(mode, base, bp, px, py);
    base & 0xFF
}

/// Emit the effective IMEM address and the raw registers used for BpPx/BpPy modes.
/// Fires when perfetto is active or TRACE_IMEM_ADDR=1 is set.
fn trace_imem_addr(mode: AddressingMode, base: u32, bp: u32, px: u32, py: u32) {
    if PREFLIGHT_DEPTH.with(|depth| depth.get() != 0) {
        return;
    }
    // Optional perfetto emit when the builder is available (llama-tests builds).
    let mut guard = crate::PERFETTO_TRACER.enter();
    guard.with_some(|tracer| {
        let op_idx = PERF_CURRENT_OP.with(|value| value.get());
        let pc = PERF_CURRENT_PC.with(|value| value.get());
        let op = if op_idx == u64::MAX {
            None
        } else {
            Some(op_idx)
        };
        let pc_val = if pc == u32::MAX { None } else { Some(pc) };
        tracer.record_imem_addr(
            &format!("{mode:?}"),
            base & 0xFF,
            bp & 0xFF,
            px & 0xFF,
            py & 0xFF,
            op,
            pc_val,
        );
    });
}

fn imem_addr_for_mode<B: LlamaBus>(bus: &mut B, mode: AddressingMode, raw: u8) -> u32 {
    INTERNAL_MEMORY_START + imem_offset_for_mode(bus, mode, raw)
}

fn enter_low_power_state<B: LlamaBus>(
    bus: &mut B,
    state: &mut LlamaState,
    power_state: PowerState,
) {
    // Mirror pysc62015.intrinsics._enter_low_power_state: adjust USR/SSR and halt.
    let mut usr = read_imem_byte(bus, IMEM_USR_OFFSET);
    usr &= !0x3F;
    usr |= 0x18;
    write_imem_byte(bus, IMEM_USR_OFFSET, usr);

    let mut ssr = read_imem_byte(bus, IMEM_SSR_OFFSET);
    ssr |= 0x04;
    write_imem_byte(bus, IMEM_SSR_OFFSET, ssr);

    if power_state == PowerState::Off {
        state.record_off_transition(state.pc());
    }
    state.set_power_state(power_state);
}

/// Apply power-on reset side effects (IMEM init, PC jump to reset vector).
pub fn power_on_reset<B: LlamaBus>(
    bus: &mut B,
    state: &mut LlamaState,
) -> Result<(), &'static str> {
    // Resolve and validate the complete transfer before RESET changes any SFR,
    // call-page, power, or PC state. The fixed vector remains an architectural
    // read, but even a volatile mismatch is rejected before the first write.
    let transfer = fetch_validated_vector(ROM_RESET_VECTOR_ADDR, state, bus)?;
    apply_power_on_reset(bus, state, transfer.target());
    Ok(())
}

/// Commit RESET using a vector that the caller already silently validated and
/// architecturally fetched before entering this layer. This path deliberately
/// performs no vector or destination read: machine wrappers may have reset
/// RAM/LCD state after the fetch, so any later rejectable bus access would make
/// the wrapper-level operation non-atomic.
pub fn power_on_reset_with_transfer<B: LlamaBus>(
    bus: &mut B,
    state: &mut LlamaState,
    transfer: ValidatedVectorTransfer,
) -> Result<(), &'static str> {
    if !transfer.architectural_fetch_validated()
        || !transfer.matches(ROM_RESET_VECTOR_ADDR, state, bus)
    {
        return Err(PREPARED_VECTOR_MISMATCH_ERROR);
    }
    apply_power_on_reset(bus, state, transfer.target());
    Ok(())
}

fn apply_power_on_reset<B: LlamaBus>(bus: &mut B, state: &mut LlamaState, reset_vector: u32) {
    // RESET intrinsic side-effects (see pysc62015.intrinsics.eval_intrinsic_reset)
    // Parity: IMR is intentionally left unchanged.
    let mut lcc = read_imem_byte(bus, IMEM_LCC_OFFSET);
    lcc &= !0x80;
    write_imem_byte(bus, IMEM_LCC_OFFSET, lcc);

    write_imem_byte(bus, IMEM_UCR_OFFSET, 0);
    write_imem_byte(bus, IMEM_ISR_OFFSET, 0);
    write_imem_byte(bus, IMEM_SCR_OFFSET, 0);

    let mut usr = read_imem_byte(bus, IMEM_USR_OFFSET);
    usr &= !0x3F;
    usr |= 0x18;
    write_imem_byte(bus, IMEM_USR_OFFSET, usr);

    let mut ssr = read_imem_byte(bus, IMEM_SSR_OFFSET);
    ssr &= !0x04;
    write_imem_byte(bus, IMEM_SSR_OFFSET, ssr);

    // Parity: keep register/flag values intact; only adjust IMEM/PC. Drop any saved
    // call-page context so near returns fall back to the current page like Python.
    state.clear_call_page_stack();
    state.set_pc(reset_vector);
    state.set_halted(false);
}

/// Validate a fixed three-byte vector and its destination using only silent
/// memory peeks. This is an emulator-integrity quarantine for unverified upper
/// bits, not a claim that silicon traps on the encoding.
pub fn validate_vector_transfer<B: LlamaBus>(
    vector_addr: u32,
    state: &LlamaState,
    bus: &mut B,
) -> Result<u32, &'static str> {
    LlamaExecutor
        .validate_vector_transfer_inner(vector_addr, state, bus)
        .map(|(target, _length)| target)
}

pub fn validate_vector_transfer_with_length<B: LlamaBus>(
    vector_addr: u32,
    state: &LlamaState,
    bus: &mut B,
) -> Result<(u32, u8), &'static str> {
    LlamaExecutor.validate_vector_transfer_inner(vector_addr, state, bus)
}

/// Prepare an IRQ/IR vector transfer using only side-effect-free peeks.
/// The architectural low-to-high vector reads must be performed later, after
/// the complete interrupt frame has been written.
pub fn prepare_validated_vector<B: LlamaBus>(
    vector_addr: u32,
    state: &LlamaState,
    bus: &mut B,
) -> Result<ValidatedVectorTransfer, &'static str> {
    let provenance = bus.vector_transfer_provenance();
    let (target, target_len) =
        LlamaExecutor.validate_vector_transfer_inner(vector_addr, state, bus)?;
    if bus.vector_transfer_provenance() != provenance {
        return Err(PREPARED_VECTOR_MISMATCH_ERROR);
    }
    Ok(ValidatedVectorTransfer {
        vector_addr,
        source_pc: state.pc() & mask_for(RegName::PC),
        target,
        target_len,
        provenance,
        architectural_fetch_validated: false,
    })
}

/// Perform the one architectural vector fetch after silent validation and
/// validate the fetched value again if a volatile bus disagrees with the peek.
/// All rejection points occur before the caller mutates architectural state.
pub fn fetch_validated_vector<B: LlamaBus>(
    vector_addr: u32,
    state: &LlamaState,
    bus: &mut B,
) -> Result<ValidatedVectorTransfer, &'static str> {
    let provenance = bus.vector_transfer_provenance();
    let (silent_vector, target_len) =
        LlamaExecutor.validate_vector_transfer_inner(vector_addr, state, bus)?;
    let fetched_vector = bus.load(vector_addr, 8)
        | (bus.load(vector_addr.wrapping_add(1), 8) << 8)
        | (bus.load(vector_addr.wrapping_add(2), 8) << 16);
    if bus.vector_transfer_provenance() != provenance {
        return Err(PREPARED_VECTOR_MISMATCH_ERROR);
    }
    if fetched_vector == silent_vector {
        return Ok(ValidatedVectorTransfer {
            vector_addr,
            source_pc: state.pc() & mask_for(RegName::PC),
            target: fetched_vector,
            target_len,
            provenance,
            architectural_fetch_validated: true,
        });
    }
    if fetched_vector & 0xF0_0000 != 0 {
        return Err(VECTOR_UPPER_NIBBLE_ERROR);
    }

    // A volatile but canonical vector still needs the same static destination
    // validation as the silently observed value.
    let opcode = bus
        .peek_byte_silent_at(fetched_vector, fetched_vector)
        .ok_or(SILENT_PEEK_UNAVAILABLE_ERROR)?;
    let mut target_state = state.clone();
    target_state.set_pc(fetched_vector);
    LlamaExecutor.validate_before_scheduling_with_options(
        opcode,
        &target_state,
        bus,
        false,
        false,
    )?;
    Err(VECTOR_CHANGED_DURING_PREFLIGHT_ERROR)
}

pub struct LlamaExecutor;

impl LlamaExecutor {
    pub fn new() -> Self {
        Self
    }

    pub fn lookup(&self, opcode: u8) -> Option<&'static OpcodeEntry> {
        dispatch::lookup(opcode)
    }

    /// Reject instruction forms whose execution contract is already known to
    /// fail before a host scheduler advances timers or devices.
    ///
    /// The main evaluator repeats these checks at its architectural boundary;
    /// runners that tick peripherals before calling `execute` use this narrow
    /// preflight to preserve the stronger no-timing-mutation guarantee.
    pub fn validate_before_scheduling<B: LlamaBus>(
        &self,
        opcode: u8,
        state: &LlamaState,
        bus: &mut B,
    ) -> Result<(), &'static str> {
        self.validate_before_scheduling_with_options(opcode, state, bus, true, true)
            .map(|_| ())
    }

    pub fn validate_before_scheduling_with_length<B: LlamaBus>(
        &self,
        opcode: u8,
        state: &LlamaState,
        bus: &mut B,
    ) -> Result<u8, &'static str> {
        self.validate_before_scheduling_with_options(opcode, state, bus, true, true)
    }

    fn validate_before_scheduling_with_options<B: LlamaBus>(
        &self,
        opcode: u8,
        state: &LlamaState,
        bus: &mut B,
        validate_data_dependent: bool,
        validate_vector_target: bool,
    ) -> Result<u8, &'static str> {
        let _preflight_guard = PreflightGuard::enter();
        let mut bus = SilentPreflightBus {
            inner: bus,
            unavailable: false,
            context_pc: state.pc() & mask_for(RegName::PC),
        };
        self.validate_before_scheduling_silent(
            opcode,
            state,
            &mut bus,
            validate_data_dependent,
            validate_vector_target,
        )
    }

    fn validate_before_scheduling_silent<B: LlamaBus>(
        &self,
        opcode: u8,
        state: &LlamaState,
        bus: &mut SilentPreflightBus<'_, B>,
        validate_data_dependent: bool,
        validate_vector_target: bool,
    ) -> Result<u8, &'static str> {
        let mut exec_pc = state.pc() & mask_for(RegName::PC);
        let mut entry = self.lookup(opcode);
        let mut prefix_len = 0u8;
        let mut pre_modes_opt: Option<PreModes> = None;
        let mut effective_pre_opcode = None;

        while let Some(resolved) = entry {
            if resolved.kind != InstrKind::Pre {
                break;
            }
            if prefix_len >= 2 {
                return Err("more than two consecutive PRE prefixes are unverified");
            }
            let pre_modes = pre_modes_for(resolved.opcode).ok_or("unknown PRE opcode")?;
            exec_pc = exec_pc.wrapping_add(1) & mask_for(RegName::PC);
            let next_opcode = Self::fetch_byte(bus, exec_pc);
            if bus.unavailable {
                return Err(SILENT_PEEK_UNAVAILABLE_ERROR);
            }
            prefix_len = prefix_len.saturating_add(1);
            pre_modes_opt = Some(pre_modes);
            effective_pre_opcode = Some(resolved.opcode);
            entry = self.lookup(next_opcode);
        }

        let resolved = entry.ok_or("invalid or reserved opcode")?;
        if let Some(pre_modes) = pre_modes_opt.as_ref() {
            validate_canonical_pre(
                effective_pre_opcode.ok_or("missing PRE opcode")?,
                pre_modes,
                resolved,
            )?;
        }
        if resolved.kind == InstrKind::Unknown {
            return Err("invalid or reserved opcode");
        }
        if resolved.kind == InstrKind::Tcl && !bus.supports_timer_phase_clear() {
            return Err(TCL_UNIMPLEMENTED_ERROR);
        }
        if resolved.kind == InstrKind::Wait && !bus.supports_wait_cycles() {
            return Err("WAIT requires a cycle-capable bus");
        }

        // Decode the complete operand shape against a cloned register image.
        // This catches malformed JP/E3/EB selectors, ignored selector bytes,
        // and noncanonical encoded 20-bit operands before scheduler mutation.
        let mut state_candidate = state.clone();
        let decoded_result = self.decode_operands(
            resolved,
            &mut state_candidate,
            bus,
            pre_modes_opt.as_ref(),
            Some(exec_pc),
        );
        if bus.unavailable {
            return Err(SILENT_PEEK_UNAVAILABLE_ERROR);
        }
        let decoded = decoded_result?;
        let instruction_len = decoded.len.saturating_add(prefix_len);

        if validate_vector_target {
            let vector_addr = match resolved.kind {
                InstrKind::Ir => Some(INTERRUPT_VECTOR_ADDR),
                InstrKind::Reset => Some(ROM_RESET_VECTOR_ADDR),
                _ => None,
            };
            if let Some(vector_addr) = vector_addr {
                self.validate_vector_transfer_silent(vector_addr, state, bus)?;
            }
        }

        // A vector destination is checked only for instruction forms that can
        // be rejected without reading mutable architectural data. Stack-image
        // checks remain the destination instruction's job.
        if !validate_data_dependent {
            return Ok(instruction_len);
        }

        Ok(instruction_len)
    }

    fn validate_vector_transfer_inner<B: LlamaBus>(
        &self,
        vector_addr: u32,
        state: &LlamaState,
        bus: &mut B,
    ) -> Result<(u32, u8), &'static str> {
        if vector_addr > mask_for(RegName::PC) {
            return Err("SC62015 vector address must be canonical 20-bit");
        }
        let _preflight_guard = PreflightGuard::enter();
        let mut bus = SilentPreflightBus {
            inner: bus,
            unavailable: false,
            context_pc: state.pc() & mask_for(RegName::PC),
        };
        self.validate_vector_transfer_silent(vector_addr, state, &mut bus)
    }

    fn validate_vector_transfer_silent<B: LlamaBus>(
        &self,
        vector_addr: u32,
        state: &LlamaState,
        bus: &mut SilentPreflightBus<'_, B>,
    ) -> Result<(u32, u8), &'static str> {
        let b0 = bus.peek_byte_silent(vector_addr).unwrap_or(0);
        let b1 = bus
            .peek_byte_silent(vector_addr.wrapping_add(1))
            .unwrap_or(0);
        let b2 = bus
            .peek_byte_silent(vector_addr.wrapping_add(2))
            .unwrap_or(0);
        if bus.unavailable {
            return Err(SILENT_PEEK_UNAVAILABLE_ERROR);
        }
        let target = u32::from(b0) | (u32::from(b1) << 8) | (u32::from(b2) << 16);
        if target & 0xF0_0000 != 0 {
            return Err(VECTOR_UPPER_NIBBLE_ERROR);
        }

        let previous_context = bus.context_pc;
        bus.context_pc = target;
        let target_result = (|| {
            let opcode = bus.peek_byte_silent(target).unwrap_or(0);
            if bus.unavailable {
                return Err(SILENT_PEEK_UNAVAILABLE_ERROR);
            }
            let mut target_state = state.clone();
            target_state.set_pc(target);
            self.validate_before_scheduling_silent(opcode, &target_state, bus, false, false)
        })();
        bus.context_pc = previous_context;
        let target_len = target_result?;
        Ok((target, target_len))
    }

    fn push_stack<B: LlamaBus>(
        state: &mut LlamaState,
        bus: &mut B,
        sp_reg: RegName,
        value: u32,
        bits: u8,
        big_endian: bool,
    ) {
        let bytes = bits.div_ceil(8);
        let mask = mask_for(sp_reg);
        let mut sp = state.get_reg(sp_reg) & mask;
        for i in (0..bytes).rev() {
            sp = sp.wrapping_sub(1) & mask;
            let shift = if big_endian {
                8 * (bytes.saturating_sub(1) - i)
            } else {
                8 * i
            };
            let byte = (value >> shift) & 0xFF;
            Self::store_traced(bus, sp, 8, byte);
        }
        state.set_reg(sp_reg, sp);
    }

    fn pop_stack<B: LlamaBus>(
        state: &mut LlamaState,
        bus: &mut B,
        sp_reg: RegName,
        bits: u8,
        big_endian: bool,
    ) -> u32 {
        let bytes = bits.div_ceil(8);
        let mut value = 0u32;
        let mask = mask_for(sp_reg);
        let mut sp = state.get_reg(sp_reg);
        for i in 0..bytes {
            let byte = bus.load(sp, 8) & 0xFF;
            let shift = if big_endian {
                8 * (bytes.saturating_sub(1) - i)
            } else {
                8 * i
            };
            value |= byte << shift;
            sp = sp.wrapping_add(1) & mask;
        }
        state.set_reg(sp_reg, sp);
        value & Self::mask_for_width(bits)
    }

    fn cond_pass(entry: &OpcodeEntry, state: &LlamaState) -> Result<bool, &'static str> {
        match entry.cond {
            None => Ok(true),
            Some("Z") => Ok(state.get_reg(RegName::FZ) & 1 == 1),
            Some("NZ") => Ok(state.get_reg(RegName::FZ) & 1 == 0),
            Some("C") => Ok(state.get_reg(RegName::FC) & 1 == 1),
            Some("NC") => Ok(state.get_reg(RegName::FC) & 1 == 0),
            _ => Err("unsupported branch condition"),
        }
    }

    fn reg_name_for_trace(reg: RegName) -> &'static str {
        match reg {
            RegName::A => "A",
            RegName::B => "B",
            RegName::BA => "BA",
            RegName::IL => "IL",
            RegName::IH => "IH",
            RegName::I => "I",
            RegName::X => "X",
            RegName::Y => "Y",
            RegName::U => "U",
            RegName::S => "S",
            RegName::PC => "PC",
            RegName::F => "F",
            RegName::FC => "FC",
            RegName::FZ => "FZ",
            RegName::IMR => "IMR",
            RegName::Temp(_) => "TEMP",
            RegName::Unknown(_) => "UNKNOWN",
        }
    }

    fn emit_control_flow_event(
        label: &str,
        kind: &str,
        instr_index: u64,
        pc: u32,
        payload: HashMap<String, AnnotationValue>,
    ) {
        let mut guard = PERFETTO_TRACER.enter();
        guard.with_some(|tracer| {
            let mut payload = payload;
            payload.insert(
                "cf_kind".to_string(),
                AnnotationValue::Str(kind.to_string()),
            );
            tracer.record_control_flow(label, instr_index, pc & mask_for(RegName::PC), payload);
        });
    }

    fn trace_instr<B: LlamaBus>(
        &self,
        opcode: u8,
        regs: &HashMap<String, u32>,
        bus: &mut B,
        instr_index: u64,
        pc_trace: u32,
    ) {
        let mut guard = PERFETTO_TRACER.enter();
        guard.with_some(|tracer| {
            let (mem_imr, mem_isr) = with_imr_read_suppressed(|| {
                (
                    bus.peek_imem_silent(IMEM_IMR_OFFSET),
                    bus.peek_imem_silent(IMEM_ISR_OFFSET),
                )
            });
            let timer = bus.timer_trace();
            let cycle = bus.cycle_count();
            let mnemonic = dispatch::lookup(opcode).map(|entry| entry.name);
            tracer.record_regs(
                instr_index,
                pc_trace & mask_for(RegName::PC),
                pc_trace & mask_for(RegName::PC),
                opcode,
                mnemonic,
                regs,
                mem_imr,
                mem_isr,
                timer.map(|t| t.mti_ticks),
                timer.map(|t| t.sti_ticks),
                cycle,
            );
        });
        PERF_LAST_PC.with(|value| value.set(pc_trace));
    }

    fn estimated_length(entry: &OpcodeEntry) -> u8 {
        let mut len = 1u8; // opcode byte
        for op in entry.operands.iter() {
            len = len.saturating_add(match op {
                OperandKind::Imm(bits) => bits.div_ceil(8),
                OperandKind::ImmOffset => 1,
                OperandKind::IMem(_) | OperandKind::IMemWidth(_) => 1,
                OperandKind::EMemAddrWidth(_) | OperandKind::EMemAddrWidthOp(_) => 3,
                // EMemReg/IMem variants encode a mode byte plus an optional displacement.
                OperandKind::EMemReg(_)
                | OperandKind::EMemRegWidth(_)
                | OperandKind::EMemRegWidthMode(_)
                | OperandKind::EMemRegModePostPre => 2,
                // EMemIMem uses a mode byte + base + optional displacement.
                OperandKind::EMemIMem(_) | OperandKind::EMemIMemWidth(_) => 3,
                // Offset IMEM/EMEM transfer forms consume mode + two IMEM bytes + optional disp.
                OperandKind::EMemImemOffsetDestIntMem | OperandKind::EMemImemOffsetDestExtMem => 4,
                // Reg+IMEM offset encodings carry a mode/displacement byte plus IMEM selector.
                OperandKind::RegIMemOffset(_) => 3,
                // Reg pair selector is always a single byte regardless of data width.
                OperandKind::RegPair(_) => 1,
                _ => 0,
            });
        }
        len
    }

    fn set_flags_for_result(state: &mut LlamaState, result: u32, carry: Option<bool>) {
        state.set_reg(RegName::FZ, if result == 0 { 1 } else { 0 });
        if let Some(c) = carry {
            state.set_reg(RegName::FC, if c { 1 } else { 0 });
        }
    }

    fn trace_mem_write(addr: u32, bits: u8, value: u32) {
        let mut guard = PERFETTO_TRACER.enter();
        guard.with_some(|tracer| {
            let op_index = PERF_CURRENT_OP.with(|value| value.get());
            let pc = PERF_CURRENT_PC.with(|value| value.get());
            let substep = perfetto_next_substep();
            let masked = if bits == 0 || bits >= 32 {
                value
            } else {
                value & ((1u32 << bits) - 1)
            };
            let space = if (INTERNAL_MEMORY_START..(INTERNAL_MEMORY_START + 0x100)).contains(&addr)
            {
                "internal"
            } else {
                "external"
            };
            tracer.record_mem_write_with_substep(op_index, pc, addr, masked, space, bits, substep);
        });
    }

    fn store_traced<B: LlamaBus>(bus: &mut B, addr: u32, bits: u8, value: u32) {
        let bytes = bits.div_ceil(8).max(1);
        // SC62015 memory transfers are observable as ordered byte accesses,
        // including when a wide operand does not cross an address boundary.
        // Splitting here also prevents a byte-wide host callback from silently
        // receiving only the low byte of a word or pointer store.
        for index in 0..bytes {
            let byte_addr = Self::advance_internal_addr(addr, index as u32);
            let byte = (value >> (8 * index)) & 0xFF;
            bus.store(byte_addr, 8, byte);
            Self::trace_mem_write(byte_addr, 8, byte);
        }
    }

    fn set_flags_cmp(state: &mut LlamaState, lhs: u32, rhs: u32, bits: u8) {
        let mask = Self::mask_for_width(bits);
        let res = lhs.wrapping_sub(rhs) & mask;
        let borrow = (lhs & mask) < (rhs & mask);
        state.set_reg(RegName::FZ, if res == 0 { 1 } else { 0 });
        state.set_reg(RegName::FC, if borrow { 1 } else { 0 });
    }

    fn alu_unary<F: Fn(u32, u8) -> u32>(
        state: &mut LlamaState,
        reg: RegName,
        bits: u8,
        op: F,
    ) -> u32 {
        let mask = Self::mask_for_width(bits);
        let val = state.get_reg(reg) & mask;
        let res = op(val, bits) & mask;
        state.set_reg(reg, res);
        Self::set_flags_for_result(state, res, None);
        res
    }

    fn mask_for_width(bits: u8) -> u32 {
        if bits == 0 {
            0
        } else if bits >= 32 {
            u32::MAX
        } else {
            (1u32 << bits) - 1
        }
    }

    fn bits_from_bytes(bytes: u8) -> u8 {
        bytes.saturating_mul(8)
    }

    fn read_reg<B: LlamaBus>(state: &mut LlamaState, bus: &mut B, reg: RegName) -> u32 {
        if reg == RegName::IMR {
            let val = bus.peek_imem(IMEM_IMR_OFFSET) as u32;
            state.set_reg(RegName::IMR, val);
            return val;
        }
        state.get_reg(reg)
    }

    fn fetch_byte<B: LlamaBus>(bus: &mut B, addr: u32) -> u8 {
        // The instruction stream is in the 20-bit external address space.
        // Fetching an operand after PC=0xFFFFF wraps to external 0x00000;
        // it must not fall through into the distinct IMEM window at 0x100000.
        (bus.load(addr & mask_for(RegName::PC), 8) & 0xFF) as u8
    }

    fn fetch_imm<B: LlamaBus>(bus: &mut B, addr: u32, bits: u8) -> u32 {
        let bytes = bits.div_ceil(8);
        let mut value = 0u32;
        for index in 0..bytes {
            value |= u32::from(Self::fetch_byte(bus, addr + u32::from(index))) << (8 * index);
        }
        value & Self::mask_for_width(bits)
    }

    fn fetch_encoded_20bit<B: LlamaBus>(bus: &mut B, addr: u32) -> Result<u32, &'static str> {
        let lo = u32::from(Self::fetch_byte(bus, addr));
        let mid = u32::from(Self::fetch_byte(bus, addr.wrapping_add(1)));
        let high = u32::from(Self::fetch_byte(bus, addr.wrapping_add(2)));
        if high & 0xF0 != 0 {
            return Err(ENCODED_20BIT_UPPER_NIBBLE_ERROR);
        }
        Ok(lo | (mid << 8) | (high << 16))
    }

    fn fetch_register_20bit<B: LlamaBus>(bus: &mut B, addr: u32) -> u32 {
        // X/Y hardware captures execute a high byte of 0x3C and subsequently
        // push 0x0C. The register-load opcodes consume three bytes but retain
        // bits 19-0. Do not generalize this silicon result to vectors or
        // arbitrary external-address operands. Separately measured HW-009
        // opcode families have their own explicitly scoped masked fetches.
        Self::fetch_imm(bus, addr, 24) & mask_for(RegName::X)
    }

    fn fetch_hw009_far_control_address<B: LlamaBus>(bus: &mut B, addr: u32) -> u32 {
        // PC-E500 HW-009 executed otherwise-identical JPF/CALLF pairs with
        // operand high bytes 0x01 and 0x81. Both reached 0x101E0; CALLF also
        // produced the same return frame and RETF path. Only these two far
        // control opcodes inherit that masking result here.
        Self::fetch_imm(bus, addr, 24) & mask_for(RegName::PC)
    }

    fn fetch_hw009_masked_data_address<B: LlamaBus>(bus: &mut B, addr: u32) -> u32 {
        // PC-E500 HW-009 executed 62/66/6A/72/7A with encoded high byte 0x84
        // like their 0x04 controls, the complete 88-8F direct-read block with
        // high byte 0x81 like 0x01, every A8-AF direct store to encoded
        // 0x8406D0 like 0x0406D0, and fixed/counted D0-D3/D8-DB transfers in
        // canonical/noncanonical pairs. These prove only the named
        // data-address consumers mask bits 23-20. Other absolute-memory
        // families and control-flow operands remain fail-closed.
        Self::fetch_imm(bus, addr, 24) & mask_for(RegName::PC)
    }

    fn read_imm<B: LlamaBus>(bus: &mut B, addr: u32, bits: u8) -> u32 {
        let bytes = bits.div_ceil(8);
        let mut value = 0u32;
        for i in 0..bytes {
            let byte_addr = Self::advance_internal_addr(addr, i as u32);
            value |= (bus.load(byte_addr, 8) & 0xFF) << (8 * i);
        }
        if bits >= 32 {
            value
        } else {
            let mask = if bits == 0 { 0 } else { (1u32 << bits) - 1 };
            value & mask
        }
    }

    fn is_internal_addr(addr: u32) -> bool {
        (INTERNAL_MEMORY_START..(INTERNAL_MEMORY_START + 0x100)).contains(&addr)
    }

    fn load_wrapped<B: LlamaBus>(bus: &mut B, addr: u32, bits: u8) -> u32 {
        let bytes = bits.div_ceil(8).max(1);
        // PC-E500 HW-007 observes F1/F2 pointer-source reads as one bus access
        // per byte even away from a wrap boundary. Always splitting also makes
        // byte-wide host callbacks see the complete architectural access.
        let mut value = 0u32;
        for index in 0..bytes {
            let byte_addr = Self::advance_internal_addr(addr, index as u32);
            value |= (bus.load(byte_addr, 8) & 0xFF) << (8 * index);
        }
        value & Self::mask_for_width(bits)
    }

    fn advance_internal_addr(addr: u32, step: u32) -> u32 {
        if Self::is_internal_addr(addr) {
            let offset = addr.wrapping_sub(INTERNAL_MEMORY_START);
            let wrapped = offset.wrapping_add(step) & 0xFF;
            INTERNAL_MEMORY_START + wrapped
        } else {
            addr.wrapping_add(step) & mask_for(RegName::X)
        }
    }

    fn addr_step_from_side_effect(reg: RegName, curr: u32, new_val: u32) -> i32 {
        let mask = mask_for(reg);
        let masked_curr = curr & mask;
        let masked_new = new_val & mask;
        if masked_new == masked_curr {
            return 0;
        }
        if masked_new > masked_curr {
            (masked_new.wrapping_sub(masked_curr)) as i32
        } else {
            -(masked_curr.wrapping_sub(masked_new) as i32)
        }
    }

    fn advance_internal_addr_signed(addr: u32, step: i32) -> u32 {
        if Self::is_internal_addr(addr) {
            let offset = addr.wrapping_sub(INTERNAL_MEMORY_START);
            let next = if step >= 0 {
                offset.wrapping_add(step as u32)
            } else {
                offset.wrapping_sub((-step) as u32)
            } & 0xFF;
            INTERNAL_MEMORY_START + next
        } else if step >= 0 {
            addr.wrapping_add(step as u32) & mask_for(RegName::X)
        } else {
            addr.wrapping_sub((-step) as u32) & mask_for(RegName::X)
        }
    }

    fn bcd_add_byte(a: u8, b: u8, carry_in: bool) -> (u8, bool) {
        let mut low_sum = (a & 0x0F)
            .wrapping_add(b & 0x0F)
            .wrapping_add(carry_in as u8);
        let low_adjust = if low_sum > 9 { 6 } else { 0 };
        low_sum = low_sum.wrapping_add(low_adjust);
        let carry_to_high = (low_sum & 0x10) != 0;
        let res_low = low_sum & 0x0F;

        let mut high_sum = ((a >> 4) & 0x0F)
            .wrapping_add((b >> 4) & 0x0F)
            .wrapping_add(carry_to_high as u8);
        let high_adjust = if high_sum > 9 { 6 } else { 0 };
        high_sum = high_sum.wrapping_add(high_adjust);
        let carry_out = (high_sum & 0x10) != 0;
        let res_high = high_sum & 0x0F;

        (((res_high << 4) | res_low), carry_out)
    }

    fn bcd_sub_byte(a: u8, b: u8, borrow_in: bool) -> (u8, bool) {
        let sub_low = (b & 0x0F).wrapping_add(borrow_in as u8);
        let mut low_res = (a & 0x0F).wrapping_sub(sub_low);
        let borrow_low = (a & 0x0F) < sub_low;
        if borrow_low {
            low_res = low_res.wrapping_sub(6);
        }
        let res_low = low_res & 0x0F;

        let sub_high = ((b >> 4) & 0x0F).wrapping_add(borrow_low as u8);
        let mut high_res = ((a >> 4) & 0x0F).wrapping_sub(sub_high);
        let borrow_out = ((a >> 4) & 0x0F) < sub_high;
        if borrow_out {
            high_res = high_res.wrapping_sub(6);
        }
        let res_high = high_res & 0x0F;

        (((res_high << 4) | res_low), borrow_out)
    }

    fn reg_from_selector(sel: u8) -> Option<RegName> {
        match sel & 0x07 {
            0 => Some(RegName::A),
            1 => Some(RegName::IL),
            2 => Some(RegName::BA),
            3 => Some(RegName::I),
            4 => Some(RegName::X),
            5 => Some(RegName::Y),
            6 => Some(RegName::U),
            7 => Some(RegName::S),
            _ => unreachable!("selector is masked to three bits"),
        }
    }

    fn decode_ext_reg_ptr<B: LlamaBus>(
        &self,
        state: &mut LlamaState,
        bus: &mut B,
        pc: u32,
        width_bytes: u8,
    ) -> Result<(MemOperand, u32), &'static str> {
        let reg_byte = Self::fetch_byte(bus, pc);
        if reg_byte & 0x07 < 4 {
            return Err("external-memory pointer requires a three-byte register");
        }
        let raw_mode = (reg_byte >> 4) & 0x0F;
        let (mode, needs_disp, disp_sign) = match raw_mode {
            0x0 => (ExtRegMode::Simple, false, 0),
            0x2 => (ExtRegMode::PostInc, false, 0),
            0x3 => (ExtRegMode::PreDec, false, 0),
            0x8 => (ExtRegMode::Offset, true, 1),
            0xC => (ExtRegMode::Offset, true, -1),
            _ => return Err("unsupported EMEM reg mode"),
        };
        let reg = Self::reg_from_selector(reg_byte).ok_or("invalid reg selector")?;

        let mut consumed = 1u32;
        let mut disp: i16 = 0;
        if needs_disp {
            let magnitude = Self::fetch_byte(bus, pc + 1);
            disp = if disp_sign >= 0 {
                magnitude as i16
            } else {
                -(magnitude as i16)
            };
            consumed += 1;
        }

        let base = state.get_reg(reg);
        let step = width_bytes as u32;
        let mask = mask_for(reg);
        let mut addr = base;
        let mut side_effect: Option<(RegName, u32)> = None;
        match mode {
            ExtRegMode::Simple => {}
            ExtRegMode::Offset => {
                addr = base.wrapping_add(disp as u32);
            }
            ExtRegMode::PreDec => {
                addr = base.wrapping_sub(step) & mask;
                side_effect = Some((reg, addr));
            }
            ExtRegMode::PostInc => {
                side_effect = Some((reg, (base.wrapping_add(step)) & mask));
            }
        }
        let bits = Self::bits_from_bytes(width_bytes);
        Ok((
            MemOperand {
                addr: bus.resolve_emem(addr & mask_for(RegName::X)),
                bits,
                side_effect,
            },
            consumed,
        ))
    }

    fn decode_imem_ptr<B: LlamaBus>(
        &self,
        bus: &mut B,
        pc: u32,
        width_bytes: u8,
        mode: AddressingMode,
    ) -> Result<(MemOperand, u32), &'static str> {
        let mode_byte = Self::fetch_byte(bus, pc);
        let (needs_disp, sign) = match mode_byte {
            0x00 => (false, 0),
            0x80 => (true, 1),
            0xC0 => (true, -1),
            _ => return Err("unsupported EMEM/IMEM mode"),
        };
        let base_raw = Self::fetch_byte(bus, pc + 1);
        validate_imem_selector(mode, base_raw)?;
        let base = imem_addr_for_mode(bus, mode, base_raw);
        let mut consumed = 2u32;
        let mut disp: i16 = 0;
        if needs_disp {
            let magnitude = Self::fetch_byte(bus, pc + 2);
            disp = if sign >= 0 {
                magnitude as i16
            } else {
                -(magnitude as i16)
            };
            consumed += 1;
        }
        let pointer = Self::read_imm(bus, base, 24);
        let addr = bus.resolve_emem(pointer.wrapping_add(disp as u32) & mask_for(RegName::X));
        let bits = Self::bits_from_bytes(width_bytes);
        Ok((
            MemOperand {
                addr,
                bits,
                side_effect: None,
            },
            consumed,
        ))
    }

    fn width_bits_for_kind(kind: InstrKind) -> u8 {
        match kind {
            InstrKind::Mvw => 16,
            InstrKind::Mvp => 24,
            InstrKind::Mvl => 8, // byte stride per spec
            _ => 8,
        }
    }

    fn decode_emem_imem_offset<B: LlamaBus>(
        &self,
        entry: &OpcodeEntry,
        bus: &mut B,
        pc: u32,
        mode_first: AddressingMode,
        mode_second: AddressingMode,
        dest_is_internal: bool,
    ) -> Result<(EmemImemTransfer, u32), &'static str> {
        let mode_byte = Self::fetch_byte(bus, pc);
        let (needs_offset, sign) = match mode_byte {
            0x00 => (false, 0),
            0x80 => (true, 1),
            0xC0 => (true, -1),
            _ => return Err("unsupported EMEM/IMEM mode"),
        };
        let first = Self::fetch_byte(bus, pc + 1);
        let second = Self::fetch_byte(bus, pc + 2);
        validate_imem_selector(mode_first, first)?;
        validate_imem_selector(mode_second, second)?;
        let mut consumed = 3u32;
        let mut disp: i32 = 0;
        if needs_offset {
            let magnitude = Self::fetch_byte(bus, pc + 3);
            disp = if sign >= 0 {
                magnitude as i32
            } else {
                -(magnitude as i32)
            };
            consumed += 1;
        }
        let width_bits = Self::width_bits_for_kind(entry.kind);
        let first_addr = imem_addr_for_mode(bus, mode_first, first);
        let second_addr = imem_addr_for_mode(bus, mode_second, second);
        let (dst_addr, src_addr, dst_is_internal) = if dest_is_internal {
            // MV (m),[(n)] - dst is internal addr=first, ptr lives at second
            let pointer = Self::read_imm(bus, second_addr, 24);
            let ext_addr =
                bus.resolve_emem(pointer.wrapping_add(disp as u32) & mask_for(RegName::X));
            (first_addr, ext_addr, true)
        } else {
            // MV [(n)],(m) - dst is external pointer from first, src is internal second
            let pointer = Self::read_imm(bus, first_addr, 24);
            let ext_addr =
                bus.resolve_emem(pointer.wrapping_add(disp as u32) & mask_for(RegName::X));
            (ext_addr, second_addr, false)
        };
        Ok((
            EmemImemTransfer {
                dst_addr,
                src_addr,
                bits: width_bits,
                dst_is_internal,
                side_effect: None,
            },
            consumed,
        ))
    }

    fn decode_operands<B: LlamaBus>(
        &self,
        entry: &OpcodeEntry,
        state: &mut LlamaState,
        bus: &mut B,
        pre: Option<&PreModes>,
        pc_override: Option<u32>,
    ) -> Result<DecodedOperands, &'static str> {
        let pc = pc_override.unwrap_or(state.pc());
        let mut offset = 1u32; // opcode consumed
        let mut decoded = DecodedOperands::default();
        let single_pre = SINGLE_ADDRESSABLE_OPCODES.contains(&entry.opcode);
        let single_pre_operand = entry
            .operands
            .iter()
            .filter(|op| operand_uses_pre_mode(op))
            .count()
            == 1;
        // Opcode-specific decoding quirks
        if entry.opcode == 0xE3 {
            // Encoding order is EMemReg mode byte then IMem8.
            let raw_mode = (Self::fetch_byte(bus, pc + offset) >> 4) & 0x0F;
            if !matches!(raw_mode, 0x2 | 0x3) {
                return Err("E3 requires post-increment or pre-decrement addressing");
            }
            let (mem_src, consumed) = self.decode_ext_reg_ptr(state, bus, pc + offset, 1)?;
            decoded.mem2 = Some(mem_src);
            offset += consumed;
            let raw_imem = u32::from(Self::fetch_byte(bus, pc + offset));
            let imem_mode = mode_for_operand(pre, 0);
            validate_imem_selector(imem_mode, raw_imem as u8)?;
            let imem_addr = imem_addr_for_mode(bus, imem_mode, raw_imem as u8);
            decoded.mem = Some(MemOperand {
                addr: imem_addr,
                bits: 8,
                side_effect: None,
            });
            offset += 1;
            decoded.len = offset as u8;
            return Ok(decoded);
        }
        let coding_order: Vec<(usize, &OperandKind)> = if entry.ops_reversed.unwrap_or(false) {
            entry.operands.iter().enumerate().rev().collect()
        } else {
            entry.operands.iter().enumerate().collect()
        };
        for (operand_index, op) in coding_order {
            match op {
                OperandKind::Imm(bits) => {
                    let val = if *bits == 20 {
                        if matches!(entry.opcode, 0x03 | 0x05) {
                            Self::fetch_hw009_far_control_address(bus, pc + offset)
                        } else if matches!(entry.opcode, 0x0C..=0x0F) {
                            Self::fetch_register_20bit(bus, pc + offset)
                        } else {
                            Self::fetch_encoded_20bit(bus, pc + offset)?
                        }
                    } else {
                        Self::fetch_imm(bus, pc + offset, *bits)
                    };
                    decoded.imm = Some((val, *bits));
                    offset += (*bits as u32).div_ceil(8);
                }
                OperandKind::ImmOffset => {
                    let byte = Self::fetch_byte(bus, pc + offset);
                    decoded.imm = Some((byte as u32, 8));
                    offset += 1;
                }
                OperandKind::IMem(bits) => {
                    let raw = u32::from(Self::fetch_byte(bus, pc + offset));
                    let slot = if decoded.mem.is_none() {
                        &mut decoded.mem
                    } else {
                        &mut decoded.mem2
                    };
                    let mode_index = if single_pre || single_pre_operand {
                        0
                    } else {
                        operand_index
                    };
                    let imem_mode = mode_for_operand(pre, mode_index);
                    validate_imem_selector(imem_mode, raw as u8)?;
                    *slot = Some(MemOperand {
                        addr: imem_addr_for_mode(bus, imem_mode, raw as u8),
                        bits: *bits,
                        side_effect: None,
                    });
                    offset += 1;
                }
                OperandKind::IMemWidth(bytes) => {
                    let bits = Self::bits_from_bytes(*bytes);
                    let raw = u32::from(Self::fetch_byte(bus, pc + offset));
                    let slot = if decoded.mem.is_none() {
                        &mut decoded.mem
                    } else {
                        &mut decoded.mem2
                    };
                    let mode_index = if single_pre || single_pre_operand {
                        0
                    } else {
                        operand_index
                    };
                    let imem_mode = mode_for_operand(pre, mode_index);
                    validate_imem_selector(imem_mode, raw as u8)?;
                    *slot = Some(MemOperand {
                        addr: imem_addr_for_mode(bus, imem_mode, raw as u8),
                        bits,
                        side_effect: None,
                    });
                    offset += 1;
                }
                OperandKind::EMemAddrWidth(bytes) | OperandKind::EMemAddrWidthOp(bytes) => {
                    let bits = Self::bits_from_bytes(*bytes);
                    let base = if matches!(
                        entry.opcode,
                        0x62
                            | 0x66
                            | 0x6A
                            | 0x72
                            | 0x7A
                            | 0x88..=0x8F
                            | 0xA8..=0xAF
                            | 0xD0..=0xD3
                            | 0xD8..=0xDB
                    ) {
                        Self::fetch_hw009_masked_data_address(bus, pc + offset)
                    } else {
                        Self::fetch_encoded_20bit(bus, pc + offset)?
                    };
                    let slot = if decoded.mem.is_none() {
                        &mut decoded.mem
                    } else {
                        &mut decoded.mem2
                    };
                    *slot = Some(MemOperand {
                        addr: bus.resolve_emem(base),
                        bits,
                        side_effect: None,
                    });
                    offset += 3;
                }
                OperandKind::EMemRegWidth(bytes) | OperandKind::EMemRegWidthMode(bytes) => {
                    if entry.opcode == 0xEB {
                        let raw_mode = (Self::fetch_byte(bus, pc + offset) >> 4) & 0x0F;
                        if !matches!(raw_mode, 0x2 | 0x3) {
                            return Err("EB requires post-increment or pre-decrement addressing");
                        }
                    }
                    let (mem, consumed) =
                        self.decode_ext_reg_ptr(state, bus, pc + offset, *bytes)?;
                    if decoded.mem.is_none() {
                        decoded.mem = Some(mem);
                    } else {
                        decoded.mem2 = Some(mem);
                    }
                    offset += consumed;
                }
                OperandKind::EMemRegModePostPre => {
                    let raw_mode = (Self::fetch_byte(bus, pc + offset) >> 4) & 0x0F;
                    if !matches!(raw_mode, 0x2 | 0x3) {
                        return Err("operand requires post-increment or pre-decrement addressing");
                    }
                    let (mem, consumed) = self.decode_ext_reg_ptr(state, bus, pc + offset, 1)?;
                    if decoded.mem.is_none() {
                        decoded.mem = Some(mem);
                    } else {
                        decoded.mem2 = Some(mem);
                    }
                    offset += consumed;
                }
                OperandKind::EMemIMemWidth(bytes) => {
                    let mode_index = if single_pre || single_pre_operand {
                        0
                    } else {
                        operand_index
                    };
                    let mode = mode_for_operand(pre, mode_index);
                    let (mem, consumed) = self.decode_imem_ptr(bus, pc + offset, *bytes, mode)?;
                    if decoded.mem.is_none() {
                        decoded.mem = Some(mem);
                    } else {
                        decoded.mem2 = Some(mem);
                    }
                    offset += consumed;
                }
                OperandKind::EMemImemOffsetDestIntMem => {
                    let mode_first_index = if single_pre { 0 } else { operand_index };
                    let mode_second_index = if single_pre { 0 } else { operand_index + 1 };
                    let mode_first = mode_for_operand(pre, mode_first_index);
                    let mode_second = mode_for_operand(pre, mode_second_index);
                    let (transfer, consumed) = self.decode_emem_imem_offset(
                        entry,
                        bus,
                        pc + offset,
                        mode_first,
                        mode_second,
                        true,
                    )?;
                    decoded.transfer = Some(transfer);
                    offset += consumed;
                }
                OperandKind::EMemImemOffsetDestExtMem => {
                    let mode_first_index = if single_pre { 0 } else { operand_index };
                    let mode_second_index = if single_pre { 0 } else { operand_index + 1 };
                    let mode_first = mode_for_operand(pre, mode_first_index);
                    let mode_second = mode_for_operand(pre, mode_second_index);
                    let (transfer, consumed) = self.decode_emem_imem_offset(
                        entry,
                        bus,
                        pc + offset,
                        mode_first,
                        mode_second,
                        false,
                    )?;
                    decoded.transfer = Some(transfer);
                    offset += consumed;
                }
                OperandKind::RegIMemOffset(kind) => {
                    let width_bits = Self::width_bits_for_kind(entry.kind);
                    let width_bytes = width_bits.div_ceil(8);
                    let reg_byte = Self::fetch_byte(bus, pc + offset);
                    if reg_byte & 0x07 < 4 {
                        return Err("external-memory pointer requires a three-byte register");
                    }
                    let raw_mode = (reg_byte >> 4) & 0x0F;
                    if matches!(entry.opcode, 0x56 | 0x5E) && !matches!(raw_mode, 0x8 | 0xC) {
                        return Err("RegIMemOffset requires positive or negative offset mode");
                    }
                    let (mode, needs_disp, disp_sign) = match raw_mode {
                        0x0 => (ExtRegMode::Simple, false, 0),
                        0x2 => (ExtRegMode::PostInc, false, 0),
                        0x3 => (ExtRegMode::PreDec, false, 0),
                        0x8 => (ExtRegMode::Offset, true, 1),
                        0xC => (ExtRegMode::Offset, true, -1),
                        _ => return Err("unsupported EMEM reg mode"),
                    };
                    let reg = Self::reg_from_selector(reg_byte).ok_or("invalid reg selector")?;

                    // RegIMemOffset encoding places the IMEM byte before any displacement.
                    let mut consumed_ptr = 1u32;
                    let raw_imem = u32::from(Self::fetch_byte(bus, pc + offset + consumed_ptr));
                    consumed_ptr += 1;
                    let mut disp: i16 = 0;
                    if needs_disp {
                        let magnitude = Self::fetch_byte(bus, pc + offset + consumed_ptr);
                        disp = if disp_sign >= 0 {
                            magnitude as i16
                        } else {
                            -(magnitude as i16)
                        };
                        consumed_ptr += 1;
                    }

                    let base = state.get_reg(reg);
                    let step = width_bytes as u32;
                    let mask = mask_for(reg);
                    let mut addr = base;
                    let mut side_effect: Option<(RegName, u32)> = None;
                    match mode {
                        ExtRegMode::Simple => {}
                        ExtRegMode::Offset => {
                            addr = base.wrapping_add(disp as u32);
                        }
                        ExtRegMode::PreDec => {
                            addr = base.wrapping_sub(step) & mask;
                            side_effect = Some((reg, addr));
                        }
                        ExtRegMode::PostInc => {
                            side_effect = Some((reg, (base.wrapping_add(step)) & mask));
                        }
                    }
                    let ptr_mem = MemOperand {
                        addr: bus.resolve_emem(addr & mask_for(RegName::X)),
                        bits: width_bits,
                        side_effect,
                    };

                    // Parity: RegIMemOffset uses the first PRE mode for the IMEM selector.
                    let mode_index = if single_pre { 0 } else { operand_index };
                    let imem_mode = mode_for_operand(pre, mode_index);
                    validate_imem_selector(imem_mode, raw_imem as u8)?;
                    let imem_addr = imem_addr_for_mode(bus, imem_mode, raw_imem as u8);
                    offset += consumed_ptr;
                    let transfer = match kind {
                        super::opcodes::RegImemOffsetKind::DestImem => EmemImemTransfer {
                            dst_addr: imem_addr,
                            src_addr: ptr_mem.addr,
                            bits: width_bits,
                            dst_is_internal: true,
                            side_effect: ptr_mem.side_effect,
                        },
                        super::opcodes::RegImemOffsetKind::DestRegOffset => EmemImemTransfer {
                            dst_addr: ptr_mem.addr,
                            src_addr: imem_addr,
                            bits: width_bits,
                            dst_is_internal: false,
                            side_effect: ptr_mem.side_effect,
                        },
                    };
                    decoded.transfer = Some(transfer);
                }
                OperandKind::Reg3 => {
                    let selector = Self::fetch_byte(bus, pc + offset);
                    let reg = Self::reg_from_selector(selector)
                        .ok_or("invalid three-byte register selector")?;
                    if entry.opcode == 0x11 && !(0x04..=0x07).contains(&selector) {
                        return Err("JP requires X, Y, U, or S");
                    }
                    if matches!(entry.opcode, 0x6C | 0x7C) && selector > 0x07 {
                        return Err("INC/DEC register selector has reserved upper bits");
                    }
                    if entry.kind == InstrKind::Cmpw && !matches!(selector, 0x02 | 0x03) {
                        return Err("CMPW requires a two-byte register");
                    }
                    if entry.kind == InstrKind::Cmpp && !(0x04..=0x07).contains(&selector) {
                        return Err("CMPP requires a three-byte register");
                    }
                    decoded.reg3 = Some(reg);
                    offset += 1;
                }
                OperandKind::RegPair(size) => {
                    let raw = Self::fetch_byte(bus, pc + offset);
                    if raw & 0x88 != 0 {
                        return Err("invalid register-pair selector");
                    }
                    let use_r2 = matches!(entry.kind, InstrKind::Mv | InstrKind::Ex);
                    let r1_code = (raw >> 4) & 0x7;
                    let r2_code = raw & 0x7;
                    let legal = match entry.opcode {
                        0x44 | 0x4C => (2..=3).contains(&r1_code) && r2_code <= 3,
                        0x45 | 0x4D => (4..=7).contains(&r1_code),
                        0x46 | 0x4E => r1_code <= 1 && r2_code <= 1,
                        _ => true,
                    };
                    if !legal {
                        return Err("invalid register-pair selector for opcode");
                    }
                    let r1 = Self::regpair_name(r1_code, use_r2);
                    let r2 = Self::regpair_name(r2_code, use_r2);
                    let bits = if matches!(entry.kind, InstrKind::Mv | InstrKind::Ex) {
                        Self::regpair_bits(*size, r1, r2)
                    } else {
                        match *size {
                            1 => 8,
                            2 => 16,
                            3 => 20,
                            _ => 8,
                        }
                    };
                    decoded.reg_pair = Some((r1, r2, bits));
                    offset += 1;
                }
                OperandKind::Reg(_, _)
                | OperandKind::RegB
                | OperandKind::RegIL
                | OperandKind::RegIMR
                | OperandKind::RegF => {}
                _ => return Err("unsupported operand kind"),
            }
        }
        decoded.len = offset as u8;
        Ok(decoded)
    }

    fn decode_with_prefix<B: LlamaBus>(
        &mut self,
        entry: &OpcodeEntry,
        state: &mut LlamaState,
        bus: &mut B,
        pre: Option<&PreModes>,
        pc_override: Option<u32>,
        prefix_len: u8,
    ) -> Result<DecodedOperands, &'static str> {
        let mut decoded = self.decode_operands(entry, state, bus, pre, pc_override)?;
        decoded.len = decoded.len.saturating_add(prefix_len);
        Ok(decoded)
    }

    fn operand_reg(op: &OperandKind) -> Option<RegName> {
        match op {
            OperandKind::Reg(name, _) => Some(*name),
            OperandKind::RegB => Some(RegName::B),
            OperandKind::RegIL => Some(RegName::IL),
            OperandKind::RegIMR => Some(RegName::IMR),
            OperandKind::RegF => Some(RegName::F),
            _ => None,
        }
    }

    fn resolved_reg(op: &OperandKind, decoded: &DecodedOperands) -> Option<RegName> {
        match op {
            OperandKind::Reg(_, _)
            | OperandKind::RegB
            | OperandKind::RegIL
            | OperandKind::RegIMR
            | OperandKind::RegF => Self::operand_reg(op),
            OperandKind::Reg3 => decoded.reg3,
            _ => None,
        }
    }

    fn regpair_name(code: u8, use_r2: bool) -> RegName {
        let idx = code & 0x7;
        if use_r2 {
            match idx {
                0 | 2 => RegName::BA,
                1 | 3 => RegName::I,
                4 => RegName::X,
                5 => RegName::Y,
                6 => RegName::U,
                7 => RegName::S,
                _ => RegName::Unknown("regpair"),
            }
        } else {
            match idx {
                0 => RegName::A,
                1 => RegName::IL,
                2 => RegName::BA,
                3 => RegName::I,
                4 => RegName::X,
                5 => RegName::Y,
                6 => RegName::U,
                7 => RegName::S,
                _ => RegName::Unknown("regpair"),
            }
        }
    }

    fn regpair_bits(size: u8, r1: RegName, r2: RegName) -> u8 {
        match size {
            1 => 8,
            2 => {
                if Self::regpair_is_20bit(r1) || Self::regpair_is_20bit(r2) {
                    20
                } else {
                    16
                }
            }
            3 => 20,
            _ => 8,
        }
    }

    fn regpair_is_20bit(reg: RegName) -> bool {
        matches!(reg, RegName::X | RegName::Y | RegName::U | RegName::S)
    }

    fn reg3_bits(reg: RegName) -> u8 {
        match reg {
            RegName::A | RegName::IL => 8,
            RegName::BA | RegName::I => 16,
            RegName::X | RegName::Y | RegName::U | RegName::S => 20,
            _ => 24,
        }
    }

    fn execute_mv_generic<B: LlamaBus>(
        &mut self,
        entry: &OpcodeEntry,
        state: &mut LlamaState,
        bus: &mut B,
        pre: Option<&PreModes>,
        pc_override: Option<u32>,
        prefix_len: u8,
    ) -> Result<u8, &'static str> {
        let prev_fc = state.get_reg(RegName::FC);
        let decoded = self.decode_with_prefix(entry, state, bus, pre, pc_override, prefix_len)?;
        if matches!(entry.kind, InstrKind::Mvl | InstrKind::Mvld) {
            let length = effective_i_count(state);
            let (mem_dst, mem_src) = decoded
                .mem
                .zip(decoded.mem2)
                .ok_or("MVL/MVLD requires two memory operands")?;
            let mut dst_addr = mem_dst.addr;
            let mut src_addr = mem_src.addr;
            let dst_step = mem_dst
                .side_effect
                .map(|(reg, new_val)| {
                    Self::addr_step_from_side_effect(reg, state.get_reg(reg), new_val)
                })
                .unwrap_or_else(|| {
                    let base = mem_dst.bits.div_ceil(8) as i32;
                    if entry.kind == InstrKind::Mvld {
                        -base
                    } else {
                        base
                    }
                });
            let src_step = mem_src
                .side_effect
                .map(|(reg, new_val)| {
                    Self::addr_step_from_side_effect(reg, state.get_reg(reg), new_val)
                })
                .unwrap_or_else(|| {
                    let base = mem_src.bits.div_ceil(8) as i32;
                    if entry.kind == InstrKind::Mvld {
                        -base
                    } else {
                        base
                    }
                });
            for _ in 0..length {
                let val = Self::load_wrapped(bus, src_addr, mem_dst.bits);
                Self::store_traced(bus, dst_addr, mem_dst.bits, val);
                src_addr = Self::advance_internal_addr_signed(src_addr, src_step);
                dst_addr = Self::advance_internal_addr_signed(dst_addr, dst_step);
            }

            for mem in [decoded.mem, decoded.mem2].into_iter().flatten() {
                if let Some((reg, new_val)) = mem.side_effect {
                    Self::apply_pointer_side_effect(state, reg, new_val, length);
                }
            }
            state.set_reg(RegName::I, 0);
            state.set_reg(RegName::FC, prev_fc);
            let start_pc = state.pc();
            if state.pc() == start_pc {
                state.set_pc(start_pc.wrapping_add(decoded.len as u32));
            }
            return Ok(decoded.len);
        }
        // Special-case RegPair-only move (e.g., opcode 0xFD)
        if entry.operands.len() == 1 {
            if let Some((dst, src, bits)) = decoded.reg_pair {
                // Copy the complete selected architectural register. X/Y/U/S
                // occupy three transfer bytes but contain only 20 significant
                // bits; state.set_reg applies that register-specific mask.
                // Do not use the generic operand annotation because opcode FD
                // shares one encoding across 8-, 16-, and three-byte classes.
                let _ = bits;
                let val = state.get_reg(src);
                state.set_reg(dst, val);
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(decoded.len as u32));
                }
                return Ok(decoded.len);
            }
            return Err("unsupported single-operand MV pattern");
        }
        let dst_op = &entry.operands[0];
        let src_op = &entry.operands[1];

        // Helpers to resolve registers from operands
        let dst_reg = Self::resolved_reg(dst_op, &decoded);
        let src_reg = Self::resolved_reg(src_op, &decoded);

        // Source value resolution
        let mut src_val: Option<(u32, u8)> = None;
        if let Some(reg) = src_reg {
            let bits = match src_op {
                OperandKind::Reg(_, bits) => *bits,
                OperandKind::RegB
                | OperandKind::RegIL
                | OperandKind::RegIMR
                | OperandKind::RegF => 8,
                _ => 8,
            };
            src_val = Some((Self::read_reg(state, bus, reg), bits));
        } else if matches!(
            src_op,
            OperandKind::Imm(_)
                | OperandKind::IMem(_)
                | OperandKind::IMemWidth(_)
                | OperandKind::EMemAddrWidth(_)
                | OperandKind::EMemAddrWidthOp(_)
                | OperandKind::EMemRegWidth(_)
                | OperandKind::EMemRegWidthMode(_)
                | OperandKind::EMemIMemWidth(_)
                | OperandKind::EMemRegModePostPre
        ) {
            if let Some((imm, bits)) = decoded.imm {
                src_val = Some((imm, bits));
            } else if let Some(mem) = decoded.mem2.or(decoded.mem) {
                let bits = mem.bits;
                let val = Self::load_wrapped(bus, mem.addr, bits);
                src_val = Some((val, bits));
            }
        }

        // Destination handling
        if let Some(reg) = dst_reg {
            // Register destination
            let (val, bits) = src_val.ok_or("missing source")?;
            let masked = val & Self::mask_for_width(bits);
            state.set_reg(reg, masked);
            if reg == RegName::IMR {
                Self::store_traced(
                    bus,
                    INTERNAL_MEMORY_START + IMEM_IMR_OFFSET,
                    8,
                    masked & 0xFF,
                );
            }
            // Preserve flags for MV-to-reg; zero flag is only updated for MVL handling below.
        } else if matches!(
            dst_op,
            OperandKind::IMem(_)
                | OperandKind::IMemWidth(_)
                | OperandKind::EMemAddrWidth(_)
                | OperandKind::EMemAddrWidthOp(_)
                | OperandKind::EMemRegWidth(_)
                | OperandKind::EMemRegWidthMode(_)
                | OperandKind::EMemIMemWidth(_)
                | OperandKind::EMemRegModePostPre
        ) {
            let mem = if entry.ops_reversed.unwrap_or(false) {
                decoded.mem.or(decoded.mem2)
            } else {
                decoded.mem
            }
            .ok_or("missing mem operand")?;
            let (val, bits) = src_val.ok_or("missing source")?;
            Self::store_traced(bus, mem.addr, bits, val);
            if let Some((reg, new_val)) = mem.side_effect {
                if !matches!(entry.kind, InstrKind::Mvl | InstrKind::Mvld) {
                    state.set_reg(reg, new_val);
                }
            }
        } else {
            // A generated table shape that reaches this branch is not implemented.
            // Do not invent an accumulator destination or a zero source.
            let (val, bits) = src_val.ok_or("missing source for MV pattern")?;
            let masked = val & Self::mask_for_width(bits);
            if let Some(mem) = decoded.mem.or(decoded.mem2) {
                Self::store_traced(bus, mem.addr, mem.bits, masked);
                if let Some((reg, new_val)) = mem.side_effect {
                    if !matches!(entry.kind, InstrKind::Mvl | InstrKind::Mvld) {
                        state.set_reg(reg, new_val);
                    }
                }
            } else {
                return Err("unsupported MV destination pattern");
            }
            let start_pc = state.pc();
            if state.pc() == start_pc {
                state.set_pc(start_pc.wrapping_add(decoded.len as u32));
            }
            return Ok(decoded.len);
        }

        // Apply any pointer side-effects even if the memory operand was a source.
        let pointer_steps = 1;
        for m in [decoded.mem, decoded.mem2].into_iter().flatten() {
            if let Some((reg, new_val)) = m.side_effect {
                if Some(reg) != dst_reg {
                    // Use the same signed-step logic as other multi-byte helpers so pre-dec
                    // addressing walks backward instead of wrapping forward.
                    Self::apply_pointer_side_effect(state, reg, new_val, pointer_steps);
                }
            }
        }
        state.set_reg(RegName::FC, prev_fc);

        let start_pc = state.pc();
        if state.pc() == start_pc {
            state.set_pc(start_pc.wrapping_add(decoded.len as u32));
        }
        Ok(decoded.len)
    }

    fn execute_reg_imm<B: LlamaBus>(
        &mut self,
        entry: &OpcodeEntry,
        state: &mut LlamaState,
        bus: &mut B,
        pre: Option<&PreModes>,
        pc_override: Option<u32>,
        prefix_len: u8,
    ) -> Result<u8, &'static str> {
        let _ = pre;
        let pc = pc_override.unwrap_or(state.pc());
        let mut offset = 1u32; // opcode byte consumed
        let mut imm: Option<u32> = None;
        for op in entry.operands.iter() {
            match op {
                OperandKind::Imm(bits) => {
                    imm = Some(Self::fetch_imm(bus, pc + offset, *bits));
                    offset += (*bits as u32).div_ceil(8);
                }
                OperandKind::Reg(RegName::A, _) => {
                    // nothing to fetch
                }
                // Parity: Python never decodes other reg/imm shapes.
                _ => return Err("unsupported operand pattern"),
            }
        }
        let rhs = imm.ok_or("missing immediate")?;
        let a = state.get_reg(RegName::A);
        let mut carry_flag: Option<bool> = None;
        let bits: u8 = 8;
        let mask = Self::mask_for_width(bits);
        let result = match entry.kind {
            InstrKind::Add => {
                let full = (a & mask) as u64 + (rhs & mask) as u64;
                carry_flag = Some(full > mask as u64);
                (full as u32) & mask
            }
            InstrKind::Sub => {
                let lhs = a & mask;
                let rhs_masked = rhs & mask;
                let borrow = lhs < rhs_masked;
                carry_flag = Some(borrow);
                lhs.wrapping_sub(rhs_masked) & mask
            }
            InstrKind::And => {
                carry_flag = None;
                (a & rhs) & mask
            }
            InstrKind::Or => {
                carry_flag = None;
                (a | rhs) & mask
            }
            InstrKind::Xor => {
                carry_flag = None;
                (a ^ rhs) & mask
            }
            InstrKind::Cmp => {
                Self::set_flags_cmp(state, a & mask, rhs & mask, bits);
                let len = offset as u8 + prefix_len;
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(len as u32));
                }
                return Ok(len);
            }
            InstrKind::Test => {
                let res = (a & rhs) & mask;
                Self::set_flags_for_result(state, res, None);
                let len = offset as u8 + prefix_len;
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(len as u32));
                }
                return Ok(len);
            }
            InstrKind::Mv => rhs & mask,
            InstrKind::Adc => {
                let c = state.get_reg(RegName::FC) & 1;
                let full = (a & mask) as u64 + (rhs & mask) as u64 + (c as u64);
                carry_flag = Some(full > mask as u64);
                (full as u32) & mask
            }
            InstrKind::Sbc => {
                let c = state.get_reg(RegName::FC) & 1;
                let lhs = a & mask;
                let rhs_masked = rhs & mask;
                let borrow = (lhs as u64) < (rhs_masked as u64 + c as u64);
                carry_flag = Some(borrow);
                lhs.wrapping_sub(rhs_masked).wrapping_sub(c) & mask
            }
            // Parity: Python only defines these A+imm arithmetic opcodes.
            _ => return Err("unsupported reg/imm kind"),
        };
        state.set_reg(RegName::A, result);
        if entry.kind != InstrKind::Mv {
            Self::set_flags_for_result(state, result, carry_flag);
            if let Some(c) = carry_flag {
                state.set_reg(RegName::FC, if c { 1 } else { 0 });
            }
        }
        let len = offset as u8 + prefix_len;
        let start_pc = state.pc();
        if state.pc() == start_pc {
            state.set_pc(start_pc.wrapping_add(len as u32));
        }
        Ok(len)
    }

    fn execute_simple_mem<B: LlamaBus>(
        &mut self,
        entry: &OpcodeEntry,
        state: &mut LlamaState,
        bus: &mut B,
        pre: Option<&PreModes>,
        pc_override: Option<u32>,
        prefix_len: u8,
    ) -> Result<u8, &'static str> {
        let start_pc = state.pc();
        let decoded = self.decode_with_prefix(entry, state, bus, pre, pc_override, prefix_len)?;
        let mem = decoded.mem.ok_or("missing mem operand")?;
        let len = decoded.len;
        match entry.kind {
            InstrKind::Mv => {
                match entry.operands {
                    // MV A, [mem]
                    [OperandKind::Reg(RegName::A, _), _] => {
                        let val = bus.load(mem.addr, 8) & 0xFF;
                        state.set_reg(RegName::A, val);
                    }
                    // MV [mem], A
                    [OperandKind::IMem(_), OperandKind::Reg(RegName::A, _)]
                    | [OperandKind::IMemWidth(_), OperandKind::Reg(RegName::A, _)]
                    | [OperandKind::EMemAddrWidth(_), OperandKind::Reg(RegName::A, _)]
                    | [OperandKind::EMemAddrWidthOp(_), OperandKind::Reg(RegName::A, _)]
                    | [OperandKind::EMemRegWidth(_), OperandKind::Reg(RegName::A, _)]
                    | [OperandKind::EMemRegWidthMode(_), OperandKind::Reg(RegName::A, _)] => {
                        let val = state.get_reg(RegName::A) & 0xFF;
                        Self::store_traced(bus, mem.addr, 8, val);
                    }
                    // MV [mem], imm
                    [OperandKind::IMem(_), OperandKind::Imm(bits)]
                    | [OperandKind::IMemWidth(_), OperandKind::Imm(bits)]
                    | [OperandKind::EMemAddrWidth(_), OperandKind::Imm(bits)]
                    | [OperandKind::EMemAddrWidthOp(_), OperandKind::Imm(bits)] => {
                        let (val, _) = decoded.imm.ok_or("missing immediate")?;
                        Self::store_traced(bus, mem.addr, *bits, val);
                    }
                    // Generic fallback: handle Reg<->Mem moves not covered above.
                    _ => {
                        // mem -> reg
                        if matches!(
                            entry.operands.first(),
                            Some(
                                OperandKind::IMem(_)
                                    | OperandKind::IMemWidth(_)
                                    | OperandKind::EMemAddrWidth(_)
                                    | OperandKind::EMemAddrWidthOp(_)
                                    | OperandKind::EMemRegWidth(_)
                                    | OperandKind::EMemRegWidthMode(_)
                            )
                        ) {
                            if let Some(reg) = entry
                                .operands
                                .get(1)
                                .and_then(|op| Self::resolved_reg(op, &decoded))
                            {
                                let val = Self::load_wrapped(bus, mem.addr, mem.bits);
                                let mask = Self::mask_for_width(mem.bits);
                                state.set_reg(reg, val & mask);
                                return Ok(decoded.len);
                            }
                        }
                        // reg -> mem
                        if matches!(
                            entry.operands.get(1),
                            Some(
                                OperandKind::IMem(_)
                                    | OperandKind::IMemWidth(_)
                                    | OperandKind::EMemAddrWidth(_)
                                    | OperandKind::EMemAddrWidthOp(_)
                                    | OperandKind::EMemRegWidth(_)
                                    | OperandKind::EMemRegWidthMode(_)
                            )
                        ) {
                            if let Some(reg) = entry
                                .operands
                                .first()
                                .and_then(|op| Self::resolved_reg(op, &decoded))
                            {
                                let val = Self::read_reg(state, bus, reg)
                                    & Self::mask_for_width(mem.bits);
                                Self::store_traced(bus, mem.addr, mem.bits, val);
                                return Ok(decoded.len);
                            }
                        }
                        return Err("mv pattern not supported");
                    }
                }
            }
            InstrKind::Add
            | InstrKind::Sub
            | InstrKind::And
            | InstrKind::Or
            | InstrKind::Xor
            | InstrKind::Adc
            | InstrKind::Sbc
            | InstrKind::Cmp
            | InstrKind::Test => {
                // Resolve operands based on ordering
                let lhs_is_mem = matches!(
                    entry.operands[0],
                    OperandKind::IMem(_)
                        | OperandKind::IMemWidth(_)
                        | OperandKind::EMemAddrWidth(_)
                        | OperandKind::EMemAddrWidthOp(_)
                        | OperandKind::EMemRegWidth(_)
                        | OperandKind::EMemRegWidthMode(_)
                        | OperandKind::EMemIMemWidth(_)
                );
                let rhs_is_mem = entry
                    .operands
                    .get(1)
                    .map(|op| {
                        matches!(
                            op,
                            OperandKind::IMem(_)
                                | OperandKind::IMemWidth(_)
                                | OperandKind::EMemAddrWidth(_)
                                | OperandKind::EMemAddrWidthOp(_)
                                | OperandKind::EMemRegWidth(_)
                                | OperandKind::EMemRegWidthMode(_)
                                | OperandKind::EMemIMemWidth(_)
                        )
                    })
                    .unwrap_or(false);

                let mask = Self::mask_for_width(mem.bits);
                let lhs_val = if lhs_is_mem {
                    Self::load_wrapped(bus, mem.addr, mem.bits) & mask
                } else if let Some(reg) = entry
                    .operands
                    .first()
                    .and_then(|op| Self::resolved_reg(op, &decoded))
                {
                    Self::read_reg(state, bus, reg) & mask
                } else {
                    return Err("missing left arithmetic operand");
                };
                let rhs_val = if rhs_is_mem {
                    let rhs_mem = decoded
                        .mem2
                        .or(decoded.mem)
                        .ok_or("missing right memory operand")?;
                    Self::load_wrapped(bus, rhs_mem.addr, rhs_mem.bits)
                        & Self::mask_for_width(rhs_mem.bits)
                } else if let Some((imm, _)) = decoded.imm {
                    imm
                } else if let Some(r) = entry
                    .operands
                    .get(1)
                    .and_then(|op| Self::resolved_reg(op, &decoded))
                {
                    Self::read_reg(state, bus, r)
                } else {
                    return Err("missing right arithmetic operand");
                };

                let (result, carry) = match entry.kind {
                    InstrKind::Add => {
                        let full = (lhs_val as u64) + (rhs_val as u64);
                        ((full as u32) & mask, Some(full > mask as u64))
                    }
                    InstrKind::Sub => {
                        let borrow = (lhs_val & mask) < (rhs_val & mask);
                        ((lhs_val.wrapping_sub(rhs_val)) & mask, Some(borrow))
                    }
                    InstrKind::And => ((lhs_val & rhs_val) & mask, None),
                    InstrKind::Or => ((lhs_val | rhs_val) & mask, None),
                    InstrKind::Xor => ((lhs_val ^ rhs_val) & mask, None),
                    InstrKind::Adc => {
                        let c = state.get_reg(RegName::FC) & 1;
                        let full = (lhs_val as u64) + (rhs_val as u64) + (c as u64);
                        ((full as u32) & mask, Some(full > mask as u64))
                    }
                    InstrKind::Sbc => {
                        let c = state.get_reg(RegName::FC) & 1;
                        let borrow = (lhs_val as u64) < (rhs_val as u64 + c as u64);
                        (
                            lhs_val.wrapping_sub(rhs_val).wrapping_sub(c) & mask,
                            Some(borrow),
                        )
                    }
                    InstrKind::Cmp => {
                        Self::set_flags_cmp(state, lhs_val & mask, rhs_val & mask, mem.bits);
                        ((lhs_val.wrapping_sub(rhs_val)) & mask, None)
                    }
                    InstrKind::Test => {
                        let res = (lhs_val & rhs_val) & mask;
                        Self::set_flags_for_result(state, res, None);
                        (res, None)
                    }
                    _ => unreachable!(),
                };
                if !matches!(entry.kind, InstrKind::Cmp | InstrKind::Test) {
                    if lhs_is_mem {
                        Self::store_traced(bus, mem.addr, mem.bits, result);
                    } else {
                        state.set_reg(RegName::A, result);
                    }
                    Self::set_flags_for_result(state, result, carry);
                }
            }
            _ => return Err("unsupported simple-memory instruction kind"),
        }

        if let Some((reg, new_val)) = mem.side_effect {
            state.set_reg(reg, new_val);
        }
        if let Some(mem2) = decoded.mem2 {
            if let Some((reg, new_val)) = mem2.side_effect {
                state.set_reg(reg, new_val);
            }
        }
        if state.pc() == start_pc {
            state.set_pc(start_pc.wrapping_add(len as u32));
        }
        Ok(len)
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_multi_byte_binary<B: LlamaBus>(
        &mut self,
        entry: &OpcodeEntry,
        state: &mut LlamaState,
        bus: &mut B,
        pre: Option<&PreModes>,
        pc_override: Option<u32>,
        prefix_len: u8,
        subtract: bool,
    ) -> Result<u8, &'static str> {
        let decoded = self.decode_with_prefix(entry, state, bus, pre, pc_override, prefix_len)?;
        let mem_dst = decoded.mem.ok_or("missing destination")?;
        let mut dst_addr = mem_dst.addr;
        let mut src_addr = decoded.mem2.map(|m| m.addr);
        let src_reg = entry
            .operands
            .get(1)
            .and_then(|op| Self::resolved_reg(op, &decoded));
        let src_bits = decoded
            .mem2
            .map(|m| m.bits)
            .or_else(|| {
                entry.operands.get(1).and_then(|op| {
                    if let OperandKind::Reg(_, b) = op {
                        Some(*b)
                    } else {
                        None
                    }
                })
            })
            .unwrap_or(mem_dst.bits);
        let mask_dst = Self::mask_for_width(mem_dst.bits);
        let mask_src = Self::mask_for_width(src_bits);
        let initial_i_zero = state.get_reg(RegName::I) & mask_for(RegName::I) == 0;
        let length = effective_i_count(state);
        let dst_step_signed = mem_dst
            .side_effect
            .map(|(reg, new_val)| {
                Self::addr_step_from_side_effect(reg, state.get_reg(reg), new_val)
            })
            .unwrap_or_else(|| mem_dst.bits.div_ceil(8) as i32);
        let src_step_signed = decoded
            .mem2
            .and_then(|m| {
                m.side_effect
                    .map(|(reg, new_val)| {
                        Self::addr_step_from_side_effect(reg, state.get_reg(reg), new_val)
                    })
                    .or_else(|| Some(m.bits.div_ceil(8) as i32))
            })
            .unwrap_or_else(|| mem_dst.bits.div_ceil(8) as i32);
        let mut overall_zero: u32 = 0;
        let mut carry = (state.get_reg(RegName::FC) & 1) != 0;

        for _ in 0..length {
            let lhs = Self::load_wrapped(bus, dst_addr, mem_dst.bits) & mask_dst;
            let rhs = match src_addr {
                Some(addr) => Self::load_wrapped(bus, addr, src_bits) & mask_src,
                None => src_reg
                    .map(|r| Self::read_reg(state, bus, r) & mask_src)
                    .ok_or("missing source")?,
            };
            let (res, new_carry) = if subtract {
                let borrow = (lhs as u64) < (rhs as u64 + carry as u64);
                (
                    lhs.wrapping_sub(rhs).wrapping_sub(carry as u32) & mask_dst,
                    borrow,
                )
            } else {
                let full = (lhs as u64) + (rhs as u64) + (carry as u64);
                (((full as u32) & mask_dst), full > mask_dst as u64)
            };
            Self::store_traced(bus, dst_addr, mem_dst.bits, res);
            overall_zero |= res;
            carry = new_carry;

            if let Some(addr) = src_addr.as_mut() {
                *addr = Self::advance_internal_addr_signed(*addr, src_step_signed);
            }
            dst_addr = Self::advance_internal_addr_signed(dst_addr, dst_step_signed);
        }

        // Apply register side-effects for EMemReg modes (pre-dec/post-inc) across the whole length.
        if length > 0 {
            for m in [decoded.mem, decoded.mem2].into_iter().flatten() {
                if let Some((reg, new_val)) = m.side_effect {
                    Self::apply_pointer_side_effect(state, reg, new_val, length);
                }
            }
        }

        state.set_reg(RegName::I, 0);
        state.set_reg(RegName::FC, if carry { 1 } else { 0 });
        state.set_reg(
            RegName::FZ,
            if !subtract && initial_i_zero {
                // HW-002 measured Z=0 for all-zero ADCL with I=0. SBCL
                // retains the ordinary aggregate-zero result.
                0
            } else if (overall_zero & mask_dst) == 0 {
                1
            } else {
                0
            },
        );
        let start_pc = state.pc();
        if state.pc() == start_pc {
            state.set_pc(start_pc.wrapping_add(decoded.len as u32));
        }
        Ok(decoded.len)
    }

    fn apply_pointer_side_effect(
        state: &mut LlamaState,
        reg: RegName,
        new_val: u32,
        iterations: u32,
    ) {
        if iterations == 0 {
            return;
        }
        let mask = mask_for(reg);
        let curr = state.get_reg(reg) & mask;
        let new_val_masked = new_val & mask;
        if new_val_masked == curr {
            return;
        }
        let step = if new_val_masked > curr {
            new_val_masked.wrapping_sub(curr) & mask
        } else {
            curr.wrapping_sub(new_val_masked) & mask
        };
        if step == 0 {
            return;
        }
        let total = step.wrapping_mul(iterations) & mask;
        let final_val = if new_val_masked > curr {
            curr.wrapping_add(total) & mask
        } else {
            curr.wrapping_sub(total) & mask
        };
        state.set_reg(reg, final_val);
    }

    /// Stub execute entrypoint; wires length estimation and recognizes WAIT/RET/HALT placeholders.
    pub fn execute<B: LlamaBus>(
        &mut self,
        opcode: u8,
        state: &mut LlamaState,
        bus: &mut B,
    ) -> Result<u8, &'static str> {
        self.execute_with_vector_transfer(opcode, state, bus, None)
    }

    /// Execute one instruction, consuming an optional vector-transfer proof
    /// prepared by a machine wrapper before it advanced timers or devices.
    pub fn execute_with_vector_transfer<B: LlamaBus>(
        &mut self,
        opcode: u8,
        state: &mut LlamaState,
        bus: &mut B,
        prepared_transfer: Option<ValidatedVectorTransfer>,
    ) -> Result<u8, &'static str> {
        if let Some(transfer) = prepared_transfer.as_ref() {
            let expected_vector = match self.lookup(opcode).map(|entry| entry.kind) {
                Some(InstrKind::Ir) => INTERRUPT_VECTOR_ADDR,
                Some(InstrKind::Reset) => ROM_RESET_VECTOR_ADDR,
                _ => return Err(PREPARED_VECTOR_MISMATCH_ERROR),
            };
            if !transfer.matches(expected_vector, state, bus) {
                return Err(PREPARED_VECTOR_MISMATCH_ERROR);
            }
        }
        // Complete every rejection-capable decode/data check before reserving
        // a trace index or publishing last-PC/call-stack context.
        self.validate_before_scheduling_with_options(
            opcode,
            state,
            bus,
            true,
            prepared_transfer.is_none(),
        )?;
        let start_pc = state.pc() & mask_for(RegName::PC);
        // For perfetto parity with Python, keep tracing anchored to the prefix byte/PC.
        let trace_pc_snapshot = start_pc;
        let trace_opcode_snapshot = opcode;
        // Execute using the resolved opcode after any PRE bytes.
        let mut exec_pc = start_pc;
        let mut exec_opcode = opcode;
        let mut prefix_len = 0u8;
        let mut pre_modes_opt: Option<PreModes> = None;
        let mut effective_pre_opcode = None;
        let mut pc_override = None;
        let mut entry = self.lookup(exec_opcode);

        while let Some(e) = entry {
            if e.kind != InstrKind::Pre {
                break;
            }
            if prefix_len >= 2 {
                return Err("more than two consecutive PRE prefixes are unverified");
            }
            let pre_modes = pre_modes_for(exec_opcode).ok_or("unknown PRE opcode")?;
            effective_pre_opcode = Some(exec_opcode);
            let next_pc = exec_pc.wrapping_add(1) & mask_for(RegName::PC);
            let next_opcode = Self::fetch_byte(bus, next_pc);
            exec_opcode = next_opcode;
            exec_pc = next_pc;
            prefix_len = prefix_len.saturating_add(1);
            pre_modes_opt = Some(pre_modes);
            pc_override = Some(next_pc);
            entry = self.lookup(next_opcode);
        }

        if let (Some(pre_modes), Some(resolved_entry)) = (pre_modes_opt.as_ref(), entry) {
            validate_canonical_pre(
                effective_pre_opcode.ok_or("missing PRE opcode")?,
                pre_modes,
                resolved_entry,
            )?;
        }

        // Reject reserved and hardware-quarantined behavior before even
        // synchronizing the IMR mirror. No architectural state, memory,
        // pointer, flag, or timing callback may change on these error paths.
        let resolved_entry = entry.ok_or("invalid or reserved opcode")?;
        if resolved_entry.kind == InstrKind::Unknown {
            return reject_unknown();
        }
        if resolved_entry.kind == InstrKind::Tcl && !bus.supports_timer_phase_clear() {
            return Err(TCL_UNIMPLEMENTED_ERROR);
        }
        // IRQ/IR retains a silent-only proof until after the frame writes;
        // RESET retains its historical fail-closed fetch-before-mutation
        // contract.
        let vector_transfer = match (resolved_entry.kind, prepared_transfer) {
            (InstrKind::Ir, Some(transfer)) => {
                if transfer.architectural_fetch_validated() {
                    return Err(PREPARED_VECTOR_MISMATCH_ERROR);
                }
                Some(transfer)
            }
            (InstrKind::Reset, Some(transfer)) => {
                if !transfer.architectural_fetch_validated() {
                    return Err(PREPARED_VECTOR_MISMATCH_ERROR);
                }
                Some(transfer)
            }
            (InstrKind::Ir, None) => {
                Some(prepare_validated_vector(INTERRUPT_VECTOR_ADDR, state, bus)?)
            }
            (InstrKind::Reset, None) => {
                Some(fetch_validated_vector(ROM_RESET_VECTOR_ADDR, state, bus)?)
            }
            (_, Some(_)) => return Err(PREPARED_VECTOR_MISMATCH_ERROR),
            (_, None) => None,
        };

        // Only a fully validated instruction may reserve or publish trace
        // state. In particular, a volatile IR/RESET vector mismatch must not
        // consume an instruction index or replace the last successful PC.
        // The trace clock belongs to an active trace, not to executor object
        // construction or untraced execution. This also prevents unrelated
        // parallel runtimes from perturbing a trace-state atomicity proof.
        let tracing_active = PERFETTO_TRACER.enter().with_some(|_tracer| ()).is_some();
        let instr_index = if tracing_active {
            PERF_INSTR_COUNTER.fetch_add(1, Ordering::Relaxed)
        } else {
            PERF_INSTR_COUNTER.load(Ordering::Relaxed)
        };
        PERF_LAST_PC.with(|value| value.set(trace_pc_snapshot));
        PERF_LAST_CALL_STACK.with(|value| {
            let mut snapshot = PerfettoCallStack::default();
            let frames = state.call_stack();
            let take = PERFETTO_CALL_STACK_MAX_FRAMES.min(frames.len());
            snapshot.len = take as u8;
            for (dst, src) in snapshot.frames.iter_mut().take(take).zip(frames.iter()) {
                *dst = *src & mask_for(RegName::PC);
            }
            value.set(snapshot);
        });
        perfetto_reset_substep();

        // Defer even the IMR mirror update until the vector's architectural
        // fetch agrees with preflight so a volatile mismatch remains atomic.
        if !matches!(resolved_entry.kind, InstrKind::Ir | InstrKind::Reset) {
            let mem_imr = with_imr_read_suppressed(|| bus.peek_imem_silent(IMEM_IMR_OFFSET));
            state.set_reg(RegName::IMR, mem_imr as u32);
        }

        PERF_CURRENT_OP.with(|value| value.set(instr_index));
        PERF_CURRENT_PC.with(|value| value.set(trace_pc_snapshot));
        let _ctx_guard = PerfettoContextGuard;
        let trace_regs = {
            let mut guard = PERFETTO_TRACER.enter();
            guard.with_some(|_| ()).is_some()
        }
        .then(|| {
            let mut regs = HashMap::new();
            for (name, reg) in [
                ("A", RegName::A),
                ("B", RegName::B),
                ("BA", RegName::BA),
                ("IL", RegName::IL),
                ("IH", RegName::IH),
                ("I", RegName::I),
                ("X", RegName::X),
                ("Y", RegName::Y),
                ("U", RegName::U),
                ("S", RegName::S),
                ("PC", RegName::PC),
                ("F", RegName::F),
                ("FC", RegName::FC),
                ("FZ", RegName::FZ),
            ] {
                regs.insert(name.to_string(), state.get_reg(reg) & mask_for(reg));
            }
            regs
        });

        let entry_kind = entry.map(|entry| entry.kind);
        let entry_name = entry.map(|entry| entry.name);
        let stack_s_before = state.get_reg(RegName::S) & mask_for(RegName::S);
        let stack_u_before = state.get_reg(RegName::U) & mask_for(RegName::U);

        let result = match entry {
            Some(entry) => self.execute_with(
                exec_opcode,
                entry,
                state,
                bus,
                pre_modes_opt.as_ref(),
                pc_override,
                prefix_len,
                instr_index,
                vector_transfer,
            ),
            None => reject_unknown(),
        };
        // Parity: record InstructionTrace after executing, but using the pre-execution
        // register snapshot (Python captures regs before execution, IMR/ISR after).
        if let Some(regs) = trace_regs.as_ref() {
            self.trace_instr(
                trace_opcode_snapshot,
                regs,
                bus,
                instr_index,
                trace_pc_snapshot,
            );
        }

        if let Some(kind) = entry_kind {
            if !matches!(
                kind,
                InstrKind::Call | InstrKind::Ret | InstrKind::RetF | InstrKind::RetI
            ) {
                let stack_s_after = state.get_reg(RegName::S) & mask_for(RegName::S);
                let stack_u_after = state.get_reg(RegName::U) & mask_for(RegName::U);
                if stack_s_before != stack_s_after {
                    let mut payload = HashMap::new();
                    payload.insert(
                        "stack_reg".to_string(),
                        AnnotationValue::Str("S".to_string()),
                    );
                    payload.insert(
                        "stack_before".to_string(),
                        AnnotationValue::Pointer(stack_s_before as u64),
                    );
                    payload.insert(
                        "stack_after".to_string(),
                        AnnotationValue::Pointer(stack_s_after as u64),
                    );
                    if let Some(name) = entry_name {
                        payload.insert(
                            "mnemonic".to_string(),
                            AnnotationValue::Str(name.to_string()),
                        );
                    }
                    Self::emit_control_flow_event(
                        "STACK_REG_WRITE",
                        "stack_write",
                        instr_index,
                        start_pc,
                        payload,
                    );
                }
                if stack_u_before != stack_u_after {
                    let mut payload = HashMap::new();
                    payload.insert(
                        "stack_reg".to_string(),
                        AnnotationValue::Str("U".to_string()),
                    );
                    payload.insert(
                        "stack_before".to_string(),
                        AnnotationValue::Pointer(stack_u_before as u64),
                    );
                    payload.insert(
                        "stack_after".to_string(),
                        AnnotationValue::Pointer(stack_u_after as u64),
                    );
                    if let Some(name) = entry_name {
                        payload.insert(
                            "mnemonic".to_string(),
                            AnnotationValue::Str(name.to_string()),
                        );
                    }
                    Self::emit_control_flow_event(
                        "STACK_REG_WRITE",
                        "stack_write",
                        instr_index,
                        start_pc,
                        payload,
                    );
                }
            }
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_with<B: LlamaBus>(
        &mut self,
        _opcode: u8,
        entry: &OpcodeEntry,
        state: &mut LlamaState,
        bus: &mut B,
        pre: Option<&PreModes>,
        pc_override: Option<u32>,
        prefix_len: u8,
        instr_index: u64,
        vector_transfer: Option<ValidatedVectorTransfer>,
    ) -> Result<u8, &'static str> {
        match entry.kind {
            InstrKind::Nop => {
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add((1 + prefix_len) as u32));
                }
                Ok(1 + prefix_len)
            }
            InstrKind::Wait => {
                let wait_cycles = wait_cycle_count(state);
                if !bus.supports_wait_cycles() {
                    return Err("WAIT requires a cycle-capable bus");
                }
                // If the host does not expose wait_cycles, tick timers/keyboard locally to avoid
                // stalling MTI/STI/KEYI.
                bus.wait_cycles(wait_cycles);
                state.set_reg(RegName::I, 0);
                // WAIT does not alter flags on this core.
                let len = 1 + prefix_len;
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(len as u32));
                }
                Ok(len)
            }
            InstrKind::Off => {
                enter_low_power_state(bus, state, PowerState::Off);
                let len = 1 + prefix_len;
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(len as u32));
                }
                Ok(len)
            }
            InstrKind::Halt => {
                enter_low_power_state(bus, state, PowerState::Halted);
                let len = 1 + prefix_len;
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(len as u32));
                }
                Ok(len)
            }
            InstrKind::Mv | InstrKind::Mvw | InstrKind::Mvp | InstrKind::Mvl | InstrKind::Mvld
                if entry.operands.len() == 1
                    && matches!(
                        entry.operands[0],
                        OperandKind::EMemImemOffsetDestIntMem
                            | OperandKind::EMemImemOffsetDestExtMem
                            | OperandKind::RegIMemOffset(_)
                    ) =>
            {
                let decoded =
                    self.decode_with_prefix(entry, state, bus, pre, pc_override, prefix_len)?;
                let transfer = decoded.transfer.ok_or("missing transfer operand")?;
                if entry.kind == InstrKind::Mvl {
                    let length = effective_i_count(state);
                    let step = transfer.bits.div_ceil(8) as i32;
                    let mut src_addr = transfer.src_addr;
                    let mut dst_addr = transfer.dst_addr;
                    for _ in 0..length {
                        // Snapshot the source before storing so an overlapping
                        // transfer cannot turn into a lazy read of the new byte.
                        let value = Self::load_wrapped(bus, src_addr, transfer.bits);
                        Self::store_traced(bus, dst_addr, transfer.bits, value);
                        src_addr = Self::advance_internal_addr_signed(src_addr, step);
                        dst_addr = Self::advance_internal_addr_signed(dst_addr, step);
                    }
                    if let Some((reg, new_val)) = transfer.side_effect {
                        Self::apply_pointer_side_effect(state, reg, new_val, length);
                    }
                    state.set_reg(RegName::I, 0);
                } else {
                    let value = Self::load_wrapped(bus, transfer.src_addr, transfer.bits);
                    Self::store_traced(bus, transfer.dst_addr, transfer.bits, value);
                    if let Some((reg, new_val)) = transfer.side_effect {
                        state.set_reg(reg, new_val);
                    }
                }
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(decoded.len as u32));
                }
                Ok(decoded.len)
            }
            InstrKind::Inc => {
                let decoded =
                    self.decode_with_prefix(entry, state, bus, pre, pc_override, prefix_len)?;
                let op = entry.operands.first().ok_or("missing operand")?;
                if let Some(reg) = Self::resolved_reg(op, &decoded) {
                    let bits = match op {
                        OperandKind::Reg(_, b) => *b,
                        OperandKind::Reg3 => Self::reg3_bits(reg),
                        _ => 8,
                    };
                    Self::alu_unary(state, reg, bits, |v, _| v.wrapping_add(1));
                } else if let Some(mem) = decoded.mem {
                    let val = Self::load_wrapped(bus, mem.addr, mem.bits);
                    let res = (val.wrapping_add(1)) & Self::mask_for_width(mem.bits);
                    Self::store_traced(bus, mem.addr, mem.bits, res);
                    Self::set_flags_for_result(state, res, None);
                } else {
                    return Err("missing operand");
                }
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(decoded.len as u32));
                }
                Ok(decoded.len)
            }
            InstrKind::Dec => {
                let decoded =
                    self.decode_with_prefix(entry, state, bus, pre, pc_override, prefix_len)?;
                let op = entry.operands.first().ok_or("missing operand")?;
                if let Some(reg) = Self::resolved_reg(op, &decoded) {
                    let bits = match op {
                        OperandKind::Reg(_, b) => *b,
                        OperandKind::Reg3 => Self::reg3_bits(reg),
                        _ => 8,
                    };
                    Self::alu_unary(state, reg, bits, |v, _| v.wrapping_sub(1));
                } else if let Some(mem) = decoded.mem {
                    let val = Self::load_wrapped(bus, mem.addr, mem.bits);
                    let res = (val.wrapping_sub(1)) & Self::mask_for_width(mem.bits);
                    Self::store_traced(bus, mem.addr, mem.bits, res);
                    Self::set_flags_for_result(state, res, None);
                } else {
                    return Err("missing operand");
                }
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(decoded.len as u32));
                }
                Ok(decoded.len)
            }
            InstrKind::Pmdf => {
                let decoded =
                    self.decode_with_prefix(entry, state, bus, pre, pc_override, prefix_len)?;
                let mem = decoded.mem.ok_or("missing mem operand")?;
                let src_val = match entry.operands.get(1) {
                    Some(OperandKind::Imm(_)) => decoded.imm.ok_or("missing immediate")?.0,
                    Some(OperandKind::Reg(RegName::A, _)) => state.get_reg(RegName::A) & 0xFF,
                    // Parity: Python only encodes PMDF with Imm8 or A sources.
                    _ => return Err("unsupported PMDF operands"),
                } & 0xFF;
                let dst = bus.load(mem.addr, 8) & 0xFF;
                let res = (dst + src_val) & 0xFF;
                Self::store_traced(bus, mem.addr, 8, res);
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(decoded.len as u32));
                }
                Ok(decoded.len)
            }
            InstrKind::Dadl | InstrKind::Dsbl => {
                let decoded =
                    self.decode_with_prefix(entry, state, bus, pre, pc_override, prefix_len)?;
                let mem_dst = decoded.mem.ok_or("missing destination")?;
                if mem_dst.bits != 8 {
                    // Parity: Python only supports byte-wide DADL/DSBL encodings.
                    return Err("unsupported width for DADL/DSBL");
                }
                let mut dst_addr = mem_dst.addr;
                let mut src_addr = decoded.mem2.map(|m| m.addr);
                let src_bits = decoded.mem2.map(|m| m.bits);
                let mut src_reg_byte = if src_bits.is_none() {
                    entry
                        .operands
                        .get(1)
                        .and_then(|op| Self::resolved_reg(op, &decoded))
                        .map(|reg| (Self::read_reg(state, bus, reg) & 0xFF) as u8)
                } else {
                    None
                };
                let initial_i_zero = state.get_reg(RegName::I) & mask_for(RegName::I) == 0;
                let length = effective_i_count(state);
                let mut carry = match entry.kind {
                    InstrKind::Dadl => {
                        state.set_reg(RegName::FC, 0);
                        false
                    }
                    InstrKind::Dsbl => (state.get_reg(RegName::FC) & 1) != 0,
                    _ => false,
                };
                let dst_step = mem_dst.bits.div_ceil(8) as u32;
                let src_step = src_bits.map_or(0, |b| b.div_ceil(8) as u32);
                let mut overall_zero: u32 = 0;
                let mut executed = false;
                for _ in 0..length {
                    let dst_byte = (Self::load_wrapped(bus, dst_addr, mem_dst.bits) & 0xFF) as u8;
                    let src_byte = if let Some(bits) = src_bits {
                        let addr = src_addr.ok_or("missing source")?;
                        (Self::load_wrapped(bus, addr, bits) & 0xFF) as u8
                    } else if let Some(byte) = src_reg_byte {
                        // DADL/DSBL with register source consumes the register byte once, then
                        // uses zero for subsequent iterations.
                        src_reg_byte = Some(0);
                        byte
                    } else {
                        return Err("missing source");
                    };
                    let (res, new_carry) = if entry.kind == InstrKind::Dadl {
                        Self::bcd_add_byte(dst_byte, src_byte, carry)
                    } else {
                        Self::bcd_sub_byte(dst_byte, src_byte, carry)
                    };
                    Self::store_traced(bus, dst_addr, mem_dst.bits, res as u32);
                    carry = new_carry;
                    overall_zero |= res as u32;
                    if let Some(addr) = src_addr.as_mut() {
                        *addr = Self::advance_internal_addr_signed(*addr, -(src_step as i32));
                    }
                    dst_addr = Self::advance_internal_addr_signed(dst_addr, -(dst_step as i32));
                    executed = true;
                }
                state.set_reg(RegName::I, 0);
                let zero_mask = Self::mask_for_width(mem_dst.bits);
                state.set_reg(
                    RegName::FZ,
                    if entry.kind == InstrKind::Dadl && initial_i_zero {
                        // HW-002 measured Z=0 for all-zero DADL with I=0;
                        // DSBL retains the ordinary aggregate-zero result.
                        0
                    } else if (overall_zero & zero_mask) == 0 {
                        1
                    } else {
                        0
                    },
                );
                if executed || entry.kind == InstrKind::Dadl {
                    state.set_reg(RegName::FC, if carry { 1 } else { 0 });
                }
                if length > 0 {
                    for m in [decoded.mem, decoded.mem2].into_iter().flatten() {
                        if let Some((reg, new_val)) = m.side_effect {
                            Self::apply_pointer_side_effect(state, reg, new_val, length);
                        }
                    }
                }
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(decoded.len as u32));
                }
                Ok(decoded.len)
            }
            InstrKind::Shl | InstrKind::Shr | InstrKind::Rol | InstrKind::Ror => {
                let decoded =
                    self.decode_with_prefix(entry, state, bus, pre, pc_override, prefix_len)?;
                let op = entry.operands.first().ok_or("missing operand")?;
                let (val, bits, dest_mem, dest_reg) =
                    if let Some(reg) = Self::resolved_reg(op, &decoded) {
                        let bits = match op {
                            OperandKind::Reg(_, b) => *b,
                            OperandKind::Reg3 => Self::reg3_bits(reg),
                            _ => 8,
                        };
                        (state.get_reg(reg), bits, None, Some(reg))
                    } else if let Some(mem) = decoded.mem {
                        (
                            Self::load_wrapped(bus, mem.addr, mem.bits),
                            mem.bits,
                            Some(mem),
                            None,
                        )
                    } else {
                        return Err("missing operand");
                    };
                let mask = Self::mask_for_width(bits);
                let carry_in = state.get_reg(RegName::FC) & 1;
                let (res, carry_out) = match entry.kind {
                    // SHL/SHR are rotate-through-carry; ROL/ROR ignore incoming carry.
                    InstrKind::Shl => (
                        ((val << 1) | carry_in) & mask,
                        ((val >> (bits.saturating_sub(1) as u32)) & 1) != 0,
                    ),
                    InstrKind::Shr => (
                        ((val >> 1) | (carry_in << (bits.saturating_sub(1) as u32))) & mask,
                        (val & 1) != 0,
                    ),
                    InstrKind::Rol => (
                        ((val << 1) | (val >> (bits as u32 - 1))) & mask,
                        ((val >> (bits.saturating_sub(1) as u32)) & 1) != 0,
                    ),
                    InstrKind::Ror => (
                        ((val >> 1) | ((val & 1) << (bits as u32 - 1))) & mask,
                        (val & 1) != 0,
                    ),
                    _ => (val, false),
                };
                if let Some(reg) = dest_reg {
                    state.set_reg(reg, res & mask);
                } else if let Some(mem) = dest_mem {
                    Self::store_traced(bus, mem.addr, bits, res & mask);
                }
                let carry_flag = match entry.kind {
                    InstrKind::Shl | InstrKind::Shr => carry_out,
                    InstrKind::Rol => ((val >> (bits.saturating_sub(1) as u32)) & 1) != 0,
                    InstrKind::Ror => (val & 1) != 0,
                    _ => false,
                };
                Self::set_flags_for_result(state, res & mask, Some(carry_flag));
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(decoded.len as u32));
                }
                Ok(decoded.len)
            }
            InstrKind::Dsll | InstrKind::Dsrl => {
                let decoded =
                    self.decode_with_prefix(entry, state, bus, pre, pc_override, prefix_len)?;
                let mem = decoded.mem.ok_or("missing mem operand")?;
                if mem.bits != 8 {
                    return Err("DSLL/DSRL only support byte operands");
                }
                let initial_i_zero = state.get_reg(RegName::I) & mask_for(RegName::I) == 0;
                let length = effective_i_count(state);
                let mut addr = mem.addr;
                let is_left = entry.kind == InstrKind::Dsll;
                let mut carry_nibble: u8 = 0;
                let mut overall_zero: u8 = 0;
                for _ in 0..length {
                    let val = bus.load(addr, 8) as u8;
                    let low = val & 0x0F;
                    let high = (val >> 4) & 0x0F;
                    let new_val = if is_left {
                        let res = (low << 4) | carry_nibble;
                        carry_nibble = high;
                        res
                    } else {
                        let res = high | (carry_nibble << 4);
                        carry_nibble = low;
                        res
                    };
                    Self::store_traced(bus, addr, 8, new_val as u32);
                    overall_zero |= new_val;
                    if Self::is_internal_addr(addr) {
                        let offset = addr.wrapping_sub(INTERNAL_MEMORY_START);
                        let next = if is_left {
                            offset.wrapping_sub(1) & 0xFF
                        } else {
                            offset.wrapping_add(1) & 0xFF
                        };
                        addr = INTERNAL_MEMORY_START + next;
                    } else if is_left {
                        addr = addr.wrapping_sub(1);
                    } else {
                        addr = addr.wrapping_add(1);
                    }
                }
                state.set_reg(RegName::I, 0);
                state.set_reg(
                    RegName::FZ,
                    if initial_i_zero {
                        // HW-002 measured Z=0 for an all-zero full-ring shift.
                        0
                    } else if overall_zero == 0 {
                        1
                    } else {
                        0
                    },
                );
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(decoded.len as u32));
                }
                Ok(decoded.len)
            }
            InstrKind::Adc if entry.name == "ADCL" => self.execute_multi_byte_binary(
                entry,
                state,
                bus,
                pre,
                pc_override,
                prefix_len,
                false,
            ),
            InstrKind::Sbcl => self.execute_multi_byte_binary(
                entry,
                state,
                bus,
                pre,
                pc_override,
                prefix_len,
                true,
            ),
            InstrKind::Add
            | InstrKind::Sub
            | InstrKind::And
            | InstrKind::Or
            | InstrKind::Xor
            | InstrKind::Adc
            | InstrKind::Sbc => {
                if entry.operands.len() == 1 && matches!(entry.operands[0], OperandKind::RegPair(_))
                {
                    let decoded =
                        self.decode_with_prefix(entry, state, bus, pre, pc_override, prefix_len)?;
                    let (r1, r2, bits) = decoded.reg_pair.ok_or("missing operand")?;
                    let mask = Self::mask_for_width(bits);
                    let lhs = state.get_reg(r1) & mask;
                    let rhs = state.get_reg(r2) & mask;
                    let (res, carry) = match entry.kind {
                        InstrKind::Add => {
                            let full = lhs as u64 + rhs as u64;
                            (((full as u32) & mask), Some(full > mask as u64))
                        }
                        InstrKind::Sub => {
                            let borrow = lhs < rhs;
                            (lhs.wrapping_sub(rhs) & mask, Some(borrow))
                        }
                        // Parity: only ADD/SUB reg-reg forms are valid in Python decode.
                        _ => return Err("unsupported operand pattern"),
                    };
                    state.set_reg(r1, res);
                    Self::set_flags_for_result(state, res, carry);
                    let start_pc = state.pc();
                    if state.pc() == start_pc {
                        state.set_pc(start_pc.wrapping_add(decoded.len as u32));
                    }
                    return Ok(decoded.len);
                }
                if entry.operands.iter().any(|op| {
                    matches!(
                        op,
                        OperandKind::IMem(_)
                            | OperandKind::IMemWidth(_)
                            | OperandKind::EMemAddrWidth(_)
                            | OperandKind::EMemAddrWidthOp(_)
                            | OperandKind::EMemRegWidth(_)
                            | OperandKind::EMemRegWidthMode(_)
                            | OperandKind::EMemIMemWidth(_)
                    )
                }) {
                    self.execute_simple_mem(entry, state, bus, pre, pc_override, prefix_len)
                } else {
                    self.execute_reg_imm(entry, state, bus, pre, pc_override, prefix_len)
                }
            }
            InstrKind::Mv | InstrKind::Mvw | InstrKind::Mvp | InstrKind::Mvl | InstrKind::Mvld => {
                let saved_fc = state.get_reg(RegName::FC);
                let len =
                    self.execute_mv_generic(entry, state, bus, pre, pc_override, prefix_len)?;
                state.set_reg(RegName::FC, saved_fc);
                Ok(len)
            }
            InstrKind::Reset => {
                let reset_vector = vector_transfer
                    .ok_or("RESET vector was not prefetched")?
                    .consume_after_architectural_fetch(ROM_RESET_VECTOR_ADDR, state, bus)?;
                apply_power_on_reset(bus, state, reset_vector);
                let mem_imr = with_imr_read_suppressed(|| bus.peek_imem_silent(IMEM_IMR_OFFSET));
                state.set_reg(RegName::IMR, mem_imr as u32);
                Ok(1 + prefix_len)
            }
            InstrKind::Pre => unreachable!("PRE should be handled before dispatch"),
            InstrKind::Unknown => reject_unknown(),
            InstrKind::Sc => {
                state.set_reg(RegName::FC, 1);
                let len = prefix_len + Self::estimated_length(entry);
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(len as u32));
                }
                Ok(len)
            }
            InstrKind::Rc => {
                state.set_reg(RegName::FC, 0);
                let len = prefix_len + Self::estimated_length(entry);
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(len as u32));
                }
                Ok(len)
            }
            InstrKind::Ir => {
                let instr_len = prefix_len + Self::estimated_length(entry);
                let pc_before = state.pc() & mask_for(RegName::PC);
                // IR is synchronous: the ROM dispatcher inspects the opcode at the
                // saved PC and advances that frame itself before RETI. Saving the
                // fallthrough here would make the software-interrupt slot unreachable.
                let saved_pc = pc_before;
                let fallthrough = pc_before.wrapping_add(instr_len as u32) & mask_for(RegName::PC);
                Self::push_stack(state, bus, RegName::S, saved_pc, 24, false);
                let f = state.get_reg(RegName::F) & 0xFF;
                Self::push_stack(state, bus, RegName::S, f, 8, false);
                let imr_addr = INTERNAL_MEMORY_START + IMEM_IMR_OFFSET;
                let imr = bus.load(imr_addr, 8) & 0xFF;
                Self::push_stack(state, bus, RegName::S, imr, 8, false);
                // Clear IRM bit in IMR (bit 7)
                let cleared_imr = imr & 0x7F;
                Self::store_traced(bus, imr_addr, 8, cleared_imr);
                state.set_reg(RegName::IMR, cleared_imr);
                let vec = vector_transfer
                    .ok_or("IR vector was not prepared")?
                    .consume_after_architectural_fetch(INTERRUPT_VECTOR_ADDR, state, bus)?;
                state.call_depth_inc();
                state.set_pc(vec);
                // Parity: emit perfetto IRQ entry like Python IR intrinsic.
                let mut guard = crate::PERFETTO_TRACER.enter();
                guard.with_some(|tracer| {
                    let mut payload = std::collections::HashMap::new();
                    payload.insert("pc".to_string(), AnnotationValue::Pointer(saved_pc as u64));
                    payload.insert("vector".to_string(), AnnotationValue::Pointer(vec as u64));
                    payload.insert("imr_before".to_string(), AnnotationValue::UInt(imr as u64));
                    payload.insert(
                        "imr_after".to_string(),
                        AnnotationValue::UInt(cleared_imr as u64),
                    );
                    payload.insert("src".to_string(), AnnotationValue::Str("IR".to_string()));
                    tracer.record_irq_event("IRQ_Enter", payload);
                });
                let vector = vec & mask_for(RegName::PC);
                let mut cf_payload = HashMap::new();
                cf_payload.insert(
                    "pc_next".to_string(),
                    AnnotationValue::Pointer(vector as u64),
                );
                cf_payload.insert(
                    "pc_target".to_string(),
                    AnnotationValue::Pointer(vector as u64),
                );
                cf_payload.insert(
                    "pc_fallthrough".to_string(),
                    AnnotationValue::Pointer(fallthrough as u64),
                );
                cf_payload.insert(
                    "ret_addr".to_string(),
                    AnnotationValue::Pointer(saved_pc as u64),
                );
                cf_payload.insert(
                    "instr_len".to_string(),
                    AnnotationValue::UInt(instr_len as u64),
                );
                Self::emit_control_flow_event(
                    entry.name,
                    "irq",
                    instr_index,
                    pc_before,
                    cf_payload,
                );
                Ok(instr_len)
            }
            InstrKind::Tcl => {
                if !bus.supports_timer_phase_clear() {
                    return Err(TCL_UNIMPLEMENTED_ERROR);
                }
                let lcc = bus.load(INTERNAL_MEMORY_START + IMEM_LCC_OFFSET, 8) & 0xFF;
                bus.clear_timer_phases(lcc & 0x02 != 0, lcc & 0x01 != 0);
                let len = prefix_len + Self::estimated_length(entry);
                state.set_pc(state.pc().wrapping_add(u32::from(len)));
                Ok(len)
            }
            InstrKind::JpAbs => {
                // Absolute jump; operand may be 16-bit (low bits) or 20-bit.
                let pc_mask = mask_for(RegName::PC);
                let pc_before = state.pc() & pc_mask;
                let cond_ok = Self::cond_pass(entry, state)?;
                let decoded =
                    self.decode_with_prefix(entry, state, bus, pre, pc_override, prefix_len)?;
                let fallthrough = pc_before.wrapping_add(decoded.len as u32) & pc_mask;
                let mut target = None;
                let mut target_addr = None;
                let mut target_reg = None;
                let target_src = if let Some((val, bits)) = decoded.imm {
                    let instr_pc = pc_override.unwrap_or(pc_before) & pc_mask;
                    let dest = if bits == 16 {
                        // JP 16-bit keeps current page (mask to PC width)
                        // Use the address of the JP instruction (Python uses addr & 0xFF0000).
                        (instr_pc & 0xFF0000) | (val & 0xFFFF)
                    } else {
                        val & pc_mask
                    };
                    target = Some(dest);
                    "imm"
                } else if let Some(mem) = decoded.mem {
                    target_addr = Some(mem.addr);
                    if cond_ok {
                        target = Some(Self::load_wrapped(bus, mem.addr, mem.bits) & pc_mask);
                    }
                    "mem"
                } else if let Some(r) = decoded.reg3 {
                    target = Some(state.get_reg(r) & pc_mask);
                    target_reg = Some(Self::reg_name_for_trace(r));
                    "reg"
                } else {
                    return Err("missing jump target");
                };

                if cond_ok && target.is_none() {
                    return Err("missing jump target");
                }

                let dest = if cond_ok {
                    target.unwrap_or(fallthrough)
                } else {
                    fallthrough
                };
                state.set_pc(dest);

                let mut payload = HashMap::new();
                payload.insert("pc_next".to_string(), AnnotationValue::Pointer(dest as u64));
                payload.insert(
                    "pc_fallthrough".to_string(),
                    AnnotationValue::Pointer(fallthrough as u64),
                );
                payload.insert(
                    "instr_len".to_string(),
                    AnnotationValue::UInt(decoded.len as u64),
                );
                if let Some(cond) = entry.cond {
                    payload.insert(
                        "cf_cond".to_string(),
                        AnnotationValue::Str(cond.to_string()),
                    );
                    payload.insert(
                        "cf_taken".to_string(),
                        AnnotationValue::UInt(cond_ok as u64),
                    );
                }
                if let Some(value) = target {
                    payload.insert(
                        "pc_target".to_string(),
                        AnnotationValue::Pointer(value as u64),
                    );
                }
                payload.insert(
                    "pc_target_src".to_string(),
                    AnnotationValue::Str(target_src.to_string()),
                );
                if let Some(addr) = target_addr {
                    payload.insert(
                        "pc_target_addr".to_string(),
                        AnnotationValue::Pointer(addr as u64),
                    );
                }
                if let Some(reg) = target_reg {
                    payload.insert(
                        "pc_target_reg".to_string(),
                        AnnotationValue::Str(reg.to_string()),
                    );
                }
                let kind = if entry.cond.is_some() {
                    "cond_branch"
                } else {
                    "jump"
                };
                Self::emit_control_flow_event(entry.name, kind, instr_index, pc_before, payload);
                Ok(decoded.len)
            }
            InstrKind::JpRel => {
                let pc_mask = mask_for(RegName::PC);
                let pc_before = state.pc() & pc_mask;
                let cond_ok = Self::cond_pass(entry, state)?;
                let decoded =
                    self.decode_with_prefix(entry, state, bus, pre, pc_override, prefix_len)?;
                let imm_raw = decoded.imm.ok_or("missing relative")?.0 as u8;
                let imm = if matches!(entry.opcode, 0x13 | 0x19 | 0x1B | 0x1D | 0x1F) {
                    -(imm_raw as i32)
                } else if matches!(entry.opcode, 0x12 | 0x18 | 0x1A | 0x1C | 0x1E) {
                    imm_raw as i32
                } else {
                    (imm_raw as i8) as i32
                };
                let fallthrough = pc_before.wrapping_add(decoded.len as u32) & pc_mask;
                let target = fallthrough.wrapping_add_signed(imm) & pc_mask;
                let dest = if cond_ok { target } else { fallthrough };
                state.set_pc(dest);

                let mut payload = HashMap::new();
                payload.insert("pc_next".to_string(), AnnotationValue::Pointer(dest as u64));
                payload.insert(
                    "pc_fallthrough".to_string(),
                    AnnotationValue::Pointer(fallthrough as u64),
                );
                payload.insert(
                    "pc_target".to_string(),
                    AnnotationValue::Pointer(target as u64),
                );
                payload.insert(
                    "instr_len".to_string(),
                    AnnotationValue::UInt(decoded.len as u64),
                );
                if let Some(cond) = entry.cond {
                    payload.insert(
                        "cf_cond".to_string(),
                        AnnotationValue::Str(cond.to_string()),
                    );
                    payload.insert(
                        "cf_taken".to_string(),
                        AnnotationValue::UInt(cond_ok as u64),
                    );
                }
                let kind = if entry.cond.is_some() {
                    "cond_branch"
                } else {
                    "jump"
                };
                Self::emit_control_flow_event(entry.name, kind, instr_index, pc_before, payload);
                Ok(decoded.len)
            }
            InstrKind::Call => {
                let decoded =
                    self.decode_with_prefix(entry, state, bus, pre, pc_override, prefix_len)?;
                let (target, bits) = decoded.imm.ok_or("missing jump target")?;
                let pc_before = state.pc();
                let pc_mask = mask_for(RegName::PC);
                let ret_addr = pc_before.wrapping_add(decoded.len as u32) & pc_mask;
                let mut dest = target;
                let push_bits = if bits == 16 {
                    // Push 16-bit return; retain high page from current PC (Python parity).
                    let high = pc_before & 0xFF0000;
                    dest = high | (target & 0xFFFF);
                    16
                } else {
                    24
                };
                // Use S stack for CALL (matches PUSHU/POPU? here sticking with S per CPU specs)
                Self::push_stack(state, bus, RegName::S, ret_addr, push_bits, false);
                if push_bits == 16 {
                    // Track call page so RET can restore the original page even if PC page changes.
                    state.push_call_page(pc_before);
                }
                state.push_call_frame(dest & 0xFFFFF, push_bits);
                state.set_pc(dest & 0xFFFFF);
                state.call_depth_inc();
                let mut guard = crate::PERFETTO_TRACER.enter();
                guard.with_some(|tracer| {
                    tracer.record_call_flow(
                        "CALL",
                        pc_before & mask_for(RegName::PC),
                        dest & 0xFFFFF,
                        state.call_depth(),
                    );
                });
                let mut payload = HashMap::new();
                payload.insert(
                    "pc_next".to_string(),
                    AnnotationValue::Pointer((dest & pc_mask) as u64),
                );
                payload.insert(
                    "pc_target".to_string(),
                    AnnotationValue::Pointer((dest & pc_mask) as u64),
                );
                payload.insert(
                    "pc_fallthrough".to_string(),
                    AnnotationValue::Pointer((ret_addr & pc_mask) as u64),
                );
                payload.insert(
                    "ret_addr".to_string(),
                    AnnotationValue::Pointer((ret_addr & pc_mask) as u64),
                );
                payload.insert(
                    "call_target".to_string(),
                    AnnotationValue::Pointer((dest & pc_mask) as u64),
                );
                payload.insert(
                    "instr_len".to_string(),
                    AnnotationValue::UInt(decoded.len as u64),
                );
                payload.insert(
                    "call_depth".to_string(),
                    AnnotationValue::Int(state.call_depth() as i64),
                );
                Self::emit_control_flow_event(
                    entry.name,
                    "call",
                    instr_index,
                    pc_before & pc_mask,
                    payload,
                );
                Ok(decoded.len)
            }
            InstrKind::Ret => {
                let pc_before = state.pc();
                let pc_mask = mask_for(RegName::PC);
                let ret = Self::pop_stack(state, bus, RegName::S, 16, false);
                let current_page = state.pc() & 0xFF0000;
                // Parity: Python RET combines the low 16-bit return with the *current* page, even
                // if CALL pushed a different page. Pop the saved page for bookkeeping but prefer
                // the current execution page for the return address.
                let _ = state.pop_call_page();
                let page = current_page;
                let dest = (page | (ret & 0xFFFF)) & 0xFFFFF;
                state.set_pc(dest);
                state.call_depth_dec();
                let _ = state.pop_call_stack();
                let mut guard = crate::PERFETTO_TRACER.enter();
                guard.with_some(|tracer| {
                    tracer.record_call_flow(
                        "RET",
                        pc_before & mask_for(RegName::PC),
                        dest,
                        state.call_depth(),
                    );
                });
                let mut payload = HashMap::new();
                payload.insert(
                    "pc_next".to_string(),
                    AnnotationValue::Pointer((dest & pc_mask) as u64),
                );
                payload.insert(
                    "ret_target".to_string(),
                    AnnotationValue::Pointer((dest & pc_mask) as u64),
                );
                payload.insert("instr_len".to_string(), AnnotationValue::UInt(1));
                payload.insert(
                    "call_depth".to_string(),
                    AnnotationValue::Int(state.call_depth() as i64),
                );
                Self::emit_control_flow_event(
                    entry.name,
                    "ret",
                    instr_index,
                    pc_before & pc_mask,
                    payload,
                );
                Ok(1)
            }
            InstrKind::RetF => {
                let pc_before = state.pc();
                let pc_mask = mask_for(RegName::PC);
                let ret = Self::pop_stack(state, bus, RegName::S, 24, false);
                let dest = ret & 0xFFFFF;
                state.set_pc(dest);
                state.call_depth_dec();
                let _ = state.pop_call_stack();
                let mut guard = crate::PERFETTO_TRACER.enter();
                guard.with_some(|tracer| {
                    tracer.record_call_flow(
                        "RETF",
                        pc_before & mask_for(RegName::PC),
                        dest,
                        state.call_depth(),
                    );
                });
                let mut payload = HashMap::new();
                payload.insert(
                    "pc_next".to_string(),
                    AnnotationValue::Pointer((dest & pc_mask) as u64),
                );
                payload.insert(
                    "ret_target".to_string(),
                    AnnotationValue::Pointer((dest & pc_mask) as u64),
                );
                payload.insert("instr_len".to_string(), AnnotationValue::UInt(1));
                payload.insert(
                    "call_depth".to_string(),
                    AnnotationValue::Int(state.call_depth() as i64),
                );
                Self::emit_control_flow_event(
                    entry.name,
                    "ret",
                    instr_index,
                    pc_before & pc_mask,
                    payload,
                );
                Ok(1)
            }
            InstrKind::RetI => {
                // Stack layout: IMR (1), F(1), three-byte 20-bit PC (little-endian).
                let pc_mask = mask_for(RegName::PC);
                let pc_before = state.pc() & pc_mask;
                let mask_s = mask_for(RegName::S);
                let mut sp = state.get_reg(RegName::S) & mask_s;
                let _sp_before = sp;
                let imr = bus.load(sp, 8) & 0xFF;
                sp = sp.wrapping_add(1) & mask_s;
                // Silicon accepts all eight stacked bits but only restores the
                // architecturally modeled carry/zero pair.
                let f = bus.load(sp, 8) & 0x03;
                sp = sp.wrapping_add(1) & mask_s;
                let pc_lo = bus.load(sp, 8) & 0xFF;
                let pc_mid = bus.load(sp.wrapping_add(1) & mask_s, 8) & 0xFF;
                let pc_hi = bus.load(sp.wrapping_add(2) & mask_s, 8) & 0xFF;
                let ret = ((pc_hi << 16) | (pc_mid << 8) | pc_lo) & pc_mask;
                sp = sp.wrapping_add(3) & mask_s;
                state.set_reg(RegName::S, sp);
                let imr_restored = imr;
                Self::store_traced(
                    bus,
                    INTERNAL_MEMORY_START + IMEM_IMR_OFFSET,
                    8,
                    imr_restored & 0xFF,
                );
                state.set_reg(RegName::IMR, imr_restored);
                state.set_reg(RegName::F, f);
                state.set_pc(ret);
                state.call_depth_dec();
                let instr_len = 1 + prefix_len;
                let mut payload = HashMap::new();
                payload.insert(
                    "pc_next".to_string(),
                    AnnotationValue::Pointer((ret & pc_mask) as u64),
                );
                payload.insert(
                    "ret_target".to_string(),
                    AnnotationValue::Pointer((ret & pc_mask) as u64),
                );
                payload.insert(
                    "instr_len".to_string(),
                    AnnotationValue::UInt(instr_len as u64),
                );
                payload.insert(
                    "call_depth".to_string(),
                    AnnotationValue::Int(state.call_depth() as i64),
                );
                Self::emit_control_flow_event(
                    entry.name,
                    "reti",
                    instr_index,
                    pc_before & pc_mask,
                    payload,
                );
                Ok(instr_len)
            }
            InstrKind::PushU | InstrKind::PushS => {
                let decoded =
                    self.decode_with_prefix(entry, state, bus, pre, pc_override, prefix_len)?;
                let reg = Self::operand_reg(entry.operands.first().ok_or("missing operand")?)
                    .ok_or("missing source")?;
                let bits = match entry.operands.first().copied().ok_or("missing operand")? {
                    OperandKind::Reg(_, b) => b,
                    OperandKind::RegB
                    | OperandKind::RegIL
                    | OperandKind::RegIMR
                    | OperandKind::RegF => 8,
                    _ => 8,
                };
                let value = Self::read_reg(state, bus, reg);
                let sp_reg = if entry.kind == InstrKind::PushU {
                    RegName::U
                } else {
                    RegName::S
                };
                Self::push_stack(state, bus, sp_reg, value, bits, false);
                if reg == RegName::IMR {
                    // Parity: `PUSH{U,S} IMR` is used by the ROM as a critical-section
                    // primitive. The Python emulator clears the IRM bit (bit 7) as part of the
                    // PUSH, and the corresponding `POP{U,S} IMR` restores the original value.
                    let imr_addr = INTERNAL_MEMORY_START + IMEM_IMR_OFFSET;
                    let cleared = (value & 0xFF) & 0x7F;
                    Self::store_traced(bus, imr_addr, 8, cleared);
                    state.set_reg(RegName::IMR, cleared);
                }
                let len = decoded.len;
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(len as u32));
                }
                Ok(len)
            }
            InstrKind::PopU | InstrKind::PopS => {
                let saved_fc = state.get_reg(RegName::FC);
                let saved_fz = state.get_reg(RegName::FZ);
                let decoded =
                    self.decode_with_prefix(entry, state, bus, pre, pc_override, prefix_len)?;
                let reg = Self::operand_reg(entry.operands.first().ok_or("missing operand")?)
                    .ok_or("missing destination")?;
                let restore_flags = !matches!(reg, RegName::F | RegName::FC | RegName::FZ);
                let bits = match entry.operands.first().copied().ok_or("missing operand")? {
                    OperandKind::Reg(_, b) => b,
                    OperandKind::RegB
                    | OperandKind::RegIL
                    | OperandKind::RegIMR
                    | OperandKind::RegF => 8,
                    _ => 8,
                };
                let sp_reg = if entry.kind == InstrKind::PopU {
                    RegName::U
                } else {
                    RegName::S
                };
                // LlamaState masks RegName::F to C/Z, matching the measured
                // POPU and POPS normalization behavior.
                let value = Self::pop_stack(state, bus, sp_reg, bits, false);
                state.set_reg(reg, value);
                if reg == RegName::IMR {
                    Self::store_traced(
                        bus,
                        INTERNAL_MEMORY_START + IMEM_IMR_OFFSET,
                        8,
                        value & 0xFF,
                    );
                }
                let len = decoded.len;
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(len as u32));
                }
                if restore_flags {
                    state.set_reg(RegName::FC, saved_fc);
                    state.set_reg(RegName::FZ, saved_fz);
                }
                Ok(len)
            }
            InstrKind::Cmp | InstrKind::Test => {
                let decoded =
                    self.decode_with_prefix(entry, state, bus, pre, pc_override, prefix_len)?;
                // Two-operand compare/test; handle reg/mem/immediate combos
                let (lhs, rhs, bits) = if entry.operands.len() == 2 {
                    let op1 = &entry.operands[0];
                    let op2 = &entry.operands[1];
                    let bits = match (op1, op2) {
                        (OperandKind::Reg(_, b1), _) => *b1,
                        (_, OperandKind::Reg(_, b2)) => *b2,
                        (OperandKind::IMemWidth(b), _) => b * 8,
                        (_, OperandKind::IMemWidth(b)) => b * 8,
                        (OperandKind::IMem(bits), _) => *bits,
                        (_, OperandKind::IMem(bits)) => *bits,
                        _ => 8,
                    };
                    let op_is_mem = |op: &OperandKind| {
                        matches!(
                            op,
                            OperandKind::IMem(_)
                                | OperandKind::IMemWidth(_)
                                | OperandKind::EMemAddrWidth(_)
                                | OperandKind::EMemAddrWidthOp(_)
                                | OperandKind::EMemRegWidth(_)
                                | OperandKind::EMemRegWidthMode(_)
                                | OperandKind::EMemIMemWidth(_)
                        )
                    };
                    let op_is_imm = |op: &OperandKind| {
                        matches!(op, OperandKind::Imm(_) | OperandKind::ImmOffset)
                    };

                    let lhs = if op_is_mem(op1) {
                        let mem = decoded.mem.ok_or("missing mem operand")?;
                        Self::load_wrapped(bus, mem.addr, mem.bits)
                    } else if op_is_imm(op1) {
                        decoded.imm.ok_or("missing immediate")?.0
                    } else if let Some(r) = Self::resolved_reg(op1, &decoded) {
                        Self::read_reg(state, bus, r)
                    } else {
                        decoded.imm.map(|v| v.0).unwrap_or(0)
                    };

                    let rhs = if op_is_mem(op2) {
                        let mem = decoded.mem2.or(decoded.mem).ok_or("missing mem operand")?;
                        Self::load_wrapped(bus, mem.addr, mem.bits)
                    } else if op_is_imm(op2) {
                        decoded.imm.ok_or("missing immediate")?.0
                    } else if let Some(r) = Self::resolved_reg(op2, &decoded) {
                        Self::read_reg(state, bus, r)
                    } else {
                        decoded.imm.map(|v| v.0).unwrap_or(0)
                    };
                    (lhs, rhs, bits)
                } else {
                    // Parity: Python decode does not emit other CMP/TEST operand shapes.
                    return Err("unsupported operand pattern");
                };
                let mask = Self::mask_for_width(bits);
                let lhs_m = lhs & mask;
                let rhs_m = rhs & mask;
                if entry.kind == InstrKind::Cmp {
                    Self::set_flags_cmp(state, lhs_m, rhs_m, bits);
                } else {
                    // TEST is logical AND setting flags
                    let res = lhs_m & rhs_m;
                    Self::set_flags_for_result(state, res, None);
                }
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(decoded.len as u32));
                }
                Ok(decoded.len)
            }
            InstrKind::Cmpw | InstrKind::Cmpp => {
                // C6/D6 compare 16-bit words. C7 and D7 compare three-byte
                // values. D7's memory operand retains all 24 bits while its
                // architectural X/Y/U/S operand is zero-extended from 20 bits.
                let decoded =
                    self.decode_with_prefix(entry, state, bus, pre, pc_override, prefix_len)?;
                let bits = if entry.kind == InstrKind::Cmpw {
                    16
                } else {
                    24
                };
                let mask = Self::mask_for_width(bits);
                let mut lhs = 0u32;
                let mut rhs = 0u32;
                if let (Some(m1), Some(m2)) = (decoded.mem, decoded.mem2) {
                    lhs = Self::load_wrapped(bus, m1.addr, bits) & mask;
                    rhs = Self::load_wrapped(bus, m2.addr, bits) & mask;
                } else if let Some((r1, r2, _)) = decoded.reg_pair {
                    lhs = state.get_reg(r1) & mask;
                    rhs = state.get_reg(r2) & mask;
                } else if let Some(r) = decoded.reg3 {
                    if let Some(mem) = decoded.mem {
                        // IMem(24), Reg3 form: memory is the left-hand side.
                        lhs = Self::load_wrapped(bus, mem.addr, bits) & mask;
                        rhs = state.get_reg(r) & mask_for(r);
                    } else {
                        lhs = state.get_reg(r) & mask;
                    }
                }
                Self::set_flags_cmp(state, lhs, rhs, bits);
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(decoded.len as u32));
                }
                Ok(decoded.len)
            }
            InstrKind::Ex | InstrKind::Exl => {
                let decoded =
                    self.decode_with_prefix(entry, state, bus, pre, pc_override, prefix_len)?;
                if entry.kind == InstrKind::Exl {
                    let (m1, m2) = decoded
                        .mem
                        .zip(decoded.mem2)
                        .ok_or("EXL requires two internal-memory operands")?;
                    let bits = m1.bits.min(m2.bits);
                    let mut addr1 = m1.addr;
                    let mut addr2 = m2.addr;
                    let length = effective_i_count(state);
                    for _ in 0..length {
                        let v1 = Self::load_wrapped(bus, addr1, bits);
                        let v2 = Self::load_wrapped(bus, addr2, bits);
                        Self::store_traced(bus, addr1, bits, v2);
                        Self::store_traced(bus, addr2, bits, v1);
                        addr1 = Self::advance_internal_addr_signed(addr1, 1);
                        addr2 = Self::advance_internal_addr_signed(addr2, 1);
                    }
                    state.set_reg(RegName::I, 0);
                    let start_pc = state.pc();
                    if state.pc() == start_pc {
                        state.set_pc(start_pc.wrapping_add(decoded.len as u32));
                    }
                    return Ok(decoded.len);
                }
                if entry.opcode == 0xC2 {
                    let (m1, m2) = decoded
                        .mem
                        .zip(decoded.mem2)
                        .ok_or("EXP requires two internal-memory operands")?;
                    let mut addr1 = m1.addr;
                    let mut addr2 = m2.addr;
                    // PC-E500 exact/+1/-1 captures establish sequential
                    // pairwise byte exchange, not whole-triple snapshots.
                    for _ in 0..3 {
                        let v1 = Self::load_wrapped(bus, addr1, 8);
                        let v2 = Self::load_wrapped(bus, addr2, 8);
                        Self::store_traced(bus, addr1, 8, v2);
                        Self::store_traced(bus, addr2, 8, v1);
                        addr1 = Self::advance_internal_addr_signed(addr1, 1);
                        addr2 = Self::advance_internal_addr_signed(addr2, 1);
                    }
                    let start_pc = state.pc();
                    if state.pc() == start_pc {
                        state.set_pc(start_pc.wrapping_add(decoded.len as u32));
                    }
                    return Ok(decoded.len);
                }
                // Swap two memory operands or two registers.
                if entry.operands.len() >= 2 {
                    if let (Some(dst_reg), Some(src_reg)) = (
                        Self::resolved_reg(&entry.operands[0], &decoded),
                        Self::resolved_reg(&entry.operands[1], &decoded),
                    ) {
                        let bits = match (&entry.operands[0], &entry.operands[1]) {
                            (OperandKind::Reg(_, b1), _) => *b1,
                            (_, OperandKind::Reg(_, b2)) => *b2,
                            _ => 8,
                        };
                        let mask = Self::mask_for_width(bits);
                        let v1 = Self::read_reg(state, bus, dst_reg) & mask;
                        let v2 = Self::read_reg(state, bus, src_reg) & mask;
                        state.set_reg(dst_reg, v2);
                        state.set_reg(src_reg, v1);
                        let start_pc = state.pc();
                        if state.pc() == start_pc {
                            state.set_pc(start_pc.wrapping_add(decoded.len as u32));
                        }
                        return Ok(decoded.len);
                    }
                }
                if let (Some(m1), Some(m2)) = (decoded.mem, decoded.mem2) {
                    let bits = m1.bits.min(m2.bits);
                    let v1 = Self::load_wrapped(bus, m1.addr, bits);
                    let v2 = Self::load_wrapped(bus, m2.addr, bits);
                    Self::store_traced(bus, m1.addr, bits, v2);
                    Self::store_traced(bus, m2.addr, bits, v1);
                    let start_pc = state.pc();
                    if state.pc() == start_pc {
                        state.set_pc(start_pc.wrapping_add(decoded.len as u32));
                    }
                    Ok(decoded.len)
                } else if let Some((r1, r2, bits)) = decoded.reg_pair {
                    let mask = Self::mask_for_width(bits);
                    let v1 = state.get_reg(r1) & mask;
                    let v2 = state.get_reg(r2) & mask;
                    state.set_reg(r1, v2);
                    state.set_reg(r2, v1);
                    let start_pc = state.pc();
                    if state.pc() == start_pc {
                        state.set_pc(start_pc.wrapping_add(decoded.len as u32));
                    }
                    Ok(decoded.len)
                } else if entry.operands.len() == 1
                    && matches!(entry.operands[0], OperandKind::RegB)
                {
                    // EX A,B variant (0xDD)
                    let a = state.get_reg(RegName::A);
                    let b = state.get_reg(RegName::B);
                    state.set_reg(RegName::A, b);
                    state.set_reg(RegName::B, a);
                    let len = decoded.len;
                    let start_pc = state.pc();
                    if state.pc() == start_pc {
                        state.set_pc(start_pc.wrapping_add(len as u32));
                    }
                    Ok(len)
                } else {
                    Err("unsupported EX operand pattern")
                }
            }
            InstrKind::Swap => {
                let decoded =
                    self.decode_with_prefix(entry, state, bus, pre, pc_override, prefix_len)?;
                let val = state.get_reg(RegName::A) & 0xFF;
                let swapped = ((val & 0x0F) << 4) | ((val >> 4) & 0x0F);
                state.set_reg(RegName::A, swapped);
                state.set_reg(RegName::FZ, if swapped == 0 { 1 } else { 0 });
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(decoded.len as u32));
                }
                Ok(decoded.len)
            }
        }
    }
}

impl Default for LlamaExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama::opcodes::OPCODES;
    use std::collections::{HashMap, HashSet};

    struct NullBus;
    impl LlamaBus for NullBus {
        fn load(&mut self, _addr: u32, _bits: u8) -> u32 {
            0
        }
        fn store(&mut self, _addr: u32, _bits: u8, _value: u32) {}
        fn peek_byte_silent(&mut self, _addr: u32) -> Option<u8> {
            Some(0)
        }
        fn wait_cycles(&mut self, _cycles: u32) {}
    }

    struct ResetBus {
        bytes: std::collections::HashMap<u32, u8>,
    }

    impl ResetBus {
        fn new(vector: [u8; 3]) -> Self {
            let mut bytes = std::collections::HashMap::new();
            // Interrupt vector is at 0xFFFFA; reset vector is at 0xFFFFD (little-endian).
            bytes.insert(INTERRUPT_VECTOR_ADDR, 0xEE);
            bytes.insert(INTERRUPT_VECTOR_ADDR + 1, 0xDD);
            bytes.insert(INTERRUPT_VECTOR_ADDR + 2, 0xCC);
            bytes.insert(ROM_RESET_VECTOR_ADDR, vector[0]);
            bytes.insert(ROM_RESET_VECTOR_ADDR + 1, vector[1]);
            bytes.insert(ROM_RESET_VECTOR_ADDR + 2, vector[2]);
            Self { bytes }
        }
    }

    impl LlamaBus for ResetBus {
        fn load(&mut self, addr: u32, _bits: u8) -> u32 {
            *self.bytes.get(&addr).unwrap_or(&0) as u32
        }
        fn store(&mut self, _addr: u32, _bits: u8, _value: u32) {}
        fn peek_byte_silent(&mut self, addr: u32) -> Option<u8> {
            Some(*self.bytes.get(&addr).unwrap_or(&0))
        }
    }

    #[test]
    fn trace_imr_peek_does_not_bump_memory_reads() {
        use crate::memory::MemoryImage;
        let mut mem = MemoryImage::new();
        mem.write_external_byte(0x0000, 0x00); // NOP
        let mut state = LlamaState::new();
        state.set_pc(0x0000);
        let mut exec = LlamaExecutor::new();
        struct Bus<'a> {
            mem: &'a mut MemoryImage,
        }
        impl<'a> LlamaBus for Bus<'a> {
            fn load(&mut self, addr: u32, bits: u8) -> u32 {
                self.mem.load(addr, bits).unwrap_or(0)
            }
            fn store(&mut self, addr: u32, bits: u8, value: u32) {
                let _ = self.mem.store(addr, bits, value);
            }
            fn peek_byte_silent(&mut self, addr: u32) -> Option<u8> {
                self.mem.read_byte_silent(addr)
            }
            fn peek_imem_silent(&mut self, offset: u32) -> u8 {
                self.mem.read_internal_byte_silent(offset).unwrap_or(0)
            }
        }
        // Simulate an opcode fetch to mirror the runtime bus.
        let _ = mem.load(0x0000, 8);
        let reads_before = mem.memory_read_count();
        {
            let mut bus = Bus { mem: &mut mem };
            exec.execute(0x00, &mut state, &mut bus)
                .expect("execute nop");
        }
        let reads_after = mem.memory_read_count();
        assert_eq!(
            reads_after.saturating_sub(reads_before),
            0,
            "IMR peeks should not bump memory reads beyond the opcode fetch"
        );
    }

    #[test]
    fn opcode_table_has_coverage() {
        assert_eq!(OPCODES.len(), 256, "expected dense opcode table");
        let mut seen = HashSet::new();
        for entry in OPCODES {
            assert!(
                seen.insert(entry.opcode),
                "duplicate opcode 0x{:02X}",
                entry.opcode
            );
        }
        assert_eq!(OPCODES.first().map(|e| e.opcode), Some(0x00));
        assert_eq!(OPCODES.last().map(|e| e.opcode), Some(0xFF));
    }

    #[test]
    fn opcode_table_marks_only_reserved_bytes_unknown() {
        let unknown: Vec<u8> = OPCODES
            .iter()
            .filter(|entry| entry.kind == InstrKind::Unknown)
            .map(|entry| entry.opcode)
            .collect();
        assert_eq!(unknown, vec![0x20, 0xBF]);
    }

    #[test]
    fn reserved_opcodes_fail_without_advancing_pc() {
        let mut exec = LlamaExecutor::new();
        for opcode in [0x20, 0xBF] {
            let mut state = LlamaState::new();
            let mut bus = MemBus::with_size(0x200);
            bus.mem[IMEM_IMR_OFFSET as usize] = 0xAA;
            state.set_pc(0x10);
            state.set_reg(RegName::IMR, 0x55);
            let err = exec
                .execute(opcode, &mut state, &mut bus)
                .expect_err("reserved opcode must fail closed");
            assert_eq!(err, "invalid or reserved opcode");
            assert_eq!(state.pc(), 0x10, "failure must not advance PC");
            assert_eq!(state.get_reg(RegName::IMR), 0x55);
            assert_eq!(bus.mem[IMEM_IMR_OFFSET as usize], 0xAA);
            assert!(bus.writes.is_empty());
        }
    }

    #[test]
    fn malformed_operand_encodings_fail_closed() {
        let cases: &[(u8, &[u8])] = &[
            (0x11, &[0x00]),       // JP requires X/Y/U/S, not A/IL/BA/I
            (0x11, &[0xA4]),       // JP selector upper bits are reserved
            (0x90, &[0x10]),       // invalid external-register mode
            (0x90, &[0x00]),       // A is not a three-byte pointer register
            (0x98, &[0x01, 0x10]), // invalid internal-indirect mode
            (0xFD, &[0x80]),       // forbidden register-pair high bit
            (0x44, &[0x41]),       // 16-bit ADD requires BA/I destination
            (0x6C, &[0x84]),       // INC selector upper bits are reserved
            (0x7C, &[0xFF]),       // DEC selector upper bits are reserved
            (0xD6, &[0x00, 0x10]), // CMPW requires BA/I
            (0xD6, &[0xA3, 0x10]), // selector upper bits are reserved
            (0xD7, &[0x02, 0x10]), // CMPP requires X/Y/U/S
            (0xD7, &[0xA4, 0x10]), // selector upper bits are reserved
            (0xE3, &[0x04, 0x10]), // MVL requires post-inc/pre-dec
            (0xEB, &[0x04, 0x10]), // reverse MVL has the same restriction
        ];

        for &(opcode, operands) in cases {
            let mut exec = LlamaExecutor::new();
            let mut state = LlamaState::new();
            let mut bus = MemBus::with_size(16);
            state.set_pc(0);
            bus.mem[0] = opcode;
            bus.mem[1..1 + operands.len()].copy_from_slice(operands);

            assert!(
                exec.execute(opcode, &mut state, &mut bus).is_err(),
                "malformed {opcode:02X} {operands:02X?} must fail"
            );
            assert_eq!(state.pc(), 0, "failure must not advance PC");
        }
    }

    #[test]
    fn narrow_register_jump_fails_before_cross_page_semantics_diverge() {
        let pc = 0xE1234usize;
        let mut exec = LlamaExecutor::new();
        let mut state = LlamaState::new();
        let mut bus = MemBus::with_size(pc + 2);
        bus.mem[pc..pc + 2].copy_from_slice(&[0x11, 0x00]); // invalid JP A
        state.set_pc(pc as u32);
        state.set_reg(RegName::A, 0x56);

        let err = exec
            .execute(0x11, &mut state, &mut bus)
            .expect_err("narrow JP register must fail closed");

        assert_eq!(err, "JP requires X, Y, U, or S");
        assert_eq!(state.pc(), pc as u32);
        assert!(bus.writes.is_empty());
    }

    #[test]
    fn redundant_or_irrelevant_pre_prefixes_fail_closed() {
        let cases: &[&[u8]] = &[
            &[0x30, 0x00],             // NOP has no IMEM selector
            &[0x30, 0xEF],             // WAIT has no IMEM selector
            &[0x30, 0xCE],             // TCL has no IMEM selector
            &[0x30, 0xDE],             // HALT has no IMEM selector
            &[0x30, 0xDF],             // OFF has no IMEM selector
            &[0x30, 0xFF],             // RESET has no IMEM selector
            &[0x32, 0xFE],             // misaligned apparent PRE+IR
            &[0x23, 0x07, 0x00, 0x00], // misaligned apparent PRE+JP
            &[0x36, 0x01],             // redundant PRE+RETI
            &[0x23, 0x84, 0x00],       // unrelated PRE+MV form
            &[0x25, 0x7C, 0x01],       // 25 is an operand byte at the alleged ROM site
            &[0x23, 0x48, 0x3F],       // F0002 is the operand byte of FD 23, not an entry
        ];

        for bytes in cases {
            let mut exec = LlamaExecutor::new();
            let mut state = LlamaState::new();
            let mut bus = MemBus::with_size(16);
            bus.mem[..bytes.len()].copy_from_slice(bytes);

            let err = exec
                .execute(bytes[0], &mut state, &mut bus)
                .expect_err("noncanonical PRE must fail closed");

            assert!(
                err.contains("PRE"),
                "unexpected error for {bytes:02X?}: {err}"
            );
            assert_eq!(state.pc(), 0, "failure must not advance PC");
            assert!(bus.writes.is_empty(), "failure must not write memory");
        }
    }

    #[test]
    fn unverified_bp_py_selector_bytes_must_be_zero() {
        let cases: &[&[u8]] = &[
            &[0x21, 0xC8, 0x00, 0x01], // BP+PY second selector
        ];

        for bytes in cases {
            let mut exec = LlamaExecutor::new();
            let mut state = LlamaState::new();
            let mut bus = MemBus::with_size(16);
            bus.mem[..bytes.len()].copy_from_slice(bytes);

            let err = exec
                .execute(bytes[0], &mut state, &mut bus)
                .expect_err("ignored nonzero selector must fail closed");

            assert!(err.contains("selector byte"));
            assert_eq!(state.pc(), 0, "failure must not advance PC");
            assert!(bus.writes.is_empty(), "failure must not write memory");
        }
    }

    #[test]
    fn two_consecutive_pre_prefixes_use_the_second_latch() {
        let mut exec = LlamaExecutor::new();
        let mut state = LlamaState::new();
        let mut bus = MemBus::with_size(0x100);
        bus.mem[..4].copy_from_slice(&[0x30, 0x31, 0xA0, 0x05]);
        state.set_reg(RegName::A, 0xE0);

        let len = exec
            .execute(0x30, &mut state, &mut bus)
            .expect("the measured two-PRE form must execute");

        assert_eq!(len, 4);
        assert_eq!(state.pc(), 4);
        assert_eq!(bus.mem[0x05], 0xE0);
    }

    #[test]
    fn three_consecutive_pre_prefixes_remain_unverified() {
        let mut exec = LlamaExecutor::new();
        let mut state = LlamaState::new();
        let mut bus = MemBus::with_size(8);
        bus.mem[..5].copy_from_slice(&[0x30, 0x31, 0x32, 0xA0, 0x05]);

        let err = exec
            .execute(0x30, &mut state, &mut bus)
            .expect_err("three PRE prefixes were not measured");

        assert_eq!(err, "more than two consecutive PRE prefixes are unverified");
        assert_eq!(state.pc(), 0);
    }

    #[test]
    fn measured_single_selector_pre_aliases_execute() {
        for bytes in [
            [0x30_u8, 0xA0, 0x05],
            [0x31, 0xA0, 0x05],
            [0x32, 0xA0, 0x05],
            [0x33, 0xA0, 0x05],
            [0x34, 0xA0, 0x05],
            [0x36, 0xA0, 0x05],
            [0x24, 0xA0, 0x05],
            [0x26, 0xA0, 0x05],
            [0x22, 0xA0, 0x05],
        ] {
            let mut exec = LlamaExecutor::new();
            let mut state = LlamaState::new();
            let mut bus = MemBus::with_size(0x100);
            bus.mem[..3].copy_from_slice(&bytes);
            bus.mem[IMEM_BP_OFFSET as usize] = 0x10;
            bus.mem[IMEM_PX_OFFSET as usize] = 0x20;
            state.set_reg(RegName::A, u32::from(bytes[0]));

            let len = exec
                .execute(bytes[0], &mut state, &mut bus)
                .expect("silicon-proven one-selector PRE alias");

            let expected = match bytes[0] {
                0x30..=0x33 => 0x05,
                0x34 | 0x36 => 0x25,
                0x24 | 0x26 => 0x30,
                0x22 => 0x15,
                _ => unreachable!(),
            };
            assert_eq!(len, 3);
            assert_eq!(bus.mem[expected], bytes[0]);
        }
    }

    #[test]
    fn reset_vector_matches_python_reset_vector() {
        let mut state = LlamaState::new();
        let mut bus = ResetBus::new([0xAA, 0xBB, 0x0C]);
        power_on_reset(&mut bus, &mut state).expect("valid reset vector");
        // Expected little-endian vector at 0xFFFFD.
        assert_eq!(state.pc(), 0x0C_BB_AA & mask_for(RegName::PC));
    }

    #[test]
    fn reset_rejects_noncanonical_vector_before_any_mutation() {
        let mut bus = MemBus::with_size((ROM_RESET_VECTOR_ADDR + 3) as usize);
        bus.mem[ROM_RESET_VECTOR_ADDR as usize] = 0x56;
        bus.mem[ROM_RESET_VECTOR_ADDR as usize + 1] = 0x34;
        bus.mem[ROM_RESET_VECTOR_ADDR as usize + 2] = 0xF2;
        for (offset, value) in [
            (IMEM_LCC_OFFSET, 0xA5),
            (IMEM_UCR_OFFSET, 0x5A),
            (IMEM_ISR_OFFSET, 0x3C),
            (IMEM_SCR_OFFSET, 0xC3),
            (IMEM_USR_OFFSET, 0xE7),
            (IMEM_SSR_OFFSET, 0x19),
        ] {
            bus.mem[MemBus::translate(INTERNAL_MEMORY_START + offset)] = value;
        }
        let memory_before = bus.mem.clone();
        let mut state = LlamaState::new();
        state.set_pc(0x45678);
        state.set_reg(RegName::S, 0x23456);
        state.set_reg(RegName::F, 0x03);
        state.set_reg(RegName::IMR, 0xA5);
        state.halt();

        let err = power_on_reset(&mut bus, &mut state)
            .expect_err("reserved reset-vector upper bits must be quarantined");

        assert_eq!(err, VECTOR_UPPER_NIBBLE_ERROR);
        assert_eq!(state.pc(), 0x45678);
        assert_eq!(state.get_reg(RegName::S), 0x23456);
        assert_eq!(state.get_reg(RegName::F), 0x03);
        assert_eq!(state.get_reg(RegName::IMR), 0xA5);
        assert!(state.is_halted());
        assert_eq!(bus.mem, memory_before);
        assert!(bus.writes.is_empty());
    }

    #[test]
    fn prefetched_reset_path_never_reads_vector_or_destination_again() {
        struct NoVectorReadBus {
            inner: MemBus,
        }

        impl LlamaBus for NoVectorReadBus {
            fn load(&mut self, addr: u32, bits: u8) -> u32 {
                assert!(
                    !(ROM_RESET_VECTOR_ADDR..=ROM_RESET_VECTOR_ADDR + 2).contains(&addr),
                    "prefetched RESET must not re-read its vector"
                );
                assert_ne!(
                    addr, 0x00200,
                    "prefetched RESET must not re-read its target"
                );
                self.inner.load(addr, bits)
            }

            fn store(&mut self, addr: u32, bits: u8, value: u32) {
                self.inner.store(addr, bits, value);
            }

            fn peek_byte_silent(&mut self, addr: u32) -> Option<u8> {
                assert!(
                    !(ROM_RESET_VECTOR_ADDR..=ROM_RESET_VECTOR_ADDR + 2).contains(&addr),
                    "prefetched RESET must not silently re-read its vector"
                );
                assert_ne!(
                    addr, 0x00200,
                    "prefetched RESET must not silently re-read its target"
                );
                self.inner.peek_byte_silent(addr)
            }

            fn vector_transfer_provenance(&self) -> (usize, u64) {
                (self.inner.mem.as_ptr() as usize, 0)
            }
        }

        let mut state = LlamaState::new();
        state.set_pc(0x12345);
        state.halt();
        let mut inner = MemBus::with_size((ROM_RESET_VECTOR_ADDR + 3) as usize);
        inner.mem[ROM_RESET_VECTOR_ADDR as usize] = 0x00;
        inner.mem[ROM_RESET_VECTOR_ADDR as usize + 1] = 0x02;
        inner.mem[ROM_RESET_VECTOR_ADDR as usize + 2] = 0x00;
        inner.mem[0x00200] = 0x00;
        let transfer = fetch_validated_vector(ROM_RESET_VECTOR_ADDR, &state, &mut inner)
            .expect("prepare validated RESET transfer");
        let mut bus = NoVectorReadBus { inner };

        power_on_reset_with_transfer(&mut bus, &mut state, transfer)
            .expect("opaque validated RESET transfer");

        assert_eq!(state.pc(), 0x00200);
        assert!(!state.is_halted());
        assert!(!bus.inner.writes.is_empty());
    }

    #[test]
    fn synchronous_ir_rejects_bad_vector_or_target_before_frame_mutation() {
        for (vector_high, target_opcode, expected_error) in [
            (0xF0, 0x00, VECTOR_UPPER_NIBBLE_ERROR),
            (0x00, 0x20, "invalid or reserved opcode"),
        ] {
            let mut bus = MemBus::with_size((INTERRUPT_VECTOR_ADDR + 3) as usize);
            let pc = 0x00100usize;
            let target = 0x00300usize;
            bus.mem[pc] = 0xFE;
            bus.mem[target] = target_opcode;
            bus.mem[INTERRUPT_VECTOR_ADDR as usize] = target as u8;
            bus.mem[INTERRUPT_VECTOR_ADDR as usize + 1] = (target >> 8) as u8;
            bus.mem[INTERRUPT_VECTOR_ADDR as usize + 2] = vector_high;
            let imr_index = MemBus::translate(INTERNAL_MEMORY_START + IMEM_IMR_OFFSET);
            bus.mem[imr_index] = 0xA5;
            let memory_before = bus.mem.clone();

            let mut state = LlamaState::new();
            state.set_pc(pc as u32);
            state.set_reg(RegName::S, 0x00400);
            state.set_reg(RegName::F, 0x03);
            state.set_reg(RegName::IMR, 0xA5);
            let mut exec = LlamaExecutor::new();

            let err = exec
                .execute(0xFE, &mut state, &mut bus)
                .expect_err("invalid IR transfer must fail atomically");

            assert_eq!(err, expected_error);
            assert_eq!(state.pc(), pc as u32);
            assert_eq!(state.get_reg(RegName::S), 0x00400);
            assert_eq!(state.get_reg(RegName::F), 0x03);
            assert_eq!(state.get_reg(RegName::IMR), 0xA5);
            assert_eq!(state.call_depth(), 0);
            assert_eq!(bus.mem, memory_before);
            assert!(bus.writes.is_empty());
        }
    }

    #[test]
    fn vector_destination_validation_does_not_recurse_or_read_stack_images() {
        for opcode in [0xFE, 0xFF, 0x01, 0xEF] {
            let target = 0x00300u32;
            let mut bus = MemBus::with_size((INTERRUPT_VECTOR_ADDR + 3) as usize);
            bus.mem[target as usize] = opcode;
            bus.mem[INTERRUPT_VECTOR_ADDR as usize] = target as u8;
            bus.mem[INTERRUPT_VECTOR_ADDR as usize + 1] = (target >> 8) as u8;
            bus.mem[INTERRUPT_VECTOR_ADDR as usize + 2] = (target >> 16) as u8;
            let mut state = LlamaState::new();
            state.set_reg(RegName::S, 0x00400);
            // RETI's eventual F image is deliberately unsupported. Static
            // vector validation must leave that data-dependent check to RETI.
            bus.mem[0x401] = 0xFC;

            assert_eq!(
                validate_vector_transfer(INTERRUPT_VECTOR_ADDR, &state, &mut bus),
                Ok(target),
                "target opcode 0x{opcode:02X} must not recursively follow vectors"
            );
        }
    }

    #[test]
    fn apply_pointer_side_effect_handles_predec_over_multiple_iterations() {
        let mut state = LlamaState::new();
        state.set_reg(RegName::X, 0x10);
        // Side-effect encodes pre-dec by one; three iterations should land at 0x0D.
        LlamaExecutor::apply_pointer_side_effect(&mut state, RegName::X, 0x0F, 3);
        assert_eq!(state.get_reg(RegName::X), 0x0D);
    }

    #[test]
    fn perfetto_trace_anchors_to_prefix_pc_and_opcode() {
        let _perfetto_lock = crate::perfetto::perfetto_test_guard();
        use crate::PerfettoTracer;
        // Program: canonical PRE (0x30) followed by MV (n),A (0xA0).
        let mut bus = MemBus::with_size(4);
        bus.mem[..3].copy_from_slice(&[0x30, 0xA0, 0x10]);

        let mut exec = LlamaExecutor::new();
        let mut state = LlamaState::new();
        state.set_pc(0);

        let path = std::env::temp_dir().join("llama_pref_trace.perfetto-trace");
        let _ = std::fs::remove_file(&path);
        let mut guard = crate::PERFETTO_TRACER.enter();
        guard.replace(Some(PerfettoTracer::new(path)));

        let len = exec
            .execute(0x30, &mut state, &mut bus)
            .expect("execute canonical PRE+MV");
        assert_eq!(len, 3, "PRE + MV should consume three bytes");

        let tracer = guard.take().expect("tracer should be installed");
        let events = tracer.test_exec_events();
        assert!(
            !events.is_empty(),
            "perfetto tracer should record at least one Exec event"
        );
        let (pc, opcode, _idx) = events[0];
        assert_eq!(pc, 0, "Exec PC should use prefix address");
        assert_eq!(opcode, 0x30, "Exec opcode should reflect prefix byte");
    }

    struct OffsetBus {
        data: HashMap<u32, u8>,
        reads: Vec<u32>,
        writes: Vec<(u32, u8)>,
    }

    impl OffsetBus {
        fn new() -> Self {
            Self {
                data: HashMap::new(),
                reads: Vec::new(),
                writes: Vec::new(),
            }
        }

        fn seed_pointer(&mut self, ptr_base: u32, pointer: u32) {
            self.data.insert(ptr_base, (pointer & 0xFF) as u8);
            self.data
                .insert(ptr_base + 1, ((pointer >> 8) & 0xFF) as u8);
            self.data
                .insert(ptr_base + 2, ((pointer >> 16) & 0xFF) as u8);
        }
    }

    impl LlamaBus for OffsetBus {
        fn load(&mut self, addr: u32, _bits: u8) -> u32 {
            self.reads.push(addr);
            *self.data.get(&addr).unwrap_or(&0) as u32
        }
        fn store(&mut self, addr: u32, _bits: u8, value: u32) {
            self.data.insert(addr, (value & 0xFF) as u8);
            self.writes.push((addr, (value & 0xFF) as u8));
        }
        fn peek_byte_silent(&mut self, addr: u32) -> Option<u8> {
            Some(*self.data.get(&addr).unwrap_or(&0))
        }
        fn resolve_emem(&mut self, base: u32) -> u32 {
            base
        }
    }

    #[test]
    fn exl_exchanges_all_i_bytes_and_wraps_internal_addresses() {
        let mut exec = LlamaExecutor::new();
        let mut state = LlamaState::new();
        let mut bus = MemBus::with_size(0x100);
        bus.mem[..3].copy_from_slice(&[0xC3, 0xFE, 0x20]);
        bus.mem[0xFE] = 0x11;
        bus.mem[0xFF] = 0x22;
        bus.mem[0x00] = 0x33;
        bus.mem[0x20..0x23].copy_from_slice(&[0xAA, 0xBB, 0xCC]);
        state.set_reg(RegName::I, 3);

        let len = exec.execute(0xC3, &mut state, &mut bus).unwrap();

        assert_eq!(len, 3);
        assert_eq!(
            [bus.mem[0xFE], bus.mem[0xFF], bus.mem[0x00]],
            [0xAA, 0xBB, 0xCC]
        );
        assert_eq!(&bus.mem[0x20..0x23], &[0x11, 0x22, 0x33]);
        assert_eq!(state.get_reg(RegName::I), 0);
    }

    #[test]
    fn ir_frame_saves_ir_address_for_rom_dispatcher() {
        let mut exec = LlamaExecutor::new();
        let mut state = LlamaState::new();
        let mut bus = OffsetBus::new();
        let ir_pc = 0x012345;
        let vector = 0x0ABCDE;
        bus.data.insert(ir_pc, 0xFE);
        bus.data
            .insert(INTERRUPT_VECTOR_ADDR, (vector & 0xFF) as u8);
        bus.data
            .insert(INTERRUPT_VECTOR_ADDR + 1, ((vector >> 8) & 0xFF) as u8);
        bus.data
            .insert(INTERRUPT_VECTOR_ADDR + 2, ((vector >> 16) & 0xFF) as u8);
        bus.data
            .insert(INTERNAL_MEMORY_START + IMEM_IMR_OFFSET, 0x80);
        state.set_pc(ir_pc);
        state.set_reg(RegName::S, 0x200);

        exec.execute(0xFE, &mut state, &mut bus).unwrap();

        let sp = state.get_reg(RegName::S);
        let saved_pc = u32::from(*bus.data.get(&(sp + 2)).unwrap_or(&0))
            | (u32::from(*bus.data.get(&(sp + 3)).unwrap_or(&0)) << 8)
            | (u32::from(*bus.data.get(&(sp + 4)).unwrap_or(&0)) << 16);
        assert_eq!(saved_pc, ir_pc);
        assert_eq!(bus.load(saved_pc, 8), 0xFE);
        assert_eq!(state.pc(), vector);
    }

    #[test]
    fn emem_imem_offset_respects_sign() {
        let entry = dispatch::lookup(0xF0).expect("opcode present");
        let exec = LlamaExecutor::new();

        // Positive offset (+5)
        let mut bus = OffsetBus::new();
        bus.data.insert(0, 0x80); // mode = positive offset
        bus.data.insert(1, 0x10); // first IMEM addr
        bus.data.insert(2, 0x20); // second IMEM addr (pointer)
        bus.data.insert(3, 0x05); // offset magnitude
        let base_ptr = 0x001000;
        let ptr_base = INTERNAL_MEMORY_START + 0x20;
        bus.seed_pointer(ptr_base, base_ptr);
        let (transfer, consumed) = exec
            .decode_emem_imem_offset(
                entry,
                &mut bus,
                0,
                AddressingMode::N,
                AddressingMode::N,
                true,
            )
            .expect("positive offset should decode");
        assert_eq!(consumed, 4, "positive offset should consume mode+ptr+disp");
        assert_eq!(
            transfer.src_addr,
            base_ptr + 5,
            "positive offset should add displacement"
        );
        assert_eq!(
            transfer.dst_addr,
            INTERNAL_MEMORY_START + 0x10,
            "dest should use first IMEM byte"
        );

        // Negative offset (-3)
        let mut bus_neg = OffsetBus::new();
        bus_neg.data.insert(0, 0xC0); // mode = negative offset
        bus_neg.data.insert(1, 0x11);
        bus_neg.data.insert(2, 0x21);
        bus_neg.data.insert(3, 0x03);
        let base_ptr_neg = 0x000900;
        let ptr_base_neg = INTERNAL_MEMORY_START + 0x21;
        bus_neg.seed_pointer(ptr_base_neg, base_ptr_neg);
        let (transfer_neg, consumed_neg) = exec
            .decode_emem_imem_offset(
                entry,
                &mut bus_neg,
                0,
                AddressingMode::N,
                AddressingMode::N,
                true,
            )
            .expect("negative offset should decode");
        assert_eq!(consumed_neg, 4);
        assert_eq!(
            transfer_neg.src_addr,
            base_ptr_neg - 3,
            "negative offset should subtract displacement"
        );
        assert_eq!(
            transfer_neg.dst_addr,
            INTERNAL_MEMORY_START + 0x11,
            "dest should use first IMEM byte"
        );
    }

    #[test]
    fn emem_imem_rejects_unknown_offset_mode() {
        let entry = dispatch::lookup(0xF0).expect("opcode present");
        let exec = LlamaExecutor::new();
        let mut bus = OffsetBus::new();
        bus.data.insert(0, 0x40); // invalid mode per spec
        bus.data.insert(1, 0x00);
        bus.data.insert(2, 0x00);
        bus.data.insert(3, 0x01);
        let ptr_base = INTERNAL_MEMORY_START;
        bus.seed_pointer(ptr_base, 0x000100);
        let res = exec.decode_emem_imem_offset(
            entry,
            &mut bus,
            0,
            AddressingMode::N,
            AddressingMode::N,
            true,
        );
        assert!(res.is_err(), "invalid mode should be rejected");
    }

    fn run_composite_mvl_case(
        pc: u32,
        program: &[u8],
        length: u32,
        regs: &[(RegName, u32)],
        seed: &[(u32, u8)],
    ) -> (OffsetBus, LlamaState, u8) {
        let mut bus = OffsetBus::new();
        for (index, byte) in program.iter().enumerate() {
            bus.data.insert(pc + index as u32, *byte);
        }
        for (addr, value) in seed {
            bus.data.insert(*addr, *value);
        }
        let mut state = LlamaState::new();
        state.set_pc(pc);
        state.set_reg(RegName::I, length);
        for (reg, value) in regs {
            state.set_reg(*reg, *value);
        }
        let mut exec = LlamaExecutor::new();
        let consumed = exec.execute(program[0], &mut state, &mut bus).unwrap();
        (bus, state, consumed)
    }

    #[test]
    fn mvl_56_copies_exact_i_external_to_internal() {
        // MVL (E0),[X+00]
        let program = [0x56, 0x84, 0xE0, 0x00];
        let seed = [(0x200, 0x11), (0x201, 0x22)];
        let (bus, state, consumed) =
            run_composite_mvl_case(0, &program, 2, &[(RegName::X, 0x200)], &seed);
        assert_eq!(consumed, 4);
        assert_eq!(state.get_reg(RegName::I), 0);
        assert_eq!(state.get_reg(RegName::X), 0x200);
        assert_eq!(
            bus.writes,
            vec![
                (INTERNAL_MEMORY_START + 0xE0, 0x11),
                (INTERNAL_MEMORY_START + 0xE1, 0x22),
            ]
        );
    }

    #[test]
    fn mvl_5e_copies_exact_i_internal_to_external() {
        // MVL [X+00],(E0)
        let program = [0x5E, 0x84, 0xE0, 0x00];
        let seed = [
            (INTERNAL_MEMORY_START + 0xE0, 0x33),
            (INTERNAL_MEMORY_START + 0xE1, 0x44),
        ];
        let (bus, state, consumed) =
            run_composite_mvl_case(0, &program, 2, &[(RegName::X, 0x300)], &seed);
        assert_eq!(consumed, 4);
        assert_eq!(state.get_reg(RegName::I), 0);
        assert_eq!(state.get_reg(RegName::X), 0x300);
        assert_eq!(bus.writes, vec![(0x300, 0x33), (0x301, 0x44)]);
    }

    #[test]
    fn mvl_f3_copies_exact_i_external_pointer_to_internal() {
        // MVL (D0),[(F0)] with [(F0)] = 0x00400.
        let program = [0xF3, 0x00, 0xD0, 0xF0];
        let seed = [
            (INTERNAL_MEMORY_START + 0xF0, 0x00),
            (INTERNAL_MEMORY_START + 0xF1, 0x04),
            (INTERNAL_MEMORY_START + 0xF2, 0x00),
            (0x400, 0x55),
            (0x401, 0x66),
        ];
        let (bus, state, consumed) = run_composite_mvl_case(0, &program, 2, &[], &seed);
        assert_eq!(consumed, 4);
        assert_eq!(state.get_reg(RegName::I), 0);
        assert_eq!(
            bus.writes,
            vec![
                (INTERNAL_MEMORY_START + 0xD0, 0x55),
                (INTERNAL_MEMORY_START + 0xD1, 0x66),
            ]
        );
    }

    #[test]
    fn mvl_f3_snapshots_external_pointer_before_overlapping_destination_writes() {
        // Hardware evidence: PC-E500 runs 20260831-175849-0001 through
        // 20260831-175854-0003 and 20260831-181618-0002 through
        // 20260831-181621-0003.  The FT600 bus trace observed sequential CE1
        // reads from the initial effective pointer even while each destination
        // write overwrote part of that pointer in IMEM.
        let cases = [
            (
                vec![0xF3, 0x00, 0x40, 0x40],
                3,
                vec![0x40680, 0x40681, 0x40682],
            ),
            (
                vec![0xF3, 0x00, 0x3F, 0x40],
                3,
                vec![0x40680, 0x40681, 0x40682],
            ),
            (vec![0xF3, 0x00, 0x41, 0x40], 2, vec![0x40680, 0x40681]),
            (
                vec![0xF3, 0x80, 0x40, 0x40, 0x10],
                3,
                vec![0x40690, 0x40691, 0x40692],
            ),
            (
                vec![0xF3, 0xC0, 0x40, 0x40, 0x10],
                3,
                vec![0x40670, 0x40671, 0x40672],
            ),
        ];

        for (program, count, expected_external_reads) in cases {
            let seed = [
                (INTERNAL_MEMORY_START + 0x3F, 0xFF),
                (INTERNAL_MEMORY_START + 0x40, 0x80),
                (INTERNAL_MEMORY_START + 0x41, 0x06),
                (INTERNAL_MEMORY_START + 0x42, 0x04),
                (INTERNAL_MEMORY_START + 0x43, 0xFF),
            ];
            let (bus, state, consumed) = run_composite_mvl_case(0, &program, count, &[], &seed);
            let external_reads: Vec<u32> = bus
                .reads
                .iter()
                .copied()
                .filter(|address| (0x40000..=0x7FFFF).contains(address))
                .collect();

            assert_eq!(usize::from(consumed), program.len());
            assert_eq!(state.get_reg(RegName::I), 0);
            assert_eq!(external_reads, expected_external_reads);
        }
    }

    #[test]
    fn fixed_width_external_pointer_moves_snapshot_before_overlapping_writes() {
        // Hardware evidence: PC-E500 runs 20260831-181830-0001 and
        // 20260831-181832-0002 plus displaced runs 20260831-184053-0001
        // through 20260831-184100-0004.
        let cases = [
            (
                vec![0xF1, 0x00, 0x40, 0x40],
                vec![0x40680, 0x40681],
                [0x00, 0x00, 0x04],
            ),
            (
                vec![0xF2, 0x00, 0x40, 0x40],
                vec![0x40680, 0x40681, 0x40682],
                [0x00, 0x00, 0x00],
            ),
            (
                vec![0xF1, 0x80, 0x40, 0x40, 0x10],
                vec![0x40690, 0x40691],
                [0x00, 0x00, 0x04],
            ),
            (
                vec![0xF1, 0xC0, 0x40, 0x40, 0x10],
                vec![0x40670, 0x40671],
                [0x00, 0x00, 0x04],
            ),
            (
                vec![0xF2, 0x80, 0x40, 0x40, 0x10],
                vec![0x40690, 0x40691, 0x40692],
                [0x00, 0x00, 0x00],
            ),
            (
                vec![0xF2, 0xC0, 0x40, 0x40, 0x10],
                vec![0x40670, 0x40671, 0x40672],
                [0x00, 0x00, 0x00],
            ),
        ];

        for (program, expected_external_reads, expected_pointer) in cases {
            let seed = [
                (INTERNAL_MEMORY_START + 0x40, 0x80),
                (INTERNAL_MEMORY_START + 0x41, 0x06),
                (INTERNAL_MEMORY_START + 0x42, 0x04),
            ];
            let (bus, state, consumed) = run_composite_mvl_case(0, &program, 0x5AA5, &[], &seed);
            let external_reads: Vec<u32> = bus
                .reads
                .iter()
                .copied()
                .filter(|address| (0x40000..=0x7FFFF).contains(address))
                .collect();
            let pointer_after = [
                *bus.data.get(&(INTERNAL_MEMORY_START + 0x40)).unwrap(),
                *bus.data.get(&(INTERNAL_MEMORY_START + 0x41)).unwrap(),
                *bus.data.get(&(INTERNAL_MEMORY_START + 0x42)).unwrap(),
            ];

            assert_eq!(usize::from(consumed), program.len());
            assert_eq!(state.get_reg(RegName::I), 0x5AA5);
            assert_eq!(external_reads, expected_external_reads);
            assert_eq!(pointer_after, expected_pointer);
        }
    }

    #[test]
    fn external_pointer_destinations_emit_hardware_write_address_order() {
        // Hardware evidence: PC-E500 direct runs 20260831-190602-0001 through
        // 20260831-190609-0004 and displaced runs 20260831-191516-0001 through
        // 20260831-191531-0008 observed exactly 1/2/3/4 low-to-high CE1 write
        // phases for every valid F8/F9/FA/FB pointer mode. The loaded gateware
        // did not preserve or expose write data, so byte values here remain an
        // ISA contract.
        let cases = [
            (vec![0xF8, 0x00, 0x40, 0x60], 0x5AA5, vec![(0x406A0, 0xA5)]),
            (
                vec![0xF9, 0x00, 0x40, 0x60],
                0x5AA5,
                vec![(0x406A0, 0xA5), (0x406A1, 0x5A)],
            ),
            (
                vec![0xFA, 0x00, 0x40, 0x60],
                0x5AA5,
                vec![(0x406A0, 0xA5), (0x406A1, 0x5A), (0x406A2, 0x3C)],
            ),
            (
                vec![0xFB, 0x00, 0x40, 0x60],
                4,
                vec![
                    (0x406A0, 0xA5),
                    (0x406A1, 0x5A),
                    (0x406A2, 0x3C),
                    (0x406A3, 0xC3),
                ],
            ),
            (
                vec![0xF8, 0x80, 0x40, 0x60, 0x10],
                0x5AA5,
                vec![(0x406B0, 0xA5)],
            ),
            (
                vec![0xF8, 0xC0, 0x40, 0x60, 0x10],
                0x5AA5,
                vec![(0x40690, 0xA5)],
            ),
            (
                vec![0xF9, 0x80, 0x40, 0x60, 0x10],
                0x5AA5,
                vec![(0x406B0, 0xA5), (0x406B1, 0x5A)],
            ),
            (
                vec![0xF9, 0xC0, 0x40, 0x60, 0x10],
                0x5AA5,
                vec![(0x40690, 0xA5), (0x40691, 0x5A)],
            ),
            (
                vec![0xFA, 0x80, 0x40, 0x60, 0x10],
                0x5AA5,
                vec![(0x406B0, 0xA5), (0x406B1, 0x5A), (0x406B2, 0x3C)],
            ),
            (
                vec![0xFA, 0xC0, 0x40, 0x60, 0x10],
                0x5AA5,
                vec![(0x40690, 0xA5), (0x40691, 0x5A), (0x40692, 0x3C)],
            ),
            (
                vec![0xFB, 0x80, 0x40, 0x60, 0x10],
                4,
                vec![
                    (0x406B0, 0xA5),
                    (0x406B1, 0x5A),
                    (0x406B2, 0x3C),
                    (0x406B3, 0xC3),
                ],
            ),
            (
                vec![0xFB, 0xC0, 0x40, 0x60, 0x10],
                4,
                vec![
                    (0x40690, 0xA5),
                    (0x40691, 0x5A),
                    (0x40692, 0x3C),
                    (0x40693, 0xC3),
                ],
            ),
        ];

        for (program, count, expected_writes) in cases {
            let seed = [
                (INTERNAL_MEMORY_START + 0x40, 0xA0),
                (INTERNAL_MEMORY_START + 0x41, 0x06),
                (INTERNAL_MEMORY_START + 0x42, 0x04),
                (INTERNAL_MEMORY_START + 0x60, 0xA5),
                (INTERNAL_MEMORY_START + 0x61, 0x5A),
                (INTERNAL_MEMORY_START + 0x62, 0x3C),
                (INTERNAL_MEMORY_START + 0x63, 0xC3),
            ];
            let (bus, state, consumed) = run_composite_mvl_case(0, &program, count, &[], &seed);

            assert_eq!(usize::from(consumed), program.len());
            assert_eq!(
                state.get_reg(RegName::I),
                if program[0] == 0xFB { 0 } else { count }
            );
            assert_eq!(bus.writes, expected_writes);
        }
    }

    #[test]
    fn mvl_fb_copies_exact_i_internal_to_external_pointer() {
        // MVL [(F0)],(D0) with [(F0)] = 0x00500.
        let program = [0xFB, 0x00, 0xF0, 0xD0];
        let seed = [
            (INTERNAL_MEMORY_START + 0xF0, 0x00),
            (INTERNAL_MEMORY_START + 0xF1, 0x05),
            (INTERNAL_MEMORY_START + 0xF2, 0x00),
            (INTERNAL_MEMORY_START + 0xD0, 0x77),
            (INTERNAL_MEMORY_START + 0xD1, 0x88),
        ];
        let (bus, state, consumed) = run_composite_mvl_case(0, &program, 2, &[], &seed);
        assert_eq!(consumed, 4);
        assert_eq!(state.get_reg(RegName::I), 0);
        assert_eq!(bus.writes, vec![(0x500, 0x77), (0x501, 0x88)]);
    }

    #[test]
    fn mvl_56_wraps_internal_and_external_byte_streams() {
        // Place the instruction away from external zero because X starts at
        // 0xFFFFF and the second source byte wraps to address zero.
        let pc = 0x100;
        let program = [0x56, 0x84, 0xFF, 0x00];
        let seed = [(0xFFFFF, 0x91), (0x00000, 0x92), (0x00001, 0x93)];
        let (bus, state, _) =
            run_composite_mvl_case(pc, &program, 3, &[(RegName::X, 0xFFFFF)], &seed);
        assert_eq!(state.get_reg(RegName::I), 0);
        assert_eq!(
            bus.writes,
            vec![
                (INTERNAL_MEMORY_START + 0xFF, 0x91),
                (INTERNAL_MEMORY_START, 0x92),
                (INTERNAL_MEMORY_START + 1, 0x93),
            ]
        );
    }

    #[test]
    fn mvl_5e_wraps_internal_and_external_byte_streams() {
        let pc = 0x100;
        let program = [0x5E, 0x84, 0xFF, 0x00];
        let seed = [
            (INTERNAL_MEMORY_START + 0xFF, 0xA1),
            (INTERNAL_MEMORY_START, 0xA2),
            (INTERNAL_MEMORY_START + 1, 0xA3),
        ];
        let (bus, state, _) =
            run_composite_mvl_case(pc, &program, 3, &[(RegName::X, 0xFFFFF)], &seed);
        assert_eq!(state.get_reg(RegName::I), 0);
        assert_eq!(
            bus.writes,
            vec![(0xFFFFF, 0xA1), (0x00000, 0xA2), (0x00001, 0xA3)]
        );
    }

    #[test]
    fn multi_byte_memory_access_wraps_imem_ff_and_emem_fffff() {
        let mut bus = OffsetBus::new();
        bus.data.insert(INTERNAL_MEMORY_START + 0xFF, 0x11);
        bus.data.insert(INTERNAL_MEMORY_START, 0x22);
        bus.data.insert(INTERNAL_MEMORY_START + 1, 0x33);
        bus.data.insert(0xFFFFF, 0x44);
        bus.data.insert(0, 0x55);
        bus.data.insert(1, 0x66);

        assert_eq!(
            LlamaExecutor::load_wrapped(&mut bus, INTERNAL_MEMORY_START + 0xFF, 24),
            0x33_22_11
        );
        assert_eq!(
            LlamaExecutor::load_wrapped(&mut bus, 0xFFFFF, 24),
            0x66_55_44
        );
        assert_eq!(
            LlamaExecutor::read_imm(&mut bus, INTERNAL_MEMORY_START + 0xFF, 24),
            0x33_22_11
        );

        bus.writes.clear();
        LlamaExecutor::store_traced(&mut bus, INTERNAL_MEMORY_START + 0xFF, 24, 0xAA_BB_CC);
        assert_eq!(
            bus.writes,
            vec![
                (INTERNAL_MEMORY_START + 0xFF, 0xCC),
                (INTERNAL_MEMORY_START, 0xBB),
                (INTERNAL_MEMORY_START + 1, 0xAA),
            ]
        );

        bus.writes.clear();
        LlamaExecutor::store_traced(&mut bus, 0xFFFFF, 24, 0xDD_EE_FF);
        assert_eq!(
            bus.writes,
            vec![(0xFFFFF, 0xFF), (0x00000, 0xEE), (0x00001, 0xDD)]
        );
    }

    #[test]
    fn instruction_operand_fetch_wraps_pc_fffff_to_external_zero() {
        let mut bus = OffsetBus::new();
        bus.data.insert(0xFFFFF, 0x40); // ADD A,imm8
        bus.data.insert(0x00000, 0x05); // wrapped immediate
        bus.data.insert(INTERNAL_MEMORY_START, 0xE0); // must not be fetched
        let mut state = LlamaState::new();
        state.set_pc(0xFFFFF);
        state.set_reg(RegName::A, 1);
        let mut exec = LlamaExecutor::new();

        let consumed = exec.execute(0x40, &mut state, &mut bus).unwrap();

        assert_eq!(consumed, 2);
        assert_eq!(state.get_reg(RegName::A), 6);
        assert_eq!(state.pc(), 1);
    }

    #[test]
    fn wait_without_cycle_hook_fails_without_mutation() {
        let mut exec = LlamaExecutor::new();
        let mut state = LlamaState::new();
        let mut bus = NullBus;
        state.set_reg(RegName::FC, 1);
        state.set_reg(RegName::FZ, 1);
        state.set_reg(RegName::I, 5);
        let err = exec.execute(0xEF, &mut state, &mut bus).unwrap_err(); // WAIT
        assert_eq!(err, "WAIT requires a cycle-capable bus");
        assert_eq!(state.pc(), 0);
        assert_eq!(state.get_reg(RegName::I), 5);
        assert_eq!(state.get_reg(RegName::FC), 1, "WAIT should preserve C");
        assert_eq!(state.get_reg(RegName::FZ), 1, "WAIT should preserve Z");
    }

    struct WaitBus {
        spins: u32,
        calls: u32,
    }

    impl LlamaBus for WaitBus {
        fn load(&mut self, _addr: u32, _bits: u8) -> u32 {
            0
        }

        fn store(&mut self, _addr: u32, _bits: u8, _value: u32) {}

        fn peek_byte_silent(&mut self, _addr: u32) -> Option<u8> {
            Some(0)
        }

        fn supports_wait_cycles(&self) -> bool {
            true
        }

        fn wait_cycles(&mut self, cycles: u32) {
            self.calls = self.calls.saturating_add(1);
            self.spins = self.spins.saturating_add(cycles);
        }
    }

    #[test]
    fn wait_does_not_tick_timers() {
        let mut exec = LlamaExecutor::new();
        let mut state = LlamaState::new();
        let mut bus = WaitBus { spins: 0, calls: 0 };
        state.set_reg(RegName::I, 5);
        let len = exec.execute(0xEF, &mut state, &mut bus).unwrap(); // WAIT
        assert_eq!(len, 1);
        assert_eq!(state.pc(), 1);
        assert_eq!(bus.calls, 1, "WAIT should tick timers via wait_cycles");
        assert_eq!(bus.spins, 5, "WAIT should consume the requested cycles");
        assert_eq!(state.get_reg(RegName::I), 0);
        assert_eq!(state.get_reg(RegName::FC), 0);
        assert_eq!(state.get_reg(RegName::FZ), 0);
    }

    #[test]
    fn hw002_wait_i_zero_consumes_full_16bit_wrap() {
        let mut exec = LlamaExecutor::new();
        let mut state = LlamaState::new();
        let mut bus = WaitBus { spins: 0, calls: 0 };
        state.set_pc(0x34567);
        state.set_reg(RegName::I, 0);
        state.set_reg(RegName::FC, 1);
        state.set_reg(RegName::FZ, 1);
        state.set_reg(RegName::S, 0x45678);

        let len = exec.execute(0xEF, &mut state, &mut bus).unwrap();

        assert_eq!(len, 1);
        assert_eq!(state.pc(), 0x34568);
        assert_eq!(state.get_reg(RegName::I), 0);
        assert_eq!(state.get_reg(RegName::FC), 1);
        assert_eq!(state.get_reg(RegName::FZ), 1);
        assert_eq!(state.get_reg(RegName::S), 0x45678);
        assert_eq!(bus.calls, 1);
        assert_eq!(bus.spins, 0x1_0000);
    }

    struct MemBus {
        mem: Vec<u8>,
        writes: Vec<(u32, u8, u32)>,
    }
    impl MemBus {
        fn with_size(size: usize) -> Self {
            Self {
                mem: vec![0; size],
                writes: Vec::new(),
            }
        }

        fn translate(addr: u32) -> usize {
            if addr >= INTERNAL_MEMORY_START {
                (addr - INTERNAL_MEMORY_START) as usize
            } else {
                addr as usize
            }
        }
    }
    impl LlamaBus for MemBus {
        fn supports_wait_cycles(&self) -> bool {
            true
        }

        fn load(&mut self, addr: u32, bits: u8) -> u32 {
            let mut val = 0u32;
            let bytes = bits.div_ceil(8);
            for i in 0..bytes {
                let idx = Self::translate(addr).saturating_add(i as usize);
                let b = *self.mem.get(idx).unwrap_or(&0) as u32;
                val |= b << (8 * i);
            }
            val & ((1u32 << bits) - 1)
        }

        fn peek_byte_silent(&mut self, addr: u32) -> Option<u8> {
            Some(self.mem.get(Self::translate(addr)).copied().unwrap_or(0))
        }

        fn vector_transfer_provenance(&self) -> (usize, u64) {
            (self.mem.as_ptr() as usize, 0)
        }

        fn store(&mut self, addr: u32, bits: u8, value: u32) {
            self.writes.push((addr, bits, value));
            let bytes = bits.div_ceil(8);
            for i in 0..bytes {
                let idx = Self::translate(addr).saturating_add(i as usize);
                if let Some(slot) = self.mem.get_mut(idx) {
                    *slot = ((value >> (8 * i)) & 0xFF) as u8;
                }
            }
        }

        fn resolve_emem(&mut self, base: u32) -> u32 {
            base
        }

        fn wait_cycles(&mut self, _cycles: u32) {}
    }

    struct VolatileVectorBus {
        silent: Vec<u8>,
        fetched: [u8; 3],
        vector_reads: Vec<u32>,
        writes: Vec<(u32, u8, u32)>,
        events: Vec<(&'static str, u32)>,
    }

    impl LlamaBus for VolatileVectorBus {
        fn load(&mut self, addr: u32, bits: u8) -> u32 {
            if bits == 8 && (INTERRUPT_VECTOR_ADDR..=INTERRUPT_VECTOR_ADDR + 2).contains(&addr) {
                self.vector_reads.push(addr);
                self.events.push(("vector-read", addr));
                return u32::from(self.fetched[(addr - INTERRUPT_VECTOR_ADDR) as usize]);
            }
            u32::from(self.silent.get(addr as usize).copied().unwrap_or(0))
        }

        fn store(&mut self, addr: u32, bits: u8, value: u32) {
            self.writes.push((addr, bits, value));
            self.events.push(("write", addr));
        }

        fn peek_byte_silent(&mut self, addr: u32) -> Option<u8> {
            Some(self.silent.get(addr as usize).copied().unwrap_or(0))
        }

        fn supports_wait_cycles(&self) -> bool {
            true
        }
    }

    #[test]
    fn synchronous_ir_detects_volatile_vector_change_after_frame_mutation() {
        let _perfetto_lock = crate::perfetto::perfetto_test_guard();
        let mut exec = LlamaExecutor::new();
        reset_perf_counters();
        set_perf_instr_counter(17);
        let mut silent = vec![0; (INTERRUPT_VECTOR_ADDR + 3) as usize];
        let pc = 0x00100usize;
        let silent_target = 0x00300usize;
        let fetched_target = 0x00400u32;
        silent[pc] = 0xFE;
        silent[INTERRUPT_VECTOR_ADDR as usize] = silent_target as u8;
        silent[INTERRUPT_VECTOR_ADDR as usize + 1] = (silent_target >> 8) as u8;
        silent[INTERRUPT_VECTOR_ADDR as usize + 2] = (silent_target >> 16) as u8;
        let mut bus = VolatileVectorBus {
            silent,
            fetched: [
                fetched_target as u8,
                (fetched_target >> 8) as u8,
                (fetched_target >> 16) as u8,
            ],
            vector_reads: Vec::new(),
            writes: Vec::new(),
            events: Vec::new(),
        };
        let mut state = LlamaState::new();
        state.set_pc(pc as u32);
        state.set_reg(RegName::S, 0x00400);
        state.set_reg(RegName::F, 0x03);
        state.set_reg(RegName::IMR, 0xA5);

        let err = exec
            .execute(0xFE, &mut state, &mut bus)
            .expect_err("volatile vector mismatch must fail after the frame");

        assert_eq!(err, VECTOR_CHANGED_DURING_PREFLIGHT_ERROR);
        assert_eq!(state.pc(), pc as u32);
        assert_eq!(state.get_reg(RegName::S), 0x003FB);
        assert_eq!(state.get_reg(RegName::F), 0x03);
        assert_eq!(state.get_reg(RegName::IMR), 0x00);
        assert_eq!(state.call_depth(), 0);
        assert_eq!(perfetto_last_instr_index(), 17);
        assert_eq!(perfetto_last_pc(), pc as u32);
        assert_eq!(perfetto_instr_context(), None);
        assert_eq!(bus.writes.len(), 6, "five frame bytes plus IMR.IRM clear");
        assert!(
            bus.events[..6].iter().all(|(kind, _)| *kind == "write"),
            "all frame/IMR writes must precede the vector fetch"
        );
        assert_eq!(
            bus.vector_reads,
            vec![
                INTERRUPT_VECTOR_ADDR,
                INTERRUPT_VECTOR_ADDR + 1,
                INTERRUPT_VECTOR_ADDR + 2
            ],
            "vector must have exactly one architectural byte fetch"
        );
        reset_perf_counters();
    }

    #[test]
    fn untraced_executor_construction_and_execution_preserve_trace_clock() {
        let _perfetto_lock = crate::perfetto::perfetto_test_guard();
        reset_perf_counters();
        set_perf_instr_counter(17);
        let mut exec = LlamaExecutor::new();
        let mut state = LlamaState::new();
        let mut bus = MemBus::with_size(1);

        exec.execute(0x00, &mut state, &mut bus)
            .expect("untraced NOP");

        assert_eq!(perfetto_last_instr_index(), 17);
        reset_perf_counters();
    }

    #[test]
    fn add_reg_imm_executes() {
        // Program: 0x40 (ADD A, imm8) imm=0x05
        let mut bus = MemBus::with_size(4);
        bus.mem[0] = 0x40;
        bus.mem[1] = 0x05;
        let mut state = LlamaState::new();
        state.set_reg(RegName::A, 1);
        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0x40, &mut state, &mut bus).unwrap();
        assert_eq!(len, 2);
        assert_eq!(state.get_reg(RegName::A), 6);
        assert_eq!(state.pc(), 2);
    }

    #[test]
    fn exp_exchanges_all_24_bits_and_preserves_flags() {
        let mut bus = MemBus::with_size(0x200);
        let pc = 0x100;
        bus.mem[pc] = 0xC2;
        bus.mem[pc + 1] = 0x20;
        bus.mem[pc + 2] = 0x30;
        bus.mem[0x20..0x23].copy_from_slice(&[0x11, 0x22, 0xA3]);
        bus.mem[0x30..0x33].copy_from_slice(&[0x44, 0x55, 0xB6]);
        let mut state = LlamaState::new();
        state.set_pc(pc as u32);
        state.set_reg(RegName::F, 0x03);
        let mut exec = LlamaExecutor::new();

        let len = exec.execute(0xC2, &mut state, &mut bus).unwrap();

        assert_eq!(len, 3);
        assert_eq!(state.pc(), pc as u32 + 3);
        assert_eq!(&bus.mem[0x20..0x23], &[0x44, 0x55, 0xB6]);
        assert_eq!(&bus.mem[0x30..0x33], &[0x11, 0x22, 0xA3]);
        assert_eq!(state.get_reg(RegName::F), 0x03);
    }

    #[test]
    fn exp_overlap_matches_pc_e500_pairwise_byte_order() {
        for (first, second, expected) in [
            (0x40, 0x40, [0xA1, 0xB2, 0xC3, 0xD4, 0xE5]),
            (0x41, 0x40, [0xB2, 0xC3, 0xD4, 0xA1, 0xE5]),
            (0x40, 0x41, [0xB2, 0xC3, 0xD4, 0xA1, 0xE5]),
        ] {
            let mut bus = MemBus::with_size(0x200);
            let pc = 0x100;
            bus.mem[pc..pc + 4].copy_from_slice(&[0x32, 0xC2, first, second]);
            bus.mem[0x40..0x45].copy_from_slice(&[0xA1, 0xB2, 0xC3, 0xD4, 0xE5]);
            let mut state = LlamaState::new();
            state.set_pc(pc as u32);
            state.set_reg(RegName::I, 0x5AA5);
            state.set_reg(RegName::F, 0x03);
            let mut exec = LlamaExecutor::new();

            let len = exec.execute(0x32, &mut state, &mut bus).unwrap();

            assert_eq!(len, 4);
            assert_eq!(state.pc(), pc as u32 + 4);
            assert_eq!(&bus.mem[0x40..0x45], &expected);
            assert_eq!(state.get_reg(RegName::I), 0x5AA5);
            assert_eq!(state.get_reg(RegName::F), 0x03);
        }
    }

    #[test]
    fn execute_add_regpair_20bit_carry() {
        // Program: 0x45 (ADD regpair size=3) with selector byte choosing X += Y.
        let mut bus = MemBus::with_size(4);
        bus.mem[0] = 0x45;
        bus.mem[1] = 0x45; // dst=X (4), src=Y (5)
        let mut state = LlamaState::new();
        state.set_reg(RegName::X, 0x0F_FFFF);
        state.set_reg(RegName::Y, 0x000001);
        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0x45, &mut state, &mut bus).unwrap();
        assert_eq!(len, 2);
        assert_eq!(state.get_reg(RegName::X), 0x000000);
        assert_eq!(
            state.get_reg(RegName::FC),
            1,
            "20-bit overflow should set carry"
        );
        assert_eq!(
            state.get_reg(RegName::FZ),
            1,
            "wrapped zero should set zero flag"
        );
        assert_eq!(state.pc(), 2);
    }

    #[test]
    fn add_20bit_destination_accepts_16bit_source_rom_form() {
        // ROM at F2B62 uses 45 52: ADD Y,BA. Register-pair operands are
        // independently selected; the opcode width controls the destination.
        let mut bus = MemBus::with_size(4);
        bus.mem[..2].copy_from_slice(&[0x45, 0x52]);
        let mut state = LlamaState::new();
        state.set_reg(RegName::Y, 0x010000);
        state.set_reg(RegName::BA, 0x2345);
        let mut exec = LlamaExecutor::new();

        let len = exec.execute(0x45, &mut state, &mut bus).unwrap();

        assert_eq!(len, 2);
        assert_eq!(state.get_reg(RegName::Y), 0x012345);
        assert_eq!(state.pc(), 2);
    }

    #[test]
    fn mixed_width_add_sub_register_pairs_follow_rom_selector_classes() {
        let cases = [
            (0x44, 0x30, RegName::I, 0x1000, RegName::A, 0x34, 0x1034),
            (
                0x45,
                0x52,
                RegName::Y,
                0x10000,
                RegName::BA,
                0x2345,
                0x12345,
            ),
            (0x4D, 0x60, RegName::U, 0x10000, RegName::A, 0x34, 0x0FFCC),
            (0x4E, 0x01, RegName::A, 0x80, RegName::IL, 0x01, 0x7F),
        ];

        for (opcode, selector, dst, dst_value, src, src_value, expected) in cases {
            let mut bus = MemBus::with_size(4);
            bus.mem[..2].copy_from_slice(&[opcode, selector]);
            let mut state = LlamaState::new();
            state.set_reg(dst, dst_value);
            state.set_reg(src, src_value);
            let mut exec = LlamaExecutor::new();

            exec.execute(opcode, &mut state, &mut bus).unwrap();

            assert_eq!(state.get_reg(dst), expected, "opcode {opcode:02X}");
        }
    }

    #[test]
    fn pmdf_uses_wrapping_binary_add_and_preserves_flags() {
        let mut bus = MemBus::with_size(INTERNAL_MEMORY_START as usize + 0x100);
        bus.mem[..3].copy_from_slice(&[0x47, 0x10, 0xF5]);
        let data_index = MemBus::translate(INTERNAL_MEMORY_START + 0x10);
        bus.mem[data_index] = 0x20;
        let mut state = LlamaState::new();
        state.set_reg(RegName::FC, 1);
        state.set_reg(RegName::FZ, 1);
        let mut exec = LlamaExecutor::new();

        let len = exec.execute(0x47, &mut state, &mut bus).unwrap();

        assert_eq!(len, 3);
        assert_eq!(bus.mem[data_index], 0x15);
        assert_eq!(state.get_reg(RegName::FC), 1);
        assert_eq!(state.get_reg(RegName::FZ), 1);
    }

    #[test]
    fn execute_inc_reg3_x_wraps_20bit() {
        // Program: 0x6C (INC reg3), selector=4 => X.
        let mut bus = MemBus::with_size(4);
        bus.mem[0] = 0x6C;
        bus.mem[1] = 0x04;
        let mut state = LlamaState::new();
        state.set_reg(RegName::X, 0x0F_FFFF);
        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0x6C, &mut state, &mut bus).unwrap();
        assert_eq!(len, 2);
        assert_eq!(state.get_reg(RegName::X), 0x000000);
        assert_eq!(state.get_reg(RegName::FZ), 1);
        assert_eq!(state.pc(), 2);
    }

    #[test]
    fn mv_reg_imm_executes() {
        // Program: 0x08 (MV A, imm8) imm=0xAA
        let mut bus = MemBus::with_size(4);
        bus.mem[0] = 0x08;
        bus.mem[1] = 0xAA;
        let mut state = LlamaState::new();
        state.set_reg(RegName::A, 0);
        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0x08, &mut state, &mut bus).unwrap();
        assert_eq!(len, 2);
        assert_eq!(state.get_reg(RegName::A), 0xAA);
        assert_eq!(state.pc(), 2);
        assert_eq!(state.get_reg(RegName::FZ), 0);
    }

    #[test]
    fn mv_reg_imem_executes() {
        // Program: 0x80 (MV A, IMem8) addr=0x10
        let mut bus = MemBus::with_size(0x40);
        bus.mem[0] = 0x80;
        bus.mem[1] = 0x10;
        bus.mem[0x10] = 0x22;
        let mut state = LlamaState::new();
        state.set_reg(RegName::A, 0);
        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0x80, &mut state, &mut bus).unwrap();
        assert_eq!(len, 2);
        assert_eq!(state.get_reg(RegName::A), 0x22);
        assert_eq!(state.pc(), 2);
    }

    #[test]
    fn mv_imem_reg_executes() {
        // Program: 0xA0 (MV IMem8, A) addr=0x20
        let mut bus = MemBus::with_size(0x40);
        bus.mem[0] = 0xA0;
        bus.mem[1] = 0x20;
        let mut state = LlamaState::new();
        state.set_reg(RegName::A, 0x33);
        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0xA0, &mut state, &mut bus).unwrap();
        assert_eq!(len, 2);
        assert_eq!(bus.mem[0x20], 0x33);
        assert_eq!(state.pc(), 2);
    }

    #[test]
    fn cmpp_imem_reg_compares_raw24_memory_with_zero_extended_register() {
        // D7 04 10: CMPP (BP+10), X. The memory image is 0xF00080,
        // while X is 0x000080. Hardware compares the full memory triple with
        // the zero-extended 20-bit register, so the operands are unequal.
        let mut bus = MemBus {
            mem: vec![0xFF; 0x40],
            writes: Vec::new(),
        };
        bus.mem[..3].copy_from_slice(&[0xD7, 0x04, 0x10]);
        bus.mem[0x10..0x13].copy_from_slice(&[0x80, 0x00, 0xF0]);

        let mut state = LlamaState::new();
        state.set_reg(RegName::X, 0x000080);
        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0xD7, &mut state, &mut bus).unwrap();

        assert_eq!(
            len, 3,
            "encoding without PRE consumes opcode + reg + 1-byte IMEM slot"
        );
        assert_eq!(state.pc(), len as u32);
        assert_eq!(
            state.get_reg(RegName::FC) & 1,
            0,
            "the larger raw memory image must not borrow"
        );
        assert_eq!(
            state.get_reg(RegName::FZ) & 1,
            0,
            "raw 24-bit values differ"
        );
    }

    #[test]
    fn cmpp_imem_imem_keeps_all_24_bits() {
        // C7 10 20: CMPP (BP+10), (BP+20). Unlike D7, C7 compares raw
        // three-byte images, so 0xF00080 is not equal to 0x000080.
        let mut bus = MemBus {
            mem: vec![0; 0x40],
            writes: Vec::new(),
        };
        bus.mem[..3].copy_from_slice(&[0xC7, 0x10, 0x20]);
        bus.mem[0x10..0x13].copy_from_slice(&[0x80, 0x00, 0xF0]);
        bus.mem[0x20..0x23].copy_from_slice(&[0x80, 0x00, 0x00]);

        let mut state = LlamaState::new();
        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0xC7, &mut state, &mut bus).unwrap();

        assert_eq!(len, 3);
        assert_eq!(state.pc(), 3);
        assert_eq!(state.get_reg(RegName::FC) & 1, 0);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 0, "24-bit values differ");
    }

    #[test]
    fn mv_imem_imm_executes() {
        // Program: 0xCC (MV IMem8, imm8) addr=0x21, val=0x44
        let mut bus = MemBus::with_size(0x40);
        bus.mem[0] = 0xCC;
        bus.mem[1] = 0x21;
        bus.mem[2] = 0x44;
        let mut state = LlamaState::new();
        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0xCC, &mut state, &mut bus).unwrap();
        assert_eq!(len, 3);
        assert_eq!(bus.mem[0x21], 0x44);
        assert_eq!(state.pc(), 3);
    }

    #[test]
    fn add_reg_imem_executes() {
        // Program: 0x42 (ADD A, IMem8) addr=0x30, mem=0x05, A=1
        let mut bus = MemBus::with_size(0x40);
        bus.mem[0] = 0x42;
        bus.mem[1] = 0x30;
        bus.mem[0x30] = 0x05;
        let mut state = LlamaState::new();
        state.set_reg(RegName::A, 1);
        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0x42, &mut state, &mut bus).unwrap();
        assert_eq!(len, 2);
        assert_eq!(state.get_reg(RegName::A), 6);
        assert_eq!(state.pc(), 2);
    }

    #[test]
    fn adc_reg_imm_sets_carry() {
        // Program: 0x50 (ADC A, imm8) imm=0x01 with C=1, A=0xFF => result 0x01, carry out
        let mut bus = MemBus::with_size(4);
        bus.mem[0] = 0x50;
        bus.mem[1] = 0x01;
        let mut state = LlamaState::new();
        state.set_reg(RegName::A, 0xFF);
        state.set_reg(RegName::FC, 1);
        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0x50, &mut state, &mut bus).unwrap();
        assert_eq!(len, 2);
        assert_eq!(state.get_reg(RegName::A), 0x01);
        assert_eq!(state.get_reg(RegName::FZ), 0);
    }

    #[test]
    fn ir_stacks_imr_f_pc_little_endian() {
        let mut exec = LlamaExecutor::new();
        let mut state = LlamaState::new();
        let mut bus = MemBus::with_size((INTERNAL_MEMORY_START as usize) + 0x200);

        // Seed IMR and interrupt vector.
        let imr_saved: u8 = 0xAA;
        bus.store(
            INTERNAL_MEMORY_START + IMEM_IMR_OFFSET,
            8,
            u32::from(imr_saved),
        );
        bus.mem[INTERRUPT_VECTOR_ADDR as usize] = 0x21; // vector low
        bus.mem[INTERRUPT_VECTOR_ADDR as usize + 1] = 0x43; // mid
        bus.mem[INTERRUPT_VECTOR_ADDR as usize + 2] = 0x05; // high -> 0x054321

        assert_eq!(
            bus.load(INTERNAL_MEMORY_START + IMEM_IMR_OFFSET, 8) as u8,
            imr_saved
        );

        let pc_start = 0x001234;
        let f_input: u8 = 0x03;
        let f_saved = f_input;
        let sp_start = 0x0200;
        state.set_pc(pc_start);
        state.set_reg(RegName::F, u32::from(f_input));
        state.set_reg(RegName::S, sp_start);

        let len = exec.execute(0xFE, &mut state, &mut bus).unwrap(); // IR
        assert_eq!(len, 1);

        // The ROM dispatcher examines the opcode at the saved PC and advances
        // the frame after recognizing IR.
        let expected_pc = pc_start & mask_for(RegName::PC);
        let expected_sp = sp_start.wrapping_sub(5) & mask_for(RegName::S);
        assert_eq!(state.get_reg(RegName::S), expected_sp);

        let base = MemBus::translate(expected_sp);
        assert_eq!(
            &bus.mem[base..base + 5],
            &[
                imr_saved,
                f_saved,
                (expected_pc & 0xFF) as u8,
                ((expected_pc >> 8) & 0xFF) as u8,
                ((expected_pc >> 16) & 0xFF) as u8
            ]
        );

        // IMR should be cleared in state/memory after the push.
        assert_eq!(state.get_reg(RegName::IMR), u32::from(imr_saved & 0x7F));
        assert_eq!(
            bus.mem[MemBus::translate(INTERNAL_MEMORY_START + IMEM_IMR_OFFSET)],
            imr_saved & 0x7F
        );
        assert_eq!(state.pc(), 0x054321);
    }

    #[test]
    fn reti_restores_imr_f_pc_little_endian() {
        let mut exec = LlamaExecutor::new();
        let mut state = LlamaState::new();
        let mut bus = MemBus::with_size((INTERNAL_MEMORY_START as usize) + 0x200);

        let sp_start = 0x0200;
        let imr_saved: u8 = 0x00;
        let f_saved: u8 = 0x03;
        let ret_pc: u32 = 0x053412;

        let base = MemBus::translate(sp_start);
        bus.mem[base] = imr_saved;
        bus.mem[base + 1] = f_saved;
        bus.mem[base + 2] = (ret_pc & 0xFF) as u8;
        bus.mem[base + 3] = ((ret_pc >> 8) & 0xFF) as u8;
        bus.mem[base + 4] = ((ret_pc >> 16) & 0xFF) as u8;

        state.set_reg(RegName::S, sp_start);
        state.call_depth_inc();

        let len = exec.execute(0x01, &mut state, &mut bus).unwrap(); // RETI
        assert_eq!(len, 1);
        assert_eq!(
            state.get_reg(RegName::S),
            (sp_start + 5) & mask_for(RegName::S)
        );
        assert_eq!(state.pc(), ret_pc & mask_for(RegName::PC));
        assert_eq!(state.get_reg(RegName::IMR), u32::from(imr_saved));
        assert_eq!(state.get_reg(RegName::F), u32::from(f_saved));
        assert_eq!(
            bus.mem[MemBus::translate(INTERNAL_MEMORY_START + IMEM_IMR_OFFSET)],
            imr_saved
        );
    }

    #[test]
    fn reti_normalizes_stacked_f_to_carry_and_zero() {
        for f_saved in [0x00_u8, 0x03, 0x04, 0xA5, 0xFC, 0xFF] {
            let mut exec = LlamaExecutor::new();
            let mut state = LlamaState::new();
            let mut bus = MemBus::with_size((INTERNAL_MEMORY_START as usize) + 0x200);
            let sp_start = 0x0200;
            let base = MemBus::translate(sp_start);
            bus.mem[base] = 0xA5;
            bus.mem[base + 1] = f_saved;
            bus.mem[base + 2..base + 5].copy_from_slice(&[0x12, 0x34, 0x05]);
            bus.mem[MemBus::translate(INTERNAL_MEMORY_START + IMEM_IMR_OFFSET)] = 0x5A;
            state.set_reg(RegName::S, sp_start);
            state.set_reg(RegName::F, 0x02);
            state.call_depth_inc();

            exec.execute(0x01, &mut state, &mut bus).unwrap();

            assert_eq!(state.get_reg(RegName::S), sp_start + 5);
            assert_eq!(state.get_reg(RegName::F), u32::from(f_saved & 0x03));
            assert_eq!(state.pc(), 0x53412);
            assert_eq!(state.call_depth(), 0);
            assert_eq!(
                bus.mem[MemBus::translate(INTERNAL_MEMORY_START + IMEM_IMR_OFFSET)],
                0xA5
            );
        }
    }

    #[test]
    fn popu_f_normalizes_upper_bits_to_carry_and_zero() {
        for f_saved in [0x04_u8, 0x80, 0xA4, 0xFC, 0xFF] {
            let mut exec = LlamaExecutor::new();
            let mut state = LlamaState::new();
            let mut bus = MemBus::with_size(0x400);
            bus.mem[0] = 0x3E;
            bus.mem[0x200] = f_saved;
            state.set_reg(RegName::U, 0x200);
            state.set_reg(RegName::F, 0x01);

            exec.execute(0x3E, &mut state, &mut bus).unwrap();

            assert_eq!(state.get_reg(RegName::U), 0x201);
            assert_eq!(state.get_reg(RegName::F), u32::from(f_saved & 0x03));
            assert_eq!(state.pc(), 1);
        }
    }

    #[test]
    fn pops_f_normalizes_upper_bits_to_carry_and_zero() {
        for f_saved in [0x00_u8, 0x03, 0x04, 0xA5, 0xFC, 0xFF] {
            let mut exec = LlamaExecutor::new();
            let mut state = LlamaState::new();
            let mut bus = MemBus::with_size(0x400);
            bus.mem[0] = 0x5F;
            bus.mem[0x200] = f_saved;
            state.set_reg(RegName::S, 0x200);
            state.set_reg(RegName::F, 0x01);

            exec.execute(0x5F, &mut state, &mut bus).unwrap();

            assert_eq!(state.get_reg(RegName::S), 0x201);
            assert_eq!(state.get_reg(RegName::F), u32::from(f_saved & 0x03));
            assert_eq!(state.pc(), 1);
        }
    }

    #[test]
    fn mvl_predec_updates_pointer_without_wrap() {
        // Opcode 0xE3: MVL IMem8, EMemReg (mode byte), uses pre-dec when reg byte upper nibble is 0x3.
        let mut exec = LlamaExecutor::new();
        let mut state = LlamaState::new();
        let mut bus = MemBus::with_size((INTERNAL_MEMORY_START as usize) + 0x400);

        // Program: [0]=0xE3 (MVL), [1]=0x36 (pre-dec, reg=U), [2]=0x10 (IMEM offset)
        bus.mem[0x0000] = 0xE3;
        bus.mem[0x0001] = 0x36; // mode=PreDec, reg selector=6 (U)
        bus.mem[0x0002] = 0x10; // IMEM destination offset

        // Source byte resides at U-1 after pre-dec.
        state.set_reg(RegName::U, 0x0030);
        bus.mem[0x002F] = 0xAB;

        // Set transfer length to 1.
        state.set_reg(RegName::I, 1);

        let len = exec.execute(0xE3, &mut state, &mut bus).unwrap();
        assert_eq!(len, 3);
        assert_eq!(state.get_reg(RegName::PC), 3);

        // Destination should receive the byte.
        let dst_addr = MemBus::translate(INTERNAL_MEMORY_START + 0x10);
        assert_eq!(bus.mem[dst_addr], 0xAB);

        // U should pre-decrement by 1, not wrap to a huge value.
        assert_eq!(state.get_reg(RegName::U), 0x002F);
    }

    #[test]
    fn execute_mvl_emem_reg_imem_updates_pointer_by_length() {
        // Opcode 0xE3: MVL IMem8, EMemReg(post-inc X), with I=3 should advance X by 3.
        let mut exec = LlamaExecutor::new();
        let mut state = LlamaState::new();
        let mut bus = MemBus::with_size((INTERNAL_MEMORY_START as usize) + 0x400);

        bus.mem[0x0000] = 0xE3;
        bus.mem[0x0001] = 0x24; // mode=PostInc, reg=X
        bus.mem[0x0002] = 0x20; // IMEM destination offset

        state.set_reg(RegName::X, 0x0030);
        bus.mem[0x0030] = 0x11;
        bus.mem[0x0031] = 0x22;
        bus.mem[0x0032] = 0x33;
        state.set_reg(RegName::I, 3);

        let len = exec.execute(0xE3, &mut state, &mut bus).unwrap();
        assert_eq!(len, 3);
        assert_eq!(state.get_reg(RegName::PC), 3);
        assert_eq!(bus.mem[0x20], 0x11);
        assert_eq!(bus.mem[0x21], 0x22);
        assert_eq!(bus.mem[0x22], 0x33);
        assert_eq!(state.get_reg(RegName::X), 0x0033);
        assert_eq!(state.get_reg(RegName::I), 0);
    }

    #[test]
    fn mvl_overlap_emits_exactly_i_ordered_writes() {
        let mut bus = MemBus::with_size(0x200);
        bus.mem[..3].copy_from_slice(&[0xCB, 0x50, 0x51]);
        bus.mem[0x51] = 0xAA;
        bus.mem[0x52] = 0xBB;
        let mut state = LlamaState::new();
        state.set_reg(RegName::I, 2);
        let mut exec = LlamaExecutor::new();

        exec.execute(0xCB, &mut state, &mut bus).unwrap();

        assert_eq!(bus.mem[0x50], 0xAA);
        assert_eq!(bus.mem[0x51], 0xBB);
        assert_eq!(
            bus.writes,
            vec![
                (INTERNAL_MEMORY_START + 0x50, 8, 0xAA),
                (INTERNAL_MEMORY_START + 0x51, 8, 0xBB),
            ]
        );
        assert_eq!(state.get_reg(RegName::I), 0);
    }

    #[test]
    fn hw002_mvl_zero_count_predec_forms_run_full_count() {
        for opcode in [0xE3, 0xEB] {
            let code_base = 0x1000usize;
            let mut bus = MemBus::with_size(0x2000);
            bus.mem[code_base..code_base + 3].copy_from_slice(&[opcode, 0x37, 0x20]);
            let mut state = LlamaState::new();
            state.set_pc(code_base as u32);
            state.set_reg(RegName::I, 0);
            state.set_reg(RegName::S, 0x80000);
            state.set_reg(RegName::FC, 0);
            state.set_reg(RegName::FZ, 1);
            let mut exec = LlamaExecutor::new();

            let len = exec.execute(opcode, &mut state, &mut bus).unwrap();

            assert_eq!(len, 3);
            assert_eq!(state.pc(), code_base as u32 + 3);
            assert_eq!(state.get_reg(RegName::I), 0);
            assert_eq!(state.get_reg(RegName::S), 0x70000);
            assert_eq!(state.get_reg(RegName::FC), 0);
            assert_eq!(state.get_reg(RegName::FZ), 1);
            assert_eq!(bus.writes.len(), 0x10000);
        }
    }

    #[test]
    fn sbc_reg_imm_sets_carry_on_no_borrow() {
        // Program: 0x58 (SBC A, imm8) imm=0x01 with C=1, A=0x03 => result 0x01, carry stays set
        let mut bus = MemBus::with_size(4);
        bus.mem[0] = 0x58;
        bus.mem[1] = 0x01;
        let mut state = LlamaState::new();
        state.set_reg(RegName::A, 0x03);
        state.set_reg(RegName::FC, 1);
        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0x58, &mut state, &mut bus).unwrap();
        assert_eq!(len, 2);
        assert_eq!(state.get_reg(RegName::A), 0x01);
    }

    #[test]
    fn adcl_multibyte_uses_incoming_carry() {
        // Program: 0x54 (ADCL (m),(n)) with I=2, carry propagates across bytes
        let mut bus = MemBus::with_size(0x200);
        bus.mem[0] = 0x54;
        bus.mem[1] = 0x10; // dst
        bus.mem[2] = 0x20; // src
        bus.mem[0x10] = 0xFF;
        bus.mem[0x11] = 0x00;
        bus.mem[0x20] = 0x01;
        bus.mem[0x21] = 0x02;
        let mut state = LlamaState::new();
        state.set_reg(RegName::I, 2);
        state.set_reg(RegName::FC, 1);
        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0x54, &mut state, &mut bus).unwrap();
        assert_eq!(len, 3);
        assert_eq!(bus.mem[0x10], 0x01);
        assert_eq!(bus.mem[0x11], 0x03);
        assert_eq!(state.get_reg(RegName::FC), 0);
        assert_eq!(state.get_reg(RegName::FZ), 0);
        assert_eq!(state.get_reg(RegName::I), 0);
        assert_eq!(state.pc(), 3);
    }

    #[test]
    fn stacked_pre_prefixes_reject_second_prefix() {
        // A second PRE is not an instruction operand; accepting it would hide
        // stream corruption and disagree with the canonical Python decoder.
        let mut bus = MemBus::with_size(8);
        bus.mem[0] = 0x32; // PRE
        bus.mem[1] = 0x32; // PRE
        bus.mem[2] = 0x02; // JP abs16
        bus.mem[3] = 0x78;
        bus.mem[4] = 0x9A;
        let mut state = LlamaState::new();
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        assert!(exec.execute(0x32, &mut state, &mut bus).is_err());
        assert_eq!(state.pc(), 0);
    }

    #[test]
    fn stacked_pre_prefix_runs_fail_closed() {
        // Prefix runs must not be treated as an unbounded skip mechanism.
        let mut bus = MemBus::with_size(10);
        bus.mem[0] = 0x32;
        bus.mem[1] = 0x32;
        bus.mem[2] = 0x32;
        bus.mem[3] = 0x32;
        bus.mem[4] = 0x02; // JP abs16
        bus.mem[5] = 0x34;
        bus.mem[6] = 0x12;
        let mut state = LlamaState::new();
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        assert!(exec.execute(0x32, &mut state, &mut bus).is_err());
        assert_eq!(state.pc(), 0);
    }

    #[test]
    fn dadl_wraps_internal_addresses_and_updates_pointers() {
        // Opcode 0xC4: DADL (m),(n) with IMEM offsets. Length=2, should wrap within IMEM space.
        let mut bus = MemBus::with_size(0x400);
        // Program layout
        bus.mem[0] = 0xC4;
        bus.mem[1] = 0x80; // dst offset 0x80 (avoid overlapping program)
        bus.mem[2] = 0x82; // src offset 0x82 (will wrap to 0x81)
                           // Seed IMEM bytes
        bus.mem[0x80] = 0x00; // dst low
        bus.mem[0x81] = 0x07; // src after wrap
        bus.mem[0x82] = 0x05; // src first iteration
        bus.mem[0x7F] = 0x00; // dst high (will be touched after wrap)

        let mut state = LlamaState::new();
        state.set_reg(RegName::I, 2); // two bytes
        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0xC4, &mut state, &mut bus).unwrap();
        assert_eq!(len, 3);
        // First byte: 0x00 + 0x05 = 0x05; second wraps dst to 0x7F, src to 0x81 (0x07).
        assert_eq!(bus.mem[0x80], 0x05);
        assert_eq!(
            bus.mem[0x7F], 0x07,
            "second iteration should write wrapped dst"
        );
        // IMEM wrapping should keep addresses inside 0x100-byte window.
        assert_eq!(state.get_reg(RegName::I), 0);
    }

    #[test]
    fn dadl_reg_source_only_first_byte() {
        // Opcode 0xC5: DADL (m),A should consume A once, then use zero for remaining bytes.
        let mut bus = MemBus::with_size(0x200);
        bus.mem[0] = 0xC5;
        bus.mem[1] = 0x10; // destination starts at IMEM[0x10], then decrements
        bus.mem[0x10] = 0x00;
        bus.mem[0x0F] = 0x00;

        let mut state = LlamaState::new();
        state.set_reg(RegName::A, 0x01);
        state.set_reg(RegName::I, 2);
        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0xC5, &mut state, &mut bus).unwrap();
        assert_eq!(len, 2);
        assert_eq!(bus.mem[0x10], 0x01);
        assert_eq!(
            bus.mem[0x0F], 0x00,
            "second byte should add zero, not reuse A"
        );
        assert_eq!(state.get_reg(RegName::I), 0);
        assert_eq!(state.pc(), 2);
    }

    #[test]
    fn adcl_ememreg_side_effect_updates_pointers() {
        // Use a synthetic ADCL entry with EMemReg operands that carry pre/post side effects.
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0x54; // opcode (ignored for operand shapes)
        bus.mem[1] = 0x24; // dst: X post-inc (raw_mode=2, reg=4)
        bus.mem[2] = 0x35; // src: Y pre-dec (raw_mode=3, reg=5)
        let mut state = LlamaState::new();
        state.set_reg(RegName::X, 0x10);
        state.set_reg(RegName::Y, 0x20);
        state.set_reg(RegName::I, 2); // length
        let mut exec = LlamaExecutor::new();
        let entry = OpcodeEntry {
            opcode: 0x54,
            kind: InstrKind::Adc,
            name: "ADCL",
            cond: None,
            ops_reversed: None,
            operands: &[OperandKind::EMemRegWidth(1), OperandKind::EMemRegWidth(1)],
        };
        let len = exec
            .execute_with(0x54, &entry, &mut state, &mut bus, None, None, 0, 0, None)
            .unwrap();
        assert_eq!(len, 3);
        // Post-inc X advances by length, pre-dec Y decrements by length.
        assert_eq!(state.get_reg(RegName::X), 0x12);
        assert_eq!(state.get_reg(RegName::Y), 0x1E);
    }

    #[test]
    fn sbcl_multibyte_propagates_borrow_forward() {
        // Program: 0x5C (SBCL (m),(n)) with I=2, borrow chains across bytes
        let mut bus = MemBus::with_size(0x200);
        bus.mem[0] = 0x5C;
        bus.mem[1] = 0x10; // dst
        bus.mem[2] = 0x20; // src
        bus.mem[0x10] = 0x00;
        bus.mem[0x11] = 0x02;
        bus.mem[0x20] = 0x01;
        bus.mem[0x21] = 0x01;
        let mut state = LlamaState::new();
        state.set_reg(RegName::I, 2);
        state.set_reg(RegName::FC, 1);
        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0x5C, &mut state, &mut bus).unwrap();
        assert_eq!(len, 3);
        assert_eq!(bus.mem[0x10], 0xFE);
        assert_eq!(bus.mem[0x11], 0x00);
        assert_eq!(state.get_reg(RegName::FC), 0);
        assert_eq!(state.get_reg(RegName::FZ), 0);
        assert_eq!(state.get_reg(RegName::I), 0);
        assert_eq!(state.pc(), 3);
    }

    #[test]
    fn swap_nibbles_updates_zero_flag() {
        let mut bus = MemBus::with_size(2);
        bus.mem[0] = 0xEE;
        let mut state = LlamaState::new();
        state.set_reg(RegName::A, 0x3C);
        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0xEE, &mut state, &mut bus).unwrap();
        assert_eq!(len, 1);
        assert_eq!(state.get_reg(RegName::A), 0xC3);
        assert_eq!(state.get_reg(RegName::FZ), 0);
        assert_eq!(state.pc(), 1);
    }

    #[test]
    fn ir_stack_pc_is_big_endian_and_reti_matches() {
        // Verify IR pushes PC big-endian (high->low) and RETI reassembles the same order.
        let mut bus = MemBus::with_size(0x80);
        let mut state = LlamaState::new();
        state.set_reg(RegName::S, 0x20);
        let pc = 0x0ABCDE;

        LlamaExecutor::push_stack(&mut state, &mut bus, RegName::S, pc, 24, true);
        // Push F and IMR (single-byte, endian-neutral).
        LlamaExecutor::push_stack(&mut state, &mut bus, RegName::S, 0x03, 8, false);
        LlamaExecutor::push_stack(&mut state, &mut bus, RegName::S, 0xC3, 8, false);

        let sp = state.get_reg(RegName::S) as usize;
        assert_eq!(sp, 0x20 - 5);
        // Layout from low to high addresses after pushes: IMR, F,
        // PC[19:16] (in a byte), PC[15:8], PC[7:0].
        assert_eq!(bus.mem[sp], 0xC3); // IMR
        assert_eq!(bus.mem[sp + 1], 0x03); // modeled F image
        assert_eq!(bus.mem[sp + 2], 0x0A);
        assert_eq!(bus.mem[sp + 3], 0xBC);
        assert_eq!(bus.mem[sp + 4], 0xDE);

        // Simulate RETI: pop IMR, F, then PC big-endian.
        let mut sp_iter = sp as u32;
        let imr = bus.load(sp_iter, 8) & 0xFF;
        sp_iter = sp_iter.wrapping_add(1);
        let f = bus.load(sp_iter, 8) & 0xFF;
        sp_iter = sp_iter.wrapping_add(1);
        let pc_hi = bus.load(sp_iter, 8) & 0xFF;
        let pc_mid = bus.load(sp_iter + 1, 8) & 0xFF;
        let pc_lo = bus.load(sp_iter + 2, 8) & 0xFF;
        let ret_pc = ((pc_hi << 16) | (pc_mid << 8) | pc_lo) & mask_for(RegName::PC);
        sp_iter = sp_iter.wrapping_add(3);

        assert_eq!(imr, 0xC3);
        assert_eq!(f, 0x03);
        assert_eq!(ret_pc, pc & mask_for(RegName::PC));
        assert_eq!(sp_iter as usize, 0x20);
    }

    #[test]
    fn call_stack_is_little_endian_with_measured_descending_write_order() {
        // The final frame is little-endian at S, but PC-E500 hardware writes
        // high byte first while pre-decrementing S once per byte.
        let mut bus = MemBus::with_size(0x80);
        let mut state = LlamaState::new();
        state.set_reg(RegName::S, 0x40);
        let ret = 0x012345;

        LlamaExecutor::push_stack(&mut state, &mut bus, RegName::S, ret, 24, false);
        let sp = state.get_reg(RegName::S) as usize;
        assert_eq!(sp, 0x40 - 3);
        assert_eq!(bus.mem[sp], 0x45); // low byte first
        assert_eq!(bus.mem[sp + 1], 0x23);
        assert_eq!(bus.mem[sp + 2], 0x01); // high byte last
        assert_eq!(
            bus.writes,
            vec![(0x3F, 8, 0x01), (0x3E, 8, 0x23), (0x3D, 8, 0x45)]
        );

        let popped = LlamaExecutor::pop_stack(&mut state, &mut bus, RegName::S, 24, false);
        assert_eq!(popped, ret);
        assert_eq!(state.get_reg(RegName::S) as usize, 0x40);
    }

    #[test]
    fn hw009_far_control_masks_every_encoded_upper_nibble() {
        for opcode in [0x03, 0x05] {
            for upper_nibble in 0..=0x0F {
                let mut bus = MemBus::with_size(0x200);
                bus.mem[..4].copy_from_slice(&[opcode, 0x80, 0x00, upper_nibble << 4]);
                let mut state = LlamaState::new();
                state.set_reg(RegName::S, 0x00190);
                let mut exec = LlamaExecutor::new();

                let len = exec
                    .execute(opcode, &mut state, &mut bus)
                    .expect("all hardware-measured far-control aliases must execute");

                assert_eq!(len, 4, "opcode {opcode:02X}, upper {upper_nibble:X}");
                assert_eq!(
                    state.pc(),
                    0x80,
                    "opcode {opcode:02X}, upper {upper_nibble:X}"
                );
                if opcode == 0x03 {
                    assert_eq!(state.get_reg(RegName::S), 0x190);
                    assert!(bus.writes.is_empty());
                } else {
                    assert_eq!(state.get_reg(RegName::S), 0x18D);
                    assert_eq!(
                        bus.writes,
                        vec![(0x18F, 8, 0x00), (0x18E, 8, 0x00), (0x18D, 8, 0x04)]
                    );
                    assert_eq!(state.call_depth(), 1);
                }
            }
        }
    }

    #[test]
    fn register_immediates_discard_the_encoded_upper_nibble() {
        for (opcode, register) in [
            (0x0C, RegName::X),
            (0x0D, RegName::Y),
            (0x0E, RegName::U),
            (0x0F, RegName::S),
        ] {
            let mut bus = MemBus::with_size(0x200);
            bus.mem[..4].copy_from_slice(&[opcode, 0xA5, 0x5A, 0x3C]);
            let mut state = LlamaState::new();
            state.set_reg(RegName::S, 0x00190);
            let mut exec = LlamaExecutor::new();

            let len = exec
                .execute(opcode, &mut state, &mut bus)
                .expect("architectural 20-bit register immediate must execute");

            assert_eq!(len, 4, "opcode {opcode:02X}");
            assert_eq!(state.get_reg(register), 0x0C_5AA5, "opcode {opcode:02X}");
            assert_eq!(state.pc(), 4, "opcode {opcode:02X}");
        }
    }

    #[test]
    fn hw009_88_8f_direct_reads_discard_address_upper_nibble() {
        for (opcode, register, expected) in [
            (0x88, RegName::A, 0xA5),
            (0x89, RegName::IL, 0xA5),
            (0x8A, RegName::BA, 0x5AA5),
            (0x8B, RegName::I, 0x5AA5),
            (0x8C, RegName::X, 0xC5AA5),
            (0x8D, RegName::Y, 0xC5AA5),
            (0x8E, RegName::U, 0xC5AA5),
            (0x8F, RegName::S, 0xC5AA5),
        ] {
            for high in [0x81, 0x01] {
                let mut bus = MemBus::with_size(0x1_0200);
                bus.mem[..5].copy_from_slice(&[opcode, 0xF0, 0x01, high, 0x00]);
                bus.mem[0x1_01F0..0x1_01F3].copy_from_slice(&[0xA5, 0x5A, 0x3C]);
                let mut state = LlamaState::new();
                state.set_reg(RegName::F, 0x03);
                let mut exec = LlamaExecutor::new();

                exec.validate_before_scheduling(opcode, &state, &mut bus)
                    .expect("HW-009 88-8F address alias must pass preflight");
                let len = exec
                    .execute(opcode, &mut state, &mut bus)
                    .expect("HW-009 88-8F address alias must execute");

                assert_eq!(len, 4, "opcode {opcode:02X}, high {high:02X}");
                assert_eq!(
                    state.get_reg(register),
                    expected,
                    "opcode {opcode:02X}, high {high:02X}"
                );
                assert_eq!(state.get_reg(RegName::F), 0x03, "opcode {opcode:02X}");
                assert_eq!(state.pc(), 4, "opcode {opcode:02X}, high {high:02X}");

                exec.execute(0x00, &mut state, &mut bus)
                    .expect("following NOP must execute");
                assert_eq!(state.pc(), 5, "opcode {opcode:02X}, high {high:02X}");
            }
        }
    }

    #[test]
    fn hw009_absolute_byte_ops_discard_address_upper_nibble() {
        for (opcode, immediate, writes) in [
            (0x62, 0x00, false),
            (0x66, 0xFF, false),
            (0x6A, 0x00, true),
            (0x72, 0xFF, true),
            (0x7A, 0x00, true),
        ] {
            for high in [0x84, 0x04] {
                let mut bus = MemBus::with_size(0x4_0700);
                bus.mem[..6].copy_from_slice(&[opcode, 0xD0, 0x06, high, immediate, 0x00]);
                bus.mem[0x4_06D0] = 0xA5;
                let mut state = LlamaState::new();
                state.set_reg(RegName::F, 0x03);
                let mut exec = LlamaExecutor::new();

                exec.validate_before_scheduling(opcode, &state, &mut bus)
                    .expect("HW-009 absolute-byte address alias must pass preflight");
                let len = exec
                    .execute(opcode, &mut state, &mut bus)
                    .expect("HW-009 absolute-byte address alias must execute");

                assert_eq!(len, 5, "opcode {opcode:02X}, high {high:02X}");
                assert_eq!(state.pc(), 5, "opcode {opcode:02X}, high {high:02X}");
                assert_eq!(bus.mem[0x4_06D0], 0xA5, "opcode {opcode:02X}");
                if writes {
                    assert_eq!(bus.writes, vec![(0x4_06D0, 8, 0xA5)]);
                } else {
                    assert!(bus.writes.is_empty(), "opcode {opcode:02X}");
                }

                exec.execute(0x00, &mut state, &mut bus)
                    .expect("following NOP must execute");
                assert_eq!(state.pc(), 6, "opcode {opcode:02X}, high {high:02X}");
            }
        }
    }

    #[test]
    fn hw009_a8_af_direct_writes_discard_address_upper_nibble() {
        for (opcode, register, value, width) in [
            (0xA8, RegName::A, 0xA5, 1_u32),
            (0xA9, RegName::IL, 0xA5, 1),
            (0xAA, RegName::BA, 0x5AA5, 2),
            (0xAB, RegName::I, 0x5AA5, 2),
            (0xAC, RegName::X, 0xC3C5A, 3),
            (0xAD, RegName::Y, 0xC3C5A, 3),
            (0xAE, RegName::U, 0xC3C5A, 3),
            (0xAF, RegName::S, 0xC3C5A, 3),
        ] {
            for high in [0x84, 0x04] {
                let mut bus = MemBus::with_size(0x4_0700);
                bus.mem[..5].copy_from_slice(&[opcode, 0xD0, 0x06, high, 0x00]);
                let mut state = LlamaState::new();
                state.set_reg(register, value);
                let mut exec = LlamaExecutor::new();

                exec.validate_before_scheduling(opcode, &state, &mut bus)
                    .expect("HW-009 A8-AF address alias must pass preflight");
                let len = exec
                    .execute(opcode, &mut state, &mut bus)
                    .expect("HW-009 A8-AF address alias must execute");

                assert_eq!(len, 4, "opcode {opcode:02X}, high {high:02X}");
                assert_eq!(state.pc(), 4, "opcode {opcode:02X}, high {high:02X}");
                let expected: Vec<_> = (0..width)
                    .map(|index| (0x4_06D0 + index, 8, (value >> (8 * index)) & 0xFF))
                    .collect();
                assert_eq!(bus.writes, expected, "opcode {opcode:02X}, high {high:02X}");

                exec.execute(0x00, &mut state, &mut bus)
                    .expect("following NOP must execute");
                assert_eq!(state.pc(), 5, "opcode {opcode:02X}, high {high:02X}");
            }
        }
    }

    #[test]
    fn hw009_absolute_transfers_discard_address_upper_nibble() {
        let sentinel = [0xA5, 0x5A, 0x3C];
        for (opcode, transfer_len, counted, to_external) in [
            (0xD0, 1_usize, false, false),
            (0xD1, 2, false, false),
            (0xD2, 3, false, false),
            (0xD3, 3, true, false),
            (0xD8, 1, false, true),
            (0xD9, 2, false, true),
            (0xDA, 3, false, true),
            (0xDB, 3, true, true),
        ] {
            for high in if to_external {
                [0x84, 0x04]
            } else {
                [0x81, 0x01]
            } {
                let mut bus = MemBus::with_size(0x4_0700);
                if to_external {
                    bus.mem[..6].copy_from_slice(&[opcode, 0xD0, 0x06, high, 0x60, 0x00]);
                    bus.mem[0x60..0x63].copy_from_slice(&sentinel);
                } else {
                    bus.mem[..6].copy_from_slice(&[opcode, 0x60, 0xF0, 0x01, high, 0x00]);
                    bus.mem[0x1_01F0..0x1_01F3].copy_from_slice(&sentinel);
                    bus.mem[0x60..0x63].copy_from_slice(&[0x11, 0x22, 0x33]);
                }
                bus.mem[IMEM_BP_OFFSET as usize] = 0;
                let mut state = LlamaState::new();
                state.set_reg(RegName::I, if counted { 3 } else { 0x5AA5 });
                state.set_reg(RegName::F, 0x03);
                let mut exec = LlamaExecutor::new();

                exec.validate_before_scheduling(opcode, &state, &mut bus)
                    .expect("HW-009 D0-D3/D8-DB alias must pass preflight");
                let len = exec
                    .execute(opcode, &mut state, &mut bus)
                    .expect("HW-009 D0-D3/D8-DB alias must execute");

                assert_eq!(len, 5, "opcode {opcode:02X}, high {high:02X}");
                assert_eq!(state.pc(), 5, "opcode {opcode:02X}, high {high:02X}");
                assert_eq!(
                    state.get_reg(RegName::I),
                    if counted { 0 } else { 0x5AA5 },
                    "opcode {opcode:02X}, high {high:02X}"
                );
                assert_eq!(state.get_reg(RegName::F), 0x03, "opcode {opcode:02X}");
                if to_external {
                    assert_eq!(
                        &bus.mem[0x4_06D0..0x4_06D0 + transfer_len],
                        &sentinel[..transfer_len],
                        "opcode {opcode:02X}, high {high:02X}"
                    );
                } else {
                    assert_eq!(
                        &bus.mem[0x60..0x60 + transfer_len],
                        &sentinel[..transfer_len],
                        "opcode {opcode:02X}, high {high:02X}"
                    );
                }

                exec.execute(0x00, &mut state, &mut bus)
                    .expect("following NOP must execute");
                assert_eq!(state.pc(), 6, "opcode {opcode:02X}, high {high:02X}");
            }
        }
    }

    #[test]
    fn callf_boundary_pushes_masked_20bit_return_address() {
        let mut bus = OffsetBus::new();
        bus.data.insert(0xFFFFF, 0x05); // CALLF
        bus.data.insert(0x00000, 0x34);
        bus.data.insert(0x00001, 0x12);
        bus.data.insert(0x00002, 0x00);
        let mut state = LlamaState::new();
        state.set_pc(0xFFFFF);
        state.set_reg(RegName::S, 0x100);
        let mut exec = LlamaExecutor::new();

        let len = exec.execute(0x05, &mut state, &mut bus).unwrap();

        assert_eq!(len, 4);
        assert_eq!(state.pc(), 0x01234);
        assert_eq!(state.get_reg(RegName::S), 0xFD);
        assert_eq!(bus.data.get(&0xFD), Some(&0x03));
        assert_eq!(bus.data.get(&0xFE), Some(&0x00));
        assert_eq!(bus.data.get(&0xFF), Some(&0x00));
    }

    #[test]
    fn ret_near_uses_current_page_at_return() {
        // CALL from page 0x30000 to 0x0020, then RET executed after callee changes PC page.
        let mut bus = MemBus::with_size(0x50000);
        let call_pc = 0x30000u32;
        let ret_target = 0x0020u32;
        bus.mem[call_pc as usize] = 0x04; // CALL imm16
        bus.mem[call_pc as usize + 1] = (ret_target & 0xFF) as u8;
        bus.mem[call_pc as usize + 2] = ((ret_target >> 8) & 0xFF) as u8;
        // Place RET at both the original callee page and an alternate page.
        bus.mem[call_pc as usize + ret_target as usize] = 0x06; // RET
        let alt_pc = 0x40000 + ret_target;
        bus.mem[alt_pc as usize] = 0x06; // RET on a different page

        let mut state = LlamaState::new();
        state.set_pc(call_pc);
        state.set_reg(RegName::S, 0x0100);
        let mut exec = LlamaExecutor::new();

        // Execute CALL.
        let len_call = exec.execute(0x04, &mut state, &mut bus).unwrap();
        assert_eq!(len_call, 3);
        assert_eq!(state.get_reg(RegName::PC), call_pc + ret_target);

        // Pretend callee jumped to a different page before RET.
        state.set_pc(alt_pc);
        let len_ret = exec.execute(0x06, &mut state, &mut bus).unwrap();
        assert_eq!(len_ret, 1);
        // Return PC should use the RET page (0x40000) like Python, not the CALL-site page.
        assert_eq!(state.get_reg(RegName::PC), 0x40003);
    }

    #[test]
    fn reti_decrements_call_depth() {
        let mut bus = MemBus::with_size(0x80);
        let mut state = LlamaState::new();
        state.set_reg(RegName::S, 0x20);
        // Simulate an interrupt frame to return from.
        LlamaExecutor::push_stack(&mut state, &mut bus, RegName::S, 0x123456, 24, false);
        LlamaExecutor::push_stack(&mut state, &mut bus, RegName::S, 0x03, 8, false);
        LlamaExecutor::push_stack(&mut state, &mut bus, RegName::S, 0xAA, 8, false);
        state.call_depth_inc();
        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0x01, &mut state, &mut bus).unwrap();
        assert_eq!(len, 1);
        assert_eq!(state.call_depth(), 0, "RETI should reduce call depth");
    }

    #[test]
    fn pushu_imr_reads_from_imem() {
        let mut bus = MemBus::with_size(0x200);
        // Preload IMR in internal memory to a value different from the register snapshot.
        let imr_idx = IMEM_IMR_OFFSET as usize;
        bus.mem[imr_idx] = 0xAA;
        let mut state = LlamaState::new();
        state.set_reg(RegName::IMR, 0x11);
        // Point U into internal space so push lands in the test buffer.
        let sp = INTERNAL_MEMORY_START + 0x40;
        let sp_masked = sp & mask_for(RegName::U);
        state.set_reg(RegName::U, sp);
        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0x2F, &mut state, &mut bus).unwrap(); // PUSHU IMR
        assert_eq!(len, 1);
        let new_sp = state.get_reg(RegName::U);
        assert_eq!(new_sp, sp_masked.wrapping_sub(1) & mask_for(RegName::U));
        let stored = bus.load(new_sp, 8) & 0xFF;
        let imr_after = bus.peek_imem(IMEM_IMR_OFFSET) as u32;
        let expected_cleared = 0xAAu32 & 0x7F;
        assert_eq!(stored, 0xAA, "stack should capture IMR from memory");
        assert_eq!(
            imr_after, expected_cleared,
            "PUSHU IMR should clear IRM (bit 7) after saving"
        );
        assert_eq!(state.get_reg(RegName::IMR), imr_after);
    }

    #[test]
    fn pushu_imr_clears_irm_and_popu_restores() {
        let mut bus = MemBus::with_size(0x200);
        let imr_saved: u8 = 0xAA;
        bus.mem[IMEM_IMR_OFFSET as usize] = imr_saved;

        let sp = INTERNAL_MEMORY_START + 0x40;
        let sp_masked = sp & mask_for(RegName::U);
        let mut state = LlamaState::new();
        state.set_reg(RegName::U, sp);

        let mut exec = LlamaExecutor::new();
        let len_push = exec.execute(0x2F, &mut state, &mut bus).unwrap(); // PUSHU IMR
        assert_eq!(len_push, 1);
        assert_eq!(state.get_reg(RegName::IMR), u32::from(imr_saved & 0x7F));
        assert_eq!(bus.peek_imem(IMEM_IMR_OFFSET), imr_saved & 0x7F);

        let len_pop = exec.execute(0x3F, &mut state, &mut bus).unwrap(); // POPU IMR
        assert_eq!(len_pop, 1);
        assert_eq!(state.get_reg(RegName::IMR), u32::from(imr_saved));
        assert_eq!(bus.peek_imem(IMEM_IMR_OFFSET), imr_saved);
        assert_eq!(state.get_reg(RegName::U), sp_masked);
    }

    #[test]
    fn pushu_imr_clears_irm_with_perfetto_enabled() {
        let _perfetto_lock = crate::perfetto::perfetto_test_guard();
        use crate::PerfettoTracer;

        let mut bus = MemBus::with_size(0x200);
        let imr_saved: u8 = 0xAA;
        bus.mem[IMEM_IMR_OFFSET as usize] = imr_saved;

        let sp = INTERNAL_MEMORY_START + 0x40;
        let mut state = LlamaState::new();
        state.set_reg(RegName::U, sp);

        let path = std::env::temp_dir().join("llama_pushu_imr.perfetto-trace");
        let _ = std::fs::remove_file(&path);
        let mut guard = crate::PERFETTO_TRACER.enter();
        guard.replace(Some(PerfettoTracer::new(path)));

        let mut exec = LlamaExecutor::new();
        let len_push = exec.execute(0x2F, &mut state, &mut bus).unwrap(); // PUSHU IMR
        assert_eq!(len_push, 1);
        assert_eq!(state.get_reg(RegName::IMR), u32::from(imr_saved & 0x7F));
        assert_eq!(bus.peek_imem(IMEM_IMR_OFFSET), imr_saved & 0x7F);

        let _ = guard.take();
    }

    #[test]
    fn mv_regpair_copies_full_register_value() {
        let mut bus = MemBus::with_size(4);
        bus.mem[0] = 0xFD;
        // RegPair encoding: upper nibble selects dst, lower bits select src (bit 3 ignored).
        // dst=Y (5), src=X (4).
        bus.mem[1] = 0x54;
        let mut state = LlamaState::new();
        state.set_reg(RegName::X, 0x123456);
        state.set_reg(RegName::Y, 0);
        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0xFD, &mut state, &mut bus).unwrap();
        assert_eq!(len, 2);
        assert_eq!(state.get_reg(RegName::Y), 0x123456 & mask_for(RegName::Y));
        assert_eq!(state.pc(), 2);
    }

    #[test]
    fn mv_regpair_low_codes_map_to_ba_i_for_mv() {
        let mut bus = MemBus::with_size(4);
        bus.mem[0] = 0xFD;
        bus.mem[1] = 0x01; // dst code 0 => BA, src code 1 => I (MV/EX mapping)
        let mut state = LlamaState::new();
        state.set_reg(RegName::BA, 0xAA55);
        state.set_reg(RegName::I, 0x1234);
        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0xFD, &mut state, &mut bus).unwrap();
        assert_eq!(len, 2);
        assert_eq!(state.get_reg(RegName::BA), 0x1234);
        assert_eq!(state.get_reg(RegName::I), 0x1234);
        assert_eq!(state.pc(), 2);
    }

    #[test]
    fn jp_abs_16_keeps_page() {
        // JP_Abs (0x02), imm16=0x9A78; PC page should be preserved.
        let mut bus = MemBus::with_size(0x400000);
        let start_pc = 0x34567;
        bus.mem[start_pc as usize] = 0x02; // JP abs (16-bit)
        bus.mem[start_pc as usize + 1] = 0x78;
        bus.mem[start_pc as usize + 2] = 0x9A;

        let mut state = LlamaState::new();
        state.set_pc(start_pc);

        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0x02, &mut state, &mut bus).unwrap();
        assert_eq!(len, 3);
        // Page (0x30000) comes from current PC; low bits from immediate.
        assert_eq!(state.pc(), 0x039A78);
    }

    #[test]
    fn jp_abs_16_uses_instruction_page() {
        let mut bus = MemBus::with_size(0x400000);
        let start_pc = 0x12FFFE;
        let base = (start_pc - INTERNAL_MEMORY_START) as usize;
        bus.mem[base] = 0x02; // JP abs (16-bit)
        bus.mem[base + 1] = 0x34;
        bus.mem[base + 2] = 0x12;

        let mut state = LlamaState::new();
        state.set_pc(start_pc);

        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0x02, &mut state, &mut bus).unwrap();
        assert_eq!(len, 3);
        // Instruction page comes from JP address masked to 20 bits (0x02FFFF -> 0x020000).
        assert_eq!(state.pc(), 0x021234);
    }

    #[test]
    fn halt_sets_flag() {
        let mut bus = MemBus::with_size(1);
        bus.mem[0] = 0xDE; // HALT
        let mut state = LlamaState::new();
        let mut exec = LlamaExecutor::new();
        let _ = exec.execute(0xDE, &mut state, &mut bus).unwrap();
        assert!(state.is_halted());
        assert!(!state.is_off(), "HALT should not enter OFF state");
    }

    #[test]
    fn off_sets_state() {
        let mut bus = MemBus::with_size(1);
        bus.mem[0] = 0xDF; // OFF
        let mut state = LlamaState::new();
        let mut exec = LlamaExecutor::new();
        let _ = exec.execute(0xDF, &mut state, &mut bus).unwrap();
        assert!(state.is_halted(), "OFF should enter low-power state");
        assert!(state.is_off(), "OFF should mark power state as off");
    }

    #[test]
    fn wait_clears_i_and_advances_pc() {
        let mut bus = MemBus::with_size(1);
        bus.mem[0] = 0xEF; // WAIT
        let mut state = LlamaState::new();
        state.set_reg(RegName::FC, 1);
        state.set_reg(RegName::FZ, 1);
        state.set_reg(RegName::I, 0xFFFF);
        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0xEF, &mut state, &mut bus).unwrap();
        assert_eq!(len, 1);
        assert_eq!(state.get_reg(RegName::I), 0);
        assert_eq!(state.pc(), 1);
        assert_eq!(state.get_reg(RegName::FC), 1, "WAIT should preserve C");
        assert_eq!(state.get_reg(RegName::FZ), 1, "WAIT should preserve Z");
    }

    #[test]
    fn external_address_advances_wrap_20bit() {
        // External addresses are carried in 20-bit r3 registers.  Do not let
        // block stepping leak into the 24-bit LLIL temporary/container space.
        let top = mask_for(RegName::X);
        assert_eq!(
            LlamaExecutor::advance_internal_addr_signed(top, 1),
            0x000000
        );
        assert_eq!(
            LlamaExecutor::advance_internal_addr_signed(0, -1),
            mask_for(RegName::X)
        );
    }

    #[test]
    fn reset_clears_halt_and_pc() {
        let mut bus = MemBus::with_size(1);
        bus.mem[0] = 0xFF; // RESET
        let mut state = LlamaState::new();
        state.set_pc(0x1234);
        state.halt();
        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0xFF, &mut state, &mut bus).unwrap();
        assert_eq!(len, 1);
        assert_eq!(state.pc(), 0);
        assert!(!state.is_halted());
    }

    #[test]
    fn power_on_reset_uses_rom_vector_and_preserves_imr() {
        let mut bus = MemBus::with_size((INTERNAL_MEMORY_START as usize) + 0x400);
        // Interrupt vector set to a different value to catch regressions.
        bus.mem[INTERRUPT_VECTOR_ADDR as usize] = 0x00;
        bus.mem[INTERRUPT_VECTOR_ADDR as usize + 1] = 0x00;
        bus.mem[INTERRUPT_VECTOR_ADDR as usize + 2] = 0x01; // would decode to 0x010000
                                                            // ROM reset vector (0xFFFFD) -> 0x054321
        bus.mem[ROM_RESET_VECTOR_ADDR as usize] = 0x21;
        bus.mem[ROM_RESET_VECTOR_ADDR as usize + 1] = 0x43;
        bus.mem[ROM_RESET_VECTOR_ADDR as usize + 2] = 0x05;

        // Seed IMR/ISR to ensure reset clears them.
        let imr_idx = MemBus::translate(INTERNAL_MEMORY_START + IMEM_IMR_OFFSET);
        let isr_idx = MemBus::translate(INTERNAL_MEMORY_START + IMEM_ISR_OFFSET);
        bus.mem[imr_idx] = 0xAA;
        bus.mem[isr_idx] = 0x55;

        let mut state = LlamaState::new();
        state.set_reg(RegName::IMR, 0xCC);
        state.halt();

        power_on_reset(&mut bus, &mut state).expect("valid reset vector");

        assert_eq!(state.pc(), 0x054321);
        assert_eq!(
            bus.mem[imr_idx], 0xAA,
            "power_on_reset should preserve IMR in memory"
        );
        assert_eq!(
            bus.mem[isr_idx], 0,
            "power_on_reset should clear ISR in memory"
        );
        assert_eq!(
            state.get_reg(RegName::IMR),
            0xCC,
            "power_on_reset should leave IMR register intact"
        );
        assert!(!state.is_halted());
    }

    #[test]
    fn reset_opcode_uses_rom_vector_and_clears_irq_state() {
        let mut bus = MemBus::with_size((INTERNAL_MEMORY_START as usize) + 0x400);
        // Opcode stream: RESET at PC 0 (already seeded by default zeroed mem)
        bus.mem[0] = 0xFF;
        // ROM reset vector -> 0x00ABCDE
        bus.mem[ROM_RESET_VECTOR_ADDR as usize] = 0xDE;
        bus.mem[ROM_RESET_VECTOR_ADDR as usize + 1] = 0xBC;
        bus.mem[ROM_RESET_VECTOR_ADDR as usize + 2] = 0x0A;

        let imr_idx = MemBus::translate(INTERNAL_MEMORY_START + IMEM_IMR_OFFSET);
        let isr_idx = MemBus::translate(INTERNAL_MEMORY_START + IMEM_ISR_OFFSET);
        bus.mem[imr_idx] = 0xF0;
        bus.mem[isr_idx] = 0x0F;

        let mut state = LlamaState::new();
        state.set_pc(0);
        state.set_reg(RegName::IMR, 0xAA);
        state.halt();
        let mut exec = LlamaExecutor::new();

        let len = exec.execute(0xFF, &mut state, &mut bus).unwrap();
        assert_eq!(len, 1);
        assert_eq!(state.pc(), 0x0ABCDE & mask_for(RegName::PC));
        assert_eq!(
            bus.mem[imr_idx], 0xF0,
            "RESET opcode should not modify IMR in memory"
        );
        assert_eq!(bus.mem[isr_idx], 0);
        assert_eq!(state.get_reg(RegName::IMR), 0xF0);
        assert!(!state.is_halted());
    }

    #[test]
    fn imr_is_synced_from_memory_even_without_tracer() {
        // No perfetto tracer is initialized in this test environment.
        let mut bus = MemBus::with_size((INTERNAL_MEMORY_START as usize) + 0x200);
        let imr_idx = MemBus::translate(INTERNAL_MEMORY_START + IMEM_IMR_OFFSET);
        bus.mem[imr_idx] = 0xAA;
        bus.mem[0] = 0x00; // NOP

        let mut state = LlamaState::new();
        state.set_reg(RegName::IMR, 0x11);
        let mut exec = LlamaExecutor::new();

        let len = exec.execute(0x00, &mut state, &mut bus).unwrap();
        assert_eq!(len, 1);
        assert_eq!(
            state.get_reg(RegName::IMR),
            0xAA,
            "IMR register should mirror IMEM even when tracing is disabled"
        );
    }

    #[test]
    fn mv_emem_post_inc_updates_reg() {
        // 0xB0: MV [r3],A with reg selector byte encoding post-inc X
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0xB0;
        bus.mem[1] = 0x24; // raw_mode=2 (post-inc), reg=X (index 4)
        let mut state = LlamaState::new();
        state.set_reg(RegName::X, 0x10);
        state.set_reg(RegName::A, 0xAB);
        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0xB0, &mut state, &mut bus).unwrap();
        assert_eq!(len, 2);
        assert_eq!(bus.mem[0x10], 0xAB);
        assert_eq!(state.get_reg(RegName::X), 0x11);
        assert_eq!(state.pc(), 2);
    }

    #[test]
    fn mv_emem_pre_dec_loads_and_updates_reg() {
        // 0x90: MV A,[--r3] (pre-dec) encoded via mode nibble 0x3
        let mut bus = MemBus::with_size(0x40);
        bus.mem[0] = 0x90;
        bus.mem[1] = 0x34; // raw_mode=3 (pre-dec), reg=X (index 4)
        bus.mem[0x1F] = 0x66;
        let mut state = LlamaState::new();
        state.set_reg(RegName::X, 0x20);
        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0x90, &mut state, &mut bus).unwrap();
        assert_eq!(len, 2);
        assert_eq!(state.get_reg(RegName::A), 0x66);
        assert_eq!(state.get_reg(RegName::X), 0x1F);
        assert_eq!(state.pc(), 2);
    }

    #[test]
    fn mvw_reg_imem_offset_round_trips_internal_pair_through_u_stack() {
        let mut bus = MemBus::with_size(0x300);
        // PRE30 E9 36 D4: MVW [--U], (D4)
        bus.mem[0] = 0x30;
        bus.mem[1] = 0xE9;
        bus.mem[2] = 0x36;
        bus.mem[3] = 0xD4;
        // PRE30 E1 26 D4: MVW (D4), [U++]
        bus.mem[4] = 0x30;
        bus.mem[5] = 0xE1;
        bus.mem[6] = 0x26;
        bus.mem[7] = 0xD4;
        bus.store(INTERNAL_MEMORY_START + 0xD4, 16, 0x0100);

        let mut state = LlamaState::new();
        state.set_reg(RegName::U, 0x210);
        let mut exec = LlamaExecutor::new();

        let len = exec.execute(0x30, &mut state, &mut bus).unwrap();
        assert_eq!(len, 4);
        assert_eq!(state.get_reg(RegName::U), 0x20E);
        assert_eq!(bus.load(0x20E, 16), 0x0100);

        bus.store(INTERNAL_MEMORY_START + 0xD4, 16, 0x9F9F);
        let len = exec.execute(0x30, &mut state, &mut bus).unwrap();
        assert_eq!(len, 4);
        assert_eq!(state.get_reg(RegName::U), 0x210);
        assert_eq!(bus.load(INTERNAL_MEMORY_START + 0xD4, 16), 0x0100);
    }

    #[test]
    fn mvw_external_absolute_uses_first_pre_mode_for_lone_imem_source() {
        let mut bus = MemBus::with_size(0x300);
        // PRE30 D9 00 02 00 D4: MVW [0x200], (D4)
        bus.mem[0] = 0x30;
        bus.mem[1] = 0xD9;
        bus.mem[2] = 0x00;
        bus.mem[3] = 0x02;
        bus.mem[4] = 0x00;
        bus.mem[5] = 0xD4;
        bus.store(INTERNAL_MEMORY_START + 0xEC, 8, 0xAF);
        bus.store(INTERNAL_MEMORY_START + 0xD4, 16, 0x0101);
        bus.store(INTERNAL_MEMORY_START + 0x83, 16, 0x9F9F);

        let mut state = LlamaState::new();
        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0x30, &mut state, &mut bus).unwrap();

        assert_eq!(len, 6);
        assert_eq!(bus.load(0x200, 16), 0x0101);
    }

    #[test]
    fn mv_emem_offset_uses_displacement_without_mutating_reg() {
        // 0x90: MV A,[r3+disp] encoded via mode nibble 0x8 and displacement byte
        let mut bus = MemBus::with_size(0x60);
        bus.mem[0] = 0x90;
        bus.mem[1] = 0x84; // raw_mode=8 (offset +), reg=X (index 4)
        bus.mem[2] = 0x02; // +2 displacement
        bus.mem[0x32] = 0x77;
        let mut state = LlamaState::new();
        state.set_reg(RegName::X, 0x30);
        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0x90, &mut state, &mut bus).unwrap();
        assert_eq!(len, 3);
        assert_eq!(state.get_reg(RegName::A), 0x77);
        assert_eq!(state.get_reg(RegName::X), 0x30);
        assert_eq!(state.pc(), 3);
    }

    #[test]
    fn mv_emem_negative_offset_handles_0x80() {
        // 0x90: MV A,[r3+disp] with negative displacement encoded via mode nibble 0xC and disp=0x80.
        let mut bus = MemBus::with_size(0x300);
        bus.mem[0] = 0x90;
        bus.mem[1] = 0xC4; // raw_mode=0xC (offset -), reg=X (index 4)
        bus.mem[2] = 0x80; // -128 displacement
        let base = 0x200u32;
        let target = base.wrapping_add(-(0x80i16) as u32);
        bus.mem[target as usize] = 0x55;
        let mut state = LlamaState::new();
        state.set_reg(RegName::X, base);
        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0x90, &mut state, &mut bus).unwrap();
        assert_eq!(len, 3);
        assert_eq!(state.get_reg(RegName::A), 0x55);
        assert_eq!(
            state.get_reg(RegName::X),
            base,
            "offset load should not mutate X"
        );
        assert_eq!(state.pc(), 3);
    }

    struct LoggingBus {
        mem: Vec<u8>,
        log: Vec<u32>,
    }

    impl LoggingBus {
        fn with_bytes(bytes: &[u8]) -> Self {
            let mut mem = bytes.to_vec();
            if mem.is_empty() {
                mem.push(0);
            }
            Self {
                mem,
                log: Vec::new(),
            }
        }
    }

    impl LlamaBus for LoggingBus {
        fn load(&mut self, addr: u32, bits: u8) -> u32 {
            if addr < INTERNAL_MEMORY_START {
                self.log.push(addr);
            }
            let bytes = bits.div_ceil(8);
            let mut val = 0u32;
            for i in 0..bytes {
                let idx = addr as usize + i as usize;
                let b = *self.mem.get(idx).unwrap_or(&0) as u32;
                val |= b << (8 * i);
            }
            if bits == 0 || bits >= 32 {
                val
            } else {
                val & ((1u32 << bits) - 1)
            }
        }

        fn store(&mut self, _addr: u32, _bits: u8, _value: u32) {}

        fn peek_byte_silent(&mut self, addr: u32) -> Option<u8> {
            Some(*self.mem.get(addr as usize).unwrap_or(&0))
        }

        fn resolve_emem(&mut self, base: u32) -> u32 {
            base
        }
    }

    #[test]
    fn cmpw_prefixed_reads_coding_order_for_ops_reversed() {
        // PRE should be followed by the opcode byte, then the operands in coding order (ops_reversed flips them).
        // Bytes: [canonical lone-selector PRE 0x30][opcode 0xD6 CMPW]
        // [reg selector][IMem offset].
        let program = [0x30u8, 0xD6, 0x03, 0x10];
        let mut bus = LoggingBus::with_bytes(&program);
        let mut state = LlamaState::new();
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();

        let _ = exec.execute(0x30, &mut state, &mut bus).unwrap();

        // The program-order loads (excluding IMEM perfetto sampling) should see opcode, reg selector, then IMem offset.
        assert!(
            bus.log.starts_with(&[1, 2, 3]),
            "expected coding-order fetches after PRE, got {:?}",
            bus.log
        );
    }

    #[test]
    fn mv_a_from_imem_pointer() {
        // 0x98: MV A,[(n)] simple mode -> mode=0x00, base=0x10 pointing to ext 0x20
        let mut bus = MemBus::with_size(0x80);
        bus.mem[0] = 0x98;
        bus.mem[1] = 0x00; // simple
        bus.mem[2] = 0x10; // IMEM slot containing pointer
                           // pointer at IMEM 0x10 -> 0x000020
        bus.mem[0x10] = 0x20;
        bus.mem[0x11] = 0x00;
        bus.mem[0x12] = 0x00;
        bus.mem[0x20] = 0x55;
        let mut state = LlamaState::new();
        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0x98, &mut state, &mut bus).unwrap();
        assert_eq!(len, 3);
        assert_eq!(state.get_reg(RegName::A), 0x55);
        assert_eq!(state.pc(), 3);
    }

    #[test]
    fn mv_emem_imem_offset_dest_int() {
        // 0xF0: MV (m),[(n)] with simple mode, dst=0x05, ptr at 0x10 -> ext 0x30
        let mut bus = MemBus::with_size(0x80);
        bus.mem[0] = 0xF0;
        bus.mem[1] = 0x00; // mode simple
        bus.mem[2] = 0x05; // dst internal
        bus.mem[3] = 0x10; // ptr location
        bus.mem[0x10] = 0x30; // pointer -> 0x30
        bus.mem[0x11] = 0x00;
        bus.mem[0x12] = 0x00;
        bus.mem[0x30] = 0xAB;
        let mut state = LlamaState::new();
        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0xF0, &mut state, &mut bus).unwrap();
        assert_eq!(len, 4);
        assert_eq!(bus.mem[0x05], 0xAB);
        assert_eq!(state.pc(), 4);
    }

    #[test]
    fn mv_emem_imem_offset_dest_ext() {
        // 0xF8: MV [(n)],(m) simple mode, ptr at 0x10 -> ext 0x40, src IMEM=0x05
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0xF8;
        bus.mem[1] = 0x00; // mode simple
        bus.mem[2] = 0x10; // ptr location
        bus.mem[3] = 0x05; // src internal
        bus.mem[0x10] = 0x40;
        bus.mem[0x11] = 0x00;
        bus.mem[0x12] = 0x00;
        bus.mem[0x05] = 0xCD;
        let mut state = LlamaState::new();
        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0xF8, &mut state, &mut bus).unwrap();
        assert_eq!(len, 4);
        assert_eq!(bus.mem[0x40], 0xCD);
        assert_eq!(state.pc(), 4);
    }

    #[test]
    fn estimated_length_tracks_encoded_sizes_for_complex_operands() {
        fn assert_length(opcode: u8, bytes: &[u8], expected: u8) {
            let mut bus = MemBus::with_size(512);
            for (idx, byte) in bytes.iter().enumerate() {
                bus.mem[idx] = *byte;
            }
            let mut state = LlamaState::new();
            state.set_pc(0);
            let mut exec = LlamaExecutor::new();
            let entry = exec.lookup(opcode).expect("opcode entry");
            let decoded = exec
                .decode_with_prefix(entry, &mut state, &mut bus, None, None, 0)
                .expect("decode should succeed");
            assert_eq!(
                decoded.len, expected,
                "decoded length should match encoded bytes for opcode 0x{opcode:02X}"
            );
            assert_eq!(
                LlamaExecutor::estimated_length(entry),
                expected,
                "estimated length should reflect encoded size for opcode 0x{opcode:02X}"
            );
        }

        // RegIMemOffset (offset form): opcode + reg/mode + disp + IMEM selector.
        assert_length(0x56, &[0x56, 0x8F, 0x12, 0x34], 4);
        // EMemImemOffset (offset form): opcode + mode + first IMEM + second IMEM + disp.
        assert_length(0xF0, &[0xF0, 0x80, 0x01, 0x02, 0x03], 5);
        // RegPair selector is always a single byte regardless of data width.
        assert_length(0xED, &[0xED, 0x12], 2);
        // ImmOffset (JR-style): opcode + 1-byte relative offset.
        let jr_op = OpcodeEntry {
            opcode: 0x99,
            kind: InstrKind::Unknown,
            name: "JR",
            cond: None,
            ops_reversed: None,
            operands: &[OperandKind::ImmOffset],
        };
        assert_eq!(
            LlamaExecutor::estimated_length(&jr_op),
            2,
            "ImmOffset operands should add one byte to estimated length"
        );
    }

    #[test]
    fn hw002_i_zero_counted_instruction_matrix_matches_hardware() {
        let cases: &[(&str, &[u8], u32, usize)] = &[
            ("ADCL", &[0x54, 0x10, 0x20], 0, 1),
            ("SBCL", &[0x5C, 0x10, 0x20], 1, 1),
            ("DADL", &[0xC4, 0x10, 0x20], 0, 1),
            ("DSBL", &[0xD4, 0x10, 0x20], 1, 1),
            ("MVL", &[0xCB, 0x10, 0x20], 1, 1),
            ("MVLD", &[0xCF, 0x10, 0x20], 1, 1),
            ("EXL", &[0xC3, 0x10, 0x20], 1, 2),
            ("DSLL", &[0xEC, 0x10], 0, 1),
            ("DSRL", &[0xFC, 0x10], 0, 1),
        ];

        for &(name, program, expected_fz, write_multiplier) in cases {
            let code_base = 0x1000usize;
            let mut bus = MemBus::with_size(0x2000);
            bus.mem[code_base..code_base + program.len()].copy_from_slice(program);
            let mut state = LlamaState::new();
            state.set_pc(code_base as u32);
            state.set_reg(RegName::I, 0);
            state.set_reg(RegName::FC, 0);
            state.set_reg(RegName::FZ, 1);
            let mut exec = LlamaExecutor::new();

            let len = exec.execute(program[0], &mut state, &mut bus).unwrap();

            assert_eq!(usize::from(len), program.len(), "program {name}");
            assert_eq!(
                state.pc(),
                code_base as u32 + program.len() as u32,
                "program {name}"
            );
            assert_eq!(state.get_reg(RegName::I), 0, "program {name}");
            assert_eq!(state.get_reg(RegName::FC), 0, "program {name}");
            assert_eq!(state.get_reg(RegName::FZ), expected_fz, "program {name}");
            assert_eq!(
                bus.writes.len(),
                0x1_0000 * write_multiplier,
                "program {name}"
            );
            assert_eq!(bus.mem[0x10], 0, "program {name}");
            assert_eq!(bus.mem[0x20], 0, "program {name}");
        }
    }

    #[test]
    fn tcl_fails_closed_until_timer_clear_is_modeled() {
        let mut bus = MemBus::with_size(0x200);
        bus.mem[IMEM_IMR_OFFSET as usize] = 0xAA;
        bus.mem[0] = 0xCE;
        let mut state = LlamaState::new();
        state.set_pc(0);
        state.set_reg(RegName::S, 0x0100);
        state.set_reg(RegName::IMR, 0x55);
        let mut exec = LlamaExecutor::new();
        let err = exec.execute(0xCE, &mut state, &mut bus).unwrap_err();
        assert_eq!(err, TCL_UNIMPLEMENTED_ERROR);
        assert_eq!(
            state.pc(),
            0,
            "a quarantined instruction must not advance PC"
        );
        assert_eq!(state.get_reg(RegName::S), 0x0100, "stack pointer unchanged");
        assert_eq!(state.get_reg(RegName::IMR), 0x55, "IMR unchanged");
        assert_eq!(bus.mem[IMEM_IMR_OFFSET as usize], 0xAA);
        assert!(bus.writes.is_empty());
    }

    #[test]
    fn preflight_fails_closed_without_a_silent_memory_view() {
        struct ObservableBus {
            loads: usize,
        }

        impl LlamaBus for ObservableBus {
            fn load(&mut self, _addr: u32, _bits: u8) -> u32 {
                self.loads += 1;
                0x05
            }

            fn store(&mut self, _addr: u32, _bits: u8, _value: u32) {}
        }

        let mut bus = ObservableBus { loads: 0 };
        let mut state = LlamaState::new();
        state.set_reg(RegName::A, 0x44);
        let mut exec = LlamaExecutor::new();

        let error = exec
            .execute(0x40, &mut state, &mut bus)
            .expect_err("operand decode must require a silent view");

        assert_eq!(error, SILENT_PEEK_UNAVAILABLE_ERROR);
        assert_eq!(bus.loads, 0, "preflight must not fall back to load");
        assert_eq!(state.get_reg(RegName::A), 0x44);
        assert_eq!(state.pc(), 0);
    }

    #[test]
    fn rejected_instructions_do_not_publish_perfetto_context() {
        let _perfetto_lock = crate::perfetto::perfetto_test_guard();
        reset_perf_counters();
        let mut exec = LlamaExecutor::new();
        let mut state = LlamaState::new();
        let mut bus = MemBus::with_size(0x200);

        state.set_pc(0x10);
        state.push_call_frame(0x12345, 16);
        bus.mem[0x10] = 0x00;
        exec.execute(0x00, &mut state, &mut bus)
            .expect("seed valid trace context");
        let pc_before = perfetto_last_pc();
        let stack_before = perfetto_last_call_stack();
        assert_eq!(pc_before, 0x10);
        assert_eq!(stack_before.len, 1);

        state.push_call_frame(0x23456, 24);
        state.set_pc(0x20);
        bus.mem[0x20] = 0x20;
        exec.execute(0x20, &mut state, &mut bus)
            .expect_err("reserved opcode");
        assert_eq!(perfetto_last_pc(), pc_before);
        assert_eq!(perfetto_last_call_stack(), stack_before);
        assert_eq!(perfetto_instr_context(), None);
    }

    #[test]
    fn perfetto_instruction_context_is_thread_local() {
        let _perfetto_lock = crate::perfetto::perfetto_test_guard();
        reset_perf_counters();
        PERF_CURRENT_OP.with(|value| value.set(7));
        PERF_CURRENT_PC.with(|value| value.set(0x12345));

        std::thread::spawn(|| {
            assert_eq!(perfetto_instr_context(), None);
            PERF_CURRENT_OP.with(|value| value.set(9));
            PERF_CURRENT_PC.with(|value| value.set(0x54321));
            assert_eq!(perfetto_instr_context(), Some((9, 0x54321)));
            drop(PerfettoContextGuard);
            assert_eq!(perfetto_instr_context(), None);
        })
        .join()
        .expect("context isolation thread");

        assert_eq!(perfetto_instr_context(), Some((7, 0x12345)));
        drop(PerfettoContextGuard);
        assert_eq!(perfetto_instr_context(), None);
    }

    #[test]
    fn perfetto_last_pc_tracks_executed_instruction_pc() {
        let _perfetto_lock = crate::perfetto::perfetto_test_guard();
        let _perfetto_handle = PERFETTO_TRACER.enter();
        reset_perf_counters();
        let mut exec = LlamaExecutor::new();
        let mut state = LlamaState::new();
        state.set_pc(0x0123);
        let mut bus = MemBus::with_size(0x0200);
        bus.mem[0x0123] = 0x00; // NOP

        let _ = exec
            .execute(0x00, &mut state, &mut bus)
            .expect("execute NOP");

        assert_eq!(
            perfetto_last_pc(),
            0x0123 & mask_for(RegName::PC),
            "perfetto_last_pc should reflect the executed instruction PC"
        );
        assert_eq!(
            state.pc(),
            0x0124 & mask_for(RegName::PC),
            "state PC should advance independently of perfetto_last_pc"
        );
    }
}
