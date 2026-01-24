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
    llama::async_eval::{tick, TickHelper},
    memory::{
        with_imr_read_suppressed, ADDRESS_MASK, IMEM_BP_OFFSET, IMEM_IMR_OFFSET, IMEM_ISR_OFFSET,
        IMEM_LCC_OFFSET, IMEM_PX_OFFSET, IMEM_PY_OFFSET, IMEM_SCR_OFFSET, IMEM_SSR_OFFSET,
        IMEM_UCR_OFFSET, IMEM_USR_OFFSET, INTERNAL_MEMORY_START,
    },
    perfetto::AnnotationValue,
    PERFETTO_TRACER,
};
use serde::{Deserialize, Serialize};
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

static PERF_INSTR_COUNTER: AtomicU64 = AtomicU64::new(0);
static PERF_CURRENT_PC: AtomicU32 = AtomicU32::new(u32::MAX);
static PERF_CURRENT_OP: AtomicU64 = AtomicU64::new(u64::MAX);
static PERF_SUBSTEP: AtomicU32 = AtomicU32::new(0);

pub const PERFETTO_CALL_STACK_MAX_FRAMES: usize = 8;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerfettoCallStack {
    pub len: u8,
    pub frames: [u32; PERFETTO_CALL_STACK_MAX_FRAMES],
}

thread_local! {
    static PERF_LAST_PC: Cell<u32> = const { Cell::new(0) };
    static PERF_LAST_CALL_STACK: Cell<PerfettoCallStack> = const { Cell::new(PerfettoCallStack { len: 0, frames: [0; PERFETTO_CALL_STACK_MAX_FRAMES] }) };
}

struct PerfettoContextGuard;
impl Drop for PerfettoContextGuard {
    fn drop(&mut self) {
        PERF_CURRENT_OP.store(u64::MAX, Ordering::Relaxed);
        PERF_CURRENT_PC.store(u32::MAX, Ordering::Relaxed);
        PERF_SUBSTEP.store(0, Ordering::Relaxed);
    }
}

/// Expose current instruction context for Perfetto correlation outside the executor.
pub fn perfetto_instr_context() -> Option<(u64, u32)> {
    let op = PERF_CURRENT_OP.load(Ordering::Relaxed);
    let pc = PERF_CURRENT_PC.load(Ordering::Relaxed);
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
    PERF_CURRENT_PC.store(u32::MAX, Ordering::Relaxed);
    PERF_CURRENT_OP.store(u64::MAX, Ordering::Relaxed);
    PERF_LAST_PC.with(|value| value.set(0));
    PERF_LAST_CALL_STACK.with(|value| value.set(PerfettoCallStack::default()));
    PERF_SUBSTEP.store(0, Ordering::Relaxed);
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
    PERF_SUBSTEP.fetch_add(1, Ordering::Relaxed) as u64 + 1
}

fn perfetto_reset_substep() {
    PERF_SUBSTEP.store(0, Ordering::Relaxed);
}

fn fallback_unknown(
    state: &mut LlamaState,
    prefix_len: u8,
    estimated_len: Option<u8>,
) -> Result<u8, &'static str> {
    // Match Python decoder: consume the full estimated instruction length (if known) even
    // for unknown/unsupported opcodes to keep the stream aligned for tracing.
    let opcode_len = estimated_len.unwrap_or(1);
    let len = opcode_len.saturating_add(prefix_len);
    let start_pc = state.pc();
    if state.pc() == start_pc {
        state.set_pc(start_pc.wrapping_add(len as u32));
    }
    Ok(len)
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
    0xBC, 0xBD, 0xBE, 0xC5, 0xCC, 0xCD, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xDC, 0xE3, 0xE5, 0xE7,
    0xEB, 0xEC, 0xF5, 0xF7, 0xFC,
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
    fn load(&mut self, _addr: u32, _bits: u8) -> u32 {
        0
    }
    fn store(&mut self, _addr: u32, _bits: u8, _value: u32) {}
    fn resolve_emem(&mut self, base: u32) -> u32 {
        base
    }
    fn peek_imem(&mut self, offset: u32) -> u8 {
        let addr = INTERNAL_MEMORY_START + offset;
        (self.load(addr, 8) & 0xFF) as u8
    }
    /// Peek IMEM without emitting tracing side-effects (IMR/ISR sampling).
    fn peek_imem_silent(&mut self, offset: u32) -> u8 {
        with_imr_read_suppressed(|| self.peek_imem(offset))
    }
    /// Optional hook for WAIT to spin timers/keyboard for `cycles` iterations (unused for Python parity WAIT).
    fn wait_cycles(&mut self, _cycles: u32) {}
    /// Optional timer snapshot for perfetto tracing (ticks since last MTI/STI fire).
    fn timer_trace(&mut self) -> Option<TimerTrace> {
        None
    }
    /// Optional timer periods (MTI, STI) for chunk clamp decisions.
    fn timer_periods(&mut self) -> Option<(u64, u64)> {
        None
    }
    /// Optional hook to finalize timer state at instruction boundaries.
    fn finalize_instruction(&mut self) {}
    /// Optional hook to finalize timer state at internal timing chunk boundaries.
    fn finalize_timer_chunk(&mut self) {}
    /// Optional hook to control timer boundary clamping for the current instruction.
    fn set_timer_finalize_clamp(&mut self, _clamp: bool) {}
    /// Optional hook to mark the cycle at which an instruction starts (post-opcode tick).
    fn mark_instruction_start(&mut self) {}
    /// Optional hook to surface the current cycle count for tracing.
    fn cycle_count(&mut self) -> Option<u64> {
        None
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
    // Optional perfetto emit when the builder is available (llama-tests builds).
    let mut guard = crate::PERFETTO_TRACER.enter();
    guard.with_some(|tracer| {
        let op_idx = PERF_CURRENT_OP.load(Ordering::Relaxed);
        let pc = PERF_CURRENT_PC.load(Ordering::Relaxed);
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
    _bus: &mut B,
    state: &mut LlamaState,
    power_state: PowerState,
) {
    // Correctness: HALT/OFF should not adjust USR/SSR; only enter the power state.

    if power_state == PowerState::Off {
        state.record_off_transition(state.pc());
    }
    state.set_power_state(power_state);
}

/// Apply power-on reset side effects (IMEM init, PC jump to reset vector).
pub fn power_on_reset<B: LlamaBus>(bus: &mut B, state: &mut LlamaState) {
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

    // Use the ROM reset vector at 0xFFFFD (interrupt vector remains at 0xFFFFA).
    let reset_vector = bus.load(ROM_RESET_VECTOR_ADDR, 8)
        | (bus.load(ROM_RESET_VECTOR_ADDR + 1, 8) << 8)
        | (bus.load(ROM_RESET_VECTOR_ADDR + 2, 8) << 16);
    // Parity: keep register/flag values intact; only adjust IMEM/PC. Drop any saved
    // call-page context so near returns fall back to the current page like Python.
    state.clear_call_page_stack();
    state.set_pc(reset_vector & mask_for(RegName::PC));
    state.set_halted(false);
}

pub struct LlamaExecutor {
    chunk_start_timer: Option<TimerTrace>,
    deferred_ticks: u64,
}

pub(crate) struct PerfettoInstrGuard {
    instr_index: u64,
    trace_pc: u32,
    trace_opcode: u8,
    regs: Option<HashMap<String, u32>>,
    _ctx: PerfettoContextGuard,
}

impl PerfettoInstrGuard {
    pub(crate) fn begin(state: &LlamaState, opcode: u8) -> Self {
        let instr_index = PERF_INSTR_COUNTER.fetch_add(1, Ordering::Relaxed);
        let trace_pc_snapshot = state.pc() & mask_for(RegName::PC);
        let trace_opcode_snapshot = opcode;

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
        PERF_CURRENT_OP.store(instr_index, Ordering::Relaxed);
        PERF_CURRENT_PC.store(trace_pc_snapshot, Ordering::Relaxed);
        let ctx = PerfettoContextGuard;

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

        Self {
            instr_index,
            trace_pc: trace_pc_snapshot,
            trace_opcode: trace_opcode_snapshot,
            regs: trace_regs,
            _ctx: ctx,
        }
    }

    pub(crate) fn finish<B: LlamaBus>(self, exec: &LlamaExecutor, bus: &mut B) {
        bus.finalize_instruction();
        if let Some(regs) = self.regs.as_ref() {
            exec.trace_instr(
                self.trace_opcode,
                regs,
                bus,
                self.instr_index,
                self.trace_pc,
            );
        }
    }
}

impl LlamaExecutor {
    pub fn new() -> Self {
        reset_perf_counters();
        Self {
            chunk_start_timer: None,
            deferred_ticks: 0,
        }
    }

    fn record_timer_chunk_start<B: LlamaBus>(&mut self, bus: &mut B) {
        self.chunk_start_timer = bus.timer_trace();
    }

    fn finalize_timer_chunk<B: LlamaBus>(&mut self, bus: &mut B) {
        bus.finalize_timer_chunk();
        self.record_timer_chunk_start(bus);
    }

    fn defer_ticks(&mut self, count: u64) {
        self.deferred_ticks = self.deferred_ticks.saturating_add(count);
    }

    async fn flush_deferred_ticks(&mut self, ticker: &mut TickHelper<'_>) {
        if self.deferred_ticks > 0 {
            tick!(ticker, self.deferred_ticks);
            self.deferred_ticks = 0;
        }
    }

    fn maybe_finalize_timer_chunk<B: LlamaBus>(&mut self, bus: &mut B, chunk_cycles: u64) {
        let Some(start) = self.chunk_start_timer else {
            self.finalize_timer_chunk(bus);
            return;
        };
        let Some((mti_period, sti_period)) = bus.timer_periods() else {
            self.finalize_timer_chunk(bus);
            return;
        };
        let mut should_finalize = false;
        if mti_period > 0 {
            let start_mti = start.mti_ticks % mti_period;
            if start_mti.saturating_add(chunk_cycles) >= mti_period {
                should_finalize = true;
            }
        }
        if sti_period > 0 {
            let start_sti = start.sti_ticks % sti_period;
            if start_sti.saturating_add(chunk_cycles) >= sti_period {
                should_finalize = true;
            }
        }
        if should_finalize {
            self.finalize_timer_chunk(bus);
        }
    }

    pub fn lookup(&self, opcode: u8) -> Option<&'static OpcodeEntry> {
        dispatch::lookup(opcode)
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
        let new_sp = state.get_reg(sp_reg).wrapping_sub(bytes as u32) & mask;
        for i in 0..bytes {
            let addr = new_sp.wrapping_add(i as u32) & mask;
            let shift = if big_endian {
                8 * (bytes.saturating_sub(1) - i)
            } else {
                8 * i
            };
            let byte = (value >> shift) & 0xFF;
            Self::store_traced(bus, addr, 8, byte);
        }
        state.set_reg(sp_reg, new_sp);
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

    fn cond_pass(entry: &OpcodeEntry, state: &LlamaState) -> bool {
        match entry.cond {
            None => true,
            Some("Z") => state.get_reg(RegName::FZ) & 1 == 1,
            Some("NZ") => state.get_reg(RegName::FZ) & 1 == 0,
            Some("C") => state.get_reg(RegName::FC) & 1 == 1,
            Some("NC") => state.get_reg(RegName::FC) & 1 == 0,
            _ => true,
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
            let op_index = PERF_CURRENT_OP.load(Ordering::Relaxed);
            let pc = PERF_CURRENT_PC.load(Ordering::Relaxed);
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
        bus.store(addr, bits, value);
        Self::trace_mem_write(addr, bits, value);
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

    fn read_imm<B: LlamaBus>(bus: &mut B, addr: u32, bits: u8) -> u32 {
        let bytes = bits.div_ceil(8);
        let mut value = 0u32;
        for i in 0..bytes {
            value |= (bus.load(addr + i as u32, 8) & 0xFF) << (8 * i);
        }
        if bits >= 32 {
            value
        } else {
            let mask = if bits == 0 { 0 } else { (1u32 << bits) - 1 };
            value & mask
        }
    }

    async fn load_u8_async<B: LlamaBus>(
        bus: &mut B,
        addr: u32,
        ticker: &mut TickHelper<'_>,
    ) -> u8 {
        tick!(ticker);
        bus.load(addr, 8) as u8
    }

    async fn read_imem_byte_async<B: LlamaBus>(
        bus: &mut B,
        offset: u32,
        ticker: &mut TickHelper<'_>,
    ) -> u8 {
        Self::load_u8_async(bus, INTERNAL_MEMORY_START + offset, ticker).await
    }

    async fn store_u8_async<B: LlamaBus>(
        bus: &mut B,
        addr: u32,
        value: u8,
        ticker: &mut TickHelper<'_>,
    ) {
        tick!(ticker);
        Self::store_traced(bus, addr, 8, value as u32);
    }

    async fn read_imm_async<B: LlamaBus>(
        bus: &mut B,
        addr: u32,
        bits: u8,
        ticker: &mut TickHelper<'_>,
    ) -> u32 {
        let bytes = bits.div_ceil(8);
        let mut value = 0u32;
        for i in 0..bytes {
            let byte = Self::load_u8_async(bus, addr + i as u32, ticker).await;
            value |= u32::from(byte & 0xFF) << (8 * i);
        }
        if bits >= 32 {
            value
        } else {
            let mask = if bits == 0 { 0 } else { (1u32 << bits) - 1 };
            value & mask
        }
    }

    async fn store_imm_async<B: LlamaBus>(
        bus: &mut B,
        addr: u32,
        bits: u8,
        value: u32,
        ticker: &mut TickHelper<'_>,
    ) {
        let bytes = bits.div_ceil(8);
        let masked = value & Self::mask_for_width(bits);
        for i in 0..bytes {
            let byte = ((masked >> (8 * i)) & 0xFF) as u8;
            Self::store_u8_async(bus, addr + i as u32, byte, ticker).await;
        }
    }

    fn store_imm_no_tick<B: LlamaBus>(bus: &mut B, addr: u32, bits: u8, value: u32) {
        let bytes = bits.div_ceil(8);
        let masked = value & Self::mask_for_width(bits);
        for i in 0..bytes {
            let byte = ((masked >> (8 * i)) & 0xFF) as u8;
            Self::store_traced(bus, addr + i as u32, 8, byte as u32);
        }
    }

    async fn store_imm_internal_single_tick_async<B: LlamaBus>(
        bus: &mut B,
        addr: u32,
        bits: u8,
        value: u32,
        ticker: &mut TickHelper<'_>,
    ) {
        // Baseline emulator charges a single cycle for internal writes when sourcing
        // from an immediate external address (MV/MVW/MVP).
        tick!(ticker);
        Self::store_imm_no_tick(bus, addr, bits, value);
    }

    async fn push_stack_async<B: LlamaBus>(
        state: &mut LlamaState,
        bus: &mut B,
        sp_reg: RegName,
        value: u32,
        bits: u8,
        big_endian: bool,
        ticker: &mut TickHelper<'_>,
    ) {
        let bytes = bits.div_ceil(8);
        let mask = mask_for(sp_reg);
        let new_sp = state.get_reg(sp_reg).wrapping_sub(bytes as u32) & mask;
        for i in 0..bytes {
            let addr = new_sp.wrapping_add(i as u32) & mask;
            let shift = if big_endian {
                8 * (bytes.saturating_sub(1) - i)
            } else {
                8 * i
            };
            let byte = ((value >> shift) & 0xFF) as u8;
            Self::store_u8_async(bus, addr, byte, ticker).await;
        }
        state.set_reg(sp_reg, new_sp);
    }

    async fn pop_stack_async<B: LlamaBus>(
        state: &mut LlamaState,
        bus: &mut B,
        sp_reg: RegName,
        bits: u8,
        big_endian: bool,
        ticker: &mut TickHelper<'_>,
    ) -> u32 {
        let bytes = bits.div_ceil(8);
        let mut value = 0u32;
        let mask = mask_for(sp_reg);
        let mut sp = state.get_reg(sp_reg);
        for i in 0..bytes {
            let byte = Self::load_u8_async(bus, sp, ticker).await as u32;
            let shift = if big_endian {
                8 * (bytes.saturating_sub(1) - i)
            } else {
                8 * i
            };
            value |= (byte & 0xFF) << shift;
            sp = sp.wrapping_add(1) & mask;
        }
        state.set_reg(sp_reg, sp);
        value & Self::mask_for_width(bits)
    }

    async fn enter_low_power_state_async<B: LlamaBus>(
        _bus: &mut B,
        state: &mut LlamaState,
        power_state: PowerState,
        ticker: &mut TickHelper<'_>,
    ) {
        // Low-power entry costs one extra cycle beyond the opcode fetch.
        tick!(ticker);

        if power_state == PowerState::Off {
            state.record_off_transition(state.pc());
        }
        state.set_power_state(power_state);
    }

    async fn power_on_reset_async<B: LlamaBus>(
        bus: &mut B,
        state: &mut LlamaState,
        ticker: &mut TickHelper<'_>,
    ) {
        let mut lcc = Self::read_imem_byte_async(bus, IMEM_LCC_OFFSET, ticker).await;
        lcc &= !0x80;
        Self::store_u8_async(bus, INTERNAL_MEMORY_START + IMEM_LCC_OFFSET, lcc, ticker).await;

        Self::store_u8_async(bus, INTERNAL_MEMORY_START + IMEM_UCR_OFFSET, 0, ticker).await;
        Self::store_u8_async(bus, INTERNAL_MEMORY_START + IMEM_ISR_OFFSET, 0, ticker).await;
        Self::store_u8_async(bus, INTERNAL_MEMORY_START + IMEM_SCR_OFFSET, 0, ticker).await;

        let mut usr = Self::read_imem_byte_async(bus, IMEM_USR_OFFSET, ticker).await;
        usr &= !0x3F;
        usr |= 0x18;
        Self::store_u8_async(bus, INTERNAL_MEMORY_START + IMEM_USR_OFFSET, usr, ticker).await;

        let mut ssr = Self::read_imem_byte_async(bus, IMEM_SSR_OFFSET, ticker).await;
        ssr &= !0x04;
        Self::store_u8_async(bus, INTERNAL_MEMORY_START + IMEM_SSR_OFFSET, ssr, ticker).await;

        let reset_vector = Self::read_imm_async(bus, ROM_RESET_VECTOR_ADDR, 24, ticker).await;
        state.clear_call_page_stack();
        state.set_pc(reset_vector & mask_for(RegName::PC));
        state.set_halted(false);
    }

    async fn decode_simple_operands_async<B: LlamaBus>(
        &mut self,
        state: &mut LlamaState,
        bus: &mut B,
        entry: &OpcodeEntry,
        exec_pc: u32,
        prefix_len: u8,
        pre: Option<&PreModes>,
        ticker: &mut TickHelper<'_>,
    ) -> Result<DecodedOperands, &'static str> {
        let mut decoded = DecodedOperands::default();
        let mut offset = 1u32;
        let single_pre = SINGLE_ADDRESSABLE_OPCODES.contains(&entry.opcode);
        let skip_imm_tick_for_imem = Self::skip_immediate_tick_for_imem(entry);

        if entry.opcode == 0xE3 {
            let (mem_src, consumed) = self
                .decode_ext_reg_ptr_async(state, bus, exec_pc + offset, 1, ticker)
                .await?;
            decoded.mem2 = Some(mem_src);
            offset = offset.saturating_add(consumed);
            let raw_imem = Self::load_u8_async(bus, exec_pc + offset, ticker).await;
            let imem_addr =
                Self::imem_addr_for_mode_async(bus, mode_for_operand(pre, 0), raw_imem, ticker)
                    .await;
            decoded.mem = Some(MemOperand {
                addr: imem_addr,
                bits: 8,
                side_effect: None,
            });
            offset = offset.saturating_add(1);
            decoded.len = (offset as u8).saturating_add(prefix_len);
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
                    let imm_addr = exec_pc.wrapping_add(offset) & mask_for(RegName::PC);
                    let imm = if skip_imm_tick_for_imem {
                        Self::read_imm(bus, imm_addr, *bits)
                    } else {
                        Self::read_imm_async(bus, imm_addr, *bits, ticker).await
                    };
                    decoded.imm = Some((imm, *bits));
                    offset = offset.saturating_add((*bits as u32).div_ceil(8));
                }
                OperandKind::ImmOffset => {
                    let imm_addr = exec_pc.wrapping_add(offset) & mask_for(RegName::PC);
                    let imm = Self::load_u8_async(bus, imm_addr, ticker).await as u32;
                    decoded.imm = Some((imm, 8));
                    offset = offset.saturating_add(1);
                }
                OperandKind::IMem(bits) => {
                    let raw_addr = exec_pc.wrapping_add(offset) & mask_for(RegName::PC);
                    let raw = Self::load_u8_async(bus, raw_addr, ticker).await;
                    let mode_index = if single_pre { 0 } else { operand_index };
                    let mode = mode_for_operand(pre, mode_index);
                    let addr = Self::imem_addr_for_mode_async(bus, mode, raw, ticker).await;
                    let slot = if decoded.mem.is_none() {
                        &mut decoded.mem
                    } else {
                        &mut decoded.mem2
                    };
                    *slot = Some(MemOperand {
                        addr,
                        bits: *bits,
                        side_effect: None,
                    });
                    offset = offset.saturating_add(1);
                }
                OperandKind::IMemWidth(bytes) => {
                    let raw_addr = exec_pc.wrapping_add(offset) & mask_for(RegName::PC);
                    let raw = Self::load_u8_async(bus, raw_addr, ticker).await;
                    let mode_index = if single_pre { 0 } else { operand_index };
                    let mode = mode_for_operand(pre, mode_index);
                    let addr = Self::imem_addr_for_mode_async(bus, mode, raw, ticker).await;
                    let slot = if decoded.mem.is_none() {
                        &mut decoded.mem
                    } else {
                        &mut decoded.mem2
                    };
                    *slot = Some(MemOperand {
                        addr,
                        bits: Self::bits_from_bytes(*bytes),
                        side_effect: None,
                    });
                    offset = offset.saturating_add(1);
                }
                OperandKind::Reg(_, _)
                | OperandKind::RegB
                | OperandKind::RegIL
                | OperandKind::RegIMR
                | OperandKind::RegF => {}
                OperandKind::RegPair(size) => {
                    let selector = Self::load_u8_async(bus, exec_pc + offset, ticker).await;
                    let use_r2 = matches!(entry.kind, InstrKind::Mv | InstrKind::Ex);
                    let r1 = Self::regpair_name((selector >> 4) & 0x7, use_r2);
                    let r2 = Self::regpair_name(selector & 0x7, use_r2);
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
                    offset = offset.saturating_add(1);
                }
                OperandKind::Reg3 => {
                    let selector = Self::load_u8_async(bus, exec_pc + offset, ticker).await;
                    let reg = match selector & 0x7 {
                        0 => RegName::A,
                        1 => RegName::IL,
                        2 => RegName::BA,
                        3 => RegName::I,
                        4 => RegName::X,
                        5 => RegName::Y,
                        6 => RegName::U,
                        7 => RegName::S,
                        _ => RegName::Unknown("reg3"),
                    };
                    decoded.reg3 = Some(reg);
                    offset = offset.saturating_add(1);
                }
                OperandKind::EMemAddrWidth(bytes) | OperandKind::EMemAddrWidthOp(bytes) => {
                    let base = Self::read_imm_async(bus, exec_pc + offset, 24, ticker).await;
                    let bits = Self::bits_from_bytes(*bytes);
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
                    offset = offset.saturating_add(3);
                }
                OperandKind::EMemRegWidth(bytes) | OperandKind::EMemRegWidthMode(bytes) => {
                    let (mem, consumed) = self
                        .decode_ext_reg_ptr_async(state, bus, exec_pc + offset, *bytes, ticker)
                        .await?;
                    if decoded.mem.is_none() {
                        decoded.mem = Some(mem);
                    } else {
                        decoded.mem2 = Some(mem);
                    }
                    offset = offset.saturating_add(consumed);
                }
                OperandKind::EMemRegModePostPre => {
                    let (mem, consumed) = self
                        .decode_ext_reg_ptr_async(state, bus, exec_pc + offset, 1, ticker)
                        .await?;
                    if decoded.mem.is_none() {
                        decoded.mem = Some(mem);
                    } else {
                        decoded.mem2 = Some(mem);
                    }
                    offset = offset.saturating_add(consumed);
                }
                OperandKind::EMemIMemWidth(bytes) => {
                    let mode_index = if single_pre { 0 } else { operand_index };
                    let mode = mode_for_operand(pre, mode_index);
                    let (mem, consumed) =
                        self.decode_imem_ptr_async(bus, exec_pc + offset, *bytes, mode, ticker)
                            .await?;
                    if decoded.mem.is_none() {
                        decoded.mem = Some(mem);
                    } else {
                        decoded.mem2 = Some(mem);
                    }
                    offset = offset.saturating_add(consumed);
                }
                OperandKind::EMemImemOffsetDestIntMem => {
                    let mode_first_index = if single_pre { 0 } else { operand_index };
                    let mode_second_index = if single_pre { 0 } else { operand_index + 1 };
                    let mode_first = mode_for_operand(pre, mode_first_index);
                    let mode_second = mode_for_operand(pre, mode_second_index);
                    let (transfer, consumed) = self
                        .decode_emem_imem_offset_async(
                            entry,
                            bus,
                            exec_pc + offset,
                            mode_first,
                            mode_second,
                            true,
                            ticker,
                        )
                        .await?;
                    decoded.transfer = Some(transfer);
                    offset = offset.saturating_add(consumed);
                }
                OperandKind::EMemImemOffsetDestExtMem => {
                    let mode_first_index = if single_pre { 0 } else { operand_index };
                    let mode_second_index = if single_pre { 0 } else { operand_index + 1 };
                    let mode_first = mode_for_operand(pre, mode_first_index);
                    let mode_second = mode_for_operand(pre, mode_second_index);
                    let (transfer, consumed) = self
                        .decode_emem_imem_offset_async(
                            entry,
                            bus,
                            exec_pc + offset,
                            mode_first,
                            mode_second,
                            false,
                            ticker,
                        )
                        .await?;
                    decoded.transfer = Some(transfer);
                    offset = offset.saturating_add(consumed);
                }
                OperandKind::RegIMemOffset(kind) => {
                    let width_bits = Self::width_bits_for_kind(entry.kind);
                    let width_bytes = width_bits.div_ceil(8);
                    let reg_byte = bus.load(exec_pc + offset, 8) as u8;
                    let mode_code = Self::normalize_ext_reg_mode((reg_byte >> 4) & 0x0F);
                    let (mode, needs_disp, disp_sign) = match mode_code {
                        0x0 | 0x1 => (ExtRegMode::Simple, false, 0),
                        0x2 => (ExtRegMode::PostInc, false, 0),
                        0x3 => (ExtRegMode::PreDec, false, 0),
                        0x8 => (ExtRegMode::Offset, true, 1),
                        0xC => (ExtRegMode::Offset, true, -1),
                        _ => return Err("unsupported EMEM reg mode"),
                    };
                    if needs_disp || matches!(mode, ExtRegMode::PreDec) {
                        self.defer_ticks(1);
                    } else if entry.kind != InstrKind::Mvp {
                        tick!(ticker);
                    }
                    let reg = Self::reg_from_selector(reg_byte).ok_or("invalid reg selector")?;

                    let mut consumed_ptr = 1u32;
                    let raw_imem = if needs_disp || matches!(mode, ExtRegMode::PreDec) {
                        self.defer_ticks(1);
                        bus.load(exec_pc + offset + consumed_ptr, 8) as u8
                    } else {
                        Self::load_u8_async(bus, exec_pc + offset + consumed_ptr, ticker).await
                    };
                    consumed_ptr = consumed_ptr.saturating_add(1);
                    let mut disp: i16 = 0;
                    if needs_disp {
                        let magnitude = Self::load_u8_async(
                            bus,
                            exec_pc + offset + consumed_ptr,
                            ticker,
                        )
                        .await;
                        disp = if disp_sign >= 0 {
                            magnitude as i16
                        } else {
                            -(magnitude as i16)
                        };
                        consumed_ptr = consumed_ptr.saturating_add(1);
                        if width_bits <= 16 {
                            self.defer_ticks(1);
                        }
                        self.maybe_finalize_timer_chunk(bus, 1);
                        self.flush_deferred_ticks(ticker).await;
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
                            // Baseline timing: pre-dec RegIMemOffset charges this extra tick
                            // for MV/MVW, but MVP already accounts for the 3-byte transfer.
                            if width_bits <= 16 {
                                tick!(ticker);
                            }
                            self.maybe_finalize_timer_chunk(bus, 1);
                            self.flush_deferred_ticks(ticker).await;
                        }
                        ExtRegMode::PostInc => {
                            side_effect = Some((reg, (base.wrapping_add(step)) & mask));
                        }
                    }
                    let ptr_mem = MemOperand {
                        addr: bus.resolve_emem(addr),
                        bits: width_bits,
                        side_effect,
                    };

                    // Parity: RegIMemOffset uses the first PRE mode for the IMEM selector.
                    let mode_index = if single_pre { 0 } else { operand_index };
                    let imem_addr =
                        Self::imem_addr_for_mode_async(bus, mode_for_operand(pre, mode_index), raw_imem, ticker)
                            .await;
                    offset = offset.saturating_add(consumed_ptr);
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
                _ => return Err("unsupported async operand"),
            }
        }
        self.flush_deferred_ticks(ticker).await;
        decoded.len = (offset as u8).saturating_add(prefix_len);
        Ok(decoded)
    }

    fn skip_immediate_tick_for_imem(entry: &OpcodeEntry) -> bool {
        let has_imm = entry
            .operands
            .iter()
            .any(|op| matches!(op, OperandKind::Imm(_)));
        if !has_imm {
            return false;
        }
        let has_imem = entry.operands.iter().any(|op| {
            matches!(
                op,
                OperandKind::IMem(_) | OperandKind::IMemWidth(_)
            )
        });
        if !has_imem {
            return false;
        }
        matches!(
            entry.kind,
            InstrKind::Mv
                | InstrKind::Mvw
                | InstrKind::Mvp
                | InstrKind::Add
                | InstrKind::Sub
                | InstrKind::And
                | InstrKind::Or
                | InstrKind::Xor
                | InstrKind::Adc
                | InstrKind::Sbc
                | InstrKind::Pmdf
        )
    }

    async fn imem_addr_for_mode_async<B: LlamaBus>(
        bus: &mut B,
        mode: AddressingMode,
        raw: u8,
        _ticker: &mut TickHelper<'_>,
    ) -> u32 {
        let bp = bus.peek_imem_silent(IMEM_BP_OFFSET) as u32;
        let px = bus.peek_imem_silent(IMEM_PX_OFFSET) as u32;
        let py = bus.peek_imem_silent(IMEM_PY_OFFSET) as u32;
        let base = match mode {
            AddressingMode::N => raw as u32,
            AddressingMode::BpN => bp.wrapping_add(raw as u32),
            AddressingMode::PxN => px.wrapping_add(raw as u32),
            AddressingMode::PyN => py.wrapping_add(raw as u32),
            AddressingMode::BpPx => bp.wrapping_add(px),
            AddressingMode::BpPy => bp.wrapping_add(py),
        };
        trace_imem_addr(mode, base, bp, px, py);
        INTERNAL_MEMORY_START + (base & 0xFF)
    }

    fn is_internal_addr(addr: u32) -> bool {
        (INTERNAL_MEMORY_START..(INTERNAL_MEMORY_START + 0x100)).contains(&addr)
    }

    fn advance_internal_addr(addr: u32, step: u32) -> u32 {
        if Self::is_internal_addr(addr) {
            let offset = addr.wrapping_sub(INTERNAL_MEMORY_START);
            let wrapped = offset.wrapping_add(step) & 0xFF;
            INTERNAL_MEMORY_START + wrapped
        } else {
            addr.wrapping_add(step) & ADDRESS_MASK
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
            addr.wrapping_add(step as u32) & ADDRESS_MASK
        } else {
            addr.wrapping_sub((-step) as u32) & ADDRESS_MASK
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

    fn normalize_ext_reg_mode(raw: u8) -> u8 {
        if raw & 0x8 != 0 {
            if raw & 0x4 != 0 {
                return 0xC;
            }
            return 0x8;
        }
        if raw & 0x3 == 0x3 {
            return 0x3;
        }
        if raw & 0x2 != 0 {
            return 0x2;
        }
        0x0
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
            _ => None,
        }
    }

    fn decode_ext_reg_ptr<B: LlamaBus>(
        &mut self,
        state: &mut LlamaState,
        bus: &mut B,
        pc: u32,
        width_bytes: u8,
    ) -> Result<(MemOperand, u32), &'static str> {
        let reg_byte = bus.load(pc, 8) as u8;
        let raw_mode = (reg_byte >> 4) & 0x0F;
        let mode_code = Self::normalize_ext_reg_mode(raw_mode);
        let (mode, needs_disp, disp_sign) = match mode_code {
            0x0 | 0x1 => (ExtRegMode::Simple, false, 0),
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
            let magnitude = bus.load(pc + 1, 8) as u8;
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
                addr: bus.resolve_emem(addr),
                bits,
                side_effect,
            },
            consumed,
        ))
    }

    async fn decode_ext_reg_ptr_async<B: LlamaBus>(
        &mut self,
        state: &mut LlamaState,
        bus: &mut B,
        pc: u32,
        width_bytes: u8,
        ticker: &mut TickHelper<'_>,
    ) -> Result<(MemOperand, u32), &'static str> {
        let reg_byte = bus.load(pc, 8) as u8;
        let raw_mode = (reg_byte >> 4) & 0x0F;
        let mode_code = Self::normalize_ext_reg_mode(raw_mode);
        let (mode, needs_disp, disp_sign) = match mode_code {
            0x0 | 0x1 => (ExtRegMode::Simple, false, 0),
            0x2 => (ExtRegMode::PostInc, false, 0),
            0x3 => (ExtRegMode::PreDec, false, 0),
            0x8 => (ExtRegMode::Offset, true, 1),
            0xC => (ExtRegMode::Offset, true, -1),
            _ => return Err("unsupported EMEM reg mode"),
        };
        if needs_disp || matches!(mode, ExtRegMode::PreDec) {
            self.defer_ticks(1);
        } else {
            tick!(ticker);
        }
        let reg = Self::reg_from_selector(reg_byte).ok_or("invalid reg selector")?;

        let mut consumed = 1u32;
        let mut disp: i16 = 0;
        if needs_disp {
            let magnitude = Self::load_u8_async(bus, pc + 1, ticker).await;
            disp = if disp_sign >= 0 {
                magnitude as i16
            } else {
                -(magnitude as i16)
            };
            consumed += 1;
            self.maybe_finalize_timer_chunk(bus, 2);
            tick!(ticker);
            self.flush_deferred_ticks(ticker).await;
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
                            self.maybe_finalize_timer_chunk(bus, 1);
                            tick!(ticker);
                            self.flush_deferred_ticks(ticker).await;
                        }
            ExtRegMode::PostInc => {
                side_effect = Some((reg, (base.wrapping_add(step)) & mask));
            }
        }
        let bits = Self::bits_from_bytes(width_bytes);
        Ok((
            MemOperand {
                addr: bus.resolve_emem(addr),
                bits,
                side_effect,
            },
            consumed,
        ))
    }

    fn decode_imem_ptr<B: LlamaBus>(
        &mut self,
        bus: &mut B,
        pc: u32,
        width_bytes: u8,
        mode: AddressingMode,
    ) -> Result<(MemOperand, u32), &'static str> {
        let mode_byte = bus.load(pc, 8) as u8;
        let (needs_disp, sign) = match mode_byte & 0xC0 {
            0x00 => (false, 0),
            0x80 => (true, 1),
            0xC0 => (true, -1),
            _ => return Err("unsupported EMEM/IMEM mode"),
        };
        let base_raw = bus.load(pc + 1, 8) as u8;
        let base = imem_addr_for_mode(bus, mode, base_raw);
        let mut consumed = 2u32;
        let mut disp: i16 = 0;
        if needs_disp {
            let magnitude = bus.load(pc + 2, 8) as u8;
            disp = if sign >= 0 {
                magnitude as i16
            } else {
                -(magnitude as i16)
            };
            consumed += 1;
        }
        let pointer = Self::read_imm(bus, base, 24);
        let addr = bus.resolve_emem(pointer.wrapping_add(disp as u32));
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

    async fn decode_imem_ptr_async<B: LlamaBus>(
        &mut self,
        bus: &mut B,
        pc: u32,
        width_bytes: u8,
        mode: AddressingMode,
        ticker: &mut TickHelper<'_>,
    ) -> Result<(MemOperand, u32), &'static str> {
        let mode_byte = bus.load(pc, 8) as u8;
        let (needs_disp, sign) = match mode_byte & 0xC0 {
            0x00 => (false, 0),
            0x80 => (true, 1),
            0xC0 => (true, -1),
            _ => return Err("unsupported EMEM/IMEM mode"),
        };
        if needs_disp {
            self.defer_ticks(1);
        } else {
            tick!(ticker);
        }
        let base_raw = if needs_disp {
            self.defer_ticks(1);
            bus.load(pc + 1, 8) as u8
        } else {
            Self::load_u8_async(bus, pc + 1, ticker).await
        };
        let base = Self::imem_addr_for_mode_async(bus, mode, base_raw, ticker).await;
        let mut consumed = 2u32;
        let mut disp: i16 = 0;
        if needs_disp {
            if self.deferred_ticks > 2 {
                self.deferred_ticks = self.deferred_ticks.saturating_sub(1);
                tick!(ticker);
            }
            let magnitude = Self::load_u8_async(bus, pc + 2, ticker).await;
            disp = if sign >= 0 {
                magnitude as i16
            } else {
                -(magnitude as i16)
            };
            consumed += 1;
            let chunk_cycles = self.deferred_ticks.max(1);
            self.maybe_finalize_timer_chunk(bus, chunk_cycles);
            self.flush_deferred_ticks(ticker).await;
            tick!(ticker);
        }
        let pointer = Self::read_imm_async(bus, base, 24, ticker).await;
        let addr = bus.resolve_emem(pointer.wrapping_add(disp as u32));
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
        &mut self,
        entry: &OpcodeEntry,
        bus: &mut B,
        pc: u32,
        mode_first: AddressingMode,
        mode_second: AddressingMode,
        dest_is_internal: bool,
    ) -> Result<(EmemImemTransfer, u32), &'static str> {
        let mode_byte = bus.load(pc, 8) as u8;
        let top = mode_byte & 0xC0;
        let (needs_offset, sign) = match top {
            0x00 => (false, 0),
            0x80 => (true, 1),
            0xC0 => (true, -1),
            _ => return Err("unsupported EMEM/IMEM mode"),
        };
        let first = bus.load(pc + 1, 8) as u8;
        let second = bus.load(pc + 2, 8) as u8;
        let mut consumed = 3u32;
        let mut disp: i32 = 0;
        if needs_offset {
            let magnitude = bus.load(pc + 3, 8) as u8;
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
            let ext_addr = bus.resolve_emem(pointer.wrapping_add(disp as u32));
            (first_addr, ext_addr, true)
        } else {
            // MV [(n)],(m) - dst is external pointer from first, src is internal second
            let pointer = Self::read_imm(bus, first_addr, 24);
            let ext_addr = bus.resolve_emem(pointer.wrapping_add(disp as u32));
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

    async fn decode_emem_imem_offset_async<B: LlamaBus>(
        &mut self,
        entry: &OpcodeEntry,
        bus: &mut B,
        pc: u32,
        mode_first: AddressingMode,
        mode_second: AddressingMode,
        dest_is_internal: bool,
        ticker: &mut TickHelper<'_>,
    ) -> Result<(EmemImemTransfer, u32), &'static str> {
        let mode_byte = bus.load(pc, 8) as u8;
        let top = mode_byte & 0xC0;
        let (needs_offset, sign) = match top {
            0x00 => (false, 0),
            0x80 => (true, 1),
            0xC0 => (true, -1),
            _ => return Err("unsupported EMEM/IMEM mode"),
        };
        if needs_offset {
            self.defer_ticks(1);
        } else {
            tick!(ticker);
        }
        let first = if needs_offset {
            self.defer_ticks(1);
            bus.load(pc + 1, 8) as u8
        } else {
            Self::load_u8_async(bus, pc + 1, ticker).await
        };
        let second = if needs_offset {
            self.defer_ticks(1);
            bus.load(pc + 2, 8) as u8
        } else {
            Self::load_u8_async(bus, pc + 2, ticker).await
        };
        let mut consumed = 3u32;
        let mut disp: i32 = 0;
        if needs_offset {
            if self.deferred_ticks > 3 {
                self.deferred_ticks = self.deferred_ticks.saturating_sub(1);
                tick!(ticker);
            }
            let magnitude = Self::load_u8_async(bus, pc + 3, ticker).await;
            disp = if sign >= 0 {
                magnitude as i32
            } else {
                -(magnitude as i32)
            };
            consumed += 1;
            let chunk_cycles = self.deferred_ticks.max(1);
            self.maybe_finalize_timer_chunk(bus, chunk_cycles);
            self.flush_deferred_ticks(ticker).await;
            tick!(ticker);
        }
        let width_bits = Self::width_bits_for_kind(entry.kind);
        let first_addr = Self::imem_addr_for_mode_async(bus, mode_first, first, ticker).await;
        let second_addr = Self::imem_addr_for_mode_async(bus, mode_second, second, ticker).await;
        let (dst_addr, src_addr, dst_is_internal) = if dest_is_internal {
            let pointer = Self::read_imm_async(bus, second_addr, 24, ticker).await;
            let ext_addr = bus.resolve_emem(pointer.wrapping_add(disp as u32));
            (first_addr, ext_addr, true)
        } else {
            let pointer = Self::read_imm_async(bus, first_addr, 24, ticker).await;
            let ext_addr = bus.resolve_emem(pointer.wrapping_add(disp as u32));
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
        &mut self,
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
        // Opcode-specific decoding quirks
        if entry.opcode == 0xE3 {
            // Encoding order is EMemReg mode byte then IMem8.
            let (mem_src, consumed) = self.decode_ext_reg_ptr(state, bus, pc + offset, 1)?;
            decoded.mem2 = Some(mem_src);
            offset += consumed;
            let raw_imem = bus.load(pc + offset, 8) & 0xFF;
            let imem_addr = imem_addr_for_mode(bus, mode_for_operand(pre, 0), raw_imem as u8);
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
                    let val = Self::read_imm(bus, pc + offset, *bits);
                    decoded.imm = Some((val, *bits));
                    offset += (*bits as u32).div_ceil(8);
                }
                OperandKind::ImmOffset => {
                    let byte = bus.load(pc + offset, 8) as u8;
                    decoded.imm = Some((byte as u32, 8));
                    offset += 1;
                }
                OperandKind::IMem(bits) => {
                    let raw = bus.load(pc + offset, 8) & 0xFF;
                    let slot = if decoded.mem.is_none() {
                        &mut decoded.mem
                    } else {
                        &mut decoded.mem2
                    };
                    let mode_index = if single_pre { 0 } else { operand_index };
                    *slot = Some(MemOperand {
                        addr: imem_addr_for_mode(bus, mode_for_operand(pre, mode_index), raw as u8),
                        bits: *bits,
                        side_effect: None,
                    });
                    offset += 1;
                }
                OperandKind::IMemWidth(bytes) => {
                    let bits = Self::bits_from_bytes(*bytes);
                    let raw = bus.load(pc + offset, 8) & 0xFF;
                    let slot = if decoded.mem.is_none() {
                        &mut decoded.mem
                    } else {
                        &mut decoded.mem2
                    };
                    let mode_index = if single_pre { 0 } else { operand_index };
                    *slot = Some(MemOperand {
                        addr: imem_addr_for_mode(bus, mode_for_operand(pre, mode_index), raw as u8),
                        bits,
                        side_effect: None,
                    });
                    offset += 1;
                }
                OperandKind::EMemAddrWidth(bytes) | OperandKind::EMemAddrWidthOp(bytes) => {
                    let bits = Self::bits_from_bytes(*bytes);
                    let base = Self::read_imm(bus, pc + offset, 24);
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
                    let (mem, consumed) = self.decode_ext_reg_ptr(state, bus, pc + offset, 1)?;
                    if decoded.mem.is_none() {
                        decoded.mem = Some(mem);
                    } else {
                        decoded.mem2 = Some(mem);
                    }
                    offset += consumed;
                }
                OperandKind::EMemIMemWidth(bytes) => {
                    let mode_index = if single_pre { 0 } else { operand_index };
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
                    let reg_byte = bus.load(pc + offset, 8) as u8;
                    let mode_code = Self::normalize_ext_reg_mode((reg_byte >> 4) & 0x0F);
                    let (mode, needs_disp, disp_sign) = match mode_code {
                        0x0 | 0x1 => (ExtRegMode::Simple, false, 0),
                        0x2 => (ExtRegMode::PostInc, false, 0),
                        0x3 => (ExtRegMode::PreDec, false, 0),
                        0x8 => (ExtRegMode::Offset, true, 1),
                        0xC => (ExtRegMode::Offset, true, -1),
                        _ => return Err("unsupported EMEM reg mode"),
                    };
                    let reg = Self::reg_from_selector(reg_byte).ok_or("invalid reg selector")?;

                    // RegIMemOffset encoding places the IMEM byte before any displacement.
                    let mut consumed_ptr = 1u32;
                    let raw_imem = bus.load(pc + offset + consumed_ptr, 8) & 0xFF;
                    consumed_ptr += 1;
                    let mut disp: i16 = 0;
                    if needs_disp {
                        let magnitude = bus.load(pc + offset + consumed_ptr, 8) as u8;
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
                        addr: bus.resolve_emem(addr),
                        bits: width_bits,
                        side_effect,
                    };

                    // Parity: RegIMemOffset uses the first PRE mode for the IMEM selector.
                    let mode_index = if single_pre { 0 } else { operand_index };
                    let imem_addr =
                        imem_addr_for_mode(bus, mode_for_operand(pre, mode_index), raw_imem as u8);
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
                    let selector = bus.load(pc + offset, 8) as u8;
                    let reg = match selector & 0x7 {
                        0 => RegName::A,
                        1 => RegName::IL,
                        2 => RegName::BA,
                        3 => RegName::I,
                        4 => RegName::X,
                        5 => RegName::Y,
                        6 => RegName::U,
                        7 => RegName::S,
                        _ => RegName::Unknown("reg3"),
                    };
                    decoded.reg3 = Some(reg);
                    offset += 1;
                }
                OperandKind::RegPair(size) => {
                    let raw = bus.load(pc + offset, 8) as u8;
                    let use_r2 = matches!(entry.kind, InstrKind::Mv | InstrKind::Ex);
                    let r1 = Self::regpair_name((raw >> 4) & 0x7, use_r2);
                    let r2 = Self::regpair_name(raw & 0x7, use_r2);
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
                _ => {}
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

    pub async fn execute_async<B: LlamaBus>(
        &mut self,
        opcode: u8,
        state: &mut LlamaState,
        bus: &mut B,
        ticker: &mut TickHelper<'_>,
    ) -> Result<u8, &'static str> {
        let mem_imr = with_imr_read_suppressed(|| bus.peek_imem_silent(IMEM_IMR_OFFSET));
        state.set_reg(RegName::IMR, mem_imr as u32);
        let instr_guard = PerfettoInstrGuard::begin(state, opcode);
        macro_rules! finish {
            ($res:expr) => {{
                let res = $res;
                instr_guard.finish(self, bus);
                return res;
            }};
        }
        self.deferred_ticks = 0;
        self.record_timer_chunk_start(bus);
        tick!(ticker);
        bus.mark_instruction_start();
        // Default to clamping timer remainder at the end of the instruction.
        // Correctness: AddState() single-chunk timing.
        bus.set_timer_finalize_clamp(true);

        let start_pc = state.pc() & mask_for(RegName::PC);
        let mut exec_pc = start_pc;
        let mut exec_opcode = opcode;
        let mut prefix_len = 0u8;
        let mut pre_modes_opt: Option<PreModes> = None;
        let mut entry = self.lookup(exec_opcode);

        while let Some(e) = entry {
            if e.kind != InstrKind::Pre {
                break;
            }
            self.finalize_timer_chunk(bus);
            let pre_modes = pre_modes_for(exec_opcode).ok_or("unknown PRE opcode")?;
            let next_pc = exec_pc.wrapping_add(1) & mask_for(RegName::PC);
            let next_opcode = bus.load(next_pc, 8) as u8;
            self.defer_ticks(1);
            exec_opcode = next_opcode;
            exec_pc = next_pc;
            prefix_len = prefix_len.saturating_add(1);
            pre_modes_opt = Some(pre_modes);
            entry = self.lookup(exec_opcode);
        }

        let entry = entry.ok_or("unknown opcode")?;
        let result = match entry.kind {
            InstrKind::Nop => {
                self.flush_deferred_ticks(ticker).await;
                let len = 1u8.saturating_add(prefix_len);
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(len as u32));
                }
                Ok(len)
            }
            InstrKind::Wait => {
                self.flush_deferred_ticks(ticker).await;
                // WAIT advances in per-cycle loops; do not clamp away timer remainder.
                bus.set_timer_finalize_clamp(false);
                let raw_i = state.get_reg(RegName::I) & mask_for(RegName::I);
                let wait_cycles = if raw_i == 0 {
                    (mask_for(RegName::I) + 1) as u32
                } else {
                    raw_i as u32
                };
                let wait_cycles = wait_cycles.max(1);
                // Base opcode tick already accounted for at entry; WAIT consumes the remaining cycles.
                tick!(ticker, wait_cycles.saturating_sub(1) as u64);
                state.set_reg(RegName::I, 0);
                let len = 1u8.saturating_add(prefix_len);
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(len as u32));
                }
                Ok(len)
            }
            InstrKind::Off => {
                self.flush_deferred_ticks(ticker).await;
                Self::enter_low_power_state_async(bus, state, PowerState::Off, ticker).await;
                let len = 1u8.saturating_add(prefix_len);
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(len as u32));
                }
                Ok(len)
            }
            InstrKind::Halt => {
                self.flush_deferred_ticks(ticker).await;
                Self::enter_low_power_state_async(bus, state, PowerState::Halted, ticker).await;
                let len = 1u8.saturating_add(prefix_len);
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(len as u32));
                }
                Ok(len)
            }
            InstrKind::Reset => {
                self.flush_deferred_ticks(ticker).await;
                Self::power_on_reset_async(bus, state, ticker).await;
                Ok(1u8.saturating_add(prefix_len))
            }
            InstrKind::Sc => {
                self.flush_deferred_ticks(ticker).await;
                state.set_reg(RegName::FC, 1);
                let len = prefix_len.saturating_add(Self::estimated_length(entry));
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(len as u32));
                }
                Ok(len)
            }
            InstrKind::Rc => {
                self.flush_deferred_ticks(ticker).await;
                state.set_reg(RegName::FC, 0);
                let len = prefix_len.saturating_add(Self::estimated_length(entry));
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(len as u32));
                }
                Ok(len)
            }
            InstrKind::Tcl => {
                self.flush_deferred_ticks(ticker).await;
                let len = prefix_len.saturating_add(Self::estimated_length(entry));
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(len as u32));
                }
                Ok(len)
            }
            InstrKind::Ir => {
                self.flush_deferred_ticks(ticker).await;
                let instr_len = prefix_len.saturating_add(Self::estimated_length(entry));
                let pc_before = state.pc() & mask_for(RegName::PC);
                let pc = state
                    .pc()
                    .wrapping_add(instr_len as u32)
                    & mask_for(RegName::PC);
                Self::push_stack_async(state, bus, RegName::S, pc, 24, false, ticker).await;
                let f = state.get_reg(RegName::F) & 0xFF;
                Self::push_stack_async(state, bus, RegName::S, f, 8, false, ticker).await;
                let imr = Self::read_imem_byte_async(bus, IMEM_IMR_OFFSET, ticker).await;
                Self::push_stack_async(state, bus, RegName::S, imr as u32, 8, false, ticker)
                    .await;
                let cleared_imr = imr & 0x7F;
                Self::store_u8_async(
                    bus,
                    INTERNAL_MEMORY_START + IMEM_IMR_OFFSET,
                    cleared_imr,
                    ticker,
                )
                .await;
                state.set_reg(RegName::IMR, cleared_imr as u32);
                state.call_depth_inc();
                let vec = Self::read_imm_async(bus, INTERRUPT_VECTOR_ADDR, 24, ticker).await;
                state.set_pc(vec & mask_for(RegName::PC));
                let mut guard = crate::PERFETTO_TRACER.enter();
                guard.with_some(|tracer| {
                    let mut payload = HashMap::new();
                    payload.insert(
                        "pc".to_string(),
                        AnnotationValue::Pointer((pc & mask_for(RegName::PC)) as u64),
                    );
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
                    AnnotationValue::Pointer(pc as u64),
                );
                cf_payload.insert("ret_addr".to_string(), AnnotationValue::Pointer(pc as u64));
                cf_payload.insert(
                    "instr_len".to_string(),
                    AnnotationValue::UInt(instr_len as u64),
                );
                Self::emit_control_flow_event(
                    entry.name,
                    "irq",
                    PERF_INSTR_COUNTER.load(Ordering::Relaxed),
                    pc_before,
                    cf_payload,
                );
                Ok(instr_len)
            }
            InstrKind::JpAbs => {
                let pc_mask = mask_for(RegName::PC);
                let pc_before = state.pc() & pc_mask;
                let cond_ok = Self::cond_pass(entry, state);
                if entry.cond.is_some() && cond_ok {
                    // Conditional jumps take a 1-cycle AddState chunk before the base cost.
                    // Correctness: clamp after first chunk.
                    self.finalize_timer_chunk(bus);
                }
                let decoded = self
                    .decode_simple_operands_async(
                        state,
                        bus,
                        entry,
                        exec_pc,
                        prefix_len,
                        pre_modes_opt.as_ref(),
                        ticker,
                    )
                    .await?;
                let fallthrough = pc_before.wrapping_add(decoded.len as u32) & pc_mask;
                let mut target = None;
                let mut target_addr = None;
                let mut target_reg = None;
                let target_src = if let Some((val, bits)) = decoded.imm {
                    let instr_pc = exec_pc & pc_mask;
                    let dest = if bits == 16 {
                        (instr_pc & 0xFF0000) | (val & 0xFFFF)
                    } else {
                        val & pc_mask
                    };
                    target = Some(dest);
                    "imm"
                } else if let Some(mem) = decoded.mem {
                    target_addr = Some(mem.addr);
                    if cond_ok {
                        let val = Self::read_imm_async(bus, mem.addr, mem.bits, ticker).await;
                        target = Some(val & pc_mask);
                    }
                    "mem"
                } else if let Some(r) = decoded.reg3 {
                    target = Some(state.get_reg(r) & pc_mask);
                    target_reg = Some(Self::reg_name_for_trace(r));
                    "reg"
                } else {
                    finish!(Err("missing jump target"));
                };

                if cond_ok && target.is_none() {
                    finish!(Err("missing jump target"));
                }
                let dest = if cond_ok {
                    target.unwrap_or(fallthrough)
                } else {
                    fallthrough
                };
                state.set_pc(dest);
                if cond_ok {
                    let extra_ticks = match target_src {
                        "imm" | "mem" => 1,
                        "reg" => 2,
                        _ => 0,
                    };
                    if extra_ticks > 0 {
                        tick!(ticker, extra_ticks as u64);
                    }
                }
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
                Self::emit_control_flow_event(entry.name, kind, PERF_INSTR_COUNTER.load(Ordering::Relaxed), pc_before, payload);
                Ok(decoded.len)
            }
            InstrKind::JpRel => {
                let pc_mask = mask_for(RegName::PC);
                let pc_before = state.pc() & pc_mask;
                let cond_ok = Self::cond_pass(entry, state);
                if entry.cond.is_some() && cond_ok {
                    // Conditional relative branches take a 1-cycle AddState chunk before the base cost.
                    // Correctness: clamp after first chunk.
                    self.finalize_timer_chunk(bus);
                }
                let decoded = self
                    .decode_simple_operands_async(
                        state,
                        bus,
                        entry,
                        exec_pc,
                        prefix_len,
                        pre_modes_opt.as_ref(),
                        ticker,
                    )
                    .await?;
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
                if cond_ok {
                    tick!(ticker);
                }
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
                Self::emit_control_flow_event(entry.name, kind, PERF_INSTR_COUNTER.load(Ordering::Relaxed), pc_before, payload);
                Ok(decoded.len)
            }
            InstrKind::Call => {
                let decoded = self
                    .decode_simple_operands_async(
                        state,
                        bus,
                        entry,
                        exec_pc,
                        prefix_len,
                        pre_modes_opt.as_ref(),
                        ticker,
                    )
                    .await?;
                let (target, bits) = decoded.imm.ok_or("missing jump target")?;
                let pc_before = state.pc();
                let pc_mask = mask_for(RegName::PC);
                let ret_addr = pc_before.wrapping_add(decoded.len as u32);
                let mut dest = target;
                let push_bits = if bits == 16 {
                    let high = pc_before & 0xFF0000;
                    dest = high | (target & 0xFFFF);
                    16
                } else {
                    24
                };
                tick!(ticker);
                Self::push_stack_async(state, bus, RegName::S, ret_addr, push_bits, false, ticker)
                    .await;
                if push_bits == 16 {
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
                    PERF_INSTR_COUNTER.load(Ordering::Relaxed),
                    pc_before & pc_mask,
                    payload,
                );
                Ok(decoded.len)
            }
            InstrKind::Ret => {
                let pc_before = state.pc();
                let pc_mask = mask_for(RegName::PC);
                tick!(ticker);
                let ret = Self::pop_stack_async(state, bus, RegName::S, 16, false, ticker).await;
                let current_page = state.pc() & 0xFF0000;
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
                    PERF_INSTR_COUNTER.load(Ordering::Relaxed),
                    pc_before & pc_mask,
                    payload,
                );
                Ok(1)
            }
            InstrKind::RetF => {
                let pc_before = state.pc();
                let pc_mask = mask_for(RegName::PC);
                tick!(ticker);
                let ret = Self::pop_stack_async(state, bus, RegName::S, 24, false, ticker).await;
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
                    PERF_INSTR_COUNTER.load(Ordering::Relaxed),
                    pc_before & pc_mask,
                    payload,
                );
                Ok(1)
            }
            InstrKind::RetI => {
                let pc_mask = mask_for(RegName::PC);
                let pc_before = state.pc() & pc_mask;
                let mask_s = mask_for(RegName::S);
                let mut sp = state.get_reg(RegName::S) & mask_s;
                let imr = Self::load_u8_async(bus, sp, ticker).await;
                sp = sp.wrapping_add(1) & mask_s;
                let f = Self::load_u8_async(bus, sp, ticker).await;
                sp = sp.wrapping_add(1) & mask_s;
                let pc_lo = Self::load_u8_async(bus, sp, ticker).await;
                let pc_mid = Self::load_u8_async(bus, sp.wrapping_add(1) & mask_s, ticker).await;
                let pc_hi = Self::load_u8_async(bus, sp.wrapping_add(2) & mask_s, ticker).await;
                let ret = ((pc_hi as u32) << 16) | ((pc_mid as u32) << 8) | (pc_lo as u32);
                sp = sp.wrapping_add(3) & mask_s;
                state.set_reg(RegName::S, sp);
                let imr_restored = imr as u32;
                Self::store_u8_async(
                    bus,
                    INTERNAL_MEMORY_START + IMEM_IMR_OFFSET,
                    imr,
                    ticker,
                )
                .await;
                state.set_reg(RegName::IMR, imr_restored);
                state.set_reg(RegName::F, f as u32);
                state.set_pc(ret & 0xFFFFF);
                state.call_depth_dec();
                let instr_len = 1u8.saturating_add(prefix_len);
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
                    PERF_INSTR_COUNTER.load(Ordering::Relaxed),
                    pc_before & pc_mask,
                    payload,
                );
                Ok(instr_len)
            }
            InstrKind::PushU | InstrKind::PushS => {
                let decoded = self
                    .decode_simple_operands_async(
                        state,
                        bus,
                        entry,
                        exec_pc,
                        prefix_len,
                        pre_modes_opt.as_ref(),
                        ticker,
                    )
                    .await?;
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
                let value = if reg == RegName::IMR {
                    let imr = bus.peek_imem_silent(IMEM_IMR_OFFSET);
                    state.set_reg(RegName::IMR, imr as u32);
                    imr as u32
                } else {
                    state.get_reg(reg)
                };
                let sp_reg = if entry.kind == InstrKind::PushU {
                    RegName::U
                } else {
                    RegName::S
                };
                Self::push_stack_async(state, bus, sp_reg, value, bits, false, ticker).await;
                if reg != RegName::IMR {
                    tick!(ticker);
                }
                if reg == RegName::IMR {
                    let imr_addr = INTERNAL_MEMORY_START + IMEM_IMR_OFFSET;
                    let cleared = (value & 0xFF) & 0x7F;
                    Self::store_u8_async(bus, imr_addr, cleared as u8, ticker).await;
                    state.set_reg(RegName::IMR, cleared);
                }
                let len = decoded.len;
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(len as u32));
                }
                Ok(len)
            }
            InstrKind::PopU | InstrKind::PopS => {
                let saved_fc = state.get_reg(RegName::FC);
                let saved_fz = state.get_reg(RegName::FZ);
                let decoded = self
                    .decode_simple_operands_async(
                        state,
                        bus,
                        entry,
                        exec_pc,
                        prefix_len,
                        pre_modes_opt.as_ref(),
                        ticker,
                    )
                    .await?;
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
                let value =
                    Self::pop_stack_async(state, bus, sp_reg, bits, false, ticker).await;
                state.set_reg(reg, value);
                if reg == RegName::IL {
                    tick!(ticker);
                }
                if reg == RegName::IMR {
                    Self::store_traced(
                        bus,
                        INTERNAL_MEMORY_START + IMEM_IMR_OFFSET,
                        8,
                        (value & 0xFF) as u32,
                    );
                }
                let len = decoded.len;
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(len as u32));
                }
                if restore_flags {
                    state.set_reg(RegName::FC, saved_fc);
                    state.set_reg(RegName::FZ, saved_fz);
                }
                Ok(len)
            }
            InstrKind::Pmdf => {
                let decoded = self
                    .decode_simple_operands_async(
                        state,
                        bus,
                        entry,
                        exec_pc,
                        prefix_len,
                        pre_modes_opt.as_ref(),
                        ticker,
                    )
                    .await?;
                let mem = decoded.mem.ok_or("missing mem operand")?;
                let src_val = match entry.operands.get(1) {
                    Some(OperandKind::Imm(_)) => decoded.imm.ok_or("missing immediate")?.0,
                    Some(OperandKind::Reg(RegName::A, _)) => state.get_reg(RegName::A) & 0xFF,
                    _ => finish!(Err("unsupported PMDF operands")),
                } & 0xFF;
                let dst = Self::load_u8_async(bus, mem.addr, ticker).await;
                let res = dst.wrapping_add(src_val as u8);
                Self::store_u8_async(bus, mem.addr, res, ticker).await;
                let len = decoded.len;
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(len as u32));
                }
                Ok(len)
            }
            InstrKind::Dadl | InstrKind::Dsbl => {
                let decoded = self
                    .decode_simple_operands_async(
                        state,
                        bus,
                        entry,
                        exec_pc,
                        prefix_len,
                        pre_modes_opt.as_ref(),
                        ticker,
                    )
                    .await?;
                let mem_dst = decoded.mem.ok_or("missing destination")?;
                if mem_dst.bits != 8 {
                    finish!(Err("unsupported width for DADL/DSBL"));
                }
                let mut dst_addr = mem_dst.addr;
                let mut src_addr = decoded.mem2.map(|m| m.addr);
                let src_bits = decoded.mem2.map(|m| m.bits);
                let src_reg = if src_bits.is_none() {
                    entry
                        .operands
                        .get(1)
                        .and_then(|op| Self::resolved_reg(op, &decoded))
                } else {
                    None
                };
                let mut src_reg_byte = src_reg.map(|reg| (state.get_reg(reg) & 0xFF) as u8);
                let length = state.get_reg(RegName::I) & mask_for(RegName::I);
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
                    let dst_byte = Self::load_u8_async(bus, dst_addr, ticker).await;
                    let src_byte = if let Some(bits) = src_bits {
                        let addr = src_addr.ok_or("missing source")?;
                        Self::read_imm_async(bus, addr, bits, ticker).await as u8
                    } else if let Some(reg_byte) = src_reg_byte {
                        let val = reg_byte;
                        src_reg_byte = Some(0);
                        val
                    } else {
                        finish!(Err("missing source"));
                    };
                    let (res, new_carry) = if entry.kind == InstrKind::Dadl {
                        Self::bcd_add_byte(dst_byte, src_byte, carry)
                    } else {
                        Self::bcd_sub_byte(dst_byte, src_byte, carry)
                    };
                    Self::store_imm_no_tick(bus, dst_addr, mem_dst.bits, res as u32);
                    carry = new_carry;
                    overall_zero |= res as u32;
                    if let Some(addr) = src_addr.as_mut() {
                        *addr = Self::advance_internal_addr_signed(*addr, -(src_step as i32));
                    }
                    dst_addr = Self::advance_internal_addr_signed(dst_addr, -(dst_step as i32));
                    executed = true;
                }
                if length > 0 && src_bits.is_none() {
                    tick!(ticker);
                }
                state.set_reg(RegName::I, 0);
                let zero_mask = Self::mask_for_width(mem_dst.bits);
                state.set_reg(
                    RegName::FZ,
                    if (overall_zero & zero_mask) == 0 { 1 } else { 0 },
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
                let len = decoded.len;
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(len as u32));
                }
                Ok(len)
            }
            InstrKind::Cmpw | InstrKind::Cmpp => {
                let decoded = self
                    .decode_simple_operands_async(
                        state,
                        bus,
                        entry,
                        exec_pc,
                        prefix_len,
                        pre_modes_opt.as_ref(),
                        ticker,
                    )
                    .await?;
                let bits = if entry.kind == InstrKind::Cmpw { 16 } else { 24 };
                let mask = Self::mask_for_width(bits);
                let bytes = bits.div_ceil(8) as u64;
                let mem_lhs = decoded.mem;
                let mem_rhs = decoded.mem2;
                let has_mem_mem = mem_lhs.is_some() && mem_rhs.is_some();
                let has_mem_reg = mem_lhs.is_some() && decoded.reg3.is_some();
                let mut lhs = 0u32;
                let mut rhs = 0u32;
                if let (Some(m1), Some(m2)) = (mem_lhs, mem_rhs) {
                    lhs = Self::read_imm_async(bus, m1.addr, bits, ticker).await & mask;
                    rhs = Self::read_imm_async(bus, m2.addr, bits, ticker).await & mask;
                } else if let Some((r1, r2, _)) = decoded.reg_pair {
                    lhs = state.get_reg(r1) & mask;
                    rhs = state.get_reg(r2) & mask;
                } else if let Some(r) = decoded.reg3 {
                    if let Some(mem) = mem_lhs {
                        lhs = Self::read_imm_async(bus, mem.addr, bits, ticker).await & mask;
                        rhs = state.get_reg(r) & mask;
                    } else {
                        lhs = state.get_reg(r) & mask;
                    }
                }
                if has_mem_mem {
                    tick!(ticker);
                } else if has_mem_reg {
                    tick!(ticker, bytes);
                }
                Self::set_flags_cmp(state, lhs & mask, rhs & mask, bits);
                let len = decoded.len;
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(len as u32));
                }
                Ok(len)
            }
            InstrKind::Ex | InstrKind::Exl => {
                let decoded = self
                    .decode_simple_operands_async(
                        state,
                        bus,
                        entry,
                        exec_pc,
                        prefix_len,
                        pre_modes_opt.as_ref(),
                        ticker,
                    )
                    .await?;
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
                        let v1 = state.get_reg(dst_reg) & mask;
                        let v2 = state.get_reg(src_reg) & mask;
                        state.set_reg(dst_reg, v2);
                        state.set_reg(src_reg, v1);
                        let len = decoded.len;
                        if len == 1 {
                            tick!(ticker, 2);
                        }
                        if state.pc() == start_pc {
                            state.set_pc(start_pc.wrapping_add(len as u32));
                        }
                        finish!(Ok(len));
                    }
                }
                if let (Some(m1), Some(m2)) = (decoded.mem, decoded.mem2) {
                    let bits = m1.bits.min(m2.bits);
                    let v1 = Self::read_imm_async(bus, m1.addr, bits, ticker).await;
                    let bytes = bits.div_ceil(8);
                    let v2 = if bytes > 1 {
                        let first = Self::load_u8_async(bus, m2.addr, ticker).await as u32;
                        let mut value = first;
                        for i in 1..bytes {
                            let byte = bus.load(m2.addr + i as u32, 8) as u32;
                            value |= (byte & 0xFF) << (8 * i);
                        }
                        value & Self::mask_for_width(bits)
                    } else {
                        Self::read_imm_async(bus, m2.addr, bits, ticker).await
                    };
                    Self::store_imm_async(bus, m1.addr, bits, v2, ticker).await;
                    Self::store_imm_async(bus, m2.addr, bits, v1, ticker).await;
                    let len = decoded.len;
                    if state.pc() == start_pc {
                        state.set_pc(start_pc.wrapping_add(len as u32));
                    }
                    Ok(len)
                } else if let Some((r1, r2, bits)) = decoded.reg_pair {
                    let mask = Self::mask_for_width(bits);
                    let v1 = state.get_reg(r1) & mask;
                    let v2 = state.get_reg(r2) & mask;
                    state.set_reg(r1, v2);
                    state.set_reg(r2, v1);
                    tick!(ticker, 2);
                    let len = decoded.len;
                    if state.pc() == start_pc {
                        state.set_pc(start_pc.wrapping_add(len as u32));
                    }
                    Ok(len)
                } else {
                    let len = decoded.len.max(Self::estimated_length(entry));
                    if state.pc() == start_pc {
                        state.set_pc(start_pc.wrapping_add(len as u32));
                    }
                    Ok(len)
                }
            }
            InstrKind::Swap => {
                let decoded = self
                    .decode_simple_operands_async(
                        state,
                        bus,
                        entry,
                        exec_pc,
                        prefix_len,
                        pre_modes_opt.as_ref(),
                        ticker,
                    )
                    .await?;
                let val = state.get_reg(RegName::A) & 0xFF;
                let swapped = ((val & 0x0F) << 4) | ((val >> 4) & 0x0F);
                state.set_reg(RegName::A, swapped);
                state.set_reg(RegName::FZ, if swapped == 0 { 1 } else { 0 });
                tick!(ticker, 2);
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(decoded.len as u32));
                }
                Ok(decoded.len)
            }
            InstrKind::Shl | InstrKind::Shr | InstrKind::Rol | InstrKind::Ror => {
                let decoded = self
                    .decode_simple_operands_async(
                        state,
                        bus,
                        entry,
                        exec_pc,
                        prefix_len,
                        pre_modes_opt.as_ref(),
                        ticker,
                    )
                    .await?;
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
                        let val = Self::read_imm_async(bus, mem.addr, mem.bits, ticker).await;
                        (val, mem.bits, Some(mem), None)
                    } else {
                        finish!(Err("missing operand"));
                    };
                let mask = Self::mask_for_width(bits);
                let carry_in = state.get_reg(RegName::FC) & 1;
                let (res, carry_out) = match entry.kind {
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
                    // Baseline cycles: register shifts/rotates take an extra cycle.
                    tick!(ticker);
                } else if let Some(mem) = dest_mem {
                    if bits == 8 {
                        Self::store_traced(bus, mem.addr, bits, res & mask);
                    } else {
                        Self::store_imm_async(bus, mem.addr, bits, res & mask, ticker).await;
                    }
                }
                let carry_flag = match entry.kind {
                    InstrKind::Shl | InstrKind::Shr => carry_out,
                    InstrKind::Rol => ((val >> (bits.saturating_sub(1) as u32)) & 1) != 0,
                    InstrKind::Ror => (val & 1) != 0,
                    _ => false,
                };
                Self::set_flags_for_result(state, res & mask, Some(carry_flag));
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(decoded.len as u32));
                }
                Ok(decoded.len)
            }
            InstrKind::Dsll | InstrKind::Dsrl => {
                let decoded = self
                    .decode_simple_operands_async(
                        state,
                        bus,
                        entry,
                        exec_pc,
                        prefix_len,
                        pre_modes_opt.as_ref(),
                        ticker,
                    )
                    .await?;
                let mem = decoded.mem.ok_or("missing mem operand")?;
                if mem.bits != 8 {
                    finish!(Err("DSLL/DSRL only support byte operands"));
                }
                let length = state.get_reg(RegName::I) & mask_for(RegName::I);
                let mut addr = mem.addr;
                let is_left = entry.kind == InstrKind::Dsll;
                let mut carry_nibble: u8 = 0;
                let mut overall_zero: u8 = 0;
                for _ in 0..length {
                    let val = Self::load_u8_async(bus, addr, ticker).await;
                    let low = val & 0x0F;
                    let high = (val >> 4) & 0x0F;
                    let new_val = if is_left {
                        let res = (low << 4) | carry_nibble;
                        carry_nibble = low;
                        res
                    } else {
                        let res = high | (carry_nibble << 4);
                        carry_nibble = high;
                        res
                    };
                    Self::store_u8_async(bus, addr, new_val, ticker).await;
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
                state.set_reg(RegName::FZ, if overall_zero == 0 { 1 } else { 0 });
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(decoded.len as u32));
                }
                Ok(decoded.len)
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
                let decoded = self
                    .decode_simple_operands_async(
                        state,
                        bus,
                        entry,
                        exec_pc,
                        prefix_len,
                        pre_modes_opt.as_ref(),
                        ticker,
                    )
                    .await?;
                let transfer = decoded.transfer.ok_or("missing transfer operand")?;
                if entry.kind == InstrKind::Mvl {
                    let length = state.get_reg(RegName::I) & mask_for(RegName::I);
                    if length == 0 {
                        if state.pc() == start_pc {
                            state.set_pc(start_pc.wrapping_add(decoded.len as u32));
                        }
                        finish!(Ok(decoded.len));
                    }
                }
                let value =
                    Self::read_imm_async(bus, transfer.src_addr, transfer.bits, ticker).await;
                Self::store_imm_async(
                    bus,
                    transfer.dst_addr,
                    transfer.bits,
                    value,
                    ticker,
                )
                .await;
                if let Some((reg, new_val)) = transfer.side_effect {
                    state.set_reg(reg, new_val);
                }
                if matches!(entry.kind, InstrKind::Mv | InstrKind::Mvw | InstrKind::Mvp)
                    && matches!(
                        entry.operands[0],
                        OperandKind::EMemImemOffsetDestIntMem | OperandKind::EMemImemOffsetDestExtMem
                    )
                {
                    let bytes = transfer.bits.div_ceil(8) as u64;
                    let extra = 3u64.saturating_sub(bytes);
                    if extra > 0 {
                        tick!(ticker, extra);
                    }
                }
                if matches!(entry.kind, InstrKind::Mv)
                    && entry
                        .operands
                        .iter()
                        .any(|op| matches!(op, OperandKind::RegIMemOffset(_)))
                {
                    tick!(ticker);
                }
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(decoded.len as u32));
                }
                Ok(decoded.len)
            }
            InstrKind::Mv | InstrKind::Mvw | InstrKind::Mvp | InstrKind::Mvl | InstrKind::Mvld => {
                self.execute_mv_generic_async(
                    entry,
                    state,
                    bus,
                    exec_pc,
                    prefix_len,
                    pre_modes_opt.as_ref(),
                    ticker,
                )
                .await
            }
            InstrKind::Inc | InstrKind::Dec => {
                let decoded = self
                    .decode_simple_operands_async(
                        state,
                        bus,
                        entry,
                        exec_pc,
                        prefix_len,
                        pre_modes_opt.as_ref(),
                        ticker,
                    )
                    .await?;
                let op = entry.operands.first().ok_or("missing operand")?;
                if let Some(reg) = Self::resolved_reg(op, &decoded) {
                    let bits = match op {
                        OperandKind::Reg(_, b) => *b,
                        OperandKind::Reg3 => Self::reg3_bits(reg),
                        _ => 8,
                    };
                    match entry.kind {
                        InstrKind::Inc => {
                            Self::alu_unary(state, reg, bits, |v, _| v.wrapping_add(1));
                        }
                        InstrKind::Dec => {
                            Self::alu_unary(state, reg, bits, |v, _| v.wrapping_sub(1));
                        }
                        _ => finish!(Err("unsupported async inc/dec kind")),
                    }
                    tick!(ticker);
                } else if let Some(mem) = decoded.mem {
                    if mem.bits != 8 {
                        finish!(Err("unsupported async inc/dec mem width"));
                    }
                    let val = Self::load_u8_async(bus, mem.addr, ticker).await;
                    let res = match entry.kind {
                        InstrKind::Inc => val.wrapping_add(1),
                        InstrKind::Dec => val.wrapping_sub(1),
                        _ => finish!(Err("unsupported async inc/dec kind")),
                    };
                    Self::store_traced(bus, mem.addr, 8, res as u32);
                    Self::set_flags_for_result(state, res as u32, None);
                } else {
                    finish!(Err("missing operand"));
                }
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(decoded.len as u32));
                }
                Ok(decoded.len)
            }
            InstrKind::Adc if entry.name == "ADCL" => {
                let decoded = self
                    .decode_simple_operands_async(
                        state,
                        bus,
                        entry,
                        exec_pc,
                        prefix_len,
                        pre_modes_opt.as_ref(),
                        ticker,
                    )
                    .await?;
                let mem_dst = decoded.mem.ok_or("missing ADCL dst")?;
                if mem_dst.bits != 8 {
                    finish!(Err("unsupported async ADCL dst width"));
                }
                let mut dst_addr = mem_dst.addr;
                let mut src_addr = decoded.mem2.map(|m| m.addr);
                let src_reg = entry
                    .operands
                    .get(1)
                    .and_then(|op| Self::operand_reg(op));
                if src_addr.is_none() && src_reg != Some(RegName::A) {
                    finish!(Err("unsupported async ADCL src"));
                }
                let length = state.get_reg(RegName::I) & mask_for(RegName::I);
                if length == 0 {
                    if state.pc() == start_pc {
                        state.set_pc(start_pc.wrapping_add(decoded.len as u32));
                    }
                    finish!(Ok(decoded.len));
                }
                let dst_step = mem_dst.bits.div_ceil(8) as i32;
                let src_step = decoded
                    .mem2
                    .map(|m| m.bits.div_ceil(8) as i32)
                    .unwrap_or(dst_step);
                let mut overall_zero: u32 = 0;
                let mut carry = (state.get_reg(RegName::FC) & 1) != 0;

                for _ in 0..length {
                    let lhs = Self::load_u8_async(bus, dst_addr, ticker).await as u32;
                    let rhs = match src_addr {
                        Some(addr) => bus.load(addr, 8) as u32,
                        None => state.get_reg(RegName::A) & 0xFF,
                    };
                    let full = lhs as u64 + rhs as u64 + carry as u64;
                    let res = (full as u32) & 0xFF;
                    let new_carry = full > 0xFF;
                    Self::store_u8_async(bus, dst_addr, res as u8, ticker).await;
                    overall_zero |= res;
                    carry = new_carry;
                    if let Some(addr) = src_addr.as_mut() {
                        *addr = Self::advance_internal_addr_signed(*addr, src_step);
                    }
                    dst_addr = Self::advance_internal_addr_signed(dst_addr, dst_step);
                }

                state.set_reg(RegName::I, 0);
                state.set_reg(RegName::FC, if carry { 1 } else { 0 });
                state.set_reg(RegName::FZ, if (overall_zero & 0xFF) == 0 { 1 } else { 0 });
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(decoded.len as u32));
                }
                Ok(decoded.len)
            }
            InstrKind::Sbcl => {
                let decoded = self
                    .decode_simple_operands_async(
                        state,
                        bus,
                        entry,
                        exec_pc,
                        prefix_len,
                        pre_modes_opt.as_ref(),
                        ticker,
                    )
                    .await?;
                let mem_dst = decoded.mem.ok_or("missing SBCL dst")?;
                if mem_dst.bits != 8 {
                    finish!(Err("unsupported async SBCL dst width"));
                }
                let mut dst_addr = mem_dst.addr;
                let mut src_addr = decoded.mem2.map(|m| m.addr);
                let src_reg = entry
                    .operands
                    .get(1)
                    .and_then(|op| Self::operand_reg(op));
                if src_addr.is_none() && src_reg != Some(RegName::A) {
                    finish!(Err("unsupported async SBCL src"));
                }
                let length = state.get_reg(RegName::I) & mask_for(RegName::I);
                if length == 0 {
                    if state.pc() == start_pc {
                        state.set_pc(start_pc.wrapping_add(decoded.len as u32));
                    }
                    finish!(Ok(decoded.len));
                }
                let dst_step = mem_dst.bits.div_ceil(8) as i32;
                let src_step = decoded
                    .mem2
                    .map(|m| m.bits.div_ceil(8) as i32)
                    .unwrap_or(dst_step);
                let mut overall_zero: u32 = 0;
                let mut carry = (state.get_reg(RegName::FC) & 1) != 0;

                for _ in 0..length {
                    let lhs = Self::load_u8_async(bus, dst_addr, ticker).await as u32;
                    let rhs = match src_addr {
                        Some(addr) => bus.load(addr, 8) as u32,
                        None => state.get_reg(RegName::A) & 0xFF,
                    };
                    let borrow = (lhs as u64) < (rhs as u64 + carry as u64);
                    let res = lhs.wrapping_sub(rhs).wrapping_sub(carry as u32) & 0xFF;
                    Self::store_u8_async(bus, dst_addr, res as u8, ticker).await;
                    overall_zero |= res;
                    carry = borrow;
                    if let Some(addr) = src_addr.as_mut() {
                        *addr = Self::advance_internal_addr_signed(*addr, src_step);
                    }
                    dst_addr = Self::advance_internal_addr_signed(dst_addr, dst_step);
                }

                state.set_reg(RegName::I, 0);
                state.set_reg(RegName::FC, if carry { 1 } else { 0 });
                state.set_reg(RegName::FZ, if (overall_zero & 0xFF) == 0 { 1 } else { 0 });
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(decoded.len as u32));
                }
                Ok(decoded.len)
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
                if entry.ops_reversed.unwrap_or(false) {
                    finish!(Err("unsupported async ALU ops_reversed"));
                }
                if entry.operands.len() == 1 {
                    if !matches!(entry.operands[0], OperandKind::RegPair(_)) {
                        finish!(Err("unsupported async ALU form"));
                    }
                    if !matches!(entry.kind, InstrKind::Add | InstrKind::Sub) {
                        finish!(Err("unsupported async ALU regpair kind"));
                    }
                    let decoded = self
                        .decode_simple_operands_async(
                            state,
                            bus,
                            entry,
                            exec_pc,
                            prefix_len,
                            pre_modes_opt.as_ref(),
                            ticker,
                        )
                        .await?;
                    let (r1, r2, bits) = decoded.reg_pair.ok_or("missing regpair")?;
                    let mask = Self::mask_for_width(bits);
                    let lhs = state.get_reg(r1) & mask;
                    let rhs = state.get_reg(r2) & mask;
                    let (result, carry) = match entry.kind {
                        InstrKind::Add => {
                            let full = lhs as u64 + rhs as u64;
                            ((full as u32) & mask, Some(full > mask as u64))
                        }
                        InstrKind::Sub => {
                            let borrow = lhs < rhs;
                            (lhs.wrapping_sub(rhs) & mask, Some(borrow))
                        }
                        _ => finish!(Err("unsupported async ALU regpair kind")),
                    };
                    state.set_reg(r1, result);
                    Self::set_flags_for_result(state, result, carry);
                    let bytes = bits.div_ceil(8) as u64;
                    let extra_ticks = bytes.saturating_mul(2).saturating_sub(1);
                    tick!(ticker, extra_ticks);
                    if state.pc() == start_pc {
                        state.set_pc(start_pc.wrapping_add(decoded.len as u32));
                    }
                    finish!(Ok(decoded.len));
                }
                if entry.operands.len() != 2 {
                    finish!(Err("unsupported async ALU form"));
                }
                let decoded = self
                    .decode_simple_operands_async(
                        state,
                        bus,
                        entry,
                        exec_pc,
                        prefix_len,
                        pre_modes_opt.as_ref(),
                        ticker,
                    )
                    .await?;
                if decoded.mem.is_some()
                    && decoded.mem2.is_some()
                    && !matches!(
                        entry.kind,
                        InstrKind::And | InstrKind::Or | InstrKind::Xor | InstrKind::Cmp
                    )
                {
                    finish!(Err("unsupported async ALU multiple mem operands"));
                }

                let dst_op = entry.operands.get(0).ok_or("missing ALU dst")?;
                let src_op = entry.operands.get(1).ok_or("missing ALU src")?;
                let mask = Self::mask_for_width(8);
                let mut lhs_is_mem = false;
                let mem_lhs = decoded.mem;
                let mem_rhs = decoded.mem2;
                let mem_any = mem_lhs.or(mem_rhs);
                let lhs_val = if let Some(reg) = Self::operand_reg(dst_op) {
                    if reg != RegName::A {
                        finish!(Err("unsupported async ALU dst reg"));
                    }
                    state.get_reg(RegName::A) & mask
                } else if matches!(dst_op, OperandKind::IMem(_) | OperandKind::IMemWidth(_)) {
                    let mem_val = mem_any.ok_or("missing ALU mem")?;
                    if mem_val.bits != 8 {
                        finish!(Err("unsupported async ALU mem width"));
                    }
                    lhs_is_mem = true;
                    let val = Self::load_u8_async(bus, mem_val.addr, ticker).await;
                    u32::from(val)
                } else if matches!(dst_op, OperandKind::EMemAddrWidth(_) | OperandKind::EMemAddrWidthOp(_))
                {
                    let mem_val = mem_any.ok_or("missing ALU mem")?;
                    if mem_val.bits != 8 {
                        finish!(Err("unsupported async ALU mem width"));
                    }
                    lhs_is_mem = true;
                    let val = Self::load_u8_async(bus, mem_val.addr, ticker).await;
                    u32::from(val)
                } else {
                    finish!(Err("unsupported async ALU dst"));
                };

                let rhs_val = if let Some(reg) = Self::operand_reg(src_op) {
                    state.get_reg(reg) & mask
                } else if matches!(src_op, OperandKind::Imm(_)) {
                    decoded.imm.ok_or("missing ALU imm")?.0 & mask
                } else if matches!(src_op, OperandKind::IMem(_) | OperandKind::IMemWidth(_)) {
                    let mem_src = mem_rhs.or(mem_lhs).ok_or("missing ALU mem")?;
                    if mem_src.bits != 8 {
                        finish!(Err("unsupported async ALU mem width"));
                    }
                    let val = Self::load_u8_async(bus, mem_src.addr, ticker).await;
                    u32::from(val)
                } else if matches!(src_op, OperandKind::EMemAddrWidth(_) | OperandKind::EMemAddrWidthOp(_))
                {
                    let mem_src = mem_rhs.or(mem_lhs).ok_or("missing ALU mem")?;
                    if mem_src.bits != 8 {
                        finish!(Err("unsupported async ALU mem width"));
                    }
                    let val = Self::load_u8_async(bus, mem_src.addr, ticker).await;
                    u32::from(val)
                } else {
                    finish!(Err("unsupported async ALU src"));
                };

                let cmp_imem_imem = matches!(entry.kind, InstrKind::Cmp)
                    && matches!(dst_op, OperandKind::IMem(_) | OperandKind::IMemWidth(_))
                    && matches!(src_op, OperandKind::IMem(_) | OperandKind::IMemWidth(_));

                if !lhs_is_mem {
                    tick!(ticker);
                } else if matches!(entry.kind, InstrKind::Cmp)
                    && matches!(
                        src_op,
                        OperandKind::Reg(_, _)
                            | OperandKind::RegB
                            | OperandKind::RegIL
                            | OperandKind::RegIMR
                            | OperandKind::RegF
                            | OperandKind::Reg3
                    )
                {
                    tick!(ticker);
                } else if matches!(entry.kind, InstrKind::Test)
                    && matches!(
                        src_op,
                        OperandKind::Reg(_, _)
                            | OperandKind::RegB
                            | OperandKind::RegIL
                            | OperandKind::RegIMR
                            | OperandKind::RegF
                            | OperandKind::Reg3
                    )
                {
                    tick!(ticker);
                } else if cmp_imem_imem {
                    tick!(ticker);
                }

                let (result, carry) = match entry.kind {
                    InstrKind::Add => {
                        let full = (lhs_val as u64) + (rhs_val as u64);
                        ((full as u32) & mask, Some(full > mask as u64))
                    }
                    InstrKind::Sub => {
                        let borrow = (lhs_val & mask) < (rhs_val & mask);
                        ((lhs_val.wrapping_sub(rhs_val)) & mask, Some(borrow))
                    }
                    InstrKind::And => {
                        let res = (lhs_val & rhs_val) & mask;
                        (res, None)
                    }
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
                        Self::set_flags_cmp(state, lhs_val & mask, rhs_val & mask, 8);
                        (lhs_val.wrapping_sub(rhs_val) & mask, None)
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
                        let mem_dst = mem_any.ok_or("missing ALU mem")?;
                        Self::store_u8_async(bus, mem_dst.addr, result as u8, ticker).await;
                    } else {
                        state.set_reg(RegName::A, result);
                    }
                    Self::set_flags_for_result(state, result, carry);
                }

                if matches!(entry.kind, InstrKind::Cmp) {
                    if let Some(c) = carry {
                        state.set_reg(RegName::FC, if c { 1 } else { 0 });
                    }
                }

                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(decoded.len as u32));
                }
                Ok(decoded.len)
            }
            InstrKind::Unknown => {
                let est_len = Self::estimated_length(entry);
                fallback_unknown(state, prefix_len, Some(est_len))
            }
            _ => Err("unsupported async opcode"),
        };
        instr_guard.finish(self, bus);
        result
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

    async fn execute_mv_generic_async<B: LlamaBus>(
        &mut self,
        entry: &OpcodeEntry,
        state: &mut LlamaState,
        bus: &mut B,
        exec_pc: u32,
        prefix_len: u8,
        pre: Option<&PreModes>,
        ticker: &mut TickHelper<'_>,
    ) -> Result<u8, &'static str> {
        let prev_fc = state.get_reg(RegName::FC);
        let decoded = self
            .decode_simple_operands_async(state, bus, entry, exec_pc, prefix_len, pre, ticker)
            .await?;
        let extra_emem_reg_tick = matches!(entry.kind, InstrKind::Mv | InstrKind::Mvw | InstrKind::Mvp)
            && entry.operands.iter().any(|op| {
                matches!(
                    op,
                    OperandKind::EMemRegWidth(_) | OperandKind::EMemRegWidthMode(_)
                )
            });
        let extra_emem_imem_tick = entry.kind == InstrKind::Mv
            && entry
                .operands
                .iter()
                .any(|op| matches!(op, OperandKind::EMemIMemWidth(_)));
        let extra_mvl_emem_reg_tick =
            matches!(entry.kind, InstrKind::Mvl | InstrKind::Mvld)
                && entry.operands.iter().any(|op| {
                    matches!(
                        op,
                        OperandKind::EMemRegWidth(_)
                            | OperandKind::EMemRegWidthMode(_)
                            | OperandKind::EMemRegModePostPre
                    )
                });
        let mut mvl_length: Option<u32> = None;
        let mut mvl_skip_post_move = false;
        if matches!(entry.kind, InstrKind::Mvl | InstrKind::Mvld) {
            let length = state.get_reg(RegName::I) & mask_for(RegName::I);
            if length == 0 {
                for m in [decoded.mem, decoded.mem2].into_iter().flatten() {
                    if let Some((reg, new_val)) = m.side_effect {
                        let curr = state.get_reg(reg);
                        if new_val < curr {
                            state.set_reg(reg, new_val);
                        }
                    }
                }
                state.set_reg(RegName::FC, prev_fc);
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(decoded.len as u32));
                }
                return Ok(decoded.len);
            }
            mvl_length = Some(length);
            let mvl_emem_addr_imem = matches!(
                entry.operands.get(0),
                Some(OperandKind::EMemAddrWidth(_) | OperandKind::EMemAddrWidthOp(_))
            ) && matches!(
                entry.operands.get(1),
                Some(OperandKind::IMem(_) | OperandKind::IMemWidth(_))
            );
            if mvl_emem_addr_imem {
                // MVL [lmn],(n) base timing is one cycle longer than IMEM/IMEM.
                tick!(ticker);
                mvl_skip_post_move = true;
            }
            if extra_mvl_emem_reg_tick {
                tick!(ticker, 5);
            }
            if let (Some(mem_dst), Some(mem_src)) = (decoded.mem, decoded.mem2) {
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
                for idx in 0..length {
                    let is_last = idx + 1 == length;
                    let val = if is_last {
                        Self::read_imm(bus, src_addr, mem_dst.bits)
                    } else {
                        Self::read_imm_async(bus, src_addr, mem_dst.bits, ticker).await
                    };
                    if is_last {
                        Self::store_imm_no_tick(bus, dst_addr, mem_dst.bits, val);
                    } else {
                        Self::store_imm_async(bus, dst_addr, mem_dst.bits, val, ticker).await;
                    }
                    src_addr = Self::advance_internal_addr_signed(src_addr, src_step);
                    dst_addr = Self::advance_internal_addr_signed(dst_addr, dst_step);
                }
            }
            if mvl_skip_post_move {
                let pointer_steps = mvl_length.unwrap_or(1);
                for m in [decoded.mem, decoded.mem2].into_iter().flatten() {
                    if let Some((reg, new_val)) = m.side_effect {
                        Self::apply_pointer_side_effect(state, reg, new_val, pointer_steps);
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
        }
        if entry.operands.len() == 1 {
            if let Some((dst, src, _)) = decoded.reg_pair {
                let val = state.get_reg(src);
                state.set_reg(dst, val);
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(decoded.len as u32));
                }
                return Ok(decoded.len);
            }
            let start_pc = state.pc();
            if state.pc() == start_pc {
                state.set_pc(start_pc.wrapping_add(decoded.len as u32));
            }
            return Ok(decoded.len);
        }

        let dst_op = &entry.operands[0];
        let src_op = &entry.operands[1];
        let dst_reg = Self::resolved_reg(dst_op, &decoded);
        let src_reg = Self::resolved_reg(src_op, &decoded);
        let dst_is_il = matches!(dst_op, OperandKind::RegIL | OperandKind::Reg(RegName::IL, _));
        let extra_emem_addr_tick = matches!(entry.kind, InstrKind::Mv | InstrKind::Mvw | InstrKind::Mvp)
            && matches!(
                src_op,
                OperandKind::EMemAddrWidth(_) | OperandKind::EMemAddrWidthOp(_)
            )
            && dst_reg.is_some()
            && !dst_is_il;
        let src_emem_addr = matches!(
            src_op,
            OperandKind::EMemAddrWidth(_) | OperandKind::EMemAddrWidthOp(_)
        );
        let dst_imem = matches!(dst_op, OperandKind::IMem(_) | OperandKind::IMemWidth(_));
        let dst_emem_addr = matches!(
            dst_op,
            OperandKind::EMemAddrWidth(_) | OperandKind::EMemAddrWidthOp(_)
        );
        let single_tick_imem_store = matches!(entry.kind, InstrKind::Mv | InstrKind::Mvw | InstrKind::Mvp)
            && src_emem_addr
            && dst_imem;
        let no_tick_emem_store = matches!(entry.kind, InstrKind::Mv | InstrKind::Mvw | InstrKind::Mvp)
            && dst_emem_addr
            && matches!(src_op, OperandKind::IMem(_) | OperandKind::IMemWidth(_));
        let extra_imem_mem_tick = matches!(entry.kind, InstrKind::Mv | InstrKind::Mvw | InstrKind::Mvp)
            && dst_imem
            && matches!(src_op, OperandKind::IMem(_) | OperandKind::IMemWidth(_));

        let mut src_val: Option<(u32, u8)> = None;
        if let Some(reg) = src_reg {
            let bits = match src_op {
                OperandKind::Reg(_, bits) => *bits,
                OperandKind::RegB | OperandKind::RegIL | OperandKind::RegIMR | OperandKind::RegF => 8,
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
                let val = Self::read_imm_async(bus, mem.addr, bits, ticker).await;
                src_val = Some((val, bits));
            }
        }

        if let Some(reg) = dst_reg {
            let (val, bits) = src_val.ok_or("missing source")?;
            let masked = val & Self::mask_for_width(bits);
            state.set_reg(reg, masked);
            if reg == RegName::IL {
                tick!(ticker);
            }
            if reg == RegName::IMR {
                Self::store_traced(
                    bus,
                    INTERNAL_MEMORY_START + IMEM_IMR_OFFSET,
                    8,
                    masked & 0xFF,
                );
            }
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
            if single_tick_imem_store {
                Self::store_imm_internal_single_tick_async(bus, mem.addr, bits, val, ticker).await;
            } else if no_tick_emem_store {
                Self::store_imm_no_tick(bus, mem.addr, bits, val);
            } else {
                Self::store_imm_async(bus, mem.addr, bits, val, ticker).await;
            }
            if let Some((reg, new_val)) = mem.side_effect {
                if !matches!(entry.kind, InstrKind::Mvl | InstrKind::Mvld) {
                    state.set_reg(reg, new_val);
                }
            }
        } else {
            let (val, bits) = src_val.unwrap_or((0, 8));
            let masked = val & Self::mask_for_width(bits);
            if let Some(mem) = decoded.mem.or(decoded.mem2) {
                Self::store_imm_async(bus, mem.addr, mem.bits, masked, ticker).await;
                if let Some((reg, new_val)) = mem.side_effect {
                    if !matches!(entry.kind, InstrKind::Mvl | InstrKind::Mvld) {
                        state.set_reg(reg, new_val);
                    }
                }
            } else if let Some(reg) = dst_reg.or(src_reg) {
                state.set_reg(reg, masked);
            } else {
                state.set_reg(RegName::A, masked);
            }
            let start_pc = state.pc();
            if state.pc() == start_pc {
                state.set_pc(start_pc.wrapping_add(decoded.len as u32));
            }
            return Ok(decoded.len);
        }

        if extra_emem_reg_tick {
            tick!(ticker);
        }
        if extra_emem_addr_tick {
            tick!(ticker);
        }
        if extra_emem_imem_tick {
            tick!(ticker, 2);
        }
        if extra_imem_mem_tick {
            tick!(ticker);
        }
        let pointer_steps = mvl_length.unwrap_or(1);
        for m in [decoded.mem, decoded.mem2].into_iter().flatten() {
            if let Some((reg, new_val)) = m.side_effect {
                if Some(reg) != dst_reg {
                    Self::apply_pointer_side_effect(state, reg, new_val, pointer_steps);
                }
            }
        }
        if mvl_length.is_some() {
            state.set_reg(RegName::I, 0);
        }
        state.set_reg(RegName::FC, prev_fc);

        let start_pc = state.pc();
        if state.pc() == start_pc {
            state.set_pc(start_pc.wrapping_add(decoded.len as u32));
        }
        Ok(decoded.len)
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
        let mut mvl_length: Option<u32> = None;
        if matches!(entry.kind, InstrKind::Mvl | InstrKind::Mvld) {
            let length = state.get_reg(RegName::I) & mask_for(RegName::I);
            if length == 0 {
                // Apply pointer side-effects even when nothing moves; Python still updates
                // pre/post addressing registers for MVL with zero length, but only for
                // pre-dec modes (including wraparound).
                for m in [decoded.mem, decoded.mem2].into_iter().flatten() {
                    if let Some((reg, new_val)) = m.side_effect {
                        let mask = mask_for(reg);
                        let curr = state.get_reg(reg) & mask;
                        let step = m.bits.div_ceil(8) as u32;
                        let expected = curr.wrapping_sub(step) & mask;
                        if (new_val & mask) == expected {
                            state.set_reg(reg, expected);
                        }
                    }
                }
                state.set_reg(RegName::FC, prev_fc);
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(decoded.len as u32));
                }
                return Ok(decoded.len);
            }
            mvl_length = Some(length);
            if let (Some(mem_dst), Some(mem_src)) = (decoded.mem, decoded.mem2) {
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
                    let val = bus.load(src_addr, mem_dst.bits);
                    Self::store_traced(bus, dst_addr, mem_dst.bits, val);
                    src_addr = Self::advance_internal_addr_signed(src_addr, src_step);
                    dst_addr = Self::advance_internal_addr_signed(dst_addr, dst_step);
                }
            }
        }
        // Special-case RegPair-only move (e.g., opcode 0xFD)
        if entry.operands.len() == 1 {
            if let Some((dst, src, bits)) = decoded.reg_pair {
                // Parity: Python treats `MV` reg-pair as a full register move for the selected
                // registers (e.g. `MV Y, X` copies all 24 bits). Do not mask to the operand-width
                // annotation because opcode 0xFD uses the same encoding for 8/16/24-bit regs.
                let _ = bits;
                let val = state.get_reg(src);
                state.set_reg(dst, val);
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(decoded.len as u32));
                }
                return Ok(decoded.len);
            }
            // Fallback: advance past the instruction even if the pattern is unusual.
            let start_pc = state.pc();
            if state.pc() == start_pc {
                state.set_pc(start_pc.wrapping_add(decoded.len as u32));
            }
            return Ok(decoded.len);
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
                let val = bus.load(mem.addr, bits);
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
            // Generic move fallback: prefer decoded.mem as destination, otherwise mem2, otherwise A.
            let (val, bits) = src_val.unwrap_or((0, 8));
            let masked = val & Self::mask_for_width(bits);
            if let Some(mem) = decoded.mem.or(decoded.mem2) {
                Self::store_traced(bus, mem.addr, mem.bits, masked);
                if let Some((reg, new_val)) = mem.side_effect {
                    if !matches!(entry.kind, InstrKind::Mvl | InstrKind::Mvld) {
                        state.set_reg(reg, new_val);
                    }
                }
            } else if let Some(reg) = dst_reg.or(src_reg) {
                state.set_reg(reg, masked);
            } else {
                state.set_reg(RegName::A, masked);
            }
            let start_pc = state.pc();
            if state.pc() == start_pc {
                state.set_pc(start_pc.wrapping_add(decoded.len as u32));
            }
            return Ok(decoded.len);
        }

        // Apply any pointer side-effects even if the memory operand was a source.
        let pointer_steps = mvl_length.unwrap_or(1);
        for m in [decoded.mem, decoded.mem2].into_iter().flatten() {
            if let Some((reg, new_val)) = m.side_effect {
                if Some(reg) != dst_reg {
                    // Use the same signed-step logic as other multi-byte helpers so pre-dec
                    // addressing walks backward instead of wrapping forward.
                    Self::apply_pointer_side_effect(state, reg, new_val, pointer_steps);
                }
            }
        }
        if mvl_length.is_some() {
            state.set_reg(RegName::I, 0);
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
                    imm = Some(Self::read_imm(bus, pc + offset, *bits));
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
                                let val = bus.load(mem.addr, mem.bits);
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
                    bus.load(mem.addr, mem.bits) & mask
                } else {
                    state.get_reg(RegName::A) & mask
                };
                let rhs_val = if rhs_is_mem {
                    decoded
                        .mem2
                        .or(decoded.mem)
                        .map(|m| bus.load(m.addr, m.bits) & Self::mask_for_width(m.bits))
                        .unwrap_or(0)
                } else if let Some((imm, _)) = decoded.imm {
                    imm
                } else if let Some(r) = Self::resolved_reg(
                    entry.operands.get(1).unwrap_or(&OperandKind::Placeholder),
                    &decoded,
                ) {
                    Self::read_reg(state, bus, r)
                } else {
                    state.get_reg(RegName::A) & mask
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
            _ => {
                // Generic fallback: re-evaluate with simple ALU semantics and write back to lhs.
                let lhs_is_mem = matches!(
                    entry.operands.first().unwrap_or(&OperandKind::Placeholder),
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
                    bus.load(mem.addr, mem.bits) & mask
                } else {
                    state.get_reg(RegName::A) & mask
                };
                let rhs_val = if rhs_is_mem {
                    decoded
                        .mem2
                        .or(decoded.mem)
                        .map(|m| bus.load(m.addr, m.bits) & Self::mask_for_width(m.bits))
                        .unwrap_or(0)
                } else if let Some((imm, _)) = decoded.imm {
                    imm
                } else if let Some(r) = Self::resolved_reg(
                    entry.operands.get(1).unwrap_or(&OperandKind::Placeholder),
                    &decoded,
                ) {
                    Self::read_reg(state, bus, r)
                } else {
                    state.get_reg(RegName::A) & mask
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
                    _ => (lhs_val, None),
                };
                if lhs_is_mem {
                    Self::store_traced(bus, mem.addr, mem.bits, result);
                } else {
                    state.set_reg(RegName::A, result);
                }
                if !matches!(entry.kind, InstrKind::Cmp | InstrKind::Test) {
                    Self::set_flags_for_result(state, result, carry);
                }
            }
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
        let length = state.get_reg(RegName::I) & mask_for(RegName::I);
        if length == 0 {
            // Parity: zero-length loops are a no-op (no flag/pointer updates), only advance PC.
            let start_pc = state.pc();
            if state.pc() == start_pc {
                state.set_pc(start_pc.wrapping_add(decoded.len as u32));
            }
            return Ok(decoded.len);
        }
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
            let lhs = bus.load(dst_addr, mem_dst.bits) & mask_dst;
            let rhs = match src_addr {
                Some(addr) => bus.load(addr, src_bits) & mask_src,
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
            if (overall_zero & mask_dst) == 0 { 1 } else { 0 },
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
        // Keep IMR in sync with memory regardless of tracing so state is architecture-accurate.
        let mem_imr = with_imr_read_suppressed(|| bus.peek_imem_silent(IMEM_IMR_OFFSET));
        state.set_reg(RegName::IMR, mem_imr as u32);

        let instr_index = PERF_INSTR_COUNTER.fetch_add(1, Ordering::Relaxed);
        let start_pc = state.pc() & mask_for(RegName::PC);
        // For perfetto parity with Python, keep tracing anchored to the prefix byte/PC.
        let trace_pc_snapshot = start_pc;
        let trace_opcode_snapshot = opcode;
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
        // Execute using the resolved opcode after any PRE bytes.
        let mut exec_pc = start_pc;
        let mut exec_opcode = opcode;
        let mut prefix_len = 0u8;
        let mut pre_modes_opt: Option<PreModes> = None;
        let mut pc_override = None;
        let mut entry = self.lookup(exec_opcode);
        perfetto_reset_substep();

        while let Some(e) = entry {
            if e.kind != InstrKind::Pre {
                break;
            }
            let pre_modes = pre_modes_for(exec_opcode).ok_or("unknown PRE opcode")?;
            let next_pc = exec_pc.wrapping_add(1) & mask_for(RegName::PC);
            let next_opcode = bus.load(next_pc, 8) as u8;
            exec_opcode = next_opcode;
            exec_pc = next_pc;
            prefix_len = prefix_len.saturating_add(1);
            pre_modes_opt = Some(pre_modes);
            pc_override = Some(next_pc);
            entry = self.lookup(next_opcode);
        }

        PERF_CURRENT_OP.store(instr_index, Ordering::Relaxed);
        PERF_CURRENT_PC.store(trace_pc_snapshot, Ordering::Relaxed);
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
            ),
            None => fallback_unknown(state, prefix_len, None),
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
                // WAIT drains I to zero.
                let raw_i = state.get_reg(RegName::I) & mask_for(RegName::I);
                let wait_cycles = if raw_i == 0 {
                    mask_for(RegName::I) + 1
                } else {
                    raw_i
                };
                // If the host does not expose wait_cycles, tick timers/keyboard locally to avoid
                // stalling MTI/STI/KEYI.
                bus.wait_cycles(wait_cycles.max(1));
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
                    let length = state.get_reg(RegName::I) & mask_for(RegName::I);
                    if length == 0 {
                        let start_pc = state.pc();
                        if state.pc() == start_pc {
                            state.set_pc(start_pc.wrapping_add(decoded.len as u32));
                        }
                        return Ok(decoded.len);
                    }
                }
                let value = bus.load(transfer.src_addr, transfer.bits);
                Self::store_traced(bus, transfer.dst_addr, transfer.bits, value);
                if let Some((reg, new_val)) = transfer.side_effect {
                    state.set_reg(reg, new_val);
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
                    let val = bus.load(mem.addr, mem.bits);
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
                    let val = bus.load(mem.addr, mem.bits);
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
                let length = state.get_reg(RegName::I) & mask_for(RegName::I);
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
                    let dst_byte = (bus.load(dst_addr, mem_dst.bits) & 0xFF) as u8;
                    let src_byte = if let Some(bits) = src_bits {
                        let addr = src_addr.ok_or("missing source")?;
                        (bus.load(addr, bits) & 0xFF) as u8
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
                    if (overall_zero & zero_mask) == 0 {
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
                        (bus.load(mem.addr, mem.bits), mem.bits, Some(mem), None)
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
                let length = state.get_reg(RegName::I) & mask_for(RegName::I);
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
                        carry_nibble = low;
                        res
                    } else {
                        let res = high | (carry_nibble << 4);
                        carry_nibble = high;
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
                state.set_reg(RegName::FZ, if overall_zero == 0 { 1 } else { 0 });
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
                power_on_reset(bus, state);
                Ok(1 + prefix_len)
            }
            InstrKind::Pre => unreachable!("PRE should be handled before dispatch"),
            InstrKind::Unknown => {
                // Known-unknown opcodes should advance without failure to mirror Python decode.
                let est_len = Self::estimated_length(entry);
                fallback_unknown(state, prefix_len, Some(est_len))
            }
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
                // Push IMR, F, PC (little-endian) to match Python RETI stack layout, clear IRM, and jump to vector.
                let pc = state.pc().wrapping_add(instr_len as u32) & mask_for(RegName::PC);
                Self::push_stack(state, bus, RegName::S, pc, 24, false);
                let f = state.get_reg(RegName::F) & 0xFF;
                Self::push_stack(state, bus, RegName::S, f, 8, false);
                let imr_addr = INTERNAL_MEMORY_START + IMEM_IMR_OFFSET;
                let imr = bus.load(imr_addr, 8) & 0xFF;
                Self::push_stack(state, bus, RegName::S, imr, 8, false);
                // Clear IRM bit in IMR (bit 7)
                let cleared_imr = imr & 0x7F;
                Self::store_traced(bus, imr_addr, 8, cleared_imr);
                state.set_reg(RegName::IMR, cleared_imr);
                state.call_depth_inc();
                let vec = bus.load(INTERRUPT_VECTOR_ADDR, 8)
                    | (bus.load(INTERRUPT_VECTOR_ADDR + 1, 8) << 8)
                    | (bus.load(INTERRUPT_VECTOR_ADDR + 2, 8) << 16);
                state.set_pc(vec & mask_for(RegName::PC));
                // Parity: emit perfetto IRQ entry like Python IR intrinsic.
                let mut guard = crate::PERFETTO_TRACER.enter();
                guard.with_some(|tracer| {
                    let mut payload = std::collections::HashMap::new();
                    payload.insert(
                        "pc".to_string(),
                        AnnotationValue::Pointer((pc & mask_for(RegName::PC)) as u64),
                    );
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
                    AnnotationValue::Pointer(pc as u64),
                );
                cf_payload.insert("ret_addr".to_string(), AnnotationValue::Pointer(pc as u64));
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
                // Python TCL intrinsic is a no-op; just advance PC by length.
                let len = prefix_len + Self::estimated_length(entry);
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(len as u32));
                }
                Ok(len)
            }
            InstrKind::JpAbs => {
                // Absolute jump; operand may be 16-bit (low bits) or 20-bit.
                let pc_mask = mask_for(RegName::PC);
                let pc_before = state.pc() & pc_mask;
                let cond_ok = Self::cond_pass(entry, state);
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
                        target = Some(bus.load(mem.addr, mem.bits) & pc_mask);
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
                let cond_ok = Self::cond_pass(entry, state);
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
                let ret_addr = pc_before.wrapping_add(decoded.len as u32);
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
                // Stack layout: IMR (1), F(1), 24-bit PC (little-endian).
                let pc_mask = mask_for(RegName::PC);
                let pc_before = state.pc() & pc_mask;
                let mask_s = mask_for(RegName::S);
                let mut sp = state.get_reg(RegName::S) & mask_s;
                let _sp_before = sp;
                let imr = bus.load(sp, 8) & 0xFF;
                sp = sp.wrapping_add(1) & mask_s;
                let f = bus.load(sp, 8) & 0xFF;
                sp = sp.wrapping_add(1) & mask_s;
                let pc_lo = bus.load(sp, 8) & 0xFF;
                let pc_mid = bus.load(sp.wrapping_add(1) & mask_s, 8) & 0xFF;
                let pc_hi = bus.load(sp.wrapping_add(2) & mask_s, 8) & 0xFF;
                let ret = ((pc_hi << 16) | (pc_mid << 8) | pc_lo) & 0xFF_FFFF;
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
                state.set_pc(ret & 0xFFFFF);
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
                let lhs;
                let rhs;
                let bits: u8;
                if entry.operands.len() == 2 {
                    let op1 = &entry.operands[0];
                    let op2 = &entry.operands[1];
                    bits = match (op1, op2) {
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

                    lhs = if op_is_mem(op1) {
                        let mem = decoded.mem.ok_or("missing mem operand")?;
                        bus.load(mem.addr, mem.bits)
                    } else if op_is_imm(op1) {
                        decoded.imm.ok_or("missing immediate")?.0
                    } else if let Some(r) = Self::resolved_reg(op1, &decoded) {
                        Self::read_reg(state, bus, r)
                    } else {
                        decoded.imm.map(|v| v.0).unwrap_or(0)
                    };

                    rhs = if op_is_mem(op2) {
                        let mem = decoded.mem2.or(decoded.mem).ok_or("missing mem operand")?;
                        bus.load(mem.addr, mem.bits)
                    } else if op_is_imm(op2) {
                        decoded.imm.ok_or("missing immediate")?.0
                    } else if let Some(r) = Self::resolved_reg(op2, &decoded) {
                        Self::read_reg(state, bus, r)
                    } else {
                        decoded.imm.map(|v| v.0).unwrap_or(0)
                    };
                } else {
                    // Parity: Python decode does not emit other CMP/TEST operand shapes.
                    return Err("unsupported operand pattern");
                }
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
                // Compare wider operands (16/24 bits depending on mnemonic)
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
                    lhs = bus.load(m1.addr, bits) & mask;
                    rhs = bus.load(m2.addr, bits) & mask;
                } else if let Some((r1, r2, _)) = decoded.reg_pair {
                    lhs = state.get_reg(r1) & mask;
                    rhs = state.get_reg(r2) & mask;
                } else if let Some(r) = decoded.reg3 {
                    if let Some(mem) = decoded.mem {
                        // IMem20, Reg3 form: the first operand (memory) is the left-hand side.
                        lhs = bus.load(mem.addr, bits) & mask;
                        rhs = state.get_reg(r) & mask;
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
                    let v1 = bus.load(m1.addr, bits);
                    let v2 = bus.load(m2.addr, bits);
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
                    // Graceful no-op swap for unsupported patterns
                    let len = decoded.len.max(Self::estimated_length(entry));
                    let start_pc = state.pc();
                    if state.pc() == start_pc {
                        state.set_pc(start_pc.wrapping_add(len as u32));
                    }
                    Ok(len)
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
    use crate::async_driver::block_on;
    use crate::llama::async_eval::{AsyncLlamaExecutor, TickHelper};
    use crate::llama::opcodes::{InstrKind, OpcodeEntry, OPCODES};
    use crate::memory::ADDRESS_MASK;
    use crate::memory::MemoryImage;
    use crate::timer::TimerContext;
    use std::collections::{HashMap, HashSet};

    struct NullBus;
    impl LlamaBus for NullBus {
        fn load(&mut self, _addr: u32, _bits: u8) -> u32 {
            0
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
    }

    struct MapBus {
        mem: HashMap<u32, u8>,
    }

    impl MapBus {
        fn new() -> Self {
            Self {
                mem: HashMap::new(),
            }
        }

        fn set(&mut self, addr: u32, value: u8) {
            self.mem.insert(addr, value);
        }
    }

    impl LlamaBus for MapBus {
        fn load(&mut self, addr: u32, _bits: u8) -> u32 {
            *self.mem.get(&addr).unwrap_or(&0) as u32
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
    fn all_opcodes_execute_without_error() {
        let mut exec = LlamaExecutor::new();
        for entry in OPCODES {
            let mut state = LlamaState::new();
            state.set_pc(0);
            let mut bus = NullBus;
            let res = exec.execute(entry.opcode, &mut state, &mut bus);
            assert!(
                res.is_ok(),
                "opcode 0x{:02X} failed with {:?}",
                entry.opcode,
                res
            );
        }
    }

    #[test]
    fn unknown_opcodes_fallback_advances_pc() {
        let mut exec = LlamaExecutor::new();
        let mut state = LlamaState::new();
        let mut bus = NullBus;
        state.set_pc(0x10);
        let res = exec.execute(0x20, &mut state, &mut bus);
        assert!(
            res.is_ok(),
            "default mode should fall back on unknown opcodes"
        );
        assert_eq!(state.pc(), 0x11, "fallback should advance PC by length");
    }

    #[test]
    fn reset_vector_matches_python_irq_vector() {
        let mut state = LlamaState::new();
        let mut bus = ResetBus::new([0xAA, 0xBB, 0x0C]);
        power_on_reset(&mut bus, &mut state);
        // Expected little-endian vector at 0xFFFFD.
        assert_eq!(state.pc(), 0x0C_BB_AA & mask_for(RegName::PC));
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
        let _lock = crate::perfetto::perfetto_test_guard();
        // Program: PRE (0x32) followed by NOP (0x00)
        let mut bus = MemBus::with_size(4);
        bus.mem[0] = 0x32;
        bus.mem[1] = 0x00;

        let mut exec = LlamaExecutor::new();
        let mut state = LlamaState::new();
        state.set_pc(0);

        let path = std::env::temp_dir().join("llama_pref_trace.perfetto-trace");
        let _ = std::fs::remove_file(&path);
        let mut guard = crate::PERFETTO_TRACER.enter();
        guard.replace(Some(PerfettoTracer::new(path)));

        let len = exec
            .execute(0x32, &mut state, &mut bus)
            .expect("execute PRE+NOP");
        assert_eq!(len, 2, "PRE + NOP should consume two bytes");

        let tracer = guard.take().expect("tracer should be installed");
        let events = tracer.test_exec_events();
        assert!(
            !events.is_empty(),
            "perfetto tracer should record at least one Exec event"
        );
        let (pc, opcode, _idx) = events[0];
        assert_eq!(pc, 0, "Exec PC should use prefix address");
        assert_eq!(opcode, 0x32, "Exec opcode should reflect prefix byte");
    }

    struct OffsetBus {
        data: HashMap<u32, u8>,
    }

    impl OffsetBus {
        fn new() -> Self {
            Self {
                data: HashMap::new(),
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
            *self.data.get(&addr).unwrap_or(&0) as u32
        }
        fn store(&mut self, addr: u32, _bits: u8, value: u32) {
            self.data.insert(addr, (value & 0xFF) as u8);
        }
        fn resolve_emem(&mut self, base: u32) -> u32 {
            base
        }
    }

    #[test]
    fn emem_imem_offset_respects_sign() {
        let entry = dispatch::lookup(0xF0).expect("opcode present");
        let mut exec = LlamaExecutor::new();

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
        let mut exec = LlamaExecutor::new();
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

    #[test]
    fn wait_advances_pc() {
        let mut exec = LlamaExecutor::new();
        let mut state = LlamaState::new();
        let mut bus = NullBus;
        state.set_reg(RegName::FC, 1);
        state.set_reg(RegName::FZ, 1);
        state.set_reg(RegName::I, 5);
        let len = exec.execute(0xEF, &mut state, &mut bus).unwrap(); // WAIT
        assert_eq!(len, 1);
        assert_eq!(state.pc(), 1);
        assert_eq!(state.get_reg(RegName::I), 0);
        assert_eq!(state.get_reg(RegName::FC), 1, "WAIT should preserve C");
        assert_eq!(state.get_reg(RegName::FZ), 1, "WAIT should preserve Z");
    }

    struct WaitBus {
        spins: u32,
        calls: u32,
    }

    impl LlamaBus for WaitBus {
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
        state.set_reg(RegName::FC, 1);
        state.set_reg(RegName::FZ, 1);
        let len = exec.execute(0xEF, &mut state, &mut bus).unwrap(); // WAIT
        assert_eq!(len, 1);
        assert_eq!(state.pc(), 1);
        assert_eq!(bus.calls, 1, "WAIT should tick timers via wait_cycles");
        assert_eq!(bus.spins, 5, "WAIT should consume the requested cycles");
        assert_eq!(state.get_reg(RegName::I), 0);
        assert_eq!(state.get_reg(RegName::FC), 1);
        assert_eq!(state.get_reg(RegName::FZ), 1);
    }

    struct MemBus {
        mem: Vec<u8>,
    }
    impl MemBus {
        fn with_size(size: usize) -> Self {
            Self { mem: vec![0; size] }
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

        fn store(&mut self, addr: u32, bits: u8, value: u32) {
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

    struct ClampBus {
        mem: Vec<u8>,
        last_clamp: Option<bool>,
        finalize_chunks: u32,
    }

    impl ClampBus {
        fn with_size(size: usize) -> Self {
            Self {
                mem: vec![0; size],
                last_clamp: None,
                finalize_chunks: 0,
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

    impl LlamaBus for ClampBus {
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

        fn store(&mut self, addr: u32, bits: u8, value: u32) {
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

        fn peek_imem_silent(&mut self, offset: u32) -> u8 {
            *self.mem.get(offset as usize).unwrap_or(&0)
        }

        fn set_timer_finalize_clamp(&mut self, clamp: bool) {
            self.last_clamp = Some(clamp);
        }

        fn finalize_timer_chunk(&mut self) {
            self.finalize_chunks = self.finalize_chunks.saturating_add(1);
        }

        fn wait_cycles(&mut self, _cycles: u32) {}
    }

    struct TimerClampBus {
        periods: (u64, u64),
        trace: TimerTrace,
        finalize_chunks: u32,
    }

    impl TimerClampBus {
        fn new(periods: (u64, u64), trace: TimerTrace) -> Self {
            Self {
                periods,
                trace,
                finalize_chunks: 0,
            }
        }
    }

    impl LlamaBus for TimerClampBus {
        fn timer_trace(&mut self) -> Option<TimerTrace> {
            Some(self.trace)
        }

        fn timer_periods(&mut self) -> Option<(u64, u64)> {
            Some(self.periods)
        }

        fn finalize_timer_chunk(&mut self) {
            self.finalize_chunks = self.finalize_chunks.saturating_add(1);
        }
    }

    struct TimerBus {
        memory: MemoryImage,
        timer: TimerContext,
        cycle_count: u64,
        finalize_clamp: bool,
    }

    impl TimerBus {
        fn new(mti_period: u64, sti_period: u64, start_cycle: u64) -> Self {
            let mut timer = TimerContext::new(true, mti_period as i32, sti_period as i32);
            timer.next_mti = start_cycle.wrapping_add(1);
            if sti_period > 0 {
                timer.next_sti = start_cycle.wrapping_add(sti_period);
            }
            Self {
                memory: MemoryImage::new(),
                timer,
                cycle_count: start_cycle,
                finalize_clamp: true,
            }
        }

        fn tick_timers_only(&mut self, cycle: u64) {
            let _ = self.timer.tick_timers(&mut self.memory, cycle, None);
        }
    }

    impl LlamaBus for TimerBus {
        fn load(&mut self, addr: u32, bits: u8) -> u32 {
            self.memory.load(addr, bits).unwrap_or(0)
        }

        fn store(&mut self, addr: u32, bits: u8, value: u32) {
            let _ = self.memory.store(addr, bits, value);
        }

        fn resolve_emem(&mut self, base: u32) -> u32 {
            base & ADDRESS_MASK
        }

        fn peek_imem_silent(&mut self, offset: u32) -> u8 {
            self.memory.read_internal_byte(offset).unwrap_or(0)
        }

        fn timer_trace(&mut self) -> Option<TimerTrace> {
            let (mti, sti) = self.timer.tick_counts(self.cycle_count);
            Some(TimerTrace {
                mti_ticks: mti,
                sti_ticks: sti,
            })
        }

        fn timer_periods(&mut self) -> Option<(u64, u64)> {
            Some((self.timer.mti_period, self.timer.sti_period))
        }

        fn finalize_instruction(&mut self) {
            self.timer
                .finalize_instruction_with_clamp(self.cycle_count, self.finalize_clamp);
        }

        fn finalize_timer_chunk(&mut self) {
            self.timer
                .finalize_instruction_with_clamp(self.cycle_count, self.finalize_clamp);
        }

        fn set_timer_finalize_clamp(&mut self, clamp: bool) {
            self.finalize_clamp = clamp;
        }

        fn mark_instruction_start(&mut self) {
            self.timer.set_instruction_start_cycle(self.cycle_count);
        }

        fn wait_cycles(&mut self, _cycles: u32) {}
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
    fn cmpp_imem_reg_uses_mem_as_lhs_and_resets_carry_when_mem_ge_reg() {
        // Program bytes (little-endian 20-bit IMEM):
        //   D7           CMPP (IMem20), Reg3 (ops_reversed encoding order)
        //   04           Reg3 selector: X
        //   10 00 00     IMem addr = 0x000010
        let mut bus = MemBus {
            mem: vec![0xFF; 0x40],
        };
        bus.mem[..5].copy_from_slice(&[0xD7, 0x04, 0x10, 0x00, 0x00]);
        // IMem bytes remain 0xFF → lhs is maximal; X = 0x000080 (rhs) → borrow should be clear.

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
            "carry/borrow should clear when lhs >= rhs"
        );
        assert_eq!(state.get_reg(RegName::FZ) & 1, 0, "result is non-zero");
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
        let f_saved: u8 = 0xA5;
        let sp_start = 0x0200;
        state.set_pc(pc_start);
        state.set_reg(RegName::F, u32::from(f_saved));
        state.set_reg(RegName::S, sp_start);

        let len = exec.execute(0xFE, &mut state, &mut bus).unwrap(); // IR
        assert_eq!(len, 1);

        let expected_pc = (pc_start.wrapping_add(len as u32)) & mask_for(RegName::PC);
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
        let f_saved: u8 = 0x7C;
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
    fn stacked_pre_prefixes_decode_next_opcode() {
        // PRE + PRE + JP (16-bit) should consume both prefixes and fetch operands from the final opcode.
        let mut bus = MemBus::with_size(8);
        bus.mem[0] = 0x32; // PRE
        bus.mem[1] = 0x32; // PRE
        bus.mem[2] = 0x02; // JP abs16
        bus.mem[3] = 0x78;
        bus.mem[4] = 0x9A;
        let mut state = LlamaState::new();
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0x32, &mut state, &mut bus).unwrap();
        assert_eq!(len, 5);
        assert_eq!(state.pc(), 0x009A78);
    }

    #[test]
    fn stacked_pre_prefixes_unbounded() {
        // Four PRE bytes followed by JP should still resolve the JP and operands.
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
        let len = exec.execute(0x32, &mut state, &mut bus).unwrap();
        assert_eq!(len, 7);
        assert_eq!(state.pc(), 0x001234);
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
            .execute_with(0x54, &entry, &mut state, &mut bus, None, None, 0, 0)
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
        LlamaExecutor::push_stack(&mut state, &mut bus, RegName::S, 0xF0, 8, false);
        LlamaExecutor::push_stack(&mut state, &mut bus, RegName::S, 0xC3, 8, false);

        let sp = state.get_reg(RegName::S) as usize;
        assert_eq!(sp, 0x20 - 5);
        // Layout from low to high addresses after pushes: IMR, F, PC[23:16], PC[15:8], PC[7:0]
        assert_eq!(bus.mem[sp], 0xC3); // IMR
        assert_eq!(bus.mem[sp + 1], 0xF0); // F
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
        let ret_pc = ((pc_hi << 16) | (pc_mid << 8) | pc_lo) & 0xFF_FFFF;
        sp_iter = sp_iter.wrapping_add(3);

        assert_eq!(imr, 0xC3);
        assert_eq!(f, 0xF0);
        assert_eq!(ret_pc, pc & 0xFF_FFFF);
        assert_eq!(sp_iter as usize, 0x20);
    }

    #[test]
    fn call_stack_is_little_endian() {
        // CALL pushes return address low byte first (little-endian).
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

        let popped = LlamaExecutor::pop_stack(&mut state, &mut bus, RegName::S, 24, false);
        assert_eq!(popped, ret);
        assert_eq!(state.get_reg(RegName::S) as usize, 0x40);
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
        LlamaExecutor::push_stack(&mut state, &mut bus, RegName::S, 0xF0, 8, false);
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
        let sp = 0x40;
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

        let sp = 0x40;
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
        let _lock = crate::perfetto::perfetto_test_guard();

        let mut bus = MemBus::with_size(0x200);
        let imr_saved: u8 = 0xAA;
        bus.mem[IMEM_IMR_OFFSET as usize] = imr_saved;

        let sp = 0x40;
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
        assert_eq!(state.get_reg(RegName::Y), 0x023456);
        assert_eq!(state.pc(), 2);
    }

    #[test]
    fn mv_regpair_low_codes_map_to_ba_i_for_mv() {
        let mut bus = MemBus::with_size(4);
        bus.mem[0] = 0xFD;
        // Correctness: dst=BA (0), src=I (1) for MV regpair.
        bus.mem[1] = 0x01;
        let mut state = LlamaState::new();
        state.set_reg(RegName::BA, 0xFF00);
        state.set_reg(RegName::I, 0x0040);
        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0xFD, &mut state, &mut bus).unwrap();
        assert_eq!(len, 2);
        assert_eq!(state.get_reg(RegName::BA), 0x0040);
        assert_eq!(state.get_reg(RegName::A), 0x40);
        assert_eq!(state.get_reg(RegName::B), 0x00);
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
    fn jp_abs_16_with_pre_keeps_page() {
        // PRE (0x32), JP_Abs (0x02), imm16=0x9A78; PC page should be preserved.
        let mut bus = MemBus::with_size(0x400000);
        let start_pc = 0x34567;
        bus.mem[start_pc as usize] = 0x32; // PRE N,N
        bus.mem[start_pc as usize + 1] = 0x02; // JP abs (16-bit)
        bus.mem[start_pc as usize + 2] = 0x78;
        bus.mem[start_pc as usize + 3] = 0x9A;

        let mut state = LlamaState::new();
        state.set_pc(start_pc);

        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0x32, &mut state, &mut bus).unwrap();
        // Length should include prefix + opcode + imm16
        assert_eq!(len, 4);
        // Page (0x30000) comes from current PC; low bits from immediate.
        assert_eq!(state.pc(), 0x039A78);
    }

    #[test]
    fn jp_abs_16_prefixed_uses_instruction_page() {
        // Place PRE at 0x12FFFE so pc_override=pc+1 would wrap; we must keep the JP's own page.
        let mut bus = MemBus::with_size(0x400000);
        let start_pc = 0x12FFFE;
        let base = (start_pc - INTERNAL_MEMORY_START) as usize;
        bus.mem[base] = 0x32; // PRE
        bus.mem[base + 1] = 0x02; // JP abs (16-bit)
        bus.mem[base + 2] = 0x34;
        bus.mem[base + 3] = 0x12;

        let mut state = LlamaState::new();
        state.set_pc(start_pc);

        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0x32, &mut state, &mut bus).unwrap();
        assert_eq!(len, 4);
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
    fn external_address_advances_wrap_24bit() {
        // External post-inc/pre-dec addressing should stay masked to 24 bits like Python.
        let top = ADDRESS_MASK;
        assert_eq!(
            LlamaExecutor::advance_internal_addr_signed(top, 1),
            0x000000
        );
        assert_eq!(
            LlamaExecutor::advance_internal_addr_signed(0, -1),
            ADDRESS_MASK
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

        power_on_reset(&mut bus, &mut state);

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

        fn resolve_emem(&mut self, base: u32) -> u32 {
            base
        }
    }

    #[test]
    fn cmpw_prefixed_reads_coding_order_for_ops_reversed() {
        // PRE should be followed by the opcode byte, then the operands in coding order (ops_reversed flips them).
        // Bytes: [PRE 0x32][opcode 0xD6 CMPW][reg selector][IMem offset]
        let program = [0x32u8, 0xD6, 0xAB, 0x10];
        let mut bus = LoggingBus::with_bytes(&program);
        let mut state = LlamaState::new();
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();

        let _ = exec.execute(0x32, &mut state, &mut bus).unwrap();

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
    fn adcl_zero_length_is_noop() {
        let mut bus = MemBus::with_size(0x40);
        bus.mem[0] = 0x54; // opcode for context; execute() is driven by opcode param
        bus.mem[1] = 0x10;
        bus.mem[2] = 0x20;
        bus.mem[0x10] = 0x12;
        bus.mem[0x20] = 0x34;
        let mut state = LlamaState::new();
        state.set_pc(0);
        state.set_reg(RegName::I, 0);
        state.set_reg(RegName::FC, 1);
        state.set_reg(RegName::FZ, 0);
        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0x54, &mut state, &mut bus).unwrap();
        assert_eq!(len, 3, "ADCL length should match encoded bytes");
        assert_eq!(state.pc(), 3, "PC should advance by decoded length");
        assert_eq!(bus.mem[0x10], 0x12, "destination should remain unchanged");
        assert_eq!(bus.mem[0x20], 0x34, "source should remain unchanged");
        assert_eq!(
            state.get_reg(RegName::FC),
            1,
            "carry flag should be preserved"
        );
        assert_eq!(
            state.get_reg(RegName::FZ),
            0,
            "zero flag should be preserved"
        );
        assert_eq!(state.get_reg(RegName::I), 0, "I should remain zero");
    }

    #[test]
    fn tcl_is_no_op_and_advances_pc() {
        // TCL (0xCE) should be a no-op intrinsic: no stack/IMR changes, just PC advance.
        let mut bus = MemBus::with_size(0x200);
        bus.mem[IMEM_IMR_OFFSET as usize] = 0xAA;
        bus.mem[0] = 0xCE;
        let mut state = LlamaState::new();
        state.set_pc(0);
        state.set_reg(RegName::S, 0x0100);
        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0xCE, &mut state, &mut bus).unwrap();
        assert_eq!(len, 1);
        assert_eq!(state.pc(), 1, "PC should advance by instruction length");
        assert_eq!(state.get_reg(RegName::S), 0x0100, "stack pointer unchanged");
        assert_eq!(state.get_reg(RegName::IMR), 0xAA, "IMR unchanged");
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

    #[test]
    fn perfetto_last_pc_tracks_async_instruction_pc() {
        let _lock = PERFETTO_TRACER.enter();
        reset_perf_counters();
        let mut exec = AsyncLlamaExecutor::new();
        let mut state = LlamaState::new();
        state.set_pc(0x0123);
        let mut bus = MemBus::with_size(0x0200);
        bus.mem[0x0123] = 0x00; // NOP
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let _ = block_on(exec.execute(0x00, &mut state, &mut bus, &mut ticker))
            .expect("execute async NOP");

        assert_eq!(
            perfetto_last_pc(),
            0x0123 & mask_for(RegName::PC),
            "perfetto_last_pc should reflect the executed instruction PC"
        );
    }

    #[test]
    fn async_decode_ext_reg_ptr_ticks() {
        let mut bus = MapBus::new();
        let pc = 0x1000;
        bus.set(pc, 0x04); // X register, simple mode
        let mut state = LlamaState::new();
        state.set_reg(RegName::X, 0x001234);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let (mem, consumed) = block_on(exec.decode_ext_reg_ptr_async(
            &mut state,
            &mut bus,
            pc,
            1,
            &mut ticker,
        ))
        .expect("decode ext reg ptr async");

        assert_eq!(consumed, 1);
        assert_eq!(mem.addr, 0x001234);
        assert_eq!(cycles, 1);
    }

    #[test]
    fn async_decode_ext_reg_ptr_offset_ticks() {
        let mut bus = MapBus::new();
        let pc = 0x1000;
        bus.set(pc, 0x84); // offset +, X register
        bus.set(pc + 1, 0x05);
        let mut state = LlamaState::new();
        state.set_reg(RegName::X, 0x001000);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let (mem, consumed) = block_on(exec.decode_ext_reg_ptr_async(
            &mut state,
            &mut bus,
            pc,
            1,
            &mut ticker,
        ))
        .expect("decode ext reg ptr offset async");

        assert_eq!(consumed, 2);
        assert_eq!(mem.addr, 0x001005);
        assert_eq!(cycles, 3);
    }

    #[test]
    fn async_decode_ext_reg_ptr_predec_ticks() {
        let mut bus = MapBus::new();
        let pc = 0x1000;
        bus.set(pc, 0x34); // pre-dec, X register
        let mut state = LlamaState::new();
        state.set_reg(RegName::X, 0x001000);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let (mem, consumed) = block_on(exec.decode_ext_reg_ptr_async(
            &mut state,
            &mut bus,
            pc,
            1,
            &mut ticker,
        ))
        .expect("decode ext reg ptr predec async");

        assert_eq!(consumed, 1);
        assert_eq!(mem.addr, 0x000FFF);
        assert_eq!(cycles, 2);
    }

    #[test]
    fn async_decode_ext_reg_ptr_offset_clamps_timer_chunk() {
        let mut bus = ClampBus::with_size(0x200);
        bus.mem[0] = 0x84; // offset +, X register
        bus.mem[1] = 0x05;
        let mut state = LlamaState::new();
        state.set_reg(RegName::X, 0x001000);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let _ = block_on(exec.decode_ext_reg_ptr_async(
            &mut state,
            &mut bus,
            0,
            1,
            &mut ticker,
        ))
        .expect("decode ext reg ptr offset async");

        assert_eq!(bus.finalize_chunks, 1);
    }

    #[test]
    fn async_decode_ext_reg_ptr_predec_clamps_timer_chunk() {
        let mut bus = ClampBus::with_size(0x200);
        bus.mem[0] = 0x34; // pre-dec, X register
        let mut state = LlamaState::new();
        state.set_reg(RegName::X, 0x001000);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let _ = block_on(exec.decode_ext_reg_ptr_async(
            &mut state,
            &mut bus,
            0,
            1,
            &mut ticker,
        ))
        .expect("decode ext reg ptr predec async");

        assert_eq!(bus.finalize_chunks, 1);
    }

    #[test]
    fn async_decode_imem_ptr_ticks() {
        let mut bus = MapBus::new();
        let pc = 0x2000;
        bus.set(pc, 0x00); // mode byte
        bus.set(pc + 1, 0x10); // IMEM base
        bus.set(INTERNAL_MEMORY_START + IMEM_BP_OFFSET, 0);
        bus.set(INTERNAL_MEMORY_START + IMEM_PX_OFFSET, 0);
        bus.set(INTERNAL_MEMORY_START + IMEM_PY_OFFSET, 0);
        bus.set(INTERNAL_MEMORY_START + 0x10, 0x34);
        bus.set(INTERNAL_MEMORY_START + 0x11, 0x12);
        bus.set(INTERNAL_MEMORY_START + 0x12, 0x00);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let (mem, consumed) = block_on(exec.decode_imem_ptr_async(
            &mut bus,
            pc,
            1,
            AddressingMode::N,
            &mut ticker,
        ))
        .expect("decode imem ptr async");

        assert_eq!(consumed, 2);
        assert_eq!(mem.addr, 0x001234);
        assert_eq!(cycles, 5);
    }

    #[test]
    fn async_decode_imem_ptr_offset_clamps_timer_chunk() {
        let mut bus = ClampBus::with_size(0x400);
        bus.mem[0] = 0x80; // mode byte with offset +
        bus.mem[1] = 0x10; // IMEM base
        bus.mem[2] = 0x05; // displacement
        bus.mem[IMEM_BP_OFFSET as usize] = 0;
        bus.mem[IMEM_PX_OFFSET as usize] = 0;
        bus.mem[IMEM_PY_OFFSET as usize] = 0;
        bus.mem[0x10] = 0x34;
        bus.mem[0x11] = 0x12;
        bus.mem[0x12] = 0x00;
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let _ = block_on(exec.decode_imem_ptr_async(
            &mut bus,
            0,
            1,
            AddressingMode::N,
            &mut ticker,
        ))
        .expect("decode imem ptr offset async");

        assert_eq!(bus.finalize_chunks, 1);
    }

    #[test]
    fn async_decode_emem_imem_offset_ticks() {
        let entry = OpcodeEntry {
            opcode: 0x00,
            kind: InstrKind::Mv,
            name: "MV",
            cond: None,
            ops_reversed: None,
            operands: &[],
        };
        let mut bus = MapBus::new();
        let pc = 0x3000;
        bus.set(pc, 0x00); // mode byte
        bus.set(pc + 1, 0x10); // IMEM dst
        bus.set(pc + 2, 0x20); // IMEM src ptr
        bus.set(INTERNAL_MEMORY_START + IMEM_BP_OFFSET, 0);
        bus.set(INTERNAL_MEMORY_START + IMEM_PX_OFFSET, 0);
        bus.set(INTERNAL_MEMORY_START + IMEM_PY_OFFSET, 0);
        bus.set(INTERNAL_MEMORY_START + 0x20, 0x78);
        bus.set(INTERNAL_MEMORY_START + 0x21, 0x56);
        bus.set(INTERNAL_MEMORY_START + 0x22, 0x34);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let (transfer, consumed) = block_on(exec.decode_emem_imem_offset_async(
            &entry,
            &mut bus,
            pc,
            AddressingMode::N,
            AddressingMode::N,
            true,
            &mut ticker,
        ))
        .expect("decode emem/imem offset async");

        assert_eq!(consumed, 3);
        assert_eq!(transfer.bits, 8);
        assert_eq!(transfer.dst_addr, INTERNAL_MEMORY_START + 0x10);
        assert_eq!(transfer.src_addr, 0x00345678);
        assert_eq!(cycles, 6);
    }

    #[test]
    fn async_decode_emem_imem_offset_clamps_timer_chunk() {
        let entry = OpcodeEntry {
            opcode: 0x00,
            kind: InstrKind::Mv,
            name: "MV",
            cond: None,
            ops_reversed: None,
            operands: &[],
        };
        let mut bus = ClampBus::with_size(0x400);
        bus.mem[0] = 0x80; // mode byte with offset +
        bus.mem[1] = 0x10; // IMEM dst
        bus.mem[2] = 0x20; // IMEM src ptr
        bus.mem[3] = 0x05; // displacement
        bus.mem[IMEM_BP_OFFSET as usize] = 0;
        bus.mem[IMEM_PX_OFFSET as usize] = 0;
        bus.mem[IMEM_PY_OFFSET as usize] = 0;
        bus.mem[0x20] = 0x78;
        bus.mem[0x21] = 0x56;
        bus.mem[0x22] = 0x34;
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let _ = block_on(exec.decode_emem_imem_offset_async(
            &entry,
            &mut bus,
            0,
            AddressingMode::N,
            AddressingMode::N,
            true,
            &mut ticker,
        ))
        .expect("decode emem/imem offset async");

        assert_eq!(bus.finalize_chunks, 1);
    }

    #[test]
    fn maybe_finalize_timer_chunk_skips_when_chunk_does_not_cross_period() {
        let mut exec = LlamaExecutor::new();
        let mut bus = TimerClampBus::new(
            (2048, 5000),
            TimerTrace {
                mti_ticks: 100,
                sti_ticks: 200,
            },
        );
        exec.chunk_start_timer = Some(bus.trace);

        exec.maybe_finalize_timer_chunk(&mut bus, 2);

        assert_eq!(bus.finalize_chunks, 0);
    }

    #[test]
    fn maybe_finalize_timer_chunk_clamps_when_chunk_crosses_period() {
        let mut exec = LlamaExecutor::new();
        let mut bus = TimerClampBus::new(
            (2048, 0),
            TimerTrace {
                mti_ticks: 2047,
                sti_ticks: 0,
            },
        );
        exec.chunk_start_timer = Some(bus.trace);

        exec.maybe_finalize_timer_chunk(&mut bus, 2);

        assert_eq!(bus.finalize_chunks, 1);
    }

    #[test]
    fn execute_async_mv_reg_imm_ticks() {
        let mut bus = MapBus::new();
        let pc = 0x1000;
        bus.set(pc + 1, 0xAB);
        let mut state = LlamaState::new();
        state.set_pc(pc);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x08, &mut state, &mut bus, &mut ticker))
            .expect("execute async mv reg imm");

        assert_eq!(len, 2);
        assert_eq!(state.get_reg(RegName::A) & 0xFF, 0xAB);
        assert_eq!(state.pc(), pc + 2);
        assert_eq!(cycles, 2);
    }

    #[test]
    fn execute_async_mv_il_imm_ticks() {
        let mut bus = MapBus::new();
        let pc = 0x1000;
        bus.set(pc + 1, 0xAB);
        let mut state = LlamaState::new();
        state.set_pc(pc);
        state.set_reg(RegName::I, 0x1200);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x09, &mut state, &mut bus, &mut ticker))
            .expect("execute async mv il imm");

        assert_eq!(len, 2);
        assert_eq!(state.get_reg(RegName::IL) & 0xFF, 0xAB);
        assert_eq!(state.get_reg(RegName::IH) & 0xFF, 0x00);
        assert_eq!(state.pc(), pc + 2);
        assert_eq!(cycles, 3);
    }

    #[test]
    fn execute_async_nop_advances_pc() {
        let mut bus = MapBus::new();
        let pc = 0x2000;
        let mut state = LlamaState::new();
        state.set_pc(pc);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x00, &mut state, &mut bus, &mut ticker))
            .expect("execute async nop");

        assert_eq!(len, 1);
        assert_eq!(state.pc(), pc + 1);
        assert_eq!(cycles, 1);
    }

    #[test]
    fn execute_async_mv_reg_imem_ticks() {
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0x80; // MV A, IMem8
        bus.mem[1] = 0x10;
        bus.mem[0x10] = 0x22;
        let mut state = LlamaState::new();
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x80, &mut state, &mut bus, &mut ticker))
            .expect("execute async mv reg imem");

        assert_eq!(len, 2);
        assert_eq!(state.get_reg(RegName::A) & 0xFF, 0x22);
        assert_eq!(state.pc(), 2);
        assert_eq!(cycles, 3);
    }

    #[test]
    fn execute_async_mv_imem_reg_ticks() {
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0xA0; // MV IMem8, A
        bus.mem[1] = 0x20;
        let mut state = LlamaState::new();
        state.set_reg(RegName::A, 0x33);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0xA0, &mut state, &mut bus, &mut ticker))
            .expect("execute async mv imem reg");

        assert_eq!(len, 2);
        assert_eq!(bus.mem[0x20], 0x33);
        assert_eq!(state.pc(), 2);
        assert_eq!(cycles, 3);
    }

    #[test]
    fn execute_async_mv_x_emem_addr_ticks() {
        let mut bus = MemBus::with_size(0x200);
        bus.mem[0] = 0x8C; // MV X, [lmn]
        bus.mem[1] = 0x10;
        bus.mem[2] = 0x00;
        bus.mem[3] = 0x00; // address 0x000010
        bus.mem[0x10] = 0x11;
        bus.mem[0x11] = 0x22;
        bus.mem[0x12] = 0x33;
        let mut state = LlamaState::new();
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x8C, &mut state, &mut bus, &mut ticker))
            .expect("execute async mv x emem addr");

        assert_eq!(len, 4);
        assert_eq!(state.get_reg(RegName::X), 0x0003_2211);
        assert_eq!(state.pc(), 4);
        assert_eq!(cycles, 8);
    }

    #[test]
    fn execute_async_mv_a_emem_addr_width1_ticks() {
        let mut bus = MemBus::with_size(0x200);
        bus.mem[0] = 0x88; // MV A, [lmn]
        bus.mem[1] = 0x10;
        bus.mem[2] = 0x00;
        bus.mem[3] = 0x00; // address 0x000010
        bus.mem[0x10] = 0x44;
        let mut state = LlamaState::new();
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x88, &mut state, &mut bus, &mut ticker))
            .expect("execute async mv a emem addr width1");

        assert_eq!(len, 4);
        assert_eq!(state.get_reg(RegName::A) & 0xFF, 0x44);
        assert_eq!(state.pc(), 4);
        assert_eq!(cycles, 6);
    }

    #[test]
    fn execute_async_mv_il_emem_addr_width1_ticks() {
        let mut bus = MemBus::with_size(0x200);
        bus.mem[0] = 0x89; // MV IL, [lmn]
        bus.mem[1] = 0x10;
        bus.mem[2] = 0x00;
        bus.mem[3] = 0x00; // address 0x000010
        bus.mem[0x10] = 0x55;
        let mut state = LlamaState::new();
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x89, &mut state, &mut bus, &mut ticker))
            .expect("execute async mv il emem addr width1");

        assert_eq!(len, 4);
        assert_eq!(state.get_reg(RegName::IL) & 0xFF, 0x55);
        assert_eq!(state.pc(), 4);
        assert_eq!(cycles, 6);
    }

    #[test]
    fn execute_async_mv_imem_imm_ticks() {
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0xCC; // MV IMem8, imm8
        bus.mem[1] = 0x21;
        bus.mem[2] = 0x44;
        let mut state = LlamaState::new();
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0xCC, &mut state, &mut bus, &mut ticker))
            .expect("execute async mv imem imm");

        assert_eq!(len, 3);
        assert_eq!(bus.mem[0x21], 0x44);
        assert_eq!(state.pc(), 3);
        assert_eq!(cycles, 3);
    }

    #[test]
    fn execute_async_mv_imem_imm_pre_mode_ticks() {
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0x30; // PRE (N, BpN)
        bus.mem[1] = 0xCC; // MV IMem8, imm8
        bus.mem[2] = IMEM_IMR_OFFSET as u8;
        bus.mem[3] = 0x12;
        let mut state = LlamaState::new();
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x30, &mut state, &mut bus, &mut ticker))
            .expect("execute async mv imem imm pre mode");

        assert_eq!(len, 4);
        assert_eq!(bus.mem[IMEM_IMR_OFFSET as usize], 0x12);
        assert_eq!(state.pc(), 4);
        assert_eq!(cycles, 4);
    }

    #[test]
    fn execute_async_pre_mv_imem_imem_ticks() {
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0x32; // PRE (N, N)
        bus.mem[1] = 0xC8; // MV IMem8, IMem8
        bus.mem[2] = 0x10;
        bus.mem[3] = 0x20;
        bus.mem[0x20] = 0x77;
        let mut state = LlamaState::new();
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x32, &mut state, &mut bus, &mut ticker))
            .expect("execute async pre mv imem imem");

        assert_eq!(len, 4);
        assert_eq!(bus.mem[0x10], 0x77);
        assert_eq!(state.pc(), 4);
        assert_eq!(cycles, 7);
    }

    #[test]
    fn execute_async_prefix_timer_clamp_respects_imr() {
        for (imr, expect_clamp) in [
            (0u8, true),
            (0x01u8, true),
            (0x80u8, true),
            (0x81u8, true),
        ] {
            let mut bus = ClampBus::with_size(0x200);
            bus.mem[0] = 0x30; // PRE (N, BpN)
            bus.mem[1] = 0x71; // AND IMem8, imm8
            bus.mem[2] = 0x10;
            bus.mem[3] = 0x01;
            bus.mem[0x10] = 0x02;
            bus.mem[IMEM_IMR_OFFSET as usize] = imr;
            let mut state = LlamaState::new();
            state.set_pc(0);
            let mut exec = LlamaExecutor::new();
            let mut cycles = 0u64;
            let mut ticker = TickHelper::new(&mut cycles, false, None);

            block_on(exec.execute_async(0x30, &mut state, &mut bus, &mut ticker))
                .expect("execute async prefixed and");

            assert_eq!(bus.last_clamp, Some(expect_clamp));
        }
    }

    #[test]
    fn execute_async_conditional_jump_clamp_is_enabled() {
        for (imr, fz, expect_clamp) in [
            (0u8, 0u8, true),   // taken -> clamp remains enabled
            (0x80u8, 0u8, true), // taken (IMR master on) -> clamp remains enabled
            (0u8, 1u8, true),   // not taken -> clamp remains enabled
        ] {
            let mut bus = ClampBus::with_size(0x200);
            bus.mem[0] = 0x1B; // JRNZ -n
            bus.mem[1] = 0x01; // displacement
            bus.mem[IMEM_IMR_OFFSET as usize] = imr;
            let mut state = LlamaState::new();
            state.set_pc(0);
            state.set_reg(RegName::FZ, fz as u32);
            let mut exec = LlamaExecutor::new();
            let mut cycles = 0u64;
            let mut ticker = TickHelper::new(&mut cycles, false, None);

            block_on(exec.execute_async(0x1B, &mut state, &mut bus, &mut ticker))
                .expect("execute async jrnz");

            assert_eq!(bus.last_clamp, Some(expect_clamp));
        }
    }

    #[test]
    fn execute_async_conditional_jump_does_not_mark_timer_chunk() {
        let mut bus = ClampBus::with_size(0x200);
        bus.mem[0] = 0x1B; // JRNZ -n
        bus.mem[1] = 0x01;
        let mut state = LlamaState::new();
        state.set_pc(0);
        state.set_reg(RegName::FZ, 0); // take branch
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        block_on(exec.execute_async(0x1B, &mut state, &mut bus, &mut ticker))
            .expect("execute async jrnz");

        assert_eq!(bus.finalize_chunks, 1);
    }

    #[test]
    fn execute_async_mvw_imem_imem_ticks() {
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0xC9; // MVW IMem16, IMem16
        bus.mem[1] = 0x10;
        bus.mem[2] = 0x12;
        bus.mem[0x12] = 0xEF;
        bus.mem[0x13] = 0xBE;
        let mut state = LlamaState::new();
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0xC9, &mut state, &mut bus, &mut ticker))
            .expect("execute async mvw imem imem");

        assert_eq!(len, 3);
        assert_eq!(bus.mem[0x10], 0xEF);
        assert_eq!(bus.mem[0x11], 0xBE);
        assert_eq!(state.pc(), 3);
        assert_eq!(cycles, 8);
    }

    #[test]
    fn execute_async_mvw_imem_imem_bp_mode_ticks() {
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0x22; // PRE (BpN, N)
        bus.mem[1] = 0xC9; // MVW IMem16, IMem16
        bus.mem[2] = 0x03; // dest raw
        bus.mem[3] = 0xD4; // src raw
        bus.mem[IMEM_BP_OFFSET as usize] = 0x10;
        bus.mem[0xD4] = 0xEF;
        bus.mem[0xD5] = 0xBE;
        let mut state = LlamaState::new();
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x22, &mut state, &mut bus, &mut ticker))
            .expect("execute async mvw imem imem bp mode");

        assert_eq!(len, 4);
        assert_eq!(bus.mem[0x13], 0xEF);
        assert_eq!(bus.mem[0x14], 0xBE);
        assert_eq!(state.pc(), 4);
        assert_eq!(cycles, 9);
    }

    #[test]
    fn execute_async_mv_imem_emem_addr_ticks() {
        let mut bus = MemBus::with_size(0x200);
        bus.mem[0] = 0xD0; // MV IMem8, [lmn]
        bus.mem[1] = 0x10; // dest raw
        bus.mem[2] = 0x30;
        bus.mem[3] = 0x00;
        bus.mem[4] = 0x00; // src address 0x000030
        bus.mem[0x30] = 0xAB;
        let mut state = LlamaState::new();
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0xD0, &mut state, &mut bus, &mut ticker))
            .expect("execute async mv imem emem addr");

        assert_eq!(len, 5);
        assert_eq!(bus.mem[0x10], 0xAB);
        assert_eq!(state.pc(), 5);
        assert_eq!(cycles, 7);
    }

    #[test]
    fn execute_async_mvw_imem_emem_addr_ticks() {
        let mut bus = MemBus::with_size(0x200);
        bus.mem[0] = 0xD1; // MVW IMem16, [lmn]
        bus.mem[1] = 0x10; // dest raw
        bus.mem[2] = 0x40;
        bus.mem[3] = 0x00;
        bus.mem[4] = 0x00; // src address 0x000040
        bus.mem[0x40] = 0xEF;
        bus.mem[0x41] = 0xBE;
        let mut state = LlamaState::new();
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0xD1, &mut state, &mut bus, &mut ticker))
            .expect("execute async mvw imem emem addr");

        assert_eq!(len, 5);
        assert_eq!(bus.mem[0x10], 0xEF);
        assert_eq!(bus.mem[0x11], 0xBE);
        assert_eq!(state.pc(), 5);
        assert_eq!(cycles, 8);
    }

    #[test]
    fn execute_async_mvp_imem_emem_addr_ticks() {
        let mut bus = MemBus::with_size(0x200);
        bus.mem[0] = 0xD2; // MVP IMem20, [lmn]
        bus.mem[1] = 0x10; // dest raw
        bus.mem[2] = 0x50;
        bus.mem[3] = 0x00;
        bus.mem[4] = 0x00; // src address 0x000050
        bus.mem[0x50] = 0x11;
        bus.mem[0x51] = 0x22;
        bus.mem[0x52] = 0x33;
        let mut state = LlamaState::new();
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0xD2, &mut state, &mut bus, &mut ticker))
            .expect("execute async mvp imem emem addr");

        assert_eq!(len, 5);
        assert_eq!(bus.mem[0x10], 0x11);
        assert_eq!(bus.mem[0x11], 0x22);
        assert_eq!(bus.mem[0x12], 0x33);
        assert_eq!(state.pc(), 5);
        assert_eq!(cycles, 9);
    }

    #[test]
    fn execute_async_mv_emem_addr_imem_ticks() {
        let mut bus = MemBus::with_size(0x200);
        bus.mem[0] = 0xD8; // MV [lmn], IMem8
        bus.mem[1] = 0x30;
        bus.mem[2] = 0x00;
        bus.mem[3] = 0x00; // dest address 0x000030
        bus.mem[4] = 0x10; // src IMEM
        bus.mem[0x10] = 0xAB;
        let mut state = LlamaState::new();
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0xD8, &mut state, &mut bus, &mut ticker))
            .expect("execute async mv emem addr imem");

        assert_eq!(len, 5);
        assert_eq!(bus.mem[0x30], 0xAB);
        assert_eq!(state.pc(), 5);
        assert_eq!(cycles, 6);
    }

    #[test]
    fn execute_async_mv_emem_addr_imem_uses_first_pre_mode() {
        let mut bus = MemBus::with_size(0x2000);
        bus.mem[0] = 0x30; // PRE (N, BpN)
        bus.mem[1] = 0xD8; // MV [klm], (n)
        bus.mem[2] = 0x00;
        bus.mem[3] = 0x10;
        bus.mem[4] = 0x00; // dest address 0x001000
        bus.mem[5] = 0xE0; // src IMEM
        bus.mem[IMEM_BP_OFFSET as usize] = 0x10;
        bus.mem[0xE0] = 0xAA; // raw (n)
        bus.mem[0xF0] = 0xBB; // BP+n (should not be used)
        let mut state = LlamaState::new();
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x30, &mut state, &mut bus, &mut ticker))
            .expect("execute async mv emem addr imem with pre");

        assert_eq!(len, 6);
        assert_eq!(bus.mem[0x1000], 0xAA);
        assert_eq!(state.pc(), 6);
    }

    #[test]
    fn execute_async_mvw_emem_addr_imem_ticks() {
        let mut bus = MemBus::with_size(0x200);
        bus.mem[0] = 0xD9; // MVW [lmn], IMem16
        bus.mem[1] = 0x40;
        bus.mem[2] = 0x00;
        bus.mem[3] = 0x00; // dest address 0x000040
        bus.mem[4] = 0x10; // src IMEM
        bus.mem[0x10] = 0xEF;
        bus.mem[0x11] = 0xBE;
        let mut state = LlamaState::new();
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0xD9, &mut state, &mut bus, &mut ticker))
            .expect("execute async mvw emem addr imem");

        assert_eq!(len, 5);
        assert_eq!(bus.mem[0x40], 0xEF);
        assert_eq!(bus.mem[0x41], 0xBE);
        assert_eq!(state.pc(), 5);
        assert_eq!(cycles, 7);
    }

    #[test]
    fn execute_async_mvp_emem_addr_imem_ticks() {
        let mut bus = MemBus::with_size(0x200);
        bus.mem[0] = 0xDA; // MVP [lmn], IMem20
        bus.mem[1] = 0x50;
        bus.mem[2] = 0x00;
        bus.mem[3] = 0x00; // dest address 0x000050
        bus.mem[4] = 0x10; // src IMEM
        bus.mem[0x10] = 0x11;
        bus.mem[0x11] = 0x22;
        bus.mem[0x12] = 0x33;
        let mut state = LlamaState::new();
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0xDA, &mut state, &mut bus, &mut ticker))
            .expect("execute async mvp emem addr imem");

        assert_eq!(len, 5);
        assert_eq!(bus.mem[0x50], 0x11);
        assert_eq!(bus.mem[0x51], 0x22);
        assert_eq!(bus.mem[0x52], 0x03);
        assert_eq!(state.pc(), 5);
        assert_eq!(cycles, 8);
    }

    #[test]
    fn execute_async_mvl_imem_imem_ticks() {
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0xCB; // MVL IMem8, IMem8
        bus.mem[1] = 0x10;
        bus.mem[2] = 0x20;
        bus.mem[0x20] = 0x11;
        bus.mem[0x21] = 0x22;
        let mut state = LlamaState::new();
        state.set_pc(0);
        state.set_reg(RegName::I, 2);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0xCB, &mut state, &mut bus, &mut ticker))
            .expect("execute async mvl imem imem");

        assert_eq!(len, 3);
        assert_eq!(bus.mem[0x10], 0x11);
        assert_eq!(bus.mem[0x11], 0x22);
        assert_eq!(state.get_reg(RegName::I), 0);
        assert_eq!(state.pc(), 3);
        assert_eq!(cycles, 7);
    }

    #[test]
    fn execute_async_mvl_emem_addr_imem_ticks() {
        let mut bus = MemBus::with_size(0x200);
        bus.mem[0] = 0xDB; // MVL [lmn], IMem8
        bus.mem[1] = 0x40;
        bus.mem[2] = 0x00;
        bus.mem[3] = 0x00; // dest address 0x000040
        bus.mem[4] = 0x10; // src IMEM
        bus.mem[0x10] = 0x11;
        bus.mem[0x11] = 0x22;
        bus.mem[0x12] = 0x33;
        let mut state = LlamaState::new();
        state.set_pc(0);
        state.set_reg(RegName::I, 3);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0xDB, &mut state, &mut bus, &mut ticker))
            .expect("execute async mvl emem addr imem");

        assert_eq!(len, 5);
        assert_eq!(bus.mem[0x40], 0x11);
        assert_eq!(bus.mem[0x41], 0x22);
        assert_eq!(bus.mem[0x42], 0x33);
        assert_eq!(state.get_reg(RegName::I), 0);
        assert_eq!(state.pc(), 5);
        assert_eq!(cycles, 10);
    }

    #[test]
    fn execute_async_mvl_imem_emem_reg_ticks() {
        let mut bus = MemBus::with_size(0x200);
        bus.mem[0] = 0xE3; // MVL IMem8, EMemReg (mode byte)
        bus.mem[1] = 0x24; // post-inc, reg=X
        bus.mem[2] = 0x10; // IMEM destination offset
        bus.mem[0x20] = 0xAA;
        bus.mem[0x21] = 0xBB;
        let mut state = LlamaState::new();
        state.set_pc(0);
        state.set_reg(RegName::I, 2);
        state.set_reg(RegName::X, 0x20);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0xE3, &mut state, &mut bus, &mut ticker))
            .expect("execute async mvl imem emem reg");

        assert_eq!(len, 3);
        assert_eq!(bus.mem[0x10], 0xAA);
        assert_eq!(bus.mem[0x11], 0xBB);
        assert_eq!(state.get_reg(RegName::X), 0x22);
        assert_eq!(state.get_reg(RegName::I), 0);
        assert_eq!(state.pc(), 3);
        assert_eq!(cycles, 12);
    }

    #[test]
    fn execute_async_mvl_emem_reg_imem_updates_pointer_by_length() {
        let mut bus = MemBus::with_size(0x200);
        bus.mem[0] = 0xEB; // MVL EMemReg, IMem8
        bus.mem[1] = 0x24; // post-inc, reg=X
        bus.mem[2] = 0x10; // IMEM source offset
        bus.mem[0x10] = 0xAA;
        bus.mem[0x11] = 0xBB;
        bus.mem[0x12] = 0xCC;
        let mut state = LlamaState::new();
        state.set_pc(0);
        state.set_reg(RegName::I, 3);
        state.set_reg(RegName::X, 0x20);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0xEB, &mut state, &mut bus, &mut ticker))
            .expect("execute async mvl emem reg imem");

        assert_eq!(len, 3);
        assert_eq!(bus.mem[0x20], 0xAA);
        assert_eq!(bus.mem[0x21], 0xBB);
        assert_eq!(bus.mem[0x22], 0xCC);
        assert_eq!(state.get_reg(RegName::X), 0x23);
        assert_eq!(state.get_reg(RegName::I), 0);
        assert_eq!(state.pc(), 3);
    }

    #[test]
    fn execute_mvl_emem_reg_imem_updates_pointer_by_length() {
        let mut bus = MemBus::with_size(0x200);
        bus.mem[0] = 0xEB; // MVL EMemReg, IMem8
        bus.mem[1] = 0x24; // post-inc, reg=X
        bus.mem[2] = 0x10; // IMEM source offset
        bus.mem[0x10] = 0xAA;
        bus.mem[0x11] = 0xBB;
        bus.mem[0x12] = 0xCC;
        let mut state = LlamaState::new();
        state.set_pc(0);
        state.set_reg(RegName::I, 3);
        state.set_reg(RegName::X, 0x20);
        let mut exec = LlamaExecutor::new();

        let len = exec
            .execute(0xEB, &mut state, &mut bus)
            .expect("execute mvl emem reg imem");

        assert_eq!(len, 3);
        assert_eq!(bus.mem[0x20], 0xAA);
        assert_eq!(bus.mem[0x21], 0xBB);
        assert_eq!(bus.mem[0x22], 0xCC);
        assert_eq!(state.get_reg(RegName::X), 0x23);
        assert_eq!(state.get_reg(RegName::I), 0);
        assert_eq!(state.pc(), 3);
    }

    #[test]
    fn execute_async_mv_reg_imem_offset_ticks() {
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0xE0; // MV RegIMemOffset (dest IMem)
        bus.mem[1] = 0x04; // X register, simple mode
        bus.mem[2] = 0x30;
        bus.mem[0x20] = 0x5A;
        let mut state = LlamaState::new();
        state.set_reg(RegName::X, 0x20);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0xE0, &mut state, &mut bus, &mut ticker))
            .expect("execute async mv reg/imem offset");

        assert_eq!(len, 3);
        assert_eq!(bus.mem[0x30], 0x5A);
        assert_eq!(state.get_reg(RegName::X), 0x20);
        assert_eq!(state.pc(), 3);
        assert_eq!(cycles, 6);
    }

    #[test]
    fn execute_async_mv_reg_imem_offset_predec_ticks() {
        let mut bus = MemBus::with_size(0x200);
        bus.mem[0] = 0x30; // PRE (N, BpN)
        bus.mem[1] = 0xE8; // MV RegIMemOffset (dest RegOffset)
        bus.mem[2] = 0x36; // U register, predec mode
        bus.mem[3] = 0xD4; // raw IMEM
        bus.mem[IMEM_BP_OFFSET as usize] = 0x10;
        bus.mem[0xD4] = 0xAB; // raw IMEM (pre_1)
        let mut state = LlamaState::new();
        state.set_reg(RegName::U, 0x20);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x30, &mut state, &mut bus, &mut ticker))
            .expect("execute async mv reg/imem offset predec");

        assert_eq!(len, 4);
        assert_eq!(bus.mem[0x1F], 0xAB);
        assert_eq!(state.get_reg(RegName::U), 0x1F);
        assert_eq!(state.pc(), 4);
        assert_eq!(cycles, 8);
    }

    #[test]
    fn execute_async_mvp_reg_imem_offset_predec_ticks() {
        let mut bus = MemBus::with_size(0x200);
        bus.mem[0] = 0x30; // PRE (N, BpN)
        bus.mem[1] = 0xEA; // MVP RegIMemOffset (dest RegOffset)
        bus.mem[2] = 0x36; // U register, predec mode
        bus.mem[3] = 0xD4; // raw IMEM
        bus.mem[IMEM_BP_OFFSET as usize] = 0x10;
        bus.mem[0xD4] = 0x11; // raw IMEM (pre_1)
        bus.mem[0xD5] = 0x22;
        bus.mem[0xD6] = 0x33;
        let mut state = LlamaState::new();
        state.set_reg(RegName::U, 0x20);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x30, &mut state, &mut bus, &mut ticker))
            .expect("execute async mvp reg/imem offset predec");

        assert_eq!(len, 4);
        assert_eq!(bus.mem[0x1D], 0x11);
        assert_eq!(bus.mem[0x1E], 0x22);
        assert_eq!(bus.mem[0x1F], 0x33);
        assert_eq!(state.get_reg(RegName::U), 0x1D);
        assert_eq!(state.pc(), 4);
        assert_eq!(cycles, 10);
    }

    #[test]
    fn execute_async_mvp_reg_imem_offset_postinc_ticks() {
        let mut bus = MemBus::with_size(0x200);
        bus.mem[0] = 0x30; // PRE (N, BpN)
        bus.mem[1] = 0xE2; // MVP RegIMemOffset (dest IMem)
        bus.mem[2] = 0x27; // S register, post-inc mode
        bus.mem[3] = 0xDD; // raw IMEM destination offset
        bus.mem[0x100] = 0x11;
        bus.mem[0x101] = 0x22;
        bus.mem[0x102] = 0x33;
        let mut state = LlamaState::new();
        state.set_reg(RegName::S, 0x100);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x30, &mut state, &mut bus, &mut ticker))
            .expect("execute async mvp reg/imem offset postinc");

        assert_eq!(len, 4);
        assert_eq!(bus.mem[0xDD], 0x11);
        assert_eq!(bus.mem[0xDE], 0x22);
        assert_eq!(bus.mem[0xDF], 0x33);
        assert_eq!(state.get_reg(RegName::S), 0x103);
        assert_eq!(state.pc(), 4);
        assert_eq!(cycles, 9);
    }

    #[test]
    fn execute_async_mv_reg_imem_offset_disp_ticks() {
        let mut bus = MemBus::with_size(0x200);
        bus.mem[0] = 0x30; // PRE (N, BpN)
        bus.mem[1] = 0xE0; // MV RegIMemOffset (dest IMem)
        bus.mem[2] = 0x85; // offset +, reg=Y
        bus.mem[3] = 0xD6; // raw IMEM
        bus.mem[4] = 0x03; // displacement
        bus.mem[0x103] = 0xAB; // external source @ Y + 3
        let mut state = LlamaState::new();
        state.set_reg(RegName::Y, 0x100);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x30, &mut state, &mut bus, &mut ticker))
            .expect("execute async mv reg/imem offset disp");

        assert_eq!(len, 5);
        assert_eq!(bus.mem[0xD6], 0xAB);
        assert_eq!(state.get_reg(RegName::Y), 0x100);
        assert_eq!(state.pc(), 5);
        assert_eq!(cycles, 9);
    }

    #[test]
    fn execute_async_mvp_reg_imem_offset_with_disp_ticks() {
        let mut bus = MemBus::with_size(0x300);
        bus.mem[0] = 0xE2; // MVP RegIMemOffset (dest IMem)
        bus.mem[1] = 0x84; // offset +, reg=X
        bus.mem[2] = 0x00; // IMEM destination offset
        bus.mem[3] = 0x05; // displacement
        bus.mem[0x25] = 0x11;
        bus.mem[0x26] = 0x22;
        bus.mem[0x27] = 0x33;
        let mut state = LlamaState::new();
        state.set_reg(RegName::X, 0x20);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0xE2, &mut state, &mut bus, &mut ticker))
            .expect("execute async mvp reg/imem offset with disp");

        assert_eq!(len, 4);
        assert_eq!(bus.mem[0x00], 0x11);
        assert_eq!(bus.mem[0x01], 0x22);
        assert_eq!(bus.mem[0x02], 0x33);
        assert_eq!(state.get_reg(RegName::X), 0x20);
        assert_eq!(state.pc(), 4);
        assert_eq!(cycles, 10);
    }

    #[test]
    fn execute_async_mv_regpair_low_codes_map_to_ba_i() {
        let mut bus = MemBus::with_size(0x10);
        bus.mem[0] = 0xFD; // MV RegPair
        bus.mem[1] = 0x01; // dst=BA, src=I
        let mut state = LlamaState::new();
        state.set_reg(RegName::BA, 0xFF00);
        state.set_reg(RegName::I, 0x0040);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0xFD, &mut state, &mut bus, &mut ticker))
            .expect("execute async mv regpair low codes");

        assert_eq!(len, 2);
        assert_eq!(state.get_reg(RegName::BA), 0x0040);
        assert_eq!(state.get_reg(RegName::A), 0x40);
        assert_eq!(state.get_reg(RegName::B), 0x00);
        assert_eq!(state.pc(), 2);
    }

    #[test]
    fn execute_async_mv_a_from_imem_pointer_ticks() {
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0x98; // MV A, [(n)]
        bus.mem[1] = 0x00; // simple mode
        bus.mem[2] = 0x10; // IMEM base
        bus.mem[0x10] = 0x30;
        bus.mem[0x11] = 0x00;
        bus.mem[0x12] = 0x00;
        bus.mem[0x30] = 0x55;
        let mut state = LlamaState::new();
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x98, &mut state, &mut bus, &mut ticker))
            .expect("execute async mv a, [(n)]");

        assert_eq!(len, 3);
        assert_eq!(state.get_reg(RegName::A) & 0xFF, 0x55);
        assert_eq!(state.pc(), 3);
        assert_eq!(cycles, 9);
    }

    #[test]
    fn execute_async_mv_a_from_imem_pointer_offset_ticks() {
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0x98; // MV A, [(n)]
        bus.mem[1] = 0x80; // offset mode (+disp)
        bus.mem[2] = 0x10; // IMEM base
        bus.mem[3] = 0x02; // displacement
        bus.mem[0x10] = 0x30;
        bus.mem[0x11] = 0x00;
        bus.mem[0x12] = 0x00;
        bus.mem[0x32] = 0x77;
        let mut state = LlamaState::new();
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x98, &mut state, &mut bus, &mut ticker))
            .expect("execute async mv a, [(n)] offset");

        assert_eq!(len, 4);
        assert_eq!(state.get_reg(RegName::A) & 0xFF, 0x77);
        assert_eq!(state.pc(), 4);
        assert_eq!(cycles, 11);
    }

    #[test]
    fn execute_async_mv_emem_imem_offset_dest_int_ticks() {
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0xF0; // MV (m), [(n)]
        bus.mem[1] = 0x00; // mode simple
        bus.mem[2] = 0x05; // dst IMEM
        bus.mem[3] = 0x10; // ptr IMEM
        bus.mem[0x10] = 0x30;
        bus.mem[0x11] = 0x00;
        bus.mem[0x12] = 0x00;
        bus.mem[0x30] = 0xAB;
        let mut state = LlamaState::new();
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0xF0, &mut state, &mut bus, &mut ticker))
            .expect("execute async mv emem/imem offset dest int");

        assert_eq!(len, 4);
        assert_eq!(bus.mem[0x05], 0xAB);
        assert_eq!(state.pc(), 4);
        assert_eq!(cycles, 11);
    }

    #[test]
    fn execute_async_mv_emem_imem_offset_dest_int_disp_ticks() {
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0xF0; // MV (m), [(n)]
        bus.mem[1] = 0x80; // mode = positive offset
        bus.mem[2] = 0x05; // dst IMEM
        bus.mem[3] = 0x10; // ptr IMEM
        bus.mem[4] = 0x02; // displacement
        bus.mem[0x10] = 0x30;
        bus.mem[0x11] = 0x00;
        bus.mem[0x12] = 0x00;
        bus.mem[0x32] = 0xAB;
        let mut state = LlamaState::new();
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0xF0, &mut state, &mut bus, &mut ticker))
            .expect("execute async mv emem/imem offset dest int disp");

        assert_eq!(len, 5);
        assert_eq!(bus.mem[0x05], 0xAB);
        assert_eq!(state.pc(), 5);
        assert_eq!(cycles, 13);
    }

    #[test]
    fn execute_async_mv_emem_imem_offset_dest_ext_ticks() {
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0xF8; // MV [(n)], (m)
        bus.mem[1] = 0x00; // mode simple
        bus.mem[2] = 0x10; // ptr IMEM
        bus.mem[3] = 0x05; // src IMEM
        bus.mem[0x10] = 0x40;
        bus.mem[0x11] = 0x00;
        bus.mem[0x12] = 0x00;
        bus.mem[0x05] = 0xCD;
        let mut state = LlamaState::new();
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0xF8, &mut state, &mut bus, &mut ticker))
            .expect("execute async mv emem/imem offset dest ext");

        assert_eq!(len, 4);
        assert_eq!(bus.mem[0x40], 0xCD);
        assert_eq!(state.pc(), 4);
        assert_eq!(cycles, 11);
    }

    #[test]
    fn execute_async_mv_emem_imem_offset_clamps_mti_phase() {
        let start_cycle = 2_000_000u64;
        let mut bus = TimerBus::new(2048, 0, start_cycle);
        // Program: MV [(n)],A with offset mode
        bus.memory.write_external_byte(0x0000, 0xB8);
        bus.memory.write_external_byte(0x0001, 0x80); // mode byte (offset +)
        bus.memory.write_external_byte(0x0002, 0x00); // IMEM base
        bus.memory.write_external_byte(0x0003, 0x02); // displacement
        bus.memory.write_internal_byte(IMEM_BP_OFFSET, 0x00);
        bus.memory.write_internal_byte(IMEM_PX_OFFSET, 0x00);
        bus.memory.write_internal_byte(IMEM_PY_OFFSET, 0x00);
        // Pointer at IMEM[0x00] -> 0x001234
        bus.memory.write_internal_byte(0x00, 0x34);
        bus.memory.write_internal_byte(0x01, 0x12);
        bus.memory.write_internal_byte(0x02, 0x00);

        let mut state = LlamaState::new();
        state.set_pc(0);
        state.set_reg(RegName::A, 0xBE);

        let mut exec = AsyncLlamaExecutor::new();
        let bus_ptr: *mut TimerBus = &mut bus;
        let mut cycle_count = bus.cycle_count;
        let mut tick_cb = move |cycle| unsafe {
            (*bus_ptr).cycle_count = cycle;
            (*bus_ptr).tick_timers_only(cycle);
        };
        let mut ticker = TickHelper::new(&mut cycle_count, true, Some(&mut tick_cb));

        let len = block_on(exec.execute(0xB8, &mut state, &mut bus, &mut ticker))
            .expect("execute async mv emem/imem offset");

        bus.cycle_count = cycle_count;
        assert_eq!(len, 4);
        assert_eq!(bus.memory.read_byte(0x001236), Some(0xBE));
        assert_eq!(bus.cycle_count, start_cycle + 11);
        let (mti, _) = bus.timer.tick_counts(bus.cycle_count);
        assert_eq!(mti, 9);
    }

    #[test]
    fn execute_async_mv_ba_imem_ptr_offset_clamps_mti_phase() {
        let mut bus = TimerBus::new(2048, 0, 2046);
        bus.timer.next_mti = bus.cycle_count.wrapping_add(2);
        // Program: PRE + MV BA,[(n)] with offset mode
        bus.memory.write_external_byte(0x0000, 0x30); // PRE
        bus.memory.write_external_byte(0x0001, 0x9A); // MV BA,[(n)]
        bus.memory.write_external_byte(0x0002, 0x80); // mode byte (offset +)
        bus.memory.write_external_byte(0x0003, 0xE6); // IMEM base
        bus.memory.write_external_byte(0x0004, 0x06); // displacement
        bus.memory.write_internal_byte(IMEM_BP_OFFSET, 0x00);
        bus.memory.write_internal_byte(IMEM_PX_OFFSET, 0x00);
        bus.memory.write_internal_byte(IMEM_PY_OFFSET, 0x00);
        // Pointer at IMEM[0xE6] -> 0x001000
        bus.memory.write_internal_byte(0xE6, 0x00);
        bus.memory.write_internal_byte(0xE7, 0x10);
        bus.memory.write_internal_byte(0xE8, 0x00);
        // External data at 0x001006 (pointer + disp)
        bus.memory.write_external_byte(0x001006, 0xCD);
        bus.memory.write_external_byte(0x001007, 0xAB);

        let mut state = LlamaState::new();
        state.set_pc(0);

        let mut exec = AsyncLlamaExecutor::new();
        let bus_ptr: *mut TimerBus = &mut bus;
        let mut cycle_count = bus.cycle_count;
        let mut tick_cb = move |cycle| unsafe {
            (*bus_ptr).cycle_count = cycle;
            (*bus_ptr).tick_timers_only(cycle);
        };
        let mut ticker = TickHelper::new(&mut cycle_count, true, Some(&mut tick_cb));

        let len = block_on(exec.execute(0x30, &mut state, &mut bus, &mut ticker))
            .expect("execute async mv ba,[(n)] offset");

        bus.cycle_count = cycle_count;
        assert_eq!(len, 5);
        assert_eq!(state.get_reg(RegName::BA) & 0xFFFF, 0xABCD);
        assert_eq!(bus.cycle_count, 2059);
        let (mti, _) = bus.timer.tick_counts(bus.cycle_count);
        assert_eq!(mti, 10);
    }

    #[test]
    fn execute_async_pre_mv_a_imem_ptr_offset_preserves_mti_phase() {
        let mut bus = TimerBus::new(2048, 0, 2045);
        bus.timer.next_mti = bus.cycle_count.wrapping_add(3);
        // Program: PRE + MV A,[(n)] with offset mode
        bus.memory.write_external_byte(0x0000, 0x30); // PRE
        bus.memory.write_external_byte(0x0001, 0x98); // MV A,[(n)]
        bus.memory.write_external_byte(0x0002, 0x80); // mode byte (offset +)
        bus.memory.write_external_byte(0x0003, 0xE6); // IMEM base
        bus.memory.write_external_byte(0x0004, 0x04); // displacement
        bus.memory.write_internal_byte(IMEM_BP_OFFSET, 0x00);
        bus.memory.write_internal_byte(IMEM_PX_OFFSET, 0x00);
        bus.memory.write_internal_byte(IMEM_PY_OFFSET, 0x00);
        // Pointer at IMEM[0xE6] -> 0x001000
        bus.memory.write_internal_byte(0xE6, 0x00);
        bus.memory.write_internal_byte(0xE7, 0x10);
        bus.memory.write_internal_byte(0xE8, 0x00);
        // External data at 0x001004 (pointer + disp)
        bus.memory.write_external_byte(0x001004, 0x55);

        let mut state = LlamaState::new();
        state.set_pc(0);

        let mut exec = AsyncLlamaExecutor::new();
        let bus_ptr: *mut TimerBus = &mut bus;
        let mut cycle_count = bus.cycle_count;
        let mut tick_cb = move |cycle| unsafe {
            (*bus_ptr).cycle_count = cycle;
            (*bus_ptr).tick_timers_only(cycle);
        };
        let mut ticker = TickHelper::new(&mut cycle_count, true, Some(&mut tick_cb));

        let len = block_on(exec.execute(0x30, &mut state, &mut bus, &mut ticker))
            .expect("execute async pre mv a,[(n)] offset");

        bus.cycle_count = cycle_count;
        assert_eq!(len, 5);
        assert_eq!(state.get_reg(RegName::A) & 0xFF, 0x55);
        assert_eq!(bus.cycle_count, 2057);
        let (mti, _) = bus.timer.tick_counts(bus.cycle_count);
        assert_eq!(mti, 9);
    }

    #[test]
    fn execute_async_mv_a_emem_reg_offset_clamps_mti_phase() {
        let start_cycle = 2047u64;
        let mut bus = TimerBus::new(2048, 0, start_cycle);
        // Program: MV A,[X+disp] with offset mode
        bus.memory.write_external_byte(0x0000, 0x90); // MV A, EMemRegWidth(1)
        bus.memory.write_external_byte(0x0001, 0x84); // offset +, X
        bus.memory.write_external_byte(0x0002, 0x03); // displacement
        bus.memory.write_external_byte(0x001003, 0x5A);

        let mut state = LlamaState::new();
        state.set_pc(0);
        state.set_reg(RegName::X, 0x001000);

        let mut exec = AsyncLlamaExecutor::new();
        let bus_ptr: *mut TimerBus = &mut bus;
        let mut cycle_count = bus.cycle_count;
        let mut tick_cb = move |cycle| unsafe {
            (*bus_ptr).cycle_count = cycle;
            (*bus_ptr).tick_timers_only(cycle);
        };
        let mut ticker = TickHelper::new(&mut cycle_count, true, Some(&mut tick_cb));

        let len = block_on(exec.execute(0x90, &mut state, &mut bus, &mut ticker))
            .expect("execute async mv a, [x+disp]");

        bus.cycle_count = cycle_count;
        assert_eq!(len, 3);
        assert_eq!(state.get_reg(RegName::A) & 0xFF, 0x5A);
        assert_eq!(bus.cycle_count, start_cycle + 6);
        let (mti, _) = bus.timer.tick_counts(bus.cycle_count);
        assert_eq!(mti, 4);
    }

    #[test]
    fn execute_async_add_reg_imm_ticks() {
        let mut bus = MemBus::with_size(0x10);
        bus.mem[0] = 0x40; // ADD A, imm8
        bus.mem[1] = 0x05;
        let mut state = LlamaState::new();
        state.set_reg(RegName::A, 1);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x40, &mut state, &mut bus, &mut ticker))
            .expect("execute async add reg imm");

        assert_eq!(len, 2);
        assert_eq!(state.get_reg(RegName::A) & 0xFF, 6);
        assert_eq!(state.get_reg(RegName::FC) & 1, 0);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 0);
        assert_eq!(state.pc(), 2);
        assert_eq!(cycles, 3);
    }

    #[test]
    fn execute_async_add_reg_imem_ticks() {
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0x42; // ADD A, IMem8
        bus.mem[1] = 0x10;
        bus.mem[0x10] = 0x05;
        let mut state = LlamaState::new();
        state.set_reg(RegName::A, 1);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x42, &mut state, &mut bus, &mut ticker))
            .expect("execute async add reg imem");

        assert_eq!(len, 2);
        assert_eq!(state.get_reg(RegName::A) & 0xFF, 6);
        assert_eq!(state.get_reg(RegName::FC) & 1, 0);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 0);
        assert_eq!(state.pc(), 2);
        assert_eq!(cycles, 4);
    }

    #[test]
    fn execute_async_add_imem_imm_ticks() {
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0x41; // ADD IMem8, imm8
        bus.mem[1] = 0x30;
        bus.mem[2] = 0x04;
        bus.mem[0x30] = 0x03;
        let mut state = LlamaState::new();
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x41, &mut state, &mut bus, &mut ticker))
            .expect("execute async add imem imm");

        assert_eq!(len, 3);
        assert_eq!(bus.mem[0x30], 0x07);
        assert_eq!(state.get_reg(RegName::FC) & 1, 0);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 0);
        assert_eq!(state.pc(), 3);
        assert_eq!(cycles, 4);
    }

    #[test]
    fn execute_async_add_imem_reg_ticks() {
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0x43; // ADD IMem8, A
        bus.mem[1] = 0x20;
        bus.mem[0x20] = 0x01;
        let mut state = LlamaState::new();
        state.set_reg(RegName::A, 2);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x43, &mut state, &mut bus, &mut ticker))
            .expect("execute async add imem reg");

        assert_eq!(len, 2);
        assert_eq!(bus.mem[0x20], 0x03);
        assert_eq!(state.get_reg(RegName::FC) & 1, 0);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 0);
        assert_eq!(state.pc(), 2);
        assert_eq!(cycles, 4);
    }

    #[test]
    fn execute_async_sub_reg_imm_ticks() {
        let mut bus = MemBus::with_size(0x10);
        bus.mem[0] = 0x48; // SUB A, imm8
        bus.mem[1] = 0x02;
        let mut state = LlamaState::new();
        state.set_reg(RegName::A, 5);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x48, &mut state, &mut bus, &mut ticker))
            .expect("execute async sub reg imm");

        assert_eq!(len, 2);
        assert_eq!(state.get_reg(RegName::A) & 0xFF, 3);
        assert_eq!(state.get_reg(RegName::FC) & 1, 0);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 0);
        assert_eq!(state.pc(), 2);
        assert_eq!(cycles, 3);
    }

    #[test]
    fn execute_async_sub_imem_reg_ticks() {
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0x4B; // SUB IMem8, A
        bus.mem[1] = 0x10;
        bus.mem[0x10] = 5;
        let mut state = LlamaState::new();
        state.set_reg(RegName::A, 2);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x4B, &mut state, &mut bus, &mut ticker))
            .expect("execute async sub imem reg");

        assert_eq!(len, 2);
        assert_eq!(bus.mem[0x10], 3);
        assert_eq!(state.get_reg(RegName::FC) & 1, 0);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 0);
        assert_eq!(state.pc(), 2);
        assert_eq!(cycles, 4);
    }

    #[test]
    fn execute_async_and_reg_imem_ticks() {
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0x77; // AND A, IMem8
        bus.mem[1] = 0x10;
        bus.mem[0x10] = 0x0F;
        let mut state = LlamaState::new();
        state.set_reg(RegName::A, 0xF0);
        state.set_reg(RegName::FC, 1);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x77, &mut state, &mut bus, &mut ticker))
            .expect("execute async and reg imem");

        assert_eq!(len, 2);
        assert_eq!(state.get_reg(RegName::A) & 0xFF, 0);
        assert_eq!(state.get_reg(RegName::FC) & 1, 1);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 1);
        assert_eq!(state.pc(), 2);
        assert_eq!(cycles, 4);
    }

    #[test]
    fn execute_async_or_reg_imm_ticks() {
        let mut bus = MemBus::with_size(0x10);
        bus.mem[0] = 0x78; // OR A, imm8
        bus.mem[1] = 0x01;
        let mut state = LlamaState::new();
        state.set_reg(RegName::A, 0x10);
        state.set_reg(RegName::FC, 1);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x78, &mut state, &mut bus, &mut ticker))
            .expect("execute async or reg imm");

        assert_eq!(len, 2);
        assert_eq!(state.get_reg(RegName::A) & 0xFF, 0x11);
        assert_eq!(state.get_reg(RegName::FC) & 1, 1);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 0);
        assert_eq!(state.pc(), 2);
        assert_eq!(cycles, 3);
    }

    #[test]
    fn execute_async_xor_reg_imm_ticks() {
        let mut bus = MemBus::with_size(0x10);
        bus.mem[0] = 0x68; // XOR A, imm8
        bus.mem[1] = 0xF0;
        let mut state = LlamaState::new();
        state.set_reg(RegName::A, 0x0F);
        state.set_reg(RegName::FC, 1);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x68, &mut state, &mut bus, &mut ticker))
            .expect("execute async xor reg imm");

        assert_eq!(len, 2);
        assert_eq!(state.get_reg(RegName::A) & 0xFF, 0xFF);
        assert_eq!(state.get_reg(RegName::FC) & 1, 1);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 0);
        assert_eq!(state.pc(), 2);
        assert_eq!(cycles, 3);
    }

    #[test]
    fn execute_async_adc_reg_imm_ticks() {
        let mut bus = MemBus::with_size(0x10);
        bus.mem[0] = 0x50; // ADC A, imm8
        bus.mem[1] = 0x01;
        let mut state = LlamaState::new();
        state.set_reg(RegName::A, 1);
        state.set_reg(RegName::FC, 1);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x50, &mut state, &mut bus, &mut ticker))
            .expect("execute async adc reg imm");

        assert_eq!(len, 2);
        assert_eq!(state.get_reg(RegName::A) & 0xFF, 3);
        assert_eq!(state.get_reg(RegName::FC) & 1, 0);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 0);
        assert_eq!(state.pc(), 2);
        assert_eq!(cycles, 3);
    }

    #[test]
    fn execute_async_sbc_reg_imm_ticks() {
        let mut bus = MemBus::with_size(0x10);
        bus.mem[0] = 0x58; // SBC A, imm8
        bus.mem[1] = 0x01;
        let mut state = LlamaState::new();
        state.set_reg(RegName::A, 3);
        state.set_reg(RegName::FC, 1);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x58, &mut state, &mut bus, &mut ticker))
            .expect("execute async sbc reg imm");

        assert_eq!(len, 2);
        assert_eq!(state.get_reg(RegName::A) & 0xFF, 1);
        assert_eq!(state.get_reg(RegName::FC) & 1, 0);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 0);
        assert_eq!(state.pc(), 2);
        assert_eq!(cycles, 3);
    }

    #[test]
    fn execute_async_cmp_reg_imm_ticks() {
        let mut bus = MemBus::with_size(0x10);
        bus.mem[0] = 0x60; // CMP A, imm8
        bus.mem[1] = 0x02;
        let mut state = LlamaState::new();
        state.set_reg(RegName::A, 2);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x60, &mut state, &mut bus, &mut ticker))
            .expect("execute async cmp reg imm");

        assert_eq!(len, 2);
        assert_eq!(state.get_reg(RegName::A) & 0xFF, 2);
        assert_eq!(state.get_reg(RegName::FC) & 1, 0);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 1);
        assert_eq!(state.pc(), 2);
        assert_eq!(cycles, 3);
    }

    #[test]
    fn execute_async_cmp_imem_reg_ticks() {
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0x63; // CMP IMem8, A
        bus.mem[1] = 0x20;
        bus.mem[0x20] = 0x01;
        let mut state = LlamaState::new();
        state.set_reg(RegName::A, 2);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x63, &mut state, &mut bus, &mut ticker))
            .expect("execute async cmp imem reg");

        assert_eq!(len, 2);
        assert_eq!(state.get_reg(RegName::A) & 0xFF, 2);
        assert_eq!(state.get_reg(RegName::FC) & 1, 1);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 0);
        assert_eq!(state.pc(), 2);
        assert_eq!(cycles, 4);
    }

    #[test]
    fn execute_async_cmp_imem_imm_ticks() {
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0x61; // CMP IMem8, imm8
        bus.mem[1] = 0x20;
        bus.mem[2] = 0x05;
        bus.mem[0x20] = 0x07;
        let mut state = LlamaState::new();
        state.set_reg(RegName::A, 0x00);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x61, &mut state, &mut bus, &mut ticker))
            .expect("execute async cmp imem imm");

        assert_eq!(len, 3);
        assert_eq!(state.get_reg(RegName::FC) & 1, 0);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 0);
        assert_eq!(state.pc(), 3);
        assert_eq!(cycles, 4);
    }

    #[test]
    fn execute_async_cmp_imem_imem_ticks() {
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0xB7; // CMP IMem8, IMem8
        bus.mem[1] = 0x10;
        bus.mem[2] = 0x12;
        bus.mem[0x10] = 0x05;
        bus.mem[0x12] = 0x07;
        let mut state = LlamaState::new();
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0xB7, &mut state, &mut bus, &mut ticker))
            .expect("execute async cmp imem imem");

        assert_eq!(len, 3);
        assert_eq!(state.get_reg(RegName::FC) & 1, 1);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 0);
        assert_eq!(state.pc(), 3);
        assert_eq!(cycles, 6);
    }

    #[test]
    fn execute_async_test_reg_imm_ticks() {
        let mut bus = MemBus::with_size(0x10);
        bus.mem[0] = 0x64; // TEST A, imm8
        bus.mem[1] = 0x0F;
        let mut state = LlamaState::new();
        state.set_reg(RegName::A, 0xF0);
        state.set_reg(RegName::FC, 1);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x64, &mut state, &mut bus, &mut ticker))
            .expect("execute async test reg imm");

        assert_eq!(len, 2);
        assert_eq!(state.get_reg(RegName::A) & 0xFF, 0xF0);
        assert_eq!(state.get_reg(RegName::FC) & 1, 1);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 1);
        assert_eq!(state.pc(), 2);
        assert_eq!(cycles, 3);
    }

    #[test]
    fn execute_async_test_imem_reg_ticks() {
        let mut bus = MemBus::with_size(0x40);
        bus.mem[0] = 0x67; // TEST (n), A
        bus.mem[1] = 0x10;
        bus.mem[0x10] = 0x0F;
        let mut state = LlamaState::new();
        state.set_reg(RegName::A, 0xF0);
        state.set_reg(RegName::FC, 1);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x67, &mut state, &mut bus, &mut ticker))
            .expect("execute async test imem reg");

        assert_eq!(len, 2);
        assert_eq!(state.get_reg(RegName::A) & 0xFF, 0xF0);
        assert_eq!(state.get_reg(RegName::FC) & 1, 1);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 1);
        assert_eq!(state.pc(), 2);
        assert_eq!(cycles, 4);
    }

    #[test]
    fn execute_async_test_imem_imm_ticks() {
        let mut bus = MemBus::with_size(0x40);
        bus.mem[0] = 0x65; // TEST (n), imm8
        bus.mem[1] = 0x10;
        bus.mem[2] = 0x0F;
        bus.mem[0x10] = 0xF0;
        let mut state = LlamaState::new();
        state.set_reg(RegName::FC, 1);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x65, &mut state, &mut bus, &mut ticker))
            .expect("execute async test imem imm");

        assert_eq!(len, 3);
        assert_eq!(state.get_reg(RegName::FC) & 1, 1);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 1);
        assert_eq!(state.pc(), 3);
        assert_eq!(cycles, 4);
    }

    #[test]
    fn execute_async_xor_imem_imem_ticks() {
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0x6E; // XOR IMem8, IMem8
        bus.mem[1] = 0x10;
        bus.mem[2] = 0x20;
        bus.mem[0x10] = 0xAA;
        bus.mem[0x20] = 0x0F;
        let mut state = LlamaState::new();
        state.set_reg(RegName::FC, 1);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x6E, &mut state, &mut bus, &mut ticker))
            .expect("execute async xor imem imem");

        assert_eq!(len, 3);
        assert_eq!(bus.mem[0x10], 0xA5);
        assert_eq!(state.get_reg(RegName::FC) & 1, 1);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 0);
        assert_eq!(state.pc(), 3);
        assert_eq!(cycles, 6);
    }

    #[test]
    fn execute_async_and_imem_imem_ticks() {
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0x76; // AND IMem8, IMem8
        bus.mem[1] = 0x12;
        bus.mem[2] = 0x22;
        bus.mem[0x12] = 0xF0;
        bus.mem[0x22] = 0x0F;
        let mut state = LlamaState::new();
        state.set_reg(RegName::FC, 1);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x76, &mut state, &mut bus, &mut ticker))
            .expect("execute async and imem imem");

        assert_eq!(len, 3);
        assert_eq!(bus.mem[0x12], 0x00);
        assert_eq!(state.get_reg(RegName::FC) & 1, 1);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 1);
        assert_eq!(state.pc(), 3);
        assert_eq!(cycles, 6);
    }

    #[test]
    fn execute_async_or_imem_imem_ticks() {
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0x7E; // OR IMem8, IMem8
        bus.mem[1] = 0x14;
        bus.mem[2] = 0x24;
        bus.mem[0x14] = 0xF0;
        bus.mem[0x24] = 0x0F;
        let mut state = LlamaState::new();
        state.set_reg(RegName::FC, 1);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x7E, &mut state, &mut bus, &mut ticker))
            .expect("execute async or imem imem");

        assert_eq!(len, 3);
        assert_eq!(bus.mem[0x14], 0xFF);
        assert_eq!(state.get_reg(RegName::FC) & 1, 1);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 0);
        assert_eq!(state.pc(), 3);
        assert_eq!(cycles, 6);
    }

    #[test]
    fn execute_async_add_regpair_ticks() {
        let mut bus = MemBus::with_size(0x10);
        bus.mem[0] = 0x44; // ADD RegPair(2)
        bus.mem[1] = 0x23; // BA <- BA + I
        let mut state = LlamaState::new();
        state.set_reg(RegName::BA, 0x0010);
        state.set_reg(RegName::I, 0x0001);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x44, &mut state, &mut bus, &mut ticker))
            .expect("execute async add regpair");

        assert_eq!(len, 2);
        assert_eq!(state.get_reg(RegName::BA) & 0xFFFF, 0x0011);
        assert_eq!(state.get_reg(RegName::FC) & 1, 0);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 0);
        assert_eq!(state.pc(), 2);
        assert_eq!(cycles, 5);
    }

    #[test]
    fn execute_async_add_regpair_20bit_carry() {
        let mut bus = MemBus::with_size(0x10);
        bus.mem[0] = 0x45; // ADD RegPair(3)
        bus.mem[1] = 0x45; // X <- X + Y
        let mut state = LlamaState::new();
        state.set_reg(RegName::X, 0x0F_FFFF);
        state.set_reg(RegName::Y, 0x000001);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x45, &mut state, &mut bus, &mut ticker))
            .expect("execute async add regpair 20-bit");

        assert_eq!(len, 2);
        assert_eq!(state.get_reg(RegName::X), 0x00000);
        assert_eq!(state.get_reg(RegName::FC) & 1, 1);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 1);
        assert_eq!(state.pc(), 2);
        assert_eq!(cycles, 7);
    }

    #[test]
    fn execute_async_sub_regpair_ticks() {
        let mut bus = MemBus::with_size(0x10);
        bus.mem[0] = 0x4C; // SUB RegPair(2)
        bus.mem[1] = 0x23; // BA <- BA - I
        let mut state = LlamaState::new();
        state.set_reg(RegName::BA, 0x0010);
        state.set_reg(RegName::I, 0x0001);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x4C, &mut state, &mut bus, &mut ticker))
            .expect("execute async sub regpair");

        assert_eq!(len, 2);
        assert_eq!(state.get_reg(RegName::BA) & 0xFFFF, 0x000F);
        assert_eq!(state.get_reg(RegName::FC) & 1, 0);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 0);
        assert_eq!(state.pc(), 2);
        assert_eq!(cycles, 5);
    }

    #[test]
    fn execute_add_regpair_20bit_carry() {
        let mut bus = MemBus::with_size(0x10);
        bus.mem[0] = 0x45; // ADD RegPair(3)
        bus.mem[1] = 0x45; // X <- X + Y
        let mut state = LlamaState::new();
        state.set_reg(RegName::X, 0x0F_FFFF);
        state.set_reg(RegName::Y, 0x000001);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();

        let len = exec.execute(0x45, &mut state, &mut bus).expect("execute add regpair 20-bit");

        assert_eq!(len, 2);
        assert_eq!(state.get_reg(RegName::X), 0x00000);
        assert_eq!(state.get_reg(RegName::FC) & 1, 1);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 1);
        assert_eq!(state.pc(), 2);
    }

    #[test]
    fn execute_async_adcl_imem_imem_ticks() {
        let mut bus = MemBus::with_size(0x200);
        bus.mem[0] = 0x54; // ADCL IMem8, IMem8
        bus.mem[1] = 0x10;
        bus.mem[2] = 0x20;
        bus.mem[0x10] = 0x01;
        bus.mem[0x11] = 0xFF;
        bus.mem[0x20] = 0x01;
        bus.mem[0x21] = 0x00;
        let mut state = LlamaState::new();
        state.set_reg(RegName::I, 2);
        state.set_reg(RegName::FC, 1);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x54, &mut state, &mut bus, &mut ticker))
            .expect("execute async adcl imem imem");

        assert_eq!(len, 3);
        assert_eq!(bus.mem[0x10], 0x03);
        assert_eq!(bus.mem[0x11], 0xFF);
        assert_eq!(state.get_reg(RegName::I), 0);
        assert_eq!(state.get_reg(RegName::FC) & 1, 0);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 0);
        assert_eq!(state.pc(), 3);
        assert_eq!(cycles, 7);
    }

    #[test]
    fn execute_async_sbcl_imem_reg_ticks() {
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0x5D; // SBCL IMem8, A
        bus.mem[1] = 0x10;
        bus.mem[0x10] = 0x05;
        let mut state = LlamaState::new();
        state.set_reg(RegName::A, 0x02);
        state.set_reg(RegName::I, 1);
        state.set_reg(RegName::FC, 1);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x5D, &mut state, &mut bus, &mut ticker))
            .expect("execute async sbcl imem reg");

        assert_eq!(len, 2);
        assert_eq!(bus.mem[0x10], 0x02);
        assert_eq!(state.get_reg(RegName::I), 0);
        assert_eq!(state.get_reg(RegName::FC) & 1, 0);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 0);
        assert_eq!(state.pc(), 2);
        assert_eq!(cycles, 4);
    }

    #[test]
    fn execute_async_pre_sbcl_imem_imem_ticks() {
        let mut bus = MemBus::with_size(0x200);
        bus.mem[0] = 0x22; // PRE (BP+n, n)
        bus.mem[1] = 0x5C; // SBCL (m),(n)
        bus.mem[2] = 0x03; // dst IMEM offset
        bus.mem[3] = 0xCE; // src IMEM offset
        bus.mem[IMEM_BP_OFFSET as usize] = 0x10;
        bus.mem[IMEM_PX_OFFSET as usize] = 0x00;
        bus.mem[IMEM_PY_OFFSET as usize] = 0x00;
        bus.mem[0x13] = 0x05;
        bus.mem[0x14] = 0x06;
        bus.mem[0x15] = 0x07;
        bus.mem[0xCE] = 0x01;
        bus.mem[0xCF] = 0x02;
        bus.mem[0xD0] = 0x03;
        let mut state = LlamaState::new();
        state.set_reg(RegName::I, 3);
        state.set_reg(RegName::FC, 0);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x22, &mut state, &mut bus, &mut ticker))
            .expect("execute async pre sbcl imem/imem");

        assert_eq!(len, 4);
        assert_eq!(bus.mem[0x13], 0x04);
        assert_eq!(bus.mem[0x14], 0x04);
        assert_eq!(bus.mem[0x15], 0x04);
        assert_eq!(state.get_reg(RegName::I), 0);
        assert_eq!(state.pc(), 4);
        assert_eq!(cycles, 10);
    }

    #[test]
    fn execute_async_pre_exw_imem_imem_ticks() {
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0x22; // PRE (BP+n, n)
        bus.mem[1] = 0xC1; // EXW (m),(n)
        bus.mem[2] = 0x12; // dst IMEM offset
        bus.mem[3] = 0x34; // src IMEM offset
        bus.mem[IMEM_BP_OFFSET as usize] = 0x00;
        bus.mem[IMEM_PX_OFFSET as usize] = 0x00;
        bus.mem[IMEM_PY_OFFSET as usize] = 0x00;
        bus.mem[0x12] = 0xAA;
        bus.mem[0x13] = 0xBB;
        bus.mem[0x34] = 0x11;
        bus.mem[0x35] = 0x22;
        let mut state = LlamaState::new();
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x22, &mut state, &mut bus, &mut ticker))
            .expect("execute async pre exw imem/imem");

        assert_eq!(len, 4);
        assert_eq!(bus.mem[0x12], 0x11);
        assert_eq!(bus.mem[0x13], 0x22);
        assert_eq!(bus.mem[0x34], 0xAA);
        assert_eq!(bus.mem[0x35], 0xBB);
        assert_eq!(state.pc(), 4);
        assert_eq!(cycles, 11);
    }

    #[test]
    fn execute_async_adcl_zero_length_is_noop() {
        let mut bus = MemBus::with_size(0x200);
        bus.mem[0] = 0x54; // ADCL IMem8, IMem8
        bus.mem[1] = 0x10;
        bus.mem[2] = 0x20;
        bus.mem[0x10] = 0x11;
        bus.mem[0x20] = 0x22;
        let mut state = LlamaState::new();
        state.set_reg(RegName::I, 0);
        state.set_reg(RegName::FC, 1);
        state.set_reg(RegName::FZ, 1);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x54, &mut state, &mut bus, &mut ticker))
            .expect("execute async adcl zero length");

        assert_eq!(len, 3);
        assert_eq!(bus.mem[0x10], 0x11);
        assert_eq!(bus.mem[0x20], 0x22);
        assert_eq!(state.get_reg(RegName::I), 0);
        assert_eq!(state.get_reg(RegName::FC) & 1, 1);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 1);
        assert_eq!(state.pc(), 3);
        assert_eq!(cycles, 3);
    }

    #[test]
    fn execute_async_and_emem_imm_ticks() {
        let mut bus = MemBus::with_size(0x200);
        bus.mem[0] = 0x72; // AND EMemAddrWidth(1), imm8
        bus.mem[1] = 0x10;
        bus.mem[2] = 0x00;
        bus.mem[3] = 0x00;
        bus.mem[4] = 0x0F;
        bus.mem[0x10] = 0xF0;
        let mut state = LlamaState::new();
        state.set_reg(RegName::FC, 1);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x72, &mut state, &mut bus, &mut ticker))
            .expect("execute async and emem imm");

        assert_eq!(len, 5);
        assert_eq!(bus.mem[0x10], 0x00);
        assert_eq!(state.get_reg(RegName::FC) & 1, 1);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 1);
        assert_eq!(state.pc(), 5);
        assert_eq!(cycles, 7);
    }

    #[test]
    fn execute_async_cmp_emem_imm_ticks() {
        let mut bus = MemBus::with_size(0x200);
        bus.mem[0] = 0x62; // CMP EMemAddrWidth(1), imm8
        bus.mem[1] = 0x20;
        bus.mem[2] = 0x00;
        bus.mem[3] = 0x00;
        bus.mem[4] = 0x10;
        bus.mem[0x20] = 0x08;
        let mut state = LlamaState::new();
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x62, &mut state, &mut bus, &mut ticker))
            .expect("execute async cmp emem imm");

        assert_eq!(len, 5);
        assert_eq!(bus.mem[0x20], 0x08);
        assert_eq!(state.get_reg(RegName::FC) & 1, 1);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 0);
        assert_eq!(state.pc(), 5);
        assert_eq!(cycles, 6);
    }

    #[test]
    fn execute_async_inc_reg3_ticks() {
        let mut bus = MemBus::with_size(0x10);
        bus.mem[0] = 0x6C; // INC Reg3
        bus.mem[1] = 0x00; // A
        let mut state = LlamaState::new();
        state.set_reg(RegName::A, 0xFF);
        state.set_reg(RegName::FC, 1);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x6C, &mut state, &mut bus, &mut ticker))
            .expect("execute async inc reg3");

        assert_eq!(len, 2);
        assert_eq!(state.get_reg(RegName::A) & 0xFF, 0x00);
        assert_eq!(state.get_reg(RegName::FC) & 1, 1);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 1);
        assert_eq!(state.pc(), 2);
        assert_eq!(cycles, 3);
    }

    #[test]
    fn execute_async_inc_reg3_x_wraps_20bit() {
        let mut bus = MemBus::with_size(0x10);
        bus.mem[0] = 0x6C; // INC Reg3
        bus.mem[1] = 0x04; // X
        let mut state = LlamaState::new();
        state.set_reg(RegName::X, 0x0F_FFFF);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x6C, &mut state, &mut bus, &mut ticker))
            .expect("execute async inc reg3 x");

        assert_eq!(len, 2);
        assert_eq!(state.get_reg(RegName::X), 0x00000);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 1);
        assert_eq!(state.pc(), 2);
        assert_eq!(cycles, 3);
    }

    #[test]
    fn execute_async_dec_reg3_x_wraps_20bit() {
        let mut bus = MemBus::with_size(0x10);
        bus.mem[0] = 0x7C; // DEC Reg3
        bus.mem[1] = 0x04; // X
        let mut state = LlamaState::new();
        state.set_reg(RegName::X, 0x00000);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x7C, &mut state, &mut bus, &mut ticker))
            .expect("execute async dec reg3 x");

        assert_eq!(len, 2);
        assert_eq!(state.get_reg(RegName::X), 0x0F_FFFF);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 0);
        assert_eq!(state.pc(), 2);
        assert_eq!(cycles, 3);
    }

    #[test]
    fn execute_inc_reg3_x_wraps_20bit() {
        let mut bus = MemBus::with_size(0x10);
        bus.mem[0] = 0x6C; // INC Reg3
        bus.mem[1] = 0x04; // X
        let mut state = LlamaState::new();
        state.set_reg(RegName::X, 0x0F_FFFF);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();

        let len = exec.execute(0x6C, &mut state, &mut bus).expect("execute inc reg3 x");

        assert_eq!(len, 2);
        assert_eq!(state.get_reg(RegName::X), 0x00000);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 1);
        assert_eq!(state.pc(), 2);
    }

    #[test]
    fn execute_async_dec_imem_ticks() {
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0x7D; // DEC IMem8
        bus.mem[1] = 0x10;
        bus.mem[0x10] = 0x00;
        let mut state = LlamaState::new();
        state.set_reg(RegName::FC, 1);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x7D, &mut state, &mut bus, &mut ticker))
            .expect("execute async dec imem");

        assert_eq!(len, 2);
        assert_eq!(bus.mem[0x10], 0xFF);
        assert_eq!(state.get_reg(RegName::FC) & 1, 1);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 0);
        assert_eq!(state.pc(), 2);
        assert_eq!(cycles, 3);
    }

    #[test]
    fn execute_async_wait_ticks() {
        let mut bus = MemBus::with_size(0x10);
        bus.mem[0] = 0xEF; // WAIT
        let mut state = LlamaState::new();
        state.set_reg(RegName::I, 3);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0xEF, &mut state, &mut bus, &mut ticker))
            .expect("execute async wait");

        assert_eq!(len, 1);
        assert_eq!(state.get_reg(RegName::I), 0);
        assert_eq!(state.pc(), 1);
        assert_eq!(cycles, 3);
    }

    #[test]
    fn execute_async_halt_sets_flag() {
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0xDE; // HALT
        let mut state = LlamaState::new();
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0xDE, &mut state, &mut bus, &mut ticker))
            .expect("execute async halt");

        assert_eq!(len, 1);
        assert_eq!(state.pc(), 1);
        assert!(state.is_halted());
        assert!(!state.is_off());
        assert_eq!(cycles, 2);
    }

    #[test]
    fn execute_async_off_sets_state() {
        let mut bus = MemBus::with_size(1);
        bus.mem[0] = 0xDF; // OFF
        let mut state = LlamaState::new();
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let _ = block_on(exec.execute_async(0xDF, &mut state, &mut bus, &mut ticker))
            .expect("execute async off");

        assert!(state.is_halted());
        assert!(state.is_off());
    }

    #[test]
    fn execute_async_jp_abs_imm16() {
        let mut bus = MemBus::with_size(0x20000);
        let pc = 0x10000;
        bus.mem[pc as usize] = 0x02; // JP abs imm16
        bus.mem[pc as usize + 1] = 0x34;
        bus.mem[pc as usize + 2] = 0x12;
        let mut state = LlamaState::new();
        state.set_pc(pc);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x02, &mut state, &mut bus, &mut ticker))
            .expect("execute async jp abs");

        assert_eq!(len, 3);
        assert_eq!(state.pc(), 0x011234);
        assert_eq!(cycles, 4);
    }

    #[test]
    fn execute_async_jpf_abs_imm20_ticks() {
        let mut bus = MemBus::with_size(0x20000);
        let pc = 0x10000;
        bus.mem[pc as usize] = 0x03; // JPF abs imm20
        bus.mem[pc as usize + 1] = 0xDE;
        bus.mem[pc as usize + 2] = 0xBC;
        bus.mem[pc as usize + 3] = 0x0A; // 0x0ABCDE
        let mut state = LlamaState::new();
        state.set_pc(pc);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x03, &mut state, &mut bus, &mut ticker))
            .expect("execute async jpf abs");

        assert_eq!(len, 4);
        assert_eq!(state.pc(), 0x0ABCDE);
        assert_eq!(cycles, 5);
    }

    #[test]
    fn execute_async_jp_rel_not_taken() {
        let mut bus = MemBus::with_size(0x10);
        bus.mem[0] = 0x18; // JP_Rel Z
        bus.mem[1] = 0x02;
        let mut state = LlamaState::new();
        state.set_reg(RegName::FZ, 0);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x18, &mut state, &mut bus, &mut ticker))
            .expect("execute async jp rel");

        assert_eq!(len, 2);
        assert_eq!(state.pc(), 2);
        assert_eq!(cycles, 2);
    }

    #[test]
    fn execute_async_jr_taken_ticks() {
        let mut bus = MemBus::with_size(0x10);
        bus.mem[0] = 0x12; // JR +n
        bus.mem[1] = 0x02;
        let mut state = LlamaState::new();
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x12, &mut state, &mut bus, &mut ticker))
            .expect("execute async jr");

        assert_eq!(len, 2);
        assert_eq!(state.pc(), 4);
        assert_eq!(cycles, 3);
    }

    #[test]
    fn execute_async_jrnz_taken_ticks() {
        let mut bus = MemBus::with_size(0x20);
        let pc = 0x10;
        bus.mem[pc as usize] = 0x1B; // JRNZ -n
        bus.mem[pc as usize + 1] = 0x02; // jump back to same PC
        let mut state = LlamaState::new();
        state.set_pc(pc);
        state.set_reg(RegName::FZ, 0); // Z=0 so NZ is taken
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x1B, &mut state, &mut bus, &mut ticker))
            .expect("execute async jrnz");

        assert_eq!(len, 2);
        assert_eq!(state.pc(), pc);
        assert_eq!(cycles, 3);
    }

    #[test]
    fn execute_async_call_pushes_return() {
        let mut bus = MemBus::with_size(0x20000);
        let pc = 0x10000;
        bus.mem[pc as usize] = 0x04; // CALL imm16
        bus.mem[pc as usize + 1] = 0x34;
        bus.mem[pc as usize + 2] = 0x12;
        let mut state = LlamaState::new();
        state.set_pc(pc);
        state.set_reg(RegName::S, 0x0100);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x04, &mut state, &mut bus, &mut ticker))
            .expect("execute async call");

        assert_eq!(len, 3);
        assert_eq!(state.pc(), 0x011234);
        assert_eq!(state.get_reg(RegName::S), 0x00FE);
        assert_eq!(bus.mem[0x00FE], 0x03);
        assert_eq!(bus.mem[0x00FF], 0x00);
        assert_eq!(cycles, 6);
    }

    #[test]
    fn execute_async_ret_pops_return() {
        let mut bus = MemBus::with_size(0x200);
        bus.mem[0x00FE] = 0x34;
        bus.mem[0x00FF] = 0x12;
        let mut state = LlamaState::new();
        state.set_pc(0x020000);
        state.set_reg(RegName::S, 0x00FE);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x06, &mut state, &mut bus, &mut ticker))
            .expect("execute async ret");

        assert_eq!(len, 1);
        assert_eq!(state.pc(), 0x021234);
        assert_eq!(state.get_reg(RegName::S), 0x0100);
        assert_eq!(cycles, 4);
    }

    #[test]
    fn execute_async_reti_restores_state() {
        let mut bus = MemBus::with_size(0x400);
        bus.mem[0x0100] = 0xAA;
        bus.mem[0x0101] = 0x55;
        bus.mem[0x0102] = 0x11;
        bus.mem[0x0103] = 0x22;
        bus.mem[0x0104] = 0x33;
        let mut state = LlamaState::new();
        state.set_pc(0x000100);
        state.set_reg(RegName::S, 0x0100);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x01, &mut state, &mut bus, &mut ticker))
            .expect("execute async reti");

        assert_eq!(len, 1);
        assert_eq!(state.pc(), 0x032211);
        assert_eq!(state.get_reg(RegName::S), 0x0105);
        assert_eq!(state.get_reg(RegName::IMR) & 0xFF, 0xAA);
        assert_eq!(state.get_reg(RegName::F) & 0xFF, 0x55);
        assert_eq!(bus.mem[IMEM_IMR_OFFSET as usize], 0xAA);
        assert_eq!(cycles, 7);
    }

    #[test]
    fn execute_async_pushu_imr_ticks() {
        let mut bus = MemBus::with_size(0x200);
        bus.mem[IMEM_IMR_OFFSET as usize] = 0xAA;
        let mut state = LlamaState::new();
        state.set_pc(0);
        state.set_reg(RegName::U, 0x40);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x2F, &mut state, &mut bus, &mut ticker))
            .expect("execute async pushu imr");

        assert_eq!(len, 1);
        assert_eq!(state.get_reg(RegName::U), 0x3F);
        assert_eq!(bus.mem[0x3F], 0xAA);
        assert_eq!(cycles, 3);
    }

    #[test]
    fn execute_async_pushu_ba_ticks() {
        let mut bus = MemBus::with_size(0x200);
        let mut state = LlamaState::new();
        state.set_pc(0);
        state.set_reg(RegName::U, 0x40);
        state.set_reg(RegName::BA, 0x1234);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x2A, &mut state, &mut bus, &mut ticker))
            .expect("execute async pushu ba");

        assert_eq!(len, 1);
        assert_eq!(state.get_reg(RegName::U), 0x3E);
        assert_eq!(bus.mem[0x3E], 0x34);
        assert_eq!(bus.mem[0x3F], 0x12);
        assert_eq!(cycles, 4);
    }

    #[test]
    fn execute_async_popu_imr_ticks() {
        let mut bus = MemBus::with_size(0x200);
        bus.mem[0x3F] = 0xAB;
        let mut state = LlamaState::new();
        state.set_pc(0);
        state.set_reg(RegName::U, 0x3F);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x3F, &mut state, &mut bus, &mut ticker))
            .expect("execute async popu imr");

        assert_eq!(len, 1);
        assert_eq!(state.get_reg(RegName::U), 0x40);
        assert_eq!(state.get_reg(RegName::IMR) & 0xFF, 0xAB);
        assert_eq!(bus.mem[IMEM_IMR_OFFSET as usize], 0xAB);
        assert_eq!(cycles, 2);
    }

    #[test]
    fn execute_async_popu_il_ticks() {
        let mut bus = MemBus::with_size(0x200);
        bus.mem[0x3F] = 0xAB;
        let mut state = LlamaState::new();
        state.set_pc(0);
        state.set_reg(RegName::U, 0x3F);
        state.set_reg(RegName::I, 0x1200);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x39, &mut state, &mut bus, &mut ticker))
            .expect("execute async popu il");

        assert_eq!(len, 1);
        assert_eq!(state.get_reg(RegName::U), 0x40);
        assert_eq!(state.get_reg(RegName::IL), 0xAB);
        assert_eq!(state.get_reg(RegName::IH), 0x00);
        assert_eq!(cycles, 3);
    }

    #[test]
    fn execute_async_ir_pushes_stack() {
        let mut bus = MemBus::with_size(0x100000);
        let vec_idx = INTERRUPT_VECTOR_ADDR as usize;
        bus.mem[vec_idx] = 0x34;
        bus.mem[vec_idx + 1] = 0x12;
        bus.mem[vec_idx + 2] = 0x00;
        bus.mem[IMEM_IMR_OFFSET as usize] = 0x80;
        let mut state = LlamaState::new();
        state.set_pc(0x020000);
        state.set_reg(RegName::S, 0x0105);
        state.set_reg(RegName::F, 0x22);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0xFE, &mut state, &mut bus, &mut ticker))
            .expect("execute async ir");

        assert_eq!(len, 1);
        assert_eq!(state.pc(), 0x001234);
        assert_eq!(state.get_reg(RegName::S), 0x0100);
        assert_eq!(bus.mem[0x0100], 0x80);
        assert_eq!(bus.mem[0x0101], 0x22);
        assert_eq!(bus.mem[0x0102], 0x01);
        assert_eq!(bus.mem[0x0103], 0x00);
        assert_eq!(bus.mem[0x0104], 0x02);
        assert_eq!(bus.mem[IMEM_IMR_OFFSET as usize], 0x00);
        assert_eq!(cycles, 11);
    }

    #[test]
    fn execute_async_swap_ticks() {
        let mut bus = MemBus::with_size(1);
        bus.mem[0] = 0xEE; // SWAP A
        let mut state = LlamaState::new();
        state.set_reg(RegName::A, 0x3C);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0xEE, &mut state, &mut bus, &mut ticker))
            .expect("execute async swap");

        assert_eq!(len, 1);
        assert_eq!(state.get_reg(RegName::A) & 0xFF, 0xC3);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 0);
        assert_eq!(state.pc(), 1);
        assert_eq!(cycles, 3);
    }

    #[test]
    fn execute_async_shl_reg_ticks() {
        let mut bus = MemBus::with_size(1);
        bus.mem[0] = 0xF6; // SHL A
        let mut state = LlamaState::new();
        state.set_reg(RegName::A, 0x80);
        state.set_reg(RegName::FC, 1);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0xF6, &mut state, &mut bus, &mut ticker))
            .expect("execute async shl");

        assert_eq!(len, 1);
        assert_eq!(state.get_reg(RegName::A) & 0xFF, 0x01);
        assert_eq!(state.get_reg(RegName::FC) & 1, 1);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 0);
        assert_eq!(state.pc(), 1);
        assert_eq!(cycles, 2);
    }

    #[test]
    fn execute_async_ror_reg_ticks() {
        let mut bus = MemBus::with_size(1);
        bus.mem[0] = 0xE4; // ROR A
        let mut state = LlamaState::new();
        state.set_reg(RegName::A, 0x01);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0xE4, &mut state, &mut bus, &mut ticker))
            .expect("execute async ror");

        assert_eq!(len, 1);
        assert_eq!(state.get_reg(RegName::A) & 0xFF, 0x80);
        assert_eq!(state.get_reg(RegName::FC) & 1, 1);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 0);
        assert_eq!(state.pc(), 1);
        assert_eq!(cycles, 2);
    }

    #[test]
    fn execute_async_shr_imem_ticks() {
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0xF5; // SHR IMem8
        bus.mem[1] = 0x10;
        bus.mem[0x10] = 0x01;
        let mut state = LlamaState::new();
        state.set_reg(RegName::FC, 1);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0xF5, &mut state, &mut bus, &mut ticker))
            .expect("execute async shr");

        assert_eq!(len, 2);
        assert_eq!(bus.mem[0x10], 0x80);
        assert_eq!(state.get_reg(RegName::FC) & 1, 1);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 0);
        assert_eq!(state.pc(), 2);
        assert_eq!(cycles, 3);
    }

    #[test]
    fn execute_async_dsll_ticks() {
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0xEC; // DSLL IMem8
        bus.mem[1] = 0x10;
        bus.mem[0x10] = 0x12;
        bus.mem[0x0F] = 0x34;
        let mut state = LlamaState::new();
        state.set_reg(RegName::I, 2);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0xEC, &mut state, &mut bus, &mut ticker))
            .expect("execute async dsll");

        assert_eq!(len, 2);
        assert_eq!(state.get_reg(RegName::I), 0);
        assert_eq!(bus.mem[0x10], 0x20);
        assert_eq!(bus.mem[0x0F], 0x42);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 0);
        assert_eq!(state.pc(), 2);
        assert_eq!(cycles, 6);
    }

    #[test]
    fn execute_async_ex_imem_ticks() {
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0xC0; // EX IMem8, IMem8
        bus.mem[1] = 0x10;
        bus.mem[2] = 0x11;
        bus.mem[0x10] = 0xAA;
        bus.mem[0x11] = 0xBB;
        let mut state = LlamaState::new();
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0xC0, &mut state, &mut bus, &mut ticker))
            .expect("execute async ex imem");

        assert_eq!(len, 3);
        assert_eq!(bus.mem[0x10], 0xBB);
        assert_eq!(bus.mem[0x11], 0xAA);
        assert_eq!(state.pc(), 3);
        assert_eq!(cycles, 7);
    }

    #[test]
    fn execute_async_ex_regpair_ticks() {
        let mut bus = MemBus::with_size(0x10);
        bus.mem[0] = 0xED; // EX RegPair
        bus.mem[1] = 0x23; // BA <-> I
        let mut state = LlamaState::new();
        state.set_reg(RegName::BA, 0x1122);
        state.set_reg(RegName::I, 0x3344);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0xED, &mut state, &mut bus, &mut ticker))
            .expect("execute async ex regpair");

        assert_eq!(len, 2);
        assert_eq!(state.get_reg(RegName::BA), 0x3344);
        assert_eq!(state.get_reg(RegName::I), 0x1122);
        assert_eq!(state.pc(), 2);
        assert_eq!(cycles, 4);
    }

    #[test]
    fn execute_async_ex_ab_ticks() {
        let mut bus = MemBus::with_size(0x10);
        bus.mem[0] = 0xDD; // EX A, B
        let mut state = LlamaState::new();
        state.set_reg(RegName::A, 0x12);
        state.set_reg(RegName::B, 0x34);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0xDD, &mut state, &mut bus, &mut ticker))
            .expect("execute async ex ab");

        assert_eq!(len, 1);
        assert_eq!(state.get_reg(RegName::A) & 0xFF, 0x34);
        assert_eq!(state.get_reg(RegName::B) & 0xFF, 0x12);
        assert_eq!(state.pc(), 1);
        assert_eq!(cycles, 3);
    }

    #[test]
    fn execute_async_pmdf_imm_ticks() {
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0x47; // PMDF IMem8, imm8
        bus.mem[1] = 0x10;
        bus.mem[2] = 0x05;
        bus.mem[0x10] = 0x03;
        let mut state = LlamaState::new();
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0x47, &mut state, &mut bus, &mut ticker))
            .expect("execute async pmdf");

        assert_eq!(len, 3);
        assert_eq!(bus.mem[0x10], 0x08);
        assert_eq!(state.pc(), 3);
        assert_eq!(cycles, 4);
    }

    #[test]
    fn execute_async_dadl_imem_ticks() {
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0xC4; // DADL IMem8, IMem8
        bus.mem[1] = 0x10;
        bus.mem[2] = 0x11;
        bus.mem[0x10] = 0x09;
        bus.mem[0x11] = 0x01;
        let mut state = LlamaState::new();
        state.set_reg(RegName::I, 1);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0xC4, &mut state, &mut bus, &mut ticker))
            .expect("execute async dadl");

        assert_eq!(len, 3);
        assert_eq!(bus.mem[0x10], 0x10);
        assert_eq!(state.get_reg(RegName::FC) & 1, 0);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 0);
        assert_eq!(state.get_reg(RegName::I), 0);
        assert_eq!(state.pc(), 3);
        assert_eq!(cycles, 5);
    }

    #[test]
    fn execute_async_dadl_reg_source_only_first_byte() {
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0xC5; // DADL IMem8, A
        bus.mem[1] = 0x10; // destination base
        bus.mem[0x10] = 0x00;
        bus.mem[0x0F] = 0x00;
        bus.mem[0x0E] = 0x00;
        let mut state = LlamaState::new();
        state.set_reg(RegName::A, 0x25);
        state.set_reg(RegName::I, 3);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0xC5, &mut state, &mut bus, &mut ticker))
            .expect("execute async dadl reg source");

        assert_eq!(len, 2);
        assert_eq!(bus.mem[0x10], 0x25);
        assert_eq!(bus.mem[0x0F], 0x00);
        assert_eq!(bus.mem[0x0E], 0x00);
        assert_eq!(state.get_reg(RegName::I), 0);
        assert_eq!(state.pc(), 2);
        assert_eq!(cycles, 6);
    }

    #[test]
    fn execute_async_dsbl_reg_ticks() {
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0xD5; // DSBL IMem8, A
        bus.mem[1] = 0x10;
        bus.mem[0x10] = 0x10;
        let mut state = LlamaState::new();
        state.set_reg(RegName::A, 0x01);
        state.set_reg(RegName::FC, 0);
        state.set_reg(RegName::I, 1);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0xD5, &mut state, &mut bus, &mut ticker))
            .expect("execute async dsbl");

        assert_eq!(len, 2);
        assert_eq!(bus.mem[0x10], 0x09);
        assert_eq!(state.get_reg(RegName::FC) & 1, 0);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 0);
        assert_eq!(state.get_reg(RegName::I), 0);
        assert_eq!(state.pc(), 2);
        assert_eq!(cycles, 4);
    }

    #[test]
    fn execute_async_cmpw_imem_ticks() {
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0xC6; // CMPW IMem16, IMem16
        bus.mem[1] = 0x10;
        bus.mem[2] = 0x12;
        bus.mem[0x10] = 0x34;
        bus.mem[0x11] = 0x12;
        bus.mem[0x12] = 0x78;
        bus.mem[0x13] = 0x12;
        let mut state = LlamaState::new();
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0xC6, &mut state, &mut bus, &mut ticker))
            .expect("execute async cmpw");

        assert_eq!(len, 3);
        assert_eq!(state.get_reg(RegName::FC) & 1, 1);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 0);
        assert_eq!(state.pc(), 3);
        assert_eq!(cycles, 8);
    }

    #[test]
    fn execute_async_cmpw_reg3_ticks() {
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0xD6; // CMPW IMem16, Reg3
        bus.mem[1] = 0x02; // BA
        bus.mem[2] = 0x10;
        bus.mem[0x10] = 0x34;
        bus.mem[0x11] = 0x12;
        let mut state = LlamaState::new();
        state.set_reg(RegName::BA, 0x1234);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0xD6, &mut state, &mut bus, &mut ticker))
            .expect("execute async cmpw reg3");

        assert_eq!(len, 3);
        assert_eq!(state.get_reg(RegName::FC) & 1, 0);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 1);
        assert_eq!(state.pc(), 3);
        assert_eq!(cycles, 7);
    }

    #[test]
    fn execute_async_cmpp_reg3_ticks() {
        let mut bus = MemBus::with_size(0x100);
        bus.mem[0] = 0xD7; // CMPP IMem20, Reg3
        bus.mem[1] = 0x04; // X
        bus.mem[2] = 0x10;
        bus.mem[0x10] = 0x01;
        bus.mem[0x11] = 0x02;
        bus.mem[0x12] = 0x03;
        let mut state = LlamaState::new();
        state.set_reg(RegName::X, 0x030201);
        state.set_pc(0);
        let mut exec = LlamaExecutor::new();
        let mut cycles = 0u64;
        let mut ticker = TickHelper::new(&mut cycles, false, None);

        let len = block_on(exec.execute_async(0xD7, &mut state, &mut bus, &mut ticker))
            .expect("execute async cmpp");

        assert_eq!(len, 3);
        assert_eq!(state.get_reg(RegName::FC) & 1, 0);
        assert_eq!(state.get_reg(RegName::FZ) & 1, 1);
        assert_eq!(state.pc(), 3);
        assert_eq!(cycles, 9);
    }
}
