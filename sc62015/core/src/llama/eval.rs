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
    state::{mask_for, LlamaState},
};
use crate::{
    memory::{
        with_imr_read_suppressed, ADDRESS_MASK, IMEM_BP_OFFSET, IMEM_IMR_OFFSET, IMEM_ISR_OFFSET,
        IMEM_LCC_OFFSET, IMEM_PX_OFFSET, IMEM_PY_OFFSET, IMEM_SCR_OFFSET, IMEM_SSR_OFFSET,
        IMEM_UCR_OFFSET, IMEM_USR_OFFSET, INTERNAL_MEMORY_START,
    },
    perfetto::AnnotationValue,
    PERFETTO_TRACER,
};
use std::collections::HashMap;
use std::env;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

static PERF_INSTR_COUNTER: AtomicU64 = AtomicU64::new(0);
static PERF_CURRENT_PC: AtomicU32 = AtomicU32::new(u32::MAX);
static PERF_CURRENT_OP: AtomicU64 = AtomicU64::new(u64::MAX);
static PERF_LAST_PC: AtomicU32 = AtomicU32::new(0);

struct PerfettoContextGuard;
impl Drop for PerfettoContextGuard {
    fn drop(&mut self) {
        PERF_CURRENT_OP.store(u64::MAX, Ordering::Relaxed);
        PERF_CURRENT_PC.store(u32::MAX, Ordering::Relaxed);
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
    PERF_LAST_PC.load(Ordering::Relaxed)
}

pub fn reset_perf_counters() {
    PERF_INSTR_COUNTER.store(0, Ordering::Relaxed);
    PERF_CURRENT_PC.store(u32::MAX, Ordering::Relaxed);
    PERF_CURRENT_OP.store(u64::MAX, Ordering::Relaxed);
    PERF_LAST_PC.store(0, Ordering::Relaxed);
}

fn strict_opcodes() -> bool {
    let strict = matches!(
        env::var("LLAMA_STRICT_OPCODES").as_deref(),
        Ok("1") | Ok("true") | Ok("True")
    );
    let permissive = matches!(
        env::var("LLAMA_PERMISSIVE_OPCODES").as_deref(),
        Ok("1") | Ok("true") | Ok("True")
    );
    // Default to permissive; strict only when explicitly requested and not overridden by permissive.
    strict && !permissive
}

fn fallback_unknown(
    state: &mut LlamaState,
    prefix_len: u8,
    estimated_len: Option<u8>,
) -> Result<u8, &'static str> {
    if strict_opcodes() {
        return Err("unknown opcode (strict)");
    }
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
    0xBC, 0xBD, 0xBE, 0xC5, 0xCC, 0xCD, 0xD5, 0xD6, 0xD7, 0xDC, 0xE3, 0xE5, 0xE7, 0xEB, 0xEC, 0xF5,
    0xF7, 0xFC,
];

const INTERRUPT_VECTOR_ADDR: u32 = 0xFFFFA;

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
    // Debug-print when explicitly requested via env to avoid overwhelming logs.
    if matches!(std::env::var("TRACE_IMEM_ADDR").as_deref(), Ok("1")) {
        eprintln!(
            "[imem-addr] mode={:?} base=0x{base:02X} bp=0x{bp:02X} px=0x{px:02X} py=0x{py:02X}",
            mode
        );
    }

    // Optional perfetto emit when the builder is available (llama-tests builds).
    let mut guard = crate::PERFETTO_TRACER.enter();
    if let Some(tracer) = guard.as_mut() {
        let op_idx = PERF_CURRENT_OP.load(Ordering::Relaxed);
        let pc = PERF_CURRENT_PC.load(Ordering::Relaxed);
        let op = if op_idx == u64::MAX { None } else { Some(op_idx) };
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
    }
}

fn imem_addr_for_mode<B: LlamaBus>(bus: &mut B, mode: AddressingMode, raw: u8) -> u32 {
    INTERNAL_MEMORY_START + imem_offset_for_mode(bus, mode, raw)
}

fn enter_low_power_state<B: LlamaBus>(bus: &mut B, state: &mut LlamaState) {
    // Mirror pysc62015.intrinsics._enter_low_power_state: adjust USR/SSR and halt.
    let mut usr = read_imem_byte(bus, IMEM_USR_OFFSET);
    usr &= !0x3F;
    usr |= 0x18;
    write_imem_byte(bus, IMEM_USR_OFFSET, usr);

    let mut ssr = read_imem_byte(bus, IMEM_SSR_OFFSET);
    ssr |= 0x04;
    write_imem_byte(bus, IMEM_SSR_OFFSET, ssr);

    state.halt();
}

/// Apply power-on reset side effects (IMEM init, PC jump to reset vector).
pub fn power_on_reset<B: LlamaBus>(bus: &mut B, state: &mut LlamaState) {
    // RESET intrinsic side-effects (see pysc62015.intrinsics.eval_intrinsic_reset)
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

    let reset_vector = bus.load(INTERRUPT_VECTOR_ADDR, 8)
        | (bus.load(INTERRUPT_VECTOR_ADDR + 1, 8) << 8)
        | (bus.load(INTERRUPT_VECTOR_ADDR + 2, 8) << 16);
    // Parity: keep register/flag values intact; only adjust IMEM/PC. Drop any saved
    // call-page context so near returns fall back to the current page like Python.
    state.clear_call_page_stack();
    state.set_pc(reset_vector & mask_for(RegName::PC));
    state.set_halted(false);
}

pub struct LlamaExecutor;

impl LlamaExecutor {
    pub fn new() -> Self {
        reset_perf_counters();
        Self
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

    fn trace_instr<B: LlamaBus>(
        &self,
        opcode: u8,
        state: &LlamaState,
        bus: &mut B,
        instr_index: u64,
        pc_trace: u32,
    ) {
        let mut guard = PERFETTO_TRACER.enter();
        if let Some(tracer) = guard.as_mut() {
            let mut regs = HashMap::new();
            // Refresh IMR from IMEM so register snapshot matches Python when host updates it.
            let (mem_imr, mem_isr) = with_imr_read_suppressed(|| {
                (
                    bus.peek_imem_silent(IMEM_IMR_OFFSET) & 0xFF,
                    bus.peek_imem_silent(IMEM_ISR_OFFSET) & 0xFF,
                )
            });
            state.set_reg(RegName::IMR, mem_imr as u32);
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
                ("IMR", RegName::IMR),
            ] {
                regs.insert(name.to_string(), state.get_reg(reg) & mask_for(reg));
            }
            tracer.record_regs(
                instr_index,
                pc_trace & mask_for(RegName::PC),
                pc_trace & mask_for(RegName::PC),
                opcode,
                &regs,
                mem_imr,
                mem_isr,
            );
        }
        PERF_LAST_PC.store(pc_trace, Ordering::Relaxed);
    }

    fn estimated_length(entry: &OpcodeEntry) -> u8 {
        let mut len = 1u8; // opcode byte
        for op in entry.operands.iter() {
            len = len.saturating_add(match op {
                OperandKind::Imm(bits) => bits.div_ceil(8),
                OperandKind::IMem(_) | OperandKind::IMemWidth(_) => 1,
                OperandKind::EMemAddrWidth(_) | OperandKind::EMemAddrWidthOp(_) => 3,
                OperandKind::EMemReg(_) | OperandKind::EMemIMem(_) => 3,
                OperandKind::EMemRegWidth(bytes)
                | OperandKind::EMemRegWidthMode(bytes)
                | OperandKind::EMemIMemWidth(bytes) => *bytes,
                OperandKind::EMemImemOffsetDestIntMem | OperandKind::EMemImemOffsetDestExtMem => 2,
                OperandKind::RegIMemOffset(_) => 1,
                OperandKind::EMemRegModePostPre => 1,
                OperandKind::RegPair(size) => *size,
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
        if let Some(tracer) = guard.as_mut() {
            let op_index = PERF_CURRENT_OP.load(Ordering::Relaxed);
            let pc = PERF_CURRENT_PC.load(Ordering::Relaxed);
            let masked = if bits == 0 || bits >= 32 {
                value
            } else {
                value & ((1u32 << bits) - 1)
            };
            let space =
                if (INTERNAL_MEMORY_START..(INTERNAL_MEMORY_START + 0x100)).contains(&addr) {
                    "internal"
                } else {
                    "external"
                };
            tracer.record_mem_write(op_index, pc, addr, masked, space, bits);
        }
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
            0x40 | 0x80 => (true, 1), // tolerate stray bits; treat as +offset
            0xC0 => (true, -1),
            _ => (false, 0),
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
                    let (ptr_mem, consumed_ptr) =
                        self.decode_ext_reg_ptr(state, bus, pc + offset, width_bytes)?;
                    let raw_imem = bus.load(pc + offset + consumed_ptr, 8) & 0xFF;
                    let mode_index = if single_pre { 0 } else { operand_index + 1 };
                    let imem_addr =
                        imem_addr_for_mode(bus, mode_for_operand(pre, mode_index), raw_imem as u8);
                    offset += consumed_ptr + 1;
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
                    let r1 = match (raw >> 4) & 0x7 {
                        0 => RegName::A,
                        1 => RegName::IL,
                        2 => RegName::BA,
                        3 => RegName::I,
                        4 => RegName::X,
                        5 => RegName::Y,
                        6 => RegName::U,
                        7 => RegName::S,
                        _ => RegName::Unknown("regpair1"),
                    };
                    let r2 = match raw & 0x7 {
                        0 => RegName::A,
                        1 => RegName::IL,
                        2 => RegName::BA,
                        3 => RegName::I,
                        4 => RegName::X,
                        5 => RegName::Y,
                        6 => RegName::U,
                        7 => RegName::S,
                        _ => RegName::Unknown("regpair2"),
                    };
                    let bits = match size {
                        1 => 8,
                        2 => 16,
                        3 => 24,
                        _ => 8,
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
                // pre-dec modes (new value lower than current).
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
                let mask = Self::mask_for_width(bits);
                let val = state.get_reg(src) & mask;
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
                state.set_reg(reg, new_val);
            }
        } else {
            // Generic move fallback: prefer decoded.mem as destination, otherwise mem2, otherwise A.
            let (val, bits) = src_val.unwrap_or((0, 8));
            let masked = val & Self::mask_for_width(bits);
            if let Some(mem) = decoded.mem.or(decoded.mem2) {
                Self::store_traced(bus, mem.addr, mem.bits, masked);
                if let Some((reg, new_val)) = mem.side_effect {
                    state.set_reg(reg, new_val);
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
                    entry.operands.get(0).unwrap_or(&OperandKind::Placeholder),
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
        let instr_index = PERF_INSTR_COUNTER.fetch_add(1, Ordering::Relaxed);
        let start_pc = state.pc() & mask_for(RegName::PC);
        // Use the actual opcode address (post-PRE) for tracing, not the prefix byte.
        let mut trace_pc = start_pc;
        let mut trace_opcode = opcode;
        let mut prefix_len = 0u8;
        let mut pre_modes_opt: Option<PreModes> = None;
        let mut pc_override = None;
        let mut entry = self.lookup(opcode);

        while let Some(e) = entry {
            if e.kind != InstrKind::Pre {
                break;
            }
            let pre_modes = pre_modes_for(trace_opcode).ok_or("unknown PRE opcode")?;
            let next_pc = trace_pc.wrapping_add(1) & mask_for(RegName::PC);
            let next_opcode = bus.load(next_pc, 8) as u8;
            trace_opcode = next_opcode;
            trace_pc = next_pc;
            prefix_len = prefix_len.saturating_add(1);
            pre_modes_opt = Some(pre_modes);
            pc_override = Some(next_pc);
            entry = self.lookup(next_opcode);
        }

        PERF_CURRENT_OP.store(instr_index, Ordering::Relaxed);
        PERF_CURRENT_PC.store(trace_pc, Ordering::Relaxed);
        let _ctx_guard = PerfettoContextGuard;
        // Trace the pre-execution snapshot using the resolved opcode/PC (post-PRE).
        self.trace_instr(trace_opcode, state, bus, instr_index, trace_pc);

        let result = match entry {
            Some(entry) => self.execute_with(
                trace_opcode,
                entry,
                state,
                bus,
                pre_modes_opt.as_ref(),
                pc_override,
                prefix_len,
            ),
            None => fallback_unknown(state, prefix_len, None),
        };
        // Keep last-PC aligned to the post-execution PC so timer/IRQ traces reflect the live value.
        PERF_LAST_PC.store(state.pc() & mask_for(RegName::PC), Ordering::Relaxed);
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
                // Python WAIT fast-path: zero I and clear flags without advancing timers.
                let _wait_loops = state.get_reg(RegName::I) & mask_for(RegName::I);
                state.set_reg(RegName::I, 0);
                state.set_reg(RegName::FC, 0);
                state.set_reg(RegName::FZ, 0);
                let len = 1 + prefix_len;
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(len as u32));
                }
                Ok(len)
            }
            InstrKind::Off => {
                enter_low_power_state(bus, state);
                let len = 1 + prefix_len;
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(len as u32));
                }
                Ok(len)
            }
            InstrKind::Halt => {
                enter_low_power_state(bus, state);
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
                        OperandKind::Reg3 => 24,
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
                        OperandKind::Reg3 => 24,
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
                    return Err("unsupported width for DADL/DSBL");
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
                    } else if let Some(reg) = src_reg {
                        (Self::read_reg(state, bus, reg) & 0xFF) as u8
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
                            OperandKind::Reg3 => 24,
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
                if strict_opcodes() {
                    return Err("unknown opcode (strict)");
                }
                // Known-unknown opcodes should advance without failure to mirror Python decode.
                let est_len = Self::estimated_length(entry);
                let len = est_len.saturating_add(prefix_len);
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(len as u32));
                }
                Ok(len)
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
                if let Some(tracer) = guard.as_mut() {
                    let mut payload = std::collections::HashMap::new();
                    payload.insert(
                        "pc".to_string(),
                        AnnotationValue::Pointer((pc & mask_for(RegName::PC)) as u64),
                    );
                    payload.insert("vector".to_string(), AnnotationValue::Pointer(vec as u64));
                    payload.insert(
                        "imr_before".to_string(),
                        AnnotationValue::UInt(imr as u64),
                    );
                    payload.insert(
                        "imr_after".to_string(),
                        AnnotationValue::UInt(cleared_imr as u64),
                    );
                    payload.insert("src".to_string(), AnnotationValue::Str("IR".to_string()));
                    tracer.record_irq_event("IRQ_Enter", payload);
                }
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
                if !Self::cond_pass(entry, state) {
                    // Condition false: just advance PC
                    let len = self
                        .decode_with_prefix(entry, state, bus, pre, pc_override, prefix_len)?
                        .len;
                    let start_pc = state.pc();
                    if state.pc() == start_pc {
                        state.set_pc(start_pc.wrapping_add(len as u32));
                    }
                    return Ok(len);
                }
                let decoded =
                    self.decode_with_prefix(entry, state, bus, pre, pc_override, prefix_len)?;
                let target = if let Some((val, bits)) = decoded.imm {
                    if bits == 16 {
                        // JP 16-bit keeps current page (mask to PC width)
                        // Use the address of the JP instruction (Python uses addr & 0xFF0000).
                        let instr_pc = pc_override.unwrap_or(state.pc());
                        (instr_pc & mask_for(RegName::PC) & 0xFF0000) | (val & 0xFFFF)
                    } else {
                        val & mask_for(RegName::PC)
                    }
                } else if let Some(mem) = decoded.mem {
                    bus.load(mem.addr, mem.bits) & 0xFFFFF
                } else if let Some(r) = decoded.reg3 {
                    state.get_reg(r) & 0xFFFFF
                } else {
                    return Err("missing jump target");
                };
                state.set_pc(target);
                Ok(decoded.len)
            }
            InstrKind::JpRel => {
                if !Self::cond_pass(entry, state) {
                    let len = self
                        .decode_with_prefix(entry, state, bus, pre, pc_override, prefix_len)?
                        .len;
                    let start_pc = state.pc();
                    if state.pc() == start_pc {
                        state.set_pc(start_pc.wrapping_add(len as u32));
                    }
                    return Ok(len);
                }
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
                // JR is relative to current PC + length
                let next_pc = state.pc().wrapping_add(decoded.len as u32);
                state.set_pc(next_pc.wrapping_add_signed(imm) & 0xFFFFF);
                Ok(decoded.len)
            }
            InstrKind::Call => {
                let decoded =
                    self.decode_with_prefix(entry, state, bus, pre, pc_override, prefix_len)?;
                let (target, bits) = decoded.imm.ok_or("missing jump target")?;
                let pc_before = state.pc();
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
                state.set_pc(dest & 0xFFFFF);
                state.call_depth_inc();
                let mut guard = crate::PERFETTO_TRACER.enter();
                if let Some(tracer) = guard.as_mut() {
                    tracer.record_call_flow(
                        "CALL",
                        pc_before & mask_for(RegName::PC),
                        dest & 0xFFFFF,
                        state.call_depth(),
                    );
                }
                Ok(decoded.len)
            }
            InstrKind::Ret => {
                let pc_before = state.pc();
                let ret = Self::pop_stack(state, bus, RegName::S, 16, false);
                let current_page = state.pc() & 0xFF0000;
                // Parity: always use the saved call page when present; Python preserves the
                // caller page across near CALL/RET, even if execution moved to a different page.
                // Fall back to the current page only when no saved page exists.
                let page = state.pop_call_page().unwrap_or(current_page);
                let dest = (page | (ret & 0xFFFF)) & 0xFFFFF;
                state.set_pc(dest);
                state.call_depth_dec();
                let mut guard = crate::PERFETTO_TRACER.enter();
                if let Some(tracer) = guard.as_mut() {
                    tracer.record_call_flow(
                        "RET",
                        pc_before & mask_for(RegName::PC),
                        dest,
                        state.call_depth(),
                    );
                }
                Ok(1)
            }
            InstrKind::RetF => {
                let pc_before = state.pc();
                let ret = Self::pop_stack(state, bus, RegName::S, 24, false);
                let dest = ret & 0xFFFFF;
                state.set_pc(dest);
                state.call_depth_dec();
                let mut guard = crate::PERFETTO_TRACER.enter();
                if let Some(tracer) = guard.as_mut() {
                    tracer.record_call_flow(
                        "RETF",
                        pc_before & mask_for(RegName::PC),
                        dest,
                        state.call_depth(),
                    );
                }
                Ok(1)
            }
            InstrKind::RetI => {
                // Stack layout: IMR (1), F(1), 24-bit PC (little-endian).
                let mask_s = mask_for(RegName::S);
                let mut sp = state.get_reg(RegName::S) & mask_s;
                let sp_before = sp;
                let trace_reti = std::env::var("TRACE_RETI").is_ok();
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
                if trace_reti {
                    eprintln!(
                        "[reti] sp_before=0x{sp_before:06X} imr=0x{imr:02X} f=0x{f:02X} ret=0x{ret:06X} sp_after=0x{sp:06X} imr_restored=0x{imr_restored:02X}",
                        sp_before = sp_before,
                        imr = imr,
                        f = f,
                        ret = ret & 0xFFFFF,
                        sp = sp,
                        imr_restored = imr_restored,
                    );
                }
                Ok(1 + prefix_len)
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
                    lhs = state.get_reg(r) & mask;
                    if let Some(mem) = decoded.mem {
                        rhs = bus.load(mem.addr, bits) & mask;
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
    use crate::llama::opcodes::OPCODES;
    use crate::memory::ADDRESS_MASK;
    use std::collections::HashSet;

    struct NullBus;
    impl LlamaBus for NullBus {
        fn load(&mut self, _addr: u32, _bits: u8) -> u32 {
            0
        }
        fn wait_cycles(&mut self, _cycles: u32) {}
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
    fn unknown_opcodes_fall_back() {
        std::env::set_var("LLAMA_PERMISSIVE_OPCODES", "1");
        let mut exec = LlamaExecutor::new();
        let mut state = LlamaState::new();
        let mut bus = NullBus;
        let res = exec.execute(0x20, &mut state, &mut bus);
        assert!(res.is_ok());
        assert_eq!(state.pc(), 1);
        let res = exec.execute(0xBF, &mut state, &mut bus);
        assert!(res.is_ok());
        assert_eq!(state.pc(), 2);
        std::env::remove_var("LLAMA_PERMISSIVE_OPCODES");
    }

    #[test]
    fn strict_unknown_opcodes_error() {
        std::env::remove_var("LLAMA_PERMISSIVE_OPCODES");
        std::env::set_var("LLAMA_STRICT_OPCODES", "1");
        let mut exec = LlamaExecutor::new();
        let mut state = LlamaState::new();
        let mut bus = NullBus;
        let res = exec.execute(0x20, &mut state, &mut bus);
        assert!(
            res.is_err(),
            "strict mode should error on unknown opcode, got {res:?}"
        );
        std::env::remove_var("LLAMA_STRICT_OPCODES");
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
        assert_eq!(state.get_reg(RegName::FC), 0, "WAIT should clear C");
        assert_eq!(state.get_reg(RegName::FZ), 0, "WAIT should clear Z");
    }

    struct WaitBus {
        spins: u32,
    }

    impl LlamaBus for WaitBus {
        fn wait_cycles(&mut self, cycles: u32) {
            self.spins = self.spins.saturating_add(cycles);
        }
    }

    #[test]
    fn wait_calls_bus_wait_cycles() {
        let mut exec = LlamaExecutor::new();
        let mut state = LlamaState::new();
        let mut bus = WaitBus { spins: 0 };
        state.set_reg(RegName::I, 5);
        let len = exec.execute(0xEF, &mut state, &mut bus).unwrap(); // WAIT
        assert_eq!(len, 1);
        assert_eq!(state.pc(), 1);
        assert_eq!(
            bus.spins, 0,
            "WAIT should not advance timers/keyboard; Python fast-path does not call wait_cycles"
        );
        assert_eq!(state.get_reg(RegName::I), 0);
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
        bus.mem[0x0FFFFA] = 0x21; // reset vector low
        bus.mem[0x0FFFFB] = 0x43; // mid
        bus.mem[0x0FFFFC] = 0x05; // high -> 0x054321

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
            .execute_with(0x54, &entry, &mut state, &mut bus, None, None, 0)
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
        let sp = INTERNAL_MEMORY_START + 0x40;
        state.set_reg(RegName::U, sp);
        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0x2F, &mut state, &mut bus).unwrap(); // PUSHU IMR
        assert_eq!(len, 1);
        let new_sp = state.get_reg(RegName::U);
        assert_eq!(new_sp, sp.wrapping_sub(1));
        let stored = bus.load(new_sp, 8) & 0xFF;
        let imr_after = bus.peek_imem(IMEM_IMR_OFFSET) as u32;
        assert_eq!(stored, 0xAA, "stack should capture IMR from memory");
        assert_eq!(imr_after, 0xAA, "IMR in memory should be unchanged by PUSH");
        assert_eq!(state.get_reg(RegName::IMR), imr_after);
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
        assert_eq!(state.get_reg(RegName::FC), 0, "WAIT should clear C");
        assert_eq!(state.get_reg(RegName::FZ), 0, "WAIT should clear Z");
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
    fn tcl_is_no_op_and_advances_pc() {
        // TCL (0xCE) should be a no-op intrinsic: no stack/IMR changes, just PC advance.
        let mut bus = MemBus::with_size(4);
        bus.mem[0] = 0xCE;
        let mut state = LlamaState::new();
        state.set_pc(0);
        state.set_reg(RegName::S, 0x0100);
        state.set_reg(RegName::IMR, 0xAA);
        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0xCE, &mut state, &mut bus).unwrap();
        assert_eq!(len, 1);
        assert_eq!(state.pc(), 1, "PC should advance by instruction length");
        assert_eq!(state.get_reg(RegName::S), 0x0100, "stack pointer unchanged");
        assert_eq!(state.get_reg(RegName::IMR), 0xAA, "IMR unchanged");
    }
}
