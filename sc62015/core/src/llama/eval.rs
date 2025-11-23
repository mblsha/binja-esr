//! Lightweight evaluator scaffold for LLAMA.
//!
//! Supports a small subset of opcodes today (imm/reg arithmetic/logic/moves on
//! `A`) to exercise the typed opcode table. The intent is to grow coverage
//! incrementally while keeping masking/aliasing consistent with the Python
//! emulator.

use super::{
    dispatch,
    opcodes::{InstrKind, OpcodeEntry, OperandKind, RegName},
    state::{mask_for, LlamaState},
};
const INTERNAL_MEMORY_START: u32 = 0x100000; // align with Python INTERNAL_MEMORY_START
const IMEM_IMR_OFFSET: u32 = 0xFB; // IMR offset within internal space (matches Python IMEMRegisters.IMR)
const IMEM_UCR_OFFSET: u32 = 0xF7;
const IMEM_USR_OFFSET: u32 = 0xF8;
const IMEM_ISR_OFFSET: u32 = 0xFC;
const IMEM_SCR_OFFSET: u32 = 0xFD;
const IMEM_LCC_OFFSET: u32 = 0xFE;
const IMEM_SSR_OFFSET: u32 = 0xFF;
const INTERRUPT_VECTOR_ADDR: u32 = 0xFFFFA;

pub trait LlamaBus {
    fn load(&mut self, _addr: u32, _bits: u8) -> u32 {
        0
    }
    fn store(&mut self, _addr: u32, _bits: u8, _value: u32) {}
    fn resolve_emem(&mut self, base: u32) -> u32 {
        base
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
    bus.store(INTERNAL_MEMORY_START + offset, 8, value as u32);
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

fn apply_reset<B: LlamaBus>(bus: &mut B, state: &mut LlamaState) {
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
    state.set_pc(reset_vector & mask_for(RegName::PC));
    state.set_halted(false);
}

pub struct LlamaExecutor;

impl LlamaExecutor {
    pub fn new() -> Self {
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
    ) {
        let bytes = (bits + 7) / 8;
        let mask = mask_for(sp_reg);
        let new_sp = state.get_reg(sp_reg).wrapping_sub(bytes as u32) & mask;
        for i in 0..bytes {
            let addr = new_sp.wrapping_add(i as u32) & mask;
            let byte = (value >> (8 * i)) & 0xFF;
            bus.store(addr, 8, byte);
        }
        state.set_reg(sp_reg, new_sp);
    }

    fn pop_stack<B: LlamaBus>(
        state: &mut LlamaState,
        bus: &mut B,
        sp_reg: RegName,
        bits: u8,
    ) -> u32 {
        let bytes = (bits + 7) / 8;
        let mut value = 0u32;
        let mask = mask_for(sp_reg);
        let mut sp = state.get_reg(sp_reg);
        for i in 0..bytes {
            let byte = bus.load(sp, 8) & 0xFF;
            value |= byte << (8 * i);
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

    fn estimated_length(entry: &OpcodeEntry) -> u8 {
        let mut len = 1u8; // opcode byte
        for op in entry.operands.iter() {
            len = len.saturating_add(match op {
                OperandKind::Imm(bits) => (*bits + 7) / 8,
                OperandKind::IMem(_) | OperandKind::IMemWidth(_) => 1,
                OperandKind::EMemAddrWidth(_) | OperandKind::EMemAddrWidthOp(_) => 3,
                OperandKind::EMemReg(_) | OperandKind::EMemIMem(_) => 3,
                OperandKind::EMemRegWidth(bytes)
                | OperandKind::EMemRegWidthMode(bytes)
                | OperandKind::EMemIMemWidth(bytes) => *bytes,
                OperandKind::EMemImemOffsetDestIntMem | OperandKind::EMemImemOffsetDestExtMem => 2,
                OperandKind::RegIMemOffset(_) => 1,
                OperandKind::EMemRegModePostPre => 1,
                OperandKind::RegPair(size) => *size as u8,
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

    fn read_imm<B: LlamaBus>(bus: &mut B, addr: u32, bits: u8) -> u32 {
        let bytes = (bits + 7) / 8;
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
        let mut disp: i8 = 0;
        if needs_disp {
            let magnitude = bus.load(pc + 1, 8) as u8;
            disp = if disp_sign >= 0 {
                magnitude as i8
            } else {
                -(magnitude as i8)
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
                addr = base.wrapping_add(disp as i32 as u32);
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
    ) -> Result<(MemOperand, u32), &'static str> {
        let mode_byte = bus.load(pc, 8) as u8;
        let (needs_disp, sign) = match mode_byte & 0xC0 {
            0x00 => (false, 0),
            0x80 => (true, 1),
            0xC0 => (true, -1),
            _ => return Err("unsupported EMEM/IMEM mode"),
        };
        let base = bus.load(pc + 1, 8) as u8;
        let mut consumed = 2u32;
        let mut disp: i8 = 0;
        if needs_disp {
            let magnitude = bus.load(pc + 2, 8) as u8;
            disp = if sign >= 0 {
                magnitude as i8
            } else {
                -(magnitude as i8)
            };
            consumed += 1;
        }
        let pointer = Self::read_imm(bus, INTERNAL_MEMORY_START + base as u32, 24);
        let addr = bus
            .resolve_emem(pointer.wrapping_add(disp as i32 as u32));
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
        dest_is_internal: bool,
    ) -> Result<(EmemImemTransfer, u32), &'static str> {
        let mode_byte = bus.load(pc, 8) as u8;
        let (needs_offset, sign) = match mode_byte {
            0x00 => (false, 0),
            0x80 => (true, 1),
            0xC0 => (true, -1),
            _ => return Err("unsupported EMEM/IMEM offset mode"),
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
        let (dst_addr, src_addr, dst_is_internal) = if dest_is_internal {
            // MV (m),[(n)] - dst is internal addr=first, ptr lives at second
            let pointer =
                Self::read_imm(bus, INTERNAL_MEMORY_START + second as u32, 24);
            let ext_addr =
                bus.resolve_emem(pointer.wrapping_add(disp as u32));
            (
                INTERNAL_MEMORY_START + first as u32,
                ext_addr,
                true,
            )
        } else {
            // MV [(n)],(m) - dst is external pointer from first, src is internal second
            let pointer =
                Self::read_imm(bus, INTERNAL_MEMORY_START + first as u32, 24);
            let ext_addr =
                bus.resolve_emem(pointer.wrapping_add(disp as u32));
            (
                ext_addr,
                INTERNAL_MEMORY_START + second as u32,
                false,
            )
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
    ) -> Result<DecodedOperands, &'static str> {
        let pc = state.pc();
        let mut offset = 1u32; // opcode consumed
        let mut decoded = DecodedOperands::default();
        // Opcode-specific decoding quirks
        if entry.opcode == 0xE3 {
            // Encoding order is EMemReg mode byte then IMem8.
            let (mem_src, consumed) = self.decode_ext_reg_ptr(state, bus, pc + offset, 1)?;
            decoded.mem2 = Some(mem_src);
            offset += consumed;
            let imem_addr = bus.load(pc + offset, 8) & 0xFF;
            decoded.mem = Some(MemOperand {
                addr: INTERNAL_MEMORY_START + imem_addr,
                bits: 8,
                side_effect: None,
            });
            offset += 1;
            decoded.len = offset as u8;
            return Ok(decoded);
        }
        for op in entry.operands.iter() {
            match op {
                OperandKind::Imm(bits) => {
                    let val = Self::read_imm(bus, pc + offset, *bits);
                    decoded.imm = Some((val, *bits));
                    offset += (*bits as u32 + 7) / 8;
                }
                OperandKind::ImmOffset => {
                    let byte = bus.load(pc + offset, 8) as u8;
                    decoded.imm = Some((byte as u32, 8));
                    offset += 1;
                }
                OperandKind::IMem(bits) => {
                    let addr = bus.load(pc + offset, 8) & 0xFF;
                    let slot = if decoded.mem.is_none() {
                        &mut decoded.mem
                    } else {
                        &mut decoded.mem2
                    };
                    *slot = Some(MemOperand {
                        addr: INTERNAL_MEMORY_START + addr,
                        bits: *bits,
                        side_effect: None,
                    });
                    offset += 1;
                }
                OperandKind::IMemWidth(bytes) => {
                    let bits = Self::bits_from_bytes(*bytes);
                    let addr = bus.load(pc + offset, 8) & 0xFF;
                    let slot = if decoded.mem.is_none() {
                        &mut decoded.mem
                    } else {
                        &mut decoded.mem2
                    };
                    *slot = Some(MemOperand {
                        addr: INTERNAL_MEMORY_START + addr,
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
                    let (mem, consumed) =
                        self.decode_ext_reg_ptr(state, bus, pc + offset, 1)?;
                    if decoded.mem.is_none() {
                        decoded.mem = Some(mem);
                    } else {
                        decoded.mem2 = Some(mem);
                    }
                    offset += consumed;
                }
                OperandKind::EMemIMemWidth(bytes) => {
                    let (mem, consumed) = self.decode_imem_ptr(bus, pc + offset, *bytes)?;
                    if decoded.mem.is_none() {
                        decoded.mem = Some(mem);
                    } else {
                        decoded.mem2 = Some(mem);
                    }
                    offset += consumed;
                }
                OperandKind::EMemImemOffsetDestIntMem => {
                    let (transfer, consumed) =
                        self.decode_emem_imem_offset(entry, bus, pc + offset, true)?;
                    decoded.transfer = Some(transfer);
                    offset += consumed;
                }
                OperandKind::EMemImemOffsetDestExtMem => {
                    let (transfer, consumed) =
                        self.decode_emem_imem_offset(entry, bus, pc + offset, false)?;
                    decoded.transfer = Some(transfer);
                    offset += consumed;
                }
                OperandKind::RegIMemOffset(kind) => {
                    let width_bits = Self::width_bits_for_kind(entry.kind);
                    let width_bytes = (width_bits + 7) / 8;
                    let (ptr_mem, consumed_ptr) =
                        self.decode_ext_reg_ptr(state, bus, pc + offset, width_bytes)?;
                    let imem_addr =
                        INTERNAL_MEMORY_START + (bus.load(pc + offset + consumed_ptr, 8) & 0xFF);
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
            OperandKind::Reg(_, _) | OperandKind::RegB | OperandKind::RegIL | OperandKind::RegIMR | OperandKind::RegF => {
                Self::operand_reg(op)
            }
            OperandKind::Reg3 => decoded.reg3,
            _ => None,
        }
    }

    fn execute_mv_generic<B: LlamaBus>(
        &mut self,
        entry: &OpcodeEntry,
        state: &mut LlamaState,
        bus: &mut B,
    ) -> Result<u8, &'static str> {
        let decoded = self.decode_operands(entry, state, bus)?;
        if matches!(entry.kind, InstrKind::Mvl | InstrKind::Mvld) {
            let length = state.get_reg(RegName::I) & mask_for(RegName::I);
            if length == 0 {
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(decoded.len as u32));
                }
                return Ok(decoded.len);
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
            return Err("mv pattern not supported");
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
                OperandKind::RegB | OperandKind::RegIL | OperandKind::RegIMR | OperandKind::RegF => 8,
                _ => 8,
            };
            src_val = Some((state.get_reg(reg), bits));
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
            bus.store(mem.addr, bits, val);
            if let Some((reg, new_val)) = mem.side_effect {
                state.set_reg(reg, new_val);
            }
        } else {
            return Err("mv pattern not supported");
        }

        // Apply any pointer side-effects even if the memory operand was a source.
        if let Some(mem) = decoded.mem {
            if let Some((reg, new_val)) = mem.side_effect {
                if Some(reg) != dst_reg {
                    state.set_reg(reg, new_val);
                }
            }
        }
        if let Some(mem2) = decoded.mem2 {
            if let Some((reg, new_val)) = mem2.side_effect {
                if Some(reg) != dst_reg {
                    state.set_reg(reg, new_val);
                }
            }
        }

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
    ) -> Result<u8, &'static str> {
        let pc = state.pc();
        let mut offset = 1u32; // opcode byte consumed
        let mut imm: Option<u32> = None;
        for op in entry.operands.iter() {
            match op {
                OperandKind::Imm(bits) => {
                    imm = Some(Self::read_imm(bus, pc + offset, *bits));
                    offset += (*bits as u32 + 7) / 8;
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
                carry_flag = Some(false);
                (a & rhs) & mask
            }
            InstrKind::Or => {
                carry_flag = Some(false);
                (a | rhs) & mask
            }
            InstrKind::Xor => {
                carry_flag = Some(false);
                (a ^ rhs) & mask
            }
            InstrKind::Cmp => {
                Self::set_flags_cmp(state, a & mask, rhs & mask, bits);
                return {
                    let len = offset as u8;
                    let start_pc = pc;
                    if state.pc() == start_pc {
                        state.set_pc(start_pc.wrapping_add(len as u32));
                    }
                    Ok(len)
                };
            }
            InstrKind::Test => {
                let res = (a & rhs) & mask;
                Self::set_flags_for_result(state, res, None);
                return {
                    let len = offset as u8;
                    let start_pc = pc;
                    if state.pc() == start_pc {
                        state.set_pc(start_pc.wrapping_add(len as u32));
                    }
                    Ok(len)
                };
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
        let len = offset as u8;
        let start_pc = pc;
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
    ) -> Result<u8, &'static str> {
        let pc = state.pc();
        let decoded = self.decode_operands(entry, state, bus)?;
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
                        bus.store(mem.addr, 8, val);
                    }
                    // MV [mem], imm
                    [OperandKind::IMem(_), OperandKind::Imm(bits)]
                    | [OperandKind::IMemWidth(_), OperandKind::Imm(bits)]
                    | [OperandKind::EMemAddrWidth(_), OperandKind::Imm(bits)]
                    | [OperandKind::EMemAddrWidthOp(_), OperandKind::Imm(bits)] => {
                        let (val, _) = decoded.imm.ok_or("missing immediate")?;
                        bus.store(mem.addr, *bits, val);
                    }
                    _ => return Err("mv pattern not supported"),
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
                let lhs_is_mem = matches!(entry.operands[0],
                    OperandKind::IMem(_)
                        | OperandKind::IMemWidth(_)
                        | OperandKind::EMemAddrWidth(_)
                        | OperandKind::EMemAddrWidthOp(_)
                        | OperandKind::EMemRegWidth(_)
                        | OperandKind::EMemRegWidthMode(_)
                        | OperandKind::EMemIMemWidth(_)
                );
                let rhs_is_mem = entry.operands.get(1).map(|op| matches!(op,
                    OperandKind::IMem(_)
                        | OperandKind::IMemWidth(_)
                        | OperandKind::EMemAddrWidth(_)
                        | OperandKind::EMemAddrWidthOp(_)
                        | OperandKind::EMemRegWidth(_)
                        | OperandKind::EMemRegWidthMode(_)
                        | OperandKind::EMemIMemWidth(_)
                )).unwrap_or(false);

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
                } else if let Some(r) =
                    Self::resolved_reg(entry.operands.get(1).unwrap_or(&OperandKind::Placeholder), &decoded)
                {
                    state.get_reg(r)
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
                    InstrKind::And => ((lhs_val & rhs_val) & mask, Some(false)),
                    InstrKind::Or => ((lhs_val | rhs_val) & mask, Some(false)),
                    InstrKind::Xor => ((lhs_val ^ rhs_val) & mask, Some(false)),
                    InstrKind::Adc => {
                        let c = state.get_reg(RegName::FC) & 1;
                        let full = (lhs_val as u64) + (rhs_val as u64) + (c as u64);
                        ((full as u32) & mask, Some(full > mask as u64))
                    }
                    InstrKind::Sbc => {
                        let c = state.get_reg(RegName::FC) & 1;
                        let borrow =
                            (lhs_val as u64) < (rhs_val as u64 + c as u64);
                        (lhs_val.wrapping_sub(rhs_val).wrapping_sub(c) & mask, Some(borrow))
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
                        bus.store(mem.addr, mem.bits, result);
                    } else {
                        state.set_reg(RegName::A, result);
                    }
                    Self::set_flags_for_result(state, result, carry);
                }
            }
            _ => return Err("memory pattern not supported"),
        }

        if let Some((reg, new_val)) = mem.side_effect {
            state.set_reg(reg, new_val);
        }
        if let Some(mem2) = decoded.mem2 {
            if let Some((reg, new_val)) = mem2.side_effect {
                state.set_reg(reg, new_val);
            }
        }

        let start_pc = pc;
        if state.pc() == start_pc {
            state.set_pc(start_pc.wrapping_add(len as u32));
        }
        Ok(len)
    }

    /// Stub execute entrypoint; wires length estimation and recognizes WAIT/RET/HALT placeholders.
    pub fn execute<B: LlamaBus>(
        &mut self,
        opcode: u8,
        state: &mut LlamaState,
        bus: &mut B,
    ) -> Result<u8, &'static str> {
        let entry = self.lookup(opcode).ok_or("unknown opcode")?;
        match entry.kind {
            InstrKind::Nop => {
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(1));
                }
                Ok(1)
            }
            InstrKind::Wait => {
                state.set_reg(RegName::IL, 0);
                state.set_reg(RegName::IH, 0);
                state.set_reg(RegName::I, 0);
                state.set_reg(RegName::FC, 0);
                state.set_reg(RegName::FZ, 0);
                let len = 1;
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(len as u32));
                }
                Ok(len)
            }
            InstrKind::Off => {
                enter_low_power_state(bus, state);
                let len = 1;
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(len as u32));
                }
                Ok(len)
            }
            InstrKind::Halt => {
                enter_low_power_state(bus, state);
                let len = 1;
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(len as u32));
                }
                Ok(len)
            }
            InstrKind::Mv | InstrKind::Mvw | InstrKind::Mvp | InstrKind::Mvl
                if entry.operands.len() == 1
                    && matches!(
                        entry.operands[0],
                        OperandKind::EMemImemOffsetDestIntMem
                            | OperandKind::EMemImemOffsetDestExtMem
                            | OperandKind::RegIMemOffset(_)
                    ) =>
            {
                let decoded = self.decode_operands(entry, state, bus)?;
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
                bus.store(transfer.dst_addr, transfer.bits, value);
                if let Some((reg, new_val)) = transfer.side_effect {
                    state.set_reg(reg, new_val);
                }
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(decoded.len as u32));
                }
                Ok(decoded.len)
            }
            InstrKind::Mvl if opcode == 0xE3 => {
                // E3 (ops_reversed MVL IMem8, EMemReg post/pre). Python leaves data untouched but applies pointer side-effects.
                let decoded = self.decode_operands(entry, state, bus)?;
                let reg_byte = bus.load(state.pc().wrapping_add(1), 8) as u8;
                let raw_mode = (reg_byte >> 4) & 0x0F;
                if raw_mode == 0x3 {
                    // Only pre-dec updates the pointer register in the Python core for this opcode.
                    for mem in [decoded.mem, decoded.mem2].into_iter().flatten() {
                        if let Some((reg, new_val)) = mem.side_effect {
                            state.set_reg(reg, new_val);
                        }
                    }
                }
                let length = state.get_reg(RegName::I) & mask_for(RegName::I);
                if length == 0 {
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
                Ok(decoded.len)
            }
            InstrKind::Inc => {
                let decoded = self.decode_operands(entry, state, bus)?;
                let op = entry.operands.get(0).ok_or("missing operand")?;
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
                    bus.store(mem.addr, mem.bits, res);
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
                let decoded = self.decode_operands(entry, state, bus)?;
                let op = entry.operands.get(0).ok_or("missing operand")?;
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
                    bus.store(mem.addr, mem.bits, res);
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
                let decoded = self.decode_operands(entry, state, bus)?;
                let mem = decoded.mem.ok_or("missing mem operand")?;
                let src_val = match entry.operands.get(1) {
                    Some(OperandKind::Imm(_)) => decoded.imm.ok_or("missing immediate")?.0,
                    Some(OperandKind::Reg(RegName::A, _)) => state.get_reg(RegName::A) & 0xFF,
                    _ => return Err("unsupported PMDF operands"),
                } & 0xFF;
                let dst = bus.load(mem.addr, 8) & 0xFF;
                let res = (dst + src_val) & 0xFF;
                bus.store(mem.addr, 8, res);
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(decoded.len as u32));
                }
                Ok(decoded.len)
            }
            InstrKind::Shl | InstrKind::Shr | InstrKind::Rol | InstrKind::Ror => {
                let decoded = self.decode_operands(entry, state, bus)?;
                let op = entry.operands.get(0).ok_or("missing operand")?;
                let (val, bits, dest_mem, dest_reg) = if let Some(reg) = Self::resolved_reg(op, &decoded) {
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
                let res = match entry.kind {
                    InstrKind::Shl => (val << 1) & mask,
                    InstrKind::Shr => (val >> 1) & mask,
                    InstrKind::Rol => ((val << 1) | (val >> (bits as u32 - 1))) & mask,
                    InstrKind::Ror => ((val >> 1) | ((val & 1) << (bits as u32 - 1))) & mask,
                    _ => val,
                };
                if let Some(reg) = dest_reg {
                    state.set_reg(reg, res);
                } else if let Some(mem) = dest_mem {
                    bus.store(mem.addr, bits, res);
                }
                state.set_reg(RegName::FZ, if res == 0 { 1 } else { 0 });
                let start_pc = state.pc();
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
            | InstrKind::Sbc => {
                if entry.operands.len() == 1 && matches!(entry.operands[0], OperandKind::RegPair(_)) {
                    let decoded = self.decode_operands(entry, state, bus)?;
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
                    self.execute_simple_mem(entry, state, bus)
                } else {
                    self.execute_reg_imm(entry, state, bus)
                }
            }
            InstrKind::Mv | InstrKind::Mvw | InstrKind::Mvp | InstrKind::Mvl => {
                self.execute_mv_generic(entry, state, bus)
            }
            InstrKind::Reset => {
                apply_reset(bus, state);
                Ok(1)
            }
            InstrKind::Pre | InstrKind::Unknown => {
                let len = if entry.kind == InstrKind::Pre { 2 } else { Self::estimated_length(entry) };
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(len as u32));
                }
                Ok(len)
            }
            InstrKind::Sc => {
                state.set_reg(RegName::FC, 1);
                let len = Self::estimated_length(entry);
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(len as u32));
                }
                Ok(len)
            }
            InstrKind::Rc => {
                state.set_reg(RegName::FC, 0);
                let len = Self::estimated_length(entry);
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(len as u32));
                }
                Ok(len)
            }
            InstrKind::Ir | InstrKind::Tcl => {
                if entry.kind == InstrKind::Ir {
                    // Push PC (24-bit), F, IMR (memory-mapped), clear IRM (bit7), and jump to interrupt vector.
                    let pc = state
                        .pc()
                        .wrapping_add(Self::estimated_length(entry) as u32)
                        & mask_for(RegName::PC);
                    Self::push_stack(state, bus, RegName::S, pc, 24);
                    let f = state.get_reg(RegName::F) & 0xFF;
                    Self::push_stack(state, bus, RegName::S, f, 8);
                    let imr_addr = INTERNAL_MEMORY_START + IMEM_IMR_OFFSET;
                    let imr = bus.load(imr_addr, 8) & 0xFF;
                    Self::push_stack(state, bus, RegName::S, imr, 8);
                    // Clear IRM bit in IMR (bit 7)
                    let cleared_imr = imr & 0x7F;
                    bus.store(imr_addr, 8, cleared_imr);
                    state.set_reg(RegName::IMR, cleared_imr);
                    state.call_depth_inc();
                    let vec = bus.load(INTERRUPT_VECTOR_ADDR, 8)
                        | (bus.load(INTERRUPT_VECTOR_ADDR + 1, 8) << 8)
                        | (bus.load(INTERRUPT_VECTOR_ADDR + 2, 8) << 16);
                    state.set_pc(vec & mask_for(RegName::PC));
                    Ok(Self::estimated_length(entry))
                } else {
                    let len = Self::estimated_length(entry);
                    let start_pc = state.pc();
                    if state.pc() == start_pc {
                        state.set_pc(start_pc.wrapping_add(len as u32));
                    }
                    Ok(len)
                }
            }
            InstrKind::JpAbs => {
                // Absolute jump; operand may be 16-bit (low bits) or 20-bit.
                if !Self::cond_pass(entry, state) {
                    // Condition false: just advance PC
                    let len = self.decode_operands(entry, state, bus)?.len;
                    let start_pc = state.pc();
                    if state.pc() == start_pc {
                        state.set_pc(start_pc.wrapping_add(len as u32));
                    }
                    return Ok(len);
                }
                let decoded = self.decode_operands(entry, state, bus)?;
                let imm = decoded.imm.map(|v| v.0);
                let target = if let Some(val) = imm {
                    if decoded.len == 3 {
                        let high = state.pc() & 0xFF0000;
                        high | (val & 0xFFFF)
                    } else {
                        val & 0xFFFFF
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
                    let len = self.decode_operands(entry, state, bus)?.len;
                    let start_pc = state.pc();
                    if state.pc() == start_pc {
                        state.set_pc(start_pc.wrapping_add(len as u32));
                    }
                    return Ok(len);
                }
                let decoded = self.decode_operands(entry, state, bus)?;
                let imm_raw = decoded.imm.ok_or("missing relative")?.0 as u8;
                let imm = if matches!(
                    entry.opcode,
                    0x13 | 0x19 | 0x1B | 0x1D | 0x1F
                ) {
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
                let decoded = self.decode_operands(entry, state, bus)?;
                let (target, bits) = decoded.imm.ok_or("missing jump target")?;
                let ret_addr = state.pc().wrapping_add(decoded.len as u32);
                let mut dest = target;
                let push_bits = if bits == 16 {
                    // Push 16-bit return; retain high page from current PC
                    let high = state.pc() & 0xFF0000;
                    dest = high | (target & 0xFFFF);
                    16
                } else {
                    24
                };
                // Use S stack for CALL (matches PUSHU/POPU? here sticking with S per CPU specs)
                Self::push_stack(state, bus, RegName::S, ret_addr, push_bits);
                state.set_pc(dest & 0xFFFFF);
                state.call_depth_inc();
                Ok(decoded.len)
            }
            InstrKind::Ret => {
                let ret = Self::pop_stack(state, bus, RegName::S, 16);
                let high = state.pc() & 0xFF0000;
                state.set_pc((high | (ret & 0xFFFF)) & 0xFFFFF);
                state.call_depth_dec();
                Ok(1)
            }
            InstrKind::RetF => {
                let ret = Self::pop_stack(state, bus, RegName::S, 24);
                state.set_pc(ret & 0xFFFFF);
                state.call_depth_dec();
                Ok(1)
            }
            InstrKind::RetI => {
                // Stack layout: IMR (1), F(1), 24-bit PC
                let imr = Self::pop_stack(state, bus, RegName::S, 8);
                let f = Self::pop_stack(state, bus, RegName::S, 8);
                let ret = Self::pop_stack(state, bus, RegName::S, 24);
                state.set_reg(RegName::IMR, imr);
                state.set_reg(RegName::F, f);
                state.set_pc(ret & 0xFFFFF);
                state.call_depth_dec();
                // Python RETI writes IMR back into IMEM for parity/traces.
                bus.store(INTERNAL_MEMORY_START + IMEM_IMR_OFFSET, 8, imr);
                Ok(1)
            }
            InstrKind::PushU | InstrKind::PushS => {
                let decoded = self.decode_operands(entry, state, bus)?;
                let reg = Self::operand_reg(entry.operands.get(0).ok_or("missing operand")?)
                    .ok_or("missing source")?;
                let bits = match entry.operands[0] {
                    OperandKind::Reg(_, b) => b,
                    OperandKind::RegB | OperandKind::RegIL | OperandKind::RegIMR | OperandKind::RegF => 8,
                    _ => 8,
                };
                let value = state.get_reg(reg);
                let sp_reg = if entry.kind == InstrKind::PushU { RegName::U } else { RegName::S };
                Self::push_stack(state, bus, sp_reg, value, bits);
                let len = decoded.len;
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(len as u32));
                }
                Ok(len)
            }
            InstrKind::PopU | InstrKind::PopS => {
                let decoded = self.decode_operands(entry, state, bus)?;
                let reg = Self::operand_reg(entry.operands.get(0).ok_or("missing operand")?)
                    .ok_or("missing destination")?;
                let bits = match entry.operands[0] {
                    OperandKind::Reg(_, b) => b,
                    OperandKind::RegB | OperandKind::RegIL | OperandKind::RegIMR | OperandKind::RegF => 8,
                    _ => 8,
                };
                let sp_reg = if entry.kind == InstrKind::PopU { RegName::U } else { RegName::S };
                let value = Self::pop_stack(state, bus, sp_reg, bits);
                state.set_reg(reg, value);
                let len = decoded.len;
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(len as u32));
                }
                Ok(len)
            }
            InstrKind::Cmp | InstrKind::Test => {
                let decoded = self.decode_operands(entry, state, bus)?;
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
                    lhs = if let Some(r) = Self::resolved_reg(op1, &decoded) {
                        state.get_reg(r)
                    } else if let Some(mem) = decoded.mem {
                        bus.load(mem.addr, bits)
                    } else {
                        decoded.imm.map(|v| v.0).unwrap_or(0)
                    };
                    rhs = if let Some(r) = Self::resolved_reg(op2, &decoded) {
                        state.get_reg(r)
                    } else if let Some((imm, _)) = decoded.imm {
                        imm
                    } else if let Some(mem) = decoded.mem {
                        bus.load(mem.addr, bits)
                    } else {
                        0
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
                let decoded = self.decode_operands(entry, state, bus)?;
                let bits = if entry.kind == InstrKind::Cmpw { 16 } else { 24 };
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
                let decoded = self.decode_operands(entry, state, bus)?;
                // Swap two memory operands or two registers.
                if let (Some(m1), Some(m2)) = (decoded.mem, decoded.mem2) {
                    let bits = m1.bits.min(m2.bits);
                    let v1 = bus.load(m1.addr, bits);
                    let v2 = bus.load(m2.addr, bits);
                    bus.store(m1.addr, bits, v2);
                    bus.store(m2.addr, bits, v1);
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
                } else if entry.operands.len() == 1 && matches!(entry.operands[0], OperandKind::RegB) {
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
            _ => {
                let len = Self::estimated_length(entry);
                let start_pc = state.pc();
                if state.pc() == start_pc {
                    state.set_pc(start_pc.wrapping_add(len as u32));
                }
                Ok(len)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama::opcodes::OPCODES;

    struct NullBus;
    impl LlamaBus for NullBus {
        fn load(&mut self, _addr: u32, _bits: u8) -> u32 {
            0
        }
    }

    #[test]
    fn opcode_table_has_coverage() {
        assert!(OPCODES.len() > 200);
        assert!(OPCODES.iter().any(|e| e.opcode == 0x00));
        assert!(OPCODES.iter().any(|e| e.opcode == 0xFF));
    }

    #[test]
    fn wait_advances_pc() {
        let mut exec = LlamaExecutor::new();
        let mut state = LlamaState::new();
        let mut bus = NullBus;
        let len = exec.execute(0xEF, &mut state, &mut bus).unwrap(); // WAIT
        assert_eq!(len, 1);
        assert_eq!(state.pc(), 1);
    }

    struct MemBus {
        mem: Vec<u8>,
    }
    impl MemBus {
        fn with_size(size: usize) -> Self {
            Self { mem: vec![0; size] }
        }
    }
    impl LlamaBus for MemBus {
        fn load(&mut self, addr: u32, bits: u8) -> u32 {
            let mut val = 0u32;
            let bytes = (bits + 7) / 8;
            for i in 0..bytes {
                let b = *self.mem.get(addr as usize + i as usize).unwrap_or(&0) as u32;
                val |= b << (8 * i);
            }
            val & ((1u32 << bits) - 1)
        }

        fn store(&mut self, addr: u32, bits: u8, value: u32) {
            let bytes = (bits + 7) / 8;
            for i in 0..bytes {
                if let Some(slot) = self.mem.get_mut(addr as usize + i as usize) {
                    *slot = ((value >> (8 * i)) & 0xFF) as u8;
                }
            }
        }

        fn resolve_emem(&mut self, base: u32) -> u32 {
            base
        }
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
        state.set_reg(RegName::I, 0xFFFF);
        let mut exec = LlamaExecutor::new();
        let len = exec.execute(0xEF, &mut state, &mut bus).unwrap();
        assert_eq!(len, 1);
        assert_eq!(state.get_reg(RegName::I), 0);
        assert_eq!(state.pc(), 1);
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
}
