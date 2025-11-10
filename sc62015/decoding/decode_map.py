from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

from .bind import (
    Addr16Page,
    Addr24,
    DecodedInstr,
    Disp8,
    Imm16,
    Imm24,
    Imm8,
    IntAddrCalc,
    PreLatch,
    ExtRegPtr,
    ImemPtr,
    RegSel,
)
from .reader import StreamCtx

DecoderFunc = Callable[[int, StreamCtx], DecodedInstr]

_REG_TABLE: Tuple[Tuple[str, str, int], ...] = (
    ("A", "r1", 8),
    ("IL", "r1", 8),
    ("BA", "r2", 16),
    ("I", "r2", 16),
    ("X", "r3", 24),
    ("Y", "r3", 24),
    ("U", "r3", 24),
    ("S", "r3", 24),
)

_GROUP_BITS = {"r1": 8, "r2": 16, "r3": 24}


def _length(ctx: StreamCtx) -> int:
    return ctx.total_length()


def _dec_mv_a_n(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    imm = Imm8(ctx.read_u8())
    return DecodedInstr(
        opcode=opcode,
        mnemonic="MV A,n",
        length=_length(ctx),
        family="imm8",
        binds={
            "dst": RegSel("r1", "A"),
            "n": imm,
        },
    )


def _dec_jr_cond(opcode: int, ctx: StreamCtx, cond: str, direction: int) -> DecodedInstr:
    offset = Imm8(ctx.read_u8())
    return DecodedInstr(
        opcode=opcode,
        mnemonic=f"JR{cond} ±n",
        length=_length(ctx),
        family="rel8",
        binds={"disp": offset, "cond": cond, "dir": direction},
    )


def _dec_jp_mn(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    lo, hi = ctx.read_u16_mn()
    addr = Addr16Page(Imm16(lo, hi), ctx.page20())
    return DecodedInstr(
        opcode=opcode,
        mnemonic="JP mn",
        length=_length(ctx),
        family="jp_mn",
        binds={"addr16_page": addr},
    )


def _dec_mv_a_abs24(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    lo, mid, hi = ctx.read_u24_lmn()
    addr = Addr24(Imm24(lo, mid, hi))
    return DecodedInstr(
        opcode=opcode,
        mnemonic="MV A,[lmn]",
        length=_length(ctx),
        family="ext24",
        binds={"addr24": addr},
    )

def _dec_call(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    lo, hi = ctx.read_u16_mn()
    addr = Addr16Page(Imm16(lo, hi), ctx.page20())
    return DecodedInstr(
        opcode=opcode,
        mnemonic="CALL mn",
        length=_length(ctx),
        family="call_near",
        binds={"addr16_page": addr},
    )


def _dec_callf(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    lo, mid, hi = ctx.read_u24_lmn()
    addr = Addr24(Imm24(lo, mid, hi))
    return DecodedInstr(
        opcode=opcode,
        mnemonic="CALLF lmn",
        length=_length(ctx),
        family="call_far",
        binds={"addr24": addr},
    )


def _dec_ret(opcode: int, ctx: StreamCtx, mnemonic: str, family: str) -> DecodedInstr:
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family=family,
        binds={},
    )


PRE_LATCHES: Dict[int, PreLatch] = {
    0x32: PreLatch("(n)", "(n)"),
    0x22: PreLatch("(BP+n)", "(n)"),
    0x36: PreLatch("(PX+n)", "(n)"),
    0x26: PreLatch("(BP+PX)", "(n)"),
    0x30: PreLatch("(n)", "(BP+n)"),
    0x33: PreLatch("(n)", "(PY+n)"),
    0x31: PreLatch("(n)", "(BP+PY)"),
}


def _dec_pre(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    latch = PRE_LATCHES.get(opcode)
    if latch is None:
        raise ValueError(f"Unsupported PRE opcode {opcode:#x}")
    # No operand bytes for PRE; ctx.idx stays 0 so total length == base_len
    return DecodedInstr(
        opcode=opcode,
        mnemonic=f"PRE{opcode:02X}",
        length=_length(ctx),
        family="pre",
        pre_latch=latch,
        binds={},
    )


_ALU_IMM8_OPS: Dict[int, str] = {
    0x40: "ADD A,n",
    0x48: "SUB A,n",
    0x50: "ADC A,n",
    0x58: "SBC A,n",
    0x64: "TEST A,n",
    0x68: "XOR A,n",
    0x70: "AND A,n",
    0x78: "OR A,n",
}

_STACK_SYS: Dict[int, Tuple[str, RegSel]] = {
    0x4F: ("PUSHS F", RegSel("r1", "F")),
    0x5F: ("POPS F", RegSel("r1", "F")),
}

_PUSHU_REGS: Dict[int, Tuple[str, RegSel]] = {
    0x28: ("PUSHU A", RegSel("r1", "A")),
    0x29: ("PUSHU IL", RegSel("r1", "IL")),
    0x2A: ("PUSHU BA", RegSel("r2", "BA")),
    0x2B: ("PUSHU I", RegSel("r2", "I")),
    0x2C: ("PUSHU X", RegSel("r3", "X")),
    0x2D: ("PUSHU Y", RegSel("r3", "Y")),
    0x2E: ("PUSHU F", RegSel("r1", "F")),
}

_POPU_REGS: Dict[int, Tuple[str, RegSel]] = {
    0x38: ("POPU A", RegSel("r1", "A")),
    0x39: ("POPU IL", RegSel("r1", "IL")),
    0x3A: ("POPU BA", RegSel("r2", "BA")),
    0x3B: ("POPU I", RegSel("r2", "I")),
    0x3C: ("POPU X", RegSel("r3", "X")),
    0x3D: ("POPU Y", RegSel("r3", "Y")),
    0x3E: ("POPU F", RegSel("r1", "F")),
}

_EXT_REG_LOADS: Dict[int, Tuple[str, RegSel]] = {
    0x90: ("MV r,[r3]", RegSel("r1", "A")),
    0x91: ("MV r,[r3]", RegSel("r1", "IL")),
    0x92: ("MV r,[r3]", RegSel("r2", "BA")),
    0x93: ("MV r,[r3]", RegSel("r2", "I")),
    0x94: ("MV r,[r3]", RegSel("r3", "X")),
    0x95: ("MV r,[r3]", RegSel("r3", "Y")),
    0x96: ("MV r,[r3]", RegSel("r3", "U")),
}

_EXT_REG_STORES: Dict[int, Tuple[str, RegSel]] = {
    0xB0: ("MV [r3],r", RegSel("r1", "A")),
    0xB1: ("MV [r3],r", RegSel("r1", "IL")),
    0xB2: ("MV [r3],r", RegSel("r2", "BA")),
    0xB3: ("MV [r3],r", RegSel("r2", "I")),
    0xB4: ("MV [r3],r", RegSel("r3", "X")),
    0xB5: ("MV [r3],r", RegSel("r3", "Y")),
    0xB6: ("MV [r3],r", RegSel("r3", "U")),
}

_EXT_PTR_LOADS: Dict[int, Tuple[str, RegSel]] = {
    0x98: ("MV r,[(n)]", RegSel("r1", "A")),
    0x99: ("MV r,[(n)]", RegSel("r1", "IL")),
    0x9A: ("MV r,[(n)]", RegSel("r2", "BA")),
    0x9B: ("MV r,[(n)]", RegSel("r2", "I")),
    0x9C: ("MV r,[(n)]", RegSel("r3", "X")),
    0x9D: ("MV r,[(n)]", RegSel("r3", "Y")),
    0x9E: ("MV r,[(n)]", RegSel("r3", "U")),
}

_EXT_PTR_STORES: Dict[int, Tuple[str, RegSel]] = {
    0xB8: ("MV [(n)],r", RegSel("r1", "A")),
    0xB9: ("MV [(n)],r", RegSel("r1", "IL")),
    0xBA: ("MV [(n)],r", RegSel("r2", "BA")),
    0xBB: ("MV [(n)],r", RegSel("r2", "I")),
    0xBC: ("MV [(n)],r", RegSel("r3", "X")),
    0xBD: ("MV [(n)],r", RegSel("r3", "Y")),
    0xBE: ("MV [(n)],r", RegSel("r3", "U")),
}

_IMEM_MOVES: Dict[int, Tuple[str, int]] = {
    0xC8: ("MV (m),(n)", 8),
    0xC9: ("MVW (m),(n)", 16),
    0xCA: ("MVP (m),(n)", 24),
}

_IMEM_LOOP_MOVES: Dict[int, Tuple[str, int]] = {
    0xCB: ("MVL (m),(n)", +1),
    0xCF: ("MVLD (m),(n)", -1),
}

_IMEM_EXCHANGES: Dict[int, Tuple[str, int]] = {
    0xC0: ("EX (m),(n)", 8),
    0xC1: ("EXW (m),(n)", 16),
    0xC2: ("EXP (m),(n)", 24),
}

_IMEM_FROM_EXT: Dict[int, Tuple[str, int]] = {
    0xE0: ("MV (n),[r3]", 8),
    0xE1: ("MVW (n),[r3]", 16),
    0xE2: ("MVP (n),[r3]", 24),
}

_EXT_FROM_IMEM: Dict[int, Tuple[str, int]] = {
    0xE8: ("MV [r3],(n)", 8),
    0xE9: ("MVW [r3],(n)", 16),
    0xEA: ("MVP [r3],(n)", 24),
}


def _dec_alu_imm(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    imm = Imm8(ctx.read_u8())
    return DecodedInstr(
        opcode=opcode,
        mnemonic=_ALU_IMM8_OPS[opcode],
        length=_length(ctx),
        family="imm8",
        binds={"n": imm},
    )


def _dec_mv_ext_store(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    lo, mid, hi = ctx.read_u24_lmn()
    addr = Addr24(Imm24(lo, mid, hi))
    return DecodedInstr(
        opcode=opcode,
        mnemonic="MV [lmn],A",
        length=_length(ctx),
        family="ext24",
        binds={"addr24": addr},
    )


def _dec_stack_sys(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    mnemonic, reg = _STACK_SYS[opcode]
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family="stack_sys",
        binds={"reg": reg},
    )


def _dec_pushu(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    mnemonic, reg = _PUSHU_REGS[opcode]
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family="pushu",
        binds={"reg": reg},
    )


def _dec_popu(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    mnemonic, reg = _POPU_REGS[opcode]
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family="popu",
        binds={"reg": reg},
    )


def _dec_pushu_imr(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    return DecodedInstr(
        opcode=opcode,
        mnemonic="PUSHU IMR",
        length=_length(ctx),
        family="pushu",
        binds={},
    )


def _dec_popu_imr(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    return DecodedInstr(
        opcode=opcode,
        mnemonic="POPU IMR",
        length=_length(ctx),
        family="popu",
        binds={},
    )


def _group_bytes(size_group: str) -> int:
    bits = _GROUP_BITS[size_group]
    return bits // 8


def _dec_ext_reg_load(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    mnemonic, dst = _EXT_REG_LOADS[opcode]
    ptr = _decode_ext_reg_ptr(ctx, _group_bytes(dst.size_group))
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family="ext_reg",
        binds={"dst": dst, "ptr": ptr},
    )


def _dec_ext_reg_store(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    mnemonic, src = _EXT_REG_STORES[opcode]
    ptr = _decode_ext_reg_ptr(ctx, _group_bytes(src.size_group))
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family="ext_reg",
        binds={"src": src, "ptr": ptr},
    )


def _dec_ext_ptr_load(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    mnemonic, dst = _EXT_PTR_LOADS[opcode]
    ptr = _decode_emem_imem(ctx)
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family="ext_ptr",
        binds={"dst": dst, "ptr": ptr},
    )


def _dec_ext_ptr_store(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    mnemonic, src = _EXT_PTR_STORES[opcode]
    ptr = _decode_emem_imem(ctx)
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family="ext_ptr",
        binds={"src": src, "ptr": ptr},
    )


def _dec_imem_move(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    mnemonic, width = _IMEM_MOVES[opcode]
    dst = Imm8(ctx.read_u8())
    src = Imm8(ctx.read_u8())
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family="imem_move",
        binds={"dst": dst, "src": src, "width": width},
    )


def _dec_imem_swap(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    mnemonic, width = _IMEM_EXCHANGES[opcode]
    left = Imm8(ctx.read_u8())
    right = Imm8(ctx.read_u8())
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family="imem_swap",
        binds={"left": left, "right": right, "width": width},
    )


def _dec_imem_from_ext(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    mnemonic, width = _IMEM_FROM_EXT[opcode]
    ptr = _decode_ext_reg_ptr(ctx, width // 8)
    addr = Imm8(ctx.read_u8())
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family="imem_ext",
        binds={"imem": addr, "ptr": ptr, "width": width},
    )


def _dec_ext_from_imem(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    mnemonic, width = _EXT_FROM_IMEM[opcode]
    ptr = _decode_ext_reg_ptr(ctx, width // 8)
    addr = Imm8(ctx.read_u8())
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family="imem_ext",
        binds={"imem": addr, "ptr": ptr, "width": width},
    )


def _dec_imem_loop_move(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    mnemonic, step = _IMEM_LOOP_MOVES[opcode]
    dst = Imm8(ctx.read_u8())
    src = Imm8(ctx.read_u8())
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family="loop_move",
        binds={"dst": dst, "src": src, "step": step},
    )


def _reg_from_byte(byte: int) -> Tuple[str, str]:
    idx = byte & 0x07
    try:
        name, size_group, _ = _REG_TABLE[idx]
    except IndexError:
        raise ValueError(f"Unsupported register index {idx}")
    return name, size_group


_EMEM_MODE_MAP: Dict[int, Tuple[str, bool, int]] = {
    0x0: ("simple", False, 0),
    0x2: ("post_inc", False, 0),
    0x3: ("pre_dec", False, 0),
    0x8: ("offset", True, +1),
    0xC: ("offset", True, -1),
}

_EMEM_IMEM_MODE_MAP: Dict[int, Tuple[str, bool, int]] = {
    0x00: ("simple", False, 0),
    0x80: ("pos", True, +1),
    0xC0: ("neg", True, -1),
}


def _decode_ext_reg_ptr(ctx: StreamCtx, width_bytes: int) -> ExtRegPtr:
    reg_byte = ctx.read_u8()
    mode_code = (reg_byte >> 4) & 0x0F
    try:
        mode_name, needs_disp, disp_sign = _EMEM_MODE_MAP[mode_code]
    except KeyError as exc:
        raise ValueError(f"Unsupported EMemReg mode {mode_code:#x}") from exc

    ptr_name, ptr_group = _reg_from_byte(reg_byte)
    if ptr_group != "r3":
        raise ValueError(f"Pointer register must be r3, got {ptr_group}")

    disp: Optional[Disp8]
    if needs_disp:
        magnitude = ctx.read_u8()
        signed = magnitude if disp_sign > 0 else -magnitude
        disp = Disp8(signed)
    else:
        disp = None

    return ExtRegPtr(ptr=RegSel(ptr_group, ptr_name), mode=mode_name, disp=disp)


def _decode_emem_imem(ctx: StreamCtx) -> ImemPtr:
    mode_byte = ctx.read_u8()
    try:
        mode_name, needs_disp, disp_sign = _EMEM_IMEM_MODE_MAP[mode_byte]
    except KeyError as exc:
        raise ValueError(f"Unsupported EMemIMem mode {mode_byte:#x}") from exc
    base = Imm8(ctx.read_u8())
    disp: Optional[Disp8] = None
    if needs_disp:
        magnitude = ctx.read_u8()
        disp = Disp8(magnitude if disp_sign > 0 else -magnitude)
    return ImemPtr(base=base, mode=mode_name, disp=disp)


def _dec_inc_dec(opcode: int, ctx: StreamCtx, mnemonic: str) -> DecodedInstr:
    reg_byte = ctx.read_u8()
    name, size_group = _reg_from_byte(reg_byte)
    if size_group != "r1":
        raise ValueError(f"{mnemonic} unsupported for register {name}")
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family="incdec",
        binds={"reg": RegSel(size_group, name)},
    )


def _dec_mv_imem(opcode: int, ctx: StreamCtx, mnemonic: str) -> DecodedInstr:
    imm = Imm8(ctx.read_u8())
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family="imem",
        binds={"n": imm},
    )


def _dec_jp_cond(opcode: int, ctx: StreamCtx, cond: str) -> DecodedInstr:
    lo, hi = ctx.read_u16_mn()
    addr = Addr16Page(Imm16(lo, hi), ctx.page20())
    return DecodedInstr(
        opcode=opcode,
        mnemonic=f"JP{cond} mn",
        length=_length(ctx),
        family="jp_mn",
        binds={"addr16_page": addr, "cond": cond},
    )


def _dec_jp_imem(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    addr = Imm8(ctx.read_u8())
    return DecodedInstr(
        opcode=opcode,
        mnemonic="JP (n)",
        length=_length(ctx),
        family="jp_imem",
        binds={"n": addr},
    )


def _dec_jp_reg(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    reg_byte = ctx.read_u8()
    name, size_group = _reg_from_byte(reg_byte)
    if size_group != "r3":
        raise ValueError(f"JP r3 requires r3 register, got {name}")
    return DecodedInstr(
        opcode=opcode,
        mnemonic="JP r3",
        length=_length(ctx),
        family="jp_reg",
        binds={"reg": RegSel(size_group, name)},
    )


DECODERS: Dict[int, DecoderFunc] = {
    0x01: lambda opcode, ctx: _dec_ret(opcode, ctx, "RETI", "reti"),
    0x02: _dec_jp_mn,
    0x08: _dec_mv_a_n,
    0x04: _dec_call,
    0x05: _dec_callf,
    0x06: lambda opcode, ctx: _dec_ret(opcode, ctx, "RET", "ret_near"),
    0x07: lambda opcode, ctx: _dec_ret(opcode, ctx, "RETF", "ret_far"),
    0x10: _dec_jp_imem,
    0x11: _dec_jp_reg,
    0x14: lambda opcode, ctx: _dec_jp_cond(opcode, ctx, "Z"),
    0x15: lambda opcode, ctx: _dec_jp_cond(opcode, ctx, "NZ"),
    0x16: lambda opcode, ctx: _dec_jp_cond(opcode, ctx, "C"),
    0x17: lambda opcode, ctx: _dec_jp_cond(opcode, ctx, "NC"),
    0x18: lambda opcode, ctx: _dec_jr_cond(opcode, ctx, "Z", +1),
    0x19: lambda opcode, ctx: _dec_jr_cond(opcode, ctx, "Z", -1),
    0x1A: lambda opcode, ctx: _dec_jr_cond(opcode, ctx, "NZ", +1),
    0x1B: lambda opcode, ctx: _dec_jr_cond(opcode, ctx, "NZ", -1),
    0x1C: lambda opcode, ctx: _dec_jr_cond(opcode, ctx, "C", +1),
    0x1D: lambda opcode, ctx: _dec_jr_cond(opcode, ctx, "C", -1),
    0x1E: lambda opcode, ctx: _dec_jr_cond(opcode, ctx, "NC", +1),
    0x1F: lambda opcode, ctx: _dec_jr_cond(opcode, ctx, "NC", -1),
    0x22: _dec_pre,
    0x26: _dec_pre,
    0x30: _dec_pre,
    0x31: _dec_pre,
    0x32: _dec_pre,
    0x33: _dec_pre,
    0x36: _dec_pre,
    0x28: _dec_pushu,
    0x29: _dec_pushu,
    0x2A: _dec_pushu,
    0x2B: _dec_pushu,
    0x2C: _dec_pushu,
    0x2D: _dec_pushu,
    0x2E: _dec_pushu,
    0x2F: _dec_pushu_imr,
    0x38: _dec_popu,
    0x39: _dec_popu,
    0x3A: _dec_popu,
    0x3B: _dec_popu,
    0x3C: _dec_popu,
    0x3D: _dec_popu,
    0x3E: _dec_popu,
    0x3F: _dec_popu_imr,
    0x90: _dec_ext_reg_load,
    0x91: _dec_ext_reg_load,
    0x92: _dec_ext_reg_load,
    0x93: _dec_ext_reg_load,
    0x94: _dec_ext_reg_load,
    0x95: _dec_ext_reg_load,
    0x96: _dec_ext_reg_load,
    0x98: _dec_ext_ptr_load,
    0x99: _dec_ext_ptr_load,
    0x9A: _dec_ext_ptr_load,
    0x9B: _dec_ext_ptr_load,
    0x9C: _dec_ext_ptr_load,
    0x9D: _dec_ext_ptr_load,
    0x9E: _dec_ext_ptr_load,
    0x40: _dec_alu_imm,
    0x48: _dec_alu_imm,
    0x4F: _dec_stack_sys,
    0x50: _dec_alu_imm,
    0x58: _dec_alu_imm,
    0x5F: _dec_stack_sys,
    0x64: _dec_alu_imm,
    0x68: _dec_alu_imm,
    0x70: _dec_alu_imm,
    0x78: _dec_alu_imm,
    0x6C: lambda opcode, ctx: _dec_inc_dec(opcode, ctx, "INC r"),
    0x7C: lambda opcode, ctx: _dec_inc_dec(opcode, ctx, "DEC r"),
    0x80: lambda opcode, ctx: _dec_mv_imem(opcode, ctx, "MV A,(n)"),
    0x88: _dec_mv_a_abs24,
    0xA0: lambda opcode, ctx: _dec_mv_imem(opcode, ctx, "MV (n),A"),
    0xA8: _dec_mv_ext_store,
    0xB0: _dec_ext_reg_store,
    0xB1: _dec_ext_reg_store,
    0xB2: _dec_ext_reg_store,
    0xB3: _dec_ext_reg_store,
    0xB4: _dec_ext_reg_store,
    0xB5: _dec_ext_reg_store,
    0xB6: _dec_ext_reg_store,
    0xB8: _dec_ext_ptr_store,
    0xB9: _dec_ext_ptr_store,
    0xBA: _dec_ext_ptr_store,
    0xBB: _dec_ext_ptr_store,
    0xBC: _dec_ext_ptr_store,
    0xBD: _dec_ext_ptr_store,
    0xBE: _dec_ext_ptr_store,
    0xC0: _dec_imem_swap,
    0xC1: _dec_imem_swap,
    0xC2: _dec_imem_swap,
    0xC8: _dec_imem_move,
    0xCB: _dec_imem_loop_move,
    0xC9: _dec_imem_move,
    0xCA: _dec_imem_move,
    0xCF: _dec_imem_loop_move,
    0xE0: _dec_imem_from_ext,
    0xE1: _dec_imem_from_ext,
    0xE2: _dec_imem_from_ext,
    0xE8: _dec_ext_from_imem,
    0xE9: _dec_ext_from_imem,
    0xEA: _dec_ext_from_imem,
}


def decode_opcode(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    try:
        decoder = DECODERS[opcode]
    except KeyError as exc:
        raise KeyError(f"No decoder registered for opcode {opcode:#x}") from exc
    return decoder(opcode, ctx)
