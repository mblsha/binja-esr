from __future__ import annotations

from typing import Callable, Dict, Literal, Optional, Tuple

from .bind import (
    Addr16Page,
    Addr24,
    DecodedInstr,
    Disp8,
    Imm16,
    Imm20,
    Imm24,
    Imm8,
    ExtRegPtr,
    ImemPtr,
    RegSel,
)
from .pre_modes import iter_all_pre_variants, prelatch_for_opcode
from .reader import StreamCtx


def _record(
    ctx: StreamCtx, key: str, kind: str, *, start: Optional[int] = None, **meta
) -> None:
    if start is not None:
        meta = dict(meta)
        meta.setdefault("offset", start)
        meta.setdefault("length_bytes", ctx.bytes_consumed() - start)
    ctx.record_operand(key, kind, **meta)


def _read_imm8(ctx: StreamCtx, key: str) -> Imm8:
    start = ctx.bytes_consumed()
    value = ctx.read_u8()
    _record(ctx, key, "imm8", start=start, width=8)
    return Imm8(value)


def _read_disp8(ctx: StreamCtx, key: str) -> Disp8:
    start = ctx.bytes_consumed()
    raw = ctx.read_u8()
    signed = raw - 0x100 if raw & 0x80 else raw
    _record(ctx, key, "disp8", start=start, width=8)
    return Disp8(signed)


def _read_addr16(ctx: StreamCtx, key: str, *, kind: str = "imm16") -> Imm16:
    start = ctx.bytes_consumed()
    lo = ctx.read_u8()
    hi = ctx.read_u8()
    _record(ctx, key, kind, start=start, order="mn")
    return Imm16(lo, hi)


def _read_addr24(ctx: StreamCtx, key: str, *, kind: str = "imm24") -> Imm24:
    start = ctx.bytes_consumed()
    lo = ctx.read_u8()
    mid = ctx.read_u8()
    hi = ctx.read_u8()
    _record(ctx, key, kind, start=start, order="lmn")
    return Imm24(lo, mid, hi)


def _read_imm20(ctx: StreamCtx, key: str) -> Imm20:
    start = ctx.bytes_consumed()
    lo = ctx.read_u8()
    mid = ctx.read_u8()
    hi = ctx.read_u8() & 0x0F
    _record(ctx, key, "imm20", start=start, width=20)
    return Imm20(lo, mid, hi)


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


def _dec_nop(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    return DecodedInstr(
        opcode=opcode,
        mnemonic="NOP",
        length=_length(ctx),
        family="nop",
        binds={},
    )


def _dec_mv_a_n(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    imm = _read_imm8(ctx, "n")
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


def _dec_mv_reg_n(
    opcode: int,
    ctx: StreamCtx,
    mnemonic: str,
    dst: RegSel,
    imm_reader: Callable[[StreamCtx, str], object],
) -> DecodedInstr:
    imm = imm_reader(ctx, "n")
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family="imm",
        binds={"dst": dst, "n": imm},
    )


def _dec_jr_cond(
    opcode: int, ctx: StreamCtx, cond: str, direction: int
) -> DecodedInstr:
    offset = _read_imm8(ctx, "disp")
    return DecodedInstr(
        opcode=opcode,
        mnemonic=f"JR{cond} ±n",
        length=_length(ctx),
        family="rel8",
        binds={"disp": offset, "cond": cond, "dir": direction},
    )


def _dec_jr_rel(opcode: int, ctx: StreamCtx, direction: int, mnemonic: str) -> DecodedInstr:
    offset = _read_imm8(ctx, "disp")
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family="rel8",
        binds={"disp": offset, "dir": direction},
    )


def _dec_jp_mn(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    imm = _read_addr16(ctx, "addr16_page", kind="addr16_page")
    addr = Addr16Page(imm, ctx.page20())
    return DecodedInstr(
        opcode=opcode,
        mnemonic="JP mn",
        length=_length(ctx),
        family="jp_mn",
        binds={"addr16_page": addr},
    )


def _dec_jpf(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    addr = _read_imm20(ctx, "addr20")
    return DecodedInstr(
        opcode=opcode,
        mnemonic="JPF lmn",
        length=_length(ctx),
        family="jp_far",
        binds={"addr20": addr},
    )


def _dec_mv_a_abs24(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    imm = _read_addr24(ctx, "addr24", kind="addr24")
    addr = Addr24(imm)
    return DecodedInstr(
        opcode=opcode,
        mnemonic="MV A,[lmn]",
        length=_length(ctx),
        family="ext24",
        binds={"addr24": addr},
    )


def _dec_call(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    addr = Addr16Page(
        _read_addr16(ctx, "addr16_page", kind="addr16_page"), ctx.page20()
    )
    return DecodedInstr(
        opcode=opcode,
        mnemonic="CALL mn",
        length=_length(ctx),
        family="call_near",
        binds={"addr16_page": addr},
    )


def _dec_callf(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    addr = Addr24(_read_addr24(ctx, "addr24", kind="addr24"))
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


def _dec_pre(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    latch = prelatch_for_opcode(opcode)
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
    0xC3: ("EXL (m),(n)", 8),
}

_IMEM_FROM_EXT: Dict[int, Tuple[str, int]] = {
    0xE0: ("MV (n),[r3]", 8),
    0xE1: ("MVW (n),[r3]", 16),
    0xE2: ("MVP (n),[r3]", 24),
    0xE3: ("MVL (n),[r3]", 8),
}

_EXT_FROM_IMEM: Dict[int, Tuple[str, int]] = {
    0xE8: ("MV [r3],(n)", 8),
    0xE9: ("MVW [r3],(n)", 16),
    0xEA: ("MVP [r3],(n)", 24),
    0xEB: ("MVL [r3],(n)", 8),
}

_MV_EXT_REGS: Dict[int, Tuple[str, RegSel]] = {
    0x88: ("MV A,[lmn]", RegSel("r1", "A")),
    0x89: ("MV IL,[lmn]", RegSel("r1", "IL")),
    0x8A: ("MV BA,[lmn]", RegSel("r2", "BA")),
    0x8B: ("MV I,[lmn]", RegSel("r2", "I")),
    0x8C: ("MV X,[lmn]", RegSel("r3", "X")),
    0x8D: ("MV Y,[lmn]", RegSel("r3", "Y")),
    0x8E: ("MV U,[lmn]", RegSel("r3", "U")),
    0x8F: ("MV S,[lmn]", RegSel("r3", "S")),
}

_MV_EXT_STORE_REGS: Dict[int, Tuple[str, RegSel]] = {
    0xA8: ("MV [lmn],A", RegSel("r1", "A")),
    0xA9: ("MV [lmn],IL", RegSel("r1", "IL")),
    0xAA: ("MV [lmn],BA", RegSel("r2", "BA")),
    0xAB: ("MV [lmn],I", RegSel("r2", "I")),
    0xAC: ("MV [lmn],X", RegSel("r3", "X")),
    0xAD: ("MV [lmn],Y", RegSel("r3", "Y")),
    0xAE: ("MV [lmn],U", RegSel("r3", "U")),
    0xAF: ("MV [lmn],S", RegSel("r3", "S")),
}

_MV_IMEM_REGS: Dict[int, Tuple[str, RegSel]] = {
    0x80: ("MV A,(n)", RegSel("r1", "A")),
    0x81: ("MV IL,(n)", RegSel("r1", "IL")),
    0x82: ("MV BA,(n)", RegSel("r2", "BA")),
    0x83: ("MV I,(n)", RegSel("r2", "I")),
    0x84: ("MV X,(n)", RegSel("r3", "X")),
    0x85: ("MV Y,(n)", RegSel("r3", "Y")),
    0x86: ("MV U,(n)", RegSel("r3", "U")),
    0x87: ("MV S,(n)", RegSel("r3", "S")),
}

_MV_REG_IMEM: Dict[int, RegSel] = {
    0xA0: RegSel("r1", "A"),
    0xA1: RegSel("r1", "IL"),
    0xA2: RegSel("r2", "BA"),
    0xA3: RegSel("r2", "I"),
    0xA4: RegSel("r3", "X"),
    0xA5: RegSel("r3", "Y"),
    0xA6: RegSel("r3", "U"),
    0xA7: RegSel("r3", "S"),
}


def _dec_alu_imm(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    imm = _read_imm8(ctx, "n")
    return DecodedInstr(
        opcode=opcode,
        mnemonic=_ALU_IMM8_OPS[opcode],
        length=_length(ctx),
        family="imm8",
        binds={"n": imm},
    )


def _dec_mv_ext_store(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    entry = _MV_EXT_STORE_REGS.get(opcode)
    if entry is None:
        raise KeyError(f"MV ext store decoder missing for opcode {opcode:#x}")
    mnemonic, reg = entry
    addr = Addr24(_read_addr24(ctx, "addr24"))
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family="ext24",
        binds={"addr24": addr, "dst_reg": reg},
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


def _decode_emem_imem_offset(ctx: StreamCtx) -> tuple[str, Imm8, Imm8, Optional[Disp8]]:
    start = ctx.bytes_consumed()
    mode_byte = ctx.read_u8()
    try:
        mode_name, needs_offset = _EMEM_IMEM_OFFSET_MODES[mode_byte]
    except KeyError as exc:
        raise ValueError(f"Unsupported EMemIMem offset mode {mode_byte:#x}") from exc
    first = _read_imm8(ctx, "first")
    second = _read_imm8(ctx, "second")
    offset: Optional[Disp8] = None
    if needs_offset:
        magnitude = ctx.read_u8()
        signed = magnitude if mode_name == "pos" else -magnitude
        offset = Disp8(signed)
    _record(ctx, "ptr", "imem_ptr", start=start)
    return mode_name, first, second, offset


def _dec_emem_imem_offset(
    opcode: int,
    ctx: StreamCtx,
    mnemonic: str,
    width_bits: int,
    order: Literal["int", "ext"],
) -> DecodedInstr:
    mode, first, second, offset = _decode_emem_imem_offset(ctx)
    if order == "int":
        dst = first
        ptr = second
    else:
        dst = second
        ptr = first
    binds: Dict[str, object] = {
        "dst": dst,
        "ptr": ptr,
        "width_bits": width_bits,
        "mode": mode,
    }
    if offset is not None:
        binds["offset"] = offset
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family="emem_imem_offset",
        binds=binds,
    )


def _dec_imem_move(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    mnemonic, width = _IMEM_MOVES[opcode]
    dst = _read_imm8(ctx, "dst")
    src = _read_imm8(ctx, "src")
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family="imem_move",
        binds={"dst": dst, "src": src, "width": width},
    )


def _dec_imem_swap(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    mnemonic, width = _IMEM_EXCHANGES[opcode]
    left = _read_imm8(ctx, "left")
    right = _read_imm8(ctx, "right")
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
    addr = _read_imm8(ctx, "imem")
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family="imem_ext",
        binds={"imem": addr, "ptr": ptr, "width": width},
    )


def _dec_tcl(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    return DecodedInstr(
        opcode=opcode,
        mnemonic="TCL",
        length=_length(ctx),
        family="system",
        binds={},
    )


def _dec_swap_a(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    return DecodedInstr(
        opcode=opcode,
        mnemonic="SWAP A",
        length=_length(ctx),
        family="system",
        binds={},
    )


def _dec_ext_from_imem(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    mnemonic, width = _EXT_FROM_IMEM[opcode]
    ptr = _decode_ext_reg_ptr(ctx, width // 8)
    addr = _read_imm8(ctx, "imem")
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family="imem_ext",
        binds={"imem": addr, "ptr": ptr, "width": width},
    )


def _dec_imem_loop_move(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    mnemonic, step = _IMEM_LOOP_MOVES[opcode]
    dst = _read_imm8(ctx, "dst")
    src = _read_imm8(ctx, "src")
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family="loop_move",
        binds={"dst": dst, "src": src, "step": step},
    )


def _dec_loop_arith_mem(opcode: int, ctx: StreamCtx, mnemonic: str) -> DecodedInstr:
    dst = _read_imm8(ctx, "dst")
    src = _read_imm8(ctx, "src")
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family="loop_arith",
        binds={"dst": dst, "src": src},
    )


def _dec_loop_arith_reg(opcode: int, ctx: StreamCtx, mnemonic: str) -> DecodedInstr:
    dst = _read_imm8(ctx, "dst")
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family="loop_arith",
        binds={"dst": dst},
    )


def _dec_loop_bcd_mem(opcode: int, ctx: StreamCtx, mnemonic: str) -> DecodedInstr:
    dst = _read_imm8(ctx, "dst")
    src = _read_imm8(ctx, "src")
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family="loop_bcd",
        binds={"dst": dst, "src": src},
    )


def _dec_loop_bcd_reg(opcode: int, ctx: StreamCtx, mnemonic: str) -> DecodedInstr:
    dst = _read_imm8(ctx, "dst")
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family="loop_bcd",
        binds={"dst": dst},
    )


def _dec_decimal_shift(opcode: int, ctx: StreamCtx, mnemonic: str) -> DecodedInstr:
    addr = _read_imm8(ctx, "dst")
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family="decimal_shift",
        binds={"dst": addr},
    )


def _dec_simple(
    opcode: int, ctx: StreamCtx, mnemonic: str, family: str
) -> DecodedInstr:
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family=family,
        binds={},
    )


def _dec_rc(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    return DecodedInstr(
        opcode=opcode,
        mnemonic="RC",
        length=_length(ctx),
        family="rc",
        binds={},
    )


def _dec_sc(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    return DecodedInstr(
        opcode=opcode,
        mnemonic="SC",
        length=_length(ctx),
        family="sc",
        binds={},
    )


def _dec_ex_ab(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    return DecodedInstr(
        opcode=opcode,
        mnemonic="EX A,B",
        length=_length(ctx),
        family="ex_ab",
        binds={},
    )


def _dec_pmdf_imm(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    addr = _read_imm8(ctx, "dst")
    imm = _read_imm8(ctx, "imm")
    return DecodedInstr(
        opcode=opcode,
        mnemonic="PMDF (m),n",
        length=_length(ctx),
        family="pmdf",
        binds={"dst": addr, "imm": imm},
    )


def _dec_pmdf_reg(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    addr = _read_imm8(ctx, "dst")
    return DecodedInstr(
        opcode=opcode,
        mnemonic="PMDF (m),A",
        length=_length(ctx),
        family="pmdf",
        binds={"dst": addr},
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
    0x1: ("simple", False, 0),
    0x2: ("post_inc", False, 0),
    0x3: ("pre_dec", False, 0),
    0x8: ("offset", True, +1),
    0x9: ("offset", True, +1),
    0xA: ("offset", True, +1),
    0xB: ("offset", True, +1),
    0xC: ("offset", True, -1),
    0xD: ("offset", True, -1),
    0xE: ("offset", True, -1),
    0xF: ("offset", True, -1),
}


def _normalize_ext_reg_mode(code: int) -> int:
    if code & 0x8:
        return 0xC if code & 0x4 else 0x8
    if code & 0x3 == 0x3:
        return 0x3
    if code & 0x2:
        return 0x2
    return 0x0

def _decode_emem_imem_mode(mode_byte: int) -> Tuple[str, bool, int]:
    base = mode_byte & 0xC0
    if base == 0x00:
        return "simple", False, 0
    if base == 0x80:
        return "pos", True, +1
    if base == 0xC0:
        return "neg", True, -1
    raise ValueError(f"Unexpected EMemIMem mode {mode_byte:#x}")


_EMEM_IMEM_OFFSET_MODES: Dict[int, Tuple[str, bool]] = {
    0x00: ("simple", False),
    0x80: ("pos", True),
    0xC0: ("neg", True),
}


def _decode_ext_reg_ptr(ctx: StreamCtx, width_bytes: int) -> ExtRegPtr:
    start = ctx.bytes_consumed()
    reg_byte = ctx.read_u8()
    raw_mode = (reg_byte >> 4) & 0x0F
    mode_code = _normalize_ext_reg_mode(raw_mode)
    try:
        mode_name, needs_disp, disp_sign = _EMEM_MODE_MAP[mode_code]
    except KeyError as exc:
        raise ValueError(f"Unsupported EMemReg mode {mode_code:#x}") from exc

    ptr_name, ptr_group = _reg_from_byte(reg_byte)

    disp: Optional[Disp8]
    if needs_disp:
        magnitude = ctx.read_u8()
        signed = magnitude if disp_sign > 0 else -magnitude
        disp = Disp8(signed)
    else:
        disp = None

    ptr = ExtRegPtr(ptr=RegSel(ptr_group, ptr_name), mode=mode_name, disp=disp)
    _record(
        ctx,
        "ptr",
        "ext_reg_ptr",
        start=start,
        width_bytes=width_bytes,
    )
    return ptr


def _decode_emem_imem(ctx: StreamCtx) -> ImemPtr:
    start = ctx.bytes_consumed()
    mode_byte = ctx.read_u8()
    mode_name, needs_disp, disp_sign = _decode_emem_imem_mode(mode_byte)
    base = Imm8(ctx.read_u8())
    disp: Optional[Disp8] = None
    if needs_disp:
        raw = ctx.read_u8()
        value = raw - 0x100 if raw & 0x80 else raw
        signed = value if disp_sign > 0 else -value
        disp = Disp8(signed)
    _record(ctx, "ptr", "imem_ptr", start=start)
    return ImemPtr(base=base, mode=mode_name, disp=disp)


def _dec_inc_dec(opcode: int, ctx: StreamCtx, mnemonic: str) -> DecodedInstr:
    start = ctx.bytes_consumed()
    reg_byte = ctx.read_u8()
    idx = reg_byte & 0x0F
    reg_sel = _reg_from_index(idx)
    allowed = ("r1", "r2", "r3")
    if reg_sel.size_group not in allowed:
        raise ValueError(f"{mnemonic} unsupported for register {reg_sel.name}")
    _record(ctx, "reg", "regsel", start=start, allowed_groups=allowed)
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family="incdec",
        binds={"reg": reg_sel},
    )


def _dec_mv_imem(opcode: int, ctx: StreamCtx, mnemonic: str) -> DecodedInstr:
    imm = _read_imm8(ctx, "n")
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family="imem",
        binds={"n": imm},
    )


def _dec_mv_imem_const(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    dst = _read_imm8(ctx, "dst")
    imm = _read_imm8(ctx, "imm")
    return DecodedInstr(
        opcode=opcode,
        mnemonic="MV (n),n",
        length=_length(ctx),
        family="imem_const",
        binds={"dst": dst, "imm": imm},
    )


def _dec_mv_imem_from_ext(
    opcode: int, ctx: StreamCtx, mnemonic: str, width_bits: int
) -> DecodedInstr:
    dst = _read_imm8(ctx, "dst")
    addr = Addr24(_read_addr24(ctx, "addr24"))
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family="mv_imem_from_ext",
        binds={"dst": dst, "addr24": addr, "width_bits": width_bits},
    )


def _dec_mv_imem_from_ext_const(
    opcode: int, ctx: StreamCtx, mnemonic: str, width_bits: int
) -> DecodedInstr:
    dst = _read_imm8(ctx, "dst")
    lo = _read_imm8(ctx, "lo")
    mid = _read_imm8(ctx, "mid")
    hi = _read_imm8(ctx, "hi")
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family="mv_imem_long_const",
        binds={"dst": dst, "lo": lo, "mid": mid, "hi": hi, "width_bits": width_bits},
    )


def _dec_unknown_bf(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    return DecodedInstr(
        opcode=opcode,
        mnemonic="BF",
        length=_length(ctx),
        family="unknown",
        binds={},
    )


def _dec_mv_ext_from_imem(
    opcode: int, ctx: StreamCtx, mnemonic: str, width_bits: int
) -> DecodedInstr:
    addr = Addr24(_read_addr24(ctx, "addr24"))
    src = _read_imm8(ctx, "src")
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family="mv_ext_from_imem",
        binds={"addr24": addr, "src": src, "width_bits": width_bits},
    )


def _dec_mv_a_a(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    return DecodedInstr(
        opcode=opcode,
        mnemonic="MV A,A",
        length=_length(ctx),
        family="mov_reg",
        binds={},
    )


def _dec_cmp_mem_mem(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    dst = _read_imm8(ctx, "dst")
    src = _read_imm8(ctx, "src")
    return DecodedInstr(
        opcode=opcode,
        mnemonic="CMP (m),(n)",
        length=_length(ctx),
        family="cmp_mem_mem",
        binds={"dst": dst, "src": src},
    )


def _dec_cmpw_mem(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    dst = _read_imm8(ctx, "dst")
    src = _read_imm8(ctx, "src")
    return DecodedInstr(
        opcode=opcode,
        mnemonic="CMPW (m),(n)",
        length=_length(ctx),
        family="cmpw",
        binds={"dst": dst, "src": src},
    )


def _dec_rotate_mem(opcode: int, ctx: StreamCtx, mnemonic: str, op: str) -> DecodedInstr:
    dst = _read_imm8(ctx, "dst")
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family="rotate_mem",
        binds={"dst": dst, "op": op},
    )


def _dec_shift_mem(opcode: int, ctx: StreamCtx, mnemonic: str, op: str) -> DecodedInstr:
    dst = _read_imm8(ctx, "dst")
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family="shift_mem",
        binds={"dst": dst, "op": op},
    )


def _dec_cmpp_mem(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    dst = _read_imm8(ctx, "dst")
    src = _read_imm8(ctx, "src")
    return DecodedInstr(
        opcode=opcode,
        mnemonic="CMPP (m),(n)",
        length=_length(ctx),
        family="cmpp",
        binds={"dst": dst, "src": src},
    )


def _dec_mv_ext_reg(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    entry = _MV_EXT_REGS[opcode]
    imm = _read_addr24(ctx, "addr24", kind="addr24")
    addr = Addr24(imm)
    return DecodedInstr(
        opcode=opcode,
        mnemonic=entry[0],
        length=_length(ctx),
        family="mv_ext_reg",
        binds={"dst": entry[1], "addr24": addr},
    )


def _dec_add_imem_const(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    dst = _read_imm8(ctx, "dst")
    imm = _read_imm8(ctx, "imm")
    return DecodedInstr(
        opcode=opcode,
        mnemonic="ADD (m),n",
        length=_length(ctx),
        family="add_imem",
        binds={"dst": dst, "imm": imm},
    )


def _dec_adc_imem_const(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    dst = _read_imm8(ctx, "dst")
    imm = _read_imm8(ctx, "imm")
    return DecodedInstr(
        opcode=opcode,
        mnemonic="ADC (m),n",
        length=_length(ctx),
        family="adc_imem",
        binds={"dst": dst, "imm": imm},
    )


def _dec_sub_imem_const(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    dst = _read_imm8(ctx, "dst")
    imm = _read_imm8(ctx, "imm")
    return DecodedInstr(
        opcode=opcode,
        mnemonic="SUB (m),n",
        length=_length(ctx),
        family="sub_imem",
        binds={"dst": dst, "imm": imm},
    )


def _dec_adc_a_mem(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    dst = _read_imm8(ctx, "dst")
    return DecodedInstr(
        opcode=opcode,
        mnemonic="ADC A,(n)",
        length=_length(ctx),
        family="adc_a_mem",
        binds={"dst": dst},
    )


def _dec_adc_mem_a(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    dst = _read_imm8(ctx, "dst")
    return DecodedInstr(
        opcode=opcode,
        mnemonic="ADC (n),A",
        length=_length(ctx),
        family="adc_mem_a",
        binds={"dst": dst},
    )


def _dec_sbc_imem_const(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    dst = _read_imm8(ctx, "dst")
    imm = _read_imm8(ctx, "imm")
    return DecodedInstr(
        opcode=opcode,
        mnemonic="SBC (m),n",
        length=_length(ctx),
        family="sbc_imem",
        binds={"dst": dst, "imm": imm},
    )


def _dec_sbc_a_mem(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    dst = _read_imm8(ctx, "dst")
    return DecodedInstr(
        opcode=opcode,
        mnemonic="SBC A,(n)",
        length=_length(ctx),
        family="sbc_a_mem",
        binds={"dst": dst},
    )


def _dec_sbc_mem_a(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    dst = _read_imm8(ctx, "dst")
    return DecodedInstr(
        opcode=opcode,
        mnemonic="SBC (n),A",
        length=_length(ctx),
        family="sbc_mem_a",
        binds={"dst": dst},
    )


def _dec_test_imem_reg(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    dst = _read_imm8(ctx, "dst")
    return DecodedInstr(
        opcode=opcode,
        mnemonic="TEST (n),A",
        length=_length(ctx),
        family="test_imem",
        binds={"dst": dst},
    )


def _dec_emem_logic_const(opcode: int, ctx: StreamCtx, mnemonic: str, family: str) -> DecodedInstr:
    addr = Addr24(_read_addr24(ctx, "addr24", kind="addr24"))
    imm = _read_imm8(ctx, "imm")
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family=family,
        binds={"addr24": addr, "imm": imm},
    )


def _dec_mov_special(opcode: int, ctx: StreamCtx, dst: RegSel, src: RegSel) -> DecodedInstr:
    return DecodedInstr(
        opcode=opcode,
        mnemonic=f"MV {dst.name},{src.name}",
        length=_length(ctx),
        family="mv_reg",
        binds={"dst": dst, "src": src},
    )


def _dec_test_imem_reg(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    dst = _read_imm8(ctx, "dst")
    return DecodedInstr(
        opcode=opcode,
        mnemonic="TEST (n),A",
        length=_length(ctx),
        family="test_imem",
        binds={"dst": dst},
    )


def _dec_xor_emem_const(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    addr = Addr24(_read_addr24(ctx, "addr24", kind="addr24"))
    imm = _read_imm8(ctx, "imm")
    return DecodedInstr(
        opcode=opcode,
        mnemonic="XOR [lmn],n",
        length=_length(ctx),
        family="xor_emem",
        binds={"addr24": addr, "imm": imm},
    )


def _dec_xor_a_mem(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    dst = _read_imm8(ctx, "dst")
    return DecodedInstr(
        opcode=opcode,
        mnemonic="XOR (n),A",
        length=_length(ctx),
        family="xor_a_mem",
        binds={"dst": dst},
    )


def _dec_xor_mem_mem(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    dst = _read_imm8(ctx, "dst")
    src = _read_imm8(ctx, "src")
    return DecodedInstr(
        opcode=opcode,
        mnemonic="XOR (m),(n)",
        length=_length(ctx),
        family="xor_mem_mem",
        binds={"dst": dst, "src": src},
    )


def _dec_mv_reg_imem(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    dst = _MV_REG_IMEM[opcode]
    addr = _read_imm8(ctx, "dst")
    return DecodedInstr(
        opcode=opcode,
        mnemonic=f"MV (n),{dst.name}",
        length=_length(ctx),
        family="mv_reg_imem",
        binds={"dst": dst, "dst_off": addr},
    )


def _dec_mv_imem_to_reg(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    dst_desc = _MV_IMEM_REGS.get(opcode)
    if dst_desc is None:
        raise KeyError(f"MV imem to reg decoder missing for opcode {opcode:#x}")
    mnemonics, regsel = dst_desc
    imm = _read_imm8(ctx, "addr")
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonics,
        length=_length(ctx),
        family="mv_imem_reg",
        binds={"dst": regsel, "addr": imm, "dst_off": imm},
    )


def _dec_logic_mem_mem(opcode: int, ctx: StreamCtx, mnemonic: str, family: str) -> DecodedInstr:
    dst = _read_imm8(ctx, "dst")
    src = _read_imm8(ctx, "src")
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family=family,
        binds={"dst": dst, "src": src},
    )


def _dec_sub_imem_const(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    dst = _read_imm8(ctx, "dst")
    imm = _read_imm8(ctx, "imm")
    return DecodedInstr(
        opcode=opcode,
        mnemonic="SUB (m),n",
        length=_length(ctx),
        family="sub_imem",
        binds={"dst": dst, "imm": imm},
    )


def _dec_imem_logic_const(
    opcode: int, ctx: StreamCtx, mnemonic: str
) -> DecodedInstr:
    dst = _read_imm8(ctx, "dst")
    imm = _read_imm8(ctx, "imm")
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family="imem_logic",
        binds={"dst": dst, "imm": imm},
    )


def _dec_imem_logic_reg(opcode: int, ctx: StreamCtx, mnemonic: str) -> DecodedInstr:
    dst = _read_imm8(ctx, "dst")
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family="imem_logic",
        binds={"dst": dst},
    )


def _dec_and_emem_const(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    return _dec_emem_logic_const(opcode, ctx, "AND [lmn],n", "and_emem")


def _dec_or_emem_const(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    return _dec_emem_logic_const(opcode, ctx, "OR [lmn],n", "or_emem")


def _dec_mv_a_b(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    return _dec_mov_special(opcode, ctx, RegSel("r1", "A"), RegSel("r1", "B"))


def _dec_mv_b_a(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    return _dec_mov_special(opcode, ctx, RegSel("r1", "B"), RegSel("r1", "A"))

def _dec_test_imem_const(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    dst = _read_imm8(ctx, "dst")
    imm = _read_imm8(ctx, "imm")
    return DecodedInstr(
        opcode=opcode,
        mnemonic="TEST (n),n",
        length=_length(ctx),
        family="imem",
        binds={"dst": dst, "imm": imm},
    )


def _dec_test_emem_const(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    addr24 = _read_addr24(ctx, "addr24", kind="addr24")
    imm = _read_imm8(ctx, "imm")
    return DecodedInstr(
        opcode=opcode,
        mnemonic="TEST [lmn],n",
        length=_length(ctx),
        family="test_emem",
        binds={"addr24": Addr24(addr24), "imm": imm},
    )


def _dec_cmp_a_imm(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    imm = _read_imm8(ctx, "imm")
    return DecodedInstr(
        opcode=opcode,
        mnemonic="CMP A,n",
        length=_length(ctx),
        family="cmp_imm",
        binds={"imm": imm},
    )


def _dec_cmp_imem_const(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    dst = _read_imm8(ctx, "dst")
    imm = _read_imm8(ctx, "imm")
    return DecodedInstr(
        opcode=opcode,
        mnemonic="CMP (m),n",
        length=_length(ctx),
        family="cmp_imem",
        binds={"dst": dst, "imm": imm},
    )


def _dec_cmp_imem_reg(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    dst = _read_imm8(ctx, "dst")
    return DecodedInstr(
        opcode=opcode,
        mnemonic="CMP (m),A",
        length=_length(ctx),
        family="cmp_imem",
        binds={"dst": dst},
    )


def _dec_cmp_emem_const(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    addr = Addr24(_read_addr24(ctx, "addr24", kind="addr24"))
    imm = _read_imm8(ctx, "imm")
    return DecodedInstr(
        opcode=opcode,
        mnemonic="CMP [lmn],n",
        length=_length(ctx),
        family="cmp_emem",
        binds={"addr24": addr, "imm": imm},
    )


def _dec_alu_a_mem(opcode: int, ctx: StreamCtx, mnemonic: str, op: str) -> DecodedInstr:
    dst = _read_imm8(ctx, "dst")
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family="alu_a_mem",
        binds={"dst": dst, "op": op},
    )


def _dec_alu_mem_a(opcode: int, ctx: StreamCtx, mnemonic: str, op: str) -> DecodedInstr:
    dst = _read_imm8(ctx, "dst")
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family="alu_mem",
        binds={"dst": dst, "op": op},
    )


def _dec_inc_dec_imem(opcode: int, ctx: StreamCtx, mnemonic: str) -> DecodedInstr:
    imm = _read_imm8(ctx, "dst")
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family="incdec_imem",
        binds={"dst": imm},
    )


def _dec_mv_imem_word_const(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    dst = _read_imm8(ctx, "dst")
    lo = _read_imm8(ctx, "lo")
    hi = _read_imm8(ctx, "hi")
    return DecodedInstr(
        opcode=opcode,
        mnemonic="MVW (l),mn",
        length=_length(ctx),
        family="mv_imem_word",
        binds={"dst": dst, "lo": lo, "hi": hi},
    )


def _dec_jp_cond(opcode: int, ctx: StreamCtx, cond: str) -> DecodedInstr:
    addr = Addr16Page(
        _read_addr16(ctx, "addr16_page", kind="addr16_page"), ctx.page20()
    )
    return DecodedInstr(
        opcode=opcode,
        mnemonic=f"JP{cond} mn",
        length=_length(ctx),
        family="jp_mn",
        binds={"addr16_page": addr, "cond": cond},
    )


def _dec_jp_imem(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    addr = _read_imm8(ctx, "n")
    return DecodedInstr(
        opcode=opcode,
        mnemonic="JP (n)",
        length=_length(ctx),
        family="jp_imem",
        binds={"n": addr},
    )


def _dec_jp_reg(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    start = ctx.bytes_consumed()
    reg_byte = ctx.read_u8()
    _record(ctx, "reg", "regsel", start=start, allowed_groups=("r3",))
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


def _read_regsel(
    ctx: StreamCtx,
    key: str,
    *,
    allowed_groups: tuple[str, ...] | None = None,
) -> RegSel:
    start = ctx.bytes_consumed()
    reg_byte = ctx.read_u8()
    meta: dict[str, object] = {}
    if allowed_groups:
        meta["allowed_groups"] = tuple(allowed_groups)
    _record(ctx, key, "regsel", start=start, **meta)
    name, size_group = _reg_from_byte(reg_byte)
    if allowed_groups and size_group not in allowed_groups:
        raise ValueError(f"Register {name} not allowed for {key}")
    return RegSel(size_group, name)


def _reg_from_index(idx: int) -> RegSel:
    try:
        name, size_group, _ = _REG_TABLE[idx]
    except IndexError:
        raise ValueError(f"Unsupported register index {idx}")
    return RegSel(size_group, name)


def _decode_reg_pair(ctx: StreamCtx) -> tuple[RegSel, RegSel]:
    start = ctx.bytes_consumed()
    raw = ctx.read_u8()
    if raw & 0x88:
        raise ValueError(f"Invalid register pair encoding: {raw:#02x}")
    lhs = _reg_from_index((raw >> 4) & 0x07)
    rhs = _reg_from_index(raw & 0x07)
    _record(
        ctx,
        "reg_pair",
        "reg_pair",
        start=start,
        dst_key="dst",
        src_key="src",
    )
    return lhs, rhs


def _dec_reg_arith(
    opcode: int, ctx: StreamCtx, mnemonic: str, opcode_family: str, *, op: str
) -> DecodedInstr:
    lhs, rhs = _decode_reg_pair(ctx)
    return DecodedInstr(
        opcode=opcode,
        mnemonic=mnemonic,
        length=_length(ctx),
        family=opcode_family,
        binds={"dst": lhs, "src": rhs, "op": op},
    )


def _dec_mv_reg_pair(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    dst, src = _decode_reg_pair(ctx)
    return DecodedInstr(
        opcode=opcode,
        mnemonic="MV r,r",
        length=_length(ctx),
        family="mv_reg",
        binds={"dst": dst, "src": src},
    )
def _dec_ex_regs(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    lhs, rhs = _decode_reg_pair(ctx)
    if lhs.size_group != rhs.size_group:
        raise ValueError("EX instruction requires registers of the same size group")
    return DecodedInstr(
        opcode=opcode,
        mnemonic="EX r,r",
        length=_length(ctx),
        family="ex_reg",
        binds={"lhs": lhs, "rhs": rhs},
    )


DECODERS: Dict[int, DecoderFunc] = {
    0x00: _dec_nop,
    0x01: lambda opcode, ctx: _dec_ret(opcode, ctx, "RETI", "reti"),
    0x02: _dec_jp_mn,
    0x03: _dec_jpf,
    0x08: _dec_mv_a_n,
    0x09: lambda opcode, ctx: _dec_mv_reg_n(
        opcode,
        ctx,
        "MV IL,n",
        RegSel("r1", "IL"),
        _read_imm8,
    ),
    0x0A: lambda opcode, ctx: _dec_mv_reg_n(
        opcode,
        ctx,
        "MVW BA,n",
        RegSel("r2", "BA"),
        lambda c, key: _read_addr16(c, key, kind="imm16"),
    ),
    0x0B: lambda opcode, ctx: _dec_mv_reg_n(
        opcode,
        ctx,
        "MVW I,n",
        RegSel("r2", "I"),
        lambda c, key: _read_addr16(c, key, kind="imm16"),
    ),
    0x0C: lambda opcode, ctx: _dec_mv_reg_n(
        opcode,
        ctx,
        "MV X,n",
        RegSel("r3", "X"),
        _read_imm20,
    ),
    0x0D: lambda opcode, ctx: _dec_mv_reg_n(
        opcode,
        ctx,
        "MV Y,n",
        RegSel("r3", "Y"),
        _read_imm20,
    ),
    0x0E: lambda opcode, ctx: _dec_mv_reg_n(
        opcode,
        ctx,
        "MV U,n",
        RegSel("r3", "U"),
        _read_imm20,
    ),
    0x0F: lambda opcode, ctx: _dec_mv_reg_n(
        opcode,
        ctx,
        "MV S,n",
        RegSel("r3", "S"),
        _read_imm20,
    ),
    0x04: _dec_call,
    0x05: _dec_callf,
    0x06: lambda opcode, ctx: _dec_ret(opcode, ctx, "RET", "ret_near"),
    0x07: lambda opcode, ctx: _dec_ret(opcode, ctx, "RETF", "ret_far"),
    0x10: _dec_jp_imem,
    0x11: _dec_jp_reg,
    0x12: lambda opcode, ctx: _dec_jr_rel(opcode, ctx, +1, "JR +n"),
    0x13: lambda opcode, ctx: _dec_jr_rel(opcode, ctx, -1, "JR -n"),
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
    0x21: _dec_pre,
    0x22: _dec_pre,
    0x23: _dec_pre,
    0x24: _dec_pre,
    0x25: _dec_pre,
    0x26: _dec_pre,
    0x27: _dec_pre,
    0x30: _dec_pre,
    0x31: _dec_pre,
    0x32: _dec_pre,
    0x33: _dec_pre,
    0x34: _dec_pre,
    0x35: _dec_pre,
    0x36: _dec_pre,
    0x37: _dec_pre,
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
    0x97: _dec_sc,
    0x98: _dec_ext_ptr_load,
    0x99: _dec_ext_ptr_load,
    0x9A: _dec_ext_ptr_load,
    0x9B: _dec_ext_ptr_load,
    0x9C: _dec_ext_ptr_load,
    0x9D: _dec_ext_ptr_load,
    0x9E: _dec_ext_ptr_load,
    0x40: _dec_alu_imm,
    0x41: _dec_add_imem_const,
    0x42: lambda opcode, ctx: _dec_alu_a_mem(opcode, ctx, "ADD A,(n)", "add"),
    0x43: lambda opcode, ctx: _dec_alu_mem_a(opcode, ctx, "ADD (n),A", "add"),
    0x44: lambda opcode, ctx: _dec_reg_arith(opcode, ctx, "ADD r,r", "reg_add", op="add"),
    0x45: lambda opcode, ctx: _dec_reg_arith(opcode, ctx, "ADD r,r", "reg_add", op="add"),
    0x46: lambda opcode, ctx: _dec_reg_arith(opcode, ctx, "ADD r,r", "reg_add", op="add"),
    0x47: _dec_pmdf_imm,
    0x48: _dec_alu_imm,
    0x49: _dec_sub_imem_const,
    0x4A: lambda opcode, ctx: _dec_alu_a_mem(opcode, ctx, "SUB A,(n)", "sub"),
    0x4B: lambda opcode, ctx: _dec_alu_mem_a(opcode, ctx, "SUB (n),A", "sub"),
    0x4C: lambda opcode, ctx: _dec_reg_arith(opcode, ctx, "SUB r,r", "reg_sub", op="sub"),
    0x4D: lambda opcode, ctx: _dec_reg_arith(opcode, ctx, "SUB r,r", "reg_sub", op="sub"),
    0x4E: lambda opcode, ctx: _dec_reg_arith(opcode, ctx, "SUB r,r", "reg_sub", op="sub"),
    0x4F: _dec_stack_sys,
    0x50: _dec_alu_imm,
    0x51: _dec_adc_imem_const,
    0x52: _dec_adc_a_mem,
    0x53: _dec_adc_mem_a,
    0x54: lambda opcode, ctx: _dec_loop_arith_mem(opcode, ctx, "ADCL (m),(n)"),
    0x55: lambda opcode, ctx: _dec_loop_arith_reg(opcode, ctx, "ADCL (m),A"),
    0x56: _dec_mv_imem_word_const,
    0x57: _dec_pmdf_reg,
    0x58: _dec_alu_imm,
    0x59: _dec_sbc_imem_const,
    0x5A: _dec_sbc_a_mem,
    0x5B: _dec_sbc_mem_a,
    0x5C: lambda opcode, ctx: _dec_loop_arith_mem(opcode, ctx, "SBCL (m),(n)"),
    0x5D: lambda opcode, ctx: _dec_loop_arith_reg(opcode, ctx, "SBCL (m),A"),
    0x5E: _dec_mv_imem_word_const,
    0x5F: _dec_stack_sys,
    0x60: _dec_cmp_a_imm,
    0x61: _dec_cmp_imem_const,
    0x62: _dec_cmp_emem_const,
    0x63: _dec_cmp_imem_reg,
    0x64: _dec_alu_imm,
    0x65: _dec_test_imem_const,
    0x66: _dec_test_emem_const,
    0x67: _dec_test_imem_reg,
    0x68: _dec_alu_imm,
    0x69: lambda opcode, ctx: _dec_imem_logic_const(opcode, ctx, "XOR (n),n"),
    0x6A: _dec_xor_emem_const,
    0x6B: _dec_xor_a_mem,
    0x6E: _dec_xor_mem_mem,
    0x6F: lambda opcode, ctx: _dec_alu_a_mem(opcode, ctx, "XOR A,(n)", "xor"),
    0x70: _dec_alu_imm,
    0x71: lambda opcode, ctx: _dec_imem_logic_const(opcode, ctx, "AND (n),n"),
    0x72: _dec_and_emem_const,
    0x73: lambda opcode, ctx: _dec_imem_logic_reg(opcode, ctx, "AND (n),A"),
    0x74: lambda opcode, ctx: _dec_mov_special(opcode, ctx, RegSel("r1","A"), RegSel("r1","B")),
    0x75: lambda opcode, ctx: _dec_mov_special(opcode, ctx, RegSel("r1","B"), RegSel("r1","A")),
    0x76: lambda opcode, ctx: _dec_logic_mem_mem(opcode, ctx, "AND (m),(n)", "imem_logic_mem"),
    0x77: lambda opcode, ctx: _dec_imem_logic_reg(opcode, ctx, "AND A,(n)"),
    0x78: _dec_alu_imm,
    0x79: lambda opcode, ctx: _dec_imem_logic_const(opcode, ctx, "OR (n),n"),
    0x7A: _dec_or_emem_const,
    0x7B: lambda opcode, ctx: _dec_imem_logic_reg(opcode, ctx, "OR (n),A"),
    0x7E: lambda opcode, ctx: _dec_logic_mem_mem(opcode, ctx, "OR (m),(n)", "imem_logic_mem"),
    0x7F: lambda opcode, ctx: _dec_imem_logic_reg(opcode, ctx, "OR A,(n)"),
    0x6C: lambda opcode, ctx: _dec_inc_dec(opcode, ctx, "INC r"),
    0x6D: lambda opcode, ctx: _dec_inc_dec_imem(opcode, ctx, "INC (n)"),
    0x7C: lambda opcode, ctx: _dec_inc_dec(opcode, ctx, "DEC r"),
    0x7D: lambda opcode, ctx: _dec_inc_dec_imem(opcode, ctx, "DEC (n)"),
    0x80: _dec_mv_imem_to_reg,
    0x81: _dec_mv_imem_to_reg,
    0x82: _dec_mv_imem_to_reg,
    0x83: _dec_mv_imem_to_reg,
    0x84: _dec_mv_imem_to_reg,
    0x85: _dec_mv_imem_to_reg,
    0x86: _dec_mv_imem_to_reg,
    0x87: _dec_mv_imem_to_reg,
    0x88: _dec_mv_ext_reg,
    0x89: _dec_mv_ext_reg,
    0x8A: _dec_mv_ext_reg,
    0x8B: _dec_mv_ext_reg,
    0x8C: _dec_mv_ext_reg,
    0x8D: _dec_mv_ext_reg,
    0x8E: _dec_mv_ext_reg,
    0x8F: _dec_mv_ext_reg,
    0xA0: _dec_mv_reg_imem,
    0xA1: _dec_mv_reg_imem,
    0xA2: _dec_mv_reg_imem,
    0xA3: _dec_mv_reg_imem,
    0xA4: _dec_mv_reg_imem,
    0xA5: _dec_mv_reg_imem,
    0xA6: _dec_mv_reg_imem,
    0xA7: _dec_mv_reg_imem,
    0xA8: _dec_mv_ext_store,
    0xA9: _dec_mv_ext_store,
    0xAA: _dec_mv_ext_store,
    0xAB: _dec_mv_ext_store,
    0xAC: _dec_mv_ext_store,
    0xAD: _dec_mv_ext_store,
    0xAE: _dec_mv_ext_store,
    0xAF: _dec_mv_ext_store,
    0xB0: _dec_ext_reg_store,
    0xB1: _dec_ext_reg_store,
    0xB2: _dec_ext_reg_store,
    0xB3: _dec_ext_reg_store,
    0xB4: _dec_ext_reg_store,
    0xB5: _dec_ext_reg_store,
    0xB6: _dec_ext_reg_store,
    0xB7: _dec_cmp_mem_mem,
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
    0xC3: _dec_imem_swap,
    0xC4: lambda opcode, ctx: _dec_loop_bcd_mem(opcode, ctx, "DADL (m),(n)"),
    0xC5: lambda opcode, ctx: _dec_loop_bcd_reg(opcode, ctx, "DADL (m),A"),
    0xC6: _dec_cmpw_mem,
    0xC7: _dec_cmpp_mem,
    0xC8: _dec_imem_move,
    0xCB: _dec_imem_loop_move,
    0xC9: _dec_imem_move,
    0xCA: _dec_imem_move,
    0xCC: _dec_mv_imem_const,
    0xCD: _dec_mv_imem_word_const,
    0xCF: _dec_imem_loop_move,
    0xD0: lambda opcode, ctx: _dec_mv_imem_from_ext(opcode, ctx, "MV (n),[lmn]", 8),
    0xD1: lambda opcode, ctx: _dec_mv_imem_from_ext(opcode, ctx, "MVW (n),[lmn]", 16),
    0xD2: lambda opcode, ctx: _dec_mv_imem_from_ext(opcode, ctx, "MVP (n),[lmn]", 24),
    0xD3: lambda opcode, ctx: _dec_mv_imem_from_ext(opcode, ctx, "MVL (n),[lmn]", 8),
    0xDC: lambda opcode, ctx: _dec_mv_imem_from_ext_const(
        opcode, ctx, "MVP (n),const", 24
    ),
    0xD6: _dec_cmpw_mem,
    0xD7: _dec_cmpp_mem,
    0xD8: lambda opcode, ctx: _dec_mv_ext_from_imem(opcode, ctx, "MV [lmn],(n)", 8),
    0xD9: lambda opcode, ctx: _dec_mv_ext_from_imem(opcode, ctx, "MVW [lmn],(n)", 16),
    0xDA: lambda opcode, ctx: _dec_mv_ext_from_imem(opcode, ctx, "MVP [lmn],(n)", 24),
    0xDB: lambda opcode, ctx: _dec_mv_ext_from_imem(opcode, ctx, "MVL [lmn],(n)", 8),
    0xC3: _dec_imem_swap,
    0xCE: _dec_tcl,
    0xEC: lambda opcode, ctx: _dec_decimal_shift(opcode, ctx, "DSLL (m)"),
    0xD4: lambda opcode, ctx: _dec_loop_bcd_mem(opcode, ctx, "DSBL (m),(n)"),
    0xD5: lambda opcode, ctx: _dec_loop_bcd_reg(opcode, ctx, "DSBL (m),A"),
    0xFC: lambda opcode, ctx: _dec_decimal_shift(opcode, ctx, "DSRL (m)"),
    0xE0: _dec_imem_from_ext,
    0xE1: _dec_imem_from_ext,
    0xE2: _dec_imem_from_ext,
    0xE3: _dec_imem_from_ext,
    0xE4: lambda opcode, ctx: _dec_simple(opcode, ctx, "ROR A", "system"),
    0xE5: lambda opcode, ctx: _dec_rotate_mem(opcode, ctx, "ROR (n)", "ror"),
    0xE6: lambda opcode, ctx: _dec_simple(opcode, ctx, "ROL A", "system"),
    0xE7: lambda opcode, ctx: _dec_rotate_mem(opcode, ctx, "ROL (n)", "rol"),
    0xE8: _dec_ext_from_imem,
    0xE9: _dec_ext_from_imem,
    0xEA: _dec_ext_from_imem,
    0xEB: _dec_ext_from_imem,
    0xED: _dec_ex_regs,
    0xDD: _dec_ex_ab,
    0xDE: lambda opcode, ctx: _dec_simple(opcode, ctx, "HALT", "system"),
    0xDF: lambda opcode, ctx: _dec_simple(opcode, ctx, "OFF", "system"),
    0xEE: _dec_swap_a,
    0xEF: lambda opcode, ctx: _dec_simple(opcode, ctx, "WAIT", "system"),
    0xF0: lambda opcode, ctx: _dec_emem_imem_offset(
        opcode, ctx, "MV (m),[(n)]", 8, order="int"
    ),
    0xF1: lambda opcode, ctx: _dec_emem_imem_offset(
        opcode, ctx, "MVW (m),[(n)]", 16, order="int"
    ),
    0xF2: lambda opcode, ctx: _dec_emem_imem_offset(
        opcode, ctx, "MVP (m),[(n)]", 24, order="int"
    ),
    0xF3: lambda opcode, ctx: _dec_emem_imem_offset(
        opcode, ctx, "MVL (m),[(n)]", 8, order="int"
    ),
    0xF4: lambda opcode, ctx: _dec_simple(opcode, ctx, "SHR A", "system"),
    0xF5: lambda opcode, ctx: _dec_shift_mem(opcode, ctx, "SHR (n)", "shr"),
    0xF6: lambda opcode, ctx: _dec_simple(opcode, ctx, "SHL A", "system"),
    0xF7: lambda opcode, ctx: _dec_shift_mem(opcode, ctx, "SHL (n)", "shl"),
    0xF8: lambda opcode, ctx: _dec_emem_imem_offset(
        opcode, ctx, "MV [(n)],(m)", 8, order="ext"
    ),
    0xF9: lambda opcode, ctx: _dec_emem_imem_offset(
        opcode, ctx, "MVW [(n)],(m)", 16, order="ext"
    ),
    0xFA: lambda opcode, ctx: _dec_emem_imem_offset(
        opcode, ctx, "MVP [(n)],(m)", 24, order="ext"
    ),
    0xFB: lambda opcode, ctx: _dec_emem_imem_offset(
        opcode, ctx, "MVL [(n)],(m)", 8, order="ext"
    ),
    0xFD: _dec_mv_reg_pair,
    0xFE: lambda opcode, ctx: _dec_simple(opcode, ctx, "IR", "system"),
    0xFF: lambda opcode, ctx: _dec_simple(opcode, ctx, "RESET", "system"),
    0x9F: _dec_rc,
    0xBF: _dec_unknown_bf,
}


def decode_opcode(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    try:
        decoder = DECODERS[opcode]
    except KeyError as exc:
        raise KeyError(f"No decoder registered for opcode {opcode:#x}") from exc
    return decoder(opcode, ctx)


def decode_with_pre_variants(opcode: int, ctx: StreamCtx) -> Tuple[DecodedInstr, ...]:
    """Decode an opcode and return all PRE variants (baseline + prefixes)."""

    decoded = decode_opcode(opcode, ctx)
    return tuple(iter_all_pre_variants(decoded))
