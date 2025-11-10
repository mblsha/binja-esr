from __future__ import annotations

from typing import Callable, Dict

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
    RegSel,
)
from .reader import StreamCtx

DecoderFunc = Callable[[int, StreamCtx], DecodedInstr]


def _length(ctx: StreamCtx) -> int:
    return ctx.total_length()


def _dec_mv_a_n(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    imm = Imm8(ctx.read_u8())
    return DecodedInstr(
        opcode=opcode,
        mnemonic="MV A,n",
        length=_length(ctx),
        binds={
            "dst": RegSel("r1", "A"),
            "n": imm,
        },
    )


def _dec_jrz(opcode: int, ctx: StreamCtx, direction: int) -> DecodedInstr:
    magnitude = ctx.read_u8()
    disp = Disp8(direction * magnitude)
    return DecodedInstr(
        opcode=opcode,
        mnemonic="JRZ ±n",
        length=_length(ctx),
        binds={
            "disp": disp,
        },
    )


def _dec_jp_mn(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    lo, hi = ctx.read_u16_mn()
    addr = Addr16Page(Imm16(lo, hi), ctx.page20())
    return DecodedInstr(
        opcode=opcode,
        mnemonic="JP mn",
        length=_length(ctx),
        binds={"addr16_page": addr},
    )


def _dec_mv_a_abs24(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    lo, mid, hi = ctx.read_u24_lmn()
    addr = Addr24(Imm24(lo, mid, hi))
    return DecodedInstr(
        opcode=opcode,
        mnemonic="MV A,[lmn]",
        length=_length(ctx),
        binds={"addr24": addr},
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
        pre_latch=latch,
        binds={},
    )


DECODERS: Dict[int, DecoderFunc] = {
    0x08: _dec_mv_a_n,
    0x18: lambda opcode, ctx: _dec_jrz(opcode, ctx, +1),
    0x19: lambda opcode, ctx: _dec_jrz(opcode, ctx, -1),
    0x02: _dec_jp_mn,
    0x88: _dec_mv_a_abs24,
    0x32: _dec_pre,
}


def decode_opcode(opcode: int, ctx: StreamCtx) -> DecodedInstr:
    try:
        decoder = DECODERS[opcode]
    except KeyError as exc:
        raise KeyError(f"No decoder registered for opcode {opcode:#x}") from exc
    return decoder(opcode, ctx)
