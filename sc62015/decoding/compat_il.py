from __future__ import annotations

from typing import Callable, Dict

from binaryninja import FlagName, RegisterName  # type: ignore
from binaryninja.lowlevelil import LowLevelILFunction, LowLevelILLabel  # type: ignore

from .bind import Addr16Page, Addr24, DecodedInstr, Disp8, Imm8

Z_FLAG = FlagName("Z")


def _emit_mv_a_n(di: DecodedInstr, il: LowLevelILFunction, addr: int) -> None:
    imm = di.binds["n"]
    assert isinstance(imm, Imm8)
    il.append(il.set_reg(1, RegisterName("A"), il.const(1, imm.value)))


def _emit_jrz(di: DecodedInstr, il: LowLevelILFunction, addr: int) -> None:
    disp = di.binds["disp"]
    assert isinstance(disp, Disp8)
    target = (addr + di.length + disp.value) & 0xFFFFF

    if_true = LowLevelILLabel()
    if_false = LowLevelILLabel()

    flag_expr = il.flag(Z_FLAG)
    cond = il.compare_equal(1, flag_expr, il.const(1, 1))
    il.append(il.if_expr(cond, if_true, if_false))

    il.mark_label(if_true)
    il.append(il.jump(il.const(3, target)))
    il.mark_label(if_false)


def _emit_jp_mn(di: DecodedInstr, il: LowLevelILFunction, addr: int) -> None:
    addr_info = di.binds["addr16_page"]
    assert isinstance(addr_info, Addr16Page)

    low = il.const(2, addr_info.offs16.u16)
    high = il.const(3, addr_info.page20)
    dest = il.or_expr(3, low, high)

    if_true = LowLevelILLabel()
    if_false = LowLevelILLabel()
    il.mark_label(if_true)
    il.append(il.jump(dest))
    il.mark_label(if_false)


def _emit_mv_a_abs24(di: DecodedInstr, il: LowLevelILFunction, addr: int) -> None:
    addr_info = di.binds["addr24"]
    assert isinstance(addr_info, Addr24)
    ptr = il.const_pointer(3, addr_info.v.u24)
    il.append(il.set_reg(1, RegisterName("A"), il.load(1, ptr)))


EMITTERS: Dict[int, Callable[[DecodedInstr, LowLevelILFunction, int], None]] = {
    0x08: _emit_mv_a_n,
    0x18: _emit_jrz,
    0x19: _emit_jrz,
    0x02: _emit_jp_mn,
    0x88: _emit_mv_a_abs24,
}


def emit_instruction(di: DecodedInstr, il: LowLevelILFunction, addr: int) -> None:
    try:
        emitter = EMITTERS[di.opcode]
    except KeyError as exc:
        raise NotImplementedError(f"No compat emitter for opcode {di.opcode:#x}") from exc
    emitter(di, il, addr)
