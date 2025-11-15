from __future__ import annotations

from typing import Callable, Dict, Tuple

from binaryninja import FlagName, RegisterName  # type: ignore
from binaryninja.lowlevelil import LowLevelILFunction, LowLevelILLabel  # type: ignore

from sc62015.scil.compat_builder import CompatLLILBuilder
from sc62015.scil import from_decoded
from sc62015.scil.backend_llil import emit_llil as emit_scil_llil
from sc62015.decoding.bind import (
    Addr16Page,
    Addr24,
    DecodedInstr,
    Imm8,
    IntAddrCalc,
    RegSel,
    PreLatch,
)

Z_FLAG = FlagName("Z")
C_FLAG = FlagName("C")


def _builder(il: LowLevelILFunction) -> CompatLLILBuilder:
    return CompatLLILBuilder(il)


def _emit_pre(di: DecodedInstr, il: LowLevelILFunction, addr: int) -> None:
    return None


def _emit_mv_a_n(di: DecodedInstr, il: LowLevelILFunction, addr: int) -> None:
    imm = di.binds["n"]
    assert isinstance(imm, Imm8)
    il.append(il.set_reg(1, RegisterName("A"), il.const(1, imm.value)))


def _emit_jr_cond(di: DecodedInstr, il: LowLevelILFunction, addr: int) -> None:
    disp = di.binds["disp"]
    direction = di.binds.get("dir", 1)
    cond = di.binds.get("cond", "Z")
    assert isinstance(disp, Imm8)
    displacement = disp.value if direction > 0 else -disp.value
    target = (addr + di.length + displacement) & 0xFFFFF

    if_true = LowLevelILLabel()
    if_false = LowLevelILLabel()

    flag = Z_FLAG if "Z" in cond else C_FLAG
    expect = 0 if "N" in cond else 1
    cond_expr = il.compare_equal(1, il.flag(flag), il.const(1, expect))
    il.append(il.if_expr(cond_expr, if_true, if_false))

    il.mark_label(if_true)
    il.append(il.jump(il.const(3, target)))
    il.mark_label(if_false)


def _jp_dest(di: DecodedInstr, il: LowLevelILFunction) -> int:
    addr_info = di.binds["addr16_page"]
    assert isinstance(addr_info, Addr16Page)
    low = il.const(2, addr_info.offs16.u16)
    high = il.const(3, addr_info.page20)
    return il.or_expr(3, low, high)


def _emit_jp_mn(di: DecodedInstr, il: LowLevelILFunction, addr: int) -> None:
    dest = _jp_dest(di, il)
    if_true = LowLevelILLabel()
    if_false = LowLevelILLabel()
    il.mark_label(if_true)
    il.append(il.jump(dest))
    il.mark_label(if_false)


def _emit_jp_cond(di: DecodedInstr, il: LowLevelILFunction, addr: int) -> None:
    cond = di.binds.get("cond", "Z")
    flag = Z_FLAG if "Z" in cond else C_FLAG
    expect = 0 if "N" in cond else 1

    if_true = LowLevelILLabel()
    if_false = LowLevelILLabel()
    cond_expr = il.compare_equal(1, il.flag(flag), il.const(1, expect))
    il.append(il.if_expr(cond_expr, if_true, if_false))

    il.mark_label(if_true)
    il.append(il.jump(_jp_dest(di, il)))
    il.mark_label(if_false)


def _emit_mv_a_abs24(di: DecodedInstr, il: LowLevelILFunction, addr: int) -> None:
    addr_info = di.binds["addr24"]
    assert isinstance(addr_info, Addr24)
    ptr = il.const_pointer(3, addr_info.v.u24)
    il.append(il.set_reg(1, RegisterName("A"), il.load(1, ptr)))


def _emit_mv_ext_store(di: DecodedInstr, il: LowLevelILFunction, addr: int) -> None:
    addr_info = di.binds["addr24"]
    assert isinstance(addr_info, Addr24)
    ptr = il.const_pointer(3, addr_info.v.u24)
    il.append(il.store(1, ptr, il.reg(1, RegisterName("A"))))


def _emit_alu_imm(
    di: DecodedInstr,
    il: LowLevelILFunction,
    op: str,
    flags: Tuple[str, ...],
    with_carry: bool = False,
) -> None:
    builder = _builder(il)
    imm = di.binds["n"]
    assert isinstance(imm, Imm8)
    lhs = il.reg(1, RegisterName("A"))
    rhs = il.const(1, imm.value)
    if with_carry:
        rhs = il.add(1, rhs, il.flag(C_FLAG))
    expr = builder.binop_with_flags(op, 8, lhs, rhs, flags)
    il.append(il.set_reg(1, RegisterName("A"), expr))


def _emit_test(di: DecodedInstr, il: LowLevelILFunction, addr: int) -> None:
    imm = di.binds["n"]
    assert isinstance(imm, Imm8)
    and_expr = il.and_expr(
        3,
        il.reg(1, RegisterName("A")),
        il.const(1, imm.value),
    )
    il.append(il.set_flag(Z_FLAG, il.compare_equal(3, and_expr, il.const(3, 0))))


def _emit_inc_dec(di: DecodedInstr, il: LowLevelILFunction, op: str) -> None:
    builder = _builder(il)
    reg_sel = di.binds["reg"]
    assert isinstance(reg_sel, RegSel)
    reg = RegisterName(reg_sel.name)
    lhs = il.reg(1, reg)
    rhs = il.const(1, 1)
    expr = builder.binop_with_flags(op, 8, lhs, rhs, ("Z",))
    il.append(il.set_reg(1, reg, expr))


def _imem_mode(di: DecodedInstr, *, slot: int) -> IntAddrCalc:
    latch: PreLatch | None = getattr(di, "pre_applied", None)
    if latch is None:
        return IntAddrCalc.BP_N
    if slot == 0:
        return latch.first
    return latch.second


def _emit_imem_load(di: DecodedInstr, il: LowLevelILFunction) -> None:
    builder = _builder(il)
    imm = di.binds["n"]
    assert isinstance(imm, Imm8)
    ptr = builder.imem_address(_imem_mode(di, slot=0), il.const(1, imm.value))
    il.append(il.set_reg(1, RegisterName("A"), il.load(1, ptr)))


def _emit_imem_store(di: DecodedInstr, il: LowLevelILFunction) -> None:
    builder = _builder(il)
    imm = di.binds["n"]
    assert isinstance(imm, Imm8)
    ptr = builder.imem_address(_imem_mode(di, slot=0), il.const(1, imm.value))
    il.append(il.store(1, ptr, il.reg(1, RegisterName("A"))))


EMITTERS: Dict[int, Callable[[DecodedInstr, LowLevelILFunction, int], None]] = {
    0x02: _emit_jp_mn,
    0x08: _emit_mv_a_n,
    0x14: _emit_jp_cond,
    0x15: _emit_jp_cond,
    0x16: _emit_jp_cond,
    0x17: _emit_jp_cond,
    0x18: _emit_jr_cond,
    0x19: _emit_jr_cond,
    0x1A: _emit_jr_cond,
    0x1B: _emit_jr_cond,
    0x1C: _emit_jr_cond,
    0x1D: _emit_jr_cond,
    0x1E: _emit_jr_cond,
    0x1F: _emit_jr_cond,
    0x32: _emit_pre,
    0x40: lambda di, il, addr: _emit_alu_imm(di, il, "add", ("C", "Z")),
    0x48: lambda di, il, addr: _emit_alu_imm(di, il, "sub", ("C", "Z")),
    0x50: lambda di, il, addr: _emit_alu_imm(
        di, il, "add", ("C", "Z"), with_carry=True
    ),
    0x58: lambda di, il, addr: _emit_alu_imm(
        di, il, "sub", ("C", "Z"), with_carry=True
    ),
    0x64: _emit_test,
    0x68: lambda di, il, addr: _emit_alu_imm(di, il, "xor", ("Z",)),
    0x70: lambda di, il, addr: _emit_alu_imm(di, il, "and", ("Z",)),
    0x78: lambda di, il, addr: _emit_alu_imm(di, il, "or", ("Z",)),
    0x6C: lambda di, il, addr: _emit_inc_dec(di, il, "add"),
    0x7C: lambda di, il, addr: _emit_inc_dec(di, il, "sub"),
    0x80: lambda di, il, addr: _emit_imem_load(di, il),
    0x88: _emit_mv_a_abs24,
    0xA0: lambda di, il, addr: _emit_imem_store(di, il),
    0xA8: _emit_mv_ext_store,
}


def emit_instruction(di: DecodedInstr, il: LowLevelILFunction, addr: int) -> None:
    emitter = EMITTERS.get(di.opcode)
    if emitter is None:
        _emit_via_scil(di, il, addr)
        return
    emitter(di, il, addr)


def _emit_via_scil(di: DecodedInstr, il: LowLevelILFunction, addr: int) -> None:
    payload = from_decoded.build(di)
    emit_scil_llil(
        il,
        payload.instr,
        payload.binder,
        CompatLLILBuilder(il),
        addr,
        pre_applied=payload.pre_applied,
    )
