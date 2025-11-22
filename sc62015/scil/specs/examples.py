from __future__ import annotations

from typing import Any

from ..ast import (
    BinOp,
    Expr,
    Cond,
    Const,
    Effect,
    Fetch,
    Flag,
    Goto,
    If,
    Instr,
    Join24,
    Mem,
    PcRel,
    Reg,
    Store,
    SetFlag,
    SetReg,
    Tmp,
    TernOp,
    UnOp,
    LoopIntPtr,
)
from ...pysc62015.constants import INTERNAL_MEMORY_START
from ...pysc62015.instr.opcodes import IMEMRegisters

_IMR_OFFSET = IMEMRegisters["IMR"].value & 0xFF
_IMR_ABS_ADDR = INTERNAL_MEMORY_START + _IMR_OFFSET


def _imr_mem() -> Mem:
    return Mem("int", Const(_IMR_ABS_ADDR, 24), 8)


def nop_instr() -> Instr:
    return Instr(
        name="NOP",
        length=1,
        semantics=(Effect("nop", ()),),
    )


def mv_a_imm() -> Instr:
    imm = Tmp("imm8", 8)
    return Instr(
        name="MV_A_IMM",
        length=2,
        semantics=(
            Fetch("u8", imm),
            SetReg(Reg("A", 8), imm),
        ),
    )


def _flag_cond(flag: str, expect_one: bool) -> Cond:
    target = Const(1 if expect_one else 0, 1)
    return Cond(kind="eq", a=Flag(flag), b=target)


def jr_rel(name: str, flag: str, expect_one: bool) -> Instr:
    disp = Tmp("disp8", 8)
    taken = PcRel(base_advance=2, disp=UnOp("sext", disp, 20), out_size=20)
    return Instr(
        name=name,
        length=2,
        semantics=(
            Fetch("disp8", disp),
            If(
                cond=_flag_cond(flag, expect_one),
                then_ops=(Goto(taken),),
            ),
        ),
    )


def mv_a_abs_ext() -> Instr:
    addr = Tmp("addr_ptr", 24)
    return Instr(
        name="MV_A_ABS_EXT",
        length=4,
        semantics=(
            Fetch("addr24", addr),
            SetReg(Reg("A", 8), Mem("ext", addr, 8)),
        ),
    )


def jp_paged() -> Instr:
    addr_lo = Tmp("addr16", 16)
    page_hi = Tmp("page_hi", 20)
    joined = BinOp(
        "or",
        UnOp("zext", addr_lo, 20),
        UnOp("zext", page_hi, 20),
        20,
    )
    return Instr(
        name="JP_PAGED",
        length=3,
        semantics=(
            Fetch("addr16_page", addr_lo),
            Fetch("addr16_page_hi", page_hi),
            Goto(joined),
        ),
    )


def jp_far() -> Instr:
    addr = Tmp("addr20", 20)
    return Instr(
        name="JPF",
        length=4,
        semantics=(
            Fetch("imm20", addr),
            Goto(addr),
        ),
    )


def jp_cond(name: str, cond: Cond) -> Instr:
    addr_lo = Tmp("addr16", 16)
    page_hi = Tmp("page_hi", 20)
    joined = BinOp(
        "or",
        UnOp("zext", addr_lo, 20),
        UnOp("zext", page_hi, 20),
        20,
    )
    return Instr(
        name=name,
        length=3,
        semantics=(
            Fetch("addr16_page", addr_lo),
            Fetch("addr16_page_hi", page_hi),
            If(cond=cond, then_ops=(Goto(joined),)),
        ),
    )


def alu_a_imm(
    name: str,
    op: str,
    *,
    include_carry: bool = False,
    flags: tuple[str, ...],
) -> Instr:
    imm = Tmp("imm8", 8)
    rhs = imm
    if include_carry:
        rhs = BinOp("add", imm, Flag("C"), 8)
    value = BinOp(op, Reg("A", 8), rhs, 8)
    return Instr(
        name=name,
        length=2,
        semantics=(
            Fetch("u8", imm),
            SetReg(Reg("A", 8), value, flags=flags),
        ),
    )


def test_a_imm() -> Instr:
    imm = Tmp("imm8", 8)
    and_expr = BinOp("and", Reg("A", 8), imm, 24)
    cmp_zero = BinOp("eq", and_expr, Const(0, 24), 1)
    return Instr(
        name="TEST A,n",
        length=2,
        semantics=(
            Fetch("u8", imm),
            SetFlag("Z", cmp_zero),
        ),
    )


def mv_ext_store() -> Instr:
    addr = Tmp("addr_ptr", 24)
    return Instr(
        name="MV_[LMN]_A",
        length=4,
        semantics=(
            Fetch("addr24", addr),
            Store(Mem("ext", addr, 8), Reg("A", 8)),
        ),
    )


def mv_ext_store_reg(name: str, width: int) -> Instr:
    addr = Tmp("addr_ptr", 24)
    return Instr(
        name=f"MV_[LMN]_{name}",
        length=4,
        semantics=(
            Fetch("addr24", addr),
            Store(Mem("ext", addr, width), Reg(name, width)),
        ),
    )


def mv_imem_from_ext(name: str, width: int) -> Instr:
    dst = Tmp("dst_off", 8)
    addr = Tmp("addr24", 24)
    dst_mem = Mem("int", dst, width)
    src_mem = Mem("ext", addr, width)
    return Instr(
        name=name,
        length=4,
        semantics=(
            Fetch("u8", dst),
            Fetch("addr24", addr),
            Store(dst_mem, src_mem),
        ),
    )


def mv_ext_from_imem(name: str, width: int) -> Instr:
    addr = Tmp("addr24", 24)
    src = Tmp("src_off", 8)
    dst_mem = Mem("ext", addr, width)
    src_mem = Mem("int", src, width)
    return Instr(
        name=name,
        length=4,
        semantics=(
            Fetch("addr24", addr),
            Fetch("u8", src),
            Store(dst_mem, src_mem),
        ),
    )


def mv_ext_store_reg(name: str, width: int) -> Instr:
    addr = Tmp("addr_ptr", 24)
    return Instr(
        name=f"MV_[LMN]_{name}",
        length=4,
        semantics=(
            Fetch("addr24", addr),
            Store(Mem("ext", addr, width), Reg(name, width)),
        ),
    )


def call_near() -> Instr:
    lo = Tmp("call_addr16", 16)
    page = Tmp("call_page_hi", 20)
    next_pc = PcRel(base_advance=3, out_size=20)
    return Instr(
        name="CALL mn",
        length=3,
        semantics=(
            Fetch("addr16_page", lo),
            Fetch("addr16_page_hi", page),
            Effect("push_ret16", (next_pc,)),
            Effect("goto_page_join", (lo, page)),
        ),
    )


def call_far() -> Instr:
    addr = Tmp("call_addr24", 24)
    next_pc = PcRel(base_advance=4, out_size=20)
    return Instr(
        name="CALLF lmn",
        length=4,
        semantics=(
            Fetch("addr24", addr),
            Effect("push_ret24", (next_pc,)),
            Effect("goto_far24", (addr,)),
        ),
    )


def ret_near() -> Instr:
    return Instr(
        name="RET",
        length=1,
        semantics=(Effect("ret_near", ()),),
    )


def ret_far() -> Instr:
    return Instr(
        name="RETF",
        length=1,
        semantics=(Effect("ret_far", ()),),
    )


def reti() -> Instr:
    return Instr(
        name="RETI",
        length=1,
        semantics=(Effect("reti", ()),),
    )


def mv_a_imem() -> Instr:
    offset = Tmp("imem_off", 8)
    return Instr(
        name="MV_A_(N)",
        length=2,
        semantics=(
            Fetch("u8", offset),
            SetReg(Reg("A", 8), Mem("int", offset, 8)),
        ),
    )


def mv_imem_a() -> Instr:
    offset = Tmp("imem_off", 8)
    return Instr(
        name="MV_(N)_A",
        length=2,
        semantics=(
            Fetch("u8", offset),
            Store(Mem("int", offset, 8), Reg("A", 8)),
        ),
    )


def test_imem_const() -> Instr:
    dst = Tmp("dst_off", 8)
    value = Tmp("imm8", 8)
    mem = Mem("int", dst, 8)
    and_expr = BinOp("and", mem, value, 24)
    cmp_zero = BinOp("eq", and_expr, Const(0, 24), 1)
    return Instr(
        name="TEST_(N),N",
        length=3,
        semantics=(
            Fetch("u8", dst),
            Fetch("u8", value),
            SetFlag("Z", cmp_zero),
        ),
    )


def cmp_imem_const() -> Instr:
    dst = Tmp("dst_off", 8)
    value = Tmp("imm8", 8)
    mem = Mem("int", dst, 8)
    eq_zero = BinOp("eq", mem, value, 1)
    cond = Cond(kind="ltu", a=mem, b=value)
    c_expr = TernOp(
        op="select",
        cond=cond,
        t=Const(1, 1),
        f=Const(0, 1),
        out_size=1,
    )
    return Instr(
        name="CMP_(M),N",
        length=3,
        semantics=(
            Fetch("u8", dst),
            Fetch("u8", value),
            SetFlag("Z", eq_zero),
            SetFlag("C", c_expr),
        ),
    )


def cmp_emem_const() -> Instr:
    addr = Tmp("addr24", 24)
    value = Tmp("imm8", 8)
    mem = Mem("ext", addr, 8)
    eq_zero = BinOp("eq", mem, value, 1)
    cond = Cond(kind="ltu", a=mem, b=value)
    c_expr = TernOp(
        op="select",
        cond=cond,
        t=Const(1, 1),
        f=Const(0, 1),
        out_size=1,
    )
    return Instr(
        name="CMP_[LMN],N",
        length=4,
        semantics=(
            Fetch("addr24", addr),
            Fetch("u8", value),
            SetFlag("Z", eq_zero),
            SetFlag("C", c_expr),
        ),
    )


def _cmp_imem_mem(name: str, width: int) -> Instr:
    dst = Tmp("dst_off", 8)
    src = Tmp("src_off", 8)
    dst_val = Mem("int", dst, width)
    src_val = Mem("int", src, width)
    diff = BinOp("sub", dst_val, src_val, width)
    eq_zero = BinOp("eq", diff, Const(0, width), 1)
    cond = Cond(kind="ltu", a=dst_val, b=src_val)
    borrow = TernOp(
        op="select",
        cond=cond,
        t=Const(1, 1),
        f=Const(0, 1),
        out_size=1,
    )
    return Instr(
        name=name,
        length=3,
        semantics=(
            Fetch("u8", dst),
            Fetch("u8", src),
            SetFlag("Z", eq_zero),
            SetFlag("C", borrow),
        ),
    )


def cmpw_imem_mem() -> Instr:
    return _cmp_imem_mem("CMPW (m),(n)", 16)


def cmpp_imem_mem() -> Instr:
    return _cmp_imem_mem("CMPP (m),(n)", 24)


def cmp_imem_mem() -> Instr:
    return _cmp_imem_mem("CMP (m),(n)", 8)


def cmp_a_imm() -> Instr:
    value = Tmp("imm8", 8)
    eq_zero = BinOp("eq", Reg("A", 8), value, 1)
    cond = Cond(kind="ltu", a=Reg("A", 8), b=value)
    c_expr = TernOp(
        op="select",
        cond=cond,
        t=Const(1, 1),
        f=Const(0, 1),
        out_size=1,
    )
    return Instr(
        name="CMP_A_N",
        length=2,
        semantics=(
            Fetch("u8", value),
            SetFlag("Z", eq_zero),
            SetFlag("C", c_expr),
        ),
    )


def cmp_imem_reg() -> Instr:
    dst = Tmp("dst_off", 8)
    mem = Mem("int", dst, 8)
    eq_zero = BinOp("eq", mem, Reg("A", 8), 1)
    cond = Cond(kind="ltu", a=mem, b=Reg("A", 8))
    c_expr = TernOp(
        op="select",
        cond=cond,
        t=Const(1, 1),
        f=Const(0, 1),
        out_size=1,
    )
    return Instr(
        name="CMP_(M),A",
        length=3,
        semantics=(
            Fetch("u8", dst),
            SetFlag("Z", eq_zero),
            SetFlag("C", c_expr),
        ),
    )


def mv_imem_const() -> Instr:
    dst = Tmp("dst_off", 8)
    value = Tmp("imm8", 8)
    return Instr(
        name="MV_(N)_N",
        length=3,
        semantics=(
            Fetch("u8", dst),
            Fetch("u8", value),
            Store(Mem("int", dst, 8), value),
        ),
    )


def mv_imem_word_const() -> Instr:
    dst = Tmp("dst_off", 8)
    lo = Tmp("imm8_lo", 8)
    hi = Tmp("imm8_hi", 8)
    hi_addr = BinOp("add", dst, Const(1, 8), 8)
    return Instr(
        name="MVW_(L),MN",
        length=4,
        semantics=(
            Fetch("u8", dst),
            Fetch("u8", lo),
            Fetch("u8", hi),
            Store(Mem("int", dst, 8), lo),
            Store(Mem("int", hi_addr, 8), hi),
        ),
    )


def mv_imem_long_const() -> Instr:
    dst = Tmp("dst_off", 8)
    lo = Tmp("imm8_lo", 8)
    mid = Tmp("imm8_mid", 8)
    hi = Tmp("imm8_hi", 8)
    value = Join24(hi, mid, lo)
    mem = Mem("int", dst, 24)
    return Instr(
        name="MVP_(L),MN",
        length=5,
        semantics=(
            Fetch("u8", dst),
            Fetch("u8", lo),
            Fetch("u8", mid),
            Fetch("u8", hi),
            Store(mem, value),
        ),
    )


def alu_a_imem(name: str, op: str, flags: tuple[str, ...] = ("Z",)) -> Instr:
    dst = Tmp("dst_off", 8)
    mem = Mem("int", dst, 8)
    a_reg = Reg("A", 8)
    result = BinOp(op, a_reg, mem, 8)
    semantics: list[Any] = [Fetch("u8", dst)]
    if "Z" in flags:
        semantics.append(SetFlag("Z", BinOp("eq", result, Const(0, 8), 1)))
    if "C" in flags:
        if op == "add":
            carry_cond = Cond(kind="ltu", a=result, b=a_reg)
        elif op == "sub":
            carry_cond = Cond(kind="ltu", a=a_reg, b=mem)
        else:
            carry_cond = None
        if carry_cond is not None:
            semantics.append(
                SetFlag(
                    "C",
                    TernOp(
                        op="select",
                        cond=carry_cond,
                        t=Const(1, 1),
                        f=Const(0, 1),
                        out_size=1,
                    ),
                )
            )
    semantics.append(SetReg(a_reg, result))
    return Instr(name=name, length=2, semantics=tuple(semantics))


def alu_mem_a(name: str, op: str) -> Instr:
    dst = Tmp("dst_off", 8)
    mem = Mem("int", dst, 8)
    a_reg = Reg("A", 8)
    result = BinOp(op, mem, a_reg, 8)
    zero = BinOp("eq", result, Const(0, 8), 1)
    if op == "add":
        carry_cond = Cond(kind="ltu", a=result, b=mem)
    elif op == "sub":
        carry_cond = Cond(kind="ltu", a=mem, b=a_reg)
    else:
        carry_cond = None
    carry_stmt: list[Any] = []
    if carry_cond is not None:
        carry_stmt.append(
            SetFlag(
                "C",
                TernOp(
                    op="select",
                    cond=carry_cond,
                    t=Const(1, 1),
                    f=Const(0, 1),
                    out_size=1,
                ),
            )
        )
    return Instr(
        name=name,
        length=2,
        semantics=(
            Fetch("u8", dst),
            SetFlag("Z", zero),
            *carry_stmt,
            Store(mem, result),
        ),
    )


def test_emem_const() -> Instr:
    addr = Tmp("addr24", 24)
    value = Tmp("imm8", 8)
    mem = Mem("ext", addr, 8)
    and_expr = BinOp("and", mem, value, 8)
    cmp_zero = BinOp("eq", and_expr, Const(0, 8), 1)
    return Instr(
        name="TEST_[LMN],N",
        length=4,
        semantics=(
            Fetch("addr24", addr),
            Fetch("u8", value),
            SetFlag("Z", cmp_zero),
        ),
    )


def test_imem_reg() -> Instr:
    dst = Tmp("dst_off", 8)
    mem = Mem("int", dst, 8)
    and_expr = BinOp("and", mem, Reg("A", 8), 8)
    eq_zero = BinOp("eq", and_expr, Const(0, 8), 1)
    return Instr(
        name="TEST_(M),A",
        length=2,
        semantics=(
            Fetch("u8", dst),
            SetFlag("Z", eq_zero),
        ),
    )


def add_imem_const() -> Instr:
    dst = Tmp("dst_off", 8)
    value = Tmp("imm8", 8)
    mem = Mem("int", dst, 8)
    sum_expr = BinOp("add", mem, value, 8)
    eq_zero = BinOp("eq", sum_expr, Const(0, 8), 1)
    carry_cond = Cond(kind="ltu", a=sum_expr, b=mem)
    carry = TernOp(
        op="select",
        cond=carry_cond,
        t=Const(1, 1),
        f=Const(0, 1),
        out_size=1,
    )
    return Instr(
        name="ADD_(M),N",
        length=3,
        semantics=(
            Fetch("u8", dst),
            Fetch("u8", value),
            SetFlag("Z", eq_zero),
            SetFlag("C", carry),
            Store(mem, sum_expr),
        ),
    )


def mv_reg_imem(name: str) -> Instr:
    dst = Tmp("dst_off", 8)
    mem = Mem("int", dst, 8)
    src = Reg("A", 8)  # placeholder; actual register injected via builder via Reg argument
    instr = Instr(
        name=name,
        length=2,
        semantics=(
            Fetch("u8", dst),
            Store(mem, src),
        ),
    )
    return instr


def _with_carry(right: Expr) -> Expr:
    return BinOp("add", right, Flag("C"), 8)


def adc_imem_const() -> Instr:
    dst = Tmp("dst_off", 8)
    value = Tmp("imm8", 8)
    mem = Mem("int", dst, 8)
    rhs = _with_carry(value)
    sum_expr = BinOp("add", mem, rhs, 8)
    eq_zero = BinOp("eq", sum_expr, Const(0, 8), 1)
    carry_cond = Cond(kind="ltu", a=sum_expr, b=mem)
    carry = TernOp(
        op="select",
        cond=carry_cond,
        t=Const(1, 1),
        f=Const(0, 1),
        out_size=1,
    )
    return Instr(
        name="ADC_(M),N",
        length=3,
        semantics=(
            Fetch("u8", dst),
            Fetch("u8", value),
            SetFlag("Z", eq_zero),
            SetFlag("C", carry),
            Store(mem, sum_expr),
        ),
    )


def adc_a_imem() -> Instr:
    dst = Tmp("dst_off", 8)
    mem = Mem("int", dst, 8)
    rhs = _with_carry(mem)
    result = BinOp("add", Reg("A", 8), rhs, 8)
    eq_zero = BinOp("eq", result, Const(0, 8), 1)
    carry_cond = Cond(kind="ltu", a=result, b=Reg("A", 8))
    carry = TernOp(
        op="select",
        cond=carry_cond,
        t=Const(1, 1),
        f=Const(0, 1),
        out_size=1,
    )
    return Instr(
        name="ADC_A_(N)",
        length=2,
        semantics=(
            Fetch("u8", dst),
            SetReg(Reg("A", 8), result, flags=("Z",)),
            SetFlag("C", carry),
        ),
    )


def adc_mem_a() -> Instr:
    dst = Tmp("dst_off", 8)
    mem = Mem("int", dst, 8)
    rhs = _with_carry(Reg("A", 8))
    result = BinOp("add", mem, rhs, 8)
    eq_zero = BinOp("eq", result, Const(0, 8), 1)
    carry_cond = Cond(kind="ltu", a=result, b=mem)
    carry = TernOp(
        op="select",
        cond=carry_cond,
        t=Const(1, 1),
        f=Const(0, 1),
        out_size=1,
    )
    return Instr(
        name="ADC_(N),A",
        length=2,
        semantics=(
            Fetch("u8", dst),
            SetFlag("Z", eq_zero),
            SetFlag("C", carry),
            Store(mem, result),
        ),
    )


def sbc_imem_const() -> Instr:
    dst = Tmp("dst_off", 8)
    value = Tmp("imm8", 8)
    mem = Mem("int", dst, 8)
    rhs = _with_carry(value)
    diff = BinOp("sub", mem, rhs, 8)
    eq_zero = BinOp("eq", diff, Const(0, 8), 1)
    borrow_cond = Cond(kind="ltu", a=mem, b=rhs)
    borrow = TernOp(
        op="select",
        cond=borrow_cond,
        t=Const(1, 1),
        f=Const(0, 1),
        out_size=1,
    )
    return Instr(
        name="SBC_(M),N",
        length=3,
        semantics=(
            Fetch("u8", dst),
            Fetch("u8", value),
            SetFlag("Z", eq_zero),
            SetFlag("C", borrow),
            Store(mem, diff),
        ),
    )


def sbc_a_imem() -> Instr:
    dst = Tmp("dst_off", 8)
    mem = Mem("int", dst, 8)
    rhs = _with_carry(mem)
    result = BinOp("sub", Reg("A", 8), rhs, 8)
    eq_zero = BinOp("eq", result, Const(0, 8), 1)
    borrow_cond = Cond(kind="ltu", a=Reg("A", 8), b=rhs)
    borrow = TernOp(
        op="select",
        cond=borrow_cond,
        t=Const(1, 1),
        f=Const(0, 1),
        out_size=1,
    )
    return Instr(
        name="SBC_A_(N)",
        length=2,
        semantics=(
            Fetch("u8", dst),
            SetReg(Reg("A", 8), result, flags=("Z",)),
            SetFlag("C", borrow),
        ),
    )


def sbc_mem_a() -> Instr:
    dst = Tmp("dst_off", 8)
    mem = Mem("int", dst, 8)
    rhs = _with_carry(Reg("A", 8))
    result = BinOp("sub", mem, rhs, 8)
    eq_zero = BinOp("eq", result, Const(0, 8), 1)
    borrow_cond = Cond(kind="ltu", a=mem, b=rhs)
    borrow = TernOp(
        op="select",
        cond=borrow_cond,
        t=Const(1, 1),
        f=Const(0, 1),
        out_size=1,
    )
    return Instr(
        name="SBC_(N),A",
        length=2,
        semantics=(
            Fetch("u8", dst),
            SetFlag("Z", eq_zero),
            SetFlag("C", borrow),
            Store(mem, result),
        ),
    )


def xor_emem_const() -> Instr:
    addr = Tmp("addr24", 24)
    value = Tmp("imm8", 8)
    mem = Mem("ext", addr, 8)
    xor_expr = BinOp("xor", mem, value, 8)
    eq_zero = BinOp("eq", xor_expr, Const(0, 8), 1)
    return Instr(
        name="XOR_[LMN],N",
        length=4,
        semantics=(
            Fetch("addr24", addr),
            Fetch("u8", value),
            Store(mem, xor_expr),
            SetFlag("Z", eq_zero),
        ),
    )


def and_emem_const() -> Instr:
    addr = Tmp("addr24", 24)
    value = Tmp("imm8", 8)
    mem = Mem("ext", addr, 8)
    and_expr = BinOp("and", mem, value, 8)
    eq_zero = BinOp("eq", and_expr, Const(0, 8), 1)
    return Instr(
        name="AND_[LMN],N",
        length=4,
        semantics=(
            Fetch("addr24", addr),
            Fetch("u8", value),
            Store(mem, and_expr),
            SetFlag("Z", eq_zero),
        ),
    )


def or_emem_const() -> Instr:
    addr = Tmp("addr24", 24)
    value = Tmp("imm8", 8)
    mem = Mem("ext", addr, 8)
    or_expr = BinOp("or", mem, value, 8)
    eq_zero = BinOp("eq", or_expr, Const(0, 8), 1)
    return Instr(
        name="OR_[LMN],N",
        length=4,
        semantics=(
            Fetch("addr24", addr),
            Fetch("u8", value),
            Store(mem, or_expr),
            SetFlag("Z", eq_zero),
        ),
    )


def xor_a_imem() -> Instr:
    dst = Tmp("dst_off", 8)
    mem = Mem("int", dst, 8)
    xor_expr = BinOp("xor", Reg("A", 8), mem, 8)
    eq_zero = BinOp("eq", xor_expr, Const(0, 8), 1)
    return Instr(
        name="XOR_A_(N)",
        length=2,
        semantics=(
            Fetch("u8", dst),
            SetReg(Reg("A", 8), xor_expr, flags=("Z",)),
        ),
    )


def xor_mem_mem() -> Instr:
    dst = Tmp("dst_off", 8)
    src = Tmp("src_off", 8)
    dst_mem = Mem("int", dst, 8)
    src_mem = Mem("int", src, 8)
    xor_expr = BinOp("xor", dst_mem, src_mem, 8)
    eq_zero = BinOp("eq", xor_expr, Const(0, 8), 1)
    return Instr(
        name="XOR_(M),(N)",
        length=3,
        semantics=(
            Fetch("u8", dst),
            Fetch("u8", src),
            Store(dst_mem, xor_expr),
            SetFlag("Z", eq_zero),
        ),
    )
    return Instr(
        name="SBC_(N),A",
        length=2,
        semantics=(
            Fetch("u8", dst),
            Store(mem, result),
            SetFlag("Z", eq_zero),
            SetFlag("C", borrow),
        ),
    )


def sub_imem_const() -> Instr:
    dst = Tmp("dst_off", 8)
    value = Tmp("imm8", 8)
    mem = Mem("int", dst, 8)
    diff = BinOp("sub", mem, value, 8)
    eq_zero = BinOp("eq", diff, Const(0, 8), 1)
    borrow_cond = Cond(kind="ltu", a=mem, b=value)
    borrow = TernOp(
        op="select",
        cond=borrow_cond,
        t=Const(1, 1),
        f=Const(0, 1),
        out_size=1,
    )
    return Instr(
        name="SUB_(M),N",
        length=3,
        semantics=(
            Fetch("u8", dst),
            Fetch("u8", value),
            SetFlag("Z", eq_zero),
            SetFlag("C", borrow),
            Store(mem, diff),
        ),
    )


def rc_instr() -> Instr:
    return Instr(
        name="RC",
        length=1,
        semantics=(SetFlag("C", Const(0, 1)),),
    )


def sc_instr() -> Instr:
    return Instr(
        name="SC",
        length=1,
        semantics=(SetFlag("C", Const(1, 1)),),
    )


def tcl_instr() -> Instr:
    return Instr(
        name="TCL",
        length=1,
        semantics=(Effect("tcl", ()),),
    )


def swap_a_instr() -> Instr:
    val = Reg("A", 8)
    high = BinOp("and", val, Const(0xF0, 8), 8)
    low = BinOp("and", val, Const(0x0F, 8), 8)
    high_shifted = BinOp("shr", high, Const(4, 8), 8)
    low_shifted = BinOp("shl", low, Const(4, 8), 8)
    swapped = BinOp("or", high_shifted, low_shifted, 8)
    return Instr(
        name="SWAP A",
        length=1,
        semantics=(SetReg(Reg("A", 8), swapped),),
    )


def _rot_imem_instr(name: str, op: str) -> Instr:
    dst = Tmp("dst_off", 8)
    mem = Mem("int", dst, 8)
    result = BinOp(op, mem, Const(1, 8), 8)
    return Instr(
        name=name,
        length=2,
        semantics=(
            Fetch("u8", dst),
            Store(mem, result),
        ),
    )


def rotate_a_instr(name: str, op: str) -> Instr:
    val = Reg("A", 8)
    result = BinOp(op, val, Const(1, 8), 8)
    return Instr(
        name=name,
        length=1,
        semantics=(SetReg(Reg("A", 8), result),),
    )


def shift_a_instr(name: str, op: str) -> Instr:
    # shift only affects A.
    return rotate_a_instr(name, op)


def mv_a_a_instr() -> Instr:
    return Instr(
        name="MV A,A",
        length=1,
        semantics=(SetReg(Reg("A", 8), Reg("A", 8)),),
    )


def imem_logic_instr(
    name: str, op: str, flags: tuple[str, ...] = ("Z",)
) -> Instr:
    dst = Tmp("dst_off", 8)
    value = Tmp("imm8", 8)
    mem = Mem("int", dst, 8)
    binop = BinOp(op, Mem("int", dst, 8), value, 8)
    cmp_zero = BinOp("eq", binop, Const(0, 8), 1)
    semantics: list[Any] = [
        Fetch("u8", dst),
        Fetch("u8", value),
        Store(mem, binop),
    ]
    if "Z" in flags:
        semantics.append(SetFlag("Z", cmp_zero))
    return Instr(name=name, length=3, semantics=tuple(semantics))


def imem_logic_mem(
    name: str, op: str, flags: tuple[str, ...] = ("Z",)
) -> Instr:
    dst = Tmp("dst_off", 8)
    src = Tmp("src_off", 8)
    dst_mem = Mem("int", dst, 8)
    src_mem = Mem("int", src, 8)
    result = BinOp(op, dst_mem, src_mem, 8)
    eq_zero = BinOp("eq", result, Const(0, 8), 1)
    semantics: list[Any] = [
        Fetch("u8", dst),
        Fetch("u8", src),
        Store(dst_mem, result),
    ]
    if "Z" in flags:
        semantics.append(SetFlag("Z", eq_zero))
    return Instr(name=name, length=3, semantics=tuple(semantics))


def imem_logic_reg(name: str, op: str, flags: tuple[str, ...] = ("Z",)) -> Instr:
    dst = Tmp("dst_off", 8)
    mem = Mem("int", dst, 8)
    result = BinOp(op, Reg("A", 8), mem, 8)
    instr = Instr(
        name=name,
        length=2,
        semantics=(
            Fetch("u8", dst),
            SetReg(Reg("A", 8), result, flags=flags),
        ),
    )
    return instr


def imem_logic_store(
    name: str, op: str, flags: tuple[str, ...] = ("Z",)
) -> Instr:
    dst = Tmp("dst_off", 8)
    mem = Mem("int", dst, 8)
    src = Reg("A", 8)
    result = BinOp(op, mem, src, 8)
    semantics: list[Any] = [
        Fetch("u8", dst),
        Store(mem, result),
    ]
    if flags:
        flag_exprs: dict[str, Expr] = {
            "Z": BinOp("eq", result, Const(0, 8), 1),
        }
        for flag in flags:
            if flag not in flag_exprs:
                raise ValueError(f"Unsupported flag '{flag}' for {name}")
            semantics.append(SetFlag(flag, flag_exprs[flag]))
    return Instr(
        name=name,
        length=2,
        semantics=tuple(semantics),
    )


def inc_dec_imem(name: str, op: str) -> Instr:
    dst = Tmp("dst_off", 8)
    mem = Mem("int", dst, 8)
    one = Const(1, 8)
    result = BinOp(op, mem, one, 8)
    cmp_zero = BinOp("eq", Mem("int", dst, 8), Const(0, 8), 1)
    return Instr(
        name=name,
        length=2,
        semantics=(
            Fetch("u8", dst),
            Store(mem, result),
            SetFlag("Z", cmp_zero),
        ),
    )


def mvl_imem_imem(name: str, step: int) -> Instr:
    dst = Tmp("dst_off", 8)
    src = Tmp("src_off", 8)
    stride = Const(step & 0xFF, 8)
    return Instr(
        name=name,
        length=3,
        semantics=(
            Fetch("u8", dst),
            Fetch("u8", src),
            Effect(
                "loop_move",
                (
                    Reg("I", 16),
                    LoopIntPtr(dst),
                    LoopIntPtr(src),
                    stride,
                    Const(8, 8),
                ),
            ),
        ),
    )


def loop_carry_instr(name: str, effect_kind: str, *, src_is_mem: bool) -> Instr:
    dst = Tmp("dst_off", 8)
    semantics: list[object] = [Fetch("u8", dst)]
    args: list[object] = [Reg("I", 16), LoopIntPtr(dst)]
    if src_is_mem:
        src = Tmp("src_off", 8)
        semantics.append(Fetch("u8", src))
        args.append(LoopIntPtr(src))
    else:
        args.append(Reg("A", 8))
    args.extend([Flag("C"), Const(8, 8)])
    semantics.append(Effect(effect_kind, tuple(args)))  # type: ignore[arg-type]
    length = 3 if src_is_mem else 2
    return Instr(name=name, length=length, semantics=tuple(semantics))


def loop_bcd_instr(
    name: str,
    effect_kind: str,
    *,
    src_is_mem: bool,
    direction: int,
    clear_carry: bool,
) -> Instr:
    dst = Tmp("dst_off", 8)
    semantics: list[object] = [Fetch("u8", dst)]
    args: list[object] = [Reg("I", 16), LoopIntPtr(dst)]
    if src_is_mem:
        src = Tmp("src_off", 8)
        semantics.append(Fetch("u8", src))
        args.append(LoopIntPtr(src))
    else:
        args.append(Reg("A", 8))
    args.extend(
        [
            Flag("C"),
            Const(8, 8),
            Const(direction & 0xFF, 8),
            Const(1 if clear_carry else 0, 1),
        ]
    )
    semantics.append(Effect(effect_kind, tuple(args)))  # type: ignore[arg-type]
    length = 3 if src_is_mem else 2
    return Instr(name=name, length=length, semantics=tuple(semantics))


def decimal_shift_instr(name: str, is_left: bool) -> Instr:
    base = Tmp("dst_off", 8)
    direction = Const((-1 if is_left else 1) & 0xFF, 8)
    flag = Const(1 if is_left else 0, 1)
    return Instr(
        name=name,
        length=2,
        semantics=(
            Fetch("u8", base),
            Effect(
                "decimal_shift",
                (
                    Reg("I", 16),
                    LoopIntPtr(base),
                    direction,
                    flag,
                ),
            ),
        ),
    )


def pmdf_immediate() -> Instr:
    addr = Tmp("dst_off", 8)
    value = Tmp("imm8", 8)
    return Instr(
        name="PMDF (m),n",
        length=3,
        semantics=(
            Fetch("u8", addr),
            Fetch("u8", value),
            Effect("pmdf", (LoopIntPtr(addr), value)),
        ),
    )


def pmdf_reg() -> Instr:
    addr = Tmp("dst_off", 8)
    return Instr(
        name="PMDF (m),A",
        length=2,
        semantics=(
            Fetch("u8", addr),
            Effect("pmdf", (LoopIntPtr(addr), Reg("A", 8))),
        ),
    )


def halt_instr() -> Instr:
    return Instr(name="HALT", length=1, semantics=(Effect("halt", ()),))


def off_instr() -> Instr:
    return Instr(name="OFF", length=1, semantics=(Effect("off", ()),))


def wait_instr() -> Instr:
    return Instr(name="WAIT", length=1, semantics=(Effect("wait", ()),))


def reset_instr() -> Instr:
    return Instr(name="RESET", length=1, semantics=(Effect("reset", ()),))


def ir_instr() -> Instr:
    return Instr(name="IR", length=1, semantics=(Effect("interrupt_enter", ()),))


def inc_dec_reg(name: str, reg_name: str, size: int, op: str) -> Instr:
    reg = Reg(reg_name, size)
    const_one = Const(1, size)
    expr = BinOp(op, reg, const_one, size)
    return Instr(
        name=name,
        length=2,
        semantics=(SetReg(reg, expr, flags=("Z",)),),
    )


def jrz_rel() -> Instr:
    return jr_rel("JRZ_REL", "Z", True)


def inc_a() -> Instr:
    return inc_dec_reg("INC_A", "A", 8, "add")


def _flag_byte_expr() -> BinOp:
    z_shift = BinOp("shl", Flag("Z"), Const(1, 8), 8)
    return BinOp("or", Flag("C"), z_shift, 8)


def pushs_f() -> Instr:
    return Instr(
        name="PUSHS F",
        length=1,
        semantics=(
            Effect(
                "push_bytes",
                (
                    Reg("S", 24),
                    _flag_byte_expr(),
                    Const(8, 8),
                    Const(0, 1),
                ),
            ),
        ),
    )


def pops_f() -> Instr:
    return Instr(
        name="POPS F",
        length=1,
        semantics=(
            Effect(
                "pop_bytes",
                (
                    Reg("S", 24),
                    Reg("F", 8),
                    Const(8, 8),
                    Const(0, 1),
                ),
            ),
        ),
    )


def pushu_reg(name: str, reg_name: str, bits: int) -> Instr:
    value: Expr
    if reg_name == "F":
        value = _flag_byte_expr()
    else:
        value = Reg(reg_name, bits)
    return Instr(
        name=name,
        length=1,
        semantics=(
            Effect(
                "push_bytes",
                (
                    Reg("U", 24),
                    value,
                    Const(bits, 8),
                    Const(1, 1),
                ),
            ),
        ),
    )


def popu_reg(name: str, reg_name: str, bits: int) -> Instr:
    dest = Reg(reg_name, bits)
    return Instr(
        name=name,
        length=1,
        semantics=(
            Effect(
                "pop_bytes",
                (
                    Reg("U", 24),
                    dest,
                    Const(bits, 8),
                    Const(1, 1),
                ),
            ),
        ),
    )


def pushu_imr() -> Instr:
    imr_mem = _imr_mem()
    masked = BinOp("and", _imr_mem(), Const(0x7F, 8), 8)
    return Instr(
        name="PUSHU IMR",
        length=1,
        semantics=(
            Effect(
                "push_bytes",
                (
                    Reg("U", 24),
                    imr_mem,
                    Const(8, 8),
                    Const(1, 1),
                ),
            ),
            Store(_imr_mem(), masked),
        ),
    )


def popu_imr() -> Instr:
    return Instr(
        name="POPU IMR",
        length=1,
        semantics=(
            Effect(
                "pop_bytes",
                (
                    Reg("U", 24),
                    _imr_mem(),
                    Const(8, 8),
                    Const(1, 1),
                ),
            ),
        ),
    )
