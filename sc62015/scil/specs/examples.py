from __future__ import annotations

from ..ast import (
    BinOp,
    Cond,
    Const,
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
    UnOp,
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
