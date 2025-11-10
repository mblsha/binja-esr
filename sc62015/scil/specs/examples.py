from __future__ import annotations

from ..ast import (
    BinOp,
    Const,
    Fetch,
    Goto,
    If,
    Instr,
    Join24,
    Mem,
    PcRel,
    Reg,
    SetReg,
    Tmp,
    UnOp,
    Cond,
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


def jrz_rel() -> Instr:
    disp = Tmp("disp8", 8)
    taken = PcRel(base_advance=2, disp=UnOp("sext", disp, 20), out_size=20)
    return Instr(
        name="JRZ_REL",
        length=2,
        semantics=(
            Fetch("disp8", disp),
            If(
                cond=Cond(kind="flag", flag="Z"),
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


def inc_a() -> Instr:
    value = BinOp("add", Reg("A", 8), Const(1, 8), 8)
    return Instr(
        name="INC_A",
        length=2,
        semantics=(
            SetReg(Reg("A", 8), value, flags=("Z",)),
        ),
    )
