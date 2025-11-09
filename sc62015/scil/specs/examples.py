from __future__ import annotations

from ..ast import (
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
    fallthrough = PcRel(base_advance=2, out_size=20)
    return Instr(
        name="JRZ_REL",
        length=2,
        semantics=(
            Fetch("disp8", disp),
            If(
                cond=Cond(kind="flag", flag="Z"),
                then_ops=(Goto(taken),),
                else_ops=(Goto(fallthrough),),
            ),
        ),
    )


def mv_a_abs_ext() -> Instr:
    b0 = Tmp("addr_lo", 8)
    b1 = Tmp("addr_mid", 8)
    b2 = Tmp("addr_hi", 8)
    addr = Join24(hi=b2, mid=b1, lo=b0)
    return Instr(
        name="MV_A_ABS_EXT",
        length=4,
        semantics=(
            Fetch("u8", b0),
            Fetch("u8", b1),
            Fetch("u8", b2),
            SetReg(Reg("A", 8), Mem("ext", addr, 8)),
        ),
    )
