from __future__ import annotations

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
    Mem,
    PcRel,
    Reg,
    Store,
    SetFlag,
    SetReg,
    Tmp,
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
