from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

from ..decoding.bind import (
    Addr16Page,
    Addr24,
    DecodedInstr,
    Disp8,
    ExtRegPtr,
    Imm8,
    PreLatch,
    RegSel,
    ImemPtr,
)
from . import ast
from .specs import examples


Binder = Dict[str, ast.Const]


@dataclass(frozen=True)
class BuildResult:
    instr: ast.Instr
    binder: Binder
    pre_applied: Optional[PreLatch] = None


_REG_BITS = {
    "r1": 8,
    "r2": 16,
    "r3": 24,
}

def _const(value: int, bits: int) -> ast.Const:
    mask = (1 << bits) - 1
    return ast.Const(value & mask, bits)


def _disp_const(disp: Optional[Disp8]) -> Optional[ast.Const]:
    if disp is None:
        return None
    return _const(disp.value, 8)


def _imem_ptr_value(ptr: ImemPtr) -> ast.Expr:
    base_const = _const(ptr.base.value, 8)
    location = base_const
    pointer = ast.Mem("int", location, 24)
    if ptr.mode == "simple":
        return pointer
    disp = _disp_const(ptr.disp)
    if disp is None:
        raise ValueError("Pointer displacement missing")
    signed = ast.UnOp("sext", disp, 24)
    if ptr.mode == "pos":
        return ast.BinOp("add", pointer, signed, 24)
    return ast.BinOp("sub", pointer, signed, 24)


def _flag_cond(flag: str, expect_one: bool) -> ast.Cond:
    return ast.Cond(
        kind="eq",
        a=ast.Flag(flag),
        b=_const(1 if expect_one else 0, 1),
    )


def _with_pre(instr: ast.Instr, binder: Binder, decoded: DecodedInstr) -> BuildResult:
    return BuildResult(instr=instr, binder=binder, pre_applied=decoded.pre_applied)


def _imm8(decoded: DecodedInstr, key: str = "imm8") -> Dict[str, ast.Const]:
    imm = decoded.binds["n"]
    assert isinstance(imm, Imm8)
    return {key: _const(imm.value, 8)}


def _mv_a_n(decoded: DecodedInstr) -> BuildResult:
    return _with_pre(examples.mv_a_imm(), _imm8(decoded), decoded)


def _alu(decoded: DecodedInstr, op: str, include_carry: bool, flags: tuple[str, ...]) -> BuildResult:
    spec = examples.alu_a_imm(decoded.mnemonic, op, include_carry=include_carry, flags=flags)
    return _with_pre(spec, _imm8(decoded), decoded)


def _test(decoded: DecodedInstr) -> BuildResult:
    return _with_pre(examples.test_a_imm(), _imm8(decoded), decoded)


def _jr(decoded: DecodedInstr, flag: str, expect_one: bool) -> BuildResult:
    offset = decoded.binds["disp"]
    direction = decoded.binds.get("dir", 1)
    assert isinstance(offset, Imm8)
    displacement = offset.value if direction > 0 else -offset.value
    target = ast.PcRel(
        base_advance=decoded.length,
        disp=_const(displacement, 20),
        out_size=20,
    )
    instr = ast.Instr(
        name=decoded.mnemonic,
        length=decoded.length,
        semantics=(
            ast.If(
                cond=_flag_cond(flag, expect_one),
                then_ops=(ast.Goto(target),),
            ),
        ),
    )
    raw = offset.value if direction > 0 else (-offset.value & 0xFF)
    binder = {"disp8": _const(raw, 8)}
    return _with_pre(instr, binder, decoded)


def _jp(decoded: DecodedInstr) -> BuildResult:
    spec = examples.jp_paged()
    addr = decoded.binds["addr16_page"]
    assert isinstance(addr, Addr16Page)
    binder = {
        "addr16": _const(addr.offs16.u16, 16),
        "page_hi": _const(addr.page20, 20),
    }
    return _with_pre(spec, binder, decoded)


def _jp_cond(decoded: DecodedInstr, flag: str, expect_one: bool) -> BuildResult:
    spec = examples.jp_cond(decoded.mnemonic, _flag_cond(flag, expect_one))
    addr = decoded.binds["addr16_page"]
    assert isinstance(addr, Addr16Page)
    binder = {
        "addr16": _const(addr.offs16.u16, 16),
        "page_hi": _const(addr.page20, 20),
    }
    return _with_pre(spec, binder, decoded)


def _mv_a_abs(decoded: DecodedInstr) -> BuildResult:
    spec = examples.mv_a_abs_ext()
    addr = decoded.binds["addr24"]
    assert isinstance(addr, Addr24)
    binder = {"addr_ptr": _const(addr.v.u24, 24)}
    return _with_pre(spec, binder, decoded)


def _call_near(decoded: DecodedInstr) -> BuildResult:
    spec = examples.call_near()
    addr = decoded.binds["addr16_page"]
    assert isinstance(addr, Addr16Page)
    binder = {
        "call_addr16": _const(addr.offs16.u16, 16),
        "call_page_hi": _const(addr.page20, 20),
    }
    return _with_pre(spec, binder, decoded)


def _call_far(decoded: DecodedInstr) -> BuildResult:
    spec = examples.call_far()
    addr = decoded.binds["addr24"]
    assert isinstance(addr, Addr24)
    binder = {"call_addr24": _const(addr.v.u24, 24)}
    return _with_pre(spec, binder, decoded)


def _ret_near(decoded: DecodedInstr) -> BuildResult:
    return _with_pre(examples.ret_near(), {}, decoded)


def _ret_far(decoded: DecodedInstr) -> BuildResult:
    return _with_pre(examples.ret_far(), {}, decoded)


def _reti(decoded: DecodedInstr) -> BuildResult:
    return _with_pre(examples.reti(), {}, decoded)


def _mv_ext_store(decoded: DecodedInstr) -> BuildResult:
    spec = examples.mv_ext_store()
    addr = decoded.binds["addr24"]
    assert isinstance(addr, Addr24)
    binder = {"addr_ptr": _const(addr.v.u24, 24)}
    return _with_pre(spec, binder, decoded)


def _jp_r3(decoded: DecodedInstr) -> BuildResult:
    reg_sel = decoded.binds["reg"]
    assert isinstance(reg_sel, RegSel)
    target = ast.Reg(reg_sel.name, 24)
    instr = ast.Instr(
        name=decoded.mnemonic,
        length=decoded.length,
        semantics=(ast.Goto(target),),
    )
    return _with_pre(instr, {}, decoded)


def _jp_imem_ptr(decoded: DecodedInstr) -> BuildResult:
    base = decoded.binds["n"]
    assert isinstance(base, Imm8)
    ptr = ImemPtr(base=base, mode="simple", disp=None)
    addr_expr = _imem_ptr_value(ptr)
    instr = ast.Instr(
        name=decoded.mnemonic,
        length=decoded.length,
        semantics=(ast.Goto(addr_expr),),
    )
    return _with_pre(instr, {}, decoded)


def _ext_reg_load(decoded: DecodedInstr) -> BuildResult:
    dst = decoded.binds["dst"]
    ptr = decoded.binds["ptr"]
    assert isinstance(dst, RegSel)
    assert isinstance(ptr, ExtRegPtr)
    dst_bits = _REG_BITS.get(dst.size_group)
    if dst_bits is None:
        raise ValueError(f"Unsupported dest group {dst.size_group}")
    semantics = (
        ast.ExtRegLoad(
            dst=ast.Reg(dst.name, dst_bits),
            ptr=ast.Reg(ptr.ptr.name, 24),
            mode=ptr.mode,
            disp=_disp_const(ptr.disp),
        ),
    )
    instr = ast.Instr(name=decoded.mnemonic, length=decoded.length, semantics=semantics)
    return BuildResult(instr, {}, decoded.pre_applied)


def _ext_reg_store(decoded: DecodedInstr) -> BuildResult:
    src = decoded.binds["src"]
    ptr = decoded.binds["ptr"]
    assert isinstance(src, RegSel)
    assert isinstance(ptr, ExtRegPtr)
    src_bits = _REG_BITS.get(src.size_group)
    if src_bits is None:
        raise ValueError(f"Unsupported source group {src.size_group}")
    semantics = (
        ast.ExtRegStore(
            src=ast.Reg(src.name, src_bits),
            ptr=ast.Reg(ptr.ptr.name, 24),
            mode=ptr.mode,
            disp=_disp_const(ptr.disp),
        ),
    )
    instr = ast.Instr(name=decoded.mnemonic, length=decoded.length, semantics=semantics)
    return BuildResult(instr, {}, decoded.pre_applied)


def _ext_ptr_load(decoded: DecodedInstr) -> BuildResult:
    dst = decoded.binds["dst"]
    ptr = decoded.binds["ptr"]
    assert isinstance(dst, RegSel)
    assert isinstance(ptr, ImemPtr)
    bits = _REG_BITS.get(dst.size_group)
    if bits is None:
        raise ValueError(f"Unsupported destination group {dst.size_group}")
    addr_expr = _imem_ptr_value(ptr)
    semantics = (
        ast.SetReg(
            reg=ast.Reg(dst.name, bits),
            value=ast.Mem("ext", addr_expr, bits),
        ),
    )
    instr = ast.Instr(name=decoded.mnemonic, length=decoded.length, semantics=semantics)
    return BuildResult(instr, {}, decoded.pre_applied)


def _ext_ptr_store(decoded: DecodedInstr) -> BuildResult:
    src = decoded.binds["src"]
    ptr = decoded.binds["ptr"]
    assert isinstance(src, RegSel)
    assert isinstance(ptr, ImemPtr)
    bits = _REG_BITS.get(src.size_group)
    if bits is None:
        raise ValueError(f"Unsupported source group {src.size_group}")
    addr_expr = _imem_ptr_value(ptr)
    semantics = (
        ast.Store(
            dst=ast.Mem("ext", addr_expr, bits),
            value=ast.Reg(src.name, bits),
        ),
    )
    instr = ast.Instr(name=decoded.mnemonic, length=decoded.length, semantics=semantics)
    return BuildResult(instr, {}, decoded.pre_applied)


def _int_mem_operand(imm: Imm8, width: int) -> ast.Mem:
    return ast.Mem("int", _const(imm.value, 8), width)


def _mv_imem_from_ext(decoded: DecodedInstr) -> BuildResult:
    ptr = decoded.binds["ptr"]
    imem = decoded.binds["imem"]
    width = decoded.binds["width"]
    assert isinstance(ptr, ExtRegPtr)
    assert isinstance(imem, Imm8)
    dst_mem = _int_mem_operand(imem, width)
    stmt = ast.ExtRegToIntMem(
        ptr=ast.Reg(ptr.ptr.name, 24),
        mode=ptr.mode,
        disp=_disp_const(ptr.disp),
        dst=dst_mem,
    )
    instr = ast.Instr(name=decoded.mnemonic, length=decoded.length, semantics=(stmt,))
    return BuildResult(instr, {}, decoded.pre_applied)


def _mv_ext_from_imem(decoded: DecodedInstr) -> BuildResult:
    ptr = decoded.binds["ptr"]
    imem = decoded.binds["imem"]
    width = decoded.binds["width"]
    assert isinstance(ptr, ExtRegPtr)
    assert isinstance(imem, Imm8)
    src_mem = _int_mem_operand(imem, width)
    stmt = ast.IntMemToExtReg(
        src=src_mem,
        ptr=ast.Reg(ptr.ptr.name, 24),
        mode=ptr.mode,
        disp=_disp_const(ptr.disp),
    )
    instr = ast.Instr(name=decoded.mnemonic, length=decoded.length, semantics=(stmt,))
    return BuildResult(instr, {}, decoded.pre_applied)


def _mv_imem_load(decoded: DecodedInstr) -> BuildResult:
    spec = examples.mv_a_imem()
    imm = decoded.binds["n"]
    assert isinstance(imm, Imm8)
    binder = {"imem_off": _const(imm.value, 8)}
    return _with_pre(spec, binder, decoded)


def _imem_move(decoded: DecodedInstr) -> BuildResult:
    dst = decoded.binds["dst"]
    src = decoded.binds["src"]
    width = decoded.binds["width"]
    assert isinstance(dst, Imm8)
    assert isinstance(src, Imm8)
    dst_mem = ast.Mem("int", _const(dst.value, 8), width)
    src_mem = ast.Mem("int", _const(src.value, 8), width)
    instr = ast.Instr(
        name=decoded.mnemonic,
        length=decoded.length,
        semantics=(ast.Store(dst=dst_mem, value=src_mem),),
    )
    return _with_pre(instr, {}, decoded)


def _mvl_imem(decoded: DecodedInstr) -> BuildResult:
    dst = decoded.binds["dst"]
    src = decoded.binds["src"]
    step = decoded.binds["step"]
    assert isinstance(dst, Imm8)
    assert isinstance(src, Imm8)
    assert isinstance(step, int)
    spec = examples.mvl_imem_imem(decoded.mnemonic, step)
    binder = {
        "dst_off": _const(dst.value, 8),
        "src_off": _const(src.value, 8),
    }
    return _with_pre(spec, binder, decoded)


def _loop_carry(decoded: DecodedInstr, effect_kind: str, *, src_is_mem: bool) -> BuildResult:
    dst = decoded.binds["dst"]
    assert isinstance(dst, Imm8)
    binder = {"dst_off": _const(dst.value, 8)}
    if src_is_mem:
        src = decoded.binds["src"]
        assert isinstance(src, Imm8)
        binder["src_off"] = _const(src.value, 8)
    spec = examples.loop_carry_instr(decoded.mnemonic, effect_kind, src_is_mem=src_is_mem)
    return _with_pre(spec, binder, decoded)


def _loop_bcd(
    decoded: DecodedInstr,
    effect_kind: str,
    *,
    src_is_mem: bool,
    clear_carry: bool,
) -> BuildResult:
    dst = decoded.binds["dst"]
    assert isinstance(dst, Imm8)
    binder = {"dst_off": _const(dst.value, 8)}
    if src_is_mem:
        src = decoded.binds["src"]
        assert isinstance(src, Imm8)
        binder["src_off"] = _const(src.value, 8)
    spec = examples.loop_bcd_instr(
        decoded.mnemonic,
        effect_kind,
        src_is_mem=src_is_mem,
        direction=-1,
        clear_carry=clear_carry,
    )
    return _with_pre(spec, binder, decoded)


def _imem_swap(decoded: DecodedInstr) -> BuildResult:
    left = decoded.binds["left"]
    right = decoded.binds["right"]
    width = decoded.binds["width"]
    assert isinstance(left, Imm8)
    assert isinstance(right, Imm8)
    semantics = (
        ast.IntMemSwap(
            left=_const(left.value, 8),
            right=_const(right.value, 8),
            width=width,
        ),
    )
    instr = ast.Instr(
        name=decoded.mnemonic,
        length=decoded.length,
        semantics=semantics,
    )
    return _with_pre(instr, {}, decoded)


def _mv_imem_store(decoded: DecodedInstr) -> BuildResult:
    spec = examples.mv_imem_a()
    imm = decoded.binds["n"]
    assert isinstance(imm, Imm8)
    binder = {"imem_off": _const(imm.value, 8)}
    return _with_pre(spec, binder, decoded)


def _inc_dec(decoded: DecodedInstr, op: str) -> BuildResult:
    reg_sel = decoded.binds["reg"]
    assert isinstance(reg_sel, RegSel)
    bits = _REG_BITS.get(reg_sel.size_group)
    if bits is None:
        raise ValueError(f"{decoded.mnemonic} unsupported for {reg_sel.size_group}")
    spec = examples.inc_dec_reg(decoded.mnemonic, reg_sel.name, bits, op)
    return _with_pre(spec, {}, decoded)


def _pushu(decoded: DecodedInstr) -> BuildResult:
    reg_sel = decoded.binds["reg"]
    assert isinstance(reg_sel, RegSel)
    bits = _REG_BITS.get(reg_sel.size_group, 8)
    spec = examples.pushu_reg(decoded.mnemonic, reg_sel.name, bits)
    return _with_pre(spec, {}, decoded)


def _popu(decoded: DecodedInstr) -> BuildResult:
    reg_sel = decoded.binds["reg"]
    assert isinstance(reg_sel, RegSel)
    bits = _REG_BITS.get(reg_sel.size_group, 8)
    spec = examples.popu_reg(decoded.mnemonic, reg_sel.name, bits)
    return _with_pre(spec, {}, decoded)


def _pushu_imr(decoded: DecodedInstr) -> BuildResult:
    return _with_pre(examples.pushu_imr(), {}, decoded)


def _popu_imr(decoded: DecodedInstr) -> BuildResult:
    return _with_pre(examples.popu_imr(), {}, decoded)


def _pushs(decoded: DecodedInstr) -> BuildResult:
    return _with_pre(examples.pushs_f(), {}, decoded)


def _pops(decoded: DecodedInstr) -> BuildResult:
    return _with_pre(examples.pops_f(), {}, decoded)


BUILDERS: Dict[str, Callable[[DecodedInstr], BuildResult]] = {
    "MV A,n": _mv_a_n,
    "ADD A,n": lambda di: _alu(di, "add", False, ("C", "Z")),
    "SUB A,n": lambda di: _alu(di, "sub", False, ("C", "Z")),
    "ADC A,n": lambda di: _alu(di, "add", True, ("C", "Z")),
    "SBC A,n": lambda di: _alu(di, "sub", True, ("C", "Z")),
    "AND A,n": lambda di: _alu(di, "and", False, ("Z",)),
    "OR A,n": lambda di: _alu(di, "or", False, ("Z",)),
    "XOR A,n": lambda di: _alu(di, "xor", False, ("Z",)),
    "TEST A,n": _test,
    "JRZ ±n": lambda di: _jr(di, "Z", True),
    "JRNZ ±n": lambda di: _jr(di, "Z", False),
    "JRC ±n": lambda di: _jr(di, "C", True),
    "JRNC ±n": lambda di: _jr(di, "C", False),
    "JP mn": _jp,
    "JPZ mn": lambda di: _jp_cond(di, "Z", True),
    "JPNZ mn": lambda di: _jp_cond(di, "Z", False),
    "JPC mn": lambda di: _jp_cond(di, "C", True),
    "JPNC mn": lambda di: _jp_cond(di, "C", False),
    "JP r3": _jp_r3,
    "JP (n)": _jp_imem_ptr,
    "MV A,[lmn]": _mv_a_abs,
    "MV [lmn],A": _mv_ext_store,
    "MV A,(n)": _mv_imem_load,
    "MV (n),A": _mv_imem_store,
    "CALL mn": _call_near,
    "CALLF lmn": _call_far,
    "RET": _ret_near,
    "RETF": _ret_far,
    "RETI": _reti,
    "MV r,[r3]": _ext_reg_load,
    "MV [r3],r": _ext_reg_store,
    "MV r,[(n)]": _ext_ptr_load,
    "MV [(n)],r": _ext_ptr_store,
    "INC r": lambda di: _inc_dec(di, "add"),
    "DEC r": lambda di: _inc_dec(di, "sub"),
    "MV (n),[r3]": _mv_imem_from_ext,
    "MVW (n),[r3]": _mv_imem_from_ext,
    "MVP (n),[r3]": _mv_imem_from_ext,
    "MV [r3],(n)": _mv_ext_from_imem,
    "MVW [r3],(n)": _mv_ext_from_imem,
    "MVP [r3],(n)": _mv_ext_from_imem,
    "MV (m),(n)": _imem_move,
    "MVW (m),(n)": _imem_move,
    "MVP (m),(n)": _imem_move,
    "MVL (m),(n)": _mvl_imem,
    "MVLD (m),(n)": _mvl_imem,
    "ADCL (m),(n)": lambda di: _loop_carry(di, "loop_add_carry", src_is_mem=True),
    "ADCL (m),A": lambda di: _loop_carry(di, "loop_add_carry", src_is_mem=False),
    "SBCL (m),(n)": lambda di: _loop_carry(di, "loop_sub_borrow", src_is_mem=True),
    "SBCL (m),A": lambda di: _loop_carry(di, "loop_sub_borrow", src_is_mem=False),
    "DADL (m),(n)": lambda di: _loop_bcd(di, "loop_bcd_add", src_is_mem=True, clear_carry=True),
    "DADL (m),A": lambda di: _loop_bcd(di, "loop_bcd_add", src_is_mem=False, clear_carry=True),
    "DSBL (m),(n)": lambda di: _loop_bcd(di, "loop_bcd_sub", src_is_mem=True, clear_carry=False),
    "DSBL (m),A": lambda di: _loop_bcd(di, "loop_bcd_sub", src_is_mem=False, clear_carry=False),
    "EX (m),(n)": _imem_swap,
    "EXW (m),(n)": _imem_swap,
    "EXP (m),(n)": _imem_swap,
    "PUSHS F": _pushs,
    "POPS F": _pops,
    "PUSHU A": _pushu,
    "PUSHU IL": _pushu,
    "PUSHU BA": _pushu,
    "PUSHU I": _pushu,
    "PUSHU X": _pushu,
    "PUSHU Y": _pushu,
    "PUSHU F": _pushu,
    "PUSHU IMR": _pushu_imr,
    "POPU A": _popu,
    "POPU IL": _popu,
    "POPU BA": _popu,
    "POPU I": _popu,
    "POPU X": _popu,
    "POPU Y": _popu,
    "POPU F": _popu,
    "POPU IMR": _popu_imr,
}


def build(decoded: DecodedInstr) -> BuildResult:
    try:
        builder = BUILDERS[decoded.mnemonic]
    except KeyError as exc:
        raise KeyError(f"No SCIL builder for mnemonic {decoded.mnemonic}") from exc
    return builder(decoded)
