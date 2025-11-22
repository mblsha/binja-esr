from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, cast

from ..decoding.bind import (
    Addr16Page,
    Addr24,
    DecodedInstr,
    Disp8,
    ExtRegPtr,
    Imm8,
    Imm16,
    Imm20,
    Imm24,
    PreLatch,
    RegSel,
    ImemPtr,
    IntAddrCalc,
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

_LOOP_EXT_MODE_IMM = 0xFF
_LOOP_EXT_MODE_SIMPLE = 0
_LOOP_EXT_MODE_POST_INC = 1
_LOOP_EXT_MODE_PRE_DEC = 2
_LOOP_EXT_MODE_OFFSET = 3


def _is_loop_mv(mnemonic: str) -> bool:
    """Return True for MVL variants that iterate via IL/I counters."""

    return mnemonic.startswith("MVL")


def _loop_stride(width_bits: int) -> int:
    return max(1, width_bits // 8)


def _loop_mode_tag(mode: Optional[str]) -> int:
    if mode == "simple":
        return _LOOP_EXT_MODE_SIMPLE
    if mode == "post_inc":
        return _LOOP_EXT_MODE_POST_INC
    if mode == "pre_dec":
        return _LOOP_EXT_MODE_PRE_DEC
    if mode == "offset":
        return _LOOP_EXT_MODE_OFFSET
    return _LOOP_EXT_MODE_IMM


def _loop_disp_const(value: Optional[ast.Const]) -> ast.Const:
    if value is not None:
        return value
    return _const(0, 8)


def _build_loop_ext_instr(
    decoded: DecodedInstr,
    *,
    effect_kind: str,
    loop_ptr: ast.LoopIntPtr,
    other_ptr: ast.Expr,
    mode_tag: int,
    disp_const: Optional[ast.Const],
    width_bits: int,
    fetches: Sequence[ast.Fetch] = (),
    binder: Optional[Binder] = None,
) -> BuildResult:
    step = _loop_stride(width_bits)
    effect = ast.Effect(
        effect_kind,
        (
            ast.Reg("I", 16),
            loop_ptr,
            other_ptr,
            _const(mode_tag, 8),
            _loop_disp_const(disp_const),
            _const(width_bits, 8),
            _const(step & 0xFFFF, 16),
        ),
    )
    semantics = tuple(fetches) + (effect,)
    instr = ast.Instr(name=decoded.mnemonic, length=decoded.length, semantics=semantics)
    return BuildResult(instr, binder or {}, decoded.pre_applied)


def _const(value: int, bits: int) -> ast.Const:
    mask = (1 << bits) - 1
    return ast.Const(value & mask, bits)


def _imm_value(imm: object) -> int:
    if isinstance(imm, Imm8):
        return imm.value
    if isinstance(imm, Imm16):
        return imm.u16
    if isinstance(imm, Imm20):
        return imm.value
    if isinstance(imm, Imm24):
        return imm.u24
    raise TypeError(f"Unsupported immediate type: {type(imm)!r}")


def _imm_size(imm: object) -> int:
    if isinstance(imm, Imm8):
        return 8
    if isinstance(imm, Imm16):
        return 16
    if isinstance(imm, Imm20):
        return 20
    if isinstance(imm, Imm24):
        return 24
    raise TypeError(f"Unsupported immediate type: {type(imm)!r}")


def _imm_kind(imm: object) -> str:
    size = _imm_size(imm)
    return "u8" if size == 8 else f"imm{size}"


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


def _dynamic_imem_ptr(ptr: ImemPtr, *, prefix: str = "ptr") -> tuple[Tuple[ast.Fetch, ...], ast.Expr, Binder]:
    base_tmp = ast.Tmp(f"{prefix}_base", 8)
    disp_tmp = ast.Tmp(f"{prefix}_disp", 8)
    fetches = (
        ast.Fetch("u8", base_tmp),
        ast.Fetch("u8", disp_tmp),
    )
    pointer = ast.Mem("int", base_tmp, 24)
    signed_disp = ast.UnOp("sext", disp_tmp, 24)
    addr = ast.BinOp("add", pointer, signed_disp, 24)
    binder = {
        f"{prefix}_base": _const(ptr.base.value, 8),
        f"{prefix}_disp": _const(ptr.disp.value if ptr.disp else 0, 8),
    }
    return fetches, addr, binder


def _int_addr_delta(base: ast.Expr, delta: int) -> ast.Expr:
    if delta == 0:
        return base
    return ast.BinOp("add", base, _const(delta, 8), 8)


def _emem_imem_pointer(ptr: Imm8, offset: Optional[Disp8]) -> ast.Expr:
    base = _const(ptr.value, 8)
    lo = ast.Mem("int", base, 8)
    mid = ast.Mem("int", _int_addr_delta(base, 1), 8)
    hi = ast.Mem("int", _int_addr_delta(base, 2), 8)
    addr = ast.Join24(hi, mid, lo)
    if offset is not None:
        addr = ast.BinOp("add", addr, _const(offset.value, 24), 24)
    return addr


def _flag_cond(flag: str, expect_one: bool) -> ast.Cond:
    return ast.Cond(
        kind="eq",
        a=ast.Flag(flag),
        b=_const(1 if expect_one else 0, 1),
    )


def _zero_expr(value: ast.Expr, bits: int) -> ast.Expr:
    return ast.BinOp("eq", value, _const(0, bits), 1)


def _carry_expr(op: str, lhs: ast.Expr, rhs: ast.Expr, result: ast.Expr) -> Optional[ast.Expr]:
    if op == "add":
        cond = ast.Cond(kind="ltu", a=result, b=lhs)
    elif op == "sub":
        cond = ast.Cond(kind="ltu", a=lhs, b=rhs)
    else:
        return None
    return ast.TernOp(
        op="select",
        cond=cond,
        t=_const(1, 1),
        f=_const(0, 1),
        out_size=1,
    )


def _with_pre(instr: ast.Instr, binder: Binder, decoded: DecodedInstr) -> BuildResult:
    return BuildResult(instr=instr, binder=binder, pre_applied=decoded.pre_applied)


def _imm8(decoded: DecodedInstr, key: str = "imm8") -> Dict[str, ast.Const]:
    imm = decoded.binds["n"]
    assert isinstance(imm, Imm8)
    return {key: _const(imm.value, 8)}


def _mv_a_n(decoded: DecodedInstr) -> BuildResult:
    return _with_pre(examples.mv_a_imm(), _imm8(decoded), decoded)


def _mv_reg_imm(decoded: DecodedInstr) -> BuildResult:
    dst = decoded.binds["dst"]
    assert isinstance(dst, RegSel)
    bits = _REG_BITS[dst.size_group]
    imm = decoded.binds["n"]
    size = _imm_size(imm)
    value = _imm_value(imm)
    kind = _imm_kind(imm)
    tmp = ast.Tmp("n", size)
    instr = ast.Instr(
        name=decoded.mnemonic,
        length=decoded.length,
        semantics=(
            ast.Fetch(kind, tmp),
            ast.SetReg(reg=ast.Reg(dst.name, bits), value=tmp),
        ),
    )
    return _with_pre(instr, {"n": _const(value, size)}, decoded)


def _nop(decoded: DecodedInstr) -> BuildResult:
    return _with_pre(examples.nop_instr(), {}, decoded)


def _alu(
    decoded: DecodedInstr, op: str, include_carry: bool, flags: tuple[str, ...]
) -> BuildResult:
    imm = ast.Tmp("imm8", 8)
    rhs: ast.Expr = imm
    if include_carry:
        rhs = ast.BinOp("add", imm, ast.Flag("C"), 8)
    lhs = ast.Reg("A", 8)
    result = ast.BinOp(op, lhs, rhs, 8)
    stmts: List[ast.Stmt] = [ast.Fetch("u8", imm)]
    if "C" in flags:
        carry = _carry_expr(op, lhs, rhs, result)
        if carry is not None:
            stmts.append(ast.SetFlag("C", carry))
    if "Z" in flags:
        stmts.append(ast.SetFlag("Z", _zero_expr(result, 8)))
    stmts.append(ast.SetReg(lhs, result))
    instr = ast.Instr(
        name=decoded.mnemonic,
        length=decoded.length,
        semantics=tuple(stmts),
    )
    return _with_pre(instr, _imm8(decoded), decoded)


def _alu_reg_reg(
    decoded: DecodedInstr, op: str, flags: tuple[str, ...] | None = ("Z", "C")
) -> BuildResult:
    dst = decoded.binds["dst"]
    src = decoded.binds["src"]
    assert isinstance(dst, RegSel)
    assert isinstance(src, RegSel)
    dst_bits = _REG_BITS.get(dst.size_group)
    src_bits = _REG_BITS.get(src.size_group)
    if dst_bits is None or src_bits is None:
        raise ValueError("Unsupported register group in reg/reg ALU")
    lhs = ast.Reg(dst.name, dst_bits)
    rhs_expr = ast.Reg(src.name, src_bits)
    if dst_bits == src_bits:
        rhs = rhs_expr
    elif dst_bits > src_bits:
        rhs = ast.UnOp("zext", rhs_expr, dst_bits)
    else:
        rhs = ast.UnOp("low_part", rhs_expr, dst_bits)
    result = ast.BinOp(op, lhs, rhs, dst_bits)
    stmts: List[ast.Stmt] = []
    if flags:
        if "C" in flags:
            carry = _carry_expr(op, lhs, rhs, result)
            if carry is not None:
                stmts.append(ast.SetFlag("C", carry))
        if "Z" in flags:
            stmts.append(ast.SetFlag("Z", _zero_expr(result, dst_bits)))
    instr = ast.Instr(
        name=decoded.mnemonic,
        length=decoded.length,
        semantics=tuple(stmts + [ast.SetReg(lhs, result)]),
    )
    return _with_pre(instr, {}, decoded)


def _mv_reg_pair(decoded: DecodedInstr) -> BuildResult:
    dst = decoded.binds["dst"]
    src = decoded.binds["src"]
    assert isinstance(dst, RegSel)
    assert isinstance(src, RegSel)
    bits = _REG_BITS.get(dst.size_group)
    src_bits = _REG_BITS.get(src.size_group)
    if bits is None or src_bits is None:
        raise ValueError("Unsupported register group for MV r,r")
    src_reg = ast.Reg(src.name, src_bits)
    value: ast.Expr = src_reg
    if bits > src_bits:
        value = ast.UnOp("zext", src_reg, bits)
    elif bits < src_bits:
        value = ast.UnOp("low_part", src_reg, bits)
    instr = ast.Instr(
        name="MV r,r",
        length=decoded.length,
        semantics=(
            ast.SetReg(ast.Reg(dst.name, bits), value),
        ),
    )
    return _with_pre(instr, {}, decoded)


def _mv_reg_imem(decoded: DecodedInstr) -> BuildResult:
    dst = decoded.binds["dst"]
    dst_off = decoded.binds["dst_off"]
    assert isinstance(dst, RegSel)
    assert isinstance(dst_off, Imm8)
    bits = _REG_BITS.get(dst.size_group)
    if bits is None:
        raise ValueError("unsupported register width for mv_reg_imem")
    offset = ast.Tmp("dst_off", 8)
    mem = ast.Mem("int", offset, bits)
    instr = ast.Instr(
        name=decoded.mnemonic,
        length=decoded.length,
        semantics=(
            ast.Fetch("u8", offset),
            ast.Store(dst=mem, value=ast.Reg(dst.name, bits)),
        ),
    )
    binder = {"dst_off": _const(dst_off.value, 8)}
    return _with_pre(instr, binder, decoded)


def _test(decoded: DecodedInstr) -> BuildResult:
    return _with_pre(examples.test_a_imm(), _imm8(decoded), decoded)


def _jr(decoded: DecodedInstr, flag: str, expect_one: bool) -> BuildResult:
    offset = decoded.binds["disp"]
    assert isinstance(offset, Imm8)
    direction = decoded.binds.get("dir", 1)
    disp = ast.Tmp("disp8", 8)
    signed = ast.UnOp("sext", disp, 20)
    if direction and direction < 0:
        signed = ast.UnOp("neg", signed, 20)
    target = ast.PcRel(
        base_advance=decoded.length,
        disp=signed,
        out_size=20,
    )
    instr = ast.Instr(
        name=decoded.mnemonic,
        length=decoded.length,
        semantics=(
            ast.Fetch("disp8", disp),
            ast.If(
                cond=_flag_cond(flag, expect_one),
                then_ops=(ast.Goto(target),),
            ),
        ),
    )
    binder = {"disp8": _const(offset.value, 8)}
    return _with_pre(instr, binder, decoded)


def _jr_rel(decoded: DecodedInstr, direction: int) -> BuildResult:
    offset = decoded.binds.get("disp")
    assert isinstance(offset, Imm8)
    disp = ast.Tmp("disp8", 8)
    signed = ast.UnOp("sext", disp, 20)
    if direction < 0:
        signed = ast.UnOp("neg", signed, 20)
    target = ast.PcRel(
        base_advance=decoded.length,
        disp=signed,
        out_size=20,
    )
    instr = ast.Instr(
        name=decoded.mnemonic,
        length=decoded.length,
        semantics=(
            ast.Fetch("disp8", disp),
            ast.Goto(target),
        ),
    )
    binder = {"disp8": _const(offset.value, 8)}
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


def _jp_far(decoded: DecodedInstr) -> BuildResult:
    spec = examples.jp_far()
    addr = decoded.binds["addr20"]
    assert isinstance(addr, Imm20)
    binder = {"addr20": _const(addr.value, 20)}
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
    reg = decoded.binds.get("dst_reg")
    if reg is None:
        spec = examples.mv_ext_store()
    else:
        assert isinstance(reg, RegSel)
        bits = _REG_BITS.get(reg.size_group)
        if bits is None:
            raise ValueError(f"Unsupported register group for MV [lmn],{reg.name}")
        spec = examples.mv_ext_store_reg(reg.name, bits)
    addr = decoded.binds["addr24"]
    assert isinstance(addr, Addr24)
    binder = {"addr_ptr": _const(addr.v.u24, 24)}
    return _with_pre(spec, binder, decoded)


def _mv_imem_from_ext(decoded: DecodedInstr) -> BuildResult:
    dst = decoded.binds["dst"]
    addr = decoded.binds["addr24"]
    width = decoded.binds["width_bits"]
    assert isinstance(dst, Imm8)
    assert isinstance(addr, Addr24)
    if _is_loop_mv(decoded.mnemonic):
        dst_tmp = ast.Tmp("dst_off", 8)
        addr_tmp = ast.Tmp("addr_ptr", 24)
        return _build_loop_ext_instr(
            decoded,
            effect_kind="loop_ext_to_int",
            loop_ptr=ast.LoopIntPtr(dst_tmp),
            other_ptr=addr_tmp,
            mode_tag=_LOOP_EXT_MODE_IMM,
            disp_const=None,
            width_bits=width,
            fetches=(ast.Fetch("u8", dst_tmp), ast.Fetch("addr24", addr_tmp)),
            binder={
                "dst_off": _const(0, 8),
                "addr_ptr": _const(0, 24),
            },
        )
    spec = examples.mv_imem_from_ext(decoded.mnemonic, width)
    binder = {"dst_off": _const(dst.value, 8), "addr24": _const(addr.v.u24, 24)}
    return _with_pre(spec, binder, decoded)


def _mv_ext_from_imem(decoded: DecodedInstr) -> BuildResult:
    addr = decoded.binds["addr24"]
    src = decoded.binds["src"]
    width = decoded.binds["width_bits"]
    assert isinstance(addr, Addr24)
    assert isinstance(src, Imm8)
    if _is_loop_mv(decoded.mnemonic):
        src_tmp = ast.Tmp("src_off", 8)
        addr_tmp = ast.Tmp("addr_ptr", 24)
        return _build_loop_ext_instr(
            decoded,
            effect_kind="loop_int_to_ext",
            loop_ptr=ast.LoopIntPtr(src_tmp),
            other_ptr=addr_tmp,
            mode_tag=_LOOP_EXT_MODE_IMM,
            disp_const=None,
            width_bits=width,
            fetches=(ast.Fetch("u8", src_tmp), ast.Fetch("addr24", addr_tmp)),
            binder={
                "src_off": _const(0, 8),
                "addr_ptr": _const(0, 24),
            },
        )
    spec = examples.mv_ext_from_imem(decoded.mnemonic, width)
    binder = {"addr24": _const(addr.v.u24, 24), "src_off": _const(src.value, 8)}
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
    fetches, addr_expr, binder = _dynamic_imem_ptr(ptr)
    semantics = fetches + (
        ast.SetReg(
            reg=ast.Reg(dst.name, bits),
            value=ast.Mem("ext", addr_expr, bits),
        ),
    )
    instr = ast.Instr(name=decoded.mnemonic, length=decoded.length, semantics=semantics)
    return BuildResult(instr, binder, decoded.pre_applied)


def _ext_ptr_store(decoded: DecodedInstr) -> BuildResult:
    src = decoded.binds["src"]
    ptr = decoded.binds["ptr"]
    assert isinstance(src, RegSel)
    assert isinstance(ptr, ImemPtr)
    bits = _REG_BITS.get(src.size_group)
    if bits is None:
        raise ValueError(f"Unsupported source group {src.size_group}")
    fetches, addr_expr, binder = _dynamic_imem_ptr(ptr)
    semantics = fetches + (
        ast.Store(
            dst=ast.Mem("ext", addr_expr, bits),
            value=ast.Reg(src.name, bits),
        ),
    )
    instr = ast.Instr(name=decoded.mnemonic, length=decoded.length, semantics=semantics)
    return BuildResult(instr, binder, decoded.pre_applied)


def _emem_imem_to_int(decoded: DecodedInstr) -> BuildResult:
    dst = decoded.binds["dst"]
    ptr = decoded.binds["ptr"]
    width = decoded.binds["width_bits"]
    offset = decoded.binds.get("offset")
    assert isinstance(dst, Imm8)
    assert isinstance(ptr, Imm8)
    pointer = _emem_imem_pointer(ptr, cast(Optional[Disp8], offset))
    dst_mem = ast.Mem("int", _const(dst.value, 8), width)
    src_mem = ast.Mem("ext", pointer, width)
    instr = ast.Instr(
        name=decoded.mnemonic,
        length=decoded.length,
        semantics=(ast.Store(dst=dst_mem, value=src_mem),),
    )
    return _with_pre(instr, {}, decoded)


def _emem_imem_to_ext(decoded: DecodedInstr) -> BuildResult:
    src = decoded.binds["dst"]
    ptr = decoded.binds["ptr"]
    width = decoded.binds["width_bits"]
    offset = decoded.binds.get("offset")
    assert isinstance(src, Imm8)
    assert isinstance(ptr, Imm8)
    pointer = _emem_imem_pointer(ptr, cast(Optional[Disp8], offset))
    src_mem = ast.Mem("int", _const(src.value, 8), width)
    dst_mem = ast.Mem("ext", pointer, width)
    instr = ast.Instr(
        name=decoded.mnemonic,
        length=decoded.length,
        semantics=(ast.Store(dst=dst_mem, value=src_mem),),
    )
    return _with_pre(instr, {}, decoded)


def _int_mem_operand(imm: Imm8, width: int) -> ast.Mem:
    return ast.Mem("int", _const(imm.value, 8), width)


def _mv_imem_from_ext_regptr(decoded: DecodedInstr) -> BuildResult:
    ptr = decoded.binds["ptr"]
    imem = decoded.binds["imem"]
    width = decoded.binds["width"]
    assert isinstance(ptr, ExtRegPtr)
    assert isinstance(imem, Imm8)
    if _is_loop_mv(decoded.mnemonic):
        loop_tmp = ast.Tmp("dst_off", 8)
        src_reg = ast.Reg(ptr.ptr.name, 24)
        mode_tag = _loop_mode_tag(ptr.mode)
        disp_const = _disp_const(ptr.disp)
        return _build_loop_ext_instr(
            decoded,
            effect_kind="loop_ext_to_int",
            loop_ptr=ast.LoopIntPtr(loop_tmp),
            other_ptr=src_reg,
            mode_tag=mode_tag,
            disp_const=disp_const,
            width_bits=width,
            fetches=(ast.Fetch("u8", loop_tmp),),
            binder={"dst_off": _const(0, 8)},
        )
    dst_tmp = ast.Tmp("dst_off", 8)
    dst_mem = ast.Mem("int", dst_tmp, width)
    stmt = ast.ExtRegToIntMem(
        ptr=ast.Reg(ptr.ptr.name, 24),
        mode=ptr.mode,
        disp=_disp_const(ptr.disp),
        dst=dst_mem,
    )
    instr = ast.Instr(
        name=decoded.mnemonic,
        length=decoded.length,
        semantics=(ast.Fetch("u8", dst_tmp), stmt),
    )
    binder = {"dst_off": _const(imem.value, 8)}
    return BuildResult(instr, binder, decoded.pre_applied)


def _mv_ext_from_imem_regptr(decoded: DecodedInstr) -> BuildResult:
    ptr = decoded.binds["ptr"]
    imem = decoded.binds["imem"]
    width = decoded.binds["width"]
    assert isinstance(ptr, ExtRegPtr)
    assert isinstance(imem, (Imm8, ImemPtr))
    if _is_loop_mv(decoded.mnemonic):
        loop_tmp = ast.Tmp("src_off", 8)
        dst_reg = ast.Reg(ptr.ptr.name, 24)
        mode_tag = _loop_mode_tag(ptr.mode)
        disp_const = _disp_const(ptr.disp)
        return _build_loop_ext_instr(
            decoded,
            effect_kind="loop_int_to_ext",
            loop_ptr=ast.LoopIntPtr(loop_tmp),
            other_ptr=dst_reg,
            mode_tag=mode_tag,
            disp_const=disp_const,
            width_bits=width,
            fetches=(ast.Fetch("u8", loop_tmp),),
            binder={"src_off": _const(0, 8)},
        )
    if isinstance(imem, ImemPtr):
        fetches, addr_expr, ptr_binder = _dynamic_imem_ptr(imem, prefix="src")
        src_mem = ast.Mem("int", addr_expr, width)
        semantics = fetches + (
            ast.IntMemToExtReg(
                src=src_mem,
                ptr=ast.Reg(ptr.ptr.name, 24),
                mode=ptr.mode,
                disp=_disp_const(ptr.disp),
            ),
        )
        binder = dict(ptr_binder)
    else:
        src_tmp = ast.Tmp("src_off", 8)
        src_mem = ast.Mem("int", src_tmp, width)
        semantics = (
            ast.Fetch("u8", src_tmp),
            ast.IntMemToExtReg(
                src=src_mem,
                ptr=ast.Reg(ptr.ptr.name, 24),
                mode=ptr.mode,
                disp=_disp_const(ptr.disp),
            ),
        )
        binder = {"src_off": _const(imem.value, 8)}
    binder["src_pre_slot"] = _const(2, 8)
    instr = ast.Instr(name=decoded.mnemonic, length=decoded.length, semantics=semantics)
    return BuildResult(instr, binder, decoded.pre_applied)


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
    dst_tmp = ast.Tmp("dst_off", 8)
    src_tmp = ast.Tmp("src_off", 8)
    dst_mem = ast.Mem("int", dst_tmp, width)
    src_mem = ast.Mem("int", src_tmp, width)
    instr = ast.Instr(
        name=decoded.mnemonic,
        length=decoded.length,
        semantics=(
            ast.Fetch("u8", dst_tmp),
            ast.Fetch("u8", src_tmp),
            ast.Store(dst=dst_mem, value=src_mem),
        ),
    )
    binder = {
        "dst_off": _const(dst.value, 8),
        "src_off": _const(src.value, 8),
    }
    return _with_pre(instr, binder, decoded)


def _ex_registers(decoded: DecodedInstr) -> BuildResult:
    lhs = decoded.binds["lhs"]
    rhs = decoded.binds["rhs"]
    assert isinstance(lhs, RegSel)
    assert isinstance(rhs, RegSel)
    lhs_bits = _REG_BITS[lhs.size_group]
    rhs_bits = _REG_BITS[rhs.size_group]
    if lhs_bits != rhs_bits:
        raise ValueError("EX registers must share the same width")
    instr = ast.Instr(
        name="EX r,r",
        length=decoded.length,
        semantics=(
            ast.Effect(
                "swap_reg",
                (
                    ast.Reg(lhs.name, lhs_bits),
                    ast.Reg(rhs.name, rhs_bits),
                ),
            ),
        ),
    )
    return _with_pre(instr, {}, decoded)


def _ex_a_b(decoded: DecodedInstr) -> BuildResult:
    instr = ast.Instr(
        name="EX A,B",
        length=decoded.length,
        semantics=(
            ast.Effect(
                "swap_reg",
                (
                    ast.Reg("A", 8),
                    ast.Reg("B", 8),
                ),
            ),
        ),
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


def _loop_carry(
    decoded: DecodedInstr, effect_kind: str, *, src_is_mem: bool
) -> BuildResult:
    dst = decoded.binds["dst"]
    assert isinstance(dst, Imm8)
    binder = {"dst_off": _const(dst.value, 8)}
    if src_is_mem:
        src = decoded.binds["src"]
        assert isinstance(src, Imm8)
        binder["src_off"] = _const(src.value, 8)
    spec = examples.loop_carry_instr(
        decoded.mnemonic, effect_kind, src_is_mem=src_is_mem
    )
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


def _decimal_shift(decoded: DecodedInstr, *, is_left: bool) -> BuildResult:
    dst = decoded.binds["dst"]
    assert isinstance(dst, Imm8)
    spec = examples.decimal_shift_instr(decoded.mnemonic, is_left=is_left)
    binder = {"dst_off": _const(dst.value, 8)}
    return _with_pre(spec, binder, decoded)


def _pmdf_imm(decoded: DecodedInstr) -> BuildResult:
    dst = decoded.binds["dst"]
    imm = decoded.binds["imm"]
    assert isinstance(dst, Imm8)
    assert isinstance(imm, Imm8)
    spec = examples.pmdf_immediate()
    binder = {
        "dst_off": _const(dst.value, 8),
        "imm8": _const(imm.value, 8),
    }
    return _with_pre(spec, binder, decoded)


def _pmdf_reg(decoded: DecodedInstr) -> BuildResult:
    dst = decoded.binds["dst"]
    assert isinstance(dst, Imm8)
    spec = examples.pmdf_reg()
    binder = {"dst_off": _const(dst.value, 8)}
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


def _mv_imem_const(decoded: DecodedInstr) -> BuildResult:
    spec = examples.mv_imem_const()
    dst = decoded.binds["dst"]
    value = decoded.binds["imm"]
    assert isinstance(dst, Imm8)
    assert isinstance(value, Imm8)
    binder = {
        "dst_off": _const(dst.value, 8),
        "imm8": _const(value.value, 8),
    }
    return _with_pre(spec, binder, decoded)


def _mv_imem_reg(decoded: DecodedInstr) -> BuildResult:
    dst = decoded.binds["dst"]
    addr = decoded.binds["dst_off"]
    assert isinstance(dst, RegSel)
    assert isinstance(addr, Imm8)
    bits = _REG_BITS.get(dst.size_group)
    if bits is None:
        raise ValueError("Unsupported register group for mv_imem_reg")
    offset = ast.Tmp("dst_off", 8)
    mem_expr = ast.Mem("int", offset, bits)
    instr = ast.Instr(
        name=decoded.mnemonic,
        length=decoded.length,
        semantics=(
            ast.Fetch("u8", offset),
            ast.SetReg(ast.Reg(dst.name, bits), mem_expr),
        ),
    )
    binder = {"dst_off": _const(addr.value, 8)}
    return _with_pre(instr, binder, decoded)


def _mv_ext_reg(decoded: DecodedInstr) -> BuildResult:
    dst = decoded.binds["dst"]
    assert isinstance(dst, RegSel)
    bits = _REG_BITS.get(dst.size_group)
    if bits is None:
        raise ValueError("Unsupported register group for MV ext reg")
    addr_tmp = ast.Tmp("addr24", 24)
    mem_expr = ast.Mem("ext", addr_tmp, bits)
    instr = ast.Instr(
        name=decoded.mnemonic,
        length=decoded.length,
        semantics=(
            ast.Fetch("addr24", addr_tmp),
            ast.SetReg(ast.Reg(dst.name, bits), mem_expr),
        ),
    )
    return _with_pre(instr, {}, decoded)


def _mv_imem_word_const(decoded: DecodedInstr) -> BuildResult:
    dst = decoded.binds["dst"]
    lo = decoded.binds["lo"]
    hi = decoded.binds["hi"]
    assert isinstance(dst, Imm8)
    assert isinstance(lo, Imm8)
    assert isinstance(hi, Imm8)
    dst_tmp = ast.Tmp("dst_off", 8)
    lo_tmp = ast.Tmp("imm8_lo", 8)
    hi_tmp = ast.Tmp("imm8_hi", 8)
    lo_zext = ast.UnOp("zext", lo_tmp, 16)
    hi_zext = ast.UnOp("zext", hi_tmp, 16)
    hi_shift = ast.BinOp("shl", hi_zext, ast.Const(8, 8), 16)
    combined = ast.BinOp("or", lo_zext, hi_shift, 16)
    mem = ast.Mem("int", dst_tmp, 16)
    instr = ast.Instr(
        name="MVW_(L),MN",
        length=decoded.length,
        semantics=(
            ast.Fetch("u8", dst_tmp),
            ast.Fetch("u8", lo_tmp),
            ast.Fetch("u8", hi_tmp),
            ast.Store(mem, combined),
        ),
    )
    binder = {
        "dst_off": _const(dst.value, 8),
        "imm8_lo": _const(lo.value, 8),
        "imm8_hi": _const(hi.value, 8),
    }
    return _with_pre(instr, binder, decoded)


def _mv_imem_long_const(decoded: DecodedInstr) -> BuildResult:
    dst = decoded.binds["dst"]
    lo = decoded.binds["lo"]
    mid = decoded.binds["mid"]
    hi = decoded.binds["hi"]
    assert isinstance(dst, Imm8)
    assert isinstance(lo, Imm8)
    assert isinstance(mid, Imm8)
    assert isinstance(hi, Imm8)
    dst_tmp = ast.Tmp("dst_off", 8)
    lo_tmp = ast.Tmp("imm8_lo", 8)
    mid_tmp = ast.Tmp("imm8_mid", 8)
    hi_tmp = ast.Tmp("imm8_hi", 8)
    mem = ast.Mem("int", dst_tmp, 24)
    instr = ast.Instr(
        name=decoded.mnemonic,
        length=decoded.length,
        semantics=(
            ast.Fetch("u8", dst_tmp),
            ast.Fetch("u8", lo_tmp),
            ast.Fetch("u8", mid_tmp),
            ast.Fetch("u8", hi_tmp),
            ast.Store(mem, ast.Join24(hi_tmp, mid_tmp, lo_tmp)),
        ),
    )
    binder = {
        "dst_off": _const(dst.value, 8),
        "imm8_lo": _const(lo.value, 8),
        "imm8_mid": _const(mid.value, 8),
        "imm8_hi": _const(hi.value, 8),
    }
    return _with_pre(instr, binder, decoded)


def _rc(decoded: DecodedInstr) -> BuildResult:
    return _with_pre(examples.rc_instr(), {}, decoded)


def _sc(decoded: DecodedInstr) -> BuildResult:
    return _with_pre(examples.sc_instr(), {}, decoded)


def _swap_a(decoded: DecodedInstr) -> BuildResult:
    return _with_pre(examples.swap_a_instr(), {}, decoded)


def _tcl(decoded: DecodedInstr) -> BuildResult:
    return _with_pre(examples.tcl_instr(), {}, decoded)


def _rotate_a(decoded: DecodedInstr, op: str) -> BuildResult:
    spec = examples.rotate_a_instr(decoded.mnemonic, op)
    return _with_pre(spec, {}, decoded)


def _mem_rotate(decoded: DecodedInstr, op: str) -> BuildResult:
    dst = decoded.binds["dst"]
    assert isinstance(dst, Imm8)
    spec = examples._rot_imem_instr(decoded.mnemonic, op)
    binder = {"dst_off": _const(dst.value, 8)}
    return _with_pre(spec, binder, decoded)


def _shift_a(decoded: DecodedInstr, op: str) -> BuildResult:
    return _rotate_a(decoded, op)


def _shift_mem(decoded: DecodedInstr, op: str) -> BuildResult:
    return _mem_rotate(decoded, op)


def _mv_a_a(decoded: DecodedInstr) -> BuildResult:
    return _with_pre(examples.mv_a_a_instr(), {}, decoded)


def _test_imem_const(decoded: DecodedInstr) -> BuildResult:
    spec = examples.test_imem_const()
    dst = decoded.binds["dst"]
    imm = decoded.binds["imm"]
    assert isinstance(dst, Imm8)
    assert isinstance(imm, Imm8)
    binder = {
        "dst_off": _const(dst.value, 8),
        "imm8": _const(imm.value, 8),
    }
    return _with_pre(spec, binder, decoded)


def _test_emem_const(decoded: DecodedInstr) -> BuildResult:
    addr = decoded.binds["addr24"]
    imm = decoded.binds["imm"]
    assert isinstance(addr, Addr24)
    assert isinstance(imm, Imm8)
    spec = examples.test_emem_const()
    binder = {
        "addr24": _const(addr.v.u24, 24),
        "imm8": _const(imm.value, 8),
    }
    return _with_pre(spec, binder, decoded)


def _test_imem_reg(decoded: DecodedInstr) -> BuildResult:
    dst = decoded.binds["dst"]
    assert isinstance(dst, Imm8)
    spec = examples.test_imem_reg()
    binder = {"dst_off": _const(dst.value, 8)}
    return _with_pre(spec, binder, decoded)


def _xor_emem_const(decoded: DecodedInstr) -> BuildResult:
    addr = decoded.binds["addr24"]
    imm = decoded.binds["imm"]
    assert isinstance(addr, Addr24)
    assert isinstance(imm, Imm8)
    spec = examples.xor_emem_const()
    binder = {"addr24": _const(addr.v.u24, 24), "imm8": _const(imm.value, 8)}
    return _with_pre(spec, binder, decoded)


def _and_emem_const(decoded: DecodedInstr) -> BuildResult:
    addr = decoded.binds["addr24"]
    imm = decoded.binds["imm"]
    assert isinstance(addr, Addr24)
    assert isinstance(imm, Imm8)
    spec = examples.and_emem_const()
    binder = {"addr24": _const(addr.v.u24, 24), "imm8": _const(imm.value, 8)}
    return _with_pre(spec, binder, decoded)


def _or_emem_const(decoded: DecodedInstr) -> BuildResult:
    addr = decoded.binds["addr24"]
    imm = decoded.binds["imm"]
    assert isinstance(addr, Addr24)
    assert isinstance(imm, Imm8)
    spec = examples.or_emem_const()
    binder = {"addr24": _const(addr.v.u24, 24), "imm8": _const(imm.value, 8)}
    return _with_pre(spec, binder, decoded)


def _xor_a_imem(decoded: DecodedInstr) -> BuildResult:
    dst = decoded.binds["dst"]
    assert isinstance(dst, Imm8)
    spec = examples.xor_a_imem()
    binder = {"dst_off": _const(dst.value, 8)}
    return _with_pre(spec, binder, decoded)


def _xor_imem_mem(decoded: DecodedInstr) -> BuildResult:
    dst = decoded.binds["dst"]
    src = decoded.binds["src"]
    assert isinstance(dst, Imm8)
    assert isinstance(src, Imm8)
    spec = examples.xor_mem_mem()
    binder = {"dst_off": _const(dst.value, 8), "src_off": _const(src.value, 8)}
    return _with_pre(spec, binder, decoded)


def _add_imem_const(decoded: DecodedInstr) -> BuildResult:
    dst = decoded.binds["dst"]
    imm = decoded.binds["imm"]
    assert isinstance(dst, Imm8)
    assert isinstance(imm, Imm8)
    spec = examples.add_imem_const()
    binder = {"dst_off": _const(dst.value, 8), "imm8": _const(imm.value, 8)}
    return _with_pre(spec, binder, decoded)


def _adc_imem_const(decoded: DecodedInstr) -> BuildResult:
    dst = decoded.binds["dst"]
    imm = decoded.binds["imm"]
    assert isinstance(dst, Imm8)
    assert isinstance(imm, Imm8)
    spec = examples.adc_imem_const()
    binder = {"dst_off": _const(dst.value, 8), "imm8": _const(imm.value, 8)}
    return _with_pre(spec, binder, decoded)


def _adc_a_mem(decoded: DecodedInstr) -> BuildResult:
    dst = decoded.binds["dst"]
    assert isinstance(dst, Imm8)
    spec = examples.adc_a_imem()
    binder = {"dst_off": _const(dst.value, 8)}
    return _with_pre(spec, binder, decoded)


def _adc_mem_a(decoded: DecodedInstr) -> BuildResult:
    dst = decoded.binds["dst"]
    assert isinstance(dst, Imm8)
    spec = examples.adc_mem_a()
    binder = {"dst_off": _const(dst.value, 8)}
    return _with_pre(spec, binder, decoded)


def _sbc_imem_const(decoded: DecodedInstr) -> BuildResult:
    dst = decoded.binds["dst"]
    imm = decoded.binds["imm"]
    assert isinstance(dst, Imm8)
    assert isinstance(imm, Imm8)
    spec = examples.sbc_imem_const()
    binder = {"dst_off": _const(dst.value, 8), "imm8": _const(imm.value, 8)}
    return _with_pre(spec, binder, decoded)


def _sbc_a_mem(decoded: DecodedInstr) -> BuildResult:
    dst = decoded.binds["dst"]
    assert isinstance(dst, Imm8)
    spec = examples.sbc_a_imem()
    binder = {"dst_off": _const(dst.value, 8)}
    return _with_pre(spec, binder, decoded)


def _sbc_mem_a(decoded: DecodedInstr) -> BuildResult:
    dst = decoded.binds["dst"]
    assert isinstance(dst, Imm8)
    spec = examples.sbc_mem_a()
    binder = {"dst_off": _const(dst.value, 8)}
    return _with_pre(spec, binder, decoded)


def _sub_imem_const(decoded: DecodedInstr) -> BuildResult:
    dst = decoded.binds["dst"]
    imm = decoded.binds["imm"]
    assert isinstance(dst, Imm8)
    assert isinstance(imm, Imm8)
    spec = examples.sub_imem_const()
    binder = {"dst_off": _const(dst.value, 8), "imm8": _const(imm.value, 8)}
    return _with_pre(spec, binder, decoded)


def _imem_logic_const(decoded: DecodedInstr, op: str) -> BuildResult:
    spec = examples.imem_logic_instr(decoded.mnemonic, op)
    dst = decoded.binds["dst"]
    imm = decoded.binds["imm"]
    assert isinstance(dst, Imm8)
    assert isinstance(imm, Imm8)
    binder = {
        "dst_off": _const(dst.value, 8),
        "imm8": _const(imm.value, 8),
    }
    return _with_pre(spec, binder, decoded)


def _imem_logic_mem(decoded: DecodedInstr, op: str) -> BuildResult:
    dst = decoded.binds["dst"]
    src = decoded.binds["src"]
    assert isinstance(dst, Imm8)
    assert isinstance(src, Imm8)
    spec = examples.imem_logic_mem(decoded.mnemonic, op)
    binder = {"dst_off": _const(dst.value, 8), "src_off": _const(src.value, 8)}
    return _with_pre(spec, binder, decoded)


def _imem_logic_reg(decoded: DecodedInstr, op: str) -> BuildResult:
    dst = decoded.binds["dst"]
    assert isinstance(dst, Imm8)
    spec = examples.imem_logic_reg(decoded.mnemonic, op, ("Z",))
    binder = {"dst_off": _const(dst.value, 8)}
    return _with_pre(spec, binder, decoded)


def _imem_logic_store(decoded: DecodedInstr, op: str) -> BuildResult:
    dst = decoded.binds["dst"]
    assert isinstance(dst, Imm8)
    spec = examples.imem_logic_store(decoded.mnemonic, op)
    binder = {"dst_off": _const(dst.value, 8)}
    return _with_pre(spec, binder, decoded)


def _alu_a_mem(decoded: DecodedInstr, op: str, flags: tuple[str, ...] = ("Z",)) -> BuildResult:
    dst = decoded.binds["dst"]
    assert isinstance(dst, Imm8)
    spec = examples.alu_a_imem(decoded.mnemonic, op, flags)
    binder = {"dst_off": _const(dst.value, 8)}
    return _with_pre(spec, binder, decoded)


def _alu_mem_a(decoded: DecodedInstr, op: str) -> BuildResult:
    dst = decoded.binds["dst"]
    assert isinstance(dst, Imm8)
    spec = examples.alu_mem_a(decoded.mnemonic, op)
    binder = {"dst_off": _const(dst.value, 8)}
    return _with_pre(spec, binder, decoded)


def _cmp_imem_const(decoded: DecodedInstr) -> BuildResult:
    dst = decoded.binds["dst"]
    imm = decoded.binds["imm"]
    assert isinstance(dst, Imm8)
    assert isinstance(imm, Imm8)
    spec = examples.cmp_imem_const()
    binder = {"dst_off": _const(dst.value, 8), "imm8": _const(imm.value, 8)}
    return _with_pre(spec, binder, decoded)


def _cmpw_imem_mem(decoded: DecodedInstr) -> BuildResult:
    dst = decoded.binds["dst"]
    src = decoded.binds["src"]
    assert isinstance(dst, Imm8)
    assert isinstance(src, Imm8)
    spec = examples.cmpw_imem_mem()
    binder = {"dst_off": _const(dst.value, 8), "src_off": _const(src.value, 8)}
    return _with_pre(spec, binder, decoded)


def _cmpp_imem_mem(decoded: DecodedInstr) -> BuildResult:
    dst = decoded.binds["dst"]
    src = decoded.binds["src"]
    assert isinstance(dst, Imm8)
    assert isinstance(src, Imm8)
    spec = examples.cmpp_imem_mem()
    binder = {"dst_off": _const(dst.value, 8), "src_off": _const(src.value, 8)}
    return _with_pre(spec, binder, decoded)


def _cmp_a_imm(decoded: DecodedInstr) -> BuildResult:
    imm = decoded.binds["imm"]
    assert isinstance(imm, Imm8)
    spec = examples.cmp_a_imm()
    binder = {"imm8": _const(imm.value, 8)}
    return _with_pre(spec, binder, decoded)


def _cmp_imem_reg(decoded: DecodedInstr) -> BuildResult:
    dst = decoded.binds["dst"]
    assert isinstance(dst, Imm8)
    spec = examples.cmp_imem_reg()
    binder = {"dst_off": _const(dst.value, 8)}
    return _with_pre(spec, binder, decoded)


def _cmp_emem_const(decoded: DecodedInstr) -> BuildResult:
    addr = decoded.binds["addr24"]
    imm = decoded.binds["imm"]
    assert isinstance(addr, Addr24)
    assert isinstance(imm, Imm8)
    spec = examples.cmp_emem_const()
    binder = {"addr24": _const(addr.v.u24, 24), "imm8": _const(imm.value, 8)}
    return _with_pre(spec, binder, decoded)


def _cmp_mem_mem(decoded: DecodedInstr) -> BuildResult:
    dst = decoded.binds["dst"]
    src = decoded.binds["src"]
    assert isinstance(dst, Imm8)
    assert isinstance(src, Imm8)
    spec = examples.cmp_imem_mem()
    binder = {"dst_off": _const(dst.value, 8), "src_off": _const(src.value, 8)}
    return _with_pre(spec, binder, decoded)


def _inc_dec_imem(decoded: DecodedInstr, op: str) -> BuildResult:
    dst = decoded.binds["dst"]
    assert isinstance(dst, Imm8)
    spec = examples.inc_dec_imem(decoded.mnemonic, op)
    binder = {"dst_off": _const(dst.value, 8)}
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
    "NOP": _nop,
    "MV A,n": _mv_a_n,
    "MV IL,n": _mv_reg_imm,
    "MVW BA,n": _mv_reg_imm,
    "MVW I,n": _mv_reg_imm,
    "MV X,n": _mv_reg_imm,
    "MV Y,n": _mv_reg_imm,
    "MV U,n": _mv_reg_imm,
    "MV S,n": _mv_reg_imm,
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
    "JR +n": lambda di: _jr_rel(di, +1),
    "JR -n": lambda di: _jr_rel(di, -1),
    "JP mn": _jp,
    "JPF lmn": _jp_far,
    "JPZ mn": lambda di: _jp_cond(di, "Z", True),
    "JPNZ mn": lambda di: _jp_cond(di, "Z", False),
    "JPC mn": lambda di: _jp_cond(di, "C", True),
    "JPNC mn": lambda di: _jp_cond(di, "C", False),
    "JP r3": _jp_r3,
    "JP (n)": _jp_imem_ptr,
    "MV A,[lmn]": _mv_ext_reg,
    "MV [lmn],A": _mv_ext_store,
    "MV A,(n)": _mv_imem_reg,
    "MV IL,(n)": _mv_imem_reg,
    "MV BA,(n)": _mv_imem_reg,
    "MV I,(n)": _mv_imem_reg,
    "MV X,(n)": _mv_imem_reg,
    "MV Y,(n)": _mv_imem_reg,
    "MV U,(n)": _mv_imem_reg,
    "MV S,(n)": _mv_imem_reg,
    "MV X,(n)": _mv_imem_reg,
    "MV Y,(n)": _mv_imem_reg,
    "MV U,(n)": _mv_imem_reg,
    "MV S,(n)": _mv_imem_reg,
    "MV (n),A": _mv_reg_imem,
    "MV (n),n": _mv_imem_const,
    "MV (n),IL": _mv_reg_imem,
    "MV (n),BA": _mv_reg_imem,
    "MV (n),I": _mv_reg_imem,
    "MV (n),X": _mv_reg_imem,
    "MV (n),Y": _mv_reg_imem,
    "MV (n),U": _mv_reg_imem,
    "MV (n),S": _mv_reg_imem,
    "MV (n),[lmn]": _mv_imem_from_ext,
    "MVW (n),[lmn]": _mv_imem_from_ext,
    "MVP (n),[lmn]": _mv_imem_from_ext,
    "MVP (n),const": _mv_imem_long_const,
    "MVL (n),[lmn]": _mv_imem_from_ext,
    "MV IL,[lmn]": _mv_ext_reg,
    "MV BA,[lmn]": _mv_ext_reg,
    "MV I,[lmn]": _mv_ext_reg,
    "MV X,[lmn]": _mv_ext_reg,
    "MV Y,[lmn]": _mv_ext_reg,
    "MV U,[lmn]": _mv_ext_reg,
    "MV S,[lmn]": _mv_ext_reg,
    "MV [lmn],(n)": _mv_ext_from_imem,
    "MVW [lmn],(n)": _mv_ext_from_imem,
    "MVP [lmn],(n)": _mv_ext_from_imem,
    "MVL [lmn],(n)": _mv_ext_from_imem,
    "MV (m),[(n)]": _emem_imem_to_int,
    "MVW (m),[(n)]": _emem_imem_to_int,
    "MVP (m),[(n)]": _emem_imem_to_int,
    "MVL (m),[(n)]": _emem_imem_to_int,
    "MV [lmn],A": _mv_ext_store,
    "MV [lmn],IL": _mv_ext_store,
    "MV [lmn],BA": _mv_ext_store,
    "MV [lmn],I": _mv_ext_store,
    "MV [lmn],X": _mv_ext_store,
    "MV [lmn],Y": _mv_ext_store,
    "MV [lmn],U": _mv_ext_store,
    "MV [lmn],S": _mv_ext_store,
    "TEST (n),n": _test_imem_const,
    "TEST (n),A": _test_imem_reg,
    "XOR (n),n": lambda di: _imem_logic_const(di, "xor"),
    "AND (n),n": lambda di: _imem_logic_const(di, "and"),
    "OR (n),n": lambda di: _imem_logic_const(di, "or"),
    "AND (n),A": lambda di: _imem_logic_store(di, "and"),
    "AND A,(n)": lambda di: _imem_logic_reg(di, "and"),
    "OR (n),A": lambda di: _imem_logic_store(di, "or"),
    "OR A,(n)": lambda di: _imem_logic_reg(di, "or"),
    "XOR (n),A": lambda di: _imem_logic_store(di, "xor"),
    "AND (m),(n)": lambda di: _imem_logic_mem(di, "and"),
    "OR (m),(n)": lambda di: _imem_logic_mem(di, "or"),
    "XOR [lmn],n": _xor_emem_const,
    "AND [lmn],n": _and_emem_const,
    "OR [lmn],n": _or_emem_const,
    "XOR (m),(n)": _xor_imem_mem,
    "ADD (m),n": _add_imem_const,
    "ADC (m),n": _adc_imem_const,
    "SBC (m),n": _sbc_imem_const,
    "CALL mn": _call_near,
    "CALLF lmn": _call_far,
    "CMP (m),(n)": _cmp_mem_mem,
    "RET": _ret_near,
    "RETF": _ret_far,
    "RETI": _reti,
    "MV r,[r3]": _ext_reg_load,
    "MV [r3],r": _ext_reg_store,
    "MV r,[(n)]": _ext_ptr_load,
    "MV [(n)],r": _ext_ptr_store,
    "INC r": lambda di: _inc_dec(di, "add"),
    "DEC r": lambda di: _inc_dec(di, "sub"),
    "MV (n),[r3]": _mv_imem_from_ext_regptr,
    "MVW (n),[r3]": _mv_imem_from_ext_regptr,
    "MVP (n),[r3]": _mv_imem_from_ext_regptr,
    "MVL (n),[r3]": _mv_imem_from_ext_regptr,
    "MV [r3],(n)": _mv_ext_from_imem_regptr,
    "MVW [r3],(n)": _mv_ext_from_imem_regptr,
    "MVP [r3],(n)": _mv_ext_from_imem_regptr,
    "MVL [r3],(n)": _mv_ext_from_imem_regptr,
    "MV [(n)],(m)": _emem_imem_to_ext,
    "MVW [(n)],(m)": _emem_imem_to_ext,
    "MVP [(n)],(m)": _emem_imem_to_ext,
    "MVL [(n)],(m)": _emem_imem_to_ext,
    "MV (m),(n)": _imem_move,
    "MVW (m),(n)": _imem_move,
    "MVP (m),(n)": _imem_move,
    "MVL (m),(n)": _mvl_imem,
    "ADC A,(n)": _adc_a_mem,
    "ADC (n),A": _adc_mem_a,
    "SBC A,(n)": _sbc_a_mem,
    "SBC (n),A": _sbc_mem_a,
    "MVLD (m),(n)": _mvl_imem,
    "ADCL (m),(n)": lambda di: _loop_carry(di, "loop_add_carry", src_is_mem=True),
    "ADCL (m),A": lambda di: _loop_carry(di, "loop_add_carry", src_is_mem=False),
    "SBCL (m),(n)": lambda di: _loop_carry(di, "loop_sub_borrow", src_is_mem=True),
    "SBCL (m),A": lambda di: _loop_carry(di, "loop_sub_borrow", src_is_mem=False),
    "DADL (m),(n)": lambda di: _loop_bcd(
        di, "loop_bcd_add", src_is_mem=True, clear_carry=True
    ),
    "DADL (m),A": lambda di: _loop_bcd(
        di, "loop_bcd_add", src_is_mem=False, clear_carry=True
    ),
    "DSBL (m),(n)": lambda di: _loop_bcd(
        di, "loop_bcd_sub", src_is_mem=True, clear_carry=False
    ),
    "DSBL (m),A": lambda di: _loop_bcd(
        di, "loop_bcd_sub", src_is_mem=False, clear_carry=False
    ),
    "DSLL (m)": lambda di: _decimal_shift(di, is_left=True),
    "DSRL (m)": lambda di: _decimal_shift(di, is_left=False),
    "PMDF (m),n": _pmdf_imm,
    "PMDF (m),A": _pmdf_reg,
    "HALT": lambda di: _with_pre(examples.halt_instr(), {}, di),
    "OFF": lambda di: _with_pre(examples.off_instr(), {}, di),
    "WAIT": lambda di: _with_pre(examples.wait_instr(), {}, di),
    "RESET": lambda di: _with_pre(examples.reset_instr(), {}, di),
    "IR": lambda di: _with_pre(examples.ir_instr(), {}, di),
    "EX (m),(n)": _imem_swap,
    "EXW (m),(n)": _imem_swap,
    "EXP (m),(n)": _imem_swap,
    "EXL (m),(n)": _imem_swap,
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
    "EX r,r": _ex_registers,
    "EX A,B": _ex_a_b,
    "INC (n)": lambda di: _inc_dec_imem(di, "add"),
    "DEC (n)": lambda di: _inc_dec_imem(di, "sub"),
    "ADD r,r": lambda di: _alu_reg_reg(di, "add", ("C", "Z")),
    "MV A,B": _mv_reg_pair,
    "MV B,A": _mv_reg_pair,
    "SUB r,r": lambda di: _alu_reg_reg(di, "sub", ("C", "Z")),
    "MV r,r": _mv_reg_pair,
    "XOR A,(n)": lambda di: _alu_a_mem(di, "xor", ("Z",)),
    "ADD A,(n)": lambda di: _alu_a_mem(di, "add", ("C", "Z")),
    "ADD (n),A": lambda di: _alu_mem_a(di, "add"),
    "SUB A,(n)": lambda di: _alu_a_mem(di, "sub", ("C", "Z")),
    "SUB (n),A": lambda di: _alu_mem_a(di, "sub"),
    "ADD (m),n": _add_imem_const,
    "SUB (m),n": _sub_imem_const,
    "CMP (m),n": _cmp_imem_const,
    "CMP [lmn],n": _cmp_emem_const,
    "CMP (m),A": _cmp_imem_reg,
    "CMPW (m),(n)": _cmpw_imem_mem,
    "CMPP (m),(n)": _cmpp_imem_mem,
    "CMP A,n": _cmp_a_imm,
    "TEST [lmn],n": _test_emem_const,
    "MVW (l),mn": _mv_imem_word_const,
    "RC": _rc,
    "SC": _sc,
    "TCL": _tcl,
    "SWAP A": _swap_a,
    "ROR A": lambda di: _rotate_a(di, "ror"),
    "ROR (n)": lambda di: _mem_rotate(di, "ror"),
    "ROL A": lambda di: _rotate_a(di, "rol"),
    "ROL (n)": lambda di: _mem_rotate(di, "rol"),
    "SHR A": lambda di: _shift_a(di, "shr"),
    "SHR (n)": lambda di: _shift_mem(di, "shr"),
    "SHL A": lambda di: _shift_a(di, "shl"),
    "SHL (n)": lambda di: _shift_mem(di, "shl"),
"MV A,A": _mv_a_a,
"BF": _nop,
}


def build(decoded: DecodedInstr) -> BuildResult:
    try:
        builder = BUILDERS[decoded.mnemonic]
    except KeyError as exc:
        raise KeyError(f"No SCIL builder for mnemonic {decoded.mnemonic}") from exc
    return builder(decoded)
