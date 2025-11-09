from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from . import ast

_ADDR_SIZES = {
    "int": 8,
    "ext": 24,
    "code": 20,
}

_FETCH_SIZES = {
    "u8": 8,
    "s8": 8,
    "disp8": 8,
    "u16": 16,
    "u24": 24,
    "addr24": 24,
    "addr16_page": 16,
}

_FLAG_SET = {"Z", "C"}


def bits_to_bytes(bits: int) -> int:
    return (bits + 7) // 8


def expr_size(expr: ast.Expr) -> int:
    if isinstance(expr, ast.Const):
        return expr.size
    if isinstance(expr, ast.Tmp):
        return expr.size
    if isinstance(expr, ast.Reg):
        return expr.size
    if isinstance(expr, ast.Mem):
        return expr.size
    if isinstance(expr, ast.UnOp):
        return expr.out_size
    if isinstance(expr, ast.BinOp):
        return expr.out_size
    if isinstance(expr, ast.TernOp):
        return expr.out_size
    if isinstance(expr, ast.PcRel):
        return expr.out_size
    if isinstance(expr, ast.Join24):
        return 24
    raise TypeError(f"Unsupported expression: {expr!r}")


def _iter_expr_children(expr: ast.Expr) -> Iterable[ast.Expr]:
    if isinstance(expr, ast.Mem):
        yield expr.addr
    elif isinstance(expr, ast.UnOp):
        yield expr.a
    elif isinstance(expr, ast.BinOp):
        yield expr.a
        yield expr.b
    elif isinstance(expr, ast.TernOp):
        if expr.cond.a is not None:
            yield expr.cond.a
        if expr.cond.b is not None:
            yield expr.cond.b
        yield expr.t
        yield expr.f
    elif isinstance(expr, ast.PcRel):
        if expr.disp is not None:
            yield expr.disp
    elif isinstance(expr, ast.Join24):
        yield expr.hi
        yield expr.mid
        yield expr.lo


@dataclass
class _State:
    pre_latch_seen: bool = False


def _err(errors: List[str], instr: ast.Instr, message: str) -> None:
    errors.append(f"{instr.name}: {message}")


def _validate_expr(expr: ast.Expr, instr: ast.Instr, errors: List[str]) -> None:
    if isinstance(expr, ast.Mem):
        _validate_mem(expr, instr, errors)
    elif isinstance(expr, ast.Join24):
        for part_name, part in (("hi", expr.hi), ("mid", expr.mid), ("lo", expr.lo)):
            if expr_size(part) != 8:
                _err(errors, instr, f"join24 {part_name} operand must be 8 bits")
    elif isinstance(expr, ast.TernOp):
        _validate_cond(expr.cond, instr, errors)
    elif isinstance(expr, ast.PcRel):
        if expr.out_size != 20:
            _err(errors, instr, f"pc-relative expressions must be 20 bits (got {expr.out_size})")
        if expr.disp is not None:
            disp_bits = expr_size(expr.disp)
            if disp_bits not in {8, expr.out_size}:
                _err(
                    errors,
                    instr,
                    f"pc-relative displacement must be 8 or {expr.out_size} bits "
                    f"(got {disp_bits})",
                )
    for child in _iter_expr_children(expr):
        _validate_expr(child, instr, errors)


def _validate_fetch(stmt: ast.Fetch, instr: ast.Instr, errors: List[str]) -> None:
    expected = _FETCH_SIZES.get(stmt.kind)
    if expected is None:
        _err(errors, instr, f"unknown fetch kind {stmt.kind}")
        return
    if stmt.dst.size != expected:
        _err(
            errors,
            instr,
            f"fetch {stmt.kind} expects {expected} bits but tmp "
            f"{stmt.dst.name} is {stmt.dst.size} bits",
        )


def _validate_mem(mem: ast.Mem, instr: ast.Instr, errors: List[str]) -> None:
    required = _ADDR_SIZES[mem.space]
    addr_bits = expr_size(mem.addr)
    if addr_bits != required:
        _err(
            errors,
            instr,
            f"{mem.space} address must be {required} bits (got {addr_bits})",
        )


def _validate_cond(cond: ast.Cond, instr: ast.Instr, errors: List[str]) -> None:
    if cond.kind == "flag":
        if not cond.flag:
            _err(errors, instr, "flag condition missing flag name")
        elif cond.flag not in _FLAG_SET:
            _err(errors, instr, f"unsupported flag {cond.flag}")
        return
    if cond.a is None or cond.b is None:
        _err(errors, instr, f"{cond.kind} condition missing operands")
        return
    size_a = expr_size(cond.a)
    size_b = expr_size(cond.b)
    if size_a != size_b:
        _err(errors, instr, f"{cond.kind} operands must match in size ({size_a} vs {size_b})")
    _validate_expr(cond.a, instr, errors)
    _validate_expr(cond.b, instr, errors)


def _validate_stmt(stmt: ast.Stmt, instr: ast.Instr, state: _State, errors: List[str]) -> None:
    if isinstance(stmt, ast.Fetch):
        _validate_fetch(stmt, instr, errors)
        return

    if isinstance(stmt, ast.SetReg):
        reg_bits = stmt.reg.size
        value_bits = expr_size(stmt.value)
        if reg_bits != value_bits:
            _err(
                errors,
                instr,
                f"set_reg {stmt.reg.name} expects {reg_bits} bits (value is {value_bits})",
            )
        if stmt.flags:
            unsupported = [flag for flag in stmt.flags if flag not in _FLAG_SET]
            if unsupported:
                _err(errors, instr, f"unsupported flags for SetReg: {', '.join(unsupported)}")
        _validate_expr(stmt.value, instr, errors)
        return

    if isinstance(stmt, ast.Store):
        if expr_size(stmt.value) != stmt.dst.size:
            _err(
                errors,
                instr,
                "store width mismatch "
                f"({expr_size(stmt.value)} vs {stmt.dst.size})",
            )
        _validate_mem(stmt.dst, instr, errors)
        _validate_expr(stmt.value, instr, errors)
        return

    if isinstance(stmt, ast.SetFlag):
        if stmt.flag not in _FLAG_SET:
            _err(errors, instr, f"unsupported flag {stmt.flag}")
        _validate_expr(stmt.value, instr, errors)
        return

    if isinstance(stmt, ast.If):
        _validate_cond(stmt.cond, instr, errors)
        for inner in stmt.then_ops:
            _validate_stmt(inner, instr, state, errors)
        for inner in stmt.else_ops:
            _validate_stmt(inner, instr, state, errors)
        return

    if isinstance(stmt, (ast.Goto, ast.Call)):
        target_bits = expr_size(stmt.target)
        if target_bits != 20:
            kind = "call" if isinstance(stmt, ast.Call) else "goto"
            _err(errors, instr, f"{kind} target must be 20 bits (got {target_bits})")
        _validate_expr(stmt.target, instr, errors)
        return

    if isinstance(stmt, ast.Effect):
        if stmt.kind == "pre_latch":
            if state.pre_latch_seen:
                _err(errors, instr, "multiple pre_latch effects in a single instruction")
            state.pre_latch_seen = True
        for arg in stmt.args:
            _validate_expr(arg, instr, errors)
        return

    if isinstance(stmt, ast.Comment | ast.Label | ast.Ret):
        return


def validate(instr: ast.Instr) -> List[str]:
    errors: List[str] = []
    if instr.length <= 0:
        _err(errors, instr, "instruction length must be positive")
    state = _State()
    for stmt in instr.semantics:
        _validate_stmt(stmt, instr, state, errors)
    return errors
