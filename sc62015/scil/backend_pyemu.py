from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Protocol, Tuple

from . import ast

class Bus(Protocol):
    def load(self, space: ast.Space, addr: int, size: int) -> int: ...
    def store(self, space: ast.Space, addr: int, value: int, size: int) -> None: ...


class StreamReader(Protocol):
    def read(self, kind: ast.FetchKind) -> int: ...


@dataclass
class CPUState:
    regs: Dict[str, int] = field(default_factory=dict)
    flags: Dict[str, int] = field(default_factory=dict)
    pc: int = 0

    def get_reg(self, name: str, default_bits: int) -> int:
        return self.regs.get(name, 0) & ((1 << default_bits) - 1)

    def set_reg(self, name: str, value: int, bits: int) -> None:
        self.regs[name] = value & ((1 << bits) - 1)

    def set_flag(self, name: str, value: int) -> None:
        self.flags[name] = value & 1

    def get_flag(self, name: str) -> int:
        return self.flags.get(name, 0) & 1


class _TmpSpace:
    def __init__(self) -> None:
        self.values: Dict[str, Tuple[int, int]] = {}

    def set(self, tmp: ast.Tmp, value: int) -> None:
        mask = (1 << tmp.size) - 1
        self.values[tmp.name] = (value & mask, tmp.size)

    def get(self, tmp: ast.Tmp) -> Tuple[int, int]:
        if tmp.name not in self.values:
            raise KeyError(f"Temporary {tmp.name} not populated")
        return self.values[tmp.name]


def _mask(bits: int) -> int:
    return (1 << bits) - 1


def _to_signed(value: int, bits: int) -> int:
    mask = _mask(bits)
    value &= mask
    sign_bit = 1 << (bits - 1)
    return value - (1 << bits) if value & sign_bit else value


def _eval_expr(expr: ast.Expr, state: CPUState, tmp: _TmpSpace, bus: Bus) -> Tuple[int, int]:
    if isinstance(expr, ast.Const):
        return expr.value & _mask(expr.size), expr.size
    if isinstance(expr, ast.Tmp):
        return tmp.get(expr)
    if isinstance(expr, ast.Reg):
        return state.get_reg(expr.name, expr.size), expr.size
    if isinstance(expr, ast.Mem):
        addr, _ = _eval_expr(expr.addr, state, tmp, bus)
        value = bus.load(expr.space, addr, expr.size)
        return value & _mask(expr.size), expr.size
    if isinstance(expr, ast.UnOp):
        inner, _ = _eval_expr(expr.a, state, tmp, bus)
        if expr.op == "neg":
            return (-inner) & _mask(expr.out_size), expr.out_size
        if expr.op == "not":
            return (~inner) & _mask(expr.out_size), expr.out_size
        if expr.op == "sext":
            return _to_signed(inner, expr.a.size) & _mask(expr.out_size), expr.out_size
        if expr.op == "zext":
            return inner & _mask(expr.out_size), expr.out_size
        if expr.op == "low_part":
            return inner & _mask(expr.out_size), expr.out_size
        if expr.op == "high_part":
            shift = expr.a.size - expr.out_size
            return (inner >> shift) & _mask(expr.out_size), expr.out_size
        if expr.op in {"band", "bor", "bxor"} and expr.param is not None:
            param = expr.param & _mask(expr.out_size)
            if expr.op == "band":
                return (inner & param) & _mask(expr.out_size), expr.out_size
            if expr.op == "bor":
                return (inner | param) & _mask(expr.out_size), expr.out_size
            return (inner ^ param) & _mask(expr.out_size), expr.out_size
        raise NotImplementedError(f"Unary op {expr.op} not implemented")
    if isinstance(expr, ast.BinOp):
        left, _ = _eval_expr(expr.a, state, tmp, bus)
        right, _ = _eval_expr(expr.b, state, tmp, bus)
        if expr.op == "add":
            return (left + right) & _mask(expr.out_size), expr.out_size
        if expr.op == "sub":
            return (left - right) & _mask(expr.out_size), expr.out_size
        if expr.op == "and":
            return (left & right) & _mask(expr.out_size), expr.out_size
        if expr.op == "or":
            return (left | right) & _mask(expr.out_size), expr.out_size
        if expr.op == "xor":
            return (left ^ right) & _mask(expr.out_size), expr.out_size
        if expr.op == "shl":
            return (left << right) & _mask(expr.out_size), expr.out_size
        if expr.op == "shr":
            return (left >> right) & _mask(expr.out_size), expr.out_size
        if expr.op == "sar":
            return (_to_signed(left, expr.out_size) >> right) & _mask(expr.out_size), expr.out_size
        raise NotImplementedError(f"Binary op {expr.op} not implemented")
    if isinstance(expr, ast.PcRel):
        base = (state.pc + expr.base_advance) & _mask(expr.out_size)
        if expr.disp is None:
            return base, expr.out_size
        disp_value, disp_bits = _eval_expr(expr.disp, state, tmp, bus)
        signed = _to_signed(disp_value, disp_bits)
        return (base + signed) & _mask(expr.out_size), expr.out_size
    if isinstance(expr, ast.Join24):
        hi, _ = _eval_expr(expr.hi, state, tmp, bus)
        mid, _ = _eval_expr(expr.mid, state, tmp, bus)
        lo, _ = _eval_expr(expr.lo, state, tmp, bus)
        value = ((hi & 0xFF) << 16) | ((mid & 0xFF) << 8) | (lo & 0xFF)
        return value & _mask(24), 24
    if isinstance(expr, ast.TernOp):
        if expr.op != "select":
            raise NotImplementedError(f"Ternary op {expr.op} unsupported")
        cond = _eval_condition(expr.cond, state, tmp, bus)
        return _eval_expr(expr.t if cond else expr.f, state, tmp, bus)
    raise NotImplementedError(f"Expression {expr} not supported in emulator backend")


def _eval_condition(cond: ast.Cond, state: CPUState, tmp: _TmpSpace, bus: Bus) -> bool:
    if cond.kind == "flag":
        if cond.flag is None:
            raise ValueError("Flag condition missing flag name")
        return bool(state.get_flag(cond.flag))
    if cond.a is None or cond.b is None:
        raise ValueError(f"{cond.kind} condition missing operands")
    lhs, lhs_bits = _eval_expr(cond.a, state, tmp, bus)
    rhs, rhs_bits = _eval_expr(cond.b, state, tmp, bus)
    if cond.kind == "eq":
        return lhs == rhs
    if cond.kind == "ne":
        return lhs != rhs
    if cond.kind == "ltu":
        return lhs < rhs
    if cond.kind == "geu":
        return lhs >= rhs
    if cond.kind == "lts":
        return _to_signed(lhs, lhs_bits) < _to_signed(rhs, rhs_bits)
    if cond.kind == "ges":
        return _to_signed(lhs, lhs_bits) >= _to_signed(rhs, rhs_bits)
    raise NotImplementedError(f"Condition {cond.kind} unsupported")


def _exec_stmt(stmt: ast.Stmt, state: CPUState, tmp: _TmpSpace, bus: Bus, stream: StreamReader) -> None:
    if isinstance(stmt, ast.Fetch):
        tmp.set(stmt.dst, stream.read(stmt.kind))
        return
    if isinstance(stmt, ast.SetReg):
        value, _ = _eval_expr(stmt.value, state, tmp, bus)
        state.set_reg(stmt.reg.name, value, stmt.reg.size)
        if stmt.flags:
            if "Z" in stmt.flags:
                state.set_flag("Z", int(value == 0))
            if "C" in stmt.flags:
                state.set_flag("C", (value >> (stmt.reg.size - 1)) & 1)
        return
    if isinstance(stmt, ast.Store):
        addr, _ = _eval_expr(stmt.dst.addr, state, tmp, bus)
        value, _ = _eval_expr(stmt.value, state, tmp, bus)
        bus.store(stmt.dst.space, addr, value, stmt.dst.size)
        return
    if isinstance(stmt, ast.SetFlag):
        value, _ = _eval_expr(stmt.value, state, tmp, bus)
        state.set_flag(stmt.flag, value)
        return
    if isinstance(stmt, ast.If):
        branch = _eval_condition(stmt.cond, state, tmp, bus)
        block = stmt.then_ops if branch else stmt.else_ops
        for inner in block:
            _exec_stmt(inner, state, tmp, bus, stream)
        return
    if isinstance(stmt, ast.Goto):
        value, bits = _eval_expr(stmt.target, state, tmp, bus)
        state.pc = value & _mask(bits)
        return
    if isinstance(stmt, ast.Call):
        value, bits = _eval_expr(stmt.target, state, tmp, bus)
        state.pc = value & _mask(bits)
        return
    if isinstance(stmt, ast.Ret):
        return
    if isinstance(stmt, ast.Effect):
        return
    if isinstance(stmt, (ast.Label, ast.Comment)):
        return
    raise NotImplementedError(f"Statement {stmt} unsupported in emulator backend")


def step(state: CPUState, bus: Bus, instr: ast.Instr, stream: StreamReader) -> None:
    tmps = _TmpSpace()
    for stmt in instr.semantics:
        _exec_stmt(stmt, state, tmps, bus, stream)
