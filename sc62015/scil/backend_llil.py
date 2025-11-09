from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Protocol, Tuple

from binaryninja import RegisterName  # type: ignore
from binaryninja.lowlevelil import LowLevelILFunction, LowLevelILLabel  # type: ignore

from . import ast
from .validate import bits_to_bytes, expr_size

ExpressionResult = Tuple[int, int]  # (LLIL expression index, bit width)


class StreamReader(Protocol):
    def read(self, kind: ast.FetchKind) -> int: ...


def _mask(bits: int) -> int:
    return (1 << bits) - 1


def _reg_name(name: object) -> object:
    return name if not isinstance(name, str) else RegisterName(name)


@dataclass
class _Env:
    il: LowLevelILFunction
    stream: StreamReader
    tmps: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    def set_tmp(self, tmp: ast.Tmp, value: int) -> None:
        self.tmps[tmp.name] = (value & _mask(tmp.size), tmp.size)

    def get_tmp(self, tmp: ast.Tmp) -> Tuple[int, int]:
        if tmp.name not in self.tmps:
            raise KeyError(f"Temporary {tmp.name} not defined")
        return self.tmps[tmp.name]


def _emit_expr(expr: ast.Expr, env: _Env) -> Tuple[int, int]:
    if isinstance(expr, ast.Const):
        width_bytes = bits_to_bytes(expr.size)
        return env.il.const(width_bytes, expr.value & _mask(expr.size)), expr.size

    if isinstance(expr, ast.Tmp):
        value, size = env.get_tmp(expr)
        width_bytes = bits_to_bytes(size)
        return env.il.const(width_bytes, value), size

    if isinstance(expr, ast.Reg):
        width_bytes = bits_to_bytes(expr.size)
        return env.il.reg(width_bytes, _reg_name(expr.name)), expr.size

    if isinstance(expr, ast.Mem):
        addr_expr, addr_bits = _emit_expr(expr.addr, env)
        width_bytes = bits_to_bytes(expr.size)
        # Pointer width is inferred by BN; we simply pass the address expression.
        return env.il.load(width_bytes, addr_expr), expr.size

    if isinstance(expr, ast.UnOp):
        inner, _ = _emit_expr(expr.a, env)
        width_bytes = bits_to_bytes(expr.out_size)
        if expr.op == "neg":
            return env.il.neg_expr(width_bytes, inner), expr.out_size
        if expr.op == "not":
            return env.il.not_expr(width_bytes, inner), expr.out_size
        if expr.op == "sext":
            return env.il.sign_extend(width_bytes, inner), expr.out_size
        if expr.op == "zext":
            return env.il.zero_extend(width_bytes, inner), expr.out_size
        if expr.op == "low_part":
            return env.il.low_part(width_bytes, inner), expr.out_size
        if expr.op == "high_part":
            return env.il.high_part(width_bytes, inner), expr.out_size
        if expr.op in {"band", "bor", "bxor"}:
            if expr.param is None:
                raise ValueError(f"{expr.op} requires a constant parameter")
            const = env.il.const(width_bytes, expr.param & _mask(expr.out_size))
            op_map = {
                "band": env.il.and_expr,
                "bor": env.il.or_expr,
                "bxor": env.il.xor_expr,
            }
            return op_map[expr.op](width_bytes, inner, const), expr.out_size
        raise NotImplementedError(f"Unsupported unary op {expr.op}")

    if isinstance(expr, ast.BinOp):
        left, _ = _emit_expr(expr.a, env)
        right, _ = _emit_expr(expr.b, env)
        width_bytes = bits_to_bytes(expr.out_size)
        op_map = {
            "add": env.il.add,
            "sub": env.il.sub,
            "and": env.il.and_expr,
            "or": env.il.or_expr,
            "xor": env.il.xor_expr,
            "shl": env.il.shift_left,
            "shr": env.il.logical_shift_right,
            "sar": env.il.arith_shift_right,
            "rol": env.il.rotate_left,
            "ror": env.il.rotate_right,
        }
        if expr.op == "cmp":
            return env.il.compare_unsigned_less_than(width_bytes, left, right), expr.out_size
        if expr.op not in op_map:
            raise NotImplementedError(f"Unsupported binary op {expr.op}")
        return op_map[expr.op](width_bytes, left, right), expr.out_size

    if isinstance(expr, ast.PcRel):
        width_bytes = bits_to_bytes(expr.out_size)
        pc_expr = env.il.reg(width_bytes, RegisterName("PC"))
        base = env.il.const(width_bytes, expr.base_advance & _mask(expr.out_size))
        acc = env.il.add(width_bytes, pc_expr, base)
        if expr.disp is not None:
            disp_expr, _ = _emit_expr(expr.disp, env)
            acc = env.il.add(width_bytes, acc, disp_expr)
        return acc, expr.out_size

    if isinstance(expr, ast.Join24):
        hi_expr, _ = _emit_expr(expr.hi, env)
        mid_expr, _ = _emit_expr(expr.mid, env)
        lo_expr, _ = _emit_expr(expr.lo, env)
        width_bytes = bits_to_bytes(24)
        hi = env.il.zero_extend(width_bytes, hi_expr)
        mid = env.il.zero_extend(width_bytes, mid_expr)
        lo = env.il.zero_extend(width_bytes, lo_expr)
        hi_shifted = env.il.shift_left(width_bytes, hi, env.il.const(width_bytes, 16))
        mid_shifted = env.il.shift_left(width_bytes, mid, env.il.const(width_bytes, 8))
        partial = env.il.or_expr(width_bytes, hi_shifted, mid_shifted)
        return env.il.or_expr(width_bytes, partial, lo), 24

    if isinstance(expr, ast.TernOp):
        if expr.op != "select":
            raise NotImplementedError(f"Unsupported ternary op {expr.op}")
        cond_expr = _emit_condition(expr.cond, env)
        true_expr, _ = _emit_expr(expr.t, env)
        false_expr, _ = _emit_expr(expr.f, env)
        width_bytes = bits_to_bytes(expr.out_size)
        return env.il.if_expr(cond_expr, true_expr, false_expr), expr.out_size

    raise NotImplementedError(f"Expression {expr} not supported yet")


def _emit_condition(cond: ast.Cond, env: _Env) -> int:
    if cond.kind == "flag":
        if cond.flag is None:
            raise ValueError("flag condition missing flag name")
        return env.il.flag(cond.flag)
    if cond.a is None or cond.b is None:
        raise ValueError(f"{cond.kind} condition missing operands")
    lhs, lhs_bits = _emit_expr(cond.a, env)
    rhs, rhs_bits = _emit_expr(cond.b, env)
    width_bits = max(lhs_bits, rhs_bits)
    width_bytes = bits_to_bytes(width_bits)
    op_map = {
        "eq": env.il.compare_equal,
        "ne": env.il.compare_not_equal,
        "ltu": env.il.compare_unsigned_less_than,
        "geu": env.il.compare_unsigned_greater_equal,
        "lts": env.il.compare_signed_less_than,
        "ges": env.il.compare_signed_greater_equal,
    }
    if cond.kind not in op_map:
        raise NotImplementedError(f"Unsupported condition {cond.kind}")
    return op_map[cond.kind](width_bytes, lhs, rhs)


def _emit_if(stmt: ast.If, env: _Env) -> None:
    cond_expr = _emit_condition(stmt.cond, env)
    true_label = LowLevelILLabel()
    false_label = LowLevelILLabel()
    end_label = LowLevelILLabel() if stmt.else_ops else None

    env.il.append(env.il.if_expr(cond_expr, true_label, false_label))
    env.il.mark_label(true_label)
    for inner in stmt.then_ops:
        _emit_stmt(inner, env)
    if stmt.else_ops:
        env.il.append(env.il.goto(end_label))
    env.il.mark_label(false_label)
    if stmt.else_ops:
        for inner in stmt.else_ops:
            _emit_stmt(inner, env)
        env.il.mark_label(end_label)


def _emit_stmt(stmt: ast.Stmt, env: _Env) -> None:
    if isinstance(stmt, ast.Fetch):
        value = env.stream.read(stmt.kind)
        env.set_tmp(stmt.dst, value)
        return

    if isinstance(stmt, ast.SetReg):
        value_expr, _ = _emit_expr(stmt.value, env)
        width_bytes = bits_to_bytes(stmt.reg.size)
        env.il.append(env.il.set_reg(width_bytes, _reg_name(stmt.reg.name), value_expr))
        return

    if isinstance(stmt, ast.Store):
        addr_expr, _ = _emit_expr(stmt.dst.addr, env)
        value_expr, _ = _emit_expr(stmt.value, env)
        width_bytes = bits_to_bytes(stmt.dst.size)
        env.il.append(env.il.store(width_bytes, addr_expr, value_expr))
        return

    if isinstance(stmt, ast.SetFlag):
        # Flags are not materialized in Phase 1 backends; emit a comment for visibility.
        env.il.append(env.il.nop())
        return

    if isinstance(stmt, ast.If):
        _emit_if(stmt, env)
        return

    if isinstance(stmt, ast.Goto):
        target_expr, _ = _emit_expr(stmt.target, env)
        env.il.append(env.il.jump(target_expr))
        return

    if isinstance(stmt, ast.Call):
        target_expr, _ = _emit_expr(stmt.target, env)
        env.il.append(env.il.call(target_expr))
        return

    if isinstance(stmt, ast.Ret):
        env.il.append(env.il.ret())
        return

    if isinstance(stmt, ast.Effect):
        # Structured effects are stubs during Phase 1; keep the IL stream aligned with a nop.
        env.il.append(env.il.nop())
        return

    if isinstance(stmt, (ast.Label, ast.Comment)):
        return

    raise NotImplementedError(f"Statement {stmt} not supported")


def emit_llil(il: LowLevelILFunction, instr: ast.Instr, stream: StreamReader) -> None:
    env = _Env(il=il, stream=stream)
    for stmt in instr.semantics:
        _emit_stmt(stmt, env)
