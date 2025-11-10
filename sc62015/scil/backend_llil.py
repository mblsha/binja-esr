from __future__ import annotations

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

from binaryninja import RegisterName  # type: ignore
from binaryninja.lowlevelil import LowLevelILFunction, LowLevelILLabel  # type: ignore

from .compat_builder import CompatLLILBuilder
from . import ast
from .validate import bits_to_bytes, expr_size

ExpressionResult = Tuple[int, int]


def _mask(bits: int) -> int:
    return (1 << bits) - 1


@dataclass
class _Env:
    il: LowLevelILFunction
    binder: Dict[str, ast.Const]
    compat: CompatLLILBuilder
    addr: int
    tmps: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    def bind_fetches(self) -> None:
        for name, const in self.binder.items():
            self.tmps[name] = (const.value, const.size)

    def set_tmp(self, tmp: ast.Tmp, value: int, size: int) -> None:
        self.tmps[tmp.name] = (value, size)

    def get_tmp(self, tmp: ast.Tmp) -> Tuple[int, int]:
        if tmp.name not in self.tmps:
            raise KeyError(f"Temporary {tmp.name} not bound")
        return self.tmps[tmp.name]


def _bound_const(expr: ast.Expr, env: _Env) -> Optional[ast.Const]:
    if isinstance(expr, ast.Tmp):
        return env.binder.get(expr.name)
    if isinstance(expr, ast.Const):
        return expr
    if isinstance(expr, ast.UnOp) and expr.op in {"zext", "sext"}:
        base = _bound_const(expr.a, env)
        if base is None:
            return None
        value = base.value
        if expr.op == "sext":
            if value & (1 << (base.size - 1)):
                value = value - (1 << base.size)
        return ast.Const(value & _mask(expr.out_size), expr.out_size)
    return None


def _const_value(expr: ast.Expr, env: _Env) -> Optional[Tuple[int, int]]:
    if isinstance(expr, ast.UnOp) and expr.op in {"zext", "sext"}:
        inner = _const_value(expr.a, env)
        if inner is None:
            return None
        value, bits = inner
        if expr.op == "sext" and bits > 0:
            if value & (1 << (bits - 1)):
                value = value - (1 << bits)
        return value, bits
    bound = _bound_const(expr, env)
    if bound is None:
        return None
    return bound.value, bound.size


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
        if expr.space == "ext":
            bound = _bound_const(expr.addr, env)
            if bound is not None:
                addr_expr = env.il.const_pointer(3, bound.value & _mask(24))
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
            bound = _bound_const(expr.a, env)
            if bound is not None:
                return (
                    env.il.const(width_bytes, bound.value & _mask(expr.out_size)),
                    expr.out_size,
                )
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
        width_bytes = bits_to_bytes(expr.out_size)
        if expr.op == "or":
            left_const = _const_value(expr.a, env)
            right_const = _const_value(expr.b, env)
            if left_const and right_const:
                lo_val, lo_bits = left_const
                hi_val, hi_bits = right_const
                # Ensure low operand is first (16-bit)
                if lo_bits > hi_bits:
                    lo_val, hi_val = hi_val, lo_val
                    lo_bits, hi_bits = hi_bits, lo_bits
                lo_expr = env.il.const(bits_to_bytes(lo_bits), lo_val & _mask(lo_bits))
                hi_expr = env.il.const(bits_to_bytes(hi_bits), hi_val & _mask(hi_bits))
                return env.il.or_expr(width_bytes, lo_expr, hi_expr), expr.out_size
        left, _ = _emit_expr(expr.a, env)
        right, _ = _emit_expr(expr.b, env)
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
        base = (env.addr + expr.base_advance) & _mask(expr.out_size)
        if expr.disp is not None:
            const_info = _const_value(expr.disp, env)
            if const_info is not None:
                signed, bits = const_info
                if signed & (1 << (bits - 1)):
                    signed -= 1 << bits
                target = (base + signed) & _mask(expr.out_size)
                return env.il.const(bits_to_bytes(expr.out_size), target), expr.out_size
            disp_expr, _ = _emit_expr(expr.disp, env)
            return env.compat.pc_relative(expr.base_advance, disp_expr), expr.out_size
        return env.il.const(bits_to_bytes(expr.out_size), base), expr.out_size

    if isinstance(expr, ast.Join24):
        parts = []
        for sub in (expr.hi, expr.mid, expr.lo):
            bound = _bound_const(sub, env)
            if bound is None:
                break
            parts.append(bound.value & 0xFF)
        if len(parts) == 3:
            value = (parts[0] << 16) | (parts[1] << 8) | parts[2]
            return env.il.const_pointer(3, value), 24
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
        flag_expr = env.il.flag(cond.flag)
        one = env.il.const(1, 1)
        return env.il.compare_equal(1, flag_expr, one)
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
        bound = env.binder.get(stmt.dst.name)
        if bound is None:
            raise KeyError(f"Fetch {stmt.dst.name} not provided")
        env.set_tmp(stmt.dst, bound.value, bound.size)
        return

    if isinstance(stmt, ast.SetReg):
        value_expr, _ = _emit_expr(stmt.value, env)
        env.compat.set_reg_with_flags(stmt.reg.name, stmt.reg.size, value_expr, stmt.flags)
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


def emit_llil(
    il: LowLevelILFunction,
    instr: ast.Instr,
    binder: Dict[str, ast.Const],
    compat: CompatLLILBuilder,
    addr: int,
) -> None:
    env = _Env(il=il, binder=binder, compat=compat, addr=addr)
    env.bind_fetches()
    for stmt in instr.semantics:
        _emit_stmt(stmt, env)
