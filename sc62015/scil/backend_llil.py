from __future__ import annotations

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

from binaryninja import FlagName, RegisterName  # type: ignore
from binaryninja.lowlevelil import LowLevelILFunction, LowLevelILLabel  # type: ignore

from .compat_builder import CompatLLILBuilder
from . import ast
from .validate import bits_to_bytes, expr_size
from ..decoding.bind import PreLatch
from ..pysc62015.instr.opcodes import (
    RegF,
    RegIMR,
    TempExchange,
    TempIncDecHelper,
    TempMvlSrc,
    TempMvlDst,
    TempMultiByte1,
    TempMultiByte2,
    TempLoopByteResult,
    TempOverallZeroAcc,
    TempBcdDigitCarry,
    CFlag,
    ZFlag,
    CZFlag,
    lift_loop,
)
from ..pysc62015.instr.instructions import bcd_add_emul, bcd_sub_emul
from ..pysc62015.constants import INTERNAL_MEMORY_START

ExpressionResult = Tuple[int, int]


def _mask(bits: int) -> int:
    return (1 << bits) - 1


def _to_signed(value: int, bits: int) -> int:
    sign = 1 << (bits - 1)
    return (value & (sign - 1)) - (value & sign)


def _const_to_signed(const: Optional[ast.Const]) -> int:
    if const is None:
        return 0
    return _to_signed(const.value, const.size)


def _const_pointer_expr(expr: ast.Expr, env: _Env) -> Optional[int]:
    bound = _bound_const(expr, env)
    if bound is None:
        return None
    width_bytes = bits_to_bytes(bound.size)
    return env.il.const_pointer(width_bytes, bound.value & _mask(bound.size))


def _coerce_width_expr(expr: int, expr_bits: int, target_bits: int, env: _Env) -> int:
    if expr_bits == target_bits:
        return expr
    width_bytes = bits_to_bytes(target_bits)
    if expr_bits > target_bits:
        return env.il.low_part(width_bytes, expr)
    return env.il.zero_extend(width_bytes, expr)


def _capture_user_stack_pointer(env: _Env, stack_name: str) -> None:
    env.il.append(
        env.il.set_reg(
            bits_to_bytes(24),
            TempIncDecHelper,
            env.il.reg(bits_to_bytes(24), RegisterName(stack_name)),
        )
    )


def _emit_user_stack_push(env: _Env, stack_name: str, value_expr: int, width_bits: int) -> None:
    width_bytes = bits_to_bytes(width_bits)
    _capture_user_stack_pointer(env, stack_name)
    temp_expr = env.il.reg(bits_to_bytes(24), TempIncDecHelper)
    offset = env.il.const(bits_to_bytes(24), width_bytes)
    new_ptr = env.il.sub(bits_to_bytes(24), temp_expr, offset)
    env.il.append(
        env.il.set_reg(bits_to_bytes(24), RegisterName(stack_name), new_ptr)
    )
    env.il.append(env.il.store(width_bytes, new_ptr, value_expr))


def _emit_user_stack_pop(env: _Env, stack_name: str, width_bits: int) -> int:
    width_bytes = bits_to_bytes(width_bits)
    _capture_user_stack_pointer(env, stack_name)
    temp_expr = env.il.reg(bits_to_bytes(24), TempIncDecHelper)
    return env.il.load(width_bytes, temp_expr)


def _finish_user_stack_pop(env: _Env, stack_name: str, width_bits: int) -> None:
    width_bytes = bits_to_bytes(width_bits)
    temp_expr = env.il.reg(bits_to_bytes(24), TempIncDecHelper)
    env.il.append(
        env.il.set_reg(
            bits_to_bytes(24),
            RegisterName(stack_name),
            env.il.add(
                bits_to_bytes(24),
                temp_expr,
                env.il.const(bits_to_bytes(24), width_bytes),
            ),
        )
    )


def _assign_stack_pop_dest(env: _Env, dest: ast.Expr, value_expr: int, value_bits: int) -> None:
    if isinstance(dest, ast.Reg):
        if dest.name == "F":
            reg_helper = RegF()
            reg_helper.lift_assign(env.il, value_expr)
            return
        env.il.append(
            env.il.set_reg(
                bits_to_bytes(dest.size),
                RegisterName(dest.name),
                _coerce_width_expr(value_expr, value_bits, dest.size, env),
            )
        )
        return
    if isinstance(dest, ast.Mem):
        ptr_expr, _ = _resolve_mem(dest, env)
        env.il.append(
            env.il.store(
                bits_to_bytes(dest.size),
                ptr_expr,
                _coerce_width_expr(value_expr, value_bits, dest.size, env),
            )
        )
        return
    raise NotImplementedError("Unsupported pop destination")


def _loop_int_offset(ptr: ast.LoopIntPtr, env: _Env) -> int:
    info = _const_value(ptr.offset, env)
    if info is None:
        raise ValueError("Loop pointer offset must be constant")
    return info[0] & 0xFF


def _update_loop_int_pointer(env: _Env, temp_reg, step_signed: int) -> None:
    width_bytes = bits_to_bytes(24)
    current = env.il.reg(width_bytes, temp_reg)
    base = env.il.const(width_bytes, INTERNAL_MEMORY_START)
    delta = abs(step_signed)
    step_expr = env.il.const(width_bytes, delta)
    update_op = env.il.add if step_signed >= 0 else env.il.sub
    new_addr = update_op(width_bytes, current, step_expr)
    offset = env.il.sub(width_bytes, new_addr, base)
    wrapped_offset = env.il.and_expr(width_bytes, offset, env.il.const(width_bytes, 0xFF))
    wrapped_addr = env.il.add(width_bytes, base, wrapped_offset)
    env.il.append(env.il.set_reg(width_bytes, temp_reg, wrapped_addr))


def _advance_temp_pointer(env: _Env, temp_reg, direction: int) -> None:
    width_bytes = bits_to_bytes(24)
    current = env.il.reg(width_bytes, temp_reg)
    step = env.il.const(width_bytes, abs(direction))
    op = env.il.add if direction >= 0 else env.il.sub
    updated = op(width_bytes, current, step)
    env.il.append(env.il.set_reg(width_bytes, temp_reg, updated))


def _emit_loop_move_int_body(env: _Env, width_bits: int, step_signed: int) -> None:
    loop_bytes = bits_to_bytes(16)
    zero = env.il.const(loop_bytes, 0)
    loop_label = LowLevelILLabel()
    exit_label = LowLevelILLabel()
    i_reg = RegisterName("I")

    env.il.append(
        env.il.if_expr(
            env.il.compare_equal(loop_bytes, env.il.reg(loop_bytes, i_reg), zero),
            exit_label,
            loop_label,
        )
    )
    env.il.mark_label(loop_label)

    width_bytes = bits_to_bytes(width_bits)
    src_ptr = env.il.reg(bits_to_bytes(24), TempMvlSrc)
    dst_ptr = env.il.reg(bits_to_bytes(24), TempMvlDst)
    value = env.il.load(width_bytes, src_ptr)
    env.il.append(env.il.store(width_bytes, dst_ptr, value))

    _update_loop_int_pointer(env, TempMvlDst, step_signed)
    _update_loop_int_pointer(env, TempMvlSrc, step_signed)

    decremented = env.il.sub(
        loop_bytes,
        env.il.reg(loop_bytes, i_reg),
        env.il.const(1, 1),
    )
    env.il.append(env.il.set_reg(loop_bytes, i_reg, decremented))
    env.il.append(
        env.il.if_expr(
            env.il.compare_equal(loop_bytes, env.il.reg(loop_bytes, i_reg), zero),
            exit_label,
            loop_label,
        )
    )
    env.il.mark_label(exit_label)


@dataclass
class _Env:
    il: LowLevelILFunction
    binder: Dict[str, ast.Const]
    compat: CompatLLILBuilder
    addr: int
    pre_latch: Optional[PreLatch] = None
    tmps: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    imem_index: int = 0

    def bind_fetches(self) -> None:
        for name, const in self.binder.items():
            self.tmps[name] = (const.value, const.size)

    def set_tmp(self, tmp: ast.Tmp, value: int, size: int) -> None:
        self.tmps[tmp.name] = (value, size)

    def get_tmp(self, tmp: ast.Tmp) -> Tuple[int, int]:
        if tmp.name not in self.tmps:
            raise KeyError(f"Temporary {tmp.name} not bound")
        return self.tmps[tmp.name]

    def next_imem_mode(self) -> str:
        mode = "(BP+n)"
        if self.pre_latch is not None:
            if self.imem_index == 0 and self.pre_latch.first:
                mode = self.pre_latch.first
            elif self.imem_index == 1 and self.pre_latch.second:
                mode = self.pre_latch.second
        self.imem_index += 1
        return mode


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
            bits = expr.out_size
        elif expr.op == "zext":
            bits = expr.a.size
        return value, bits
    if isinstance(expr, ast.PcRel):
        base = (env.addr + expr.base_advance) & _mask(expr.out_size)
        if expr.disp is None:
            return base, expr.out_size
        disp_info = _const_value(expr.disp, env)
        if disp_info is None:
            return None
        signed, bits = disp_info
        if signed & (1 << (bits - 1)):
            signed -= 1 << bits
        return (base + signed) & _mask(expr.out_size), expr.out_size
    bound = _bound_const(expr, env)
    if bound is None:
        return None
    return bound.value, bound.size


def _resolve_mem(mem: ast.Mem, env: _Env) -> Tuple[int, int]:
    if mem.space == "ext":
        addr_expr, _ = _emit_expr(mem.addr, env)
        bound = _bound_const(mem.addr, env)
        if bound is not None:
            addr_expr = env.il.const_pointer(bits_to_bytes(24), bound.value & _mask(24))
        return addr_expr, 24
    if mem.space == "int":
        if isinstance(mem.addr, ast.Const) and mem.addr.size > 8:
            return env.il.const_pointer(bits_to_bytes(mem.addr.size), mem.addr.value), mem.addr.size
        offset_expr, _ = _emit_expr(mem.addr, env)
        mode = env.next_imem_mode()
        ptr = env.compat.imem_address(mode, offset_expr)
        return ptr, 24
    raise NotImplementedError(f"Unknown memory space {mem.space}")


def _emit_value_with_flags(
    expr: ast.Expr, env: _Env, flags: Optional[Tuple[str, ...]]
) -> Tuple[int, int]:
    if not flags:
        return _emit_expr(expr, env)
    if isinstance(expr, ast.BinOp) and expr.op in {"add", "sub", "and", "or", "xor"}:
        left, _ = _emit_expr(expr.a, env)
        right, _ = _emit_expr(expr.b, env)
        value = env.compat.binop_with_flags(expr.op, expr.out_size, left, right, flags)
        return value, expr.out_size
    raise NotImplementedError(f"Flagged emission not supported for {expr}")


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
        return env.il.reg(width_bytes, RegisterName(expr.name)), expr.size

    if isinstance(expr, ast.Flag):
        return env.il.flag(FlagName(expr.name)), 1

    if isinstance(expr, ast.Mem):
        ptr_expr, _ = _resolve_mem(expr, env)
        width_bytes = bits_to_bytes(expr.size)
        return env.il.load(width_bytes, ptr_expr), expr.size

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
                if isinstance(expr.a, ast.UnOp) and expr.a.op == "zext" and isinstance(expr.a.a, ast.Const):
                    lo_bits = expr.a.a.size
                if isinstance(expr.b, ast.UnOp) and expr.b.op == "zext" and isinstance(expr.b.a, ast.Const):
                    hi_bits = expr.b.a.size
                # Ensure low operand is first (16-bit)
                if lo_bits > hi_bits:
                    lo_val, hi_val = hi_val, lo_val
                    lo_bits, hi_bits = hi_bits, lo_bits
                lo_expr = env.il.const(bits_to_bytes(lo_bits), lo_val & _mask(lo_bits))
                hi_expr = env.il.const(bits_to_bytes(hi_bits), hi_val & _mask(hi_bits))
                return env.il.or_expr(width_bytes, lo_expr, hi_expr), expr.out_size
        if expr.op in {"add", "sub"}:
            left_const = _const_value(expr.a, env)
            right_const = _const_value(expr.b, env)
            if left_const and right_const:
                left_val, _ = left_const
                right_val, _ = right_const
                value = left_val + right_val if expr.op == "add" else left_val - right_val
                return (
                    env.il.const(width_bytes, value & _mask(expr.out_size)),
                    expr.out_size,
                )
        left, left_bits = _emit_expr(expr.a, env)
        right, right_bits = _emit_expr(expr.b, env)
        if expr.op in {"eq", "ne"}:
            width_bits = max(left_bits, right_bits)
            cmp_map = {
                "eq": env.il.compare_equal,
                "ne": env.il.compare_not_equal,
            }
            return cmp_map[expr.op](bits_to_bytes(width_bits), left, right), 1
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
        flag_expr = env.il.flag(FlagName(cond.flag))
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


def _emit_ret_near(env: _Env) -> None:
    pop_val = env.il.pop(2)
    high = env.il.and_expr(
        3,
        env.il.reg(3, RegisterName("PC")),
        env.il.const(3, 0xFF0000),
    )
    env.il.append(env.il.ret(env.il.or_expr(3, pop_val, high)))


def _emit_reti(env: _Env) -> None:
    reg_imr = RegIMR()
    reg_imr.lift_assign(env.il, env.il.pop(1))
    reg_f = RegF()
    reg_f.lift_assign(env.il, env.il.pop(1))
    env.il.append(env.il.ret(env.il.pop(3)))


def _emit_effect_stmt(stmt: ast.Effect, env: _Env) -> None:
    kind = stmt.kind
    if kind == "push_ret16":
        const_info = _const_value(stmt.args[0], env)
        if const_info is not None:
            value, _ = const_info
            env.il.append(
                env.il.push(2, env.il.const(bits_to_bytes(16), value))
            )
            return
        value_expr, bits = _emit_expr(stmt.args[0], env)
        coerced = _coerce_width_expr(value_expr, bits, 16, env)
        env.il.append(env.il.push(2, coerced))
        return
    if kind == "push_ret24":
        # Legacy CALLF emitted BN's call primitive which already models the stack effect.
        # Keep the legacy LLIL shape by omitting an explicit push here.
        return
    if kind == "goto_page_join":
        lo_const = _const_value(stmt.args[0], env)
        page_const = _const_value(stmt.args[1], env)
        if lo_const is not None and page_const is not None:
            lo_val, _ = lo_const
            page_val, _ = page_const
            joined = ((page_val & 0xF0000) | (lo_val & 0xFFFF)) & _mask(20)
            env.il.append(env.il.jump(env.il.const_pointer(3, joined)))
            return
        lo_expr, _ = _emit_expr(stmt.args[0], env)
        page_expr, _ = _emit_expr(stmt.args[1], env)
        target = env.compat.paged_join(page_expr, lo_expr)
        env.il.append(env.il.jump(target))
        return
    if kind == "goto_far24":
        ptr_expr = _const_pointer_expr(stmt.args[0], env)
        if ptr_expr is None:
            ptr_expr, _ = _emit_expr(stmt.args[0], env)
        env.il.append(env.il.call(ptr_expr))
        return
    if kind == "ret_near":
        _emit_ret_near(env)
        return
    if kind == "ret_far":
        env.il.append(env.il.ret(env.il.pop(3)))
        return
    if kind == "reti":
        _emit_reti(env)
        return
    if kind == "push_bytes":
        width_info = _const_value(stmt.args[2], env)
        if width_info is None:
            raise ValueError("push_bytes requires constant width")
        width_bits, _ = width_info
        value_expr, value_bits = _emit_expr(stmt.args[1], env)
        coerced = _coerce_width_expr(value_expr, value_bits, width_bits, env)
        stack = stmt.args[0]
        if not isinstance(stack, ast.Reg):
            raise TypeError("push_bytes requires stack register")
        if stack.name == "S":
            env.il.append(env.il.push(bits_to_bytes(width_bits), coerced))
            return
        if stack.name == "U":
            _emit_user_stack_push(env, stack.name, coerced, width_bits)
            return
        raise NotImplementedError(f"push_bytes unsupported stack {stack.name}")
    if kind == "pop_bytes":
        width_info = _const_value(stmt.args[2], env)
        if width_info is None:
            raise ValueError("pop_bytes requires constant width")
        width_bits, _ = width_info
        stack = stmt.args[0]
        if not isinstance(stack, ast.Reg):
            raise TypeError("pop_bytes requires stack register")
        if stack.name == "S":
            value_expr = env.il.pop(bits_to_bytes(width_bits))
            _assign_stack_pop_dest(env, stmt.args[1], value_expr, width_bits)
            return
        if stack.name == "U":
            value_expr = _emit_user_stack_pop(env, stack.name, width_bits)
            _assign_stack_pop_dest(env, stmt.args[1], value_expr, width_bits)
            _finish_user_stack_pop(env, stack.name, width_bits)
            return
        else:
            raise NotImplementedError(f"pop_bytes unsupported stack {stack.name}")
    if kind == "loop_move":
        dst_ptr = stmt.args[1]
        src_ptr = stmt.args[2]
        if not isinstance(dst_ptr, ast.LoopIntPtr) or not isinstance(src_ptr, ast.LoopIntPtr):
            raise NotImplementedError("loop_move currently supports internal-memory operands only")
        dst_offset = _loop_int_offset(dst_ptr, env)
        src_offset = _loop_int_offset(src_ptr, env)
        dst_mode = env.next_imem_mode()
        src_mode = env.next_imem_mode()
        dst_addr = env.compat.imem_address(dst_mode, env.il.const(1, dst_offset))
        src_addr = env.compat.imem_address(src_mode, env.il.const(1, src_offset))
        env.il.append(env.il.set_reg(bits_to_bytes(24), TempMvlDst, dst_addr))
        env.il.append(env.il.set_reg(bits_to_bytes(24), TempMvlSrc, src_addr))
        step_info = _const_value(stmt.args[3], env)
        width_info = _const_value(stmt.args[4], env)
        if step_info is None or width_info is None:
            raise ValueError("loop_move requires constant step and width")
        step_signed = _to_signed(step_info[0], stmt.args[3].size)
        width_bits = width_info[0]
        _emit_loop_move_int_body(env, width_bits, step_signed)
        return
    if kind in {"loop_add_carry", "loop_sub_borrow"}:
        dst_ptr = stmt.args[1]
        src_ptr = stmt.args[2]
        if not isinstance(dst_ptr, ast.LoopIntPtr):
            raise NotImplementedError("loop_add_carry requires internal memory destination")
        dst_offset = _loop_int_offset(dst_ptr, env)
        dst_mode = env.next_imem_mode()
        dst_addr = env.compat.imem_address(dst_mode, env.il.const(1, dst_offset))
        env.il.append(env.il.set_reg(bits_to_bytes(24), TempMultiByte1, dst_addr))

        src_is_mem = isinstance(src_ptr, ast.LoopIntPtr)
        src_expr = None
        src_bits = 0
        if src_is_mem:
            src_offset = _loop_int_offset(src_ptr, env)
            src_mode = env.next_imem_mode()
            src_addr = env.compat.imem_address(src_mode, env.il.const(1, src_offset))
            env.il.append(env.il.set_reg(bits_to_bytes(24), TempMultiByte2, src_addr))
        else:
            src_expr, src_bits = _emit_expr(src_ptr, env)

        width_info = _const_value(stmt.args[4], env)
        if width_info is None:
            raise ValueError("loop carry width missing")
        width_bits = width_info[0]
        width_bytes = bits_to_bytes(width_bits)
        env.il.append(
            env.il.set_reg(
                width_bytes,
                TempOverallZeroAcc,
                env.il.const(width_bytes, 0),
            )
        )

        with lift_loop(env.il):
            dst_ptr_reg = env.il.reg(bits_to_bytes(24), TempMultiByte1)
            dst_byte = env.il.load(width_bytes, dst_ptr_reg)

            if src_is_mem:
                src_ptr_reg = env.il.reg(bits_to_bytes(24), TempMultiByte2)
                src_byte = env.il.load(width_bytes, src_ptr_reg)
            else:
                assert src_expr is not None
                src_byte = _coerce_width_expr(src_expr, src_bits, width_bits, env)

            initial_carry = env.il.flag(CFlag)
            if kind == "loop_sub_borrow":
                term = env.il.add(width_bytes, src_byte, initial_carry)
                main = env.il.sub(width_bytes, dst_byte, term, CZFlag)
            else:
                term = env.il.add(width_bytes, src_byte, initial_carry)
                main = env.il.add(width_bytes, dst_byte, term, CZFlag)

            env.il.append(env.il.set_reg(width_bytes, TempLoopByteResult, main))
            byte_value = env.il.reg(width_bytes, TempLoopByteResult)
            env.il.append(env.il.store(width_bytes, dst_ptr_reg, byte_value))

            overall = env.il.reg(width_bytes, TempOverallZeroAcc)
            env.il.append(
                env.il.set_reg(
                    width_bytes,
                    TempOverallZeroAcc,
                    env.il.or_expr(width_bytes, overall, byte_value),
                )
            )

            env.il.append(
                env.il.set_reg(
                    bits_to_bytes(24),
                    TempMultiByte1,
                    env.il.add(
                        bits_to_bytes(24),
                        dst_ptr_reg,
                        env.il.const(bits_to_bytes(24), width_bytes),
                    ),
                )
            )

            if src_is_mem:
                src_ptr_reg = env.il.reg(bits_to_bytes(24), TempMultiByte2)
                env.il.append(
                    env.il.set_reg(
                        bits_to_bytes(24),
                        TempMultiByte2,
                        env.il.add(
                            bits_to_bytes(24),
                            src_ptr_reg,
                            env.il.const(bits_to_bytes(24), width_bytes),
                        ),
                    )
                )

        overall = env.il.reg(width_bytes, TempOverallZeroAcc)
        env.il.append(
            env.il.set_flag(
                ZFlag,
                env.il.compare_equal(
                    width_bytes,
                    overall,
                    env.il.const(width_bytes, 0),
                ),
            )
        )
        return
    if kind in {"loop_bcd_add", "loop_bcd_sub"}:
        dst_ptr = stmt.args[1]
        src_op = stmt.args[2]
        if not isinstance(dst_ptr, ast.LoopIntPtr):
            raise NotImplementedError("loop_bcd requires internal memory destination")
        dst_offset = _loop_int_offset(dst_ptr, env)
        dst_mode = env.next_imem_mode()
        dst_addr = env.compat.imem_address(dst_mode, env.il.const(1, dst_offset))
        env.il.append(env.il.set_reg(bits_to_bytes(24), TempMultiByte1, dst_addr))

        src_is_mem = isinstance(src_op, ast.LoopIntPtr)
        if src_is_mem:
            src_offset = _loop_int_offset(src_op, env)
            src_mode = env.next_imem_mode()
            src_addr = env.compat.imem_address(src_mode, env.il.const(1, src_offset))
            env.il.append(env.il.set_reg(bits_to_bytes(24), TempMultiByte2, src_addr))
        else:
            if not isinstance(src_op, ast.Reg):
                raise NotImplementedError("loop_bcd register source required")
            src_reg = RegisterName(src_op.name)
            src_bits = src_op.size

        width_info = _const_value(stmt.args[4], env)
        if width_info is None:
            raise ValueError("loop_bcd width missing")
        width_bits = width_info[0]
        width_bytes = bits_to_bytes(width_bits)

        direction_info = _const_value(stmt.args[5], env)
        if direction_info is None:
            raise ValueError("loop_bcd direction missing")
        direction = _to_signed(direction_info[0], stmt.args[5].size)

        clear_info = _const_value(stmt.args[6], env)
        clear_carry = bool(clear_info[0] if clear_info else 0)
        if clear_carry:
            env.il.append(env.il.set_flag(CFlag, env.il.const(1, 0)))

        env.il.append(
            env.il.set_reg(
                width_bytes,
                TempOverallZeroAcc,
                env.il.const(width_bytes, 0),
            )
        )

        with lift_loop(env.il):
            dst_ptr_reg = env.il.reg(bits_to_bytes(24), TempMultiByte1)
            dst_byte = env.il.load(width_bytes, dst_ptr_reg)

            if src_is_mem:
                src_ptr_reg = env.il.reg(bits_to_bytes(24), TempMultiByte2)
                src_byte = env.il.load(width_bytes, src_ptr_reg)
            else:
                src_byte = env.il.reg(width_bytes, src_reg)

            if kind == "loop_bcd_sub":
                result_operand = bcd_sub_emul(env.il, width_bytes, dst_byte, src_byte)
            else:
                result_operand = bcd_add_emul(env.il, width_bytes, dst_byte, src_byte)
            result_expr = result_operand.lift(env.il)

            env.il.append(env.il.store(width_bytes, dst_ptr_reg, result_expr))

            overall = env.il.reg(width_bytes, TempOverallZeroAcc)
            env.il.append(
                env.il.set_reg(
                    width_bytes,
                    TempOverallZeroAcc,
                    env.il.or_expr(width_bytes, overall, result_expr),
                )
            )

            _advance_temp_pointer(env, TempMultiByte1, direction)
            if src_is_mem:
                _advance_temp_pointer(env, TempMultiByte2, direction)

        env.il.append(
            env.il.set_flag(
                ZFlag,
                env.il.compare_equal(
                    width_bytes,
                    env.il.reg(width_bytes, TempOverallZeroAcc),
                    env.il.const(width_bytes, 0),
                ),
            )
        )
        return
    if kind == "decimal_shift":
        ptr = stmt.args[1]
        if not isinstance(ptr, ast.LoopIntPtr):
            raise NotImplementedError("decimal_shift requires internal memory operand")
        offset = _loop_int_offset(ptr, env)
        mode = env.next_imem_mode()
        addr = env.compat.imem_address(mode, env.il.const(1, offset))
        env.il.append(env.il.set_reg(bits_to_bytes(24), TempMultiByte1, addr))
        env.il.append(env.il.set_reg(1, TempBcdDigitCarry, env.il.const(1, 0)))
        env.il.append(env.il.set_reg(1, TempOverallZeroAcc, env.il.const(1, 0)))

        direction_info = _const_value(stmt.args[2], env)
        if direction_info is None:
            raise ValueError("decimal_shift direction missing")
        direction = _to_signed(direction_info[0], stmt.args[2].size)
        left_info = _const_value(stmt.args[3], env)
        if left_info is None:
            raise ValueError("decimal_shift needs direction flag")
        is_left = bool(left_info[0])

        with lift_loop(env.il):
            addr_reg = env.il.reg(bits_to_bytes(24), TempMultiByte1)
            current_byte = env.il.load(1, addr_reg)
            low_nibble = env.il.and_expr(1, current_byte, env.il.const(1, 0x0F))
            high_nibble = env.il.logical_shift_right(1, current_byte, env.il.const(1, 4))

            shift_part = env.il.shift_left(1, low_nibble, env.il.const(1, 4))
            carry_part = env.il.reg(1, TempBcdDigitCarry)
            next_carry = high_nibble
            addr_update = env.il.sub(bits_to_bytes(24), addr_reg, env.il.const(bits_to_bytes(24), 1))

            if not is_left:
                shift_part, high_nibble = high_nibble, shift_part
                carry_part = env.il.shift_left(1, carry_part, env.il.const(1, 4))
                next_carry = low_nibble
                addr_update = env.il.add(bits_to_bytes(24), addr_reg, env.il.const(bits_to_bytes(24), 1))

            shifted = env.il.or_expr(1, shift_part, carry_part)
            env.il.append(env.il.store(1, addr_reg, shifted))
            env.il.append(env.il.set_reg(1, TempBcdDigitCarry, next_carry))

            overall = env.il.reg(1, TempOverallZeroAcc)
            env.il.append(
                env.il.set_reg(
                    1,
                    TempOverallZeroAcc,
                    env.il.or_expr(1, overall, shifted),
                )
            )

            env.il.append(env.il.set_reg(bits_to_bytes(24), TempMultiByte1, addr_update))

        env.il.append(
            env.il.set_flag(
                ZFlag,
                env.il.compare_equal(
                    1,
                    env.il.reg(1, TempOverallZeroAcc),
                    env.il.const(1, 0),
                ),
            )
        )
        return
    raise NotImplementedError(f"Effect {kind} not supported")


def _emit_stmt(stmt: ast.Stmt, env: _Env) -> None:
    if isinstance(stmt, ast.Fetch):
        bound = env.binder.get(stmt.dst.name)
        if bound is None:
            raise KeyError(f"Fetch {stmt.dst.name} not provided")
        env.set_tmp(stmt.dst, bound.value, bound.size)
        return

    if isinstance(stmt, ast.SetReg):
        value_expr, _ = _emit_value_with_flags(stmt.value, env, stmt.flags)
        env.compat.set_reg_with_flags(
            stmt.reg.name, stmt.reg.size, value_expr, stmt.flags
        )
        return

    if isinstance(stmt, ast.Store):
        addr_expr, _ = _resolve_mem(stmt.dst, env)
        value_expr, _ = _emit_expr(stmt.value, env)
        width_bytes = bits_to_bytes(stmt.dst.size)
        env.il.append(env.il.store(width_bytes, addr_expr, value_expr))
        return

    if isinstance(stmt, ast.SetFlag):
        value_expr, _ = _emit_expr(stmt.value, env)
        env.il.append(env.il.set_flag(FlagName(stmt.flag), value_expr))
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
        _emit_effect_stmt(stmt, env)
        return

    if isinstance(stmt, ast.ExtRegLoad):
        disp = _const_to_signed(stmt.disp)
        env.compat.ext_reg_load(
            stmt.dst.name,
            stmt.dst.size,
            stmt.ptr.name,
            stmt.mode,
            disp,
        )
        return

    if isinstance(stmt, ast.ExtRegStore):
        disp = _const_to_signed(stmt.disp)
        env.compat.ext_reg_store(
            stmt.src.name,
            stmt.src.size,
            stmt.ptr.name,
            stmt.mode,
            disp,
        )
        return

    if isinstance(stmt, ast.IntMemSwap):
        _emit_int_mem_swap(stmt, env)
        return

    if isinstance(stmt, ast.ExtRegToIntMem):
        disp = _const_to_signed(stmt.disp)
        value = env.compat.ext_reg_read_value(stmt.ptr.name, stmt.mode, stmt.dst.size, disp)
        addr_expr, _ = _resolve_mem(stmt.dst, env)
        env.il.append(env.il.store(bits_to_bytes(stmt.dst.size), addr_expr, value))
        return

    if isinstance(stmt, ast.IntMemToExtReg):
        disp = _const_to_signed(stmt.disp)
        value_expr, _ = _emit_expr(stmt.src, env)
        env.compat.ext_reg_store_value(stmt.ptr.name, stmt.mode, stmt.src.size, disp, value_expr)
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
    pre_applied: Optional[PreLatch] = None,
) -> None:
    env = _Env(il=il, binder=binder, compat=compat, addr=addr, pre_latch=pre_applied)
    env.bind_fetches()
    for stmt in instr.semantics:
        _emit_stmt(stmt, env)


def _emit_int_mem_swap(stmt: ast.IntMemSwap, env: _Env) -> None:
    width_bytes = bits_to_bytes(stmt.width)
    left_mem = ast.Mem("int", stmt.left, stmt.width)
    right_mem = ast.Mem("int", stmt.right, stmt.width)

    # Load left operand into temp register
    left_ptr, _ = _resolve_mem(left_mem, env)
    left_value = env.il.load(width_bytes, left_ptr)
    env.il.append(env.il.set_reg(width_bytes, TempExchange, left_value))

    # Store left <- right
    left_ptr_store, _ = _resolve_mem(left_mem, env)
    right_value, _ = _emit_expr(ast.Mem("int", stmt.right, stmt.width), env)
    env.il.append(env.il.store(width_bytes, left_ptr_store, right_value))

    # Store right <- temp
    right_ptr_store, _ = _resolve_mem(right_mem, env)
    env.il.append(
        env.il.store(
            width_bytes,
            right_ptr_store,
            env.il.reg(width_bytes, TempExchange),
        )
    )
