from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, Tuple

from ..decoding.bind import PreLatch
from ..pysc62015.instr.opcodes import IMEMRegisters, INTERRUPT_VECTOR_ADDR
from ..pysc62015.constants import PC_MASK, INTERNAL_MEMORY_START
from . import ast


class Bus(Protocol):
    def load(self, space: ast.Space, addr: int, size: int) -> int: ...
    def store(self, space: ast.Space, addr: int, value: int, size: int) -> None: ...


class StreamReader(Protocol):
    def read(self, tmp: ast.Tmp, kind: ast.FetchKind) -> int: ...


@dataclass
class CPUState:
    regs: Dict[str, int] = field(default_factory=dict)
    flags: Dict[str, int] = field(default_factory=dict)
    pc: int = 0

    def get_reg(self, name: str, default_bits: int) -> int:
        if name == "PC":
            return self.pc & PC_MASK
        return self.regs.get(name, 0) & ((1 << default_bits) - 1)

    def set_reg(self, name: str, value: int, bits: int) -> None:
        if name == "PC":
            self.pc = value & PC_MASK
            return
        self.regs[name] = value & ((1 << bits) - 1)

    def set_flag(self, name: str, value: int) -> None:
        self.flags[name] = value & 1

    def get_flag(self, name: str) -> int:
        return self.flags.get(name, 0) & 1


@dataclass
class _Env:
    state: CPUState
    bus: Bus
    binder: Dict[str, ast.Const]
    pre_latch: Optional[PreLatch] = None
    tmps: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    imem_index: int = 0

    def set_tmp(self, tmp: ast.Tmp, value: int) -> None:
        mask = _mask(tmp.size)
        self.tmps[tmp.name] = (value & mask, tmp.size)

    def get_tmp(self, tmp: ast.Tmp) -> Tuple[int, int]:
        if tmp.name not in self.tmps:
            raise KeyError(f"Temporary {tmp.name} not populated")
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


def _mask(bits: int) -> int:
    return (1 << bits) - 1


_PTR_MODE_SIMPLE = 0
_PTR_MODE_POST_INC = 1
_PTR_MODE_PRE_DEC = 2
_PTR_MODE_OFFSET = 3
_ADDR_MASK = (1 << 24) - 1


def _to_signed(value: int, bits: int) -> int:
    mask = _mask(bits)
    value &= mask
    sign_bit = 1 << (bits - 1)
    return value - (1 << bits) if value & sign_bit else value


def _read_imem_reg(bus: Bus, name: str) -> int:
    addr = IMEMRegisters[name].value & 0xFF
    return bus.load("int", addr, 8) & 0xFF


def _resolve_imem_addr(env: _Env, offset: int) -> int:
    mode = env.next_imem_mode()
    if mode == "(n)":
        return offset & 0xFF
    bp = _read_imem_reg(env.bus, "BP")
    if mode == "(BP+n)":
        return (bp + offset) & 0xFF
    if mode == "(PX+n)":
        px = _read_imem_reg(env.bus, "PX")
        return (px + offset) & 0xFF
    if mode == "(PY+n)":
        py = _read_imem_reg(env.bus, "PY")
        return (py + offset) & 0xFF
    if mode == "(BP+PX)":
        px = _read_imem_reg(env.bus, "PX")
        return (bp + px) & 0xFF
    if mode == "(BP+PY)":
        py = _read_imem_reg(env.bus, "PY")
        return (bp + py) & 0xFF
    # Default fallback
    return (bp + offset) & 0xFF


def _eval_expr(expr: ast.Expr, env: _Env) -> Tuple[int, int]:
    state = env.state
    bus = env.bus
    if isinstance(expr, ast.Const):
        return expr.value & _mask(expr.size), expr.size
    if isinstance(expr, ast.Tmp):
        return env.get_tmp(expr)
    if isinstance(expr, ast.Reg):
        return state.get_reg(expr.name, expr.size), expr.size
    if isinstance(expr, ast.Flag):
        return state.get_flag(expr.name), 1
    if isinstance(expr, ast.Mem):
        addr, addr_bits = _eval_expr(expr.addr, env)
        if expr.space == "int":
            if addr_bits > 8:
                offset = (addr - INTERNAL_MEMORY_START) & 0xFF
                value = bus.load("int", offset, expr.size)
                return value & _mask(expr.size), expr.size
            addr = _resolve_imem_addr(env, addr & 0xFF)
        value = bus.load(expr.space, addr, expr.size)
        return value & _mask(expr.size), expr.size
    if isinstance(expr, ast.UnOp):
        inner, _ = _eval_expr(expr.a, env)
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
        left, _ = _eval_expr(expr.a, env)
        right, _ = _eval_expr(expr.b, env)
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
            return (_to_signed(left, expr.out_size) >> right) & _mask(
                expr.out_size
            ), expr.out_size
        if expr.op == "eq":
            return (1 if left == right else 0, 1)
        if expr.op == "ne":
            return (1 if left != right else 0, 1)
        raise NotImplementedError(f"Binary op {expr.op} not implemented")
    if isinstance(expr, ast.PcRel):
        base = (state.pc + expr.base_advance) & _mask(expr.out_size)
        if expr.disp is None:
            return base, expr.out_size
        disp_value, disp_bits = _eval_expr(expr.disp, env)
        signed = _to_signed(disp_value, disp_bits)
        return (base + signed) & _mask(expr.out_size), expr.out_size
    if isinstance(expr, ast.Join24):
        hi, _ = _eval_expr(expr.hi, env)
        mid, _ = _eval_expr(expr.mid, env)
        lo, _ = _eval_expr(expr.lo, env)
        value = ((hi & 0xFF) << 16) | ((mid & 0xFF) << 8) | (lo & 0xFF)
        return value & _mask(24), 24
    if isinstance(expr, ast.TernOp):
        if expr.op != "select":
            raise NotImplementedError(f"Ternary op {expr.op} unsupported")
        cond = _eval_condition(expr.cond, env)
        return _eval_expr(expr.t if cond else expr.f, env)
    raise NotImplementedError(f"Expression {expr} not supported in emulator backend")


def _const_arg(expr: ast.Expr) -> int:
    if isinstance(expr, ast.Const):
        return expr.value
    raise TypeError("Effect argument must be constant")


def _eval_condition(cond: ast.Cond, env: _Env) -> bool:
    state = env.state
    if cond.kind == "flag":
        if cond.flag is None:
            raise ValueError("Flag condition missing flag name")
        return bool(state.get_flag(cond.flag))
    if cond.a is None or cond.b is None:
        raise ValueError(f"{cond.kind} condition missing operands")
    lhs, lhs_bits = _eval_expr(cond.a, env)
    rhs, rhs_bits = _eval_expr(cond.b, env)
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


def _exec_stmt(
    stmt: ast.Stmt,
    env: _Env,
    stream: Optional[StreamReader],
) -> None:
    state = env.state
    bus = env.bus
    if isinstance(stmt, ast.Fetch):
        value: Optional[int] = None
        const = env.binder.get(stmt.dst.name)
        if const is not None:
            value = const.value
        elif stream is not None:
            value = stream.read(stmt.dst, stmt.kind)
        if value is None:
            raise KeyError(f"No binder or stream value for fetch {stmt.dst.name}")
        env.set_tmp(stmt.dst, value)
        return
    if isinstance(stmt, ast.SetReg):
        value, _ = _eval_expr(stmt.value, env)
        state.set_reg(stmt.reg.name, value, stmt.reg.size)
        if stmt.flags:
            if "Z" in stmt.flags:
                state.set_flag("Z", int(value == 0))
            if "C" in stmt.flags:
                width = stmt.reg.size
                state.set_flag("C", (value >> (width - 1)) & 1)
        return
    if isinstance(stmt, ast.Store):
        addr, _ = _eval_expr(stmt.dst.addr, env)
        if stmt.dst.space == "int":
            addr = _resolve_imem_addr(env, addr & 0xFF)
        value, _ = _eval_expr(stmt.value, env)
        bus.store(stmt.dst.space, addr, value, stmt.dst.size)
        return
    if isinstance(stmt, ast.SetFlag):
        value, _ = _eval_expr(stmt.value, env)
        state.set_flag(stmt.flag, value)
        return
    if isinstance(stmt, ast.If):
        branch = _eval_condition(stmt.cond, env)
        block = stmt.then_ops if branch else stmt.else_ops
        for inner in block:
            _exec_stmt(inner, env, stream)
        return
    if isinstance(stmt, ast.Goto):
        value, bits = _eval_expr(stmt.target, env)
        state.pc = value & _mask(bits)
        return
    if isinstance(stmt, ast.Call):
        value, bits = _eval_expr(stmt.target, env)
        state.pc = value & _mask(bits)
        return
    if isinstance(stmt, ast.Ret):
        return
    if isinstance(stmt, ast.ExtRegLoad):
        _exec_ext_reg_op(
            state,
            bus,
            stmt.ptr.name,
            stmt.mode,
            stmt.disp,
            stmt.dst.size,
            load_reg=stmt.dst.name,
        )
        return
    if isinstance(stmt, ast.ExtRegStore):
        _exec_ext_reg_op(
            state,
            bus,
            stmt.ptr.name,
            stmt.mode,
            stmt.disp,
            stmt.src.size,
            store_reg=stmt.src.name,
        )
        return
    if isinstance(stmt, ast.IntMemSwap):
        _exec_int_mem_swap(stmt, env)
        return
    if isinstance(stmt, ast.ExtRegToIntMem):
        disp = _ext_disp_value(stmt.disp)
        value = _ext_pointer_read_value(
            state, bus, stmt.ptr.name, stmt.mode, disp, stmt.dst.size
        )
        addr, _ = _eval_expr(stmt.dst.addr, env)
        if stmt.dst.space == "int":
            addr = _resolve_imem_addr(env, addr & 0xFF)
        bus.store(stmt.dst.space, addr, value, stmt.dst.size)
        return
    if isinstance(stmt, ast.IntMemToExtReg):
        disp = _ext_disp_value(stmt.disp)
        value, _ = _eval_expr(stmt.src, env)
        _ext_pointer_store_value(
            state, bus, stmt.ptr.name, stmt.mode, disp, stmt.src.size, value
        )
        return
    if isinstance(stmt, (ast.Label, ast.Comment)):
        return
    if isinstance(stmt, ast.Effect):
        _exec_effect(stmt, env)
        return
    raise NotImplementedError(f"Statement {stmt} unsupported in emulator backend")


def step(
    state: CPUState,
    bus: Bus,
    instr: ast.Instr,
    *,
    binder: Optional[Dict[str, ast.Const]] = None,
    pre_latch: Optional[PreLatch] = None,
    stream: Optional[StreamReader] = None,
) -> None:
    env = _Env(state=state, bus=bus, binder=binder or {}, pre_latch=pre_latch)
    for stmt in instr.semantics:
        _exec_stmt(stmt, env, stream)


def _ext_disp_value(const: Optional[ast.Const]) -> int:
    if const is None:
        return 0
    return _to_signed(const.value, const.size)


def _stack_push(env: _Env, register: str, value: int, count: int) -> None:
    state = env.state
    bus = env.bus
    sp = state.get_reg(register, 24)
    new_sp = (sp - count) & _ADDR_MASK
    state.set_reg(register, new_sp, 24)
    for i in range(count):
        byte = (value >> (8 * i)) & 0xFF
        bus.store("ext", (new_sp + i) & _ADDR_MASK, byte, 8)


def _stack_pop(env: _Env, register: str, count: int) -> int:
    state = env.state
    bus = env.bus
    sp = state.get_reg(register, 24)
    result = 0
    for i in range(count):
        byte = bus.load("ext", (sp + i) & _ADDR_MASK, 8) & 0xFF
        result |= byte << (8 * i)
    state.set_reg(register, (sp + count) & _ADDR_MASK, 24)
    return result


def _loop_int_pointer(ptr: ast.LoopIntPtr, env: _Env) -> int:
    offset, _ = _eval_expr(ptr.offset, env)
    return _resolve_imem_addr(env, offset & 0xFF)


def _imem_addr(name: str) -> int:
    return IMEMRegisters[name].value & 0xFF


def _enter_low_power_state(env: _Env) -> None:
    bus = env.bus
    usr_addr = _imem_addr("USR")
    usr = bus.load("int", usr_addr, 8) & 0xFF
    usr = (usr & ~0x3F) | 0x18
    bus.store("int", usr_addr, usr, 8)

    ssr_addr = _imem_addr("SSR")
    ssr = bus.load("int", ssr_addr, 8) & 0xFF
    ssr |= 0x04
    bus.store("int", ssr_addr, ssr, 8)


def _perform_reset(env: _Env) -> None:
    bus = env.bus
    for reg in ("UCR", "ISR", "SCR"):
        bus.store("int", _imem_addr(reg), 0x00, 8)

    lcc_addr = _imem_addr("LCC")
    lcc = bus.load("int", lcc_addr, 8) & 0xFF
    lcc &= ~0x80
    bus.store("int", lcc_addr, lcc, 8)

    usr_addr = _imem_addr("USR")
    usr = bus.load("int", usr_addr, 8) & 0xFF
    usr = (usr & ~0x3F) | 0x18
    bus.store("int", usr_addr, usr, 8)

    ssr_addr = _imem_addr("SSR")
    ssr = bus.load("int", ssr_addr, 8) & 0xFF
    ssr &= ~0x04
    bus.store("int", ssr_addr, ssr, 8)

    lo = bus.load("code", INTERRUPT_VECTOR_ADDR, 8) & 0xFF
    mid = bus.load("code", INTERRUPT_VECTOR_ADDR + 1, 8) & 0xFF
    hi = bus.load("code", INTERRUPT_VECTOR_ADDR + 2, 8) & 0xFF
    vector = ((hi << 16) | (mid << 8) | lo) & PC_MASK
    env.state.set_reg("PC", vector, 20)


def _interrupt_enter(env: _Env) -> None:
    state = env.state
    imr_addr = _imem_addr("IMR")
    imr = env.bus.load("int", imr_addr, 8) & 0xFF
    _stack_push(env, "S", imr, 1)
    env.bus.store("int", imr_addr, imr & 0x7F, 8)
    _stack_push(env, "S", state.get_reg("F", 8), 1)
    _stack_push(env, "S", state.get_reg("PC", 24), 3)

    lo = env.bus.load("code", INTERRUPT_VECTOR_ADDR, 8) & 0xFF
    mid = env.bus.load("code", INTERRUPT_VECTOR_ADDR + 1, 8) & 0xFF
    hi = env.bus.load("code", INTERRUPT_VECTOR_ADDR + 2, 8) & 0xFF
    vector = ((hi << 16) | (mid << 8) | lo) & PC_MASK
    state.set_reg("PC", vector, 20)


def _bcd_add_byte(a: int, b: int, carry_in: int) -> tuple[int, int]:
    low = (a & 0xF) + (b & 0xF) + (carry_in & 1)
    carry_high = 0
    if low >= 10:
        low -= 10
        carry_high = 1
    high = ((a >> 4) & 0xF) + ((b >> 4) & 0xF) + carry_high
    carry_out = 0
    if high >= 10:
        high -= 10
        carry_out = 1
    result = ((high & 0xF) << 4) | (low & 0xF)
    return result & 0xFF, carry_out


def _bcd_sub_byte(a: int, b: int, borrow_in: int) -> tuple[int, int]:
    low = (a & 0xF) - (b & 0xF) - (borrow_in & 1)
    borrow_high = 0
    if low < 0:
        low += 10
        borrow_high = 1
    high = ((a >> 4) & 0xF) - ((b >> 4) & 0xF) - borrow_high
    borrow_out = 0
    if high < 0:
        high += 10
        borrow_out = 1
    result = ((high & 0xF) << 4) | (low & 0xF)
    return result & 0xFF, borrow_out


def _exec_effect(stmt: ast.Effect, env: _Env) -> None:
    kind = stmt.kind
    state = env.state
    if kind == "push_ret16":
        value, _ = _eval_expr(stmt.args[0], env)
        _stack_push(env, "S", value, 2)
        return
    if kind == "push_ret24":
        value, _ = _eval_expr(stmt.args[0], env)
        _stack_push(env, "S", value, 3)
        return
    if kind == "goto_page_join":
        lo, _ = _eval_expr(stmt.args[0], env)
        page, _ = _eval_expr(stmt.args[1], env)
        target = ((page & 0xF0000) | (lo & 0xFFFF)) & PC_MASK
        state.pc = target
        return
    if kind == "goto_far24":
        value, _ = _eval_expr(stmt.args[0], env)
        state.pc = value & PC_MASK
        return
    if kind == "ret_near":
        value = _stack_pop(env, "S", 2)
        page = state.pc & 0xF0000
        state.pc = (page | (value & 0xFFFF)) & PC_MASK
        return
    if kind == "ret_far":
        value = _stack_pop(env, "S", 3)
        state.pc = value & PC_MASK
        return
    if kind == "reti":
        sp = state.get_reg("S", 24)
        imr = env.bus.load("ext", sp & _ADDR_MASK, 8) & 0xFF
        f_val = env.bus.load("ext", (sp + 1) & _ADDR_MASK, 8) & 0xFF
        lo = env.bus.load("ext", (sp + 2) & _ADDR_MASK, 8) & 0xFF
        hi = env.bus.load("ext", (sp + 3) & _ADDR_MASK, 8) & 0xFF
        page = env.bus.load("ext", (sp + 4) & _ADDR_MASK, 8) & 0xFF
        state.set_reg("S", (sp + 5) & _ADDR_MASK, 24)
        env.bus.store("int", IMEMRegisters["IMR"].value & 0xFF, imr, 8)
        state.set_reg("F", f_val, 8)
        state.set_flag("C", f_val & 1)
        state.set_flag("Z", (f_val >> 1) & 1)
        target = (((page & 0xFF) << 16) | (hi << 8) | lo) & PC_MASK
        state.pc = target
        return
    if kind == "push_bytes":
        stack_reg = stmt.args[0]
        if not isinstance(stack_reg, ast.Reg):
            raise TypeError("push_bytes requires stack register arg")
        value, value_bits = _eval_expr(stmt.args[1], env)
        width_bits = _const_arg(stmt.args[2])
        width_bytes = width_bits // 8
        mask = _mask(width_bits)
        _stack_push(env, stack_reg.name, value & mask, width_bytes)
        return
    if kind == "pop_bytes":
        stack_reg = stmt.args[0]
        dest = stmt.args[1]
        if not isinstance(stack_reg, ast.Reg):
            raise TypeError("pop_bytes requires stack register arg")
        width_bits = _const_arg(stmt.args[2])
        width_bytes = width_bits // 8
        value = _stack_pop(env, stack_reg.name, width_bytes) & _mask(width_bits)
        if isinstance(dest, ast.Reg):
            state.set_reg(dest.name, value, dest.size)
            if dest.name == "F":
                state.set_flag("C", value & 1)
                state.set_flag("Z", (value >> 1) & 1)
            return
        if isinstance(dest, ast.Mem):
            addr, _ = _eval_expr(dest.addr, env)
            if dest.space == "int":
                addr = _resolve_imem_addr(env, addr & 0xFF)
            env.bus.store(dest.space, addr, value, dest.size)
            return
        raise TypeError("pop_bytes destination must be register or memory")
    if kind == "loop_move":
        count_value, count_bits = _eval_expr(stmt.args[0], env)
        dst_ptr = stmt.args[1]
        src_ptr = stmt.args[2]
        if not isinstance(dst_ptr, ast.LoopIntPtr) or not isinstance(
            src_ptr, ast.LoopIntPtr
        ):
            raise NotImplementedError(
                "loop_move currently supports internal-memory pointers only"
            )
        step_raw = _const_arg(stmt.args[3])
        step_signed = _to_signed(step_raw, stmt.args[3].size)
        width_bits = _const_arg(stmt.args[4])
        width_bytes = max(1, width_bits // 8)
        remaining = count_value & 0xFFFF
        dst_offset = _loop_int_pointer(dst_ptr, env)
        src_offset = _loop_int_pointer(src_ptr, env)
        while remaining:
            for byte in range(width_bytes):
                value = env.bus.load("int", (src_offset + byte) & 0xFF, 8)
                env.bus.store("int", (dst_offset + byte) & 0xFF, value, 8)
            dst_offset = (dst_offset + step_signed) & 0xFF
            src_offset = (src_offset + step_signed) & 0xFF
            remaining = (remaining - 1) & 0xFFFF
        env.state.set_reg("I", remaining, 16)
        return
    if kind in {"loop_add_carry", "loop_sub_borrow"}:
        count_value, _ = _eval_expr(stmt.args[0], env)
        dst_ptr = stmt.args[1]
        src_ptr = stmt.args[2]
        carry_flag = stmt.args[3]
        width_bits = _const_arg(stmt.args[4])
        remaining = count_value & 0xFFFF
        dst_offset = _loop_int_pointer(dst_ptr, env)
        src_is_mem = isinstance(src_ptr, ast.LoopIntPtr)
        if src_is_mem:
            src_offset = _loop_int_pointer(src_ptr, env)
        elif isinstance(src_ptr, ast.Reg):
            src_reg_name = src_ptr.name
            src_reg_bits = src_ptr.size
        else:
            raise NotImplementedError("loop carry source must be memory or register")
        carry = env.state.get_flag(
            carry_flag.name if isinstance(carry_flag, ast.Flag) else "C"
        )
        overall_zero = 0
        while remaining:
            dst_byte = env.bus.load("int", dst_offset & 0xFF, 8) & 0xFF
            if src_is_mem:
                src_byte = env.bus.load("int", src_offset & 0xFF, 8) & 0xFF
            else:
                src_byte = env.state.get_reg(src_reg_name, src_reg_bits) & 0xFF
            if kind == "loop_sub_borrow":
                diff = dst_byte - src_byte - carry
                if diff < 0:
                    carry = 1
                    result = (diff + 0x100) & 0xFF
                else:
                    carry = 0
                    result = diff & 0xFF
            else:
                total = dst_byte + src_byte + carry
                carry = 1 if total > 0xFF else 0
                result = total & 0xFF
            env.bus.store("int", dst_offset & 0xFF, result, 8)
            overall_zero |= result
            dst_offset = (dst_offset + 1) & 0xFF
            if src_is_mem:
                src_offset = (src_offset + 1) & 0xFF
            remaining = (remaining - 1) & 0xFFFF
        env.state.set_reg("I", remaining, 16)
        env.state.set_flag("C", carry)
        env.state.set_flag("Z", 1 if (overall_zero & 0xFF) == 0 else 0)
        return
    if kind in {"loop_bcd_add", "loop_bcd_sub"}:
        count_value, _ = _eval_expr(stmt.args[0], env)
        dst_ptr = stmt.args[1]
        src_operand = stmt.args[2]
        if not isinstance(dst_ptr, ast.LoopIntPtr):
            raise NotImplementedError("loop_bcd requires internal-memory destination")
        direction_raw = _const_arg(stmt.args[5])
        direction = _to_signed(direction_raw, stmt.args[5].size)
        clear_carry = bool(_const_arg(stmt.args[6]))
        dst_offset = _loop_int_pointer(dst_ptr, env)
        src_is_mem = isinstance(src_operand, ast.LoopIntPtr)
        if src_is_mem:
            src_offset = _loop_int_pointer(src_operand, env)
        elif isinstance(src_operand, ast.Reg):
            src_reg_name = src_operand.name
            src_reg_bits = src_operand.size
        else:
            raise NotImplementedError("loop_bcd source must be memory or register")
        carry = 0 if clear_carry else env.state.get_flag("C")
        remaining = count_value & 0xFFFF
        overall_zero = 0
        while remaining:
            dst_byte = env.bus.load("int", dst_offset & 0xFF, 8) & 0xFF
            if src_is_mem:
                src_byte = env.bus.load("int", src_offset & 0xFF, 8) & 0xFF
            else:
                src_byte = env.state.get_reg(src_reg_name, src_reg_bits) & 0xFF
            if kind == "loop_bcd_sub":
                result, carry = _bcd_sub_byte(dst_byte, src_byte, carry)
            else:
                result, carry = _bcd_add_byte(dst_byte, src_byte, carry)
            env.bus.store("int", dst_offset & 0xFF, result, 8)
            overall_zero |= result
            dst_offset = (dst_offset + direction) & 0xFF
            if src_is_mem:
                src_offset = (src_offset + direction) & 0xFF
            remaining = (remaining - 1) & 0xFFFF
        env.state.set_reg("I", remaining, 16)
        env.state.set_flag("C", carry)
        env.state.set_flag("Z", 1 if (overall_zero & 0xFF) == 0 else 0)
        return
    if kind == "decimal_shift":
        count_value, _ = _eval_expr(stmt.args[0], env)
        ptr = stmt.args[1]
        if not isinstance(ptr, ast.LoopIntPtr):
            raise NotImplementedError("decimal_shift requires internal-memory operand")
        direction = _to_signed(_const_arg(stmt.args[2]), stmt.args[2].size)
        is_left = bool(_const_arg(stmt.args[3]))
        remaining = count_value & 0xFFFF
        current_addr = _loop_int_pointer(ptr, env)
        digit_carry = 0
        overall_zero = 0
        while remaining:
            byte = env.bus.load("int", current_addr & 0xFF, 8) & 0xFF
            low = byte & 0x0F
            high = (byte >> 4) & 0x0F
            if is_left:
                shifted = ((low << 4) & 0xF0) | (digit_carry & 0x0F)
                digit_carry = low
                next_addr = (current_addr - 1) & 0xFF
            else:
                shifted = (high & 0x0F) | ((digit_carry & 0x0F) << 4)
                digit_carry = high
                next_addr = (current_addr + 1) & 0xFF
            env.bus.store("int", current_addr & 0xFF, shifted & 0xFF, 8)
            overall_zero |= shifted
            current_addr = next_addr
            remaining = (remaining - 1) & 0xFFFF
        env.state.set_reg("I", remaining, 16)
        env.state.set_flag("Z", 1 if (overall_zero & 0xFF) == 0 else 0)
        return
    if kind == "pmdf":
        ptr = stmt.args[0]
        if not isinstance(ptr, ast.LoopIntPtr):
            raise NotImplementedError("pmdf requires internal-memory operand")
        addr = _loop_int_pointer(ptr, env)
        value, _ = _eval_expr(stmt.args[1], env)
        current = env.bus.load("int", addr & 0xFF, 8) & 0xFF
        env.bus.store("int", addr & 0xFF, (current + value) & 0xFF, 8)
        return
    if kind in {"halt", "off"}:
        _enter_low_power_state(env)
        state.halted = True
        return
    if kind == "reset":
        _perform_reset(env)
        state.halted = False
        return
    if kind == "wait":
        return
    if kind == "interrupt_enter":
        _interrupt_enter(env)
        state.halted = False
        return
    raise NotImplementedError(f"Effect {kind} not supported")


def _exec_ext_reg_op(
    state: CPUState,
    bus: Bus,
    ptr_name: str,
    mode: str,
    disp_const: Optional[ast.Const],
    width_bits: int,
    *,
    load_reg: Optional[str] = None,
    store_reg: Optional[str] = None,
    store_value: Optional[int] = None,
) -> None:
    if (load_reg is None) == (store_reg is None and store_value is None):
        raise ValueError("Invalid ext_reg operation configuration")
    width_bytes = width_bits // 8
    if width_bytes == 0:
        raise ValueError("Pointer width must be at least 1 byte")
    disp = _ext_disp_value(disp_const)

    if load_reg is not None:
        value = _ext_pointer_read_value(state, bus, ptr_name, mode, disp, width_bits)
        state.set_reg(load_reg, value, width_bits)
        return

    value = store_value
    if value is None and store_reg is not None:
        value = state.get_reg(store_reg, width_bits)
    if value is None:
        raise ValueError("Store value missing for ext_reg operation")
    _ext_pointer_store_value(state, bus, ptr_name, mode, disp, width_bits, value)


def _ext_pointer_base(
    state: CPUState,
    ptr_name: str,
    mode: str,
    disp: int,
    width_bits: int,
) -> int:
    mask24 = _mask(24)
    width_bytes = width_bits // 8
    ptr_val = state.get_reg(ptr_name, 24)
    base = ptr_val
    if mode == "offset":
        base = (ptr_val + disp) & mask24
    elif mode == "pre_dec":
        ptr_val = (ptr_val - width_bytes) & mask24
        state.set_reg(ptr_name, ptr_val, 24)
        base = ptr_val
    else:
        base = ptr_val

    if mode == "post_inc":
        state.set_reg(ptr_name, (ptr_val + width_bytes) & mask24, 24)
    return base


def _ext_pointer_read_value(
    state: CPUState,
    bus: Bus,
    ptr_name: str,
    mode: str,
    disp: int,
    width_bits: int,
) -> int:
    base = _ext_pointer_base(state, ptr_name, mode, disp, width_bits)
    return bus.load("ext", base, width_bits)


def _ext_pointer_store_value(
    state: CPUState,
    bus: Bus,
    ptr_name: str,
    mode: str,
    disp: int,
    width_bits: int,
    value: int,
) -> None:
    base = _ext_pointer_base(state, ptr_name, mode, disp, width_bits)
    bus.store("ext", base, value, width_bits)


def _exec_int_mem_swap(stmt: ast.IntMemSwap, env: _Env) -> None:
    left_addr, _ = _eval_expr(stmt.left, env)
    right_addr, _ = _eval_expr(stmt.right, env)
    left_addr = _resolve_imem_addr(env, left_addr & 0xFF)
    right_addr = _resolve_imem_addr(env, right_addr & 0xFF)
    width = stmt.width
    left_value = env.bus.load("int", left_addr, width)
    right_value = env.bus.load("int", right_addr, width)
    env.bus.store("int", left_addr, right_value, width)
    env.bus.store("int", right_addr, left_value, width)
