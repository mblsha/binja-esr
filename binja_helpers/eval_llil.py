# Evaluate mock LowLevelIL expressions without Binary Ninja.

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Protocol, Tuple, TypedDict

from .mock_llil import MockIntrinsic, MockLLIL


class RegistersLike(Protocol):
    """Minimal register access interface used by the evaluator."""

    def get_by_name(self, name: str) -> int:  # pragma: no cover - protocol
        ...

    def set_by_name(self, name: str, value: int) -> None:  # pragma: no cover - protocol
        ...

    # Optional, for architectures that want to customise flag handling
    def get_flag(self, name: str) -> int:  # pragma: no cover - protocol
        ...

    def set_flag(self, name: str, value: int) -> None:  # pragma: no cover - protocol
        ...


ReadMemType = Callable[[int], int]
WriteMemType = Callable[[int, int], None]


class Memory:
    """Simple memory helper used by the LLIL evaluator."""

    def __init__(self, read_mem: ReadMemType, write_mem: WriteMemType) -> None:
        self.read_mem = read_mem
        self.write_mem = write_mem

    def read_byte(self, address: int) -> int:
        return self.read_mem(address)

    def write_byte(self, address: int, value: int) -> None:
        assert 0 <= value < 256, "Value must be a byte (0-255)"
        self.write_mem(address, value & 0xFF)

    def read_bytes(self, address: int, size: int) -> int:
        assert 0 < size <= 3, "Size must be between 1 and 3 bytes"
        value = 0
        for i in range(size):
            value |= self.read_byte(address + i) << (i * 8)
        return value

    def write_bytes(self, size: int, address: int, value: int) -> None:
        assert 0 < size <= 3
        for i in range(size):
            byte_value = (value >> (i * 8)) & 0xFF
            self.write_byte(address + i, byte_value)


@dataclass
class State:
    halted: bool = False


class ResultFlags(TypedDict, total=False):
    C: Optional[int]
    Z: Optional[int]


FlagGetter = Callable[[str], int]
FlagSetter = Callable[[str, int], None]


EvalLLILType = Callable[
    [MockLLIL, Optional[int], RegistersLike, Memory, State, FlagGetter, FlagSetter],
    Tuple[Optional[int], Optional[ResultFlags]],
]


def evaluate_llil(
    llil: MockLLIL,
    regs: RegistersLike,
    memory: Memory,
    state: State,
    get_flag: Optional[FlagGetter] = None,
    set_flag: Optional[FlagSetter] = None,
) -> Tuple[Optional[int], Optional[ResultFlags]]:
    op_name_bare = llil.bare_op()
    llil_flags_spec = llil.flags()  # e.g., "CZ", "Z", or None
    size = llil.width()
    current_op_name_for_eval = op_name_bare

    if get_flag is None:
        if hasattr(regs, "get_flag"):
            def get_flag_default(name: str) -> int:
                return regs.get_flag(name)
        else:
            def get_flag_default(name: str) -> int:
                return regs.get_by_name(f"F{name}")

        get_flag = get_flag_default

    if set_flag is None:
        if hasattr(regs, "set_flag"):
            def set_flag_default(name: str, value: int) -> None:
                regs.set_flag(name, value)
        else:
            def set_flag_default(name: str, value: int) -> None:
                regs.set_by_name(f"F{name}", value)

        set_flag = set_flag_default

    if isinstance(llil, MockIntrinsic):
        intrinsic = llil
        current_op_name_for_eval = f"INTRINSIC_{intrinsic.name}"

    f = EVAL_LLIL.get(current_op_name_for_eval)
    if f is None:
        raise NotImplementedError(
            f"Eval for {current_op_name_for_eval} not implemented"
        )

    result_value, op_defined_flags = f(llil, size, regs, memory, state, get_flag, set_flag)

    if llil_flags_spec is not None and llil_flags_spec != "0":
        if "Z" in llil_flags_spec:
            fz_val_to_set: Optional[int] = None
            if op_defined_flags:
                fz_val_to_set = op_defined_flags.get("Z")

            if fz_val_to_set is not None:
                set_flag("Z", fz_val_to_set)
            elif isinstance(result_value, int):
                assert size is not None, (
                    f"FZ flag setting from result_value requires size for {current_op_name_for_eval}"  # noqa: E501
                )
                zero_mask = (1 << (size * 8)) - 1
                set_flag("Z", int((result_value & zero_mask) == 0))

        if "C" in llil_flags_spec:
            fc_val_to_set: Optional[int] = None
            if op_defined_flags:
                fc_val_to_set = op_defined_flags.get("C")

            if fc_val_to_set is not None:
                set_flag("C", fc_val_to_set)
            elif isinstance(result_value, int):
                assert size is not None, (
                    f"FC flag setting from result_value requires size for {current_op_name_for_eval}"  # noqa: E501
                )
                unsigned_max_for_size = (1 << (size * 8)) - 1
                carry_flag_val = 0
                if result_value > unsigned_max_for_size:
                    carry_flag_val = 1
                set_flag("C", carry_flag_val)

    return result_value, op_defined_flags


def eval_const(
    llil: MockLLIL,
    size: Optional[int],
    regs: RegistersLike,
    memory: Memory,
    state: State,
    get_flag: FlagGetter,
    set_flag: FlagSetter,
) -> Tuple[int, Optional[ResultFlags]]:
    result = llil.ops[0]
    assert isinstance(result, int)
    return result, None


def eval_const_ptr(
    llil: MockLLIL,
    size: Optional[int],
    regs: RegistersLike,
    memory: Memory,
    state: State,
    get_flag: FlagGetter,
    set_flag: FlagSetter,
) -> Tuple[int, Optional[ResultFlags]]:
    result = llil.ops[0]
    assert isinstance(result, int)
    return result, None


def eval_reg(
    llil: MockLLIL,
    size: Optional[int],
    regs: RegistersLike,
    memory: Memory,
    state: State,
    get_flag: FlagGetter,
    set_flag: FlagSetter,
) -> Tuple[int, Optional[ResultFlags]]:
    return regs.get_by_name(llil.ops[0].name), None


def eval_set_reg(
    llil: MockLLIL,
    size: Optional[int],
    regs: RegistersLike,
    memory: Memory,
    state: State,
    get_flag: FlagGetter,
    set_flag: FlagSetter,
) -> Tuple[None, Optional[ResultFlags]]:
    assert isinstance(llil.ops[1], MockLLIL)
    value_to_set, _ = evaluate_llil(llil.ops[1], regs, memory, state, get_flag, set_flag)
    assert value_to_set is not None
    regs.set_by_name(llil.ops[0].name, value_to_set)
    return None, None


def eval_flag(
    llil: MockLLIL,
    size: Optional[int],
    regs: RegistersLike,
    memory: Memory,
    state: State,
    get_flag: FlagGetter,
    set_flag: FlagSetter,
) -> Tuple[int, Optional[ResultFlags]]:
    return get_flag(llil.ops[0].name), None


def eval_set_flag(
    llil: MockLLIL,
    size: Optional[int],
    regs: RegistersLike,
    memory: Memory,
    state: State,
    get_flag: FlagGetter,
    set_flag: FlagSetter,
) -> Tuple[None, Optional[ResultFlags]]:
    assert isinstance(llil.ops[1], MockLLIL)
    value_to_set, _ = evaluate_llil(llil.ops[1], regs, memory, state, get_flag, set_flag)
    assert value_to_set is not None
    set_flag(llil.ops[0].name, 1 if value_to_set != 0 else 0)
    return None, None


def _create_logical_eval(op_func: Callable[[int, int], int]) -> EvalLLILType:
    def _eval(
        llil: MockLLIL,
        size: Optional[int],
        regs: RegistersLike,
        memory: Memory,
        state: State,
        get_flag: FlagGetter,
        set_flag: FlagSetter,
    ) -> Tuple[int, Optional[ResultFlags]]:
        assert isinstance(llil.ops[0], MockLLIL) and isinstance(llil.ops[1], MockLLIL)
        op1_val, _ = evaluate_llil(llil.ops[0], regs, memory, state, get_flag, set_flag)
        op2_val, _ = evaluate_llil(llil.ops[1], regs, memory, state, get_flag, set_flag)
        assert op1_val is not None and op2_val is not None
        result = op_func(int(op1_val), int(op2_val))
        return result, {"Z": 1 if result == 0 else 0, "C": 0}

    return _eval


def _create_arithmetic_eval(
    op_func: Callable[[int, int], int],
    carry_func: Callable[[int, int, int, int], int],
) -> EvalLLILType:
    def _eval(
        llil: MockLLIL,
        size: Optional[int],
        regs: RegistersLike,
        memory: Memory,
        state: State,
        get_flag: FlagGetter,
        set_flag: FlagSetter,
    ) -> Tuple[int, Optional[ResultFlags]]:
        assert size is not None
        assert isinstance(llil.ops[0], MockLLIL) and isinstance(llil.ops[1], MockLLIL)
        op1_val, _ = evaluate_llil(llil.ops[0], regs, memory, state, get_flag, set_flag)
        op2_val, _ = evaluate_llil(llil.ops[1], regs, memory, state, get_flag, set_flag)
        assert op1_val is not None and op2_val is not None

        result_full = op_func(int(op1_val), int(op2_val))

        width_bits = size * 8
        mask = (1 << width_bits) - 1
        result_masked = result_full & mask

        flag_z = 1 if result_masked == 0 else 0
        flag_c = carry_func(int(op1_val), int(op2_val), result_full, mask)

        return result_masked, {"C": flag_c, "Z": flag_z}

    return _eval


def _create_shift_eval(op_func: Callable[..., Tuple[int, ResultFlags]]) -> EvalLLILType:
    def _eval(
        llil: MockLLIL,
        size: Optional[int],
        regs: RegistersLike,
        memory: Memory,
        state: State,
        get_flag: FlagGetter,
        set_flag: FlagSetter,
    ) -> Tuple[int, Optional[ResultFlags]]:
        assert size is not None
        values = []
        for operand in llil.ops:
            assert isinstance(operand, MockLLIL)
            val, _ = evaluate_llil(operand, regs, memory, state, get_flag, set_flag)
            assert val is not None
            values.append(int(val))

        result, flags = op_func(size, *values)
        return result, flags

    return _eval


def eval_pop(
    llil: MockLLIL,
    size: Optional[int],
    regs: RegistersLike,
    memory: Memory,
    state: State,
    get_flag: FlagGetter,
    set_flag: FlagSetter,
) -> Tuple[int, Optional[ResultFlags]]:
    assert size
    addr = regs.get_by_name("S")
    result = memory.read_bytes(addr, size)
    regs.set_by_name("S", addr + size)
    return result, None


def eval_push(
    llil: MockLLIL,
    size: Optional[int],
    regs: RegistersLike,
    memory: Memory,
    state: State,
    get_flag: FlagGetter,
    set_flag: FlagSetter,
) -> Tuple[None, Optional[ResultFlags]]:
    assert size
    assert isinstance(llil.ops[0], MockLLIL)
    value_to_push, _ = evaluate_llil(llil.ops[0], regs, memory, state, get_flag, set_flag)
    assert value_to_push is not None
    addr = regs.get_by_name("S") - size
    memory.write_bytes(size, addr, value_to_push)
    regs.set_by_name("S", addr)
    return None, None


def eval_nop(
    llil: MockLLIL,
    size: Optional[int],
    regs: RegistersLike,
    memory: Memory,
    state: State,
    get_flag: FlagGetter,
    set_flag: FlagSetter,
) -> Tuple[None, Optional[ResultFlags]]:
    return None, None


def eval_unimpl(
    llil: MockLLIL,
    size: Optional[int],
    regs: RegistersLike,
    memory: Memory,
    state: State,
    get_flag: FlagGetter,
    set_flag: FlagSetter,
) -> Tuple[None, Optional[ResultFlags]]:
    raise NotImplementedError(f"Low-level IL operation {llil.op} is not implemented")


def eval_store(
    llil: MockLLIL,
    size: Optional[int],
    regs: RegistersLike,
    memory: Memory,
    state: State,
    get_flag: FlagGetter,
    set_flag: FlagSetter,
) -> Tuple[None, Optional[ResultFlags]]:
    assert size
    assert isinstance(llil.ops[0], MockLLIL) and isinstance(llil.ops[1], MockLLIL)
    dest_addr, _ = evaluate_llil(llil.ops[0], regs, memory, state, get_flag, set_flag)
    value_to_store, _ = evaluate_llil(llil.ops[1], regs, memory, state, get_flag, set_flag)
    assert dest_addr is not None and value_to_store is not None
    memory.write_bytes(size, dest_addr, value_to_store)
    return None, None


def eval_load(
    llil: MockLLIL,
    size: Optional[int],
    regs: RegistersLike,
    memory: Memory,
    state: State,
    get_flag: FlagGetter,
    set_flag: FlagSetter,
) -> Tuple[int, Optional[ResultFlags]]:
    assert size
    assert isinstance(llil.ops[0], MockLLIL)
    addr, _ = evaluate_llil(llil.ops[0], regs, memory, state, get_flag, set_flag)
    assert addr is not None
    return memory.read_bytes(addr, size), None


def eval_ret(
    llil: MockLLIL,
    size: Optional[int],
    regs: RegistersLike,
    memory: Memory,
    state: State,
    get_flag: FlagGetter,
    set_flag: FlagSetter,
) -> Tuple[None, Optional[ResultFlags]]:
    assert isinstance(llil.ops[0], MockLLIL)
    addr_val, _ = evaluate_llil(llil.ops[0], regs, memory, state, get_flag, set_flag)
    assert isinstance(addr_val, int)
    regs.set_by_name("PC", addr_val)
    return None, None


def eval_jump(
    llil: MockLLIL,
    size: Optional[int],
    regs: RegistersLike,
    memory: Memory,
    state: State,
    get_flag: FlagGetter,
    set_flag: FlagSetter,
) -> Tuple[None, Optional[ResultFlags]]:
    assert isinstance(llil.ops[0], MockLLIL)
    addr, _ = evaluate_llil(llil.ops[0], regs, memory, state, get_flag, set_flag)
    assert isinstance(addr, int)
    regs.set_by_name("PC", addr)
    return None, None


def eval_call(
    llil: MockLLIL,
    size: Optional[int],
    regs: RegistersLike,
    memory: Memory,
    state: State,
    get_flag: FlagGetter,
    set_flag: FlagSetter,
) -> Tuple[None, Optional[ResultFlags]]:
    assert isinstance(llil.ops[0], MockLLIL)
    addr, _ = evaluate_llil(llil.ops[0], regs, memory, state, get_flag, set_flag)
    assert isinstance(addr, int)

    ret_addr = regs.get_by_name("PC")

    push_size = 3
    if (
        llil.ops[0].op == "CONST_PTR.w"
        or (llil.ops[0].op == "CONST.w")
        or (llil.ops[0].op == "OR.l" and llil.ops[0].ops[0].op == "CONST.w")
    ):
        push_size = 2

    stack_addr = regs.get_by_name("S") - push_size
    if push_size == 2:
        memory.write_bytes(push_size, stack_addr, ret_addr & 0xFFFF)
    else:
        memory.write_bytes(push_size, stack_addr, ret_addr & 0xFFFFF)

    regs.set_by_name("S", stack_addr)
    regs.set_by_name("PC", addr)
    return None, None


def eval_cmp_e(
    llil: MockLLIL,
    size: Optional[int],
    regs: RegistersLike,
    memory: Memory,
    state: State,
    get_flag: FlagGetter,
    set_flag: FlagSetter,
) -> Tuple[int, Optional[ResultFlags]]:
    assert isinstance(llil.ops[0], MockLLIL) and isinstance(llil.ops[1], MockLLIL)
    op1_val, _ = evaluate_llil(llil.ops[0], regs, memory, state, get_flag, set_flag)
    op2_val, _ = evaluate_llil(llil.ops[1], regs, memory, state, get_flag, set_flag)
    assert op1_val is not None and op2_val is not None
    return int(int(op1_val) == int(op2_val)), None


def eval_cmp_ugt(
    llil: MockLLIL,
    size: Optional[int],
    regs: RegistersLike,
    memory: Memory,
    state: State,
    get_flag: FlagGetter,
    set_flag: FlagSetter,
) -> Tuple[int, Optional[ResultFlags]]:
    assert isinstance(llil.ops[0], MockLLIL) and isinstance(llil.ops[1], MockLLIL)
    op1_val, _ = evaluate_llil(llil.ops[0], regs, memory, state, get_flag, set_flag)
    op2_val, _ = evaluate_llil(llil.ops[1], regs, memory, state, get_flag, set_flag)
    assert op1_val is not None and op2_val is not None
    return int(int(op1_val) > int(op2_val)), None


def to_signed(value: int, size_bytes: int) -> int:
    width_bits = size_bytes * 8
    mask = (1 << width_bits) - 1
    sign_bit_mask = 1 << (width_bits - 1)
    value &= mask
    if (value & sign_bit_mask) != 0:
        return value - (1 << width_bits)
    return value


def eval_cmp_slt(
    llil: MockLLIL,
    size: Optional[int],
    regs: RegistersLike,
    memory: Memory,
    state: State,
    get_flag: FlagGetter,
    set_flag: FlagSetter,
) -> Tuple[int, Optional[ResultFlags]]:
    assert size is not None, "Size must be provided for signed comparison"
    assert isinstance(llil.ops[0], MockLLIL) and isinstance(llil.ops[1], MockLLIL)
    op1_val, _ = evaluate_llil(llil.ops[0], regs, memory, state, get_flag, set_flag)
    op2_val, _ = evaluate_llil(llil.ops[1], regs, memory, state, get_flag, set_flag)
    assert op1_val is not None and op2_val is not None

    signed_op1 = to_signed(int(op1_val), size)
    signed_op2 = to_signed(int(op2_val), size)

    return int(signed_op1 < signed_op2), None


def _lsl_impl(size: int, val: int, count: int) -> Tuple[int, ResultFlags]:
    width = size * 8
    mask = (1 << width) - 1
    if count == 0:
        arith_result = val & mask
        return arith_result, {"C": 0, "Z": 1 if arith_result == 0 else 0}

    carry_out = 0
    if count <= width and width > 0:
        carry_out = (val >> (width - count)) & 1

    arith_result = (val << count) & mask
    zero_flag = 1 if arith_result == 0 else 0

    return arith_result, {"C": carry_out, "Z": zero_flag}


def _lsr_impl(size: int, val: int, count: int) -> Tuple[int, ResultFlags]:
    width = size * 8
    if count == 0:
        arith_result = val & ((1 << width) - 1 if width > 0 else 0)
        return arith_result, {"C": 0, "Z": 1 if arith_result == 0 else 0}

    carry_out = 0
    if count > 0 and count <= width and width > 0:
        carry_out = (val >> (count - 1)) & 1

    arith_result = val >> count
    zero_flag_val = arith_result & ((1 << width) - 1 if width > 0 else 0)
    zero_flag = 1 if zero_flag_val == 0 else 0

    return arith_result, {"C": carry_out, "Z": zero_flag}


def _ror_impl(size: int, val: int, count: int) -> Tuple[int, ResultFlags]:
    width = size * 8
    mask = (1 << width) - 1 if width > 0 else 0
    if width == 0:
        return val & mask, {"C": 0, "Z": 1 if (val & mask) == 0 else 0}

    count %= width
    if count == 0:
        arith_result = val & mask
        return arith_result, {"C": val & 1, "Z": 1 if arith_result == 0 else 0}

    shifted_part = val >> count
    rotated_part = val << (width - count)
    arith_result = (shifted_part | rotated_part) & mask

    carry_out = (val >> (count - 1)) & 1
    zero_flag = 1 if arith_result == 0 else 0

    return arith_result, {"C": carry_out, "Z": zero_flag}


def _rol_impl(size: int, val: int, count: int) -> Tuple[int, ResultFlags]:
    width = size * 8
    mask = (1 << width) - 1 if width > 0 else 0
    if width == 0:
        return val & mask, {"C": 0, "Z": 1 if (val & mask) == 0 else 0}

    count %= width
    if count == 0:
        arith_result = val & mask
        return arith_result, {
            "C": (val >> (width - 1)) & 1 if width > 0 else 0,
            "Z": 1 if arith_result == 0 else 0,
        }

    shifted_part = val << count
    rotated_part = val >> (width - count)
    arith_result = (shifted_part | rotated_part) & mask

    carry_out = (val >> (width - count)) & 1
    zero_flag = 1 if arith_result == 0 else 0

    return arith_result, {"C": carry_out, "Z": zero_flag}


def _rrc_impl(
    size: int, val: int, count: int, carry_in: int
) -> Tuple[int, ResultFlags]:
    width = size * 8
    mask = (1 << width) - 1 if width > 0 else 0
    assert count == 1, "RRC count should be 1 for standard definition"

    if width == 0:
        return val & mask, {"C": 0, "Z": 1 if (val & mask) == 0 else 0}

    new_carry_out = val & 1
    arith_result = (val >> count) | (carry_in << (width - count))
    arith_result &= mask

    zero_flag = 1 if arith_result == 0 else 0
    return arith_result, {"C": new_carry_out, "Z": zero_flag}


def _rlc_impl(
    size: int, val: int, count: int, carry_in: int
) -> Tuple[int, ResultFlags]:
    width = size * 8
    mask = (1 << width) - 1 if width > 0 else 0
    assert count == 1, "RLC count should be 1 for standard definition"

    if width == 0:
        return val & mask, {"C": 0, "Z": 1 if (val & mask) == 0 else 0}

    new_carry_out = (val >> (width - 1)) & 1 if width > 0 else 0
    arith_result = (val << count) | carry_in
    arith_result &= mask

    zero_flag = 1 if arith_result == 0 else 0
    return arith_result, {"C": new_carry_out, "Z": zero_flag}


def eval_intrinsic_tcl(
    llil: MockLLIL,
    size: Optional[int],
    regs: RegistersLike,
    memory: Memory,
    state: State,
    get_flag: FlagGetter,
    set_flag: FlagSetter,
) -> Tuple[None, Optional[ResultFlags]]:
    return None, None


def eval_intrinsic_halt(
    llil: MockLLIL,
    size: Optional[int],
    regs: RegistersLike,
    memory: Memory,
    state: State,
    get_flag: FlagGetter,
    set_flag: FlagSetter,
) -> Tuple[None, Optional[ResultFlags]]:
    state.halted = True
    return None, None


def eval_intrinsic_off(
    llil: MockLLIL,
    size: Optional[int],
    regs: RegistersLike,
    memory: Memory,
    state: State,
    get_flag: FlagGetter,
    set_flag: FlagSetter,
) -> Tuple[None, Optional[ResultFlags]]:
    state.halted = True
    return None, None


EVAL_LLIL: Dict[str, EvalLLILType] = {
    "CONST": eval_const,
    "CONST_PTR": eval_const_ptr,
    "REG": eval_reg,
    "SET_REG": eval_set_reg,
    "FLAG": eval_flag,
    "SET_FLAG": eval_set_flag,
    "AND": _create_logical_eval(lambda a, b: a & b),
    "OR": _create_logical_eval(lambda a, b: a | b),
    "XOR": _create_logical_eval(lambda a, b: a ^ b),
    "POP": eval_pop,
    "PUSH": eval_push,
    "NOP": eval_nop,
    "UNIMPL": eval_unimpl,
    "STORE": eval_store,
    "LOAD": eval_load,
    "RET": eval_ret,
    "JUMP": eval_jump,
    "CALL": eval_call,
    "ADD": _create_arithmetic_eval(
        lambda a, b: a + b,
        lambda _a, _b, result, mask: 1 if result > mask else 0,
    ),
    "SUB": _create_arithmetic_eval(
        lambda a, b: a - b,
        lambda _a, _b, result, _mask: 1 if result < 0 else 0,
    ),
    "CMP_E": eval_cmp_e,
    "CMP_UGT": eval_cmp_ugt,
    "CMP_SLT": eval_cmp_slt,
    "LSL": _create_shift_eval(lambda size, val, count: _lsl_impl(size, val, count)),
    "LSR": _create_shift_eval(lambda size, val, count: _lsr_impl(size, val, count)),
    "ROR": _create_shift_eval(lambda size, val, count: _ror_impl(size, val, count)),
    "RRC": _create_shift_eval(
        lambda size, val, count, carry: _rrc_impl(size, val, count, carry)
    ),
    "ROL": _create_shift_eval(lambda size, val, count: _rol_impl(size, val, count)),
    "RLC": _create_shift_eval(
        lambda size, val, count, carry: _rlc_impl(size, val, count, carry)
    ),
    "INTRINSIC_TCL": eval_intrinsic_tcl,
    "INTRINSIC_HALT": eval_intrinsic_halt,
    "INTRINSIC_OFF": eval_intrinsic_off,
}
