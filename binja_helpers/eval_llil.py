# Evaluate mock LowLevelIL expressions without Binary Ninja.

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Protocol, Tuple, TypedDict, cast

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


@dataclass
class FlagInfo:
    register: str
    bit: int


# Default bit positions for generic flags in the combined flag register "F".
FLAG_LAYOUT: Dict[str, FlagInfo] = {
    "C": FlagInfo("F", 0),
    "Z": FlagInfo("F", 1),
}


FlagGetter = Callable[[str], int]
FlagSetter = Callable[[str, int], None]


EvalLLILType = Callable[
    [MockLLIL, Optional[int], RegistersLike, Memory, State, FlagGetter, FlagSetter],
    Tuple[Optional[int], Optional[ResultFlags]],
]


# Global registry for architecture-specific intrinsic evaluators
_INTRINSIC_REGISTRY: Dict[str, EvalLLILType] = {}


def register_intrinsic(name: str, evaluator: EvalLLILType) -> None:
    """Register an architecture-specific intrinsic evaluator.
    
    Args:
        name: The intrinsic name (e.g., "TCL", "HALT", "OFF")
        evaluator: The evaluation function that implements the intrinsic behavior
    """
    _INTRINSIC_REGISTRY[name] = evaluator


def get_intrinsic_evaluator(name: str) -> Optional[EvalLLILType]:
    """Get the evaluator for a specific intrinsic.
    
    Args:
        name: The intrinsic name
        
    Returns:
        The evaluator function if registered, None otherwise
    """
    return _INTRINSIC_REGISTRY.get(name)


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
        def get_flag_default(name: str) -> int:
            info = FLAG_LAYOUT.get(name)
            if info is None:
                return regs.get_by_name(f"F{name}")
            reg_val = regs.get_by_name(info.register)
            return (reg_val >> info.bit) & 1

        get_flag = get_flag_default

    if set_flag is None:
        def set_flag_default(name: str, value: int) -> None:
            info = FLAG_LAYOUT.get(name)
            if info is None:
                regs.set_by_name(f"F{name}", value)
                return
            reg_val = regs.get_by_name(info.register)
            if value:
                reg_val |= 1 << info.bit
            else:
                reg_val &= ~(1 << info.bit)
            regs.set_by_name(info.register, reg_val)

        set_flag = set_flag_default

    if isinstance(llil, MockIntrinsic):
        intrinsic = llil
        # Look up intrinsic in the registry first
        f = get_intrinsic_evaluator(intrinsic.name)
        if f is None:
            raise NotImplementedError(
                f"Intrinsic '{intrinsic.name}' not registered. "
                f"Architecture-specific intrinsics must be registered using register_intrinsic()."
            )
    else:
        f = EVAL_LLIL.get(current_op_name_for_eval)
        if f is None:
            raise NotImplementedError(
                f"Eval for {current_op_name_for_eval} not implemented"
            )

    result_value, op_defined_flags = f(llil, size, regs, memory, state, get_flag, set_flag)

    if llil_flags_spec is not None and llil_flags_spec != "0":
        flag_from_result: Dict[str, Callable[[int, int], int]] = {
            "Z": lambda val, sz: int((val & ((1 << (sz * 8)) - 1)) == 0),
            "C": lambda val, sz: int(val > ((1 << (sz * 8)) - 1)),
        }

        for flag_name in llil_flags_spec:
            value_to_set: Optional[int] = None
            if op_defined_flags:
                value_to_set = cast(Optional[int], op_defined_flags.get(flag_name))

            if value_to_set is None and isinstance(result_value, int):
                assert size is not None, (
                    f"F{flag_name} flag setting from result_value requires size for {current_op_name_for_eval}"
                )
                update_func = flag_from_result.get(flag_name)
                if update_func is not None:
                    value_to_set = update_func(int(result_value), size)

            if value_to_set is not None:
                set_flag(flag_name, value_to_set)

    return result_value, op_defined_flags


def _create_const_eval() -> EvalLLILType:
    def _eval(
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

    return _eval


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


def to_signed(value: int, size_bytes: int) -> int:
    width_bits = size_bytes * 8
    mask = (1 << width_bits) - 1
    sign_bit_mask = 1 << (width_bits - 1)
    value &= mask
    if (value & sign_bit_mask) != 0:
        return value - (1 << width_bits)
    return value


def _create_comparison_eval(op_func: Callable[[int, int], bool], signed: bool = False) -> EvalLLILType:
    def _eval(
        llil: MockLLIL,
        size: Optional[int],
        regs: RegistersLike,
        memory: Memory,
        state: State,
        get_flag: FlagGetter,
        set_flag: FlagSetter,
    ) -> Tuple[int, Optional[ResultFlags]]:
        if signed:
            assert size is not None, "Size must be provided for signed comparison"
        assert isinstance(llil.ops[0], MockLLIL) and isinstance(llil.ops[1], MockLLIL)
        op1_val, _ = evaluate_llil(llil.ops[0], regs, memory, state, get_flag, set_flag)
        op2_val, _ = evaluate_llil(llil.ops[1], regs, memory, state, get_flag, set_flag)
        assert op1_val is not None and op2_val is not None
        
        if signed:
            assert size is not None  # Already checked above
            op1_val = to_signed(int(op1_val), size)
            op2_val = to_signed(int(op2_val), size)
        else:
            op1_val = int(op1_val)
            op2_val = int(op2_val)
        
        return int(op_func(op1_val, op2_val)), None

    return _eval


def _shift_impl(size: int, val: int, count: int, *, left: bool) -> Tuple[int, ResultFlags]:
    """Common implementation for logical shifts."""
    width = size * 8
    mask = (1 << width) - 1 if width > 0 else 0

    if count == 0:
        result = val & mask
        return result, {"C": 0, "Z": 1 if result == 0 else 0}

    carry_out = 0
    if left:
        if count <= width and width > 0:
            carry_out = (val >> (width - count)) & 1
        result = (val << count) & mask
    else:
        if count > 0 and count <= width and width > 0:
            carry_out = (val >> (count - 1)) & 1
        result = (val >> count) & mask

    return result, {"C": carry_out, "Z": 1 if result == 0 else 0}


def _lsl_impl(size: int, val: int, count: int) -> Tuple[int, ResultFlags]:
    return _shift_impl(size, val, count, left=True)


def _lsr_impl(size: int, val: int, count: int) -> Tuple[int, ResultFlags]:
    return _shift_impl(size, val, count, left=False)


def _rotate_impl(size: int, val: int, count: int, *, left: bool) -> Tuple[int, ResultFlags]:
    """Rotate ``val`` left or right by ``count`` bits."""
    width = size * 8
    mask = (1 << width) - 1 if width > 0 else 0
    if width == 0:
        return val & mask, {"C": 0, "Z": 1 if (val & mask) == 0 else 0}

    count %= width
    if count == 0:
        result = val & mask
        carry_out = (val >> (width - 1)) & 1 if left and width > 0 else val & 1
        return result, {"C": carry_out, "Z": 1 if result == 0 else 0}

    if left:
        shifted_part = val << count
        rotated_part = val >> (width - count)
        carry_out = (val >> (width - count)) & 1
    else:
        shifted_part = val >> count
        rotated_part = val << (width - count)
        carry_out = (val >> (count - 1)) & 1

    result = (shifted_part | rotated_part) & mask
    return result, {"C": carry_out, "Z": 1 if result == 0 else 0}


def _ror_impl(size: int, val: int, count: int) -> Tuple[int, ResultFlags]:
    return _rotate_impl(size, val, count, left=False)


def _rol_impl(size: int, val: int, count: int) -> Tuple[int, ResultFlags]:
    return _rotate_impl(size, val, count, left=True)


def _rotate_through_carry_impl(
    size: int, val: int, count: int, carry_in: int, *, left: bool
) -> Tuple[int, ResultFlags]:
    """Rotate ``val`` through carry left or right by ``count`` bits."""
    width = size * 8
    mask = (1 << width) - 1 if width > 0 else 0
    assert count == 1, "RRC/RLC count should be 1 for standard definition"

    if width == 0:
        return val & mask, {"C": 0, "Z": 1 if (val & mask) == 0 else 0}

    if left:
        new_carry_out = (val >> (width - 1)) & 1 if width > 0 else 0
        result = ((val << count) | carry_in) & mask
    else:
        new_carry_out = val & 1
        result = ((val >> count) | (carry_in << (width - count))) & mask

    return result, {"C": new_carry_out, "Z": 1 if result == 0 else 0}


def _rrc_impl(size: int, val: int, count: int, carry_in: int) -> Tuple[int, ResultFlags]:
    return _rotate_through_carry_impl(size, val, count, carry_in, left=False)


def _rlc_impl(size: int, val: int, count: int, carry_in: int) -> Tuple[int, ResultFlags]:
    return _rotate_through_carry_impl(size, val, count, carry_in, left=True)


def _create_intrinsic_eval(action: Optional[Callable[[State], None]] = None) -> EvalLLILType:
    def _eval(
        llil: MockLLIL,
        size: Optional[int],
        regs: RegistersLike,
        memory: Memory,
        state: State,
        get_flag: FlagGetter,
        set_flag: FlagSetter,
    ) -> Tuple[None, Optional[ResultFlags]]:
        if action is not None:
            action(state)
        return None, None

    return _eval


EVAL_LLIL: Dict[str, EvalLLILType] = {
    "CONST": _create_const_eval(),
    "CONST_PTR": _create_const_eval(),
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
    "MUL": _create_arithmetic_eval(
        lambda a, b: a * b,
        lambda _a, _b, result, mask: 1 if result > mask else 0,
    ),
    "DIVU": _create_arithmetic_eval(
        lambda a, b: a // b if b != 0 else 0,
        lambda _a, _b, result, _mask: 0,  # Division typically doesn't set carry
    ),
    "MODU": _create_arithmetic_eval(
        lambda a, b: a % b if b != 0 else 0,
        lambda _a, _b, result, _mask: 0,  # Modulo typically doesn't set carry
    ),
    "CMP_E": _create_comparison_eval(lambda a, b: a == b),
    "CMP_NE": _create_comparison_eval(lambda a, b: a != b),
    "CMP_UGT": _create_comparison_eval(lambda a, b: a > b),
    "CMP_UGE": _create_comparison_eval(lambda a, b: a >= b),
    "CMP_ULT": _create_comparison_eval(lambda a, b: a < b),
    "CMP_ULE": _create_comparison_eval(lambda a, b: a <= b),
    "CMP_SGT": _create_comparison_eval(lambda a, b: a > b, signed=True),
    "CMP_SGE": _create_comparison_eval(lambda a, b: a >= b, signed=True),
    "CMP_SLT": _create_comparison_eval(lambda a, b: a < b, signed=True),
    "CMP_SLE": _create_comparison_eval(lambda a, b: a <= b, signed=True),
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
}
