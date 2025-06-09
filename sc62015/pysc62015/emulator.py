from typing import Dict, Set, Callable, Optional, Any, cast, Tuple, TypedDict
import enum
from .coding import FetchDecoder
from dataclasses import dataclass

from .instr import (
    decode,
    OPCODES,
    Instruction,
)
from .mock_llil import (
    MockLowLevelILFunction,
    MockLLIL,
    MockLabel,
    MockIfExpr,
    MockIntrinsic,
    MockGoto,
)
from binaryninja import (
    InstructionInfo,
)


NUM_TEMP_REGISTERS = 14


class RegisterName(enum.Enum):
    """CPU register names."""

    _ignore_ = ["_i"]

    # 8-bit
    A = "A"
    B = "B"
    IL = "IL"
    IH = "IH"
    # 16-bit
    I = "I"  # noqa: E741
    BA = "BA"
    # 24-bit (3 bytes)
    X = "X"
    Y = "Y"
    U = "U"
    S = "S"
    # 20-bit (stored in 3 bytes, masked)
    PC = "PC"
    # Flags
    FC = "FC"  # Carry
    FZ = "FZ"  # Zero
    F = "F"
    # Temp Register
    for _i in range(NUM_TEMP_REGISTERS):
        locals()[f"TEMP{_i}"] = f"TEMP{_i}"
    del _i


REGISTER_SIZE: Dict[RegisterName, int] = {
    RegisterName.A: 1,  # 8-bit
    RegisterName.B: 1,  # 8-bit
    RegisterName.IL: 1,  # 8-bit
    RegisterName.IH: 1,  # 8-bit
    RegisterName.I: 2,  # 16-bit
    RegisterName.BA: 2,  # 16-bit
    RegisterName.X: 3,  # 24-bit
    RegisterName.Y: 3,  # 24-bit
    RegisterName.U: 3,  # 24-bit
    RegisterName.S: 3,  # 24-bit
    RegisterName.PC: 3,  # 20-bit (stored in 3 bytes)
    RegisterName.FC: 1,  # 1-bit
    RegisterName.FZ: 1,  # 1-bit
    RegisterName.F: 1,  # 8-bit (general flags register)
    **{getattr(RegisterName, f"TEMP{i}"): 3 for i in range(NUM_TEMP_REGISTERS)},
}


class Registers:
    BASE: Set[RegisterName] = {
        RegisterName.BA,
        RegisterName.I,
        RegisterName.X,
        RegisterName.Y,
        RegisterName.U,
        RegisterName.S,
        RegisterName.PC,
        RegisterName.F,
    } | {getattr(RegisterName, f"TEMP{i}") for i in range(NUM_TEMP_REGISTERS)}

    def __init__(self) -> None:
        self._values: Dict[RegisterName, int] = {reg: 0 for reg in self.BASE}

    def get(self, reg: RegisterName) -> int:
        if reg in self.BASE:
            return self._values[reg]

        match reg:
            case RegisterName.A:
                return self._values[RegisterName.BA] & 0xFF
            case RegisterName.B:
                return (self._values[RegisterName.BA] >> 8) & 0xFF
            case RegisterName.IL:
                return self._values[RegisterName.I] & 0xFF
            case RegisterName.IH:
                return (self._values[RegisterName.I] >> 8) & 0xFF
            case RegisterName.FC:
                return self._values[RegisterName.F] & 0x01
            case RegisterName.FZ:
                return (self._values[RegisterName.F] >> 1) & 0x01
            case _:
                raise ValueError(
                    f"Attempted to get unknown or non-base register: {reg}"
                )

    def set(self, reg: RegisterName, value: int) -> None:
        if reg in self.BASE:
            self._values[reg] = value & (1 << (REGISTER_SIZE[reg] * 8)) - 1
            return

        match reg:
            case RegisterName.A:
                self._values[RegisterName.BA] = (
                    self._values[RegisterName.BA] & 0xFF00
                ) | (value & 0xFF)
            case RegisterName.B:
                self._values[RegisterName.BA] = (
                    self._values[RegisterName.BA] & 0x00FF
                ) | ((value & 0xFF) << 8)
            case RegisterName.IL:
                self._values[RegisterName.I] = (
                    self._values[RegisterName.I] & 0xFF00
                ) | (value & 0xFF)
            case RegisterName.IH:
                self._values[RegisterName.I] = (
                    self._values[RegisterName.I] & 0x00FF
                ) | ((value & 0xFF) << 8)
            case RegisterName.FC:
                self._values[RegisterName.F] = (self._values[RegisterName.F] & 0xFE) | (
                    value & 0x01
                )
            case RegisterName.FZ:
                self._values[RegisterName.F] = (self._values[RegisterName.F] & 0xFD) | (
                    (value & 0x01) << 1
                )
            case _:
                raise ValueError(
                    f"Attempted to set unknown or non-base register: {reg}"
                )


ReadMemType = Callable[[int], int]
WriteMemType = Callable[[int, int], None]


class Memory:
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


EvalLLILType = Callable[
    [MockLLIL, Optional[int], Registers, Memory, State],
    Tuple[Optional[int], Optional[ResultFlags]],
]


def evaluate_llil(
    llil: MockLLIL, regs: Registers, memory: Memory, state: State
) -> Tuple[Optional[int], Optional[ResultFlags]]:
    op_name_bare = llil.bare_op()
    llil_flags_spec = llil.flags()  # e.g., "CZ", "Z", or None
    size = llil.width()
    current_op_name_for_eval = op_name_bare  # Will be updated for intrinsics

    if isinstance(llil, MockIntrinsic):
        intrinsic = cast(MockIntrinsic, llil)
        current_op_name_for_eval = f"INTRINSIC_{intrinsic.name}"

    f = EVAL_LLIL.get(current_op_name_for_eval)
    if f is None:
        raise NotImplementedError(
            f"Eval for {current_op_name_for_eval} not implemented"
        )

    result_value, op_defined_flags = f(llil, size, regs, memory, state)

    if llil_flags_spec is not None and llil_flags_spec != "0":
        if "Z" in llil_flags_spec:
            # We must set FZ based on either op_defined_flags or result_value
            fz_val_to_set: Optional[int] = None
            if op_defined_flags:
                fz_val_to_set = op_defined_flags.get("Z")

            if fz_val_to_set is not None:
                regs.set(RegisterName.FZ, fz_val_to_set)
            elif isinstance(result_value, int):
                assert (
                    size is not None
                ), f"FZ flag setting from result_value requires size for {current_op_name_for_eval}"
                zero_mask = (1 << (size * 8)) - 1
                regs.set(RegisterName.FZ, int((result_value & zero_mask) == 0))
            # If neither, FZ is not set by this explicit mechanism (might be implicitly correct or an oversight in arch spec)

        if "C" in llil_flags_spec:
            # We must set FC based on either op_defined_flags or result_value
            fc_val_to_set: Optional[int] = None
            if op_defined_flags:
                fc_val_to_set = op_defined_flags.get("C")

            if fc_val_to_set is not None:
                regs.set(RegisterName.FC, fc_val_to_set)
            elif isinstance(result_value, int):
                assert (
                    size is not None
                ), f"FC flag setting from result_value requires size for {current_op_name_for_eval}"
                unsigned_max_for_size = (1 << (size * 8)) - 1
                carry_flag_val = 0
                if result_value > unsigned_max_for_size:  # Unsigned overflow
                    carry_flag_val = 1
                regs.set(RegisterName.FC, carry_flag_val)
            # If neither, FC is not set by this explicit mechanism.

    return result_value, op_defined_flags


class Emulator:
    def __init__(self, memory: Memory) -> None:
        self.regs = Registers()
        self.memory = memory
        self.state = State()

    def decode_instruction(self, address: int) -> Instruction:
        def fecher(offset: int) -> int:
            return self.memory.read_byte(address + offset)

        decoder = FetchDecoder(fecher)
        return decode(decoder, address, OPCODES)  # type: ignore

    def execute_instruction(self, address: int) -> None:
        self.regs.set(RegisterName.PC, address)
        instr = self.decode_instruction(address)
        assert instr is not None, f"Failed to decode instruction at {address:04X}"

        il = MockLowLevelILFunction()
        instr.lift(il, address)

        info = InstructionInfo()
        instr.analyze(info, address)

        # MyPy Fix for line 244: Cast info.length to int.
        # Although type-hinted as int, MyPy might not be able to prove it in all contexts.
        current_instr_length = cast(int, info.length)
        assert (
            current_instr_length is not None
        ), "InstructionInfo.length was not set by analyze()"
        self.regs.set(RegisterName.PC, address + current_instr_length)

        label_to_index: Dict[Any, int] = {}
        for idx, node in enumerate(il.ils):
            if isinstance(node, MockLabel):
                label_to_index[node.label] = idx

        pc_llil = 0
        while pc_llil < len(il.ils):
            node = il.ils[pc_llil]

            if isinstance(node, MockLabel):
                pc_llil += 1
                continue

            if isinstance(node, MockIfExpr):
                # MyPy Fix for line 253: Ensure node.cond is MockLLIL for eval
                assert isinstance(
                    node.cond, MockLLIL
                ), "Condition for IF expression must be MockLLIL"
                cond_val, _ = self.evaluate(node.cond)
                assert (
                    cond_val is not None
                ), "Condition for IF expression evaluated to None"
                target_label = node.t if cond_val else node.f
                assert target_label in label_to_index, f"Unknown label {target_label}"
                pc_llil = label_to_index[target_label]
                continue

            if isinstance(node, MockGoto):
                assert (
                    node.label in label_to_index
                ), f"Unknown goto target label {node.label}"
                pc_llil = label_to_index[node.label]
                continue

            assert isinstance(node, MockLLIL), f"Expected MockLLIL, got {type(node)}"
            self.evaluate(node)
            pc_llil += 1

    def evaluate(self, llil: MockLLIL) -> Tuple[Optional[int], Optional[ResultFlags]]:
        return evaluate_llil(llil, self.regs, self.memory, self.state)


def eval_const(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> Tuple[int, Optional[ResultFlags]]:
    result = llil.ops[0]
    assert isinstance(result, int)
    return result, None


def eval_const_ptr(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> Tuple[int, Optional[ResultFlags]]:
    result = llil.ops[0]
    assert isinstance(result, int)
    return result, None


def eval_reg(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> Tuple[int, Optional[ResultFlags]]:
    reg_name_enum = RegisterName(llil.ops[0].name)
    return regs.get(reg_name_enum), None


def eval_set_reg(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> Tuple[None, Optional[ResultFlags]]:
    reg_name_enum = RegisterName(llil.ops[0].name)
    # Ensure llil.ops[1] is MockLLIL for eval
    assert isinstance(
        llil.ops[1], MockLLIL
    ), "Source operand for SET_REG must be MockLLIL"
    value_to_set, _ = evaluate_llil(llil.ops[1], regs, memory, state)
    assert value_to_set is not None, "Value for SET_REG cannot be None"
    regs.set(reg_name_enum, value_to_set)
    return None, None


def eval_flag(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> Tuple[int, Optional[ResultFlags]]:
    flag_name_enum = RegisterName(f"F{llil.ops[0].name}")
    return regs.get(flag_name_enum), None


def eval_set_flag(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> Tuple[None, Optional[ResultFlags]]:
    flag_name_enum = RegisterName(f"F{llil.ops[0].name}")
    assert isinstance(
        llil.ops[1], MockLLIL
    ), "Source operand for SET_FLAG must be MockLLIL"
    value_to_set, _ = evaluate_llil(llil.ops[1], regs, memory, state)
    assert value_to_set is not None, "Value for SET_FLAG cannot be None"
    regs.set(flag_name_enum, value_to_set != 0)
    return None, None


def _create_logical_eval(op_func: Callable[[int, int], int]) -> EvalLLILType:
    """Factory for simple logical operations."""

    def _eval(
        llil: MockLLIL,
        size: Optional[int],
        regs: Registers,
        memory: Memory,
        state: State,
    ) -> Tuple[int, Optional[ResultFlags]]:
        assert isinstance(llil.ops[0], MockLLIL) and isinstance(llil.ops[1], MockLLIL)
        op1_val, _ = evaluate_llil(llil.ops[0], regs, memory, state)
        op2_val, _ = evaluate_llil(llil.ops[1], regs, memory, state)
        assert op1_val is not None and op2_val is not None
        result = op_func(int(op1_val), int(op2_val))
        return result, {"Z": 1 if result == 0 else 0, "C": 0}

    return _eval


def _create_arithmetic_eval(
    op_func: Callable[[int, int], int],
    carry_func: Callable[[int, int, int, int], int],
) -> EvalLLILType:
    """Factory for arithmetic operations like ADD and SUB."""

    def _eval(
        llil: MockLLIL,
        size: Optional[int],
        regs: Registers,
        memory: Memory,
        state: State,
    ) -> Tuple[int, Optional[ResultFlags]]:
        assert size is not None
        assert isinstance(llil.ops[0], MockLLIL) and isinstance(llil.ops[1], MockLLIL)
        op1_val, _ = evaluate_llil(llil.ops[0], regs, memory, state)
        op2_val, _ = evaluate_llil(llil.ops[1], regs, memory, state)
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
    """Factory for shift and rotate operations."""

    def _eval(
        llil: MockLLIL,
        size: Optional[int],
        regs: Registers,
        memory: Memory,
        state: State,
    ) -> Tuple[int, Optional[ResultFlags]]:
        assert size is not None
        values = []
        for operand in llil.ops:
            assert isinstance(operand, MockLLIL)
            val, _ = evaluate_llil(operand, regs, memory, state)
            assert val is not None
            values.append(int(val))

        result, flags = op_func(size, *values)
        return result, flags

    return _eval





def eval_pop(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> Tuple[int, Optional[ResultFlags]]:
    assert size
    addr = regs.get(RegisterName.S)
    result = memory.read_bytes(addr, size)
    regs.set(RegisterName.S, addr + size)
    return result, None


def eval_push(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> Tuple[None, Optional[ResultFlags]]:
    assert size
    assert isinstance(llil.ops[0], MockLLIL)
    value_to_push, _ = evaluate_llil(llil.ops[0], regs, memory, state)
    assert value_to_push is not None
    addr = regs.get(RegisterName.S) - size
    memory.write_bytes(size, addr, value_to_push)
    regs.set(RegisterName.S, addr)
    return None, None


def eval_nop(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> Tuple[None, Optional[ResultFlags]]:
    return None, None


def eval_unimpl(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> Tuple[None, Optional[ResultFlags]]:
    raise NotImplementedError(f"Low-level IL operation {llil.op} is not implemented")


def eval_store(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> Tuple[None, Optional[ResultFlags]]:
    assert size
    assert isinstance(llil.ops[0], MockLLIL) and isinstance(llil.ops[1], MockLLIL)
    dest_addr, _ = evaluate_llil(llil.ops[0], regs, memory, state)
    value_to_store, _ = evaluate_llil(llil.ops[1], regs, memory, state)
    assert dest_addr is not None and value_to_store is not None
    memory.write_bytes(size, dest_addr, value_to_store)
    return None, None


def eval_load(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> Tuple[int, Optional[ResultFlags]]:
    assert size
    assert isinstance(llil.ops[0], MockLLIL)
    addr, _ = evaluate_llil(llil.ops[0], regs, memory, state)
    assert addr is not None
    return memory.read_bytes(addr, size), None


def eval_ret(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> Tuple[None, Optional[ResultFlags]]:
    assert isinstance(llil.ops[0], MockLLIL)
    addr_val, _ = evaluate_llil(llil.ops[0], regs, memory, state)
    assert isinstance(
        addr_val, int
    ), f"Address for RET must be an integer, got {type(addr_val)}"

    # The lifting phase already expands a 16-bit return address to a 20-bit
    # value using the current PC page, so both RET and RETF simply jump to the
    # evaluated address here.
    regs.set(RegisterName.PC, addr_val)
    return None, None


def eval_jump(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> Tuple[None, Optional[ResultFlags]]:
    assert isinstance(llil.ops[0], MockLLIL)
    addr, _ = evaluate_llil(llil.ops[0], regs, memory, state)
    assert isinstance(
        addr, int
    ), f"Address for JUMP must be an integer, got {type(addr)}"
    regs.set(RegisterName.PC, addr)
    return None, None


def eval_call(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> Tuple[None, Optional[ResultFlags]]:
    assert isinstance(llil.ops[0], MockLLIL)
    addr, _ = evaluate_llil(llil.ops[0], regs, memory, state)
    assert isinstance(
        addr, int
    ), f"Address for CALL must be an integer, got {type(addr)}"

    ret_addr = regs.get(RegisterName.PC)

    # Determine push size for return address
    # SC62015: CALL mn (2-byte target) pushes 2 bytes; CALLF lmn (3-byte target) pushes 3 bytes.
    # This distinction should ideally be made during lifting.
    # If llil.ops[0] (target addr expr) resulted from a 16-bit const, it's likely a short CALL.
    push_size = 3  # Default for CALLF / system stack operations
    if (
        llil.ops[0].op == "CONST_PTR.w"
        or (llil.ops[0].op == "CONST.w")
        or (llil.ops[0].op == "OR.l" and llil.ops[0].ops[0].op == "CONST.w")
    ):  # Heuristic for CALL mn
        push_size = 2

    stack_addr = regs.get(RegisterName.S) - push_size
    # Ensure ret_addr is masked correctly for the push size
    if push_size == 2:
        memory.write_bytes(push_size, stack_addr, ret_addr & 0xFFFF)
    else:  # push_size == 3
        memory.write_bytes(push_size, stack_addr, ret_addr & 0xFFFFF)

    regs.set(RegisterName.S, stack_addr)
    regs.set(RegisterName.PC, addr)
    return None, None




def eval_cmp_e(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> Tuple[int, Optional[ResultFlags]]:
    assert isinstance(llil.ops[0], MockLLIL) and isinstance(llil.ops[1], MockLLIL)
    op1_val, _ = evaluate_llil(llil.ops[0], regs, memory, state)
    op2_val, _ = evaluate_llil(llil.ops[1], regs, memory, state)
    assert op1_val is not None and op2_val is not None
    return int(int(op1_val) == int(op2_val)), None


def eval_cmp_ugt(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> Tuple[int, Optional[ResultFlags]]:
    assert isinstance(llil.ops[0], MockLLIL) and isinstance(llil.ops[1], MockLLIL)
    op1_val, _ = evaluate_llil(llil.ops[0], regs, memory, state)
    op2_val, _ = evaluate_llil(llil.ops[1], regs, memory, state)
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
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> Tuple[int, Optional[ResultFlags]]:
    assert size is not None, "Size must be provided for signed comparison"
    assert isinstance(llil.ops[0], MockLLIL) and isinstance(llil.ops[1], MockLLIL)
    op1_val, _ = evaluate_llil(llil.ops[0], regs, memory, state)
    op2_val, _ = evaluate_llil(llil.ops[1], regs, memory, state)
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
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> Tuple[None, Optional[ResultFlags]]:
    return None, None


def eval_intrinsic_halt(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> Tuple[None, Optional[ResultFlags]]:
    state.halted = True
    return None, None


def eval_intrinsic_off(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
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
