from typing import Dict, Set, Callable, Optional, Any, cast
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
)
from binaryninja import (
    InstructionInfo,
)


class RegisterName(enum.Enum):
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
    # Temp Registers
    TEMP0 = "TEMP0"
    TEMP1 = "TEMP1"
    TEMP10 = "TEMP10"


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
    RegisterName.TEMP0: 3,
    RegisterName.TEMP1: 3,
    RegisterName.TEMP10: 3,
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
        RegisterName.TEMP0,
        RegisterName.TEMP1,
        RegisterName.TEMP10,
    }

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


EvalLLILType = Callable[[MockLLIL, Optional[int], Registers, Memory, State], Any]


def eval(llil: MockLLIL, regs: Registers, memory: Memory, state: State) -> Any:
    op = llil.bare_op()
    flags = llil.flags()
    size = llil.width()

    if isinstance(llil, MockIntrinsic):
        intrinsic = cast(MockIntrinsic, llil)
        op = f"INTRINSIC_{intrinsic.name}"

    f = EVAL_LLIL.get(op)
    if f is None:
        raise NotImplementedError(f"Eval for {op} not implemented")

    result = f(llil, size, regs, memory, state)

    # NOTE We must handle flags here, as depending on the context some operations
    # may not set flags, while others do. This also minimizes the number of
    # places where we need to check for flags.
    if flags is not None and flags != "0":
        if "Z" in flags:
            assert size
            zero_mask = (1 << (size * 8)) - 1
            regs.set(RegisterName.FZ, int(result & zero_mask == 0))
        if "C" in flags:
            assert size
            over_limit = int(result > (1 << (size * 8)) - 1)
            under_limit = int(result < 0)
            carry_flag = int(over_limit or under_limit)
            regs.set(RegisterName.FC, carry_flag)
        pass
    return result


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
        # set PC to the next instruction
        self.regs.set(RegisterName.PC, address + info.length)

        # Build a map: LowLevelILLabel → index in il.ils
        label_to_index: Dict[Any, int] = {}
        for idx, node in enumerate(il.ils):
            if isinstance(node, MockLabel):
                label_to_index[node.label] = idx

        # Now iterate *with* label‐aware control flow:
        pc_llil = 0
        while pc_llil < len(il.ils):
            node = il.ils[pc_llil]

            # 1) Skip over plain labels:
            if isinstance(node, MockLabel):
                pc_llil += 1
                continue

            # 2) Handle an IF‐expression:
            if isinstance(node, MockIfExpr):
                # Evaluate the condition
                cond_val = self.eval(node.cond)
                # If non‐zero: jump to the “true” label; else to the “false” label
                target_label = node.t if cond_val else node.f
                assert target_label in label_to_index, f"Unknown label {target_label}"
                pc_llil = label_to_index[target_label]
                continue

            # 4) Otherwise, it’s a “normal” MockLLIL (CONST, REG, ADD, JUMP, etc.)
            self.eval(node)
            pc_llil += 1

    def eval(self, llil: MockLLIL) -> Any:
        return eval(llil, self.regs, self.memory, self.state)


def eval_const(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> int:
    result = llil.ops[0]
    assert isinstance(result, int)
    return result


def eval_const_ptr(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> int:
    result = llil.ops[0]
    assert isinstance(result, int)
    return result


def eval_reg(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> int:
    reg = RegisterName(llil.ops[0].name)
    return regs.get(reg)


def eval_set_reg(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> None:
    reg = RegisterName(llil.ops[0].name)
    value = eval(llil.ops[1], regs, memory, state)
    regs.set(reg, value)


def eval_flag(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> int:
    flag = RegisterName(f"F{llil.ops[0].name}")
    return regs.get(flag)


def eval_set_flag(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> None:
    flag = RegisterName(f"F{llil.ops[0].name}")
    value = eval(llil.ops[1], regs, memory, state)
    regs.set(flag, value != 0)


def eval_and(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> int:
    op1, op2 = [eval(op, regs, memory, state) for op in llil.ops]
    return int(op1) & int(op2)


def eval_or(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> int:
    op1, op2 = [eval(op, regs, memory, state) for op in llil.ops]
    return int(op1) | int(op2)


def eval_xor(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> int:
    op1, op2 = [eval(op, regs, memory, state) for op in llil.ops]
    return int(op1) ^ int(op2)


def eval_pop(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> int:
    assert size
    addr = regs.get(RegisterName.S)
    result = memory.read_bytes(addr, size)
    regs.set(RegisterName.S, addr + size)
    return result


def eval_push(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> None:
    assert size
    value = eval(llil.ops[0], regs, memory, state)
    addr = regs.get(RegisterName.S) - size
    memory.write_bytes(size, addr, value)
    regs.set(RegisterName.S, addr)


def eval_nop(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> None:
    # NOP does nothing
    pass


def eval_unimpl(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> None:
    raise NotImplementedError(f"Low-level IL operation {llil.op} is not implemented")


def eval_store(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> None:
    assert size
    dest, value = [eval(i, regs, memory, state) for i in llil.ops]
    memory.write_bytes(size, dest, value)


def eval_load(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> int:
    assert size
    addr = eval(llil.ops[0], regs, memory, state)
    return memory.read_bytes(addr, size)


def eval_ret(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> None:
    addr = eval(llil.ops[0], regs, memory, state)
    if llil.ops[0].width() == 2:
        addr = regs.get(RegisterName.PC) & 0xFF0000 | addr
    regs.set(RegisterName.PC, addr)


def eval_jump(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> None:
    addr = eval(llil.ops[0], regs, memory, state)
    regs.set(RegisterName.PC, addr)


def eval_call(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> None:
    addr = eval(llil.ops[0], regs, memory, state)
    # expect the execute_instruction to advance PC to after the CALL instruction
    ret_addr = regs.get(RegisterName.PC)
    memory.write_bytes(3, regs.get(RegisterName.S) - 3, ret_addr)
    regs.set(RegisterName.S, regs.get(RegisterName.S) - 3)
    regs.set(RegisterName.PC, addr)


def eval_add(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> int:
    assert size
    op1, op2 = [eval(op, regs, memory, state) for op in llil.ops]
    return int(op1) + int(op2)


def eval_sub(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> int:
    assert size
    op1, op2 = [eval(op, regs, memory, state) for op in llil.ops]
    return int(op1) - int(op2)


def eval_cmp_e(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> int:
    assert size
    op1, op2 = [eval(op, regs, memory, state) for op in llil.ops]
    return int(op1) == int(op2)


def eval_lsl(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> int:
    assert size
    val, count = [eval(op, regs, memory, state) for op in llil.ops]
    width = size * 8
    if count == 0:
        return val

    new_carry_out = (val >> (width - count)) & 1 if count <= width else 0
    return (val << count) & ((1 << width) - 1) | (new_carry_out << width)


def eval_ror(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> int:
    assert size == 1
    val, count = [int(eval(op, regs, memory, state)) for op in llil.ops]
    width = size * 8
    new_carry_out = val & 1
    result = (val >> count) | (val << (width - count))
    result &= (1 << width) - 1  # Ensure we don't overflow the width
    return result | (new_carry_out << width)


def eval_rol(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> int:
    assert size == 1
    val, count = [int(eval(op, regs, memory, state)) for op in llil.ops]
    width = size * 8
    msb = val >> (width - count)
    new_carry_out = msb & 1
    return (val << count) | msb | (new_carry_out << width)


def eval_rrc(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> int:
    # RRC is similar to ROR but with the carry bit
    assert size == 1
    val, count, carry_in = [int(eval(op, regs, memory, state)) for op in llil.ops]
    assert count == 1
    width = size * 8
    new_carry_out = val & 1 # LSB of val
    return (val >> count) | (carry_in << (width - count)) | (new_carry_out << width)


def eval_rlc(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> int:
    # RLC is similar to ROL but with the carry bit
    assert size == 1
    val, count, carry_in = [int(eval(op, regs, memory, state)) for op in llil.ops]
    width = size * 8
    new_carry_out = (val >> (width - 1)) & 1 # MSB of val
    return (val << 1) | carry_in | (new_carry_out << width)


def eval_intrinsic_tcl(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> None:
    pass


def eval_intrinsic_halt(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> None:
    state.halted = True


def eval_intrinsic_off(
    llil: MockLLIL, size: Optional[int], regs: Registers, memory: Memory, state: State
) -> None:
    # This is a no-op in the emulator context, but could be used for debugging or logging
    state.halted = True


EVAL_LLIL: Dict[str, EvalLLILType] = {
    "CONST": eval_const,
    "CONST_PTR": eval_const_ptr,
    "REG": eval_reg,
    "SET_REG": eval_set_reg,
    "FLAG": eval_flag,
    "SET_FLAG": eval_set_flag,
    "AND": eval_and,
    "OR": eval_or,
    "XOR": eval_xor,
    "POP": eval_pop,
    "PUSH": eval_push,
    "NOP": eval_nop,
    "UNIMPL": eval_unimpl,
    "STORE": eval_store,
    "LOAD": eval_load,
    "RET": eval_ret,
    "JUMP": eval_jump,
    "CALL": eval_call,
    "ADD": eval_add,
    "SUB": eval_sub,
    "CMP_E": eval_cmp_e,
    "LSL": eval_lsl,
    "ROR": eval_ror,
    "RRC": eval_rrc,
    "ROL": eval_rol,
    "RLC": eval_rlc,
    "INTRINSIC_TCL": eval_intrinsic_tcl,
    "INTRINSIC_HALT": eval_intrinsic_halt,
    "INTRINSIC_OFF": eval_intrinsic_off,
}
