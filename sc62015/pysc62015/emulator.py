from typing import Dict, Set, Optional, Any, cast, Tuple
import enum
from .coding import FetchDecoder
from .constants import PC_MASK

from .instr import (
    decode,
    OPCODES,
    Instruction,
)
from binja_helpers.mock_llil import (
    MockLowLevelILFunction,
    MockLLIL,
    MockLabel,
    MockIfExpr,
    MockGoto,
)
from binja_helpers.eval_llil import (
    Memory,
    State,
    ResultFlags,
    evaluate_llil,
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
    # Temp registers
    #
    # These are generated dynamically so new temporary registers can
    # be added by simply adjusting ``NUM_TEMP_REGISTERS``. This keeps
    # the enum definition DRY and avoids repeating similar lines.
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
            val = self._values[reg]
            if reg is RegisterName.PC:
                return val & PC_MASK
            return val

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
            mask = (1 << (REGISTER_SIZE[reg] * 8)) - 1
            if reg is RegisterName.PC:
                mask = PC_MASK
            self._values[reg] = value & mask
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

    def get_by_name(self, name: str) -> int:
        return self.get(RegisterName[name])

    def set_by_name(self, name: str, value: int) -> None:
        self.set(RegisterName[name], value)



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

