from typing import Dict, Set, Callable, Any
import enum
from .coding import FetchDecoder

from .instr import (
    decode,
    OPCODES,
)
# from .mock_llil import MockLowLevelILFunction, MockLLIL, MockReg, MockFlag


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


class Emulator:
    read_mem: Callable[[int], int]
    write_mem: Callable[[int, int], None]

    def __init__(self) -> None:
        self.regs = Registers()

    def execute_instruction(self, address: int) -> Any:
        self.regs.set(RegisterName.PC, address)
        def fecher(offset: int) -> int:
           return self.read_mem(self.regs.get(RegisterName.PC) + offset)
        decoder = FetchDecoder(fecher)
        instr = decode(decoder, address, OPCODES)
        return instr
        # FIXME: Implement instruction execution logic
