# based on https://github.com/whitequark/binja-avnera/blob/main/mc/tokens.py
from . import binja_api # noqa: F401
from binaryninja import InstructionTextToken
from binaryninja.enums import InstructionTextTokenType

import enum
from typing import List, Tuple




class Token:
    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return False
        return self.__dict__ == getattr(other, "__dict__", {})

    def binja(self) -> tuple[InstructionTextTokenType, str]:
        raise NotImplementedError("binja() not implemented for {}".format(type(self)))

    def to_binja(self) -> InstructionTextToken:
        kind, data = self.binja()
        return InstructionTextToken(kind, data)


def asm(parts: List[Token]) -> List[InstructionTextToken]:
    # map all tokens using to_binja()
    return [part.to_binja() for part in parts]


def asm_str(parts: List[Token]) -> str:
    return "".join(str(part) for part in parts)


class TInstr(Token):
    def __init__(self, instr: str) -> None:
        self.instr = instr

    def __repr__(self) -> str:
        return f"TInstr({self.instr})"

    def __str__(self) -> str:
        return self.instr

    def binja(self) -> Tuple[InstructionTextTokenType, str]:
        return (InstructionTextTokenType.InstructionToken, self.instr)


class TSep(Token):
    def __init__(self, sep: str) -> None:
        self.sep = sep

    def __repr__(self) -> str:
        return f"TSep({self.sep})"

    def __str__(self) -> str:
        return self.sep

    def binja(self) -> Tuple[InstructionTextTokenType, str]:
        return (InstructionTextTokenType.OperandSeparatorToken, self.sep)


class TText(Token):
    def __init__(self, text: str) -> None:
        self.text = text

    def __repr__(self) -> str:
        return f"TText({self.text})"

    def __str__(self) -> str:
        return self.text

    def binja(self) -> Tuple[InstructionTextTokenType, str]:
        return (InstructionTextTokenType.TextToken, self.text)


class TInt(Token):
    def __init__(self, value: str) -> None:
        self.value = value

    def __repr__(self) -> str:
        return f"TInt({self.value})"

    def __str__(self) -> str:
        return str(self.value)

    def binja(self) -> Tuple[InstructionTextTokenType, str]:
        return (InstructionTextTokenType.IntegerToken, str(self.value))


class MemType(enum.Enum):
    INTERNAL = 0
    EXTERNAL = 1


class TBegMem(Token):
    def __init__(self, mem_type: MemType) -> None:
        self.mem_type = mem_type

    def __repr__(self) -> str:
        return f"TBegMem({self.mem_type})"

    def __str__(self) -> str:
        if self.mem_type == MemType.EXTERNAL:
            return "["
        return "("

    def binja(self) -> Tuple[InstructionTextTokenType, str]:
        return (InstructionTextTokenType.BeginMemoryOperandToken, self.__str__())


class TEndMem(Token):
    def __init__(self, mem_type: MemType) -> None:
        self.mem_type = mem_type

    def __repr__(self) -> str:
        return f"TEndMem({self.mem_type})"

    def __str__(self) -> str:
        if self.mem_type == MemType.EXTERNAL:
            return "]"
        return ")"

    def binja(self) -> Tuple[InstructionTextTokenType, str]:
        return (InstructionTextTokenType.EndMemoryOperandToken, self.__str__())

class TAddr(Token):
    def __init__(self, value: int) -> None:
        self.value = value

    def __repr__(self) -> str:
        return f"TAddr({self.value})"

    def __str__(self) -> str:
        return f"{self.value:05X}"

    def binja(self) -> Tuple[InstructionTextTokenType, str]:
        return (InstructionTextTokenType.PossibleAddressToken, f"{self.value:05X}")

class TReg(Token):
    def __init__(self, reg: str) -> None:
        self.reg = reg

    def __repr__(self) -> str:
        return f"TReg({self.reg})"

    def __str__(self) -> str:
        return self.reg

    def binja(self) -> Tuple[InstructionTextTokenType, str]:
        return (InstructionTextTokenType.RegisterToken, self.reg)
