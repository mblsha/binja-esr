"""Utility helpers for working with Binary Ninja instruction tokens."""

# based on https://github.com/whitequark/binja-avnera/blob/main/mc/tokens.py


from . import binja_api  # noqa: F401 -- make sure Binary Ninja stubs are loaded
from binaryninja import InstructionTextToken
from binaryninja.enums import InstructionTextTokenType

import enum
from typing import Iterable, Any, List, Tuple, ClassVar
from dataclasses import dataclass


class Token:
    """Base class for all renderable tokens."""

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return False
        return self.__dict__ == getattr(other, "__dict__", {})

    def binja(self) -> Tuple[InstructionTextTokenType, str]:
        raise NotImplementedError(f"binja() not implemented for {type(self)}")

    def to_binja(self) -> InstructionTextToken:
        kind, data = self.binja()
        return InstructionTextToken(kind, data)


def asm(parts: List[Token]) -> List[InstructionTextToken]:
    """Convert tokens to Binary Ninja ``InstructionTextToken`` objects."""

    return [part.to_binja() for part in parts]


def asm_str(parts: Iterable[Any]) -> str:
    """Return a human readable assembly string for a sequence of tokens."""

    return "".join(str(part) for part in parts)


@dataclass
class _BaseToken(Token):
    """Simple token carrying a string value."""

    value: str
    token_type: ClassVar[InstructionTextTokenType]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value})"

    def __str__(self) -> str:
        return self.value

    def binja(self) -> Tuple[InstructionTextTokenType, str]:
        return (self.token_type, self.__str__())


@dataclass
class TInstr(_BaseToken):
    token_type: ClassVar[InstructionTextTokenType] = InstructionTextTokenType.InstructionToken


@dataclass
class TSep(_BaseToken):
    token_type: ClassVar[InstructionTextTokenType] = InstructionTextTokenType.OperandSeparatorToken


@dataclass
class TText(_BaseToken):
    token_type: ClassVar[InstructionTextTokenType] = InstructionTextTokenType.TextToken


@dataclass
class TInt(_BaseToken):
    token_type: ClassVar[InstructionTextTokenType] = InstructionTextTokenType.IntegerToken


class MemType(enum.Enum):
    INTERNAL = 0
    EXTERNAL = 1


@dataclass
class TBegMem(Token):
    mem_type: MemType

    token_type: ClassVar[InstructionTextTokenType] = InstructionTextTokenType.BeginMemoryOperandToken

    def __repr__(self) -> str:
        return f"TBegMem({self.mem_type})"

    def __str__(self) -> str:
        return "[" if self.mem_type == MemType.EXTERNAL else "("

    def binja(self) -> Tuple[InstructionTextTokenType, str]:
        return (self.token_type, self.__str__())


@dataclass
class TEndMem(Token):
    mem_type: MemType

    token_type: ClassVar[InstructionTextTokenType] = InstructionTextTokenType.EndMemoryOperandToken

    def __repr__(self) -> str:
        return f"TEndMem({self.mem_type})"

    def __str__(self) -> str:
        return "]" if self.mem_type == MemType.EXTERNAL else ")"

    def binja(self) -> Tuple[InstructionTextTokenType, str]:
        return (self.token_type, self.__str__())


@dataclass
class TAddr(Token):
    value: int

    token_type: ClassVar[InstructionTextTokenType] = InstructionTextTokenType.PossibleAddressToken

    def __repr__(self) -> str:
        return f"TAddr({self.value})"

    def __str__(self) -> str:
        return f"{self.value:05X}"

    def binja(self) -> Tuple[InstructionTextTokenType, str]:
        return (self.token_type, self.__str__())

@dataclass
class TReg(_BaseToken):
    token_type: ClassVar[InstructionTextTokenType] = InstructionTextTokenType.RegisterToken

