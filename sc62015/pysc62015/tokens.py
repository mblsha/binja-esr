# based on https://github.com/whitequark/binja-avnera/blob/main/mc/tokens.py
from .binja_api import *
from binaryninja import InstructionTextToken
from binaryninja.enums import InstructionTextTokenType

import enum


def token(kind, text, *data):
    if InstructionTextToken is None:
        return text
    else:
        if kind == "instr":
            tokenType = InstructionTextTokenType.InstructionToken
        elif kind == "opsep":
            tokenType = InstructionTextTokenType.OperandSeparatorToken
        elif kind == "reg":
            tokenType = InstructionTextTokenType.RegisterToken
        elif kind == "int":
            tokenType = InstructionTextTokenType.IntegerToken
        elif kind == "addr":
            tokenType = InstructionTextTokenType.PossibleAddressToken
        elif kind == "begmem":
            tokenType = InstructionTextTokenType.BeginMemoryOperandToken
        elif kind == "endmem":
            tokenType = InstructionTextTokenType.EndMemoryOperandToken
        elif kind == "text":
            tokenType = InstructionTextTokenType.TextToken
        else:
            raise ValueError("Invalid token kind {}".format(kind))
        return InstructionTextToken(tokenType, text, *data)


def asm(parts):
    # map all tokens using to_binja()
    return [part.to_binja() for part in parts]


def asm_str(parts):
    return "".join(str(part) for part in parts)


class Token:
    def __eq__(self, other):
        return repr(self) == repr(other)

    def binja(self):
        raise NotImplementedError("binja() not implemented for {}".format(type(self)))

    def to_binja(self):
        kind, data = self.binja()
        return InstructionTextToken(kind, data)


class TInstr(Token):
    def __init__(self, instr):
        self.instr = instr

    def __repr__(self):
        return f"TInstr({self.instr})"

    def __str__(self):
        return self.instr

    def binja(self):
        return (InstructionTextTokenType.InstructionToken, self.instr)


class TSep(Token):
    def __init__(self, sep):
        self.sep = sep

    def __repr__(self):
        return f"TSep({self.sep})"

    def __str__(self):
        return self.sep

    def binja(self):
        return (InstructionTextTokenType.OperandSeparatorToken, self.sep)


class TText(Token):
    def __init__(self, text):
        self.text = text

    def __repr__(self):
        return f"TText({self.text})"

    def __str__(self):
        return self.text

    def binja(self):
        return (InstructionTextTokenType.TextToken, self.text)


class TInt(Token):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"TInt({self.value})"

    def __str__(self):
        return str(self.value)

    def binja(self):
        return (InstructionTextTokenType.IntegerToken, str(self.value))


class MemType(enum.Enum):
    INTERNAL = 0
    EXTERNAL = 1


class TBegMem(Token):
    def __init__(self, mem_type):
        self.mem_type = mem_type

    def __repr__(self):
        return f"TBegMem({self.mem_type})"

    def __str__(self):
        if self.mem_type == MemType.EXTERNAL:
            return "["
        return "("

    def binja(self):
        return (InstructionTextTokenType.BeginMemoryOperandToken, self.__str__())


class TEndMem(Token):
    def __init__(self, mem_type):
        self.mem_type = mem_type

    def __repr__(self):
        return f"TEndMem({self.mem_type})"

    def __str__(self):
        if self.mem_type == MemType.EXTERNAL:
            return "]"
        return ")"

    def binja(self):
        return (InstructionTextTokenType.EndMemoryOperandToken, self.__str__())


class TAddr(Token):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"TAddr({self.value})"

    def __str__(self):
        return str(self.value)

    def binja(self):
        return (InstructionTextTokenType.PossibleAddressToken, str(self.value))


class TReg(Token):
    def __init__(self, reg):
        self.reg = reg

    def __repr__(self):
        return f"TReg({self.reg})"

    def __str__(self):
        return self.reg

    def binja(self):
        return (InstructionTextTokenType.RegisterToken, self.reg)
