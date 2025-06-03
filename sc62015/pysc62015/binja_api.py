"""Helper for importing the Binary Ninja Python API during tests.

The real Binary Ninja installation might not be available when running unit
tests.  This module tries several strategies to make the ``binaryninja``
package importable:

1.  If a local Binary Ninja installation exists, its ``python`` directory is
    added to ``sys.path``.
2.  If that fails, the open source API is downloaded from GitHub and extracted
    into a temporary directory which is then added to ``sys.path``.
3.  As a last resort a very small stub implementation providing only the
    classes and enums used in the tests is installed.

Import this module before importing anything from ``binaryninja``.
"""

from __future__ import annotations

import enum
import io
import os
import sys
import tempfile
import types
from dataclasses import dataclass

_binja_install = os.path.expanduser(
    "~/Applications/Binary Ninja.app/Contents/Resources/python/"
)
if os.path.isdir(_binja_install) and _binja_install not in sys.path:
    sys.path.append(_binja_install)


def _has_binja() -> bool:
    try:
        import binaryninja  # noqa: F401

        return True
    except Exception:
        return False


if not _has_binja():
    try:  # Attempt to fetch the API from GitHub
        import urllib.request
        import zipfile

        url = (
            "https://github.com/Vector35/binaryninja-api/"
            "archive/refs/heads/dev.zip"
        )
        cache_dir = os.path.join(tempfile.gettempdir(), "binaryninja_api_dev")
        if not os.path.exists(cache_dir):
            data = urllib.request.urlopen(url).read()
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                zf.extractall(cache_dir)
        for sub in os.listdir(cache_dir):
            candidate = os.path.join(cache_dir, sub, "python")
            if os.path.isdir(candidate):
                sys.path.append(candidate)
                break
    except Exception:
        pass


if not _has_binja():
    # Final fallback: provide a tiny stub with the pieces the tests rely on.
    bn = types.ModuleType("binaryninja")
    sys.modules["binaryninja"] = bn

    enums_mod = types.ModuleType("binaryninja.enums")

    class BranchType(enum.Enum):
        UnconditionalBranch = 0
        TrueBranch = 1
        FalseBranch = 2
        CallDestination = 3
        FunctionReturn = 4

    class InstructionTextTokenType(enum.Enum):
        InstructionToken = 0
        OperandSeparatorToken = 1
        RegisterToken = 2
        IntegerToken = 3
        PossibleAddressToken = 4
        BeginMemoryOperandToken = 5
        EndMemoryOperandToken = 6
        TextToken = 7

    class SegmentFlag(enum.IntFlag):
        SegmentReadable = 1
        SegmentWritable = 2
        SegmentExecutable = 4

    class SectionSemantics(enum.Enum):
        ReadOnlyCodeSectionSemantics = 0
        ReadWriteDataSectionSemantics = 1

    class SymbolType(enum.Enum):
        FunctionSymbol = 0

    class Endianness(enum.Enum):
        LittleEndian = 0
        BigEndian = 1

    class FlagRole(enum.Enum):
        ZeroFlagRole = 0
        CarryFlagRole = 1

    enums_mod.BranchType = BranchType
    enums_mod.InstructionTextTokenType = InstructionTextTokenType
    enums_mod.SegmentFlag = SegmentFlag
    enums_mod.SectionSemantics = SectionSemantics
    enums_mod.SymbolType = SymbolType
    enums_mod.Endianness = Endianness
    enums_mod.FlagRole = FlagRole

    bn.enums = enums_mod
    sys.modules["binaryninja.enums"] = enums_mod

    @dataclass
    class InstructionTextToken:
        type: InstructionTextTokenType
        text: str

    bn.InstructionTextToken = InstructionTextToken

    types_mod = types.ModuleType("binaryninja.types")

    @dataclass
    class Symbol:
        type: SymbolType
        addr: int
        name: str

    types_mod.Symbol = Symbol
    bn.types = types_mod
    sys.modules["binaryninja.types"] = types_mod

    binaryview_mod = types.ModuleType("binaryninja.binaryview")

    class BinaryView:
        def __init__(self, *args, **kwargs) -> None:
            pass

    binaryview_mod.BinaryView = BinaryView
    bn.binaryview = binaryview_mod
    sys.modules["binaryninja.binaryview"] = binaryview_mod

    arch_mod = types.ModuleType("binaryninja.architecture")

    class Architecture:
        name = ""

        @classmethod
        def __getitem__(cls, _name: str) -> "Architecture":
            return cls()

    class RegisterName(str):
        def __new__(cls, name: str):
            obj = str.__new__(cls, name)
            obj.name = name
            return obj

    class IntrinsicName(str):
        def __new__(cls, name: str):
            obj = str.__new__(cls, name)
            obj.name = name
            return obj

    class FlagName(str):
        def __new__(cls, name: str):
            obj = str.__new__(cls, name)
            obj.name = name
            return obj

    arch_mod.Architecture = Architecture
    arch_mod.RegisterName = RegisterName
    arch_mod.IntrinsicName = IntrinsicName
    arch_mod.FlagName = FlagName
    bn.architecture = arch_mod
    sys.modules["binaryninja.architecture"] = arch_mod

    llil_mod = types.ModuleType("binaryninja.lowlevelil")

    class ExpressionIndex(int):
        pass

    def LLIL_TEMP(n: int) -> ExpressionIndex:
        return ExpressionIndex(0x80000000 + n)

    class LowLevelILFunction:
        def _op(self, name: str, size: int | None, *ops: object, flags: object | None = None) -> object:
            from types import SimpleNamespace

            return self.expr(SimpleNamespace(name=f"LLIL_{name}"), *ops, size=size, flags=flags)

        def unimplemented(self) -> object:
            return self._op("UNIMPL", None)

        def nop(self) -> object:
            return self._op("NOP", None)

        def const(self, size: int, value: int) -> object:
            return self._op("CONST", size, value)

        def const_pointer(self, size: int, value: int) -> object:
            return self._op("CONST_PTR", size, value)

        def reg(self, size: int, reg: object) -> object:
            if isinstance(reg, llil_mod.ExpressionIndex):
                reg = mreg(f"TEMP{reg - 0x80000000}")
            elif isinstance(reg, str):
                reg = mreg(reg)
            return self._op("REG", size, reg)

        def set_reg(self, size: int, reg: object, value: object) -> object:
            if isinstance(reg, llil_mod.ExpressionIndex):
                reg = mreg(f"TEMP{reg - 0x80000000}")
            elif isinstance(reg, str):
                reg = mreg(reg)
            return self._op("SET_REG", size, reg, value, flags=0)

        def add(self, size: int, a: object, b: object, flags: object | None = None) -> object:
            return self._op("ADD", size, a, b, flags=flags)

        def sub(self, size: int, a: object, b: object, flags: object | None = None) -> object:
            return self._op("SUB", size, a, b, flags=flags)

        def and_expr(self, size: int, a: object, b: object, flags: object | None = None) -> object:
            return self._op("AND", size, a, b, flags=flags)

        def or_expr(self, size: int, a: object, b: object, flags: object | None = None) -> object:
            return self._op("OR", size, a, b, flags=flags)

        def xor_expr(self, size: int, a: object, b: object, flags: object | None = None) -> object:
            return self._op("XOR", size, a, b, flags=flags)

        def shift_left(self, size: int, a: object, b: object, flags: object | None = None) -> object:
            return self._op("LSL", size, a, b, flags=flags)

        def logical_shift_right(self, size: int, a: object, b: object, flags: object | None = None) -> object:
            return self._op("LSR", size, a, b, flags=flags)

        def rotate_left(self, size: int, a: object, b: object, flags: object | None = None) -> object:
            return self._op("ROL", size, a, b, flags=flags)

        def rotate_right(self, size: int, a: object, b: object, flags: object | None = None) -> object:
            return self._op("ROR", size, a, b, flags=flags)

        def rotate_left_carry(
            self, size: int, a: object, b: object, carry: object, flags: object | None = None
        ) -> object:
            return self._op("RLC", size, a, b, carry, flags=flags)

        def rotate_right_carry(
            self, size: int, a: object, b: object, carry: object, flags: object | None = None
        ) -> object:
            return self._op("RRC", size, a, b, carry, flags=flags)

        def compare_equal(self, size: int, a: object, b: object) -> object:
            return self._op("CMP_E", size, a, b)

        def compare_signed_less_than(self, size: int, a: object, b: object) -> object:
            return self._op("CMP_SLT", size, a, b)

        def compare_unsigned_greater_than(self, size: int, a: object, b: object) -> object:
            return self._op("CMP_UGT", size, a, b)

        def flag(self, flag: object) -> object:
            if isinstance(flag, str):
                flag = MockFlag(flag)
            return self._op("FLAG", None, flag)

        def set_flag(self, flag: object, value: object) -> object:
            if isinstance(flag, str):
                flag = MockFlag(flag)
            return self._op("SET_FLAG", None, flag, value)

        def load(self, size: int, addr: object) -> object:
            return self._op("LOAD", size, addr)

        def store(self, size: int, addr: object, value: object) -> object:
            return self._op("STORE", size, addr, value)

        def push(self, size: int, value: object) -> object:
            return self._op("PUSH", size, value)

        def pop(self, size: int) -> object:
            return self._op("POP", size)

        def jump(self, dest: object) -> object:
            return self._op("JUMP", None, dest)

        def call(self, dest: object) -> object:
            return self._op("CALL", None, dest)

        def ret(self, dest: object | None = None) -> object:
            ops = [] if dest is None else [dest]
            return self._op("RET", None, *ops)

    class LowLevelILLabel:
        pass

    @dataclass
    class ILSourceLocation:
        instr_index: int = 0

    llil_mod.ExpressionIndex = ExpressionIndex
    llil_mod.LLIL_TEMP = LLIL_TEMP
    llil_mod.LowLevelILFunction = LowLevelILFunction
    llil_mod.LowLevelILLabel = LowLevelILLabel
    llil_mod.ILSourceLocation = ILSourceLocation

    bn.lowlevelil = llil_mod
    sys.modules["binaryninja.lowlevelil"] = llil_mod
    from .mock_llil import mreg, MockFlag

    @dataclass
    class InstructionInfo:
        length: int = 0

        def add_branch(self, *args: object, **kwargs: object) -> None:
            pass

    bn.InstructionInfo = InstructionInfo

    @dataclass
    class RegisterInfo:
        name: str
        size: int
        offset: int = 0

    bn.RegisterInfo = RegisterInfo

    @dataclass
    class IntrinsicInfo:
        inputs: list
        outputs: list

    bn.IntrinsicInfo = IntrinsicInfo

    class CallingConvention:
        pass

    bn.CallingConvention = CallingConvention
    bn.Architecture = Architecture

    def log_error(msg: str) -> None:
        print(msg, file=sys.stderr)

    bn.log_error = log_error
    sys.modules["binaryninja"] = bn

