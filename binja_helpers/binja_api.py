"""Helper for importing the Binary Ninja Python API during tests.

1.  If a local Binary Ninja installation exists, its ``python`` directory is
    added to ``sys.path``.
2.  As a last resort a very small stub implementation providing only the
    classes and enums used in the tests is installed.

Import this module before importing anything from ``binaryninja``.
"""

from __future__ import annotations

import enum
import os
import sys
import types
from dataclasses import dataclass
from typing import Any, Optional

# Force use of mock when FORCE_BINJA_MOCK environment variable is set
_force_mock = os.environ.get("FORCE_BINJA_MOCK", "").lower() in ("1", "true", "yes")

if not _force_mock:
    _binja_install = os.path.expanduser(
        "~/Applications/Binary Ninja.app/Contents/Resources/python/"
    )
    if os.path.isdir(_binja_install) and _binja_install not in sys.path:
        sys.path.append(_binja_install)


def _has_binja() -> bool:
    if _force_mock:
        return False
    try:
        import binaryninja  # noqa: F401

        return True
    except ImportError:
        return False


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
        NegativeSignFlagRole = 0
        ZeroFlagRole = 1
        OverflowFlagRole = 2
        CarryFlagRole = 3

    class ImplicitRegisterExtend(enum.Enum):
        SignExtendToFullWidth = 0

    enums_mod.BranchType = BranchType  # type: ignore [attr-defined]
    enums_mod.InstructionTextTokenType = InstructionTextTokenType  # type: ignore [attr-defined]
    enums_mod.SegmentFlag = SegmentFlag  # type: ignore [attr-defined]
    enums_mod.SectionSemantics = SectionSemantics  # type: ignore [attr-defined]
    enums_mod.SymbolType = SymbolType  # type: ignore [attr-defined]
    enums_mod.Endianness = Endianness  # type: ignore [attr-defined]
    enums_mod.FlagRole = FlagRole  # type: ignore [attr-defined]
    enums_mod.ImplicitRegisterExtend = ImplicitRegisterExtend  # type: ignore [attr-defined]

    bn.enums = enums_mod  # type: ignore [attr-defined]
    sys.modules["binaryninja.enums"] = enums_mod

    @dataclass
    class InstructionTextToken:
        type: InstructionTextTokenType
        text: str

    bn.InstructionTextToken = InstructionTextToken  # type: ignore [attr-defined]

    types_mod = types.ModuleType("binaryninja.types")

    @dataclass
    class Symbol:
        type: SymbolType
        addr: int
        name: str

    types_mod.Symbol = Symbol  # type: ignore [attr-defined]
    bn.types = types_mod  # type: ignore [attr-defined]
    sys.modules["binaryninja.types"] = types_mod

    binaryview_mod = types.ModuleType("binaryninja.binaryview")

    # Mock configuration registry for architecture-specific settings
    _mock_config = {
        "default_filename": "test.bin",
        "default_memory_size": 0x100000,  # 1MB
    }

    def configure_mock_binaryview(filename: Optional[str] = None, memory_size: Optional[int] = None) -> None:
        """Configure mock BinaryView defaults for testing."""
        if filename is not None:
            _mock_config["default_filename"] = filename
        if memory_size is not None:
            _mock_config["default_memory_size"] = memory_size

    class BinaryView:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            # Extract configuration from kwargs with fallbacks to global config
            filename = kwargs.get("filename", _mock_config["default_filename"])
            memory_size = kwargs.get("memory_size", _mock_config["default_memory_size"])

            # Mock file object - configurable filename
            self.file = types.SimpleNamespace(filename=filename)
            # Memory buffer for testing - configurable size
            self._memory = bytearray(memory_size)
            self.state: Any = None

        def read(self, addr: int, length: int) -> bytes:
            """Read bytes from mock memory."""
            if addr + length > len(self._memory):
                return b'\x00' * length
            return bytes(self._memory[addr:addr+length])

        def write_memory(self, addr: int, data: bytes) -> None:
            """Write data to mock memory for testing."""
            if addr + len(data) <= len(self._memory):
                self._memory[addr:addr+len(data)] = data

    # Add configuration function to the module
    binaryview_mod.configure_mock_binaryview = configure_mock_binaryview  # type: ignore [attr-defined]
    binaryview_mod.BinaryView = BinaryView  # type: ignore [attr-defined]
    bn.binaryview = binaryview_mod  # type: ignore [attr-defined]

    # Also expose configuration at top level for easy access
    bn.configure_mock_binaryview = configure_mock_binaryview  # type: ignore [attr-defined]

    sys.modules["binaryninja.binaryview"] = binaryview_mod

    arch_mod = types.ModuleType("binaryninja.architecture")

    class Architecture:
        name: str = ""  # Added type hint

        @classmethod
        def __class_getitem__(cls, _name: str) -> "Architecture":
            return cls()

    class RegisterName(str):
        name: str  # Declare the attribute

        def __new__(cls, name: str) -> "RegisterName":
            obj = str.__new__(cls, name)
            obj.name = name
            return obj

    class IntrinsicName(str):
        name: str  # Declare the attribute

        def __new__(cls, name: str) -> "IntrinsicName":
            obj = str.__new__(cls, name)
            obj.name = name
            return obj

    class FlagName(str):
        name: str  # Declare the attribute

        def __new__(cls, name: str) -> "FlagName":
            obj = str.__new__(cls, name)
            obj.name = name
            return obj

    class FlagWriteTypeName(str):
        name: str  # Declare the attribute

        def __new__(cls, name: str) -> "FlagWriteTypeName":
            obj = str.__new__(cls, name)
            obj.name = name
            return obj

    arch_mod.Architecture = Architecture  # type: ignore [attr-defined]
    arch_mod.RegisterName = RegisterName  # type: ignore [attr-defined]
    arch_mod.IntrinsicName = IntrinsicName  # type: ignore [attr-defined]
    arch_mod.FlagName = FlagName  # type: ignore [attr-defined]
    arch_mod.FlagWriteTypeName = FlagWriteTypeName  # type: ignore [attr-defined]
    bn.architecture = arch_mod  # type: ignore [attr-defined]
    sys.modules["binaryninja.architecture"] = arch_mod

    # Expose common architecture types at the module root so imports like
    # ``from binaryninja import RegisterName`` succeed when using this mock.
    bn.RegisterName = RegisterName  # type: ignore[attr-defined]
    bn.IntrinsicName = IntrinsicName  # type: ignore[attr-defined]
    bn.FlagName = FlagName  # type: ignore[attr-defined]
    bn.FlagWriteTypeName = FlagWriteTypeName  # type: ignore[attr-defined]

    llil_mod = types.ModuleType("binaryninja.lowlevelil")

    class ExpressionIndex(int):
        pass

    def LLIL_TEMP(n: int) -> ExpressionIndex:
        return ExpressionIndex(0x80000000 + n)

    class LowLevelILFunction:
        def expr(
            self,
            op_ns: Any,
            *ops: object,
            size: int | None,
            flags: object | None = None,
        ) -> object:
            from types import SimpleNamespace

            op_name_raw = getattr(op_ns, "name", "LLIL_UNKNOWN")

            name_to_process = op_name_raw
            if name_to_process.startswith("LLIL_"):
                name_to_process = name_to_process[5:]

            SZ_LOOKUP_DICT = {1: ".b", 2: ".w", 3: ".l"}

            final_op_name_parts = [name_to_process]
            if size is not None and size in SZ_LOOKUP_DICT:
                final_op_name_parts.append(SZ_LOOKUP_DICT[size])
            elif size is not None:
                final_op_name_parts.append(f".{size}")

            if flags is not None and str(flags) != "0":
                final_op_name_parts.append(f"{{{flags}}}")

            final_op_name = "".join(final_op_name_parts)
            # Return a SimpleNamespace that might be compatible with what eval expects from a MockLLIL
            # This provides .op, .ops, and lambda-based .width(), .flags(), .bare_op()
            return SimpleNamespace(
                op=final_op_name,
                ops=list(ops),
                width=lambda: size,
                flags=lambda: str(flags) if flags is not None else None,
                bare_op=lambda: name_to_process,
            )

        def _op(
            self, name: str, size: int | None, *ops: object, flags: object | None = None
        ) -> object:
            from types import SimpleNamespace  # Ensure SimpleNamespace is available

            return self.expr(
                SimpleNamespace(name=f"LLIL_{name}"), *ops, size=size, flags=flags
            )

        def unimplemented(self) -> object:
            return self._op("UNIMPL", None)

        def nop(self) -> object:
            return self._op("NOP", None)

        def const(self, size: int, value: int) -> object:
            return self._op("CONST", size, value)

        def const_pointer(self, size: int, value: int) -> object:
            return self._op("CONST_PTR", size, value)

        def reg(
            self, size: int, reg_obj: object
        ) -> object:  # Renamed reg to reg_obj to avoid conflict
            # Ensure llil_mod is accessible if used here for ExpressionIndex comparison
            # The check `isinstance(reg, llil_mod.ExpressionIndex)` might need llil_mod to be defined earlier
            # or this logic should be guarded if llil_mod is this current module being defined.
            # For the stub, we can assume `reg` is either str or our `ExpressionIndex`.
            processed_reg_obj = reg_obj
            if isinstance(
                reg_obj, ExpressionIndex
            ):  # No need for llil_mod. here as ExpressionIndex is local
                processed_reg_obj = mreg(
                    f"TEMP{reg_obj - 0x80000000}"
                )  # mreg is from .mock_llil
            elif isinstance(reg_obj, str):
                processed_reg_obj = mreg(reg_obj)  # mreg is from .mock_llil
            return self._op("REG", size, processed_reg_obj)

        def set_reg(self, size: int, reg_obj: object, value: object) -> object:
            processed_reg_obj = reg_obj
            if isinstance(reg_obj, ExpressionIndex):
                processed_reg_obj = mreg(f"TEMP{reg_obj - 0x80000000}")
            elif isinstance(reg_obj, str):
                processed_reg_obj = mreg(reg_obj)
            return self._op(
                "SET_REG", size, processed_reg_obj, value, flags="0"
            )  # Explicitly pass flags="0"

        def add(
            self, size: int, a: object, b: object, flags: object | None = None
        ) -> object:
            return self._op("ADD", size, a, b, flags=flags)

        def sub(
            self, size: int, a: object, b: object, flags: object | None = None
        ) -> object:
            return self._op("SUB", size, a, b, flags=flags)

        def mult(
            self, size: int, a: object, b: object, flags: object | None = None
        ) -> object:
            return self._op("MUL", size, a, b, flags=flags)

        def div_signed(
            self, size: int, a: object, b: object, flags: object | None = None
        ) -> object:
            return self._op("DIVS", size, a, b, flags=flags)

        def and_expr(
            self, size: int, a: object, b: object, flags: object | None = None
        ) -> object:
            return self._op("AND", size, a, b, flags=flags)

        def or_expr(
            self, size: int, a: object, b: object, flags: object | None = None
        ) -> object:
            return self._op("OR", size, a, b, flags=flags)

        def xor_expr(
            self, size: int, a: object, b: object, flags: object | None = None
        ) -> object:
            return self._op("XOR", size, a, b, flags=flags)

        def shift_left(
            self, size: int, a: object, b: object, flags: object | None = None
        ) -> object:
            return self._op("LSL", size, a, b, flags=flags)

        def logical_shift_right(
            self, size: int, a: object, b: object, flags: object | None = None
        ) -> object:
            return self._op("LSR", size, a, b, flags=flags)

        def rotate_left(
            self, size: int, a: object, b: object, flags: object | None = None
        ) -> object:
            return self._op("ROL", size, a, b, flags=flags)

        def rotate_right(
            self, size: int, a: object, b: object, flags: object | None = None
        ) -> object:
            return self._op("ROR", size, a, b, flags=flags)

        def rotate_left_carry(
            self,
            size: int,
            a: object,
            b: object,
            carry: object,
            flags: object | None = None,
        ) -> object:
            return self._op("RLC", size, a, b, carry, flags=flags)

        def rotate_right_carry(
            self,
            size: int,
            a: object,
            b: object,
            carry: object,
            flags: object | None = None,
        ) -> object:
            return self._op("RRC", size, a, b, carry, flags=flags)

        def compare_equal(self, size: int, a: object, b: object) -> object:
            return self._op("CMP_E", size, a, b)

        def compare_not_equal(self, size: int, a: object, b: object) -> object:
            return self._op("CMP_NE", size, a, b)

        def compare_signed_less_than(self, size: int, a: object, b: object) -> object:
            return self._op("CMP_SLT", size, a, b)

        def compare_signed_greater_than(self, size: int, a: object, b: object) -> object:
            return self._op("CMP_SGT", size, a, b)

        def compare_signed_less_equal(self, size: int, a: object, b: object) -> object:
            return self._op("CMP_SLE", size, a, b)

        def compare_signed_greater_equal(self, size: int, a: object, b: object) -> object:
            return self._op("CMP_SGE", size, a, b)

        def compare_unsigned_greater_than(
            self, size: int, a: object, b: object
        ) -> object:
            return self._op("CMP_UGT", size, a, b)

        def flag(self, flag_obj: object) -> object:  # Renamed flag to flag_obj
            processed_flag_obj = flag_obj
            if isinstance(flag_obj, str):
                processed_flag_obj = MockFlag(flag_obj)  # MockFlag is from .mock_llil
            return self._op("FLAG", None, processed_flag_obj)

        def set_flag(self, flag_obj: object, value: object) -> object:
            processed_flag_obj = flag_obj
            if isinstance(flag_obj, str):
                processed_flag_obj = MockFlag(flag_obj)
            return self._op("SET_FLAG", None, processed_flag_obj, value)

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

        def intrinsic(self, outputs: list[Any], name: str, inputs: list[Any]) -> object:
            return self._op("INTRINSIC", None, outputs, name, inputs)

    class LowLevelILLabel:
        pass

    @dataclass
    class ILSourceLocation:
        instr_index: int = 0

    llil_mod.ExpressionIndex = ExpressionIndex  # type: ignore [attr-defined]
    llil_mod.LLIL_TEMP = LLIL_TEMP  # type: ignore [attr-defined]
    llil_mod.LowLevelILFunction = LowLevelILFunction  # type: ignore [attr-defined]
    llil_mod.LowLevelILLabel = LowLevelILLabel  # type: ignore [attr-defined]
    llil_mod.ILSourceLocation = ILSourceLocation  # type: ignore [attr-defined]

    bn.lowlevelil = llil_mod  # type: ignore [attr-defined]
    sys.modules["binaryninja.lowlevelil"] = llil_mod
    from .mock_llil import (
        mreg,
        MockFlag,
    )  # mreg, MockFlag are used in the stub LowLevelILFunction

    @dataclass
    class InstructionInfo:
        length: int = 0

        def add_branch(self, *args: object, **kwargs: object) -> None:
            pass

    bn.InstructionInfo = InstructionInfo  # type: ignore [attr-defined]

    @dataclass
    class RegisterInfo:
        def __init__(self, name: str, size: int, offset: int = 0, extend: Any = None) -> None:
            self.name = name
            self.size = size
            self.offset = offset
            self.extend = extend

    bn.RegisterInfo = RegisterInfo  # type: ignore [attr-defined]

    @dataclass
    class IntrinsicInfo:
        inputs: list[Any]  # Use list[Any] for stub
        outputs: list[Any]  # Use list[Any] for stub

    bn.IntrinsicInfo = IntrinsicInfo  # type: ignore [attr-defined]

    class CallingConvention:
        pass

    bn.CallingConvention = CallingConvention  # type: ignore [attr-defined]
    # bn.Architecture = Architecture was listed, but arch_mod.Architecture is already set
    # and bn.architecture = arch_mod. If bn.Architecture (the class) is needed directly:
    bn.Architecture = Architecture  # type: ignore [attr-defined]

    def log_error(msg: str) -> None:
        print(msg, file=sys.stderr)

    bn.log_error = log_error  # type: ignore [attr-defined]

    log_mod = types.ModuleType("binaryninja.log")
    log_mod.log_error = log_error  # type: ignore [attr-defined]
    bn.log = log_mod  # type: ignore [attr-defined]
    sys.modules["binaryninja.log"] = log_mod

    # Add UIContext mock for interaction module
    interaction_mod = types.ModuleType("binaryninja.interaction")

    class UIContext:
        @staticmethod
        def activeContext() -> None:
            return None

    interaction_mod.UIContext = UIContext  # type: ignore [attr-defined]
    bn.interaction = interaction_mod  # type: ignore [attr-defined]
    bn.UIContext = UIContext  # type: ignore [attr-defined]  # Also add to main module
    sys.modules["binaryninja.interaction"] = interaction_mod

    sys.modules["binaryninja"] = bn
