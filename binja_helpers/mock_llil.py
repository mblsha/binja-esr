# Make LLIL unit-testable.

from . import binja_api  # noqa: F401
from binaryninja.lowlevelil import (
    LowLevelILFunction,
    LowLevelILLabel,
    ILSourceLocation,
    ExpressionIndex,
)
from dataclasses import dataclass
from typing import Any, List, Optional, Union


# Default size lookup table - can be overridden per architecture
DEFAULT_SZ_LOOKUP = {1: ".b", 2: ".w", 3: ".l", 4: ".error"}
DEFAULT_SUFFIX_SZ = {"b": 1, "w": 2, "l": 3}

# Current active size lookup - can be modified by set_size_lookup()
SZ_LOOKUP = DEFAULT_SZ_LOOKUP.copy()
SUFFIX_SZ = DEFAULT_SUFFIX_SZ.copy()


def set_size_lookup(size_lookup: dict[int, str], suffix_sz: dict[str, int] = None) -> None:
    """
    Set custom size lookup tables for architecture-specific width suffixes.
    
    Args:
        size_lookup: Map from byte size to suffix string (e.g., {4: ".4"})
        suffix_sz: Optional reverse mapping from suffix to size (e.g., {"4": 4})
    """
    global SZ_LOOKUP, SUFFIX_SZ
    SZ_LOOKUP = size_lookup.copy()
    if suffix_sz is not None:
        SUFFIX_SZ = suffix_sz.copy()


def reset_size_lookup() -> None:
    """Reset size lookup tables to defaults."""
    global SZ_LOOKUP, SUFFIX_SZ
    SZ_LOOKUP = DEFAULT_SZ_LOOKUP.copy()
    SUFFIX_SZ = DEFAULT_SUFFIX_SZ.copy()


@dataclass
class MockReg:
    name: str


@dataclass
class MockFlag:
    name: str


# we cannot use real Architecture (it requires a Binary Ninja license)
class MockArch:
    address_size = 4  # SCUMM6 uses 4-byte addresses
    
    def get_reg_index(self, name: object) -> Any:
        assert name != "IMR"

        if isinstance(name, int):
            result = MockReg(f"TEMP{name - 2147483648}")
        else:
            result = MockReg(str(name))
        return result

    def get_flag_by_name(self, name: str) -> Any:
        return MockFlag(name)


class MockHandle:
    pass


@dataclass
class MockLLIL:
    """Simplified stand-in for Binary Ninja's LLIL expression.

    WARNING: On a real Binary Ninja instance these expressions are
    represented by ``ExpressionIndex`` integers.  This class exists only
    for testing and should not be confused with the real API.
    """

    op: str
    ops: List[Any]

    def width(self) -> Optional[int]:
        op = self.op.split("{")[0]
        opsplit = op.split(".")
        op = opsplit[0]
        if len(opsplit) > 1:
            suffix = opsplit[1]
            size = SUFFIX_SZ.get(suffix, None)
        else:
            size = None
        return size

    def flags(self) -> Optional[str]:
        flagssplit = self.op.split("{")
        if len(flagssplit) > 1:
            flags = flagssplit[1].rstrip("}")
        else:
            flags = None
        return flags

    def bare_op(self) -> str:
        return self.op.split("{")[0].split(".")[0]


ExprType = Union[MockLLIL, ExpressionIndex]


def mreg(name: str) -> MockReg:
    return MockReg(name)


def mllil(op: str, ops: Optional[List[object]] = None) -> MockLLIL:
    if ops is None:
        ops = []
    return MockLLIL(op, ops)


@dataclass
class MockIfExpr(MockLLIL):
    cond: Any
    t: Any
    f: Any

    def __init__(self, cond: Any, t: Any, f: Any) -> None:
        super().__init__("IF", [])
        self.cond = cond
        self.t = t
        self.f = f


@dataclass
class MockLabel(MockLLIL):
    label: LowLevelILLabel

    def __init__(self, label: LowLevelILLabel) -> None:
        super().__init__("LABEL", [])
        self.label = label


@dataclass
class MockIntrinsic(MockLLIL):
    name: str
    outputs: Any
    params: Any

    def __init__(self, name: str, outputs: Any, params: Any) -> None:
        super().__init__("INTRINSIC", [])
        self.name = name
        self.outputs = outputs
        self.params = params


@dataclass
class MockGoto:
    label: Any


class MockSourceFunction:
    """Mock source function with arch attribute."""
    def __init__(self) -> None:
        self.arch = MockArch()


class MockLowLevelILFunction(LowLevelILFunction):  # type: ignore[misc]
    def __init__(self) -> None:
        # self.handle = MockHandle()
        self._arch = MockArch()
        self.ils: List[MockLLIL] = []
        self.source_function = MockSourceFunction()
    
    @property
    def arch(self) -> MockArch:
        """The architecture for this LLIL function."""
        return self._arch

    def __del__(self) -> None:
        pass

    def mark_label(self, label: LowLevelILLabel) -> Any:
        # remove source_location from kwargs
        result = MockLabel(label)
        self.append(result)
        return result

    def goto(self, label: LowLevelILLabel, loc: Optional[ILSourceLocation] = None) -> Any:
        return MockGoto(label)

    def if_expr(self, cond, t, f) -> Any:  # type: ignore
        return MockIfExpr(cond, t, f)

    def intrinsic(self, outputs, name: str, params) -> Any:  # type: ignore
        return MockIntrinsic(name, outputs, params)

    def append(self, il: Any) -> Any:
        self.ils.append(il)
        return len(self.ils) - 1

    def expr(self, *args, **kwargs) -> ExprType:  # type: ignore
        llil, *ops = args
        kwargs.pop("source_location", None)
        size = kwargs.get("size", None)
        flags = kwargs.get("flags", None)

        name = llil.name
        # remove the "LLIL_" prefix
        name = name[5:]
        if isinstance(size, int):
            suffix = SZ_LOOKUP.get(size, "")
        else:
            suffix = ""
        name = name + suffix
        name = name + f"{{{flags}}}" if flags is not None else name
        return MockLLIL(name, ops)

    def get_label_for_address(self, arch: Any, addr: int) -> LowLevelILLabel:
        """Mock implementation for getting a label for an address."""
        return LowLevelILLabel()
