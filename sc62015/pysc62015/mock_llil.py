# Make LLIL unit-testable.

from .binja_api import *
from binaryninja.lowlevelil import (
    LowLevelILFunction,
    LowLevelILLabel,
    ILSourceLocation,
)
from binaryninja import Architecture
from binaryninja.enums import LowLevelILOperation
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Tuple


SZ_LOOKUP = {1:'.b', 2:'.w', 3:'.l'}


@dataclass
class MockReg:
    name: str


@dataclass
class MockFlag:
    name: str


class MockArch:
    def get_reg_index(self, name: object) -> Any:
        if name == 2147483648:
            return MockReg('TEMP0')
        return MockReg(str(name))

    def get_flag_by_name(self, name: str) -> Any:
        return MockFlag(name)


class MockHandle:
    pass


@dataclass
class MockLLIL:
    op: str
    ops: List[Any]


def mreg(name: str) -> MockReg:
    return MockReg(name)


def mllil(op: str, ops:List[object]=[]) -> MockLLIL:
    return MockLLIL(op, ops)


@dataclass
class MockIfExpr:
    cond: Any
    t: Any
    f: Any


@dataclass
class MockLabel:
    label: LowLevelILLabel

@dataclass
class MockGoto:
    label: Any


class MockLowLevelILFunction(LowLevelILFunction):
    def __init__(self) -> None:
        # self.handle = MockHandle()
        self._arch = MockArch()
        self.ils: List[MockLLIL] = []

    def __del__(self) -> None:
        pass

    def mark_label(self, label: LowLevelILLabel) -> Any:
        # remove source_location from kwargs
        return MockLabel(label)

    def goto(self, label: LowLevelILLabel, loc: Optional[ILSourceLocation] =
             None) -> Any: # type: ignore
        return MockGoto(label)

    def if_expr(self, cond, t, f) -> Any:  # type: ignore
        return MockIfExpr(cond, t, f)

    def append(self, il: Any) -> int:
        self.ils.append(il)
        return len(self.ils) - 1

    def expr(self, *args, **kwargs) -> Any:  # type: ignore
        llil, *ops = args
        del kwargs["source_location"]
        size = kwargs.get("size", None)
        flags = kwargs.get("flags", None)

        name = llil.name
        # remove the "LLIL_" prefix
        name = name[5:]
        name = name + SZ_LOOKUP.get(size, "")
        name = name + f"{{{flags}}}" if flags is not None else name
        return MockLLIL(name, ops)
