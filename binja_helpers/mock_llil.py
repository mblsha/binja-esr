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


SZ_LOOKUP = {1: ".b", 2: ".w", 3: ".l", 4: ".error"}
SUFFIX_SZ = {"b": 1, "w": 2, "l": 3}


@dataclass
class MockReg:
    name: str


@dataclass
class MockFlag:
    name: str


# we cannot use real Architecture (it requires a Binary Ninja license)
class MockArch:
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
    op: str
    ops: List[Any]

    def width(self) -> Optional[int]:
        op = self.op.split("{")[0]
        opsplit = op.split(".")
        op = opsplit[0]
        if len(opsplit) > 1:
            size = SUFFIX_SZ[opsplit[1]]
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


class MockLowLevelILFunction(LowLevelILFunction):
    def __init__(self) -> None:
        # self.handle = MockHandle()
        self._arch = MockArch()  # type: ignore
        self.ils: List[MockLLIL] = []

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
