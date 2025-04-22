from .binja_api import *
from binaryninja.lowlevelil import (
    LowLevelILFunction,
)
from binaryninja.enums import LowLevelILOperation
from dataclasses import dataclass
from typing import Any, List, Dict


@dataclass
class MockReg:
    name: str


@dataclass
class MockFlag:
    name: str


class MockArch:
    def get_reg_index(self, name):
        return MockReg(name)

    def get_flag_by_name(self, name):
        return MockFlag(name)


class MockHandle:
    pass


@dataclass
class MockLLIL:
    op: str
    ops: List[Any]
    meta: Dict[str, Any]


def mreg(name):
    return MockReg(name)


def mllil(op, ops=[], meta={}):
    if meta is None:
        meta = {}
    return MockLLIL(op, ops, meta)


@dataclass
class MockIfExpr:
    cond: Any
    t: Any
    f: Any


@dataclass
class MockLabel:
    args: List[Any]
    meta: Dict[str, Any]


class MockLowLevelILFunction(LowLevelILFunction):
    def __init__(self):
        # self.handle = MockHandle()
        self._arch = MockArch()
        self.ils = []

    def __del__(self):
        pass

    def mark_label(self, *args, **kwargs):
        # remove source_location from kwargs
        return MockLabel(args, kwargs)

    def if_expr(self, cond, t, f):
        return MockIfExpr(cond, t, f)

    def append(self, il):
        self.ils.append(il)

    def expr(self, *args, **kwargs):
        llil, *ops = args
        del kwargs["source_location"]
        name = llil.name
        # remove the "LLIL_" prefix
        name = name[5:]
        return MockLLIL(name, ops, kwargs)
