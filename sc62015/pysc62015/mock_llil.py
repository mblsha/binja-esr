from .binja_api import *
from binaryninja.lowlevelil import (
    LowLevelILFunction,
)

class MockArch:
    def get_reg_index(self, *args, **kwargs):
        return 0

    def get_flag_by_name(self, *args, **kwargs):
        return 0

class MockHandle:
    pass

class MockLowLevelILFunction(LowLevelILFunction):
    def __init__(self):
        # self.handle = MockHandle()
        self._arch = MockArch()
        pass

    def __del__(self):
        pass

    def mark_label(self, *args, **kwargs):
        pass

    def if_expr(self, cond, t, f):
        pass

    def append(self, *args, **kwargs):
        pass

    def expr(self, *args, **kwargs):
        pass
