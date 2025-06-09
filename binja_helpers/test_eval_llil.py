from dataclasses import dataclass, field
from typing import Dict, Optional

from binja_helpers import binja_api  # noqa: F401
from binja_helpers.eval_llil import evaluate_llil, Memory, State
from binja_helpers.mock_llil import MockLLIL, MockFlag, mllil, mreg
import pytest

class SimpleRegs:
    def __init__(self):
        self.values = {}

    def get_by_name(self, name: str) -> int:
        return self.values.get(name, 0)

    def set_by_name(self, name: str, value: int) -> None:
        self.values[name] = value & 0xFFFFFFFF



def make_mem(size=0x100):
    buf = bytearray(size)
    return (
        buf,
        lambda addr: buf[addr],
        lambda addr, val: buf.__setitem__(addr, val & 0xFF),
    )


@dataclass
class LlilEvalTestCase:
    """Container for a single LLIL evaluation test case."""

    test_id: str
    llil_expr: MockLLIL
    initial_regs: Dict[str, int] = field(default_factory=dict)
    initial_mem: Dict[int, int] = field(default_factory=dict)
    expected_result: Optional[int] = None
    expected_regs: Dict[str, int] = field(default_factory=dict)
    expected_mem_writes: Dict[int, int] = field(default_factory=dict)
    expected_flags_in_f: Optional[int] = None


eval_llil_test_cases = [
    LlilEvalTestCase(
        test_id="add_byte_sets_carry_and_zero_flags",
        llil_expr=mllil(
            "ADD.b{CZ}",
            [mllil("CONST.b", [0x01]), mllil("CONST.b", [0xFF])],
        ),
        expected_result=0x00,
        expected_flags_in_f=0b11,
    ),
    LlilEvalTestCase(
        test_id="set_reg_updates_register",
        llil_expr=mllil(
            "SET_REG.b",
            [mreg("A"), mllil("CONST.b", [0x42])],
        ),
        expected_regs={"A": 0x42},
    ),
    LlilEvalTestCase(
        test_id="reg_returns_value",
        llil_expr=mllil("REG", [mreg("A")]),
        initial_regs={"A": 0x42},
        expected_result=0x42,
    ),
    LlilEvalTestCase(
        test_id="push_byte_decrements_sp_and_writes_memory",
        llil_expr=mllil("PUSH.b", [mllil("CONST.b", [0xAA])]),
        initial_regs={"S": 0x10},
        expected_regs={"S": 0x0F},
        expected_mem_writes={0x0F: 0xAA},
    ),
    LlilEvalTestCase(
        test_id="pop_byte_restores_sp_and_returns_value",
        llil_expr=mllil("POP.b"),
        initial_regs={"S": 0x0F},
        initial_mem={0x0F: 0xAA},
        expected_result=0xAA,
        expected_regs={"S": 0x10},
    ),
    LlilEvalTestCase(
        test_id="load_byte_reads_memory",
        llil_expr=mllil("LOAD.b", [mllil("CONST_PTR", [0x20])]),
        initial_mem={0x20: 0x77},
        expected_result=0x77,
    ),
    LlilEvalTestCase(
        test_id="store_byte_writes_memory",
        llil_expr=mllil(
            "STORE.b",
            [mllil("CONST_PTR", [0x21]), mllil("CONST.b", [0x99])],
        ),
        expected_mem_writes={0x21: 0x99},
    ),
    LlilEvalTestCase(
        test_id="set_flag_sets_bit",
        llil_expr=mllil("SET_FLAG", [MockFlag("C"), mllil("CONST.b", [1])]),
        expected_flags_in_f=0b1,
    ),
    LlilEvalTestCase(
        test_id="flag_returns_value",
        llil_expr=mllil("FLAG", [MockFlag("C")]),
        initial_regs={"F": 0b1},
        expected_result=1,
    ),
    LlilEvalTestCase(
        test_id="lsl_sets_carry_and_zero_flags",
        llil_expr=mllil(
            "LSL.b{CZ}",
            [mllil("CONST.b", [0x80]), mllil("CONST.b", [1])],
        ),
        expected_result=0x00,
        expected_flags_in_f=0b11,
    ),
]


@pytest.mark.parametrize(
    "case",
    eval_llil_test_cases,
    ids=[c.test_id for c in eval_llil_test_cases],
)
def test_llil_evaluation(case: LlilEvalTestCase) -> None:
    regs = SimpleRegs()
    for name, value in case.initial_regs.items():
        regs.set_by_name(name, value)

    buf = bytearray(256)
    for addr, value in case.initial_mem.items():
        buf[addr] = value

    def read_mem(addr: int) -> int:
        return buf[addr]

    def write_mem(addr: int, val: int) -> None:
        buf[addr] = val & 0xFF
        if addr in case.expected_mem_writes:
            assert val == case.expected_mem_writes[addr]

    memory = Memory(read_mem, write_mem)
    state = State()

    result, _flags = evaluate_llil(case.llil_expr, regs, memory, state)

    if case.expected_result is not None:
        assert result == case.expected_result

    for name, value in case.expected_regs.items():
        assert regs.get_by_name(name) == value

    if case.expected_flags_in_f is not None:
        assert regs.get_by_name("F") == case.expected_flags_in_f

    for addr, value in case.expected_mem_writes.items():
        assert buf[addr] == value
