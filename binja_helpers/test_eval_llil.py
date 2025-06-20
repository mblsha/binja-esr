from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

from binja_helpers import binja_api  # noqa: F401
from binja_helpers.eval_llil import evaluate_llil, Memory, State
from binja_helpers.mock_llil import MockLLIL, MockFlag, mllil, mreg
import pytest

class SimpleRegs:
    def __init__(self) -> None:
        self.values: dict[str, int] = {}

    def get_by_name(self, name: str) -> int:
        return self.values.get(name, 0) or 0

    def set_by_name(self, name: str, value: int) -> None:
        self.values[name] = value & 0xFFFFFFFF

    def get_flag(self, name: str) -> int:
        return self.values.get(name, 0) or 0

    def set_flag(self, name: str, value: int) -> None:
        self.values[name] = value & 1



def make_mem(size: int = 0x100) -> Tuple[bytearray, Callable[[int], int], Callable[[int, int], None]]:
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
    # Logical operations
    *(
        LlilEvalTestCase(
            test_id=f"and_{sz}_sets_flags",
            llil_expr=mllil(
                f"AND.{sz}{{CZ}}",
                [mllil(f"CONST.{sz}", [0]), mllil(f"CONST.{sz}", [0])],
            ),
            expected_result=0,
            expected_flags_in_f=0b10,
        )
        for sz in ["b", "w", "l"]
    ),
    *(
        LlilEvalTestCase(
            test_id=f"or_{sz}_sets_carry_only",
            llil_expr=mllil(
                f"OR.{sz}{'{C}'}",
                [mllil(f"CONST.{sz}", [1]), mllil(f"CONST.{sz}", [2])],
            ),
            expected_result=3,
            expected_flags_in_f=0,
        )
        for sz in ["b", "w", "l"]
    ),
    *(
        LlilEvalTestCase(
            test_id=f"xor_{sz}_sets_zero_only",
            llil_expr=mllil(
                f"XOR.{sz}{'{Z}'}",
                [mllil(f"CONST.{sz}", [0xAA]), mllil(f"CONST.{sz}", [0xAA])],
            ),
            expected_result=0,
            expected_flags_in_f=0b10,
        )
        for sz in ["b", "w", "l"]
    ),
    *(
        LlilEvalTestCase(
            test_id=f"sub_{sz}_borrow_sets_carry",
            llil_expr=mllil(
                f"SUB.{sz}{{CZ}}",
                [mllil(f"CONST.{sz}", [0]), mllil(f"CONST.{sz}", [1])],
            ),
            expected_result=(1 << ({"b":8,"w":16,"l":24}[sz])) - 1,
            expected_flags_in_f=0b01,
        )
        for sz in ["b", "w", "l"]
    ),
    # Shift and rotate operations
    *(
        LlilEvalTestCase(
            test_id=f"lsr_{sz}_carry_and_zero",
            llil_expr=mllil(
                f"LSR.{sz}{{CZ}}",
                [mllil(f"CONST.{sz}", [1]), mllil(f"CONST.{sz}", [1])],
            ),
            expected_result=0,
            expected_flags_in_f=0b11,
        )
        for sz in ["b", "w", "l"]
    ),
    *(
        LlilEvalTestCase(
            test_id=f"ror_{sz}_updates_flags",
            llil_expr=mllil(
                f"ROR.{sz}{{CZ}}",
                [mllil(f"CONST.{sz}", [1]), mllil(f"CONST.{sz}", [1])],
            ),
            expected_result=(1 << ({"b":8,"w":16,"l":24}[sz])) >> 1,
            expected_flags_in_f=0b01,
        )
        for sz in ["b", "w", "l"]
    ),
    *(
        LlilEvalTestCase(
            test_id=f"rol_{sz}_updates_flags",
            llil_expr=mllil(
                f"ROL.{sz}{{CZ}}",
                [mllil(f"CONST.{sz}", [1 << ({"b":8,"w":16,"l":24}[sz] - 1)]), mllil(f"CONST.{sz}", [1])],
            ),
            expected_result=1,
            expected_flags_in_f=0b01,
        )
        for sz in ["b", "w", "l"]
    ),
    *(
        LlilEvalTestCase(
            test_id=f"rrc_{sz}_carry_out",
            llil_expr=mllil(
                f"RRC.{sz}{{CZ}}",
                [
                    mllil(f"CONST.{sz}", [2]),
                    mllil(f"CONST.{sz}", [1]),
                    mllil(f"CONST.{sz}", [1]),
                ],
            ),
            expected_result=(1 << ({"b":8,"w":16,"l":24}[sz] - 1)) | 1,
            expected_flags_in_f=0,
        )
        for sz in ["b", "w", "l"]
    ),
    *(
        LlilEvalTestCase(
            test_id=f"rlc_{sz}_carry_out",
            llil_expr=mllil(
                f"RLC.{sz}{{CZ}}",
                [
                    mllil(f"CONST.{sz}", [1]),
                    mllil(f"CONST.{sz}", [1]),
                    mllil(f"CONST.{sz}", [1]),
                ],
            ),
            expected_result=3,
            expected_flags_in_f=0,
        )
        for sz in ["b", "w", "l"]
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


def test_nop_does_nothing() -> None:
    llil = mllil("NOP")
    regs = SimpleRegs()
    regs.set_by_name("F", 0x3)
    buf, read_mem, write_mem = make_mem()
    memory = Memory(read_mem, write_mem)
    state = State()
    result, _ = evaluate_llil(llil, regs, memory, state)
    assert result is None
    assert regs.get_by_name("F") == 0x3


def test_unimpl_raises() -> None:
    with pytest.raises(NotImplementedError):
        llil = mllil("UNIMPL")
        regs = SimpleRegs()
        buf, read_mem, write_mem = make_mem()
        memory = Memory(read_mem, write_mem)
        state = State()
        evaluate_llil(llil, regs, memory, state)


def test_jump_sets_pc() -> None:
    llil = mllil("JUMP", [mllil("CONST_PTR", [0x123])])
    regs = SimpleRegs()
    regs.set_by_name("PC", 0)
    buf, read_mem, write_mem = make_mem()
    memory = Memory(read_mem, write_mem)
    state = State()
    evaluate_llil(llil, regs, memory, state)
    assert regs.get_by_name("PC") == 0x123


def test_call_pushes_return_address_and_updates_pc() -> None:
    llil = mllil("CALL", [mllil("CONST_PTR.w", [0x200])])
    regs = SimpleRegs()
    regs.set_by_name("PC", 0x1111)
    regs.set_by_name("S", 0x10)
    buf, read_mem, write_mem = make_mem()
    memory = Memory(read_mem, write_mem)
    state = State()
    evaluate_llil(llil, regs, memory, state)
    assert regs.get_by_name("PC") == 0x200
    assert regs.get_by_name("S") == 0x0E
    assert buf[0x0E] == 0x11 and buf[0x0F] == 0x11


def test_ret_sets_pc_from_operand() -> None:
    llil = mllil("RET", [mllil("CONST_PTR", [0x333])])
    regs = SimpleRegs()
    buf, read_mem, write_mem = make_mem()
    memory = Memory(read_mem, write_mem)
    state = State()
    evaluate_llil(llil, regs, memory, state)
    assert regs.get_by_name("PC") == 0x333


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (1, 1, 1),
        (1, 2, 0),
    ],
)
def test_cmp_e(a: int, b: int, expected: int) -> None:
    llil = mllil(
        "CMP_E.b",
        [mllil("CONST.b", [a]), mllil("CONST.b", [b])],
    )
    regs = SimpleRegs()
    buf, read_mem, write_mem = make_mem()
    memory = Memory(read_mem, write_mem)
    state = State()
    result, _ = evaluate_llil(llil, regs, memory, state)
    assert result == expected


def test_cmp_ugt() -> None:
    llil = mllil(
        "CMP_UGT.w",
        [mllil("CONST.w", [3]), mllil("CONST.w", [2])],
    )
    regs = SimpleRegs()
    buf, read_mem, write_mem = make_mem()
    memory = Memory(read_mem, write_mem)
    state = State()
    result, _ = evaluate_llil(llil, regs, memory, state)
    assert result == 1


def test_cmp_slt_signed() -> None:
    llil = mllil(
        "CMP_SLT.b",
        [mllil("CONST.b", [0xFF]), mllil("CONST.b", [0x01])],
    )
    regs = SimpleRegs()
    buf, read_mem, write_mem = make_mem()
    memory = Memory(read_mem, write_mem)
    state = State()
    result, _ = evaluate_llil(llil, regs, memory, state)
    assert result == 1





def test_operation_without_flag_spec_leaves_flags() -> None:
    llil = mllil(
        "AND.w",
        [mllil("CONST.w", [1]), mllil("CONST.w", [1])],
    )
    regs = SimpleRegs()
    regs.set_by_name("F", 0x3)
    buf, read_mem, write_mem = make_mem()
    memory = Memory(read_mem, write_mem)
    state = State()
    evaluate_llil(llil, regs, memory, state)
    assert regs.get_by_name("F") == 0x3


def test_custom_flag_handlers() -> None:
    regs = SimpleRegs()
    buf, read_mem, write_mem = make_mem()
    memory = Memory(read_mem, write_mem)
    state = State()
    flag_store: Dict[str, int] = {}

    def get_flag(name: str) -> int:
        return flag_store.get(name, 0)

    def set_flag(name: str, value: int) -> None:
        flag_store[name] = value

    expr = mllil(
        "ADD.b{CZ}",
        [mllil("CONST.b", [1]), mllil("CONST.b", [0xFF])],
    )
    evaluate_llil(expr, regs, memory, state, get_flag=get_flag, set_flag=set_flag)

    assert flag_store.get("C") == 1
    assert flag_store.get("Z") == 1
    assert regs.get_by_name("F") == 0
