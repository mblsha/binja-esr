from binja_helpers import binja_api  # noqa: F401
from binja_helpers.eval_llil import evaluate_llil, Memory, State
from binja_helpers.mock_llil import mllil, mreg, MockFlag

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


def test_add_sets_flags() -> None:
    regs = SimpleRegs()
    mem, read_mem, write_mem = make_mem()
    state = State()
    expr = mllil(
        "ADD.b{CZ}",
        [mllil("CONST.b", [0x01]), mllil("CONST.b", [0xFF])],
    )
    result, flags = evaluate_llil(expr, regs, Memory(read_mem, write_mem), state)
    assert result == 0x00
    assert flags == {"C": 1, "Z": 1}
    assert regs.get_by_name("F") == 0b11


def test_set_and_get_reg() -> None:
    regs = SimpleRegs()
    mem, read_mem, write_mem = make_mem()
    state = State()
    expr_set = mllil(
        "SET_REG.b",
        [mreg("A"), mllil("CONST.b", [0x42])],
    )
    evaluate_llil(expr_set, regs, Memory(read_mem, write_mem), state)
    expr_get = mllil("REG", [mreg("A")])
    val, _ = evaluate_llil(expr_get, regs, Memory(read_mem, write_mem), state)
    assert val == 0x42


def test_push_and_pop() -> None:
    regs = SimpleRegs()
    regs.set_by_name("S", 0x10)
    mem, read_mem, write_mem = make_mem()
    state = State()
    expr_push = mllil("PUSH.b", [mllil("CONST.b", [0xAA])])
    evaluate_llil(expr_push, regs, Memory(read_mem, write_mem), state)
    assert regs.get_by_name("S") == 0x0F
    assert mem[0x0F] == 0xAA
    expr_pop = mllil("POP.b")
    val, _ = evaluate_llil(expr_pop, regs, Memory(read_mem, write_mem), state)
    assert val == 0xAA
    assert regs.get_by_name("S") == 0x10


def test_load_and_store() -> None:
    regs = SimpleRegs()
    mem, read_mem, write_mem = make_mem()
    mem[0x20] = 0x77
    state = State()
    expr_load = mllil("LOAD.b", [mllil("CONST_PTR", [0x20])])
    val, _ = evaluate_llil(expr_load, regs, Memory(read_mem, write_mem), state)
    assert val == 0x77
    expr_store = mllil(
        "STORE.b",
        [mllil("CONST_PTR", [0x21]), mllil("CONST.b", [0x99])],
    )
    evaluate_llil(expr_store, regs, Memory(read_mem, write_mem), state)
    assert mem[0x21] == 0x99


def test_set_and_read_flag() -> None:
    regs = SimpleRegs()
    mem, read_mem, write_mem = make_mem()
    state = State()
    expr_set = mllil("SET_FLAG", [MockFlag("C"), mllil("CONST.b", [1])])
    evaluate_llil(expr_set, regs, Memory(read_mem, write_mem), state)
    assert regs.get_by_name("F") & 1 == 1
    expr_get = mllil("FLAG", [MockFlag("C")])
    val, _ = evaluate_llil(expr_get, regs, Memory(read_mem, write_mem), state)
    assert val == 1


def test_lsl_flag_behavior() -> None:
    regs = SimpleRegs()
    mem, read_mem, write_mem = make_mem()
    state = State()
    expr = mllil(
        "LSL.b{CZ}",
        [mllil("CONST.b", [0x80]), mllil("CONST.b", [1])],
    )
    result, _ = evaluate_llil(expr, regs, Memory(read_mem, write_mem), state)
    assert result == 0x00
    assert regs.get_by_name("F") == 0b11
