from binja_helpers import binja_api  # noqa: F401
from binja_helpers.eval_llil import evaluate_llil, Memory, State
from binja_helpers.mock_llil import mllil, mreg, MockIntrinsic


class DummyRegisters:
    def __init__(self) -> None:
        self.values: dict[str, int] = {}

    def get_by_name(self, name: str) -> int:
        return self.values.get(name, 0)

    def set_by_name(self, name: str, value: int) -> None:
        self.values[name] = value & 0xFFFFFFFF

    def get_flag(self, name: str) -> int:
        return self.get_by_name(f"F{name}")

    def set_flag(self, name: str, value: int) -> None:
        self.set_by_name(f"F{name}", value & 1)


def make_memory() -> tuple[Memory, dict[int, int]]:
    mem: dict[int, int] = {}

    def read(addr: int) -> int:
        return mem.get(addr, 0)

    def write(addr: int, value: int) -> None:
        mem[addr] = value & 0xFF

    return Memory(read, write), mem


def test_eval_const() -> None:
    regs = DummyRegisters()
    memory, _ = make_memory()
    state = State()

    result, flags = evaluate_llil(mllil("CONST.w", [0x1234]), regs, memory, state)
    assert result == 0x1234
    assert flags is None


def test_set_and_get_reg() -> None:
    regs = DummyRegisters()
    memory, _ = make_memory()
    state = State()

    evaluate_llil(
        mllil("SET_REG.b", [mreg("A"), mllil("CONST.b", [0x56])]),
        regs,
        memory,
        state,
    )
    result, _ = evaluate_llil(mllil("REG.b", [mreg("A")]), regs, memory, state)
    assert result == 0x56


def test_add_sets_flags() -> None:
    regs = DummyRegisters()
    memory, _ = make_memory()
    state = State()

    result, _ = evaluate_llil(
        mllil("ADD.b{CZ}", [mllil("CONST.b", [0xFF]), mllil("CONST.b", [1])]),
        regs,
        memory,
        state,
    )
    assert result == 0x00
    assert regs.get_by_name("FC") == 1
    assert regs.get_by_name("FZ") == 1


def test_sub_sets_carry() -> None:
    regs = DummyRegisters()
    memory, _ = make_memory()
    state = State()

    result, _ = evaluate_llil(
        mllil("SUB.b{CZ}", [mllil("CONST.b", [0]), mllil("CONST.b", [1])]),
        regs,
        memory,
        state,
    )
    assert result == 0xFF
    assert regs.get_by_name("FC") == 1
    assert regs.get_by_name("FZ") == 0


def test_shift_lsl() -> None:
    regs = DummyRegisters()
    memory, _ = make_memory()
    state = State()

    result, _ = evaluate_llil(
        mllil("LSL.b{CZ}", [mllil("CONST.b", [0x80]), mllil("CONST.b", [1])]),
        regs,
        memory,
        state,
    )
    assert result == 0x00
    assert regs.get_by_name("FC") == 1
    assert regs.get_by_name("FZ") == 1


def test_memory_load_store() -> None:
    regs = DummyRegisters()
    memory, mem = make_memory()
    state = State()

    evaluate_llil(
        mllil(
            "STORE.b",
            [mllil("CONST_PTR.b", [0x10]), mllil("CONST.b", [0xAB])],
        ),
        regs,
        memory,
        state,
    )
    assert mem[0x10] == 0xAB

    result, _ = evaluate_llil(
        mllil("LOAD.b", [mllil("CONST_PTR.b", [0x10])]),
        regs,
        memory,
        state,
    )
    assert result == 0xAB


def test_intrinsic_halt() -> None:
    regs = DummyRegisters()
    memory, _ = make_memory()
    state = State()

    evaluate_llil(MockIntrinsic("HALT", [], []), regs, memory, state)
    assert state.halted
