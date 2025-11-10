from sc62015.scil import backend_pyemu, specs


class _DictBus:
    def __init__(self) -> None:
        self.storage = {}

    def load(self, space: str, addr: int, size: int) -> int:
        return self.storage.get((space, addr), 0)

    def store(self, space: str, addr: int, value: int, size: int) -> None:
        mask = (1 << size) - 1
        self.storage[(space, addr)] = value & mask


class _ImmediateStream:
    def __init__(self, *values: int) -> None:
        self._values = list(values)
        self._index = 0

    def read(self, kind: str) -> int:
        if kind == "addr24":
            lo = self.read("u8")
            mid = self.read("u8")
            hi = self.read("u8")
            return lo | (mid << 8) | (hi << 16)
        value = self._values[self._index]
        self._index += 1
        return value


def test_mv_a_abs_ext_reads_memory() -> None:
    instr = specs.mv_a_abs_ext()
    state = backend_pyemu.CPUState(pc=0x80000)
    bus = _DictBus()
    bus.storage[("ext", 0x001234)] = 0xA5
    stream = _ImmediateStream(0x34, 0x12, 0x00)

    backend_pyemu.step(state, bus, instr, stream)

    assert state.regs["A"] == 0xA5


def test_jrz_updates_program_counter_when_zero_flag_set() -> None:
    instr = specs.jrz_rel()
    state = backend_pyemu.CPUState(pc=0x1000, flags={"Z": 1})
    bus = _DictBus()
    stream = _ImmediateStream(0x04)  # Jump forward by 4

    backend_pyemu.step(state, bus, instr, stream)

    assert state.pc == 0x1006
