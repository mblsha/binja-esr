from __future__ import annotations

import json
from typing import Iterable

import pytest

from sc62015.decoding.dispatcher import CompatDispatcher
from sc62015.pysc62015._rust_bridge import BridgeCPU, MemoryAdapter  # noqa: E402
from sc62015.pysc62015.constants import ADDRESS_SPACE_SIZE, PC_MASK  # noqa: E402
from sc62015.scil import from_decoded, serde  # noqa: E402
from sc62015.scil.pyemu import CPUState  # noqa: E402
from sc62015.scil.pyemu.eval import execute_build  # noqa: E402

rustcore = pytest.importorskip(
    "_sc62015_rustcore", reason="Rust SCIL backend not built in this environment"
)


class _ArrayMemory:
    """Minimal memory stub compatible with the CPU facade."""

    def __init__(self, program: Iterable[int]) -> None:
        self._data = bytearray(ADDRESS_SPACE_SIZE)
        for idx, value in enumerate(program):
            self._data[idx & 0xFFFFFF] = value & 0xFF
        self.cpu = None
        self.set_cpu_calls: list[object | None] = []

    def read_byte(self, address: int) -> int:
        return self._data[address & 0xFFFFFF]

    def write_byte(self, address: int, value: int) -> None:
        self._data[address & 0xFFFFFF] = value & 0xFF

    def set_cpu(self, cpu) -> None:
        self.cpu = cpu
        self.set_cpu_calls.append(cpu)

    def export_flat_memory(self):
        return bytes(self._data[: 0x100000]), tuple()

    def apply_external_writes(self, writes):
        for address, value in writes:
            self.write_byte(address, value)


@pytest.mark.parametrize(
    "program",
    [
        bytes([0x00]),  # NOP
        bytes([0x08, 0x7F]),  # MV A,n
        bytes([0x18, 0x04]),  # JRZ +4
        bytes([0x40, 0x01]),  # ADD A,n
    ],
)
def test_rust_execution_matches_pyemu(program: bytes) -> None:
    """Ensure PyEMU and the Rust SCIL interpreter stay in lock-step."""

    dispatcher = CompatDispatcher()
    result = dispatcher.try_decode(program, 0x1000)
    assert result is not None
    length, decoded = result
    assert decoded is not None, "fixture emitted only PRE prefix"
    assert length == len(program)

    build = from_decoded.build(decoded)

    def _init_state() -> CPUState:
        template = CPUState()
        template.pc = 0xE000
        template.set_reg("BA", 0x55AA, 16)
        template.set_reg("X", 0x120304, 24)
        template.set_flag("C", 1)
        return template

    state_py = _clone_state(_init_state())
    state_rs = _clone_state(_init_state())

    py_bus = MemoryAdapter(_ArrayMemory(program))
    rs_bus = MemoryAdapter(_ArrayMemory(program))

    execute_build(state_py, py_bus, build, advance_pc=True)

    start_pc = state_rs.pc & PC_MASK

    state_json = json.dumps(state_rs.to_dict())
    instr_json = json.dumps(serde.instr_to_dict(build.instr))
    binder_json = json.dumps(serde.binder_to_dict(build.binder))
    pre_json = (
        json.dumps(serde.prelatch_to_dict(build.pre_applied))
        if build.pre_applied
        else None
    )

    new_state_json = rustcore.scil_step_json(
        state_json, instr_json, binder_json, rs_bus, pre_json
    )
    state_rs.load_dict(json.loads(new_state_json))
    if (state_rs.pc & PC_MASK) == start_pc:
        state_rs.pc = (start_pc + length) & PC_MASK

    assert state_rs.to_dict() == state_py.to_dict()


def _clone_state(state: CPUState) -> CPUState:
    clone = CPUState()
    clone.load_dict(state.to_dict())
    return clone


def test_rust_bridge_fallback_steps_once() -> None:
    """Decode failures should route through the python facade safely."""

    memory = _ArrayMemory(bytes([0x00]))  # Legacy decoder: NOP
    bridge = BridgeCPU(memory, reset_on_init=False)

    class _FailRuntime:
        def __init__(self, inner):
            self._inner = inner

        def execute_instruction(self):
            raise RuntimeError("boom")

        def __getattr__(self, name):
            return getattr(self._inner, name)

    bridge._runtime = _FailRuntime(bridge._runtime)

    stats_before = bridge.get_stats().copy()
    opcode, length = bridge.execute_instruction(0)

    stats_after = bridge.get_stats()
    decode_miss_before = stats_before["decode_miss"]
    fallback_before = stats_before["fallback_steps"]
    steps_before = stats_before["steps_rust"]
    decode_miss_after = stats_after["decode_miss"]
    fallback_after = stats_after["fallback_steps"]
    steps_after = stats_after["steps_rust"]
    assert isinstance(decode_miss_before, int)
    assert isinstance(fallback_before, int)
    assert isinstance(steps_before, int)
    assert isinstance(decode_miss_after, int)
    assert isinstance(fallback_after, int)
    assert isinstance(steps_after, int)
    assert decode_miss_after == decode_miss_before
    assert fallback_after == fallback_before + 1
    assert steps_after == steps_before
    assert stats_after["rust_errors"] == stats_before["rust_errors"] + 1

    assert opcode == 0x00
    assert length == 1
    assert bridge.read_register("PC") & PC_MASK == 1  # Legacy path advanced PC
    # Bridge should reattach itself after fallback
    assert memory.set_cpu_calls[-1] is bridge
