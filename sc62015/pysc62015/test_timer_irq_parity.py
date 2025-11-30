"""Parity tests for timer/IRQ events across Python and LLAMA backends."""

from __future__ import annotations

import pytest

from sc62015.pysc62015 import CPU, RegisterName, available_backends
from sc62015.pysc62015.constants import ADDRESS_SPACE_SIZE, INTERNAL_MEMORY_START


@pytest.mark.skipif("llama" not in available_backends(), reason="LLAMA backend not available")
@pytest.mark.parametrize("backend", ("python", "llama"))
def test_timer_irq_sets_isr_and_traces(backend: str) -> None:
    # Minimal memory; scheduler lives in Python emulator, but LLAMA backend should still mirror IRQ writes.
    raw = bytearray(ADDRESS_SPACE_SIZE)
    # Place a NOP so execute_instruction doesn't crash.
    raw[0] = 0x00

    def read(addr: int) -> int:
        return raw[addr & 0xFFFFFF]

    def write(addr: int, value: int) -> None:
        raw[addr & 0xFFFFFF] = value & 0xFF

    class Mem:
        def __init__(self) -> None:
            self._raw = raw
            self.irq_traces: list[tuple[str, dict[str, int]]] = []

        def read_byte(self, addr: int) -> int:
            return read(addr)

        def write_byte(self, addr: int, value: int) -> None:
            write(addr, value)

        def trace_irq_from_rust(self, name: str, payload: dict[str, int]) -> None:
            self.irq_traces.append((name, payload))

        def read_bytes(self, address: int, size: int) -> int:
            result = 0
            for i in range(size):
                result |= (self.read_byte(address + i) & 0xFF) << (8 * i)
            return result

        def write_bytes(self, size: int, address: int, value: int) -> None:
            for i in range(size):
                self.write_byte(address + i, (value >> (8 * i)) & 0xFF)

    mem = Mem()
    cpu = CPU(mem, reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.PC, 0)

    # Program: MV (ISR), #0x01 to simulate timer-set ISR via bus store
    raw[0] = 0xCC  # MV IMem8, imm8
    raw[1] = 0xFC  # ISR offset
    raw[2] = 0x01
    cpu.execute_instruction(0)

    # ISR bit set and trace emitted (for LLAMA path via trace_irq_from_rust).
    isr_addr = INTERNAL_MEMORY_START + 0xFC
    assert mem.read_byte(isr_addr) & 0x01 == 0x01
    if backend == "llama":
        assert mem.irq_traces != []
