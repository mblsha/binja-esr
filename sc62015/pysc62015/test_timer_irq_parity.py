"""Parity tests for timer/IRQ events across Python and LLAMA backends."""

from __future__ import annotations

import pytest
from typing import Any, cast

from sc62015.pysc62015 import CPU, RegisterName
from sc62015.pysc62015.constants import (
    ADDRESS_SPACE_SIZE,
    INTERNAL_MEMORY_START,
    ISRFlag,
)
from sc62015.pysc62015.instr.opcodes import IMEMRegisters
from pce500.emulator import PCE500Emulator


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


def test_timer_irq_cadence_matches_between_backends(monkeypatch) -> None:
    backends = ("python", "llama")
    results = {}

    import _sc62015_rustcore as rustcore
    rustcore = cast(Any, rustcore)

    for backend in backends:
        captured: list[tuple[str, dict[str, int]]] = []

        def fake_irq(name: str, payload: dict[str, int]) -> None:
            captured.append((name, dict(payload)))

        monkeypatch.setenv("SC62015_CPU_BACKEND", backend)
        orig_irq = rustcore.record_irq_event
        rustcore.record_irq_event = fake_irq  # type: ignore[assignment]
        try:
            emu = PCE500Emulator(
                trace_enabled=False,
                perfetto_trace=False,
                enable_new_tracing=False,
                keyboard_columns_active_high=True,
            )
            # Enable MTI/STI/KEYI in IMR
            imr_addr = INTERNAL_MEMORY_START + IMEMRegisters.IMR
            emu.memory.write_byte(imr_addr, 0x07)
            # Short timer periods for deterministic cadence
            emu._scheduler.mti_period = 1  # type: ignore[attr-defined]
            emu._scheduler.sti_period = 1  # type: ignore[attr-defined]
            # Run a few ticks to generate MTI/STI, then assert KEYI via ISR bit.
            for _ in range(3):
                emu._tick_timers()
            emu._set_isr_bits(int(ISRFlag.KEYI))
            # Snapshot ISR/IMR after events
            isr_addr = INTERNAL_MEMORY_START + IMEMRegisters.ISR
            results[backend] = {
                "irq_events": captured.copy(),
                "imr": emu.memory.read_byte(imr_addr - 1) & 0xFF,
                "isr": emu.memory.read_byte(isr_addr) & 0xFF,
            }
        finally:
            rustcore.record_irq_event = orig_irq  # type: ignore[assignment]
            monkeypatch.delenv("SC62015_CPU_BACKEND", raising=False)

    assert results["python"]["imr"] == results["llama"]["imr"]
    assert results["python"]["isr"] == results["llama"]["isr"]
    # Event sequences should match by name order (payloads may differ in pc but should exist).
    python_names = [name for name, _ in results["python"]["irq_events"]]
    llama_names = [name for name, _ in results["llama"]["irq_events"]]
    assert python_names == llama_names
