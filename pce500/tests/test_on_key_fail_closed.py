"""Fail-closed ON-key contracts shared by the PCE Python and LLAMA paths."""

from __future__ import annotations

from typing import Any

import pytest

from pce500.emulator import IRQSource, PCE500Emulator
from pce500.keyboard_handler import PCE500KeyboardHandler
from sc62015.pysc62015.constants import INTERNAL_MEMORY_START
from sc62015.pysc62015.cpu import available_backends, select_backend
from sc62015.pysc62015.instr.opcodes import IMEMRegisters


ONK_MASK = 0x08


@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize("remaining_isr", [0x00, 0x01])
def test_pce_on_key_press_and_release_keep_status_and_irq_state_in_sync(
    backend: str,
    remaining_isr: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if backend == "llama" and "llama" not in available_backends():
        pytest.skip("LLAMA backend not available")
    monkeypatch.setenv("SC62015_CPU_BACKEND", backend)
    emulator = PCE500Emulator()

    ssr_addr = INTERNAL_MEMORY_START + IMEMRegisters.SSR
    isr_addr = INTERNAL_MEMORY_START + IMEMRegisters.ISR
    emulator.memory.write_byte(ssr_addr, 0x04)
    emulator.memory.write_byte(isr_addr, remaining_isr)
    emulator._irq_pending = False
    emulator._irq_source = None

    assert emulator.press_key("KEY_ON")
    assert emulator.memory.read_byte(ssr_addr) == 0x04 | ONK_MASK
    assert emulator.memory.read_byte(isr_addr) == remaining_isr | ONK_MASK
    assert emulator._irq_pending
    assert emulator._irq_source == IRQSource.ONK
    assert emulator.keyboard._on_key_pressed

    emulator.release_key("KEY_ON")
    assert emulator.memory.read_byte(ssr_addr) == 0x04
    assert emulator.memory.read_byte(isr_addr) == remaining_isr | ONK_MASK
    assert emulator._irq_pending
    assert emulator._irq_source is IRQSource.ONK
    assert not emulator.keyboard._on_key_pressed


class _FailingHostMemory:
    def __init__(self) -> None:
        self.data: dict[int, int] = {}
        self.write_calls = 0

    def read_byte(self, address: int, cpu_pc: int | None = None) -> int:
        del cpu_pc
        return self.data.get(address & 0xFFFFFF, 0)

    def write_byte(self, address: int, value: int, cpu_pc: int | None = None) -> None:
        del address, value, cpu_pc
        self.write_calls += 1
        raise RuntimeError("ON bridge write failed")


@pytest.mark.parametrize("release", [False, True])
def test_llama_on_bridge_callback_failure_propagates_once_and_poison_gates_retry(
    release: bool,
) -> None:
    if "llama" not in available_backends():
        pytest.skip("LLAMA backend not available")
    _, rust_module = select_backend("llama")
    assert rust_module is not None
    memory = _FailingHostMemory()
    cpu_type: Any = rust_module.LlamaCPU
    cpu = cpu_type(memory=memory, reset_on_init=False)
    handler = PCE500KeyboardHandler()
    # Matrix forwarding is intentionally disabled; ON still uses the native
    # transaction whenever a bridge CPU is attached.
    handler.set_bridge_cpu(cpu, enabled=False)

    operation = (
        (lambda: handler.release_key("KEY_ON"))
        if release
        else (lambda: handler.press_key("KEY_ON"))
    )
    with pytest.raises(RuntimeError, match="ON bridge write failed"):
        operation()
    assert memory.write_calls == 1
    assert not handler._on_key_pressed

    with pytest.raises(RuntimeError, match="poisoned.*ON bridge write failed"):
        operation()
    assert memory.write_calls == 1
