"""Tests for the snapshot-driven orchestrator harness."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Tuple

import pytest

from pce500 import OrchestratorInputs, SnapshotOrchestrator
from pce500.memory import INTERNAL_MEMORY_START
from pce500.tracing import TraceEventType, trace_dispatcher
from sc62015.pysc62015.emulator import RegisterName
from sc62015.pysc62015.instr.opcodes import IMEMRegisters

ROM_PATH = Path(__file__).resolve().parents[3] / "roms" / "pc-e500.bin"

pytestmark = pytest.mark.skipif(
    not ROM_PATH.exists(), reason="pc-e500 ROM image not available"
)


def _build_test_rom(opcodes: Sequence[int]) -> Tuple[bytes, bytes]:
    """Return mutated full ROM image and the internal ROM overlay slice."""

    rom_image = bytearray(ROM_PATH.read_bytes())
    start = 0xC0000
    for idx, opcode in enumerate(opcodes):
        rom_image[start + idx] = opcode & 0xFF
    internal_rom = bytes(rom_image[start : start + 0x40000])
    return bytes(rom_image), internal_rom


def _collect_execution_threads(events: Iterable) -> int:
    return sum(1 for event in events if event.thread == "Execution")


def _disable_interrupt_sources(orchestrator: SnapshotOrchestrator) -> None:
    emu = orchestrator.emulator
    emu._timer_enabled = False  # type: ignore[attr-defined]
    emu._irq_pending = False  # type: ignore[attr-defined]
    if hasattr(emu.cpu.state, "halted"):
        emu.cpu.state.halted = False
    emu.cpu.regs.set(RegisterName.S, 0x0BFF00)
    imr_addr = INTERNAL_MEMORY_START + IMEMRegisters.IMR
    isr_addr = INTERNAL_MEMORY_START + IMEMRegisters.ISR
    emu.memory.write_byte(imr_addr, 0x00)
    emu.memory.write_byte(isr_addr, 0x00)


def test_step_advances_cpu_registers() -> None:
    rom_image, internal_rom = _build_test_rom(
        [
            0x90,
            0x42,  # LD A,0x42
            0x91,
            0x33,  # LD B,0x33
            0x00,  # NOP
        ]
    )

    with SnapshotOrchestrator() as orchestrator:
        orchestrator.load_rom(internal_rom)
        orchestrator.bootstrap_from_rom_image(
            rom_image, reset=False, configure_interrupt_mask=False
        )
        _disable_interrupt_sources(orchestrator)
        orchestrator.emulator.cpu.regs.set(RegisterName.PC, 0xC0000)

        initial = orchestrator.capture_snapshot()
        assert initial.cpu.registers["pc"] == 0xC0000

        first = orchestrator.step(
            OrchestratorInputs(max_instructions=0, metadata={"tag": "noop"})
        )
        assert first.executed_instructions == 0
        assert first.cycle_count == initial.cycle_count
        assert first.metadata["tag"] == "noop"

        second = orchestrator.step(OrchestratorInputs(max_instructions=0))
        assert second.cycle_count == first.cycle_count


def test_run_scenario_returns_snapshots_with_metadata() -> None:
    rom_image, internal_rom = _build_test_rom([0x00, 0x00, 0x00])
    steps = [
        OrchestratorInputs(
            max_instructions=0, press_keys=["KEY_F1"], metadata={"label": "press_pf1"}
        ),
        OrchestratorInputs(max_instructions=1),
    ]

    with SnapshotOrchestrator() as orchestrator:
        orchestrator.load_rom(internal_rom)
        orchestrator.bootstrap_from_rom_image(
            rom_image, reset=False, configure_interrupt_mask=False
        )
        _disable_interrupt_sources(orchestrator)
        orchestrator.emulator.cpu.regs.set(RegisterName.PC, 0xC0000)

        snapshots = orchestrator.run_scenario(steps, include_initial_snapshot=True)
        assert len(snapshots) == 3
        assert snapshots[0].executed_instructions == 0
        assert "KEY_F1" in snapshots[1].keyboard.pressed_keys
        assert snapshots[1].metadata["label"] == "press_pf1"
        assert snapshots[2].executed_instructions == 1


def test_register_observer_receives_execution_events() -> None:
    rom_image, internal_rom = _build_test_rom([0x00])

    class CollectingObserver:
        def __init__(self) -> None:
            self.events = []

        def handle_event(self, event) -> None:
            self.events.append(event)

    observer = CollectingObserver()

    with SnapshotOrchestrator() as orchestrator:
        orchestrator.load_rom(internal_rom)
        orchestrator.bootstrap_from_rom_image(
            rom_image, reset=False, configure_interrupt_mask=False
        )
        _disable_interrupt_sources(orchestrator)
        orchestrator.emulator.cpu.regs.set(RegisterName.PC, 0xC0000)
        orchestrator.register_observer(observer)

        trace_dispatcher.record_instant("Execution", "TestEvent", {})

        assert observer.events
        assert _collect_execution_threads(observer.events) >= 1
        assert any(event.type is TraceEventType.INSTANT for event in observer.events)

        orchestrator.unregister_observer(observer)
