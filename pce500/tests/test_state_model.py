"""Tests for the canonical state snapshot and diff utilities."""

from __future__ import annotations

from sc62015.pysc62015.emulator import RegisterName

from pce500.emulator import PCE500Emulator
from pce500.state_model import (
    FieldDiff,
    capture_state,
    diff_states,
    empty_state_diff,
)


def test_capture_state_contains_core_components() -> None:
    emu = PCE500Emulator(perfetto_trace=False)
    state = capture_state(emu)

    assert "pc" in state.cpu.registers
    assert isinstance(state.memory.imr, int)
    assert isinstance(state.keyboard.pressed_keys, tuple)
    assert state.timers.mti_period > 0
    # Serial snapshot should be captured even when idle.
    assert hasattr(state.peripherals.serial, "ucr")


def test_diff_states_detects_cpu_register_change() -> None:
    emu = PCE500Emulator(perfetto_trace=False)
    before = capture_state(emu)

    emu.cpu.regs.set(RegisterName.A, 0x12)
    after = capture_state(emu)

    diff = diff_states(before, after)
    assert not diff.is_empty()
    assert FieldDiff("registers.a", before.cpu.registers["a"], 0x12) in diff.cpu


def test_empty_state_diff_reports_no_changes() -> None:
    diff = empty_state_diff()
    assert diff.is_empty()
