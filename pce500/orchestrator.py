"""Snapshot-driven orchestrator harness wrapping the PC-E500 emulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

from .emulator import PCE500Emulator
from .state_model import (
    EmulatorState,
    StateDiff,
    capture_state,
    diff_states,
    empty_state_diff,
)
from .tracing import TraceObserver, trace_dispatcher


@dataclass
class OrchestratorSnapshot:
    """Composite snapshot and diff captured after each orchestrator step."""

    state: EmulatorState
    diff: StateDiff
    executed_instructions: int
    cycle_count: int
    memory_access_log: Dict[str, Dict[str, list[tuple[int, int]]]] = field(
        default_factory=dict
    )
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class OrchestratorInputs:
    """Inputs applied before advancing the system."""

    max_instructions: Optional[int] = 1
    press_keys: Sequence[str] = ()
    release_keys: Sequence[str] = ()
    release_all_keys: bool = False
    metadata: Dict[str, object] = field(default_factory=dict)


class SnapshotOrchestrator:
    """High-level orchestrator that produces deterministic snapshots."""

    def __init__(
        self,
        *,
        emulator: Optional[PCE500Emulator] = None,
        trace_enabled: bool = False,
        perfetto_trace: bool = False,
        keyboard_columns_active_high: bool = True,
        enable_new_tracing: bool = False,
    ) -> None:
        self._emulator = emulator or PCE500Emulator(
            trace_enabled=trace_enabled,
            perfetto_trace=perfetto_trace,
            save_lcd_on_exit=False,
            keyboard_columns_active_high=keyboard_columns_active_high,
            enable_new_tracing=enable_new_tracing,
        )
        self._registered_observers: set[TraceObserver] = set()
        self._last_state: Optional[EmulatorState] = None

    # ------------------------------------------------------------------ #
    # Lifecycle helpers
    # ------------------------------------------------------------------ #
    @property
    def emulator(self) -> PCE500Emulator:
        return self._emulator

    def reset(self) -> None:
        """Reset all subsystems to their power-on state."""

        self._emulator.reset()
        self._last_state = None

    def load_rom(self, rom_data: bytes, *, start_address: Optional[int] = None) -> None:
        """Load a ROM blob into memory and reset to the entry vector."""

        self._emulator.load_rom(rom_data, start_address=start_address)
        self.reset()

    def bootstrap_from_rom_image(
        self,
        rom_image: bytes,
        *,
        reset: bool = True,
        restore_internal_ram: bool = True,
        configure_interrupt_mask: bool = True,
        imr_value: int = 0x43,
        isr_value: int = 0x00,
    ) -> None:
        """Delegate to :meth:`PCE500Emulator.bootstrap_from_rom_image`."""

        self._emulator.bootstrap_from_rom_image(
            rom_image,
            reset=reset,
            restore_internal_ram=restore_internal_ram,
            configure_interrupt_mask=configure_interrupt_mask,
            imr_value=imr_value,
            isr_value=isr_value,
        )
        self._last_state = None

    def close(self) -> None:
        """Release registered observers and stop tracing."""

        for observer in tuple(self._registered_observers):
            trace_dispatcher.unregister(observer)
        self._registered_observers.clear()
        self._last_state = None
        self._emulator.stop_tracing()

    def __enter__(self) -> "SnapshotOrchestrator":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    # ------------------------------------------------------------------ #
    # Observer management
    # ------------------------------------------------------------------ #
    def register_observer(self, observer: TraceObserver) -> None:
        """Subscribe a tracing observer to execution events."""

        trace_dispatcher.register(observer)
        self._registered_observers.add(observer)

    def unregister_observer(self, observer: TraceObserver) -> None:
        """Remove a previously registered observer."""

        if observer in self._registered_observers:
            trace_dispatcher.unregister(observer)
            self._registered_observers.discard(observer)

    # ------------------------------------------------------------------ #
    # Snapshot + step API
    # ------------------------------------------------------------------ #
    def capture_snapshot(
        self,
        *,
        executed_instructions: int = 0,
        metadata: Optional[Dict[str, object]] = None,
    ) -> OrchestratorSnapshot:
        """Capture a composite snapshot of emulator subsystems."""

        if metadata is None:
            metadata = {}
        return self._build_snapshot(executed_instructions, metadata)

    def apply_inputs(self, inputs: OrchestratorInputs) -> int:
        """Apply inputs and return number of instructions executed."""

        if inputs.release_all_keys:
            self._emulator.keyboard.release_all_keys()

        for key in inputs.release_keys:
            self._emulator.keyboard.release_key(key)

        for key in inputs.press_keys:
            self._emulator.keyboard.press_key(key)

        executed = 0
        if inputs.max_instructions is not None:
            executed = self._emulator.run(max_instructions=inputs.max_instructions)
        return executed

    def step(self, inputs: Optional[OrchestratorInputs] = None) -> OrchestratorSnapshot:
        """Apply inputs, advance the system, and return the resulting snapshot."""

        step_inputs = inputs or OrchestratorInputs()
        executed = self.apply_inputs(step_inputs)
        return self._build_snapshot(executed, dict(step_inputs.metadata))

    def run_scenario(
        self,
        steps: Sequence[OrchestratorInputs],
        *,
        include_initial_snapshot: bool = False,
    ) -> list[OrchestratorSnapshot]:
        """Run a multi-step scenario returning snapshots after each step."""

        snapshots: list[OrchestratorSnapshot] = []
        if include_initial_snapshot:
            snapshots.append(self.capture_snapshot())

        for step_inputs in steps:
            snapshots.append(self.step(step_inputs))
        return snapshots

    # ------------------------------------------------------------------ #
    # Internal snapshot helpers
    # ------------------------------------------------------------------ #
    def _build_snapshot(
        self, executed: int, metadata: Dict[str, object]
    ) -> OrchestratorSnapshot:
        state = capture_state(self._emulator)
        if self._last_state is not None:
            diff = diff_states(self._last_state, state)
        else:
            diff = empty_state_diff()
        snapshot = OrchestratorSnapshot(
            state=state,
            diff=diff,
            executed_instructions=executed,
            cycle_count=self._emulator.cycle_count,
            memory_access_log=self._emulator.memory.get_imem_access_tracking(),
            metadata=dict(metadata),
        )
        self._last_state = state
        return snapshot


__all__ = [
    "OrchestratorInputs",
    "OrchestratorSnapshot",
    "SnapshotOrchestrator",
]
