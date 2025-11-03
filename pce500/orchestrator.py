"""Snapshot-driven orchestrator harness wrapping the PC-E500 emulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from sc62015.pysc62015.instr.opcodes import IMEMRegisters

from .display.pipeline import LCDSnapshot
from .emulator import PCE500Emulator
from .memory import INTERNAL_MEMORY_START
from .peripherals import (
    CassetteSnapshot,
    PeripheralManager,
    SerialSnapshot,
    StdIOSnapshot,
)
from .tracing import TraceObserver, trace_dispatcher


@dataclass
class CPUSnapshot:
    """Registers and flag state captured from the SC62015 core."""

    registers: Dict[str, int]
    flags: Dict[str, int]
    cycles: int
    instruction_count: int


@dataclass
class MemorySnapshot:
    """Subset of memory relevant to snapshot comparisons."""

    internal_ram: bytes
    imr: int
    isr: int


@dataclass
class KeyboardSnapshot:
    """Keyboard latch, FIFO, and pressed key information."""

    pressed_keys: List[str]
    fifo: List[int]
    kol: int
    koh: int
    kil: int


@dataclass
class TimerSnapshot:
    """State of the programmable timers."""

    enabled: bool
    next_mti: int
    next_sti: int
    mti_period: int
    sti_period: int


@dataclass
class PeripheralSnapshots:
    """Snapshots of peripheral adapters."""

    serial: SerialSnapshot
    cassette: CassetteSnapshot
    stdio: StdIOSnapshot


@dataclass
class OrchestratorSnapshot:
    """Composite snapshot captured after each orchestrator step."""

    cpu: CPUSnapshot
    memory: MemorySnapshot
    keyboard: KeyboardSnapshot
    display: LCDSnapshot
    peripherals: PeripheralSnapshots
    timers: TimerSnapshot
    memory_access_log: Dict[str, Dict[str, List[tuple[int, int]]]]
    executed_instructions: int = 0
    cycle_count: int = 0
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

    # ------------------------------------------------------------------ #
    # Lifecycle helpers
    # ------------------------------------------------------------------ #
    @property
    def emulator(self) -> PCE500Emulator:
        return self._emulator

    def reset(self) -> None:
        """Reset all subsystems to their power-on state."""

        self._emulator.reset()

    def load_rom(self, rom_data: bytes, *, start_address: Optional[int] = None) -> None:
        """Load a ROM blob into memory and reset to the entry vector."""

        self._emulator.load_rom(rom_data, start_address=start_address)
        self._emulator.reset()

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

    def close(self) -> None:
        """Release registered observers and stop tracing."""

        for observer in tuple(self._registered_observers):
            trace_dispatcher.unregister(observer)
        self._registered_observers.clear()
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

        cpu_state = self._capture_cpu_state(executed_instructions)
        memory_state = self._capture_memory_state()
        keyboard_state = self._capture_keyboard_state()
        display_state = self._emulator.lcd.get_snapshot()
        peripheral_state = self._capture_peripheral_state()
        timer_state = self._capture_timer_state()
        memory_access = self._emulator.memory.get_imem_access_tracking()

        return OrchestratorSnapshot(
            cpu=cpu_state,
            memory=memory_state,
            keyboard=keyboard_state,
            display=display_state,
            peripherals=peripheral_state,
            timers=timer_state,
            memory_access_log=memory_access,
            executed_instructions=executed_instructions,
            cycle_count=self._emulator.cycle_count,
            metadata=dict(metadata or {}),
        )

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
        return self.capture_snapshot(
            executed_instructions=executed, metadata=step_inputs.metadata
        )

    def run_scenario(
        self,
        steps: Sequence[OrchestratorInputs],
        *,
        include_initial_snapshot: bool = False,
    ) -> List[OrchestratorSnapshot]:
        """Run a multi-step scenario returning snapshots after each step."""

        snapshots: List[OrchestratorSnapshot] = []
        if include_initial_snapshot:
            snapshots.append(self.capture_snapshot())

        for step_inputs in steps:
            snapshots.append(self.step(step_inputs))
        return snapshots

    # ------------------------------------------------------------------ #
    # Internal snapshot helpers
    # ------------------------------------------------------------------ #
    def _capture_cpu_state(self, executed: int) -> CPUSnapshot:
        state = self._emulator.get_cpu_state()
        registers = {
            key: int(state[key])
            for key in ("pc", "a", "b", "ba", "i", "x", "y", "u", "s")
        }
        flags = {name: int(value) for name, value in state["flags"].items()}
        return CPUSnapshot(
            registers=registers,
            flags=flags,
            cycles=int(state.get("cycles", 0)),
            instruction_count=self._emulator.instruction_count,
        )

    def _capture_memory_state(self) -> MemorySnapshot:
        emu = self._emulator
        ram_start = emu.INTERNAL_RAM_START
        ram_end = ram_start + emu.INTERNAL_RAM_SIZE
        internal_ram = bytes(emu.memory.external_memory[ram_start:ram_end])
        imr_addr = INTERNAL_MEMORY_START + IMEMRegisters.IMR
        isr_addr = INTERNAL_MEMORY_START + IMEMRegisters.ISR
        imr = emu.memory.read_byte(imr_addr) & 0xFF
        isr = emu.memory.read_byte(isr_addr) & 0xFF
        return MemorySnapshot(internal_ram=internal_ram, imr=imr, isr=isr)

    def _capture_keyboard_state(self) -> KeyboardSnapshot:
        keyboard = self._emulator.keyboard
        pressed = sorted(keyboard.get_pressed_keys())
        fifo = keyboard.fifo_snapshot()
        kol = keyboard.kol_value
        koh = keyboard.koh_value
        kil = keyboard.peek_keyboard_input()
        return KeyboardSnapshot(
            pressed_keys=pressed, fifo=fifo, kol=kol, koh=koh, kil=kil
        )

    def _capture_peripheral_state(self) -> PeripheralSnapshots:
        peripherals: PeripheralManager = self._emulator.peripherals
        return PeripheralSnapshots(
            serial=peripherals.serial.snapshot(),
            cassette=peripherals.cassette.snapshot(),
            stdio=peripherals.stdio.snapshot(),
        )

    def _capture_timer_state(self) -> TimerSnapshot:
        scheduler = self._emulator._scheduler  # pylint: disable=protected-access
        return TimerSnapshot(
            enabled=bool(scheduler.enabled),
            next_mti=int(scheduler.next_mti),
            next_sti=int(scheduler.next_sti),
            mti_period=int(scheduler.mti_period),
            sti_period=int(scheduler.sti_period),
        )


__all__ = [
    "CPUSnapshot",
    "MemorySnapshot",
    "KeyboardSnapshot",
    "TimerSnapshot",
    "PeripheralSnapshots",
    "OrchestratorSnapshot",
    "OrchestratorInputs",
    "SnapshotOrchestrator",
]
