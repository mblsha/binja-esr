"""Canonical emulator state snapshots and diff utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple
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


@dataclass(frozen=True)
class CPUState:
    """Registers, flags, and execution counters captured from the SC62015 core."""

    registers: Dict[str, int]
    flags: Dict[str, int]
    cycles: int
    instruction_count: int


@dataclass(frozen=True)
class MemoryState:
    """Internal RAM and interrupt mask/status bytes."""

    internal_ram: bytes
    imr: int
    isr: int


@dataclass(frozen=True)
class KeyboardState:
    """Keyboard latch, FIFO, and debounced key data."""

    pressed_keys: Tuple[str, ...]
    fifo: Tuple[int, ...]
    kol: int
    koh: int
    kil: int


@dataclass(frozen=True)
class TimerState:
    """Programmable timer scheduler state."""

    enabled: bool
    next_mti: int
    next_sti: int
    mti_period: int
    sti_period: int


@dataclass(frozen=True)
class PeripheralState:
    """Snapshots of serial, cassette, and stdio adapters."""

    serial: SerialSnapshot
    cassette: CassetteSnapshot
    stdio: StdIOSnapshot


@dataclass(frozen=True)
class EmulatorState:
    """Composite immutable snapshot of emulator subsystems."""

    cpu: CPUState
    memory: MemoryState
    keyboard: KeyboardState
    timers: TimerState
    peripherals: PeripheralState
    display: LCDSnapshot


@dataclass(frozen=True)
class FieldDiff:
    """Difference for a single named field."""

    name: str
    before: object
    after: object


@dataclass(frozen=True)
class StateDiff:
    """Aggregated differences between two emulator states."""

    cpu: Tuple[FieldDiff, ...] = field(default_factory=tuple)
    memory: Tuple[FieldDiff, ...] = field(default_factory=tuple)
    keyboard: Tuple[FieldDiff, ...] = field(default_factory=tuple)
    timers: Tuple[FieldDiff, ...] = field(default_factory=tuple)
    display_changed: bool = False
    peripherals_changed: Dict[str, bool] = field(default_factory=dict)

    def is_empty(self) -> bool:
        """Return True when no differences were recorded."""

        return (
            not self.cpu
            and not self.memory
            and not self.keyboard
            and not self.timers
            and not self.display_changed
            and not any(self.peripherals_changed.values())
        )


def empty_state_diff() -> StateDiff:
    """Return a reusable empty diff instance."""

    return StateDiff()


def capture_state(emulator: PCE500Emulator) -> EmulatorState:
    """Capture the current emulator state as canonical snapshot."""

    cpu_state = _capture_cpu_state(emulator)
    memory_state = _capture_memory_state(emulator)
    keyboard_state = _capture_keyboard_state(emulator)
    timer_state = _capture_timer_state(emulator)
    peripheral_state = _capture_peripheral_state(emulator.peripherals)
    display_state = emulator.lcd.get_snapshot()
    return EmulatorState(
        cpu=cpu_state,
        memory=memory_state,
        keyboard=keyboard_state,
        timers=timer_state,
        peripherals=peripheral_state,
        display=display_state,
    )


def diff_states(before: Optional[EmulatorState], after: EmulatorState) -> StateDiff:
    """Compute structured differences between two emulator states."""

    if before is None:
        return empty_state_diff()

    cpu_diffs = _diff_cpu(before.cpu, after.cpu)
    memory_diffs = _diff_memory(before.memory, after.memory)
    keyboard_diffs = _diff_keyboard(before.keyboard, after.keyboard)
    timer_diffs = _diff_timers(before.timers, after.timers)
    display_changed = before.display != after.display
    peripherals_changed = {
        "serial": before.peripherals.serial != after.peripherals.serial,
        "cassette": before.peripherals.cassette != after.peripherals.cassette,
        "stdio": before.peripherals.stdio != after.peripherals.stdio,
    }
    return StateDiff(
        cpu=cpu_diffs,
        memory=memory_diffs,
        keyboard=keyboard_diffs,
        timers=timer_diffs,
        display_changed=display_changed,
        peripherals_changed=peripherals_changed,
    )


def _capture_cpu_state(emulator: PCE500Emulator) -> CPUState:
    snapshot = emulator.get_cpu_state()
    registers = {
        "pc": int(snapshot["pc"]),
        "a": int(snapshot["a"]),
        "b": int(snapshot["b"]),
        "ba": int(snapshot["ba"]),
        "i": int(snapshot["i"]),
        "x": int(snapshot["x"]),
        "y": int(snapshot["y"]),
        "u": int(snapshot["u"]),
        "s": int(snapshot["s"]),
    }
    flags = {name: int(value) for name, value in snapshot["flags"].items()}
    return CPUState(
        registers=registers,
        flags=flags,
        cycles=int(snapshot["cycles"]),
        instruction_count=int(emulator.instruction_count),
    )


def _capture_memory_state(emulator: PCE500Emulator) -> MemoryState:
    ram_start = emulator.INTERNAL_RAM_START
    ram_end = ram_start + emulator.INTERNAL_RAM_SIZE
    internal_ram = bytes(emulator.memory.external_memory[ram_start:ram_end])
    imr_addr = INTERNAL_MEMORY_START + IMEMRegisters.IMR
    isr_addr = INTERNAL_MEMORY_START + IMEMRegisters.ISR
    imr = emulator.memory.read_byte(imr_addr) & 0xFF
    isr = emulator.memory.read_byte(isr_addr) & 0xFF
    return MemoryState(internal_ram=internal_ram, imr=imr, isr=isr)


def _capture_keyboard_state(emulator: PCE500Emulator) -> KeyboardState:
    keyboard = emulator.keyboard
    pressed = tuple(sorted(keyboard.get_pressed_keys()))
    fifo = tuple(keyboard.fifo_snapshot())
    return KeyboardState(
        pressed_keys=pressed,
        fifo=fifo,
        kol=keyboard.kol_value,
        koh=keyboard.koh_value,
        kil=keyboard.peek_keyboard_input(),
    )


def _capture_timer_state(emulator: PCE500Emulator) -> TimerState:
    scheduler = emulator._scheduler  # pylint: disable=protected-access
    return TimerState(
        enabled=bool(scheduler.enabled),
        next_mti=int(scheduler.next_mti),
        next_sti=int(scheduler.next_sti),
        mti_period=int(scheduler.mti_period),
        sti_period=int(scheduler.sti_period),
    )


def _capture_peripheral_state(peripherals: PeripheralManager) -> PeripheralState:
    return PeripheralState(
        serial=peripherals.serial.snapshot(),
        cassette=peripherals.cassette.snapshot(),
        stdio=peripherals.stdio.snapshot(),
    )


def _diff_cpu(before: CPUState, after: CPUState) -> Tuple[FieldDiff, ...]:
    diffs: list[FieldDiff] = []
    diffs.extend(_diff_mapping("registers", before.registers, after.registers))
    diffs.extend(_diff_mapping("flags", before.flags, after.flags))
    if before.cycles != after.cycles:
        diffs.append(FieldDiff("cycles", before.cycles, after.cycles))
    if before.instruction_count != after.instruction_count:
        diffs.append(
            FieldDiff(
                "instruction_count",
                before.instruction_count,
                after.instruction_count,
            )
        )
    return tuple(diffs)


def _diff_memory(before: MemoryState, after: MemoryState) -> Tuple[FieldDiff, ...]:
    diffs: list[FieldDiff] = []
    if before.imr != after.imr:
        diffs.append(FieldDiff("imr", before.imr, after.imr))
    if before.isr != after.isr:
        diffs.append(FieldDiff("isr", before.isr, after.isr))
    if before.internal_ram != after.internal_ram:
        diffs.append(FieldDiff("internal_ram", "<bytes>", "<bytes>"))
    return tuple(diffs)


def _diff_keyboard(
    before: KeyboardState, after: KeyboardState
) -> Tuple[FieldDiff, ...]:
    diffs: list[FieldDiff] = []
    if before.pressed_keys != after.pressed_keys:
        diffs.append(FieldDiff("pressed_keys", before.pressed_keys, after.pressed_keys))
    if before.fifo != after.fifo:
        diffs.append(FieldDiff("fifo", before.fifo, after.fifo))
    if before.kol != after.kol:
        diffs.append(FieldDiff("kol", before.kol, after.kol))
    if before.koh != after.koh:
        diffs.append(FieldDiff("koh", before.koh, after.koh))
    if before.kil != after.kil:
        diffs.append(FieldDiff("kil", before.kil, after.kil))
    return tuple(diffs)


def _diff_timers(before: TimerState, after: TimerState) -> Tuple[FieldDiff, ...]:
    diffs: list[FieldDiff] = []
    if before.enabled != after.enabled:
        diffs.append(FieldDiff("enabled", before.enabled, after.enabled))
    if before.next_mti != after.next_mti:
        diffs.append(FieldDiff("next_mti", before.next_mti, after.next_mti))
    if before.next_sti != after.next_sti:
        diffs.append(FieldDiff("next_sti", before.next_sti, after.next_sti))
    if before.mti_period != after.mti_period:
        diffs.append(FieldDiff("mti_period", before.mti_period, after.mti_period))
    if before.sti_period != after.sti_period:
        diffs.append(FieldDiff("sti_period", before.sti_period, after.sti_period))
    return tuple(diffs)


def _diff_mapping(
    prefix: str, before: Dict[str, int], after: Dict[str, int]
) -> Iterable[FieldDiff]:
    for key in sorted(set(before.keys()) | set(after.keys())):
        previous = before.get(key)
        current = after.get(key)
        if previous != current:
            yield FieldDiff(f"{prefix}.{key}", previous, current)


__all__ = [
    "CPUState",
    "MemoryState",
    "KeyboardState",
    "TimerState",
    "PeripheralState",
    "EmulatorState",
    "FieldDiff",
    "StateDiff",
    "capture_state",
    "diff_states",
    "empty_state_diff",
]
