"""Snapshot-driven SC62015 CPU stepper.

This module exposes a pure stepping helper that accepts a register snapshot and
an in-memory image, executes a single instruction using the existing
``Emulator`` implementation, and returns an updated snapshot together with the
side effects that occurred during the step.  The goal is to decouple instruction
execution from the full PC-E500 emulator so unit tests can feed deterministic
state fixtures and assert the resulting deltas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, MutableMapping, Tuple, List

from binja_test_mocks.eval_llil import Memory

from .emulator import (
    Emulator,
    NUM_TEMP_REGISTERS,
    RegisterName,
    Registers,
)


_CORE_REGISTER_FIELDS: Tuple[str, ...] = (
    "pc",
    "ba",
    "i",
    "x",
    "y",
    "u",
    "s",
    "f",
)


@dataclass(slots=True)
class CPURegistersSnapshot:
    """Minimal register file snapshot for the SC62015 core."""

    pc: int
    ba: int = 0
    i: int = 0
    x: int = 0
    y: int = 0
    u: int = 0
    s: int = 0
    f: int = 0
    temps: Dict[int, int] = field(default_factory=dict)
    call_sub_level: int = 0

    @classmethod
    def from_registers(cls, regs: Registers) -> "CPURegistersSnapshot":
        temps: Dict[int, int] = {}
        for index in range(NUM_TEMP_REGISTERS):
            reg = getattr(RegisterName, f"TEMP{index}")
            value = regs.get(reg)
            if value:
                temps[index] = value

        return cls(
            pc=regs.get(RegisterName.PC),
            ba=regs.get(RegisterName.BA),
            i=regs.get(RegisterName.I),
            x=regs.get(RegisterName.X),
            y=regs.get(RegisterName.Y),
            u=regs.get(RegisterName.U),
            s=regs.get(RegisterName.S),
            f=regs.get(RegisterName.F),
            temps=temps,
            call_sub_level=regs.call_sub_level,
        )

    def apply_to(self, regs: Registers) -> None:
        regs.set(RegisterName.PC, self.pc)
        regs.set(RegisterName.BA, self.ba)
        regs.set(RegisterName.I, self.i)
        regs.set(RegisterName.X, self.x)
        regs.set(RegisterName.Y, self.y)
        regs.set(RegisterName.U, self.u)
        regs.set(RegisterName.S, self.s)
        regs.set(RegisterName.F, self.f)

        for index in range(NUM_TEMP_REGISTERS):
            value = self.temps.get(index, 0)
            regs.set(getattr(RegisterName, f"TEMP{index}"), value)

        regs.call_sub_level = self.call_sub_level

    def to_dict(self) -> Dict[str, int]:
        values = {
            "pc": self.pc,
            "ba": self.ba,
            "i": self.i,
            "x": self.x,
            "y": self.y,
            "u": self.u,
            "s": self.s,
            "f": self.f,
        }
        for index, value in self.temps.items():
            values[f"TEMP{index}"] = value
        values["call_sub_level"] = self.call_sub_level
        return values

    def diff(self, other: "CPURegistersSnapshot") -> Dict[str, Tuple[int, int]]:
        diffs: Dict[str, Tuple[int, int]] = {}
        for field_name in _CORE_REGISTER_FIELDS:
            before = getattr(self, field_name)
            after = getattr(other, field_name)
            if before != after:
                diffs[field_name.upper()] = (before, after)

        all_temps: Iterable[int] = set(self.temps) | set(other.temps)
        for index in sorted(all_temps):
            before = self.temps.get(index, 0)
            after = other.temps.get(index, 0)
            if before != after:
                diffs[f"TEMP{index}"] = (before, after)

        if self.call_sub_level != other.call_sub_level:
            diffs["call_sub_level"] = (self.call_sub_level, other.call_sub_level)

        return diffs


@dataclass(slots=True)
class MemoryWrite:
    """Memory mutation captured during a CPU step."""

    address: int
    value: int
    previous: int
    size: int = 1


class _SnapshotMemory(Memory):
    """Memory adapter that records mutations while servicing CPU fetches."""

    def __init__(self, image: Mapping[int, int], default_value: int = 0) -> None:
        self._backing: MutableMapping[int, int] = dict(image)
        self._default = default_value & 0xFF
        self._writes: List[MemoryWrite] = []
        super().__init__(self._read_byte, self._write_byte)

    def _read_byte(self, address: int) -> int:
        return self._backing.get(address, self._default)

    def _write_byte(self, address: int, value: int) -> None:
        value &= 0xFF
        previous = self._backing.get(address, self._default)
        self._backing[address] = value
        self._writes.append(
            MemoryWrite(address=address, value=value, previous=previous)
        )

    @property
    def writes(self) -> Tuple[MemoryWrite, ...]:
        return tuple(self._writes)

    def snapshot(self) -> Dict[int, int]:
        return dict(self._backing)


@dataclass(slots=True)
class CPUStepResult:
    registers: CPURegistersSnapshot
    changed_registers: Dict[str, Tuple[int, int]]
    memory_writes: Tuple[MemoryWrite, ...]
    memory_image: Dict[int, int]
    instruction_name: str
    instruction_length: int


class CPUStepper:
    """Utility that executes a single SC62015 instruction from a snapshot."""

    def __init__(self, *, default_memory_value: int = 0) -> None:
        self._default_memory_value = default_memory_value & 0xFF

    def step(
        self,
        registers: CPURegistersSnapshot,
        memory_image: Mapping[int, int],
    ) -> CPUStepResult:
        snapshot_memory = _SnapshotMemory(
            memory_image,
            default_value=self._default_memory_value,
        )
        emulator = Emulator(snapshot_memory, reset_on_init=False)
        registers.apply_to(emulator.regs)

        eval_info = emulator.execute_instruction(registers.pc)

        new_registers = CPURegistersSnapshot.from_registers(emulator.regs)
        changed_registers = registers.diff(new_registers)
        instruction_length = int(eval_info.instruction_info.length or 0)

        return CPUStepResult(
            registers=new_registers,
            changed_registers=changed_registers,
            memory_writes=snapshot_memory.writes,
            memory_image=snapshot_memory.snapshot(),
            instruction_name=eval_info.instruction.name(),
            instruction_length=instruction_length,
        )


__all__ = [
    "CPUStepper",
    "CPUStepResult",
    "CPURegistersSnapshot",
    "MemoryWrite",
]
