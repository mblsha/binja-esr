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
import operator
from typing import Dict, Iterable, Mapping, MutableMapping, Tuple, List

from binja_test_mocks.eval_llil import Memory

from .emulator import NUM_TEMP_REGISTERS, RegisterName, Registers
from .constants import validate_f_image


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
        # Build the complete register image in isolation.  Dataclass type
        # annotations are not runtime validation: a malformed later field must
        # not leave earlier registers committed, and unknown temporaries must
        # not disappear silently on one backend while Rust rejects them.
        def snapshot_u32(field_name: str, value: object) -> int:
            try:
                result = operator.index(value)
            except TypeError as exc:
                raise TypeError(f"snapshot {field_name} must be an integer") from exc
            if not 0 <= result <= 0xFFFF_FFFF:
                raise ValueError(
                    f"snapshot {field_name} must fit in an unsigned 32-bit value"
                )
            return result

        core_values = {
            field_name: snapshot_u32(field_name, getattr(self, field_name))
            for field_name in _CORE_REGISTER_FIELDS
        }
        validate_f_image(core_values["f"])

        if not isinstance(self.temps, Mapping):
            raise TypeError("snapshot temps must be a mapping")
        validated_temps: Dict[int, int] = {}
        for raw_index, raw_value in self.temps.items():
            index = snapshot_u32("temporary register index", raw_index)
            if index >= NUM_TEMP_REGISTERS:
                raise ValueError(
                    f"snapshot contains unknown temporary register TEMP{index}"
                )
            validated_temps[index] = snapshot_u32(f"TEMP{index}", raw_value)

        call_sub_level = snapshot_u32("call_sub_level", self.call_sub_level)
        if call_sub_level > 0x7FFF_FFFF:
            raise ValueError(
                "snapshot call_sub_level must fit in a signed 32-bit value"
            )

        candidate = Registers()
        for field_name in _CORE_REGISTER_FIELDS:
            candidate.set(RegisterName[field_name.upper()], core_values[field_name])
        for index in range(NUM_TEMP_REGISTERS):
            candidate.set(
                getattr(RegisterName, f"TEMP{index}"), validated_temps.get(index, 0)
            )
        candidate.call_sub_level = call_sub_level

        regs._values.clear()
        regs._values.update(candidate._values)
        regs.call_sub_level = candidate.call_sub_level

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
        self._wait_cycles: List[int] = []
        super().__init__(self._read_byte, self._write_byte)

    def wait_cycles(self, cycles: int) -> None:
        """Account for WAIT timing even though the pure stepper has no clock."""
        self._wait_cycles.append(int(cycles))

    def _read_byte(self, address: int) -> int:
        return self._backing.get(address, self._default)

    def peek_byte_for_preflight(self, address: int, _pc: int | None = None) -> int:
        """Read the immutable snapshot image without recording a bus access."""

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

    def __init__(
        self,
        *,
        default_memory_value: int = 0,
        backend: str | None = None,
    ) -> None:
        self._default_memory_value = default_memory_value & 0xFF
        self._backend = backend

    def step(
        self,
        registers: CPURegistersSnapshot,
        memory_image: Mapping[int, int],
    ) -> CPUStepResult:
        from .cpu import CPU  # Local import to avoid circular dependency

        snapshot_memory = _SnapshotMemory(
            memory_image,
            default_value=self._default_memory_value,
        )
        cpu = CPU(snapshot_memory, reset_on_init=False, backend=self._backend)
        cpu.apply_snapshot(registers)

        eval_info = cpu.execute_instruction(registers.pc)

        new_registers = cpu.snapshot_registers()
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
