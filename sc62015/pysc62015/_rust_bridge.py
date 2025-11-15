from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import logging
from typing import Dict, Tuple

import _sc62015_rustcore as rustcore
from sc62015.pysc62015 import emulator as _emulator
from sc62015.pysc62015.constants import INTERNAL_MEMORY_START, PC_MASK
from sc62015.pysc62015.stepper import CPURegistersSnapshot

InstructionInfo = _emulator.InstructionInfo  # type: ignore[attr-defined]


logger = logging.getLogger(__name__)


def _mask(bits: int) -> int:
    return (1 << bits) - 1


@dataclass
class _Snapshot:
    registers: CPURegistersSnapshot
    temps: Dict[int, int]


class MemoryAdapter:
    """Bridge SCIL space-aware loads/stores to the emulator memory object."""

    def __init__(self, memory) -> None:
        self._memory = memory

    def load(self, space: str, addr: int, size: int) -> int:
        base = self._resolve(space, addr)
        value = 0
        width = max(1, size // 8)
        for offset in range(width):
            byte = self._memory.read_byte((base + offset) & 0xFFFFFF) & 0xFF
            value |= byte << (offset * 8)
        return value & _mask(size or (width * 8))

    def store(self, space: str, addr: int, size: int, value: int) -> None:
        base = self._resolve(space, addr)
        width = max(1, size // 8)
        for offset in range(width):
            byte = (value >> (offset * 8)) & 0xFF
            self._memory.write_byte((base + offset) & 0xFFFFFF, byte)

    def _resolve(self, space: str, addr: int) -> int:
        if space == "int":
            return INTERNAL_MEMORY_START + (addr & 0xFF)
        return addr & 0xFFFFFF


class BridgeCPU:
    """Python helper that executes instructions via the Rust SCIL backend."""

    def __init__(self, memory, reset_on_init: bool = True) -> None:
        self.memory = memory
        self._runtime = rustcore.Runtime(memory=memory, reset_on_init=reset_on_init)
        self.call_sub_level = 0
        self.halted = False
        self._temps: Dict[int, int] = {}
        self._fallback_cpu = None  # Lazily instantiated façade (python backend)
        self.stats_steps_rust = 0
        self.stats_decode_miss = 0
        self.stats_fallback_steps = 0
        self.stats_rust_errors = 0
        if reset_on_init:
            self.power_on_reset()

    def power_on_reset(self) -> None:
        self._runtime.power_on_reset()
        self.call_sub_level = 0
        self.halted = False
        self._temps.clear()

    # ------------------------------------------------------------------ #
    # Register / flag access

    def read_register(self, name: str) -> int:
        name = name.upper()
        if name.startswith("TEMP"):
            index = int(name[4:])
            return self._temps.get(index, 0)
        return int(self._runtime.read_register(name))

    def write_register(self, name: str, value: int) -> None:
        name = name.upper()
        if name.startswith("TEMP"):
            index = int(name[4:])
            if value:
                self._temps[index] = value & 0xFFFFFF
            elif index in self._temps:
                del self._temps[index]
            return
        self._runtime.write_register(name, int(value))

    def read_flag(self, name: str) -> int:
        return int(self._runtime.read_flag(name.upper()))

    def write_flag(self, name: str, value: int) -> None:
        self._runtime.write_flag(name.upper(), int(value))

    # ------------------------------------------------------------------ #
    # Execution helpers

    def execute_instruction(self, address: int) -> Tuple[int, int]:
        address &= PC_MASK
        if address != self._read_pc():
            self._runtime.write_register("PC", address)
        snapshot = self.snapshot_registers()
        try:
            opcode, length = self._runtime.execute_instruction()
        except Exception as exc:  # pragma: no cover - exercised via integration
            self.stats_rust_errors += 1
            logger.warning(
                "Rust SCIL execution failed at PC=%06X: %s",
                address & PC_MASK,
                exc,
            )
            self.load_snapshot(snapshot)
            opcode, length = self._execute_via_fallback(address)
            return opcode, length

        self.halted = bool(getattr(self._runtime, "halted", False))
        self.stats_steps_rust += 1
        return opcode, length

    # ------------------------------------------------------------------ #
    # Snapshots

    def _read_pc(self) -> int:
        return int(self._runtime.read_register("PC")) & PC_MASK

    def snapshot_cpu_registers(self) -> CPURegistersSnapshot:
        temps = dict(self._temps)
        snapshot = CPURegistersSnapshot(
            pc=self._read_pc(),
            ba=self.read_register("BA"),
            i=self.read_register("I"),
            x=self.read_register("X"),
            y=self.read_register("Y"),
            u=self.read_register("U"),
            s=self.read_register("S"),
            f=self.read_register("F"),
            temps=temps,
            call_sub_level=self.call_sub_level,
        )
        return snapshot

    def load_cpu_snapshot(self, snapshot: CPURegistersSnapshot) -> None:
        self._runtime.write_register("PC", snapshot.pc & PC_MASK)
        self._runtime.write_register("BA", snapshot.ba)
        self._runtime.write_register("I", snapshot.i)
        self._runtime.write_register("X", snapshot.x)
        self._runtime.write_register("Y", snapshot.y)
        self._runtime.write_register("U", snapshot.u)
        self._runtime.write_register("S", snapshot.s)
        self._runtime.write_register("F", snapshot.f)
        self._temps = dict(snapshot.temps)
        self.call_sub_level = snapshot.call_sub_level

    # ------------------------------------------------------------------ #
    # Misc helpers (used by CPU facade proxies)

    def snapshot_registers(self) -> _Snapshot:
        snapshot = self.snapshot_cpu_registers()
        return _Snapshot(registers=snapshot, temps=dict(self._temps))

    def load_snapshot(self, snapshot: _Snapshot) -> None:
        self.load_cpu_snapshot(snapshot.registers)
        self._temps = dict(snapshot.temps)

    # ------------------------------------------------------------------ #
    # Fallback helpers

    def _get_fallback_cpu(self):
        if self._fallback_cpu is None:
            from sc62015.pysc62015.cpu import CPU as _FacadeCPU

            self._fallback_cpu = _FacadeCPU(
                self.memory, reset_on_init=False, backend="python"
            )
        return self._fallback_cpu

    def _execute_via_fallback(self, address: int) -> Tuple[int, int]:
        fallback = self._get_fallback_cpu()
        snapshot = self.snapshot_registers()
        fallback.apply_snapshot(snapshot.registers)

        previous_cpu = getattr(self.memory, "cpu", None)
        self.memory.set_cpu(fallback)
        eval_info = None
        try:
            eval_info = fallback.execute_instruction(address)
            new_regs = fallback.snapshot_registers()
            self.stats_fallback_steps += 1
        finally:
            if previous_cpu is not None:
                self.memory.set_cpu(previous_cpu)
            else:
                self.memory.set_cpu(self)

        self.load_cpu_snapshot(new_regs)
        self._runtime.write_register("PC", new_regs.pc & PC_MASK)
        self.halted = False  # Legacy path doesn't provide halted info; assume false
        instr_info = getattr(eval_info, "instruction_info", None) if eval_info else None
        length = 0
        if instr_info is not None and getattr(instr_info, "length", None) is not None:
            length = int(instr_info.length)  # type: ignore[attr-defined]
        opcode = self.memory.read_byte(address & PC_MASK) & 0xFF
        return opcode, length or 1

    # ------------------------------------------------------------------ #
    # Stats

    def get_stats(self) -> Dict[str, int | dict[int, int]]:
        return {
            "steps_rust": self.stats_steps_rust,
            "decode_miss": self.stats_decode_miss,
            "fallback_steps": self.stats_fallback_steps,
            "rust_errors": self.stats_rust_errors,
        }

__all__ = ["BridgeCPU", "MemoryAdapter"]
