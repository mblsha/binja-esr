from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import logging
from typing import Dict, Iterable, Tuple, Optional

import _sc62015_rustcore as rustcore
from sc62015.decoding.bind import DecodedInstr
from sc62015.decoding.dispatcher import CompatDispatcher
from sc62015.pysc62015.emulator import InstructionInfo
from sc62015.pysc62015.constants import INTERNAL_MEMORY_START, PC_MASK
from sc62015.pysc62015.stepper import CPURegistersSnapshot
from sc62015.scil import from_decoded, serde
from sc62015.scil.pyemu import CPUState


logger = logging.getLogger(__name__)


def _mask(bits: int) -> int:
    return (1 << bits) - 1


_REGISTER_WIDTHS: Dict[str, int] = {
    "A": 8,
    "B": 8,
    "BA": 16,
    "I": 16,
    "X": 24,
    "Y": 24,
    "U": 24,
    "S": 24,
    "F": 8,
    "PC": 20,
}


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
    """Python helper that executes instructions via SCIL PyEMU."""

    def __init__(self, memory, reset_on_init: bool = True) -> None:
        self.memory = memory
        self.bus = MemoryAdapter(memory)
        self.state = CPUState()
        self.dispatcher = CompatDispatcher()
        self.call_sub_level = 0
        self.halted = False
        self._temps: Dict[int, int] = {}
        self._fallback_cpu = None  # Lazily instantiated façade (python backend)
        self.stats_steps_rust = 0
        self.stats_decode_miss = 0
        self.stats_fallback_steps = 0
        self.stats_rust_errors = 0
        self._decode_hist: Counter[int] = Counter()
        if reset_on_init:
            self.power_on_reset()

    def power_on_reset(self) -> None:
        self.state.reset()
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
        if name == "PC":
            return self.state.pc & PC_MASK
        width = _REGISTER_WIDTHS.get(name, 24)
        return self.state.get_reg(name, width)

    def write_register(self, name: str, value: int) -> None:
        name = name.upper()
        if name.startswith("TEMP"):
            index = int(name[4:])
            if value:
                self._temps[index] = value & 0xFFFFFF
            elif index in self._temps:
                del self._temps[index]
            return
        if name == "PC":
            self.state.pc = value & PC_MASK
            return
        width = _REGISTER_WIDTHS.get(name, 24)
        self.state.set_reg(name, value, width)

    def read_flag(self, name: str) -> int:
        return self.state.get_flag(name.upper())

    def write_flag(self, name: str, value: int) -> None:
        self.state.set_flag(name.upper(), value)

    # ------------------------------------------------------------------ #
    # Execution helpers

    def execute_instruction(self, address: int) -> Tuple[int, int]:
        decoded, length, opcode = self._decode_instruction(address)

        if isinstance(decoded, DecodedInstr):
            build = from_decoded.build(decoded)
            start_pc = self.state.pc & PC_MASK
            snapshot = self.snapshot_registers()
            state_json = json.dumps(self.state.to_dict())
            instr_json = json.dumps(serde.instr_to_dict(build.instr))
            binder_json = json.dumps(serde.binder_to_dict(build.binder))
            pre_json = (
                json.dumps(serde.prelatch_to_dict(build.pre_applied))
                if build.pre_applied
                else None
            )
            try:
                new_state_json = rustcore.scil_step_json(
                    state_json,
                    instr_json,
                    binder_json,
                    self.bus,
                    pre_json,
                )
            except Exception as exc:  # pragma: no cover - exercised via integration
                self.stats_rust_errors += 1
                logger.warning(
                    "Rust SCIL execution failed at PC=%06X opcode=%02X: %s",
                    address & PC_MASK,
                    opcode & 0xFF,
                    exc,
                )
                self.load_snapshot(snapshot)
                self._execute_via_fallback(address, length)
                return opcode, length

            self.state.load_dict(json.loads(new_state_json))
            if (self.state.pc & PC_MASK) == start_pc:
                self.state.pc = (start_pc + length) & PC_MASK
            self.halted = bool(getattr(self.state, "halted", False))
            self.stats_steps_rust += 1
            return opcode, length

        # SCIL decode miss: execute via python façade fallback
        self.stats_decode_miss += 1
        self._decode_hist[opcode & 0xFF] += 1
        self._execute_via_fallback(address, length)
        return opcode, length

    def _decode_instruction(
        self, address: int
    ) -> Tuple[Optional[DecodedInstr], int, int]:
        # Reset pending PRE latch before decoding a fresh instruction
        if hasattr(self.dispatcher, "_pending_pre"):
            self.dispatcher._pending_pre = None  # type: ignore[attr-defined]
        buf = bytearray(
            self.memory.read_byte((address + offset) & 0xFFFFFF) & 0xFF
            for offset in range(128)
        )
        total = 0
        cursor = 0
        opcode = buf[0]
        dispatcher = self.dispatcher
        while cursor < len(buf):
            data = bytes(buf[cursor:])
            result = dispatcher.try_decode(data, (address + cursor) & PC_MASK)
            if result is None:
                break
            length, decoded = result
            total += length
            cursor += length
            if decoded is not None:
                return decoded, total, opcode
        # Fallback: use legacy decoder just to recover metadata
        self._log_decode_failure(address, opcode, buf[:8])
        fallback = self._get_fallback_cpu()
        instr = fallback.decode_instruction(address)
        if instr is None:
            raise ValueError(f"Failed to decode instruction at {address:#06X}")
        info = InstructionInfo()
        instr.analyze(info, address)
        return None, int(info.length), instr.opcode or opcode

    # ------------------------------------------------------------------ #
    # Snapshots

    def snapshot_cpu_registers(self) -> CPURegistersSnapshot:
        temps = dict(self._temps)
        snapshot = CPURegistersSnapshot(
            pc=self.state.pc & PC_MASK,
            ba=self.state.get_reg("BA", 16),
            i=self.state.get_reg("I", 16),
            x=self.state.get_reg("X", 24),
            y=self.state.get_reg("Y", 24),
            u=self.state.get_reg("U", 24),
            s=self.state.get_reg("S", 24),
            f=self.state.get_reg("F", 8),
            temps=temps,
            call_sub_level=self.call_sub_level,
        )
        return snapshot

    def load_cpu_snapshot(self, snapshot: CPURegistersSnapshot) -> None:
        self.state.pc = snapshot.pc & PC_MASK
        self.state.set_reg("BA", snapshot.ba, 16)
        self.state.set_reg("I", snapshot.i, 16)
        self.state.set_reg("X", snapshot.x, 24)
        self.state.set_reg("Y", snapshot.y, 24)
        self.state.set_reg("U", snapshot.u, 24)
        self.state.set_reg("S", snapshot.s, 24)
        self.state.set_reg("F", snapshot.f, 8)
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

    def _execute_via_fallback(self, address: int, _length: int) -> None:
        fallback = self._get_fallback_cpu()
        snapshot = self.snapshot_registers()
        fallback.apply_snapshot(snapshot.registers)

        previous_cpu = getattr(self.memory, "cpu", None)
        self.memory.set_cpu(fallback)
        try:
            fallback.execute_instruction(address)
            new_regs = fallback.snapshot_registers()
            self.stats_fallback_steps += 1
        finally:
            if previous_cpu is not None:
                self.memory.set_cpu(previous_cpu)
            else:
                self.memory.set_cpu(self)

        self.load_cpu_snapshot(new_regs)
        self.state.pc = new_regs.pc & PC_MASK
        self.halted = False  # Legacy path doesn't provide halted info; assume false

    # ------------------------------------------------------------------ #
    # Stats

    def get_stats(self) -> Dict[str, int]:
        return {
            "steps_rust": self.stats_steps_rust,
            "decode_miss": self.stats_decode_miss,
            "fallback_steps": self.stats_fallback_steps,
            "rust_errors": self.stats_rust_errors,
            "decode_miss_hist": dict(self._decode_hist.most_common(10)),
        }

    # ------------------------------------------------------------------ #
    # Internal helpers

    def _log_decode_failure(
        self, address: int, opcode: int, window: Iterable[int]
    ) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return
        pre = getattr(self.dispatcher, "pending_pre", None)
        if pre is None and hasattr(self.dispatcher, "_pending_pre"):
            pre = getattr(self.dispatcher, "_pending_pre", None)
        pre_desc = "-"
        if pre is not None:
            first = getattr(pre, "first", "?")
            second = getattr(pre, "second", "?")
            pre_desc = f"{first}->{second}"
        preview = " ".join(f"{byte:02X}" for byte in window)
        logger.debug(
            "scil_decode_miss pc=%06X opcode=%02X pre=%s bytes=%s",
            address & PC_MASK,
            opcode & 0xFF,
            pre_desc,
            preview or "<none>",
        )


__all__ = ["BridgeCPU", "MemoryAdapter"]
