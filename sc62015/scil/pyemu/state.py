from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from ...pysc62015.constants import PC_MASK


_BASE_WIDTHS: Dict[str, int] = {
    "BA": 16,
    "I": 16,
    "X": 24,
    "Y": 24,
    "U": 24,
    "S": 24,
    "F": 8,
}

_SUBREG_INFO: Dict[str, tuple[str, int, int]] = {
    "A": ("BA", 0, 0xFF),
    "B": ("BA", 8, 0xFF),
    "IL": ("I", 0, 0xFF),
    "IH": ("I", 8, 0xFF),
}


def _mask(bits: int) -> int:
    return (1 << bits) - 1


@dataclass
class CPUState:
    """Architecture-aware register bank for the SCIL Python emulator."""

    _regs: Dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in _BASE_WIDTHS}
    )
    _flags: Dict[str, int] = field(default_factory=lambda: {"C": 0, "Z": 0})
    pc: int = 0
    halted: bool = False

    def reset(self) -> None:
        for name in _BASE_WIDTHS:
            self._regs[name] = 0
        for flag in self._flags:
            self._flags[flag] = 0
        self.pc = 0
        self.halted = False

    def to_dict(self) -> Dict[str, int]:
        regs = dict(self._regs)
        for name, (_, _, mask) in _SUBREG_INFO.items():
            regs[name] = self.get_reg(name, mask.bit_length())
        return {
            "pc": self.pc & PC_MASK,
            "halted": bool(self.halted),
            "regs": regs,
            "flags": dict(self._flags),
        }

    def load_dict(self, payload: Dict[str, int]) -> None:
        self.reset()
        self.pc = payload.get("pc", 0) & PC_MASK
        self.halted = bool(payload.get("halted", False))
        regs = payload.get("regs", {})
        skip_subregs = {
            name
            for name in regs
            if name in _SUBREG_INFO and _SUBREG_INFO[name][0] in regs
        }
        for name, value in regs.items():
            if name in {"FC", "FZ"}:
                continue
            if name in skip_subregs:
                continue
            bits = _BASE_WIDTHS.get(name, 24)
            if name in _SUBREG_INFO and name not in _BASE_WIDTHS:
                bits = _SUBREG_INFO[name][2].bit_length()
            self.set_reg(name, int(value), bits)
        flags = payload.get("flags", {})
        for name, value in flags.items():
            self.set_flag(name, int(value))

    # ------------------------------------------------------------------ #
    # Register access helpers

    def get_reg(self, name: str, default_bits: int) -> int:
        if name == "PC":
            return self.pc & PC_MASK
        if name in _BASE_WIDTHS:
            bits = _BASE_WIDTHS[name]
            return self._regs[name] & _mask(min(bits, default_bits))
        if name in _SUBREG_INFO:
            base, shift, mask = _SUBREG_INFO[name]
            return (self._regs[base] >> shift) & mask
        return 0

    def set_reg(self, name: str, value: int, bits: int) -> None:
        if name == "PC":
            self.pc = value & PC_MASK
            return
        if name in _BASE_WIDTHS:
            width = _BASE_WIDTHS[name]
            masked = value & _mask(min(bits, width))
            self._regs[name] = masked
            if name == "F":
                self._flags["C"] = masked & 1
                self._flags["Z"] = (masked >> 1) & 1
            return
        if name in _SUBREG_INFO:
            base, shift, mask = _SUBREG_INFO[name]
            full_mask = _mask(_BASE_WIDTHS[base])
            cur = self._regs[base] & full_mask
            cur &= ~(mask << shift)
            cur |= (value & mask) << shift
            self._regs[base] = cur & full_mask
            return
        # Fallback: create a scratch register with the requested width
        self._regs[name] = value & _mask(bits)

    # ------------------------------------------------------------------ #
    # Flags

    def set_flag(self, name: str, value: int) -> None:
        bit = value & 1
        self._flags[name] = bit
        if name == "C":
            self._update_f_bit(0, bit)
        elif name == "Z":
            self._update_f_bit(1, bit)

    def get_flag(self, name: str) -> int:
        return self._flags.get(name, 0) & 1

    # ------------------------------------------------------------------ #
    # Utilities

    def snapshot(self) -> Dict[str, int]:
        snap = {name: self.get_reg(name, width) for name, width in _BASE_WIDTHS.items()}
        snap.update({name: self.get_reg(name, 8) for name in _SUBREG_INFO})
        snap["PC"] = self.pc & PC_MASK
        snap["C"] = self.get_flag("C")
        snap["Z"] = self.get_flag("Z")
        return snap

    def _update_f_bit(self, bit_index: int, value: int) -> None:
        current = self._regs.get("F", 0)
        if value & 1:
            current |= 1 << bit_index
        else:
            current &= ~(1 << bit_index)
        self._regs["F"] = current & _mask(_BASE_WIDTHS["F"])
