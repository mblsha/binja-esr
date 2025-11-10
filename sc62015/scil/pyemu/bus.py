from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple


def _mask(bits: int) -> int:
    return (1 << bits) - 1


@dataclass
class MemoryBus:
    """Simple in-memory implementation of the SCIL bus protocol."""

    _int_mem: Dict[int, int] = field(default_factory=dict)
    _ext_mem: Dict[int, int] = field(default_factory=dict)

    def load(self, space: str, addr: int, size: int) -> int:
        if size % 8 != 0:
            raise ValueError("Bus load size must be byte-aligned")
        length = size // 8
        store = self._select_space(space)
        base = self._normalize_addr(space, addr)
        value = 0
        for offset in range(length):
            byte = store.get(base + offset, 0) & 0xFF
            value |= byte << (offset * 8)
        return value & _mask(size)

    def store(self, space: str, addr: int, value: int, size: int) -> None:
        if size % 8 != 0:
            raise ValueError("Bus store size must be byte-aligned")
        length = size // 8
        store = self._select_space(space)
        base = self._normalize_addr(space, addr)
        for offset in range(length):
            byte = (value >> (offset * 8)) & 0xFF
            store[base + offset] = byte

    def preload_internal(self, items: Iterable[Tuple[int, int]]) -> None:
        for addr, value in items:
            self._int_mem[addr & 0xFF] = value & 0xFF

    def preload_external(self, items: Iterable[Tuple[int, int]]) -> None:
        for addr, value in items:
            self._ext_mem[addr & 0xFFFFFF] = value & 0xFF

    def dump_internal(self) -> Dict[int, int]:
        return dict(self._int_mem)

    def dump_external(self) -> Dict[int, int]:
        return dict(self._ext_mem)

    def _select_space(self, space: str) -> Dict[int, int]:
        if space == "int":
            return self._int_mem
        if space in {"ext", "code"}:
            return self._ext_mem
        raise ValueError(f"Unknown space {space}")

    @staticmethod
    def _normalize_addr(space: str, addr: int) -> int:
        if space == "int":
            return addr & 0xFF
        return addr & 0xFFFFFF
