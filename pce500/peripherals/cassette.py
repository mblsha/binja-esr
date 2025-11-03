"""Cassette peripheral adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable

from ..memory import PCE500Memory
from ..scheduler import TimerScheduler

_CASSETTE_RANGES: Iterable[range] = (
    range(0x00BFE20, 0x00BFE30),
    range(0x00BFE34, 0x00BFE38),
    range(0x00BFE40, 0x00BFE48),
    range(0x00BFE5A, 0x00BFE60),
    range(0x00BFEF0, 0x00BFF00),
)

_TRACKED_ADDRS = tuple(addr for r in _CASSETTE_RANGES for addr in r)


@dataclass
class CassetteSnapshot:
    """Serialized view of cassette workspace bytes."""

    workspace: Dict[int, int] = field(default_factory=dict)


class CassetteAdapter:
    """Helper for manipulating cassette workspace fields."""

    def __init__(self, memory: PCE500Memory, scheduler: TimerScheduler) -> None:
        self._memory = memory
        self._scheduler = scheduler

    def snapshot(self) -> CassetteSnapshot:
        """Capture tracked workspace bytes."""

        return CassetteSnapshot(
            workspace={addr: self._memory.read_byte(addr) for addr in _TRACKED_ADDRS}
        )

    def restore(self, snapshot: CassetteSnapshot) -> None:
        """Restore workspace bytes from a snapshot."""

        for addr, value in snapshot.workspace.items():
            self._memory.write_byte(addr, value & 0xFF)

    def write_workspace(self, addr: int, value: int) -> None:
        """Write a byte within the tracked cassette workspace."""

        if addr not in _TRACKED_ADDRS:
            raise ValueError(f"Address 0x{addr:06X} not tracked by cassette adapter")
        self._memory.write_byte(addr, value & 0xFF)

    def read_workspace(self, addr: int) -> int:
        """Read a tracked cassette workspace byte."""

        if addr not in _TRACKED_ADDRS:
            raise ValueError(f"Address 0x{addr:06X} not tracked by cassette adapter")
        return self._memory.read_byte(addr) & 0xFF
