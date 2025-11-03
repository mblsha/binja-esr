"""Cassette peripheral adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable

from ..memory import PCE500Memory
from ..scheduler import TimerScheduler

# Address ranges derived from docs/hw/serial_port.md and iocs analysis.
_CASSETTE_RANGES: Iterable[range] = (
    range(0x00BFE20, 0x00BFE30),  # Block pointers, state flags
    range(0x00BFE34, 0x00BFE38),  # Handshake bits
    range(0x00BFE40, 0x00BFE48),  # Shared serial/cassette workspace
    range(0x00BFE5A, 0x00BFE60),  # Motor control shadows
    range(0x00BFEF0, 0x00BFF00),  # Retry counters, timestamps
)

_TRACKED_CASSETTE_ADDRS = tuple(
    addr for addr_range in _CASSETTE_RANGES for addr in addr_range
)


@dataclass
class CassetteSnapshot:
    """Serialized view of cassette workspace bytes."""

    workspace: Dict[int, int] = field(default_factory=dict)


class CassetteAdapter:
    """High-level helpers for cassette workspace fields."""

    def __init__(self, memory: PCE500Memory, scheduler: TimerScheduler) -> None:
        self._memory = memory
        self._scheduler = scheduler

    def snapshot(self) -> CassetteSnapshot:
        """Capture cassette workspace bytes."""

        return CassetteSnapshot(
            workspace={
                addr: self._memory.read_byte(addr) for addr in _TRACKED_CASSETTE_ADDRS
            }
        )

    def restore(self, snapshot: CassetteSnapshot) -> None:
        """Restore cassette workspace bytes."""

        for addr, value in snapshot.workspace.items():
            self._memory.write_byte(addr, value & 0xFF)

    def write_workspace(self, addr: int, value: int) -> None:
        """Write a byte within the tracked cassette workspace."""

        if addr not in _TRACKED_CASSETTE_ADDRS:
            raise ValueError(f"Address 0x{addr:06X} not tracked by cassette adapter")
        self._memory.write_byte(addr, value & 0xFF)

    def read_workspace(self, addr: int) -> int:
        """Read a tracked cassette workspace byte."""

        if addr not in _TRACKED_CASSETTE_ADDRS:
            raise ValueError(f"Address 0x{addr:06X} not tracked by cassette adapter")
        return self._memory.read_byte(addr) & 0xFF
