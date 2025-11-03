"""STDI/STDO peripheral adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Sequence

from ..memory import PCE500Memory
from ..scheduler import TimerScheduler

_STDIO_RANGES: Iterable[range] = (
    range(0x00BFD40, 0x00BFD60),
    range(0x00BFD80, 0x00BFDA0),
)

_TRACKED_ADDRS = tuple(addr for r in _STDIO_RANGES for addr in r)


@dataclass
class StdIOSnapshot:
    """Serialized view of STDI/STDO workspace bytes."""

    workspace: Dict[int, int] = field(default_factory=dict)


class StdIODeviceAdapter:
    """Utility for manipulating STDI/STDO buffers."""

    def __init__(self, memory: PCE500Memory, scheduler: TimerScheduler) -> None:
        self._memory = memory
        self._scheduler = scheduler

    def snapshot(self) -> StdIOSnapshot:
        """Capture tracked workspace bytes."""

        return StdIOSnapshot(
            workspace={addr: self._memory.read_byte(addr) for addr in _TRACKED_ADDRS}
        )

    def restore(self, snapshot: StdIOSnapshot) -> None:
        """Restore workspace bytes from a snapshot."""

        for addr, value in snapshot.workspace.items():
            self._memory.write_byte(addr, value & 0xFF)

    def load_output_buffer(self, data: Sequence[int], *, base: int = 0x00BFD48) -> None:
        """Populate the printer/STDO spool buffer starting at `base`."""

        for offset, byte in enumerate(data):
            addr = base + offset
            if addr not in _TRACKED_ADDRS:
                raise ValueError(f"Address 0x{addr:06X} not tracked by stdio adapter")
            self._memory.write_byte(addr, int(byte) & 0xFF)

    def read_workspace(self, addr: int) -> int:
        """Read a tracked workspace byte."""

        if addr not in _TRACKED_ADDRS:
            raise ValueError(f"Address 0x{addr:06X} not tracked by stdio adapter")
        return self._memory.read_byte(addr) & 0xFF

    def write_workspace(self, addr: int, value: int) -> None:
        """Write a tracked workspace byte."""

        if addr not in _TRACKED_ADDRS:
            raise ValueError(f"Address 0x{addr:06X} not tracked by stdio adapter")
        self._memory.write_byte(addr, value & 0xFF)
