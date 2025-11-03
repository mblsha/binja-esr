"""STDI/STDO peripheral adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Sequence

from ..memory import PCE500Memory
from ..scheduler import TimerScheduler

_STDIO_RANGES: Iterable[range] = (
    range(0x00BFD40, 0x00BFD60),  # Printer spool and flags
    range(0x00BFD80, 0x00BFDA0),  # STDI buffers
)

_TRACKED_STDIO_ADDRS = tuple(
    addr for addr_range in _STDIO_RANGES for addr in addr_range
)


@dataclass
class StdIOSnapshot:
    """Serialized view of the STDI/STDO workspace."""

    workspace: Dict[int, int] = field(default_factory=dict)


class StdIODeviceAdapter:
    """Helper for manipulating STDI/STDO memory without raw pokes."""

    def __init__(self, memory: PCE500Memory, scheduler: TimerScheduler) -> None:
        self._memory = memory
        self._scheduler = scheduler

    def snapshot(self) -> StdIOSnapshot:
        """Capture STDI/STDO workspace bytes."""

        return StdIOSnapshot(
            workspace={
                addr: self._memory.read_byte(addr) for addr in _TRACKED_STDIO_ADDRS
            }
        )

    def restore(self, snapshot: StdIOSnapshot) -> None:
        """Restore STDI/STDO workspace bytes."""

        for addr, value in snapshot.workspace.items():
            self._memory.write_byte(addr, value & 0xFF)

    def load_output_buffer(self, data: Sequence[int], *, base: int = 0x00BFD48) -> None:
        """Populate the printer/STDO spool buffer starting at `base`."""

        for offset, byte in enumerate(data):
            addr = base + offset
            if addr not in _TRACKED_STDIO_ADDRS:
                raise ValueError(
                    f"Address 0x{addr:06X} is outside the tracked STDI/STDO workspace"
                )
            self._memory.write_byte(addr, int(byte) & 0xFF)

    def read_workspace(self, addr: int) -> int:
        """Read a tracked STDI/STDO byte."""

        if addr not in _TRACKED_STDIO_ADDRS:
            raise ValueError(f"Address 0x{addr:06X} not tracked by stdio adapter")
        return self._memory.read_byte(addr) & 0xFF

    def write_workspace(self, addr: int, value: int) -> None:
        """Write a tracked STDI/STDO byte."""

        if addr not in _TRACKED_STDIO_ADDRS:
            raise ValueError(f"Address 0x{addr:06X} not tracked by stdio adapter")
        self._memory.write_byte(addr, value & 0xFF)
