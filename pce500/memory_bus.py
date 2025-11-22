"""Overlay-aware memory bus for the PC-E500 emulator."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Deque, Iterable, List, Literal, Optional, Tuple
from collections import deque

ReadHandler = Callable[[int, Optional[int]], int]
WriteHandler = Callable[[int, int, Optional[int]], None]


@dataclass
class MemoryOverlay:
    """Descriptor for a memory-mapped overlay."""

    start: int
    end: int
    name: str
    data: Optional[bytearray] = None
    read_only: bool = True
    read_handler: Optional[ReadHandler] = None
    write_handler: Optional[WriteHandler] = None
    perfetto_thread: str = "Memory"

    def contains(self, address: int) -> bool:
        return self.start <= address <= self.end


@dataclass
class MemoryAccessLog:
    kind: Literal["read", "write"]
    address: int
    value: int
    overlay: str
    pc: Optional[int]
    previous: Optional[int] = None


@dataclass
class ReadResult:
    value: int
    overlay: MemoryOverlay


@dataclass
class WriteResult:
    overlay: MemoryOverlay
    value: int
    previous: Optional[int]


class MemoryBus:
    """Overlay resolver with lightweight access logging."""

    def __init__(self, *, log_limit: int = 256) -> None:
        self._overlays: List[MemoryOverlay] = []
        self._read_log: Deque[MemoryAccessLog] = deque(maxlen=log_limit)
        self._write_log: Deque[MemoryAccessLog] = deque(maxlen=log_limit)

    def add_overlay(self, overlay: MemoryOverlay) -> None:
        self._overlays.append(overlay)
        self._overlays.sort(key=lambda ov: (ov.start, ov.end, ov.name))

    def remove_overlay(self, name: str) -> None:
        self._overlays = [ov for ov in self._overlays if ov.name != name]

    def iter_overlays(self) -> Iterable[MemoryOverlay]:
        return tuple(self._overlays)

    def overlay_count(self) -> int:
        return len(self._overlays)

    def read_log(self) -> Tuple[MemoryAccessLog, ...]:
        return tuple(self._read_log)

    def write_log(self) -> Tuple[MemoryAccessLog, ...]:
        return tuple(self._write_log)

    def clear_logs(self) -> None:
        self._read_log.clear()
        self._write_log.clear()

    def read(self, address: int, cpu_pc: Optional[int] = None) -> Optional[ReadResult]:
        for overlay in self._overlays:
            if not overlay.contains(address):
                continue

            value = self._read_from_overlay(overlay, address, cpu_pc)
            if value is None:
                continue

            value &= 0xFF
            self._read_log.append(
                MemoryAccessLog(
                    kind="read",
                    address=address,
                    value=value,
                    overlay=overlay.name,
                    pc=cpu_pc,
                )
            )
            return ReadResult(value=value, overlay=overlay)

        return None

    def write(
        self,
        address: int,
        value: int,
        cpu_pc: Optional[int] = None,
    ) -> Optional[WriteResult]:
        value &= 0xFF

        for overlay in self._overlays:
            if not overlay.contains(address):
                continue

            if os.getenv("LCD_WRITE_TRACE"):
                print(f"[bus] overlay {overlay.name} handling addr=0x{address:05X}")

            handled, previous = self._write_to_overlay(overlay, address, value, cpu_pc)
            if not handled:
                continue

            self._write_log.append(
                MemoryAccessLog(
                    kind="write",
                    address=address,
                    value=value,
                    overlay=overlay.name,
                    pc=cpu_pc,
                    previous=previous,
                )
            )
            return WriteResult(overlay=overlay, value=value, previous=previous)

        return None

    @staticmethod
    def _read_from_overlay(
        overlay: MemoryOverlay, address: int, cpu_pc: Optional[int]
    ) -> Optional[int]:
        if overlay.read_handler is not None:
            return overlay.read_handler(address, cpu_pc) & 0xFF

        if overlay.data is not None:
            offset = address - overlay.start
            if 0 <= offset < len(overlay.data):
                return overlay.data[offset]

        return None

    @staticmethod
    def _write_to_overlay(
        overlay: MemoryOverlay,
        address: int,
        value: int,
        cpu_pc: Optional[int],
    ) -> Tuple[bool, Optional[int]]:
        if overlay.write_handler is not None:
            overlay.write_handler(address, value, cpu_pc)
            return True, None

        if overlay.data is not None and not overlay.read_only:
            offset = address - overlay.start
            if 0 <= offset < len(overlay.data):
                previous = overlay.data[offset]
                overlay.data[offset] = value
                return True, previous

        if overlay.read_only:
            return True, None

        return False, None


__all__ = [
    "MemoryBus",
    "MemoryOverlay",
    "MemoryAccessLog",
    "ReadResult",
    "WriteResult",
]
