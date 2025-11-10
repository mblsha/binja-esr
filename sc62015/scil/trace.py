from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List


@dataclass(frozen=True)
class TraceEntry:
    event: str
    addr: int
    mnemonic: str
    detail: str


class TraceBuffer:
    def __init__(self, capacity: int = 32) -> None:
        self._entries: Deque[TraceEntry] = deque(maxlen=capacity)

    def record(self, entry: TraceEntry) -> None:
        self._entries.append(entry)

    def snapshot(self) -> List[TraceEntry]:
        return list(self._entries)


TRACE_BUFFER = TraceBuffer()


def record(event: str, addr: int, mnemonic: str, detail: str) -> None:
    TRACE_BUFFER.record(
        TraceEntry(event=event, addr=addr, mnemonic=mnemonic, detail=detail)
    )


def snapshot() -> List[TraceEntry]:
    return TRACE_BUFFER.snapshot()


__all__ = ["record", "snapshot", "TraceEntry"]
