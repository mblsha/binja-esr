"""Cassette peripheral adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Literal

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


@dataclass(frozen=True)
class CassetteBlock:
    """Deterministic cassette block used by tests and future ROM adapters."""

    kind: Literal["header", "data"]
    payload: bytes
    checksum: int

    @classmethod
    def from_payload(
        cls, kind: Literal["header", "data"], payload: bytes
    ) -> "CassetteBlock":
        if kind == "header" and len(payload) != 0x30:
            raise ValueError("cassette header blocks must be exactly 0x30 bytes")
        return cls(kind=kind, payload=bytes(payload), checksum=sum(payload) & 0xFF)

    def verify(self) -> bool:
        return self.checksum == (sum(self.payload) & 0xFF)


@dataclass
class CassetteTapeImage:
    """Simple ordered cassette image with deterministic read/verify behavior."""

    blocks: list[CassetteBlock] = field(default_factory=list)
    cursor: int = 0

    def append_header(self, payload: bytes) -> CassetteBlock:
        block = CassetteBlock.from_payload("header", payload)
        self.blocks.append(block)
        return block

    def append_data(self, payload: bytes) -> CassetteBlock:
        block = CassetteBlock.from_payload("data", payload)
        self.blocks.append(block)
        return block

    def rewind(self) -> None:
        self.cursor = 0

    def read_next(
        self, expected_kind: Literal["header", "data"] | None = None
    ) -> CassetteBlock:
        if self.cursor >= len(self.blocks):
            raise EOFError("cassette image is at end of tape")
        block = self.blocks[self.cursor]
        if expected_kind is not None and block.kind != expected_kind:
            raise ValueError(f"expected {expected_kind} block, got {block.kind}")
        if not block.verify():
            raise ValueError("cassette block checksum mismatch")
        self.cursor += 1
        return block

    def verify_next(self, payload: bytes) -> bool:
        block = self.read_next()
        return block.payload == bytes(payload)


class CassetteAdapter:
    """Helper for manipulating cassette workspace fields."""

    def __init__(self, memory: PCE500Memory, scheduler: TimerScheduler) -> None:
        self._memory = memory
        self._scheduler = scheduler
        self.tape = CassetteTapeImage()

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
