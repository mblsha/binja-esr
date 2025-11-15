from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class LayoutEntry:
    key: str
    kind: str
    meta: Dict[str, object]


@dataclass
class StreamCtx:
    """
    Sequential reader over the operand bytes of an instruction.

    `base_len` captures already-consumed bytes (opcode + prefixes) so total
    length can be reported without duplicating consumers' knowledge.
    """

    pc: int
    data: bytes
    base_len: int = 1
    idx: int = 0
    record_layout: bool = False
    _layout: List[LayoutEntry] = field(default_factory=list, init=False)

    def _require(self, count: int) -> None:
        if self.idx + count > len(self.data):
            raise ValueError(
                f"Insufficient bytes: need {count}, "
                f"have {len(self.data) - self.idx} remaining"
            )

    def record_operand(self, key: str, kind: str, **meta) -> None:
        if not self.record_layout:
            return
        self._layout.append(LayoutEntry(key=key, kind=kind, meta=dict(meta)))

    def read_u8(self) -> int:
        self._require(1)
        value = self.data[self.idx]
        self.idx += 1
        return value

    def read_s8(self) -> int:
        raw = self.read_u8()
        return raw - 0x100 if raw & 0x80 else raw

    def read_u16_mn(self) -> tuple[int, int]:
        lo = self.read_u8()
        hi = self.read_u8()
        return lo, hi

    def read_u24_lmn(self) -> tuple[int, int, int]:
        lo = self.read_u8()
        mid = self.read_u8()
        hi = self.read_u8()
        return lo, mid, hi

    def bytes_consumed(self) -> int:
        return self.idx

    def total_length(self) -> int:
        return self.base_len + self.idx

    def remaining(self) -> int:
        return len(self.data) - self.idx

    def page20(self) -> int:
        return self.pc & 0xF0000

    def snapshot_layout(self) -> tuple[LayoutEntry, ...]:
        return tuple(self._layout)
