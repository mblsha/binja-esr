from __future__ import annotations

from typing import List

from .bind import Addr16Page, Addr24, DecodedInstr, Disp8, Imm8, PreLatch


class BoundStream:
    """
    Replays operand bytes using the decoded binds so legacy code can read the
    exact sequence it expects (little-endian order, magnitude bytes, etc.).
    """

    def __init__(self, decoded: DecodedInstr):
        self._decoded = decoded
        self._buffer: List[int] = []
        self._cursor = 0
        self._materialize()

    def _append(self, *values: int) -> None:
        for value in values:
            if not 0 <= value <= 0xFF:
                raise ValueError(f"Replay byte out of range: {value:#x}")
            self._buffer.append(value)

    def _materialize(self) -> None:
        binds = self._decoded.binds

        imm = binds.get("n")
        if isinstance(imm, Imm8):
            self._append(imm.value)

        disp = binds.get("disp")
        if isinstance(disp, Disp8):
            if self._decoded.opcode == 0x19:
                self._append(abs(disp.value))
            else:
                self._append(disp.value & 0xFF)

        addr16 = binds.get("addr16_page")
        if isinstance(addr16, Addr16Page):
            self._append(addr16.offs16.lo, addr16.offs16.hi)

        addr24 = binds.get("addr24")
        if isinstance(addr24, Addr24):
            self._append(addr24.v.lo, addr24.v.mid, addr24.v.hi)

    def read_u8(self) -> int:
        if self._cursor >= len(self._buffer):
            raise EOFError("BoundStream exhausted")
        value = self._buffer[self._cursor]
        self._cursor += 1
        return value

    def read_i8(self) -> int:
        raw = self.read_u8()
        return raw - 0x100 if raw & 0x80 else raw

    def read_u16(self) -> int:
        lo = self.read_u8()
        hi = self.read_u8()
        return (hi << 8) | lo

    def read_u24(self) -> int:
        lo = self.read_u8()
        mid = self.read_u8()
        hi = self.read_u8()
        return (hi << 16) | (mid << 8) | lo
