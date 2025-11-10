from __future__ import annotations

from typing import Optional, Set, Tuple

from .decode_map import PRE_LATCHES, decode_opcode
from .reader import StreamCtx
from .bind import DecodedInstr


class CompatDispatcher:
    """Lightweight bridge that uses the SCIL-friendly decoder for pilot opcodes."""

    def __init__(self) -> None:
        self.pilot_opcodes: Set[int] = {
            0x01,
            0x02,
            0x08,
            0x04,
            0x05,
            0x06,
            0x07,
            0x10,
            0x11,
            0x14,
            0x15,
            0x16,
            0x17,
            0x18,
            0x19,
            0x1A,
            0x1B,
            0x1C,
            0x1D,
            0x1E,
            0x1F,
            0x32,
            0x40,
            0x47,
            0x48,
            0x28,
            0x29,
            0x2A,
            0x2B,
            0x2C,
            0x2D,
            0x2E,
            0x2F,
            0x38,
            0x39,
            0x3A,
            0x3B,
            0x3C,
            0x3D,
            0x3E,
            0x3F,
            0x4F,
            0x50,
            0x57,
            0x54,
            0x55,
            0x58,
            0x5C,
            0x5D,
            0x5F,
            0x64,
            0x68,
            0x70,
            0x78,
            0x6C,
            0x7C,
            0x80,
            0x88,
            0xA0,
            0xA8,
            0x90,
            0x91,
            0x92,
            0x93,
            0x94,
            0x95,
            0x96,
            0x98,
            0x99,
            0x9A,
            0x9B,
            0x9C,
            0x9D,
            0x9E,
            0xB0,
            0xB1,
            0xB2,
            0xB3,
            0xB4,
            0xB5,
            0xB6,
            0xB8,
            0xB9,
            0xBA,
            0xBB,
            0xBC,
            0xBD,
            0xBE,
            0xC0,
            0xC1,
            0xC2,
            0xC4,
            0xC5,
            0xCB,
            0xC8,
            0xC9,
            0xCA,
            0xCF,
            0xD4,
            0xD5,
            0xEC,
            0xFC,
            0xE0,
            0xE1,
            0xE2,
            0xE8,
            0xE9,
            0xEA,
            0xDE,
            0xDF,
            0xEF,
            0xFE,
            0xFF,
        }
        self.pilot_opcodes.update(PRE_LATCHES.keys())
        self._pending_pre = None

    @property
    def pending_pre(self):
        return self._pending_pre

    def try_decode(
        self, data: bytes, addr: int
    ) -> Optional[Tuple[int, Optional[DecodedInstr]]]:
        if not data:
            return None

        opcode = data[0]
        if opcode not in self.pilot_opcodes:
            return None

        ctx = StreamCtx(pc=addr, data=data[1:], base_len=1)
        try:
            decoded = decode_opcode(opcode, ctx)
        except Exception:
            return None

        if decoded.pre_latch is not None:
            self._pending_pre = decoded.pre_latch
            return decoded.length, None

        object.__setattr__(decoded, "pre_applied", self._pending_pre)
        self._pending_pre = None

        return decoded.length, decoded
