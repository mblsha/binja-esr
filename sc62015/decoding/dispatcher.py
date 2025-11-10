from __future__ import annotations

from typing import Optional, Tuple

from .decode_map import decode_opcode
from .reader import StreamCtx
from .bind import DecodedInstr


class CompatDispatcher:
    """Lightweight bridge that uses the SCIL-friendly decoder for pilot opcodes."""

    def __init__(self) -> None:
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
