from __future__ import annotations

from typing import Optional, Set, Tuple

from binaryninja.lowlevelil import LowLevelILFunction  # type: ignore

from .compat_il import emit_instruction
from .decode_map import decode_opcode
from .reader import StreamCtx
from .bind import DecodedInstr


class CompatDispatcher:
    """Lightweight bridge that uses the SCIL-friendly decoder for pilot opcodes."""

    def __init__(self) -> None:
        self.pilot_opcodes: Set[int] = {0x02, 0x08, 0x18, 0x19, 0x32, 0x88}
        self._pending_pre = None

    @property
    def pending_pre(self):
        return self._pending_pre

    def try_emit(
        self, data: bytes, addr: int, il: LowLevelILFunction
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

        emit_instruction(decoded, il, addr)
        return decoded.length, decoded
