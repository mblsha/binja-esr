"""
Typed decoding and binding helpers for SC62015 instruction bytes.

Phase 2 introduces strictly-typed operand binds that power SCIL builders.
"""

from .bind import (  # noqa: F401
    Addr16Page,
    Addr24,
    DecodedInstr,
    Disp8,
    Imm16,
    Imm24,
    Imm8,
    IntAddrCalc,
    PreLatch,
    RegSel,
)
from .reader import StreamCtx  # noqa: F401
from . import decode_map  # noqa: F401

__all__ = [
    "Addr16Page",
    "Addr24",
    "DecodedInstr",
    "Disp8",
    "Imm16",
    "Imm24",
    "Imm8",
    "IntAddrCalc",
    "PreLatch",
    "RegSel",
    "StreamCtx",
    "decode_map",
]
