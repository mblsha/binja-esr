from __future__ import annotations

from typing import Iterable, Optional, Tuple

from binaryninja import RegisterName  # type: ignore

from .validate import bits_to_bytes


class CompatLLILBuilder:
    """Utility helpers that reproduce the LLIL shapes emitted by the legacy lifter."""

    def __init__(self, il) -> None:
        self.il = il

    # ------------------------------------------------------------------
    # Address helpers

    def paged_join(self, page_expr: int, offs_expr: int) -> int:
        """Join a 20-bit page with a 16-bit offset via OR to match legacy JP."""
        width_bytes = bits_to_bytes(20)
        hi = self.il.zero_extend(width_bytes, page_expr)
        lo = self.il.zero_extend(width_bytes, offs_expr)
        return self.il.or_expr(width_bytes, hi, lo)

    def pc_relative(self, base_advance: int, disp_expr: Optional[int]) -> int:
        width_bytes = bits_to_bytes(20)
        pc = self.il.reg(width_bytes, RegisterName("PC"))
        fallthrough = self.il.add(width_bytes, pc, self.il.const(width_bytes, base_advance))
        if disp_expr is None:
            return fallthrough
        return self.il.add(width_bytes, fallthrough, disp_expr)

    # ------------------------------------------------------------------
    # Register helpers

    def set_reg_with_flags(
        self,
        reg_name: str,
        width_bits: int,
        value_expr: int,
        flags: Optional[Tuple[str, ...]] = None,
    ) -> None:
        width_bytes = bits_to_bytes(width_bits)
        self.il.append(self.il.set_reg(width_bytes, RegisterName(reg_name), value_expr))
        if flags:
            self.il.append(self.il.nop())
