from __future__ import annotations

from typing import Optional, Tuple

from binaryninja import FlagName, RegisterName  # type: ignore

from ..pysc62015.constants import INTERNAL_MEMORY_START
from ..pysc62015.instr.opcodes import IMEMRegisters, TempIncDecHelper
from .validate import bits_to_bytes


class CompatLLILBuilder:
    """Utility helpers that reproduce the LLIL shapes emitted by the legacy lifter."""

    def __init__(self, il) -> None:
        self.il = il
        self._addr_mask = (1 << 24) - 1
        self._mode_simple = 0
        self._mode_post_inc = 1
        self._mode_pre_dec = 2
        self._mode_offset = 3

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
        fallthrough = self.il.add(
            width_bytes, pc, self.il.const(width_bytes, base_advance)
        )
        if disp_expr is None:
            return fallthrough
        return self.il.add(width_bytes, fallthrough, disp_expr)

    def _resolve_ptr_mode(self, mode: str) -> int:
        mapping = {
            "simple": self._mode_simple,
            "post_inc": self._mode_post_inc,
            "pre_dec": self._mode_pre_dec,
            "offset": self._mode_offset,
        }
        if mode not in mapping:
            raise ValueError(f"Unsupported pointer mode {mode}")
        return mapping[mode]

    def _ext_ptr_address(self, ptr_name: str, mode: int, width: int, disp: int) -> int:
        ptr_reg = RegisterName(ptr_name)
        ptr_expr = self.il.reg(bits_to_bytes(24), ptr_reg)
        if mode == self._mode_simple:
            base = ptr_expr
        elif mode == self._mode_offset:
            base = self.il.add(
                bits_to_bytes(24),
                ptr_expr,
                self.il.const(bits_to_bytes(24), disp & self._addr_mask),
            )
        elif mode == self._mode_post_inc:
            self.il.append(
                self.il.set_reg(bits_to_bytes(24), TempIncDecHelper, ptr_expr)
            )
            self.il.append(
                self.il.set_reg(
                    bits_to_bytes(24),
                    ptr_reg,
                    self.il.add(
                        bits_to_bytes(24),
                        ptr_expr,
                        self.il.const(bits_to_bytes(24), width),
                    ),
                )
            )
            base = self.il.reg(bits_to_bytes(24), TempIncDecHelper)
        elif mode == self._mode_pre_dec:
            new_value = self.il.sub(
                bits_to_bytes(24), ptr_expr, self.il.const(bits_to_bytes(24), width)
            )
            self.il.append(
                self.il.set_reg(bits_to_bytes(24), TempIncDecHelper, new_value)
            )
            self.il.append(self.il.set_reg(bits_to_bytes(24), ptr_reg, new_value))
            base = self.il.reg(bits_to_bytes(24), TempIncDecHelper)
        else:
            raise ValueError(f"Unsupported pointer mode {mode}")
        return base

    def ext_reg_read_value(self, ptr_name: str, mode: str, width_bits: int, disp: int):
        mode_code = self._resolve_ptr_mode(mode)
        width_bytes = bits_to_bytes(width_bits)
        addr = self._ext_ptr_address(ptr_name, mode_code, width_bytes, disp)
        return self.il.load(width_bytes, addr)

    def ext_reg_store_value(
        self, ptr_name: str, mode: str, width_bits: int, disp: int, value
    ) -> None:
        mode_code = self._resolve_ptr_mode(mode)
        width_bytes = bits_to_bytes(width_bits)
        addr = self._ext_ptr_address(ptr_name, mode_code, width_bytes, disp)
        self.il.append(self.il.store(width_bytes, addr, value))

    def ext_reg_load(
        self, dst_name: str, dst_bits: int, ptr_name: str, mode: str, disp: int
    ) -> None:
        width_bytes = bits_to_bytes(dst_bits)
        value = self.ext_reg_read_value(ptr_name, mode, dst_bits, disp)
        self.il.append(self.il.set_reg(width_bytes, RegisterName(dst_name), value))

    def ext_reg_store(
        self, src_name: str, src_bits: int, ptr_name: str, mode: str, disp: int
    ) -> None:
        width_bytes = bits_to_bytes(src_bits)
        value = self.il.reg(width_bytes, RegisterName(src_name))
        self.ext_reg_store_value(ptr_name, mode, src_bits, disp, value)

    # ------------------------------------------------------------------
    # Internal memory helpers

    def _imem_reg(self, name: str) -> int:
        addr = INTERNAL_MEMORY_START + IMEMRegisters[name]
        return self.il.load(1, self.il.const_pointer(bits_to_bytes(24), addr))

    def _internal_base(self) -> int:
        return self.il.const(bits_to_bytes(24), INTERNAL_MEMORY_START)

    def imem_address(self, mode: str, offset_expr: int) -> int:
        """Reproduce IMemHelper tree for the requested addressing mode."""
        if mode not in {
            "(n)",
            "(BP+n)",
            "(PX+n)",
            "(PY+n)",
            "(BP+PX)",
            "(BP+PY)",
        }:
            raise ValueError(f"Unsupported IMEM addressing mode {mode}")

        def _add_byte(lhs: int, rhs: int) -> int:
            return self.il.add(1, lhs, rhs)

        if mode == "(n)":
            offset = offset_expr
        elif mode == "(BP+n)":
            offset = _add_byte(self._imem_reg("BP"), offset_expr)
        elif mode == "(PX+n)":
            offset = _add_byte(self._imem_reg("PX"), offset_expr)
        elif mode == "(PY+n)":
            offset = _add_byte(self._imem_reg("PY"), offset_expr)
        elif mode == "(BP+PX)":
            offset = _add_byte(self._imem_reg("BP"), self._imem_reg("PX"))
        else:  # (BP+PY)
            offset = _add_byte(self._imem_reg("BP"), self._imem_reg("PY"))

        return self.il.add(bits_to_bytes(24), offset, self._internal_base())

    # ------------------------------------------------------------------
    # Arithmetic helpers

    @staticmethod
    def _flag_name(flags: Optional[Tuple[str, ...]]) -> Optional[FlagName]:
        if not flags:
            return None
        return FlagName("".join(flags))

    def binop_with_flags(
        self,
        op: str,
        width_bits: int,
        lhs: int,
        rhs: int,
        flags: Optional[Tuple[str, ...]],
    ) -> int:
        width_bytes = bits_to_bytes(width_bits)
        bn_flags = self._flag_name(flags)
        mapping = {
            "add": self.il.add,
            "sub": self.il.sub,
            "and": self.il.and_expr,
            "or": self.il.or_expr,
            "xor": self.il.xor_expr,
        }
        if op not in mapping:
            raise NotImplementedError(f"Flagged op {op} not supported")
        return mapping[op](width_bytes, lhs, rhs, flags=bn_flags)

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
