from __future__ import annotations

from typing import Optional, Tuple

from binja_test_mocks.eval_llil import (
    EVAL_LLIL,
    FlagGetter,
    FlagSetter,
    Memory,
    RegistersLike,
    ResultFlags,
    State,
    evaluate_llil,
)
from binja_test_mocks.mock_llil import MockLLIL


def _operand_width(llil: MockLLIL, fallback: Optional[int]) -> int:
    width = getattr(llil, "width", None)
    if callable(width):
        maybe = width()
        if maybe:
            return maybe
    return fallback or 1


def _eval_sx(
    llil: MockLLIL,
    size: Optional[int],
    regs: RegistersLike,
    memory: Memory,
    state: State,
    get_flag: FlagGetter,
    set_flag: FlagSetter,
) -> Tuple[int, Optional[ResultFlags]]:
    dest_size = size or llil.width() or 1
    inner = llil.ops[0]
    src_size = _operand_width(inner, dest_size)
    value, _ = evaluate_llil(inner, regs, memory, state, get_flag, set_flag)
    value = int(value or 0)
    src_bits = src_size * 8
    dest_bits = dest_size * 8
    mask = (1 << src_bits) - 1
    value &= mask
    sign_bit = 1 << (src_bits - 1)
    if value & sign_bit:
        value -= 1 << src_bits
    value &= (1 << dest_bits) - 1
    return value, None


def _eval_zx(
    llil: MockLLIL,
    size: Optional[int],
    regs: RegistersLike,
    memory: Memory,
    state: State,
    get_flag: FlagGetter,
    set_flag: FlagSetter,
) -> Tuple[int, Optional[ResultFlags]]:
    dest_size = size or llil.width() or 1
    inner = llil.ops[0]
    value, _ = evaluate_llil(inner, regs, memory, state, get_flag, set_flag)
    value = int(value or 0)
    dest_bits = dest_size * 8
    value &= (1 << dest_bits) - 1
    return value, None


def _eval_low_part(
    llil: MockLLIL,
    size: Optional[int],
    regs: RegistersLike,
    memory: Memory,
    state: State,
    get_flag: FlagGetter,
    set_flag: FlagSetter,
) -> Tuple[int, Optional[ResultFlags]]:
    target_size = size or llil.width() or 1
    value, _ = evaluate_llil(llil.ops[0], regs, memory, state, get_flag, set_flag)
    value = int(value or 0)
    mask = (1 << (target_size * 8)) - 1
    return value & mask, None


def _eval_high_part(
    llil: MockLLIL,
    size: Optional[int],
    regs: RegistersLike,
    memory: Memory,
    state: State,
    get_flag: FlagGetter,
    set_flag: FlagSetter,
) -> Tuple[int, Optional[ResultFlags]]:
    target_size = size or llil.width() or 1
    inner = llil.ops[0]
    value, _ = evaluate_llil(inner, regs, memory, state, get_flag, set_flag)
    value = int(value or 0)
    src_size = _operand_width(inner, target_size)
    shift = max(0, (src_size - target_size) * 8)
    result = (value >> shift) & ((1 << (target_size * 8)) - 1)
    return result, None


def _ensure_patch(name: str, func) -> None:
    if name not in EVAL_LLIL:
        EVAL_LLIL[name] = func


_ensure_patch("SX", _eval_sx)
_ensure_patch("ZX", _eval_zx)
_ensure_patch("LOW_PART", _eval_low_part)
_ensure_patch("HIGH_PART", _eval_high_part)
