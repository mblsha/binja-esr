from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, Tuple

from .bind import DecodedInstr, IntAddrCalc, PreLatch

# Families that perform internal-memory accesses and therefore respond to PRE
# prefixes. These names mirror the `family` strings emitted by `decode_map`.
IMEM_FAMILIES = {
    "imem",
    "imem_ext",
    "imem_move",
    "imem_swap",
    "imem_const",
    "imem_logic",
    "alu_mem",
    "pmdf",
    "loop_move",
    "loop_arith",
    "loop_bcd",
    "incdec_imem",
    "jp_imem",
    "alu_a_mem",
    "cmp_imem",
    "cmpw",
    "cmpp",
    "cmp_mem_mem",
    "mv_reg",
    "ex_reg",
    "ext_reg",
    "adc_imem",
    "adc_a_mem",
    "adc_mem_a",
    "sbc_imem",
    "sbc_a_mem",
    "sbc_mem_a",
    "and_emem",
    "or_emem",
    "xor_emem",
    "mv_imem_from_ext",
    "mv_ext_from_imem",
    "mv_imem_reg",
    "mv_reg_imem",
    "mv_imem_word",
    "mv_imem_word_const",
    "mv_imem_long_const",
    "sub_imem",
    "ext_ptr",
}


@dataclass(frozen=True)
class PreMode:
    """Represents a single PRE opcode and the IMEM mappings it enforces."""

    opcode: int
    latch: PreLatch


_ROW_LABELS: Tuple[IntAddrCalc, ...] = (
    IntAddrCalc.N,
    IntAddrCalc.BP_N,
    IntAddrCalc.PX_N,
    IntAddrCalc.BP_PX,
)

_COL_LABELS: Tuple[IntAddrCalc, ...] = (
    IntAddrCalc.N,
    IntAddrCalc.BP_N,
    IntAddrCalc.PY_N,
    IntAddrCalc.BP_PY,
)

# This table defines the hexadecimal value of the PRE (Prefix) byte required for
# certain complex internal RAM addressing modes, based on the combination of
# addressing calculations used for the first and second address components.
# Rows indicate the addressing mode calculation specified by the first operand
# byte (e.g., in MV (m) (n), this corresponds to (m)).
# Columns indicate the addressing mode calculation specified by the second
# operand byte (e.g., in MV (m) (n), this corresponds to (n)).
#
# +---------------+-------+--------+--------+--------+
# | 1st op \ 2nd  | (n)   | (BP+n) | (PY+n) |(BP+PY) |
# +---------------+-------+--------+--------+--------+
# | (n)           | 32H   | 30H    | 33H    | 31H    |
# +---------------+-------+--------+--------+--------+
# | (BP+n)        | 22H   |        | 23H    | 21H    |
# +---------------+-------+--------+--------+--------+
# | (PX+n)        | 36H   | 34H    | 37H    | 35H    |
# +---------------+-------+--------+--------+--------+
# | (BP+PX)       | 26H   | 24H    | 27H    | 25H    |
# +---------------+-------+--------+--------+--------+
_PRE_OPCODE_MATRIX = {
    # First-op rows: (n), (BP+n), (PX+n), (BP+PX)
    # Second-op cols: (n), (BP+n), (PY+n), (BP+PY)
    (IntAddrCalc.N, IntAddrCalc.N): 0x32,
    (IntAddrCalc.N, IntAddrCalc.BP_N): 0x30,
    (IntAddrCalc.N, IntAddrCalc.PY_N): 0x33,
    (IntAddrCalc.N, IntAddrCalc.BP_PY): 0x31,
    (IntAddrCalc.BP_N, IntAddrCalc.N): 0x22,
    (IntAddrCalc.BP_N, IntAddrCalc.PY_N): 0x23,
    (IntAddrCalc.BP_N, IntAddrCalc.BP_PY): 0x21,
    (IntAddrCalc.PX_N, IntAddrCalc.N): 0x36,
    (IntAddrCalc.PX_N, IntAddrCalc.BP_N): 0x34,
    (IntAddrCalc.PX_N, IntAddrCalc.PY_N): 0x37,
    (IntAddrCalc.PX_N, IntAddrCalc.BP_PY): 0x35,
    (IntAddrCalc.BP_PX, IntAddrCalc.N): 0x26,
    (IntAddrCalc.BP_PX, IntAddrCalc.BP_N): 0x24,
    (IntAddrCalc.BP_PX, IntAddrCalc.PY_N): 0x27,
    (IntAddrCalc.BP_PX, IntAddrCalc.BP_PY): 0x25,
}


def _build_modes() -> Tuple[
    Dict[int, PreMode], Dict[Tuple[IntAddrCalc, IntAddrCalc], int]
]:
    by_opcode: Dict[int, PreMode] = {}
    by_pair: Dict[Tuple[IntAddrCalc, IntAddrCalc], int] = {}

    for (row, col), opcode in _PRE_OPCODE_MATRIX.items():
        if opcode == 0x20:
            # 0x20 is not a documented PRE prefix; skip reserving it as a mode.
            continue
        if opcode in by_opcode:
            raise ValueError(f"Duplicate PRE opcode {opcode:#x}")
        latch = PreLatch(row, col)
        by_opcode[opcode] = PreMode(opcode=opcode, latch=latch)
        by_pair[(row, col)] = opcode

    return by_opcode, by_pair


PRE_BY_OPCODE, PRE_BY_PAIR = _build_modes()


def prelatch_for_opcode(opcode: int) -> PreLatch:
    """Return the `PreLatch` associated with a PRE opcode byte."""

    mode = PRE_BY_OPCODE.get(opcode)
    if mode is None:
        raise ValueError(f"Unsupported PRE opcode {opcode:#x}")
    return mode.latch


def opcode_for_modes(first: IntAddrCalc, second: IntAddrCalc) -> int | None:
    """Return the PRE opcode byte for a given pair of IMEM addressing modes."""

    return PRE_BY_PAIR.get((first, second))


def iter_pre_modes() -> Iterable[PreMode]:
    """Yield all supported PRE modes sorted by opcode."""

    for opcode in sorted(PRE_BY_OPCODE):
        yield PRE_BY_OPCODE[opcode]


def iter_all_pre_variants(decoded: DecodedInstr) -> Iterable[DecodedInstr]:
    """Yield the decoded instruction with every PRE latch applied (plus baseline)."""

    base = replace(decoded, pre_applied=None)
    yield base
    if not needs_pre_variants(base):
        return
    for mode in iter_pre_modes():
        yield replace(base, pre_applied=mode.latch)


def needs_pre_variants(decoded: DecodedInstr) -> bool:
    """Return True if this instruction family uses IMEM and can be prefixed."""

    if decoded.family is None:
        return False
    return decoded.family in IMEM_FAMILIES
