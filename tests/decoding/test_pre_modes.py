from sc62015.decoding import decode_map
from sc62015.decoding.bind import IntAddrCalc
from sc62015.decoding.pre_modes import (
    iter_all_pre_variants,
    iter_pre_modes,
    needs_pre_variants,
    opcode_for_modes,
    prelatch_for_opcode,
)
from sc62015.decoding.reader import StreamCtx


def _decode(opcode: int, operands: bytes):
    return decode_map.decode_opcode(
        opcode, StreamCtx(pc=0x1000, data=operands, base_len=1)
    )


def test_all_documented_pre_modes_present() -> None:
    modes = list(iter_pre_modes())
    assert len(modes) == 15
    for mode in modes:
        assert mode.opcode == opcode_for_modes(mode.latch.first, mode.latch.second)


def test_roundtrip_lookup() -> None:
    for mode in iter_pre_modes():
        latch = prelatch_for_opcode(mode.opcode)
        assert latch is mode.latch


def test_single_operand_lookup_matches_lookup_table() -> None:
    assert opcode_for_modes(IntAddrCalc.BP_N, IntAddrCalc.N) == 0x22
    assert opcode_for_modes(IntAddrCalc.BP_PX, IntAddrCalc.BP_PY) == 0x25
    assert opcode_for_modes(IntAddrCalc.N, IntAddrCalc.PY_N) == 0x33
    assert opcode_for_modes(IntAddrCalc.PX_N, IntAddrCalc.BP_PY) == 0x35
    assert opcode_for_modes(IntAddrCalc.PX_N, IntAddrCalc.BP_N) == 0x34


def test_needs_pre_variants_detects_imem_family() -> None:
    imem = _decode(0x80, bytes([0x10]))
    assert needs_pre_variants(imem)


def test_needs_pre_variants_false_for_non_imem() -> None:
    imm = _decode(0x08, bytes([0x01]))
    assert not needs_pre_variants(imm)


def test_decode_with_pre_variants_matches_helper() -> None:
    variants = decode_map.decode_with_pre_variants(
        0x80, StreamCtx(pc=0x1000, data=bytes([0x02]), base_len=1)
    )
    helper = tuple(iter_all_pre_variants(_decode(0x80, bytes([0x02]))))
    assert variants == helper
