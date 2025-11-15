from sc62015.decoding.bind import IntAddrCalc
from sc62015.decoding.decode_map import decode_opcode
from sc62015.decoding.pre_modes import iter_all_pre_variants
from sc62015.decoding.reader import StreamCtx


def _decode(opcode: int, operands: bytes = b""):
    return decode_opcode(opcode, StreamCtx(pc=0x1000, data=operands, base_len=1))


def test_iter_all_pre_variants_includes_default() -> None:
    decoded = _decode(0x80, bytes([0x42]))  # MV A,(n)
    variants = list(iter_all_pre_variants(decoded))
    assert len(variants) == 16  # default + 15 documented PREs
    assert variants[0].pre_applied is None


def test_iter_all_pre_variants_cover_all_modes() -> None:
    decoded = _decode(0x80, bytes([0x10]))
    modes = {
        (variant.pre_applied.first, variant.pre_applied.second)
        for variant in iter_all_pre_variants(decoded)
        if variant.pre_applied
    }
    assert (IntAddrCalc.N, IntAddrCalc.PY_N) in modes
    assert (IntAddrCalc.BP_PX, IntAddrCalc.BP_PY) in modes


def test_non_imem_variants_no_expansion() -> None:
    decoded = _decode(0x08, bytes([0x01]))  # MV A,n (imm8)
    variants = list(iter_all_pre_variants(decoded))
    assert len(variants) == 1
    assert variants[0].pre_applied is None
