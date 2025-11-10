from sc62015.decoding import decode_map
from sc62015.decoding.reader import StreamCtx
from sc62015.scil import from_decoded


def _decode(opcode: int, operand_bytes: bytes, pc: int = 0x1000):
    ctx = StreamCtx(pc=pc, data=operand_bytes, base_len=1)
    return decode_map.decode_opcode(opcode, ctx)


def test_mv_a_n_binder_maps_imm8() -> None:
    di = _decode(0x08, bytes([0x5A]))
    result = from_decoded.build(di)
    assert result.instr.name == "MV_A_IMM"
    assert result.binder["imm8"].value == 0x5A


def test_jrz_binder_sign_extension() -> None:
    di = _decode(0x19, bytes([0x03]))
    result = from_decoded.build(di)
    assert result.binder["disp8"].value == 0xFD


def test_jp_binder_splits_lo_and_page() -> None:
    di = _decode(0x02, bytes([0x34, 0x12]), pc=0x34567)
    result = from_decoded.build(di)
    assert result.binder["addr16"].value == 0x1234
    assert result.binder["page_hi"].value == 0x30000


def test_mv_abs24_binder_breaks_bytes() -> None:
    di = _decode(0x88, bytes([0x10, 0x20, 0x03]))
    result = from_decoded.build(di)
    assert result.binder["addr_ptr"].value == 0x032010


def test_mv_imem_store_binder_tracks_offset() -> None:
    di = _decode(0xA0, bytes([0x2A]))
    result = from_decoded.build(di)
    assert result.binder["imem_off"].value == 0x2A


def test_inc_uses_dynamic_reg() -> None:
    di = _decode(0x6C, bytes([0x00]))
    result = from_decoded.build(di)
    assert result.instr.name == "INC r"
