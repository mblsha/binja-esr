from sc62015.decoding import decode_map
from sc62015.decoding.reader import StreamCtx
from sc62015.scil import from_decoded


def _decode(opcode: int, operand_bytes: bytes, pc: int = 0x1000):
    ctx = StreamCtx(pc=pc, data=operand_bytes, base_len=1)
    return decode_map.decode_opcode(opcode, ctx)


def test_mv_a_n_binder_maps_imm8() -> None:
    di = _decode(0x08, bytes([0x5A]))
    instr, binder = from_decoded.build(di)
    assert instr.name == "MV_A_IMM"
    assert binder["imm8"].value == 0x5A


def test_jrz_binder_sign_extension() -> None:
    di = _decode(0x19, bytes([0x03]))
    _, binder = from_decoded.build(di)
    assert binder["disp8"].value == 0xFD


def test_jp_binder_splits_lo_and_page() -> None:
    di = _decode(0x02, bytes([0x34, 0x12]), pc=0x34567)
    _, binder = from_decoded.build(di)
    assert binder["addr16"].value == 0x1234
    assert binder["page_hi"].value == 0x30000


def test_mv_abs24_binder_breaks_bytes() -> None:
    di = _decode(0x88, bytes([0x10, 0x20, 0x03]))
    _, binder = from_decoded.build(di)
    assert binder["addr_lo"].value == 0x10
    assert binder["addr_mid"].value == 0x20
    assert binder["addr_hi"].value == 0x03
