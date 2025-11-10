from sc62015.decoding import decode_map
from sc62015.decoding.reader import StreamCtx
from sc62015.decoding.bind import Disp8


def test_decode_mv_a_n() -> None:
    ctx = StreamCtx(pc=0x1000, data=bytes([0x5A]), base_len=1)
    di = decode_map.decode_opcode(0x08, ctx)
    assert di.length == 2
    assert di.mnemonic == "MV A,n"
    assert di.binds["n"].value == 0x5A


def test_decode_jrz_plus_and_minus() -> None:
    ctx_plus = StreamCtx(pc=0x2000, data=bytes([0x04]), base_len=1)
    di_plus = decode_map.decode_opcode(0x18, ctx_plus)
    assert isinstance(di_plus.binds["disp"], Disp8)
    assert di_plus.binds["disp"].value == 4

    ctx_minus = StreamCtx(pc=0x2000, data=bytes([0x03]), base_len=1)
    di_minus = decode_map.decode_opcode(0x19, ctx_minus)
    assert di_minus.binds["disp"].value == -3


def test_decode_jp_mn_tracks_page() -> None:
    ctx = StreamCtx(pc=0x34567, data=bytes([0x34, 0x12]), base_len=1)
    di = decode_map.decode_opcode(0x02, ctx)
    addr = di.binds["addr16_page"]
    assert addr.offs16.u16 == 0x1234
    assert addr.page20 == 0x30000
    assert di.length == 3


def test_decode_mv_a_abs24() -> None:
    ctx = StreamCtx(pc=0x010000, data=bytes([0x10, 0x20, 0x03]), base_len=1)
    di = decode_map.decode_opcode(0x88, ctx)
    addr = di.binds["addr24"]
    assert addr.v.u24 == 0x032010
    assert di.length == 4


def test_decode_pre_32_sets_latch() -> None:
    ctx = StreamCtx(pc=0x4000, data=b"", base_len=1)
    di = decode_map.decode_opcode(0x32, ctx)
    assert di.length == 1
    assert di.pre_latch is not None
    assert di.pre_latch.first == "(n)"
    assert di.pre_latch.second == "(n)"
