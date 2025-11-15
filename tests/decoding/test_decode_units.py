from sc62015.decoding import decode_map
from sc62015.decoding.reader import StreamCtx
from sc62015.decoding.bind import Imm8, IntAddrCalc


def test_decode_mv_a_n() -> None:
    ctx = StreamCtx(pc=0x1000, data=bytes([0x5A]), base_len=1)
    di = decode_map.decode_opcode(0x08, ctx)
    assert di.length == 2
    assert di.mnemonic == "MV A,n"
    assert di.binds["n"].value == 0x5A


def test_decode_jrz_plus_and_minus() -> None:
    ctx_plus = StreamCtx(pc=0x2000, data=bytes([0x04]), base_len=1)
    di_plus = decode_map.decode_opcode(0x18, ctx_plus)
    assert isinstance(di_plus.binds["disp"], Imm8)
    assert di_plus.binds["disp"].value == 0x04

    ctx_minus = StreamCtx(pc=0x2000, data=bytes([0x03]), base_len=1)
    di_minus = decode_map.decode_opcode(0x19, ctx_minus)
    assert di_minus.binds["disp"].value == 0x03


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
    assert di.pre_latch.first is IntAddrCalc.N
    assert di.pre_latch.second is IntAddrCalc.N


def test_decode_add_imm_sets_family() -> None:
    ctx = StreamCtx(pc=0x1111, data=bytes([0x33]), base_len=1)
    di = decode_map.decode_opcode(0x40, ctx)
    assert di.mnemonic == "ADD A,n"
    assert di.family == "imm8"
    assert di.binds["n"].value == 0x33


def test_decode_jrnz_tracks_cond() -> None:
    ctx = StreamCtx(pc=0x2000, data=bytes([0x02]), base_len=1)
    di = decode_map.decode_opcode(0x1A, ctx)
    assert di.binds["cond"] == "NZ"
    assert di.binds["disp"].value == 2


def test_decode_jp_cond_records_cond() -> None:
    ctx = StreamCtx(pc=0x34567, data=bytes([0xAA, 0x55]), base_len=1)
    di = decode_map.decode_opcode(0x16, ctx)
    assert di.binds["cond"] == "C"
    assert di.binds["addr16_page"].offs16.u16 == 0x55AA


def test_decode_inc_r_only_r1_supported() -> None:
    ctx = StreamCtx(pc=0x5000, data=bytes([0x00]), base_len=1)
    di = decode_map.decode_opcode(0x6C, ctx)
    assert di.binds["reg"].name == "A"
    assert di.family == "incdec"


def test_decode_internal_moves_capture_offset() -> None:
    ctx = StreamCtx(pc=0x4000, data=bytes([0x44]), base_len=1)
    di = decode_map.decode_opcode(0x80, ctx)
    assert di.mnemonic == "MV A,(n)"
    assert di.binds["n"].value == 0x44


def test_decode_jp_imem_pointer() -> None:
    ctx = StreamCtx(pc=0x6000, data=bytes([0x21]), base_len=1)
    di = decode_map.decode_opcode(0x10, ctx)
    assert di.mnemonic == "JP (n)"
    assert di.binds["n"].value == 0x21


def test_decode_jp_r3_register() -> None:
    ctx = StreamCtx(pc=0x6000, data=bytes([0x04]), base_len=1)
    di = decode_map.decode_opcode(0x11, ctx)
    assert di.mnemonic == "JP r3"
    assert di.binds["reg"].name == "X"


def test_decode_ext_ptr_load_positive_offset() -> None:
    ctx = StreamCtx(pc=0x7000, data=bytes([0x80, 0x05, 0x03]), base_len=1)
    di = decode_map.decode_opcode(0x98, ctx)
    ptr = di.binds["ptr"]
    assert ptr.mode == "pos"
    assert ptr.base.value == 0x05
    assert ptr.disp.value == 3


def test_decode_imem_move_width_bits() -> None:
    ctx = StreamCtx(pc=0x8000, data=bytes([0x10, 0x20]), base_len=1)
    di = decode_map.decode_opcode(0xC9, ctx)
    assert di.binds["width"] == 16


def test_decode_imem_swap_fields() -> None:
    ctx = StreamCtx(pc=0x8100, data=bytes([0x01, 0x02]), base_len=1)
    di = decode_map.decode_opcode(0xC0, ctx)
    assert di.binds["width"] == 8
    assert di.binds["left"].value == 0x01
    assert di.binds["right"].value == 0x02


def test_decode_imem_from_ext_loads_ptr_and_offset() -> None:
    ctx = StreamCtx(pc=0x8200, data=bytes([0x04, 0x10]), base_len=1)
    di = decode_map.decode_opcode(0xE0, ctx)
    assert di.binds["width"] == 8
    assert di.binds["imem"].value == 0x10


def test_decode_ext_from_imem_loads_ptr_and_offset() -> None:
    ctx = StreamCtx(pc=0x8200, data=bytes([0x04, 0x22]), base_len=1)
    di = decode_map.decode_opcode(0xE8, ctx)
    assert di.binds["width"] == 8
    assert di.binds["imem"].value == 0x22
