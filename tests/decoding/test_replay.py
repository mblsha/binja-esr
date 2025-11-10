from sc62015.decoding import decode_map
from sc62015.decoding.reader import StreamCtx
from sc62015.decoding.replay import BoundStream


def _decode(opcode: int, operands: bytes, pc: int = 0x1000) -> BoundStream:
    ctx = StreamCtx(pc=pc, data=operands, base_len=1)
    di = decode_map.decode_opcode(opcode, ctx)
    return BoundStream(di)


def test_bound_stream_mv_a_n_matches_raw_byte() -> None:
    stream = _decode(0x08, bytes([0xAA]))
    assert stream.read_u8() == 0xAA


def test_bound_stream_jrz_minus_replays_magnitude() -> None:
    stream = _decode(0x19, bytes([0x05]))
    assert stream.read_u8() == 0x05


def test_bound_stream_jp_reads_little_endian() -> None:
    stream = _decode(0x02, bytes([0x34, 0x12]), pc=0x50000)
    assert stream.read_u16() == 0x1234


def test_bound_stream_mv_a_abs24_reads_three_bytes() -> None:
    stream = _decode(0x88, bytes([0x10, 0x20, 0x03]))
    assert stream.read_u24() == 0x032010
