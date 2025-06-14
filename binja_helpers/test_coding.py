from binja_helpers.binja_helpers.coding import Decoder, Encoder, FetchDecoder, BufferTooShort
import struct
from sc62015.pysc62015.constants import ADDRESS_SPACE_SIZE
import pytest


def test_decoder() -> None:
    decoder = Decoder(bytearray([0x01, 0x02, 0x03, 0x00]))
    assert decoder.unsigned_byte() == 0x01
    assert decoder.unsigned_byte() == 0x02
    assert decoder.unsigned_byte() == 0x03
    assert decoder.unsigned_byte() == 0x00

    decoder = Decoder(bytearray([0xAA, 0xBB, 0xCC, 0xDD]))
    assert decoder.unsigned_word_le() == 0xBBAA
    assert decoder.unsigned_word_le() == 0xDDCC


def test_fetchdecoder() -> None:
    def read_mem(addr: int) -> int:
        memory = bytearray([0x01, 0x02, 0x03, 0x04, 0x05, 0x06])
        if addr < len(memory):
            return memory[addr]
        raise IndexError("Address out of bounds")

    decoder = FetchDecoder(read_mem, ADDRESS_SPACE_SIZE)

    assert decoder.peek(0) == 0x01
    assert decoder.peek(1) == 0x02
    assert decoder.peek(2) == 0x03
    assert decoder.peek(3) == 0x04

    assert decoder.unsigned_byte() == 0x01
    assert decoder.unsigned_byte() == 0x02
    assert decoder.unsigned_word_le() == 0x0403


def test_fetchdecoder_bounds() -> None:
    memory = bytearray(range(256)) + bytearray([0] * (ADDRESS_SPACE_SIZE - 256))

    def read_mem(addr: int) -> int:
        if 0 <= addr < len(memory):
            return memory[addr]
        raise IndexError("Address out of bounds")

    decoder = FetchDecoder(read_mem, ADDRESS_SPACE_SIZE)

    decoder.pos = ADDRESS_SPACE_SIZE - 1
    assert decoder.peek(0) == memory[-1]
    with pytest.raises(BufferTooShort):
        decoder.peek(1)

    decoder.pos = ADDRESS_SPACE_SIZE - 1
    assert decoder.unsigned_byte() == memory[-1]
    decoder.pos = ADDRESS_SPACE_SIZE - 2
    assert decoder.unsigned_word_le() == int.from_bytes(memory[-2:], "little")
    decoder.pos = ADDRESS_SPACE_SIZE - 1
    with pytest.raises(BufferTooShort):
        decoder.unsigned_word_le()


def test_encoder() -> None:
    encoder = Encoder()
    encoder.unsigned_byte(0x01)
    encoder.unsigned_byte(0x02)
    encoder.unsigned_byte(0x03)
    encoder.unsigned_byte(0x00)
    assert encoder.buf == bytearray([0x01, 0x02, 0x03, 0x00])

    encoder = Encoder()
    encoder.unsigned_word_le(0xBBAA)
    encoder.unsigned_word_le(0xDDCC)
    assert encoder.buf == bytearray([0xAA, 0xBB, 0xCC, 0xDD])


def test_decoder_errors() -> None:
    decoder = Decoder(bytearray([0x11]))
    assert decoder.unsigned_byte() == 0x11
    with pytest.raises(BufferTooShort):
        decoder.unsigned_byte()

    decoder = Decoder(bytearray([0x22]))
    with pytest.raises(BufferTooShort):
        decoder.unsigned_word_le()


def test_fetchdecoder_unpack_boundary() -> None:
    memory = bytearray(range(256)) + bytearray([0] * (ADDRESS_SPACE_SIZE - 256))

    def read_mem(addr: int) -> int:
        if 0 <= addr < len(memory):
            return memory[addr]
        raise IndexError("Address out of bounds")

    decoder = FetchDecoder(read_mem, ADDRESS_SPACE_SIZE)
    decoder.pos = ADDRESS_SPACE_SIZE - 1
    with pytest.raises(BufferTooShort):
        decoder._unpack("H")


def test_fetchdecoder_read_error_propagation() -> None:
    def read_mem(addr: int) -> int:
        raise RuntimeError("read failed")

    decoder = FetchDecoder(read_mem, ADDRESS_SPACE_SIZE)
    with pytest.raises(RuntimeError):
        decoder.unsigned_byte()


def test_encoder_value_too_large() -> None:
    encoder = Encoder()
    with pytest.raises(struct.error):
        encoder.unsigned_byte(256)
