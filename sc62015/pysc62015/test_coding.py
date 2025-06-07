from .coding import Decoder, Encoder, FetchDecoder, BufferTooShort
from .constants import ADDRESS_SPACE_SIZE
import pytest


def test_decoder() -> None:
    # Test decoding unsigned byte
    decoder = Decoder(bytearray([0x01, 0x02, 0x03, 0x00]))
    assert decoder.unsigned_byte() == 0x01
    assert decoder.unsigned_byte() == 0x02
    assert decoder.unsigned_byte() == 0x03
    assert decoder.unsigned_byte() == 0x00

    # Test decoding unsigned word
    decoder = Decoder(bytearray([0xAA, 0xBB, 0xCC, 0xDD]))
    assert decoder.unsigned_word_le() == 0xBBAA
    assert decoder.unsigned_word_le() == 0xDDCC


def test_fetchdecoder() -> None:
    # Mock read_mem function
    def read_mem(addr: int) -> int:
        memory = bytearray([0x01, 0x02, 0x03, 0x04, 0x05, 0x06])
        if addr < len(memory):
            return memory[addr]
        raise IndexError("Address out of bounds")

    decoder = FetchDecoder(read_mem)

    # Test fetching bytes
    assert decoder.peek(0) == 0x01
    assert decoder.peek(1) == 0x02
    assert decoder.peek(2) == 0x03
    assert decoder.peek(3) == 0x04

    # Test decoding unsigned byte
    assert decoder.unsigned_byte() == 0x01
    assert decoder.unsigned_byte() == 0x02

    # Test decoding unsigned word
    assert decoder.unsigned_word_le() == 0x0403


def test_fetchdecoder_bounds() -> None:
    memory = bytearray(range(256)) + bytearray([0] * (ADDRESS_SPACE_SIZE - 256))

    def read_mem(addr: int) -> int:
        if 0 <= addr < len(memory):
            return memory[addr]
        raise IndexError("Address out of bounds")

    decoder = FetchDecoder(read_mem)

    # reading last byte succeeds
    decoder.pos = ADDRESS_SPACE_SIZE - 1
    assert decoder.peek(0) == memory[-1]
    with pytest.raises(BufferTooShort):
        decoder.peek(1)

    decoder.pos = ADDRESS_SPACE_SIZE - 1
    assert decoder.unsigned_byte() == memory[-1]
    decoder.pos = ADDRESS_SPACE_SIZE - 2
    # word read exactly at boundary succeeds
    assert decoder.unsigned_word_le() == int.from_bytes(memory[-2:], "little")
    decoder.pos = ADDRESS_SPACE_SIZE - 1
    with pytest.raises(BufferTooShort):
        decoder.unsigned_word_le()


def test_encoder() -> None:
    # Test encoding unsigned byte
    encoder = Encoder()
    encoder.unsigned_byte(0x01)
    encoder.unsigned_byte(0x02)
    encoder.unsigned_byte(0x03)
    encoder.unsigned_byte(0x00)
    assert encoder.buf == bytearray([0x01, 0x02, 0x03, 0x00])

    # Test encoding unsigned word
    encoder = Encoder()
    encoder.unsigned_word_le(0xBBAA)
    encoder.unsigned_word_le(0xDDCC)
    assert encoder.buf == bytearray([0xAA, 0xBB, 0xCC, 0xDD])

