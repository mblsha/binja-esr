from .coding import Decoder, Encoder, FetchDecoder


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
