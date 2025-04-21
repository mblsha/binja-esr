from .coding import Decoder, Encoder


def test_decoder():
    # Test decoding unsigned byte
    decoder = Decoder(bytearray([0x01, 0x02, 0x03, 0x00]))
    assert decoder.unsigned_byte() == 0x01
    assert decoder.unsigned_byte() == 0x02
    assert decoder.unsigned_byte() == 0x03
    assert decoder.unsigned_byte() == 0x00

    # Test decoding signed byte
    decoder = Decoder(bytearray([0xFF, 0x7F, 0x80, 0x00]))
    assert decoder.signed_byte() == -1
    assert decoder.signed_byte() == 127
    assert decoder.signed_byte() == -128
    assert decoder.signed_byte() == 0

    # Test decoding unsigned word
    decoder = Decoder(bytearray([0xAA, 0xBB, 0xCC, 0xDD]))
    assert decoder.unsigned_word_le() == 0xBBAA
    assert decoder.unsigned_word_le() == 0xDDCC

    # Test decoding unsigned word big-endian
    decoder = Decoder(bytearray([0xAA, 0xBB, 0xCC, 0xDD]))
    assert decoder.unsigned_word_be() == 0xAABB
    assert decoder.unsigned_word_be() == 0xCCDD

    # Test decoding signed word
    decoder = Decoder(bytearray([0xFF, 0xFF, 0x00, 0x80]))
    assert decoder.signed_word_le() == -1
    assert decoder.signed_word_le() == -32768

    # Test decoding signed word big-endian
    decoder = Decoder(bytearray([0xFF, 0xFF, 0x80, 0x00]))
    assert decoder.signed_word_be() == -1
    assert decoder.signed_word_be() == -32768


def test_encoder():
    # Test encoding unsigned byte
    encoder = Encoder()
    encoder.unsigned_byte(0x01)
    encoder.unsigned_byte(0x02)
    encoder.unsigned_byte(0x03)
    encoder.unsigned_byte(0x00)
    assert encoder.buf == bytearray([0x01, 0x02, 0x03, 0x00])

    # Test encoding signed byte
    encoder = Encoder()
    encoder.signed_byte(-1)
    encoder.signed_byte(127)
    encoder.signed_byte(-128)
    encoder.signed_byte(0)
    assert encoder.buf == bytearray([0xFF, 0x7F, 0x80, 0x00])

    # Test encoding unsigned word
    encoder = Encoder()
    encoder.unsigned_word_le(0xBBAA)
    encoder.unsigned_word_le(0xDDCC)
    assert encoder.buf == bytearray([0xAA, 0xBB, 0xCC, 0xDD])

    # Test encoding unsigned word big-endian
    encoder = Encoder()
    encoder.unsigned_word_be(0xAABB)
    encoder.unsigned_word_be(0xCCDD)
    assert encoder.buf == bytearray([0xAA, 0xBB, 0xCC, 0xDD])

    # Test encoding signed word
    encoder = Encoder()
    encoder.signed_word_le(-1)
    encoder.signed_word_le(-32768)
    assert encoder.buf == bytearray([0xFF, 0xFF, 0x00, 0x80])

    # Test encoding signed word big-endian
    encoder = Encoder()
    encoder.signed_word_be(-1)
    encoder.signed_word_be(-32768)
    assert encoder.buf == bytearray([0xFF, 0xFF, 0x80, 0x00])


