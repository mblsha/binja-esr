from .instr import (
    decode,
    encode,
    OPCODES,
    Operand,
    EMemReg,
    EMemRegMode,
    EMemIMem,
    EMemIMemMode,
)
from .tokens import TInstr, TSep, TText, TInt, asm_str
from .coding import Decoder, Encoder
from .analysis import MockAnalysisInfo


def test_operand():
    op = Operand()
    assert op.render() == [TText("unimplemented")]


def test_nop():
    instr = decode(bytearray([0x00]), 0x1234, OPCODES)
    assert instr.name() == "NOP"
    assert instr.render() == [TInstr("NOP")]


def test_jp_abs():
    instr = decode(bytearray([0x02, 0xAA, 0xBB]), 0x1234, OPCODES)
    assert instr.name() == "JP"
    assert instr.render() == [TInstr("JP"), TSep(" "), TInt("BBAA")]

    instr = decode(bytearray([0x03, 0xAA, 0xBB, 0x0C]), 0x1234, OPCODES)
    assert instr.render() == [TInstr("JPF"), TSep(" "), TInt("CBBAA")]


def test_mvi():
    instr = decode(bytearray([0x08, 0xAA]), 0x1234, OPCODES)
    assert instr.name() == "MV"
    assert instr.render() == [
        TInstr("MV"),
        TSep(" "),
        TText("A"),
        TSep(","),
        TInt("AA"),
    ]

    instr = decode(bytearray([0x09, 0xAA]), 0x1234, OPCODES)
    assert instr.name() == "MV"
    assert instr.render() == [
        TInstr("MV"),
        TSep(" "),
        TText("IL"),
        TSep(","),
        TInt("AA"),
    ]


def test_emem_reg():
    # SIMPLE
    decoder = Decoder(bytearray([0x04]))
    op = EMemReg()
    op.decode(decoder, 0x1234)
    assert op.mode == EMemRegMode.SIMPLE
    assert op.render() == [TText("["), TText("X"), TText("]")]
    encoder = Encoder()
    op.encode(encoder, 0x1234)
    assert encoder.buf == bytearray([0x04])

    # POST_INC
    decoder = Decoder(bytearray([0x24]))
    op = EMemReg()
    op.decode(decoder, 0x1234)
    assert op.mode == EMemRegMode.POST_INC
    assert op.render() == [TText("["), TText("X"), TText("++"), TText("]")]
    encoder = Encoder()
    op.encode(encoder, 0x1234)
    assert encoder.buf == bytearray([0x24])

    # PRE_DEC
    decoder = Decoder(bytearray([0x34]))
    op = EMemReg()
    op.decode(decoder, 0x1234)
    assert op.mode == EMemRegMode.PRE_DEC
    assert op.render() == [TText("["), TText("--"), TText("X"), TText("]")]
    encoder = Encoder()
    op.encode(encoder, 0x1234)
    assert encoder.buf == bytearray([0x34])

    # POSITIVE_OFFSET
    decoder = Decoder(bytearray([0x84, 0xBB]))
    op = EMemReg()
    op.decode(decoder, 0x1234)
    assert op.mode == EMemRegMode.POSITIVE_OFFSET
    assert op.offset.value == 0xBB
    assert op.render() == [TText("["), TText("X"), TInt("+BB"), TText("]")]
    encoder = Encoder()
    op.encode(encoder, 0x1234)
    assert encoder.buf == bytearray([0x84, 0xBB])

    # NEGATIVE_OFFSET
    decoder = Decoder(bytearray([0xC4, 0xBB]))
    op = EMemReg()
    op.decode(decoder, 0x1234)
    assert op.mode == EMemRegMode.NEGATIVE_OFFSET
    assert op.offset.value == 0xBB
    assert op.render() == [TText("["), TText("X"), TInt("-BB"), TText("]")]
    encoder = Encoder()
    op.encode(encoder, 0x1234)
    assert encoder.buf == bytearray([0xC4, 0xBB])


def test_emem_imem():
    # SIMPLE
    decoder = Decoder(bytearray([0x00, 0x02]))
    op = EMemIMem()
    op.decode(decoder, 0x1234)
    assert op.mode == EMemIMemMode.SIMPLE
    assert op.render() == [TText("["), TInt("(02)"), TText("]")]

    # POSITIVE_OFFSET
    decoder = Decoder(bytearray([0x80, 0x02, 0xBB]))
    op = EMemIMem()
    op.decode(decoder, 0x1234)
    assert op.mode == EMemIMemMode.POSITIVE_OFFSET
    assert op.render() == [TText("["), TInt("(02)"), TInt("+BB"), TText("]")]
    encoder = Encoder()
    op.encode(encoder, 0x1234)
    assert encoder.buf == bytearray([0x80, 0x02, 0xBB])

    # NEGATIVE_OFFSET
    decoder = Decoder(bytearray([0xC0, 0x02, 0xBB]))
    op = EMemIMem()
    op.decode(decoder, 0x1234)
    assert op.mode == EMemIMemMode.NEGATIVE_OFFSET
    assert op.render() == [TText("["), TInt("(02)"), TInt("-BB"), TText("]")]
    encoder = Encoder()
    op.encode(encoder, 0x1234)


# Format:
# F90F0F00: MVW   [(0F)],(00)
def opcode_generator():
    with open("opcodes.txt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                yield None, None
                continue
            parts = line.split(":")
            if len(parts) != 2:
                raise ValueError(f"Invalid line: {line}")
            byte_str, expected_str = parts
            byte_array = bytearray.fromhex(byte_str)
            expected_str = expected_str.strip()
            # replace repeated spaces with a single space
            expected_str = " ".join(expected_str.split())
            yield byte_array, expected_str


def test_opcode_generator():
    gen = opcode_generator()
    b, s = next(gen)
    assert b == bytearray([0x00])
    assert s == "NOP"

    b, s = next(gen)
    assert b == bytearray([0x01])
    assert s == "RETI"

    b, s = next(gen)
    assert b == bytearray([0x02, 0x00, 0x00])
    assert s == "JP 0000"

    b, s = next(gen)
    assert b == bytearray([0x02, 0x00, 0x07])
    assert s == "JP 0700"


def test_compare_opcodes():
    # enumerate all opcodes, want index for each opcode
    for i, (b, s) in enumerate(opcode_generator()):
        if b is None:
            continue
        try:
            instr = decode(b, 0x1234, OPCODES)

            recoded = encode(instr, 0x1234)
            if b != recoded:
                opcode = hex(b[0])
                raise ValueError(
                    f"Opcode {opcode}: Encoded instruction {b} does not match recoded {recoded}"
                )

        except Exception as exc:
            raise ValueError(f"Failed to decode {b.hex()} at line {i+1}: {s}") from exc

        if not instr:
            raise ValueError(f"Failed to decode {b.hex()} at line {i+1}: {s}")
        try:
            rendered = instr.render()
        except Exception as exc:
            raise ValueError(f"Failed to render {b.hex()} at line {i+1}: {s}") from exc
        if not rendered:
            raise ValueError(f"Failed to render {b.hex()} at line {i+1}: {s}")
        # assert instr.name() == s.split()[0], f"Failed at line {i+1}: {s}"
        assert asm_str(rendered) == s, f"Failed at line {i+1}: {s}"

        info = MockAnalysisInfo()
        instr.analyze(info, 0x1234)
        assert info.length == len(b), f"Failed at line {i+1}: {s}"
