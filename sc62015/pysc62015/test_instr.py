from .instr import (
    decode,
    encode,
    OPCODES,
    Operand,
    EMemReg,
    EMemRegMode,
    EMemIMem,
    EMemIMemMode,
    IMem8,
    IMemHelper,
    Imm8,
    ImmOffset,
    EMemValueOffsetHelper,
    EMemRegOffsetHelper,
    INTERNAL_MEMORY_START,
    UnknownInstruction,
    Reg,
    PRE,
    TCL,
    HALT,
    OFF,
    IR,
)
from .tokens import TInstr, TSep, TText, TInt, asm_str, TBegMem, TEndMem, MemType, TReg
from .coding import Decoder, Encoder
from .mock_analysis import MockAnalysisInfo
from .mock_llil import MockLowLevelILFunction, MockLLIL, mllil, mreg

import os
from pprint import pprint


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
    assert instr.render() == [TInstr("JP"), TSep("    "), TInt("BBAA")]

    instr = decode(bytearray([0x03, 0xAA, 0xBB, 0x0C]), 0x1234, OPCODES)
    assert instr.render() == [TInstr("JPF"), TSep("   "), TInt("CBBAA")]


def test_mvi():
    instr = decode(bytearray([0x08, 0xAA]), 0x1234, OPCODES)
    assert instr.name() == "MV"
    assert instr.render() == [
        TInstr("MV"),
        TSep("    "),
        TReg("A"),
        TSep(", "),
        TInt("AA"),
    ]

    instr = decode(bytearray([0x09, 0xAA]), 0x1234, OPCODES)
    assert instr.name() == "MV"
    assert instr.render() == [
        TInstr("MV"),
        TSep("    "),
        TReg("IL"),
        TSep(", "),
        TInt("AA"),
    ]


def test_emem_reg():
    # SIMPLE
    instr = decode(bytearray([0x90, 0x04]), 0x1234, OPCODES)
    _, op = instr.operands()
    assert asm_str(op.render()) == "[X]"

    instr = decode(bytearray([0xB0, 0x04]), 0x1234, OPCODES)
    op, _ = instr.operands()
    assert asm_str(op.render()) == "[X]"

    # POST_INC
    instr = decode(bytearray([0x90, 0x24]), 0x1234, OPCODES)
    _, op = instr.operands()

    # PRE_DEC
    instr = decode(bytearray([0x90, 0x34]), 0x1234, OPCODES)
    _, op = instr.operands()
    assert asm_str(op.render()) == "[--X]"

    # POSITIVE_OFFSET
    instr = decode(bytearray([0x90, 0x84, 0xBB]), 0x1234, OPCODES)
    _, op = instr.operands()
    assert asm_str(op.render()) == "[X+BB]"

    # NEGATIVE_OFFSET
    instr = decode(bytearray([0x90, 0xC4, 0xBB]), 0x1234, OPCODES)
    _, op = instr.operands()
    assert asm_str(op.render()) == "[X-BB]"


def test_emem_imem():
    def render(op):
        r = []
        for o in op.operands():
            r.extend(o.render())
        return r

    # SIMPLE
    decoder = Decoder(bytearray([0x00, 0x02]))
    op = EMemIMem()
    op.decode(decoder, 0x1234)
    assert op.mode == EMemIMemMode.SIMPLE
    assert render(op) == [
        TBegMem(MemType.EXTERNAL),
        TBegMem(MemType.INTERNAL),
        TInt("02"),
        TEndMem(MemType.INTERNAL),
        TEndMem(MemType.EXTERNAL),
    ]

    # POSITIVE_OFFSET
    decoder = Decoder(bytearray([0x80, 0x02, 0xBB]))
    op = EMemIMem()
    op.decode(decoder, 0x1234)
    assert op.mode == EMemIMemMode.POSITIVE_OFFSET
    assert render(op) == [
        TBegMem(MemType.EXTERNAL),
        TBegMem(MemType.INTERNAL),
        TInt("02"),
        TEndMem(MemType.INTERNAL),
        TInt("+BB"),
        TEndMem(MemType.EXTERNAL),
    ]
    encoder = Encoder()
    op.encode(encoder, 0x1234)
    assert encoder.buf == bytearray([0x80, 0x02, 0xBB])

    # NEGATIVE_OFFSET
    decoder = Decoder(bytearray([0xC0, 0x02, 0xBB]))
    op = EMemIMem()
    op.decode(decoder, 0x1234)
    assert op.mode == EMemIMemMode.NEGATIVE_OFFSET
    assert render(op) == [
        TBegMem(MemType.EXTERNAL),
        TBegMem(MemType.INTERNAL),
        TInt("02"),
        TEndMem(MemType.INTERNAL),
        TInt("-BB"),
        TEndMem(MemType.EXTERNAL),
    ]
    encoder = Encoder()
    op.encode(encoder, 0x1234)


def test_inc_lifting():
    instr = decode(bytearray([0x6C, 0x00]), 0x1234, OPCODES)
    assert asm_str(instr.render()) == "INC   A"

    il = MockLowLevelILFunction()
    instr.lift(il, 0x1234)
    assert il.ils == [
        mllil(
            "SET_REG.b{0}",
            [
                mreg("A"),
                mllil(
                    "ADD.b{Z}",
                    [
                        mllil("REG.b", [mreg("A")]),
                        mllil("CONST.b", [1]),
                    ],
                ),
            ],
        )
    ]


def test_emem_value_offset_helper_lifting():
    imem = IMem8()
    imem.value = 0xAB

    offset = ImmOffset("+")
    offset.value = 0xCD

    h = EMemValueOffsetHelper(imem, offset)
    assert asm_str(h.render()) == "[(AB)+CD]"

    il = MockLowLevelILFunction()
    assert h.lift(il) == mllil(
        "LOAD.b",
        [
            mllil(
                "ADD.b",
                [
                    mllil(
                        "LOAD", [mllil("CONST_PTR.l", [INTERNAL_MEMORY_START + 0xAB])]
                    ),
                    mllil("CONST.b", [0xCD]),
                ],
            )
        ],
    )


def test_lift_mv():
    instr = decode(bytearray([0x08, 0xCD]), 0x1234, OPCODES)
    assert asm_str(instr.render()) == "MV    A, CD"

    il = MockLowLevelILFunction()
    instr.lift(il, 0x1234)
    assert il.ils == [
        mllil(
            "SET_REG.b{0}",
            [
                mreg("A"),
                mllil("CONST.b", [0xCD]),
            ],
        )
    ]

    instr = decode(bytearray([0xC8, 0xAB, 0xCD]), 0x1234, OPCODES)
    assert asm_str(instr.render()) == "MV    (AB), (CD)"

    il = MockLowLevelILFunction()
    instr.lift(il, 0x1234)
    assert il.ils == [
        mllil(
            "STORE",
            [
                mllil("CONST_PTR.l", [INTERNAL_MEMORY_START + 0xAB]),
                mllil(
                    "LOAD",
                    [
                        mllil("CONST_PTR.l", [INTERNAL_MEMORY_START + 0xCD]),
                    ],
                ),
            ],
        )
    ]


# Format:
# F90F0F00: MVW   [(0F)],(00)
def opcode_generator():
    dirname = os.path.dirname(__file__)
    with open(os.path.join(dirname, "opcodes.txt")) as f:
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

        rendered_str = asm_str(rendered)
        rendered_str = " ".join(rendered_str.split())
        rendered_str = rendered_str.replace(", ", ",")
        assert rendered_str == s, f"Failed at line {i+1}: {s}"

        # test that no assertions are raised
        info = MockAnalysisInfo()
        instr.analyze(info, 0x1234)
        assert info.length == len(b), f"Failed at line {i+1}: {s}"

        try:
            # test that no assertions are raised
            il = MockLowLevelILFunction()
            instr.lift(il, 0x1234)
        except Exception as exc:
            raise ValueError(f"Failed to lift {b.hex()} at line {i+1}: {s}") from exc

        def check_no_unimplemented(instr):
            if isinstance(instr, MockLLIL):
                if instr.op == "UNIMPL":
                    pprint(il.ils)
                    raise ValueError(
                        f"Unimplemented instruction: {instr} for {rendered_str} at line {i+1}"
                    )

                for op in instr.ops:
                    check_no_unimplemented(op)

        def start_check_lifting(ils):
            assert len(ils) > 0, f"Failed to lift {b.hex()} at line {i+1}: {s}"
            for instr in ils:
                check_no_unimplemented(instr)

        if not isinstance(instr, (UnknownInstruction, PRE, TCL, HALT, OFF, IR)):
            start_check_lifting(il.ils)
