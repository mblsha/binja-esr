from .instr import (
    encode,
    OPCODES,
    Operand,
    Instruction,
    JP_Abs,
    JP_Rel,
    EMemIMem,
    EMemIMemMode,
    IMem8,
    ImmOffset,
    EMemValueOffsetHelper,
    INTERNAL_MEMORY_START,
    UnknownInstruction,
    PRE,
    TCL,
    HALT,
    OFF,
    IR,
)
from .instr import decode as decode_instr
from .tokens import Token, TInstr, TSep, TText, TInt, asm_str, TBegMem, TEndMem, MemType, TReg
from .coding import Decoder, Encoder
from .mock_analysis import MockAnalysisInfo
from .mock_llil import MockLowLevelILFunction, MockLLIL, mllil, mreg

import os
from pprint import pprint

from typing import Generator, Tuple, List, Optional


def decode(data: bytearray, addr: int) -> Instruction:
    decoder = Decoder(data)
    instr = decode_instr(decoder, addr, OPCODES) # type: ignore
    if instr is None:
        raise ValueError(f"Failed to decode {data.hex()} at {addr:#x}")
    return instr



def test_operand() -> None:
    op = Operand()
    assert op.render() == [TText("unimplemented")]


def test_nop() -> None:
    instr = decode(bytearray([0x00]), 0x1234)
    assert instr.name() == "NOP"
    assert instr.render() == [TInstr("NOP")]


def test_jp_abs() -> None:
    instr = decode(bytearray([0x02, 0xAA, 0xBB]), 0xCD1234)
    assert instr.name() == "JP"
    assert instr.render() == [TInstr("JP"), TSep("    "), TInt("BBAA")]
    il = MockLowLevelILFunction()
    assert isinstance(instr, JP_Abs)
    assert instr.lift_jump_addr(il, 0xCD1234) == mllil("OR.l", [
        mllil("CONST.w", [0xBBAA]),
        mllil("CONST.l", [0xCD0000]),
    ])

    instr = decode(bytearray([0x03, 0xAA, 0xBB, 0x0C]), 0x1234)
    assert isinstance(instr, JP_Abs)
    assert instr.render() == [TInstr("JPF"), TSep("   "), TInt("CBBAA")]
    assert instr.lift_jump_addr(il, 0x1234) == mllil("CONST.l", [0xCBBAA])

    instr = decode(bytearray([0x15, 0xcd, 0x00]), 0xf0185)
    assert isinstance(instr, JP_Abs)
    assert instr.render() == [TInstr("JPNZ"), TSep("  "), TInt("00CD")]
    assert instr.lift_jump_addr(il, 0xf0185) == mllil("OR.l", [
        mllil("CONST.w", [0x00CD]),
        mllil("CONST.l", [0xf0000]),
    ])


def test_jp_rel() -> None:
    instr = decode(bytearray([0x1a, 0x06]), 0xf0163)
    assert instr.name() == "JRNZ"
    il = MockLowLevelILFunction()
    assert isinstance(instr, JP_Rel)
    assert instr.lift_jump_addr(il, 0xf0163) == mllil('CONST.l', [0xf0163 + 2 + 6])

    instr = decode(bytearray([0x1b, 0x06]), 0xf0163)
    assert instr.name() == "JRNZ"
    assert isinstance(instr, JP_Rel)
    assert instr.lift_jump_addr(il, 0xf0163) == mllil('CONST.l', [0xf0163 + 2 - 6])


def test_mvi() -> None:
    instr = decode(bytearray([0x08, 0xAA]), 0x1234)
    assert instr.name() == "MV"
    assert instr.render() == [
        TInstr("MV"),
        TSep("    "),
        TReg("A"),
        TSep(", "),
        TInt("AA"),
    ]

    instr = decode(bytearray([0x09, 0xAA]), 0x1234)
    assert instr.name() == "MV"
    assert instr.render() == [
        TInstr("MV"),
        TSep("    "),
        TReg("IL"),
        TSep(", "),
        TInt("AA"),
    ]


def test_emem_reg() -> None:
    # SIMPLE
    instr = decode(bytearray([0x90, 0x04]), 0x1234)
    _, op = instr.operands()
    assert asm_str(op.render()) == "[X]"

    instr = decode(bytearray([0xB0, 0x04]), 0x1234)
    op, _ = instr.operands()
    assert asm_str(op.render()) == "[X]"

    # POST_INC
    instr = decode(bytearray([0x90, 0x24]), 0x1234)
    _, op = instr.operands()

    # PRE_DEC
    instr = decode(bytearray([0x90, 0x34]), 0x1234)
    _, op = instr.operands()
    assert asm_str(op.render()) == "[--X]"

    # POSITIVE_OFFSET
    instr = decode(bytearray([0x90, 0x84, 0xBB]), 0x1234)
    _, op = instr.operands()
    assert asm_str(op.render()) == "[X+BB]"

    # NEGATIVE_OFFSET
    instr = decode(bytearray([0x90, 0xC4, 0xBB]), 0x1234)
    _, op = instr.operands()
    assert asm_str(op.render()) == "[X-BB]"


def test_emem_imem() -> None:
    def render(op: EMemIMem) -> List[Token]:
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


def test_inc_lifting() -> None:
    instr = decode(bytearray([0x6C, 0x00]), 0x1234)
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


def test_emem_value_offset_helper_lifting() -> None:
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
                        "LOAD.b", [mllil("CONST_PTR.l", [INTERNAL_MEMORY_START + 0xAB])]
                    ),
                    mllil("CONST.b", [0xCD]),
                ],
            )
        ],
    )


def test_lift_mv() -> None:
    instr = decode(bytearray([0x08, 0xCD]), 0x1234)
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

    instr = decode(bytearray([0xC8, 0xAB, 0xCD]), 0x1234)
    assert asm_str(instr.render()) == "MV    (AB), (CD)"

    il = MockLowLevelILFunction()
    instr.lift(il, 0x1234)
    assert il.ils == [
        mllil(
            "STORE.b",
            [
                mllil("CONST_PTR.l", [INTERNAL_MEMORY_START + 0xAB]),
                mllil(
                    "LOAD.b",
                    [
                        mllil("CONST_PTR.l", [INTERNAL_MEMORY_START + 0xCD]),
                    ],
                ),
            ],
        )
    ]


def test_invalid_instruction() -> None:
    data = bytearray([0x4d, 0x4d])
    try:
        decode(data, 0x1234)
    except ValueError as exc:
        assert str(exc) == "Failed to decode 4d4d at 0x1234"
        pass
    else:
        assert False, "Expected Exception"


def test_pre_roundtrip() -> None:
    # 3331307dec
    data = bytearray([0x33, 0x7d, 0xec])
    instr = decode(data, 0x1234)
    assert instr._pre == 0x33
    assert asm_str(instr.render()) == "DEC   (EC)"

    encoder = Encoder()
    instr.encode(encoder, 0x1234)
    assert encoder.buf == data


def test_lift_pre() -> None:
    # no PRE: MV IMem8, Imm8
    instr = decode(bytearray([0xCC, 0xFB, 0x00]), 0xf0102)
    assert asm_str(instr.render()) == "MV    (FB), 00"
    assert instr._pre is None
    assert instr.length() == 3

    il = MockLowLevelILFunction()
    instr.lift(il, 0xf0102)
    assert il.ils == [
        mllil(
            "STORE.b",
            [
                mllil("CONST_PTR.l", [INTERNAL_MEMORY_START + 0xFB]),
                mllil("CONST.b", [0x00]),
            ],
        )
    ]


    # PRE25 + MV IMem8, Imm8
    instr = decode(bytearray([0x25, 0xCC, 0xFB, 0x00]), 0xf0102)
    assert asm_str(instr.render()) == "MV    (BP+PX), 00"
    assert instr._pre == 0x25
    assert instr.length() == 4

    il = MockLowLevelILFunction()
    instr.lift(il, 0xf0102)
    assert il.ils == [
        mllil(
            "STORE.b",
            [
                mllil(
                    "ADD.l",
                    [
                        mllil("ADD.b", [
                            mllil("LOAD.b", [mllil("CONST_PTR.l", [INTERNAL_MEMORY_START + 0xEC])]),
                            mllil("LOAD.b", [mllil("CONST_PTR.l", [INTERNAL_MEMORY_START + 0xED])]),
                        ]),
                        mllil("CONST.l", [INTERNAL_MEMORY_START]),
                    ],
                ),
                mllil("CONST.b", [0x00]),
            ],
        )
    ]


# Format:
# F90F0F00: MVW   [(0F)],(00)
def opcode_generator() -> Generator[Tuple[Optional[bytearray], Optional[str]], None, None]:
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


def test_opcode_generator() -> None:
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


def test_compare_opcodes() -> None:
    # enumerate all opcodes, want index for each opcode
    for i, (b, s) in enumerate(opcode_generator()):
        if b is None:
            continue
        try:
            instr = decode(b, 0x1234)

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

        def check_no_unimplemented(instr: MockLLIL) -> None:
            if isinstance(instr, MockLLIL):
                if instr.op == "UNIMPL":
                    pprint(il.ils)
                    raise ValueError(
                        f"Unimplemented instruction: {instr} for {rendered_str} at line {i+1}"
                    )

                for op in instr.ops:
                    check_no_unimplemented(op)

        def start_check_lifting(ils: List[MockLLIL]) -> None:
            assert len(ils) > 0, f"Failed to lift {b.hex()} at line {i+1}: {s}"
            for instr in ils:
                check_no_unimplemented(instr)

        if not isinstance(instr, (UnknownInstruction, PRE, TCL, HALT, OFF, IR)):
            start_check_lifting(il.ils)
