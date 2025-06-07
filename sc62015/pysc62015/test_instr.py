from . import binja_api  # noqa: F401

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
    IMemHelper,
    IMEM_NAMES,
    TempReg,
    AddressingMode,
    Reg,
    Imm8,
    ImmOffset,
    EMemValueOffsetHelper,
    EMemRegOffsetHelper,
    EMemRegMode,
    Reg3,
    UnknownInstruction,
    PRE,
    TCL,
    HALT,
    OFF,
    IR,
)
from .instr import decode as decode_instr
from .constants import INTERNAL_MEMORY_START
from .tokens import (
    Token,
    TInstr,
    TSep,
    TText,
    TInt,
    asm_str,
    TBegMem,
    TEndMem,
    MemType,
    TReg,
)
from .coding import Decoder, Encoder
from .mock_analysis import MockAnalysisInfo
from .mock_llil import MockLowLevelILFunction, MockLLIL, MockFlag, mllil, mreg
from binaryninja.lowlevelil import (
    LLIL_TEMP,
)

import os
from pprint import pprint

from typing import Generator, Tuple, List, Optional


def decode(data: bytearray, addr: int) -> Instruction:
    decoder = Decoder(data)
    instr = decode_instr(decoder, addr, OPCODES)  # type: ignore
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
    assert instr.lift_jump_addr(il, 0xCD1234) == mllil(
        "OR.l",
        [
            mllil("CONST.w", [0xBBAA]),
            mllil("CONST.l", [0xCD0000]),
        ],
    )

    instr = decode(bytearray([0x03, 0xAA, 0xBB, 0x0C]), 0x1234)
    assert isinstance(instr, JP_Abs)
    assert instr.render() == [TInstr("JPF"), TSep("   "), TInt("CBBAA")]
    assert instr.lift_jump_addr(il, 0x1234) == mllil("CONST.l", [0xCBBAA])

    instr = decode(bytearray([0x15, 0xCD, 0x00]), 0xF0185)
    assert isinstance(instr, JP_Abs)
    assert instr.render() == [TInstr("JPNZ"), TSep("  "), TInt("00CD")]
    assert instr.lift_jump_addr(il, 0xF0185) == mllil(
        "OR.l",
        [
            mllil("CONST.w", [0x00CD]),
            mllil("CONST.l", [0xF0000]),
        ],
    )


def test_jp_rel() -> None:
    instr = decode(bytearray([0x1A, 0x06]), 0xF0163)
    assert instr.name() == "JRNZ"
    il = MockLowLevelILFunction()
    assert isinstance(instr, JP_Rel)
    assert instr.lift_jump_addr(il, 0xF0163) == mllil("CONST.l", [0xF0163 + 2 + 6])

    instr = decode(bytearray([0x1B, 0x06]), 0xF0163)
    assert instr.name() == "JRNZ"
    assert isinstance(instr, JP_Rel)
    assert instr.lift_jump_addr(il, 0xF0163) == mllil("CONST.l", [0xF0163 + 2 - 6])


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


def test_emem_value_offset_helper_widths() -> None:
    imem = IMem8()
    imem.value = 0x10

    offset = ImmOffset("+")
    offset.value = 1

    for width, suffix in [(2, "w"), (3, "l")]:
        h = EMemValueOffsetHelper(imem, offset, width=width)
        il = MockLowLevelILFunction()
        assert h.lift(il) == mllil(
            f"LOAD.{suffix}",
            [
                mllil(
                    "ADD.b",
                    [
                        mllil(
                            "LOAD.b",
                            [mllil("CONST_PTR.l", [INTERNAL_MEMORY_START + 0x10])],
                        ),
                        mllil("CONST.b", [1]),
                    ],
                )
            ],
        )


def test_emem_reg_offset_helper_widths() -> None:
    reg = Reg3()
    reg.reg = "X"

    for width, suffix in [(2, "w"), (3, "l")]:
        h = EMemRegOffsetHelper(width, reg, EMemRegMode.SIMPLE, offset=None)
        op = next(h.operands())
        il = MockLowLevelILFunction()
        assert op.lift(il) == mllil(
            f"LOAD.{suffix}",
            [mllil("REG.l", [mreg("X")])],
        )
        il2 = MockLowLevelILFunction()
        op.lift_assign(il2, il2.const(width, 0x11))
        assert il2.ils == [
            mllil(
                f"STORE.{suffix}",
                [mllil("REG.l", [mreg("X")]), mllil(f"CONST.{suffix}", [0x11])],
            )
        ]


class TestIMemHelperLifting:
    def _get_imem_addr_llil(
        self, helper: IMemHelper, pre_mode: Optional[AddressingMode] = None
    ) -> MockLLIL:
        il = MockLowLevelILFunction()
        # The imem_addr method returns an ExpressionIndex, which in MockLLIL is the MockLLIL itself
        # or an index if append was used. For direct calls like this, it should be the MockLLIL.
        addr_expr = helper.imem_addr(il, pre_mode)
        # If imem_addr directly returns an expression (like const_pointer or reg),
        # it won't be in il.ils. If it builds an expression (like add), it might be.
        # For simplicity, we'll assume addr_expr is the primary result.
        # If it complexly appends to il.ils, this might need adjustment.
        # However, imem_addr is designed to *return* the address expression.
        assert isinstance(
            addr_expr, MockLLIL
        ), f"Expected MockLLIL, got {type(addr_expr)}"
        return addr_expr

    def test_imem_helper_direct_n_mode(self) -> None:
        # IMemHelper for (0x10)
        helper = IMemHelper(width=1, value=Imm8(0x10))
        addr_llil = self._get_imem_addr_llil(helper, pre_mode=AddressingMode.N)
        expected_llil = mllil("CONST_PTR.l", [INTERNAL_MEMORY_START + 0x10])
        assert addr_llil == expected_llil

    def test_imem_helper_direct_no_pre(self) -> None:
        # IMemHelper for (0x25) with pre=None (should default to N mode behavior for Imm8)
        helper = IMemHelper(width=1, value=Imm8(0x25))
        addr_llil = self._get_imem_addr_llil(helper, pre_mode=None)
        expected_llil = mllil("CONST_PTR.l", [INTERNAL_MEMORY_START + 0x25])
        assert addr_llil == expected_llil

    def test_imem_helper_bp_plus_n_mode(self) -> None:
        # IMemHelper for (BP+0x05), assuming BP holds 0x02
        # Setup: mock that BP (IMEM[0xEC]) contains 0x02
        # The helper itself doesn't know BP's value, its `value` is Imm8(0x05)
        # The lifting of `_reg_value("BP", il)` will produce the LOAD for BP
        helper = IMemHelper(width=1, value=Imm8(0x05))  # n = 0x05
        addr_llil = self._get_imem_addr_llil(helper, pre_mode=AddressingMode.BP_N)

        # Expected: add.l( add.b( load.b( const_ptr.l(IMEM_START + BP_ADDR) ), const.b(0x05) ), const_ptr.l(IMEM_START) )
        # Simplified due to const_ptr in _imem_offset for the base of the offset calculation:
        # add.l ( add.b ( load.b (const_ptr.l (IMEM_START + 0xEC)) , const.b(0x05) ), const.l (IMEM_START) )
        # Note: The final `add` combines the 8-bit offset with INTERNAL_MEMORY_START.
        # The `_imem_offset` calculates `BP + n` as an 8-bit value.
        # `imem_addr` then adds `INTERNAL_MEMORY_START` to this.

        expected_llil = mllil(
            "ADD.l",
            [
                mllil(
                    "ADD.b",
                    [
                        mllil(
                            "LOAD.b",
                            [
                                mllil(
                                    "CONST_PTR.l",
                                    [INTERNAL_MEMORY_START + IMEM_NAMES["BP"]],
                                )
                            ],
                        ),  # BP value
                        mllil("CONST.b", [0x05]),  # n
                    ],
                ),
                mllil(
                    "CONST.l", [INTERNAL_MEMORY_START]
                ),  # Add base for internal memory
            ],
        )
        assert addr_llil == expected_llil

    def test_imem_helper_px_plus_n_mode(self) -> None:
        # IMemHelper for (PX+0x0A)
        helper = IMemHelper(width=1, value=Imm8(0x0A))  # n = 0x0A
        addr_llil = self._get_imem_addr_llil(helper, pre_mode=AddressingMode.PX_N)

        expected_llil = mllil(
            "ADD.l",
            [
                mllil(
                    "ADD.b",
                    [
                        mllil(
                            "LOAD.b",
                            [
                                mllil(
                                    "CONST_PTR.l",
                                    [INTERNAL_MEMORY_START + IMEM_NAMES["PX"]],
                                )
                            ],
                        ),  # PX value
                        mllil("CONST.b", [0x0A]),  # n
                    ],
                ),
                mllil("CONST.l", [INTERNAL_MEMORY_START]),
            ],
        )
        assert addr_llil == expected_llil

    def test_imem_helper_py_plus_n_mode(self) -> None:
        # IMemHelper for (PY+0x03)
        helper = IMemHelper(width=1, value=Imm8(0x03))  # n = 0x03
        addr_llil = self._get_imem_addr_llil(helper, pre_mode=AddressingMode.PY_N)

        expected_llil = mllil(
            "ADD.l",
            [
                mllil(
                    "ADD.b",
                    [
                        mllil(
                            "LOAD.b",
                            [
                                mllil(
                                    "CONST_PTR.l",
                                    [INTERNAL_MEMORY_START + IMEM_NAMES["PY"]],
                                )
                            ],
                        ),  # PY value
                        mllil("CONST.b", [0x03]),  # n
                    ],
                ),
                mllil("CONST.l", [INTERNAL_MEMORY_START]),
            ],
        )
        assert addr_llil == expected_llil

    def test_imem_helper_bp_plus_px_mode(self) -> None:
        # IMemHelper for (BP+PX). The Imm8 value is often 0 or ignored in this mode.
        helper = IMemHelper(
            width=1, value=Imm8(0x00)
        )  # n is ignored by _imem_offset for this pre
        addr_llil = self._get_imem_addr_llil(helper, pre_mode=AddressingMode.BP_PX)

        expected_llil = mllil(
            "ADD.l",
            [
                mllil(
                    "ADD.b",
                    [
                        mllil(
                            "LOAD.b",
                            [
                                mllil(
                                    "CONST_PTR.l",
                                    [INTERNAL_MEMORY_START + IMEM_NAMES["BP"]],
                                )
                            ],
                        ),  # BP value
                        mllil(
                            "LOAD.b",
                            [
                                mllil(
                                    "CONST_PTR.l",
                                    [INTERNAL_MEMORY_START + IMEM_NAMES["PX"]],
                                )
                            ],
                        ),  # PX value
                    ],
                ),
                mllil("CONST.l", [INTERNAL_MEMORY_START]),
            ],
        )
        assert addr_llil == expected_llil

    def test_imem_helper_bp_plus_py_mode(self) -> None:
        # IMemHelper for (BP+PY)
        helper = IMemHelper(width=1, value=Imm8(0x00))
        addr_llil = self._get_imem_addr_llil(helper, pre_mode=AddressingMode.BP_PY)

        expected_llil = mllil(
            "ADD.l",
            [
                mllil(
                    "ADD.b",
                    [
                        mllil(
                            "LOAD.b",
                            [
                                mllil(
                                    "CONST_PTR.l",
                                    [INTERNAL_MEMORY_START + IMEM_NAMES["BP"]],
                                )
                            ],
                        ),  # BP value
                        mllil(
                            "LOAD.b",
                            [
                                mllil(
                                    "CONST_PTR.l",
                                    [INTERNAL_MEMORY_START + IMEM_NAMES["PY"]],
                                )
                            ],
                        ),  # PY value
                    ],
                ),
                mllil("CONST.l", [INTERNAL_MEMORY_START]),
            ],
        )
        assert addr_llil == expected_llil

    def test_imem_helper_temp_reg_value_no_pre(self) -> None:
        # This tests the case where IMemHelper's value is a TempReg,
        # which is assumed to ALREADY hold the full internal memory address.
        # Example: TempReg(TEMP0) holds INTERNAL_MEMORY_START + 0x30

        temp_reg_operand = TempReg(LLIL_TEMP(0), width=3)
        helper = IMemHelper(width=1, value=temp_reg_operand)

        # When pre_mode is None, and value is a Reg/TempReg, it should just lift the reg.
        addr_llil = self._get_imem_addr_llil(helper, pre_mode=None)

        # Expected: REG.l (TEMP0). No additional INTERNAL_MEMORY_START should be added.
        expected_llil = mllil("REG.l", [mreg("TEMP0")])
        assert addr_llil == expected_llil

    def test_imem_helper_actual_reg_value_no_pre(self) -> None:
        # Similar to TempReg, but with an actual CPU register (e.g. X)
        # Actual Regs aren't supposed to hold full addresses ever.

        actual_reg_operand = Reg("X")  # X is 3 bytes (REG_SIZES['X'])
        helper = IMemHelper(width=1, value=actual_reg_operand)

        addr_llil = self._get_imem_addr_llil(helper, pre_mode=None)

        expected_llil = mllil("ADD.l", [
            mllil("REG.l", [mreg("X")]),
            mllil("CONST.l", [INTERNAL_MEMORY_START])
        ])
        assert addr_llil == expected_llil


class TestIMem8CurrentAddr:
    def _get_current_addr_llil(
        self, operand: IMem8, pre_mode: Optional[AddressingMode]
    ) -> MockLLIL:
        il = MockLowLevelILFunction()
        addr_expr = operand.lift_current_addr(il, pre=pre_mode, side_effects=False)
        assert isinstance(addr_expr, MockLLIL)
        return addr_expr

    def test_direct_n_mode(self) -> None:
        op = IMem8()
        op.value = 0x10
        addr_llil = self._get_current_addr_llil(op, AddressingMode.N)
        expected_llil = mllil("CONST_PTR.l", [INTERNAL_MEMORY_START + 0x10])
        assert addr_llil == expected_llil

    def test_direct_no_pre(self) -> None:
        op = IMem8()
        op.value = 0x25
        addr_llil = self._get_current_addr_llil(op, None)
        expected_llil = mllil("CONST_PTR.l", [INTERNAL_MEMORY_START + 0x25])
        assert addr_llil == expected_llil

    def test_bp_plus_n_mode(self) -> None:
        op = IMem8()
        op.value = 0x05
        addr_llil = self._get_current_addr_llil(op, AddressingMode.BP_N)
        expected_llil = mllil(
            "ADD.l",
            [
                mllil(
                    "ADD.b",
                    [
                        mllil(
                            "LOAD.b",
                            [
                                mllil(
                                    "CONST_PTR.l",
                                    [INTERNAL_MEMORY_START + IMEM_NAMES["BP"]],
                                )
                            ],
                        ),
                        mllil("CONST.b", [0x05]),
                    ],
                ),
                mllil("CONST.l", [INTERNAL_MEMORY_START]),
            ],
        )
        assert addr_llil == expected_llil

    def test_px_plus_n_mode(self) -> None:
        op = IMem8()
        op.value = 0x0A
        addr_llil = self._get_current_addr_llil(op, AddressingMode.PX_N)
        expected_llil = mllil(
            "ADD.l",
            [
                mllil(
                    "ADD.b",
                    [
                        mllil(
                            "LOAD.b",
                            [
                                mllil(
                                    "CONST_PTR.l",
                                    [INTERNAL_MEMORY_START + IMEM_NAMES["PX"]],
                                )
                            ],
                        ),
                        mllil("CONST.b", [0x0A]),
                    ],
                ),
                mllil("CONST.l", [INTERNAL_MEMORY_START]),
            ],
        )
        assert addr_llil == expected_llil

    def test_py_plus_n_mode(self) -> None:
        op = IMem8()
        op.value = 0x03
        addr_llil = self._get_current_addr_llil(op, AddressingMode.PY_N)
        expected_llil = mllil(
            "ADD.l",
            [
                mllil(
                    "ADD.b",
                    [
                        mllil(
                            "LOAD.b",
                            [
                                mllil(
                                    "CONST_PTR.l",
                                    [INTERNAL_MEMORY_START + IMEM_NAMES["PY"]],
                                )
                            ],
                        ),
                        mllil("CONST.b", [0x03]),
                    ],
                ),
                mllil("CONST.l", [INTERNAL_MEMORY_START]),
            ],
        )
        assert addr_llil == expected_llil

    def test_bp_plus_px_mode(self) -> None:
        op = IMem8()
        op.value = 0x00
        addr_llil = self._get_current_addr_llil(op, AddressingMode.BP_PX)
        expected_llil = mllil(
            "ADD.l",
            [
                mllil(
                    "ADD.b",
                    [
                        mllil(
                            "LOAD.b",
                            [
                                mllil(
                                    "CONST_PTR.l",
                                    [INTERNAL_MEMORY_START + IMEM_NAMES["BP"]],
                                )
                            ],
                        ),
                        mllil(
                            "LOAD.b",
                            [
                                mllil(
                                    "CONST_PTR.l",
                                    [INTERNAL_MEMORY_START + IMEM_NAMES["PX"]],
                                )
                            ],
                        ),
                    ],
                ),
                mllil("CONST.l", [INTERNAL_MEMORY_START]),
            ],
        )
        assert addr_llil == expected_llil

    def test_bp_plus_py_mode(self) -> None:
        op = IMem8()
        op.value = 0x00
        addr_llil = self._get_current_addr_llil(op, AddressingMode.BP_PY)
        expected_llil = mllil(
            "ADD.l",
            [
                mllil(
                    "ADD.b",
                    [
                        mllil(
                            "LOAD.b",
                            [
                                mllil(
                                    "CONST_PTR.l",
                                    [INTERNAL_MEMORY_START + IMEM_NAMES["BP"]],
                                )
                            ],
                        ),
                        mllil(
                            "LOAD.b",
                            [
                                mllil(
                                    "CONST_PTR.l",
                                    [INTERNAL_MEMORY_START + IMEM_NAMES["PY"]],
                                )
                            ],
                        ),
                    ],
                ),
                mllil("CONST.l", [INTERNAL_MEMORY_START]),
            ],
        )
        assert addr_llil == expected_llil


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
    data = bytearray([0x4D, 0x4D])
    try:
        decode(data, 0x1234)
    except ValueError as exc:
        assert str(exc) == "Failed to decode 4d4d at 0x1234"
        pass
    else:
        assert False, "Expected Exception"


def test_pre_roundtrip() -> None:
    # 3331307dec
    data = bytearray([0x33, 0x7D, 0xEC])
    instr = decode(data, 0x1234)
    assert instr._pre == 0x33
    assert asm_str(instr.render()) == "DEC   (EC)"

    encoder = Encoder()
    instr.encode(encoder, 0x1234)
    assert encoder.buf == data


def test_lift_pre() -> None:
    # no PRE: MV IMem8, Imm8
    instr = decode(bytearray([0xCC, 0xFB, 0x00]), 0xF0102)
    assert asm_str(instr.render()) == "MV    (FB), 00"
    assert instr._pre is None
    assert instr.length() == 3

    il = MockLowLevelILFunction()
    instr.lift(il, 0xF0102)
    assert il.ils == [
        mllil(
            "STORE.b",
            [
                mllil("CONST_PTR.l", [INTERNAL_MEMORY_START + 0xFB]),
                mllil("CONST.b", [0x00]),
            ],
        )
    ]


def test_cmp_with_pre() -> None:
    # PRE30 + CMP (n),(m) => first operand (n), second (BP+m)
    instr = decode(bytearray([0x30, 0xB7, 0x12, 0x34]), 0x2000)
    assert asm_str(instr.render()) == "CMP   (12), (BP+34)"
    assert instr._pre == 0x30

    il = MockLowLevelILFunction()
    instr.lift(il, 0x2000)
    assert il.ils == [
        mllil(
            "SUB.b{CZ}",
            [
                mllil("LOAD.b", [mllil("CONST_PTR.l", [INTERNAL_MEMORY_START + 0x12])]),
                mllil(
                    "LOAD.b",
                    [
                        mllil(
                            "ADD.l",
                            [
                                mllil(
                                    "ADD.b",
                                    [
                                        mllil(
                                            "LOAD.b",
                                            [mllil("CONST_PTR.l", [INTERNAL_MEMORY_START + IMEM_NAMES["BP"]])],
                                        ),
                                        mllil("CONST.b", [0x34]),
                                    ],
                                ),
                                mllil("CONST.l", [INTERNAL_MEMORY_START]),
                            ],
                        )
                    ],
                ),
            ],
        )
    ]


def test_test_with_pre() -> None:
    # PRE22 + TEST (n),00 => operand uses BP indexed addressing
    instr = decode(bytearray([0x22, 0x65, 0x12, 0x07]), 0x2000)
    assert asm_str(instr.render()) == "TEST  (BP+12), 07"
    assert instr._pre == 0x22

    il = MockLowLevelILFunction()
    instr.lift(il, 0x2000)
    assert il.ils == [
        mllil(
            "SET_FLAG",
            [
                MockFlag("Z"),
                mllil(
                    "AND.l",
                    [
                        mllil(
                            "LOAD.b",
                            [
                                mllil(
                                    "ADD.l",
                                    [
                                        mllil(
                                            "ADD.b",
                                            [
                                                mllil(
                                                    "LOAD.b",
                                                    [mllil("CONST_PTR.l", [INTERNAL_MEMORY_START + IMEM_NAMES["BP"]])],
                                                ),
                                                mllil("CONST.b", [0x12]),
                                            ],
                                        ),
                                        mllil("CONST.l", [INTERNAL_MEMORY_START]),
                                    ],
                                )
                            ],
                        ),
                        mllil("CONST.b", [0x07]),
                    ],
                ),
            ],
        )
    ]

    # PRE25 + MV IMem8, Imm8
    instr = decode(bytearray([0x25, 0xCC, 0xFB, 0x00]), 0xF0102)
    assert asm_str(instr.render()) == "MV    (BP+PX), 00"
    assert instr._pre == 0x25
    assert instr.length() == 4

    il = MockLowLevelILFunction()
    instr.lift(il, 0xF0102)
    assert il.ils == [
        mllil(
            "STORE.b",
            [
                mllil(
                    "ADD.l",
                    [
                        mllil(
                            "ADD.b",
                            [
                                mllil(
                                    "LOAD.b",
                                    [
                                        mllil(
                                            "CONST_PTR.l",
                                            [INTERNAL_MEMORY_START + 0xEC],
                                        )
                                    ],
                                ),
                                mllil(
                                    "LOAD.b",
                                    [
                                        mllil(
                                            "CONST_PTR.l",
                                            [INTERNAL_MEMORY_START + 0xED],
                                        )
                                    ],
                                ),
                            ],
                        ),
                        mllil("CONST.l", [INTERNAL_MEMORY_START]),
                    ],
                ),
                mllil("CONST.b", [0x00]),
            ],
        )
    ]


# Format:
# F90F0F00: MVW   [(0F)],(00)
def opcode_generator() -> (
    Generator[Tuple[Optional[bytearray], Optional[str]], None, None]
):
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
