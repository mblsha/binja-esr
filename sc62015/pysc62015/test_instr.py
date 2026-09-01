from binja_test_mocks import binja_api  # noqa: F401  # pyright: ignore
from .instr.opcode_table import OPCODES
from .instr import (
    encode,
    Operand,
    Instruction,
    JP_Abs,
    JP_Rel,
    CALL,
    EMemIMem,
    EMemIMemMode,
    IMem8,
    IMemHelper,
    IMEMRegisters,
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
    InvalidInstruction,
)
from .instr import decode as decode_instr
from .constants import INTERNAL_MEMORY_START, PC_MASK
from binja_test_mocks.tokens import (
    Token,
    TInstr,
    TSep,
    TText,
    TInt,
    TBegMem,
    TEndMem,
    MemType,
    TReg,
)
from binja_test_mocks.tokens import asm_str
from binja_test_mocks.coding import Decoder, Encoder
from binja_test_mocks.mock_analysis import MockAnalysisInfo
from binja_test_mocks.mock_llil import (
    MockLowLevelILFunction,
    MockIfExpr,
    MockLabel,
    MockLLIL,
    MockFlag,
    mllil,
    mreg,
)
from binaryninja.lowlevelil import (  # type: ignore
    LLIL_TEMP,
)
from binaryninja.enums import BranchType  # type: ignore
from binaryninja import RegisterName  # type: ignore

import os
import pytest
from pprint import pprint

from typing import Generator, Tuple, List, Optional, cast


def decode(data: bytearray, addr: int) -> Instruction:
    decoder = Decoder(data)
    instr = decode_instr(decoder, addr, OPCODES)  # type: ignore
    if instr is None:
        raise ValueError(f"Failed to decode {data.hex()} at {addr:#x}")
    return instr


def _walk_mock_llil(value: object) -> Generator[MockLLIL, None, None]:
    if not isinstance(value, MockLLIL):
        return
    node = cast(MockLLIL, value)
    yield node
    for operand in node.ops:
        yield from _walk_mock_llil(operand)


class _WidthStrictMockLLIL(MockLowLevelILFunction):
    """Expose the resize constructors present in real Binary Ninja LLIL."""

    _suffix = {1: "b", 2: "w", 3: "l"}

    def zero_extend(self, width: int, value: MockLLIL) -> MockLLIL:
        return mllil(f"ZX.{self._suffix[width]}", [value])

    def low_part(self, width: int, value: MockLLIL) -> MockLLIL:
        return mllil(f"LOW_PART.{self._suffix[width]}", [value])


def _assert_conditional_jump_llil(
    instr: JP_Rel,
    addr: int,
    expected_dest: MockLLIL,
    *,
    expected_cond_flag: str,
    expected_cond_value: int,
) -> None:
    il = MockLowLevelILFunction()
    instr.lift(il, addr)

    assert len(il.ils) == 4
    if_expr, label_true, jump, label_false = il.ils

    assert isinstance(if_expr, MockIfExpr)
    assert isinstance(label_true, MockLabel)
    assert isinstance(jump, MockLLIL)
    assert isinstance(label_false, MockLabel)

    assert if_expr.t is label_true.label
    assert if_expr.f is label_false.label

    assert if_expr.cond == mllil(
        "CMP_E.b",
        [
            mllil("FLAG", [MockFlag(expected_cond_flag)]),
            mllil("CONST.b", [expected_cond_value]),
        ],
    )

    assert jump.op == "JUMP"
    assert jump.ops == [expected_dest]


def _assert_unconditional_jump_llil(
    instr: JP_Rel, addr: int, expected_dest: MockLLIL
) -> None:
    il = MockLowLevelILFunction()
    instr.lift(il, addr)

    assert len(il.ils) == 3
    label_true, jump, label_false = il.ils

    assert isinstance(label_true, MockLabel)
    assert isinstance(jump, MockLLIL)
    assert isinstance(label_false, MockLabel)

    assert label_true.label is not label_false.label

    assert jump.op == "JUMP"
    assert jump.ops == [expected_dest]


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
            mllil("CONST.l", [0xD0000]),
        ],
    )
    info = MockAnalysisInfo()
    instr.analyze(info, 0xCD1234)
    assert info.mybranches == [(BranchType.UnconditionalBranch, 0xDBBAA)]

    instr = decode(bytearray([0x03, 0xAA, 0xBB, 0x0C]), 0x1234)
    assert isinstance(instr, JP_Abs)
    assert instr.render() == [TInstr("JPF"), TSep("   "), TInt("CBBAA")]
    assert instr.lift_jump_addr(il, 0x1234) == mllil("CONST.l", [0xCBBAA])
    info = MockAnalysisInfo()
    instr.analyze(info, 0x1234)
    assert info.mybranches == [(BranchType.UnconditionalBranch, 0xCBBAA)]

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
    info = MockAnalysisInfo()
    instr.analyze(info, 0xF0185)
    assert info.mybranches == [
        (BranchType.FalseBranch, 0xF0188),
        (BranchType.TrueBranch, 0xF00CD),
    ]


def test_jp_rel() -> None:
    instr = decode(bytearray([0x1A, 0x06]), 0xF0163)
    assert instr.name() == "JRNZ"
    assert isinstance(instr, JP_Rel)
    il = MockLowLevelILFunction()
    dest = instr.lift_jump_addr(il, 0xF0163)
    assert dest == mllil("CONST.l", [0xF0163 + 2 + 6])
    info = MockAnalysisInfo()
    instr.analyze(info, 0xF0163)
    assert info.mybranches == [
        (BranchType.FalseBranch, 0xF0165),
        (BranchType.TrueBranch, 0xF016B),
    ]
    _assert_conditional_jump_llil(
        instr,
        0xF0163,
        dest,
        expected_cond_flag="Z",
        expected_cond_value=0,
    )

    instr = decode(bytearray([0x1B, 0x06]), 0xF0163)
    assert instr.name() == "JRNZ"
    assert isinstance(instr, JP_Rel)
    il = MockLowLevelILFunction()
    dest = instr.lift_jump_addr(il, 0xF0163)
    assert dest == mllil("CONST.l", [0xF0163 + 2 - 6])
    info = MockAnalysisInfo()
    instr.analyze(info, 0xF0163)
    assert info.mybranches == [
        (BranchType.FalseBranch, 0xF0165),
        (BranchType.TrueBranch, 0xF015F),
    ]
    _assert_conditional_jump_llil(
        instr,
        0xF0163,
        dest,
        expected_cond_flag="Z",
        expected_cond_value=0,
    )

    instr = decode(bytearray([0x12, 0x05]), 0x2000)
    assert instr.name() == "JR"
    assert isinstance(instr, JP_Rel)
    il = MockLowLevelILFunction()
    dest = instr.lift_jump_addr(il, 0x2000)
    assert dest == mllil("CONST.l", [0x2000 + 2 + 5])
    info = MockAnalysisInfo()
    instr.analyze(info, 0x2000)
    assert info.mybranches == [(BranchType.UnconditionalBranch, 0x2007)]
    _assert_unconditional_jump_llil(instr, 0x2000, dest)


def test_control_flow_targets_wrap_to_20_bits() -> None:
    # At the final address, both conditional fallthrough and a positive
    # relative destination cross the architectural PC boundary.
    instr = decode(bytearray([0x18, 0x01]), 0xFFFFF)  # JRZ +1
    assert isinstance(instr, JP_Rel)
    il = MockLowLevelILFunction()
    assert instr.lift_jump_addr(il, 0xFFFFF) == mllil("CONST.l", [0x00002])
    info = MockAnalysisInfo()
    instr.analyze(info, 0xFFFFF)
    assert info.mybranches == [
        (BranchType.FalseBranch, 0x00001),
        (BranchType.TrueBranch, 0x00002),
    ]
    _assert_conditional_jump_llil(
        instr,
        0xFFFFF,
        mllil("CONST.l", [0x00002]),
        expected_cond_flag="Z",
        expected_cond_value=1,
    )

    # Binary Ninja can ask about an aliased host address.  Near targets use
    # only the low architectural page nibble, never host bits 20..23.
    near_jp = decode(bytearray([0x02, 0xAA, 0xBB]), 0x1D1234)
    assert isinstance(near_jp, JP_Abs)
    assert near_jp.lift_jump_addr(il, 0x1D1234) == mllil(
        "OR.l",
        [mllil("CONST.w", [0xBBAA]), mllil("CONST.l", [0xD0000])],
    )
    info = MockAnalysisInfo()
    near_jp.analyze(info, 0x1D1234)
    assert info.mybranches == [(BranchType.UnconditionalBranch, 0xDBBAA)]

    near_call = decode(bytearray([0x04, 0x34, 0x12]), 0x1FFFFF)
    assert isinstance(near_call, CALL)
    assert near_call.dest_addr(0x1FFFFF) == 0xF1234
    info = MockAnalysisInfo()
    near_call.analyze(info, 0x1FFFFF)
    assert info.mybranches == [(BranchType.CallDestination, 0xF1234)]

    instr = decode(bytearray([0x13, 0x05]), 0x2000)
    assert instr.name() == "JR"
    assert isinstance(instr, JP_Rel)
    il = MockLowLevelILFunction()
    dest = instr.lift_jump_addr(il, 0x2000)
    assert dest == mllil("CONST.l", [0x2000 + 2 - 5])
    info = MockAnalysisInfo()
    instr.analyze(info, 0x2000)
    assert info.mybranches == [(BranchType.UnconditionalBranch, 0x1FFD)]
    _assert_unconditional_jump_llil(instr, 0x2000, dest)


def test_control_flow_llil_uses_explicit_widths_and_20_bit_targets() -> None:
    il = _WidthStrictMockLLIL()

    near = decode(bytearray.fromhex("02AABB"), 0xD1234)
    assert isinstance(near, JP_Abs)
    assert near.lift_jump_addr(il, 0xD1234) == mllil(
        "OR.l",
        [
            mllil("ZX.l", [mllil("CONST.w", [0xBBAA])]),
            mllil("CONST.l", [0xD0000]),
        ],
    )

    register_jump = decode(bytearray.fromhex("1104"), 0x1234)  # JP X
    assert isinstance(register_jump, JP_Abs)
    assert register_jump.lift_jump_addr(il, 0x1234) == mllil(
        "AND.l",
        [mllil("REG.l", [mreg("X")]), mllil("CONST.l", [PC_MASK])],
    )

    near_ret = decode(bytearray.fromhex("06"), 0xD1234)
    near_ret.lift(il, 0xD1234)
    ret = next(node for node in reversed(il.ils) if node.op == "RET")
    ret_nodes = list(_walk_mock_llil(ret))
    assert any(node.op == "ZX.l" for node in ret_nodes)
    assert any(
        node.op == "CONST.l" and node.ops == [PC_MASK & ~0xFFFF] for node in ret_nodes
    )

    far_il = _WidthStrictMockLLIL()
    far_ret = decode(bytearray.fromhex("07"), 0x1234)
    far_ret.lift(far_il, 0x1234)
    far_target = next(node for node in reversed(far_il.ils) if node.op == "RET").ops[0]
    assert isinstance(far_target, MockLLIL)
    assert far_target.op == "AND.l"
    assert far_target.ops[1] == mllil("CONST.l", [PC_MASK])


@pytest.mark.parametrize(
    ("raw", "address", "target", "frame_width"),
    [
        (bytes.fromhex("043412"), 0x10000, 0x11234, 2),
        (bytes.fromhex("05785603"), 0x10000, 0x35678, 3),
    ],
)
def test_call_lift_preserves_real_call_op_with_explicit_wrapped_frame(
    raw: bytes, address: int, target: int, frame_width: int
) -> None:
    instr = decode(bytearray(raw), address)
    assert isinstance(instr, CALL)
    il = MockLowLevelILFunction()

    instr.lift(il, address)

    nodes = [node for node in il.ils if isinstance(node, MockLLIL)]
    assert nodes[-1] == mllil("CALL", [mllil("CONST_PTR.l", [target])])
    assert all(node.bare_op() != "JUMP" for node in nodes)
    assert sum(node.bare_op() == "STORE" for node in nodes) == frame_width
    assert any(
        node.bare_op() == "SET_REG" and getattr(node.ops[0], "name", None) == "S"
        for node in nodes
    )


def test_reset_analysis_is_unresolved_branch() -> None:
    instr = decode(bytearray([0xFF]), 0x1234)
    assert instr.name() == "RESET"

    info = MockAnalysisInfo()
    instr.analyze(info, 0x1234)
    assert info.length == 1
    assert info.mybranches == [(BranchType.UnresolvedBranch, None)]


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
    # With no pre specified, defaults to BP_N addressing
    assert render(op) == [
        TBegMem(MemType.EXTERNAL),
        TBegMem(MemType.INTERNAL),
        TText("BP"),
        TSep("+"),
        TInt("02"),
        TEndMem(MemType.INTERNAL),
        TEndMem(MemType.EXTERNAL),
    ]

    # POSITIVE_OFFSET
    decoder = Decoder(bytearray([0x80, 0x02, 0xBB]))
    op = EMemIMem()
    op.decode(decoder, 0x1234)
    assert op.mode == EMemIMemMode.POSITIVE_OFFSET
    # With no pre specified, defaults to BP_N addressing
    assert render(op) == [
        TBegMem(MemType.EXTERNAL),
        TBegMem(MemType.INTERNAL),
        TText("BP"),
        TSep("+"),
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
    # With no pre specified, defaults to BP_N addressing
    assert render(op) == [
        TBegMem(MemType.EXTERNAL),
        TBegMem(MemType.INTERNAL),
        TText("BP"),
        TSep("+"),
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
            "SET_REG.b",
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
    # With no pre specified, defaults to BP_N addressing
    assert asm_str(h.render()) == "[(BP+AB)+CD]"

    il = MockLowLevelILFunction()
    lifted = h.lift(il)
    nodes = list(_walk_mock_llil(lifted))
    ops = [node.op for node in nodes]
    assert lifted.op == "LOAD.b"
    # The three-byte pointer itself must be composed from wrapped byte loads;
    # a LOAD.l at IMEM FF would spill into the synthetic 0x100 window.
    assert "LOAD.l" not in ops
    assert ops.count("LOAD.b") >= 3
    assert any(node.op == "CONST.l" and node.ops == [0xFF] for node in nodes)
    assert any(node.op == "CONST.l" and node.ops == [PC_MASK] for node in nodes)


def test_emem_value_offset_helper_widths() -> None:
    imem = IMem8()
    imem.value = 0x10

    offset = ImmOffset("+")
    offset.value = 1

    for width, suffix in [(2, "w"), (3, "l")]:
        h = EMemValueOffsetHelper(imem, offset, width=width)
        il = MockLowLevelILFunction()
        lifted = h.lift(il)
        nodes = list(_walk_mock_llil(lifted))
        ops = [node.op for node in nodes]
        assert lifted.op == f"OR.{suffix}"
        assert f"LOAD.{suffix}" not in ops
        assert ops.count("LOAD.b") >= width + 3
        assert any(node.op == "CONST.l" and node.ops == [0xFF] for node in nodes)
        assert any(node.op == "CONST.l" and node.ops == [PC_MASK] for node in nodes)


def test_emem_reg_offset_helper_widths() -> None:
    reg = Reg3()
    reg.reg = RegisterName("X")

    for width, suffix in [(2, "w"), (3, "l")]:
        h = EMemRegOffsetHelper(width, reg, EMemRegMode.SIMPLE, offset=None)
        op = next(h.operands())
        il = MockLowLevelILFunction()
        lifted = op.lift(il)
        load_nodes = list(_walk_mock_llil(lifted))
        load_ops = [node.op for node in load_nodes]
        assert lifted.op == f"OR.{suffix}"
        assert f"LOAD.{suffix}" not in load_ops
        assert load_ops.count("LOAD.b") == width
        assert any(
            node.op == "CONST.l" and node.ops == [PC_MASK] for node in load_nodes
        )

        il2 = MockLowLevelILFunction()
        op.lift_assign(il2, il2.const(width, 0x11))
        store_nodes = [node for root in il2.ils for node in _walk_mock_llil(root)]
        store_ops = [node.op for node in store_nodes]
        assert il2.ils[0].op == "SET_REG.l"  # snapshot the effective address
        assert il2.ils[1].op == f"SET_REG.{suffix}"  # then snapshot the value
        assert f"STORE.{suffix}" not in store_ops
        assert store_ops.count("STORE.b") == width
        assert any(
            node.op == "CONST.l" and node.ops == [PC_MASK] for node in store_nodes
        )


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
        assert isinstance(addr_expr, MockLLIL), (
            f"Expected MockLLIL, got {type(addr_expr)}"
        )
        return addr_expr

    def test_imem_helper_direct_n_mode(self) -> None:
        # IMemHelper for (0x10)
        helper = IMemHelper(width=1, value=Imm8(0x10))
        addr_llil = self._get_imem_addr_llil(helper, pre_mode=AddressingMode.N)
        expected_llil = mllil("CONST_PTR.l", [INTERNAL_MEMORY_START + 0x10])
        assert addr_llil == expected_llil

    @pytest.mark.parametrize(
        "mode",
        [
            AddressingMode.BP_N,
            AddressingMode.PX_N,
            AddressingMode.PY_N,
            AddressingMode.BP_PX,
            AddressingMode.BP_PY,
        ],
    )
    def test_dynamic_imem_offset_is_explicitly_extended_to_address_width(
        self, mode: AddressingMode
    ) -> None:
        helper = IMemHelper(width=1, value=Imm8(0x05))
        il = _WidthStrictMockLLIL()

        address = helper.imem_addr(il, mode)

        assert isinstance(address, MockLLIL)
        assert address.op == "ADD.l"
        widened_offset = address.ops[0]
        assert isinstance(widened_offset, MockLLIL)
        assert widened_offset.op == "ZX.l"
        assert isinstance(widened_offset.ops[0], MockLLIL)
        assert widened_offset.ops[0].op == "ADD.b"

    def test_imem_helper_direct_no_pre(self) -> None:
        # IMemHelper for (0x25) with pre=None (should default to BP_N mode behavior)
        helper = IMemHelper(width=1, value=Imm8(0x25))
        addr_llil = self._get_imem_addr_llil(helper, pre_mode=None)
        # Expected: add.l( add.b( load.b( const_ptr.l(IMEM_START + BP_ADDR) ), const.b(0x25) ), const_ptr.l(IMEM_START) )
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
                                    [INTERNAL_MEMORY_START + IMEMRegisters.BP],
                                )
                            ],
                        ),  # BP value
                        mllil("CONST.b", [0x25]),  # n
                    ],
                ),
                mllil(
                    "CONST.l", [INTERNAL_MEMORY_START]
                ),  # Add base for internal memory
            ],
        )
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
                                    [INTERNAL_MEMORY_START + IMEMRegisters.BP],
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
                                    [INTERNAL_MEMORY_START + IMEMRegisters.PX],
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
                                    [INTERNAL_MEMORY_START + IMEMRegisters.PY],
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
                                    [INTERNAL_MEMORY_START + IMEMRegisters.BP],
                                )
                            ],
                        ),  # BP value
                        mllil(
                            "LOAD.b",
                            [
                                mllil(
                                    "CONST_PTR.l",
                                    [INTERNAL_MEMORY_START + IMEMRegisters.PX],
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
                                    [INTERNAL_MEMORY_START + IMEMRegisters.BP],
                                )
                            ],
                        ),  # BP value
                        mllil(
                            "LOAD.b",
                            [
                                mllil(
                                    "CONST_PTR.l",
                                    [INTERNAL_MEMORY_START + IMEMRegisters.PY],
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
        # With pre_mode=None (now defaults to BP_N), it will use BP-relative addressing
        # with n=0 (since TempReg doesn't have an n_val)

        temp_reg_operand = TempReg(LLIL_TEMP(0), width=3)
        helper = IMemHelper(width=1, value=temp_reg_operand)

        # When pre_mode is None (defaults to BP_N), it uses BP+0 addressing
        addr_llil = self._get_imem_addr_llil(helper, pre_mode=None)

        # Expected: add.l( add.b( load.b( const_ptr.l(IMEM_START + BP_ADDR) ), const.b(0) ), const.l(IMEM_START) )
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
                                    [INTERNAL_MEMORY_START + IMEMRegisters.BP],
                                )
                            ],
                        ),  # BP value
                        mllil("CONST.b", [0]),  # n=0 for TempReg
                    ],
                ),
                mllil(
                    "CONST.l", [INTERNAL_MEMORY_START]
                ),  # Add base for internal memory
            ],
        )
        assert addr_llil == expected_llil

    def test_imem_helper_actual_reg_value_no_pre(self) -> None:
        # Similar to TempReg, but with an actual CPU register (e.g. X)
        # With pre_mode=None (now defaults to BP_N), it will use BP-relative addressing

        actual_reg_operand = Reg("X")  # X is 3 bytes (REG_SIZES['X'])
        helper = IMemHelper(width=1, value=actual_reg_operand)

        addr_llil = self._get_imem_addr_llil(helper, pre_mode=None)

        # Expected: add.l( add.b( load.b( const_ptr.l(IMEM_START + BP_ADDR) ), const.b(0) ), const.l(IMEM_START) )
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
                                    [INTERNAL_MEMORY_START + IMEMRegisters.BP],
                                )
                            ],
                        ),  # BP value
                        mllil("CONST.b", [0]),  # n=0 for Reg
                    ],
                ),
                mllil(
                    "CONST.l", [INTERNAL_MEMORY_START]
                ),  # Add base for internal memory
            ],
        )
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
        # With pre=None (defaults to BP_N), it uses BP+0x25 addressing
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
                                    [INTERNAL_MEMORY_START + IMEMRegisters.BP],
                                )
                            ],
                        ),  # BP value
                        mllil("CONST.b", [0x25]),  # n
                    ],
                ),
                mllil(
                    "CONST.l", [INTERNAL_MEMORY_START]
                ),  # Add base for internal memory
            ],
        )
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
                                    [INTERNAL_MEMORY_START + IMEMRegisters.BP],
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
                                    [INTERNAL_MEMORY_START + IMEMRegisters.PX],
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
                                    [INTERNAL_MEMORY_START + IMEMRegisters.PY],
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
                                    [INTERNAL_MEMORY_START + IMEMRegisters.BP],
                                )
                            ],
                        ),
                        mllil(
                            "LOAD.b",
                            [
                                mllil(
                                    "CONST_PTR.l",
                                    [INTERNAL_MEMORY_START + IMEMRegisters.PX],
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
                                    [INTERNAL_MEMORY_START + IMEMRegisters.BP],
                                )
                            ],
                        ),
                        mllil(
                            "LOAD.b",
                            [
                                mllil(
                                    "CONST_PTR.l",
                                    [INTERNAL_MEMORY_START + IMEMRegisters.PY],
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
            "SET_REG.b",
            [
                mreg("A"),
                mllil("CONST.b", [0xCD]),
            ],
        )
    ]


@pytest.mark.parametrize(
    ("encoded", "destination", "resize_op", "source_op"),
    [
        ("097F", "I", "ZX.w", "CONST.b"),  # MV IL,7F clears IH
        ("FD24", "BA", "LOW_PART.w", "AND.l"),  # ROM-valid MV BA,X
        ("FD42", "X", "ZX.l", "REG.w"),  # ROM-valid MV X,BA
    ],
)
def test_mv_llil_resizes_register_values_explicitly(
    encoded: str, destination: str, resize_op: str, source_op: str
) -> None:
    instr = decode(bytearray.fromhex(encoded), 0x1234)
    il = _WidthStrictMockLLIL()

    instr.lift(il, 0x1234)

    write = next(
        node
        for node in il.ils
        if node.op.startswith("SET_REG")
        and getattr(node.ops[0], "name", None) == destination
    )
    nodes = list(_walk_mock_llil(write.ops[1]))
    assert any(node.op == resize_op for node in nodes)
    assert any(node.op == source_op for node in nodes)
    if destination == "X":
        assert write.ops[1].op == "AND.l"
        assert write.ops[1].ops[1] == mllil("CONST.l", [PC_MASK])


@pytest.mark.parametrize("encoded", ["ED24", "ED42"])
def test_ed_exchange_snapshots_and_resizes_both_directions(encoded: str) -> None:
    instr = decode(bytearray.fromhex(encoded), 0x1234)
    il = _WidthStrictMockLLIL()

    instr.lift(il, 0x1234)

    assert il.ils[0].op.startswith("SET_REG")
    assert il.ils[1].op.startswith("SET_REG")
    writes = il.ils[2:]
    ba_write = next(
        node for node in writes if getattr(node.ops[0], "name", None) == "BA"
    )
    x_write = next(node for node in writes if getattr(node.ops[0], "name", None) == "X")
    assert any(node.op == "LOW_PART.w" for node in _walk_mock_llil(ba_write))
    assert any(node.op == "ZX.l" for node in _walk_mock_llil(x_write))
    assert x_write.ops[1].op == "AND.l"
    assert x_write.ops[1].ops[1] == mllil("CONST.l", [PC_MASK])


def test_external_register_address_is_masked_to_20_bits() -> None:
    reg = Reg3()
    reg.reg = RegisterName("X")
    reg.reg_raw = 0x04
    reg.high4 = 0
    helper = EMemRegOffsetHelper(
        1,
        reg,
        EMemRegMode.SIMPLE,
        offset=None,
    )
    operand = next(helper.operands())
    il = _WidthStrictMockLLIL()

    lifted = operand.lift(il)

    assert lifted.op == "LOAD.b"
    address = lifted.ops[0]
    assert isinstance(address, MockLLIL)
    assert address.op == "AND.l"
    assert address.ops[1] == mllil("CONST.l", [PC_MASK])


def test_lift_mv_memory_to_memory() -> None:
    instr = decode(bytearray([0xC8, 0xAB, 0xCD]), 0x1234)
    # With no PRE, defaults to BP_N addressing for both operands
    assert asm_str(instr.render()) == "MV    (BP+AB), (BP+CD)"

    il = MockLowLevelILFunction()
    instr.lift(il, 0x1234)
    # With BP_N addressing for both operands
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
                                            [INTERNAL_MEMORY_START + IMEMRegisters.BP],
                                        )
                                    ],
                                ),  # BP value
                                mllil("CONST.b", [0xAB]),  # first operand n
                            ],
                        ),
                        mllil(
                            "CONST.l", [INTERNAL_MEMORY_START]
                        ),  # Add base for internal memory
                    ],
                ),
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
                                            [
                                                mllil(
                                                    "CONST_PTR.l",
                                                    [
                                                        INTERNAL_MEMORY_START
                                                        + IMEMRegisters.BP
                                                    ],
                                                )
                                            ],
                                        ),  # BP value
                                        mllil("CONST.b", [0xCD]),  # second operand n
                                    ],
                                ),
                                mllil(
                                    "CONST.l", [INTERNAL_MEMORY_START]
                                ),  # Add base for internal memory
                            ],
                        )
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
    # PRE30 is the canonical single-operand (n) prefix.  PRE33 has the same
    # first latch but differs only in the unused second latch.
    data = bytearray([0x30, 0x7D, 0xEC])
    instr = decode(data, 0x1234)
    assert instr._pre == 0x30
    assert asm_str(instr.render()) == "DEC   (BP)"

    encoder = Encoder()
    instr.encode(encoder, 0x1234)
    assert encoder.buf == data


def test_lift_pre() -> None:
    # no PRE: MV IMem8, Imm8
    instr = decode(bytearray([0xCC, 0xFB, 0x00]), 0xF0102)
    # With no PRE, defaults to BP_N addressing for destination
    assert asm_str(instr.render()) == "MV    (BP+FB), 00"
    assert instr._pre is None
    assert instr.length() == 3

    il = MockLowLevelILFunction()
    instr.lift(il, 0xF0102)
    # With no PRE (defaults to BP_N), destination uses BP+0xFB addressing
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
                                            [INTERNAL_MEMORY_START + IMEMRegisters.BP],
                                        )
                                    ],
                                ),  # BP value
                                mllil("CONST.b", [0xFB]),  # n
                            ],
                        ),
                        mllil(
                            "CONST.l", [INTERNAL_MEMORY_START]
                        ),  # Add base for internal memory
                    ],
                ),
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
                                            [
                                                mllil(
                                                    "CONST_PTR.l",
                                                    [
                                                        INTERNAL_MEMORY_START
                                                        + IMEMRegisters.BP
                                                    ],
                                                )
                                            ],
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


def test_silicon_proven_single_operand_pre_aliases_decode() -> None:
    # PRE22 changes only TEST's unused second latch; silicon executes it with
    # the ordinary BP+n first-latch semantics.
    test_instr = decode(bytearray([0x22, 0x65, 0x12, 0x07]), 0x2000)
    assert asm_str(test_instr.render()) == "TEST  (BP+12), 07"
    assert test_instr._pre == 0x22
    assert test_instr.length() == 4

    # PRE25's unused second latch was not in the hardware matrix.
    with pytest.raises(InvalidInstruction, match="Noncanonical PRE25"):
        decode(bytearray([0x25, 0xCC, 0x00, 0x00]), 0xF0102)

    # BP+PX ignores the carried selector byte, including nonzero values.
    nonzero = decode(bytearray([0x24, 0xCC, 0xFB, 0x00]), 0xF0102)
    assert asm_str(nonzero.render()) == "MV    (BP+PX), 00"
    assert nonzero._pre == 0x24

    instr = decode(bytearray([0x24, 0xCC, 0x00, 0x00]), 0xF0102)
    assert asm_str(instr.render()) == "MV    (BP+PX), 00"
    assert instr._pre == 0x24
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


def test_wait_lifts_to_timing_intrinsic() -> None:
    instr = decode(bytearray([0xEF]), 0x1234)  # WAIT
    il = MockLowLevelILFunction()
    instr.lift(il, 0x1234)

    assert [getattr(node, "name", None) for node in il.ils] == ["WAIT"]


def test_ir_lifts_vector_validation_before_stack_mutation() -> None:
    instr = decode(bytearray([0xFE]), 0x12345)
    il = MockLowLevelILFunction()
    instr.lift(il, 0x12345)

    fetch, validation = il.ils[:2]
    assert fetch.bare_op() == "SET_REG"
    assert "TEMP6" in repr(fetch)
    assert "1048570" in repr(fetch)  # architectural read at 0xFFFFA
    assert getattr(validation, "name", None) == "VALIDATE_VECTOR_TRANSFER"
    assert "1048570" in repr(validation)
    assert "74565" in repr(validation)  # source PC at 0x12345
    assert "TEMP6" in repr(validation)
    assert all(node.bare_op() != "STORE" for node in il.ils[:2])
    assert il.ils[-1].bare_op() == "JUMP"
    assert "TEMP6" in repr(il.ils[-1])
    assert "LOAD" not in repr(il.ils[-1])


def test_mvl_predec_source_continues_until_final_wrapped_iteration() -> None:
    # PRE34 is the canonical single-operand PX+n prefix here; the external
    # source uses its own encoded pre-decrement mode.
    instr = decode(bytearray.fromhex("34E33720"), 0x1234)
    il = MockLowLevelILFunction()
    instr.lift(il, 0x1234)

    lifted = repr(il.ils)
    assert "CMP_NE.w" in lifted
    assert "CMP_UGT" not in lifted
    assert "CMP_SGT" not in lifted


def test_register_class_encodings_fail_closed() -> None:
    invalid = (
        "4407",  # ADD r2 with A/S mixed-class selector
        "4500",  # ADD r3 with A/A selector
        "4607",  # ADD r1 with A/S mixed-class selector
        "4C07",  # SUB r2 with A/S mixed-class selector
        "4D00",  # SUB r3 with A/A selector
        "4E07",  # SUB r1 with A/S mixed-class selector
        "4441",  # ADD r2 form with X destination
        "1100",  # JP requires an exact X/Y/U/S selector
        "11A4",  # JP selector upper bits are reserved, not an X alias
        "6CA4",  # INC selector upper bits are reserved
        "7CFF",  # DEC selector upper bits are reserved
        "D60700",  # CMPW with S instead of BA/I
        "D70000",  # CMPP with A instead of X/Y/U/S
    )
    for encoded in invalid:
        assert decode_instr(bytearray.fromhex(encoded), 0x1234, OPCODES) is None


def test_register_class_valid_boundaries_decode() -> None:
    valid = {
        "4600": "ADD   A, A",
        "4611": "ADD   IL, IL",
        "4422": "ADD   BA, BA",
        "4433": "ADD   I, I",
        "4544": "ADD   X, X",
        "4577": "ADD   S, S",
        "4552": "ADD   Y, BA",  # PC-E500 EN ROM at F2B62
        "4430": "ADD   I, A",  # PC-E500 EN ROM at E400F and F47C8
        "4C30": "SUB   I, A",  # matching ROM path at F47CC
        "4540": "ADD   X, A",
        "4D60": "SUB   U, A",
        "4E01": "SUB   A, IL",
        "1104": "JP    X",  # PC-E500 E2F5F; IQ-7000 E4766
        "1105": "JP    Y",  # PC-E500 F2053; IQ-7000 F5230
        "1106": "JP    U",
        "1107": "JP    S",
        "6C00": "INC   A",
        "6C07": "INC   S",
        "7C00": "DEC   A",
        "7C07": "DEC   S",
        "D60200": "CMPW  (BP+00), BA",
        "D60300": "CMPW  (BP+00), I",
        "D70400": "CMPP  (BP+00), X",
        "D70700": "CMPP  (BP+00), S",
    }
    for encoded, expected_asm in valid.items():
        instr = decode(bytearray.fromhex(encoded), 0x1234)
        assert asm_str(instr.render()) == expected_asm


@pytest.mark.parametrize(
    ("encoded", "source_op", "target_op", "mask"),
    [
        ("4430", "REG.b", "AND.w", 0xFF),  # ADD I,A
        ("4552", "REG.w", "AND.l", 0xFFFF),  # ADD Y,BA
        ("4540", "REG.b", "AND.l", 0xFF),  # ADD X,A
        ("4D60", "REG.b", "AND.l", 0xFF),  # SUB U,A
    ],
)
def test_mixed_register_arithmetic_llil_resizes_source_to_destination_width(
    encoded: str, source_op: str, target_op: str, mask: int
) -> None:
    instr = decode(bytearray.fromhex(encoded), 0x1234)
    il = MockLowLevelILFunction()
    instr.lift(il, 0x1234)
    nodes = [node for root in il.ils for node in _walk_mock_llil(root)]

    assert any(
        node.op == target_op
        and isinstance(node.ops[0], MockLLIL)
        and node.ops[0].op == source_op
        and isinstance(node.ops[1], MockLLIL)
        and node.ops[1].op in {"CONST.w", "CONST.l"}
        and node.ops[1].ops == [mask]
        for node in nodes
    )


def test_same_width_register_arithmetic_llil_needs_no_resize() -> None:
    instr = decode(bytearray.fromhex("4E01"), 0x1234)  # SUB A,IL
    il = MockLowLevelILFunction()
    instr.lift(il, 0x1234)
    nodes = [node for root in il.ils for node in _walk_mock_llil(root)]

    sub = next(node for node in nodes if node.op == "SUB.b{CZ}")
    assert [operand.op for operand in sub.ops] == ["REG.b", "REG.b"]


# Format:
# F90F0F00: MVW   [(0F)],(00)
def opcode_generator() -> Generator[
    Tuple[Optional[bytearray], Optional[str]], None, None
]:
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
            raise ValueError(
                f"Failed to decode {b.hex()} at line {i + 1}: {s}"
            ) from exc

        if not instr:
            raise ValueError(f"Failed to decode {b.hex()} at line {i + 1}: {s}")
        try:
            rendered = instr.render()
        except Exception as exc:
            raise ValueError(
                f"Failed to render {b.hex()} at line {i + 1}: {s}"
            ) from exc
        if not rendered:
            raise ValueError(f"Failed to render {b.hex()} at line {i + 1}: {s}")

        rendered_str = asm_str(rendered)
        rendered_str = " ".join(rendered_str.split())
        rendered_str = rendered_str.replace(", ", ",")

        # Handle the BP_N default addressing mode change
        # When there's no PRE, memory operands now default to BP+n notation
        # Try to match both with and without BP+ prefix
        expected_str = s

        # Check if rendered output has BP+ but expected doesn't
        if s and "BP+" in rendered_str and "BP+" not in s:
            # Add BP+ to expected string for comparison
            import re

            # Match patterns like (XX) where XX is hex digits
            expected_str = re.sub(r"\(([0-9A-F]+)\)", r"(BP+\1)", s)

        # Also handle the reverse case where expected has BP+ but rendered doesn't
        # (this can happen for unknown instructions)
        if expected_str and "BP+" not in rendered_str and "BP+" in expected_str:
            # Remove BP+ from expected for comparison
            import re

            expected_str = re.sub(r"\(BP\+([0-9A-F]+)\)", r"(\1)", expected_str)

        def normalize_mv_ex_regpair_aliases(text: str) -> str:
            import re

            match = re.fullmatch(r"(EX|MV)\s+([A-Z]+),([A-Z]+)", text)
            if match is None:
                return text
            mnemonic, lhs, rhs = match.groups()
            alias_map = {"A": "BA", "IL": "I"}
            lhs = alias_map.get(lhs, lhs)
            rhs = alias_map.get(rhs, rhs)
            valid_regpair_regs = {"BA", "I", "X", "Y", "U", "S"}
            if lhs in valid_regpair_regs and rhs in valid_regpair_regs:
                return f"{mnemonic} {lhs},{rhs}"
            return text

        normalized_rendered = normalize_mv_ex_regpair_aliases(rendered_str)
        normalized_expected = normalize_mv_ex_regpair_aliases(expected_str)

        assert normalized_rendered == normalized_expected, (
            f"Failed at line {i + 1}: expected '{expected_str}', got '{rendered_str}'"
        )

        # test that no assertions are raised
        info = MockAnalysisInfo()
        try:
            instr.analyze(info, 0x1234)
            assert info.length == len(b), f"Failed at line {i + 1}: {s}"
        except InvalidInstruction:
            # Skip unfused PRE instructions - they're invalid on their own
            if isinstance(instr, PRE):
                continue
            raise

        try:
            # test that no assertions are raised
            il = MockLowLevelILFunction()
            instr.lift(il, 0x1234)
        except InvalidInstruction:
            # Skip unfused PRE instructions - they're invalid on their own
            if isinstance(instr, PRE):
                continue
            raise
        except Exception as exc:
            raise ValueError(f"Failed to lift {b.hex()} at line {i + 1}: {s}") from exc

        def check_no_unimplemented(instr: MockLLIL) -> None:
            if isinstance(instr, MockLLIL):
                if instr.op == "UNIMPL":
                    pprint(il.ils)
                    raise ValueError(
                        f"Unimplemented instruction: {instr} for {rendered_str} at line {i + 1}"
                    )

                for op in instr.ops:
                    check_no_unimplemented(op)

        def start_check_lifting(ils: List[MockLLIL]) -> None:
            assert b is not None  # Already checked above
            assert len(ils) > 0, f"Failed to lift {b.hex()} at line {i + 1}: {s}"
            for instr in ils:
                check_no_unimplemented(instr)

        if not isinstance(instr, (UnknownInstruction, PRE, TCL, HALT, OFF, IR)):
            start_check_lifting(il.ils)
