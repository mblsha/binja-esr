from typing import Any

import pytest
from binja_test_mocks.coding import Decoder, Encoder
from binaryninja import RegisterName  # type: ignore

from .instr import (
    EMemIMem,
    EMemIMemMode,
    EMemIMemOffset,
    EMemIMemOffsetOrder,
    EMemReg,
    EMemRegMode,
    IMem8,
    ImmOffset,
    InvalidInstruction,
    Reg3,
    RegIMemOffset,
    RegIMemOffsetOrder,
    decode,
    encode,
)
from .instr.opcode_table import OPCODES
from .sc_asm import Assembler, AssemblerError


def _decode(raw: str):  # type: ignore[no-untyped-def]
    instr = decode(Decoder(bytearray.fromhex(raw)), 0, OPCODES)
    assert instr is not None
    return instr


def _pointer_reg(mode: EMemRegMode) -> Reg3:
    reg = Reg3()
    reg.reg = RegisterName("X")
    reg.reg_raw = 0x04
    reg.high4 = mode.value
    return reg


def _offset(sign: str = "+") -> ImmOffset:
    offset = ImmOffset(sign)  # type: ignore[arg-type]
    offset.value = 1
    return offset


def test_direct_encoder_rejects_mutated_noncanonical_pre() -> None:
    instr = _decode("308005")
    instr._pre = 0x32
    with pytest.raises(InvalidInstruction, match="Noncanonical PRE32"):
        encode(instr, 0)


def test_direct_encoder_rejects_pre_on_insensitive_instruction() -> None:
    instr = _decode("00")
    instr._pre = 0x30
    with pytest.raises(InvalidInstruction, match="PRE30 cannot prefix PRE-insensitive"):
        encode(instr, 0)


@pytest.mark.parametrize("missing", [False, True], ids=["extra", "missing"])
def test_emem_reg_offset_arity_is_exact(missing: bool) -> None:
    mode = EMemRegMode.POSITIVE_OFFSET if missing else EMemRegMode.SIMPLE
    operand = EMemReg(width=1)
    operand.reg = _pointer_reg(mode)
    operand.mode = mode
    operand.offset = None if missing else _offset()
    with pytest.raises(InvalidInstruction, match="requires one offset|must not encode"):
        operand.encode(Encoder(), 0)


@pytest.mark.parametrize("missing", [False, True], ids=["extra", "missing"])
def test_reg_imem_offset_arity_is_exact(missing: bool) -> None:
    mode = EMemRegMode.POSITIVE_OFFSET if missing else EMemRegMode.SIMPLE
    operand = RegIMemOffset(RegIMemOffsetOrder.DEST_IMEM)
    operand.reg = _pointer_reg(mode)
    operand.imem = IMem8(0x20)
    operand.mode = mode
    operand.offset = None if missing else _offset()
    with pytest.raises(InvalidInstruction, match="requires one offset|must not encode"):
        operand.encode(Encoder(), 0)


@pytest.mark.parametrize("composite", [False, True], ids=["direct", "composite"])
def test_emem_imem_mode_byte_must_match_metadata(composite: bool) -> None:
    if composite:
        operand = EMemIMemOffset(EMemIMemOffsetOrder.DEST_INT_MEM)
        operand.mode = EMemIMemMode.POSITIVE_OFFSET
        operand.mode_imm.value = EMemIMemMode.SIMPLE.value
        operand.imem1.value = 0x20
        operand.imem2.value = 0x30
    else:
        operand = EMemIMem()
        operand.mode = EMemIMemMode.POSITIVE_OFFSET
        operand.value = EMemIMemMode.SIMPLE.value
        operand.imem = IMem8(0x20)
    operand.offset = _offset()
    with pytest.raises(InvalidInstruction, match="does not match encoded mode byte"):
        operand.encode(Encoder(), 0)


@pytest.mark.parametrize("composite", [False, True], ids=["direct", "composite"])
@pytest.mark.parametrize("missing", [False, True], ids=["extra", "missing"])
def test_emem_imem_offset_arity_is_exact(composite: bool, missing: bool) -> None:
    mode = EMemIMemMode.POSITIVE_OFFSET if missing else EMemIMemMode.SIMPLE
    if composite:
        operand = EMemIMemOffset(EMemIMemOffsetOrder.DEST_INT_MEM)
        operand.mode = mode
        operand.mode_imm.value = mode.value
        operand.imem1.value = 0x20
        operand.imem2.value = 0x30
    else:
        operand = EMemIMem()
        operand.mode = mode
        operand.value = mode.value
        operand.imem = IMem8(0x20)
    operand.offset = None if missing else _offset()
    with pytest.raises(InvalidInstruction, match="requires one offset|must not encode"):
        operand.encode(Encoder(), 0)


def test_imm20_encoder_rejects_value_high_nibble_disagreement() -> None:
    instr = _decode("03341205")
    (target,) = tuple(instr.operands())
    setattr(target, "value", 0x61234)
    with pytest.raises(InvalidInstruction, match="disagrees with encoded high byte"):
        encode(instr, 0)


def test_reg3_encoder_rejects_semantic_selector_disagreement() -> None:
    instr = _decode("1104")
    (target,) = tuple(instr.operands())
    setattr(target, "reg", RegisterName("Y"))
    with pytest.raises(InvalidInstruction, match="encodes X but operand names Y"):
        encode(instr, 0)


def test_regpair_encoder_rejects_semantic_selector_disagreement() -> None:
    instr = _decode("4545")
    pair: Any = instr._operands[0]
    pair.reg1.reg = RegisterName("Y")
    with pytest.raises(InvalidInstruction, match="but operand names Y,Y"):
        encode(instr, 0)


@pytest.mark.parametrize(
    "source",
    [
        "DEFB -1",
        "DEFB 0x100",
        "DEFW -1",
        "DEFW 0x10000",
        "DEFL -1",
        "DEFL 0x1000000",
    ],
)
def test_data_directives_reject_values_that_do_not_fit(source: str) -> None:
    with pytest.raises(AssemblerError, match="outside the unsigned"):
        Assembler().assemble(source)


def test_org_rejects_addresses_outside_the_20_bit_space() -> None:
    with pytest.raises(AssemblerError, match="outside the 20-bit range"):
        Assembler().assemble(".ORG 0x100000\nNOP")


def test_statement_cannot_cross_the_20_bit_space_boundary() -> None:
    with pytest.raises(AssemblerError, match="crosses the 20-bit address-space"):
        Assembler().assemble(".ORG 0xFFFFF\nMV A, 1")


def test_quoted_defb_string_does_not_resolve_as_same_named_label() -> None:
    encoded = Assembler().assemble('A: NOP\nDEFB "A"').as_binary()
    assert encoded == bytearray((0x00, 0x41))
