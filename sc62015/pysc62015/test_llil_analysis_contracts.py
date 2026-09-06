"""Lifting contracts which native/mock numeric parity alone cannot establish."""

import pytest
from binja_test_mocks.mock_llil import MockIntrinsic, MockLowLevelILFunction

from .instr import OPCODES, decode
from . import CPU, RegisterName
from .constants import ADDRESS_SPACE_SIZE, INTERNAL_MEMORY_START
from binja_test_mocks.eval_llil import Memory


@pytest.mark.parametrize(
    "raw", ["e4", "e520", "e6", "e720", "f4", "f520", "f6", "f720"]
)
def test_bit_rotations_explicitly_define_carry_and_zero(raw: str) -> None:
    instruction = decode(bytes.fromhex(raw), 0x1000, OPCODES)
    assert instruction is not None
    il = MockLowLevelILFunction()
    instruction.lift(il, 0x1000)
    written = {str(node.ops[0].name) for node in il.ils if node.bare_op() == "SET_FLAG"}
    # Binary Ninja has no default carry formula for ROL/ROR/RLC. Root-level
    # explicit assignments are also easy for both evaluators to audit.
    assert written == {"C", "Z"}


def test_wait_exposes_i_input_and_clear_to_real_analysis() -> None:
    instruction = decode(bytes.fromhex("ef"), 0x1000, OPCODES)
    assert instruction is not None
    il = MockLowLevelILFunction()
    instruction.lift(il, 0x1000)
    assert len(il.ils) == 2
    intrinsic, clear = il.ils
    assert isinstance(intrinsic, MockIntrinsic)
    assert intrinsic.name == "WAIT"
    assert len(intrinsic.params) == 1
    assert intrinsic.params[0].bare_op() == "REG"
    assert intrinsic.params[0].ops[0].name == "I"
    assert clear.bare_op() == "SET_REG"
    assert clear.ops[0].name == "I"
    assert clear.ops[1].constant == 0


def test_reset_exposes_vector_result_and_terminates_llil_block() -> None:
    instruction = decode(bytes.fromhex("ff"), 0x1000, OPCODES)
    assert instruction is not None
    il = MockLowLevelILFunction()
    instruction.lift(il, 0x1000)
    intrinsic, transfer = il.ils
    assert isinstance(intrinsic, MockIntrinsic)
    assert intrinsic.name == "RESET"
    assert [str(output) for output in intrinsic.outputs] == ["PC"]
    assert transfer.bare_op() == "JUMP"
    assert "PC" in repr(transfer)


@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize("opcode", [0xE4, 0xE5, 0xE6, 0xE7, 0xF4, 0xF5, 0xF6, 0xF7])
def test_bit_rotations_exhaustive_byte_and_flag_reference(
    backend: str, opcode: int
) -> None:
    # Independent arithmetic reference for the existing one-bit contract.
    # This is software/model evidence, not a new hardware capture.
    backing = bytearray(ADDRESS_SPACE_SIZE)
    backing[0x1000:0x1002] = bytes((opcode, 0x20))
    memory = Memory(backing.__getitem__, backing.__setitem__)
    setattr(
        memory, "peek_byte_for_preflight", lambda address, _pc=None: backing[address]
    )
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    indirect = bool(opcode & 1)
    left = bool(opcode & 2)
    through_carry = opcode >= 0xF0
    for value in range(256):
        for initial_f in range(4):
            cpu.regs.set(RegisterName.PC, 0x1000)
            cpu.regs.set(RegisterName.F, initial_f)
            cpu.regs.set(RegisterName.A, value)
            backing[INTERNAL_MEMORY_START + 0x20] = value
            outgoing = value >> 7 if left else value & 1
            inserted = initial_f & 1 if through_carry else outgoing
            expected = (
                ((value << 1) | inserted) & 0xFF
                if left
                else (value >> 1) | (inserted << 7)
            )
            cpu.execute_instruction(0x1000)
            actual = (
                backing[INTERNAL_MEMORY_START + 0x20]
                if indirect
                else cpu.regs.get(RegisterName.A)
            )
            assert actual == expected, (backend, opcode, value, initial_f)
            assert cpu.regs.get(RegisterName.F) == outgoing | (int(expected == 0) << 1)
