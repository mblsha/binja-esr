from __future__ import annotations

from typing import cast

import pytest

from binja_test_mocks.eval_llil import Memory

from sc62015.pysc62015 import CPU, RegisterName, available_backends
from sc62015.pysc62015.constants import ADDRESS_SPACE_SIZE, INTERNAL_MEMORY_START
from sc62015.pysc62015.instr.opcodes import IMEMRegisters


class MemoryWithRaw(Memory):
    _raw: bytearray


def _make_memory(
    imr: int, f: int, ret_bytes: tuple[int, int, int], sp: int
) -> MemoryWithRaw:
    raw = bytearray(ADDRESS_SPACE_SIZE)
    raw[0] = 0x01  # RETI opcode
    raw[sp] = imr & 0xFF
    raw[sp + 1] = f & 0xFF
    raw[sp + 2] = ret_bytes[0] & 0xFF
    raw[sp + 3] = ret_bytes[1] & 0xFF
    raw[sp + 4] = ret_bytes[2] & 0xFF

    def read(addr: int) -> int:
        if addr < 0 or addr >= len(raw):
            raise IndexError(f"Read address {addr:#x} out of bounds")
        return raw[addr]

    def write(addr: int, value: int) -> None:
        if addr < 0 or addr >= len(raw):
            raise IndexError(f"Write address {addr:#x} out of bounds")
        raw[addr] = value & 0xFF

    memory = Memory(read, write)
    setattr(memory, "_raw", raw)
    setattr(
        memory,
        "peek_byte_for_preflight",
        lambda address, _pc=None: raw[address & 0xFFFFFF],
    )
    setattr(memory, "instruction_byte_is_callback_free", lambda _address: True)
    setattr(memory, "vector_transfer_provenance", lambda: (id(memory), 0))
    return cast(MemoryWithRaw, memory)


@pytest.mark.parametrize("backend", ["python", "llama"])
def test_reti_restores_imr_exactly(backend: str) -> None:
    if backend == "llama":
        assert "llama" in available_backends(), "LLAMA backend not available"

    sp = 0x0100
    imr_saved = 0x00  # deliberately clear IRM bit to detect forced setting
    f_saved = 0x03
    ret_bytes = (0x12, 0x34, 0x05)  # little-endian PC
    memory = _make_memory(imr_saved, f_saved, ret_bytes, sp)
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.PC, 0x0000)
    cpu.regs.set(RegisterName.S, sp)

    cpu.execute_instruction(0x0000)

    expected_pc = ret_bytes[0] | (ret_bytes[1] << 8) | (ret_bytes[2] << 16)
    imr_addr = INTERNAL_MEMORY_START + IMEMRegisters.IMR
    assert memory._raw[imr_addr] == imr_saved
    assert cpu.regs.get(RegisterName.PC) == expected_pc
    assert cpu.regs.get(RegisterName.S) == (sp + 5)
    assert cpu.regs.get(RegisterName.F) == f_saved
    assert cpu.regs.get(RegisterName.FC) == (f_saved & 0x01)
    assert cpu.regs.get(RegisterName.FZ) == ((f_saved >> 1) & 0x01)


@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize("f_input", [0x00, 0x01, 0x02, 0x03])
@pytest.mark.parametrize(
    ("opcode", "stack_register"),
    [
        (0x2E, RegisterName.U),  # PUSHU F
        (0x4F, RegisterName.S),  # PUSHS F
    ],
)
def test_push_f_preserves_modeled_images(
    backend: str, f_input: int, opcode: int, stack_register: RegisterName
) -> None:
    if backend == "llama":
        assert "llama" in available_backends(), "LLAMA backend not available"

    memory = _make_memory(0, 0, (0, 0, 0), 0x100)
    memory._raw[0] = opcode
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.F, f_input)
    cpu.regs.set(stack_register, 0x100)

    cpu.execute_instruction(0)

    assert cpu.regs.get(RegisterName.F) == f_input
    assert cpu.regs.get(stack_register) == 0xFF
    assert memory._raw[0xFF] == f_input


@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize("f_input", [0x00, 0x01, 0x02, 0x03])
@pytest.mark.parametrize(
    ("opcode", "stack_register"),
    [
        (0x3E, RegisterName.U),  # POPU F
        (0x5F, RegisterName.S),  # POPS F
    ],
)
def test_pop_f_preserves_modeled_images(
    backend: str, f_input: int, opcode: int, stack_register: RegisterName
) -> None:
    if backend == "llama":
        assert "llama" in available_backends(), "LLAMA backend not available"

    memory = _make_memory(0, 0, (0, 0, 0), 0x100)
    memory._raw[0] = opcode
    memory._raw[0x100] = f_input
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    cpu.regs.set(stack_register, 0x100)

    cpu.execute_instruction(0)

    assert cpu.regs.get(RegisterName.F) == f_input
    assert cpu.regs.get(RegisterName.FC) == (f_input & 0x01)
    assert cpu.regs.get(RegisterName.FZ) == ((f_input >> 1) & 0x01)
    assert cpu.regs.get(stack_register) == 0x101


@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize("f_input", [0x00, 0x01, 0x02, 0x03])
def test_ir_stacks_modeled_f_image(backend: str, f_input: int) -> None:
    if backend == "llama":
        assert "llama" in available_backends(), "LLAMA backend not available"

    memory = _make_memory(0, 0, (0, 0, 0), 0x100)
    memory._raw[0] = 0xFE  # IR
    memory._raw[0xFFFFA:0xFFFFD] = bytes.fromhex("452301")
    imr_addr = INTERNAL_MEMORY_START + IMEMRegisters.IMR
    memory._raw[imr_addr] = 0xA5
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.S, 0x105)
    cpu.regs.set(RegisterName.F, f_input)

    cpu.prepare_instruction_before_scheduling(0)
    cpu.execute_instruction(0)

    assert cpu.regs.get(RegisterName.S) == 0x100
    assert memory._raw[0x100:0x105] == bytes([0xA5, f_input, 0x00, 0x00, 0x00])
    assert cpu.regs.get(RegisterName.PC) == 0x12345
    assert memory._raw[imr_addr] == 0x25


@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize("f_input", [0x04, 0x80, 0xA4, 0xFC, 0xFF])
@pytest.mark.parametrize(
    ("opcode", "stack_register"),
    [
        (0x3E, RegisterName.U),  # POPU F
        (0x5F, RegisterName.S),  # POPS F
    ],
)
def test_pop_f_rejects_unverified_image_before_advancing_stack(
    backend: str, f_input: int, opcode: int, stack_register: RegisterName
) -> None:
    if backend == "llama":
        assert "llama" in available_backends(), "LLAMA backend not available"

    memory = _make_memory(0, 0, (0, 0, 0), 0x100)
    memory._raw[0] = opcode
    memory._raw[0x100] = f_input
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    cpu.regs.set(stack_register, 0x100)
    cpu.regs.set(RegisterName.F, 0x01)

    with pytest.raises(RuntimeError, match="bits 2-7 require real-hardware tracing"):
        cpu.execute_instruction(0)

    assert cpu.regs.get(stack_register) == 0x100
    assert cpu.regs.get(RegisterName.F) == 0x01
    assert cpu.regs.get(RegisterName.PC) == 0


@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize("f_input", [0x04, 0x80, 0xA4, 0xFC, 0xFF])
def test_reti_rejects_unverified_f_before_any_architectural_write(
    backend: str, f_input: int
) -> None:
    if backend == "llama":
        assert "llama" in available_backends(), "LLAMA backend not available"

    sp = 0x100
    memory = _make_memory(0xA5, f_input, (0x12, 0x34, 0x05), sp)
    imr_addr = INTERNAL_MEMORY_START + IMEMRegisters.IMR
    memory._raw[imr_addr] = 0x5A
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.PC, 0)
    cpu.regs.set(RegisterName.S, sp)
    cpu.regs.set(RegisterName.F, 0x02)

    with pytest.raises(RuntimeError, match="bits 2-7 require real-hardware tracing"):
        cpu.execute_instruction(0)

    assert cpu.regs.get(RegisterName.PC) == 0
    assert cpu.regs.get(RegisterName.S) == sp
    assert cpu.regs.get(RegisterName.F) == 0x02
    assert memory._raw[imr_addr] == 0x5A
