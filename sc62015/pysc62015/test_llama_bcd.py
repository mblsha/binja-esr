from __future__ import annotations

import pytest

from binja_test_mocks.eval_llil import Memory

from sc62015.pysc62015 import CPU, RegisterName, available_backends
from sc62015.pysc62015.constants import ADDRESS_SPACE_SIZE, INTERNAL_MEMORY_START


def _make_memory_from_raw(raw: bytearray) -> Memory:
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
    return memory


def _build_template(
    program: list[int],
    dst_start: int,
    dst_bytes: list[int],
    src_start: int | None = None,
    src_bytes: list[int] | None = None,
) -> bytearray:
    raw = bytearray(ADDRESS_SPACE_SIZE)
    for idx, value in enumerate(program):
        raw[idx] = value & 0xFF
    for idx, value in enumerate(dst_bytes):
        raw[INTERNAL_MEMORY_START + dst_start - idx] = value & 0xFF
    if src_start is not None and src_bytes is not None:
        for idx, value in enumerate(src_bytes):
            raw[INTERNAL_MEMORY_START + src_start - idx] = value & 0xFF
    return raw


def _run_backend(
    template: bytearray,
    backend: str,
    *,
    dest_offsets: list[int],
    i_val: int,
    fc_val: int = 0,
    a_val: int | None = None,
) -> dict[str, int | list[int]]:
    raw = bytearray(template)
    memory = _make_memory_from_raw(raw)
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.PC, 0)
    cpu.regs.set(RegisterName.I, i_val)
    cpu.regs.set(RegisterName.FC, fc_val)
    if a_val is not None:
        cpu.regs.set(RegisterName.A, a_val)

    cpu.execute_instruction(0)

    dest_values = [raw[INTERNAL_MEMORY_START + offset] for offset in dest_offsets]
    return {
        "dest": dest_values,
        "fc": cpu.regs.get(RegisterName.FC),
        "fz": cpu.regs.get(RegisterName.FZ),
        "i": cpu.regs.get(RegisterName.I),
        "pc": cpu.regs.get(RegisterName.PC),
    }


@pytest.mark.parametrize("backend", ["python", "llama"])
def test_dadl_imem_reg_a_single_byte_sets_flags(backend: str) -> None:
    dst_offset = 0x12
    program = [0x32, 0xC5, dst_offset]  # PRE32 (N,N), DADL (m),A
    template = _build_template(program, dst_offset, [0x99])

    result = _run_backend(
        template, backend, dest_offsets=[dst_offset], i_val=1, a_val=0x01
    )

    assert result["dest"] == [0x00]
    assert result["fc"] == 1  # carry out of 0x99 + 0x01
    assert result["fz"] == 1  # result byte is zero
    assert result["i"] == 0
    assert result["pc"] == 3  # PRE + opcode + 1-byte operand


def test_dadl_imem_imem_two_bytes_parity_with_llama() -> None:
    assert "llama" in available_backends(), "LLAMA backend not available"

    dst_offset = 0x22
    src_offset = 0x32
    program = [0x32, 0xC4, dst_offset, src_offset]  # PRE32, DADL (m),(n)
    template = _build_template(
        program,
        dst_offset,
        dst_bytes=[0x99, 0x09],
        src_start=src_offset,
        src_bytes=[0x01, 0x99],
    )
    dest_offsets = [dst_offset, dst_offset - 1]

    python_result = _run_backend(
        template,
        "python",
        dest_offsets=dest_offsets,
        i_val=2,
    )
    assert python_result["pc"] == 4
    assert python_result["i"] == 0

    llama_result = _run_backend(
        template,
        "llama",
        dest_offsets=dest_offsets,
        i_val=2,
    )

    assert llama_result == python_result


def test_dsbl_imem_imem_two_bytes_parity_with_llama() -> None:
    assert "llama" in available_backends(), "LLAMA backend not available"

    dst_offset = 0x40
    src_offset = 0x50
    program = [0x32, 0xD4, dst_offset, src_offset]  # PRE32, DSBL (m),(n)
    template = _build_template(
        program,
        dst_offset,
        dst_bytes=[0x00, 0x10],
        src_start=src_offset,
        src_bytes=[0x01, 0x90],
    )
    dest_offsets = [dst_offset, dst_offset - 1]

    python_result = _run_backend(
        template,
        "python",
        dest_offsets=dest_offsets,
        i_val=2,
        fc_val=1,  # incoming borrow
    )
    assert python_result["pc"] == 4
    assert python_result["i"] == 0

    llama_result = _run_backend(
        template,
        "llama",
        dest_offsets=dest_offsets,
        i_val=2,
        fc_val=1,
    )

    assert llama_result == python_result
