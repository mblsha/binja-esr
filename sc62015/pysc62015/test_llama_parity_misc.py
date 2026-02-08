from __future__ import annotations

from typing import cast

import pytest

from binja_test_mocks.eval_llil import Memory

from sc62015.pysc62015 import CPU, RegisterName, available_backends
from sc62015.pysc62015.constants import ADDRESS_SPACE_SIZE, INTERNAL_MEMORY_START
from sc62015.pysc62015.instr.opcodes import IMEMRegisters
from sc62015.pysc62015.test_emulator import compute_expected_dsll


class MemoryWithRaw(Memory):
    _raw: bytearray


def _make_memory(raw: bytearray) -> MemoryWithRaw:
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
    return cast(MemoryWithRaw, memory)


def _run(cpu: CPU, addr: int = 0) -> None:
    cpu.regs.set(RegisterName.PC, addr)
    cpu.execute_instruction(addr)


@pytest.mark.parametrize("backend", ["python", "llama"])
def test_il_write_clears_ih(backend: str) -> None:
    if backend == "llama":
        assert "llama" in available_backends(), "LLAMA backend not available"

    raw = bytearray(ADDRESS_SPACE_SIZE)
    raw[0] = 0x00  # NOP
    memory = _make_memory(raw)
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.I, 0x1234)

    # Write low byte only; IH is cleared by hardware
    cpu.regs.set(RegisterName.IL, 0x56)
    _run(cpu)

    assert cpu.regs.get(RegisterName.I) == 0x0056
    assert cpu.regs.get(RegisterName.IH) == 0


@pytest.mark.parametrize("backend", ["python", "llama"])
def test_mvld_decrements_addresses(backend: str) -> None:
    if backend == "llama":
        assert "llama" in available_backends(), "LLAMA backend not available"

    dst_offset = 0x20
    src_offset = 0x30
    raw = bytearray(ADDRESS_SPACE_SIZE)
    # PRE32 (N,N), MVLD (m),(n)
    raw[0:4] = bytes([0x32, 0xCF, dst_offset, src_offset])
    # Populate source bytes at src_offset (MSB) and src_offset-1 (LSB)
    raw[INTERNAL_MEMORY_START + src_offset] = 0x11
    raw[INTERNAL_MEMORY_START + src_offset - 1] = 0x22
    # Clear destination
    raw[INTERNAL_MEMORY_START + dst_offset] = 0x00
    raw[INTERNAL_MEMORY_START + dst_offset - 1] = 0x00

    memory = _make_memory(raw)
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.I, 2)

    _run(cpu)

    assert memory._raw[INTERNAL_MEMORY_START + dst_offset] == 0x11
    assert memory._raw[INTERNAL_MEMORY_START + dst_offset - 1] == 0x22
    assert cpu.regs.get(RegisterName.I) == 0


@pytest.mark.parametrize("backend", ["python", "llama"])
def test_dsll_shifts_left_digits(backend: str) -> None:
    if backend == "llama":
        assert "llama" in available_backends(), "LLAMA backend not available"

    start = 0x10
    raw = bytearray(ADDRESS_SPACE_SIZE)
    # PRE32 (N,N), DSLL (n)
    raw[0:3] = bytes([0x32, 0xEC, start])
    # BCD 0x1234 laid out MSB at start, LSB at start-1
    raw[INTERNAL_MEMORY_START + start] = 0x12
    raw[INTERNAL_MEMORY_START + start - 1] = 0x34

    memory = _make_memory(raw)
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.I, 2)

    _run(cpu)

    expected = compute_expected_dsll([0x12, 0x34])
    assert memory._raw[INTERNAL_MEMORY_START + start] == expected[0]
    assert memory._raw[INTERNAL_MEMORY_START + start - 1] == expected[1]
    assert cpu.regs.get(RegisterName.I) == 0
    assert cpu.regs.get(RegisterName.FZ) == 0


def test_cmpp_imem_reg_matches_between_backends() -> None:
    """Ensure LLAMA matches Python for CMPP (m),r3 borrow/operand ordering."""

    assert "llama" in available_backends(), "LLAMA backend not available"

    def run_case(backend: str) -> tuple[int, int, int]:
        raw = bytearray(ADDRESS_SPACE_SIZE)
        # D7 04 10: CMPP (BP+10), X (no PRE byte; IMem defaults to BP+N).
        raw[0:3] = bytes([0xD7, 0x04, 0x10])
        # (m..m+2) = 0xFFFFFF (little-endian), so lhs >= rhs for X=0x000080.
        raw[INTERNAL_MEMORY_START + 0x10 : INTERNAL_MEMORY_START + 0x13] = (
            b"\xff\xff\xff"
        )

        memory = _make_memory(raw)
        cpu = CPU(memory, reset_on_init=False, backend=backend)
        cpu.regs.set(RegisterName.X, 0x000080)
        cpu.regs.set(RegisterName.FC, 1)
        cpu.regs.set(RegisterName.FZ, 1)
        _run(cpu)
        return (
            cpu.regs.get(RegisterName.PC),
            cpu.regs.get(RegisterName.FC) & 1,
            cpu.regs.get(RegisterName.FZ) & 1,
        )

    python_state = run_case("python")
    llama_state = run_case("llama")

    assert llama_state == python_state
    assert python_state == (3, 0, 0)


def test_wait_invokes_wait_cycles_llama() -> None:
    assert "llama" in available_backends(), "LLAMA backend not available"

    raw = bytearray(ADDRESS_SPACE_SIZE)
    raw[0] = 0xEF  # WAIT
    memory = _make_memory(raw)
    spins = {"cycles": 0}

    def wait_cycles(cycles: int) -> None:
        spins["cycles"] += int(cycles)

    setattr(memory, "wait_cycles", wait_cycles)

    cpu = CPU(memory, reset_on_init=False, backend="llama")
    cpu.regs.set(RegisterName.I, 3)

    _run(cpu)

    assert cpu.regs.get(RegisterName.I) == 0
    assert spins["cycles"] == 3


@pytest.mark.parametrize("backend", ["python", "llama"])
def test_mv_regpair_low_codes_map_to_ba_i(backend: str) -> None:
    if backend == "llama":
        assert "llama" in available_backends(), "LLAMA backend not available"

    raw = bytearray(ADDRESS_SPACE_SIZE)
    raw[0:2] = bytes([0xFD, 0x01])  # MV regpair: low codes should decode as BA/I
    memory = _make_memory(raw)
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.BA, 0xAA55)
    cpu.regs.set(RegisterName.I, 0x1234)

    _run(cpu)

    assert cpu.regs.get(RegisterName.BA) == 0x1234
    assert cpu.regs.get(RegisterName.I) == 0x1234


@pytest.mark.parametrize("backend", ["python", "llama"])
def test_add_regpair_20bit_carry_and_zero(backend: str) -> None:
    if backend == "llama":
        assert "llama" in available_backends(), "LLAMA backend not available"

    raw = bytearray(ADDRESS_SPACE_SIZE)
    raw[0:2] = bytes([0x45, 0x45])  # ADD regpair size=3: X += Y
    memory = _make_memory(raw)
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.X, 0x0FFFFF)
    cpu.regs.set(RegisterName.Y, 0x000001)

    _run(cpu)

    assert cpu.regs.get(RegisterName.X) == 0x000000
    assert cpu.regs.get(RegisterName.FC) == 1
    assert cpu.regs.get(RegisterName.FZ) == 1


@pytest.mark.parametrize("backend", ["python", "llama"])
def test_inc_reg3_x_wraps_20bit(backend: str) -> None:
    if backend == "llama":
        assert "llama" in available_backends(), "LLAMA backend not available"

    raw = bytearray(ADDRESS_SPACE_SIZE)
    raw[0:2] = bytes([0x6C, 0x04])  # INC reg3 selector=X
    memory = _make_memory(raw)
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.X, 0x0FFFFF)

    _run(cpu)

    assert cpu.regs.get(RegisterName.X) == 0x000000
    assert cpu.regs.get(RegisterName.FZ) == 1


@pytest.mark.parametrize("backend", ["python", "llama"])
def test_dadl_reg_source_only_first_byte(backend: str) -> None:
    if backend == "llama":
        assert "llama" in available_backends(), "LLAMA backend not available"

    raw = bytearray(ADDRESS_SPACE_SIZE)
    raw[0:3] = bytes([0x32, 0xC5, 0x10])  # PRE32, DADL (m),A
    raw[INTERNAL_MEMORY_START + 0x10] = 0x00
    raw[INTERNAL_MEMORY_START + 0x0F] = 0x00
    memory = _make_memory(raw)
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.I, 2)
    cpu.regs.set(RegisterName.A, 0x01)

    _run(cpu)

    assert memory._raw[INTERNAL_MEMORY_START + 0x10] == 0x01
    assert memory._raw[INTERNAL_MEMORY_START + 0x0F] == 0x00


@pytest.mark.parametrize("backend", ["python", "llama"])
def test_reg_imem_offset_imem_selector_uses_pre1(backend: str) -> None:
    if backend == "llama":
        assert "llama" in available_backends(), "LLAMA backend not available"

    raw = bytearray(ADDRESS_SPACE_SIZE)
    # PRE30: op1=N, op2=BP+N. RegIMemOffset should still use PRE1 for the IMEM selector.
    raw[0:4] = bytes([0x30, 0xE8, 0x04, 0x10])  # MV [X], (n)
    raw[INTERNAL_MEMORY_START + 0x10] = 0xAA  # direct (N)
    raw[INTERNAL_MEMORY_START + 0x30] = 0xBB  # BP+N when BP=0x20
    raw[INTERNAL_MEMORY_START + IMEMRegisters.BP] = 0x20
    memory = _make_memory(raw)
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.X, 0x000100)

    _run(cpu)

    assert memory._raw[0x000100] == 0xAA


@pytest.mark.parametrize("backend", ["python", "llama"])
def test_wait_i_zero_uses_full_counter_span(backend: str) -> None:
    if backend == "llama":
        assert "llama" in available_backends(), "LLAMA backend not available"

    raw = bytearray(ADDRESS_SPACE_SIZE)
    raw[0] = 0xEF  # WAIT
    memory = _make_memory(raw)
    spins = {"cycles": 0}

    def wait_cycles(cycles: int) -> None:
        spins["cycles"] += int(cycles)

    setattr(memory, "wait_cycles", wait_cycles)
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.I, 0)

    _run(cpu)

    assert cpu.regs.get(RegisterName.I) == 0
    assert spins["cycles"] == 0x10000
