from __future__ import annotations

from typing import cast

import pytest

from binja_test_mocks.eval_llil import Memory

from sc62015.pysc62015 import CPU, RegisterName, available_backends
from sc62015.pysc62015.constants import ADDRESS_SPACE_SIZE, INTERNAL_MEMORY_START
from sc62015.pysc62015.instr.opcodes import IMEMRegisters, InvalidInstruction
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
    setattr(
        memory,
        "peek_byte_for_preflight",
        lambda address, _pc=None: raw[address & 0xFFFFFF],
    )
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
    # PRE30 (N), DSLL (n): lone IMEM selectors use the canonical PRE1 form.
    raw[0:3] = bytes([0x30, 0xEC, start])
    # DSLL's operand points at the least-significant byte and the instruction
    # walks downward toward the most-significant byte.
    raw[INTERNAL_MEMORY_START + start] = 0x34
    raw[INTERNAL_MEMORY_START + start - 1] = 0x12

    memory = _make_memory(raw)
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.I, 2)

    _run(cpu)

    expected = compute_expected_dsll([0x12, 0x34])
    assert memory._raw[INTERNAL_MEMORY_START + start] == expected[1]
    assert memory._raw[INTERNAL_MEMORY_START + start - 1] == expected[0]
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
def test_narrow_register_jp_fails_before_cross_page_semantics_diverge(
    backend: str,
) -> None:
    """The formerly accepted JP A must fail without choosing a page policy."""

    if backend == "llama":
        assert "llama" in available_backends(), "LLAMA backend not available"

    addr = 0xE1234
    raw = bytearray(ADDRESS_SPACE_SIZE)
    raw[addr : addr + 2] = bytes([0x11, 0x00])  # invalid JP A
    memory = _make_memory(raw)
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.PC, addr)
    cpu.regs.set(RegisterName.A, 0x56)

    with pytest.raises(InvalidInstruction, match="opcode 0x11"):
        cpu.execute_instruction(addr)

    # Python formerly kept the E0000 page while Rust jumped to 00056. Rejecting
    # the malformed selector atomically prevents either invented behavior.
    assert cpu.regs.get(RegisterName.PC) == addr


@pytest.mark.parametrize("backend", ["python", "llama"])
def test_valid_register_jp_uses_full_20_bit_target(backend: str) -> None:
    if backend == "llama":
        assert "llama" in available_backends(), "LLAMA backend not available"

    addr = 0xE1234
    raw = bytearray(ADDRESS_SPACE_SIZE)
    raw[addr : addr + 2] = bytes([0x11, 0x04])  # JP X
    memory = _make_memory(raw)
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.X, 0x23456)

    _run(cpu, addr)

    assert cpu.regs.get(RegisterName.PC) == 0x23456


@pytest.mark.parametrize("backend", ["python", "llama"])
def test_dadl_reg_source_only_first_byte(backend: str) -> None:
    if backend == "llama":
        assert "llama" in available_backends(), "LLAMA backend not available"

    raw = bytearray(ADDRESS_SPACE_SIZE)
    raw[0:3] = bytes([0x30, 0xC5, 0x10])  # PRE30, DADL (m),A
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
@pytest.mark.parametrize(
    ("mnemonic", "program"),
    [
        ("ADCL", bytes.fromhex("541020")),
        ("SBCL", bytes.fromhex("5C1020")),
        ("DADL", bytes.fromhex("C41020")),
        ("DSBL", bytes.fromhex("D41020")),
        ("MVL", bytes.fromhex("E33720")),
        ("MVLD", bytes.fromhex("CF1020")),
        ("EXL", bytes.fromhex("C31020")),
        ("DSLL", bytes.fromhex("EC10")),
        ("DSRL", bytes.fromhex("FC10")),
        ("WAIT", bytes.fromhex("EF")),
    ],
)
def test_i_zero_counted_instructions_fail_atomically_across_backends(
    backend: str, mnemonic: str, program: bytes
) -> None:
    if backend == "llama":
        assert "llama" in available_backends(), "LLAMA backend not available"

    raw = bytearray(ADDRESS_SPACE_SIZE)
    raw[: len(program)] = program
    raw[INTERNAL_MEMORY_START + 0x10] = 0x12
    raw[INTERNAL_MEMORY_START + 0x20] = 0x34
    raw[INTERNAL_MEMORY_START + IMEMRegisters.BP] = 0x40
    raw[INTERNAL_MEMORY_START + IMEMRegisters.PX] = 0x50
    raw[INTERNAL_MEMORY_START + IMEMRegisters.PY] = 0x60
    memory = _make_memory(raw)
    spins = {"cycles": 0}

    def wait_cycles(cycles: int) -> None:
        spins["cycles"] += int(cycles)

    setattr(memory, "wait_cycles", wait_cycles)
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    initial_pc = 0x34567
    cpu.regs.set(RegisterName.PC, initial_pc)
    cpu.regs.set(RegisterName.I, 0)
    cpu.regs.set(RegisterName.FC, 1)
    cpu.regs.set(RegisterName.FZ, 0)
    initial_pointers = {
        RegisterName.X: 0x12345,
        RegisterName.Y: 0x23456,
        RegisterName.U: 0x34567,
        RegisterName.S: 0x45678,
    }
    for register, value in initial_pointers.items():
        cpu.regs.set(register, value)

    expected_error = RuntimeError if backend == "llama" else NotImplementedError
    with pytest.raises(expected_error, match=r"I=0.*real-hardware"):
        cpu.execute_instruction(0)

    assert cpu.regs.get(RegisterName.PC) == initial_pc
    assert cpu.regs.get(RegisterName.I) == 0
    assert cpu.regs.get(RegisterName.FC) == 1
    assert cpu.regs.get(RegisterName.FZ) == 0
    for register, value in initial_pointers.items():
        assert cpu.regs.get(register) == value
    assert spins["cycles"] == 0, mnemonic
    assert raw[INTERNAL_MEMORY_START + 0x10] == 0x12
    assert raw[INTERNAL_MEMORY_START + 0x20] == 0x34
    assert raw[INTERNAL_MEMORY_START + IMEMRegisters.BP] == 0x40
    assert raw[INTERNAL_MEMORY_START + IMEMRegisters.PX] == 0x50
    assert raw[INTERNAL_MEMORY_START + IMEMRegisters.PY] == 0x60


@pytest.mark.parametrize("backend", ["python", "llama"])
def test_wide_imem_store_snapshots_bp_destination_before_self_overlap(
    backend: str,
) -> None:
    if backend == "llama":
        assert "llama" in available_backends(), "LLAMA backend not available"

    raw = bytearray(ADDRESS_SPACE_SIZE)
    raw[0:3] = bytes.fromhex("C90010")  # MVW (BP+00),(BP+10)
    raw[INTERNAL_MEMORY_START + IMEMRegisters.BP] = 0xEC
    raw[INTERNAL_MEMORY_START + IMEMRegisters.PX] = 0xA5
    raw[INTERNAL_MEMORY_START + 0xFC] = 0x34
    raw[INTERNAL_MEMORY_START + 0xFD] = 0x12
    memory = _make_memory(raw)
    cpu = CPU(memory, reset_on_init=False, backend=backend)

    _run(cpu)

    assert raw[INTERNAL_MEMORY_START + IMEMRegisters.BP] == 0x34
    assert raw[INTERNAL_MEMORY_START + IMEMRegisters.PX] == 0x12
    assert raw[INTERNAL_MEMORY_START + 0x35] == 0x00


@pytest.mark.parametrize("backend", ["python", "llama"])
def test_wide_imem_pointer_store_snapshots_destination_across_bp_px_py(
    backend: str,
) -> None:
    if backend == "llama":
        assert "llama" in available_backends(), "LLAMA backend not available"

    raw = bytearray(ADDRESS_SPACE_SIZE)
    raw[0:3] = bytes.fromhex("CA0010")  # MVP (BP+00),(BP+10)
    raw[INTERNAL_MEMORY_START + IMEMRegisters.BP] = 0xEC
    raw[INTERNAL_MEMORY_START + IMEMRegisters.PX] = 0xA5
    raw[INTERNAL_MEMORY_START + IMEMRegisters.PY] = 0x5A
    raw[INTERNAL_MEMORY_START + 0xFC] = 0x56
    raw[INTERNAL_MEMORY_START + 0xFD] = 0x34
    raw[INTERNAL_MEMORY_START + 0xFE] = 0x02
    memory = _make_memory(raw)
    cpu = CPU(memory, reset_on_init=False, backend=backend)

    _run(cpu)

    assert raw[INTERNAL_MEMORY_START + IMEMRegisters.BP] == 0x56
    assert raw[INTERNAL_MEMORY_START + IMEMRegisters.PX] == 0x34
    assert raw[INTERNAL_MEMORY_START + IMEMRegisters.PY] == 0x02
    assert raw[INTERNAL_MEMORY_START + 0x57] == 0x00
    assert raw[INTERNAL_MEMORY_START + 0x58] == 0x00


@pytest.mark.parametrize("backend", ["python", "llama"])
def test_wide_register_indirect_store_wraps_from_snapshotted_address(
    backend: str,
) -> None:
    if backend == "llama":
        assert "llama" in available_backends(), "LLAMA backend not available"

    instr_addr = 0x100
    raw = bytearray(ADDRESS_SPACE_SIZE)
    raw[instr_addr : instr_addr + 2] = bytes.fromhex("B404")  # MV [X],X
    raw[INTERNAL_MEMORY_START] = 0x99
    memory = _make_memory(raw)
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.X, 0xFFFFF)

    _run(cpu, instr_addr)

    assert cpu.regs.get(RegisterName.X) == 0xFFFFF
    assert raw[0xFFFFF] == 0xFF
    assert raw[0x00000] == 0xFF
    assert raw[0x00001] == 0x0F
    assert raw[INTERNAL_MEMORY_START] == 0x99
