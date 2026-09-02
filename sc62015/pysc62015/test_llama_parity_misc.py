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
@pytest.mark.parametrize(
    ("program", "expected_offset"),
    [
        (bytes.fromhex("30 A0 05"), 0x05),
        (bytes.fromhex("31 A0 05"), 0x05),
        (bytes.fromhex("32 A0 05"), 0x05),
        (bytes.fromhex("33 A0 05"), 0x05),
        (bytes.fromhex("34 A0 05"), 0x25),
        (bytes.fromhex("36 A0 05"), 0x25),
        (bytes.fromhex("24 A0 05"), 0x30),
        (bytes.fromhex("26 A0 05"), 0x30),
        (bytes.fromhex("22 A0 05"), 0x15),
        (bytes.fromhex("30 31 A0 05"), 0x05),
    ],
)
def test_hw008_measured_pre_aliases_match_between_backends(
    backend: str, program: bytes, expected_offset: int
) -> None:
    if backend == "llama":
        assert "llama" in available_backends(), "LLAMA backend not available"

    raw = bytearray(ADDRESS_SPACE_SIZE)
    raw[: len(program)] = program
    raw[INTERNAL_MEMORY_START + IMEMRegisters.BP] = 0x10
    raw[INTERNAL_MEMORY_START + IMEMRegisters.PX] = 0x20
    memory = _make_memory(raw)
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.A, 0xD7)

    _run(cpu)

    assert memory._raw[INTERNAL_MEMORY_START + expected_offset] == 0xD7
    assert cpu.regs.get(RegisterName.PC) == len(program)


@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize(
    "program", [bytes.fromhex("65 12 07"), bytes.fromhex("22 65 12 07")]
)
def test_hw008_redundant_pre22_test_flags_match_between_backends(
    backend: str, program: bytes
) -> None:
    if backend == "llama":
        assert "llama" in available_backends(), "LLAMA backend not available"

    raw = bytearray(ADDRESS_SPACE_SIZE)
    raw[: len(program)] = program
    raw[INTERNAL_MEMORY_START + IMEMRegisters.BP] = 0x10
    raw[INTERNAL_MEMORY_START + 0x22] = 0x80
    memory = _make_memory(raw)
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.F, 0x01)

    _run(cpu)

    assert cpu.regs.get(RegisterName.FC) == 1
    assert cpu.regs.get(RegisterName.FZ) == 1
    assert cpu.regs.get(RegisterName.PC) == len(program)


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
@pytest.mark.parametrize(
    ("program", "i_seed", "expected_i", "expected_imem", "expected_external_reads"),
    [
        (
            bytes.fromhex("32 F3 00 40 40"),
            3,
            0,
            bytes.fromhex("FF 00 00 00 FF"),
            [0x40680, 0x40681, 0x40682],
        ),
        (
            bytes.fromhex("32 F3 00 3F 40"),
            3,
            0,
            bytes.fromhex("00 00 00 04 FF"),
            [0x40680, 0x40681, 0x40682],
        ),
        (
            bytes.fromhex("32 F3 00 41 40"),
            2,
            0,
            bytes.fromhex("FF 80 00 00 FF"),
            [0x40680, 0x40681],
        ),
        (
            bytes.fromhex("32 F3 80 40 40 10"),
            3,
            0,
            bytes.fromhex("FF 00 00 00 FF"),
            [0x40690, 0x40691, 0x40692],
        ),
        (
            bytes.fromhex("32 F3 C0 40 40 10"),
            3,
            0,
            bytes.fromhex("FF 00 00 00 FF"),
            [0x40670, 0x40671, 0x40672],
        ),
        (
            bytes.fromhex("32 F1 00 40 40"),
            0x5AA5,
            0x5AA5,
            bytes.fromhex("FF 00 00 04 FF"),
            [0x40680, 0x40681],
        ),
        (
            bytes.fromhex("32 F2 00 40 40"),
            0x5AA5,
            0x5AA5,
            bytes.fromhex("FF 00 00 00 FF"),
            [0x40680, 0x40681, 0x40682],
        ),
        (
            bytes.fromhex("32 F1 80 40 40 10"),
            0x5AA5,
            0x5AA5,
            bytes.fromhex("FF 00 00 04 FF"),
            [0x40690, 0x40691],
        ),
        (
            bytes.fromhex("32 F1 C0 40 40 10"),
            0x5AA5,
            0x5AA5,
            bytes.fromhex("FF 00 00 04 FF"),
            [0x40670, 0x40671],
        ),
        (
            bytes.fromhex("32 F2 80 40 40 10"),
            0x5AA5,
            0x5AA5,
            bytes.fromhex("FF 00 00 00 FF"),
            [0x40690, 0x40691, 0x40692],
        ),
        (
            bytes.fromhex("32 F2 C0 40 40 10"),
            0x5AA5,
            0x5AA5,
            bytes.fromhex("FF 00 00 00 FF"),
            [0x40670, 0x40671, 0x40672],
        ),
    ],
)
def test_hw007_external_pointer_moves_snapshot_before_overlapping_writes(
    backend: str,
    program: bytes,
    i_seed: int,
    expected_i: int,
    expected_imem: bytes,
    expected_external_reads: list[int],
) -> None:
    """Match the decisive PC-E500 CE1 pointer-overlap matrices."""

    if backend == "llama":
        assert "llama" in available_backends(), "LLAMA backend not available"

    raw = bytearray(ADDRESS_SPACE_SIZE)
    # The pointer at IMEM 40..42 initially contains 0x40680 and overlaps the
    # destination writes. Programs include the canonical PRE32 (N,N).
    raw[: len(program)] = program
    raw[INTERNAL_MEMORY_START + 0x3F : INTERNAL_MEMORY_START + 0x44] = bytes.fromhex(
        "FF 80 06 04 FF"
    )
    external_reads: list[int] = []

    def read(addr: int) -> int:
        if 0x40000 <= addr <= 0x7FFFF:
            external_reads.append(addr)
        return raw[addr]

    def write(addr: int, value: int) -> None:
        raw[addr] = value & 0xFF

    memory = Memory(read, write)
    setattr(
        memory,
        "peek_byte_for_preflight",
        lambda address, _pc=None: raw[address & 0xFFFFFF],
    )
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.I, i_seed)
    cpu.regs.set(RegisterName.FC, 1)
    cpu.regs.set(RegisterName.FZ, 1)

    _run(cpu)

    assert external_reads == expected_external_reads
    assert (
        raw[INTERNAL_MEMORY_START + 0x3F : INTERNAL_MEMORY_START + 0x44]
        == expected_imem
    )
    assert cpu.regs.get(RegisterName.I) == expected_i
    assert cpu.regs.get(RegisterName.F) == 0x03


@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize(
    ("program", "i_seed", "expected_i", "expected_external_writes"),
    [
        (
            bytes.fromhex("32 F8 00 40 60"),
            0x5AA5,
            0x5AA5,
            [(0x406A0, 0xA5)],
        ),
        (
            bytes.fromhex("32 F9 00 40 60"),
            0x5AA5,
            0x5AA5,
            [(0x406A0, 0xA5), (0x406A1, 0x5A)],
        ),
        (
            bytes.fromhex("32 FA 00 40 60"),
            0x5AA5,
            0x5AA5,
            [(0x406A0, 0xA5), (0x406A1, 0x5A), (0x406A2, 0x3C)],
        ),
        (
            bytes.fromhex("32 FB 00 40 60"),
            4,
            0,
            [
                (0x406A0, 0xA5),
                (0x406A1, 0x5A),
                (0x406A2, 0x3C),
                (0x406A3, 0xC3),
            ],
        ),
        (
            bytes.fromhex("32 F8 80 40 60 10"),
            0x5AA5,
            0x5AA5,
            [(0x406B0, 0xA5)],
        ),
        (
            bytes.fromhex("32 F8 C0 40 60 10"),
            0x5AA5,
            0x5AA5,
            [(0x40690, 0xA5)],
        ),
        (
            bytes.fromhex("32 F9 80 40 60 10"),
            0x5AA5,
            0x5AA5,
            [(0x406B0, 0xA5), (0x406B1, 0x5A)],
        ),
        (
            bytes.fromhex("32 F9 C0 40 60 10"),
            0x5AA5,
            0x5AA5,
            [(0x40690, 0xA5), (0x40691, 0x5A)],
        ),
        (
            bytes.fromhex("32 FA 80 40 60 10"),
            0x5AA5,
            0x5AA5,
            [(0x406B0, 0xA5), (0x406B1, 0x5A), (0x406B2, 0x3C)],
        ),
        (
            bytes.fromhex("32 FA C0 40 60 10"),
            0x5AA5,
            0x5AA5,
            [(0x40690, 0xA5), (0x40691, 0x5A), (0x40692, 0x3C)],
        ),
        (
            bytes.fromhex("32 FB 80 40 60 10"),
            4,
            0,
            [
                (0x406B0, 0xA5),
                (0x406B1, 0x5A),
                (0x406B2, 0x3C),
                (0x406B3, 0xC3),
            ],
        ),
        (
            bytes.fromhex("32 FB C0 40 60 10"),
            4,
            0,
            [
                (0x40690, 0xA5),
                (0x40691, 0x5A),
                (0x40692, 0x3C),
                (0x40693, 0xC3),
            ],
        ),
    ],
)
def test_hw007_external_pointer_destinations_emit_hardware_write_address_order(
    backend: str,
    program: bytes,
    i_seed: int,
    expected_i: int,
    expected_external_writes: list[tuple[int, int]],
) -> None:
    """Pin all F8-FB write addresses; current FT hardware cannot validate data."""

    if backend == "llama":
        assert "llama" in available_backends(), "LLAMA backend not available"

    raw = bytearray(ADDRESS_SPACE_SIZE)
    raw[: len(program)] = program
    raw[INTERNAL_MEMORY_START + 0x40 : INTERNAL_MEMORY_START + 0x43] = bytes.fromhex(
        "A0 06 04"
    )
    raw[INTERNAL_MEMORY_START + 0x60 : INTERNAL_MEMORY_START + 0x64] = bytes.fromhex(
        "A5 5A 3C C3"
    )
    external_writes: list[tuple[int, int]] = []

    def read(addr: int) -> int:
        return raw[addr]

    def write(addr: int, value: int) -> None:
        raw[addr] = value & 0xFF
        if 0x40000 <= addr <= 0x7FFFF:
            external_writes.append((addr, value & 0xFF))

    memory = Memory(read, write)
    setattr(
        memory,
        "peek_byte_for_preflight",
        lambda address, _pc=None: raw[address & 0xFFFFFF],
    )
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.I, i_seed)
    cpu.regs.set(RegisterName.FC, 1)
    cpu.regs.set(RegisterName.FZ, 1)

    _run(cpu)

    # The connected PC-E500/FT600 capture proves only the address count and
    # low-to-high order. Values remain the ISA/emulator contract because this
    # known gateware image sampled and read every CE1 write as zero.
    assert external_writes == expected_external_writes
    assert cpu.regs.get(RegisterName.I) == expected_i
    assert cpu.regs.get(RegisterName.F) == 0x03


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


def test_cmpp_imem_reg_uses_raw24_memory_between_backends() -> None:
    """Pin D7's raw memory image versus zero-extended register contract."""

    assert "llama" in available_backends(), "LLAMA backend not available"

    def run_case(backend: str) -> tuple[int, int, int]:
        raw = bytearray(ADDRESS_SPACE_SIZE)
        # D7 04 10: CMPP (BP+10), X (no PRE byte; IMem defaults to BP+N).
        raw[0:3] = bytes([0xD7, 0x04, 0x10])
        # The raw memory image is 0xF00080. D7 compares all of it with the
        # zero-extended 20-bit X register, so the operands are unequal.
        raw[INTERNAL_MEMORY_START + 0x10 : INTERNAL_MEMORY_START + 0x13] = (
            b"\x80\x00\xf0"
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


# This hardware-backed parity case executes the complete 65,536-iteration ring;
# the Python LLIL backend can legitimately exceed the global 60s guard in CI.
@pytest.mark.timeout(180)
@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize(
    ("mnemonic", "program", "expected_fz"),
    [
        ("ADCL", bytes.fromhex("541020"), 0),
        ("SBCL", bytes.fromhex("5C1020"), 1),
        ("DADL", bytes.fromhex("C41020"), 0),
        ("DSBL", bytes.fromhex("D41020"), 1),
        ("MVL", bytes.fromhex("CB1020"), 1),
        ("MVLD", bytes.fromhex("CF1020"), 1),
        ("EXL", bytes.fromhex("C31020"), 1),
        ("DSLL", bytes.fromhex("EC10"), 0),
        ("DSRL", bytes.fromhex("FC10"), 0),
    ],
)
def test_hw002_i_zero_counted_instructions_match_hardware_across_backends(
    backend: str, mnemonic: str, program: bytes, expected_fz: int
) -> None:
    if backend == "llama":
        assert "llama" in available_backends(), "LLAMA backend not available"

    raw = bytearray(ADDRESS_SPACE_SIZE)
    raw[: len(program)] = program
    memory = _make_memory(raw)
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.I, 0)
    cpu.regs.set(RegisterName.FC, 0)
    cpu.regs.set(RegisterName.FZ, 1)

    cpu.execute_instruction(0)

    assert cpu.regs.get(RegisterName.PC) == len(program), mnemonic
    assert cpu.regs.get(RegisterName.I) == 0
    assert cpu.regs.get(RegisterName.FC) == 0
    assert cpu.regs.get(RegisterName.FZ) == expected_fz
    assert raw[INTERNAL_MEMORY_START + 0x10] == 0
    assert raw[INTERNAL_MEMORY_START + 0x20] == 0


@pytest.mark.parametrize("backend", ["python", "llama"])
def test_hw002_wait_i_zero_consumes_65536_cycles_and_wraps(backend: str) -> None:
    """Match PC-E500 run 20260830-232830-0064 without fake memory reads."""

    if backend == "llama":
        assert "llama" in available_backends(), "LLAMA backend not available"

    raw = bytearray(ADDRESS_SPACE_SIZE)
    raw[0] = 0xEF
    memory = _make_memory(raw)
    calls: list[int] = []
    setattr(memory, "wait_cycles", calls.append)
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.I, 0)
    cpu.regs.set(RegisterName.FC, 1)
    cpu.regs.set(RegisterName.FZ, 1)

    cpu.execute_instruction(0)

    assert calls == [0x10000]
    assert cpu.regs.get(RegisterName.I) == 0
    assert cpu.regs.get(RegisterName.F) == 0x03
    assert cpu.regs.get(RegisterName.PC) == 1


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
