from __future__ import annotations

import pytest

from sc62015.pysc62015 import CPU, RegisterName, available_backends
from sc62015.pysc62015.constants import ADDRESS_SPACE_SIZE, INTERNAL_MEMORY_START
from sc62015.pysc62015.test_llama_parity_misc import _make_memory


def _execute_raw24_case(
    backend: str, instruction: bytes, initial_memory: dict[int, int]
) -> bytearray:
    if backend == "llama":
        assert "llama" in available_backends(), "LLAMA backend not available"

    raw = bytearray(ADDRESS_SPACE_SIZE)
    raw[: len(instruction)] = instruction
    for address, value in initial_memory.items():
        raw[address] = value

    cpu = CPU(_make_memory(raw), reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.PC, 0)
    cpu.execute_instruction(0)
    return raw


@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize(
    ("opcode", "register"),
    [
        (0x0C, RegisterName.X),
        (0x0D, RegisterName.Y),
        (0x0E, RegisterName.U),
        (0x0F, RegisterName.S),
    ],
)
def test_three_byte_register_immediate_keeps_only_20_bits(
    backend: str, opcode: int, register: RegisterName
) -> None:
    """Three fetched bytes are not a 24-bit architectural register.

    The X/Y cases reproduce the real PC-E500 capture shape: encoded high byte
    0x3C is observed as 0x0C when the register is subsequently transferred.
    U/S share the same architectural register-immediate operand class.
    """

    raw = bytearray(ADDRESS_SPACE_SIZE)
    raw[:4] = bytes((opcode, 0xA5, 0x5A, 0x3C))
    cpu = CPU(_make_memory(raw), reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.PC, 0)

    cpu.execute_instruction(0)

    assert cpu.regs.get(register) == 0x0C_5AA5


@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize(
    ("opcode", "register", "expected"),
    [
        (0x88, RegisterName.A, 0xA5),
        (0x89, RegisterName.IL, 0xA5),
        (0x8A, RegisterName.BA, 0x5AA5),
        (0x8B, RegisterName.I, 0x5AA5),
        (0x8C, RegisterName.X, 0xC5AA5),
        (0x8D, RegisterName.Y, 0xC5AA5),
        (0x8E, RegisterName.U, 0xC5AA5),
        (0x8F, RegisterName.S, 0xC5AA5),
    ],
)
@pytest.mark.parametrize(
    "address_high",
    [
        pytest.param(0x81, id="hw009-noncanonical-alias"),
        pytest.param(0x01, id="hw009-canonical-counterpart"),
    ],
)
def test_hw009_direct_reads_ignore_encoded_upper_nibble(
    backend: str,
    opcode: int,
    register: RegisterName,
    expected: int,
    address_high: int,
) -> None:
    raw = bytearray(ADDRESS_SPACE_SIZE)
    raw[:5] = bytes((opcode, 0xF0, 0x01, address_high, 0x00))
    raw[0x1_01F0:0x1_01F3] = bytes.fromhex("A5 5A 3C")
    cpu = CPU(_make_memory(raw), reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.PC, 0)
    cpu.regs.set(RegisterName.F, 0x03)

    cpu.execute_instruction(0)

    assert cpu.regs.get(register) == expected
    assert cpu.regs.get(RegisterName.F) == 0x03
    assert cpu.regs.get(RegisterName.PC) == 4

    cpu.execute_instruction(4)
    assert cpu.regs.get(RegisterName.PC) == 5


@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize("address_high", [0x84, 0x04])
@pytest.mark.parametrize(
    ("opcode", "immediate"),
    [
        (0x62, 0x00),
        (0x66, 0xFF),
        (0x6A, 0x00),
        (0x72, 0xFF),
        (0x7A, 0x00),
    ],
)
def test_hw009_absolute_byte_ops_ignore_encoded_upper_nibble(
    backend: str, address_high: int, opcode: int, immediate: int
) -> None:
    """Pin hardware-backed addresses; identity RMW values remain ISA contract."""

    raw = bytearray(ADDRESS_SPACE_SIZE)
    raw[:6] = bytes((opcode, 0xD0, 0x06, address_high, immediate, 0x00))
    raw[0x4_06D0] = 0xA5
    cpu = CPU(_make_memory(raw), reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.PC, 0)
    cpu.regs.set(RegisterName.F, 0x03)

    cpu.execute_instruction(0)

    assert raw[0x4_06D0] == 0xA5
    assert cpu.regs.get(RegisterName.PC) == 5

    cpu.execute_instruction(5)
    assert cpu.regs.get(RegisterName.PC) == 6


@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize("address_high", [0x84, 0x04])
@pytest.mark.parametrize(
    ("opcode", "register", "value", "expected"),
    [
        (0xA8, RegisterName.A, 0xA5, bytes.fromhex("A5")),
        (0xA9, RegisterName.IL, 0xA5, bytes.fromhex("A5")),
        (0xAA, RegisterName.BA, 0x5AA5, bytes.fromhex("A5 5A")),
        (0xAB, RegisterName.I, 0x5AA5, bytes.fromhex("A5 5A")),
        (0xAC, RegisterName.X, 0xC3C5A, bytes.fromhex("5A 3C 0C")),
        (0xAD, RegisterName.Y, 0xC3C5A, bytes.fromhex("5A 3C 0C")),
        (0xAE, RegisterName.U, 0xC3C5A, bytes.fromhex("5A 3C 0C")),
        (0xAF, RegisterName.S, 0xC3C5A, bytes.fromhex("5A 3C 0C")),
    ],
)
def test_hw009_direct_writes_ignore_encoded_upper_nibble(
    backend: str,
    address_high: int,
    opcode: int,
    register: RegisterName,
    value: int,
    expected: bytes,
) -> None:
    """Pin hardware-backed A8-AF addresses; byte values remain ISA contract."""

    raw = bytearray(ADDRESS_SPACE_SIZE)
    raw[:5] = bytes((opcode, 0xD0, 0x06, address_high, 0x00))
    cpu = CPU(_make_memory(raw), reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.PC, 0)
    cpu.regs.set(register, value)

    cpu.execute_instruction(0)

    assert raw[0x4_06D0 : 0x4_06D0 + len(expected)] == expected
    assert cpu.regs.get(RegisterName.PC) == 4

    cpu.execute_instruction(4)
    assert cpu.regs.get(RegisterName.PC) == 5


@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize("address_high", [0x81, 0x01])
@pytest.mark.parametrize(
    ("opcode", "transfer_length", "counted"),
    [
        (0xD0, 1, False),
        (0xD1, 2, False),
        (0xD2, 3, False),
        (0xD3, 3, True),
    ],
)
def test_hw009_absolute_transfers_from_emem_ignore_encoded_upper_nibble(
    backend: str,
    address_high: int,
    opcode: int,
    transfer_length: int,
    counted: bool,
) -> None:
    imem = INTERNAL_MEMORY_START
    raw = bytearray(ADDRESS_SPACE_SIZE)
    raw[:6] = bytes((opcode, 0x60, 0xF0, 0x01, address_high, 0x00))
    raw[0x1_01F0:0x1_01F3] = bytes.fromhex("A5 5A 3C")
    raw[imem + 0xEC] = 0  # BP: make the raw BP+60 selector deterministic
    raw[imem + 0x60 : imem + 0x63] = bytes.fromhex("11 22 33")
    cpu = CPU(_make_memory(raw), reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.PC, 0)
    cpu.regs.set(RegisterName.I, 3 if counted else 0x5AA5)
    cpu.regs.set(RegisterName.F, 0x03)

    cpu.execute_instruction(0)

    assert (
        raw[imem + 0x60 : imem + 0x60 + transfer_length]
        == bytes.fromhex("A5 5A 3C")[:transfer_length]
    )
    assert cpu.regs.get(RegisterName.I) == (0 if counted else 0x5AA5)
    assert cpu.regs.get(RegisterName.F) == 0x03
    assert cpu.regs.get(RegisterName.PC) == 5

    cpu.execute_instruction(5)
    assert cpu.regs.get(RegisterName.PC) == 6


@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize("address_high", [0x84, 0x04])
@pytest.mark.parametrize(
    ("opcode", "transfer_length", "counted"),
    [
        (0xD8, 1, False),
        (0xD9, 2, False),
        (0xDA, 3, False),
        (0xDB, 3, True),
    ],
)
def test_hw009_absolute_transfers_to_emem_ignore_encoded_upper_nibble(
    backend: str,
    address_high: int,
    opcode: int,
    transfer_length: int,
    counted: bool,
) -> None:
    imem = INTERNAL_MEMORY_START
    raw = bytearray(ADDRESS_SPACE_SIZE)
    raw[:6] = bytes((opcode, 0xD0, 0x06, address_high, 0x60, 0x00))
    raw[imem + 0xEC] = 0  # BP: make the raw BP+60 selector deterministic
    raw[imem + 0x60 : imem + 0x63] = bytes.fromhex("A5 5A 3C")
    cpu = CPU(_make_memory(raw), reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.PC, 0)
    cpu.regs.set(RegisterName.I, 3 if counted else 0x5AA5)
    cpu.regs.set(RegisterName.F, 0x03)

    cpu.execute_instruction(0)

    assert (
        raw[0x4_06D0 : 0x4_06D0 + transfer_length]
        == bytes.fromhex("A5 5A 3C")[:transfer_length]
    )
    assert cpu.regs.get(RegisterName.I) == (0 if counted else 0x5AA5)
    assert cpu.regs.get(RegisterName.F) == 0x03
    assert cpu.regs.get(RegisterName.PC) == 5

    cpu.execute_instruction(5)
    assert cpu.regs.get(RegisterName.PC) == 6


@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize("f_input", [0x00, 0x01, 0x02, 0x03])
def test_exp_c2_exchanges_all_24_bits_and_preserves_flags(
    backend: str, f_input: int
) -> None:
    imem = INTERNAL_MEMORY_START
    raw = bytearray(ADDRESS_SPACE_SIZE)
    raw[:4] = bytes.fromhex("32C22050")  # PRE (n),(n); EXP (20),(50)
    raw[imem + 0xEC] = 0x10  # BP must not affect PRE32 direct operands
    raw[imem + 0x20 : imem + 0x23] = bytes.fromhex("1122A8")
    raw[imem + 0x50 : imem + 0x53] = bytes.fromhex("3344B9")
    raw[imem + 0x30] = 0xA1  # BP-relative sentinel
    raw[imem + 0x60] = 0xB2  # BP-relative sentinel
    cpu = CPU(_make_memory(raw), reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.PC, 0)
    cpu.regs.set(RegisterName.F, f_input)

    cpu.execute_instruction(0)

    assert raw[imem + 0x20 : imem + 0x23] == bytes.fromhex("3344B9")
    assert raw[imem + 0x50 : imem + 0x53] == bytes.fromhex("1122A8")
    assert raw[imem + 0x30] == 0xA1
    assert raw[imem + 0x60] == 0xB2
    assert cpu.regs.get(RegisterName.F) == f_input


@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize(
    ("instruction", "expected"),
    [
        pytest.param("32C24040", "A1B2C3D4E5", id="exact-alias"),
        pytest.param("32C24140", "B2C3D4A1E5", id="first-plus-one"),
        pytest.param("32C24041", "B2C3D4A1E5", id="first-minus-one"),
    ],
)
def test_exp_c2_overlap_matches_pc_e500_pairwise_byte_order(
    backend: str, instruction: str, expected: str
) -> None:
    """Pin the exact three-case overlap matrix captured on a PC-E500."""

    imem = INTERNAL_MEMORY_START
    raw = bytearray(ADDRESS_SPACE_SIZE)
    raw[:4] = bytes.fromhex(instruction)
    raw[imem + 0x40 : imem + 0x45] = bytes.fromhex("A1B2C3D4E5")
    cpu = CPU(_make_memory(raw), reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.PC, 0)
    cpu.regs.set(RegisterName.I, 0x5AA5)
    cpu.regs.set(RegisterName.F, 0x03)

    cpu.execute_instruction(0)

    assert raw[imem + 0x40 : imem + 0x45] == bytes.fromhex(expected)
    assert cpu.regs.get(RegisterName.I) == 0x5AA5
    assert cpu.regs.get(RegisterName.F) == 0x03


@pytest.mark.parametrize("backend", ["python", "llama"])
def test_mvp_ca_copies_all_24_internal_memory_bits(backend: str) -> None:
    imem = INTERNAL_MEMORY_START
    raw = _execute_raw24_case(
        backend,
        bytes.fromhex("32CA2030"),  # PRE (n),(n); MVP (20),(30)
        {
            imem + 0x30: 0x33,
            imem + 0x31: 0x44,
            imem + 0x32: 0xB9,
        },
    )

    assert raw[imem + 0x20 : imem + 0x23] == bytes.fromhex("3344B9")


@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize(
    ("left", "right", "expected_c", "expected_z"),
    [
        pytest.param(0xF00080, 0x000080, 0, 0, id="raw-left-greater"),
        pytest.param(0x000080, 0xF00080, 1, 0, id="raw-left-less"),
        pytest.param(0xF00080, 0xF00080, 0, 1, id="raw-equal"),
    ],
)
def test_cmpp_c7_matches_hardware_raw24_flags(
    backend: str,
    left: int,
    right: int,
    expected_c: int,
    expected_z: int,
) -> None:
    imem = INTERNAL_MEMORY_START
    raw = bytearray(ADDRESS_SPACE_SIZE)
    raw[:4] = bytes.fromhex("32C71020")  # PRE (n),(n); CMPP (10),(20)
    raw[imem + 0x10 : imem + 0x13] = left.to_bytes(3, "little")
    raw[imem + 0x20 : imem + 0x23] = right.to_bytes(3, "little")

    cpu = CPU(_make_memory(raw), reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.PC, 0)
    cpu.regs.set(RegisterName.F, 0x03)
    cpu.execute_instruction(0)

    assert cpu.regs.get(RegisterName.FC) == expected_c
    assert cpu.regs.get(RegisterName.FZ) == expected_z


@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize(
    ("memory_value", "encoded_x", "expected_x", "expected_c", "expected_z"),
    [
        pytest.param(
            0xF00080,
            0x000080,
            0x000080,
            0,
            0,
            id="raw-memory-upper-nibble-differs",
        ),
        pytest.param(
            0x0C5AA5,
            0x3C5AA5,
            0x0C5AA5,
            0,
            1,
            id="noncanonical-x-normalizes-equal",
        ),
        pytest.param(
            0x3C5AA5,
            0x3C5AA5,
            0x0C5AA5,
            0,
            0,
            id="raw-memory-versus-normalized-x",
        ),
    ],
)
def test_cmpp_d7_matches_hardware_raw24_memory_and_normalized_x(
    backend: str,
    memory_value: int,
    encoded_x: int,
    expected_x: int,
    expected_c: int,
    expected_z: int,
) -> None:
    imem = INTERNAL_MEMORY_START
    raw = bytearray(ADDRESS_SPACE_SIZE)
    raw[:4] = bytes((0x0C, *encoded_x.to_bytes(3, "little")))  # MV X,imm20
    raw[4:7] = bytes.fromhex("D70410")  # CMPP (BP+10),X
    raw[imem + 0x10 : imem + 0x13] = memory_value.to_bytes(3, "little")

    cpu = CPU(_make_memory(raw), reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.PC, 0)
    cpu.regs.set(RegisterName.F, 0x03)
    cpu.execute_instruction(0)

    # The captured raw 0x3C5AA5 immediate is architecturally observed through
    # X as 0x0C5AA5 before D7 compares it with the unmasked memory triple.
    assert cpu.regs.get(RegisterName.X) == expected_x
    assert cpu.regs.get(RegisterName.F) == 0x03

    cpu.execute_instruction(4)

    assert cpu.regs.get(RegisterName.FC) == expected_c
    assert cpu.regs.get(RegisterName.FZ) == expected_z


@pytest.mark.parametrize("backend", ["python", "llama"])
def test_mvp_d2_copies_all_24_external_memory_bits_into_imem(backend: str) -> None:
    imem = INTERNAL_MEMORY_START
    raw = _execute_raw24_case(
        backend,
        bytes.fromhex("30D220001000"),  # PRE (n); MVP (20),[01000]
        {0x1000: 0x55, 0x1001: 0x66, 0x1002: 0xC7},
    )

    assert raw[imem + 0x20 : imem + 0x23] == bytes.fromhex("5566C7")


@pytest.mark.parametrize("backend", ["python", "llama"])
def test_mvp_da_copies_all_24_imem_bits_to_external_memory(backend: str) -> None:
    imem = INTERNAL_MEMORY_START
    raw = _execute_raw24_case(
        backend,
        bytes.fromhex("30DA00100030"),  # PRE (n); MVP [01000],(30)
        {
            imem + 0x30: 0x33,
            imem + 0x31: 0x44,
            imem + 0x32: 0xB9,
        },
    )

    assert raw[0x1000:0x1003] == bytes.fromhex("3344B9")


@pytest.mark.parametrize("backend", ["python", "llama"])
def test_mvp_dc_preserves_raw_24_bit_immediate(backend: str) -> None:
    imem = INTERNAL_MEMORY_START
    raw = _execute_raw24_case(
        backend,
        bytes.fromhex("30DC20A55A3C"),  # PRE (n); MVP (20),3C5AA5
        {},
    )

    assert raw[imem + 0x20 : imem + 0x23] == bytes.fromhex("A55A3C")
