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
def test_exp_c2_exchanges_canonical_20_bit_pointer_images(backend: str) -> None:
    imem = INTERNAL_MEMORY_START
    raw = _execute_raw24_case(
        backend,
        bytes.fromhex("32C22050"),  # PRE (n),(n); EXP (20),(50)
        {
            imem + 0xEC: 0x10,  # BP must not affect PRE32 direct operands
            imem + 0x20: 0x11,
            imem + 0x21: 0x22,
            imem + 0x22: 0x08,
            imem + 0x50: 0x33,
            imem + 0x51: 0x44,
            imem + 0x52: 0x09,
            imem + 0x30: 0xA1,  # BP-relative sentinel
            imem + 0x60: 0xB2,  # BP-relative sentinel
        },
    )

    assert raw[imem + 0x20 : imem + 0x23] == bytes.fromhex("334409")
    assert raw[imem + 0x50 : imem + 0x53] == bytes.fromhex("112208")
    assert raw[imem + 0x30] == 0xA1
    assert raw[imem + 0x60] == 0xB2


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
