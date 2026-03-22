from __future__ import annotations

from textwrap import dedent

import pytest

from binja_test_mocks import binja_api  # noqa: F401  # pyright: ignore
from binja_test_mocks.eval_llil import Memory

from sc62015.pysc62015 import CPU, RegisterName, available_backends
from sc62015.pysc62015.constants import ADDRESS_SPACE_SIZE
from sc62015.pysc62015.sc_asm import Assembler


CARD_ROM_UART_HELLO_SOURCE = dedent(
    """
    .ORG 0x10100

    start:
        PUSHU F
        PUSHU A
        PUSHU X
        MV X, message

    emit_next:
        MV A, [X++]
        CMP A, 0x00
        JPZ emit_done
        MV [0x1FFF1], A
        JP emit_next

    emit_done:
        POPU X
        POPU A
        POPU F
        RETF

    message:
        defm "HELLO FROM CE6"
        defb 0x0D, 0x0A, 0x00
    """
).strip()

# This is the current binary emitted for the hello payload above.
CARD_ROM_UART_HELLO_BYTES = bytes.fromhex(
    "2E 28 2C 0C 19 01 01 90 24 60 00 14 15 01 "
    "A8 F1 FF 01 02 07 01 3C 38 3E 07 "
    "48 45 4C 4C 4F 20 46 52 4F 4D 20 43 45 36 0D 0A 00"
)

ENTRY_POINT = 0x10100
WRAPPER_ADDR = 0x10000
ECHO_PORT_ADDR = 0x1FFF1
INITIAL_SYSTEM_SP = 0x2000
INITIAL_USER_SP = 0x3000
EXPECTED_TEXT = b"HELLO FROM CE6\r\n"
MAX_STEPS = 128


def _assemble_uart_hello() -> bytes:
    bin_file = Assembler().assemble(CARD_ROM_UART_HELLO_SOURCE)
    assert len(bin_file.segments) == 1
    segment = bin_file.segments[0]
    assert segment.minimum_address == ENTRY_POINT
    return bytes(segment.data)


def _make_memory(program: bytes) -> tuple[Memory, list[int]]:
    raw = bytearray(ADDRESS_SPACE_SIZE)
    writes_to_echo: list[int] = []

    # CALLF 0x10100; HALT
    raw[WRAPPER_ADDR : WRAPPER_ADDR + 5] = bytes((0x05, 0x00, 0x01, 0x01, 0xDE))
    raw[ENTRY_POINT : ENTRY_POINT + len(program)] = program

    def read(addr: int) -> int:
        if addr < 0 or addr >= len(raw):
            raise IndexError(f"Read address {addr:#x} out of bounds")
        return raw[addr]

    def write(addr: int, value: int) -> None:
        if addr < 0 or addr >= len(raw):
            raise IndexError(f"Write address {addr:#x} out of bounds")
        raw[addr] = value & 0xFF
        if addr == ECHO_PORT_ADDR:
            writes_to_echo.append(value & 0xFF)

    memory = Memory(read, write)
    setattr(memory, "_raw", raw)
    return memory, writes_to_echo


@pytest.mark.parametrize("backend", available_backends())
def test_card_rom_uart_hello_emits_expected_text(backend: str) -> None:
    program = _assemble_uart_hello()
    assert program == CARD_ROM_UART_HELLO_BYTES

    memory, writes_to_echo = _make_memory(program)
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.PC, WRAPPER_ADDR)
    cpu.regs.set(RegisterName.S, INITIAL_SYSTEM_SP)
    cpu.regs.set(RegisterName.U, INITIAL_USER_SP)
    cpu.regs.set(RegisterName.A, 0x5A)
    cpu.regs.set(RegisterName.X, 0x123456)
    cpu.regs.set(RegisterName.F, 0xA4)
    initial_a = cpu.regs.get(RegisterName.A)
    initial_x = cpu.regs.get(RegisterName.X)
    initial_f = cpu.regs.get(RegisterName.F)

    for _ in range(MAX_STEPS):
        cpu.execute_instruction(cpu.regs.get(RegisterName.PC))
        if cpu.state.halted:
            break
    else:
        pytest.fail("card-ROM hello wrapper did not halt within the step budget")

    assert cpu.state.halted is True
    assert bytes(writes_to_echo) == EXPECTED_TEXT
    assert cpu.regs.get(RegisterName.S) == INITIAL_SYSTEM_SP
    assert cpu.regs.get(RegisterName.U) == INITIAL_USER_SP
    assert cpu.regs.get(RegisterName.A) == initial_a
    assert cpu.regs.get(RegisterName.X) == initial_x
    assert cpu.regs.get(RegisterName.F) == initial_f
