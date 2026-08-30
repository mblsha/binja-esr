"""Test HALT/OFF/RESET instruction behavior."""

import pytest
from binja_test_mocks.eval_llil import Memory
from .emulator import Emulator, RegisterName
from .sc_asm import Assembler
from .constants import ADDRESS_SPACE_SIZE, INTERNAL_MEMORY_START
from .instr.opcodes import IMEMRegisters


def create_memory():
    """Create a memory object with backing storage."""
    raw = bytearray([0x00] * ADDRESS_SPACE_SIZE)

    def read_mem(addr: int) -> int:
        if addr < 0 or addr >= len(raw):
            raise IndexError(f"Read address {addr:04x} out of bounds")
        return raw[addr]

    def write_mem(addr: int, value: int) -> None:
        if addr < 0 or addr >= len(raw):
            raise IndexError(f"Write address {addr:04x} out of bounds")
        raw[addr] = value & 0xFF

    return Memory(read_mem, write_mem), raw


def test_reset_disabled_starts_in_running_power_state():
    memory, _ = create_memory()
    emu = Emulator(memory, reset_on_init=False)
    assert emu.state.halted is False
    assert getattr(emu.state, "power_state") == "running"


def test_halt_model_contract():
    """Pin the provisional HALT register model; this is not hardware proof."""
    memory, _ = create_memory()

    # Set up initial values
    memory.write_byte(
        INTERNAL_MEMORY_START + IMEMRegisters.USR, 0xFF
    )  # USR with all bits set
    memory.write_byte(
        INTERNAL_MEMORY_START + IMEMRegisters.SSR, 0x00
    )  # SSR with all bits clear

    # Write HALT instruction at address 0x1000
    assembler = Assembler()
    bin_file = assembler.assemble("HALT")
    for i, byte in enumerate(bin_file.segments[0].data):
        memory.write_byte(0x1000 + i, byte)

    # Create emulator without reset
    emu = Emulator(memory, reset_on_init=False)
    emu.regs.set(RegisterName.PC, 0x1000)

    # Execute HALT
    emu.execute_instruction(0x1000)

    # Check register modifications
    usr = memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.USR)
    assert (usr & 0x3F) == 0x18  # Bits 0-5: only bits 3,4 should be set

    ssr = memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.SSR)
    assert (ssr & 0x04) == 0x04  # Bit 2 should be set

    # Check halted state
    assert emu.state.halted is True
    assert getattr(emu.state, "power_state") == "halted"


def test_off_model_contract():
    """Pin the provisional OFF register model and keep it distinct from HALT."""
    memory, _ = create_memory()

    # Set up initial values
    memory.write_byte(
        INTERNAL_MEMORY_START + IMEMRegisters.USR, 0xFF
    )  # USR with all bits set
    memory.write_byte(
        INTERNAL_MEMORY_START + IMEMRegisters.SSR, 0x00
    )  # SSR with all bits clear

    # Write OFF instruction at address 0x1000
    assembler = Assembler()
    bin_file = assembler.assemble("OFF")
    for i, byte in enumerate(bin_file.segments[0].data):
        memory.write_byte(0x1000 + i, byte)

    # Create emulator without reset
    emu = Emulator(memory, reset_on_init=False)
    emu.regs.set(RegisterName.PC, 0x1000)

    # Execute OFF
    emu.execute_instruction(0x1000)

    # Check register modifications (same as HALT)
    usr = memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.USR)
    assert (usr & 0x3F) == 0x18  # Bits 0-5: only bits 3,4 should be set

    ssr = memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.SSR)
    assert (ssr & 0x04) == 0x04  # Bit 2 should be set

    # Check halted state
    assert emu.state.halted is True
    assert getattr(emu.state, "power_state") == "off"


def test_reset_model_contract_and_distinct_vector_slot():
    """Pin the synthetic reset model and its distinct vector-slot contract."""
    memory, _ = create_memory()

    # Set up initial values
    memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.UCR, 0xFF)  # UCR
    memory.write_byte(
        INTERNAL_MEMORY_START + IMEMRegisters.USR, 0xFF
    )  # USR with all bits set
    memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR, 0xFF)  # ISR
    memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.SCR, 0xFF)  # SCR
    memory.write_byte(
        INTERNAL_MEMORY_START + IMEMRegisters.LCC, 0xFF
    )  # LCC with bit 7 set
    memory.write_byte(
        INTERNAL_MEMORY_START + IMEMRegisters.SSR, 0xFF
    )  # SSR with bit 2 set

    # Keep a distinct interrupt-vector decoy at FFFFA and place a synthetic
    # reset target in the separate FFFFD vector slot.
    memory.write_byte(0xFFFFA, 0xAA)
    memory.write_byte(0xFFFFB, 0xBB)
    memory.write_byte(0xFFFFC, 0x0C)
    memory.write_byte(0xFFFFD, 0x45)  # Low byte
    memory.write_byte(0xFFFFE, 0x23)  # Middle byte
    memory.write_byte(0xFFFFF, 0x01)  # High byte

    # Write RESET instruction at address 0x1000
    assembler = Assembler()
    bin_file = assembler.assemble("RESET")
    for i, byte in enumerate(bin_file.segments[0].data):
        memory.write_byte(0x1000 + i, byte)

    # Create emulator without reset
    emu = Emulator(memory, reset_on_init=False)
    emu.regs.set(RegisterName.PC, 0x1000)

    # Set some register values that should be retained
    emu.regs.set(RegisterName.A, 0x55)
    emu.regs.set(RegisterName.B, 0xAA)
    emu.regs.set(RegisterName.FC, 1)
    emu.regs.set(RegisterName.FZ, 1)

    # Execute RESET
    emu.execute_instruction(0x1000)

    # Check register modifications
    assert (
        memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.UCR) == 0x00
    )  # UCR reset
    assert (
        memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR) == 0x00
    )  # ISR reset (clears interrupt status)
    assert (
        memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.SCR) == 0x00
    )  # SCR reset

    usr = memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.USR)
    assert (usr & 0x3F) == 0x18  # Bits 0-5: only bits 3,4 should be set

    lcc = memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.LCC)
    assert (lcc & 0x80) == 0x00  # Bit 7 should be clear

    ssr = memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.SSR)
    assert (ssr & 0x04) == 0x00  # Bit 2 should be clear

    # Check PC set to reset vector
    assert emu.regs.get(RegisterName.PC) == 0x12345

    # Check retained registers
    assert emu.regs.get(RegisterName.A) == 0x55
    assert emu.regs.get(RegisterName.B) == 0xAA
    assert emu.regs.get(RegisterName.FC) == 1
    assert emu.regs.get(RegisterName.FZ) == 1
    assert emu.state.halted is False
    assert getattr(emu.state, "power_state") == "running"


def test_power_on_reset_model_contract():
    """Pin the provisional power-on register model and reset-vector fetch."""
    memory, _ = create_memory()

    # Set up initial values
    memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.UCR, 0xFF)  # UCR
    memory.write_byte(
        INTERNAL_MEMORY_START + IMEMRegisters.USR, 0xFF
    )  # USR with all bits set
    memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR, 0xFF)  # ISR
    memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.SCR, 0xFF)  # SCR
    memory.write_byte(
        INTERNAL_MEMORY_START + IMEMRegisters.LCC, 0xFF
    )  # LCC with bit 7 set
    memory.write_byte(
        INTERNAL_MEMORY_START + IMEMRegisters.SSR, 0xFF
    )  # SSR with bit 2 set

    # FFFFA is the interrupt vector; power-on RESET reads FFFFD instead.
    memory.write_byte(0xFFFFA, 0xAA)
    memory.write_byte(0xFFFFB, 0xBB)
    memory.write_byte(0xFFFFC, 0x0C)
    memory.write_byte(0xFFFFD, 0x21)  # Low byte
    memory.write_byte(0xFFFFE, 0x43)  # Middle byte
    memory.write_byte(0xFFFFF, 0x05)  # High byte

    # Create emulator with reset
    emu = Emulator(memory, reset_on_init=True)

    assert emu.state.halted is False
    assert getattr(emu.state, "power_state") == "running"

    # Check register modifications
    assert (
        memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.UCR) == 0x00
    )  # UCR reset
    assert (
        memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR) == 0x00
    )  # ISR reset (clears interrupt status)
    assert (
        memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.SCR) == 0x00
    )  # SCR reset

    usr = memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.USR)
    assert (usr & 0x3F) == 0x18  # Bits 0-5: only bits 3,4 should be set

    lcc = memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.LCC)
    assert (lcc & 0x80) == 0x00  # Bit 7 should be clear

    ssr = memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.SSR)
    assert (ssr & 0x04) == 0x00  # Bit 2 should be clear

    # Check PC set to reset vector
    assert emu.regs.get(RegisterName.PC) == 0x54321

    # Check halted state is false
    assert emu.state.halted is False


def test_failed_power_on_reset_restores_native_state_and_requires_complete_reset():
    raw = bytearray([0x00] * ADDRESS_SPACE_SIZE)
    raw[0] = 0x00  # NOP used to prove poisoned execution is blocked.
    raw[0xFFFFD:0x100000] = bytes.fromhex("452301")
    fail_reset_write = True

    def read_mem(addr: int) -> int:
        return raw[addr]

    def write_mem(addr: int, value: int) -> None:
        nonlocal fail_reset_write
        if fail_reset_write and addr == INTERNAL_MEMORY_START + IMEMRegisters.UCR:
            raise RuntimeError("reset write failed")
        raw[addr] = value & 0xFF

    emu = Emulator(Memory(read_mem, write_mem), reset_on_init=False)
    emu.regs.set(RegisterName.PC, 0x22222)
    emu.regs.set(RegisterName.A, 0x5A)
    emu.regs.call_sub_level = 4
    emu.state.halted = True
    setattr(emu.state, "power_state", "halted")

    with pytest.raises(RuntimeError, match="reset write failed"):
        emu.power_on_reset()

    assert emu.regs.get(RegisterName.PC) == 0x22222
    assert emu.regs.get(RegisterName.A) == 0x5A
    assert emu.regs.call_sub_level == 4
    assert emu.state.halted is True
    assert getattr(emu.state, "power_state") == "halted"
    with pytest.raises(RuntimeError, match="poisoned.*reset required"):
        emu.execute_instruction(0)

    fail_reset_write = False
    emu.power_on_reset()
    assert emu.regs.get(RegisterName.PC) == 0x12345
    assert emu.state.halted is False
    assert getattr(emu.state, "power_state") == "running"
    emu.execute_instruction(0)
    assert emu.regs.get(RegisterName.PC) == 1


if __name__ == "__main__":
    pytest.main([__file__])
