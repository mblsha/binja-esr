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


def test_halt_instruction():
    """Test HALT instruction modifies registers correctly."""
    memory, _ = create_memory()
    
    # Set up initial values
    memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.USR, 0xFF)  # USR with all bits set
    memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.SSR, 0x00)  # SSR with all bits clear
    
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


def test_off_instruction():
    """Test OFF instruction modifies registers correctly."""
    memory, _ = create_memory()
    
    # Set up initial values
    memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.USR, 0xFF)  # USR with all bits set
    memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.SSR, 0x00)  # SSR with all bits clear
    
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


def test_reset_instruction():
    """Test RESET instruction modifies registers and jumps to reset vector."""
    memory, _ = create_memory()
    
    # Set up initial values
    memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.UCR, 0xFF)  # UCR
    memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.USR, 0xFF)  # USR with all bits set
    memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR, 0xFF)  # ISR
    memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.SCR, 0xFF)  # SCR
    memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.LCC, 0xFF)  # LCC with bit 7 set
    memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.SSR, 0xFF)  # SSR with bit 2 set
    
    # Set reset vector at 0xFFFFA to point to 0x12345
    memory.write_byte(0xFFFFA, 0x45)  # Low byte
    memory.write_byte(0xFFFFB, 0x23)  # Middle byte
    memory.write_byte(0xFFFFC, 0x01)  # High byte
    
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
    assert memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.UCR) == 0x00  # UCR reset
    assert memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR) == 0x00  # ISR reset (clears interrupt status)
    assert memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.SCR) == 0x00  # SCR reset
    
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


def test_power_on_reset():
    """Test power-on reset behavior."""
    memory, _ = create_memory()
    
    # Set up initial values
    memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.UCR, 0xFF)  # UCR
    memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.USR, 0xFF)  # USR with all bits set
    memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR, 0xFF)  # ISR
    memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.SCR, 0xFF)  # SCR
    memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.LCC, 0xFF)  # LCC with bit 7 set
    memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.SSR, 0xFF)  # SSR with bit 2 set
    
    # Set reset vector at 0xFFFFA to point to 0x54321
    memory.write_byte(0xFFFFA, 0x21)  # Low byte
    memory.write_byte(0xFFFFB, 0x43)  # Middle byte
    memory.write_byte(0xFFFFC, 0x05)  # High byte
    
    # Create emulator with reset
    emu = Emulator(memory, reset_on_init=True)
    
    # Check register modifications
    assert memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.UCR) == 0x00  # UCR reset
    assert memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR) == 0x00  # ISR reset (clears interrupt status)
    assert memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.SCR) == 0x00  # SCR reset
    
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


if __name__ == "__main__":
    pytest.main([__file__])