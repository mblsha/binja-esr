"""Unit tests for call stack tracking in the PC-E500 emulator."""

import pytest

from sc62015.pysc62015.emulator import RegisterName
from pce500 import PCE500Emulator


@pytest.fixture
def emulator():
    """Create a PC-E500 emulator with Perfetto tracing enabled."""
    # Create emulator with tracing
    emu = PCE500Emulator(perfetto_trace=True)
    yield emu
    # Cleanup
    emu.stop_tracing()


class TestCallStackTracking:
    """Test call stack tracking functionality."""
    
    def setup_stack(self, emulator):
        """Initialize stack pointer to a reasonable value."""
        # Initialize S register (system stack pointer) to point to valid memory
        # Stack grows downward, so set it to a high address in RAM
        emulator.cpu.regs.set(RegisterName.S, 0xBFFFF)
    
    def write_code(self, emulator, address: int, code: bytes):
        """Helper to write code bytes to memory."""
        for i, byte in enumerate(code):
            emulator.memory.write_byte(address + i, byte)
    
    def test_simple_call_return(self, emulator):
        """Test basic CALL/RET sequence."""
        # Use internal RAM region (0xB8000-0xBFFFF)
        # CALL 0xB9000 (opcode 0x04, little-endian address)
        self.write_code(emulator, 0xB8000, bytes([0x04, 0x00, 0x90]))
        # RET (opcode 0x06)
        self.write_code(emulator, 0xB9000, bytes([0x06]))
        # NOP after RET
        self.write_code(emulator, 0xB9001, bytes([0x00]))
        
        # Setup stack and initial state
        self.setup_stack(emulator)
        assert emulator.call_depth == 0
        
        # Execute CALL
        emulator.cpu.regs.set(RegisterName.PC, 0xB8000)
        emulator.step()
        assert emulator.call_depth == 1
        assert emulator.cpu.regs.get(RegisterName.PC) == 0xB9000
        
        # Execute RET
        emulator.step()
        assert emulator.call_depth == 0
    
    def test_nested_calls(self, emulator):
        """Test nested CALL instructions."""
        # Setup stack
        self.setup_stack(emulator)
        
        # Use internal RAM region
        # func1: CALL func2
        self.write_code(emulator, 0xB8000, bytes([0x04, 0x00, 0x81]))  # CALL 0x8100
        
        # func2: CALL func3  
        self.write_code(emulator, 0xB8100, bytes([0x04, 0x00, 0x82]))  # CALL 0x8200
        
        # func3: RET
        self.write_code(emulator, 0xB8200, bytes([0x06]))  # RET
        
        # func2 continuation: RET
        self.write_code(emulator, 0xB8103, bytes([0x06]))  # RET
        
        # Execute calls
        emulator.cpu.regs.set(RegisterName.PC, 0xB8000)
        emulator.step()  # CALL func2
        assert emulator.call_depth == 1
        
        emulator.step()  # CALL func3
        assert emulator.call_depth == 2
        
        # Execute returns
        emulator.step()  # RET from func3
        assert emulator.call_depth == 1
        
        emulator.cpu.regs.set(RegisterName.PC, 0xB8103)
        emulator.step()  # RET from func2
        assert emulator.call_depth == 0
    
    def test_interrupt_handling(self, emulator):
        """Test IR/RETI sequence."""
        # Create a minimal ROM with interrupt vector at 0xFFFFA pointing to 0xB8300
        rom_size = 0x40000  # 256KB
        rom_data = bytearray(b'\xFF' * rom_size)
        # Set interrupt vector at offset 0x1FFFA (0xFFFFA - 0xE0000)
        rom_data[0x1FFFA:0x1FFFD] = bytes([0x00, 0x83, 0x0B])  # Little-endian 0x0B8300
        emulator.load_rom(bytes(rom_data))
    
        # IR instruction (opcode 0xFE) in RAM
        self.write_code(emulator, 0xB8000, bytes([0xFE]))
    
        # Interrupt handler at 0xB8300
        self.write_code(emulator, 0xB8300, bytes([0x01]))  # RETI
    
        # Set PC to a known value before executing IR
        emulator.cpu.regs.set(RegisterName.PC, 0xB8000)
        self.setup_stack(emulator)
        
        # Execute interrupt
        emulator.step()  # IR
        assert emulator.call_depth == 1
        assert emulator._interrupt_stack  # Should have interrupt ID
        assert emulator.cpu.regs.get(RegisterName.PC) == 0xB8300
        
        # Execute return from interrupt
        emulator.step()  # RETI
        assert emulator.call_depth == 0
        assert not emulator._interrupt_stack  # Should be empty
    
    def test_callf_retf(self, emulator):
        """Test CALLF/RETF (far call/return) instructions."""
        # CALLF 0xB8400 (opcode 0x05, 20-bit address in little-endian)
        self.write_code(emulator, 0xB8000, bytes([0x05, 0x00, 0x84, 0x0B]))
        
        # RETF at target
        self.write_code(emulator, 0xB8400, bytes([0x07]))
        
        self.setup_stack(emulator)
        
        # Execute CALLF
        emulator.cpu.regs.set(RegisterName.PC, 0xB8000)
        emulator.step()
        assert emulator.call_depth == 1
        assert emulator.cpu.regs.get(RegisterName.PC) == 0xB8400
        
        # Execute RETF
        emulator.step()
        assert emulator.call_depth == 0
    
    def test_stack_underflow_protection(self, emulator):
        """Test that call stack depth doesn't go negative."""
        # RET without matching CALL
        self.write_code(emulator, 0xB8000, bytes([0x06]))
        
        self.setup_stack(emulator)
        assert emulator.call_depth == 0
        
        # Execute RET - should not go negative
        emulator.cpu.regs.set(RegisterName.PC, 0xB8000)
        emulator.step()
        assert emulator.call_depth == 0  # Should stay at 0, not go negative
    
    def test_tail_call_pattern(self, emulator):
        """Test tail call optimization pattern (JP after CALL)."""
        # func1: CALL func2; JP func3
        self.write_code(emulator, 0xB8000, bytes([0x04, 0x00, 0x81]))  # CALL 0x8100
        self.write_code(emulator, 0xB8003, bytes([0x0E, 0x00, 0x82]))  # JP 0x8200
        
        # func2: RET
        self.write_code(emulator, 0xB8100, bytes([0x06]))
        
        # func3: RET
        self.write_code(emulator, 0xB8200, bytes([0x06]))
        
        self.setup_stack(emulator)
        
        # Execute CALL func2
        emulator.cpu.regs.set(RegisterName.PC, 0xB8000)
        emulator.step()
        assert emulator.call_depth == 1
        
        # Execute RET from func2
        emulator.step()
        assert emulator.call_depth == 0
        # RET should return to address after CALL (0xB8003)
        pc_after_ret = emulator.cpu.regs.get(RegisterName.PC)
        assert pc_after_ret == 0xB8003, f"Expected PC=0xB8003 after RET, got 0x{pc_after_ret:06X}"
        
        # Execute JP func3 (tail call - no stack change)
        emulator.step()
        assert emulator.call_depth == 0  # JP doesn't change call depth
        # Note: We're testing call_depth tracking, not the JP instruction execution itself
    
    def test_mixed_call_types(self, emulator):
        """Test mixing regular and far calls."""
        # Regular CALL
        self.write_code(emulator, 0xB8000, bytes([0x04, 0x00, 0x81]))  # CALL 0x8100
        
        # Far CALL inside regular function
        self.write_code(emulator, 0xB8100, bytes([0x05, 0x00, 0x85, 0x0B]))  # CALLF 0xB8500
        
        # RETF from far call
        self.write_code(emulator, 0xB8500, bytes([0x07]))
        
        # RET from regular call
        self.write_code(emulator, 0xB8104, bytes([0x06]))
        
        self.setup_stack(emulator)
        
        # Execute regular CALL
        emulator.cpu.regs.set(RegisterName.PC, 0xB8000)
        emulator.step()
        assert emulator.call_depth == 1
        
        # Execute far CALL
        emulator.step()
        assert emulator.call_depth == 2
        
        # Execute RETF
        emulator.step()
        assert emulator.call_depth == 1
        assert emulator.cpu.regs.get(RegisterName.PC) == 0xB8104
        
        # Execute RET
        emulator.step()
        assert emulator.call_depth == 0
    
    def test_interrupt_during_call(self, emulator):
        """Test interrupt occurring during a function call."""
        # Create a minimal ROM with interrupt vector at 0xFFFFA pointing to 0xB8600
        rom_size = 0x40000  # 256KB
        rom_data = bytearray(b'\xFF' * rom_size)
        # Set interrupt vector at offset 0x1FFFA (0xFFFFA - 0xE0000)
        rom_data[0x1FFFA:0x1FFFD] = bytes([0x00, 0x86, 0x0B])  # Little-endian 0x0B8600
        emulator.load_rom(bytes(rom_data))
        
        # func1: CALL func2
        self.write_code(emulator, 0xB8000, bytes([0x04, 0x00, 0x81]))  # CALL 0x8100
        
        # func2: IR (interrupt), then RET
        self.write_code(emulator, 0xB8100, bytes([0xFE]))  # IR
        self.write_code(emulator, 0xB8101, bytes([0x06]))  # RET
        
        # Interrupt handler
        self.write_code(emulator, 0xB8600, bytes([0x01]))  # RETI
        
        self.setup_stack(emulator)
        
        # Execute CALL
        emulator.cpu.regs.set(RegisterName.PC, 0xB8000)
        emulator.step()
        assert emulator.call_depth == 1
        
        # Execute IR - should increase call depth and track interrupt
        emulator.step()
        assert emulator.call_depth == 2
        assert len(emulator._interrupt_stack) == 1
        
        # Execute RETI
        emulator.step()
        assert emulator.call_depth == 1
        assert len(emulator._interrupt_stack) == 0
        assert emulator.cpu.regs.get(RegisterName.PC) == 0xB8101
        
        # Execute RET
        emulator.step()
        assert emulator.call_depth == 0