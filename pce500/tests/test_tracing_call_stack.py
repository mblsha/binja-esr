"""Unit tests for call stack tracking in the tracing emulator."""

import pytest
from pathlib import Path
import tempfile

from sc62015.pysc62015.emulator import Memory, RegisterName
from pce500.tracing_emulator import TracingEmulator


class TestMemory(Memory):
    """Simple memory implementation for testing."""
    
    def __init__(self, size: int = 0x1000000):  # 16MB to cover 24-bit address space
        self._memory = bytearray(size)
    
    def read(self, address: int, size: int) -> bytes:
        # Make sure we're reading within bounds
        if address + size > len(self._memory):
            # For out of bounds reads, return zeros
            return bytes(size)
        return bytes(self._memory[address:address + size])
    
    def write(self, address: int, data: bytes) -> None:
        for i, byte in enumerate(data):
            self._memory[address + i] = byte
    
    def read_byte(self, address: int) -> int:
        return self._memory[address]
    
    def write_byte(self, address: int, value: int) -> None:
        self._memory[address] = value & 0xFF
    
    def read_bytes(self, size: int, address: int) -> int:
        """Read bytes from memory and return as integer (little-endian).
        
        Note: The signature has size before address, matching Binary Ninja's API.
        """
        data = self.read(address, size)
        result = 0
        for i, byte in enumerate(data):
            result |= byte << (i * 8)
        return result
    
    def write_bytes(self, size: int, address: int, value: int) -> None:
        """Write integer value to memory as bytes (little-endian).
        
        Note: The signature has size before address, matching Binary Ninja's API.
        """
        data = []
        for i in range(size):
            data.append((value >> (i * 8)) & 0xFF)
        self.write(address, bytes(data))


@pytest.fixture
def memory():
    """Create a test memory instance."""
    return TestMemory()


@pytest.fixture
def emulator(memory):
    """Create a tracing emulator with a temporary trace file."""
    # Create temporary trace file
    with tempfile.NamedTemporaryFile(suffix='.perfetto-trace', delete=False) as f:
        trace_path = f.name
    
    emu = TracingEmulator(memory, trace_path)
    yield emu
    
    # Cleanup
    emu.stop_tracing()
    Path(trace_path).unlink(missing_ok=True)


class TestCallStackTracking:
    """Test call stack tracking functionality."""
    
    def setup_stack(self, emulator):
        """Initialize stack pointer to a reasonable value."""
        # Initialize S register (system stack pointer) to point to valid memory
        # Stack grows downward, so set it to a high address
        emulator.regs.set(RegisterName.S, 0xFFFF)
    
    def test_simple_call_return(self, memory, emulator):
        """Test basic CALL/RET sequence."""
        # CALL 0x2000 (opcode 0x04, little-endian address)
        memory.write(0x1000, bytes([0x04, 0x00, 0x20]))
        # RET (opcode 0x06)
        memory.write(0x2000, bytes([0x06]))
        # NOP after RET
        memory.write(0x2001, bytes([0x00]))
        
        # Setup stack and initial state
        self.setup_stack(emulator)
        assert emulator.regs.call_sub_level == 0
        
        # Execute CALL
        emulator.execute_instruction(0x1000)
        assert emulator.regs.call_sub_level == 1
        assert emulator.regs.get(RegisterName.PC) == 0x2000
        
        # Execute RET
        emulator.execute_instruction(0x2000)
        assert emulator.regs.call_sub_level == 0
    
    def test_nested_calls(self, memory, emulator):
        """Test nested CALL instructions."""
        # Setup stack
        self.setup_stack(emulator)
        
        # func1: CALL func2
        memory.write(0x1000, bytes([0x04, 0x00, 0x20]))  # CALL 0x2000
        
        # func2: CALL func3
        memory.write(0x2000, bytes([0x04, 0x00, 0x30]))  # CALL 0x3000
        
        # func3: RET
        memory.write(0x3000, bytes([0x06]))  # RET
        
        # func2 continuation: RET
        memory.write(0x2003, bytes([0x06]))  # RET
        
        # Execute calls
        emulator.execute_instruction(0x1000)  # CALL func2
        assert emulator.regs.call_sub_level == 1
        
        emulator.execute_instruction(0x2000)  # CALL func3
        assert emulator.regs.call_sub_level == 2
        
        # Execute returns
        emulator.execute_instruction(0x3000)  # RET from func3
        assert emulator.regs.call_sub_level == 1
        
        emulator.execute_instruction(0x2003)  # RET from func2
        assert emulator.regs.call_sub_level == 0
    
    def test_interrupt_handling(self, memory, emulator):
        """Test IR/RETI sequence."""
        # Set interrupt vector at 0xFFFFA pointing to 0x3000
        memory.write(0xFFFFA, bytes([0x00, 0x30, 0x00]))
        
        # IR instruction (opcode 0xFE)
        memory.write(0x1000, bytes([0xFE]))
        
        # Interrupt handler at 0x3000
        memory.write(0x3000, bytes([0x01]))  # RETI
        
        # Set PC to a known value before executing IR
        emulator.regs.set(RegisterName.PC, 0x1000)
        
        # Execute interrupt
        emulator.execute_instruction(0x1000)  # IR
        assert emulator.regs.call_sub_level == 1
        assert emulator._interrupt_stack  # Should have interrupt ID
        
        # Return from interrupt
        emulator.execute_instruction(0x3000)  # RETI
        assert emulator.regs.call_sub_level == 0
        assert not emulator._interrupt_stack  # Should be empty
    
    def test_callf_retf(self, memory, emulator):
        """Test far call/return (CALLF/RETF)."""
        # Setup stack
        self.setup_stack(emulator)
        
        # CALLF 0x012345 (opcode 0x05, little-endian 20-bit address)
        memory.write(0x1000, bytes([0x05, 0x45, 0x23, 0x01]))
        
        # RETF at target (opcode 0x07)
        memory.write(0x12345, bytes([0x07]))
        
        # Execute far call
        emulator.execute_instruction(0x1000)
        assert emulator.regs.call_sub_level == 1
        assert emulator.regs.get(RegisterName.PC) == 0x12345
        
        # Execute far return
        emulator.execute_instruction(0x12345)
        assert emulator.regs.call_sub_level == 0
    
    def test_stack_underflow_protection(self, memory, emulator):
        """Test protection against call stack underflow."""
        # RET without matching CALL
        memory.write(0x1000, bytes([0x06]))  # RET
        
        # Should not go negative
        assert emulator.regs.call_sub_level == 0
        emulator.execute_instruction(0x1000)
        assert emulator.regs.call_sub_level == 0
        
        # Multiple RETs
        memory.write(0x1001, bytes([0x06]))  # RET
        memory.write(0x1002, bytes([0x06]))  # RET
        
        emulator.execute_instruction(0x1001)
        assert emulator.regs.call_sub_level == 0
        
        emulator.execute_instruction(0x1002)
        assert emulator.regs.call_sub_level == 0
    
    def test_tail_call_pattern(self, memory, emulator):
        """Test tail call optimization pattern (CALL followed by JP)."""
        # func1: CALL func2
        memory.write(0x1000, bytes([0x04, 0x00, 0x20]))  # CALL 0x2000
        
        # func2: JP func3 (tail call)
        memory.write(0x2000, bytes([0x02, 0x00, 0x30]))  # JP 0x3000
        
        # func3: RET (returns to func1's caller)
        memory.write(0x3000, bytes([0x06]))  # RET
        
        # Execute pattern
        emulator.execute_instruction(0x1000)  # CALL func2
        assert emulator.regs.call_sub_level == 1
        
        emulator.execute_instruction(0x2000)  # JP func3 (not a call)
        assert emulator.regs.call_sub_level == 1  # No change
        assert emulator.regs.get(RegisterName.PC) == 0x3000
        
        emulator.execute_instruction(0x3000)  # RET
        assert emulator.regs.call_sub_level == 0
    
    def test_mixed_call_types(self, memory, emulator):
        """Test mixing regular and far calls."""
        # CALL 0x2000
        memory.write(0x1000, bytes([0x04, 0x00, 0x20]))
        
        # At 0x2000: CALLF 0x30000
        memory.write(0x2000, bytes([0x05, 0x00, 0x00, 0x03]))
        
        # At 0x30000: RETF
        memory.write(0x30000, bytes([0x07]))
        
        # Back at 0x2004: RET
        memory.write(0x2004, bytes([0x06]))
        
        # Execute sequence
        emulator.execute_instruction(0x1000)  # CALL
        assert emulator.regs.call_sub_level == 1
        
        emulator.execute_instruction(0x2000)  # CALLF
        assert emulator.regs.call_sub_level == 2
        
        emulator.execute_instruction(0x30000)  # RETF
        assert emulator.regs.call_sub_level == 1
        
        emulator.execute_instruction(0x2004)  # RET
        assert emulator.regs.call_sub_level == 0
    
    def test_interrupt_during_call(self, memory, emulator):
        """Test interrupt occurring during a function call."""
        # Set interrupt vector
        memory.write(0xFFFFA, bytes([0x00, 0x40, 0x00]))
        
        # func1: CALL func2
        memory.write(0x1000, bytes([0x04, 0x00, 0x20]))
        
        # func2: IR (interrupt occurs)
        memory.write(0x2000, bytes([0xFE]))
        
        # ISR at 0x4000: RETI
        memory.write(0x4000, bytes([0x01]))
        
        # func2 continuation: RET
        memory.write(0x2001, bytes([0x06]))
        
        # Execute sequence
        emulator.execute_instruction(0x1000)  # CALL func2
        assert emulator.regs.call_sub_level == 1
        
        emulator.execute_instruction(0x2000)  # IR
        assert emulator.regs.call_sub_level == 2
        assert len(emulator._interrupt_stack) == 1
        
        emulator.execute_instruction(0x4000)  # RETI
        assert emulator.regs.call_sub_level == 1
        assert len(emulator._interrupt_stack) == 0
        
        emulator.execute_instruction(0x2001)  # RET from func2
        assert emulator.regs.call_sub_level == 0
    
    def test_conditional_jump_not_affecting_call_depth(self, memory, emulator):
        """Test that conditional jumps don't affect call depth."""
        # Set Zero flag for JPZ
        emulator.regs.set_flag('Z', 1)
        
        # JPZ 0x2000 (opcode 0x14)
        memory.write(0x1000, bytes([0x14, 0x00, 0x20]))
        
        # Target has RET (should not affect depth)
        memory.write(0x2000, bytes([0x06]))
        
        # Execute conditional jump
        initial_depth = emulator.regs.call_sub_level
        emulator.execute_instruction(0x1000)  # JPZ (taken)
        assert emulator.regs.call_sub_level == initial_depth  # No change
        assert emulator.regs.get(RegisterName.PC) == 0x2000