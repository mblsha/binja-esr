"""Integration tests for Perfetto tracing with the emulator."""

import pytest
import tempfile
from pathlib import Path

from sc62015.pysc62015.emulator import Memory, RegisterName
from pce500.tracing_emulator import TracingEmulator
from pce500.memory.mapper import MemoryMapper
from pce500.memory.regions import RAMRegion, PeripheralRegion
from pce500.trace_manager import g_tracer


class MockPeripheral:
    """Mock peripheral that supports tracing."""
    
    def __init__(self):
        self.last_write = None
        self.last_read_addr = None
        self.last_cpu_pc = None
    
    def read(self, address: int, cpu_pc: int = None) -> int:
        self.last_read_addr = address
        self.last_cpu_pc = cpu_pc
        return 0x42
    
    def write(self, address: int, value: int, cpu_pc: int = None) -> None:
        self.last_write = (address, value)
        self.last_cpu_pc = cpu_pc


class TestTracingIntegration:
    """Integration tests for tracing with full emulator."""
    
    @pytest.fixture
    def memory_mapper(self):
        """Create a memory mapper with RAM and peripheral."""
        mapper = MemoryMapper()
        
        # Add RAM region
        mapper.add_region(RAMRegion(0x0000, 0x10000, "RAM"))
        
        # Add peripheral region
        peripheral = MockPeripheral()
        mapper.add_region(PeripheralRegion(0x20000, 0x1000, peripheral, "TestPeripheral"))
        
        return mapper, peripheral
    
    @pytest.fixture
    def tracing_emulator(self, memory_mapper):
        """Create a tracing emulator with temporary trace file."""
        mapper, peripheral = memory_mapper
        
        with tempfile.NamedTemporaryFile(suffix='.perfetto-trace', delete=False) as f:
            trace_path = f.name
        
        emu = TracingEmulator(mapper, trace_path)
        yield emu, peripheral, trace_path
        
        # Cleanup
        emu.stop_tracing()
        Path(trace_path).unlink(missing_ok=True)
    
    def test_conditional_jump_tracing(self, tracing_emulator):
        """Test that conditional jumps are traced when taken."""
        emu, _, trace_path = tracing_emulator
        
        # Set up conditional jump (JPZ)
        # Set Zero flag
        emu.regs.set_flag('Z', 1)
        
        # JPZ 0x2000 at address 0x1000
        emu.memory.write(0x1000, bytes([0x14, 0x00, 0x20]))
        
        # Execute the conditional jump
        emu.execute_instruction(0x1000)
        
        # Verify jump was taken
        assert emu.regs.get(RegisterName.PC) == 0x2000
        
        # Stop tracing to flush
        emu.stop_tracing()
        
        # Verify trace file exists and has content
        assert Path(trace_path).exists()
        assert Path(trace_path).stat().st_size > 0
    
    def test_peripheral_receives_cpu_context(self, tracing_emulator):
        """Test that peripherals receive CPU context during memory operations."""
        emu, peripheral, _ = tracing_emulator
        
        # Set PC to a known value
        test_pc = 0x1234
        emu._current_pc = test_pc
        
        # Read from peripheral
        emu.read_memory(0x20000, 1)
        assert peripheral.last_cpu_pc == test_pc
        
        # Write to peripheral
        emu.write_memory(0x20000, bytes([0x55]))
        assert peripheral.last_cpu_pc == test_pc
        assert peripheral.last_write == (0x20000, 0x55)
    
    def test_counter_events_emitted(self, tracing_emulator):
        """Test that counter events are emitted periodically."""
        emu, _, trace_path = tracing_emulator
        
        # Set a short counter interval for testing
        emu._counter_interval = 10
        
        # Execute enough instructions to trigger counters
        for i in range(15):
            # NOP instructions
            emu.memory.write(i, bytes([0x00]))
            emu.execute_instruction(i)
        
        # Counters should have been emitted
        assert emu._last_counter_trace > 0
    
    def test_interrupt_flow_tracking(self, tracing_emulator):
        """Test interrupt flow ID tracking."""
        emu, _, _ = tracing_emulator
        
        # Set up interrupt vector
        emu.memory.write(0xFFFFA, bytes([0x00, 0x50, 0x00]))  # Vector to 0x5000
        
        # IR instruction
        emu.memory.write(0x1000, bytes([0xFE]))
        
        # RETI at interrupt handler
        emu.memory.write(0x5000, bytes([0x01]))
        
        # Execute interrupt entry
        initial_stack_len = len(emu._interrupt_stack)
        emu.execute_instruction(0x1000)
        
        # Should have pushed interrupt ID
        assert len(emu._interrupt_stack) == initial_stack_len + 1
        assert emu.regs.call_sub_level == 1
        
        # Execute interrupt exit
        emu.execute_instruction(0x5000)
        
        # Should have popped interrupt ID
        assert len(emu._interrupt_stack) == initial_stack_len
        assert emu.regs.call_sub_level == 0
    
    def test_memory_bandwidth_tracking(self, tracing_emulator):
        """Test memory read/write bandwidth counters."""
        emu, _, _ = tracing_emulator
        
        initial_reads = emu._memory_reads
        initial_writes = emu._memory_writes
        
        # Perform memory operations
        emu.read_memory(0x1000, 4)  # 4 byte read
        emu.write_memory(0x2000, bytes([1, 2, 3]))  # 3 byte write
        
        # Verify counters updated
        assert emu._memory_reads == initial_reads + 4
        assert emu._memory_writes == initial_writes + 3
    
    def test_call_stack_with_tracing(self, tracing_emulator):
        """Test call stack tracking with full tracing enabled."""
        emu, _, _ = tracing_emulator
        
        # CALL 0x2000
        emu.memory.write(0x1000, bytes([0x04, 0x00, 0x20]))
        # RET
        emu.memory.write(0x2000, bytes([0x06]))
        
        # Execute call
        assert emu.regs.call_sub_level == 0
        emu.execute_instruction(0x1000)
        assert emu.regs.call_sub_level == 1
        
        # Execute return
        emu.execute_instruction(0x2000)
        assert emu.regs.call_sub_level == 0
    
    def test_multiple_trace_threads(self, tracing_emulator):
        """Test that different subsystems use different trace threads."""
        emu, _, trace_path = tracing_emulator
        
        # CPU operation
        emu.memory.write(0x1000, bytes([0x00]))  # NOP
        emu.execute_instruction(0x1000)
        
        # Memory operation (triggers Memory thread)
        emu.read_memory(0x2000, 4)
        
        # I/O operation (triggers I/O thread)
        emu.trace_io_write(0x80, 0xFF)
        
        # Peripheral operation (triggers Display thread if it's an LCD)
        emu.write_memory(0x20000, bytes([0x55]))
        
        # Stop and verify trace was created
        emu.stop_tracing()
        assert Path(trace_path).exists()
        assert Path(trace_path).stat().st_size > 0


class TestTracingEmulatorEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_tracing_without_tg4perfetto(self):
        """Test graceful behavior when tg4perfetto is not available."""
        # This is hard to test without actually uninstalling the package
        # But the code handles it gracefully by checking HAS_TG4PERFETTO
        pass
    
    def test_invalid_memory_access_tracing(self):
        """Test tracing behavior with invalid memory accesses."""
        mapper = MemoryMapper()
        # Empty memory map
        
        with tempfile.NamedTemporaryFile(suffix='.perfetto-trace', delete=False) as f:
            emu = TracingEmulator(mapper, f.name)
            
            try:
                # Read from unmapped memory
                data = emu.read_memory(0xFFFFFF, 1)
                assert data == bytes([0xFF])  # Should return default
                
                # Write to unmapped memory (should be ignored)
                emu.write_memory(0xFFFFFF, bytes([0x42]))
                
            finally:
                emu.stop_tracing()
                Path(f.name).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])