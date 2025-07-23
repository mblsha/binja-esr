"""Tests for the simplified PC-E500 emulator."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pce500.emulator import SimplifiedPCE500Emulator
from sc62015.pysc62015.emulator import RegisterName


class TestSimplifiedEmulator:
    """Test the simplified emulator implementation."""
    
    def test_basic_initialization(self):
        """Test emulator can be created and initialized."""
        emu = SimplifiedPCE500Emulator()
        assert emu is not None
        assert emu.memory is not None
        assert emu.lcd is not None
        assert emu.cpu is not None
        
    def test_rom_loading(self):
        """Test ROM loading functionality."""
        emu = SimplifiedPCE500Emulator()
        
        # Create test ROM data
        rom_data = bytes([0x00, 0x01, 0x02, 0x03] * 64)  # 256 bytes
        emu.load_rom(rom_data)
        
        # Verify ROM is loaded at correct address
        assert emu.memory.read_byte(0xC0000) == 0x00
        assert emu.memory.read_byte(0xC0001) == 0x01
        assert emu.memory.read_byte(0xC0002) == 0x02
        assert emu.memory.read_byte(0xC0003) == 0x03
        
    def test_memory_card_loading(self):
        """Test memory card loading."""
        emu = SimplifiedPCE500Emulator()
        
        # Load 8KB memory card
        card_data = bytes(range(256)) * 32  # 8KB
        emu.load_memory_card(card_data, 8192)
        
        # Verify card is accessible
        assert emu.memory.read_byte(0x40000) == 0
        assert emu.memory.read_byte(0x40001) == 1
        assert emu.memory.read_byte(0x400FF) == 255
        
    def test_ram_access(self):
        """Test internal RAM read/write."""
        emu = SimplifiedPCE500Emulator()
        
        # Write to RAM
        emu.memory.write_byte(0xB8000, 0x42)
        emu.memory.write_byte(0xB8001, 0x43)
        
        # Read back
        assert emu.memory.read_byte(0xB8000) == 0x42
        assert emu.memory.read_byte(0xB8001) == 0x43
        
    def test_lcd_write_read(self):
        """Test LCD controller access."""
        emu = SimplifiedPCE500Emulator()
        
        # Write display on command to left chip
        # Address: 0x20008 = instruction write to left chip
        emu.memory.write_byte(0x20008, 0x3F)  # Display on
        
        # Verify display state
        assert emu.lcd.display_on[0] is True
        
    def test_reset_functionality(self):
        """Test emulator reset."""
        emu = SimplifiedPCE500Emulator()
        
        # Load ROM with reset vector
        rom_data = bytearray(256 * 1024)
        # Set reset vector at 0xFFFFD to point to 0x1234
        rom_data[0x3FFFD] = 0x34  # Low byte
        rom_data[0x3FFFE] = 0x12  # Middle byte
        rom_data[0x3FFFF] = 0x00  # High byte
        emu.load_rom(bytes(rom_data))
        
        # Write some data to RAM
        emu.memory.write_byte(0xB8000, 0xFF)
        
        # Reset
        emu.reset()
        
        # Check PC is set to reset vector
        assert emu.cpu.regs.get(RegisterName.PC) == 0x1234
        
        # Check RAM is cleared
        assert emu.memory.read_byte(0xB8000) == 0
        
    def test_cpu_state_access(self):
        """Test getting CPU state."""
        emu = SimplifiedPCE500Emulator()
        
        # Set some register values
        emu.cpu.regs.set(RegisterName.PC, 0x1234)
        emu.cpu.regs.set(RegisterName.A, 0x42)
        emu.cpu.regs.set(RegisterName.B, 0x43)
        
        # Get CPU state
        state = emu.get_cpu_state()
        
        assert state['pc'] == 0x1234
        assert state['a'] == 0x42
        assert state['b'] == 0x43
        assert 'flags' in state
        assert state['cycles'] == 0
        
    def test_breakpoint_functionality(self):
        """Test breakpoints."""
        emu = SimplifiedPCE500Emulator()
        
        # Add breakpoint
        emu.add_breakpoint(0x1000)
        assert 0x1000 in emu.breakpoints
        
        # Remove breakpoint
        emu.remove_breakpoint(0x1000)
        assert 0x1000 not in emu.breakpoints
        
    def test_simple_trace(self):
        """Test simple tracing functionality."""
        emu = SimplifiedPCE500Emulator(trace_enabled=True)
        
        # Load simple program
        rom_data = bytearray(256 * 1024)
        rom_data[0] = 0x00  # NOP at 0xC0000
        emu.load_rom(bytes(rom_data))
        
        # Set PC
        emu.cpu.regs.set(RegisterName.PC, 0xC0000)
        
        # Execute one instruction
        emu.step()
        
        # Check trace
        assert len(emu.trace) == 1
        assert emu.trace[0][0] == 'exec'
        assert emu.trace[0][1] == 0xC0000
        
    def test_memory_info(self):
        """Test memory info display."""
        emu = SimplifiedPCE500Emulator()
        
        # Load ROM
        emu.load_rom(bytes(256 * 1024))
        
        # Get memory info
        info = emu.get_memory_info()
        assert "ROM: 0xC0000-0xFFFFF (256KB)" in info
        assert "RAM: 0xB8000-0xBFFFF (32KB)" in info
        assert "LCD: 0x20000-0x2FFFF" in info
        
    def test_performance_stats(self):
        """Test performance statistics."""
        emu = SimplifiedPCE500Emulator()
        
        # Get initial stats
        stats = emu.get_performance_stats()
        assert stats['instructions_executed'] == 0
        assert stats['elapsed_time'] >= 0
        assert stats['instructions_per_second'] == 0
        assert stats['speed_ratio'] == 0