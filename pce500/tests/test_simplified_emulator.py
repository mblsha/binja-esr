"""Tests for the simplified PC-E500 emulator."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pce500.emulator import PCE500Emulator
from sc62015.pysc62015.emulator import RegisterName


class TestSimplifiedEmulator:
    """Test the simplified emulator implementation."""
    
    def test_basic_initialization(self):
        """Test emulator can be created and initialized."""
        emu = PCE500Emulator()
        assert emu is not None
        assert emu.memory is not None
        assert emu.lcd is not None
        assert emu.cpu is not None
        
    def test_rom_loading(self):
        """Test ROM loading functionality."""
        emu = PCE500Emulator()
        
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
        emu = PCE500Emulator()
        
        # Load 8KB memory card
        card_data = bytes(range(256)) * 32  # 8KB
        emu.load_memory_card(card_data, 8192)
        
        # Verify card is accessible
        assert emu.memory.read_byte(0x40000) == 0
        assert emu.memory.read_byte(0x40001) == 1
        assert emu.memory.read_byte(0x400FF) == 255
        
    def test_ram_access(self):
        """Test internal RAM read/write."""
        emu = PCE500Emulator()
        
        # Write to RAM
        emu.memory.write_byte(0xB8000, 0x42)
        emu.memory.write_byte(0xB8001, 0x43)
        
        # Read back
        assert emu.memory.read_byte(0xB8000) == 0x42
        assert emu.memory.read_byte(0xB8001) == 0x43
        
    def test_lcd_write_read(self):
        """Test LCD controller access."""
        emu = PCE500Emulator()
        
        # Write display on command to left chip
        # Address: 0x20008 = instruction write to left chip
        emu.memory.write_byte(0x20008, 0x3F)  # Display on
        
        # Verify display state
        assert emu.lcd.display_on[0] is True
        
    def test_reset_functionality(self):
        """Test emulator reset."""
        emu = PCE500Emulator()
        
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
        emu = PCE500Emulator()
        
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
        emu = PCE500Emulator()
        
        # Add breakpoint
        emu.add_breakpoint(0x1000)
        assert 0x1000 in emu.breakpoints
        
        # Remove breakpoint
        emu.remove_breakpoint(0x1000)
        assert 0x1000 not in emu.breakpoints
        
    def test_simple_trace(self):
        """Test simple tracing functionality."""
        emu = PCE500Emulator(trace_enabled=True)
        
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
        emu = PCE500Emulator()
        
        # Load ROM
        emu.load_rom(bytes(256 * 1024))
        
        # Get memory info
        info = emu.get_memory_info()
        assert "ROM: 0xC0000-0xFFFFF (256KB)" in info
        assert "RAM: 0xB8000-0xBFFFF (32KB)" in info
        assert "LCD: 0x20000-0x2FFFF" in info
        
    def test_performance_stats(self):
        """Test performance statistics."""
        emu = PCE500Emulator()
        
        # Get initial stats
        stats = emu.get_performance_stats()
        assert stats['instructions_executed'] == 0
        assert stats['elapsed_time'] >= 0
        assert stats['instructions_per_second'] == 0
        assert stats['speed_ratio'] == 0
        
    def test_mv_bp_register_instruction_execution(self):
        """Test MV (EC), C2 instruction execution - opcode 30ccecc2."""
        from sc62015.pysc62015.instr.opcodes import IMEMRegisters, INTERNAL_MEMORY_START
        
        emu = PCE500Emulator(perfetto_trace=True)
        
        # Setup ROM with the MV (EC), C2 instruction (opcode: 30ccecc2)
        rom_data = bytearray(256 * 1024)
        opcode_bytes = bytes.fromhex('30ccecc2')
        for i, byte in enumerate(opcode_bytes):
            rom_data[i] = byte
        emu.load_rom(bytes(rom_data))
        
        # Set PC to start of ROM
        initial_pc = 0xC0000
        emu.cpu.regs.set(RegisterName.PC, initial_pc)
        
        # Verify instruction is correctly loaded in ROM
        for i, expected_byte in enumerate(opcode_bytes):
            actual_byte = emu.memory.read_byte(initial_pc + i)
            assert actual_byte == expected_byte, f"ROM byte {i}: expected 0x{expected_byte:02X}, got 0x{actual_byte:02X}"
        
        # Decode instruction to verify it's correct
        instr = emu.cpu.decode_instruction(initial_pc)
        assert instr is not None, "Failed to decode instruction"
        
        # Test BP value reading using both memory access methods
        bp_addr = INTERNAL_MEMORY_START + IMEMRegisters.BP
        
        # Check initial BP values via both methods
        initial_bp_via_memory = emu.memory.read_byte(bp_addr)
        initial_bp_via_cpu = emu.cpu.memory.read_byte(bp_addr)
        assert initial_bp_via_memory == initial_bp_via_cpu, f"Both methods should give same initial value: memory={initial_bp_via_memory:02X}, cpu={initial_bp_via_cpu:02X}"
        
        # Get initial state
        initial_reads = emu.memory_read_count
        initial_writes = emu.memory_write_count
        
        # Execute the instruction
        result = emu.step()
        
        # Verify execution completed successfully
        assert result is True, "Instruction execution should continue (not hit breakpoint)"
        
        # Verify PC advanced by instruction length (4 bytes)
        final_pc = emu.cpu.regs.get(RegisterName.PC)
        assert final_pc == initial_pc + 4, f"PC should advance by 4 bytes: expected 0x{initial_pc + 4:06X}, got 0x{final_pc:06X}"
        
        # Check final BP values via both methods
        final_bp_via_memory = emu.memory.read_byte(bp_addr)
        final_bp_via_cpu = emu.cpu.memory.read_byte(bp_addr)
        assert final_bp_via_memory == final_bp_via_cpu, f"Both methods should give same final value: memory={final_bp_via_memory:02X}, cpu={final_bp_via_cpu:02X}"
        
        # Verify that BP value correctly changed to 0xC2 (the immediate value from the MV instruction)
        assert final_bp_via_memory == 0xC2, f"BP should be 0xC2 after MV (EC), C2 instruction, got 0x{final_bp_via_memory:02X}"
        assert final_bp_via_cpu == 0xC2, f"BP should be 0xC2 via cpu.memory access, got 0x{final_bp_via_cpu:02X}"
        
        # Verify memory operations occurred
        assert emu.memory_read_count > initial_reads, "Should have read memory during instruction fetch"
        assert emu.memory_write_count > initial_writes, "Should have attempted memory write for MV instruction"
        
        # Note: The BP register value correctly changes because PCE500Memory now properly
        # maps SC62015 internal memory to existing storage (reusing internal_ram array)
        # This allows the MV instruction to write to internal memory registers successfully
        
        # Verify instruction count incremented
        assert emu.instruction_count == 1, "Instruction count should increment"
        
    def test_mv_bp_register_with_sc62015_direct(self):
        """Test MV (EC), C2 with SC62015 emulator directly to verify BP value changes."""
        from sc62015.pysc62015.test_emulator import _make_cpu_and_mem
        from sc62015.pysc62015.constants import ADDRESS_SPACE_SIZE, INTERNAL_MEMORY_START
        from sc62015.pysc62015.instr.opcodes import IMEMRegisters
        
        # Create SC62015 emulator with the MV instruction
        opcode_bytes = bytes.fromhex('30ccecc2')
        init_mem = {INTERNAL_MEMORY_START + IMEMRegisters.BP: 0x00}  # Initial BP = 0x00
        
        cpu, raw, reads, writes = _make_cpu_and_mem(
            size=ADDRESS_SPACE_SIZE,
            init_data=init_mem,
            instr_bytes=opcode_bytes,
            instr_addr=0x1000
        )
        
        # Verify initial BP value
        bp_addr = INTERNAL_MEMORY_START + IMEMRegisters.BP
        initial_bp = raw[bp_addr]
        assert initial_bp == 0x00, f"Initial BP should be 0x00, got 0x{initial_bp:02X}"
        
        # Set PC and execute instruction
        cpu.regs.set(RegisterName.PC, 0x1000)
        cpu.execute_instruction(0x1000)
        
        # Verify BP value changed to 0xC2
        final_bp = raw[bp_addr]
        assert final_bp == 0xC2, f"BP should be 0xC2 after instruction, got 0x{final_bp:02X}"
        
        # Verify PC advanced correctly
        final_pc = cpu.regs.get(RegisterName.PC)
        assert final_pc == 0x1004, f"PC should advance to 0x1004, got 0x{final_pc:04X}"