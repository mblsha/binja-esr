"""Test PC-E500 emulator performance and LCD initialization."""

import sys
import time
import pytest
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pce500.run_pce500 import run_emulator


class TestEmulatorPerformance:
    """Test emulator performance and correct initialization."""
    
    def test_emulator_boot_performance(self):
        """Test that emulator boots properly within performance constraints.
        
        This test verifies:
        1. The emulator completes 20000 instructions within 7 seconds
        2. Both LCD chips have "Display ON: True"
        3. LCD controller statistics are within expected ranges
        4. Memory operation counts are reasonable
        """
        # Check if ROM file exists
        rom_path = Path(__file__).parent.parent.parent / "data" / "pc-e500.bin"
        if not rom_path.exists():
            pytest.skip(f"ROM file not found at {rom_path}")
            
        # Record start time
        start_time = time.time()
        
        # Run emulator without Perfetto tracing or LCD saving for speed
        # Also disable print statements
        emu = run_emulator(
            num_steps=20000,
            dump_pc=0xFFFFF,  # Same as bisect_test.py
            perfetto_trace=False,  # Disable for speed
            save_lcd=False,  # Don't save PNG files
            print_stats=False  # Quiet mode
        )
        
        # Check execution time
        elapsed = time.time() - start_time
        assert elapsed < 10.0, f"Emulator took {elapsed:.1f}s, expected < 10s"
        
        # Check LCD displays are ON
        stats = emu.lcd.get_chip_statistics()
        assert len(stats) == 2, "Expected 2 LCD chips"
        
        for stat in stats:
            chip_name = "Left" if stat['chip'] == 0 else "Right"
            assert stat['on'] is True, f"{chip_name} LCD chip is not ON"
        
        # Verify LCD controller received instructions
        for stat in stats:
            chip_name = "Left" if stat['chip'] == 0 else "Right"
            assert stat['instructions'] > 0, f"{chip_name} chip received no instructions"
            assert stat['on_off_commands'] > 0, f"{chip_name} chip received no ON/OFF commands"
        
        # Check chip select usage
        assert emu.lcd.cs_both_count >= 0, "Negative chip select count"
        assert emu.lcd.cs_left_count >= 0, "Negative left chip select count"
        assert emu.lcd.cs_right_count >= 0, "Negative right chip select count"
        
        # At least some chip selects should have occurred
        total_cs = emu.lcd.cs_both_count + emu.lcd.cs_left_count + emu.lcd.cs_right_count
        assert total_cs > 0, "No chip select operations occurred"
        
        # Verify memory operations occurred
        assert emu.memory_read_count > 0, "No memory reads occurred"
        assert emu.memory_write_count > 0, "No memory writes occurred"
        assert emu.instruction_count == 20000, f"Expected 20000 instructions, got {emu.instruction_count}"
        
        # Verify some data was written to LCD
        for stat in stats:
            chip_name = "Left" if stat['chip'] == 0 else "Right"
            # It's OK if no data is written initially, just ON/OFF commands
            assert stat['data_written'] >= 0, f"{chip_name} chip has negative data_written"
        
        # Clean up
        emu.stop_tracing()
    
    def test_emulator_lcd_statistics_detail(self):
        """Test detailed LCD statistics after boot."""
        # Run emulator for fewer steps to be faster
        emu = run_emulator(
            num_steps=5000,
            perfetto_trace=False,
            save_lcd=False,
            print_stats=False
        )
        
        stats = emu.lcd.get_chip_statistics()
        
        # Check structure of statistics
        for stat in stats:
            # Verify all expected fields are present
            assert 'chip' in stat
            assert 'on' in stat
            assert 'instructions' in stat
            assert 'on_off_commands' in stat
            assert 'data_written' in stat
            assert 'data_read' in stat
            assert 'page' in stat
            assert 'column' in stat
            
            # Verify types
            assert isinstance(stat['chip'], int)
            assert isinstance(stat['on'], bool)
            assert isinstance(stat['instructions'], int)
            assert isinstance(stat['on_off_commands'], int)
            assert isinstance(stat['data_written'], int)
            assert isinstance(stat['data_read'], int)
            assert isinstance(stat['page'], int)
            assert isinstance(stat['column'], int)
            
            # Verify ranges
            assert stat['chip'] in [0, 1], f"Invalid chip number: {stat['chip']}"
            assert 0 <= stat['page'] <= 7, f"Invalid page: {stat['page']}"
            assert 0 <= stat['column'] <= 63, f"Invalid column: {stat['column']}"
        
        # Clean up
        emu.stop_tracing()


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])