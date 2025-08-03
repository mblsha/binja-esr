"""Unit tests for HD61202 LCD controller."""

import pytest
import numpy as np
from PIL import Image

from . import (
    HD61202Controller, HD61202, HD61202State, Command,
    parse_command, ChipSelect, Instruction, DataInstruction, ReadWrite
)


class TestCommandParsing:
    """Test command parsing from address and value."""
    
    def test_parse_instruction_on_off(self):
        """Test ON/OFF instruction parsing."""
        # ON command to both chips (0x2000)
        cmd = parse_command(0x2000, 0b00111111)  # 0x3F
        assert cmd.cs == ChipSelect.BOTH
        assert cmd.instr == Instruction.ON_OFF
        assert cmd.data == 1
        
        # OFF command to both chips
        cmd = parse_command(0x2000, 0b00111110)  # 0x3E
        assert cmd.cs == ChipSelect.BOTH
        assert cmd.instr == Instruction.ON_OFF
        assert cmd.data == 0
        
    def test_parse_instruction_set_page(self):
        """Test SET_PAGE instruction parsing."""
        # Set page 5
        cmd = parse_command(0x2000, 0b10111101)  # 0xBD
        assert cmd.cs == ChipSelect.BOTH
        assert cmd.instr == Instruction.SET_PAGE
        assert cmd.data == 5  # Only lower 3 bits
        
    def test_parse_instruction_set_y_address(self):
        """Test SET_Y_ADDRESS instruction parsing."""
        # Set Y address to 0x25
        cmd = parse_command(0x2000, 0b01100101)  # 0x65
        assert cmd.cs == ChipSelect.BOTH
        assert cmd.instr == Instruction.SET_Y_ADDRESS
        assert cmd.data == 0x25
        
    def test_parse_instruction_start_line(self):
        """Test START_LINE instruction parsing."""
        # Set start line to 0x1A
        cmd = parse_command(0x2000, 0b11011010)  # 0xDA
        assert cmd.cs == ChipSelect.BOTH
        assert cmd.instr == Instruction.START_LINE
        assert cmd.data == 0x1A
        
    def test_parse_data_write(self):
        """Test data write parsing."""
        # Data write to both chips
        cmd = parse_command(0x2002, 0xAA)  # A1=1 for data
        assert cmd.cs == ChipSelect.BOTH
        assert cmd.instr is None
        assert cmd.data == 0xAA
        
    def test_chip_select_decoding(self):
        """Test chip select address decoding."""
        # Both chips (CS=00)
        cmd = parse_command(0x2000, 0x3F)
        assert cmd.cs == ChipSelect.BOTH
        
        # Right chip (CS=01)
        cmd = parse_command(0x2004, 0x3F)
        assert cmd.cs == ChipSelect.RIGHT
        
        # Left chip (CS=10)
        cmd = parse_command(0x2008, 0x3F)
        assert cmd.cs == ChipSelect.LEFT
        
    def test_alternate_address_range(self):
        """Test 0xAxxxx address range support."""
        cmd = parse_command(0xA000, 0x3F)  # Need full address
        assert cmd.cs == ChipSelect.BOTH
        assert cmd.instr == Instruction.ON_OFF
        
    def test_invalid_address(self):
        """Test invalid address handling."""
        with pytest.raises(ValueError):
            parse_command(0x3000, 0x3F)  # Invalid range
            
    def test_read_operation_error(self):
        """Test that read operations raise error."""
        with pytest.raises(ValueError):
            parse_command(0x2001, 0x3F)  # A0=1 is read
            
    def test_chip_select_none_error(self):
        """Test that CS=11 (NONE) raises error."""
        with pytest.raises(ValueError):
            parse_command(0x200C, 0x3F)  # CS=11


class TestHD61202:
    """Test individual HD61202 chip functionality."""
    
    def test_chip_initialization(self):
        """Test chip initialization."""
        chip = HD61202()
        assert chip.width == 120
        assert chip.height_pages == 4
        assert chip.state.on is False
        assert chip.state.page == 0
        assert chip.state.y_address == 0
        assert chip.state.start_line == 0
        
    def test_write_instruction_on_off(self):
        """Test ON/OFF instruction."""
        chip = HD61202()
        
        # Turn on
        chip.write_instruction(Instruction.ON_OFF, 1)
        assert chip.state.on is True
        
        # Turn off
        chip.write_instruction(Instruction.ON_OFF, 0)
        assert chip.state.on is False
        
    def test_write_instruction_set_page(self):
        """Test SET_PAGE instruction."""
        chip = HD61202()
        chip.write_instruction(Instruction.SET_PAGE, 3)
        assert chip.state.page == 3
        
        # Test wraparound
        chip.write_instruction(Instruction.SET_PAGE, 0xFF)
        assert chip.state.page == 7  # Masked to 3 bits
        
    def test_write_instruction_set_y_address(self):
        """Test SET_Y_ADDRESS instruction."""
        chip = HD61202()
        chip.write_instruction(Instruction.SET_Y_ADDRESS, 45)
        assert chip.state.y_address == 45
        
        # Test wraparound for 64-width chip
        chip.write_instruction(Instruction.SET_Y_ADDRESS, 100)
        assert chip.state.y_address == 36  # 100 & 63
        
    def test_write_instruction_start_line(self):
        """Test START_LINE instruction."""
        chip = HD61202()
        chip.write_instruction(Instruction.START_LINE, 25)
        assert chip.state.start_line == 25
        
        # Test masking
        chip.write_instruction(Instruction.START_LINE, 0xFF)
        assert chip.state.start_line == 0x3F  # Masked to 6 bits
        
    def test_write_data(self):
        """Test data writing and auto-increment."""
        chip = HD61202()
        chip.state.page = 2
        chip.state.y_address = 10
        
        # Write data
        chip.write_data(0xAA)
        assert chip.vram[2][10] == 0xAA
        assert chip.state.y_address == 11  # Auto-incremented
        
        # Test wraparound
        chip.state.y_address = 63
        chip.write_data(0x55)
        assert chip.vram[2][63] == 0x55
        assert chip.state.y_address == 0  # Wrapped around
        
    def test_read_data(self):
        """Test data reading and auto-increment."""
        chip = HD61202()
        chip.state.page = 1
        chip.state.y_address = 5
        chip.vram[1][5] = 0x42
        
        # Read data
        data = chip.read_data()
        assert data == 0x42
        assert chip.state.y_address == 6  # Auto-incremented
        
    def test_read_status(self):
        """Test status register reading."""
        chip = HD61202()
        
        # Initial state
        status = chip.read_status()
        assert status & 0x10  # Reset flag set
        
        # Turn on display
        chip.state.on = True
        status = chip.read_status()
        assert status & 0x20  # Display on flag
        
        # Set busy
        chip.busy = True
        status = chip.read_status()
        assert status & 0x80  # Busy flag
        
    def test_reset(self):
        """Test chip reset."""
        chip = HD61202()
        chip.state.on = True
        chip.state.page = 3
        chip.vram[0][0] = 0xFF
        
        chip.reset()
        
        assert chip.state.on is False
        assert chip.state.page == 0
        assert chip.vram[0][0] == 0
        assert chip.reset_flag is True
        
    def test_get_display_buffer(self):
        """Test display buffer generation."""
        chip = HD61202()
        
        # Write some test patterns
        chip.vram[0][0] = 0b00000001  # Pixel at (0,0)
        chip.vram[0][1] = 0b00000010  # Pixel at (1,1)
        chip.vram[1][0] = 0b10000000  # Pixel at (0,15)
        
        buffer = chip.get_display_buffer()
        
        assert buffer.shape == (64, 64)
        assert buffer[0, 0] == 1  # First pixel
        assert buffer[1, 1] == 1  # Second pixel
        assert buffer[15, 0] == 1  # Third pixel
        
    def test_get_display_buffer_with_scrolling(self):
        """Test display buffer with start line scrolling."""
        chip = HD61202()
        chip.state.start_line = 8  # Scroll by one page
        
        # Write to page 0
        chip.vram[0][0] = 0xFF  # All pixels on
        
        buffer = chip.get_display_buffer()
        
        # Due to scrolling, page 0 should appear at lines 56-63
        for y in range(56, 64):
            assert buffer[y, 0] == 1
            
    def test_get_vram_image(self):
        """Test VRAM image generation."""
        chip = HD61202()
        chip.vram[0][0] = 0xFF  # All pixels in first column of first page
        
        image = chip.get_vram_image(zoom=2)
        
        assert image.size == (128, 128)  # 64x64 * 2
        
        # Check that pixels are green
        pixels = np.array(image)
        # First 8 rows, first column should be green (0, 255, 0)
        for y in range(16):
            for x in range(2):
                assert tuple(pixels[y, x]) == (0, 255, 0)  # Green pixels


class TestHD61202Controller:
    """Test HD61202 controller with dual chips."""
    
    def test_controller_initialization(self):
        """Test controller initialization."""
        controller = HD61202Controller()
        assert len(controller.chips) == 2
        assert controller.chips[0].width == 120
        assert controller.chips[1].width == 120
        
    def test_write_command_both_chips(self):
        """Test writing command to both chips."""
        controller = HD61202Controller()
        
        # Turn on both displays
        controller.write(0x2000, 0x3F)
        
        assert controller.chips[0].state.on is True
        assert controller.chips[1].state.on is True
        
    def test_write_command_single_chip(self):
        """Test writing command to single chip."""
        controller = HD61202Controller()
        
        # Turn on left chip only
        controller.write(0x2008, 0x3F)
        assert controller.chips[0].state.on is True
        assert controller.chips[1].state.on is False
        
        # Turn on right chip only  
        controller.write(0x2004, 0x3F)
        assert controller.chips[1].state.on is True
        
    def test_write_data_both_chips(self):
        """Test writing data to both chips."""
        controller = HD61202Controller()
        
        # Write data to both chips
        controller.write(0x2002, 0xAA)
        
        assert controller.chips[0].vram[0][0] == 0xAA
        assert controller.chips[1].vram[0][0] == 0xAA
        
    def test_read_status(self):
        """Test reading status from chips."""
        controller = HD61202Controller()
        controller.chips[0].state.on = True
        controller.chips[1].busy = True
        
        # Read from both chips (OR'd together)
        status = controller.read(0x2001)
        assert status & 0x20  # Display on from chip 0
        assert status & 0x80  # Busy from chip 1
        
    def test_read_data(self):
        """Test reading data from chip."""
        controller = HD61202Controller()
        controller.chips[0].vram[0][0] = 0x42
        
        # Read from left chip: A0=1 (read), A1=1 (data), A3:A2=10 (left)
        # Binary: ...1011 = 0xB
        data = controller.read(0x2000B)  # CS=10, read, data
        assert data == 0x42
        
    def test_get_display_buffer(self):
        """Test combined display buffer."""
        controller = HD61202Controller()
        controller.chips[0].state.on = True
        controller.chips[1].state.on = True
        
        # Write test patterns
        controller.chips[0].vram[0][0] = 0xFF
        controller.chips[1].vram[0][0] = 0xFF
        
        buffer = controller.get_display_buffer()
        
        assert buffer.shape == (32, 240)
        # Check pixels from both chips
        for y in range(8):
            assert buffer[y, 0] == 1    # Left chip
            assert buffer[y, 120] == 1  # Right chip
            
    def test_reset(self):
        """Test controller reset."""
        controller = HD61202Controller()
        controller.chips[0].state.on = True
        controller.chips[1].state.page = 3
        
        controller.reset()
        
        assert controller.chips[0].state.on is False
        assert controller.chips[1].state.page == 0
        
    def test_backward_compatibility_properties(self):
        """Test backward compatibility properties."""
        controller = HD61202Controller()
        controller.chips[0].state.on = True
        controller.chips[1].state.page = 2
        
        # Test properties
        assert controller.display_on == [True, False]
        assert controller.page == [0, 2]
        assert controller.column == [0, 0]
        
    def test_save_chip_to_png(self, tmp_path):
        """Test saving individual chip displays."""
        controller = HD61202Controller()
        controller.chips[0].vram[0][0] = 0xFF
        
        # Save left chip
        png_path = tmp_path / "test_left.png"
        controller.save_chip_to_png(0, str(png_path))
        
        assert png_path.exists()
        
        # Load and verify
        img = Image.open(png_path)
        assert img.mode == 'L'  # Grayscale
        assert img.size == (120, 32)  # PC-E500 chip size
        
    def test_get_combined_display(self):
        """Test combined display image generation."""
        controller = HD61202Controller()
        
        # Set some test patterns
        # Left chip: pixels at column 0 and 64
        controller.chips[0].vram[0][0] = 0xFF
        controller.chips[0].vram[0][64] = 0xFF
        
        # Right chip: pixels at column 0 and 56
        controller.chips[1].vram[0][0] = 0xFF
        controller.chips[1].vram[0][56] = 0xFF
        
        image = controller.get_combined_display(zoom=1)
        
        assert image.size == (240, 32)
        
        # Verify the layout
        pixels = np.array(image)
        # Check green channel for lit pixels
        green = pixels[:, :, 1]
        
        # LCD0[0-63] should have pixel at column 0
        assert np.any(green[:8, 0] > 0)
        
        # LCD1[0-55] should have pixel at column 64
        assert np.any(green[:8, 64] > 0)


class TestIntegration:
    """Integration tests with real command sequences."""
    
    def test_display_initialization_sequence(self):
        """Test typical display initialization sequence."""
        controller = HD61202Controller()
        
        # Typical init sequence
        controller.write(0x2000, 0x3F)  # Display ON
        controller.write(0x2000, 0xC0)  # Start line 0
        controller.write(0x2000, 0xB0)  # Page 0
        controller.write(0x2000, 0x40)  # Y address 0
        
        # Verify state
        assert all(controller.display_on)
        assert all(chip.state.start_line == 0 for chip in controller.chips)
        assert all(p == 0 for p in controller.page)
        assert all(c == 0 for c in controller.column)
        
    def test_write_character_pattern(self):
        """Test writing a character pattern."""
        controller = HD61202Controller()
        
        # Initialize display
        controller.write(0x2000, 0x3F)  # ON
        controller.write(0x2000, 0xB0)  # Page 0
        controller.write(0x2000, 0x40)  # Column 0
        
        # Write 5x7 character 'A' pattern
        char_data = [0x7E, 0x11, 0x11, 0x11, 0x7E]  # Simple 'A'
        for data in char_data:
            controller.write(0x2002, data)
            
        # Verify data was written
        for i, data in enumerate(char_data):
            assert controller.chips[0].vram[0][i] == data
            assert controller.chips[1].vram[0][i] == data
            
    def test_selective_chip_updates(self):
        """Test updating chips selectively."""
        controller = HD61202Controller()
        
        # Turn on left chip only
        controller.write(0x2008, 0x3F)
        
        # Write data to left chip only
        controller.write(0x2000A, 0xAA)
        
        # Verify only left chip was updated
        assert controller.chips[0].state.on is True
        assert controller.chips[0].vram[0][0] == 0xAA
        assert controller.chips[1].state.on is False
        assert controller.chips[1].vram[0][0] == 0
        
    def test_address_wraparound(self):
        """Test column address wraparound."""
        controller = HD61202Controller()
        
        # Set column to near end
        controller.chips[0].state.y_address = 119
        
        # Write two bytes
        controller.write(0x2000A, 0x11)  # Left chip only
        controller.write(0x2000A, 0x22)  # Should wrap to column 0
        
        assert controller.chips[0].vram[0][119] == 0x11
        assert controller.chips[0].vram[0][0] == 0x22
        assert controller.chips[0].state.y_address == 1