"""Simplified HD61202 LCD controller for PC-E500 emulator."""

import numpy as np
from typing import Optional

from ..trace_manager import g_tracer


class HD61202Controller:
    """Simplified PC-E500 display controller using two HD61202 chips.
    
    Removes complex command pattern and namespace classes.
    """
    
    def __init__(self):
        """Initialize dual HD61202 chips for 240x32 display."""
        # Two chips: left (120x32) and right (120x32)
        self.left_vram = bytearray(4 * 120)   # 4 pages x 120 columns
        self.right_vram = bytearray(4 * 120)  # 4 pages x 120 columns
        
        # Chip states (one set per chip)
        self.page = [0, 0]          # Current page (0-3)
        self.column = [0, 0]        # Current column (0-119)
        self.start_line = [0, 0]    # Display start line for scrolling
        self.display_on = [False, False]
        
        # Status flags
        self.busy = [False, False]
        self.reset_flag = [False, False]
        
        # Perfetto tracing
        self.perfetto_enabled = False
        
    def read(self, address: int, cpu_pc: Optional[int] = None) -> int:
        """Read from LCD controller.
        
        Args:
            address: Memory address being read
            cpu_pc: Optional CPU program counter for tracing context
        """
        # Check if in LCD address range (0x2xxxx)
        if (address & 0xF0000) != 0x20000:
            return 0xFF
            
        # Decode address bits
        is_read = address & 1           # A0: Read flag
        is_data = (address >> 1) & 1    # A1: Data/Instruction
        chip_sel = (address >> 2) & 3   # A3:A2: Chip select
        
        if not is_read:
            return 0xFF  # Write operation
            
        # Determine which chip(s) to read from
        chips = []
        if chip_sel == 0:    # Both chips
            chips = [0, 1]
        elif chip_sel == 2:  # Left chip
            chips = [0]
        elif chip_sel == 1:  # Right chip
            chips = [1]
        # chip_sel == 3 means no chips
        
        if not chips:
            return 0xFF
            
        if is_data:
            # Data read - typically from one chip
            chip = chips[0]
            vram = self.left_vram if chip == 0 else self.right_vram
            addr = self.page[chip] * 120 + self.column[chip]
            data = vram[addr] if addr < len(vram) else 0xFF
            self.column[chip] = (self.column[chip] + 1) % 120
            return data
        else:
            # Status read - OR together from selected chips
            status = 0
            for chip in chips:
                if self.busy[chip]:
                    status |= 0x80
                if self.display_on[chip]:
                    status |= 0x20
                if self.reset_flag[chip]:
                    status |= 0x10
            return status
            
    def write(self, address: int, value: int, cpu_pc: Optional[int] = None) -> None:
        """Write to LCD controller.
        
        Args:
            address: Memory address being written
            value: Byte value to write
            cpu_pc: Optional CPU program counter for tracing context
        """
        # Check if in LCD address range
        if (address & 0xF0000) != 0x20000:
            return
            
        # Decode address bits
        is_read = address & 1           # A0: Read flag
        is_data = (address >> 1) & 1    # A1: Data/Instruction
        chip_sel = (address >> 2) & 3   # A3:A2: Chip select
        
        if is_read:
            return  # Read operation
            
        # Determine which chip(s) to write to
        chips = []
        if chip_sel == 0:    # Both chips
            chips = [0, 1]
        elif chip_sel == 2:  # Left chip
            chips = [0]
        elif chip_sel == 1:  # Right chip
            chips = [1]
            
        if not chips:
            return
            
        if is_data:
            # Data write
            for chip in chips:
                vram = self.left_vram if chip == 0 else self.right_vram
                addr = self.page[chip] * 120 + self.column[chip]
                if addr < len(vram):
                    vram[addr] = value & 0xFF
                self.column[chip] = (self.column[chip] + 1) % 120
        else:
            # Instruction write - decode command
            for chip in chips:
                self._execute_command(chip, value)
                
                # Perfetto tracing
                if self.perfetto_enabled:
                    chip_name = "Left" if chip == 0 else "Right"
                    trace_data = {"cmd": f"0x{value:02X}", "chip": chip_name}
                    if cpu_pc is not None:
                        trace_data["pc"] = f"0x{cpu_pc:06X}"
                    g_tracer.trace_instant("LCD", f"LCD_Cmd_{chip_name}", trace_data)
                
    def _execute_command(self, chip: int, cmd: int) -> None:
        """Execute HD61202 command on specified chip."""
        if (cmd & 0xFE) == 0x3E:
            # Display on/off (0011111X)
            self.display_on[chip] = bool(cmd & 1)
        elif (cmd & 0xC0) == 0xC0:
            # Set start line (11XXXXXX)
            self.start_line[chip] = cmd & 0x3F
        elif (cmd & 0xF8) == 0xB0:
            # Set page (10111XXX)
            self.page[chip] = cmd & 0x07
        elif (cmd & 0xC0) == 0x40:
            # Set Y address (01XXXXXX)
            self.column[chip] = cmd & 0x3F
            
    def get_display_buffer(self) -> np.ndarray:
        """Get combined display buffer as numpy array."""
        buffer = np.zeros((32, 240), dtype=np.uint8)
        
        # Process left chip (columns 0-119)
        if self.display_on[0]:
            self._render_chip(buffer, 0, 0, self.left_vram, 
                            self.start_line[0])
                            
        # Process right chip (columns 120-239)
        if self.display_on[1]:
            self._render_chip(buffer, 0, 120, self.right_vram,
                            self.start_line[1])
                            
        return buffer
        
    def _render_chip(self, buffer: np.ndarray, row_offset: int, 
                     col_offset: int, vram: bytearray, start_line: int) -> None:
        """Render one chip's VRAM to the display buffer."""
        for y in range(32):
            # Account for scrolling
            ram_line = (start_line + y) % 64
            page = ram_line // 8
            bit = ram_line % 8
            
            for x in range(120):
                addr = page * 120 + x
                if addr < len(vram):
                    byte_val = vram[addr]
                    if (byte_val >> bit) & 1:
                        buffer[row_offset + y, col_offset + x] = 1
                        
    def reset(self) -> None:
        """Reset controller state."""
        self.left_vram[:] = bytes(len(self.left_vram))
        self.right_vram[:] = bytes(len(self.right_vram))
        self.page = [0, 0]
        self.column = [0, 0]
        self.start_line = [0, 0]
        self.display_on = [False, False]
        self.busy = [False, False]
        self.reset_flag = [True, True]
        
    @property
    def width(self) -> int:
        return 240
        
    @property
    def height(self) -> int:
        return 32
        
    def set_perfetto_enabled(self, enabled: bool) -> None:
        """Enable or disable Perfetto tracing."""
        self.perfetto_enabled = enabled


# Backward compatibility aliases
SimpleHD61202 = HD61202Controller
HD61202 = HD61202Controller  # For code expecting the toolkit HD61202 class