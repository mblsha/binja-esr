"""Simplified HD61202 LCD controller for PC-E500 emulator."""

import enum
import dataclasses
import numpy as np
from typing import Optional, List
from PIL import Image, ImageDraw

from ..trace_manager import g_tracer


# Enums for command parsing
class ReadWrite(enum.Enum):
    READ = 1
    WRITE = 0


class DataInstruction(enum.Enum):
    INSTRUCTION = 0
    DATA = 1


class ChipSelect(enum.Enum):
    BOTH = 0b00
    RIGHT = 0b01
    LEFT = 0b10
    NONE = 0b11


class Instruction(enum.Enum):
    """HD61202 instruction types (DB7-DB6)"""
    ON_OFF = 0b00         # 0011111X
    SET_Y_ADDRESS = 0b01  # 01XXXXXX
    SET_PAGE = 0b10       # 10111XXX
    START_LINE = 0b11     # 11XXXXXX


@dataclasses.dataclass
class HD61202State:
    """State of a single HD61202 chip"""
    on: bool = False
    start_line: int = 0  # 0-63
    page: int = 0        # 0-7
    y_address: int = 0   # 0-63/119 depending on chip width


@dataclasses.dataclass
class Command:
    """Parsed LCD command"""
    cs: ChipSelect
    instr: Optional[Instruction] = None
    data: int = 0

    def __repr__(self):
        return f"Command(cs={self.cs}, instr={self.instr}, data=0x{self.data:02X})"


def parse_command(addr: int, value: int) -> Command:
    """Parse address and value into structured command.
    
    Address bit encoding:
    - A0: R/W (0=write, 1=read)
    - A1: D/I (0=instruction, 1=data)
    - A3:A2: Chip select
    """
    # Support both 0x2xxxx and 0xAxxxx ranges
    addr_hi = addr & 0xF0000
    if addr_hi not in [0x20000, 0xA0000]:
        raise ValueError(f"Invalid LCD address: 0x{addr:06X}")
    
    addr = addr & 0xFFF
    rw = ReadWrite(addr & 1)
    if rw != ReadWrite.WRITE:
        raise ValueError("Read operations not supported by parse_command")
    
    di = DataInstruction((addr >> 1) & 1)
    cs = ChipSelect((addr >> 2) & 0b11)
    
    if cs == ChipSelect.NONE:
        raise ValueError("ChipSelect.NONE not supported")
    
    data = value
    instr = None
    
    if di == DataInstruction.INSTRUCTION:
        # Decode instruction from DB7-DB6
        instr = Instruction(data >> 6)
        data = data & 0b111111
        
        # Apply instruction-specific data masks
        if instr == Instruction.ON_OFF:
            data = data & 1
        elif instr == Instruction.SET_PAGE:
            data = data & 0b111
            
    return Command(cs, instr, data)


class HD61202Chip:
    """Single HD61202 LCD controller chip."""
    
    def __init__(self, width: int = 64, height_pages: int = 8):
        """Initialize HD61202 chip.
        
        Args:
            width: Number of columns (64 for standard, 120 for PC-E500)
            height_pages: Number of pages (8 for 64 pixels high)
        """
        self.width = width
        self.height_pages = height_pages
        self.state = HD61202State()
        # VRAM organized as pages (rows) x columns
        self.vram = [[0 for _ in range(width)] for _ in range(height_pages)]
        self.busy = False
        self.reset_flag = True
        
    def reset(self) -> None:
        """Reset chip to initial state."""
        self.state = HD61202State()
        self.vram = [[0 for _ in range(self.width)] for _ in range(self.height_pages)]
        self.busy = False
        self.reset_flag = True
        
    def write_instruction(self, instr: Instruction, data: int) -> None:
        """Execute an instruction."""
        if instr == Instruction.ON_OFF:
            self.state.on = bool(data)
        elif instr == Instruction.START_LINE:
            self.state.start_line = data & 0x3F
        elif instr == Instruction.SET_PAGE:
            self.state.page = data & 0x07
        elif instr == Instruction.SET_Y_ADDRESS:
            self.state.y_address = data & (self.width - 1)
        else:
            raise ValueError(f"Unknown instruction: {instr}")
            
    def write_data(self, data: int) -> None:
        """Write data byte to current page/column position."""
        if self.state.page < self.height_pages and self.state.y_address < self.width:
            self.vram[self.state.page][self.state.y_address] = data & 0xFF
            # Auto-increment column
            self.state.y_address = (self.state.y_address + 1) % self.width
            
    def read_data(self) -> int:
        """Read data byte from current page/column position."""
        if self.state.page < self.height_pages and self.state.y_address < self.width:
            data = self.vram[self.state.page][self.state.y_address]
            # Auto-increment column
            self.state.y_address = (self.state.y_address + 1) % self.width
            return data
        return 0xFF
        
    def read_status(self) -> int:
        """Read status register."""
        status = 0
        if self.busy:
            status |= 0x80
        if self.state.on:
            status |= 0x20
        if self.reset_flag:
            status |= 0x10
        return status
        
    def get_display_buffer(self) -> np.ndarray:
        """Get display buffer accounting for scrolling.
        
        Returns:
            2D numpy array of pixel values (0 or 1)
        """
        height = self.height_pages * 8
        buffer = np.zeros((height, self.width), dtype=np.uint8)
        
        for y in range(height):
            # Account for scrolling
            ram_line = (self.state.start_line + y) % 64
            page = ram_line // 8
            bit = ram_line % 8
            
            if page < self.height_pages:
                for x in range(self.width):
                    byte_val = self.vram[page][x]
                    if (byte_val >> bit) & 1:
                        buffer[y, x] = 1
                        
        return buffer
        
    def get_vram_image(self, zoom: int = 1) -> Image.Image:
        """Get VRAM as PIL Image.
        
        Args:
            zoom: Scaling factor
            
        Returns:
            PIL Image in RGB format
        """
        off_color = (0, 0, 0)
        on_color = (0, 255, 0)
        
        img_width = self.width * zoom
        img_height = self.height_pages * 8 * zoom
        image = Image.new("RGB", (img_width, img_height), off_color)
        draw = ImageDraw.Draw(image)
        
        for page in range(self.height_pages):
            for col in range(self.width):
                byte_val = self.vram[page][col]
                for bit in range(8):
                    if (byte_val >> bit) & 1:
                        dx = col
                        dy = page * 8 + bit
                        draw.rectangle(
                            [
                                dx * zoom,
                                dy * zoom,
                                dx * zoom + zoom - 1,
                                dy * zoom + zoom - 1,
                            ],
                            fill=on_color,
                        )
                        
        return image


class HD61202Controller:
    """Simplified PC-E500 display controller using two HD61202 chips.
    
    Removes complex command pattern and namespace classes.
    """
    
    def __init__(self):
        """Initialize dual HD61202 chips for 240x32 display."""
        # PC-E500 uses two 120-column chips (non-standard width)
        self.chips = [
            HD61202Chip(width=120, height_pages=4),  # Left chip
            HD61202Chip(width=120, height_pages=4)   # Right chip
        ]
        
        # Perfetto tracing
        self.perfetto_enabled = False
        
    def read(self, address: int, cpu_pc: Optional[int] = None) -> int:
        """Read from LCD controller.
        
        Args:
            address: Memory address being read
            cpu_pc: Optional CPU program counter for tracing context
        """
        # Check if in LCD address range (0x2xxxx or 0xAxxxx)
        addr_hi = address & 0xF0000
        if addr_hi not in [0x20000, 0xA0000]:
            return 0xFF
            
        # Decode address bits
        addr = address & 0xFFF
        is_read = addr & 1           # A0: Read flag
        is_data = (addr >> 1) & 1    # A1: Data/Instruction
        chip_sel = (addr >> 2) & 3   # A3:A2: Chip select
        
        if not is_read:
            return 0xFF  # Write operation
            
        # Map chip select to chip indices
        chip_indices = self._get_chip_indices(ChipSelect(chip_sel))
        if not chip_indices:
            return 0xFF
            
        if is_data:
            # Data read - from first selected chip
            chip_idx = chip_indices[0]
            return self.chips[chip_idx].read_data()
        else:
            # Status read - OR together from selected chips
            status = 0
            for chip_idx in chip_indices:
                status |= self.chips[chip_idx].read_status()
            return status
            
    def write(self, address: int, value: int, cpu_pc: Optional[int] = None) -> None:
        """Write to LCD controller.
        
        Args:
            address: Memory address being written
            value: Byte value to write
            cpu_pc: Optional CPU program counter for tracing context
        """
        try:
            # Parse command from address and value
            cmd = parse_command(address, value)
            
            # Get chip indices based on chip select
            chip_indices = self._get_chip_indices(cmd.cs)
            
            # Execute command on selected chips
            for chip_idx in chip_indices:
                chip = self.chips[chip_idx]
                
                if cmd.instr is not None:
                    # Instruction write
                    chip.write_instruction(cmd.instr, cmd.data)
                    
                    # Perfetto tracing
                    if self.perfetto_enabled:
                        chip_name = "Left" if chip_idx == 0 else "Right"
                        trace_data = {
                            "cmd": f"0x{value:02X}",
                            "chip": chip_name,
                            "instr": cmd.instr.name,
                            "data": f"0x{cmd.data:02X}"
                        }
                        if cpu_pc is not None:
                            trace_data["pc"] = f"0x{cpu_pc:06X}"
                        g_tracer.trace_instant("LCD", f"LCD_Cmd_{chip_name}", trace_data)
                else:
                    # Data write
                    chip.write_data(cmd.data)
                    
        except ValueError:
            # Invalid address or command - ignore
            pass
            
    def _get_chip_indices(self, cs: ChipSelect) -> List[int]:
        """Map chip select to chip indices.
        
        Note: PC-E500 chip mapping:
        - CS=00 (BOTH): Both chips
        - CS=01 (RIGHT): Chip 1 (right)
        - CS=10 (LEFT): Chip 0 (left)
        - CS=11 (NONE): No chips
        """
        if cs == ChipSelect.BOTH:
            return [0, 1]
        elif cs == ChipSelect.LEFT:
            return [0]
        elif cs == ChipSelect.RIGHT:
            return [1]
        else:  # ChipSelect.NONE
            return []
            
    def get_display_buffer(self) -> np.ndarray:
        """Get combined display buffer as numpy array."""
        buffer = np.zeros((32, 240), dtype=np.uint8)
        
        # Process left chip (columns 0-119)
        if self.chips[0].state.on:
            chip_buffer = self.chips[0].get_display_buffer()
            buffer[:, 0:120] = chip_buffer
                            
        # Process right chip (columns 120-239)
        if self.chips[1].state.on:
            chip_buffer = self.chips[1].get_display_buffer()
            buffer[:, 120:240] = chip_buffer
                            
        return buffer
                        
    def reset(self) -> None:
        """Reset controller state."""
        for chip in self.chips:
            chip.reset()
        
    @property
    def width(self) -> int:
        return 240
        
    @property
    def height(self) -> int:
        return 32
        
    # Backward compatibility properties
    @property
    def display_on(self) -> List[bool]:
        """Get display on status for both chips."""
        return [chip.state.on for chip in self.chips]
        
    @property
    def page(self) -> List[int]:
        """Get current page for both chips."""
        return [chip.state.page for chip in self.chips]
        
    @property
    def column(self) -> List[int]:
        """Get current column (y_address) for both chips."""
        return [chip.state.y_address for chip in self.chips]
        
    def set_perfetto_enabled(self, enabled: bool) -> None:
        """Enable or disable Perfetto tracing."""
        self.perfetto_enabled = enabled
        
    def save_chip_to_png(self, chip_index: int, filename: str) -> None:
        """Save a single chip's display memory to a PNG file.
        
        Args:
            chip_index: 0 for left chip, 1 for right chip
            filename: Path to save the PNG file
        """
        if chip_index < 0 or chip_index >= len(self.chips):
            raise ValueError(f"Invalid chip index: {chip_index}")
            
        # Get the chip's VRAM image
        image = self.chips[chip_index].get_vram_image(zoom=1)
        
        # Convert to grayscale
        # Extract just the green channel since we use (0,255,0) for on pixels
        image_array = np.array(image)
        grayscale = image_array[:, :, 1]  # Green channel
        
        # Save as grayscale PNG
        Image.fromarray(grayscale, mode='L').save(filename)
        
    def save_displays_to_png(self, left_filename: str = "lcd_left.png", 
                           right_filename: str = "lcd_right.png") -> None:
        """Save both LCD chip displays to separate PNG files.
        
        Args:
            left_filename: Filename for left chip display
            right_filename: Filename for right chip display
        """
        self.save_chip_to_png(0, left_filename)
        self.save_chip_to_png(1, right_filename)
        
    def get_chip_buffer(self, chip_index: int) -> np.ndarray:
        """Get display buffer for a single chip as numpy array.
        
        Args:
            chip_index: 0 for left chip, 1 for right chip
            
        Returns:
            32x120 numpy array with pixel values (0 or 1)
        """
        if chip_index < 0 or chip_index >= len(self.chips):
            raise ValueError(f"Invalid chip index: {chip_index}")
            
        return self.chips[chip_index].get_display_buffer()
        
    def get_combined_display(self, zoom: int = 2) -> Image.Image:
        """Get combined display as stitched image matching PC-E500 layout.
        
        The PC-E500 display uses two HD61202 chips in a special arrangement:
        LCD0[0-63] | LCD1[0-55] | LCD1[56-63 flipped] | LCD0[64-119 flipped]
        
        Args:
            zoom: Scaling factor (default 2)
            
        Returns:
            PIL Image showing the combined display
        """
        # Get individual chip images
        images = [chip.get_vram_image(zoom=1) for chip in self.chips]
        
        # PC-E500 specific layout dimensions
        lcd0w = 64
        lcd1w = 56
        combined_width = lcd0w * 2 + lcd1w * 2  # 240 pixels
        combined_height = 32  # 4 pages * 8 pixels
        
        # Create combined image
        image = Image.new("RGB", (combined_width, combined_height), (0, 0, 0))
        
        # Layout according to PC-E500 display arrangement
        # Left portion: LCD0[0-63]
        image.paste(images[0].crop((0, 0, lcd0w, combined_height)), (0, 0))
        
        # Center-left portion: LCD1[0-55]
        image.paste(images[1].crop((0, 0, lcd1w, combined_height)), (lcd0w, 0))
        
        # Center-right portion: LCD1[56-63] flipped horizontally
        lcd1_right = images[1].crop((lcd1w, 0, lcd0w, combined_height))
        image.paste(lcd1_right.transpose(Image.FLIP_LEFT_RIGHT), (lcd0w + lcd1w, 0))
        
        # Right portion: LCD0[64-119] flipped horizontally
        lcd0_right = images[0].crop((lcd0w, 0, 120, combined_height))
        image.paste(lcd0_right.transpose(Image.FLIP_LEFT_RIGHT), (lcd0w + lcd1w + (lcd0w - lcd1w), 0))
        
        # Apply zoom
        if zoom > 1:
            image = image.resize((image.width * zoom, image.height * zoom), Image.NEAREST)
            
        return image


# Backward compatibility aliases
SimpleHD61202 = HD61202Controller
HD61202 = HD61202Controller  # For code expecting the toolkit HD61202 class