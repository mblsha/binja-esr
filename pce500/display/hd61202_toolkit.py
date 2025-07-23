"""HD61202 LCD controller toolkit for PC-E500 emulator."""

from dataclasses import dataclass
from typing import Union, TypeAlias, Dict, Optional
from enum import IntEnum
from PIL import Image, ImageDraw
import numpy as np


class HD61202:
    """
    A toolkit for parsing and interpreting commands for the Hitachi HD61202
    LCD Column Driver, used in the PC-E500 main display.

    This class serves as a namespace for three main components:
    1. Command objects (e.g., HD61202U.DisplayOnOff)
    2. A static Parser to convert raw bus signals into Command objects.
    3. An Interpreter to simulate the controller's state based on Commands.
    """

    # --- 1. Command Definitions ---
    # Using frozen dataclasses for immutable, self-describing command objects.
    @dataclass(frozen=True)
    class DisplayOnOff:
        """Controls the display's on/off state."""
        on: bool

    @dataclass(frozen=True)
    class SetStartLine:
        """Sets the display RAM line (0-63) for the top of the screen (scrolling)."""
        line: int

    @dataclass(frozen=True)
    class SetPage:
        """Sets the page address (X address, 0-7)."""
        page: int

    @dataclass(frozen=True)
    class SetYAddress:
        """Sets the column address (Y address, 0-63)."""
        y_addr: int

    @dataclass(frozen=True)
    class WriteData:
        """Writes a byte to the display RAM at the current address."""
        value: int

    @dataclass(frozen=True)
    class StatusReadRequest:
        """Represents a request to read the status register."""
        pass

    @dataclass(frozen=True)
    class DataReadRequest:
        """Represents a request to read data from display RAM."""
        pass

    @dataclass(frozen=True)
    class UnknownCommand:
        """Represents an unhandled or unknown command."""
        di: bool
        rw: bool
        value: int

    # A beautiful, clean type alias for any possible command.
    Command: TypeAlias = Union[
        DisplayOnOff, SetStartLine, SetPage, SetYAddress, WriteData,
        StatusReadRequest, DataReadRequest, UnknownCommand
    ]

    # --- 2. The Parser ---
    class Parser:
        """A static class to parse raw bus signals into HD61202.Command objects."""
        @staticmethod
        def parse(di: bool, rw: bool, data: int) -> "HD61202.Command":
            """
            Parses a single bus event using elegant pattern matching.

            Args:
                di (bool): Data/Instruction pin state (True=Data).
                rw (bool): Read/Write pin state (True=Read).
                data (int): The 8-bit value on the data bus.
            """
            match (di, rw):
                case (False, False):  # Instruction Write
                    match data:
                        case d if (d & 0b11111110) == 0b00111110:
                            return HD61202.DisplayOnOff(on=bool(d & 1))
                        case d if (d & 0b11000000) == 0b11000000:
                            return HD61202.SetStartLine(line=(d & 0x3F))
                        case d if (d & 0b11110000) == 0b10110000:
                            return HD61202.SetPage(page=(d & 0x0F))
                        case d if (d & 0b11000000) == 0b01000000:
                            return HD61202.SetYAddress(y_addr=(d & 0x3F))
                case (True, False):  # Data Write
                    return HD61202.WriteData(value=data)
                case (False, True):  # Status Read
                    return HD61202.StatusReadRequest()
                case (True, True):  # Data Read
                    return HD61202.DataReadRequest()

            return HD61202.UnknownCommand(di=di, rw=rw, value=data)


    # --- 3. The Interpreter ---
    class Interpreter:
        """Simulates the internal state and VRAM of an HD61202 controller."""

        def __init__(self, width: int = 64, height: int = 64):
            self.width = width
            self.height = height
            
            # Calculate page dimensions based on display size
            self.page_width = width  # Columns per page
            self.num_pages = (height + 7) // 8  # Number of 8-pixel high pages
            self.vram_size = self.num_pages * self.page_width

            # --- Internal State Simulation ---
            self.vram = bytearray(self.vram_size)
            self.page: int = 0
            self.y_addr: int = 0
            self.start_line: int = 0
            self.display_on: bool = False
            
            # Status register bits
            self.busy: bool = False
            self.on_off: bool = False
            self.reset: bool = False

        def eval(self, cmd: "HD61202.Command") -> None:
            """Evaluates a command and updates the interpreter's internal state."""
            match cmd:
                case HD61202.DisplayOnOff(on=on):
                    self.display_on = on
                    self.on_off = on
                case HD61202.SetStartLine(line=line):
                    self.start_line = line
                case HD61202.SetPage(page=page):
                    self.page = page
                case HD61202.SetYAddress(y_addr=y_addr):
                    self.y_addr = y_addr
                case HD61202.WriteData(value=value):
                    address = self.page * self.page_width + self.y_addr
                    if address < self.vram_size:
                        self.vram[address] = value
                    self.y_addr = (self.y_addr + 1) % self.page_width
                case HD61202.StatusReadRequest() | HD61202.DataReadRequest():
                    pass # Reads are handled separately
                case HD61202.UnknownCommand():
                    pass # Ignore unknown commands

        def read_status(self) -> int:
            """Returns the status register value."""
            status = 0
            if self.busy:
                status |= 0x80
            if self.on_off:
                status |= 0x20
            if self.reset:
                status |= 0x10
            return status

        def read_data(self) -> int:
            """Reads data from the current address and advances."""
            address = self.page * self.page_width + self.y_addr
            if address < self.vram_size:
                data = self.vram[address]
                self.y_addr = (self.y_addr + 1) % self.page_width
                return data
            return 0xFF

        def render(self, zoom: int = 4, on_color="black", off_color=(200, 220, 200)) -> Image.Image:
            """Renders the current VRAM state into a PIL Image."""
            img = Image.new("RGB", (self.width * zoom, self.height * zoom), off_color)
            if not self.display_on:
                return img

            draw = ImageDraw.Draw(img)
            for screen_y in range(self.height):
                # Account for hardware scrolling via the start line register
                ram_line = (self.start_line + screen_y) % self.height
                page, bit_in_page = divmod(ram_line, 8)

                for screen_x in range(self.width):
                    vram_addr = page * self.page_width + screen_x
                    if vram_addr < self.vram_size:
                        byte = self.vram[vram_addr]

                        if (byte >> bit_in_page) & 1:
                            x0, y0 = screen_x * zoom, screen_y * zoom
                            draw.rectangle((x0, y0, x0 + zoom - 1, y0 + zoom - 1), fill=on_color)
            return img

        def get_display_buffer(self) -> np.ndarray:
            """Returns display buffer as numpy array for compatibility."""
            buffer = np.zeros((self.height, self.width), dtype=np.uint8)
            if not self.display_on:
                return buffer
                
            for screen_y in range(self.height):
                ram_line = (self.start_line + screen_y) % self.height
                page, bit_in_page = divmod(ram_line, 8)
                
                for screen_x in range(self.width):
                    vram_addr = page * self.page_width + screen_x
                    if vram_addr < self.vram_size:
                        byte = self.vram[vram_addr]
                        
                        if (byte >> bit_in_page) & 1:
                            buffer[screen_y, screen_x] = 1
                        
            return buffer


class ChipSelect(IntEnum):
    """Elegant chip selection encoding from A3:A2 bits."""
    BOTH = 0b00   # Broadcast to both chips
    RIGHT = 0b01  # Right chip only (chip 2)
    LEFT = 0b10   # Left chip only (chip 1)
    NONE = 0b11   # No chips selected


@dataclass(frozen=True)
class AddressDecode:
    """Elegant representation of decoded LCD controller address."""
    rw: bool          # A0: Read/Write (0=write, 1=read)
    di: bool          # A1: Data/Instruction (0=instruction, 1=data)
    chip_select: ChipSelect  # A3:A2: Chip selection
    cs2: bool         # A12: CS2 (active low)
    cs3: bool         # A13: CS3 (active high)
    
    @property
    def is_valid(self) -> bool:
        """Check if this is a valid LCD access."""
        # CS3 must be high (active) and we must be in 0x2xxxx range
        return self.cs3
    
    @property
    def chips_enabled(self) -> bool:
        """Check if any chips are enabled."""
        if self.chip_select == ChipSelect.NONE:
            return False
        # CS2 (active low) affects chip 2 (right)
        if not self.cs2 and self.chip_select in [ChipSelect.RIGHT, ChipSelect.BOTH]:
            # CS2 is low (active), so chip 2 operations are valid
            return True
        # CS2 high disables chip 2
        return self.chip_select == ChipSelect.LEFT


class HD61202Controller:
    """PC-E500 main display controller using two HD61202 chips."""
    
    # Address bit positions
    BIT_RW = 0   # A0: Read/Write
    BIT_DI = 1   # A1: Data/Instruction  
    BIT_CS = 2   # A3:A2: Chip select (2 bits starting at position 2)
    BIT_CS2 = 12 # A12: CS2 (active low)
    BIT_CS3 = 13 # A13: CS3 (active high)
    
    def __init__(self, start_addr: int = 0x20000):
        """Initialize controller."""
        self.start_addr = start_addr
        
        # Two HD61202 chips for 240x32 display
        # Each chip handles 120x32 pixels (left/right split)
        self.chips: Dict[str, HD61202.Interpreter] = {
            'left': HD61202.Interpreter(width=120, height=32),
            'right': HD61202.Interpreter(width=120, height=32)
        }
        
    def decode_address(self, address: int) -> AddressDecode:
        """Decode memory address into LCD control signals."""
        # Extract individual control bits
        rw = bool(address & (1 << self.BIT_RW))
        di = bool(address & (1 << self.BIT_DI))
        chip_select = ChipSelect((address >> self.BIT_CS) & 0x03)
        cs2 = bool(address & (1 << self.BIT_CS2))
        cs3 = bool(address & (1 << self.BIT_CS3))
        
        return AddressDecode(rw=rw, di=di, chip_select=chip_select, cs2=cs2, cs3=cs3)
    
    def contains_address(self, address: int) -> bool:
        """Check if address is handled by this controller."""
        # Must be in 0x2xxxx range
        if (address & 0xF0000) != 0x20000:
            return False
            
        # Decode and check if it's a valid LCD access
        decode = self.decode_address(address)
        return decode.is_valid
    
    def _get_selected_chips(self, decode: AddressDecode) -> list[HD61202.Interpreter]:
        """Get list of selected chip interpreters based on address decode."""
        if not decode.chips_enabled:
            return []
            
        # CS2 (active low) affects chip 2 (right)
        cs2_active = not decode.cs2
        
        match decode.chip_select:
            case ChipSelect.BOTH:
                # Both selected, but CS2 might disable right chip
                if cs2_active:
                    return [self.chips['left'], self.chips['right']]
                else:
                    return [self.chips['left']]
            case ChipSelect.LEFT:
                return [self.chips['left']]
            case ChipSelect.RIGHT:
                # Right chip only enabled if CS2 is active (low)
                return [self.chips['right']] if cs2_active else []
            case ChipSelect.NONE:
                return []
        
    def read(self, address: int, cpu_pc: Optional[int] = None) -> int:
        """Read from LCD controller.
        
        Args:
            address: Memory address being read
            cpu_pc: Optional CPU program counter for tracing context
            
        Returns:
            8-bit value read from LCD
        """
        if not self.contains_address(address):
            return 0xFF
            
        decode = self.decode_address(address)
        
        # Only process read operations
        if not decode.rw:
            return 0xFF
            
        selected_chips = self._get_selected_chips(decode)
        if not selected_chips:
            return 0xFF
            
        if not decode.di:  # Status read (instruction mode)
            # OR together status from all selected chips
            result = sum(chip.read_status() for chip in selected_chips)
            
            # Add tracing
            try:
                from ..trace_manager import g_tracer
                if g_tracer.is_tracing() and cpu_pc is not None:
                    g_tracer.trace_instant(
                        "Display",
                        f"LCD_StatusRead@0x{cpu_pc:06X}",
                        {"status": f"0x{result:02X}"}
                    )
            except ImportError:
                pass
                
            return result
        else:  # Data read
            # For data reads, OR together data from all selected chips
            # (though typically only one chip is selected for data reads)
            result = sum(chip.read_data() for chip in selected_chips)
            
            # Add tracing
            try:
                from ..trace_manager import g_tracer
                if g_tracer.is_tracing() and cpu_pc is not None:
                    g_tracer.trace_instant(
                        "Display",
                        f"LCD_DataRead@0x{cpu_pc:06X}",
                        {"data": f"0x{result:02X}"}
                    )
            except ImportError:
                pass
                
            return result
        
    def write(self, address: int, value: int, cpu_pc: Optional[int] = None) -> None:
        """Write to LCD controller.
        
        Args:
            address: Memory address being written
            value: 8-bit value to write
            cpu_pc: Optional CPU program counter for tracing context
        """
        if not self.contains_address(address):
            return
            
        decode = self.decode_address(address)
        
        # Only process write operations
        if decode.rw:
            return
            
        # Parse command and broadcast to selected chips
        cmd = HD61202.Parser.parse(decode.di, False, value)
        
        # Add tracing support if trace manager is available
        try:
            from ..trace_manager import g_tracer
            if g_tracer.is_tracing() and cpu_pc is not None:
                # Get command name for tracing
                cmd_name = cmd.__class__.__name__
                
                # Add specific details for different command types
                details = {"cmd": cmd_name, "value": f"0x{value:02X}"}
                
                if isinstance(cmd, HD61202.SetPage):
                    details["page"] = cmd.page
                elif isinstance(cmd, HD61202.SetYAddress):
                    details["y_addr"] = cmd.y_addr
                elif isinstance(cmd, HD61202.SetStartLine):
                    details["line"] = cmd.line
                elif isinstance(cmd, HD61202.WriteData):
                    details["data"] = f"0x{cmd.value:02X}"
                
                # Emit trace event
                g_tracer.trace_instant(
                    "Display",
                    f"LCD_{cmd_name}@0x{cpu_pc:06X}",
                    details
                )
        except ImportError:
            # Tracing not available, continue normally
            pass
        
        for chip in self._get_selected_chips(decode):
            chip.eval(cmd)
                
    def get_display_buffer(self) -> np.ndarray:
        """Get combined display buffer from both chips."""
        left_buffer = self.chips['left'].get_display_buffer()
        right_buffer = self.chips['right'].get_display_buffer()
        
        # Combine into 240x32 display
        combined = np.zeros((32, 240), dtype=np.uint8)
        combined[:, :120] = left_buffer
        combined[:, 120:] = right_buffer
        
        return combined
        
    @property
    def display_on(self) -> bool:
        """Check if display is on."""
        return any(chip.display_on for chip in self.chips.values())
        
    @property
    def width(self) -> int:
        return 240
        
    @property
    def height(self) -> int:
        return 32