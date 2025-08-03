# hd61202.py
# HD61202 LCD Controller implementation extracted from lcd_visualization.py

import enum
import dataclasses
from typing import Optional, List

from PIL import Image

# --- Data Structures and Enums ---

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
    ON_OFF = 0b00
    START_LINE = 0b11
    SET_PAGE = 0b10
    SET_Y_ADDRESS = 0b01

@dataclasses.dataclass
class HD61202State:
    on: bool = False
    start_line: int = 0  # 0-63
    page: int = 0        # 0-7
    y_address: int = 0   # 0-63

@dataclasses.dataclass
class Command:
    cs: ChipSelect
    instr: Optional[Instruction] = None
    data: int = 0

    def __repr__(self):
        return f"Command(cs={self.cs}, instr={self.instr}, data=0x{self.data:02X})"

# --- HD61202 Controller Implementation ---

class HD61202:
    """Emulates the state and VRAM of a single HD61202 LCD controller."""
    LCD_WIDTH_PIXELS = 64
    LCD_PAGES = 8
    PAGE_HEIGHT_PIXELS = 8
    LCD_HEIGHT_PIXELS = LCD_PAGES * PAGE_HEIGHT_PIXELS

    def __init__(self):
        self.state = HD61202State()
        self.vram = [[0] * self.LCD_WIDTH_PIXELS for _ in range(self.LCD_PAGES)]

    def write_instruction(self, instr: Instruction, data: int):
        if instr == Instruction.ON_OFF:
            self.state.on = bool(data)
        elif instr == Instruction.START_LINE:
            self.state.start_line = data
        elif instr == Instruction.SET_PAGE:
            self.state.page = data
        elif instr == Instruction.SET_Y_ADDRESS:
            self.state.y_address = data
        else:
            raise ValueError(f"Unknown instruction: {instr}")

    def write_data(self, data: int):
        if 0 <= self.state.page < self.LCD_PAGES and 0 <= self.state.y_address < self.LCD_WIDTH_PIXELS:
            self.vram[self.state.page][self.state.y_address] = data
            self.state.y_address = (self.state.y_address + 1) % self.LCD_WIDTH_PIXELS

    def render_vram_image(self, zoom: int = 1) -> Image.Image:
        """Renders the controller's VRAM to a PIL Image."""
        img = Image.new("1", (self.LCD_WIDTH_PIXELS, self.LCD_HEIGHT_PIXELS))
        pixels = img.load()
        for page in range(self.LCD_PAGES):
            for y_addr in range(self.LCD_WIDTH_PIXELS):
                byte = self.vram[page][y_addr]
                for bit in range(self.PAGE_HEIGHT_PIXELS):
                    if (byte >> bit) & 1:
                        pixels[y_addr, page * self.PAGE_HEIGHT_PIXELS + bit] = 1
        
        if zoom > 1:
            return img.resize((img.width * zoom, img.height * zoom), Image.NEAREST)
        return img

    def read_instruction_status(self) -> int:
        """Reads the status byte (instruction register)."""
        # Bit 7: BUSY flag (0 = ready, 1 = busy)
        # Bit 6: ON/OFF state
        # Bit 5: RESET state (always 0 in normal operation)
        # Bits 4-0: Don't care
        status = 0
        if self.state.on:
            status |= 0x40  # Set bit 6
        # We always report ready (BUSY = 0)
        return status

    def read_data(self) -> int:
        """Reads data from the current address and increments Y address."""
        if 0 <= self.state.page < self.LCD_PAGES and 0 <= self.state.y_address < self.LCD_WIDTH_PIXELS:
            data = self.vram[self.state.page][self.state.y_address]
            self.state.y_address = (self.state.y_address + 1) % self.LCD_WIDTH_PIXELS
            return data
        return 0

# Compatibility alias for existing code
HD61202Interpreter = HD61202

# --- Helper Functions ---

def parse_command(addr: int, value: int) -> Command:
    """Parses a memory write address and value into an LCD command."""
    addr_hi = addr & 0xF000
    if addr_hi not in [0xA000, 0x2000]:
        raise ValueError(f"Unexpected address high bits: {hex(addr_hi)}")

    addr_lo = addr & 0xFFF
    if not (addr_lo & 1) == ReadWrite.WRITE.value:
        raise ValueError("Command parsing only supports write operations.")

    di = DataInstruction((addr_lo >> 1) & 1)
    cs = ChipSelect((addr_lo >> 2) & 0b11)
    if cs == ChipSelect.NONE:
        raise ValueError("Unexpected Chip Select value: NONE")

    data = value
    instr = None
    if di == DataInstruction.INSTRUCTION:
        instr = Instruction(data >> 6)
        data = data & 0b00111111
        if instr == Instruction.ON_OFF:
            data &= 1
        elif instr == Instruction.SET_PAGE:
            data &= 0b111
    
    return Command(cs, instr, data)

def render_combined_image(lcds: List[HD61202], zoom: int = 1) -> Image.Image:
    """Stitches images from two LCD controllers into the final wide display."""
    images = [lcd.render_vram_image(zoom=1) for lcd in lcds]
    # This layout logic is specific to the device's display configuration
    lcd0w, lcd1w = 64, 56
    height = HD61202.LCD_HEIGHT_PIXELS // 2

    combined_width = lcd0w * 2 + lcd1w * 2
    image = Image.new("RGB", (combined_width, height), "black")
    
    # Page 206 of a manual likely explains this layout
    image.paste(images[0].crop((0, 0, lcd0w, height)), (0, 0))
    image.paste(images[1].crop((0, 0, lcd1w, height)), (lcd0w, 0))
    # The right half is a flipped view of the left VRAMs
    image.paste(images[1].crop((0, height, lcd1w, height*2)).transpose(Image.FLIP_LEFT_RIGHT), (lcd0w + lcd1w, 0))
    image.paste(images[0].crop((0, height, lcd0w, height*2)).transpose(Image.FLIP_LEFT_RIGHT), (lcd0w + lcd1w + lcd1w, 0))
    
    if zoom > 1:
        return image.resize((image.width * zoom, image.height * zoom), Image.NEAREST)
    return image