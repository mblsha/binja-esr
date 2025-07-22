"""PC-E500 machine configuration and setup."""

from typing import Optional
from .memory import MemoryMapper, ROMRegion, RAMRegion, PeripheralRegion
from .display import HD61202Controller


class PCE500Machine:
    """PC-E500 machine configuration (hardcoded for PC-E500 only)."""
    
    # Memory map constants (from PC-E500 specification)
    INTERNAL_ROM_START = 0xC0000
    INTERNAL_ROM_SIZE = 0x40000    # 256KB internal ROM
    
    USER_AREA_START = 0xB8000
    USER_AREA_SIZE = 0x4C00        # Size up to BEC00H
    
    MACHINE_CODE_START = 0xBEC00  
    MACHINE_CODE_SIZE = 0x1234     # 0x1234 bytes
    
    INTERNAL_RAM_START = 0xB8000
    INTERNAL_RAM_SIZE = 0x8000     # 32KB internal RAM
    
    # Memory cards
    CARD_8KB_START = 0x40000
    CARD_16KB_START = 0x48000
    CARD_32KB_START = 0x44000
    CARD_64KB_START = 0x40000
    
    # LCD controller in 0x2xxxx space
    LCD_MAIN_START = 0x20000       # Main LCD controller (HD61202)
    
    def __init__(self):
        self.memory = MemoryMapper()
        self.main_lcd = HD61202Controller(self.LCD_MAIN_START)
        
        # Setup default memory map
        self._setup_memory_map()
    
    def _setup_memory_map(self) -> None:
        """Setup the default PC-E500 memory map."""
        # Internal RAM (32KB at 0xB8000-0xBFFFF)
        # This includes both user area and work area
        self.memory.add_region(
            RAMRegion(self.INTERNAL_RAM_START, self.INTERNAL_RAM_SIZE, "Internal RAM")
        )
        
        # LCD controller in 0x2xxxx space
        # The controller uses complex address decoding:
        # - They respond to addresses in 0x2xxxx range
        # - Address bits control chip selection and operation type
        # HD61202 dual-chip controller for 240x32 display
        self.memory.add_region(
            PeripheralRegion(0x20000, 0x10000, self.main_lcd, "LCD (HD61202)")
        )
    
    def load_rom(self, rom_data: bytes, start_address: Optional[int] = None) -> None:
        """Load ROM data at specified address."""
        if start_address is None:
            start_address = self.INTERNAL_ROM_START
        
        # Validate ROM size
        if len(rom_data) > self.INTERNAL_ROM_SIZE:
            raise ValueError(f"ROM size {len(rom_data)} exceeds maximum {self.INTERNAL_ROM_SIZE}")
        
        # Add ROM region
        self.memory.add_region(
            ROMRegion(start_address, rom_data, "Internal ROM")
        )
    
    def load_memory_card(self, card_data: bytes, card_size: int) -> None:
        """Load a memory card (8KB, 16KB, 32KB, or 64KB)."""
        card_starts = {
            8192: self.CARD_8KB_START,    # 8KB
            16384: self.CARD_16KB_START,   # 16KB
            32768: self.CARD_32KB_START,   # 32KB
            65536: self.CARD_64KB_START    # 64KB
        }
        
        if card_size not in card_starts:
            raise ValueError(f"Invalid card size {card_size}. Must be 8KB, 16KB, 32KB, or 64KB")
            
        if len(card_data) > card_size:
            raise ValueError(f"Card data size {len(card_data)} exceeds card size {card_size}")
            
        self.memory.add_region(
            ROMRegion(card_starts[card_size], card_data, f"{card_size//1024}KB Card")
        )
    
    def expand_ram(self, size: int, start_address: int) -> None:
        """Add RAM expansion module."""
        self.memory.add_region(
            RAMRegion(start_address, size, f"RAM Expansion ({size//1024}KB)")
        )
    
    def reset(self) -> None:
        """Reset machine state."""
        # Reset display controller
        # For HD61202, we need to reset the interpreters
        for chip in self.main_lcd.chips.values():
            chip.vram = bytearray(chip.vram_size)
            chip.display_on = False
            chip.page = 0
            chip.y_addr = 0
            chip.start_line = 0
        
        # Clear RAM regions
        for region in self.memory.regions:
            if isinstance(region, RAMRegion):
                region.data.fill(0)
    
    def get_memory_info(self) -> str:
        """Get information about current memory configuration."""
        return self.memory.dump_regions()