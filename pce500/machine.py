"""PC-E500 machine configuration and setup."""

from typing import Optional
from .memory import MemoryMapper, ROMRegion, RAMRegion, PeripheralRegion
from .display import T6A04Controller, HD61700Controller


class PCE500Machine:
    """PC-E500 machine configuration."""
    
    # Memory map constants
    INTERNAL_RAM_START = 0x000000
    INTERNAL_RAM_SIZE = 0x2000    # 8KB internal RAM
    
    EXTERNAL_RAM_START = 0x008000
    EXTERNAL_RAM_SIZE = 0x8000     # 32KB external RAM (base model)
    
    ROM_START = 0x040000
    ROM_SIZE = 0x40000             # 256KB ROM
    
    LCD_MAIN_START = 0x007000     # Main LCD controller
    LCD_SUB_START = 0x007800      # Sub LCD controller
    
    def __init__(self):
        self.memory = MemoryMapper()
        self.main_lcd = T6A04Controller(self.LCD_MAIN_START)
        self.sub_lcd = HD61700Controller(self.LCD_SUB_START)
        
        # Setup default memory map
        self._setup_memory_map()
    
    def _setup_memory_map(self) -> None:
        """Setup the default PC-E500 memory map."""
        # Internal RAM (8KB)
        self.memory.add_region(
            RAMRegion(self.INTERNAL_RAM_START, self.INTERNAL_RAM_SIZE, "Internal RAM")
        )
        
        # External RAM (32KB base, can be expanded)
        self.memory.add_region(
            RAMRegion(self.EXTERNAL_RAM_START, self.EXTERNAL_RAM_SIZE, "External RAM")
        )
        
        # LCD controllers as peripherals
        self.memory.add_region(
            PeripheralRegion(self.LCD_MAIN_START, 0x800, self.main_lcd, "Main LCD")
        )
        self.memory.add_region(
            PeripheralRegion(self.LCD_SUB_START, 0x100, self.sub_lcd, "Sub LCD")
        )
    
    def load_rom(self, rom_data: bytes, start_address: Optional[int] = None) -> None:
        """Load ROM data at specified address."""
        if start_address is None:
            start_address = self.ROM_START
        
        # Validate ROM size
        if len(rom_data) > self.ROM_SIZE:
            raise ValueError(f"ROM size {len(rom_data)} exceeds maximum {self.ROM_SIZE}")
        
        # Add ROM region
        self.memory.add_region(
            ROMRegion(start_address, rom_data, "System ROM")
        )
    
    def expand_ram(self, size: int, start_address: int) -> None:
        """Add RAM expansion module."""
        self.memory.add_region(
            RAMRegion(start_address, size, f"RAM Expansion ({size//1024}KB)")
        )
    
    def reset(self) -> None:
        """Reset machine state."""
        # Reset display controllers
        self.main_lcd.clear_display()
        self.main_lcd.display_on = False
        self.sub_lcd.clear_display()
        self.sub_lcd.display_on = False
        
        # Clear RAM regions
        for region in self.memory.regions:
            if isinstance(region, RAMRegion):
                region.data.fill(0)
    
    def get_memory_info(self) -> str:
        """Get information about current memory configuration."""
        return self.memory.dump_regions()