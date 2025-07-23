"""Simplified memory implementation for PC-E500 emulator."""

from typing import Optional
from dataclasses import dataclass

from .trace_manager import g_tracer


@dataclass
class MemoryRegion:
    """Simple memory region descriptor."""
    start: int
    end: int
    data: Optional[bytearray] = None  # None for ROM
    rom_data: Optional[bytes] = None  # None for RAM
    name: str = ""
    

class PCE500Memory:
    """Memory manager for PC-E500.
    
    Direct memory access implementation that replaces the original
    complex 4-layer abstraction.
    """
    
    def __init__(self):
        # Pre-allocate common regions
        self.internal_ram = bytearray(32 * 1024)  # 32KB at 0xB8000
        self.internal_rom: Optional[bytes] = None  # 256KB at 0xC0000
        
        # Optional components
        self.lcd_controller = None
        self.memory_card: Optional[bytes] = None
        self.card_start = 0
        
        # Additional regions for expansion
        self.regions: list[MemoryRegion] = []
        
        # Perfetto tracing
        self.perfetto_enabled = False
        
    def load_rom(self, rom_data: bytes) -> None:
        """Load internal ROM."""
        if len(rom_data) > 256 * 1024:
            raise ValueError(f"ROM too large: {len(rom_data)} bytes")
        self.internal_rom = rom_data
        
    def load_memory_card(self, card_data: bytes, card_size: int) -> None:
        """Load memory card."""
        card_starts = {
            8192: 0x40000,    # 8KB
            16384: 0x48000,   # 16KB
            32768: 0x44000,   # 32KB
            65536: 0x40000    # 64KB
        }
        
        if card_size not in card_starts:
            raise ValueError(f"Invalid card size: {card_size}")
            
        self.memory_card = card_data
        self.card_start = card_starts[card_size]
        
    def set_lcd_controller(self, controller) -> None:
        """Set LCD controller for memory-mapped I/O."""
        self.lcd_controller = controller
        
    def read_byte(self, address: int, cpu_pc: Optional[int] = None) -> int:
        """Read a byte from memory.
        
        Args:
            address: Memory address to read from
            cpu_pc: Optional CPU program counter for tracing context
        """
        address &= 0xFFFFFF  # 24-bit address space
        
        # Internal ROM (0xC0000-0xFFFFF)
        if 0xC0000 <= address <= 0xFFFFF:
            if self.internal_rom:
                offset = address - 0xC0000
                if offset < len(self.internal_rom):
                    return self.internal_rom[offset]
            return 0xFF
            
        # Internal RAM (0xB8000-0xBFFFF)
        elif 0xB8000 <= address <= 0xBFFFF:
            value = self.internal_ram[address - 0xB8000]
            
            # Perfetto tracing for RAM reads
            if self.perfetto_enabled and cpu_pc is not None:
                g_tracer.trace_instant("Memory", "RAM_Read", {
                    "addr": f"0x{address:06X}",
                    "value": f"0x{value:02X}",
                    "pc": f"0x{cpu_pc:06X}"
                })
            
            return value
            
        # LCD controller (0x20000-0x2FFFF)
        elif 0x20000 <= address <= 0x2FFFF and self.lcd_controller:
            return self.lcd_controller.read(address, cpu_pc)
            
        # Memory card
        elif self.memory_card and self.card_start <= address < self.card_start + len(self.memory_card):
            return self.memory_card[address - self.card_start]
            
        # Check additional regions
        for region in self.regions:
            if region.start <= address <= region.end:
                offset = address - region.start
                if region.rom_data:
                    return region.rom_data[offset] if offset < len(region.rom_data) else 0xFF
                elif region.data:
                    return region.data[offset]
                    
        return 0xFF  # Default for unmapped memory
        
    def write_byte(self, address: int, value: int, cpu_pc: Optional[int] = None) -> None:
        """Write a byte to memory.
        
        Args:
            address: Memory address to write to
            value: Byte value to write
            cpu_pc: Optional CPU program counter for tracing context
        """
        address &= 0xFFFFFF  # 24-bit address space
        value &= 0xFF
        
        # Internal RAM (0xB8000-0xBFFFF)
        if 0xB8000 <= address <= 0xBFFFF:
            self.internal_ram[address - 0xB8000] = value
            
            # Perfetto tracing for RAM writes
            if self.perfetto_enabled:
                trace_data = {"addr": f"0x{address:06X}", "value": f"0x{value:02X}"}
                if cpu_pc is not None:
                    trace_data["pc"] = f"0x{cpu_pc:06X}"
                g_tracer.trace_instant("Memory", "RAM_Write", trace_data)
            
        # LCD controller (0x20000-0x2FFFF)
        elif 0x20000 <= address <= 0x2FFFF and self.lcd_controller:
            self.lcd_controller.write(address, value, cpu_pc)
            
        # ROM regions - silently ignore writes
        elif 0xC0000 <= address <= 0xFFFFF:
            if self.perfetto_enabled:
                g_tracer.trace_instant("Memory", "ROM_Write_Ignored", 
                                     {"addr": f"0x{address:06X}", "value": f"0x{value:02X}"})
            
        # Memory card - ROM, ignore writes
        elif self.memory_card and self.card_start <= address < self.card_start + len(self.memory_card):
            pass  # Card ROM write ignored
            
        # Check additional regions
        else:
            for region in self.regions:
                if region.start <= address <= region.end:
                    if region.data:  # RAM region
                        offset = address - region.start
                        if offset < len(region.data):
                            region.data[offset] = value
                    # ROM regions ignore writes
                    break
                    
    def read_word(self, address: int) -> int:
        """Read 16-bit word (little-endian)."""
        low = self.read_byte(address)
        high = self.read_byte(address + 1)
        return low | (high << 8)
        
    def write_word(self, address: int, value: int) -> None:
        """Write 16-bit word (little-endian)."""
        self.write_byte(address, value & 0xFF)
        self.write_byte(address + 1, (value >> 8) & 0xFF)
        
    def read_long(self, address: int) -> int:
        """Read 24-bit long (little-endian)."""
        low = self.read_byte(address)
        mid = self.read_byte(address + 1)
        high = self.read_byte(address + 2)
        return low | (mid << 8) | (high << 16)
        
    def write_long(self, address: int, value: int) -> None:
        """Write 24-bit long (little-endian)."""
        self.write_byte(address, value & 0xFF)
        self.write_byte(address + 1, (value >> 8) & 0xFF)
        self.write_byte(address + 2, (value >> 16) & 0xFF)
        
    def add_ram(self, start: int, size: int, name: str = "") -> None:
        """Add additional RAM region."""
        self.regions.append(MemoryRegion(
            start=start,
            end=start + size - 1,
            data=bytearray(size),
            name=name
        ))
        
    def add_rom(self, start: int, data: bytes, name: str = "") -> None:
        """Add additional ROM region."""
        self.regions.append(MemoryRegion(
            start=start,
            end=start + len(data) - 1,
            rom_data=data,
            name=name
        ))
        
    def reset(self) -> None:
        """Reset all RAM to zero."""
        self.internal_ram[:] = bytes(len(self.internal_ram))
        for region in self.regions:
            if region.data:  # RAM region
                region.data[:] = bytes(len(region.data))
                
    def read_bytes(self, address: int, size: int) -> bytes:
        """Read multiple bytes from memory (for SC62015 emulator compatibility)."""
        result = bytearray()
        for i in range(size):
            result.append(self.read_byte(address + i))
        return bytes(result)
        
    def get_memory_info(self) -> str:
        """Get memory configuration info."""
        lines = ["Memory Configuration:"]
        
        if self.internal_rom:
            lines.append(f"  ROM: 0xC0000-0xFFFFF ({len(self.internal_rom)//1024}KB)")
        lines.append("  RAM: 0xB8000-0xBFFFF (32KB)")
        
        if self.memory_card:
            lines.append(f"  Card: 0x{self.card_start:05X} ({len(self.memory_card)//1024}KB)")
            
        if self.lcd_controller:
            lines.append("  LCD: 0x20000-0x2FFFF")
            
        for region in self.regions:
            size = (region.end - region.start + 1) // 1024
            rtype = "RAM" if region.data else "ROM"
            lines.append(f"  {rtype}: 0x{region.start:05X}-0x{region.end:05X} ({size}KB) {region.name}")
            
        return "\n".join(lines)
        
    def set_perfetto_enabled(self, enabled: bool) -> None:
        """Enable or disable Perfetto tracing."""
        self.perfetto_enabled = enabled


# Backward compatibility aliases
SimplifiedMemory = PCE500Memory
MemoryMapper = PCE500Memory  # For code expecting the old MemoryMapper class