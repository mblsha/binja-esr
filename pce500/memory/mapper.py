"""Memory mapper for PC-E500 emulator."""

from typing import List, Optional, Dict, Any
from .regions import MemoryRegion, PeripheralRegion


class MemoryMapper:
    """Maps memory addresses to appropriate regions."""
    
    def __init__(self):
        self.regions: List[MemoryRegion] = []
        self._context: Optional[Dict[str, Any]] = None
    
    def add_region(self, region: MemoryRegion) -> None:
        """Add a memory region to the mapper."""
        # Check for overlaps
        for existing in self.regions:
            if (region.start <= existing.end and region.end >= existing.start):
                raise ValueError(
                    f"Region {region.name} ({region.start:06X}-{region.end:06X}) "
                    f"overlaps with {existing.name} ({existing.start:06X}-{existing.end:06X})"
                )
        self.regions.append(region)
        # Keep regions sorted by start address
        self.regions.sort(key=lambda r: r.start)
    
    def find_region(self, address: int) -> Optional[MemoryRegion]:
        """Find the region containing the given address."""
        # Binary search could be used for better performance
        for region in self.regions:
            if region.contains(address):
                return region
        return None
    
    def set_context(self, context: Optional[Dict[str, Any]]) -> None:
        """Set the context for memory operations (e.g., CPU state).
        
        Args:
            context: Dictionary containing contextual information like cpu_pc
        """
        self._context = context
    
    def read_byte(self, address: int) -> int:
        """Read a byte from the given address."""
        address &= 0xFFFFFF  # 24-bit address space
        region = self.find_region(address)
        if region:
            # Pass context to peripherals
            if isinstance(region, PeripheralRegion) and self._context:
                cpu_pc = self._context.get('cpu_pc')
                return region.read_byte(region.offset(address), cpu_pc=cpu_pc)
            else:
                return region.read_byte(region.offset(address))
        return 0xFF  # Return 0xFF for unmapped addresses
    
    def write_byte(self, address: int, value: int) -> None:
        """Write a byte to the given address."""
        address &= 0xFFFFFF  # 24-bit address space
        region = self.find_region(address)
        if region:
            # Pass context to peripherals
            if isinstance(region, PeripheralRegion) and self._context:
                cpu_pc = self._context.get('cpu_pc')
                region.write_byte(region.offset(address), value, cpu_pc=cpu_pc)
            else:
                region.write_byte(region.offset(address), value)
        # Writes to unmapped addresses are ignored
    
    def read_word(self, address: int) -> int:
        """Read a 16-bit word (little-endian)."""
        low = self.read_byte(address)
        high = self.read_byte(address + 1)
        return low | (high << 8)
    
    def write_word(self, address: int, value: int) -> None:
        """Write a 16-bit word (little-endian)."""
        self.write_byte(address, value & 0xFF)
        self.write_byte(address + 1, (value >> 8) & 0xFF)
    
    def read_long(self, address: int) -> int:
        """Read a 24-bit long (little-endian)."""
        low = self.read_byte(address)
        mid = self.read_byte(address + 1)
        high = self.read_byte(address + 2)
        return low | (mid << 8) | (high << 16)
    
    def write_long(self, address: int, value: int) -> None:
        """Write a 24-bit long (little-endian)."""
        self.write_byte(address, value & 0xFF)
        self.write_byte(address + 1, (value >> 8) & 0xFF)
        self.write_byte(address + 2, (value >> 16) & 0xFF)
    
    def dump_regions(self) -> str:
        """Return a string representation of all memory regions."""
        lines = ["Memory Map:"]
        for region in self.regions:
            lines.append(f"  {region.start:06X}-{region.end:06X}: {region.name} ({region.size} bytes)")
        return "\n".join(lines)