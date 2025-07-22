"""Memory region definitions for PC-E500."""

from abc import ABC, abstractmethod


class MemoryRegion(ABC):
    """Base class for memory regions."""
    
    def __init__(self, start: int, size: int, name: str = ""):
        self.start = start
        self.size = size
        self.end = start + size - 1
        self.name = name
    
    def contains(self, address: int) -> bool:
        """Check if address is within this region."""
        return self.start <= address <= self.end
    
    def offset(self, address: int) -> int:
        """Get offset within region for given address."""
        return address - self.start
    
    @abstractmethod
    def read_byte(self, offset: int) -> int:
        """Read a byte at given offset within region."""
        pass
    
    @abstractmethod
    def write_byte(self, offset: int, value: int) -> None:
        """Write a byte at given offset within region."""
        pass


class ROMRegion(MemoryRegion):
    """Read-only memory region."""
    
    def __init__(self, start: int, data: bytes, name: str = "ROM"):
        super().__init__(start, len(data), name)
        self.data = bytearray(data)
    
    def read_byte(self, offset: int) -> int:
        if 0 <= offset < self.size:
            return self.data[offset]
        return 0xFF  # Return 0xFF for out-of-bounds reads
    
    def write_byte(self, offset: int, value: int) -> None:
        # ROM is read-only, writes are ignored
        pass


class RAMRegion(MemoryRegion):
    """Read-write memory region."""
    
    def __init__(self, start: int, size: int, name: str = "RAM"):
        super().__init__(start, size, name)
        self.data = bytearray(size)
    
    def read_byte(self, offset: int) -> int:
        if 0 <= offset < self.size:
            return self.data[offset]
        return 0xFF
    
    def write_byte(self, offset: int, value: int) -> None:
        if 0 <= offset < self.size:
            self.data[offset] = value & 0xFF


class PeripheralRegion(MemoryRegion):
    """Memory-mapped peripheral region."""
    
    def __init__(self, start: int, size: int, peripheral, name: str = "Peripheral"):
        super().__init__(start, size, name)
        self.peripheral = peripheral
    
    def read_byte(self, offset: int) -> int:
        return self.peripheral.read(self.start + offset)
    
    def write_byte(self, offset: int, value: int) -> None:
        self.peripheral.write(self.start + offset, value)