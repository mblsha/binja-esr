"""Memory subsystem for PC-E500 emulator."""

from .mapper import MemoryMapper
from .regions import MemoryRegion, ROMRegion, RAMRegion, PeripheralRegion

__all__ = ["MemoryMapper", "MemoryRegion", "ROMRegion", "RAMRegion", "PeripheralRegion"]