"""Memory and state inspection utilities for PC-E500 emulator."""

from typing import List, Optional
from ..memory import MemoryMapper


class MemoryInspector:
    """Memory inspection and debugging utilities."""
    
    def __init__(self, memory: MemoryMapper):
        self.memory = memory
    
    def dump_memory(self, start: int, length: int, width: int = 16) -> str:
        """Dump memory in hex format."""
        lines = []
        
        for offset in range(0, length, width):
            # Address
            addr = start + offset
            line = f"{addr:06X}: "
            
            # Hex bytes
            hex_part = ""
            ascii_part = ""
            
            for i in range(width):
                if offset + i < length:
                    byte = self.memory.read_byte(addr + i)
                    hex_part += f"{byte:02X} "
                    # ASCII representation
                    if 0x20 <= byte <= 0x7E:
                        ascii_part += chr(byte)
                    else:
                        ascii_part += "."
                else:
                    hex_part += "   "
            
            line += hex_part.ljust(width * 3 + 1) + "|" + ascii_part + "|"
            lines.append(line)
        
        return "\n".join(lines)
    
    def find_string(self, text: str, start: int = 0, end: int = 0xFFFFFF) -> List[int]:
        """Find string in memory."""
        addresses = []
        text_bytes = text.encode('ascii', errors='ignore')
        
        for addr in range(start, end - len(text_bytes) + 1):
            match = True
            for i, byte in enumerate(text_bytes):
                if self.memory.read_byte(addr + i) != byte:
                    match = False
                    break
            if match:
                addresses.append(addr)
        
        return addresses
    
    def find_pattern(self, pattern: List[Optional[int]], 
                    start: int = 0, end: int = 0xFFFFFF) -> List[int]:
        """Find byte pattern in memory (None = wildcard)."""
        addresses = []
        
        for addr in range(start, end - len(pattern) + 1):
            match = True
            for i, byte in enumerate(pattern):
                if byte is not None:
                    if self.memory.read_byte(addr + i) != byte:
                        match = False
                        break
            if match:
                addresses.append(addr)
        
        return addresses
    
    def watch_memory(self, address: int, length: int = 1) -> bytes:
        """Read memory region for watching."""
        data = bytearray()
        for i in range(length):
            data.append(self.memory.read_byte(address + i))
        return bytes(data)
    
    def compare_memory(self, addr1: int, addr2: int, length: int) -> List[int]:
        """Compare two memory regions and return differing offsets."""
        differences = []
        
        for offset in range(length):
            byte1 = self.memory.read_byte(addr1 + offset)
            byte2 = self.memory.read_byte(addr2 + offset)
            if byte1 != byte2:
                differences.append(offset)
        
        return differences