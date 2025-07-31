"""Simplified memory implementation for PC-E500 emulator."""

from typing import Optional, Callable, Union, List
from dataclasses import dataclass

from .trace_manager import g_tracer

# Import constants for accessing internal memory registers
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from sc62015.pysc62015.instr.opcodes import IMEMRegisters, INTERNAL_MEMORY_START


@dataclass
class MemoryOverlay:
    """Represents a memory overlay that can override reads/writes."""
    start: int
    end: int
    name: str
    read_only: bool = True
    data: Optional[Union[bytes, bytearray]] = None
    read_handler: Optional[Callable[[int, Optional[int]], int]] = None  # (address, cpu_pc) -> byte
    write_handler: Optional[Callable[[int, int, Optional[int]], None]] = None  # (address, value, cpu_pc) -> None
    

class PCE500Memory:
    """Memory manager for PC-E500.
    
    Direct memory access implementation that replaces the original
    complex 4-layer abstraction.
    """
    
    def __init__(self):
        # Base 1MB external memory (0x00000-0xFFFFF)
        self.external_memory = bytearray(1024 * 1024)
        
        # Overlays list (checked in order)
        self.overlays: List[MemoryOverlay] = []
        
        # Perfetto tracing
        self.perfetto_enabled = False
        
        # Reference to CPU emulator for accessing internal memory registers
        self.cpu = None
        
    # The new implementations are in the add_overlay section below
        
    def read_byte(self, address: int, cpu_pc: Optional[int] = None) -> int:
        """Read a byte from memory.
        
        Args:
            address: Memory address to read from
            cpu_pc: Optional CPU program counter for tracing context
        """
        address &= 0xFFFFFF  # 24-bit address space
        
        # Check for SC62015 internal memory (0x100000-0x1000FF)
        if address >= 0x100000:
            # Internal 256-byte RAM
            offset = address - 0x100000
            if offset < 0x100:
                # Internal memory stored at end of external_memory for compatibility
                internal_offset = len(self.external_memory) - 256 + offset
                return self.external_memory[internal_offset]
            raise ValueError(f"Invalid SC62015 internal memory address: 0x{address:06X} (offset 0x{offset:02X} >= 0x100)")
        
        # External memory space (0x00000-0xFFFFF)
        address &= 0xFFFFF
        
        # Check overlays in order
        for overlay in self.overlays:
            if overlay.start <= address <= overlay.end:
                if overlay.read_handler:
                    return overlay.read_handler(address, cpu_pc)
                elif overlay.data:
                    offset = address - overlay.start
                    if offset < len(overlay.data):
                        return overlay.data[offset]
                return 0x00
        
        # Default to external memory
        return self.external_memory[address]
        
    def write_byte(self, address: int, value: int, cpu_pc: Optional[int] = None) -> None:
        """Write a byte to memory.
        
        Args:
            address: Memory address to write to
            value: Byte value to write
            cpu_pc: Optional CPU program counter for tracing context
        """
        address &= 0xFFFFFF  # 24-bit address space
        value &= 0xFF
        
        # Check for SC62015 internal memory (0x100000-0x1000FF)
        if address >= 0x100000:
            # Internal 256-byte RAM
            offset = address - 0x100000
            if offset < 0x100:
                # Internal memory stored at end of external_memory for compatibility
                internal_offset = len(self.external_memory) - 256 + offset
                self.external_memory[internal_offset] = value
                
                if self.perfetto_enabled:
                    # Get current BP value from internal memory if CPU is available
                    bp_value = "N/A"
                    if self.cpu:
                        try:
                            bp_addr = INTERNAL_MEMORY_START + IMEMRegisters.BP
                            bp_value = f"0x{self.cpu.memory.read_byte(bp_addr):02X}"
                        except Exception:
                            bp_value = "N/A"
                    
                    # Check if this offset corresponds to a known internal memory register
                    imem_name = "N/A"
                    for reg_name in IMEMRegisters.__members__:
                        if IMEMRegisters[reg_name].value == offset:
                            imem_name = reg_name
                            break
                    
                    g_tracer.trace_instant("Memory_Internal", "", {
                        "offset": f"0x{offset:02X}",
                        "value": f"0x{value:02X}",
                        "pc": f"0x{cpu_pc:06X}" if cpu_pc is not None else "N/A",
                        "bp": bp_value,
                        "imem_name": imem_name,
                        "size": "1"  # Always 1 for byte writes
                    })
            else:
                raise ValueError(f"Invalid SC62015 internal memory address: 0x{address:06X} (offset 0x{offset:02X} >= 0x100)")
            return
        
        # External memory space (0x00000-0xFFFFF)
        address &= 0xFFFFF
        
        # Check overlays for write handlers or read-only
        for overlay in self.overlays:
            if overlay.start <= address <= overlay.end:
                if overlay.write_handler:
                    overlay.write_handler(address, value, cpu_pc)
                    return
                elif overlay.read_only:
                    # Silently ignore writes to read-only overlays
                    if self.perfetto_enabled:
                        trace_data = {"addr": f"0x{address:06X}", "value": f"0x{value:02X}"}
                        if cpu_pc is not None:
                            trace_data["pc"] = f"0x{cpu_pc:06X}"
                        trace_data["overlay"] = overlay.name
                        g_tracer.trace_instant("Memory_External", "", trace_data)
                    return
        
        # Default to external memory
        self.external_memory[address] = value
        
        # Perfetto tracing for all writes
        if self.perfetto_enabled:
            trace_data = {"addr": f"0x{address:06X}", "value": f"0x{value:02X}"}
            if cpu_pc is not None:
                trace_data["pc"] = f"0x{cpu_pc:06X}"
            g_tracer.trace_instant("Memory_External", "", trace_data)
                    
    def read_word(self, address: int) -> int:
        """Read 16-bit word (little-endian)."""
        low = self.read_byte(address)
        high = self.read_byte(address + 1)
        return low | (high << 8)
        
    def write_word(self, address: int, value: int, cpu_pc: Optional[int] = None) -> None:
        """Write 16-bit word (little-endian)."""
        self.write_byte(address, value & 0xFF, cpu_pc)
        self.write_byte(address + 1, (value >> 8) & 0xFF, cpu_pc)
        
    def read_long(self, address: int) -> int:
        """Read 24-bit long (little-endian)."""
        low = self.read_byte(address)
        mid = self.read_byte(address + 1)
        high = self.read_byte(address + 2)
        return low | (mid << 8) | (high << 16)
        
    def write_long(self, address: int, value: int, cpu_pc: Optional[int] = None) -> None:
        """Write 24-bit long (little-endian)."""
        self.write_byte(address, value & 0xFF, cpu_pc)
        self.write_byte(address + 1, (value >> 8) & 0xFF, cpu_pc)
        self.write_byte(address + 2, (value >> 16) & 0xFF, cpu_pc)
        
    def add_overlay(self, overlay: MemoryOverlay) -> None:
        """Add a memory overlay."""
        self.overlays.append(overlay)
        # Optionally sort by start address for efficiency
        self.overlays.sort(key=lambda o: o.start)
    
    def remove_overlay(self, name: str) -> None:
        """Remove overlay by name."""
        self.overlays = [o for o in self.overlays if o.name != name]
    
    def load_rom(self, rom_data: bytes) -> None:
        """Load ROM as an overlay at 0xC0000."""
        # Remove any existing ROM overlay
        self.remove_overlay("internal_rom")
        
        # Add new ROM overlay
        self.add_overlay(MemoryOverlay(
            start=0xC0000,
            end=0xC0000 + len(rom_data) - 1,
            name="internal_rom",
            read_only=True,
            data=rom_data
        ))
    
    def set_lcd_controller(self, controller) -> None:
        """Add LCD controller as an overlay."""
        # Remove any existing LCD overlay
        self.remove_overlay("lcd_controller")
        
        if controller:
            self.add_overlay(MemoryOverlay(
                start=0x20000,
                end=0x2FFFF,
                name="lcd_controller",
                read_only=False,
                read_handler=lambda addr, pc: controller.read(addr, pc),
                write_handler=lambda addr, val, pc: controller.write(addr, val, pc)
            ))
    
    def load_memory_card(self, card_data: bytes, card_size: int) -> None:
        """Load memory card as overlay."""
        card_starts = {
            8192: 0x40000,    # 8KB
            16384: 0x48000,   # 16KB
            32768: 0x44000,   # 32KB
            65536: 0x40000    # 64KB
        }
        
        if card_size not in card_starts:
            raise ValueError(f"Invalid card size: {card_size}")
        
        # Remove any existing memory card overlay
        self.remove_overlay("memory_card")
        
        # Add new memory card overlay
        start = card_starts[card_size]
        self.add_overlay(MemoryOverlay(
            start=start,
            end=start + len(card_data) - 1,
            name="memory_card",
            read_only=True,
            data=card_data
        ))
    
    def add_ram(self, start: int, size: int, name: str = "") -> None:
        """Add additional RAM overlay (writable)."""
        self.add_overlay(MemoryOverlay(
            start=start,
            end=start + size - 1,
            name=name or f"RAM_0x{start:06X}",
            read_only=False,
            data=bytearray(size)
        ))
        
    def add_rom(self, start: int, data: bytes, name: str = "") -> None:
        """Add additional ROM overlay (read-only)."""
        self.add_overlay(MemoryOverlay(
            start=start,
            end=start + len(data) - 1,
            name=name or f"ROM_0x{start:06X}",
            read_only=True,
            data=data
        ))
        
    def reset(self) -> None:
        """Reset all RAM to zero."""
        # Reset external memory (including internal memory at the end)
        self.external_memory[:] = bytes(len(self.external_memory))
        
        # Reset any writable overlays
        for overlay in self.overlays:
            if not overlay.read_only and overlay.data and isinstance(overlay.data, bytearray):
                overlay.data[:] = bytes(len(overlay.data))
                
    def read_bytes(self, address: int, size: int) -> bytes:
        """Read multiple bytes from memory (for SC62015 emulator compatibility)."""
        result = bytearray()
        for i in range(size):
            result.append(self.read_byte(address + i))
        return bytes(result)
        
    def get_memory_info(self) -> str:
        """Get memory configuration info."""
        lines = ["Memory Configuration:"]
        lines.append("  Base RAM: 0x00000-0xFFFFF (1MB)")
        lines.append("  Internal: 0x100000-0x1000FF (256B)")
        
        if self.overlays:
            lines.append("\nOverlays:")
            for overlay in sorted(self.overlays, key=lambda o: o.start):
                size = (overlay.end - overlay.start + 1)
                if size >= 1024:
                    size_str = f"{size//1024}KB"
                else:
                    size_str = f"{size}B"
                
                overlay_type = "R/O" if overlay.read_only else "R/W"
                if overlay.read_handler or overlay.write_handler:
                    overlay_type = "I/O"
                
                lines.append(f"  {overlay.name}: 0x{overlay.start:05X}-0x{overlay.end:05X} ({size_str}) [{overlay_type}]")
        
        return "\n".join(lines)
        
    def get_internal_memory_bytes(self) -> bytes:
        """Get internal memory (256 bytes) as raw bytes."""
        # Internal memory is stored in the last 256 bytes of external_memory
        start_offset = len(self.external_memory) - 256
        return bytes(self.external_memory[start_offset:])
        
    def set_perfetto_enabled(self, enabled: bool) -> None:
        """Enable or disable Perfetto tracing."""
        self.perfetto_enabled = enabled
        
    def set_cpu(self, cpu) -> None:
        """Set reference to CPU emulator for accessing internal memory registers."""
        self.cpu = cpu


# Backward compatibility aliases
SimplifiedMemory = PCE500Memory
MemoryMapper = PCE500Memory  # For code expecting the old MemoryMapper class