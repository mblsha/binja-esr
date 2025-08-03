"""Memory overlay for PC-E500 keyboard registers."""

from typing import Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from sc62015.pysc62015.instr.opcodes import IMEMRegisters, INTERNAL_MEMORY_START
from pce500.memory import MemoryOverlay


class KeyboardMemoryOverlay:
    """Creates memory overlay for keyboard register handling."""
    
    @staticmethod
    def create_overlay(keyboard_handler) -> MemoryOverlay:
        """Create a memory overlay for keyboard registers.
        
        Args:
            keyboard_handler: PCE500KeyboardHandler instance
            
        Returns:
            MemoryOverlay configured for keyboard I/O
        """
        def read_handler(address: int, cpu_pc: Optional[int] = None) -> int:
            """Handle reads from keyboard registers."""
            # Calculate register offset within internal memory
            register = address - INTERNAL_MEMORY_START
            
            # Let keyboard handler process the read
            result = keyboard_handler.handle_register_read(register)
            if result is not None:
                return result
                
            # If not a keyboard register, return 0
            return 0x00
            
        def write_handler(address: int, value: int, cpu_pc: Optional[int] = None) -> None:
            """Handle writes to keyboard registers."""
            # Calculate register offset within internal memory  
            register = address - INTERNAL_MEMORY_START
            
            # Let keyboard handler process the write
            keyboard_handler.handle_register_write(register, value)
        
        # Create overlay for keyboard registers (KOL, KOH, KIL)
        # These are at offsets 0xF0-0xF2 in internal memory
        return MemoryOverlay(
            start=INTERNAL_MEMORY_START + IMEMRegisters.KOL,
            end=INTERNAL_MEMORY_START + IMEMRegisters.KIL,
            name="keyboard_io",
            read_only=False,
            read_handler=read_handler,
            write_handler=write_handler,
            perfetto_thread="I/O"
        )