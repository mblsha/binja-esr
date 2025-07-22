"""Base LCD controller class for PC-E500 emulator."""

from abc import ABC, abstractmethod
import numpy as np


class LCDController(ABC):
    """Base class for LCD controllers."""
    
    def __init__(self, width: int, height: int, start_addr: int, size: int):
        self.width = width
        self.height = height
        self.start_addr = start_addr
        self.size = size
        self.end_addr = start_addr + size - 1
        
        # Display buffer (1 bit per pixel)
        self.display_buffer = np.zeros((height, width), dtype=np.uint8)
        
        # Control registers
        self.display_on = False
        self.cursor_x = 0
        self.cursor_y = 0
        
    def contains_address(self, address: int) -> bool:
        """Check if address is handled by this controller."""
        return self.start_addr <= address <= self.end_addr
    
    def read(self, address: int) -> int:
        """Read from LCD controller."""
        if not self.contains_address(address):
            return 0xFF
        
        offset = address - self.start_addr
        return self._handle_read(offset)
    
    def write(self, address: int, value: int) -> None:
        """Write to LCD controller."""
        if not self.contains_address(address):
            return
        
        offset = address - self.start_addr
        self._handle_write(offset, value)
    
    @abstractmethod
    def _handle_read(self, offset: int) -> int:
        """Handle read from controller-specific offset."""
        pass
    
    @abstractmethod
    def _handle_write(self, offset: int, value: int) -> None:
        """Handle write to controller-specific offset."""
        pass
    
    def get_display_buffer(self) -> np.ndarray:
        """Get the current display buffer."""
        return self.display_buffer.copy()
    
    def clear_display(self) -> None:
        """Clear the display buffer."""
        self.display_buffer.fill(0)
    
    def set_pixel(self, x: int, y: int, value: bool) -> None:
        """Set a pixel in the display buffer."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.display_buffer[y, x] = 1 if value else 0
    
    def get_pixel(self, x: int, y: int) -> bool:
        """Get a pixel from the display buffer."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return bool(self.display_buffer[y, x])
        return False