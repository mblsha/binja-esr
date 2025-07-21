"""HD61700 LCD controller implementation for sub display."""

from .lcd_controller import LCDController


class HD61700Controller(LCDController):
    """HD61700 LCD controller for PC-E500 sub display."""
    
    # Register offsets
    REG_CONTROL = 0x00
    REG_DATA = 0x01
    
    def __init__(self, start_addr: int = 0x7800):
        # HD61700: Typically smaller display, e.g., 24x2 characters (192x16 pixels)
        super().__init__(width=192, height=16, start_addr=start_addr, size=0x100)
        
        # Display RAM organized differently than T6A04
        self.display_ram = bytearray(0x100)
        
        # Control state
        self.address_counter = 0
        self.auto_increment = True
        
    def _handle_read(self, offset: int) -> int:
        """Handle read from HD61700."""
        if offset == self.REG_CONTROL:
            # Status register
            return 0x00  # Not busy
        elif offset == self.REG_DATA:
            # Read from current address
            data = self.display_ram[self.address_counter & 0xFF]
            if self.auto_increment:
                self.address_counter = (self.address_counter + 1) & 0xFF
            return data
        return 0xFF
    
    def _handle_write(self, offset: int, value: int) -> None:
        """Handle write to HD61700."""
        if offset == self.REG_CONTROL:
            self._process_control(value)
        elif offset == self.REG_DATA:
            self._write_data(value)
    
    def _process_control(self, value: int) -> None:
        """Process control register write."""
        # Simplified control logic
        if value & 0x80:
            # Set address counter
            self.address_counter = value & 0x7F
        elif value & 0x40:
            # Display control
            self.display_on = bool(value & 0x01)
        elif value & 0x20:
            # Auto increment control
            self.auto_increment = bool(value & 0x01)
    
    def _write_data(self, value: int) -> None:
        """Write data to display RAM."""
        self.display_ram[self.address_counter & 0xFF] = value
        self._update_display_region(self.address_counter)
        
        if self.auto_increment:
            self.address_counter = (self.address_counter + 1) & 0xFF
    
    def _update_display_region(self, address: int) -> None:
        """Update display buffer based on RAM write."""
        # Map RAM address to display coordinates
        # This is a simplified mapping - actual hardware may differ
        
        # Assume 24 bytes per row (192 pixels / 8 bits)
        bytes_per_row = self.width // 8
        row = address // bytes_per_row
        col_byte = address % bytes_per_row
        
        if row >= self.height // 8:
            return
        
        # Update 8 pixels
        data = self.display_ram[address & 0xFF]
        x_base = col_byte * 8
        y_base = row * 8
        
        for bit in range(8):
            x = x_base + bit
            if x < self.width:
                for y_offset in range(8):
                    y = y_base + y_offset
                    if y < self.height:
                        pixel_on = (data >> bit) & 1
                        self.set_pixel(x, y, bool(pixel_on))