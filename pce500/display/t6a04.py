"""T6A04 LCD controller implementation for main display (160x64)."""

from .lcd_controller import LCDController


class T6A04Controller(LCDController):
    """T6A04 LCD controller for PC-E500 main display."""
    
    # Command registers
    REG_COMMAND = 0x00
    REG_DATA = 0x01
    
    # Commands
    CMD_DISPLAY_ON = 0xAF
    CMD_DISPLAY_OFF = 0xAE
    CMD_SET_PAGE = 0xB0  # B0-B7 for pages 0-7
    CMD_SET_COLUMN_HIGH = 0x10  # Upper nibble of column address
    CMD_SET_COLUMN_LOW = 0x00   # Lower nibble of column address
    CMD_SET_START_LINE = 0x40  # 40-7F for lines 0-63
    
    def __init__(self, start_addr: int = 0x7000):
        # T6A04: 160x64 pixels, organized as 8 pages of 8 pixels high
        super().__init__(width=160, height=64, start_addr=start_addr, size=0x800)
        
        # Display RAM: 8 pages x 160 columns
        self.display_ram = [[0] * 160 for _ in range(8)]
        
        # Current page and column for data writes
        self.current_page = 0
        self.current_column = 0
        self.start_line = 0
        
        # Status register
        self.busy = False
        
    def _handle_read(self, offset: int) -> int:
        """Handle read from T6A04."""
        if offset == self.REG_COMMAND:
            # Status read: bit 7 = busy flag
            return 0x80 if self.busy else 0x00
        elif offset == self.REG_DATA:
            # Data read from current position
            if self.current_column < 160:
                data = self.display_ram[self.current_page][self.current_column]
                self.current_column = (self.current_column + 1) % 160
                return data
        return 0xFF
    
    def _handle_write(self, offset: int, value: int) -> None:
        """Handle write to T6A04."""
        if offset == self.REG_COMMAND:
            self._process_command(value)
        elif offset == self.REG_DATA:
            self._write_data(value)
    
    def _process_command(self, cmd: int) -> None:
        """Process LCD command."""
        if cmd == self.CMD_DISPLAY_ON:
            self.display_on = True
        elif cmd == self.CMD_DISPLAY_OFF:
            self.display_on = False
        elif (cmd & 0xF8) == self.CMD_SET_PAGE:
            # Set page address (0-7)
            self.current_page = cmd & 0x07
        elif (cmd & 0xF0) == self.CMD_SET_COLUMN_HIGH:
            # Set column address high nibble
            self.current_column = (self.current_column & 0x0F) | ((cmd & 0x0F) << 4)
        elif (cmd & 0xF0) == self.CMD_SET_COLUMN_LOW:
            # Set column address low nibble
            self.current_column = (self.current_column & 0xF0) | (cmd & 0x0F)
        elif (cmd & 0xC0) == self.CMD_SET_START_LINE:
            # Set display start line
            self.start_line = cmd & 0x3F
            self._update_display()
    
    def _write_data(self, value: int) -> None:
        """Write data to display RAM."""
        if self.current_column < 160:
            self.display_ram[self.current_page][self.current_column] = value
            self._update_pixel_column(self.current_page, self.current_column, value)
            self.current_column = (self.current_column + 1) % 160
    
    def _update_pixel_column(self, page: int, column: int, data: int) -> None:
        """Update display buffer for a column of 8 pixels."""
        if column >= self.width:
            return
            
        # Each page is 8 pixels high
        y_base = page * 8
        
        # Update 8 pixels in the column
        for bit in range(8):
            y = y_base + bit
            if y < self.height:
                pixel_on = (data >> bit) & 1
                self.set_pixel(column, y, bool(pixel_on))
    
    def _update_display(self) -> None:
        """Update entire display from RAM (used when start line changes)."""
        for page in range(8):
            for column in range(160):
                data = self.display_ram[page][column]
                self._update_pixel_column(page, column, data)