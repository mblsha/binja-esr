"""HD61202Controller wrapper to provide compatibility with the emulator."""

from typing import List, Optional, Dict
import numpy as np
from PIL import Image

from .hd61202 import HD61202, parse_command, render_combined_image
from ..tracing.perfetto_tracing import tracer as new_tracer, perf_trace


class HD61202Controller:
    """Wrapper class that provides the interface expected by the PC-E500 emulator.

    This wraps two HD61202 chips to create the 240x32 display used by PC-E500.
    """

    def __init__(self):
        """Initialize with two HD61202 chips."""
        # PC-E500 uses two standard 64x64 chips, but only uses 4 pages each
        self.chips = [HD61202(), HD61202()]
        self.perfetto_enabled = False

        # Debug counters
        self.cs_both_count = 0
        self.cs_left_count = 0
        self.cs_right_count = 0

    def read(self, address: int, cpu_pc: Optional[int] = None) -> int:
        """Read from LCD controller at given address."""
        # The new parse_command expects addresses in 0x2xxx or 0xAxxx range
        # Just return 0xFF for reads (not implemented in new version)
        return 0xFF

    @perf_trace("Display")
    def write(self, address: int, value: int, cpu_pc: Optional[int] = None) -> None:
        """Write to LCD controller at given address."""
        try:
            cmd = parse_command(address, value)

            # Map chip select to chips
            chips_to_write = []
            if cmd.cs.value == 0b00:  # BOTH
                chips_to_write = [0, 1]
                self.cs_both_count += 1
            elif cmd.cs.value == 0b01:  # RIGHT
                chips_to_write = [1]
                self.cs_right_count += 1
            elif cmd.cs.value == 0b10:  # LEFT
                chips_to_write = [0]
                self.cs_left_count += 1

            # Write to selected chips
            for chip_idx in chips_to_write:
                chip = self.chips[chip_idx]
                if cmd.instr is not None:
                    chip.write_instruction(cmd.instr, cmd.data)
                else:
                    chip.write_data(cmd.data, pc_source=cpu_pc)

        except ValueError:
            # Invalid command, ignore
            pass

    def reset(self) -> None:
        """Reset both chips and all statistics."""
        # Reset each chip (this now includes statistics counters)
        for chip in self.chips:
            chip.reset()

        # Reset chip select counters
        self.cs_both_count = 0
        self.cs_left_count = 0
        self.cs_right_count = 0

    @perf_trace("Display")
    def get_display_buffer(self) -> np.ndarray:
        """Get combined display buffer as numpy array (240x32).

        Returns display buffer compatible with the emulator's expectations.
        """
        # Create 32x240 buffer
        buffer = np.zeros((32, 240), dtype=np.uint8)

        # The PC-E500 uses only 4 pages (32 pixels) of each chip
        # and arranges them in a specific layout

        # Left chip contributes to left side of display
        if self.chips[0].state.on:
            for page in range(4):  # Only 4 pages used
                for col in range(64):
                    byte = self.chips[0].vram[page][col]
                    for bit in range(8):
                        if (byte >> bit) & 1:
                            buffer[page * 8 + bit, col] = 1

        # Right chip contributes to right side
        if self.chips[1].state.on:
            for page in range(4):  # Only 4 pages used
                for col in range(64):
                    byte = self.chips[1].vram[page][col]
                    for bit in range(8):
                        if (byte >> bit) & 1:
                            # Right chip starts at column 120
                            buffer[page * 8 + bit, 120 + col] = 1

        return buffer

    def get_combined_display(self, zoom: int = 2) -> Image.Image:
        """Get combined display using the render_combined_image function."""
        return render_combined_image(self.chips, zoom)

    def save_displays_to_png(
        self, left_filename: str = "lcd_left.png", right_filename: str = "lcd_right.png"
    ) -> None:
        """Save both LCD displays as PNG files."""
        # Get images from chips
        left_img = self.chips[0].render_vram_image(zoom=1)
        right_img = self.chips[1].render_vram_image(zoom=1)

        # Convert to grayscale
        left_img = left_img.convert("L")
        right_img = right_img.convert("L")

        # Save
        left_img.save(left_filename)
        right_img.save(right_filename)

    def set_perfetto_enabled(self, enabled: bool) -> None:
        """Enable/disable Perfetto tracing (no-op in new implementation)."""
        self.perfetto_enabled = enabled

    @property
    def display_on(self) -> List[bool]:
        """Get display on status for both chips."""
        return [chip.state.on for chip in self.chips]

    @property
    def page(self) -> List[int]:
        """Get current page for both chips."""
        return [chip.state.page for chip in self.chips]

    @property
    def column(self) -> List[int]:
        """Get current column (y_address) for both chips."""
        return [chip.state.y_address for chip in self.chips]

    @property
    def width(self) -> int:
        return 240

    @property
    def height(self) -> int:
        return 32

    def get_chip_statistics(self) -> List[Dict[str, int]]:
        """Get statistics for each chip."""
        stats = []
        for i, chip in enumerate(self.chips):
            stats.append(
                {
                    "chip": i,
                    "on": chip.state.on,
                    "instructions": chip.instruction_count,
                    "data_written": chip.data_write_count,
                    "data_read": chip.data_read_count,
                    "on_off_commands": chip.on_off_count,
                    "page": chip.state.page,
                    "column": chip.state.y_address,
                }
            )
        return stats

    def get_pixel_pc_source(self, x: int, y: int) -> Optional[int]:
        """Get the PC source for a specific pixel on the 240x32 display.

        Args:
            x: X coordinate (0-239)
            y: Y coordinate (0-31)

        Returns:
            PC address that last wrote to this pixel, or None
        """
        # Convert y to page
        page = y // 8

        # Determine which chip and column based on PC-E500 layout
        # Left half: right chip (64px) + left chip (56px)
        # Right half: left chip (56px flipped) + right chip (64px flipped)

        if x < 64:
            # Left portion: right chip[0-63]
            chip_idx = 1
            col = x
        elif x < 120:
            # Center-left portion: left chip[0-55]
            chip_idx = 0
            col = x - 64
        elif x < 176:
            # Center-right portion: left chip[0-55] (flipped)
            chip_idx = 0
            col = 55 - (x - 120)  # Flip horizontally
        else:
            # Right portion: right chip[0-63] (flipped)
            chip_idx = 1
            col = 63 - (x - 176)  # Flip horizontally

        # Get the PC source from the appropriate chip
        if 0 <= chip_idx < len(self.chips):
            return self.chips[chip_idx].get_pc_source(page, col)
        return None
