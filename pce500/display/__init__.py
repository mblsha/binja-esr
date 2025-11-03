"""Display subsystem for PC-E500 emulator."""

from .hd61202 import (
    HD61202,
    HD61202Interpreter,
    HD61202State,
    Instruction,
    ChipSelect,
    DataInstruction,
    ReadWrite,
    Command,
    parse_command,
    render_combined_image,
)

# Import the controller wrapper for emulator compatibility
from .controller_wrapper import HD61202Controller
from .font import glyph_bitmap, glyph_columns, text_columns
from .pipeline import (
    LCDChipSnapshot,
    LCDOperation,
    LCDPipeline,
    LCDSnapshot,
    replay_operations,
)

# Import visualization functionality
from .lcd_visualization import generate_lcd_image_from_trace
from .text_decoder import decode_display_text

__all__ = [
    # Controller for emulator
    "HD61202Controller",
    # Core HD61202 classes
    "HD61202",
    "HD61202Interpreter",
    "HD61202State",
    # Enums
    "Instruction",
    "ChipSelect",
    "DataInstruction",
    "ReadWrite",
    # Data structures
    "Command",
    # Helper functions
    "parse_command",
    "render_combined_image",
    "glyph_columns",
    "glyph_bitmap",
    "text_columns",
    "LCDPipeline",
    "LCDSnapshot",
    "LCDChipSnapshot",
    "LCDOperation",
    "replay_operations",
    # Visualization
    "generate_lcd_image_from_trace",
    # Text decoding
    "decode_display_text",
]
