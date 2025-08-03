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
    render_combined_image
)

# Import the controller wrapper for emulator compatibility
from .controller_wrapper import HD61202Controller

# Import visualization functionality
from .lcd_visualization import (
    generate_lcd_image_from_trace,
    HD61202Interpreter as HD61202Viz  # Alias to avoid confusion
)

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
    
    # Visualization
    "generate_lcd_image_from_trace",
]