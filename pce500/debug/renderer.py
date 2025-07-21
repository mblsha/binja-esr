"""Display rendering utilities for PC-E500 emulator."""

from typing import Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ..display import LCDController


class DisplayRenderer:
    """Renders LCD display to images for debugging."""
    
    def __init__(self, scale: int = 4, 
                 bg_color: Tuple[int, int, int] = (200, 220, 200),
                 fg_color: Tuple[int, int, int] = (20, 30, 20)):
        self.scale = scale
        self.bg_color = bg_color
        self.fg_color = fg_color
        
    def render_display(self, controller: LCDController) -> Image.Image:
        """Render LCD controller display to PIL Image."""
        # Get display buffer
        buffer = controller.get_display_buffer()
        
        # Create image with LCD-like colors
        width = controller.width * self.scale
        height = controller.height * self.scale
        img = Image.new('RGB', (width, height), self.bg_color)
        draw = ImageDraw.Draw(img)
        
        # Draw pixels
        for y in range(controller.height):
            for x in range(controller.width):
                if buffer[y, x]:
                    # Draw scaled pixel
                    x1 = x * self.scale
                    y1 = y * self.scale
                    x2 = x1 + self.scale - 1
                    y2 = y1 + self.scale - 1
                    draw.rectangle([x1, y1, x2, y2], fill=self.fg_color)
        
        return img
    
    def render_dual_display(self, main_lcd: LCDController, 
                           sub_lcd: LCDController,
                           gap: int = 20) -> Image.Image:
        """Render both LCD displays in a single image."""
        # Render individual displays
        main_img = self.render_display(main_lcd)
        sub_img = self.render_display(sub_lcd)
        
        # Calculate combined image size
        total_width = max(main_img.width, sub_img.width)
        total_height = main_img.height + gap + sub_img.height
        
        # Create combined image
        combined = Image.new('RGB', (total_width, total_height), (240, 240, 240))
        
        # Paste main display at top
        main_x = (total_width - main_img.width) // 2
        combined.paste(main_img, (main_x, 0))
        
        # Paste sub display below with gap
        sub_x = (total_width - sub_img.width) // 2
        combined.paste(sub_img, (sub_x, main_img.height + gap))
        
        return combined
    
    def save_display(self, controller: LCDController, filename: str) -> None:
        """Save display to image file."""
        img = self.render_display(controller)
        img.save(filename)
    
    def create_debug_image(self, main_lcd: LCDController,
                          sub_lcd: LCDController,
                          cpu_state: dict,
                          filename: str) -> None:
        """Create a debug image with displays and CPU state."""
        # Render displays
        display_img = self.render_dual_display(main_lcd, sub_lcd)
        
        # Add border and space for text
        border = 20
        text_height = 150
        final_width = display_img.width + 2 * border
        final_height = display_img.height + 2 * border + text_height
        
        final_img = Image.new('RGB', (final_width, final_height), (255, 255, 255))
        final_img.paste(display_img, (border, border))
        
        # Add CPU state text
        draw = ImageDraw.Draw(final_img)
        text_y = display_img.height + border + 10
        
        # Try to use a monospace font
        try:
            font = ImageFont.truetype("Courier", 12)
        except:
            font = ImageFont.load_default()
        
        # Format CPU state
        lines = [
            f"PC: {cpu_state['pc']:06X}  Cycles: {cpu_state['cycles']}",
            f"A: {cpu_state['a']:02X}  B: {cpu_state['b']:02X}  BA: {cpu_state['ba']:04X}  I: {cpu_state['i']:02X}",
            f"X: {cpu_state['x']:06X}  Y: {cpu_state['y']:06X}",
            f"U: {cpu_state['u']:06X}  S: {cpu_state['s']:06X}",
            f"Flags: Z={int(cpu_state['flags']['z'])} C={int(cpu_state['flags']['c'])}"
        ]
        
        for i, line in enumerate(lines):
            draw.text((border, text_y + i * 20), line, fill=(0, 0, 0), font=font)
        
        final_img.save(filename)