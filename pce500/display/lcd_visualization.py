# lcd_visualization.py

import sys
import enum
import dataclasses
import argparse
from typing import Optional, List, Dict

from PIL import Image, ImageDraw

# Attempt to import the Perfetto protobuf definitions
try:
    from retrobus_perfetto.proto import perfetto_pb2
except ImportError:
    print("Error: Could not import 'perfetto_pb2'.", file=sys.stderr)
    print("Please ensure 'retrobus-perfetto' is installed.", file=sys.stderr)
    print("Hint: `pip install -e ./third_party/retrobus-perfetto/py`", file=sys.stderr)
    sys.exit(1)

# --- Data Structures and Enums ---

class ReadWrite(enum.Enum):
    READ = 1
    WRITE = 0

class DataInstruction(enum.Enum):
    INSTRUCTION = 0
    DATA = 1

class ChipSelect(enum.Enum):
    BOTH = 0b00
    RIGHT = 0b01
    LEFT = 0b10
    NONE = 0b11

class Instruction(enum.Enum):
    ON_OFF = 0b00
    START_LINE = 0b11
    SET_PAGE = 0b10
    SET_Y_ADDRESS = 0b01

@dataclasses.dataclass
class HD61202State:
    on: bool = False
    start_line: int = 0  # 0-63
    page: int = 0        # 0-7
    y_address: int = 0   # 0-63

@dataclasses.dataclass
class Command:
    cs: ChipSelect
    instr: Optional[Instruction] = None
    data: int = 0

    def __repr__(self):
        return f"Command(cs={self.cs}, instr={self.instr}, data=0x{self.data:02X})"

# --- Perfetto Trace Parsing ---

def load_perfetto_trace(trace_path: str) -> Optional[perfetto_pb2.Trace]:
    """Loads a Perfetto protobuf trace file into a Trace message object."""
    trace = perfetto_pb2.Trace()
    try:
        with open(trace_path, 'rb') as f:
            trace_data = f.read()
            trace.ParseFromString(trace_data)
            return trace
    except FileNotFoundError:
        print(f"Error: Trace file not found at '{trace_path}'", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error parsing trace file: {e}", file=sys.stderr)
        return None

def find_all_threads(trace: perfetto_pb2.Trace) -> Dict[str, int]:
    """Finds all thread descriptors in a trace and maps their names to UUIDs."""
    found_threads = {}
    for packet in trace.packet:
        if packet.HasField('track_descriptor'):
            desc = packet.track_descriptor
            if desc.HasField('thread'):
                thread_name = desc.name or desc.thread.thread_name
                if thread_name:
                    found_threads[thread_name] = desc.uuid
    return found_threads

def extract_events_from_thread(trace: perfetto_pb2.Trace, target_track_uuid: int) -> List[perfetto_pb2.TracePacket]:
    """Extracts all trace packets for a given track UUID."""
    return [
        packet for packet in trace.packet
        if packet.HasField('track_event') and packet.track_event.track_uuid == target_track_uuid
    ]

def extract_annotations(event: perfetto_pb2.TrackEvent) -> Dict[str, str]:
    """Extracts debug annotations from a track event into a dictionary."""
    annotations = {}
    for annotation in event.debug_annotations:
        ann_name = annotation.name if annotation.HasField('name') else ""
        if not ann_name:
            continue

        if annotation.HasField('string_value'):
            ann_value = annotation.string_value
        elif annotation.HasField('int_value'):
            ann_value = str(annotation.int_value)
        else:
            ann_value = "<unknown>" # Simplified for brevity
        annotations[ann_name] = ann_value
    return annotations

def parse_command(addr: int, value: int) -> Command:
    """Parses a memory write address and value into an LCD command."""
    addr_hi = addr & 0xF000
    if addr_hi not in [0xA000, 0x2000]:
        raise ValueError(f"Unexpected address high bits: {hex(addr_hi)}")

    addr_lo = addr & 0xFFF
    if not (addr_lo & 1) == ReadWrite.WRITE.value:
        raise ValueError("Command parsing only supports write operations.")

    di = DataInstruction((addr_lo >> 1) & 1)
    cs = ChipSelect((addr_lo >> 2) & 0b11)
    if cs == ChipSelect.NONE:
        raise ValueError("Unexpected Chip Select value: NONE")

    data = value
    instr = None
    if di == DataInstruction.INSTRUCTION:
        instr = Instruction(data >> 6)
        data = data & 0b00111111
        if instr == Instruction.ON_OFF:
            data &= 1
        elif instr == Instruction.SET_PAGE:
            data &= 0b111
    
    return Command(cs, instr, data)

# --- LCD Emulation and Rendering ---

class HD61202Interpreter:
    """Emulates the state and VRAM of a single HD61202 LCD controller."""
    LCD_WIDTH_PIXELS = 64
    LCD_PAGES = 8
    PAGE_HEIGHT_PIXELS = 8
    LCD_HEIGHT_PIXELS = LCD_PAGES * PAGE_HEIGHT_PIXELS

    def __init__(self):
        self.state = HD61202State()
        self.vram = [[0] * self.LCD_WIDTH_PIXELS for _ in range(self.LCD_PAGES)]

    def write_instruction(self, instr: Instruction, data: int):
        if instr == Instruction.ON_OFF:
            self.state.on = bool(data)
        elif instr == Instruction.START_LINE:
            self.state.start_line = data
        elif instr == Instruction.SET_PAGE:
            self.state.page = data
        elif instr == Instruction.SET_Y_ADDRESS:
            self.state.y_address = data
        else:
            raise ValueError(f"Unknown instruction: {instr}")

    def write_data(self, data: int):
        if 0 <= self.state.page < self.LCD_PAGES and 0 <= self.state.y_address < self.LCD_WIDTH_PIXELS:
            self.vram[self.state.page][self.state.y_address] = data
            self.state.y_address = (self.state.y_address + 1) % self.LCD_WIDTH_PIXELS

    def render_vram_image(self, zoom: int = 1) -> Image.Image:
        """Renders the controller's VRAM to a PIL Image."""
        img = Image.new("1", (self.LCD_WIDTH_PIXELS, self.LCD_HEIGHT_PIXELS))
        pixels = img.load()
        for page in range(self.LCD_PAGES):
            for y_addr in range(self.LCD_WIDTH_PIXELS):
                byte = self.vram[page][y_addr]
                for bit in range(self.PAGE_HEIGHT_PIXELS):
                    if (byte >> bit) & 1:
                        pixels[y_addr, page * self.PAGE_HEIGHT_PIXELS + bit] = 1
        
        if zoom > 1:
            return img.resize((img.width * zoom, img.height * zoom), Image.NEAREST)
        return img

def _render_combined_image(lcds: List[HD61202Interpreter], zoom: int) -> Image.Image:
    """Stitches images from two LCD controllers into the final wide display."""
    images = [lcd.render_vram_image(zoom=1) for lcd in lcds]
    # This layout logic is specific to the device's display configuration
    lcd0w, lcd1w = 64, 56
    height = HD61202Interpreter.LCD_HEIGHT_PIXELS // 2

    combined_width = lcd0w * 2 + lcd1w * 2
    image = Image.new("RGB", (combined_width, height), "black")
    
    # Page 206 of a manual likely explains this layout
    image.paste(images[0].crop((0, 0, lcd0w, height)), (0, 0))
    image.paste(images[1].crop((0, 0, lcd1w, height)), (lcd0w, 0))
    # The right half is a flipped view of the left VRAMs
    image.paste(images[1].crop((0, height, lcd1w, height*2)).transpose(Image.FLIP_LEFT_RIGHT), (lcd0w + lcd1w, 0))
    image.paste(images[0].crop((0, height, lcd0w, height*2)).transpose(Image.FLIP_LEFT_RIGHT), (lcd0w + lcd1w + lcd1w, 0))
    
    return image.resize((image.width * zoom, image.height * zoom), Image.NEAREST)

# --- Main Library Function ---

def generate_lcd_image_from_trace(trace_path: str, zoom: int = 2) -> Optional[Image.Image]:
    """
    Processes a Perfetto trace file to generate an image of the final LCD state.

    Args:
        trace_path: Path to the .pftrace file.
        zoom: Integer factor to scale the output image.

    Returns:
        A PIL Image object of the final display, or None on error.
    """
    trace = load_perfetto_trace(trace_path)
    if not trace:
        return None

    threads = find_all_threads(trace)
    if 'Display' not in threads:
        print("Error: 'Display' thread not found in trace.", file=sys.stderr)
        return None
    
    display_uuid = threads['Display']
    events = extract_events_from_thread(trace, display_uuid)

    commands = []
    for packet in events:
        ann = extract_annotations(packet.track_event)
        if 'addr' in ann and 'value' in ann:
            try:
                cmd = parse_command(int(ann['addr'], 16), int(ann['value'], 16))
                commands.append(cmd)
            except ValueError as e:
                # Silently ignore parse errors for robustness
                pass

    lcds = [HD61202Interpreter(), HD61202Interpreter()]
    lcd_cs_map = {
        ChipSelect.BOTH: [lcds[0], lcds[1]],
        ChipSelect.RIGHT: [lcds[0]],
        ChipSelect.LEFT: [lcds[1]],
    }

    for c in commands:
        targets = lcd_cs_map.get(c.cs)
        if targets:
            for lcd in targets:
                if c.instr is not None:
                    lcd.write_instruction(c.instr, c.data)
                else:
                    lcd.write_data(c.data)

    # Convert to a display-friendly color scheme before returning
    final_image_bw = _render_combined_image(lcds, zoom)
    final_image_color = final_image_bw.convert("RGB")
    
    pixels = final_image_color.load()
    on_color = (50, 255, 100) # Green
    off_color = (20, 30, 40)  # Dark blue/gray
    for x in range(final_image_color.width):
        for y in range(final_image_color.height):
            if pixels[x, y] == (255, 255, 255):
                pixels[x, y] = on_color
            else:
                pixels[x, y] = off_color

    return final_image_color

# --- Command-Line Interface ---
def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Convert a Perfetto trace with LCD data to a PNG image."
    )
    parser.add_argument(
        "trace_file",
        help="Path to the input Perfetto trace file (.pftrace)."
    )
    parser.add_argument(
        "output_image",
        help="Path to save the output PNG image."
    )
    parser.add_argument(
        "--zoom",
        type=int,
        default=2,
        help="Zoom factor for the output image (default: 2)."
    )
    args = parser.parse_args()

    print(f"Processing trace file: {args.trace_file}")
    image = generate_lcd_image_from_trace(args.trace_file, zoom=args.zoom)

    if image:
        image.save(args.output_image)
        print(f"Successfully saved LCD image to: {args.output_image}")
    else:
        print("Failed to generate image.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()