# lcd_visualization.py
# Perfetto trace visualization for HD61202 LCD controllers

import sys
import argparse
from typing import Optional, List, Dict

from PIL import Image

# Import the HD61202 components from the main module
from .hd61202 import (
    HD61202 as HD61202Interpreter,
    ChipSelect,
    parse_command,
    render_combined_image,
)

from retrobus_perfetto.proto import perfetto_pb2

# --- Perfetto Trace Parsing ---


def load_perfetto_trace(trace_path: str) -> Optional["perfetto_pb2.Trace"]:
    """Loads a Perfetto protobuf trace file into a Trace message object."""
    trace = perfetto_pb2.Trace()
    try:
        with open(trace_path, "rb") as f:
            trace_data = f.read()
            trace.ParseFromString(trace_data)
            return trace
    except FileNotFoundError:
        print(f"Error: Trace file not found at '{trace_path}'", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error parsing trace file: {e}", file=sys.stderr)
        return None


def find_all_threads(trace: "perfetto_pb2.Trace") -> Dict[str, int]:
    """Finds all thread descriptors in a trace and maps their names to UUIDs."""
    found_threads = {}
    for packet in trace.packet:
        if packet.HasField("track_descriptor"):
            desc = packet.track_descriptor
            if desc.HasField("thread"):
                thread_name = desc.name or desc.thread.thread_name
                if thread_name:
                    found_threads[thread_name] = desc.uuid
    return found_threads


def extract_events_from_thread(
    trace: "perfetto_pb2.Trace", target_track_uuid: int
) -> List["perfetto_pb2.TracePacket"]:
    """Extracts all trace packets for a given track UUID."""
    return [
        packet
        for packet in trace.packet
        if packet.HasField("track_event")
        and packet.track_event.track_uuid == target_track_uuid
    ]


def extract_annotations(event: "perfetto_pb2.TrackEvent") -> Dict[str, str]:
    """Extracts debug annotations from a track event into a dictionary."""
    annotations = {}
    for annotation in event.debug_annotations:
        ann_name = annotation.name if annotation.HasField("name") else ""
        if not ann_name:
            continue

        if annotation.HasField("string_value"):
            ann_value = annotation.string_value
        elif annotation.HasField("int_value"):
            ann_value = str(annotation.int_value)
        else:
            ann_value = "<unknown>"  # Simplified for brevity
        annotations[ann_name] = ann_value
    return annotations


# --- Main Library Function ---


def generate_lcd_image_from_trace(
    trace_path: str, zoom: int = 2
) -> Optional[Image.Image]:
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
    if "Display" not in threads:
        print("Error: 'Display' thread not found in trace.", file=sys.stderr)
        return None

    display_uuid = threads["Display"]
    events = extract_events_from_thread(trace, display_uuid)

    commands = []
    for packet in events:
        ann = extract_annotations(packet.track_event)
        if "addr" in ann and "value" in ann:
            try:
                cmd = parse_command(int(ann["addr"], 16), int(ann["value"], 16))
                commands.append(cmd)
            except ValueError:
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
    final_image_bw = render_combined_image(lcds, zoom)
    final_image_color = final_image_bw.convert("RGB")

    pixels = final_image_color.load()
    on_color = (50, 255, 100)  # Green
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
        "trace_file", help="Path to the input Perfetto trace file (.pftrace)."
    )
    parser.add_argument("output_image", help="Path to save the output PNG image.")
    parser.add_argument(
        "--zoom",
        type=int,
        default=2,
        help="Zoom factor for the output image (default: 2).",
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
