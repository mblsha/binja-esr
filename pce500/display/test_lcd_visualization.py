# test_lcd_visualization.py

import pytest
from pathlib import Path
from PIL import Image

# Import the library to be tested
from . import lcd_visualization as lv
from .hd61202 import Instruction, ChipSelect


# --- Test Cases for command parsing ---
@pytest.mark.parametrize(
    "test_id, in_addr, in_data, out_cs, out_instr, out_data",
    [
        ("on_both", 0x2000, 0b00111111, ChipSelect.BOTH, Instruction.ON_OFF, 0x01),
        ("off_both", 0x2000, 0b00111110, ChipSelect.BOTH, Instruction.ON_OFF, 0x00),
        ("on_left", 0x2008, 0x3F, ChipSelect.LEFT, Instruction.ON_OFF, 0x01),
        ("on_right", 0x2004, 0x3F, ChipSelect.RIGHT, Instruction.ON_OFF, 0x01),
        (
            "set_page_2_left",
            0xA008,
            0b10111010,
            ChipSelect.LEFT,
            Instruction.SET_PAGE,
            2,
        ),
        (
            "set_y_addr_40_right",
            0xA004,
            0b01101000,
            ChipSelect.RIGHT,
            Instruction.SET_Y_ADDRESS,
            40,
        ),
        (
            "set_start_line_15_both",
            0x2000,
            0b11001111,
            ChipSelect.BOTH,
            Instruction.START_LINE,
            15,
        ),
        ("write_data_AA", 0x2002, 0xAA, ChipSelect.BOTH, None, 0xAA),
    ],
)
def test_parse_command(test_id, in_addr, in_data, out_cs, out_instr, out_data):
    """Tests the parse_command function with various inputs."""
    from .hd61202 import parse_command

    command = parse_command(in_addr, in_data)
    assert command.cs == out_cs, f"{test_id} failed: CS mismatch"
    assert command.instr == out_instr, f"{test_id} failed: Instruction mismatch"
    assert command.data == out_data, f"{test_id} failed: Data mismatch"


def test_parse_command_invalid_input():
    """Tests that parse_command raises errors for invalid inputs."""
    from .hd61202 import parse_command

    # Invalid Chip Select (NONE)
    with pytest.raises(ValueError):
        parse_command(0x200C, 0x00)
    # Read operation instead of write
    with pytest.raises(ValueError):
        parse_command(0x2001, 0x00)


def create_dummy_trace_file(path: Path) -> Path:
    """Creates a small, valid Perfetto trace file for testing."""
    trace = lv.perfetto_pb2.Trace()

    # 1. Define the 'Display' thread
    packet_thread = trace.packet.add()
    desc = packet_thread.track_descriptor
    desc.uuid = 123
    desc.name = "Display"
    desc.thread.pid = 1
    desc.thread.tid = 2
    desc.thread.thread_name = "DisplayThread"

    # 2. Add an event: Turn display ON for BOTH controllers
    packet_on = trace.packet.add()
    packet_on.track_event.track_uuid = 123
    # Add annotation for address: 0x2002 (Instruction, Both)
    ann_addr_on = packet_on.track_event.debug_annotations.add()
    ann_addr_on.name = "addr"
    ann_addr_on.string_value = "0x2002"
    # Add annotation for value: 0x3F (ON/OFF instruction, data=1)
    ann_val_on = packet_on.track_event.debug_annotations.add()
    ann_val_on.name = "value"
    ann_val_on.string_value = "0x3F"

    # 3. Add an event: Write data to the VRAM
    packet_data = trace.packet.add()
    packet_data.track_event.track_uuid = 123
    # Add annotation for address: 0x2000 (Data, Both)
    ann_addr_data = packet_data.track_event.debug_annotations.add()
    ann_addr_data.name = "addr"
    ann_addr_data.string_value = "0x2000"
    # Add annotation for value: 0xFF (all pixels in a column on)
    ann_val_data = packet_data.track_event.debug_annotations.add()
    ann_val_data.name = "value"
    ann_val_data.string_value = "0xFF"

    trace_file = path / "dummy.pftrace"
    with open(trace_file, "wb") as f:
        f.write(trace.SerializeToString())

    return trace_file


def test_generate_lcd_image_from_trace(tmp_path: Path):
    """
    Integration test: verifies the full conversion from a trace file to a PNG.
    """
    # 1. Create a dummy trace file in a temporary directory
    dummy_trace_path = create_dummy_trace_file(tmp_path)
    assert dummy_trace_path.exists()

    # 2. Run the main library function
    image = lv.generate_lcd_image_from_trace(str(dummy_trace_path))

    # 3. Verify the output
    assert image is not None
    assert isinstance(image, Image.Image)
    assert image.width > 0
    assert image.height > 0

    # 4. Save the image to a file and verify it was written
    output_png_path = tmp_path / "output.png"
    image.save(output_png_path)
    assert output_png_path.exists()
    assert output_png_path.stat().st_size > 0

    # 5. Verify the image content is readable
    try:
        reopened_image = Image.open(output_png_path)
        assert reopened_image.format == "PNG"
    except Exception as e:
        pytest.fail(f"Could not reopen the generated PNG file: {e}")
