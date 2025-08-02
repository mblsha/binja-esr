import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    # import marimo as mo
    import sys
    from typing import Optional, List, Dict
    import enum
    import dataclasses
    from PIL import Image, ImageDraw
    return Dict, Image, ImageDraw, List, Optional, dataclasses, enum, sys


@app.cell
def _(Optional, sys):
    try:
        from retrobus_perfetto.proto import perfetto_pb2
    except ImportError:
        print("Error: Could not import 'perfetto_pb2'.", file=sys.stderr)
        print("Please ensure 'retrobus-perfetto' is installed.", file=sys.stderr)
        print("Hint: `pip install -e ./third_party/retrobus-perfetto/py`", file=sys.stderr)
        sys.exit(1)

    def load_perfetto_trace(trace_path: str) -> Optional[perfetto_pb2.Trace]:
        """
        Loads a Perfetto protobuf trace file into a Trace message object.

        Args:
            trace_path: The path to the .perfetto-trace or .pftrace file.

        Returns:
            A populated perfetto_pb2.Trace object, or None on error.
        """
        trace = perfetto_pb2.Trace()
        with open(trace_path, 'rb') as f:
            trace_data = f.read()
            trace.ParseFromString(trace_data)
            return trace

    trace = load_perfetto_trace("../pc-e500.trace")
    return perfetto_pb2, trace


@app.cell
def _(Dict, List, perfetto_pb2):
    def extract_events_from_thread(trace: perfetto_pb2.Trace, target_track_uuid: int) -> List[perfetto_pb2.TracePacket]:
        thread_events = []
        for packet in trace.packet:
            if packet.HasField('track_event'):
                event = packet.track_event
                if event.track_uuid == target_track_uuid:
                    thread_events.append(packet)

        return thread_events

    def find_all_threads(trace: perfetto_pb2.Trace) -> Dict[str, int]:
        found_threads = {}

        # Iterate through all packets to find thread definitions
        for packet in trace.packet:
            # We are only interested in packets that describe tracks
            if not packet.HasField('track_descriptor'):
                continue

            desc = packet.track_descriptor

            # Check if the track descriptor is for a thread
            if desc.HasField('thread'):
                # The 'name' field in the TrackDescriptor is the one displayed in the UI
                thread_name = desc.name or desc.thread.thread_name  # Fallback to internal name
                thread_uuid = desc.uuid
                # found_threads[thread_uuid] = thread_name
                found_threads[thread_name] = thread_uuid

        return found_threads
    return extract_events_from_thread, find_all_threads


@app.cell
def _(
    Optional,
    dataclasses,
    enum,
    extract_events_from_thread,
    find_all_threads,
):
    def extract_annotations(event):
        annotations = {}
        for annotation in event.debug_annotations:
            # Handle different annotation value types
            ann_name = annotation.name if annotation.HasField('name') else ""

            if annotation.HasField('string_value'):
                ann_value = annotation.string_value
            elif annotation.HasField('int_value'):
                ann_value = str(annotation.int_value)
            elif annotation.HasField('uint_value'):
                ann_value = str(annotation.uint_value)
            elif annotation.HasField('bool_value'):
                ann_value = str(annotation.bool_value).lower()
            elif annotation.HasField('double_value'):
                ann_value = str(annotation.double_value)
            elif annotation.HasField('pointer_value'):
                ann_value = f"0x{annotation.pointer_value:X}"
            else:
                ann_value = "<unknown>"

            if ann_name:
                annotations[ann_name] = ann_value
        return annotations


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

    # upper two bytes encode Instruction Type
    class Instruction(enum.Enum):
        ON_OFF = 0b00
        START_LINE = 0b11
        SET_PAGE = 0b10
        SET_Y_ADDRESS = 0b01

    @dataclasses.dataclass
    class HD61202State:
        on: bool = False
        start_line: int = 0 # 0-63
        page: int = 0 # 0-7
        y_address: int = 0 # 0-63

    @dataclasses.dataclass
    class Command:
        cs: ChipSelect
        instr: Optional[Instruction] = None
        data: int = 0

        def __repr__(self):
            return f"Command(cs={self.cs}, instr={self.instr}, data=0x{self.data:02X})"

    def draw_commands(trace):
        commands = []

        threads = find_all_threads(trace)
        for d in extract_events_from_thread(trace, threads['Display']):
            # ann: {'addr': '0x00A000', 'value': '0x3E', 'pc': '0x0F10CC', 'overlay': 'display_memory'}
            ann = extract_annotations(d.track_event)
            addr = int(ann['addr'], 16) & 0xFFF
            rw = ReadWrite(addr & 1)
            assert rw == ReadWrite.WRITE, "Unexpected Read value"
            di = DataInstruction((addr >> 1) & 1)
            cs = ChipSelect((addr >> 2) & 0b11)
            assert cs != ChipSelect.NONE, "Unexpected Chip Select NONE"

            data = int(ann['value'], 16)
            instr = None
            if di == DataInstruction.INSTRUCTION:
                instr = Instruction(data >> 6)
                data = (data >> 2) & 0b111111
                match instr:
                    case Instruction.ON_OFF:
                        data = data & 1
                    case Instruction.SET_PAGE:
                        data = data & 0b111

            commands.append(Command(cs, instr, data))

        return commands
    return ChipSelect, HD61202State, Instruction, draw_commands


@app.cell
def _(HD61202State, Image, ImageDraw, Instruction):
    class HD61202:
        LCD_WIDTH = 64
        LCD_HEIGHT = 8

        def __init__(self):
            self.state = HD61202State()
            self.vram = [[0 for _ in range(self.LCD_WIDTH)] for _ in range(self.LCD_HEIGHT)]

        def reset(self):
            self.state = HD61202State()

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
            self.vram[self.state.page][self.state.y_address] = data
            self.state.y_address = min(self.state.y_address + 1, self.LCD_WIDTH - 1)

        def __repr__(self):
            return f"HD61202(on={self.state.on}, start_line={self.state.start_line}, page={self.state.page}, y_address={self.state.y_address})"

        def vram_image(self, zoom=4):
            off_color = (0, 0, 0)
            on_color = (0, 255, 0)

            img_width = len(self.vram[0]) * zoom
            img_height = len(self.vram) * 6 * zoom
            image = Image.new("RGB", (img_width, img_height), off_color)
            draw = ImageDraw.Draw(image)

            for row in range(len(self.vram)):
                for col in range(len(self.vram[row])):
                    byte = self.vram[row][col]
                    for bit in range(8):
                        pixel_state = (byte >> bit) & 1
                        color = on_color if pixel_state else off_color

                        dx = col
                        dy = row * 8 + bit
                        draw.rectangle(
                            [
                                dx * zoom,
                                dy * zoom,
                                dx * zoom + zoom - 1,
                                dy * zoom + zoom - 1,
                            ],
                            fill=color,
                        )

            return image #, draw
    return (HD61202,)


@app.cell
def _(draw_commands, trace):
    def rle_commands(commands):
        # Run-length encode the commands in tuples (Repeat, Command)
        result = []
        if not commands:
            return result
        current_command = commands[0]
        count = 1
        for command in commands[1:]:
            if command == current_command:
                count += 1
            else:
                result.append((count, current_command))
                current_command = command
                count = 1
        result.append((count, current_command))
        return result


    commands = draw_commands(trace)
    rle_commands(commands)
    return (commands,)


@app.cell
def _(ChipSelect, HD61202, Instruction, commands):
    def draw_lcds(commands):
        lcds = [HD61202(), HD61202()]
        lcd_cs = {
            ChipSelect.BOTH: [lcds[0], lcds[1]],
            ChipSelect.RIGHT: [lcds[1]],
            ChipSelect.LEFT: [lcds[0]],
            ChipSelect.NONE: None
        }

        for c in commands:
            for lcd in lcd_cs[c.cs]:
                if c.instr is not None:
                    # ON_OFF = 0b00
                    # START_LINE = 0b11
                    # SET_PAGE = 0b10
                    # SET_Y_ADDRESS = 0b01
                    if c.instr in [Instruction.ON_OFF, Instruction.START_LINE, Instruction.SET_Y_ADDRESS]:
                        lcd.write_instruction(c.instr, c.data)
                else:
                    if c.data != 0:
                        lcd.write_data(c.data)

        return lcds

    lcds = draw_lcds(commands)
    lcds[0].vram_image(zoom=4), lcds[1].vram_image(zoom=4)
    return


@app.cell
def _(HD61202):
    test_lcd = HD61202()
    test_lcd.write_data(0xFF)
    test_lcd.vram_image(zoom=4)
    return


if __name__ == "__main__":
    app.run()
