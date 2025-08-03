import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import sys
    from typing import Optional, List, Dict
    import enum
    import dataclasses
    from PIL import Image, ImageDraw
    return Dict, Image, ImageDraw, List, Optional, dataclasses, enum, mo, sys


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
def _(find_all_threads, mo, trace):
    threads = find_all_threads(trace)
    selected_thread = mo.ui.dropdown(threads)
    selected_thread
    return (selected_thread,)


@app.cell
def _(extract_annotations, extract_events_from_thread, selected_thread, trace):
    extract_events_from_thread(trace, selected_thread.value)

    import collections
    write_addrs = collections.defaultdict(int)
    for e in extract_events_from_thread(trace, selected_thread.value):
        ann = extract_annotations(e.track_event)
        addr = int(ann['addr'], 16)
        write_addrs[addr] += 1
        # break

    for a in sorted(list(write_addrs)):
        print(f"{hex(a)}: {write_addrs[a]} writes")
    return


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
def _(List, Optional, dataclasses, enum):
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

    # upper two bytes encode Instruction Type (DB7-DB6)
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

    def parse_command(addr, value):
        # ann: {'addr': '0x00A000', 'value': '0x3E', 'pc': '0x0F10CC', 'overlay': 'display_memory'}
        # Addr Bit Encoding: 7 6 5 4 3 2 1 0
        #                                  -
        #                                  rw
        #                                -
        #                                di
        #                            ---
        #                            cs

        addr_hi = addr & 0xF000
        assert addr_hi in [0xA000, 0x2000], f"Unexpected address high bits: {hex(addr_hi)}"
        addr = addr & 0xFFF
        rw = ReadWrite(addr & 1)
        assert rw == ReadWrite.WRITE, "Unexpected Read value"
        di = DataInstruction((addr >> 1) & 1)
        cs = ChipSelect((addr >> 2) & 0b11)
        assert cs != ChipSelect.NONE, "Unexpected Chip Select NONE"

        data = value
        instr = None
        if di == DataInstruction.INSTRUCTION:
            instr = Instruction(data >> 6)
            data = data & 0b111111
            match instr:
                case Instruction.ON_OFF:
                    data = data & 1
                case Instruction.SET_PAGE:
                    data = data & 0b111

        return Command(cs, instr, data)

    @dataclasses.dataclass
    class TestCase:
        test_id: str
        in_addr: int
        in_data: int
        out_cs: ChipSelect = ChipSelect.NONE
        out_instr: Optional[Instruction] = None
        out_data: int = 0

    test_cases: List[TestCase] = [
        TestCase(
            test_id="on_both",
            in_addr=0x2000,
            in_data=0b00111111,
            out_cs=ChipSelect.BOTH,
            out_instr=Instruction.ON_OFF,
            out_data=0x01
        ),
        TestCase(
            test_id="off_both",
            in_addr=0x2000,
            in_data=0b00111110,
            out_cs=ChipSelect.BOTH,
            out_instr=Instruction.ON_OFF,
            out_data=0x00
        ),
        TestCase(
            test_id="on_left",
            in_addr=0x2008,
            in_data=0x3F,
            out_cs=ChipSelect.LEFT,
            out_instr=Instruction.ON_OFF,
            out_data=0x01
        ),
        TestCase(
            test_id="on_right",
            in_addr=0x2004,
            in_data=0x3F,
            out_cs=ChipSelect.RIGHT,
            out_instr=Instruction.ON_OFF,
            out_data=0x01
        ),
    ]
    return (
        ChipSelect,
        HD61202State,
        Instruction,
        extract_annotations,
        parse_command,
        test_cases,
    )


@app.cell
def _(parse_command, test_cases: "List[TestCase]"):
    def run_test_cases():
        for t in test_cases:
            command = parse_command(t.in_addr, t.in_data)
            assert command.cs == t.out_cs, f"{t.test_id} failed: expected CS {t.out_cs}, got {command.cs}"
            assert command.instr == t.out_instr, f"{t.test_id} failed: expected Instr {t.out_instr}, got {command.instr}"
            assert command.data == t.out_data, f"{t.test_id} failed: expected Data {t.out_data}, got {command.data}"

    run_test_cases()

    return


@app.cell
def _(
    extract_annotations,
    extract_events_from_thread,
    find_all_threads,
    parse_command,
):
    def draw_commands(trace):
        commands = []

        threads = find_all_threads(trace)
        for d in extract_events_from_thread(trace, threads['Display']):
            ann = extract_annotations(d.track_event)
            command = parse_command(int(ann['addr'], 16), int(ann['value'], 16))
            commands.append(command)

        return commands
    return (draw_commands,)


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
def _(HD61202State, Image, ImageDraw, Instruction):
    class HD61202Interpreter:
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
            img_height = len(self.vram) * 8 * zoom
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
    return (HD61202Interpreter,)


@app.cell
def _(ChipSelect, HD61202Interpreter, commands):
    def draw_lcds(commands):
        lcds = [HD61202Interpreter(), HD61202Interpreter()]
        lcd_cs = {
            ChipSelect.BOTH: [lcds[0], lcds[1]],
            ChipSelect.RIGHT: [lcds[0]],
            ChipSelect.LEFT: [lcds[1]],
            ChipSelect.NONE: None
        }

        for c in commands:
            for lcd in lcd_cs[c.cs]:
                if c.instr is not None:
                    # ON_OFF = 0b00
                    # START_LINE = 0b11
                    # SET_PAGE = 0b10
                    # SET_Y_ADDRESS = 0b01
                    # if c.instr in [Instruction.ON_OFF, Instruction.START_LINE, Instruction.SET_Y_ADDRESS]:
                    lcd.write_instruction(c.instr, c.data)
                    pass
                else:
                    if c.data != 0:
                        lcd.write_data(c.data)

        return lcds

    lcds = draw_lcds(commands)
    lcds[0].vram_image(zoom=4), lcds[1].vram_image(zoom=4)
    return


if __name__ == "__main__":
    app.run()
