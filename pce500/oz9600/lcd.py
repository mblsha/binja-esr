"""Firmware-derived LH1553F controller, matching the Rust bus controller."""

from pce500.display.lcd_frame import LcdFrame

STORAGE_WIDTH, HEIGHT, MAIN_WIDTH = 336, 240, 320
ACCESS_REQUEST, ACCESS_READY, SAMPLE_SYNC, FRAME_PHASE = 1, 2, 4, 8
FRAME_HALF_PERIOD = 32768


class LcdController:
    def __init__(self):
        descriptor = bytes([0, 0x7C, 0, 0x7E, 0, 0x7C, 0, 0x7C])
        self.registers = bytearray(32)
        self.registers[24:] = descriptor
        self.windows = [bytearray(descriptor) for _ in range(16)]
        self.pixels = bytearray(STORAGE_WIDTH * HEIGHT)
        self.read_latch = 0
        self.data_writes = self.data_reads = self.block_operations = 0

    def window_descriptor(self, index):
        return bytes(self.windows[index])

    @staticmethod
    def coordinate_high_readback(value, field_mask):
        upper, overflow = 0x7F & ~field_mask, field_mask + 1
        if value & 0x80 == 0:
            return (value & field_mask) | (overflow if value & upper else 0)
        if value & upper == upper:
            return (value | ~field_mask) & 255
        return ((value & ~overflow) | ~(field_mask | overflow)) & 255

    def word(self, offset):
        return int.from_bytes(self.registers[offset : offset + 2], "little")

    def set_word(self, offset, value):
        self.registers[offset : offset + 2] = (value & 65535).to_bytes(2, "little")

    def pixel(self, x, y):
        return (
            0 <= x < STORAGE_WIDTH
            and 0 <= y < HEIGHT
            and bool(self.pixels[y * STORAGE_WIDTH + x])
        )

    def bit_location(self, x, y, bit):
        return (x, y + bit) if self.registers[8] & 0x80 else (x + bit, y)

    def read_pixels(self, x, y):
        value = sum(
            (0x80 >> bit)
            for bit in range(8)
            if self.pixel(*self.bit_location(x, y, bit))
        )
        return value & self.registers[3]

    @staticmethod
    def window_coordinate(low, high, mask):
        high = (high | ~mask) & 255 if high & 0x80 else high & mask
        word = low | (high << 8)
        return word - 65536 if word & 32768 else word

    def permits_write(self, x, y):
        if not 0 <= x < STORAGE_WIDTH or not 0 <= y < HEIGHT:
            return False
        window = self.windows[self.registers[9]]
        if window[7] & 2:
            return True
        coords = [
            self.window_coordinate(window[n], window[n + 1], 3 if n in (0, 4) else 1)
            for n in (0, 2, 4, 6)
        ]
        return coords[0] <= x <= coords[2] and coords[1] <= y <= coords[3]

    def write_pixels(self, x, y, value):
        for bit in range(8):
            mask = 0x80 >> bit
            px, py = self.bit_location(x, y, bit)
            if not self.registers[3] & mask or not self.permits_write(px, py):
                continue
            index = py * STORAGE_WIDTH + px
            destination, source = self.pixels[index], int(bool(value & mask))
            combined = (
                source,
                destination ^ source,
                destination | source,
                # Scrapbook live mode 3 matches the saved software OR stroke.
                # Mode 2 remains an unqualified hypothesis.
                destination | source,
            )[self.registers[8] & 3]
            self.pixels[index] = combined ^ ((self.registers[8] >> 2) & 1)

    def increment_axis(self, coordinate, nibble):
        if not nibble:
            return
        step = 8 if nibble & 8 else 1
        direction = nibble & 3
        if direction not in (1, 2):
            raise ValueError(f"unqualified LCD increment nibble {nibble:X}")
        self.set_word(
            coordinate, self.word(coordinate) + (step if direction == 1 else -step)
        )

    def check_increments(self, read):
        increments = self.registers[4 if read else 5]
        for nibble in (increments >> 4, increments & 15):
            if nibble and nibble & 3 not in (1, 2):
                raise ValueError(f"unqualified LCD increment nibble {nibble:X}")

    def advance(self, read):
        increments = self.registers[4 if read else 5]
        coordinate = 20 if read else 16
        self.increment_axis(coordinate, increments >> 4)
        self.increment_axis(coordinate + 2, increments & 15)

    def peek(self, offset, cycle):
        if offset == 1:
            phase = (cycle % FRAME_HALF_PERIOD * HEIGHT) % FRAME_HALF_PERIOD
            return (
                (ACCESS_READY if self.registers[14] & ACCESS_REQUEST else 0)
                | (SAMPLE_SYNC if phase < FRAME_HALF_PERIOD // 2 else 0)
                | (FRAME_PHASE if cycle // FRAME_HALF_PERIOD & 1 else 0)
            )
        if offset == 2:
            return self.read_latch
        if offset in (17, 21, 19, 23):
            return self.coordinate_high_readback(
                self.registers[offset], 3 if offset in (17, 21) else 1
            )
        return self.registers[offset]

    def read(self, offset, cycle):
        if offset == 0:
            self.check_increments(True)
            self.check_increments(False)
            for _ in range(self.word(6)):
                value = self.read_pixels(self.word(20), self.word(22))
                self.write_pixels(self.word(16), self.word(18), value)
                self.advance(True)
                self.advance(False)
            self.block_operations += 1
            return 0
        if offset == 2:
            self.check_increments(True)
            previous = self.read_latch
            self.read_latch = self.read_pixels(self.word(20), self.word(22))
            self.advance(True)
            self.data_reads += 1
            return previous
        return self.peek(offset, cycle)

    def write(self, offset, value):
        if offset in (0, 2):
            self.check_increments(False)
            for _ in range(self.word(6) if offset == 0 else 1):
                self.write_pixels(self.word(16), self.word(18), value)
                self.advance(False)
            if offset == 0:
                self.block_operations += 1
            else:
                self.read_latch = value
                self.data_writes += 1
        elif offset == 7:
            self.registers[offset] = value & 3
        elif offset == 9:
            self.registers[9] = value & 15
            self.registers[24:] = self.windows[value & 15]
        elif 24 <= offset <= 31:
            fixed = 0x7C if offset in (25, 29, 31) else 0x7E if offset == 27 else 0
            self.windows[self.registers[9]][offset - 24] = value | fixed
            self.registers[offset] = value | fixed
        else:
            self.registers[offset] = value

    def matrix_frame(self):
        return LcdFrame(STORAGE_WIDTH, HEIGHT, bytes(self.pixels))

    def pbm(self):
        return self.matrix_frame().pbm()
