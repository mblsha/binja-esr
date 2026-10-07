"""Raw ten-bit contact and held two-byte ADC sample; physical timing pending."""


class Tablet:
    def __init__(self):
        self.x = self.y = self.conversion_control = self.drive_control = (
            self.data_reads
        ) = 0
        self.pressed = self.low_part_next = False
        self.latched_sample = 0

    def set_contact(self, x, y, pressed):
        if not 0 <= x <= 1023 or not 0 <= y <= 1023:
            raise ValueError("tablet ADC samples must be ten-bit values (0..1023)")
        edge = pressed and not self.pressed
        self.x, self.y, self.pressed = x, y, pressed
        return edge

    def write_conversion_control(self, value):
        if value not in (6, 7):
            raise ValueError(f"unimplemented tablet ADC conversion control {value:02X}")
        self.conversion_control = value
        self.low_part_next = False

    def analog_sample(self):
        if not self.pressed:
            return 0
        return (
            self.x
            if (self.drive_control, self.conversion_control & 1) == (0xA8, 0)
            else self.y
            if (self.drive_control, self.conversion_control & 1) == (0x8A, 1)
            else 1023
        )

    def peek_data(self):
        return (
            self.latched_sample & 3 if self.low_part_next else self.analog_sample() >> 2
        )

    def read_data(self):
        if self.conversion_control not in (6, 7):
            raise ValueError("tablet ADC read before supported conversion selection")
        value = self.peek_data()
        if not self.low_part_next:
            self.latched_sample = self.analog_sample()
        self.low_part_next = not self.low_part_next
        self.data_reads += 1
        return value
