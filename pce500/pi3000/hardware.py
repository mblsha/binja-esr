"""PI logical bus reference, mirrored by the Rust CoreRuntime adapter.

Fresh RAM, empty expansion slots and auxiliary absence are explicit bootstrap
policies. Unknown I/O and unacquired mapping combinations stay unsupported.
"""

from pce500.oz9600.lcd import LcdController
from pce500.oz9600.rtc import Rtc
from pce500.oz9600.tablet import Tablet
from pce500.pi3000.auxiliary import Auxiliary

WIDTH, HEIGHT = 240, 336


class Hardware:
    def __init__(
        self, banks, *, absent_auxiliary=False, auxiliary_control=False, firmware=False
    ):
        self.banks = banks
        self.selector, self.mode, self.cycle = 0xFD, 0, 0
        self.ram, self.scratch = bytearray(0x40000), bytearray(0x4000)
        self.gate, self.gpio = bytearray(64), bytearray(18)
        self.lcd, self.rtc = LcdController(WIDTH, HEIGHT), Rtc()
        self.lcd.firmware_window_candidate = True
        self.lcd.ordered_window_clipping_candidate = firmware
        self.tablet, self.tablet_route = Tablet(), "integrated" if firmware else None
        self.tablet_interrupt_source = "gate1" if firmware else "gate3"
        self.halt_main_timer_candidate = firmware
        self.halt_main_timer_units = 0
        self.rtc_tick_period = 512_000 if firmware else None
        self.rtc_tick_phase = self.rtc_ticks = 0
        self.rtc_second_period = None
        self.rtc_second_phase = self.rtc_seconds = self.rtc_held_units = 0
        self.rtc_event_candidate = False
        self.unmapped_low_read = 0xE0 if firmware else None
        self.clock_control = self.mapping_unlock = self.lcd_writes = 0
        self.startup_latches, self.io_8a = bytearray(3), bytearray(9)
        self.absent_auxiliary = absent_auxiliary
        self.auxiliary = Auxiliary() if auxiliary_control or firmware else None
        self.auxiliary_sample_irq_candidate = firmware
        self.auxiliary_sample_period = 4096
        self.auxiliary_sample_deadline = None
        self.auxiliary_request = bytearray(32)
        self.auxiliary_commands = []
        self.auxiliary_command_count = 0
        self.fault = None

    def read(self, address):
        if 0 <= address < 0x4000:
            return self.scratch[address]
        if 0x5000 <= address < 0x8000:
            return self.unmapped_low_read
        if address == 0x4021:
            return self.selector
        if 0x4000 <= address < 0x4040:
            return self.gate[address - 0x4000]
        if 0x8000 <= address < 0x8020:
            return self.rtc.read(address - 0x8000)
        if 0x8400 <= address < 0x8420 or 0x8438 <= address < 0x8440:
            return self.lcd.peek(address & 31, self.cycle)
        if address == 0x8A00 and self.absent_auxiliary:
            return 2
        if address == 0x8A00 and self.auxiliary is not None:
            return self.auxiliary.status()
        if address in (0x8A04, 0x8A05, 0x8A09) and self.auxiliary is not None:
            return 0
        if 0x8A40 <= address <= 0x8A5F and self.auxiliary is not None:
            return self.auxiliary.response(address - 0x8A40)
        if address == 0x8B1B and self.tablet_route is not None:
            return int(self.tablet_route == "external")
        if (address, self.tablet_route) in (
            (0x8800, "external"),
            (0x8B80, "integrated"),
        ):
            return self.tablet.conversion_control
        if (address, self.tablet_route) in (
            (0x8C00, "external"),
            (0x8BC0, "integrated"),
        ):
            return self.tablet.peek_data()
        if address == 0x8B1A:
            return self.gpio[2]
        if address == 0x8B1C:
            return self.clock_control
        if address == 0x8B1D:
            return self.gpio[5]
        if 0xF01C <= address <= 0xF01E:
            return self.startup_latches[address - 0xF01C]
        if 0x20000 <= address < 0x80000:
            return 255
        if 0x80000 <= address < 0xC0000:
            return self.ram[address - 0x80000]
        if (
            0xC0000 <= address < 0xE0000
            and self.mode == 0
            and self.selector in self.banks
        ):
            return self.banks[self.selector][address - 0xC0000]
        return None

    def architectural_read(self, address, pc=None):
        try:
            if (address, self.tablet_route) in (
                (0x8C00, "external"),
                (0x8BC0, "integrated"),
            ):
                return self.tablet.read_data()
            if 0x8400 <= address < 0x8420:
                return self.lcd.read(address & 31, self.cycle)
            value = self.read(address)
            if value is None:
                raise ValueError(f"unqualified PI read {address:05X}, PC {pc}")
            return value
        except ValueError as error:
            self.fault = self.fault or str(error)
            return 255

    def write(self, address, value, pc=None):
        try:
            if 0 <= address < 0x4000:
                self.scratch[address] = value
            elif address == 0x4021:
                self.selector = value
            elif 0x4000 <= address < 0x4040:
                self.gate[address - 0x4000] = value
            elif 0x8000 <= address < 0x8020:
                self.rtc.write(address - 0x8000, value)
            elif 0x8420 <= address < 0x8440:
                self.lcd_writes += 1
                self.lcd.write(address & 31, value)
            elif address == 0x8B01:
                self.mode = value
            elif address == 0x8B00:
                self.clock_control = value
            elif address == 0x8B04:
                self.gpio[4] = self.tablet.drive_control = value
            elif 0x8B02 <= address <= 0x8B05 or 0x8B08 <= address <= 0x8B11:
                self.gpio[address - 0x8B00] = value
            elif (address, self.tablet_route) in (
                (0x8800, "external"),
                (0x8B80, "integrated"),
            ):
                self.tablet.write_conversion_control(value)
            elif address in (0x8A01, 0x8A06, 0x8A07, 0x8A08) or (
                (self.absent_auxiliary or self.auxiliary is not None)
                and address in (0x8A02, 0x8A03)
            ):
                self.io_8a[address - 0x8A00] = value
                if (
                    address == 0x8A06
                    and value & 0x80
                    and self.auxiliary_sample_irq_candidate
                ):
                    self.gate[0x12] |= 8
                    self.auxiliary_sample_deadline = (
                        self.cycle + self.auxiliary_sample_period
                    )
                elif address == 0x8A06:
                    self.auxiliary_sample_deadline = None
                if address == 0x8A02 and value == 0x80:
                    self.auxiliary_command_count += 1
                    if len(self.auxiliary_commands) < 256:
                        self.auxiliary_commands.append(bytes(self.auxiliary_request))
                if self.auxiliary is not None and address in (0x8A02, 0x8A03):
                    if address == 0x8A02 and value == 0x80:
                        self.auxiliary.start(self.auxiliary_request, self.cycle)
                    elif address == 0x8A03:
                        self.auxiliary.acknowledge(value)
                    else:
                        raise ValueError(
                            f"unimplemented PI auxiliary start value {value:02X}"
                        )
            elif (
                self.absent_auxiliary or self.auxiliary is not None
            ) and 0x8A20 <= address < 0x8A40:
                offset = address - 0x8A20
                self.auxiliary_request[offset] = value
                if self.auxiliary is not None:
                    self.auxiliary.mark_written(offset)
            elif 0xF00C <= address <= 0xF00E:
                self.startup_latches[address - 0xF00C] = value
            elif address == 0xF103:
                self.mapping_unlock = value
            elif 0x20000 <= address < 0x80000:
                pass
            elif 0x80000 <= address < 0xC0000:
                self.ram[address - 0x80000] = value
            else:
                raise ValueError(f"unqualified PI write {address:05X}, PC {pc}")
        except ValueError as error:
            self.fault = self.fault or str(error)

    def configure_tablet_route(self, route):
        if route not in ("integrated", "external"):
            raise ValueError("unknown PI tablet ADC route")
        if self.tablet_route is not None and self.tablet_route != route:
            raise ValueError("PI tablet ADC route cannot change after selection")
        self.tablet_route = route

    def set_raw_contact(self, x, y, pressed):
        if self.tablet_route is None:
            raise ValueError(
                "PI tablet ADC route is unqualified; select an explicit candidate"
            )
        if self.tablet.set_contact(x, y, pressed) and self.auxiliary is not None:
            self.gate[0x12] |= 2 if self.tablet_interrupt_source == "gate1" else 8

    def configure_tablet_interrupt_source(self, source):
        if source not in ("gate1", "gate3"):
            raise ValueError("unknown PI contact-source candidate")
        if self.tablet.pressed or self.tablet.data_reads:
            raise ValueError(
                "PI contact source must be selected before input/conversion"
            )
        self.tablet_interrupt_source = source

    def advance(self, cycle):
        self.cycle = cycle
        if self.auxiliary is not None:
            self.auxiliary.advance(cycle)
        self.advance_auxiliary_sample_irq(cycle)

    def advance_auxiliary_sample_irq(self, cycle):
        if (
            self.auxiliary_sample_deadline is not None
            and cycle >= self.auxiliary_sample_deadline
        ):
            if self.auxiliary_sample_irq_candidate and self.io_8a[6] & 0x80:
                self.gate[0x12] |= 8
                self.auxiliary_sample_deadline = cycle + self.auxiliary_sample_period
            else:
                self.auxiliary_sample_deadline = None

    def halt_main_deadline(self, *, halted, off, lcc, next_mti):
        """Boundary clock candidate; shared CPU runtime owns timer/IRQ mutation."""
        if self.halt_main_timer_candidate and halted and not off and lcc & 0x80:
            return next_mti
        return None

    def external_irq(self):
        # F15C4 RTC dispatch uses gate bit0, independently of the auxiliary.
        if self.rtc.interrupt_asserted():
            self.gate[0x12] |= 1
        return bool(self.gate[0x10] & self.gate[0x12] & 0x7F)

    def advance_rtc_ticks(self, elapsed):
        """Explicit A2 timing candidate; does not advance calendar or guest RAM."""
        if self.rtc_tick_period is not None:
            ticks, self.rtc_tick_phase = divmod(
                self.rtc_tick_phase + elapsed, self.rtc_tick_period
            )
            if ticks:
                self.rtc_ticks = (self.rtc_ticks + ticks) & (2**64 - 1)
                self.rtc.raise_causes(4, 0)

    def advance_rtc_calendar(self, elapsed):
        """Opt-in packed calendar and ROM-backed A3..A6/ordinary alarm policy."""
        if self.rtc_second_period is not None:
            if self.rtc.read(12) & 128:  # CONTROL.HOLD
                self.rtc_held_units = (self.rtc_held_units + elapsed) & (2**64 - 1)
            else:
                seconds, phase = divmod(
                    self.rtc_second_phase + elapsed, self.rtc_second_period
                )
                if seconds:
                    if self.rtc_event_candidate:
                        self.rtc.advance_experimental_seconds(seconds)
                    else:
                        self.rtc.advance_seconds(seconds)
                    self.rtc_seconds = (self.rtc_seconds + seconds) & (2**64 - 1)
                self.rtc_second_phase = phase
