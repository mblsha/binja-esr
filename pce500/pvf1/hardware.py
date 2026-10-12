"""PV-F1 logical board reference; physical routing/timing remain candidates.

The shared CPU owns native scheduling; LCD, RTC, ADC and auxiliary transport
are reused independently of the related models' mapper and command contracts.
"""

from pce500.oz9600.lcd import LcdController
from pce500.oz9600.rtc import Rtc
from pce500.oz9600.tablet import Tablet
from pce500.pvf1.auxiliary import Auxiliary, CAPTURE_SAMPLE_PERIOD_CANDIDATE
from pce500.pvf1.rtc_clock import CalendarClock, HalfSecondClock
from pce500.tablet_geometry import quarter_to_adc
from pce500.pa9500.card import Pa9c3Card
from pce500.peripherals.uart import Uart

WIDTH, HEIGHT = 336, 292


class Hardware:
    def lcd_to_raw(self, x, y, workspace):
        if (
            not isinstance(x, int)
            or not isinstance(y, int)
            or not 0 <= x < 336
            or not 0 <= y < 240
        ):
            raise ValueError("LCD contact outside PV-F1 main window")
        if not 0x80000 <= workspace <= 0xBFF66:
            raise ValueError("PV-F1 tablet workspace is unavailable")
        words = [
            int.from_bytes(
                bytes(
                    [
                        self.peek(workspace + 0x8E + 2 * i),
                        self.peek(workspace + 0x8F + 2 * i),
                    ]
                ),
                "little",
                signed=True,
            )
            for i in range(6)
        ]
        return quarter_to_adc((4 * x + 2, 4 * y + 2), words)

    def __init__(self, banks, profile="strict"):
        if profile not in (
            "strict",
            "absent-auxiliary",
            "control-candidate",
            "firmware",
        ):
            raise ValueError("unknown PV-F1 profile")
        self.banks, self.profile = banks, profile
        self.irq_imr_only_candidate = profile == "firmware"
        self.on_input_live_candidate = profile == "firmware"
        self.alarm_wake_candidate = profile == "firmware"
        self.cpu_options = {
            "block_transfer_policy": "independent"
            if profile == "strict"
            else "coupled_predecrement",
            "byte_arithmetic_source_policy": "strict"
            if profile == "strict"
            else "low_byte",
        }
        self.selector, self.cycle, self.fault = 7, 0, None
        self.gate, self.gpio = bytearray(64), bytearray(18)
        self.pixel_expansion = 0
        self.ram, self.scratch = bytearray(0x40000), bytearray(0x4000)
        self.lcd, self.rtc = LcdController(WIDTH, HEIGHT), Rtc()
        self.rtc_clock = HalfSecondClock()
        self.calendar_clock = CalendarClock()
        if profile == "firmware":
            self.rtc_clock.configure(512_000)  # Nominal half-second candidate.
            self.calendar_clock.configure(1_024_000)
        self.startup = bytearray(3)
        self.mapping_unlock = 0
        self.auxiliary = (
            None if profile == "strict" else Auxiliary(profile != "absent-auxiliary")
        )
        if profile == "firmware":
            self.auxiliary.configure_sample_period(CAPTURE_SAMPLE_PERIOD_CANDIDATE)
        self.expansion_absent = profile != "strict"
        self.card = None
        self.card_presence_candidate = False
        # Separate logical PA-9C3 input buffer; physical alias/extent unverified.
        self.card_buffer = bytearray(0x1000)
        self.uart = Uart(1_024_000, 2) if profile == "firmware" else None
        self.tablet = None if profile == "strict" else Tablet(contact_sample=1023)
        self.halt_main_timer_candidate = profile == "firmware"
        self.start_timers_at_first_halt_candidate = profile == "firmware"
        self.halt_main_timer_units = 0

    def peek(self, address):
        if 0 <= address <= 0x3FFF:
            return self.scratch[address]
        if address == 0x4021:
            return self.selector
        if address in (0x401E, 0x401F):
            return (self.pixel_expansion >> (8 * (address - 0x401E))) & 255
        if 0x4000 <= address <= 0x403F:
            return self.gate[address - 0x4000]
        if 0x8000 <= address <= 0x801F:
            return self.rtc.read(address - 0x8000)
        if (
            0x8400 <= address <= 0x841F
            or address == 0x842E
            or 0x8438 <= address <= 0x843F
        ):
            return self.lcd.peek(address & 31, self.cycle)
        if address in (0x8B1A, 0x8B1C, 0x8B1D):
            return self.gpio[{0x8B1A: 2, 0x8B1C: 0, 0x8B1D: 5}[address]]
        if 0x8A00 <= address <= 0x8A5F and self.auxiliary:
            return self.auxiliary.peek(address)
        if address == 0x8800 and self.tablet:
            return self.tablet.conversion_control
        if address == 0x8C00 and self.tablet:
            return self.tablet.peek_data()
        if 0xF01C <= address <= 0xF01E:
            return self.startup[address - 0xF01C]
        if (
            0xE000 <= address <= 0xEFFF
            and self.card_presence_candidate
            and self.startup[0] == 1
        ):
            return self.card_buffer[address - 0xE000]
        if 0x20000 <= address <= 0x7FFFF:
            if self.card:
                result = self.card.read(address)
                return 255 if result is None else result
            if self.expansion_absent:
                return 255
        if 0x80000 <= address <= 0xBFFFF:
            return self.ram[address - 0x80000]
        return None

    def configure_pa9c3_candidate(self, rom, sram, *, instructions=0, elapsed=0):
        card = Pa9c3Card(rom, sram)
        if (
            self.tablet is None
            or self.card is not None
            or self.card_presence_candidate
            or instructions != 0
            or elapsed != 0
            or self.cycle != 0
        ):
            raise ValueError(
                "PV-F1 card candidate must be configured once before execution"
            )
        self.card = card
        self.card_presence_candidate = True

    def configure_uart_candidate(self, baud_divisor, *, instructions=0, elapsed=0):
        if (
            baud_divisor <= 0
            or self.tablet is None
            or self.uart is not None
            or instructions != 0
            or elapsed != 0
            or self.cycle != 0
        ):
            raise ValueError(
                "PV-F1 UART candidate requires a positive divisor once before execution"
            )
        self.uart = Uart(1_024_000, baud_divisor)

    def card_presence_sample(self, ssr):
        if not self.card_presence_candidate:
            return ssr
        # SSR.CI (bit1) is a card-presence wiring candidate, not an IRQ event.
        return ssr | 2 if self.card is not None else ssr & ~2

    def read_mutated(self, address):
        try:
            if address == 0x8C00 and self.tablet:
                return self.tablet.read_data()
            if 0x8400 <= address <= 0x841F:
                return self.lcd.read(address & 31, self.cycle)
            value = self.peek(address)
            if value is None:
                raise ValueError(f"unresolved PV-F1 read {address:05X}")
            return value
        except ValueError as error:
            self.fault = self.fault or str(error)
            raise

    def write(self, address, value):
        try:
            if 0 <= address <= 0x3FFF:
                self.scratch[address] = value
            elif address == 0x4021:
                self.selector = value
            elif address == 0x401E:
                # Native F2CDA/F2CFA request a doubled packed pixel column.
                self.gate[0x1E] = value
                self.pixel_expansion = sum(
                    3 << (2 * bit) for bit in range(8) if value & (1 << bit)
                )
            elif 0x4000 <= address <= 0x403F:
                self.gate[address - 0x4000] = value
            elif 0x8000 <= address <= 0x801F:
                self.rtc.write(address - 0x8000, value)
            elif 0x8420 <= address <= 0x843F:
                self.lcd.write(address & 31, value)
            elif 0x8B00 <= address <= 0x8B11:
                self.gpio[address - 0x8B00] = value
                if address == 0x8B04 and self.tablet:
                    self.tablet.drive_control = value
            elif address == 0x8800 and self.tablet:
                self.tablet.write_conversion_control(value)
            elif 0x8A00 <= address <= 0x8A5F and self.auxiliary:
                self.auxiliary.write(address, value, self.cycle)
            elif 0xF00C <= address <= 0xF00E:
                self.startup[address - 0xF00C] = value
            elif (
                0xE000 <= address <= 0xEFFF
                and self.card_presence_candidate
                and self.startup[0] == 1
            ):
                self.card_buffer[address - 0xE000] = value
            elif address == 0xF103 and value in (0xA5, 0x5A):
                self.mapping_unlock = value
            elif 0x20000 <= address <= 0x7FFFF and self.card:
                self.card.write(address, value)
            elif 0x20000 <= address <= 0x7FFFF and self.expansion_absent:
                pass
            elif 0x80000 <= address <= 0xBFFFF:
                self.ram[address - 0x80000] = value
            else:
                raise ValueError(f"unresolved PV-F1 write {address:05X}={value:02X}")
        except ValueError as error:
            self.fault = self.fault or str(error)
            raise

    def set_raw_contact(self, x, y, pressed):
        if self.tablet is None:
            raise ValueError("select an explicit PV-F1 tablet candidate")
        if self.tablet.set_contact(x, y, pressed):
            self.gate[0x12] |= 2

    def halt_timer_runs(self, halted, off, lcc):
        return (
            self.halt_main_timer_candidate and halted and not off and bool(lcc & 0x80)
        )

    def configure_rtc_half_second_candidate(self, period):
        if self.tablet is None:
            raise ValueError("RTC timing requires an explicit board candidate")
        self.rtc_clock.configure(period)

    def advance_rtc_timing(self, elapsed):
        if self.rtc_clock.advance(elapsed):
            self.rtc.raise_causes(4, 0)
        self.calendar_clock.advance(self.rtc, elapsed)

    def configure_calendar_candidate(self, period):
        if self.tablet is None:
            raise ValueError("RTC calendar requires an explicit board candidate")
        self.calendar_clock.configure(period)

    def configure_recognizer(self, data):
        from pce500.handwriting import Model

        if self.auxiliary is None:
            raise ValueError("requires an auxiliary candidate")
        self.auxiliary.configure_recognizer(None if data is None else Model(data))

    def external_irq(self):
        if self.auxiliary:
            if self.auxiliary.advance(self.cycle):
                self.gate[0x12] |= 8
            if self.auxiliary.take_result_irq_edge():
                self.gate[0x12] |= 0x40
        if self.tablet and self.rtc.interrupt_asserted():
            self.gate[0x12] |= 1
        return self.tablet is not None and bool(self.gate[0x12] & self.gate[0x10])
