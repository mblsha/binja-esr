"""Reference PA reset bus, mirrored by sc62015/core/src/pa9500.

No captured live RAM is loaded. The high 256 KiB SRAM population and optional
OZ controller semantics are named hypotheses, not a physical board contract.
"""

from pce500.oz9600.lcd import LcdController
from pce500.oz9600.rtc import Rtc
from pce500.peripherals.uart import Uart

FIXED_ROM_HASH = "4e61da23861f8a007cd83116dd26b03db5843bdd6daa033aba36a68e52d6c1aa"
SELECTORS = (*range(8), *range(248, 256))


def on_irq_should_reassert(held, *, contact_edge_candidate=False):
    """Opt-in ON latch hypothesis; the physical SSR contact remains held.

    Guest acknowledgement consumes the edge latch. A duplicate host press is
    not an electrical edge. Assertion/re-latch timing is not hardware verified.
    """
    return held and not contact_edge_candidate


def register_uart_candidate():
    """Register-only transport with explicit nominal timing; firmware owns COM.

    The PA hardware constructor leaves this opt-in peripheral unattached.
    Physical clock/divisor, parity waveform and connector wiring are open.
    """
    return Uart(timebase_hz=1_024_000, baud_divisor=2)


class Hardware:
    def __init__(self, banks, controller_hypothesis=False):
        self.selector, self.cycle = 7, 0
        self.gate, self.ram = bytearray(64), bytearray(0x40000)
        self.banks = banks
        self.startup_latches = bytearray(2)
        self.compatibility_command = None
        self.external_media_enabled, self.card = False, None
        self.card_lower_write_alias = False
        # Explicit F483B strobe/input candidate, not measured board wiring.
        self.card_eil_presence_candidate = False
        self.native_zero_data = None
        self.observed_low_pair = None
        self.scratch_candidate = None
        self.lcd_coordinate_restore_candidate = False
        self.adc_control = 0 if controller_hypothesis else None
        # Explicit raw electrical candidate; no host coordinate hit testing.
        self.tablet = None
        self.adc_low_next, self.adc_latched, self.adc_data_reads = False, 0, 0
        self.lcd = LcdController(195, 145) if controller_hypothesis else None
        self.rtc = Rtc() if controller_hypothesis else None

    def read(self, address):
        if address == 0:
            return self.native_zero_data
        if 0xC <= address <= 0xD and self.observed_low_pair is not None:
            return self.observed_low_pair[address - 0xC]
        if address == 0x4012 and self.tablet is not None:
            return (self.gate[18] & ~2) | (2 if self.tablet[3] else 0)
        if address == 0x4021:
            return self.selector
        if 0x4000 <= address <= 0x403F:
            return self.gate[address - 0x4000]
        if address == 0x8800:
            return self.adc_control
        if address == 0x8C00 and self.adc_control is not None:
            return self.adc_latched & 3 if self.adc_low_next else self.adc_sample() >> 2
        if 0x80000 <= address <= 0xBFFFF:
            return self.ram[address - 0x80000]
        if (
            0xE000 <= address <= 0xEFFF
            and self.compatibility_command == 1
            and self.scratch_candidate is not None
        ):
            return self.scratch_candidate[address - 0xE000]
        if 0x20000 <= address <= 0x7FFFF and self.external_media_enabled:
            value = self.card.read(address) if self.card else None
            return 255 if value is None else value
        if 0xC0000 <= address <= 0xDFFFF and self.selector in self.banks:
            return self.banks[self.selector][address - 0xC0000]
        if 0x8000 <= address <= 0x801F and self.rtc:
            return self.rtc.read(address & 31)
        if 0x8400 <= address <= 0x843F and self.lcd:
            return self.lcd.peek(address & 31, self.cycle)
        return None

    def adc_sample(self):
        if self.tablet is None or not self.tablet[3]:
            return 0
        return {6: self.tablet[0], 7: self.tablet[1], 22: self.tablet[2] << 2}.get(
            self.adc_control, 0
        )

    def card_presence_input(self, eol):
        if not self.card_eil_presence_candidate or not self.external_media_enabled:
            return None
        return self.card is not None and bool(eol & 4)

    def architectural_read(self, address):
        value = self.read(address)
        if address == 0x8C00 and self.adc_control is not None:
            if not self.adc_low_next:
                self.adc_latched = self.adc_sample()
            self.adc_low_next = not self.adc_low_next
            self.adc_data_reads += 1
        return value

    def write(self, address, value):
        if (
            address == 0x8800
            and self.adc_control is not None
            and value in (6, 7, 14, 22, 30)
        ):
            self.adc_control = value
            self.adc_low_next = False
        elif address == 0x4021:
            self.selector = value
        elif 0x4000 <= address <= 0x403F:
            self.gate[address - 0x4000] = value
        elif 0xF00D <= address <= 0xF00E:
            self.startup_latches[address - 0xF00D] = value
        elif (
            address == 0xF00C and self.compatibility_command is not None and value == 1
        ):
            self.compatibility_command = value
        elif 0x80000 <= address <= 0xBFFFF:
            self.ram[address - 0x80000] = value
        elif (
            0xE000 <= address <= 0xEFFF
            and self.compatibility_command == 1
            and self.scratch_candidate is not None
        ):
            self.scratch_candidate[address - 0xE000] = value
        elif 0x20000 <= address <= 0x7FFFF and self.external_media_enabled:
            if self.card:
                target = (
                    address + 0x8000
                    if self.card_lower_write_alias and 0x30000 <= address <= 0x37FFF
                    else address
                )
                self.card.write(target, value)
        elif 0x8000 <= address <= 0x801F and self.rtc:
            self.rtc.write(address & 31, value)
        elif self.lcd and (
            0x8420 <= address <= 0x843F
            or self.lcd_coordinate_restore_candidate
            and address in (0x8412, 0x8413, 0x8416, 0x8417)
        ):
            self.lcd.write(address & 31, value)
        else:
            raise ValueError(f"unqualified PA write {address:05X}")

    def install_lcd_coordinate_restore_candidate(self):
        """Opt-in F2FAC/F2FB1 Y-coordinate aliases; physical decoding is open."""
        if self.cycle or self.lcd is None or self.lcd_coordinate_restore_candidate:
            raise ValueError(
                "LCD coordinate restore candidate requires a fresh PA controller"
            )
        self.lcd_coordinate_restore_candidate = True
