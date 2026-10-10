"""Reference PA reset bus, mirrored by sc62015/core/src/pa9500.

No captured live RAM is loaded. The high 256 KiB SRAM population and optional
OZ controller semantics are named hypotheses, not a physical board contract.
"""

from pce500.oz9600.lcd import LcdController
from pce500.oz9600.rtc import Rtc

FIXED_ROM_HASH = "4e61da23861f8a007cd83116dd26b03db5843bdd6daa033aba36a68e52d6c1aa"
SELECTORS = (*range(8), *range(248, 256))


class Hardware:
    def __init__(self, banks, controller_hypothesis=False):
        self.selector, self.cycle = 7, 0
        self.gate, self.ram = bytearray(64), bytearray(0x40000)
        self.banks = banks
        self.startup_latches = bytearray(2)
        self.compatibility_command = None
        self.external_media_enabled, self.card = False, None
        self.native_zero_data = None
        self.adc_control = 0 if controller_hypothesis else None
        self.lcd = LcdController(195, 145) if controller_hypothesis else None
        self.rtc = Rtc() if controller_hypothesis else None

    def read(self, address):
        if address == 0:
            return self.native_zero_data
        if address == 0x4021:
            return self.selector
        if 0x4000 <= address <= 0x403F:
            return self.gate[address - 0x4000]
        if address == 0x8800:
            return self.adc_control
        if address == 0x8C00 and self.adc_control is not None:
            return 0
        if 0x80000 <= address <= 0xBFFFF:
            return self.ram[address - 0x80000]
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

    def write(self, address, value):
        if (
            address == 0x8800
            and self.adc_control is not None
            and value in (6, 7, 14, 22, 30)
        ):
            self.adc_control = value
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
        elif 0x20000 <= address <= 0x7FFFF and self.external_media_enabled:
            if self.card:
                self.card.write(address, value)
        elif 0x8000 <= address <= 0x801F and self.rtc:
            self.rtc.write(address & 31, value)
        elif 0x8420 <= address <= 0x843F and self.lcd:
            self.lcd.write(address & 31, value)
        else:
            raise ValueError(f"unqualified PA write {address:05X}")
