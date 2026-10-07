"""Provisional logical OZ bus. No CPU, captured RAM or physical alias claim."""

from .audio import AudioCapture
from .lcd import LcdController
from .rtc import Rtc
from .tablet import Tablet

FIXED_ROM_HASH = "05fa315246a13d7d33650be7c628d803157d79c94e620aca920ab7473e98e519"
# F0D6B/F0D6E samples live ON separately from SSR.1 card presence.
ON_KEY_SSR_MASK = 8
# FE591 reserves [1FF38]+A0000: low workspace shares the final RAM quarter.
# Normal-view firmware policy; physical mode/page/chip-enable decoding pending.
WORKSPACE_RAM_OFFSET = 0x30000


def guest_internal_write_is_allowed(address: int) -> bool:
    """OZ guest EIL/EIH writes do not change device-owned input backing.

    Related ESR-L read-only register evidence and normal OZ initialization
    motivate this bus policy; physical E-port levels remain provisional.
    This reference does not intercept host/device input updates.
    """
    return address not in (0x1000F5, 0x1000F6)


class Hardware:
    def __init__(self):
        self.selector, self.cycle, self.fault = 7, 0, None
        self.gate, self.gpio, self.ram = (
            bytearray(64),
            bytearray(256),
            bytearray(0x40000),
        )
        self.banks, self.events = {}, []
        self.rtc, self.lcd, self.tablet = Rtc(), LcdController(), Tablet()
        self.audio = AudioCapture()
        self.audio_ci_input = False  # Physical/SSR connection is unqualified.
        self.retained_loaded = False
        self.card = None

    def log(self, kind, address, value, pc):
        if self.events and all(
            self.events[-1][k] == v
            for k, v in (
                ("kind", kind),
                ("address", address),
                ("value", value),
                ("pc", pc),
            )
        ):
            last = self.events[-1]
            last["repeats"] = last.get("repeats", 1) + 1
            last["last_cycle"] = self.cycle
        elif len(self.events) < 2048:
            self.events.append(
                dict(kind=kind, address=address, value=value, pc=pc, cycle=self.cycle)
            )

    def read(self, address):
        if address == 0x4021:
            return self.selector
        if 0x4000 <= address <= 0x403F:
            return self.gate[address - 0x4000]
        if 0x8000 <= address <= 0x83FF:
            return self.rtc.read(address & 31)
        if 0x8400 <= address <= 0x843F:
            return self.lcd.peek(address & 31, self.cycle)
        if address == 0x8800:
            return self.tablet.conversion_control
        if address in (0x8B1A, 0x8B1C, 0x8B1D):
            return self.gpio[{0x8B1A: 2, 0x8B1C: 0, 0x8B1D: 5}[address]]
        if 0x8B00 <= address <= 0x8BFF:
            return self.gpio[address & 255]
        if address == 0x8C00:
            return self.tablet.peek_data()
        if 0x8801 <= address <= 0x88FF or 0x8C01 <= address <= 0x8CFF:
            return 0
        if 0x10000 <= address <= 0x1FFFF:
            return self.ram[WORKSPACE_RAM_OFFSET + address - 0x10000]
        if 0x80000 <= address <= 0xBFFFF:
            return self.ram[address - 0x80000]
        if 0x30000 <= address <= 0x3FFFF:
            return self.card.read(address) if self.card else None
        if 0x40000 <= address <= 0x7FFFF:
            return self.card.read(address) if self.card else 255
        if 0xC0000 <= address <= 0xDFFFF and self.selector in self.banks:
            return self.banks[self.selector][address - 0xC0000]
        return None

    def architectural_read(self, address, pc=None):
        try:
            if address == 0x8C00:
                value = self.tablet.read_data()
            elif 0x8400 <= address <= 0x841F:
                value = self.lcd.read(address & 31, self.cycle)
            else:
                value = self.read(address)
                if value is None:
                    raise ValueError("unavailable ROM or peripheral read")
                if address >= 0x10000:
                    return value
            self.log("read", address, value, pc)
            return value
        except ValueError as error:
            self.fault = str(error)
            return 255

    def write(self, address, value, pc=None):
        kind = "write"
        try:
            if address == 0x4021:
                self.selector, kind = value, "select"
            elif 0x4000 <= address <= 0x403F:
                self.gate[address - 0x4000] = value
            elif 0x8000 <= address <= 0x83FF:
                self.rtc.write(address & 31, value)
            elif 0x8420 <= address <= 0x843F:
                self.lcd.write(address & 31, value)
            elif address == 0x8800:
                self.tablet.write_conversion_control(value)
            elif 0x8B00 <= address <= 0x8BFF:
                self.gpio[address & 255] = value
                if address == 0x8B04:
                    self.tablet.drive_control = value
            elif 0x80000 <= address <= 0xBFFFF:
                self.ram[address - 0x80000] = value
                return
            elif 0x10000 <= address <= 0x1FFFF:
                self.ram[WORKSPACE_RAM_OFFSET + address - 0x10000] = value
                return
            elif 0x30000 <= address <= 0x3FFFF:
                if self.card:
                    self.card.write(address, value)
                    return
                kind = "unimplemented_write"
            elif 0x40000 <= address <= 0x7FFFF or 0xC0000 <= address <= 0xDFFFF:
                return
            else:
                kind = "unimplemented_write"
        except ValueError as error:
            self.fault = str(error)
        self.log(kind, address, value, pc)

    def refresh_irq_inputs(self):
        if self.rtc.interrupt_asserted():
            self.gate[18] |= 1
        return bool(self.gate[18] & self.gate[16] & 127)
