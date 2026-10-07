"""Native host layout/contact reference; guest pixels stay separate from artwork.

The Rust frontend owns the OS window. This pure Python reference covers the
shared pixel, coordinate and contact-ownership contract without another GUI or
machine implementation. ADC centers come from default firmware calibration.
"""

from dataclasses import dataclass, field

from pce500.display.lcd_frame import LcdFrame

WIDTH, HEIGHT = 526, 530
LCD = (162, 36, 336, 240)
SOUND = (414, 7, 100, 22)
PAPER, INK = 0xD8DFBC, 0x263325
PANEL = (
    ("CAL", 61, 141),
    ("SCHED", 122, 141),
    ("MENU", 182, 141),
    ("TO DO", 61, 288),
    ("ANN", 122, 288),
    ("CALC", 182, 288),
    ("TEL", 61, 434),
    ("USER", 122, 434),
    ("CLOCK", 182, 434),
    ("NOTE", 61, 581),
    ("OUTLN", 122, 581),
    ("SCRAP", 182, 581),
    ("FILER", 61, 727),
    ("CARD", 122, 727),
    ("SEARCH", 122, 874),
)


def sound_artwork(enabled: bool) -> tuple[str, int]:
    """Host-only sound label/fill; the control does not drive a guest contact."""
    return ("SOUND ON", 0xB2C9A5) if enabled else ("SOUND OFF", 0x94ACBB)


def window_point(x: float, y: float, width: int, height: int) -> tuple[int, int] | None:
    """Invert physical surface scaling without assuming a platform pixel ratio."""
    if width <= 0 or height <= 0 or not (0 <= x < width and 0 <= y < height):
        return None
    return int(x * WIDTH / width), int(y * HEIGHT / height)


@dataclass
class CloseGate:
    """Once close begins, queued redraws and controls cannot run a new epoch."""

    closing: bool = False

    def begin_close(self) -> bool:
        if self.closing:
            return False
        self.closing = True
        return True


@dataclass
class PointerGate:
    """Focus/press eligibility; callers cancel Contacts on focus loss or fault."""

    active: bool = False
    down: bool = False
    blocked: bool = False

    def focus(self, active: bool):
        self.active = active
        if not active:
            self.blocked = self.down

    def button(self, pressed: bool) -> bool:
        self.down = pressed
        if not pressed:
            self.blocked = False
            return False
        if not self.active:
            self.blocked = True
            return False
        return not self.blocked

    def accepts_target(self, *, host_control: bool, fault: bool) -> bool:
        return (
            self.active
            and self.down
            and not self.blocked
            and (host_control or not fault)
        )


@dataclass
class HostKeys:
    """Keep both edges of a short host key tap; repeated down is not a new tap."""

    down: set[str] = field(default_factory=set)

    def events(self, events: list[tuple[str, bool]]) -> set[str]:
        presses = set()
        for key, pressed in events:
            if pressed:
                if key not in self.down:
                    presses.add(key)
                self.down.add(key)
            else:
                self.down.discard(key)
        return presses


def lcd_tablet(x: int, y: int) -> tuple[int, int]:
    """Inverse of the ROM's default screen conversion; not physical calibration."""
    return (
        (4 * min(335, max(0, x)) + 419) * 548 // 1008,
        (4 * min(239, max(0, y)) + 75) * 630 // 688,
    )


def contains(rect: tuple[int, int, int, int], x: int, y: int) -> bool:
    rx, ry, width, height = rect
    return rx <= x < rx + width and ry <= y < ry + height


def lcd_into(frame: LcdFrame, host_pixels: list[int]) -> list[int]:
    """Return an owned host image with every guest pixel copied unchanged last."""
    if (frame.cols, frame.rows) != LCD[2:]:
        raise ValueError("native window requires full 336x240 controller image")
    if len(host_pixels) != WIDTH * HEIGHT:
        raise ValueError("host image shape mismatch")
    output = list(host_pixels)
    for i, pixel in enumerate(frame.pixels):
        y, x = divmod(i, frame.cols)
        output[(LCD[1] + y) * WIDTH + LCD[0] + x] = INK if pixel else PAPER
    return output


@dataclass
class Contacts:
    """Union of host owners; cancel releases even an assisted short contact."""

    minimum_hold: int = 0
    keys: dict[int, int] = field(default_factory=dict)
    tablet: tuple[int, int, int] | None = None
    on_deadline: int | None = None

    def sync(
        self,
        desired: set[int],
        tablet: tuple[int, int] | None,
        at: int,
        on: bool = False,
    ) -> list:
        events = []
        for code in sorted(desired):
            if code not in self.keys:
                self.keys[code] = at + self.minimum_hold
                events.append(("matrix", code, True))
        for code, end in sorted(tuple(self.keys.items())):
            if code not in desired and at >= end:
                del self.keys[code]
                events.append(("matrix", code, False))
        if on and self.on_deadline is None:
            self.on_deadline = at + self.minimum_hold
            events.append(("on", True))
        elif not on and self.on_deadline is not None and at >= self.on_deadline:
            self.on_deadline = None
            events.append(("on", False))
        if tablet is not None:
            x, y = tablet
            if self.tablet is None:
                self.tablet = (x, y, at + self.minimum_hold)
                events.append(("tablet", x, y, True))
            elif self.tablet[:2] != tablet:
                self.tablet = (x, y, self.tablet[2])
                events.append(("tablet", x, y, True))
        elif self.tablet is not None and at >= self.tablet[2]:
            x, y, _ = self.tablet
            self.tablet = None
            events.append(("tablet", x, y, False))
        return events

    def cancel(self) -> list:
        events = [("matrix", code, False) for code in sorted(self.keys)]
        self.keys.clear()
        if self.on_deadline is not None:
            events.append(("on", False))
            self.on_deadline = None
        if self.tablet is not None:
            events.append(("tablet", *self.tablet[:2], False))
            self.tablet = None
        return events
