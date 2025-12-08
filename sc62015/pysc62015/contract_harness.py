"""Shared contract-test harness for Python and LLAMA backends.

This harness keeps the Rust core runnable without Python while letting pytest feed
identical IO/overlay vectors into both implementations. Backends implement the same
minimal surface: read/write hooks, observable events, and memory snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Protocol, Sequence

from pce500.memory import INTERNAL_MEMORY_START, PCE500Memory
from pce500.keyboard_handler import PCE500KeyboardHandler
from pce500.display.hd61202 import HD61202, parse_command, ChipSelect

try:
    import _sc62015_rustcore as rustcore
except ImportError:
    rustcore = None


AccessKind = Literal["read", "write"]

IMEM_ISR_OFFSET = 0xFC
IMEM_SSR_OFFSET = 0xFF


def _is_lcd_addr(address: int) -> bool:
    addr = address & 0xFFFFFF
    return (0x2000 <= addr <= 0x200F) or (0xA000 <= addr <= 0xAFFF)


@dataclass(frozen=True)
class AccessVector:
    """One memory operation in a contract test."""

    kind: AccessKind
    address: int
    value: int = 0
    pc: Optional[int] = None


@dataclass(frozen=True)
class ContractEvent:
    """Observable side-effect captured during contract execution."""

    kind: str
    address: int
    value: int
    pc: Optional[int] = None


@dataclass(frozen=True)
class ContractSnapshot:
    """Minimal snapshot after a contract run."""

    internal: bytes
    external: Optional[bytes]
    external_len: int
    imr: int
    isr: int
    lcd_events: tuple[ContractEvent, ...]
    lcd_status: Optional[int]
    lcd_log: tuple[tuple[int, int], ...]
    lcd_vram: Optional[bytes]
    lcd_meta: Optional[str]
    metadata: dict[str, object]


@dataclass(frozen=True)
class ContractRun:
    """Result of applying a vector sequence to a backend."""

    events: list[ContractEvent]
    snapshot: ContractSnapshot


class ContractBackend(Protocol):
    """Shared surface exercised by contract tests."""

    def read(self, address: int, pc: Optional[int] = None) -> int: ...

    def write(self, address: int, value: int, pc: Optional[int] = None) -> None: ...

    def drain_events(self) -> list[ContractEvent]: ...

    def snapshot(self) -> ContractSnapshot: ...

    def configure_timer(
        self, mti_period: int, sti_period: int, *, enabled: bool = True
    ) -> None: ...  # pragma: no cover

    def tick_timers(self, steps: int = 1) -> None: ...  # pragma: no cover

    def press_on_key(self) -> None: ...  # pragma: no cover

    def release_on_key(self) -> None: ...  # pragma: no cover


class PythonContractBackend:
    """Adapter over the Python memory bus/peripherals."""

    def __init__(self, memory: Optional[PCE500Memory] = None) -> None:
        self.memory = memory or PCE500Memory()
        self._events: list[ContractEvent] = []
        self._irq_pending = False
        self._irq_source: Optional[str] = None
        from pce500.scheduler import TimerScheduler

        self._timer = TimerScheduler(mti_period=0, sti_period=0, enabled=True)
        self._cycles = 0
        self._lcd_chips = (HD61202(), HD61202())
        # Minimal keyboard handler to mirror KIO behaviour for contract tests.
        self._keyboard = PCE500KeyboardHandler(self.memory)

        def _kb_read(addr: int, pc: Optional[int]) -> int:
            return int(self._keyboard.handle_register_read(addr))

        def _kb_write(addr: int, val: int, pc: Optional[int]) -> None:
            self._keyboard.handle_register_write(addr, val)

        self.memory.set_keyboard_handler(_kb_read, _kb_write, enable_overlay=True)

    def load_memory(
        self, *, external: Optional[bytes] = None, internal: Optional[bytes] = None
    ) -> None:
        if external is not None:
            limit = min(len(external), len(self.memory.external_memory))
            self.memory.external_memory[:limit] = external[:limit]
        if internal is not None:
            limit = min(len(internal), 256)
            start = len(self.memory.external_memory) - 256
            self.memory.external_memory[start : start + limit] = internal[:limit]
        for chip in self._lcd_chips:
            chip.reset()
        try:
            self._keyboard.release_all_keys()
        except Exception:
            pass
        self._irq_pending = False
        self._irq_source = None

    def read(self, address: int, pc: Optional[int] = None) -> int:
        value = int(self.memory.read_byte(address, pc)) & 0xFF
        self._events.append(
            ContractEvent(kind="read", address=address & 0xFFFFFF, value=value, pc=pc)
        )
        return value

    def write(self, address: int, value: int, pc: Optional[int] = None) -> None:
        self.memory.write_byte(address, value, pc)
        self._events.append(
            ContractEvent(
                kind="write", address=address & 0xFFFFFF, value=value & 0xFF, pc=pc
            )
        )
        if _is_lcd_addr(address):
            try:
                cmd = parse_command(address, value)
            except ValueError:
                cmd = None
            if cmd is not None:
                targets: tuple[int, ...]
                if cmd.cs == ChipSelect.BOTH:
                    targets = (0, 1)
                elif cmd.cs == ChipSelect.RIGHT:
                    targets = (1,)
                elif cmd.cs == ChipSelect.LEFT:
                    targets = (0,)
                else:
                    targets = ()
                for idx in targets:
                    chip = self._lcd_chips[idx]
                    if cmd.instr is None:
                        chip.write_data(cmd.data, pc_source=pc)
                    else:
                        chip.write_instruction(cmd.instr, cmd.data)

    def drain_events(self) -> list[ContractEvent]:
        events = list(self._events)
        self._events.clear()
        return events

    def press_on_key(self) -> None:
        """Latch ONK in SSR/ISR and mark pending to mirror emulator behaviour."""
        ssr_addr = INTERNAL_MEMORY_START + IMEM_SSR_OFFSET
        isr_addr = INTERNAL_MEMORY_START + IMEM_ISR_OFFSET
        ssr = self.read(ssr_addr)
        # write(...) logs a ContractEvent, so use it to keep parity with Rust bus logging.
        self.write(ssr_addr, ssr | 0x08)
        isr = self.read(isr_addr)
        self.write(isr_addr, isr | 0x08)
        self._irq_pending = True
        self._irq_source = "ONK"

    def release_on_key(self) -> None:
        ssr_addr = INTERNAL_MEMORY_START + IMEM_SSR_OFFSET
        isr_addr = INTERNAL_MEMORY_START + IMEM_ISR_OFFSET
        ssr = self.read(ssr_addr)
        self.write(ssr_addr, ssr & ~0x08)
        isr = self.read(isr_addr)
        self.write(isr_addr, isr & ~0x08)

    def snapshot(self) -> ContractSnapshot:
        internal = bytes(self.memory.get_internal_memory_bytes())
        external = bytes(self.memory.external_memory)
        lcd_events = tuple(evt for evt in self._events if _is_lcd_addr(evt.address))
        imr = internal[0xFB] if len(internal) > 0xFB else 0
        isr = internal[0xFC] if len(internal) > 0xFC else 0
        status = None
        for evt in reversed(self._events):
            if evt.kind == "read" and evt.address in (0x2001, 0xA001):
                status = evt.value
                break
        lcd_log = tuple(
            (evt.address, evt.value)
            for evt in self._events
            if evt.kind == "write" and _is_lcd_addr(evt.address)
        )
        lcd_vram: list[int] = []
        for chip in self._lcd_chips:
            for page in range(HD61202.LCD_PAGES):
                lcd_vram.extend(chip.vram[page])
        return ContractSnapshot(
            internal=internal,
            external=external,
            external_len=len(external),
            imr=imr,
            isr=isr,
            lcd_events=lcd_events,
            lcd_status=status,
            lcd_log=lcd_log,
            lcd_vram=bytes(lcd_vram),
            lcd_meta="chips=2,pages=8,width=64",
            metadata={
                "backend": "python",
                "imr": imr,
                "isr": isr,
                "lcd_status": status,
                "irq_pending": self._irq_pending,
                "irq_source": self._irq_source,
            },
        )

    def configure_timer(
        self, mti_period: int, sti_period: int, *, enabled: bool = True
    ) -> None:
        self._timer.mti_period = int(mti_period)
        self._timer.sti_period = int(sti_period)
        self._timer.enabled = bool(enabled)
        self._timer.reset(cycle_base=self._cycles)

    def tick_timers(self, steps: int = 1) -> None:
        from pce500.scheduler import TimerSource

        for _ in range(max(0, int(steps))):
            self._cycles += 1
            fired = self._timer.advance(self._cycles)
            if not fired:
                continue
            value = 0
            if TimerSource.MTI in fired:
                value |= 0x01
            if TimerSource.STI in fired:
                value |= 0x02
            # Mirror into ISR.
            current = self.memory.read_byte(0x100000 + 0xFC, None) & 0xFF
            self.memory.write_byte(0x100000 + 0xFC, current | value, None)
            self._events.append(
                ContractEvent(
                    kind="timer",
                    address=0x100000 + 0xFC,
                    value=value & 0xFF,
                    pc=None,
                )
            )


class RustContractBackend:
    """Adapter over the PyO3 contract bus exposed by the LLAMA core."""

    def __init__(self, host_memory: Optional[PCE500Memory] = None) -> None:
        if rustcore is None or not hasattr(rustcore, "LlamaContractBus"):
            raise RuntimeError("LlamaContractBus is unavailable (build rustcore first)")
        self._impl = rustcore.LlamaContractBus()  # type: ignore[attr-defined]
        if host_memory is not None and hasattr(self._impl, "set_host_memory"):
            try:
                self._impl.set_host_memory(host_memory)
            except Exception:
                # Best-effort: if host memory fails, continue with local-only model.
                pass

    def load_memory(
        self, *, external: Optional[bytes] = None, internal: Optional[bytes] = None
    ) -> None:
        if external is not None:
            self._impl.load_external(external)
        if internal is not None:
            self._impl.load_internal(internal)

    def read(self, address: int, pc: Optional[int] = None) -> int:
        return int(self._impl.read_byte(address, pc))

    def write(self, address: int, value: int, pc: Optional[int] = None) -> None:
        self._impl.write_byte(address, value, pc)

    def drain_events(self) -> list[ContractEvent]:
        raw = self._impl.drain_events()
        events: list[ContractEvent] = []
        for entry in raw:
            kind = str(entry.get("kind"))
            addr = int(entry.get("address"))
            val = int(entry.get("value"))
            pc = entry.get("pc")
            events.append(
                ContractEvent(
                    kind=kind,
                    address=addr & 0xFFFFFF,
                    value=val & 0xFF,
                    pc=int(pc) if pc is not None else None,
                )
            )
        return events

    def snapshot(self) -> ContractSnapshot:
        snap = self._impl.snapshot()
        internal = bytes(snap.get("internal", b""))
        external = bytes(snap.get("external", b""))
        imr = int(snap.get("imr", internal[0xFB] if len(internal) > 0xFB else 0))
        isr = int(snap.get("isr", internal[0xFC] if len(internal) > 0xFC else 0))
        lcd_events_raw = snap.get("lcd_events", []) or []
        lcd_events: list[ContractEvent] = []
        for entry in lcd_events_raw:
            kind = str(entry.get("kind"))
            addr = int(entry.get("address"))
            val = int(entry.get("value"))
            pc = entry.get("pc")
            lcd_events.append(
                ContractEvent(
                    kind=kind,
                    address=addr & 0xFFFFFF,
                    value=val & 0xFF,
                    pc=int(pc) if pc is not None else None,
                )
            )
        lcd_status = snap.get("lcd_status")
        if lcd_status is not None:
            try:
                lcd_status = int(lcd_status) & 0xFF
            except Exception:
                lcd_status = None
        lcd_meta_raw = snap.get("lcd_meta")
        lcd_meta = str(lcd_meta_raw) if lcd_meta_raw is not None else None
        return ContractSnapshot(
            internal=internal,
            external=external,
            external_len=int(snap.get("external_len", len(external))),
            imr=imr,
            isr=isr,
            lcd_events=tuple(lcd_events),
            lcd_status=lcd_status if isinstance(lcd_status, int) else None,
            lcd_log=tuple(
                (
                    int(entry.get("address")) & 0xFFFFFF,
                    int(entry.get("value")) & 0xFF,
                )
                for entry in snap.get("lcd_log", []) or []
            ),
            lcd_vram=bytes(snap.get("lcd_vram", b"")) if snap.get("lcd_vram") else None,
            lcd_meta=lcd_meta,
            metadata={
                "backend": "llama",
                "imr": imr,
                "isr": isr,
                "lcd_status": lcd_status,
                "irq_pending": bool(snap.get("irq_pending", False)),
                "irq_source": snap.get("irq_source"),
            },
        )

    def press_on_key(self) -> None:
        self._impl.press_on_key()

    def release_on_key(self) -> None:
        self._impl.release_on_key()

    def set_python_ranges(self, ranges: list[tuple[int, int]]) -> None:
        self._impl.set_python_ranges(ranges)

    def set_readonly_ranges(self, ranges: list[tuple[int, int]]) -> None:
        self._impl.set_readonly_ranges(ranges)

    def set_keyboard_bridge(self, enabled: bool) -> None:
        self._impl.set_keyboard_bridge(bool(enabled))

    def requires_python(self, address: int) -> bool:
        return bool(self._impl.requires_python(address))

    def configure_timer(
        self, mti_period: int, sti_period: int, *, enabled: bool = True
    ) -> None:
        self._impl.configure_timer(mti_period, sti_period, enabled=enabled)

    def tick_timers(self, steps: int = 1) -> None:
        self._impl.tick_timers(steps)


def run_vectors(
    vectors: Sequence[AccessVector], backend: ContractBackend
) -> ContractRun:
    """Apply a vector list to a backend and return its observations."""

    for step in vectors:
        if step.kind == "write":
            backend.write(step.address, step.value, step.pc)
        elif step.kind == "read":
            _ = backend.read(step.address, step.pc)
        else:
            raise ValueError(f"Unsupported vector kind: {step.kind}")
    snapshot = backend.snapshot()
    events = backend.drain_events()
    return ContractRun(events=events, snapshot=snapshot)


def run_dual(
    vectors: Sequence[AccessVector],
    *,
    python_backend: Optional[PythonContractBackend] = None,
    rust_backend: Optional[RustContractBackend] = None,
) -> dict[str, ContractRun]:
    """Drive identical vectors through Python and Rust backends."""

    py = python_backend or PythonContractBackend()
    rs = rust_backend or RustContractBackend(host_memory=py.memory)
    return {
        "python": run_vectors(vectors, py),
        "llama": run_vectors(vectors, rs),
    }
