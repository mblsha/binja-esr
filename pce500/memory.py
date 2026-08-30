"""PC-E500 memory system implementation with overlay bus integration."""

from __future__ import annotations

import inspect
from collections import deque
from typing import Optional, Callable, Dict, Tuple, Literal, Deque, Any, Iterable, List

from sc62015.pysc62015.instr.opcodes import IMEMRegisters

from .tracing import trace_dispatcher
from .tracing.perfetto_tracing import perf_trace
from .tracing.perfetto_tracing import tracer as perfetto_tracer
from .memory_bus import MemoryBus, MemoryOverlay, UnsafePreflightRead

# Import constants for accessing internal memory registers
# Define locally to avoid circular imports
INTERNAL_MEMORY_START = 0x100000
IMEM_ACCESS_HISTORY_LIMIT = 10
IMEM_OFFSET_TO_NAME: Dict[int, str] = {
    reg.value: name for name, reg in IMEMRegisters.__members__.items()
}
IMEM_TRACE_ALL = False
IMEM_TRACE_REG = ""

MEMORY_CARD_SLOT_START = 0x40000
MEMORY_CARD_SLOT_END = 0x4FFFF
PCE500_ROM_WINDOW_START = 0xC0000
PCE500_ROM_WINDOW_SIZE = 0x40000
PCE500_BASE_ROM_START = 0xE0000
PCE500_BASE_ROM_SIZE = 0x20000


def _keyboard_read_handler_arity(handler: Callable[..., int]) -> int:
    """Select the supported positional callback shape without invoking it.

    Legacy keyboard overlays may expose ``handler(address)`` while current
    handlers accept ``handler(address, cpu_pc)``.  Inspecting the signature up
    front keeps a ``TypeError`` raised by the handler body distinguishable from
    an argument-binding error, so a failing callback is never invoked twice.
    Callables without an inspectable signature follow the current two-argument
    contract and are still invoked exactly once.
    """

    try:
        signature = inspect.signature(handler)
    except (TypeError, ValueError):
        return 2

    try:
        signature.bind(0, None)
    except TypeError:
        try:
            signature.bind(0)
        except TypeError as one_arg_error:
            raise TypeError(
                "keyboard read handler must accept (address) or (address, cpu_pc)"
            ) from one_arg_error
        return 1
    return 2


def _is_lcd_region(address: int) -> bool:
    return 0x2000 <= address <= 0x200F or 0xA000 <= address <= 0xAFFF


def _trace_lcd_write(
    memory: "PCE500Memory", address: int, value: int, cpu_pc: Optional[int]
) -> None:
    if _is_lcd_region(address):
        return


def _lcd_write_wrapper(
    memory: "PCE500Memory",
    handler: Callable[[int, int, Optional[int]], None],
    address: int,
    value: int,
    cpu_pc: Optional[int],
) -> None:
    handler(address, value, cpu_pc)
    _trace_lcd_write(memory, address, value, cpu_pc)
    if False:
        pc_str = f"0x{cpu_pc:06X}" if cpu_pc is not None else "N/A"
        print(f"[LCD TRACE HANDLER] addr=0x{address:06X} val=0x{value:02X} pc={pc_str}")


_STACK_TRACE_RANGES: Optional[List[Tuple[int, int]]] = None


def _stack_trace_ranges() -> List[Tuple[int, int]]:
    return []


def _trace_stack_write(
    address: int, value: int, cpu_pc: Optional[int], *, label: str = "stack-trace"
) -> None:
    _ = (address, value, cpu_pc, label)


class PCE500Memory:
    """PC-E500 memory implementation with overlay support.

    Direct memory access implementation that replaces the original
    complex 4-layer abstraction.
    """

    def __init__(self):
        # Base 1MB external memory (0x00000-0xFFFFF)
        self.external_memory = bytearray(1024 * 1024)

        # Overlay manager
        self._bus = MemoryBus()

        # Memory card slot (0x40000..). The ROM probes this window to decide
        # whether the card path is available (e.g. toggling 0x40005).
        #
        # Default behavior matches existing harness expectations: a present,
        # writable 64KB card initialized to zeroes.
        self._card_present = True
        self._card_writable = True
        self._card_len = 65536
        self._card_data = bytearray(self._card_len)

        def _card_read(address: int, cpu_pc: Optional[int]) -> int:
            _ = cpu_pc
            if not self._card_present:
                # Empirically, the ROM's probe sequence expects that absent
                # cards do not latch writes and read back as 0.
                return 0x00
            offset = address - MEMORY_CARD_SLOT_START
            if 0 <= offset < self._card_len:
                return self._card_data[offset]
            return 0x00

        def _card_write(address: int, value: int, cpu_pc: Optional[int]) -> None:
            _ = cpu_pc
            if not self._card_present or not self._card_writable:
                return
            offset = address - MEMORY_CARD_SLOT_START
            if 0 <= offset < self._card_len:
                self._card_data[offset] = value & 0xFF

        self.add_overlay(
            MemoryOverlay(
                start=MEMORY_CARD_SLOT_START,
                end=MEMORY_CARD_SLOT_END,
                name="memory_card_slot",
                read_only=False,
                read_handler=_card_read,
                write_handler=_card_write,
                perfetto_thread="Memory_Card",
                preflight_read_handler=_card_read,
            )
        )

        # Track keyboard overlay for optimization
        self._keyboard_overlay: Optional[MemoryOverlay] = None
        self._keyboard_read_arity = 2
        self._emulator: Optional[Any] = None

        # Perfetto tracing
        self.perfetto_enabled = False

        # Reference to CPU emulator for accessing internal memory registers
        self.cpu = None

        # Track IMEMRegisters access for debugging.
        # Each entry keeps a bounded deque of ``(PC, count)`` tuples to avoid
        # repeated list reallocations when maintaining a fixed-size history.
        self.imem_access_tracking: Dict[str, Dict[str, Deque[Tuple[int, int]]]] = {}

        # Reference to emulator for tracking counters
        self._emulator = None

        # Reference to LCD controller
        self._lcd_controller = None
        self._keyboard_bridge_enabled = False

        # Callback for IMEM register access tracking
        self._imem_access_callback: Optional[Callable[[int, str, str, int], None]] = (
            None
        )
        self._suppress_llama_sync = 0
        self._defer_llama_sync = 0
        self._deferred_llama_writes: List[Tuple[int, int]] = []
        self._perf_tracer = None
        # IMR read diagnostics
        self._imr_read_zero_count = 0
        self._imr_read_nonzero_count = 0
        self._imr_cache_value: Optional[int] = None

    def set_memory_card_present(self, present: bool) -> None:
        """Enable/disable memory card emulation.

        When disabled, reads in the card window return 0 and writes are ignored.
        """

        self._card_present = bool(present)

    def load_memory_card(
        self,
        card_data: bytes,
        card_size: int,
        *,
        writable: bool = True,
    ) -> None:
        """Load a memory card (8KB, 16KB, 32KB, or 64KB) into the card slot."""

        size_map = {
            8192: 8192,
            16384: 16384,
            32768: 32768,
            65536: 65536,
        }
        if card_size not in size_map:
            raise ValueError(f"Invalid memory card size: {card_size}")

        self._card_len = size_map[card_size]
        self._card_data = bytearray(self._card_len)
        self._card_data[: min(len(card_data), self._card_len)] = card_data[
            : min(len(card_data), self._card_len)
        ]
        self._card_present = True
        self._card_writable = bool(writable)

    def set_imem_access_callback(
        self, callback: Callable[[int, str, str, int], None]
    ) -> None:
        """Set callback for IMEM register access notifications.

        Args:
            callback: Function(pc, reg_name, access_type, value) to call on IMEM access
        """
        self._imem_access_callback = callback

    def _track_imem_access(
        self,
        offset: int,
        access_type: Literal["read", "write"],
        value: int,
        effective_pc: Optional[int],
    ) -> Optional[str]:
        """Record IMEM register access counts and notify listeners."""

        if effective_pc is None:
            return None

        reg_name = IMEM_OFFSET_TO_NAME.get(offset)
        if reg_name is None:
            return None

        tracking = self.imem_access_tracking.setdefault(
            reg_name,
            {
                "reads": deque(maxlen=IMEM_ACCESS_HISTORY_LIMIT),
                "writes": deque(maxlen=IMEM_ACCESS_HISTORY_LIMIT),
            },
        )
        history = tracking["reads" if access_type == "read" else "writes"]

        if history and history[-1][0] == effective_pc:
            history[-1] = (effective_pc, history[-1][1] + 1)
        else:
            history.append((effective_pc, 1))

        if self._imem_access_callback:
            self._imem_access_callback(
                effective_pc,
                reg_name,
                access_type,
                value & 0xFF,
            )
        if IMEM_TRACE_ALL or (IMEM_TRACE_REG and reg_name == IMEM_TRACE_REG):
            pc_str = f"0x{effective_pc:06X}" if effective_pc is not None else "N/A"
            print(
                f"[imem-track] pc={pc_str} reg={reg_name} type={access_type} value=0x{value & 0xFF:02X}"
            )

        return reg_name

    def _record_perfetto_write(
        self,
        *,
        address: int,
        value: int,
        effective_pc: Optional[int],
        space: str,
        size: int = 1,
    ) -> None:
        """Emit a Perfetto instant event for a memory write when tracing is active."""

        tracer = getattr(self, "_perf_tracer", None)
        if tracer is None or not getattr(tracer, "instant", None):
            return

        payload: Dict[str, Any] = {
            "backend": "python",
            "address": address & 0xFFFFFF,
            "value": value & 0xFF,
            "space": space,
            "size": size,
        }
        if effective_pc is not None:
            payload["pc"] = effective_pc & 0xFFFFFF
        emulator = getattr(self, "_emulator", None)
        if emulator is not None:
            op_index = getattr(emulator, "_active_trace_instruction", None)
            if op_index is not None:
                payload["op_index"] = op_index
            if getattr(emulator, "_new_trace_enabled", False) and hasattr(
                emulator, "_next_memory_trace_units"
            ):
                units = emulator._next_memory_trace_units()
                setter = getattr(tracer, "set_manual_clock_units", None)
                if units is not None and callable(setter):
                    setter(units)

        track = "IWrites" if "internal" in space.lower() else "EWrites"
        tracer.instant(track, f"Write@0x{address & 0xFFFFFF:06X}", payload)

    def _record_imr_read(self, value: int, effective_pc: Optional[int]) -> None:
        """Log IMR reads for debugging/perfetto correlation."""

        tracer = getattr(self, "_perf_tracer", None)
        if tracer is None:
            return

        try:
            if (value & 0xFF) == 0:
                self._imr_read_zero_count += 1
                tracer.counter("IMR", "IMR_ReadZero", self._imr_read_zero_count)
            else:
                self._imr_read_nonzero_count += 1
                tracer.counter("IMR", "IMR_ReadNonZero", self._imr_read_nonzero_count)

            tracer.instant(
                "IMR",
                "IMR_Read",
                {
                    "pc": effective_pc & 0xFFFFFF if effective_pc is not None else None,
                    "value": value & 0xFF,
                    "zero": int((value & 0xFF) == 0),
                },
            )
        except Exception:
            pass

    def _trace_write_event(
        self,
        perfetto_thread: Optional[str],
        address: int,
        value: int,
        cpu_pc: Optional[int],
        overlay_name: Optional[str] = None,
    ) -> None:
        """Emit a legacy trace-dispatcher instant for overlay writes."""

        thread = perfetto_thread or "Memory"
        trace_dispatcher.record_instant(
            thread,
            "MemoryWrite",
            {
                "addr": f"0x{address & 0xFFFFFF:06X}",
                "value": f"0x{value & 0xFF:02X}",
                "pc": f"0x{cpu_pc & 0xFFFFFF:06X}" if cpu_pc is not None else "N/A",
                "overlay": overlay_name or thread,
            },
        )

    def _maybe_sync_llama_host_write(
        self, address: int, value: int, cpu_pc: Optional[int]
    ) -> None:
        """Mirror a committed host write into the active LLAMA backend.

        CPU-originated writes carry ``cpu_pc`` and are already visible to the
        native execution path. Host-originated writes do not, so silently
        losing this notification would leave the Python and Rust memories
        divergent.
        """

        emulator = getattr(self, "_emulator", None)
        facade_cpu = getattr(emulator, "cpu", None) if emulator else None
        cpu = facade_cpu or getattr(self, "cpu", None)
        if cpu is None or getattr(cpu, "backend", None) != "llama":
            return
        if cpu_pc is not None:
            return
        canonical_address = address & 0xFFFFFF
        if canonical_address >= INTERNAL_MEMORY_START:
            canonical_address = INTERNAL_MEMORY_START + (
                (canonical_address - INTERNAL_MEMORY_START) & 0xFF
            )
        else:
            canonical_address &= 0xFFFFF
        if self._suppress_llama_sync > 0:
            if self._defer_llama_sync > 0:
                self._deferred_llama_writes.append((canonical_address, value & 0xFF))
            return
        backend_impl = cpu.unwrap() if hasattr(cpu, "unwrap") else cpu
        notifier = getattr(backend_impl, "notify_host_write", None) or getattr(
            cpu, "notify_host_write", None
        )
        if notifier is None:
            raise RuntimeError(
                "LLAMA backend does not expose the required notify_host_write hook"
            )
        notifier(canonical_address, value & 0xFF)

    def flush_deferred_llama_host_writes(self) -> None:
        """Apply host writes deferred across a re-entrant LLAMA callback.

        The caller must invoke this only after the native PyO3 CPU method has
        returned and released its mutable borrow. Ordering is preserved because
        device-register writes can be edge-sensitive.
        """

        pending = self._deferred_llama_writes
        self._deferred_llama_writes = []
        for index, (address, value) in enumerate(pending):
            try:
                self._maybe_sync_llama_host_write(address, value, None)
            except Exception:
                self._deferred_llama_writes[0:0] = pending[index:]
                raise

    def discard_deferred_llama_host_writes(self) -> None:
        """Discard notifications superseded by a completed machine reset."""

        self._deferred_llama_writes.clear()

    @perf_trace("Memory", sample_rate=100)
    def read_byte(self, address: int, cpu_pc: Optional[int] = None) -> int:
        """Read a byte from memory.

        Args:
            address: Memory address to read from
            cpu_pc: Optional CPU program counter for tracing context
        """
        # Track memory reads
        if self._emulator:
            self._emulator.memory_read_count += 1

        address &= 0xFFFFFF  # 24-bit address space

        # Check for SC62015 internal memory (0x100000-0x1000FF)
        if address >= 0x100000:
            offset = (address - 0x100000) & 0xFF
            effective_pc = cpu_pc if cpu_pc is not None else self._get_current_pc()
            # Fast path for keyboard overlay
            if self._keyboard_overlay and 0xF0 <= offset <= 0xF2:
                value: int = 0
                effective_pc = cpu_pc if cpu_pc is not None else self._get_current_pc()
                if self._keyboard_overlay.read_handler:
                    handler = self._keyboard_overlay.read_handler
                    if self._keyboard_read_arity == 1:
                        value = int(handler(address)) & 0xFF
                    else:
                        value = int(handler(address, effective_pc)) & 0xFF
                    # Optional override for KIL to force a test value (diagnostic).
                    if offset == IMEMRegisters.KIL:
                        pass
                elif self._keyboard_overlay.data:
                    overlay_offset = address - self._keyboard_overlay.start
                    if overlay_offset < len(self._keyboard_overlay.data):
                        value = int(self._keyboard_overlay.data[overlay_offset]) & 0xFF
                    else:
                        value = 0x00
                else:
                    value = 0x00

                # Track IMEMRegisters reads (ensure disasm trace sees KOL/KOH/KIL)
                self._track_imem_access(offset, "read", value, effective_pc)
                if offset in (IMEMRegisters.KOL, IMEMRegisters.KOH, IMEMRegisters.KIL):
                    tracer = getattr(self, "_perf_tracer", None)
                    if tracer is not None:
                        pc_val = (
                            effective_pc
                            if effective_pc is not None
                            else self._get_current_pc()
                        )
                        tracer.instant(
                            "KIO",
                            f"read@{IMEM_OFFSET_TO_NAME.get(offset, f'0x{offset:02X}')}",
                            {
                                "pc": pc_val & 0xFFFFFF if pc_val is not None else None,
                                "value": value,
                                "offset": offset,
                            },
                        )
                    # Emit a generic KIO read marker to match LLAMA perfetto naming.
                    try:
                        if tracer is not None:
                            tracer.instant(
                                "KIO",
                                "read@KIO",
                                {
                                    "pc": pc_val & 0xFFFFFF
                                    if pc_val is not None
                                    else None,
                                    "value": value,
                                    "offset": offset,
                                },
                            )
                    except Exception:
                        pass
                    # Also emit via dispatcher in case tracer is absent
                    trace_dispatcher.record_instant(
                        "KIO",
                        f"read@{IMEM_OFFSET_TO_NAME.get(offset, f'0x{offset:02X}')}",
                        {
                            "pc": f"0x{effective_pc & 0xFFFFFF:06X}"
                            if effective_pc is not None
                            else "N/A",
                            "value": f"0x{value & 0xFF:02X}",
                            "offset": f"0x{offset:02X}",
                        },
                    )
                    trace_dispatcher.record_instant(
                        "KIO",
                        "read@KIO",
                        {
                            "pc": f"0x{effective_pc & 0xFFFFFF:06X}"
                            if effective_pc is not None
                            else "N/A",
                            "value": f"0x{value & 0xFF:02X}",
                            "offset": f"0x{offset:02X}",
                        },
                    )
                    if offset == IMEMRegisters.KIL:
                        pass
                return value

            # Normal internal memory access (most common case)
            internal_offset = len(self.external_memory) - 256 + offset
            value = self.external_memory[internal_offset]
            if offset == IMEMRegisters.IMR:
                if False and self._imr_cache_value is not None:
                    value = int(self._imr_cache_value) & 0xFF
                # Debug hook: log IMR reads to diagnose IMR/IRM coherence.
                self._record_imr_read(value, effective_pc)
            elif offset in (IMEMRegisters.KOL, IMEMRegisters.KOH, IMEMRegisters.KIL):
                tracer = getattr(self, "_perf_tracer", None)
                if tracer is not None:
                    pc_val = (
                        effective_pc
                        if effective_pc is not None
                        else self._get_current_pc()
                    )
                    tracer.instant(
                        "KIO",
                        f"read@{IMEM_OFFSET_TO_NAME.get(offset, f'0x{offset:02X}')}",
                        {
                            "pc": pc_val & 0xFFFFFF if pc_val is not None else None,
                            "value": value,
                            "offset": offset,
                        },
                    )
                    # Emit a generic KIO read marker to align with LLAMA trace naming.
                    try:
                        tracer.instant(
                            "KIO",
                            "read@KIO",
                            {
                                "pc": pc_val & 0xFFFFFF if pc_val is not None else None,
                                "value": value,
                                "offset": offset,
                            },
                        )
                    except Exception:
                        pass
                # Also emit via dispatcher in case matrix hooks are bypassed.
                trace_dispatcher.record_instant(
                    "KIO",
                    f"read@{IMEM_OFFSET_TO_NAME.get(offset, f'0x{offset:02X}')}",
                    {
                        "pc": f"0x{effective_pc & 0xFFFFFF:06X}"
                        if effective_pc is not None
                        else "N/A",
                        "value": f"0x{value & 0xFF:02X}",
                        "offset": f"0x{offset:02X}",
                    },
                )
                trace_dispatcher.record_instant(
                    "KIO",
                    "read@KIO",
                    {
                        "pc": f"0x{effective_pc & 0xFFFFFF:06X}"
                        if effective_pc is not None
                        else "N/A",
                        "value": f"0x{value & 0xFF:02X}",
                        "offset": f"0x{offset:02X}",
                    },
                )
                # Log explicitly when KIL is read to capture PC/value even if tracer is absent.
                if offset == IMEMRegisters.KIL:
                    pass

            # Track IMEMRegisters reads
            self._track_imem_access(offset, "read", value, effective_pc)

            return value

        # External memory space (0x00000-0xFFFFF)
        address &= 0xFFFFF

        result = self._bus.read(address, cpu_pc)
        if result is not None:
            return result.value

        # Default to external memory
        return self.external_memory[address]

    def peek_byte_for_preflight(
        self, address: int, cpu_pc: Optional[int] = None
    ) -> int:
        """Read a byte without counters, tracing, or device side effects.

        Scheduling preflight uses this path to validate an instruction before
        timers or peripherals advance.  A dynamic overlay without an explicit
        safe callback is rejected instead of being invoked twice or bypassed.
        """

        address &= 0xFFFFFF
        if address >= INTERNAL_MEMORY_START:
            offset = (address - INTERNAL_MEMORY_START) & 0xFF
            if self._keyboard_overlay and 0xF0 <= offset <= 0xF2:
                overlay = self._keyboard_overlay
                if overlay.preflight_read_handler is not None:
                    return int(overlay.preflight_read_handler(address, cpu_pc)) & 0xFF
                if overlay.read_handler is not None:
                    raise UnsafePreflightRead(
                        "side-effect-free preflight read unavailable for overlay "
                        f"{overlay.name!r} at 0x{address:06X}"
                    )
                if overlay.data is not None:
                    overlay_offset = address - overlay.start
                    if 0 <= overlay_offset < len(overlay.data):
                        return int(overlay.data[overlay_offset]) & 0xFF
                return 0

            return int(self.external_memory[-256 + offset]) & 0xFF

        address &= 0xFFFFF
        result = self._bus.peek_for_preflight(address, cpu_pc)
        if result is not None:
            return result.value
        return int(self.external_memory[address]) & 0xFF

    @perf_trace("Memory", sample_rate=100)
    def write_byte(
        self, address: int, value: int, cpu_pc: Optional[int] = None
    ) -> None:
        """Write a byte to memory.

        Args:
            address: Memory address to write to
            value: Byte value to write
            cpu_pc: Optional CPU program counter for tracing context
        """
        # Track memory writes
        if self._emulator:
            self._emulator.memory_write_count += 1

        address &= 0xFFFFFF  # 24-bit address space
        value &= 0xFF
        effective_pc = cpu_pc if cpu_pc is not None else self._get_current_pc()
        if False and 0x0BFC60 <= address <= 0x0BFC7F and effective_pc is not None:
            print(
                f"[ext-write] pc=0x{effective_pc:06X} addr=0x{address:06X} value=0x{value:02X}"
            )
        if getattr(self, "_emulator", None) is not None and getattr(
            self._emulator, "_llama_pure_lcd", False
        ):
            if (
                effective_pc is not None
                and 0x0BFCE0 <= address <= 0x0BFCFF
                and 0x0F2040 <= (effective_pc & 0xFFFFFF) <= 0x0F205F
            ):
                notify = getattr(self._emulator, "notify_lcd_interrupt", None)
                if callable(notify):
                    notify(address, value, effective_pc)

        # Check for SC62015 internal memory (0x100000-0x1000FF)
        if address >= 0x100000:
            offset = (address - 0x100000) & 0xFF
            # Fast path for keyboard overlay
            if self._keyboard_overlay and 0xF0 <= offset <= 0xF2:
                if self._keyboard_overlay.write_handler:
                    self._keyboard_overlay.write_handler(address, value, effective_pc)
                    # Track IMEMRegisters writes (ensure disasm trace sees KOL/KOH/KIL)
                    self._track_imem_access(offset, "write", value, effective_pc)
                    # Add tracing for write_handler overlays
                    if self.perfetto_enabled:
                        trace_data = {
                            "addr": f"0x{address:06X}",
                            "value": f"0x{value:02X}",
                        }
                        if effective_pc is not None:
                            trace_data["pc"] = f"0x{effective_pc:06X}"
                        trace_data["overlay"] = self._keyboard_overlay.name
                        trace_dispatcher.record_instant(
                            self._keyboard_overlay.perfetto_thread,
                            "KeyboardOverlayWrite",
                            trace_data,
                        )
                    self._record_perfetto_write(
                        address=address,
                        value=value,
                        effective_pc=effective_pc,
                        space=self._keyboard_overlay.name or "keyboard_overlay",
                    )
                    _trace_lcd_write(self, address, value, effective_pc)
                    self._maybe_sync_llama_host_write(address, value, cpu_pc)
                    return
                elif self._keyboard_overlay.read_only:
                    # Silently ignore writes to read-only overlays
                    return
                elif self._keyboard_overlay.data and isinstance(
                    self._keyboard_overlay.data, bytearray
                ):
                    # Write to writable overlay data
                    overlay_offset = address - self._keyboard_overlay.start
                    if overlay_offset < len(self._keyboard_overlay.data):
                        self._keyboard_overlay.data[overlay_offset] = value
                        self._record_perfetto_write(
                            address=address,
                            value=value,
                            effective_pc=effective_pc,
                            space=self._keyboard_overlay.name or "keyboard_overlay",
                        )
                        self._maybe_sync_llama_host_write(address, value, cpu_pc)
                        return

            # Normal internal memory write (most common case)
            # Internal memory stored at end of external_memory for compatibility
            internal_offset = len(self.external_memory) - 256 + offset
            prev_val = int(self.external_memory[internal_offset])
            self.external_memory[internal_offset] = value
            if offset == IMEMRegisters.IMR:
                self._imr_cache_value = value

            # Track IMEMRegisters writes
            reg_name = self._track_imem_access(offset, "write", value, effective_pc)

            # Interrupt bit watch for IMR/ISR
            if reg_name in ("IMR", "ISR"):
                try:
                    if (
                        self._emulator
                        and hasattr(self._emulator, "_record_irq_bit_watch")
                        and effective_pc is not None
                    ):
                        self._emulator._record_irq_bit_watch(
                            reg_name,
                            prev_val & 0xFF,
                            value & 0xFF,
                            int(effective_pc),
                        )
                except Exception:
                    pass

            if self.perfetto_enabled:
                # Get current BP value from internal memory if CPU is available
                bp_value = "N/A"
                if self.cpu:
                    try:
                        bp_addr = INTERNAL_MEMORY_START + IMEMRegisters.BP
                        bp_value = f"0x{self.cpu.memory.read_byte(bp_addr):02X}"
                    except Exception:
                        bp_value = "N/A"

                # Check if this offset corresponds to a known internal memory register
                imem_name = IMEM_OFFSET_TO_NAME.get(offset, "N/A")

                trace_dispatcher.record_instant(
                    "Memory_Internal",
                    "MemoryWrite",
                    {
                        "offset": f"0x{offset:02X}",
                        "value": f"0x{value:02X}",
                        "pc": f"0x{effective_pc:06X}"
                        if effective_pc is not None
                        else "N/A",
                        "bp": bp_value,
                        "imem_name": imem_name,
                        "size": "1",
                    },
                )
            self._record_perfetto_write(
                address=address,
                value=value,
                effective_pc=effective_pc,
                space="internal",
            )
            _trace_lcd_write(self, address, value, effective_pc)
            _trace_stack_write(address, value, effective_pc)
            self._maybe_sync_llama_host_write(address, value, cpu_pc)
            return

        # External memory space (0x00000-0xFFFFF)
        address &= 0xFFFFF

        write_result = self._bus.write(address, value, effective_pc)
        if write_result is not None:
            if self.perfetto_enabled:
                self._trace_write_event(
                    write_result.overlay.perfetto_thread,
                    address,
                    value,
                    effective_pc,
                    write_result.overlay.name,
                )
            self._record_perfetto_write(
                address=address,
                value=value,
                effective_pc=effective_pc,
                space=write_result.overlay.name,
            )
            _trace_stack_write(address, value, effective_pc)
            # A handler returning normally or a writable data overlay is the
            # bus contract's only success signal. Read-only overlays also
            # return a WriteResult, but intentionally ignore the write and
            # therefore must not alter the native mirror.
            overlay = write_result.overlay
            if overlay.write_handler is not None or (
                overlay.data is not None and not overlay.read_only
            ):
                self._maybe_sync_llama_host_write(address, value, cpu_pc)
            return

        # Default to external memory
        self.external_memory[address] = value
        _trace_stack_write(address, value, effective_pc)
        self._maybe_sync_llama_host_write(address, value, cpu_pc)

        # Perfetto tracing for all writes
        if self.perfetto_enabled:
            trace_data = {"addr": f"0x{address:06X}", "value": f"0x{value:02X}"}
            if effective_pc is not None:
                trace_data["pc"] = f"0x{effective_pc:06X}"
            trace_dispatcher.record_instant(
                "Memory_External",
                "MemoryWrite",
                trace_data,
            )
        self._record_perfetto_write(
            address=address,
            value=value,
            effective_pc=effective_pc,
            space="external",
        )

    @perf_trace("Memory", sample_rate=100)
    def read_word(self, address: int) -> int:
        """Read 16-bit word (little-endian)."""
        low = self.read_byte(address)
        high = self.read_byte(address + 1)
        return low | (high << 8)

    @perf_trace("Memory", sample_rate=100)
    def write_word(
        self, address: int, value: int, cpu_pc: Optional[int] = None
    ) -> None:
        """Write 16-bit word (little-endian)."""
        self.write_byte(address, value & 0xFF, cpu_pc)
        self.write_byte(address + 1, (value >> 8) & 0xFF, cpu_pc)

    def read_long(self, address: int) -> int:
        """Read 24-bit long (little-endian)."""
        low = self.read_byte(address)
        mid = self.read_byte(address + 1)
        high = self.read_byte(address + 2)
        return low | (mid << 8) | (high << 16)

    def write_long(
        self, address: int, value: int, cpu_pc: Optional[int] = None
    ) -> None:
        """Write 24-bit long (little-endian)."""
        self.write_byte(address, value & 0xFF, cpu_pc)
        self.write_byte(address + 1, (value >> 8) & 0xFF, cpu_pc)
        self.write_byte(address + 2, (value >> 16) & 0xFF, cpu_pc)

    def add_overlay(self, overlay: MemoryOverlay) -> None:
        """Add a memory overlay."""
        is_keyboard_overlay = overlay.start >= 0x1000F0 and overlay.end <= 0x1000F2
        keyboard_read_arity = (
            _keyboard_read_handler_arity(overlay.read_handler)
            if is_keyboard_overlay and overlay.read_handler is not None
            else 2
        )
        self._bus.add_overlay(overlay)

        # Track keyboard overlay for fast access
        if is_keyboard_overlay:
            self._keyboard_overlay = overlay
            self._keyboard_read_arity = keyboard_read_arity

    def remove_overlay(self, name: str) -> None:
        """Remove overlay by name."""
        if self._keyboard_overlay and self._keyboard_overlay.name == name:
            self._keyboard_overlay = None
            self._keyboard_read_arity = 2

        self._bus.remove_overlay(name)

    @property
    def overlays(self) -> Tuple[MemoryOverlay, ...]:
        """Compatibility view of overlays for legacy code paths."""
        return tuple(self._bus.iter_overlays())

    # ------------------------------------------------------------------ #
    # LLAMA backend helpers

    def export_flat_memory(
        self,
    ) -> Tuple[bytes, Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int], ...]]:
        """Return a flattened view plus Python-backed + read-only ranges."""

        blob = bytearray(self.external_memory)
        blob_len = len(blob)
        fallback: List[Tuple[int, int]] = []
        readonly: List[Tuple[int, int]] = []
        for overlay in self._bus.iter_overlays():
            start = overlay.start & 0xFFFFFF
            end = overlay.end & 0xFFFFFF
            # Internal-memory overlays (>= INTERNAL_MEMORY_START) always rely on Python.
            if start >= INTERNAL_MEMORY_START:
                fallback.append((start, end))
                continue

            if overlay.data is not None and start < blob_len:
                # Copy as much of the overlay payload as fits in the flattened blob.
                max_len = min(end - start + 1, blob_len - start, len(overlay.data))
                if max_len > 0:
                    blob[start : start + max_len] = overlay.data[:max_len]
            if overlay.read_only:
                clamped_end = min(end, INTERNAL_MEMORY_START - 1)
                readonly.append((start, clamped_end))

            needs_python = (
                overlay.read_handler is not None
                or overlay.write_handler is not None
                or overlay.data is None
            )
            if needs_python:
                fallback.append((start, end))

        return bytes(blob), tuple(fallback), tuple(readonly)

    def apply_external_writes(self, writes: Iterable[Tuple[int, int]]) -> None:
        """Apply external-memory writes that originated from the LLAMA backend."""
        self._suppress_llama_sync += 1
        try:
            for address, value in writes:
                self.write_byte(address & 0xFFFFFF, value & 0xFF)
        finally:
            self._suppress_llama_sync = max(0, self._suppress_llama_sync - 1)

    def apply_internal_writes(self, writes: Iterable[Tuple[int, int]]) -> None:
        """Apply internal-memory writes (IMEM registers) from the LLAMA backend."""
        self._suppress_llama_sync += 1
        try:
            for address, value in writes:
                self.write_byte(address & 0xFFFFFF, value & 0xFF)
        finally:
            self._suppress_llama_sync = max(0, self._suppress_llama_sync - 1)

    def load_rom(self, rom_data: bytes, *, start_address: Optional[int] = None) -> None:
        """Load a PC-E500 ROM payload into the read-only CE2 window.

        The original 128 KiB ROM occupies ``0xE0000..0xFFFFF``.  Extended
        256 KiB images occupy the complete ``0xC0000..0xFFFFF`` CE2 window.
        Full 1 MiB captures are accepted by selecting their top 256 KiB.

        Smaller synthetic payloads retain the historical ``0xC0000`` default;
        tests that need another placement can pass ``start_address``.
        """
        if not rom_data:
            raise ValueError("ROM payload is empty")

        payload = bytes(rom_data)
        if len(payload) > PCE500_ROM_WINDOW_SIZE:
            payload = payload[-PCE500_ROM_WINDOW_SIZE:]

        if start_address is None:
            start_address = (
                PCE500_BASE_ROM_START
                if len(payload) == PCE500_BASE_ROM_SIZE
                else PCE500_ROM_WINDOW_START
            )

        end_address = start_address + len(payload)
        rom_window_end = PCE500_ROM_WINDOW_START + PCE500_ROM_WINDOW_SIZE
        if not (
            PCE500_ROM_WINDOW_START <= start_address < rom_window_end
            and end_address <= rom_window_end
        ):
            raise ValueError(
                "ROM payload does not fit the PC-E500 CE2 window: "
                f"start=0x{start_address:05X}, size=0x{len(payload):X}"
            )

        # Keep the complete CE2 select read-only while placing the actual ROM
        # bytes at their variant-specific address.  Unpopulated bytes read as
        # zero, matching the pre-existing flat-memory default.
        window = bytearray(PCE500_ROM_WINDOW_SIZE)
        window_offset = start_address - PCE500_ROM_WINDOW_START
        window[window_offset : window_offset + len(payload)] = payload

        # Remove any existing ROM overlay
        self.remove_overlay("internal_rom")

        # Add new ROM overlay
        self.add_overlay(
            MemoryOverlay(
                start=PCE500_ROM_WINDOW_START,
                end=0xFFFFF,
                name="internal_rom",
                data=window,
                read_only=True,
                perfetto_thread="Memory_ROM",
            )
        )

    # NOTE: load_memory_card is implemented near __init__ to share the slot overlay.

    def add_ram(self, start_address: int, size: int, name: str) -> None:
        """Add RAM expansion as overlay."""
        # Create writable overlay
        self.add_overlay(
            MemoryOverlay(
                start=start_address,
                end=start_address + size - 1,
                name=name,
                data=bytearray(size),
                read_only=False,
                perfetto_thread="Memory_RAM",
            )
        )

    def add_rom(self, start_address: int, rom_data: bytes, name: str) -> None:
        """Add ROM at arbitrary address as overlay."""
        self.add_overlay(
            MemoryOverlay(
                start=start_address,
                end=start_address + len(rom_data) - 1,
                name=name,
                data=bytearray(rom_data),
                read_only=True,
                perfetto_thread="Memory_ROM",
            )
        )

    def set_lcd_controller(
        self, lcd_controller, *, enable_overlay: bool = True
    ) -> None:
        """Set LCD controller and optionally add memory-mapped I/O overlay."""
        self._lcd_controller = lcd_controller

        # Pass CPU reference to LCD controller if available
        if self.cpu and hasattr(lcd_controller, "set_cpu"):
            lcd_controller.set_cpu(self.cpu)

        if not enable_overlay:
            # Native path: forward writes to the overlay hooks so Python-side
            # peripherals (keyboard/LCD) and IMEM sync stay active even without
            # the MemoryOverlay objects.
            def _llama_lcd_write(addr: int, val: int, pc: Optional[int]) -> None:
                _lcd_write_wrapper(self, lcd_controller.write, addr, val, pc)

            setattr(self, "_llama_lcd_write", _llama_lcd_write)
            return

        def lcd_write_handler(addr: int, val: int, pc: Optional[int]) -> None:
            _lcd_write_wrapper(self, lcd_controller.write, addr, val, pc)

        # LCD controllers at 0x2000 and 0xA000
        self.add_overlay(
            MemoryOverlay(
                start=0x2000,
                end=0x200F,
                name="lcd_controller_low",
                read_only=False,
                read_handler=lambda addr, pc: lcd_controller.read(addr, pc),
                write_handler=lcd_write_handler,
                perfetto_thread="Display",
            )
        )
        self.add_overlay(
            MemoryOverlay(
                start=0xA000,
                end=0xAFFF,
                name="lcd_controller",
                read_only=False,
                read_handler=lambda addr, pc: lcd_controller.read(addr, pc),
                write_handler=lcd_write_handler,
                perfetto_thread="Display",
            )
        )

    def set_keyboard_handler(
        self,
        read_callback: Callable[[int, Optional[int]], int],
        write_callback: Callable[[int, int, Optional[int]], None],
        *,
        enable_overlay: bool = True,
    ) -> None:
        """Configure keyboard register handlers when using the Python backend."""

        self.remove_overlay("keyboard_io")
        if not enable_overlay:
            self._keyboard_bridge_enabled = True
            return

        self._keyboard_bridge_enabled = False
        self.add_overlay(
            MemoryOverlay(
                start=INTERNAL_MEMORY_START + IMEMRegisters.KOL,
                end=INTERNAL_MEMORY_START + IMEMRegisters.KIL,
                name="keyboard_io",
                read_only=False,
                read_handler=read_callback,
                write_handler=write_callback,
                perfetto_thread="I/O",
            )
        )

    def reset(self) -> None:
        """Reset all RAM to zero."""
        # Reset external memory (including internal memory at the end)
        self.external_memory[:] = bytes(len(self.external_memory))
        self._imr_read_zero_count = 0
        self._imr_read_nonzero_count = 0
        self._imr_cache_value = None

        # Reset any writable overlays
        for overlay in self._bus.iter_overlays():
            if not overlay.read_only and overlay.data:
                overlay.data[:] = bytes(len(overlay.data))

    def get_memory_info(self) -> str:
        """Get information about memory configuration."""
        lines = ["Memory Configuration:"]
        lines.append("  Base: 1MB external memory (0x00000-0xFFFFF)")
        lines.append("  Internal: 256B internal memory (0x100000-0x1000FF)")

        overlays = tuple(self._bus.iter_overlays())
        if overlays:
            lines.append(f"\nOverlays ({len(overlays)}):")
            for overlay in sorted(overlays, key=lambda o: o.start):
                size = overlay.end - overlay.start + 1
                lines.append(
                    f"  {overlay.name}: 0x{overlay.start:05X}-0x{overlay.end:05X} ({size} bytes, {'R/O' if overlay.read_only else 'R/W'})"
                )

        return "\n".join(lines)

    def get_internal_memory_bytes(self) -> bytes:
        """Get internal memory (256 bytes) as raw bytes."""
        # Internal memory is stored in the last 256 bytes of external_memory
        return bytes(self.external_memory[-256:])

    def wait_cycles(self, cycles: int) -> None:
        """Advance emulator cycle/timer state for LLAMA WAIT handling.

        The Rust LLAMA core can delegate WAIT-loop timing to the Python scheduler via this
        hook so that ISR bits and Perfetto traces stay aligned with the Python backend.
        A present hook is part of the execution contract: configuration and scheduler
        failures must propagate so the CPU cannot clear I and advance PC without time
        actually passing.
        """

        emu = getattr(self, "_emulator", None)
        if emu is None:
            raise RuntimeError("WAIT timing requires an attached emulator scheduler")
        try:
            cycles_i = int(cycles)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid WAIT cycle count {cycles!r}") from exc
        if cycles_i <= 0:
            raise ValueError(f"WAIT cycle count must be positive, got {cycles_i}")
        # LLAMA invokes this hook while its PyO3 CPU is already mutably
        # borrowed. Timer/keyboard writes performed by the Python scheduler are
        # therefore authoritative host-side writes for this callback; trying to
        # notify the same native CPU re-entrantly raises ``Already borrowed``.
        # Suppress only that recursive notification while preserving ordinary
        # host-write synchronization before and after the callback.
        self._suppress_llama_sync += 1
        self._defer_llama_sync += 1
        try:
            emu._simulate_wait(cycles_i)
        finally:
            self._defer_llama_sync = max(0, self._defer_llama_sync - 1)
            self._suppress_llama_sync = max(0, self._suppress_llama_sync - 1)

    def set_perfetto_enabled(self, enabled: bool) -> None:
        """Enable or disable Perfetto tracing for memory operations."""
        self.perfetto_enabled = enabled

    def set_cpu(self, cpu) -> None:
        """Set reference to CPU emulator for accessing internal memory registers."""
        self.cpu = cpu

        # Also pass CPU to LCD controller if already set
        if self._lcd_controller and hasattr(self._lcd_controller, "set_cpu"):
            self._lcd_controller.set_cpu(cpu)

    def _get_current_pc(self) -> Optional[int]:
        """Get current PC from CPU if available."""
        if self.cpu and hasattr(self.cpu, "regs"):
            try:
                from sc62015.pysc62015.emulator import RegisterName

                return self.cpu.regs.get(RegisterName.PC)
            except Exception:
                pass
        return None

    def get_imem_access_tracking(self) -> Dict[str, Dict[str, list[Tuple[int, int]]]]:
        """Get the IMEM register access tracking data.

        Returns:
            Dictionary with register names as keys and read/write lists as values
        """
        # Return a copy of the tracking data as plain lists for serialization
        return {
            reg_name: {
                access_type: list(entries) for access_type, entries in accesses.items()
            }
            for reg_name, accesses in self.imem_access_tracking.items()
        }

    def clear_imem_access_tracking(self) -> None:
        """Clear all IMEM register access tracking data."""
        self.imem_access_tracking.clear()

    def read_bytes(self, address: int, size: int) -> int:
        """Read bytes according to binja_test_mocks.eval_llil.Memory interface.

        Args:
            address: Memory address to read from
            size: Number of bytes to read

        Returns:
            Value as integer (little-endian)
        """
        result = 0
        for i in range(size):
            if address >= INTERNAL_MEMORY_START:
                byte_address = INTERNAL_MEMORY_START + (
                    (address - INTERNAL_MEMORY_START + i) & 0xFF
                )
            else:
                byte_address = (address + i) & 0xFFFFF
            byte_val = self.read_byte(byte_address)
            result |= byte_val << (i * 8)
        return result

    def write_bytes(self, size: int, address: int, value: int) -> None:
        """Write bytes according to binja_test_mocks.eval_llil.Memory interface.

        Args:
            size: Number of bytes to write
            address: Memory address to write to
            value: Value to write (as integer, little-endian)
        """
        for i in range(size):
            byte_value = (value >> (i * 8)) & 0xFF
            if address >= INTERNAL_MEMORY_START:
                byte_address = INTERNAL_MEMORY_START + (
                    (address - INTERNAL_MEMORY_START + i) & 0xFF
                )
            else:
                byte_address = (address + i) & 0xFFFFF
            self.write_byte(byte_address, byte_value)

    def set_context(self, context: dict) -> None:
        """Set context for memory operations (compatibility method)."""
        # Context is handled through cpu_pc parameter in read/write methods
        pass

    def set_perf_tracer(self, tracer) -> None:
        """Set performance tracer for SC62015 emulator integration."""
        self._perf_tracer = tracer

    def trace_kio_from_rust(
        self, offset: int, value: int, pc: Optional[int] = None
    ) -> None:
        """Mirror LLAMA-side KIO reads into the main Perfetto trace."""

        tracer = getattr(self, "_perf_tracer", None)
        if tracer is None:
            return
        payload: Dict[str, Any] = {
            "offset": offset & 0xFF,
            "value": value & 0xFF,
        }
        eff_pc = pc
        if eff_pc is None and self.cpu is not None:
            try:
                from sc62015.pysc62015.emulator import RegisterName

                eff_pc = self.cpu.regs.get(RegisterName.PC)
            except Exception:
                pass
        if eff_pc is not None:
            payload["pc"] = eff_pc & 0xFFFFFF
        emulator = getattr(self, "_emulator", None)
        if emulator is not None:
            op_index = getattr(emulator, "_active_trace_instruction", None)
            if op_index is not None:
                payload["op_index"] = op_index
            if getattr(emulator, "_new_trace_enabled", False) and hasattr(
                emulator, "_next_memory_trace_units"
            ):
                units = emulator._next_memory_trace_units()
                setter = getattr(tracer, "set_manual_clock_units", None)
                if units is not None and callable(setter):
                    setter(units)
        try:
            if hasattr(tracer, "instant"):
                tracer.instant("KIO", "read@KIO", payload)
            elif hasattr(tracer, "record_instant"):
                tracer.record_instant("KIO", "read@KIO", payload)
        except Exception:
            pass

    def trace_irq_from_rust(
        self, name: str, payload: Dict[str, Any], track: str = "irq.key"
    ) -> None:
        """Mirror LLAMA-side IRQ events into the main Perfetto trace."""

        # Also forward to the rustcore helper so both tracers stay aligned.
        try:
            from _sc62015_rustcore import record_irq_event as rust_irq_event
        except Exception:
            rust_irq_event = None

        tracer = getattr(self, "_perf_tracer", None)
        if tracer is None:
            return
        data: Dict[str, Any] = dict(payload)
        if "pc" not in data and self.cpu is not None:
            try:
                from sc62015.pysc62015.emulator import RegisterName

                data["pc"] = self.cpu.regs.get(RegisterName.PC) & 0xFFFFFF
            except Exception:
                pass
        emulator = getattr(self, "_emulator", None)
        if emulator is not None:
            op_index = getattr(emulator, "_active_trace_instruction", None)
            if op_index is not None:
                data["op_index"] = op_index
            if getattr(emulator, "_new_trace_enabled", False) and hasattr(
                emulator, "_next_memory_trace_units"
            ):
                units = emulator._next_memory_trace_units()
                setter = getattr(tracer, "set_manual_clock_units", None)
                if units is not None and callable(setter):
                    setter(units)
        try:
            if hasattr(tracer, "instant"):
                tracer.instant(track, name, data)
            elif hasattr(tracer, "record_instant"):
                tracer.record_instant(track, name, data)
            if rust_irq_event is not None and perfetto_tracer.enabled:
                rust_payload = {
                    k: int(v) if v is not None else 0 for k, v in data.items()
                }
                rust_irq_event(name, rust_payload)
        except Exception:
            pass
