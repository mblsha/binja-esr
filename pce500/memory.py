"""PC-E500 memory system implementation with overlay bus integration."""

from collections import deque
from typing import Optional, Callable, Dict, Tuple, Literal, Deque, Any

from sc62015.pysc62015.instr.opcodes import IMEMRegisters

from .trace_manager import g_tracer
from .tracing.perfetto_tracing import perf_trace
from .memory_bus import MemoryBus, MemoryOverlay

# Import constants for accessing internal memory registers
# Define locally to avoid circular imports
INTERNAL_MEMORY_START = 0x100000
IMEM_ACCESS_HISTORY_LIMIT = 10
IMEM_OFFSET_TO_NAME: Dict[int, str] = {
    reg.value: name for name, reg in IMEMRegisters.__members__.items()
}


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

        # Track keyboard overlay for optimization
        self._keyboard_overlay: Optional[MemoryOverlay] = None
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

        # Callback for IMEM register access tracking
        self._imem_access_callback: Optional[Callable[[int, str, str, int], None]] = (
            None
        )

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

        return reg_name

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
            offset = address - 0x100000
            if offset < 0x100:
                # Fast path for keyboard overlay
                if self._keyboard_overlay and 0xF0 <= offset <= 0xF2:
                    value: int
                    if self._keyboard_overlay.read_handler:
                        value = (
                            int(self._keyboard_overlay.read_handler(address, cpu_pc))
                            & 0xFF
                        )
                    elif self._keyboard_overlay.data:
                        overlay_offset = address - self._keyboard_overlay.start
                        if overlay_offset < len(self._keyboard_overlay.data):
                            value = (
                                int(self._keyboard_overlay.data[overlay_offset]) & 0xFF
                            )
                        else:
                            value = 0x00
                    else:
                        value = 0x00

                    # Track IMEMRegisters reads (ensure disasm trace sees KOL/KOH/KIL)
                    effective_pc = (
                        cpu_pc if cpu_pc is not None else self._get_current_pc()
                    )
                    self._track_imem_access(offset, "read", value, effective_pc)
                    return value

                # Normal internal memory access (most common case)
                internal_offset = len(self.external_memory) - 256 + offset
                value = self.external_memory[internal_offset]

                # Track IMEMRegisters reads
                # Get PC from CPU if not provided
                effective_pc = cpu_pc if cpu_pc is not None else self._get_current_pc()
                self._track_imem_access(offset, "read", value, effective_pc)

                return value
            raise ValueError(
                f"Invalid SC62015 internal memory address: 0x{address:06X} (offset 0x{offset:02X} >= 0x100)"
            )

        # External memory space (0x00000-0xFFFFF)
        address &= 0xFFFFF

        result = self._bus.read(address, cpu_pc)
        if result is not None:
            return result.value

        # Default to external memory
        return self.external_memory[address]

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

        # Check for SC62015 internal memory (0x100000-0x1000FF)
        if address >= 0x100000:
            offset = address - 0x100000
            if offset < 0x100:
                # Fast path for keyboard overlay
                if self._keyboard_overlay and 0xF0 <= offset <= 0xF2:
                    if self._keyboard_overlay.write_handler:
                        self._keyboard_overlay.write_handler(address, value, cpu_pc)
                        # Track IMEMRegisters writes (ensure disasm trace sees KOL/KOH/KIL)
                        effective_pc = (
                            cpu_pc if cpu_pc is not None else self._get_current_pc()
                        )
                        self._track_imem_access(offset, "write", value, effective_pc)
                        # Add tracing for write_handler overlays
                        if self.perfetto_enabled:
                            trace_data = {
                                "addr": f"0x{address:06X}",
                                "value": f"0x{value:02X}",
                            }
                            if cpu_pc is not None:
                                trace_data["pc"] = f"0x{cpu_pc:06X}"
                            trace_data["overlay"] = self._keyboard_overlay.name
                            g_tracer.trace_instant(
                                self._keyboard_overlay.perfetto_thread, "", trace_data
                            )
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
                            return

                # Normal internal memory write (most common case)
                # Internal memory stored at end of external_memory for compatibility
                internal_offset = len(self.external_memory) - 256 + offset
                prev_val = int(self.external_memory[internal_offset])
                self.external_memory[internal_offset] = value

                # Track IMEMRegisters writes
                # Get PC from CPU if not provided
                effective_pc = cpu_pc if cpu_pc is not None else self._get_current_pc()
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

                    g_tracer.trace_instant(
                        "Memory_Internal",
                        "",
                        {
                            "offset": f"0x{offset:02X}",
                            "value": f"0x{value:02X}",
                            "pc": f"0x{cpu_pc:06X}" if cpu_pc is not None else "N/A",
                            "bp": bp_value,
                            "imem_name": imem_name,
                            "size": "1",  # Always 1 for byte writes
                        },
                    )
            else:
                raise ValueError(
                    f"Invalid SC62015 internal memory address: 0x{address:06X} (offset 0x{offset:02X} >= 0x100)"
                )
            return

        # External memory space (0x00000-0xFFFFF)
        address &= 0xFFFFF

        write_result = self._bus.write(address, value, cpu_pc)
        if write_result is not None:
            if self.perfetto_enabled:
                self._trace_write_event(
                    write_result.overlay.perfetto_thread,
                    address,
                    value,
                    cpu_pc,
                    write_result.overlay.name,
                )
            return

        # Default to external memory
        self.external_memory[address] = value

        # Perfetto tracing for all writes
        if self.perfetto_enabled:
            trace_data = {"addr": f"0x{address:06X}", "value": f"0x{value:02X}"}
            if cpu_pc is not None:
                trace_data["pc"] = f"0x{cpu_pc:06X}"
            g_tracer.trace_instant("Memory_External", "", trace_data)

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
        self._bus.add_overlay(overlay)

        # Track keyboard overlay for fast access
        if overlay.start >= 0x1000F0 and overlay.end <= 0x1000F2:
            self._keyboard_overlay = overlay

    def remove_overlay(self, name: str) -> None:
        """Remove overlay by name."""
        if self._keyboard_overlay and self._keyboard_overlay.name == name:
            self._keyboard_overlay = None

        self._bus.remove_overlay(name)

    @property
    def overlays(self) -> Tuple[MemoryOverlay, ...]:
        """Compatibility view of overlays for legacy code paths."""
        return tuple(self._bus.iter_overlays())

    def load_rom(self, rom_data: bytes) -> None:
        """Load ROM as an overlay at 0xC0000."""
        # Remove any existing ROM overlay
        self.remove_overlay("internal_rom")

        # Add new ROM overlay
        self.add_overlay(
            MemoryOverlay(
                start=0xC0000,
                end=0xFFFFF,
                name="internal_rom",
                data=bytearray(rom_data),
                read_only=True,
                perfetto_thread="Memory_ROM",
            )
        )

    def load_memory_card(self, card_data: bytes, card_size: int) -> None:
        """Load a memory card (8KB, 16KB, 32KB, or 64KB) as overlay."""
        # Map size to standard sizes
        size_map = {
            8192: (0x40000, 0x41FFF, "8KB"),
            16384: (0x40000, 0x43FFF, "16KB"),
            32768: (0x40000, 0x47FFF, "32KB"),
            65536: (0x40000, 0x4FFFF, "64KB"),
        }

        if card_size not in size_map:
            raise ValueError(f"Invalid memory card size: {card_size}")

        start, end, size_str = size_map[card_size]

        # Remove any existing memory card overlay
        self.remove_overlay("memory_card")

        # Add new memory card overlay
        self.add_overlay(
            MemoryOverlay(
                start=start,
                end=end,
                name="memory_card",
                data=bytearray(card_data[:card_size]),
                read_only=True,
                perfetto_thread="Memory_Card",
            )
        )

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

    def set_lcd_controller(self, lcd_controller) -> None:
        """Set LCD controller and add memory-mapped I/O overlay."""
        self._lcd_controller = lcd_controller

        # Pass CPU reference to LCD controller if available
        if self.cpu and hasattr(lcd_controller, "set_cpu"):
            lcd_controller.set_cpu(self.cpu)
        # LCD controllers at 0x2000 and 0xA000
        self.add_overlay(
            MemoryOverlay(
                start=0x2000,
                end=0x200F,
                name="lcd_controller_low",
                read_only=False,
                read_handler=lambda addr, pc: lcd_controller.read(addr, pc),
                write_handler=lambda addr, val, pc: lcd_controller.write(addr, val, pc),
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
                write_handler=lambda addr, val, pc: lcd_controller.write(addr, val, pc),
                perfetto_thread="Display",
            )
        )

    def reset(self) -> None:
        """Reset all RAM to zero."""
        # Reset external memory (including internal memory at the end)
        self.external_memory[:] = bytes(len(self.external_memory))

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
            byte_val = self.read_byte(address + i)
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
            self.write_byte(address + i, byte_value)

    def set_context(self, context: dict) -> None:
        """Set context for memory operations (compatibility method)."""
        # Context is handled through cpu_pc parameter in read/write methods
        pass

    def set_perf_tracer(self, tracer) -> None:
        """Set performance tracer for SC62015 emulator integration."""
        self._perf_tracer = tracer
