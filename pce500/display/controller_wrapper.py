"""HD61202Controller wrapper to provide compatibility with the emulator."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from .hd61202 import HD61202, parse_command, render_combined_image
from .pipeline import LCDOperation, LCDPipeline, LCDSnapshot
from ..tracing.perfetto_tracing import tracer as new_tracer, perf_trace


class HD61202Controller:
    """Wrapper class that provides the interface expected by the PC-E500 emulator.

    Two HD61202 chips are stitched together to emulate the calculator's 240×32 LCD.
    """

    def __init__(self):
        self.pipeline = LCDPipeline()
        self.chips = self.pipeline.chips
        self.perfetto_enabled = False
        self._cpu = None  # Reference to CPU for getting current PC
        self._write_trace_callbacks: list = []

        # Debug counters
        self.cs_both_count = 0
        self.cs_left_count = 0
        self.cs_right_count = 0

        self.pipeline.subscribe(self._handle_pipeline_event)

    def set_cpu(self, cpu) -> None:
        """Set reference to CPU emulator for getting current PC."""
        self._cpu = cpu

    def add_write_trace_callback(self, callback) -> None:
        """Register a callback invoked for every LCD controller write."""
        if callback not in self._write_trace_callbacks:
            self._write_trace_callbacks.append(callback)

    def clear_write_trace_callbacks(self) -> None:
        """Remove all registered LCD write callbacks."""
        self._write_trace_callbacks.clear()

    def set_write_trace_callback(self, callback) -> None:
        """Backwards-compatible helper that resets to a single callback."""
        self.clear_write_trace_callbacks()
        if callback is not None:
            self.add_write_trace_callback(callback)

    def _get_current_pc(self) -> Optional[int]:
        """Get current PC from CPU if available."""
        if self._cpu and hasattr(self._cpu, "regs"):
            try:
                from sc62015.pysc62015.emulator import RegisterName

                return self._cpu.regs.get(RegisterName.PC)
            except Exception:
                pass
        return None

    def read(self, address: int, cpu_pc: Optional[int] = None) -> int:
        """Read from LCD controller at given address."""
        # Reads are not currently emulated.
        return 0xFF

    @perf_trace("Display")
    def write(self, address: int, value: int, cpu_pc: Optional[int] = None) -> None:
        """Write to LCD controller at given address."""
        try:
            command = parse_command(address, value)
        except ValueError:
            return

        if command.cs.value == 0b00:
            self.cs_both_count += 1
        elif command.cs.value == 0b01:
            self.cs_right_count += 1
        elif command.cs.value == 0b10:
            self.cs_left_count += 1

        operation_pc = cpu_pc if cpu_pc is not None else self._get_current_pc()
        self.pipeline.apply(LCDOperation(command=command, pc=operation_pc))

    def _handle_pipeline_event(
        self, event: Dict[str, object], snapshot: LCDSnapshot
    ) -> None:
        pc_value = event.get("pc")
        if pc_value is None:
            pc_value = self._get_current_pc()
            event["pc"] = pc_value

        if new_tracer.enabled:
            if event["type"] == "instruction":
                new_tracer.instant(
                    "Display",
                    f"LCD_{event['instruction']}",
                    {
                        "chip": event["chip"],
                        "data": f"0x{int(event['data']):02X}",
                        "pc": f"0x{int(pc_value or 0):06X}",
                    },
                )
            else:
                new_tracer.instant(
                    "Display",
                    "VRAM_Write",
                    {
                        "chip": event["chip"],
                        "page": event["page"],
                        "col": event["column"],
                        "data": f"0x{int(event['data']):02X}",
                        "pc": f"0x{int(pc_value or 0):06X}",
                    },
                )

        for callback in self._write_trace_callbacks:
            try:
                callback(event)
            except Exception:
                pass

    def reset(self) -> None:
        """Reset both chips and all statistics."""
        for chip in self.chips:
            chip.reset()
        self.cs_both_count = 0
        self.cs_left_count = 0
        self.cs_right_count = 0

    def get_snapshot(self) -> LCDSnapshot:
        """Return the current LCD snapshot (for diagnostics/tests)."""
        return self.pipeline.snapshot

    @perf_trace("Display")
    def get_display_buffer(self) -> np.ndarray:
        """Return a 32×240 boolean buffer representing the LCD contents."""

        def pixel_on(byte: int, bit: int) -> int:
            return 1 if not ((byte >> bit) & 1) else 0

        buffer = np.zeros((32, 240), dtype=np.uint8)
        left_chip, right_chip = self.chips[0], self.chips[1]

        def copy_region(
            chip: HD61202,
            start_page: int,
            column_range: range,
            dest_start_col: int,
            mirror: bool = False,
        ) -> None:
            for row in range(32):
                page = start_page + row // 8
                bit = row % 8
                if mirror:
                    for dest_offset, src_col in enumerate(reversed(column_range)):
                        byte = chip.vram[page][src_col]
                        buffer[row, dest_start_col + dest_offset] = pixel_on(byte, bit)
                else:
                    for dest_offset, src_col in enumerate(column_range):
                        byte = chip.vram[page][src_col]
                        buffer[row, dest_start_col + dest_offset] = pixel_on(byte, bit)

        if right_chip.state.on:
            copy_region(
                chip=right_chip,
                start_page=0,
                column_range=range(64),
                dest_start_col=0,
            )
        if left_chip.state.on:
            copy_region(
                chip=left_chip,
                start_page=0,
                column_range=range(56),
                dest_start_col=64,
            )
            copy_region(
                chip=left_chip,
                start_page=4,
                column_range=range(56),
                dest_start_col=120,
                mirror=True,
            )
        if right_chip.state.on:
            copy_region(
                chip=right_chip,
                start_page=4,
                column_range=range(64),
                dest_start_col=176,
                mirror=True,
            )

        return buffer

    def get_combined_display(self, zoom: int = 2) -> Image.Image:
        """Get combined display using the render_combined_image function."""
        return render_combined_image(self.chips, zoom)

    def save_displays_to_png(
        self, left_filename: str = "lcd_left.png", right_filename: str = "lcd_right.png"
    ) -> None:
        left_img = self.chips[0].render_vram_image(zoom=1)
        right_img = self.chips[1].render_vram_image(zoom=1)

        left_img = left_img.convert("L")
        right_img = right_img.convert("L")

        left_img.save(left_filename)
        right_img.save(right_filename)

    def set_perfetto_enabled(self, enabled: bool) -> None:
        self.perfetto_enabled = enabled

    @property
    def display_on(self) -> List[bool]:
        return [chip.state.on for chip in self.chips]

    @property
    def page(self) -> List[int]:
        return [chip.state.page for chip in self.chips]

    @property
    def column(self) -> List[int]:
        return [chip.state.y_address for chip in self.chips]

    @property
    def width(self) -> int:
        return 240

    @property
    def height(self) -> int:
        return 32

    def get_chip_statistics(self) -> List[Dict[str, int]]:
        stats = []
        for i, chip in enumerate(self.chips):
            stats.append(
                {
                    "chip": i,
                    "on": chip.state.on,
                    "instructions": chip.instruction_count,
                    "data_written": chip.data_write_count,
                    "data_read": chip.data_read_count,
                    "on_off_commands": chip.on_off_count,
                    "page": chip.state.page,
                    "column": chip.state.y_address,
                }
            )
        return stats
