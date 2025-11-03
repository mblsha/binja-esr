"""Shared emulator lifecycle management for the web API."""

from __future__ import annotations

import base64
import io
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional

from pce500 import PCE500Emulator
from pce500.tracing.perfetto_tracing import tracer as perfetto_tracer
from sc62015.pysc62015.constants import IMRFlag, INTERNAL_MEMORY_START
from sc62015.pysc62015.instr.opcodes import IMEMRegisters

RUN_BATCH_INSTRUCTIONS = 5_000
RUN_SLEEP_SECONDS = 0.002
TRACE_PATH = "pc-e500.perfetto-trace"


class EmulatorService:
    """Manage a shared emulator instance and background execution."""

    UPDATE_TIME_THRESHOLD = 0.1
    UPDATE_INSTRUCTION_THRESHOLD = 100_000

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._emulator: Optional[PCE500Emulator] = None
        self._state: Dict[str, object] = self._initial_state()
        self._run_event = threading.Event()
        self._shutdown = threading.Event()
        self._runner_thread: Optional[threading.Thread] = None

    @staticmethod
    def _initial_state() -> Dict[str, object]:
        return {
            "is_running": False,
            "last_update_time": 0.0,
            "last_update_instructions": 0,
            "screen": None,
            "registers": {},
            "flags": {},
            "instruction_count": 0,
            "instruction_history": [],
            "speed_calc_time": None,
            "speed_calc_instructions": None,
            "emulation_speed": None,
            "speed_ratio": None,
            "interrupts": None,
        }

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def ensure_emulator(self) -> PCE500Emulator:
        """Return an initialised emulator, creating it if necessary."""
        with self._lock:
            if self._emulator is None:
                self._emulator = self._create_emulator()
            return self._emulator

    def _create_emulator(self) -> PCE500Emulator:
        rom_path = Path(__file__).parent.parent / "data" / "pc-e500.bin"

        if not rom_path.exists():
            rom_data = bytearray(0x100000)
            rom_data[0xFFFFD] = 0x00
            rom_data[0xFFFFE] = 0x00
            rom_data[0xFFFFF] = 0x0C
            rom_data[0xC0000] = 0x00
        else:
            with open(rom_path, "rb") as fh:
                rom_data = fh.read()

        if len(rom_data) >= 0x100000:
            rom_portion = rom_data[0xC0000:0x100000]
        else:
            rom_portion = rom_data

        emulator = PCE500Emulator(
            trace_enabled=True,
            perfetto_trace=False,
            save_lcd_on_exit=False,
        )
        emulator.load_rom(rom_portion)
        emulator.reset()
        self._update_state_locked(emulator, force=True)
        return emulator

    def shutdown(self) -> None:
        """Stop the run thread and release resources."""
        self._shutdown.set()
        self._run_event.clear()
        if self._runner_thread and self._runner_thread.is_alive():
            self._runner_thread.join(timeout=1.0)
        with self._lock:
            if self._emulator:
                self._emulator.stop_tracing()
            self._emulator = None
            self._state = self._initial_state()

    # ------------------------------------------------------------------ #
    # Runner management
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        """Start continuous execution."""
        self.ensure_emulator()
        with self._lock:
            self._state["is_running"] = True
            self._state["speed_calc_time"] = None
            self._state["speed_calc_instructions"] = None
            self._state["emulation_speed"] = None
            self._state["speed_ratio"] = None
        self._run_event.set()
        if not self._runner_thread or not self._runner_thread.is_alive():
            self._runner_thread = threading.Thread(
                target=self._runner_loop, name="PCE500Runner", daemon=True
            )
            self._runner_thread.start()

    def pause(self) -> None:
        """Pause continuous execution."""
        self._run_event.clear()
        with self._lock:
            self._state["is_running"] = False
            self._state["speed_calc_time"] = None
            self._state["speed_calc_instructions"] = None

    def step(self) -> None:
        """Execute a single instruction."""
        with self._lock:
            emulator = self.ensure_emulator()
            emulator.step()
            self._update_state_locked(emulator, force=True)

    def reset(self) -> None:
        """Reset emulator and resume running."""
        with self._lock:
            emulator = self.ensure_emulator()
            emulator.reset()
            self._update_state_locked(emulator, force=True)
            self._state["is_running"] = True
            self._state["speed_calc_time"] = None
            self._state["speed_calc_instructions"] = None
        self._run_event.set()
        if not self._runner_thread or not self._runner_thread.is_alive():
            self._runner_thread = threading.Thread(
                target=self._runner_loop, name="PCE500Runner", daemon=True
            )
            self._runner_thread.start()

    def _runner_loop(self) -> None:
        """Background loop for paced execution."""
        while not self._shutdown.is_set():
            if not self._run_event.wait(timeout=0.1):
                continue
            while self._run_event.is_set() and not self._shutdown.is_set():
                with self._lock:
                    emulator = self._emulator
                    if emulator is None:
                        break
                    emulator.run(RUN_BATCH_INSTRUCTIONS)
                    self._maybe_update_state_locked(emulator)
                time.sleep(RUN_SLEEP_SECONDS)

    # ------------------------------------------------------------------ #
    # State helpers
    # ------------------------------------------------------------------ #

    def snapshot_state(self) -> Dict[str, object]:
        """Return the latest emulator state, forcing a refresh when paused."""
        with self._lock:
            emulator = self._emulator
            if emulator:
                self._update_state_locked(emulator, force=not self._state["is_running"])
            return dict(self._state)

    def _maybe_update_state_locked(self, emulator: PCE500Emulator) -> None:
        if not self._state["is_running"]:
            return

        now = time.time()
        last_time = self._state["last_update_time"]
        last_instr = self._state["last_update_instructions"]
        time_delta = now - last_time
        instr_delta = emulator.instruction_count - int(last_instr)

        if (
            time_delta >= self.UPDATE_TIME_THRESHOLD
            or instr_delta >= self.UPDATE_INSTRUCTION_THRESHOLD
        ):
            self._update_state_locked(emulator, force=True)

    def _update_state_locked(
        self, emulator: PCE500Emulator, *, force: bool = False
    ) -> None:
        if not force and self._state["is_running"]:
            now = time.time()
            last_time = self._state["last_update_time"]
            last_instr = self._state["last_update_instructions"]
            if (
                now - last_time < self.UPDATE_TIME_THRESHOLD
                and emulator.instruction_count - int(last_instr)
                < self.UPDATE_INSTRUCTION_THRESHOLD
            ):
                return

        lcd_image = emulator.lcd.get_combined_display(zoom=1)
        img_buffer = io.BytesIO()
        lcd_image.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        screen_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

        current_time = time.time()
        current_instructions = emulator.instruction_count
        prev_time = self._state.get("speed_calc_time")
        prev_instr = self._state.get("speed_calc_instructions")

        if self._state["is_running"]:
            speed, ratio = self._calculate_emulation_speed(
                prev_time,
                prev_instr,
                current_time,
                current_instructions,
            )
            self._state["emulation_speed"] = speed
            self._state["speed_ratio"] = ratio
        else:
            self._state["emulation_speed"] = None
            self._state["speed_ratio"] = None

        self._state["speed_calc_time"] = current_time
        self._state["speed_calc_instructions"] = current_instructions

        cpu_state = emulator.get_cpu_state()
        self._state.update(
            {
                "screen": f"data:image/png;base64,{screen_base64}",
                "registers": {
                    "pc": cpu_state["pc"],
                    "a": cpu_state["a"],
                    "b": cpu_state["b"],
                    "ba": cpu_state["ba"],
                    "i": cpu_state["i"],
                    "x": cpu_state["x"],
                    "y": cpu_state["y"],
                    "u": cpu_state["u"],
                    "s": cpu_state["s"],
                },
                "flags": {
                    "z": cpu_state["flags"]["z"],
                    "c": cpu_state["flags"]["c"],
                },
                "instruction_count": emulator.instruction_count,
                "instruction_history": list(emulator.instruction_history),
                "last_update_time": current_time,
                "last_update_instructions": current_instructions,
            }
        )
        self._state["interrupts"] = self._build_interrupt_snapshot(emulator)

    @staticmethod
    def _calculate_emulation_speed(
        previous_time: Optional[float],
        previous_instructions: Optional[int],
        current_time: float,
        current_instructions: int,
    ) -> tuple[float, float]:
        if previous_time is None or previous_instructions is None:
            return 0.0, 0.0
        delta = current_time - previous_time
        if delta <= 0:
            return 0.0, 0.0
        speed = (current_instructions - previous_instructions) / delta
        return speed, speed / 2_000_000

    def _build_interrupt_snapshot(
        self, emulator: PCE500Emulator
    ) -> Optional[Dict[str, object]]:
        try:
            ints = (
                emulator.get_interrupt_stats()
                if hasattr(emulator, "get_interrupt_stats")
                else None
            )
            snapshot = ints or {}
            imr_val = (
                emulator.memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.IMR)
                & 0xFF
            )
            isr_val = (
                emulator.memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR)
                & 0xFF
            )
            snapshot.update(
                {
                    "imr": f"0x{imr_val:02X}",
                    "isr": f"0x{isr_val:02X}",
                    "irm": 1 if (imr_val & int(IMRFlag.IRM)) else 0,
                    "keym": 1 if (imr_val & int(IMRFlag.KEYM)) else 0,
                    "isr_key": 1 if (isr_val & int(IMRFlag.KEYM)) else 0,
                    "pending": bool(getattr(emulator, "_irq_pending", False)),
                }
            )
            return snapshot
        except Exception:
            return ints if ints is not None else None

    # ------------------------------------------------------------------ #
    # API Helpers
    # ------------------------------------------------------------------ #

    def keyboard_register_state(self) -> Dict[str, str]:
        emulator = self.ensure_emulator()
        registers = emulator.get_keyboard_register_state()
        return {key: f"0x{value:02X}" for key, value in registers.items()}

    def keyboard_queue(self):
        emulator = self.ensure_emulator()
        return emulator.keyboard.get_queue_info()

    def keyboard_debug_info(self) -> Dict[str, object]:
        with self._lock:
            emulator = self.ensure_emulator()
            return emulator.keyboard.get_debug_info()

    def capture_lcd_image(self):
        with self._lock:
            emulator = self.ensure_emulator()
            return emulator.lcd.get_combined_display(zoom=1)

    def press_key(self, key_code: str) -> bool:
        with self._lock:
            emulator = self.ensure_emulator()
            queued = emulator.press_key(key_code)
            self._update_state_locked(emulator, force=True)
            return queued

    def release_key(self, key_code: str) -> None:
        with self._lock:
            emulator = self.ensure_emulator()
            emulator.release_key(key_code)
            self._update_state_locked(emulator, force=True)

    def lcd_debug(self, x: int, y: int) -> Optional[int]:
        emulator = self.ensure_emulator()
        return emulator.lcd.get_pixel_pc_source(x, y)

    def imem_watch(self):
        emulator = self.ensure_emulator()
        tracking = emulator.memory.get_imem_access_tracking()
        result = {}
        for reg_name, accesses in tracking.items():
            result[reg_name] = {
                "reads": [
                    {"pc": f"0x{pc:06X}", "count": count}
                    for pc, count in accesses["reads"]
                ],
                "writes": [
                    {"pc": f"0x{pc:06X}", "count": count}
                    for pc, count in accesses["writes"]
                ],
            }
        return result

    def lcd_stats(self):
        emulator = self.ensure_emulator()
        chip_select_stats = {
            "both": emulator.lcd.cs_both_count,
            "left": emulator.lcd.cs_left_count,
            "right": emulator.lcd.cs_right_count,
        }
        chip_stats = emulator.lcd.get_chip_statistics()
        return {"chip_select": chip_select_stats, "chips": chip_stats}

    # ------------------------------------------------------------------ #
    # Trace controls
    # ------------------------------------------------------------------ #

    def start_trace(self) -> Dict[str, object]:
        emulator = self.ensure_emulator()
        if emulator.tracing_enabled:
            return {
                "ok": True,
                "enabled": True,
                "path": TRACE_PATH,
                "message": "Already tracing",
            }
        emulator.start_tracing(TRACE_PATH)
        perfetto_tracer.start(TRACE_PATH)
        return {"ok": True, "enabled": True, "path": TRACE_PATH}

    def stop_trace(self) -> Dict[str, object]:
        emulator = self.ensure_emulator()
        emulator.stop_tracing()
        return {"ok": True, "enabled": False, "path": TRACE_PATH}

    def trace_status(self) -> Dict[str, object]:
        emulator = self.ensure_emulator()
        return {
            "enabled": emulator.tracing_enabled,
            "path": TRACE_PATH,
            "file_exists": Path(TRACE_PATH).exists(),
        }

    @contextmanager
    def emulator_context(self):
        """Provide exclusive access to the underlying emulator."""
        with self._lock:
            yield self.ensure_emulator()


service = EmulatorService()


def init_app(app) -> None:
    """Ensure the emulator is ready when the Flask app starts."""
    with app.app_context():
        service.ensure_emulator()
