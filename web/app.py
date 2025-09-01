"""Flask backend for PC-E500 web emulator."""

import base64
import io
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from flask import (
    Flask,
    jsonify,
    request,
    render_template,
    send_from_directory,
    send_file,
)
from flask_cors import CORS
from sc62015.pysc62015.instr.opcodes import IMEMRegisters
from sc62015.pysc62015.constants import IMRFlag
from PIL import Image, ImageOps

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from pce500 import PCE500Emulator
from pce500.tracing.perfetto_tracing import tracer as perfetto_tracer

app = Flask(__name__)
CORS(app)  # Enable CORS for API endpoints

# Global emulator instance and control state
emulator: Optional[PCE500Emulator] = None
emulator_lock = threading.Lock()
emulator_thread: Optional[threading.Thread] = None
emulator_state = {
    "is_running": False,
    "last_update_time": 0,
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
}

# Update thresholds
UPDATE_TIME_THRESHOLD = 0.1  # 100ms (10fps)
UPDATE_INSTRUCTION_THRESHOLD = 100000  # 100k instructions

# Trace file path
TRACE_PATH = "pc-e500.perfetto-trace"


def initialize_emulator():
    """Initialize the PC-E500 emulator with ROM."""
    global emulator

    # Load ROM file
    rom_path = Path(__file__).parent.parent / "data" / "pc-e500.bin"

    # For testing, create a minimal ROM if the file doesn't exist
    if not rom_path.exists():
        # Create a minimal test ROM with a simple program
        # This is enough to allow the emulator to initialize
        rom_data = bytearray(0x100000)  # 1MB
        # Set entry point at 0xFFFFD to 0xC0000 (start of ROM)
        rom_data[0xFFFFD] = 0x00
        rom_data[0xFFFFE] = 0x00
        rom_data[0xFFFFF] = 0x0C
        # Add a HALT instruction at 0xC0000
        rom_data[0xC0000] = 0x00  # HALT opcode
    else:
        with open(rom_path, "rb") as f:
            rom_data = f.read()

    # Extract the ROM portion from 0xC0000-0x100000 (256KB)
    # The pc-e500.bin file is a full 1MB dump, but the ROM is only the last 256KB
    if len(rom_data) >= 0x100000:
        rom_portion = rom_data[0xC0000:0x100000]
    else:
        rom_portion = rom_data  # If file is smaller, use as-is

    # Create emulator instance
    with emulator_lock:
        # Use compat keyboard implementation
        # Enable lightweight opcode/disassembly tracing so Instruction History populates
        # (emulator maintains a bounded deque, so overhead remains reasonable)
        emulator = PCE500Emulator(
            trace_enabled=True,
            perfetto_trace=False,
            save_lcd_on_exit=False,
        )
        emulator.load_rom(rom_portion)

        # Reset to properly set PC from the now-loaded ROM entry point
        emulator.reset()

        # Update initial state
        update_emulator_state()


def update_emulator_state():
    """Capture current emulator state for API responses."""
    global emulator_state

    if emulator is None:
        return

    # Get CPU state
    cpu_state = emulator.get_cpu_state()

    # Get screen as PNG
    lcd_image = emulator.lcd.get_combined_display(zoom=1)  # 240x32 pixels

    # Convert PIL image to base64 PNG
    img_buffer = io.BytesIO()
    lcd_image.save(img_buffer, format="PNG")
    img_buffer.seek(0)
    screen_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

    # Calculate emulation speed
    current_time = time.time()
    current_instructions = emulator.instruction_count

    if emulator_state["is_running"]:
        # Calculate speed only when running
        prev_time = emulator_state.get("speed_calc_time")
        prev_instructions = emulator_state.get("speed_calc_instructions")

        if prev_time is not None and prev_instructions is not None:
            time_delta = current_time - prev_time
            instruction_delta = current_instructions - prev_instructions

            if time_delta > 0:
                speed = instruction_delta / time_delta
                speed_ratio = speed / 2_000_000  # 2MHz CPU
            else:
                speed = 0
                speed_ratio = 0
        else:
            # First update while running
            speed = 0
            speed_ratio = 0

        emulator_state["emulation_speed"] = speed
        emulator_state["speed_ratio"] = speed_ratio
    else:
        # Not running, clear speed
        emulator_state["emulation_speed"] = None
        emulator_state["speed_ratio"] = None

    # Update tracking values
    emulator_state["speed_calc_time"] = current_time
    emulator_state["speed_calc_instructions"] = current_instructions

    # Update state dictionary
    emulator_state.update(
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
            "flags": {"z": cpu_state["flags"]["z"], "c": cpu_state["flags"]["c"]},
            "instruction_count": emulator.instruction_count,
            "instruction_history": list(emulator.instruction_history),
            "last_update_time": current_time,
            "last_update_instructions": current_instructions,
            # Interrupt stats (if available) + current IMR/ISR bits
            "interrupts": (
                emulator.get_interrupt_stats()
                if hasattr(emulator, "get_interrupt_stats")
                else None
            ),
        }
    )
    # Enrich interrupts with IMR/ISR flag info
    try:
        ints = emulator_state.get("interrupts") or {}
        INTERNAL_MEMORY_START = 0x100000
        imr_val = (
            emulator.memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.IMR) & 0xFF
        )
        isr_val = (
            emulator.memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR) & 0xFF
        )
        irm = 1 if (imr_val & int(IMRFlag.IRM)) else 0
        keym = 1 if (imr_val & int(IMRFlag.KEYM)) else 0
        isr_key = 1 if (isr_val & int(IMRFlag.KEYM)) else 0
        pending = bool(getattr(emulator, "_irq_pending", False))
        ints.update(
            {
                "imr": f"0x{imr_val:02X}",
                "isr": f"0x{isr_val:02X}",
                "irm": irm,
                "keym": keym,
                "isr_key": isr_key,
                "pending": pending,
            }
        )
        emulator_state["interrupts"] = ints
    except Exception:
        pass


@app.route("/api/v1/ocr", methods=["GET"])
def get_ocr_text():
    """Return OCR text extracted from the current LCD image.

    Polling target is ~2 Hz from the UI when the emulator is running.
    """
    global emulator
    with emulator_lock:
        if emulator is None:
            return jsonify({"ok": False, "error": "Emulator not initialized"}), 500
        try:
            img = emulator.lcd.get_combined_display(zoom=1)
        except Exception as e:
            return jsonify({"ok": False, "error": f"Failed to capture LCD: {e}"}), 500

    # Preprocess image for OCR
    try:
        import pytesseract  # type: ignore
    except Exception as e:
        return jsonify(
            {"ok": False, "error": f"pytesseract not available: {e}", "text": ""}
        ), 200

    try:
        im = img.convert("L")
        # Add a small white border and upscale to help OCR
        im = ImageOps.expand(im, border=4, fill=255)
        im = im.resize((im.width * 3, im.height * 3), Image.LANCZOS)
        # Binarize quickly
        im = im.point(lambda v: 255 if v > 128 else 0, mode="1")
        text = pytesseract.image_to_string(im, config="--psm 3")
        return jsonify({"ok": True, "text": text})
    except Exception as e:
        return jsonify({"ok": False, "error": f"OCR failed: {e}", "text": ""}), 200


def emulator_run_loop():
    """Background thread for running the emulator."""
    global emulator_state

    while emulator_state["is_running"]:
        if emulator is None:
            break

        with emulator_lock:
            try:
                # Execute one instruction
                emulator.step()

                # Check if we should update state
                current_time = time.time()
                current_instructions = emulator.instruction_count

                time_delta = current_time - emulator_state["last_update_time"]
                instruction_delta = (
                    current_instructions - emulator_state["last_update_instructions"]
                )

                if (
                    time_delta >= UPDATE_TIME_THRESHOLD
                    or instruction_delta >= UPDATE_INSTRUCTION_THRESHOLD
                ):
                    update_emulator_state()

            except Exception as e:
                print(f"Emulator error: {e}")
                emulator_state["is_running"] = False
                break

        # Small delay to prevent CPU spinning too fast
        time.sleep(0.0001)


@app.route("/")
def index():
    """Serve the main web interface."""
    return render_template("index.html")


@app.route("/api/v1/state", methods=["GET"])
def get_state():
    """Get current emulator state."""
    with emulator_lock:
        # Always update state when explicitly requested
        if emulator is not None:
            update_emulator_state()

        return jsonify(emulator_state)


@app.route("/api/v1/key", methods=["POST"])
def handle_key():
    """Handle keyboard input."""
    data = request.get_json()
    if not data or "key_code" not in data:
        return jsonify({"error": "Missing key_code"}), 400

    key_code = data["key_code"]
    action = data.get("action", "press")  # 'press' or 'release'

    with emulator_lock:
        if emulator is None:
            return jsonify({"error": "Emulator not initialized"}), 500

        try:
            key_queued = False
            if action == "press":
                key_queued = emulator.press_key(key_code)
                if key_queued:
                    print(f"Key pressed and queued: {key_code}")
                else:
                    print(
                        f"Key press ignored (not mapped or already queued): {key_code}"
                    )
            elif action == "release":
                # Ignore release if key isn't currently pressed to avoid noise from hover events
                try:
                    currently_pressed = set()
                    if hasattr(emulator, "keyboard"):
                        kb = emulator.keyboard
                        if hasattr(kb, "pressed_keys"):
                            currently_pressed = set(kb.pressed_keys)  # type: ignore[attr-defined]
                        elif hasattr(kb, "get_pressed_keys"):
                            currently_pressed = set(kb.get_pressed_keys())  # type: ignore[attr-defined]
                    if key_code in currently_pressed:
                        emulator.release_key(key_code)
                        print(f"Key released: {key_code}")
                    else:
                        # Silently ignore spurious release
                        pass
                except Exception:
                    emulator.release_key(key_code)
            else:
                return jsonify({"error": f"Invalid action: {action}"}), 400

            # Return debug info for development
            debug_info = emulator.keyboard.get_debug_info()
            return jsonify(
                {
                    "status": "ok",
                    "key_queued": key_queued if action == "press" else None,
                    "message": f"Key {key_code} {'queued' if key_queued else 'not queued (unmapped or already pressed)'}"
                    if action == "press"
                    else f"Key {key_code} released",
                    "debug": debug_info,
                }
            )

        except Exception as e:
            return jsonify({"error": str(e)}), 500


@app.route("/api/v1/lcd_debug", methods=["GET"])
def get_lcd_debug():
    """Get PC source for a specific LCD pixel."""
    x = request.args.get("x", type=int)
    y = request.args.get("y", type=int)

    if x is None or y is None:
        return jsonify({"error": "Missing x or y parameter"}), 400

    if not (0 <= x < 240 and 0 <= y < 32):
        return jsonify({"error": "Coordinates out of range"}), 400

    with emulator_lock:
        if emulator is None:
            return jsonify({"error": "Emulator not initialized"}), 500

        pc_source = emulator.lcd.get_pixel_pc_source(x, y)
        return jsonify({"pc": f"0x{pc_source:06X}" if pc_source is not None else None})


@app.route("/api/v1/imem_watch", methods=["GET"])
def get_imem_watch():
    """Get IMEM register access tracking data."""
    with emulator_lock:
        if emulator is None:
            return jsonify({"error": "Emulator not initialized"}), 500

        tracking_data = emulator.memory.get_imem_access_tracking()

        # Format the data for the API response
        result = {}
        for reg_name, accesses in tracking_data.items():
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

        return jsonify(result)


@app.route("/api/v1/lcd_stats", methods=["GET"])
def get_lcd_stats():
    """Get LCD controller statistics."""
    with emulator_lock:
        if emulator is None:
            return jsonify({"error": "Emulator not initialized"}), 500

        # Get chip select counts
        chip_select_stats = {
            "both": emulator.lcd.cs_both_count,
            "left": emulator.lcd.cs_left_count,
            "right": emulator.lcd.cs_right_count,
        }

        # Get per-chip statistics
        chip_stats = emulator.lcd.get_chip_statistics()

        return jsonify({"chip_select": chip_select_stats, "chips": chip_stats})


@app.route("/api/v1/key_queue", methods=["GET"])
def get_key_queue():
    """Get keyboard queue state."""
    with emulator_lock:
        if emulator is None:
            return jsonify({"error": "Emulator not initialized"}), 500

        # Get key queue info
        queue_info = emulator.keyboard.get_queue_info()

        # Get current KOL/KOH/KIL values
        keyboard_state = {
            "kol": f"0x{emulator.keyboard._last_kol:02X}",
            "koh": f"0x{emulator.keyboard._last_koh:02X}",
            "kil": f"0x{emulator.keyboard._read_keyboard_input():02X}",
        }

        return jsonify({"queue": queue_info, "registers": keyboard_state})


@app.route("/api/v1/control", methods=["POST"])
def control_emulator():
    """Control emulator execution (run/pause/step/reset)."""
    global emulator_thread, emulator_state

    data = request.get_json()
    if not data or "command" not in data:
        return jsonify({"error": "Missing command"}), 400

    command = data["command"]

    # We'll do minimal critical sections to avoid deadlocks with the run loop
    join_needed = False
    thread_to_join = None
    response = None
    with emulator_lock:
        if emulator is None and command != "reset":
            response = (jsonify({"error": "Emulator not initialized"}), 500)
        elif command == "run":
            if not emulator_state["is_running"]:
                emulator_state["is_running"] = True
                emulator_thread = threading.Thread(target=emulator_run_loop)
                emulator_thread.start()
            response = jsonify({"status": "running"})
        elif command == "pause":
            emulator_state["is_running"] = False
            # Clear speed tracking
            emulator_state["speed_calc_time"] = None
            emulator_state["speed_calc_instructions"] = None
            if emulator_thread:
                # Defer join until after releasing the lock to avoid deadlock
                thread_to_join = emulator_thread
                join_needed = True
                emulator_thread = None
            response = jsonify({"status": "paused"})
        elif command == "step":
            if not emulator_state["is_running"]:
                emulator.step()
                update_emulator_state()
            response = jsonify({"status": "stepped"})
        elif command == "reset":
            # Stop running if active
            emulator_state["is_running"] = False
            # Clear speed tracking
            emulator_state["speed_calc_time"] = None
            emulator_state["speed_calc_instructions"] = None
            if emulator_thread:
                # Defer join to outside lock
                thread_to_join = emulator_thread
                join_needed = True
                emulator_thread = None

            # Reset or reinitialize emulator
            if emulator:
                emulator.reset()
            else:
                initialize_emulator()

            # Immediately start running after reset
            emulator_state["is_running"] = True
            update_emulator_state()

            # Start background run loop
            def _start_thread():
                global emulator_thread
                emulator_thread = threading.Thread(target=emulator_run_loop)
                emulator_thread.start()

            _start_thread()
            response = jsonify({"status": "reset", "is_running": True})
        else:
            response = (jsonify({"error": f"Unknown command: {command}"}), 400)
    # Perform join outside the emulator_lock critical section
    if join_needed and thread_to_join:
        thread_to_join.join()
    return response


@app.route("/trace/start", methods=["POST"])
def trace_start():
    """Start Perfetto tracing."""
    global emulator

    with emulator_lock:
        if emulator and not perfetto_tracer.enabled:
            # Enable tracing in the emulator
            emulator._new_trace_enabled = True
            emulator._trace_path = TRACE_PATH
            perfetto_tracer.start(TRACE_PATH)
            return jsonify({"ok": True, "enabled": True, "path": TRACE_PATH})
        elif perfetto_tracer.enabled:
            return jsonify(
                {
                    "ok": True,
                    "enabled": True,
                    "path": TRACE_PATH,
                    "message": "Already tracing",
                }
            )
        else:
            return jsonify({"ok": False, "error": "Emulator not initialized"}), 500


@app.route("/trace/stop", methods=["POST"])
def trace_stop():
    """Stop Perfetto tracing."""
    if perfetto_tracer.enabled:
        perfetto_tracer.stop()
        return jsonify({"ok": True, "enabled": False, "path": TRACE_PATH})
    else:
        return jsonify(
            {"ok": True, "enabled": False, "message": "Not currently tracing"}
        )


@app.route("/trace/download", methods=["GET"])
def trace_download():
    """Download the trace file."""
    trace_file = Path(TRACE_PATH)
    if trace_file.exists():
        return send_file(
            str(trace_file), as_attachment=True, download_name="pc-e500.perfetto-trace"
        )
    else:
        return jsonify({"error": "Trace file not found"}), 404


@app.route("/trace/status", methods=["GET"])
def trace_status():
    """Get current trace status."""
    return jsonify(
        {
            "enabled": perfetto_tracer.enabled,
            "path": TRACE_PATH,
            "file_exists": Path(TRACE_PATH).exists(),
        }
    )


@app.route("/static/<path:filename>")
def static_files(filename):
    """Serve static files."""
    return send_from_directory("static", filename)


if __name__ == "__main__":
    # Initialize emulator on startup
    try:
        initialize_emulator()
        print("PC-E500 emulator initialized successfully")
    except Exception as e:
        print(f"Failed to initialize emulator: {e}")

    # Run Flask app
    app.run(debug=True, host="0.0.0.0", port=8080)
