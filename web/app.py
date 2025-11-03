"""Flask backend for PC-E500 web emulator."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_file,
    send_from_directory,
)
from flask_cors import CORS
from PIL import Image, ImageOps  # type: ignore

# Ensure imports work when running as a module or script
CURRENT_DIR = Path(__file__).parent
PARENT_DIR = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

try:  # pragma: no cover - exercised under WSGI
    from .emulator_service import TRACE_PATH, init_app, service
except ImportError:  # pragma: no cover - direct script execution
    from emulator_service import TRACE_PATH, init_app, service  # type: ignore

app = Flask(__name__)

# Restrict CORS by default; allow opt-in via config/env
allowed_origins = app.config.get("WEB_ALLOWED_ORIGINS") or os.environ.get(
    "PCE500_WEB_ALLOWED_ORIGINS"
)
if allowed_origins:
    if isinstance(allowed_origins, str):
        origins = [
            origin.strip() for origin in allowed_origins.split(",") if origin.strip()
        ]
    else:
        origins = allowed_origins
    if origins:
        CORS(app, resources={r"/api/*": {"origins": origins}})


def initialize_emulator() -> None:
    """Legacy helper used by tests/CLI."""
    service.ensure_emulator()


# Ensure emulator is ready under WSGI servers
init_app(app)


@app.route("/")
def index():
    """Serve the main web interface."""
    return render_template("index.html")


@app.route("/api/v1/state", methods=["GET"])
def get_state():
    """Return current emulator state snapshot."""
    state = service.snapshot_state()
    return jsonify(state)


@app.route("/api/v1/key", methods=["POST"])
def handle_key():
    """Handle keyboard input."""
    data = request.get_json() or {}
    key_code = data.get("key_code")
    if not key_code:
        return jsonify({"error": "Missing key_code"}), 400

    action = data.get("action", "press")
    try:
        if action == "press":
            queued = service.press_key(key_code)
            message = (
                f"Key {key_code} queued"
                if queued
                else f"Key {key_code} not queued (unmapped or already pressed)"
            )
        elif action == "release":
            debug_snapshot = service.keyboard_debug_info()
            pressed = set(debug_snapshot.get("pressed_keys", []))
            if key_code in pressed:
                service.release_key(key_code)
                message = f"Key {key_code} released"
            else:
                message = f"Key {key_code} release ignored"
            queued = None
        else:
            return jsonify({"error": f"Invalid action: {action}"}), 400
    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({"error": str(exc)}), 500

    debug_info = service.keyboard_debug_info()
    return jsonify(
        {
            "status": "ok",
            "key_queued": queued,
            "message": message,
            "debug": debug_info,
        }
    )


@app.route("/api/v1/ocr", methods=["GET"])
def get_ocr_text():
    """Return OCR text extracted from the current LCD image."""
    try:
        img = service.capture_lcd_image()
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Failed to capture LCD: {exc}"}), 500

    try:
        import pytesseract  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        return jsonify(
            {"ok": False, "error": f"pytesseract not available: {exc}", "text": ""}
        ), 200

    try:
        im = img.convert("L")
        im = ImageOps.expand(im, border=4, fill=255)
        im = im.resize((im.width * 3, im.height * 3), Image.LANCZOS)
        im = im.point(lambda v: 255 if v > 128 else 0, mode="1")
        text = pytesseract.image_to_string(im, config="--psm 3")
        return jsonify({"ok": True, "text": text})
    except Exception as exc:  # pragma: no cover - OCR best effort
        return jsonify({"ok": False, "error": f"OCR failed: {exc}", "text": ""}), 200


@app.route("/api/v1/lcd_debug", methods=["GET"])
def get_lcd_debug():
    """Get PC source for a specific LCD pixel."""
    x = request.args.get("x", type=int)
    y = request.args.get("y", type=int)
    if x is None or y is None:
        return jsonify({"error": "Missing x or y parameter"}), 400
    if not (0 <= x < 240 and 0 <= y < 32):
        return jsonify({"error": "Coordinates out of range"}), 400

    pc_source = service.lcd_debug(x, y)
    return jsonify({"pc": f"0x{pc_source:06X}" if pc_source is not None else None})


@app.route("/api/v1/imem_watch", methods=["GET"])
def get_imem_watch():
    """Get IMEM register access tracking data."""
    return jsonify(service.imem_watch())


@app.route("/api/v1/lcd_stats", methods=["GET"])
def get_lcd_stats():
    """Get LCD controller statistics."""
    return jsonify(service.lcd_stats())


@app.route("/api/v1/key_queue", methods=["GET"])
def get_key_queue():
    """Get keyboard queue state."""
    queue_info = service.keyboard_queue()
    keyboard_state = service.keyboard_register_state()
    return jsonify({"queue": queue_info, "registers": keyboard_state})


@app.route("/api/v1/control", methods=["POST"])
def control_emulator():
    """Control emulator execution (run/pause/step/reset)."""
    data = request.get_json() or {}
    command = data.get("command")
    if not command:
        return jsonify({"error": "Missing command"}), 400

    if command == "run":
        service.run()
        return jsonify({"status": "running"})
    if command == "pause":
        service.pause()
        return jsonify({"status": "paused"})
    if command == "step":
        state = service.snapshot_state()
        if not state.get("is_running"):
            service.step()
        return jsonify({"status": "stepped"})
    if command == "reset":
        service.reset()
        return jsonify({"status": "reset", "is_running": True})

    return jsonify({"error": f"Unknown command: {command}"}), 400


@app.route("/trace/start", methods=["POST"])
def trace_start():
    """Start Perfetto tracing."""
    result = service.start_trace()
    status = 200 if result.get("ok") else 500
    return jsonify(result), status


@app.route("/trace/stop", methods=["POST"])
def trace_stop():
    """Stop Perfetto tracing."""
    result = service.stop_trace()
    return jsonify(result)


@app.route("/trace/download", methods=["GET"])
def trace_download():
    """Download the trace file."""
    trace_file = Path(TRACE_PATH)
    if trace_file.exists():
        return send_file(
            str(trace_file), as_attachment=True, download_name="pc-e500.perfetto-trace"
        )
    return jsonify({"error": "Trace file not found"}), 404


@app.route("/trace/status", methods=["GET"])
def trace_status():
    """Get current trace status."""
    return jsonify(service.trace_status())


@app.route("/static/<path:filename>")
def static_files(filename):
    """Serve static files."""
    return send_from_directory("static", filename)


if __name__ == "__main__":
    initialize_emulator()
    print("PC-E500 emulator initialized successfully")
    print("Starting web server at http://localhost:8080")
    app.run(debug=True, host="0.0.0.0", port=8080)
