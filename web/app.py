"""Flask backend for PC-E500 web emulator."""

import base64
import io
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from pce500 import PCE500Emulator

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
    "instruction_count": 0
}

# Update thresholds
UPDATE_TIME_THRESHOLD = 0.1  # 100ms (10fps)
UPDATE_INSTRUCTION_THRESHOLD = 100000  # 100k instructions


def initialize_emulator():
    """Initialize the PC-E500 emulator with ROM."""
    global emulator
    
    # Load ROM file
    rom_path = Path(__file__).parent.parent / "data" / "pc-e500.bin"
    if not rom_path.exists():
        raise FileNotFoundError(f"ROM file not found at {rom_path}")
    
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
        emulator = PCE500Emulator(trace_enabled=False, perfetto_trace=False, save_lcd_on_exit=False)
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
    lcd_image.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    screen_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    
    # Update state dictionary
    emulator_state.update({
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
            "s": cpu_state["s"]
        },
        "flags": {
            "z": cpu_state["flags"]["z"],
            "c": cpu_state["flags"]["c"]
        },
        "instruction_count": emulator.instruction_count,
        "last_update_time": time.time(),
        "last_update_instructions": emulator.instruction_count
    })


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
                instruction_delta = current_instructions - emulator_state["last_update_instructions"]
                
                if (time_delta >= UPDATE_TIME_THRESHOLD or 
                    instruction_delta >= UPDATE_INSTRUCTION_THRESHOLD):
                    update_emulator_state()
                    
            except Exception as e:
                print(f"Emulator error: {e}")
                emulator_state["is_running"] = False
                break
        
        # Small delay to prevent CPU spinning too fast
        time.sleep(0.0001)


@app.route('/')
def index():
    """Serve the main web interface."""
    return render_template('index.html')


@app.route('/api/v1/state', methods=['GET'])
def get_state():
    """Get current emulator state."""
    with emulator_lock:
        # Always update state when explicitly requested
        if emulator is not None:
            update_emulator_state()
        
        return jsonify(emulator_state)


@app.route('/api/v1/key', methods=['POST'])
def handle_key():
    """Handle keyboard input."""
    data = request.get_json()
    if not data or 'key_code' not in data:
        return jsonify({"error": "Missing key_code"}), 400
    
    key_code = data['key_code']
    action = data.get('action', 'press')  # 'press' or 'release'
    
    with emulator_lock:
        if emulator is None:
            return jsonify({"error": "Emulator not initialized"}), 500
        
        try:
            if action == 'press':
                emulator.press_key(key_code)
                print(f"Key pressed: {key_code}")
            elif action == 'release':
                emulator.release_key(key_code)
                print(f"Key released: {key_code}")
            else:
                return jsonify({"error": f"Invalid action: {action}"}), 400
            
            # Return debug info for development
            debug_info = emulator.keyboard.get_debug_info()
            return jsonify({
                "status": "ok",
                "debug": debug_info
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500


@app.route('/api/v1/control', methods=['POST'])
def control_emulator():
    """Control emulator execution (run/pause/step/reset)."""
    global emulator_thread, emulator_state
    
    data = request.get_json()
    if not data or 'command' not in data:
        return jsonify({"error": "Missing command"}), 400
    
    command = data['command']
    
    with emulator_lock:
        if emulator is None and command != 'reset':
            return jsonify({"error": "Emulator not initialized"}), 500
        
        if command == 'run':
            if not emulator_state["is_running"]:
                emulator_state["is_running"] = True
                emulator_thread = threading.Thread(target=emulator_run_loop)
                emulator_thread.start()
            return jsonify({"status": "running"})
            
        elif command == 'pause':
            emulator_state["is_running"] = False
            if emulator_thread:
                emulator_thread.join()
                emulator_thread = None
            return jsonify({"status": "paused"})
            
        elif command == 'step':
            if not emulator_state["is_running"]:
                emulator.step()
                update_emulator_state()
            return jsonify({"status": "stepped"})
            
        elif command == 'reset':
            # Stop running if active
            emulator_state["is_running"] = False
            if emulator_thread:
                emulator_thread.join()
                emulator_thread = None
                
            # Reset or reinitialize emulator
            if emulator:
                emulator.reset()
            else:
                initialize_emulator()
                
            update_emulator_state()
            return jsonify({"status": "reset"})
            
        else:
            return jsonify({"error": f"Unknown command: {command}"}), 400


@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files."""
    return send_from_directory('static', filename)


if __name__ == '__main__':
    # Initialize emulator on startup
    try:
        initialize_emulator()
        print("PC-E500 emulator initialized successfully")
    except Exception as e:
        print(f"Failed to initialize emulator: {e}")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=8080)