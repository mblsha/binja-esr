# PC-E500 Web Emulator

A web-based emulator for the Sharp PC-E500 pocket computer, built with Flask and JavaScript.

## Features

- **Real-time emulation** of the PC-E500 with visual LCD display
- **Virtual keyboard** with full PC-E500 layout
- **CPU state monitoring** showing all registers and flags
- **Emulation controls** (Run, Pause, Step, Reset)
- **10 FPS screen updates** with intelligent throttling
- **Physical keyboard mapping** for convenient typing

## Architecture

The web UI talks to a single emulator implementation:

- **Flask Backend** (`app.py`): Manages the SC62015 + PC‑E500 emulator and provides REST API.
- **JavaScript Frontend** (`static/app.js`): Interactive UI with state polling.
- **Keyboard**: Handled by the emulator’s compat keyboard implementation (`pce500/keyboard_compat.py`). The web UI only calls API endpoints to press/release keys and to fetch debug/queue information.

## Quick Start

1. Install dependencies:
   ```bash
   cd web
   pip install -r requirements.txt
   ```

2. Run the server:
   ```bash
   python app.py
   ```

3. Open your browser to `http://localhost:8080`

## API Endpoints

### GET /api/v1/state
Returns the current emulator state including screen, registers, and flags.

### POST /api/v1/key
Simulates keyboard input:
```json
{
  "key_code": "KEY_A",
  "action": "press"  // or "release"
}
```

### POST /api/v1/control
Controls emulator execution:
```json
{
  "command": "run"  // or "pause", "step", "reset"
}
```

## Keyboard Mapping

The virtual keyboard mimics the PC-E500's physical layout. You can also use your physical keyboard:

- Letters A-Z → Virtual keys A-Z
- Numbers 0-9 → Virtual number keys
- Enter → ENTER key
- Backspace → Backspace key
- Arrow keys → Cursor keys
- Space → Space bar
- Tab → TAB key
- Escape → ON key

## Technical Details

### Update Logic

The emulator state updates when either condition is met:
- 100ms have elapsed (10 FPS target)
- 100,000 instructions have executed

This ensures responsive updates during both active computation and idle states.

### Keyboard Matrix

The PC‑E500 uses a matrix scanning system with three registers:
- `KOL` (0xF0): Key Output Low – selects columns KO0..KO7 (compat: active-high bits)
- `KOH` (0xF1): Key Output High – selects columns KO8..KO10 via bits 0..2 (compat: active‑high bits)
- `KIL` (0xF2): Key Input – reads rows KI0..KI7

This project now uses a single “compat” keyboard implementation that simulates debounced
press/release behavior consistent with firmware scanning patterns. To adjust layout or
debouncing, edit `pce500/keyboard_compat.py` (`KEYBOARD_LAYOUT`, `DEFAULT_*_READS`).

## Development

- To modify the keyboard layout, edit `KEYBOARD_LAYOUT` in `pce500/keyboard_compat.py`.
- To adjust update rates, modify `UPDATE_TIME_THRESHOLD` and `UPDATE_INSTRUCTION_THRESHOLD` in `web/app.py`.

## Known Limitations

- Sound emulation is not implemented
- Peripheral ports are not emulated
- Keyboard matrix mapping is based on available diagrams; minor refinements may be needed
