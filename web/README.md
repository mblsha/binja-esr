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

- **Flask Backend** (`app.py` + `emulator_service.py`): Provides the REST API and manages a shared emulator instance, including lifecycle, tracing and paced execution.
- **JavaScript Frontend** (`static/app.js`): Generated bundle built from modular sources in `static/js/`.
- **Keyboard**: Handled by the emulator’s keyboard handler implementation (`pce500/keyboard_handler.py`). The web UI only calls API endpoints to press/release keys and to fetch debug/queue information.

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

### Configuration

- By default the API is same-origin only. To opt into cross-origin requests set `PCE500_WEB_ALLOWED_ORIGINS` (comma-separated) or configure `WEB_ALLOWED_ORIGINS` in Flask config before importing `app`.

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

The emulator service emits state snapshots when either condition is met:
- 100ms have elapsed (10 FPS target)
- 100,000 instructions have executed

This ensures responsive updates during both active computation and idle states while avoiding CPU spin.

### Keyboard Matrix

The PC‑E500 uses a matrix scanning system with three registers:
- `KOL` (0xF0): Key Output Low – selects columns KO0..KO7 (handler uses active-high bits)
- `KOH` (0xF1): Key Output High – selects columns KO8..KO10 via bits 0..2 (handler uses active‑high bits)
- `KIL` (0xF2): Key Input – reads rows KI0..KI7

This project now uses a single keyboard handler implementation that simulates debounced
press/release behavior consistent with firmware scanning patterns. To adjust layout or
debouncing, edit `pce500/keyboard_handler.py` (`KEYBOARD_LAYOUT`, `DEFAULT_*_READS`).

## Development

- Frontend source lives under `static/js/`. After making changes run:
  ```bash
  uv run python scripts/build_frontend.py
  ```
  This regenerates the shipped bundle at `static/app.js`.
- To modify the keyboard layout, edit `KEYBOARD_LAYOUT` in `pce500/keyboard_handler.py`.
- To adjust update rates, modify `UPDATE_TIME_THRESHOLD` and `UPDATE_INSTRUCTION_THRESHOLD` in `web/emulator_service.py`.

## Known Limitations

- Sound emulation is not implemented
- Peripheral ports are not emulated
- Keyboard matrix mapping is based on available diagrams; minor refinements may be needed
