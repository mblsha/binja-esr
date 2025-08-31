"""Web UI keyboard-IRQ integration test.

Simulates the Web UI flow using Flask's test client:
- Ensure the backend initializes and emulator is stopped
- Start the emulator (Run)
- Poll OCR endpoint every 500ms until non-empty and stable for 1s (best-effort)
- Press PF1 and verify KEY interrupt is delivered (by interrupt counters)
"""

from __future__ import annotations

import time
from typing import Any

import app as webapp
from sc62015.pysc62015.emulator import RegisterName
from sc62015.pysc62015.instr.opcodes import IMEMRegisters


def _get_json(client, path: str) -> dict[str, Any]:
    resp = client.get(path)
    assert resp.status_code == 200
    return resp.get_json()  # type: ignore[no-any-return]


def _post_json(client, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    resp = client.post(path, json=payload)
    assert resp.status_code == 200
    return resp.get_json()  # type: ignore[no-any-return]


def test_web_keyboard_triggers_key_interrupt() -> None:
    start_time = time.time()
    deadline = start_time + 30.0  # hard cap for this test
    # Initialize app + emulator
    webapp.initialize_emulator()
    client = webapp.app.test_client()

    # Ensure stopped
    state = _get_json(client, "/api/v1/state")
    if state.get("is_running"):
        _post_json(client, "/api/v1/control", {"command": "pause"})
        state = _get_json(client, "/api/v1/state")
    assert not state.get("is_running"), "Expected emulator to be stopped"

    # Prepare IMR/ISR and stacks to allow IRQ delivery when enabled
    with webapp.emulator_lock:
        emu = webapp.emulator
        assert emu is not None
        INTERNAL_MEMORY_START = 0x100000
        # Disable timers to avoid background IRQs in this test
        try:
            setattr(emu, "_timer_enabled", False)
        except Exception:
            pass
        emu.cpu.regs.set(RegisterName.S, 0xBFF00)
        emu.cpu.regs.set(RegisterName.U, 0xBFE00)
        # Mask interrupts initially; clear ISR
        emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.IMR, 0x00)
        emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR, 0x00)

    # Start (Run)
    _post_json(client, "/api/v1/control", {"command": "run"})

    # Poll OCR endpoint every 500ms until non-empty and stable for ~1s (best effort)
    last_txt = None
    last_change = time.time()
    timeout_end = min(time.time() + 10.0, deadline)
    while time.time() < timeout_end:
        ocr = _get_json(client, "/api/v1/ocr")
        if ocr.get("ok"):
            txt = (ocr.get("text") or "").strip()
            if txt:
                if txt != last_txt:
                    last_txt = txt
                    last_change = time.time()
                else:
                    if (time.time() - last_change) >= 1.0:
                        break
        time.sleep(0.5)
    # Proceed either way; OCR stability is not required for IRQ verification

    # Baseline KEY IRQ count
    state = _get_json(client, "/api/v1/state")
    base_key = int(((state.get("interrupts") or {}).get("by_source") or {}).get("KEY", 0))

    # Press PF1, then enable IRM|KEYM to deliver the armed key IRQ
    _post_json(client, "/api/v1/key", {"key_code": "KEY_F1", "action": "press"})
    with webapp.emulator_lock:
        emu = webapp.emulator
        assert emu is not None
        INTERNAL_MEMORY_START = 0x100000
        emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.IMR, 0x80 | 0x04)

    # Wait up to 5s for KEY IRQ count to increase (bounded by total deadline)
    end = min(time.time() + 5.0, deadline)
    delivered = False
    while time.time() < end:
        s = _get_json(client, "/api/v1/state")
        cur = int(((s.get("interrupts") or {}).get("by_source") or {}).get("KEY", 0))
        if cur > base_key:
            delivered = True
            break
        time.sleep(0.1)

    # Pause emulator
    _post_json(client, "/api/v1/control", {"command": "pause"})

    assert delivered, "KEY interrupt did not trigger within 5 seconds"

    # Final total timeout guard
    assert time.time() <= deadline, "Test exceeded 30s total execution time"
