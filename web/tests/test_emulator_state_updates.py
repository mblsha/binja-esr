"""Tests for emulator service state management."""

from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Make sure app module is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["FORCE_BINJA_MOCK"] = "1"

from emulator_service import EmulatorService  # type: ignore  # noqa: E402


class EmulatorServiceStateTests(unittest.TestCase):
    """Validate the EmulatorService state update behaviour."""

    def setUp(self) -> None:
        self.service = EmulatorService()
        self.service._state = self.service._initial_state()

        # Mock emulator with minimal surface
        self.mock_emulator = Mock()
        self.mock_emulator.instruction_count = 0
        self.mock_emulator.instruction_history = []
        self.mock_emulator.get_cpu_state.return_value = {
            "pc": 0x100,
            "a": 0x01,
            "b": 0x02,
            "ba": 0x0102,
            "i": 0x0304,
            "x": 0x000500,
            "y": 0x000600,
            "u": 0x000700,
            "s": 0x000800,
            "flags": {"z": 1, "c": 0},
        }
        self.mock_emulator.memory.read_byte.return_value = 0
        self.mock_emulator.get_interrupt_stats.return_value = {"total": 0}

        mock_image = Mock()

        def fake_save(buffer, format="PNG"):
            buffer.write(b"PNG")

        mock_image.save.side_effect = fake_save
        self.mock_emulator.lcd.get_combined_display.return_value = mock_image

        self.service._emulator = self.mock_emulator

    def test_update_state_basic(self):
        """_update_state_locked populates expected fields."""
        self.service._update_state_locked(self.mock_emulator, force=True)
        state = self.service._state

        self.assertEqual(state["registers"]["pc"], 0x100)
        self.assertEqual(state["flags"]["z"], 1)
        self.assertEqual(state["flags"]["c"], 0)
        self.assertTrue(str(state["screen"]).startswith("data:image/png;base64,"))
        self.assertEqual(state["instruction_count"], 0)

    def test_maybe_update_state_respects_time_threshold(self):
        """Updates are skipped when thresholds are unmet."""
        self.service._state["is_running"] = True
        self.service._state["last_update_time"] = time.time()
        self.service._state["last_update_instructions"] = 0

        with patch.object(self.service, "_update_state_locked") as mock_update:
            self.service._maybe_update_state_locked(self.mock_emulator)
            mock_update.assert_not_called()

    def test_maybe_update_state_triggers_when_elapsed(self):
        """Updates occur when thresholds are exceeded."""
        self.service._state["is_running"] = True
        self.service._state["last_update_time"] = time.time() - (
            self.service.UPDATE_TIME_THRESHOLD + 0.01
        )
        self.service._state["last_update_instructions"] = 0

        with patch.object(self.service, "_update_state_locked") as mock_update:
            self.service._maybe_update_state_locked(self.mock_emulator)
            mock_update.assert_called_once()

    def test_snapshot_state_forces_refresh_when_paused(self):
        """snapshot_state forces refresh when emulator is paused."""
        self.service._state["is_running"] = False
        with patch.object(self.service, "_update_state_locked") as mock_update:
            self.service.snapshot_state()
            mock_update.assert_called_once()

    def test_keyboard_register_state_format(self):
        """keyboard_register_state exposes public API values."""
        self.mock_emulator.get_keyboard_register_state.return_value = {
            "kol": 0x12,
            "koh": 0x34,
            "kil": 0x56,
        }
        registers = self.service.keyboard_register_state()
        self.assertEqual(registers, {"kol": "0x12", "koh": "0x34", "kil": "0x56"})

    def test_press_key_updates_emulator(self):
        """press_key delegates to emulator and updates state."""
        self.mock_emulator.press_key.return_value = True
        queued = self.service.press_key("KEY_A")
        self.assertTrue(queued)
        self.mock_emulator.press_key.assert_called_with("KEY_A")


if __name__ == "__main__":
    unittest.main()
