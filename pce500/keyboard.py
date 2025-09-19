"""Compat-only keyboard with legacy read semantics for tests.

This module exposes a keyboard handler class with the same name as before,
implemented on top of the single compat keyboard, but with legacy read
semantics expected by tests in this repository:

- Active-high behavior when reading KIL
- While a key's column is strobed, its row bit is visible immediately,
  even before the press debounce threshold is reached (read_count increments
  on each read). Release debounce still applies before removal.
"""

from .keyboard_compat import PCE500KeyboardHandler as _CompatHandler
from .keyboard_compat import DEFAULT_DEBOUNCE_READS


class PCE500KeyboardHandler(_CompatHandler):
    """Compat keyboard with legacy immediate-visible row bit behavior."""

    def _read_keyboard_input(self) -> int:  # type: ignore[override]
        """Return KIL (active-high) with immediate row visibility while strobed.

        - If a queued key's column is currently strobed, OR its row bit into the
          result regardless of whether debounce has marked it active yet; and
          increment press debounce read_count.
        - When not strobed: if the key was released and had become active,
          count towards release debounce and deactivate when threshold is met.
        - Remove keys only after they've been released and deactivated.
        """
        result = 0x00
        completed_keys = []

        for queued_key in self.key_queue:
            strobed = queued_key.matches_output(self._last_kol, self._last_koh)
            if strobed:
                # Expose row bit immediately (legacy behavior) and count press reads
                result |= (1 << queued_key.row) & 0xFF
                queued_key.increment_read()
                # While strobed, do not count release
                queued_key.release_reads = 0
                # Legacy completion: after press debounce threshold, allow removal post-release
                if queued_key.read_count >= queued_key.target_reads:
                    completed_keys.append(queued_key)
            else:
                # Not strobed: if released and previously active, count towards release
                if queued_key.released and queued_key.active:
                    queued_key.release_reads += 1
                    if queued_key.release_reads >= queued_key.release_target_reads:
                        queued_key.active = False
                # If fully complete (released and inactive), schedule removal
                if queued_key.released and (
                    queued_key.read_count >= queued_key.target_reads
                ):
                    completed_keys.append(queued_key)

        for ck in completed_keys:
            # Remove only after physical release has been signaled
            if ck.released:
                self.key_queue.remove(ck)
                self.pressed_keys.discard(ck.key_code)

        return result & 0xFF

    def peek_keyboard_input(self) -> int:  # type: ignore[override]
        """Preview KIL without affecting debounce/queue state."""

        result = 0x00

        for queued_key in self.key_queue:
            if queued_key.matches_output(self._last_kol, self._last_koh):
                result |= (1 << queued_key.row) & 0xFF

        return result & 0xFF


__all__ = ["PCE500KeyboardHandler", "DEFAULT_DEBOUNCE_READS"]
