"""Public keyboard handler re-export."""

from .keyboard_compat import (
    DEFAULT_DEBOUNCE_READS,
    PCE500KeyboardHandler,
)

__all__ = ["PCE500KeyboardHandler", "DEFAULT_DEBOUNCE_READS"]
