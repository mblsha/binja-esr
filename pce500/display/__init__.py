"""Display subsystem for PC-E500 emulator."""

from .hd61202 import HD61202Controller, HD61202

__all__ = ["HD61202Controller", "HD61202"]