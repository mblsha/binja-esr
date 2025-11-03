"""Peripheral adapters for the PC-E500 emulator."""

from .manager import PeripheralManager
from .serial import SerialAdapter, SerialSnapshot, SerialQueuedByte
from .cassette import CassetteAdapter, CassetteSnapshot
from .stdio import StdIODeviceAdapter, StdIOSnapshot

__all__ = [
    "PeripheralManager",
    "SerialAdapter",
    "SerialSnapshot",
    "SerialQueuedByte",
    "CassetteAdapter",
    "CassetteSnapshot",
    "StdIODeviceAdapter",
    "StdIOSnapshot",
]
