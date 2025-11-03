"""Peripheral manager wiring serial, cassette, and STDI adapters."""

from __future__ import annotations

from typing import Optional

from ..memory import PCE500Memory
from ..scheduler import TimerScheduler
from .serial import SerialAdapter
from .cassette import CassetteAdapter
from .stdio import StdIODeviceAdapter


class PeripheralManager:
    """Coordinates peripheral adapters and IMEM callbacks."""

    def __init__(self, memory: PCE500Memory, scheduler: TimerScheduler) -> None:
        self.memory = memory
        self.scheduler = scheduler

        self.serial = SerialAdapter(memory, scheduler)
        self.cassette = CassetteAdapter(memory, scheduler)
        self.stdio = StdIODeviceAdapter(memory, scheduler)

    def handle_imem_access(
        self, pc: int, reg_name: Optional[str], access_type: str, value: int
    ) -> None:
        """Dispatch IMEM register notifications to interested adapters."""

        if not reg_name:
            return
        self.serial.handle_imem_access(pc, reg_name, access_type, value)
