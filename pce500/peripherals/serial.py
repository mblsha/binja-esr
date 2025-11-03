"""Serial (RS-232C) peripheral adapter."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional

from sc62015.pysc62015.constants import INTERNAL_MEMORY_START
from sc62015.pysc62015.instr.opcodes import IMEMRegisters

from ..memory import PCE500Memory
from ..scheduler import TimerScheduler

UART_REGISTER_ADDRS: Dict[IMEMRegisters, int] = {
    reg: INTERNAL_MEMORY_START + reg.value
    for reg in (
        IMEMRegisters.UCR,
        IMEMRegisters.USR,
        IMEMRegisters.RXD,
        IMEMRegisters.TXD,
    )
}

SERIAL_WORKSPACE_ADDRS: Iterable[int] = range(0x00BFE40, 0x00BFE48)
SERIAL_HANDSHAKE_ADDR = 0x00BFE46


@dataclass
class SerialQueuedByte:
    """Represents a single entry queued for reception."""

    value: int
    parity_error: bool = False
    overrun_error: bool = False
    framing_error: bool = False


@dataclass
class SerialSnapshot:
    """Snapshot of serial adapter state suitable for deterministic tests."""

    ucr: int
    usr: int
    handshake: int
    workspace: Dict[int, int] = field(default_factory=dict)
    rx_queue: List[SerialQueuedByte] = field(default_factory=list)
    tx_queue: List[int] = field(default_factory=list)


class SerialAdapter:
    """Utility wrapper for the PC-E500 serial workspace and UART registers."""

    _RX_READY_MASK = 0x20
    _TX_EMPTY_MASK = 0x10
    _TX_READY_MASK = 0x08
    _FRAMING_ERROR_MASK = 0x04
    _OVERRUN_ERROR_MASK = 0x02
    _PARITY_ERROR_MASK = 0x01

    def __init__(self, memory: PCE500Memory, scheduler: TimerScheduler) -> None:
        self._memory = memory
        self._scheduler = scheduler
        self._rx_queue: Deque[SerialQueuedByte] = deque()
        self._tx_queue: Deque[int] = deque()

    # ------------------------------------------------------------------ #
    # IMEM access hook
    # ------------------------------------------------------------------ #
    def handle_imem_access(
        self, pc: int, reg_name: str, access_type: str, value: int
    ) -> None:
        """Observe IMEM register accesses to maintain UART state."""

        if reg_name != "TXD" or access_type != "write":
            return

        self._tx_queue.append(value & 0xFF)
        self._mark_transmit_busy()

    # ------------------------------------------------------------------ #
    # Snapshot helpers
    # ------------------------------------------------------------------ #
    def snapshot(self) -> SerialSnapshot:
        """Capture the current UART and workspace state."""

        workspace = {
            addr: self._memory.read_byte(addr) for addr in SERIAL_WORKSPACE_ADDRS
        }
        return SerialSnapshot(
            ucr=self._read_register(IMEMRegisters.UCR),
            usr=self._read_register(IMEMRegisters.USR),
            handshake=self._memory.read_byte(SERIAL_HANDSHAKE_ADDR),
            workspace=workspace,
            rx_queue=list(self._rx_queue),
            tx_queue=list(self._tx_queue),
        )

    def restore(self, snapshot: SerialSnapshot) -> None:
        """Restore UART/workspace state from a previously captured snapshot."""

        self._write_register(IMEMRegisters.UCR, snapshot.ucr)
        self._write_register(IMEMRegisters.USR, snapshot.usr)
        for addr, value in snapshot.workspace.items():
            self._memory.write_byte(addr, value & 0xFF)
        self._memory.write_byte(SERIAL_HANDSHAKE_ADDR, snapshot.handshake & 0xFF)

        self._rx_queue = deque(snapshot.rx_queue)
        self._tx_queue = deque(snapshot.tx_queue)
        self._latch_next_received()
        self._update_transmit_status()

    # ------------------------------------------------------------------ #
    # Receive helpers
    # ------------------------------------------------------------------ #
    def queue_receive(
        self,
        value: int,
        *,
        parity_error: bool = False,
        overrun_error: bool = False,
        framing_error: bool = False,
    ) -> None:
        """Queue a byte for the ROM to consume via RXD."""

        queued = SerialQueuedByte(
            value=value & 0xFF,
            parity_error=parity_error,
            overrun_error=overrun_error,
            framing_error=framing_error,
        )
        self._rx_queue.append(queued)
        self._latch_next_received()

    def consume_received(self) -> Optional[SerialQueuedByte]:
        """Pop the next byte queued for reception (simulates ROM reading RXD)."""

        if not self._rx_queue:
            return None
        entry = self._rx_queue.popleft()
        self._latch_next_received()
        return entry

    def pending_receive(self) -> List[SerialQueuedByte]:
        """Return the current receive queue for inspection."""

        return list(self._rx_queue)

    # ------------------------------------------------------------------ #
    # Transmit helpers
    # ------------------------------------------------------------------ #
    def pending_transmit(self) -> List[int]:
        """Return the buffered transmit bytes."""

        return list(self._tx_queue)

    def complete_transmit(self) -> Optional[int]:
        """Simulate hardware completing a transmission and returning the byte."""

        if not self._tx_queue:
            return None
        value = self._tx_queue.popleft()
        self._update_transmit_status()
        return value

    # ------------------------------------------------------------------ #
    # Handshake utilities
    # ------------------------------------------------------------------ #
    def set_handshake(self, value: int) -> None:
        """Update the ROM's handshake shadow byte (0xBFE46)."""

        self._memory.write_byte(SERIAL_HANDSHAKE_ADDR, value & 0xFF)

    def get_handshake(self) -> int:
        """Return the handshake shadow byte."""

        return self._memory.read_byte(SERIAL_HANDSHAKE_ADDR) & 0xFF

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _mark_transmit_busy(self) -> None:
        usr = self._read_register(IMEMRegisters.USR)
        usr &= ~self._TX_READY_MASK
        usr &= ~self._TX_EMPTY_MASK
        self._write_register(IMEMRegisters.USR, usr)

    def _update_transmit_status(self) -> None:
        usr = self._read_register(IMEMRegisters.USR)
        if self._tx_queue:
            usr &= ~self._TX_READY_MASK
            usr &= ~self._TX_EMPTY_MASK
        else:
            usr |= self._TX_READY_MASK
            usr |= self._TX_EMPTY_MASK
        self._write_register(IMEMRegisters.USR, usr)

    def _latch_next_received(self) -> None:
        usr = self._read_register(IMEMRegisters.USR)
        usr &= ~(self._FRAMING_ERROR_MASK | self._OVERRUN_ERROR_MASK | self._PARITY_ERROR_MASK)

        if not self._rx_queue:
            usr &= ~self._RX_READY_MASK
            self._write_register(IMEMRegisters.USR, usr)
            return

        entry = self._rx_queue[0]
        self._memory.write_byte(
            UART_REGISTER_ADDRS[IMEMRegisters.RXD], entry.value & 0xFF
        )
        usr |= self._RX_READY_MASK
        if entry.framing_error:
            usr |= self._FRAMING_ERROR_MASK
        if entry.overrun_error:
            usr |= self._OVERRUN_ERROR_MASK
        if entry.parity_error:
            usr |= self._PARITY_ERROR_MASK

        self._write_register(IMEMRegisters.USR, usr)

    def _read_register(self, reg: IMEMRegisters) -> int:
        return self._memory.read_byte(UART_REGISTER_ADDRS[reg]) & 0xFF

    def _write_register(self, reg: IMEMRegisters, value: int) -> None:
        self._memory.write_byte(UART_REGISTER_ADDRS[reg], value & 0xFF)
