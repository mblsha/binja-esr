from __future__ import annotations

from pce500.memory import PCE500Memory, INTERNAL_MEMORY_START
from pce500.scheduler import TimerScheduler
from pce500.emulator import (
    MTI_PERIOD_CYCLES_DEFAULT,
    STI_PERIOD_CYCLES_DEFAULT,
)
from pce500.peripherals import PeripheralManager, SerialQueuedByte
from sc62015.pysc62015.instr.opcodes import IMEMRegisters


def make_manager() -> PeripheralManager:
    memory = PCE500Memory()
    scheduler = TimerScheduler(
        mti_period=MTI_PERIOD_CYCLES_DEFAULT,
        sti_period=STI_PERIOD_CYCLES_DEFAULT,
    )
    manager = PeripheralManager(memory, scheduler)
    memory.set_imem_access_callback(manager.handle_imem_access)
    return manager


def test_serial_receive_queue_sets_status_bits() -> None:
    manager = make_manager()
    serial = manager.serial
    memory = manager.memory

    serial.queue_receive(0x41, parity_error=True, framing_error=True)

    rxd_addr = INTERNAL_MEMORY_START + IMEMRegisters.RXD.value
    usr_addr = INTERNAL_MEMORY_START + IMEMRegisters.USR.value

    assert memory.read_byte(rxd_addr) == 0x41
    usr = memory.read_byte(usr_addr)
    assert usr & 0x20  # RXR
    assert usr & 0x04  # FE
    assert usr & 0x01  # PE

    consumed = serial.consume_received()
    assert isinstance(consumed, SerialQueuedByte)
    assert consumed.value == 0x41
    assert consumed.framing_error
    assert consumed.parity_error

    usr_after = memory.read_byte(usr_addr)
    assert not (usr_after & 0x20)


def test_serial_transmit_queue_tracks_writes() -> None:
    manager = make_manager()
    serial = manager.serial
    memory = manager.memory

    txd_addr = INTERNAL_MEMORY_START + IMEMRegisters.TXD.value
    memory.write_byte(txd_addr, 0x55, cpu_pc=0)

    assert serial.pending_transmit() == [0x55]
    transmitted = serial.complete_transmit()
    assert transmitted == 0x55
    assert serial.pending_transmit() == []

    usr_addr = INTERNAL_MEMORY_START + IMEMRegisters.USR.value
    usr = memory.read_byte(usr_addr)
    assert usr & 0x10  # TXE
    assert usr & 0x08  # TXR


def test_cassette_snapshot_roundtrip() -> None:
    manager = make_manager()
    cassette = manager.cassette

    cassette.write_workspace(0x00BFE20, 0x12)
    cassette.write_workspace(0x00BFEF4, 0x99)
    snap = cassette.snapshot()

    cassette.write_workspace(0x00BFE20, 0x00)
    cassette.write_workspace(0x00BFEF4, 0x00)
    cassette.restore(snap)

    assert cassette.read_workspace(0x00BFE20) == 0x12
    assert cassette.read_workspace(0x00BFEF4) == 0x99


def test_stdio_buffer_helpers() -> None:
    manager = make_manager()
    stdio = manager.stdio

    stdio.load_output_buffer([0x11, 0x22, 0x33])
    snap = stdio.snapshot()
    assert snap.workspace[0x00BFD48] == 0x11
    assert snap.workspace[0x00BFD4A] == 0x33

    stdio.write_workspace(0x00BFD48, 0x77)
    assert stdio.read_workspace(0x00BFD48) == 0x77
