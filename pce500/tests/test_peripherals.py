from __future__ import annotations

from pce500.memory import PCE500Memory, INTERNAL_MEMORY_START
from pce500.scheduler import TimerScheduler
from pce500.emulator import (
    MTI_PERIOD_CYCLES_DEFAULT,
    STI_PERIOD_CYCLES_DEFAULT,
)
from pce500.peripherals import PeripheralManager, SerialQueuedByte
from pce500.peripherals.storage import (
    BlockNotFoundError,
    DuplicateBlockError,
    InsufficientSpaceError,
    MemoryCardImage,
    RamDiskImage,
)
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


def test_serial_input_lines_model_cs_and_cd_bits() -> None:
    manager = make_manager()
    serial = manager.serial
    memory = manager.memory

    serial.set_input_lines(cs=True, cd=False)
    assert serial.input_lines() == {"cs": True, "cd": False}
    eil_addr = INTERNAL_MEMORY_START + IMEMRegisters.EIL.value
    assert memory.read_byte(eil_addr) & 0x04
    assert not (memory.read_byte(eil_addr) & 0x02)

    serial.set_input_lines(cd=True)
    assert serial.input_lines() == {"cs": True, "cd": True}


def test_serial_snapshot_restores_rx_tx_and_line_contract() -> None:
    manager = make_manager()
    serial = manager.serial
    memory = manager.memory

    serial.queue_receive(0x41)
    memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.TXD.value, 0x55, cpu_pc=0)
    serial.set_input_lines(cs=True, cd=True)
    serial.set_handshake(0xA5)
    snap = serial.snapshot()

    serial.consume_received()
    serial.complete_transmit()
    serial.set_input_lines(cs=False, cd=False)
    serial.set_handshake(0x00)
    serial.restore(snap)

    assert [entry.value for entry in serial.pending_receive()] == [0x41]
    assert serial.pending_transmit() == [0x55]
    assert serial.input_lines() == {"cs": True, "cd": True}
    assert serial.get_handshake() == 0xA5


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


def test_cassette_tape_image_reads_and_verifies_deterministically() -> None:
    manager = make_manager()
    tape = manager.cassette.tape
    header = bytes(range(0x30))
    payload = b"HELLO"

    header_block = tape.append_header(header)
    data_block = tape.append_data(payload)

    assert header_block.checksum == sum(header) & 0xFF
    assert data_block.checksum == sum(payload) & 0xFF
    assert tape.read_next("header").payload == header
    assert tape.verify_next(payload)

    tape.rewind()
    assert tape.read_next("header").kind == "header"
    assert tape.read_next("data").payload == payload


def test_cassette_tape_image_rejects_wrong_block_shape() -> None:
    manager = make_manager()
    tape = manager.cassette.tape

    try:
        tape.append_header(b"short")
    except ValueError as exc:
        assert "0x30" in str(exc)
    else:
        raise AssertionError("expected short header to fail")


def test_stdio_buffer_helpers() -> None:
    manager = make_manager()
    stdio = manager.stdio

    stdio.load_output_buffer([0x11, 0x22, 0x33])
    snap = stdio.snapshot()
    assert snap.workspace[0x00BFD48] == 0x11
    assert snap.workspace[0x00BFD4A] == 0x33

    stdio.write_workspace(0x00BFD48, 0x77)
    assert stdio.read_workspace(0x00BFD48) == 0x77


def test_memory_card_edge_cases_and_ramfile_backing() -> None:
    card = MemoryCardImage(capacity=16)
    block = card.create("foo", 4)
    block.data[:] = b"ABCD"

    assert card.used == 4
    assert card.free == 12
    assert card.find("FOO").data == bytearray(b"ABCD")

    try:
        card.create("foo", 1)
    except DuplicateBlockError:
        pass
    else:
        raise AssertionError("expected duplicate block creation to fail")

    try:
        card.create("too-big", 32)
    except InsufficientSpaceError:
        pass
    else:
        raise AssertionError("expected oversized block creation to fail")

    card.resize("foo", 2)
    assert card.find("foo").data == bytearray(b"AB")
    card.rename("foo", "bar")
    assert card.find("bar").size == 2

    ramdisk = RamDiskImage(card)
    ramdisk.format(8)
    assert ramdisk.size == 8
    assert card.blocks[0].name == "RAMFILE"

    card.delete("bar")
    try:
        card.find("bar")
    except BlockNotFoundError:
        pass
    else:
        raise AssertionError("expected deleted block lookup to fail")
