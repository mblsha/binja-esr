from __future__ import annotations

from pce500.memory_bus import MemoryBus, MemoryOverlay


def test_memory_bus_reads_data_overlay() -> None:
    bus = MemoryBus()
    data = bytearray([0xAA, 0xBB])
    bus.add_overlay(
        MemoryOverlay(start=0x2000, end=0x2001, name="rom", data=data, read_only=True)
    )

    result = bus.read(0x2000)

    assert result is not None
    assert result.value == 0xAA
    assert result.overlay.name == "rom"
    log = bus.read_log()
    assert len(log) == 1 and log[0].value == 0xAA


def test_memory_bus_write_mutable_overlay_updates_data() -> None:
    bus = MemoryBus()
    data = bytearray([0x10])
    bus.add_overlay(
        MemoryOverlay(
            start=0x4000,
            end=0x4000,
            name="ram",
            data=data,
            read_only=False,
        )
    )

    write_result = bus.write(0x4000, 0x5A)

    assert write_result is not None
    assert data[0] == 0x5A
    write_log = bus.write_log()
    assert len(write_log) == 1
    assert write_log[0].value == 0x5A
    assert write_log[0].previous == 0x10


def test_memory_bus_invokes_read_handler() -> None:
    captured: list[int] = []

    def handler(address: int, pc: int | None) -> int:
        captured.append(address)
        assert pc == 0x123456
        return 0x42

    bus = MemoryBus()
    bus.add_overlay(
        MemoryOverlay(
            start=0x6000,
            end=0x6000,
            name="handler",
            read_handler=handler,
        )
    )

    result = bus.read(0x6000, cpu_pc=0x123456)

    assert captured == [0x6000]
    assert result is not None and result.value == 0x42


def test_memory_bus_write_handler_invoked_for_overlay() -> None:
    writes: list[tuple[int, int, int | None]] = []

    def handler(address: int, value: int, pc: int | None) -> None:
        writes.append((address, value, pc))

    bus = MemoryBus()
    bus.add_overlay(
        MemoryOverlay(
            start=0x7000,
            end=0x7000,
            name="io",
            write_handler=handler,
            read_only=False,
        )
    )

    result = bus.write(0x7000, 0x3C, cpu_pc=0x777)

    assert result is not None
    assert writes == [(0x7000, 0x3C, 0x777)]
