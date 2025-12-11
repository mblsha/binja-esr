from __future__ import annotations

import pytest

from pce500.memory import PCE500Memory
from pce500.memory_bus import MemoryOverlay

from sc62015.pysc62015.contract_harness import RustContractBackend


def _has_rust() -> bool:
    try:
        RustContractBackend()
        return True
    except RuntimeError:
        return False


@pytest.mark.skipif(not _has_rust(), reason="LLAMA backend unavailable")
def test_python_ranges_and_keyboard_bridge_surface():
    backend = RustContractBackend()

    # Keyboard overlay needs Python when bridge is disabled.
    assert backend.requires_python(0x100000 + 0xF0) is True
    backend.set_keyboard_bridge(True)
    assert backend.requires_python(0x100000 + 0xF0) is False

    # Python ranges should mark addresses as host-handled.
    backend.set_python_ranges([(0x2000, 0x200F)])
    assert backend.requires_python(0x2005) is True
    # Addresses outside declared ranges stay local.
    assert backend.requires_python(0x3000) is False


@pytest.mark.skipif(not _has_rust(), reason="LLAMA backend unavailable")
def test_overlay_helpers_exposed():
    backend = RustContractBackend()
    backend.add_ram_overlay(0x4000, 4, name="ram_overlay_pytest")
    backend.write(0x4000, 0xAA)
    backend.write(0x4001, 0xBB)
    assert backend.read(0x4000) == 0xAA
    assert backend.read(0x4001) == 0xBB

    backend.add_rom_overlay(0x5000, bytes([0x12, 0x34]), name="rom_overlay_pytest")
    assert backend.read(0x5000) == 0x12
    assert backend.read(0x5001) == 0x34

    card = bytes([0xCC] * 8192)
    backend.load_memory_card(card)
    assert backend.read(0x040000) == 0xCC


@pytest.mark.skipif(not _has_rust(), reason="LLAMA backend unavailable")
def test_overlay_logs_exposed():
    backend = RustContractBackend()
    backend.add_ram_overlay(0x6000, 2, name="log_overlay")
    backend.write(0x6000, 0xAB, pc=0x0100)
    _ = backend.read(0x6000, pc=0x0200)

    writes = backend.overlay_write_log()
    reads = backend.overlay_read_log()
    assert any(entry["overlay"] == "log_overlay" for entry in writes)
    assert any(entry["overlay"] == "log_overlay" for entry in reads)
    assert any(entry.get("pc") == 0x0100 for entry in writes)
    assert any(entry.get("pc") == 0x0200 for entry in reads)


@pytest.mark.skipif(not _has_rust(), reason="LLAMA backend unavailable")
def test_overlay_log_parity_with_python():
    # Python backend overlay logs
    py_mem = PCE500Memory()
    py_mem.add_overlay(
        MemoryOverlay(
            start=0x7000,
            end=0x7001,
            name="py_overlay",
            data=bytearray(2),
            read_only=False,
            read_handler=None,
            write_handler=None,
            perfetto_thread="Memory",
        )
    )
    py_mem.write_byte(0x7000, 0xCC, cpu_pc=0x0300)
    py_mem.read_byte(0x7000, cpu_pc=0x0400)
    py_writes = [log for log in py_mem._bus.write_log() if log.overlay == "py_overlay"]  # type: ignore[attr-defined]
    py_reads = [log for log in py_mem._bus.read_log() if log.overlay == "py_overlay"]  # type: ignore[attr-defined]

    # Rust backend overlay logs
    backend = RustContractBackend()
    backend.add_ram_overlay(0x7000, 2, name="py_overlay")
    backend.write(0x7000, 0xCC, pc=0x0300)
    _ = backend.read(0x7000, pc=0x0400)
    rs_writes = [
        log for log in backend.overlay_write_log() if log["overlay"] == "py_overlay"
    ]
    rs_reads = [
        log for log in backend.overlay_read_log() if log["overlay"] == "py_overlay"
    ]

    assert len(py_writes) == len(rs_writes) == 1
    assert len(py_reads) == len(rs_reads) == 1
    assert rs_writes[0]["value"] == py_writes[0].value
    assert rs_reads[0]["value"] == py_reads[0].value


@pytest.mark.skipif(not _has_rust(), reason="LLAMA backend unavailable")
def test_overlay_removal():
    backend = RustContractBackend()
    backend.add_ram_overlay(0x7100, 1, name="temp_overlay")
    backend.write(0x7100, 0x5A)
    assert backend.read(0x7100) == 0x5A
    backend.remove_overlay("temp_overlay")
    # After removal, overlay should be gone and reads fall back to external memory default 0.
    assert backend.read(0x7100) == 0x00
