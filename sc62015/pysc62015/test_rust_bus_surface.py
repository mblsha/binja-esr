from __future__ import annotations

import pytest

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
