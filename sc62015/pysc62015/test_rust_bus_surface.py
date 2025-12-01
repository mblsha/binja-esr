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
