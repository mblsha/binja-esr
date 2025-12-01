from __future__ import annotations

import os
from pathlib import Path

import pytest

from pce500.emulator import PCE500Emulator
from sc62015.pysc62015.cpu import available_backends
from sc62015.pysc62015.emulator import RegisterName


_CALL_DEPTH = 1
_CALL_SUB_LEVEL = 1
_MEM_READS = 7
_MEM_WRITES = 5
_KB_IRQ_COUNT = 2
_KB_STROBE_COUNT = 4
_KB_COL_HIST = [1, 2]
_KB_LAST_COLS = [0, 1]
_KB_LAST_KOL = 0x12
_KB_LAST_KOH = 0x03
_KB_KIL_READS = 1


def _seed_state(emu: PCE500Emulator) -> None:
    # Prime counters/metadata that should round-trip through .pcsnap.
    emu.call_depth = _CALL_DEPTH
    emu.cpu.regs.call_sub_level = _CALL_SUB_LEVEL
    emu.memory_read_count = _MEM_READS
    emu.memory_write_count = _MEM_WRITES
    if getattr(emu.cpu, "backend", "python") == "llama":
        impl = emu.cpu.unwrap()
        try:
            impl.memory_reads = _MEM_READS
            impl.memory_writes = _MEM_WRITES
        except Exception:
            pass

    # Seed keyboard metrics and sync into snapshot metadata.
    emu._kb_irq_count = _KB_IRQ_COUNT
    emu._kb_strobe_count = _KB_STROBE_COUNT
    emu._kb_col_hist = list(_KB_COL_HIST)
    emu._last_kil_columns = list(_KB_LAST_COLS)
    emu._last_kol = _KB_LAST_KOL
    emu._last_koh = _KB_LAST_KOH
    emu._kil_read_count = _KB_KIL_READS
    emu._kb_irq_enabled = True

    # Seed LCD payload by issuing a couple of writes into the overlay window.
    emu.memory.write_byte(0x2000, 0x3F)  # instruction
    emu.memory.write_byte(0x2002, 0xAB)  # data

    # Touch a few registers to ensure state is non-zero.
    emu.cpu.regs.set(RegisterName.Y, 0x1234)


def _assert_state(emu: PCE500Emulator) -> None:
    assert emu.call_depth == _CALL_DEPTH
    assert emu.cpu.regs.call_sub_level == _CALL_SUB_LEVEL
    assert emu.memory_read_count == _MEM_READS
    assert emu.memory_write_count == _MEM_WRITES

    assert emu._kb_irq_count == _KB_IRQ_COUNT
    assert emu._kb_strobe_count == _KB_STROBE_COUNT
    assert list(emu._kb_col_hist) == _KB_COL_HIST
    assert list(emu._last_kil_columns) == _KB_LAST_COLS
    assert emu._last_kol == _KB_LAST_KOL
    assert emu._last_koh == _KB_LAST_KOH
    assert emu._kil_read_count == _KB_KIL_READS
    assert emu._kb_irq_enabled is True

    # LCD payload should have been restored.
    snap = emu.lcd.get_snapshot()
    assert any(0xAB in row for chip in snap.chips for row in chip.vram)


def _has_llama_backend() -> bool:
    return "llama" in available_backends()


@pytest.mark.skipif(not _has_llama_backend(), reason="LLAMA backend unavailable")
def test_snapshot_roundtrip_llama_to_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Build and seed a LLAMA-backed emulator snapshot.
    monkeypatch.setenv("SC62015_CPU_BACKEND", "llama")
    emu_llama = PCE500Emulator(
        trace_enabled=False,
        perfetto_trace=False,
        save_lcd_on_exit=False,
    )
    _seed_state(emu_llama)
    snap_path = tmp_path / "llama_snapshot.pcsnap"
    emu_llama.save_snapshot(snap_path)

    # Load into a Python-backed emulator and verify parity of metadata/state.
    monkeypatch.setenv("SC62015_CPU_BACKEND", "python")
    emu_python = PCE500Emulator(
        trace_enabled=False,
        perfetto_trace=False,
        save_lcd_on_exit=False,
    )
    emu_python.load_snapshot(snap_path, backend="python")
    _assert_state(emu_python)


@pytest.mark.skipif(not _has_llama_backend(), reason="LLAMA backend unavailable")
def test_snapshot_roundtrip_python_to_llama(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Build and seed a Python-backed emulator snapshot.
    monkeypatch.setenv("SC62015_CPU_BACKEND", "python")
    emu_python = PCE500Emulator(
        trace_enabled=False,
        perfetto_trace=False,
        save_lcd_on_exit=False,
    )
    _seed_state(emu_python)
    snap_path = tmp_path / "python_snapshot.pcsnap"
    emu_python.save_snapshot(snap_path)

    # Load into a LLAMA-backed emulator and verify parity of metadata/state.
    monkeypatch.setenv("SC62015_CPU_BACKEND", "llama")
    emu_llama = PCE500Emulator(
        trace_enabled=False,
        perfetto_trace=False,
        save_lcd_on_exit=False,
    )
    emu_llama.load_snapshot(snap_path, backend="llama")
    _assert_state(emu_llama)
