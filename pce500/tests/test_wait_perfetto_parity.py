from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from pce500.emulator import PCE500Emulator
from sc62015.pysc62015.cpu import available_backends
from sc62015.pysc62015.emulator import RegisterName
from sc62015.pysc62015.instr.opcodes import IMEMRegisters, INTERNAL_MEMORY_START


WAIT_OPCODE = 0xEF
WAIT_CYCLES = 600  # MTI default period is 500 cycles; ensure at least one tick fires.


def _run_wait_once(
    *,
    backend: str,
    perfetto: bool,
    trace_path: Path | None,
    monkeypatch: pytest.MonkeyPatch,
) -> int:
    monkeypatch.setenv("SC62015_CPU_BACKEND", backend)
    if os.environ.get("FORCE_BINJA_MOCK") != "1":
        monkeypatch.setenv("FORCE_BINJA_MOCK", "1")

    trace_kw = {}
    if trace_path is not None:
        trace_kw["trace_path"] = str(trace_path)

    emu = PCE500Emulator(
        trace_enabled=False,
        perfetto_trace=perfetto,
        enable_new_tracing=perfetto,
        save_lcd_on_exit=False,
        **trace_kw,
    )
    try:
        emu.load_rom(bytes([WAIT_OPCODE]), start_address=0x0000)
        emu.cpu.regs.set(RegisterName.PC, 0x0000)
        emu.cpu.regs.set(RegisterName.I, WAIT_CYCLES)
        # Clear ISR so the WAIT-induced timer tick shows up deterministically.
        emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR, 0x00)
        emu.step()
        return int(
            emu.memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR) & 0xFF
        )
    finally:
        emu.close()


def test_wait_side_effects_match_without_tracing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert "llama" in available_backends(), (
        "LLAMA backend must be available for parity tests"
    )

    isr_py = _run_wait_once(
        backend="python", perfetto=False, trace_path=None, monkeypatch=monkeypatch
    )
    isr_ll = _run_wait_once(
        backend="llama", perfetto=False, trace_path=None, monkeypatch=monkeypatch
    )
    assert isr_py == isr_ll
    assert isr_py & 0x01, "WAIT should advance timers enough to raise MTI at least once"


def test_wait_perfetto_trace_matches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert "llama" in available_backends(), (
        "LLAMA backend must be available for parity tests"
    )

    repo_root = Path(__file__).resolve().parents[2]
    compare_script = repo_root / "scripts" / "compare_perfetto_traces.py"

    trace_py = tmp_path / "wait_python.perfetto-trace"
    trace_ll = tmp_path / "wait_llama.perfetto-trace"
    rust_trace_ll = Path(str(trace_ll) + ".rust")

    _ = _run_wait_once(
        backend="python", perfetto=True, trace_path=trace_py, monkeypatch=monkeypatch
    )
    _ = _run_wait_once(
        backend="llama", perfetto=True, trace_path=trace_ll, monkeypatch=monkeypatch
    )

    assert trace_py.is_file()
    assert rust_trace_ll.is_file()

    out = subprocess.run(
        [sys.executable, str(compare_script), str(trace_py), str(rust_trace_ll)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, (
        f"Perfetto traces diverged (exit {out.returncode})\nSTDOUT:\n{out.stdout}\nSTDERR:\n{out.stderr}"
    )
