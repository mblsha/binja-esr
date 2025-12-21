from __future__ import annotations

import os

import pytest

from pce500 import PCE500Emulator
from sc62015.pysc62015.cpu import available_backends
from sc62015.pysc62015.emulator import RegisterName
from sc62015.pysc62015.instr.opcodes import IMEMRegisters, INTERNAL_MEMORY_START


WAIT_OPCODE = 0xEF
MVL_OPCODE = 0xCB
MVLD_OPCODE = 0xCF


@pytest.mark.parametrize("backend", available_backends())
def test_wait_advances_cycle_count_by_i(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SC62015_CPU_BACKEND", backend)
    if os.environ.get("FORCE_BINJA_MOCK") != "1":
        monkeypatch.setenv("FORCE_BINJA_MOCK", "1")

    emu = PCE500Emulator(
        trace_enabled=False, perfetto_trace=False, save_lcd_on_exit=False
    )
    try:
        emu.load_rom(bytes([WAIT_OPCODE]), start_address=0x0000)
        emu.cpu.regs.set(RegisterName.PC, 0x0000)
        emu.cpu.regs.set(RegisterName.I, 5)

        before = int(emu.cycle_count)
        emu.step()
        after = int(emu.cycle_count)

        assert after - before == 6
        assert int(emu.cpu.regs.get(RegisterName.I)) == 0
    finally:
        emu.close()


@pytest.mark.parametrize("backend", available_backends())
def test_mvl_advances_cycle_count_by_i(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SC62015_CPU_BACKEND", backend)
    if os.environ.get("FORCE_BINJA_MOCK") != "1":
        monkeypatch.setenv("FORCE_BINJA_MOCK", "1")

    emu = PCE500Emulator(
        trace_enabled=False, perfetto_trace=False, save_lcd_on_exit=False
    )
    try:
        emu.load_rom(bytes([MVL_OPCODE, 0x50, 0xA0]), start_address=0x0000)
        emu.cpu.regs.set(RegisterName.PC, 0x0000)
        emu.cpu.regs.set(RegisterName.I, 5)
        emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.BP, 0x00)

        src = [0x11, 0x22, 0x33, 0x44, 0x55]
        for idx, byte in enumerate(src):
            emu.memory.write_byte(INTERNAL_MEMORY_START + 0xA0 + idx, byte)
            emu.memory.write_byte(INTERNAL_MEMORY_START + 0x50 + idx, 0x00)

        before = int(emu.cycle_count)
        emu.step()
        after = int(emu.cycle_count)

        assert after - before == 6
        assert int(emu.cpu.regs.get(RegisterName.I)) == 0
        for idx, byte in enumerate(src):
            got = int(emu.memory.read_byte(INTERNAL_MEMORY_START + 0x50 + idx) & 0xFF)
            assert got == byte
    finally:
        emu.close()


# NOTE: MVLD is not yet implemented in the LLAMA backend (it currently behaves as a no-op),
# so only assert MVLD loop semantics against the Python backend.
@pytest.mark.parametrize("backend", ["python"])
def test_mvld_advances_cycle_count_by_i(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SC62015_CPU_BACKEND", backend)
    if os.environ.get("FORCE_BINJA_MOCK") != "1":
        monkeypatch.setenv("FORCE_BINJA_MOCK", "1")

    emu = PCE500Emulator(
        trace_enabled=False, perfetto_trace=False, save_lcd_on_exit=False
    )
    try:
        emu.load_rom(bytes([MVLD_OPCODE, 0x51, 0x50]), start_address=0x0000)
        emu.cpu.regs.set(RegisterName.PC, 0x0000)
        emu.cpu.regs.set(RegisterName.I, 3)
        emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.BP, 0x00)

        before = int(emu.cycle_count)
        emu.step()
        after = int(emu.cycle_count)

        assert after - before == 4
        assert int(emu.cpu.regs.get(RegisterName.I)) == 0
    finally:
        emu.close()
