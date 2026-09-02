"""Hardware-backed raw KIL/KEYI and separate host-event bookkeeping."""

from __future__ import annotations

from sc62015.pysc62015.constants import ISRFlag
from sc62015.pysc62015.instr.opcodes import IMEMRegisters

from pce500 import PCE500Emulator


INTERNAL_MEMORY_START = 0x100000


def test_raw_keyi_survives_release_until_firmware_acknowledges_it():
    emu = PCE500Emulator(perfetto_trace=False, save_lcd_on_exit=False)
    emu._timer_enabled = False  # type: ignore[attr-defined]

    # Mask interrupts so KEYI cannot be delivered yet.
    emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.IMR, 0x00)
    # Activate the KEY_F1 column (column 10 -> KOH bit 2) so scans enqueue events.
    emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.KOH, 0x04)

    assert emu.press_key("KEY_F1") is True
    emu.step()
    assert (
        emu.memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR)
        & int(ISRFlag.KEYI)
        != 0
    )

    # Releasing the key removes the electrical level but does not itself clear
    # the status bit already latched by hardware.
    emu.release_key("KEY_F1")
    assert (
        emu.memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR)
        & int(ISRFlag.KEYI)
        != 0
    )

    # Firmware acknowledgement after physical release clears raw KEYI. The
    # host FIFO latch is separate and must not manufacture another silicon
    # level request.
    emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR, 0x00)
    emu.step()
    assert (
        emu.memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR)
        & int(ISRFlag.KEYI)
        == 0
    )


def test_selected_held_key_relatches_keyi_while_handler_is_active():
    emu = PCE500Emulator(perfetto_trace=False, save_lcd_on_exit=False)
    emu._timer_enabled = False  # type: ignore[attr-defined]
    emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.IMR, 0x00)
    emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.KOH, 0x04)
    assert emu.press_key("KEY_F1") is True

    emu._in_interrupt = True  # type: ignore[attr-defined]
    emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR, 0x00)
    emu.step()

    assert (
        emu.memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR)
        & int(ISRFlag.KEYI)
        != 0
    )

    emu.release_key("KEY_F1")
    emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR, 0x00)
    emu.step()
    assert (
        emu.memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR)
        & int(ISRFlag.KEYI)
        == 0
    )
