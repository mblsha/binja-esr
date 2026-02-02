from __future__ import annotations

from sc62015.pysc62015.constants import ISRFlag
from sc62015.pysc62015.instr.opcodes import IMEMRegisters

from pce500 import PCE500Emulator


INTERNAL_MEMORY_START = 0x100000


def test_key_latch_survives_release_until_irq_delivered():
    emu = PCE500Emulator(perfetto_trace=False, save_lcd_on_exit=False)
    emu._timer_enabled = False  # type: ignore[attr-defined]

    # Mask interrupts so KEYI cannot be delivered yet.
    emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.IMR, 0x00)
    # Activate the KEY_F1 column (column 10 -> KOH bit 2) so scans enqueue events.
    emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.KOH, 0x04)

    assert emu.press_key("KEY_F1") is True
    emu.step()
    assert emu._key_irq_latched is True  # type: ignore[attr-defined]
    assert (
        emu.memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR)
        & int(ISRFlag.KEYI)
        != 0
    )

    # User releases the key before firmware unmasks IRQs; latch must remain.
    emu.release_key("KEY_F1")
    assert emu._key_irq_latched is True  # type: ignore[attr-defined]

    # Firmware clears ISR while still masked; step() should reassert KEYI from the latch.
    emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR, 0x00)
    emu.step()
    assert (
        emu.memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR)
        & int(ISRFlag.KEYI)
        != 0
    )
