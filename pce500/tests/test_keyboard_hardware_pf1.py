from pce500.keyboard_hardware import KeyboardHardware, KOL, KOH, KIL
from sc62015.pysc62015.constants import INTERNAL_MEMORY_START
from sc62015.pysc62015.instr.opcodes import IMEMRegisters


def _mem_read(addr: int) -> int:
    """Minimal memory accessor returning LCC with KSD=0 (strobing enabled)."""
    if addr == INTERNAL_MEMORY_START + IMEMRegisters.LCC:
        # KSD is bit 2; 0 means keyboard strobing enabled
        return 0x00
    return 0x00


def test_pf1_active_low_kil_and_column_mapping():
    kb = KeyboardHardware(memory_accessor=_mem_read, active_low=True)

    # Press PF1. In the hardware layout, PF1 is at column 10, row 6 (KO10, KI6)
    assert kb.press_key("KEY_F1")

    # Idle: no strobes => KIL should read as 0xFF (no rows detected)
    assert kb.read_register(KIL) == 0xFF

    # Strobe KO10 only (KOH bits 0..2 map to KO8..KO10). Active-low => clear bit 2
    kb.write_register(KOL, 0xFF)
    kb.write_register(KOH, 0xFF & ~(1 << 2))

    kil = kb.read_register(KIL)
    # Active-low KIL: row 6 cleared when PF1 (row 6) is pressed and column 10 is strobed
    assert kil == (0xFF & ~(1 << 6)), f"Expected KI6 low (0xBF), got 0x{kil:02X}"

    # Now strobe a different column (e.g., KO0 via KOL bit 0 cleared); PF1 should not appear
    kb.write_register(KOH, 0xFF)  # clear KO10 strobe
    kb.write_register(KOL, 0xFF & ~(1 << 0))  # strobe KO0
    kil2 = kb.read_register(KIL)
    assert kil2 == 0xFF, f"Expected no row detection for KO0, got 0x{kil2:02X}"

    # Verify active columns helper reports KO10 when strobed
    kb.write_register(KOL, 0xFF)
    kb.write_register(KOH, 0xFF & ~(1 << 2))
    assert 10 in kb.get_active_columns()
