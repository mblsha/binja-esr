from __future__ import annotations

from pathlib import Path

import pytest

from pce500 import PCE500Emulator
from sc62015.pysc62015.instr.opcodes import IMEMRegisters


def _rom_image() -> bytes:
    rom_path = Path(__file__).parent.parent.parent / "data" / "pc-e500.bin"
    if not rom_path.exists():
        pytest.skip(f"ROM file {rom_path} not found")
    data = rom_path.read_bytes()
    if len(data) != 0x100000:
        pytest.skip(f"Expected 1MB ROM image at {rom_path} (got {len(data)} bytes)")
    return data


def _warm_until_key_interrupt(
    emu: PCE500Emulator, target: int, chunk_steps: int = 20_000, max_chunks: int = 10
) -> None:
    """Run the emulator in chunks until KEY interrupts reach the target."""
    for _ in range(max_chunks):
        emu.run(chunk_steps)
        stats = emu.get_interrupt_stats()
        if stats["by_source"]["KEY"] >= target:
            return
    pytest.fail("Keyboard interrupt count did not increase as expected")


def test_keyboard_interrupt_delivery_from_rom():
    """Verify the ROM fast-timer ISR raises KEY interrupts after bootstrap."""
    rom_image = _rom_image()

    # Create emulator with ROM code loaded and restore the runtime RAM snapshot.
    emu = PCE500Emulator(save_lcd_on_exit=False, perfetto_trace=False)
    emu.load_rom(rom_image[0xC0000:0x100000], start_address=0xC0000)
    emu.bootstrap_from_rom_image(rom_image)

    # Ensure IMR/ISR match the expected defaults even if the caller overrides arguments.
    imr_addr = 0x100000 + IMEMRegisters.IMR
    isr_addr = 0x100000 + IMEMRegisters.ISR
    assert emu.memory.read_byte(imr_addr) == 0x43
    assert emu.memory.read_byte(isr_addr) == 0x00

    # IOCS keyboard mode byte must be writable for the fast-timer ISR to enqueue events.
    emu.memory.write_byte(0xBFD1D, 0x02)
    emu.memory.write_byte(0xBFD1E, 0x00)

    # Give the ROM a head start so timer interrupts are active.
    def _run_with_mask(total_steps: int, mask: int) -> None:
        remaining = total_steps
        while remaining > 0:
            chunk = min(5_000, remaining)
            emu.memory.write_byte(imr_addr, mask & 0xFF)
            emu.memory.write_byte(isr_addr, 0x00)
            emu.run(chunk)
            remaining -= chunk

    _run_with_mask(200_000, 0x00)
    stats_before = emu.get_interrupt_stats()
    key_before = stats_before["by_source"]["KEY"]
    kb_irq_before = emu._kb_irq_count

    # Drive a key press long enough for the ROM to observe it.
    emu.press_key("KEY_A")
    _warm_until_key_interrupt(emu, key_before + 1)

    # Release and allow the ROM to service the trailing edge.
    emu.release_key("KEY_A")
    emu.run(40_000)

    stats_after = emu.get_interrupt_stats()
    key_after = stats_after["by_source"]["KEY"]
    kb_irq_after = emu._kb_irq_count

    assert key_after > key_before, "Keyboard interrupt counter did not increment"
    assert kb_irq_after > kb_irq_before, "Emulator-level IRQ accounting did not advance"
    assert emu._kb_strobe_count > 0, "ROM never toggled keyboard columns"
    assert emu.memory.read_byte(0xBFD34) & 0x80, (
        "Keyboard scanner gate should stay enabled"
    )


def test_keyboard_interrupt_blocked_when_masked():
    """Ensure masking the KEY bit suppresses delivery even after bootstrap."""
    rom_image = _rom_image()
    emu = PCE500Emulator(save_lcd_on_exit=False, perfetto_trace=False)
    emu.load_rom(rom_image[0xC0000:0x100000], start_address=0xC0000)
    emu.bootstrap_from_rom_image(rom_image)

    imr_addr = 0x100000 + IMEMRegisters.IMR
    emu.memory.write_byte(0xBFD1D, 0x02)
    emu.memory.write_byte(0xBFD1E, 0x00)

    # Mask out all sources and clear pending bits before any warm-up.
    emu.memory.write_byte(imr_addr, 0x00)
    isr_addr = 0x100000 + IMEMRegisters.ISR
    emu.memory.write_byte(isr_addr, 0x00)

    def _run_with_mask(total_steps: int) -> None:
        for _ in range(total_steps):
            emu.memory.write_byte(imr_addr, 0x00)
            emu.memory.write_byte(isr_addr, 0x00)
            emu.step()

    _run_with_mask(120_000)
    stats_before = emu.get_interrupt_stats()
    key_before = stats_before["by_source"]["KEY"]
    kb_irq_before = emu._kb_irq_count

    emu.press_key("KEY_A")
    _run_with_mask(40_000)
    emu.release_key("KEY_A")
    _run_with_mask(20_000)

    stats_after = emu.get_interrupt_stats()
    key_after = stats_after["by_source"]["KEY"]
    kb_irq_after = emu._kb_irq_count

    assert key_after == key_before, "All-masked IMR should block dispatcher delivery"
    assert kb_irq_after == kb_irq_before, "Hardware IRQ count should remain unchanged"
