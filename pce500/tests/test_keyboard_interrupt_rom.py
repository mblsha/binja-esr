from __future__ import annotations

from pathlib import Path

import pytest

from pce500 import PCE500Emulator
from pce500.emulator import IMRFlag, ISRFlag, IRQSource
from sc62015.pysc62015 import RegisterName
from sc62015.pysc62015.instr.opcodes import IMEMRegisters


DISPATCHER = 0xF1FB5
DISPATCHER_ACK = 0xF2059
KEY_HANDLER = 0xF2061
ROM_VECTOR_TABLE = 0xF0C44
RAM_VECTOR_TABLE = 0xBFCC6
RETURN_PC = 0xB8000
SYSTEM_FRAME = 0xBFE00
USER_STACK = 0xBFF00


def _rom_image() -> bytes:
    rom_path = Path(__file__).parent.parent.parent / "data" / "pc-e500-en.bin"
    if not rom_path.exists():
        pytest.skip(f"ROM file {rom_path} not found")
    data = rom_path.read_bytes()
    if len(data) != 0x100000:
        pytest.skip(f"Expected 1MB ROM image at {rom_path} (got {len(data)} bytes)")
    return data


def _warm_until_host_key_event(
    emu: PCE500Emulator, target: int, chunk_steps: int = 20_000, max_chunks: int = 10
) -> None:
    """Run in chunks until the host keyboard bridge observes a scan event."""
    for _ in range(max_chunks):
        emu.run(chunk_steps)
        if emu._kb_irq_count >= target:
            return
    pytest.fail("Host keyboard event count did not increase as expected")


def _warm_until_keyboard_strobes(
    emu: PCE500Emulator, target: int, chunk_steps: int = 10_000, max_chunks: int = 10
) -> None:
    """Run in chunks until the ROM starts strobing keyboard columns."""
    for _ in range(max_chunks):
        emu.run(chunk_steps)
        if emu._kb_strobe_count >= target:
            return
    pytest.fail("Keyboard strobe count did not increase as expected")


def _run_dispatcher(
    rom_image: bytes, *, active_imr: int, saved_imr: int, isr: int
) -> tuple[PCE500Emulator, list[int]]:
    """Execute the stock interrupt dispatcher from a hardware-shaped frame."""
    emu = PCE500Emulator(save_lcd_on_exit=False, perfetto_trace=False)
    emu.load_rom(rom_image[0xC0000:0x100000], start_address=0xC0000)
    emu._timer_enabled = False
    emu._in_interrupt = True
    emu._irq_pending = False
    emu._irq_source = IRQSource.KEY
    emu._key_irq_latched = False

    vector_bytes = rom_image[ROM_VECTOR_TABLE : ROM_VECTOR_TABLE + 24]
    for offset, value in enumerate(vector_bytes):
        emu.memory.write_byte(RAM_VECTOR_TABLE + offset, value)

    imr_addr = 0x100000 + IMEMRegisters.IMR
    isr_addr = 0x100000 + IMEMRegisters.ISR
    emu.memory.write_byte(imr_addr, active_imr)
    emu.memory.write_byte(isr_addr, isr)

    # Hardware frame at S: saved IMR, saved F, saved 24-bit return PC.
    emu.memory.write_byte(SYSTEM_FRAME, saved_imr)
    emu.memory.write_byte(SYSTEM_FRAME + 1, 0)
    emu.memory.write_bytes(3, SYSTEM_FRAME + 2, RETURN_PC)
    emu.memory.write_byte(RETURN_PC, 0x00)
    emu.cpu.regs.set(RegisterName.S, SYSTEM_FRAME)
    emu.cpu.regs.set(RegisterName.U, USER_STACK)
    emu.cpu.regs.set(RegisterName.PC, DISPATCHER)

    visited: list[int] = []
    for _ in range(80):
        visited.append(emu.cpu.regs.get(RegisterName.PC))
        emu.step()
        if emu.cpu.regs.get(RegisterName.PC) == RETURN_PC:
            return emu, visited
    pytest.fail("ROM interrupt dispatcher did not return within 80 instructions")


def test_rom_keyboard_scan_path_observes_host_key_without_key_irq_delivery():
    """Separate ROM scan activity from the masked host KEYI bridge."""
    rom_image = _rom_image()

    # Create emulator with ROM code loaded and restore the runtime RAM snapshot.
    emu = PCE500Emulator(save_lcd_on_exit=False, perfetto_trace=False)
    emu.load_rom(rom_image[0xC0000:0x100000], start_address=0xC0000)
    emu.bootstrap_from_rom_image(rom_image)
    emu._trace_execution = lambda *_args, **_kwargs: None
    emu._emit_instruction_trace_event = lambda *_args, **_kwargs: None

    # Defaults are IRM | EXM | STM | MTM; KEYM is intentionally disabled.
    imr_addr = 0x100000 + IMEMRegisters.IMR
    isr_addr = 0x100000 + IMEMRegisters.ISR
    assert emu.memory.read_byte(imr_addr) == 0x43
    assert emu.memory.read_byte(isr_addr) == 0x00

    # IOCS keyboard mode byte must be writable for the fast-timer ISR to enqueue events.
    emu.memory.write_byte(0xBFD1D, 0x02)
    emu.memory.write_byte(0xBFD1E, 0x00)

    # Give the ROM a head start so timer interrupts and keyboard scanning are active.
    _warm_until_keyboard_strobes(emu, target=2)
    stats_before = emu.get_interrupt_stats()
    key_before = stats_before["by_source"]["KEY"]
    kb_irq_before = emu._kb_irq_count

    # Drive a key press long enough for the ROM to observe it.
    emu.press_key("KEY_A")
    _warm_until_host_key_event(emu, kb_irq_before + 1)

    # Release without waiting for the trailing edge; this test only asserts
    # positive KEY interrupt delivery from the press path.
    emu.release_key("KEY_A")

    stats_after = emu.get_interrupt_stats()
    key_after = stats_after["by_source"]["KEY"]
    kb_irq_after = emu._kb_irq_count

    assert key_after == key_before, "Masked KEYM must not be reported as delivered"
    assert kb_irq_after > kb_irq_before, "Host keyboard scan bridge did not observe key"
    assert emu._kb_strobe_count > 0, "ROM never toggled keyboard columns"
    assert emu.memory.read_byte(0xBFD34) & 0x80, (
        "Keyboard scanner gate should stay enabled"
    )


def test_rom_dispatcher_acknowledges_key_and_disables_key_mask_on_return():
    """The ROM, rather than RETI, clears KEYI and disables its default vector."""
    rom_image = _rom_image()
    emu, visited = _run_dispatcher(
        rom_image,
        active_imr=int(IMRFlag.KEYM),
        saved_imr=int(IMRFlag.IRM | IMRFlag.KEYM),
        isr=int(ISRFlag.KEYI),
    )
    imr_addr = 0x100000 + IMEMRegisters.IMR
    isr_addr = 0x100000 + IMEMRegisters.ISR
    assert KEY_HANDLER in visited
    assert DISPATCHER_ACK in visited
    assert emu.memory.read_byte(isr_addr) & int(ISRFlag.KEYI) == 0
    assert emu.memory.read_byte(imr_addr) == int(IMRFlag.IRM)
    assert emu.cpu.regs.get(RegisterName.S) == SYSTEM_FRAME + 5
    assert emu.cpu.regs.get(RegisterName.U) == USER_STACK


def test_rom_dispatcher_leaves_masked_key_status_pending():
    """A masked KEYI takes the no-dispatch RETI path and remains pending."""
    rom_image = _rom_image()
    emu, visited = _run_dispatcher(
        rom_image,
        active_imr=0,
        saved_imr=int(IMRFlag.IRM),
        isr=int(ISRFlag.KEYI),
    )
    isr_addr = 0x100000 + IMEMRegisters.ISR
    assert KEY_HANDLER not in visited
    assert DISPATCHER_ACK not in visited
    assert emu.memory.read_byte(isr_addr) & int(ISRFlag.KEYI)


def test_rom_dispatcher_prioritizes_key_over_main_timer():
    """KEYI wins over MTI, matching the stock RX→EX→TX→ON→KEY→ST→MT order."""
    rom_image = _rom_image()
    enabled = IMRFlag.IRM | IMRFlag.KEYM | IMRFlag.MTM
    pending = ISRFlag.KEYI | ISRFlag.MTI
    emu, visited = _run_dispatcher(
        rom_image,
        active_imr=int(enabled & ~IMRFlag.IRM),
        saved_imr=int(enabled),
        isr=int(pending),
    )
    imr_addr = 0x100000 + IMEMRegisters.IMR
    isr_addr = 0x100000 + IMEMRegisters.ISR
    assert KEY_HANDLER in visited
    assert emu.memory.read_byte(isr_addr) == int(ISRFlag.MTI)
    assert emu.memory.read_byte(imr_addr) == int(IMRFlag.IRM | IMRFlag.MTM)
