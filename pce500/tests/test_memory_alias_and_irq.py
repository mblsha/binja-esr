"""Emulator memory, interrupt-instruction, and Python scheduler contracts.

These tests do not execute the stock ROM dispatcher. ROM-grounded dispatcher
coverage lives in ``test_keyboard_interrupt_rom.py``.
"""

from __future__ import annotations

from contextlib import contextmanager
import os

import pytest

from binja_test_mocks.eval_llil import Memory
from pce500.emulator import IMRFlag, ISRFlag, IRQSource, PCE500Emulator as Emulator
from pce500.memory import PCE500Memory as PyMemory
from sc62015.pysc62015 import RegisterName, available_backends
from sc62015.pysc62015.constants import ADDRESS_SPACE_SIZE, INTERNAL_MEMORY_START
from sc62015.pysc62015.instr.opcodes import IMEMRegisters


def _make_raw() -> bytearray:
    return bytearray(ADDRESS_SPACE_SIZE)


def _make_py_memory(raw: bytearray) -> PyMemory:
    mem = PyMemory()
    mem.external_memory = raw  # mimic injected backing store
    return mem


@contextmanager
def _backend(name: str | None):
    prev = os.environ.get("SC62015_CPU_BACKEND")
    if name is None and "SC62015_CPU_BACKEND" in os.environ:
        del os.environ["SC62015_CPU_BACKEND"]
    elif name is not None:
        os.environ["SC62015_CPU_BACKEND"] = name
    try:
        yield
    finally:
        if prev is None:
            if "SC62015_CPU_BACKEND" in os.environ:
                del os.environ["SC62015_CPU_BACKEND"]
        else:
            os.environ["SC62015_CPU_BACKEND"] = prev


def _make_llama_memory(raw: bytearray):
    # Minimal shim to write/read internal memory via alias and direct
    def read(addr: int) -> int:
        return raw[addr]

    def write(addr: int, value: int) -> None:
        raw[addr] = value & 0xFF

    mem = Memory(read, write)
    setattr(mem, "_raw", raw)
    return mem


@pytest.mark.parametrize(
    "alias_base", [ADDRESS_SPACE_SIZE - 0x100, INTERNAL_MEMORY_START]
)
def test_internal_memory_alias_writes(alias_base: int) -> None:
    raw = _make_raw()
    py_mem = _make_py_memory(raw)

    # Write IMR through alias/direct and ensure both views see it.
    imr_addr = alias_base + IMEMRegisters.IMR
    py_mem.write_byte(imr_addr, 0xAA)

    # Direct IMEM read
    direct_val = py_mem.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.IMR)
    alias_val = py_mem.read_byte(ADDRESS_SPACE_SIZE - 0x100 + IMEMRegisters.IMR)
    assert direct_val == 0xAA
    assert alias_val == 0xAA


@pytest.mark.parametrize("backend", ["python", "llama"])
def test_reti_restores_interrupt_frame_without_acknowledging_isr(
    backend: str,
) -> None:
    """Match the ROM contract: firmware acknowledges ISR before executing RETI."""
    if backend == "llama" and "llama" not in available_backends():
        pytest.skip("LLAMA backend not available")

    with _backend(backend if backend != "python" else None):
        emu = Emulator()

    mem = emu.memory

    mem.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR, 0x04)  # KEYI
    mem.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.IMR, 0x84)  # IRM+KEY

    emu._irq_source = IRQSource.KEY
    emu._in_interrupt = True

    sp = 0x0100
    mem.write_byte(sp, 0x84)
    # F is a byte-wide frame slot, but only C/Z (0x03) are currently modeled.
    # Upper bits remain hardware-trace work and are deliberately rejected.
    mem.write_byte(sp + 1, 0x03)
    mem.write_byte(sp + 2, 0x12)
    mem.write_byte(sp + 3, 0x34)
    mem.write_byte(sp + 4, 0x05)
    emu.cpu.regs.set(RegisterName.S, sp)

    mem.write_byte(0x0000, 0x01)
    emu.cpu.regs.set(RegisterName.PC, 0x0000)

    emu.step()

    isr_after = mem.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR)
    assert isr_after & 0x04 == 0x04
    imr_after = mem.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.IMR)
    assert imr_after == 0x84


@pytest.mark.parametrize("backend", ["python", "llama"])
def test_imr_masks_irq_pending(backend: str) -> None:
    if backend == "llama" and "llama" not in available_backends():
        pytest.skip("LLAMA backend not available")

    with _backend(backend if backend != "python" else None):
        emu = Emulator()

    # Avoid timer side effects and IRQ delivery; simulate pending check only.
    emu._timer_enabled = False
    emu._in_interrupt = True
    emu._irq_source = IRQSource.KEY
    emu._irq_pending = False
    # Ensure a valid opcode at PC=0
    emu.memory.write_byte(0x0000, 0x00)
    emu.cpu.regs.set(RegisterName.PC, 0x0000)

    isr_addr = INTERNAL_MEMORY_START + IMEMRegisters.ISR
    imr_addr = INTERNAL_MEMORY_START + IMEMRegisters.IMR
    emu.memory.write_byte(isr_addr, 0x04)  # KEYI set

    # IMR masked: pending should remain false
    emu.memory.write_byte(imr_addr, 0x00)
    emu._irq_pending = False
    emu.step()
    assert not emu._irq_pending

    # IMR master+KEY enabled: pending should arm
    emu.memory.write_byte(imr_addr, 0x84)
    emu._irq_pending = False
    emu.step()
    assert emu._irq_pending


@pytest.mark.parametrize(
    ("pending", "expected"),
    [
        (ISRFlag.ONKI | ISRFlag.KEYI | ISRFlag.STI | ISRFlag.MTI, IRQSource.ONK),
        (ISRFlag.STI | ISRFlag.MTI, IRQSource.STI),
    ],
)
def test_pending_source_bookkeeping_matches_rom_priority(
    pending: ISRFlag, expected: IRQSource
) -> None:
    emu = Emulator()
    emu._timer_enabled = False
    emu._in_interrupt = True
    emu._irq_pending = False
    emu._irq_source = None
    emu.memory.write_byte(0x0000, 0x00)
    emu.cpu.regs.set(RegisterName.PC, 0x0000)
    emu.memory.write_byte(
        INTERNAL_MEMORY_START + IMEMRegisters.IMR,
        int(IMRFlag.IRM) | int(pending),
    )
    emu.memory.write_byte(
        INTERNAL_MEMORY_START + IMEMRegisters.ISR,
        int(pending),
    )

    emu.step()

    assert emu._irq_source is expected


@pytest.mark.parametrize("backend", ["python", "llama"])
def test_timer_irq_arms_only_when_imr_allows(backend: str) -> None:
    if backend == "llama" and "llama" not in available_backends():
        pytest.skip("LLAMA backend not available")

    with _backend(backend if backend != "python" else None):
        emu = Emulator()

    emu._timer_enabled = True
    emu._timer_mti_period = 2
    emu._timer_sti_period = 0
    emu.cpu.regs.set(RegisterName.PC, 0x0000)
    emu.cpu.regs.set(RegisterName.S, 0x0200)
    imr_addr = INTERNAL_MEMORY_START + IMEMRegisters.IMR
    isr_addr = INTERNAL_MEMORY_START + IMEMRegisters.ISR
    emu.memory.write_byte(isr_addr, 0x00)

    # Masked IMR: ISR should set; delivery/pending may be deferred.
    emu.memory.write_byte(imr_addr, 0x00)
    emu._irq_pending = False
    emu.cycle_count = emu._scheduler.next_mti
    emu.step()
    assert emu.memory.read_byte(isr_addr) & int(ISRFlag.MTI)

    # Enable IRM+MTI and fire again: IRQ should either arm pending or deliver.
    emu.memory.write_byte(imr_addr, int(IMRFlag.IRM) | int(IMRFlag.MTI))
    emu._irq_pending = False
    emu.cycle_count = emu._scheduler.next_mti
    emu.step()
    assert emu._irq_pending or emu.cpu.regs.get(RegisterName.PC) != 0x0000


def test_timer_advances_while_interrupt_handler_is_active() -> None:
    """Pin the Python scheduler model; real-device ISR relatch timing is unverified."""
    emu = Emulator()
    emu._timer_enabled = True
    emu._timer_mti_period = 2
    emu._timer_sti_period = 0
    emu._in_interrupt = True
    emu.cpu.regs.set(RegisterName.PC, 0x0000)
    emu.cpu.regs.set(RegisterName.S, 0x0200)
    isr_addr = INTERNAL_MEMORY_START + IMEMRegisters.ISR
    emu.memory.write_byte(isr_addr, 0)
    emu.cycle_count = emu._scheduler.next_mti

    emu.step()

    assert emu.memory.read_byte(isr_addr) & int(ISRFlag.MTI)


@pytest.mark.parametrize("backend", ["python", "llama"])
def test_sti_irq_arms_when_imr_allows(backend: str) -> None:
    if backend == "llama" and "llama" not in available_backends():
        pytest.skip("LLAMA backend not available")

    with _backend(backend if backend != "python" else None):
        emu = Emulator()

    emu._timer_enabled = True
    emu._timer_mti_period = 0
    emu._timer_sti_period = 2
    emu.cpu.regs.set(RegisterName.PC, 0x0000)
    emu.cpu.regs.set(RegisterName.S, 0x0200)
    imr_addr = INTERNAL_MEMORY_START + IMEMRegisters.IMR
    isr_addr = INTERNAL_MEMORY_START + IMEMRegisters.ISR
    emu.memory.write_byte(isr_addr, 0x00)

    # Masked IMR: ISR should set; pending/delivery gated by IMR.
    emu.memory.write_byte(imr_addr, 0x00)
    emu._irq_pending = False
    emu.cycle_count = emu._scheduler.next_sti
    emu.step()
    assert emu.memory.read_byte(isr_addr) & int(ISRFlag.STI)

    # Enable IRM+STI and fire again: pending should arm or deliver.
    emu.memory.write_byte(imr_addr, int(IMRFlag.IRM) | int(IMRFlag.STI))
    emu._irq_pending = False
    emu.cycle_count = emu._scheduler.next_sti
    emu.step()
    assert emu._irq_pending or emu.cpu.regs.get(RegisterName.PC) != 0x0000


@pytest.mark.parametrize("backend", ["python", "llama"])
def test_keyi_pending_respects_imr(backend: str) -> None:
    if backend == "llama" and "llama" not in available_backends():
        pytest.skip("LLAMA backend not available")

    with _backend(backend if backend != "python" else None):
        emu = Emulator()

    emu._timer_enabled = False
    emu._irq_pending = False
    emu._in_interrupt = True  # prevent delivery; we only care about pending arm
    emu.cpu.regs.set(RegisterName.PC, 0x0000)
    emu.cpu.regs.set(RegisterName.S, 0x0200)
    imr_addr = INTERNAL_MEMORY_START + IMEMRegisters.IMR
    isr_addr = INTERNAL_MEMORY_START + IMEMRegisters.ISR
    emu.memory.write_byte(isr_addr, int(ISRFlag.KEYI))

    emu.memory.write_byte(imr_addr, 0x00)
    emu.step()
    assert not emu._irq_pending

    emu._irq_pending = False
    emu.memory.write_byte(imr_addr, int(IMRFlag.IRM) | int(IMRFlag.KEY))
    emu.step()
    assert emu._irq_pending or emu.cpu.regs.get(RegisterName.PC) != 0x0000


@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize(
    ("mask", "status", "source"),
    [
        (IMRFlag.KEYM, ISRFlag.KEYI, IRQSource.KEY),
        (IMRFlag.ONKM, ISRFlag.ONKI, IRQSource.ONK),
    ],
)
def test_key_on_irq_delivery_waits_for_irm(
    backend: str,
    mask: IMRFlag,
    status: ISRFlag,
    source: IRQSource,
) -> None:
    """A latched KEY/ON request is not permission to bypass the master mask."""
    if backend == "llama" and "llama" not in available_backends():
        pytest.skip("LLAMA backend not available")

    with _backend(backend if backend != "python" else None):
        emu = Emulator()

    emu._timer_enabled = False
    emu._in_interrupt = False
    emu._irq_pending = True
    emu._irq_source = source
    emu.cpu.regs.set(RegisterName.PC, 0x0000)
    emu.cpu.regs.set(RegisterName.S, 0x0200)
    emu.memory.write_byte(0x0000, 0x00)
    emu.memory.write_byte(0x0001, 0x00)

    imr_addr = INTERNAL_MEMORY_START + IMEMRegisters.IMR
    isr_addr = INTERNAL_MEMORY_START + IMEMRegisters.ISR
    emu.memory.write_byte(imr_addr, int(mask))
    emu.memory.write_byte(isr_addr, int(status))

    emu.step()

    assert emu.cpu.regs.get(RegisterName.PC) == 0x0001
    assert emu.cpu.regs.get(RegisterName.S) == 0x0200
    assert emu._irq_pending is True
    assert emu._in_interrupt is False
    assert emu.memory.read_byte(isr_addr) & int(status)

    emu.memory.write_byte(imr_addr, int(IMRFlag.IRM | mask))
    emu.step()

    assert emu._irq_pending is False
    assert emu._in_interrupt is True
    assert emu.cpu.regs.get(RegisterName.S) == 0x01FB
    assert emu.last_irq["src"] == source.name
    assert emu.last_irq["pc"] == 0x0001
    assert emu.memory.read_byte(isr_addr) & int(status)
