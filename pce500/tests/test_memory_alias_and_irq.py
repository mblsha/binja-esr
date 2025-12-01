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


def test_reti_clears_isr_bit_python() -> None:
    # Use Python emulator as reference: set ISR key bit, deliver IRQ, ensure RETI clears it
    with _backend("python"):
        emu = Emulator()

    mem = emu.memory

    # Mock an ISR bit and IMR enabling KEY
    mem.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR, 0x04)  # KEYI
    mem.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.IMR, 0x84)  # IRM+KEY

    # Force IRQ delivery bookkeeping
    emu._irq_source = IRQSource.KEY
    emu._in_interrupt = True

    # Prepare stack for RETI: IMR, F, PC bytes
    sp = 0x0100
    mem.write_byte(sp, 0x84)
    mem.write_byte(sp + 1, 0x7C)
    mem.write_byte(sp + 2, 0x12)
    mem.write_byte(sp + 3, 0x34)
    mem.write_byte(sp + 4, 0x05)
    emu.cpu.regs.set(RegisterName.S, sp)

    mem.write_byte(0x0000, 0x01)  # RETI opcode
    emu.cpu.regs.set(RegisterName.PC, 0x0000)

    emu.step()

    isr_after = mem.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR)
    assert isr_after & 0x04 == 0
    imr_after = mem.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.IMR)
    assert imr_after == 0x84


@pytest.mark.parametrize("backend", ["python", "llama"])
def test_kil_respects_ksd(backend: str) -> None:
    if backend == "llama" and "llama" not in available_backends():
        pytest.skip("LLAMA backend not available")

    with _backend(backend if backend != "python" else None):
        emu = Emulator()

    mem = emu.memory

    # Strobe a column and press a key by setting latch directly
    mem.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.KOL, 0x01)
    mem.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.KOH, 0x00)
    mem.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.LCC, 0x04)  # KSD bit set

    # KIL should be masked to 0 when KSD is set
    kil_val = emu.memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.KIL)
    assert kil_val == 0x00


def test_reti_clears_isr_bit_llama() -> None:
    if "llama" not in available_backends():
        pytest.skip("LLAMA backend not available")

    with _backend("llama"):
        emu = Emulator()

    mem = emu.memory

    mem.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR, 0x04)  # KEYI
    mem.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.IMR, 0x84)  # IRM+KEY

    emu._irq_source = IRQSource.KEY
    emu._in_interrupt = True

    sp = 0x0100
    mem.write_byte(sp, 0x84)
    mem.write_byte(sp + 1, 0x7C)
    mem.write_byte(sp + 2, 0x12)
    mem.write_byte(sp + 3, 0x34)
    mem.write_byte(sp + 4, 0x05)
    emu.cpu.regs.set(RegisterName.S, sp)

    mem.write_byte(0x0000, 0x01)
    emu.cpu.regs.set(RegisterName.PC, 0x0000)

    emu.step()

    isr_after = mem.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR)
    assert isr_after & 0x04 == 0
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
    assert emu.memory.read_byte(isr_addr) & int(ISRFlag.MTI)

    # Enable IRM+MTI and fire again: IRQ should either arm pending or deliver.
    emu.memory.write_byte(imr_addr, int(IMRFlag.IRM) | int(IMRFlag.MTI))
    emu._irq_pending = False
    emu.cycle_count = emu._scheduler.next_mti
    emu.step()
    assert emu._irq_pending or emu.cpu.regs.get(RegisterName.PC) != 0x0000


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
