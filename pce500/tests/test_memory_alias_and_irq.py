from __future__ import annotations

from contextlib import contextmanager
import os

import pytest

from binja_test_mocks.eval_llil import Memory
from pce500.emulator import PCE500Emulator as Emulator, IRQSource
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
