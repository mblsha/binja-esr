from __future__ import annotations

import pytest

from binja_test_mocks.eval_llil import Memory

from sc62015.pysc62015.constants import ADDRESS_SPACE_SIZE, INTERNAL_MEMORY_START
from sc62015.pysc62015.instr.opcodes import IMEMRegisters
from pce500.memory import PCE500Memory as PyMemory
from pce500.emulator import PCE500Emulator as Emulator, IRQSource
from sc62015.pysc62015 import RegisterName


def _make_raw() -> bytearray:
    return bytearray(ADDRESS_SPACE_SIZE)


def _make_py_memory(raw: bytearray) -> PyMemory:
    mem = PyMemory()
    mem.external_memory = raw  # mimic injected backing store
    return mem


def _make_llama_memory(raw: bytearray):
    # Minimal shim to write/read internal memory via alias and direct
    def read(addr: int) -> int:
        return raw[addr]

    def write(addr: int, value: int) -> None:
        raw[addr] = value & 0xFF

    mem = Memory(read, write)
    setattr(mem, "_raw", raw)
    return mem


@pytest.mark.parametrize("alias_base", [ADDRESS_SPACE_SIZE - 0x100, INTERNAL_MEMORY_START])
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
    raw = _make_raw()
    mem = _make_py_memory(raw)
    emu = Emulator()
    emu.memory = mem
    emu.cpu.memory = mem
    emu.regs.set(RegisterName.PC, 0x0000)

    # Mock an ISR bit and IMR enabling KEY
    mem.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR, 0x04)  # KEYI
    mem.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.IMR, 0x84)  # IRM+KEY

    # Force IRQ delivery bookkeeping
    emu._irq_source = IRQSource.KEY
    emu._in_interrupt = True

    # Place RETI at PC
    mem.write_byte(0x0000, 0x01)  # RETI opcode
    emu.step()

    isr_after = mem.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR)
    assert isr_after & 0x04 == 0


def test_reti_clears_isr_bit_llama() -> None:
    # Mirror the Python check against the Rust backend via the LLAMA CPU wrapper
    raw = _make_raw()
    mem = _make_py_memory(raw)
    emu = Emulator(enable_new_tracing=False)
    emu.memory = mem
    emu.cpu = Emulator(cpu := None).cpu  # type: ignore[attr-defined]
    emu.cpu = CPU(mem, reset_on_init=False, backend="llama")
    emu.cpu.regs.set(RegisterName.PC, 0x0000)

    mem.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR, 0x04)  # KEYI
    mem.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.IMR, 0x84)  # IRM+KEY

    # In LLAMA backend, trigger RETI directly
    mem.write_byte(0x0000, 0x01)
    emu.step()

    isr_after = mem.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR)
    assert isr_after & 0x04 == 0
