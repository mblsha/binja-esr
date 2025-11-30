"""Lightweight smoke tests for the LLAMA backend.

These keep coverage shallow (NOP and a couple of basic moves) while ensuring
the backend is wired through the Python facade and respects PC/flag/register
updates. Skip when the LLAMA backend is unavailable.
"""

from __future__ import annotations

import pytest

from sc62015.pysc62015 import CPU, RegisterName, available_backends
from sc62015.pysc62015.constants import ADDRESS_SPACE_SIZE
from sc62015.pysc62015.instr.opcodes import IMEMRegisters
from binja_test_mocks.eval_llil import Memory


def _make_memory(*bytes_seq: int) -> Memory:
    """Create a simple byte-addressable memory with the given initial bytes."""
    if not bytes_seq:
        raise ValueError("memory initialiser requires at least one byte")
    raw = bytearray(ADDRESS_SPACE_SIZE)
    for idx, value in enumerate(bytes_seq):
        raw[idx] = value & 0xFF

    def read(addr: int) -> int:
        if addr < 0 or addr >= len(raw):
            raise IndexError(f"Read address {addr:#x} out of bounds")
        return raw[addr]

    def write(addr: int, value: int) -> None:
        if addr < 0 or addr >= len(raw):
            raise IndexError(f"Write address {addr:#x} out of bounds")
        raw[addr] = value & 0xFF

    memory = Memory(read, write)
    setattr(memory, "_raw", raw)
    return memory


pytestmark = pytest.mark.skipif(
    "llama" not in available_backends(),
    reason="LLAMA backend not available in this runtime",
)


def test_llama_executes_nop_and_advances_pc() -> None:
    memory = _make_memory(0x00)  # NOP
    cpu = CPU(memory, reset_on_init=False, backend="llama")
    cpu.regs.set(RegisterName.PC, 0x0000)

    info = cpu.execute_instruction(0x0000)

    assert info.instruction.name() == "NOP"
    assert cpu.regs.get(RegisterName.PC) == 0x0001


def test_llama_mv_a_imm() -> None:
    memory = _make_memory(0x08, 0x5A)  # MV A,n
    cpu = CPU(memory, reset_on_init=False, backend="llama")
    cpu.regs.set(RegisterName.PC, 0x0000)
    cpu.regs.set(RegisterName.A, 0x00)

    cpu.execute_instruction(0x0000)

    assert cpu.regs.get(RegisterName.A) == 0x5A
    assert cpu.regs.get(RegisterName.PC) == 0x0002


def test_llama_mv_a_to_ext_mem() -> None:
    # MV [00020], A
    memory = _make_memory(0xA8, 0x20, 0x00, 0x00)
    cpu = CPU(memory, reset_on_init=False, backend="llama")
    cpu.regs.set(RegisterName.PC, 0x0000)
    cpu.regs.set(RegisterName.A, 0xCD)

    cpu.execute_instruction(0x0000)

    assert memory.read_byte(0x20) == 0xCD
    assert cpu.regs.get(RegisterName.PC) == 0x0004


def test_llama_add_sets_flags() -> None:
    # ADD A,#0xFF (opcode 0x40)
    memory = _make_memory(0x40, 0xFF)
    cpu = CPU(memory, reset_on_init=False, backend="llama")
    cpu.regs.set(RegisterName.PC, 0x0000)
    cpu.regs.set(RegisterName.A, 0x01)
    cpu.regs.set(RegisterName.FC, 0)
    cpu.regs.set(RegisterName.FZ, 0)

    cpu.execute_instruction(0x0000)

    assert cpu.regs.get(RegisterName.A) == 0x00
    assert cpu.regs.get(RegisterName.FC) == 1
    assert cpu.regs.get(RegisterName.FZ) == 1
    assert cpu.regs.get(RegisterName.PC) == 0x0002


def test_llama_wait_clears_i_and_flags() -> None:
    # WAIT (0xEF) should clear I/FC/FZ and advance PC by 1
    memory = _make_memory(0xEF)
    cpu = CPU(memory, reset_on_init=False, backend="llama")
    cpu.regs.set(RegisterName.PC, 0x0000)
    cpu.regs.set(RegisterName.I, 0x1234)
    cpu.regs.set(RegisterName.FC, 1)
    cpu.regs.set(RegisterName.FZ, 1)

    cpu.execute_instruction(0x0000)

    assert cpu.regs.get(RegisterName.I) == 0
    assert cpu.regs.get(RegisterName.FC) == 0
    assert cpu.regs.get(RegisterName.FZ) == 0
    assert cpu.regs.get(RegisterName.PC) == 0x0001


@pytest.mark.parametrize("backend", ("python", "llama"))
def test_f_register_preserves_upper_bits(backend: str) -> None:
    memory = _make_memory(0x00)  # no opcode needed
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.F, 0xAA)

    assert cpu.regs.get(RegisterName.F) == 0xAA
    assert cpu.regs.get(RegisterName.FC) == 0x00
    assert cpu.regs.get(RegisterName.FZ) == 0x01


@pytest.mark.parametrize("backend", ("python", "llama"))
def test_fc_fz_updates_do_not_clobber_f_upper_bits(backend: str) -> None:
    memory = _make_memory(0x00)  # no opcode needed
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.F, 0xAA)  # 0b1010_1010

    cpu.regs.set(RegisterName.FC, 0)  # lower bit -> 0
    cpu.regs.set(RegisterName.FZ, 1)  # bit1 -> 1

    assert cpu.regs.get(RegisterName.F) == 0xAA  # upper bits unchanged


def test_llama_keyboard_bridge_updates_fifo_and_kil() -> None:
    memory = _make_memory(0x00)
    cpu = CPU(memory, reset_on_init=True, backend="llama")

    # Press matrix code 0 (row 0, col 0), then release it to populate FIFO.
    assert cpu.keyboard_press_matrix_code(0x00)
    kil_addr = (0x100000 + IMEMRegisters.KIL) & 0xFFFFFF
    assert memory._raw[kil_addr] == 0x01
    fifo_base = 0x00BFC96
    assert memory._raw[fifo_base] == 0x00

    assert cpu.keyboard_release_matrix_code(0x00)
    assert memory._raw[kil_addr] == 0x00
    assert memory._raw[fifo_base + 1] == 0x80


class _MemoryWithLcdHook:
    def __init__(self) -> None:
        self._raw = bytearray(ADDRESS_SPACE_SIZE)
        self.calls: list[tuple[int, int, int]] = []
        self._llama_lcd_write = self._hook

    def _hook(self, addr: int, val: int, pc: int) -> None:
        self.calls.append((addr & 0xFFFFFF, val & 0xFF, pc & 0xFFFFFF))

    def read_byte(self, addr: int) -> int:
        return self._raw[addr & 0xFFFFFF]

    def write_byte(self, addr: int, value: int) -> None:
        self._raw[addr & 0xFFFFFF] = value & 0xFF


def test_llama_lcd_hook_invoked_on_write() -> None:
    mem = _MemoryWithLcdHook()
    # MV [0x2000],A
    mem.write_byte(0, 0xA8)
    mem.write_byte(1, 0x00)
    mem.write_byte(2, 0x20)
    mem.write_byte(3, 0x00)
    cpu = CPU(mem, reset_on_init=False, backend="llama")
    cpu.regs.set(RegisterName.PC, 0x0000)
    cpu.regs.set(RegisterName.A, 0xCD)

    cpu.execute_instruction(0x0000)

    assert mem.calls == [(0x2000, 0xCD, 0x0000)]
