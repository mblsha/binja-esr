"""Lightweight smoke tests for the LLAMA backend.

These keep coverage shallow (NOP and a couple of basic moves) while ensuring
the backend is wired through the Python facade and respects PC/flag/register
updates. Skip when the LLAMA backend is unavailable.
"""

from __future__ import annotations

from typing import Any, cast

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


def test_llama_keyboard_bridge_updates_kil() -> None:
    memory = cast(Any, _make_memory(0x00))
    cpu = CPU(memory, reset_on_init=True, backend="llama")

    # Press matrix code 0 (row 0, col 0), then release it to populate FIFO.
    assert cpu.keyboard_press_matrix_code(0x00)
    kil_addr = (0x100000 + IMEMRegisters.KIL) & 0xFFFFFF
    assert memory._raw[kil_addr] == 0x01

    assert cpu.keyboard_release_matrix_code(0x00)
    assert memory._raw[kil_addr] == 0x00


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


class _MemoryRecordingPc:
    def __init__(self) -> None:
        self._raw = bytearray(ADDRESS_SPACE_SIZE)
        self.calls: list[tuple[str, int, int | None]] = []

    def read_byte(self, addr: int, cpu_pc: int | None = None) -> int:
        self.calls.append(("read", addr & 0xFFFFFF, cpu_pc))
        return self._raw[addr & 0xFFFFFF]

    def write_byte(self, addr: int, value: int, cpu_pc: int | None = None) -> None:
        self.calls.append(("write", addr & 0xFFFFFF, cpu_pc))
        self._raw[addr & 0xFFFFFF] = value & 0xFF


def test_llama_passes_pc_to_memory_reads_and_writes() -> None:
    mem = _MemoryRecordingPc()
    # MV [0x00020],A
    mem._raw[0] = 0xA8
    mem._raw[1] = 0x20
    mem._raw[2] = 0x00
    mem._raw[3] = 0x00
    cpu = CPU(mem, reset_on_init=False, backend="llama")
    cpu.regs.set(RegisterName.PC, 0x0000)
    cpu.regs.set(RegisterName.A, 0xCD)

    cpu.execute_instruction(0x0000)

    assert ("write", 0x20, 0) in mem.calls
    pc_values = {pc for _, _, pc in mem.calls if pc is not None}
    assert pc_values == {0}


class _MemoryWithKioTrace:
    def __init__(self) -> None:
        self._raw = bytearray(ADDRESS_SPACE_SIZE)
        self.kio_traces: list[tuple[int, int, int | None]] = []
        self.irq_traces: list[tuple[str, dict[str, int | None]]] = []

    def trace_kio_from_rust(
        self, offset: int, value: int, pc: int | None = None
    ) -> None:
        self.kio_traces.append((offset & 0xFF, value & 0xFF, pc))

    def trace_irq_from_rust(self, name: str, payload: dict[str, int | None]) -> None:
        self.irq_traces.append((name, dict(payload)))

    def read_byte(self, addr: int, cpu_pc: int | None = None) -> int:
        return self._raw[addr & 0xFFFFFF]

    def write_byte(self, addr: int, value: int, cpu_pc: int | None = None) -> None:
        self._raw[addr & 0xFFFFFF] = value & 0xFF


def test_llama_kio_write_traces_into_python_bus() -> None:
    mem = _MemoryWithKioTrace()
    # MV (KOL),A with immediate offset 0xF0
    mem._raw[0] = 0xA0  # MV (n),A
    mem._raw[1] = 0xF0  # KOL offset
    cpu = CPU(mem, reset_on_init=False, backend="llama")
    cpu.regs.set(RegisterName.PC, 0x0000)
    cpu.regs.set(RegisterName.A, 0x12)

    cpu.execute_instruction(0x0000)

    assert (0xF0, 0x12, 0) in mem.kio_traces
    assert mem._raw[(0x100000 + 0xF0) & 0xFFFFFF] == 0x12


def test_llama_imr_write_traces_irq_payload() -> None:
    mem = _MemoryWithKioTrace()
    # MV (IMR),#0xAA using IMem8 immediate
    mem._raw[0] = 0xCC  # MV IMem8,imm8
    mem._raw[1] = 0xFB  # IMR offset
    mem._raw[2] = 0xAA
    cpu = CPU(mem, reset_on_init=False, backend="llama")
    cpu.regs.set(RegisterName.PC, 0x0000)

    cpu.execute_instruction(0x0000)

    assert any(name == "IMR_Write" for name, _ in mem.irq_traces)
    assert mem._raw[(0x100000 + 0xFB) & 0xFFFFFF] == 0xAA


def test_llama_isr_write_traces_irq_payload() -> None:
    mem = _MemoryWithKioTrace()
    # MV (ISR),#0x55 using IMem8 immediate
    mem._raw[0] = 0xCC  # MV IMem8,imm8
    mem._raw[1] = 0xFC  # ISR offset
    mem._raw[2] = 0x55
    cpu = CPU(mem, reset_on_init=False, backend="llama")
    cpu.regs.set(RegisterName.PC, 0x0000)

    cpu.execute_instruction(0x0000)

    assert any(name == "ISR_Write" for name, _ in mem.irq_traces)
    assert mem._raw[(0x100000 + 0xFC) & 0xFFFFFF] == 0x55


def test_llama_ir_traces_irq_enter() -> None:
    mem = _MemoryWithKioTrace()
    # IR opcode at 0x00, vector at 0xFFFFA = 0x12345
    mem._raw[0] = 0xFE
    mem._raw[0xFFFFA] = 0x45
    mem._raw[0xFFFFB] = 0x23
    mem._raw[0xFFFFC] = 0x01
    cpu = CPU(mem, reset_on_init=False, backend="llama")
    cpu.regs.set(RegisterName.PC, 0x0000)
    cpu.regs.set(RegisterName.S, 0x0020)  # scratch stack space
    cpu.regs.set(RegisterName.F, 0x10)

    cpu.execute_instruction(0x0000)

    assert any(name == "IRQ_Enter" for name, _ in mem.irq_traces)
    entry = next(payload for name, payload in mem.irq_traces if name == "IRQ_Enter")
    assert entry["pc"] == 0
    assert entry["vector"] == 0x12345


def test_llama_reti_traces_irq_return() -> None:
    mem = _MemoryWithKioTrace()
    # RETI opcode with stack frame IMR,F,PC bytes
    mem._raw[0] = 0x01
    # stack frame at 0x30
    mem._raw[0x30] = 0xAA  # IMR
    mem._raw[0x31] = 0xBB  # F
    mem._raw[0x32] = 0x11
    mem._raw[0x33] = 0x22
    mem._raw[0x34] = 0x33
    cpu = CPU(mem, reset_on_init=False, backend="llama")
    cpu.regs.set(RegisterName.PC, 0x0000)
    cpu.regs.set(RegisterName.S, 0x0030)

    cpu.execute_instruction(0x0000)

    assert any(name in ("IRQ_Return", "IRQ_Exit") for name, _ in mem.irq_traces)
    exit_payload = next(
        payload
        for name, payload in mem.irq_traces
        if name in ("IRQ_Return", "IRQ_Exit")
    )
    assert exit_payload["pc"] == 0
    assert exit_payload["ret"] == 0x032211
