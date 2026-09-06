"""Changing callbacks expose pointer/IMR re-evaluation, not silicon micro-order."""

from collections import Counter

from binja_test_mocks.eval_llil import Memory
import pytest

from sc62015.pysc62015 import CPU, RegisterName
from sc62015.pysc62015.constants import ADDRESS_SPACE_SIZE, INTERNAL_MEMORY_START


@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize("initial", [0x00, 0x21, 0x80, 0xA1])
def test_pushu_imr_clears_irm_from_the_saved_sample(backend, initial):
    # A1 = IRM | RXRM | MTM. The second callback value is deliberately different.
    raw = bytearray(ADDRESS_SPACE_SIZE)
    raw[0x1000] = 0x2F
    imr = INTERNAL_MEMORY_START + 0xFB
    reads = []

    def read(address):
        if address == imr:
            value = initial if not reads else initial ^ 0x7F
            reads.append(value)
            return value
        return raw[address]

    memory = Memory(read, raw.__setitem__)
    setattr(memory, "peek_byte_for_preflight", lambda address, _pc=None: raw[address])
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.PC, 0x1000)
    cpu.regs.set(RegisterName.U, 0x80010)
    cpu.regs.set(RegisterName.F, 3)
    cpu.execute_instruction(0x1000)
    assert raw[0x8000F] == initial
    assert raw[imr] == initial & 0x7F
    assert reads == [initial]
    assert cpu.regs.get(RegisterName.U) == 0x8000F
    assert cpu.regs.get(RegisterName.F) == 3


@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize("selector", [0x20, 0xFF])
@pytest.mark.parametrize(
    "pointer,mode,displacement", [(0x80000, 0, 0), (0xFFFFF, 0x80, 1), (0, 0xC0, 1)]
)
@pytest.mark.parametrize(
    "target,opcode,width",
    [
        ("A", 0x98, 1),
        ("BA", 0x9A, 2),
        ("X", 0x9C, 3),
        ("mem", 0xF2, 3),
        ("block", 0xF3, 3),
    ],
)
def test_indirect_pointer_is_sampled_once_before_ordered_data_reads(
    backend, selector, pointer, mode, displacement, target, opcode, width
):
    raw = bytearray(ADDRESS_SPACE_SIZE)
    program = bytes([0x32 if target in ("mem", "block") else 0x30, opcode, mode])
    if target in ("mem", "block"):
        program += bytes([0x40, selector])
    else:
        program += bytes([selector])
    if mode:
        program += bytes([displacement])
    raw[0x1000 : 0x1000 + len(program)] = program
    pointer_addresses = [(selector + offset) & 255 for offset in range(3)]
    original_bytes = list(pointer.to_bytes(3, "little"))
    pointer_reads = Counter()
    events = []
    data_base = (
        pointer
        + (displacement if mode == 0x80 else -displacement if mode == 0xC0 else 0)
    ) & 0xFFFFF
    data_addresses = [(data_base + offset) & 0xFFFFF for offset in range(width)]
    data = [0xA5, 0x5A, 0xCF][:width]
    for address, value in zip(data_addresses, data):
        raw[address] = value

    def read(address):
        offset = address - INTERNAL_MEMORY_START
        if offset in pointer_addresses:
            index = pointer_addresses.index(offset)
            value = (
                original_bytes[index]
                if not pointer_reads[offset]
                else original_bytes[index] ^ 0x11
            )
            pointer_reads[offset] += 1
            events.append(("pointer", address, value))
            return value
        if address < 0x100000 and not 0x1000 <= address < 0x1000 + len(program):
            events.append(("data", address, raw[address]))
        return raw[address]

    memory = Memory(read, raw.__setitem__)
    setattr(memory, "peek_byte_for_preflight", lambda address, _pc=None: raw[address])
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.PC, 0x1000)
    cpu.regs.set(RegisterName.I, width)
    cpu.regs.set(RegisterName.F, 3)
    cpu.execute_instruction(0x1000)
    if target in ("mem", "block"):
        assert (
            list(
                raw[INTERNAL_MEMORY_START + 0x40 : INTERNAL_MEMORY_START + 0x40 + width]
            )
            == data
        )
    else:
        expected = int.from_bytes(bytes(data), "little") & (
            0xFFFFF if width == 3 else (1 << (8 * width)) - 1
        )
        assert cpu.regs.get(getattr(RegisterName, target)) == expected
    assert events == [
        ("pointer", INTERNAL_MEMORY_START + address, value)
        for address, value in zip(pointer_addresses, original_bytes)
    ] + [("data", address, value) for address, value in zip(data_addresses, data)]
    assert cpu.regs.get(RegisterName.F) == 3
    assert cpu.regs.get(RegisterName.I) == (0 if target == "block" else width)
