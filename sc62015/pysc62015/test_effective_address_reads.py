"""Changing address-base callbacks are software witnesses, not silicon traces."""

from collections import Counter

from binja_test_mocks.eval_llil import Memory
import pytest

from sc62015.pysc62015 import CPU, RegisterName
from sc62015.pysc62015.constants import ADDRESS_SPACE_SIZE, INTERNAL_MEMORY_START


def run(
    backend,
    program,
    contents,
    bases,
    *,
    ba=0,
    flags=3,
    stack=0x80000,
    irq_events=None,
    read_budget=None,
):
    raw = bytearray(ADDRESS_SPACE_SIZE)
    raw[0x1000 : 0x1000 + len(program)] = program
    for offset, value in contents.items():
        raw[INTERNAL_MEMORY_START + offset] = value
    for offset, values in bases.items():
        raw[INTERNAL_MEMORY_START + offset] = values[0]
    raw[stack : stack + 5] = bytes([0xA1, 0xFC, 0, 0x20, 0])
    raw[0xFFFFA:0xFFFFD] = bytes([0, 0x20, 0])
    counts, reads, writes = Counter(), [], []

    def read(address):
        offset = address - INTERNAL_MEMORY_START
        if read_budget is not None and offset in read_budget:
            assert counts[offset] < read_budget[offset], f"extra SFR read {offset:02X}"
        if offset in bases:
            values = bases[offset]
            value = values[min(counts[offset], len(values) - 1)]
        else:
            value = raw[address]
        if 0 <= offset < 256:
            counts[offset] += 1
            reads.append((offset, value))
        return value

    def write(address, value):
        raw[address] = value
        writes.append((address, value))

    memory = Memory(read, write)
    setattr(memory, "peek_byte_for_preflight", lambda address, _pc=None: raw[address])
    setattr(memory, "instruction_byte_is_callback_free", lambda _address: True)
    if irq_events is not None:
        setattr(
            memory,
            "trace_irq_from_rust",
            lambda name, payload: irq_events.append((name, payload)),
        )
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.PC, 0x1000)
    cpu.regs.set(RegisterName.BA, ba)
    cpu.regs.set(RegisterName.F, flags)
    cpu.regs.set(RegisterName.S, stack)
    if program == b"\xfe":
        cpu.prepare_instruction_before_scheduling(0x1000)
    cpu.execute_instruction(0x1000)
    return cpu, raw, reads, writes


@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize(
    "opcode,tail,result,flags",
    [
        (0x41, b"\x01", 0x82, 0),
        (0x49, b"\x01", 0x80, 0),
        (0x51, b"\x01", 0x83, 0),
        (0x59, b"\x01", 0x7F, 0),
        (0x47, b"\x01", 0x82, 3),
        (0x69, b"\x81", 0, 3),
        (0x71, b"\x01", 1, 1),
        (0x79, b"\x02", 0x83, 1),
        (0x6D, b"", 0x82, 1),
        (0x7D, b"", 0x80, 1),
        (0xE5, b"", 0xC0, 1),
        (0xE7, b"", 0x03, 1),
        # SHR/SHL shift through incoming C=1, not a zero-filled logical shift.
        (0xF5, b"", 0xC0, 1),
        (0xF7, b"", 0x03, 1),
    ],
)
def test_rmw_keeps_the_original_effective_address(backend, opcode, tail, result, flags):
    cpu, raw, reads, writes = run(
        backend,
        bytes([opcode, 0]) + tail,
        {0x20: 0x81, 0x30: 0x19},
        {0xEC: [0x20, 0x30]},
    )
    assert raw[INTERNAL_MEMORY_START + 0x20] == result
    assert raw[INTERNAL_MEMORY_START + 0x30] == 0x19
    assert cpu.regs.get(RegisterName.F) == flags
    assert reads == [(0xEC, 0x20), (0x20, 0x81)]
    assert writes == [(INTERNAL_MEMORY_START + 0x20, result)]


@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize(
    "opcode,register,expected,width",
    [(0x82, "BA", 0x5AA5, 2), (0x84, "X", 0xF5AA5, 3), (0x10, "PC", 0xF5AA5, 3)],
)
@pytest.mark.parametrize(
    "prefix,selector,bases,base_reads",
    [
        (b"\x30", 0x20, {0xEC: [0x70]}, []),
        (b"", 0, {0xEC: [0x20, 0x30]}, [(0xEC, 0x20)]),
        (b"\x34", 0, {0xED: [0x20, 0x30]}, [(0xED, 0x20)]),
        (
            b"\x24",
            0,
            {0xEC: [0x10, 0x30], 0xED: [0x10, 0x40]},
            [(0xEC, 0x10), (0xED, 0x10)],
        ),
    ],
)
def test_wide_source_samples_each_base_once(
    backend, opcode, register, expected, width, prefix, selector, bases, base_reads
):
    data = [0xA5, 0x5A, 0xCF]
    cpu, _, reads, writes = run(
        backend,
        prefix + bytes([opcode, selector]),
        {0x20 + i: value for i, value in enumerate(data)},
        bases,
    )
    assert cpu.regs.get(getattr(RegisterName, register)) == expected
    assert cpu.regs.get(RegisterName.F) == 3
    assert reads == base_reads + [
        (0x20 + i, value) for i, value in enumerate(data[:width])
    ]
    assert not writes


@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("opcode,register", [(0x84, "X"), (0x10, "PC")])
def test_latched_wide_source_wraps_inside_imem(backend, direct, opcode, register):
    program = bytes([0x30, opcode, 0xFF]) if direct else bytes([opcode, 0])
    cpu, _, reads, writes = run(
        backend, program, {0xFF: 0xA5, 0: 0x5A, 1: 0xCF}, {0xEC: [0xFF, 0x70]}
    )
    assert cpu.regs.get(getattr(RegisterName, register)) == 0xF5AA5
    assert reads == ([] if direct else [(0xEC, 0xFF)]) + [
        (0xFF, 0xA5),
        (0, 0x5A),
        (1, 0xCF),
    ]
    assert not writes


@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize(
    "opcode,width,move",
    [(0xC6, 2, False), (0xC7, 3, False), (0xC9, 2, True), (0xCA, 3, True)],
)
def test_distinct_operand_addresses_remain_latched(backend, opcode, width, move):
    # BP supplies one sample per operand, not one global sample per instruction.
    # All later accidental reads return 70 and must not redirect a byte.
    contents = {0x20 + i: v for i, v in enumerate([0xA5, 0x5A, 0xCF])}
    contents.update({0x40 + i: v for i, v in enumerate([0xA5, 0x5A, 0xCF])})
    cpu, _, reads, writes = run(
        backend, bytes([opcode, 0, 0]), contents, {0xEC: [0x20, 0x40, 0x70]}
    )
    expected_reads = [(0xEC, 0x20), (0xEC, 0x40)]
    if not move:
        expected_reads += [(0x20 + i, contents[0x20 + i]) for i in range(width)]
    expected_reads += [(0x40 + i, contents[0x40 + i]) for i in range(width)]
    assert reads == expected_reads
    assert cpu.regs.get(RegisterName.F) == (3 if move else 2)
    assert writes == (
        [(INTERNAL_MEMORY_START + 0x20 + i, contents[0x40 + i]) for i in range(width)]
        if move
        else []
    )


@pytest.mark.parametrize("backend", ["python", "llama"])
def test_reti_bridge_synchronization_does_not_read_imr_or_isr(backend):
    cpu, raw, reads, writes = run(backend, b"\x01", {0xFB: 0, 0xFC: 0x08}, {})
    assert reads == []
    assert writes == [(INTERNAL_MEMORY_START + 0xFB, 0xA1)]
    assert raw[INTERNAL_MEMORY_START + 0xFC] == 0x08
    assert cpu.regs.get(RegisterName.PC) == 0x2000
    assert cpu.regs.get(RegisterName.S) == 0x80005
    assert cpu.regs.get(RegisterName.F) == 0


@pytest.mark.parametrize("opcode", [0x01, 0xFE])
@pytest.mark.parametrize("with_hook", [False, True])
def test_native_irq_notifications_use_silent_post_instruction_state(opcode, with_hook):
    events = [] if with_hook else None
    returning = opcode == 0x01
    budget = {0xFB: 0 if returning else 1, 0xFC: 0}
    cpu, _, reads, _ = run(
        "llama",
        bytes([opcode]),
        {0xFB: 0xA1, 0xFC: 0x08},
        {},
        irq_events=events,
        read_budget=budget,
    )
    assert cpu.regs.get(RegisterName.PC) == 0x2000
    assert reads == ([] if returning else [(0xFB, 0xA1)])
    if with_hook:
        assert events is not None
        assert [name for name, _ in events] == [
            "IMR_Write",
            "IRQ_Return" if returning else "IRQ_Enter",
        ]
        assert events[1][1]["imr"] == (0xA1 if returning else 0x21)
        assert events[1][1]["isr"] == 0x08
