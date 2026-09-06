"""BCD must latch each operand once per byte, including callback-backed IMEM.

These are bus-contract tests, not measurements of internal silicon micro-order.
Expected decimal results are fixed examples, not computed by either backend.
"""

from collections import Counter

from binja_test_mocks.eval_llil import Memory
import pytest

from sc62015.pysc62015 import CPU, RegisterName
from sc62015.pysc62015.constants import ADDRESS_SPACE_SIZE, INTERNAL_MEMORY_START


def run_case(backend, family, source, left, right, outputs, flags, changing):
    count = len(left)
    dst = 0x20 + count - 1
    src = dst if source == "alias" else 0x30 + count - 1
    opcode = (0xC4 if family == "dadl" else 0xD4) + (source == "a")
    program = bytes([0x30 if source == "a" else 0x32, opcode, dst])
    if source != "a":
        program += bytes([src])
    raw = bytearray(ADDRESS_SPACE_SIZE)
    raw[0x1000 : 0x1000 + len(program)] = program
    values = {}
    expected_events = []
    for index, output in enumerate(outputs):
        address = INTERNAL_MEMORY_START + dst - index
        values.setdefault(address, []).append(left[index])
        expected_events.append(("read", address, left[index]))
        if source != "a":
            rhs_address = INTERNAL_MEMORY_START + src - index
            values.setdefault(rhs_address, []).append(right[index])
            expected_events.append(("read", rhs_address, right[index]))
        expected_events.append(("write", address, output))
    # For alias tests the two operand reads have explicitly distinct witnesses;
    # latching once per operand must not collapse them into one address read.
    reads = Counter()
    events = []

    def read(address):
        if address not in values:
            return raw[address]
        index = reads[address]
        reads[address] += 1
        expected = values[address]
        value = expected[min(index, len(expected) - 1)]
        if changing and index >= len(expected):
            value = 0x77  # any unintended subsequent read sees another byte
        events.append(("read", address, value))
        return value

    def write(address, value):
        raw[address] = value
        if address in values:
            events.append(("write", address, value))

    memory = Memory(read, write)
    setattr(memory, "peek_byte_for_preflight", lambda address, _pc=None: raw[address])
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.PC, 0x1000)
    cpu.regs.set(RegisterName.I, count)
    cpu.regs.set(RegisterName.A, right[0])
    cpu.regs.set(RegisterName.F, 3)  # ignored initial C; Z recomputed
    cpu.execute_instruction(0x1000)
    actual = [raw[INTERNAL_MEMORY_START + dst - i] for i in range(count)]
    assert actual == outputs
    assert cpu.regs.get(RegisterName.F) == flags
    assert cpu.regs.get(RegisterName.I) == 0
    assert cpu.regs.get(RegisterName.PC) == 0x1000 + len(program)
    assert cpu.regs.get(RegisterName.A) == right[0]
    assert events == expected_events


@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize("source", ["memory", "a"])
@pytest.mark.parametrize("changing", [False, True], ids=["stable", "changing"])
@pytest.mark.parametrize(
    "family,left,right,output,flags",
    [
        ("dadl", 0x09, 0x01, 0x10, 0),
        ("dadl", 0x99, 0x01, 0x00, 3),
        ("dadl", 0x0F, 0x0F, 0x14, 0),
        ("dadl", 0xFF, 0xFF, 0x54, 1),
        ("dsbl", 0x10, 0x01, 0x09, 0),
        ("dsbl", 0x00, 0x01, 0x99, 1),
        ("dsbl", 0x0A, 0x00, 0x94, 1),
        ("dsbl", 0x00, 0x00, 0x00, 2),
    ],
)
def test_each_byte_operand_is_read_once(
    backend, source, changing, family, left, right, output, flags
):
    run_case(backend, family, source, [left], [right], [output], flags, changing)


@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize("source", ["memory", "a", "alias"])
@pytest.mark.parametrize("family", ["dadl", "dsbl"])
def test_operands_relatch_each_iteration_and_keep_generated_carry(
    backend, source, family
):
    # The upper byte uses zero for both memory and consumed-once A forms.
    left, output = (
        ([0x99, 0x00], [0x00, 0x01])
        if family == "dadl"
        else ([0x00, 0x10], [0x99, 0x09])
    )
    run_case(backend, family, source, left, [0x01, 0x00], output, 0, True)
