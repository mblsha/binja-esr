from __future__ import annotations

import os

import pytest
from binja_test_mocks.eval_llil import Memory

from sc62015.pysc62015 import CPU, RegisterName, available_backends
from sc62015.pysc62015.constants import ADDRESS_SPACE_SIZE
from sc62015.pysc62015.stepper import CPURegistersSnapshot


def _make_memory(*bytes_seq: int) -> Memory:
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
    setattr(
        memory,
        "peek_byte_for_preflight",
        lambda address, _pc=None: raw[address & 0xFFFFFF],
    )
    return memory


def test_cpu_facade_executes_nop(cpu_backend: str) -> None:
    """Ensure all enabled backends can execute a trivial instruction."""

    memory = _make_memory(0x00)  # NOP

    cpu = CPU(memory, reset_on_init=False, backend=cpu_backend)

    cpu.regs.set(RegisterName.PC, 0x0000)
    info = cpu.execute_instruction(0x0000)

    assert info.instruction.name() == "NOP"
    assert cpu.backend_stats()["execution_scope"] == "cpu-only"
    assert cpu.backend_stats()["scheduler_owner"] == "python-caller"


def test_cpu_stepper_round_trip(cpu_backend: str) -> None:
    """Verify snapshot-based stepping matches per-backend execution."""

    memory = _make_memory(0x08, 0x5A)  # MV A,n

    cpu = CPU(memory, reset_on_init=False, backend=cpu_backend)

    cpu.regs.set(RegisterName.PC, 0x0000)
    cpu.regs.set(RegisterName.A, 0x00)

    snapshot = CPURegistersSnapshot.from_registers(cpu.regs)
    result = cpu.step_snapshot(snapshot, {0x0000: 0x08, 0x0001: 0x5A})

    assert result.registers.ba & 0xFF == 0x5A


@pytest.mark.parametrize("opcode", [0x46, 0x4E])
def test_byte_arithmetic_wide_source_is_rejected_by_default(
    cpu_backend: str, opcode: int
) -> None:
    cpu = CPU(_make_memory(opcode, 0x03), reset_on_init=False, backend=cpu_backend)
    initial = CPURegistersSnapshot(pc=0, ba=8, i=0x102, f=3)
    cpu.apply_snapshot(initial)
    initial = cpu.snapshot_registers()
    with pytest.raises(Exception, match="Invalid|invalid"):
        cpu.execute_instruction(0)
    assert cpu.snapshot_registers() == initial


@pytest.mark.parametrize(
    "opcode,a,i,result,flags",
    [
        (0x4E, 8, 2, 6, 0),
        (0x4E, 8, 0x102, 6, 0),
        (0x4E, 8, 9, 0xFF, 1),
        (0x4E, 0x80, 1, 0x7F, 0),
        (0x4E, 2, 0x102, 0, 2),
        (0x46, 0xFE, 0x102, 0, 3),
    ],
)
def test_opt_in_byte_arithmetic_source_is_truncated_before_flags(
    cpu_backend: str, opcode: int, a: int, i: int, result: int, flags: int
) -> None:
    cpu = CPU(
        _make_memory(opcode, 0x03),
        reset_on_init=False,
        backend=cpu_backend,
        byte_arithmetic_source_policy="low_byte",
    )
    cpu.apply_snapshot(CPURegistersSnapshot(pc=0, ba=0xAB00 | a, i=i, f=3))
    cpu.execute_instruction(0)
    assert cpu.regs.get(RegisterName.BA) == 0xAB00 | result
    assert cpu.regs.get(RegisterName.I) == i
    assert cpu.regs.get(RegisterName.F) == flags
    assert cpu.regs.get(RegisterName.PC) == 2
    assert cpu.backend_stats()["byte_arithmetic_source_policy"] == "low_byte"


@pytest.mark.parametrize("selector", [0x23, 0x43, 0x83, 0x0B])
def test_byte_source_policy_preserves_destination_and_reserved_bit_checks(
    cpu_backend: str, selector: int
) -> None:
    cpu = CPU(
        _make_memory(0x4E, selector),
        reset_on_init=False,
        backend=cpu_backend,
        byte_arithmetic_source_policy="low_byte",
    )
    initial = CPURegistersSnapshot(pc=0, ba=8, i=2, f=3)
    cpu.apply_snapshot(initial)
    initial = cpu.snapshot_registers()
    with pytest.raises(Exception, match="Invalid|invalid"):
        cpu.execute_instruction(0)
    assert cpu.snapshot_registers() == initial


def test_snapshot_step_uses_callers_byte_source_policy(cpu_backend: str) -> None:
    cpu = CPU(
        _make_memory(0),
        reset_on_init=False,
        backend=cpu_backend,
        byte_arithmetic_source_policy="low_byte",
    )
    result = cpu.step_snapshot(
        CPURegistersSnapshot(pc=0, ba=8, i=0x102, f=3), {0: 0x4E, 1: 0x03}
    )
    assert result.registers.ba == 6
    assert result.registers.i == 0x102
    assert result.registers.f == 0
    assert result.registers.pc == 2


def test_byte_source_policy_is_validated_before_execution(cpu_backend: str) -> None:
    with pytest.raises(ValueError, match="Unknown byte arithmetic source policy"):
        CPU(
            _make_memory(0),
            reset_on_init=False,
            backend=cpu_backend,
            byte_arithmetic_source_policy="unknown",
        )


@pytest.mark.parametrize(
    "selector,ba,i,y,expected_ba,expected_i,flags",
    [
        (0x05, 0xAB08, 0, 0xF0102, 0xAB06, 0, 0),  # SUB A,Y: low pointer byte
        (0x02, 0xAB08, 0, 0, 0xAB00, 0, 2),  # SUB A,BA: read alias before write
        (0x12, 0xAB02, 0xCD08, 0, 0xAB02, 6, 0),  # SUB IL,BA: existing IL alias policy
    ],
)
def test_byte_source_policy_handles_pointers_and_register_aliases(
    cpu_backend: str,
    selector: int,
    ba: int,
    i: int,
    y: int,
    expected_ba: int,
    expected_i: int,
    flags: int,
) -> None:
    cpu = CPU(
        _make_memory(0x4E, selector),
        reset_on_init=False,
        backend=cpu_backend,
        byte_arithmetic_source_policy="low_byte",
    )
    cpu.apply_snapshot(CPURegistersSnapshot(pc=0, ba=ba, i=i, y=y, f=3))
    cpu.execute_instruction(0)
    assert cpu.regs.get(RegisterName.BA) == expected_ba
    assert cpu.regs.get(RegisterName.I) == expected_i
    assert cpu.regs.get(RegisterName.Y) == y
    assert cpu.regs.get(RegisterName.F) == flags
    assert cpu.regs.get(RegisterName.PC) == 2


@pytest.mark.parametrize("policy", ["strict", "low_byte"])
def test_vector_preflight_uses_callers_byte_source_policy(
    cpu_backend: str, policy: str
) -> None:
    memory = _make_memory(0)
    raw: bytearray = getattr(memory, "_raw")
    raw[0xFFFFD:0x100000] = bytes.fromhex("00 10 00")
    raw[0x1000:0x1002] = bytes.fromhex("4e 03")
    cpu = CPU(
        memory,
        reset_on_init=False,
        backend=cpu_backend,
        byte_arithmetic_source_policy=policy,
    )
    initial = cpu.snapshot_registers()
    if policy == "strict":
        with pytest.raises(Exception, match="Invalid|invalid"):
            cpu.preflight_vector_transfer(0xFFFFD)
    else:
        assert cpu.preflight_vector_transfer(0xFFFFD) == 0x1000
    assert cpu.snapshot_registers() == initial


@pytest.mark.parametrize("policy", ["independent", "coupled_predecrement"])
@pytest.mark.parametrize("opcode", [0xE3, 0xEB])
@pytest.mark.parametrize("mode", [0x26, 0x36])
def test_scoped_mvl_policy_matches_internal_direction_and_pointer_updates(
    cpu_backend: str, policy: str, opcode: int, mode: int
) -> None:
    memory = _make_memory(0x34, opcode, mode, 0x07)
    raw: bytearray = getattr(memory, "_raw")
    raw[0x1000EC] = 0x70
    raw[0x1000ED] = 0x40
    coupled = policy == "coupled_predecrement" and mode == 0x36
    pairs = [
        (
            0x100000 + 0x47 + (-n if coupled else n),
            0x3EFF - n if mode == 0x36 else 0x3F00 + n,
        )
        for n in range(8)
    ]
    for n, (internal, external) in enumerate(pairs):
        raw[external if opcode == 0xE3 else internal] = n + 1
    cpu = CPU(
        memory, reset_on_init=False, backend=cpu_backend, block_transfer_policy=policy
    )
    cpu.regs.set(RegisterName.U, 0x3F00)
    cpu.regs.set(RegisterName.I, 8)
    cpu.regs.set(RegisterName.F, 3)

    cpu.execute_instruction(0)

    for n, (internal, external) in enumerate(pairs):
        assert raw[internal if opcode == 0xE3 else external] == n + 1
    assert cpu.regs.get(RegisterName.U) == (0x3EF8 if mode == 0x36 else 0x3F08)
    assert cpu.regs.get(RegisterName.I) == 0
    assert cpu.regs.get(RegisterName.F) == 3
    assert cpu.regs.get(RegisterName.PC) == 4
    assert cpu.backend_stats()["block_transfer_policy"] == policy


def test_snapshot_step_uses_callers_block_transfer_policy(cpu_backend: str) -> None:
    cpu = CPU(
        _make_memory(0),
        reset_on_init=False,
        backend=cpu_backend,
        block_transfer_policy="coupled_predecrement",
    )
    image = {0: 0xEB, 1: 0x36, 2: 0x47}
    image.update({0x100040 + n: n + 1 for n in range(8)})
    result = cpu.step_snapshot(CPURegistersSnapshot(pc=0, i=8, u=0x3F00), image)
    assert [result.memory_image[0x3EF8 + n] for n in range(8)] == list(range(1, 9))
    assert result.registers.u == 0x3EF8


@pytest.mark.parametrize("opcode", [0xE3, 0xEB])
def test_coupled_mvl_wraps_internal_offsets_and_external_pointer(
    cpu_backend: str, opcode: int
) -> None:
    memory = _make_memory(0)
    raw: bytearray = getattr(memory, "_raw")
    raw[0x1000:0x1003] = bytes([opcode, 0x36, 3])
    pairs = [(0x100000 + ((3 - n) & 0xFF), 0xFFFFF - n) for n in range(8)]
    for n, (internal, external) in enumerate(pairs):
        raw[external if opcode == 0xE3 else internal] = n + 1
    cpu = CPU(
        memory,
        reset_on_init=False,
        backend=cpu_backend,
        block_transfer_policy="coupled_predecrement",
    )
    cpu.regs.set(RegisterName.PC, 0x1000)
    cpu.regs.set(RegisterName.U, 0)
    cpu.regs.set(RegisterName.I, 8)
    cpu.regs.set(RegisterName.F, 3)

    cpu.execute_instruction(0x1000)

    for n, (internal, external) in enumerate(pairs):
        assert raw[internal if opcode == 0xE3 else external] == n + 1
    assert cpu.regs.get(RegisterName.U) == 0xFFFF8
    assert cpu.regs.get(RegisterName.I) == 0
    assert cpu.regs.get(RegisterName.F) == 3
    assert cpu.regs.get(RegisterName.PC) == 0x1003


def test_mvl_policy_is_validated_before_execution(cpu_backend: str) -> None:
    with pytest.raises(ValueError, match="Unknown block transfer policy"):
        CPU(
            _make_memory(0),
            reset_on_init=False,
            backend=cpu_backend,
            block_transfer_policy="unknown",
        )


@pytest.mark.parametrize("pc", [0x7FFFE, 0x7FFFF, 0xFFFFF])
def test_near_ret_uses_page_after_opcode_fetch(cpu_backend: str, pc: int) -> None:
    """RET composes its 16-bit target with the page of the advanced PC."""

    memory = _make_memory(0x00)
    raw: bytearray = getattr(memory, "_raw")
    raw[pc] = 0x06  # RET
    raw[0x80000:0x80002] = bytes.fromhex("34 12")
    cpu = CPU(memory, reset_on_init=False, backend=cpu_backend)
    cpu.apply_snapshot(CPURegistersSnapshot(pc=pc, s=0x80000))

    cpu.execute_instruction(pc)

    expected_page = ((pc + 1) & 0xFFFFF) & 0xF0000
    assert cpu.regs.get(RegisterName.PC) == expected_page | 0x1234
    assert cpu.regs.get(RegisterName.S) == 0x80002


@pytest.mark.skipif(not os.environ.get("CI"), reason="CI backend-availability guard")
def test_ci_parity_requires_llama_backend() -> None:
    assert "llama" in available_backends(), (
        "CI parity tests require the built LLAMA backend; absence must not become a skip"
    )
