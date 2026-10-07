"""Explicit ESR-P ISR compatibility hypothesis; not silicon qualification."""

import pytest
from collections.abc import Callable
from binja_test_mocks.eval_llil import Memory

from . import CPU, RegisterName
from .constants import ADDRESS_SPACE_SIZE, INTERNAL_MEMORY_START
from .stepper import CPURegistersSnapshot


class PreflightMemory(Memory):
    """Fixture with an explicit side-effect-free peek supplied by each test."""

    peek_byte_for_preflight: Callable[[int, int | None], int]


@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize("policy", ["replace", "clear_only", "clear_only_except_mti"])
@pytest.mark.parametrize("initial", [0, 0x5D, 0x80, 0xFF])
def test_guest_isr_writes_cannot_replace_hardware_latch_in_clear_only_policy(
    backend, policy, initial
):
    raw = bytearray(ADDRESS_SPACE_SIZE)
    code = bytes.fromhex("30 cd fb f0 ff 30 79 fc ff 30 71 fc ef")
    raw[0x1000 : 0x1000 + len(code)] = code
    raw[INTERNAL_MEMORY_START + 0xFC] = initial
    memory = PreflightMemory(raw.__getitem__, raw.__setitem__)
    memory.peek_byte_for_preflight = lambda address, _pc=None: raw[address]
    cpu = CPU(
        memory, reset_on_init=False, backend=backend, isr_software_write_policy=policy
    )
    initial_latch = initial | (1 if policy == "clear_only_except_mti" else 0)
    for pc, expected in [
        (0x1000, initial_latch),
        (0x1005, initial_latch),
        (0x1009, initial_latch & 0xEF),
    ]:
        cpu.execute_instruction(pc)
        assert raw[INTERNAL_MEMORY_START + 0xFC] == (
            expected if policy != "replace" else (0xEF if pc == 0x1009 else 0xFF)
        )
    # Host/peripheral writes continue to be able to set RXRI/EXI after a clear.
    memory.write_byte(INTERNAL_MEMORY_START + 0xFC, 0x60)
    cpu.notify_host_write(INTERNAL_MEMORY_START + 0xFC, 0x60)
    cpu.execute_instruction(0x1005)
    assert raw[INTERNAL_MEMORY_START + 0xFC] == (
        (0x61 if policy == "clear_only_except_mti" else 0x60)
        if policy != "replace"
        else 0xFF
    )
    assert raw[INTERNAL_MEMORY_START + 0xFB] == 0xF0


@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize("policy", ["clear_only", "clear_only_except_mti"])
def test_block_copy_masks_only_isr_byte_and_preserves_adjacent_ports(backend, policy):
    raw = bytearray(ADDRESS_SPACE_SIZE)
    raw[0x1000:0x1004] = bytes.fromhex("30 cb fa 20")
    raw[INTERNAL_MEMORY_START + 0x20 : INTERNAL_MEMORY_START + 0x24] = bytes.fromhex(
        "11 22 ff 44"
    )
    raw[INTERNAL_MEMORY_START + 0xFC] = 4
    memory = PreflightMemory(raw.__getitem__, raw.__setitem__)
    memory.peek_byte_for_preflight = lambda address, _pc=None: raw[address]
    cpu = CPU(
        memory,
        reset_on_init=False,
        backend=backend,
        isr_software_write_policy=policy,
    )
    cpu.regs.set(RegisterName.I, 4)
    cpu.execute_instruction(0x1000)
    assert raw[
        INTERNAL_MEMORY_START + 0xFA : INTERNAL_MEMORY_START + 0xFE
    ] == bytes.fromhex(
        "11 22 05 44" if policy == "clear_only_except_mti" else "11 22 04 44"
    )
    assert cpu.regs.get(RegisterName.I) == 0


@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize("policy", ["clear_only", "clear_only_except_mti"])
def test_snapshot_step_retains_the_callers_isr_write_policy(backend, policy):
    memory = PreflightMemory(lambda _address: 0, lambda _address, _value: None)
    memory.peek_byte_for_preflight = lambda _address, _pc=None: 0
    cpu = CPU(
        memory,
        reset_on_init=False,
        backend=backend,
        isr_software_write_policy=policy,
    )
    result = cpu.step_snapshot(
        CPURegistersSnapshot(pc=0x1000),
        {
            0x1000: 0x30,
            0x1001: 0xCC,
            0x1002: 0xFC,
            0x1003: 0xFF,
            INTERNAL_MEMORY_START + 0xFC: 0x12,
        },
    )
    assert result.memory_image[INTERNAL_MEMORY_START + 0xFC] == (
        0x13 if policy == "clear_only_except_mti" else 0x12
    )
    assert cpu.backend_stats()["isr_software_write_policy"] == policy


@pytest.mark.parametrize("backend", ["python", "llama"])
def test_unknown_isr_policy_is_rejected_before_reset_or_bus_access(backend):
    calls = []
    memory = Memory(
        lambda address: calls.append(address) or 0,
        lambda address, _value: calls.append(address),
    )
    with pytest.raises(ValueError, match="Unknown ISR software write policy"):
        CPU(memory, backend=backend, isr_software_write_policy="unknown")
    assert calls == []


@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize("policy", ["clear_only", "clear_only_except_mti"])
def test_clear_only_isr_does_not_filter_external_address_fc(backend, policy):
    raw = bytearray(ADDRESS_SPACE_SIZE)
    raw[0x1000:0x1005] = bytes.fromhex("d8 fc 00 00 20")
    raw[INTERNAL_MEMORY_START + 0x20] = 0xFF
    raw[INTERNAL_MEMORY_START + 0xFC] = 4
    memory = PreflightMemory(raw.__getitem__, raw.__setitem__)
    memory.peek_byte_for_preflight = lambda address, _pc=None: raw[address]
    cpu = CPU(
        memory,
        reset_on_init=False,
        backend=backend,
        isr_software_write_policy=policy,
    )
    cpu.execute_instruction(0x1000)
    assert raw[0xFC] == 0xFF
    assert raw[INTERNAL_MEMORY_START + 0xFC] == 4


@pytest.mark.parametrize("backend", ["python", "llama"])
@pytest.mark.parametrize("policy", ["clear_only", "clear_only_except_mti"])
def test_clear_only_policy_requires_a_silent_sfr_peek_before_bus_access(
    backend, policy
):
    calls = []
    memory = Memory(
        lambda address: calls.append(address) or 0,
        lambda address, _value: calls.append(address),
    )
    with pytest.raises(ValueError, match="silent internal-register peek"):
        CPU(memory, backend=backend, isr_software_write_policy=policy)
    assert calls == []
