"""Fail-closed contracts enforced by the backend-neutral CPU facade."""

from __future__ import annotations

from typing import Any

import pytest

from sc62015.pysc62015.cpu import CPU
from sc62015.pysc62015.emulator import Registers


class _DecodedOneByteInstruction:
    def analyze(self, info: Any, _address: int) -> None:
        info.length = 1


class _LegacyDecoder:
    def decode_instruction(
        self, _address: int, read_fn: Any = None
    ) -> _DecodedOneByteInstruction:
        if read_fn is not None:
            assert read_fn(0) == 0
        return _DecodedOneByteInstruction()


class _PreflightMemory:
    @staticmethod
    def peek_byte_for_preflight(_address: int, _pc: int | None = None) -> int:
        return 0


class _SideEffectingNative:
    def __init__(self) -> None:
        self.execution_count = 0
        self.reset_error: Exception | None = None
        self.reset_kwargs: dict[str, int | None] = {}

    def execute_instruction(self, _address: int) -> tuple[int, int]:
        self.execution_count += 1
        # Model a native instruction which has already committed a side effect
        # before returning a length that violates the decoder/runtime contract.
        return 0x00, 2

    def power_on_reset(self, **kwargs: int | None) -> None:
        self.reset_kwargs = kwargs
        if self.reset_error is not None:
            raise self.reset_error


def _facade_with_fake_native(native: _SideEffectingNative) -> CPU:
    cpu = CPU.__new__(CPU)
    cpu._impl = native
    setattr(cpu, "_legacy_decoder", _LegacyDecoder())
    cpu.memory = _PreflightMemory()
    cpu.regs = Registers()
    cpu.backend = "llama"
    cpu._contract_poisoned = None
    return cpu


def test_post_execution_length_mismatch_poison_blocks_double_execution() -> None:
    native = _SideEffectingNative()
    cpu = _facade_with_fake_native(native)

    with pytest.raises(RuntimeError, match="Decoded length .* disagrees"):
        cpu.execute_instruction(0)
    assert native.execution_count == 1

    with pytest.raises(RuntimeError, match="facade is poisoned"):
        cpu.execute_instruction(0)
    assert native.execution_count == 1


def test_failed_reset_preserves_facade_contract_poison() -> None:
    native = _SideEffectingNative()
    cpu = _facade_with_fake_native(native)
    with pytest.raises(RuntimeError, match="Decoded length .* disagrees"):
        cpu.execute_instruction(0)

    native.reset_error = RuntimeError("reset failed")
    with pytest.raises(RuntimeError, match="reset failed"):
        cpu.power_on_reset()
    with pytest.raises(RuntimeError, match="facade is poisoned"):
        cpu.execute_instruction(0)
    assert native.execution_count == 1

    native.reset_error = None
    cpu.power_on_reset()
    with pytest.raises(RuntimeError, match="Decoded length .* disagrees"):
        cpu.execute_instruction(0)
    assert native.execution_count == 2


def test_native_reset_facade_exposes_no_prefetched_scalar_bypass() -> None:
    native = _SideEffectingNative()
    cpu = _facade_with_fake_native(native)

    cpu.power_on_reset()

    assert native.reset_kwargs == {}
    with pytest.raises(TypeError):
        cpu.power_on_reset(prefetched_vector=0x00200)  # type: ignore[call-arg]
