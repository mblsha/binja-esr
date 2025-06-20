"""Tests for SC62015-specific intrinsic evaluators."""

import sys
import os

# Add the project root to the path to avoid circular import issues
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .intrinsics import (
    eval_intrinsic_tcl,
    eval_intrinsic_halt,
    eval_intrinsic_off,
    register_sc62015_intrinsics,
)


class SimpleRegs:
    """Simple register implementation for testing."""
    
    def __init__(self):
        self.values = {}

    def get_by_name(self, name: str) -> int:
        return self.values.get(name, 0)

    def set_by_name(self, name: str, value: int) -> None:
        self.values[name] = value & 0xFFFFFFFF


class SimpleState:
    """Simple state implementation for testing."""
    
    def __init__(self):
        self.halted = False


class SimpleMemory:
    """Simple memory implementation for testing."""
    
    def __init__(self, size=0x100):
        self.buf = bytearray(size)
    
    def read_byte(self, addr: int) -> int:
        return self.buf[addr]
    
    def write_byte(self, addr: int, value: int) -> None:
        self.buf[addr] = value & 0xFF


def dummy_flag_getter(name: str) -> int:
    """Dummy flag getter for testing."""
    return 0


def dummy_flag_setter(name: str, value: int) -> None:
    """Dummy flag setter for testing."""
    pass


def test_sc62015_intrinsic_handlers() -> None:
    """Test SC62015-specific intrinsic evaluators."""
    regs = SimpleRegs()
    memory = SimpleMemory()
    state = SimpleState()

    # Test TCL intrinsic (should be a no-op)
    result, flags = eval_intrinsic_tcl(
        None, None, regs, memory, state, dummy_flag_getter, dummy_flag_setter
    )
    assert result is None
    assert flags is None
    assert not state.halted

    # Test HALT intrinsic (should halt the processor)
    result, flags = eval_intrinsic_halt(
        None, None, regs, memory, state, dummy_flag_getter, dummy_flag_setter
    )
    assert result is None
    assert flags is None
    assert state.halted

    # Test OFF intrinsic (should also halt the processor)
    state.halted = False
    result, flags = eval_intrinsic_off(
        None, None, regs, memory, state, dummy_flag_getter, dummy_flag_setter
    )
    assert result is None
    assert flags is None
    assert state.halted


def test_intrinsic_registration() -> None:
    """Test that intrinsic registration function exists."""
    # This test verifies that the registration function exists.
    # We can't actually test the registration due to circular import issues
    # in the test environment, but we can verify the function is defined.
    
    # Just verify the function exists
    assert callable(register_sc62015_intrinsics)
    
    # The actual registration will be tested as part of integration tests
    # when the emulator is initialized
