"""SC62015-specific intrinsic evaluators."""

from typing import Optional, Tuple

# Import types from the main evaluation system to ensure compatibility
from binja_test_mocks.eval_llil import (
    RegistersLike, Memory, State, ResultFlags, FlagGetter, FlagSetter
)
from binja_test_mocks.mock_llil import MockLLIL


def eval_intrinsic_tcl(
    llil: MockLLIL,
    size: Optional[int],
    regs: RegistersLike,
    memory: Memory,
    state: State,
    get_flag: FlagGetter,
    set_flag: FlagSetter,
) -> Tuple[None, Optional[ResultFlags]]:
    """Evaluate the TCL (Test Clock) intrinsic.
    
    This is a no-op instruction that doesn't affect processor state.
    """
    return None, None


def eval_intrinsic_halt(
    llil: MockLLIL,
    size: Optional[int],
    regs: RegistersLike,
    memory: Memory,
    state: State,
    get_flag: FlagGetter,
    set_flag: FlagSetter,
) -> Tuple[None, Optional[ResultFlags]]:
    """Evaluate the HALT intrinsic.
    
    This instruction halts the processor by setting the halted state flag.
    """
    state.halted = True
    return None, None


def eval_intrinsic_off(
    llil: MockLLIL,
    size: Optional[int],
    regs: RegistersLike,
    memory: Memory,
    state: State,
    get_flag: FlagGetter,
    set_flag: FlagSetter,
) -> Tuple[None, Optional[ResultFlags]]:
    """Evaluate the OFF intrinsic.
    
    This instruction turns off the processor by setting the halted state flag.
    Similar to HALT but represents a different power state.
    """
    state.halted = True
    return None, None


def register_sc62015_intrinsics() -> None:
    """Register all SC62015-specific intrinsic evaluators with the generic evaluation system."""
    # Import here to avoid circular imports
    from binja_test_mocks.eval_llil import register_intrinsic
    
    register_intrinsic("TCL", eval_intrinsic_tcl)
    register_intrinsic("HALT", eval_intrinsic_halt)
    register_intrinsic("OFF", eval_intrinsic_off)
