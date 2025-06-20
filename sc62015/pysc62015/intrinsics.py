"""SC62015-specific intrinsic evaluators."""

from typing import Optional, Tuple, Callable, Dict, Protocol, TypedDict


class RegistersLike(Protocol):
    """Minimal register access interface used by the evaluator."""

    def get_by_name(self, name: str) -> int:  # pragma: no cover - protocol
        ...

    def set_by_name(self, name: str, value: int) -> None:  # pragma: no cover - protocol
        ...


class Memory:
    """Simple memory helper used by the LLIL evaluator."""

    def read_byte(self, address: int) -> int:
        ...

    def write_byte(self, address: int, value: int) -> None:
        ...


class State:
    """Execution state."""
    halted: bool


class ResultFlags(TypedDict, total=False):
    C: Optional[int]
    Z: Optional[int]


FlagGetter = Callable[[str], int]
FlagSetter = Callable[[str, int], None]


class MockLLIL:
    """Mock LLIL for type hints."""
    pass


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
    from binja_helpers.eval_llil import register_intrinsic
    
    register_intrinsic("TCL", eval_intrinsic_tcl)
    register_intrinsic("HALT", eval_intrinsic_halt)
    register_intrinsic("OFF", eval_intrinsic_off)
