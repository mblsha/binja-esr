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
    
    This instruction halts the processor and modifies system registers per SC62015 spec:
    - USR (F8H) bits 0 to 2/5 are reset to 0
    - SSR (FFH) bit 2 is set to 1
    - USR (F8H) bits 3 and 4 are set to 1
    """
    # Modify USR register (F8H)
    usr = memory.read_byte(0xF8)
    usr &= ~0x3F  # Clear bits 0-5 (reset to 0)
    usr |= 0x18   # Set bits 3 and 4 to 1
    memory.write_byte(0xF8, usr)
    
    # Modify SSR register (FFH)
    ssr = memory.read_byte(0xFF)
    ssr |= 0x04   # Set bit 2 to 1
    memory.write_byte(0xFF, ssr)
    
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
    
    This instruction turns off the processor and modifies system registers per SC62015 spec:
    - USR (F8H) bits 0 to 2/5 are reset to 0
    - SSR (FFH) bit 2 is set to 1
    - USR (F8H) bits 3 and 4 are set to 1
    Same as HALT but represents a different power state (main/sub clock stop).
    """
    # Modify USR register (F8H)
    usr = memory.read_byte(0xF8)
    usr &= ~0x3F  # Clear bits 0-5 (reset to 0)
    usr |= 0x18   # Set bits 3 and 4 to 1
    memory.write_byte(0xF8, usr)
    
    # Modify SSR register (FFH)
    ssr = memory.read_byte(0xFF)
    ssr |= 0x04   # Set bit 2 to 1
    memory.write_byte(0xFF, ssr)
    
    state.halted = True
    return None, None


def eval_intrinsic_reset(
    llil: MockLLIL,
    size: Optional[int],
    regs: RegistersLike,
    memory: Memory,
    state: State,
    get_flag: FlagGetter,
    set_flag: FlagSetter,
) -> Tuple[None, Optional[ResultFlags]]:
    """Evaluate the RESET intrinsic.
    
    This instruction resets the processor per SC62015 spec:
    - ACM (FEH) bit 7 is reset to 0
    - UCR (F7H) is reset to 0
    - USR (F8H) bits 0 to 2/5 are reset to 0
    - IMR (FCH) is reset to 0
    - SCR (FDH) is reset to 0 
    - SSR (FFH) bit 2 is reset to 0
    - USR (F8H) bits 3 and 4 are set to 1
    - PC reads the reset vector at 0xFFFFA (3 bytes, little-endian)
    - Other registers retain their values
    - Flags (C/Z) are retained
    """
    # Reset ACM bit 7
    acm = memory.read_byte(0xFE)
    acm &= ~0x80  # Clear bit 7
    memory.write_byte(0xFE, acm)
    
    # Reset UCR, IMR, SCR to 0
    memory.write_byte(0xF7, 0x00)  # UCR
    memory.write_byte(0xFC, 0x00)  # IMR
    memory.write_byte(0xFD, 0x00)  # SCR
    
    # Modify USR register
    usr = memory.read_byte(0xF8)
    usr &= ~0x3F  # Clear bits 0-5 (reset to 0)
    usr |= 0x18   # Set bits 3 and 4 to 1
    memory.write_byte(0xF8, usr)
    
    # Reset SSR bit 2
    ssr = memory.read_byte(0xFF)
    ssr &= ~0x04  # Clear bit 2
    memory.write_byte(0xFF, ssr)
    
    # Read reset vector at 0xFFFFA (3 bytes, little-endian)
    reset_vector = memory.read_byte(0xFFFFA)
    reset_vector |= memory.read_byte(0xFFFFB) << 8
    reset_vector |= memory.read_byte(0xFFFFC) << 16
    
    # Set PC to reset vector (masked to 20 bits)
    from .constants import PC_MASK
    regs.set_by_name('PC', reset_vector & PC_MASK)
    
    return None, None


def register_sc62015_intrinsics() -> None:
    """Register all SC62015-specific intrinsic evaluators with the generic evaluation system."""
    # Import here to avoid circular imports
    from binja_test_mocks.eval_llil import register_intrinsic
    
    register_intrinsic("TCL", eval_intrinsic_tcl)
    register_intrinsic("HALT", eval_intrinsic_halt)
    register_intrinsic("OFF", eval_intrinsic_off)
    register_intrinsic("RESET", eval_intrinsic_reset)
