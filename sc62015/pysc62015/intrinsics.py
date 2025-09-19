"""SC62015-specific intrinsic evaluators."""

from typing import Optional, Tuple

# Import types from the main evaluation system to ensure compatibility
from binja_test_mocks.eval_llil import (
    RegistersLike,
    Memory,
    State,
    ResultFlags,
    FlagGetter,
    FlagSetter,
)
from binja_test_mocks.mock_llil import MockLLIL

# Import register addresses from opcodes
from .instr.opcodes import IMEMRegisters
from .constants import INTERNAL_MEMORY_START


def _enter_low_power_state(memory: Memory, state: State) -> None:
    """Apply shared register updates for HALT/OFF low power modes."""

    usr_addr = INTERNAL_MEMORY_START + IMEMRegisters.USR
    usr = memory.read_byte(usr_addr)
    usr &= ~0x3F  # Clear bits 0-5 (reset to 0)
    usr |= 0x18  # Set bits 3 and 4 to 1
    memory.write_byte(usr_addr, usr)

    ssr_addr = INTERNAL_MEMORY_START + IMEMRegisters.SSR
    ssr = memory.read_byte(ssr_addr)
    ssr |= 0x04  # Set bit 2 to 1
    memory.write_byte(ssr_addr, ssr)

    state.halted = True


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
    _enter_low_power_state(memory, state)
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
    _enter_low_power_state(memory, state)
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
    - LCC (FEH) bit 7 is reset to 0 (documented as ACM bit 7)
    - UCR (F7H) is reset to 0
    - USR (F8H) bits 0 to 2/5 are reset to 0
    - ISR (FCH) is reset to 0 (clears interrupt status)
    - SCR (FDH) is reset to 0
    - SSR (FFH) bit 2 is reset to 0
    - USR (F8H) bits 3 and 4 are set to 1
    - PC reads the reset vector at 0xFFFFA (3 bytes, little-endian)
    - Other registers retain their values
    - Flags (C/Z) are retained
    """
    # Reset LCC bit 7 (documented as ACM bit 7 in RESET spec)
    lcc = memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.LCC)
    lcc &= ~0x80  # Clear bit 7
    memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.LCC, lcc)

    # Reset UCR, ISR, SCR to 0
    memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.UCR, 0x00)
    memory.write_byte(
        INTERNAL_MEMORY_START + IMEMRegisters.ISR, 0x00
    )  # Clear interrupt status
    memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.SCR, 0x00)

    # Modify USR register
    usr = memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.USR)
    usr &= ~0x3F  # Clear bits 0-5 (reset to 0)
    usr |= 0x18  # Set bits 3 and 4 to 1
    memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.USR, usr)

    # Reset SSR bit 2
    ssr = memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.SSR)
    ssr &= ~0x04  # Clear bit 2
    memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.SSR, ssr)

    # Read reset vector at 0xFFFFA (3 bytes, little-endian)
    reset_vector = memory.read_byte(0xFFFFA)
    reset_vector |= memory.read_byte(0xFFFFB) << 8
    reset_vector |= memory.read_byte(0xFFFFC) << 16

    # Set PC to reset vector (masked to 20 bits)
    from .constants import PC_MASK

    regs.set_by_name("PC", reset_vector & PC_MASK)

    return None, None


def register_sc62015_intrinsics() -> None:
    """Register all SC62015-specific intrinsic evaluators with the generic evaluation system."""
    # Import here to avoid circular imports
    from binja_test_mocks.eval_llil import register_intrinsic

    register_intrinsic("TCL", eval_intrinsic_tcl)
    register_intrinsic("HALT", eval_intrinsic_halt)
    register_intrinsic("OFF", eval_intrinsic_off)
    register_intrinsic("RESET", eval_intrinsic_reset)
