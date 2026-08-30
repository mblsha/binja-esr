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
from .constants import INTERNAL_MEMORY_START, validate_f_image


def eval_intrinsic_validate_f(
    llil: MockLLIL,
    size: Optional[int],
    regs: RegistersLike,
    memory: Memory,
    state: State,
    get_flag: FlagGetter,
    set_flag: FlagSetter,
) -> Tuple[None, Optional[ResultFlags]]:
    """Quarantine byte-wide F images whose upper-bit behavior is unknown."""

    from binja_test_mocks.eval_llil import evaluate_llil

    params = getattr(llil, "params", ())
    if len(params) != 1:
        raise RuntimeError("VALIDATE_F requires exactly one byte input")
    value, _flags = evaluate_llil(
        params[0], regs, memory, state, get_flag=get_flag, set_flag=set_flag
    )
    if value is None:
        raise RuntimeError("VALIDATE_F input did not produce a value")
    validate_f_image(value)
    return None, None


def eval_intrinsic_validate_exp_high_nibble(
    llil: MockLLIL,
    size: Optional[int],
    regs: RegistersLike,
    memory: Memory,
    state: State,
    get_flag: FlagGetter,
    set_flag: FlagSetter,
) -> Tuple[None, Optional[ResultFlags]]:
    """Quarantine unverified EXP behavior above the 20-bit pointer range."""

    from binja_test_mocks.eval_llil import evaluate_llil

    params = getattr(llil, "params", ())
    if len(params) != 2:
        raise RuntimeError("VALIDATE_EXP_HIGH_NIBBLE requires two pointer inputs")
    values = []
    for param in params:
        value, _flags = evaluate_llil(
            param,
            regs,
            memory,
            state,
            get_flag=get_flag,
            set_flag=set_flag,
        )
        if value is None:
            raise RuntimeError("VALIDATE_EXP_HIGH_NIBBLE input did not produce a value")
        values.append(value)
    if any(value & 0xF00000 for value in values):
        raise NotImplementedError(
            "SC62015 EXP high-nibble behavior requires real-hardware tracing"
        )
    return None, None


def eval_intrinsic_validate_i_count(
    llil: MockLLIL,
    size: Optional[int],
    regs: RegistersLike,
    memory: Memory,
    state: State,
    get_flag: FlagGetter,
    set_flag: FlagSetter,
) -> Tuple[None, Optional[ResultFlags]]:
    """Reject the hardware-unverified I=0 counted-instruction edge case."""

    from binja_test_mocks.eval_llil import evaluate_llil

    params = getattr(llil, "params", ())
    if len(params) != 1:
        raise RuntimeError("VALIDATE_I_COUNT requires exactly one word input")
    value, _flags = evaluate_llil(
        params[0], regs, memory, state, get_flag=get_flag, set_flag=set_flag
    )
    if value is None:
        raise RuntimeError("VALIDATE_I_COUNT input did not produce a value")
    if value & 0xFFFF == 0:
        raise NotImplementedError(
            "SC62015 I=0 counted-instruction semantics require real-hardware tracing"
        )
    return None, None


def eval_intrinsic_validate_vector_transfer(
    llil: MockLLIL,
    size: Optional[int],
    regs: RegistersLike,
    memory: Memory,
    state: State,
    get_flag: FlagGetter,
    set_flag: FlagSetter,
) -> Tuple[None, Optional[ResultFlags]]:
    """Reject an unverified vector or invalid target before any mutation."""

    from binja_test_mocks.eval_llil import evaluate_llil

    params = getattr(llil, "params", ())
    if len(params) != 3:
        raise RuntimeError(
            "VALIDATE_VECTOR_TRANSFER requires vector address, source PC, "
            "and the architectural vector fetch"
        )
    values: list[int] = []
    for param in params:
        value, _flags = evaluate_llil(
            param,
            regs,
            memory,
            state,
            get_flag=get_flag,
            set_flag=set_flag,
        )
        if value is None:
            raise RuntimeError("VALIDATE_VECTOR_TRANSFER input produced no value")
        values.append(int(value))

    # Runtime import avoids an instruction-module cycle during decoder setup.
    from .emulator import validate_vector_transfer

    validate_vector_transfer(
        memory,
        regs,  # type: ignore[arg-type]
        values[0],
        source_pc=values[1],
        actual_raw_vector=values[2],
    )
    return None, None


def eval_intrinsic_wait(
    llil: MockLLIL,
    size: Optional[int],
    regs: RegistersLike,
    memory: Memory,
    state: State,
    get_flag: FlagGetter,
    set_flag: FlagSetter,
) -> Tuple[None, Optional[ResultFlags]]:
    """Account for WAIT cycles, clear I, and preserve C/Z.

    I=0 is deliberately rejected because the 65,536-iteration interpretation
    remains unverified on real hardware.
    """
    wait_cycles = regs.get_by_name("I") & 0xFFFF
    if wait_cycles == 0:
        raise NotImplementedError(
            "SC62015 I=0 counted-instruction semantics require real-hardware tracing"
        )

    wait_hook = getattr(memory, "wait_cycles", None)
    if not callable(wait_hook):
        raise NotImplementedError(
            "WAIT requires a memory.wait_cycles timing hook; refusing to "
            "clear I without accounting for elapsed cycles"
        )

    wait_hook(wait_cycles)
    regs.set_by_name("I", 0)
    return None, None


def _enter_low_power_state(memory: Memory, state: State, mode: str) -> None:
    """Apply the current manual-derived HALT/OFF register model.

    ``State`` only standardises a ``halted`` boolean, so retain a lightweight
    mode tag as well.  Consumers must not silently treat OFF as HALT: their
    wake sources differ in the emulator model and still require hardware
    validation.
    """

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
    setattr(state, "power_state", mode)


def eval_intrinsic_tcl(
    llil: MockLLIL,
    size: Optional[int],
    regs: RegistersLike,
    memory: Memory,
    state: State,
    get_flag: FlagGetter,
    set_flag: FlagSetter,
) -> Tuple[None, Optional[ResultFlags]]:
    """Reject TCL until its timer-clear side effects are modeled.

    TCL conditionally resets the sub/main clock-generator timer phases based
    on ``LCC.STCL`` and ``LCC.MTCL``.  Treating it as a no-op fabricates timer
    behavior, so execution is deliberately quarantined pending a timer hook
    and real-hardware trace validation.
    """
    raise NotImplementedError(
        "TCL timer-clear side effects are not implemented; hardware trace required"
    )


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

    The register mutations below are the current manual-derived emulator
    contract, not hardware-trace evidence:
    - USR (F8H) bits 0 to 2/5 are reset to 0
    - SSR (FFH) bit 2 is set to 1
    - USR (F8H) bits 3 and 4 are set to 1
    """
    _enter_low_power_state(memory, state, "halted")
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

    The register mutations below are the current manual-derived emulator
    contract, not hardware-trace evidence:
    - USR (F8H) bits 0 to 2/5 are reset to 0
    - SSR (FFH) bit 2 is set to 1
    - USR (F8H) bits 3 and 4 are set to 1
    OFF is tagged separately from HALT so device runtimes cannot silently give
    it HALT's wake policy.  Exact clock and wake behavior needs hardware trace.
    """
    _enter_low_power_state(memory, state, "off")
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

    The reset-vector location is corroborated by the stock ROM vectors.  The
    remaining register writes/retention rules below are the current
    manual-derived emulator contract pending targeted hardware tracing:
    - LCC (FEH) bit 7 is reset to 0 (documented as ACM bit 7)
    - UCR (F7H) is reset to 0
    - USR (F8H) bits 0 to 2/5 are reset to 0
    - ISR (FCH) is reset to 0 (clears interrupt status)
    - SCR (FDH) is reset to 0
    - SSR (FFH) bit 2 is reset to 0
    - USR (F8H) bits 3 and 4 are set to 1
    - PC reads the reset vector at 0xFFFFD (3 bytes, little-endian)
    - Other registers retain their values
    - Flags (C/Z) are retained
    """
    # Validate the complete transfer before the first SFR write.  RESET can be
    # invoked either through lifted opcode FF or directly by power_on_reset(),
    # so this guard belongs in the evaluator itself.
    from .emulator import (
        _ValidatedVectorTransfer,
        fetch_validated_vector_transfer,
    )

    # Perform the architectural vector read before the first SFR mutation and
    # require it to match the side-effect-free preflight.  Reuse this value for
    # PC so a volatile or failing bus cannot change the destination after RESET
    # has partially committed.
    reset_source = regs.get_by_name("PC") & 0xFFFFF
    staged_transfer = getattr(state, "_sc62015_prepared_reset_transfer", None)
    prepared_transfer = None
    if staged_transfer is not None:
        prepared_transfer = staged_transfer
        reset_source = int(
            getattr(state, "_sc62015_prepared_reset_source_pc", reset_source)
        ) & 0xFFFFF
        # Remove the staged capability before consuming it.  An exception
        # cannot leave a retryable token attached to restored evaluator state.
        delattr(state, "_sc62015_prepared_reset_transfer")
        if hasattr(state, "_sc62015_prepared_reset_source_pc"):
            delattr(state, "_sc62015_prepared_reset_source_pc")
    if prepared_transfer is None:
        prepared_transfer = fetch_validated_vector_transfer(
            memory,
            regs,  # type: ignore[arg-type]
            0xFFFFD,
            source_pc=reset_source,
        )
    if not isinstance(prepared_transfer, _ValidatedVectorTransfer):
        raise RuntimeError("RESET requires an opaque validated vector transfer")
    target = prepared_transfer.consume(memory, 0xFFFFD, reset_source)

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

    regs.set_by_name("PC", target)
    state.halted = False
    setattr(state, "power_state", "running")

    return None, None


def register_sc62015_intrinsics() -> None:
    """Register all SC62015-specific intrinsic evaluators with the generic evaluation system."""
    # Import here to avoid circular imports
    from binja_test_mocks.eval_llil import register_intrinsic

    register_intrinsic("WAIT", eval_intrinsic_wait)
    register_intrinsic("TCL", eval_intrinsic_tcl)
    register_intrinsic("HALT", eval_intrinsic_halt)
    register_intrinsic("OFF", eval_intrinsic_off)
    register_intrinsic("RESET", eval_intrinsic_reset)
    register_intrinsic("VALIDATE_F", eval_intrinsic_validate_f)
    register_intrinsic(
        "VALIDATE_EXP_HIGH_NIBBLE", eval_intrinsic_validate_exp_high_nibble
    )
    register_intrinsic("VALIDATE_I_COUNT", eval_intrinsic_validate_i_count)
    register_intrinsic(
        "VALIDATE_VECTOR_TRANSFER", eval_intrinsic_validate_vector_transfer
    )
