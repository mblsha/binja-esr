from typing import Dict, Set, Optional, Any, cast, Tuple, Callable
import inspect
import enum
from dataclasses import dataclass
from binja_test_mocks.coding import FetchDecoder

try:
    from .cached_decoder import CachedFetchDecoder

    USE_CACHED_DECODER = True
except ImportError:
    USE_CACHED_DECODER = False
from .constants import (
    PC_MASK,
    ADDRESS_SPACE_SIZE,
    INTERNAL_MEMORY_START,
    MODELED_F_MASK,
    validate_f_image,
)

from .instr.opcode_table import OPCODES
from .instr.opcodes import ENTRY_POINT_ADDR, IMEMRegisters, INTERRUPT_VECTOR_ADDR
from .instr import (
    ADCL,
    DADL,
    DSBL,
    DSLL,
    DSRL,
    EXL,
    IR,
    MVL,
    MVLD,
    RESET,
    SBCL,
    decode,
    Instruction,
    InvalidInstruction,
    PRE,
    TCL,
    UnknownInstruction,
    WAIT,
)
from binja_test_mocks.mock_llil import (
    MockLowLevelILFunction,
    MockLLIL,
    MockLabel,
    MockIfExpr,
    MockGoto,
)
from binja_test_mocks.eval_llil import (
    Memory,
    State,
    ResultFlags,
    evaluate_llil,
)
from binaryninja import (  # type: ignore
    InstructionInfo,
)
from .intrinsics import register_sc62015_intrinsics


NUM_TEMP_REGISTERS = 16


CALL_STACK_EFFECTS = {
    0x04: 1,  # CALL mn
    0x05: 1,  # CALLF lmn
    0xFE: 1,  # IR - Interrupt entry
    0x06: -1,  # RET
    0x07: -1,  # RETF
    0x01: -1,  # RETI - Return from interrupt
}

I_COUNTED_INSTRUCTIONS = (ADCL, SBCL, DADL, DSBL, MVL, MVLD, EXL, DSLL, DSRL, WAIT)


def _decode_instruction_for_preflight(
    address: int, read_byte: Callable[[int], int]
) -> Instruction:
    """Decode through an explicitly side-effect-free program-memory reader."""

    fetched_opcode: int | None = None

    def fetch(offset: int) -> int:
        nonlocal fetched_opcode
        value = int(read_byte((address + offset) & PC_MASK)) & 0xFF
        if offset == 0:
            fetched_opcode = value
        return value

    if USE_CACHED_DECODER:
        decoder = CachedFetchDecoder(fetch, ADDRESS_SPACE_SIZE)
    else:
        decoder = FetchDecoder(fetch, ADDRESS_SPACE_SIZE)
    instr = decode(decoder, address & PC_MASK, OPCODES)  # type: ignore
    if instr is None or isinstance(instr, UnknownInstruction):
        opcode = 0 if fetched_opcode is None else fetched_opcode
        raise InvalidInstruction(
            f"Invalid, reserved, or truncated opcode 0x{opcode:02X} "
            f"at 0x{address & PC_MASK:05X}"
        )
    return cast(Instruction, instr)


def _vector_transfer_provenance(memory: Memory) -> tuple[int, int]:
    """Return the stable identity and mapping epoch for a prepared transfer."""

    provider = getattr(memory, "vector_transfer_provenance", None)
    if not callable(provider):
        return id(memory), 0
    provenance = provider()
    if (
        not isinstance(provenance, tuple)
        or len(provenance) != 2
        or isinstance(provenance[0], bool)
        or isinstance(provenance[1], bool)
        or not isinstance(provenance[0], int)
        or not isinstance(provenance[1], int)
    ):
        raise RuntimeError(
            "memory.vector_transfer_provenance() must return two integers"
        )
    return int(provenance[0]), int(provenance[1])


def _read_byte_with_pc(memory: Memory, address: int, source_pc: int) -> int:
    """Perform one architectural read using the strongest supported PC context.

    Callback arity is inspected before invocation.  Retrying after a TypeError
    would be unsafe because the callback body may already have consumed device
    state before raising.
    """

    reader = memory.read_byte
    try:
        signature = inspect.signature(reader)
    except (TypeError, ValueError):
        # Opaque/native callables follow the modern two-argument contract.
        return int(reader(address, source_pc)) & 0xFF  # type: ignore[call-arg]
    try:
        signature.bind(address, source_pc)
    except TypeError:
        try:
            signature.bind(address)
        except TypeError as exc:
            raise TypeError(
                "memory.read_byte must accept (address) or (address, cpu_pc)"
            ) from exc
        return int(reader(address)) & 0xFF
    return int(reader(address, source_pc)) & 0xFF  # type: ignore[call-arg]


def _write_byte_with_pc(
    memory: Memory, address: int, value: int, source_pc: int
) -> None:
    """Perform one architectural write without retrying an invoked callback."""

    writer = memory.write_byte
    try:
        signature = inspect.signature(writer)
    except (TypeError, ValueError):
        writer(address, value, source_pc)  # type: ignore[call-arg]
        return
    try:
        signature.bind(address, value, source_pc)
    except TypeError:
        try:
            signature.bind(address, value)
        except TypeError as exc:
            raise TypeError(
                "memory.write_byte must accept (address, value) or "
                "(address, value, cpu_pc)"
            ) from exc
        writer(address, value)
        return
    writer(address, value, source_pc)  # type: ignore[call-arg]


class _ValidatedVectorTransfer:
    """Private, one-shot proof binding a vector fetch to one memory mapping."""

    __slots__ = (
        "_memory",
        "_provenance",
        "_source_pc",
        "_scope",
        "_target",
        "_used",
        "_vector_address",
    )

    def __init__(
        self,
        memory: Memory,
        vector_address: int,
        source_pc: int,
        target: int,
        provenance: tuple[int, int],
        scope: str = "instruction",
    ) -> None:
        self._memory = memory
        self._provenance = provenance
        self._vector_address = int(vector_address)
        self._source_pc = int(source_pc) & PC_MASK
        self._scope = scope
        self._target = int(target) & PC_MASK
        self._used = False

    @property
    def target(self) -> int:
        return self._target

    def _binding_matches(
        self, memory: Memory, vector_address: int, source_pc: int
    ) -> bool:
        return (
            memory is self._memory
            and _vector_transfer_provenance(memory) == self._provenance
            and int(vector_address) == self._vector_address
            and (int(source_pc) & PC_MASK) == self._source_pc
        )

    def matches(self, memory: Memory, vector_address: int, source_pc: int) -> bool:
        return not self._used and self._binding_matches(
            memory, vector_address, source_pc
        )

    def invalidate(self) -> None:
        self._used = True

    def consume(
        self,
        memory: Memory,
        vector_address: int,
        source_pc: int,
    ) -> int:
        """Consume this proof before checking it, so failed reuse stays failed."""

        if self._used:
            raise RuntimeError("prepared SC62015 vector transfer was already consumed")
        self._used = True
        matches = self._binding_matches(memory, vector_address, source_pc)
        if not matches:
            raise RuntimeError(
                "prepared SC62015 vector transfer does not match the current "
                "instruction or memory mapping"
            )
        return self._target


def validate_vector_transfer_stability(
    memory: Memory,
    vector_address: int,
    target: int,
    *,
    require_immutable: bool = False,
    require_metadata: bool = True,
) -> None:
    stability_name = (
        "instruction_byte_is_immutable"
        if require_immutable
        else "instruction_byte_is_callback_free"
    )
    stability_check = getattr(memory, stability_name, None)
    if not callable(stability_check):
        if not require_immutable and not require_metadata:
            return
        requirement = "immutable" if require_immutable else "callback-free"
        raise RuntimeError(
            f"prepared SC62015 vector transfer requires {requirement} metadata"
        )
    vector_bytes = tuple((vector_address + index) & PC_MASK for index in range(3))

    def target_peek(address: int) -> int:
        peek_method = getattr(memory, "peek_byte_for_preflight")
        return int(peek_method(address & PC_MASK, target)) & 0xFF

    target_instruction = _decode_instruction_for_preflight(target, target_peek)
    target_info = InstructionInfo()
    target_instruction.analyze(target_info, target)
    target_length = int(target_info.length or target_instruction.length())
    target_bytes = tuple((target + index) & PC_MASK for index in range(target_length))
    unstable = next(
        (
            address
            for address in (*vector_bytes, *target_bytes)
            if not bool(stability_check(address))
        ),
        None,
    )
    if unstable is not None:
        requirement = "immutable" if require_immutable else "callback-free"
        raise RuntimeError(
            "SC62015 prepared vector transfer requires "
            f"{requirement} instruction bytes; 0x{unstable:05X} is dynamic"
        )


def fetch_validated_vector_transfer(
    memory: Memory,
    regs: "Registers",
    vector_address: int,
    *,
    source_pc: int | None = None,
    require_immutable: bool = False,
    scope: str = "instruction",
    require_stability_metadata: bool = False,
) -> _ValidatedVectorTransfer:
    """Validate, architecturally fetch once, and return an opaque proof."""

    if isinstance(vector_address, bool) or not 0 <= int(vector_address) <= PC_MASK:
        raise ValueError("SC62015 vector address must be canonical 20-bit")
    vector_address = int(vector_address)
    vector_pc = (
        regs.get(RegisterName.PC) if source_pc is None else int(source_pc)
    ) & PC_MASK
    provenance = _vector_transfer_provenance(memory)
    target = validate_vector_transfer(
        memory,
        regs,
        vector_address,
        source_pc=vector_pc,
    )
    validate_vector_transfer_stability(
        memory,
        vector_address,
        target,
        require_immutable=require_immutable,
        require_metadata=require_stability_metadata,
    )
    raw_vector = 0
    for byte_index in range(3):
        raw_vector |= _read_byte_with_pc(
            memory,
            (vector_address + byte_index) & PC_MASK,
            vector_pc,
        ) << (byte_index * 8)
    if raw_vector & 0xF00000:
        raise NotImplementedError(
            "SC62015 vector upper-nibble behavior is unverified; refusing "
            f"noncanonical architectural vector 0x{raw_vector:06X}"
        )
    if raw_vector != target:
        raise RuntimeError(
            "SC62015 architectural vector fetch disagrees with safe "
            f"preflight at 0x{vector_address & PC_MASK:05X}: "
            f"fetched 0x{raw_vector:06X}, preflight 0x{target:06X}"
        )
    if _vector_transfer_provenance(memory) != provenance:
        raise RuntimeError("SC62015 vector mapping changed during architectural fetch")
    return _ValidatedVectorTransfer(
        memory,
        vector_address,
        vector_pc,
        target,
        provenance,
        scope,
    )


def validate_vector_transfer(
    memory: Memory,
    regs: "Registers",
    vector_address: int,
    *,
    source_pc: int | None = None,
    actual_raw_vector: int | None = None,
) -> int:
    """Validate an indirect control transfer without observable bus reads.

    SC62015 vector slots contain three bytes, but the program counter is
    20-bit.  The behavior of a nonzero encoded upper nibble has not been
    established on hardware, so the emulator quarantines it instead of
    silently aliasing it.  The destination is also statically decoded before
    the caller is allowed to mutate a stack, reset register, or scheduler.
    """

    if isinstance(vector_address, bool) or not 0 <= int(vector_address) <= PC_MASK:
        raise ValueError("SC62015 vector address must be canonical 20-bit")
    vector_address = int(vector_address)
    peek_method = getattr(memory, "peek_byte_for_preflight", None)
    if not callable(peek_method):
        raise RuntimeError(
            "SC62015 vector-transfer preflight requires memory.peek_byte_for_preflight"
        )

    vector_pc = (
        regs.get(RegisterName.PC) if source_pc is None else int(source_pc)
    ) & PC_MASK

    def vector_peek(address: int) -> int:
        return int(peek_method(address & 0xFFFFFF, vector_pc)) & 0xFF

    raw_vector = 0
    for byte_index in range(3):
        raw_vector |= vector_peek((vector_address + byte_index) & PC_MASK) << (
            byte_index * 8
        )
    if raw_vector & 0xF00000:
        raise NotImplementedError(
            "SC62015 vector upper-nibble behavior is unverified; refusing "
            f"noncanonical vector 0x{raw_vector:06X} at "
            f"0x{vector_address & PC_MASK:05X}"
        )
    if actual_raw_vector is not None:
        if not 0 <= actual_raw_vector <= 0xFFFFFF:
            raise RuntimeError("architectural SC62015 vector is not a 24-bit value")
        if actual_raw_vector != raw_vector:
            raise RuntimeError(
                "SC62015 architectural vector fetch disagrees with safe "
                f"preflight at 0x{vector_address & PC_MASK:05X}: "
                f"fetched 0x{actual_raw_vector:06X}, "
                f"preflight 0x{raw_vector:06X}"
            )

    target = raw_vector & PC_MASK

    def target_peek(address: int) -> int:
        return int(peek_method(address & PC_MASK, target)) & 0xFF

    instr = _decode_instruction_for_preflight(target, target_peek)
    if isinstance(instr, PRE):
        raise InvalidInstruction(
            f"Unfused or malformed PRE instruction at 0x{target:05X}"
        )
    if isinstance(instr, TCL):
        raise NotImplementedError(
            "TCL timer-clear side effects are not implemented; hardware trace required"
        )
    return target


@dataclass
class InstructionEvalInfo:
    instruction_info: InstructionInfo
    instruction: Instruction


class RegisterName(enum.Enum):
    """CPU register names."""

    _ignore_ = ["_i"]

    # 8-bit
    A = "A"
    B = "B"
    IL = "IL"
    IH = "IH"
    # 16-bit
    I = "I"  # noqa: E741
    BA = "BA"
    # 20-bit architectural registers (stored/transferred in 3 bytes)
    X = "X"
    Y = "Y"
    U = "U"
    S = "S"
    # 20-bit (stored in 3 bytes, masked)
    PC = "PC"
    # Flags
    FC = "FC"  # Carry
    FZ = "FZ"  # Zero
    F = "F"  # byte-wide stack image; only C/Z bits are currently modeled
    # Temp registers
    #
    # These are generated dynamically so new temporary registers can
    # be added by simply adjusting ``NUM_TEMP_REGISTERS``. This keeps
    # the enum definition DRY and avoids repeating similar lines.
    for _i in range(NUM_TEMP_REGISTERS):
        locals()[f"TEMP{_i}"] = f"TEMP{_i}"
    del _i


REGISTER_SIZE: Dict[RegisterName, int] = {
    RegisterName.A: 1,  # 8-bit
    RegisterName.B: 1,  # 8-bit
    RegisterName.IL: 1,  # 8-bit
    RegisterName.IH: 1,  # 8-bit
    RegisterName.I: 2,  # 16-bit
    RegisterName.BA: 2,  # 16-bit
    RegisterName.X: 3,  # 3-byte storage, 20 significant bits
    RegisterName.Y: 3,  # 3-byte storage, 20 significant bits
    RegisterName.U: 3,  # 3-byte storage, 20 significant bits
    RegisterName.S: 3,  # 3-byte storage, 20 significant bits
    RegisterName.PC: 3,  # 20-bit (stored in 3 bytes)
    RegisterName.FC: 1,  # 1-bit
    RegisterName.FZ: 1,  # 1-bit
    RegisterName.F: 1,  # byte-wide stack image; modeled mask is C/Z (0x03)
    **{getattr(RegisterName, f"TEMP{i}"): 3 for i in range(NUM_TEMP_REGISTERS)},
}

# Mapping from generic flag names to architecture specific registers
FLAG_TO_REGISTER: Dict[str, RegisterName] = {
    "C": RegisterName.FC,
    "Z": RegisterName.FZ,
}


_LCD_LOOP_TRACE_ENABLED: bool = False
_LCD_LOOP_RANGE: tuple[int, int] | None = None
_LCD_LOOP_RANGE_DEFAULT: tuple[int, int] = (0x0F29A0, 0x0F2B00)
_LCD_LOOP_REGS = (
    RegisterName.PC,
    RegisterName.A,
    RegisterName.B,
    RegisterName.BA,
    RegisterName.I,
    RegisterName.X,
    RegisterName.Y,
    RegisterName.U,
    RegisterName.S,
)
_LCD_LOOP_FLAGS = ("C", "Z")
_LCD_TRACE_BP_ENABLED: bool = False

_STACK_SNAPSHOT_RANGE: tuple[int, int] | None = None
_STACK_SNAPSHOT_LEN: int | None = None


def _lcd_loop_range() -> tuple[int, int]:
    global _LCD_LOOP_RANGE
    if _LCD_LOOP_RANGE is not None:
        return _LCD_LOOP_RANGE
    _LCD_LOOP_RANGE = _LCD_LOOP_RANGE_DEFAULT
    return _LCD_LOOP_RANGE


def _should_trace_lcd(address: int) -> bool:
    if not _LCD_LOOP_TRACE_ENABLED:
        return False
    start, end = _lcd_loop_range()
    return start <= address <= end


def _log_lcd_loop_state(prefix: str, pc: int, regs: "Registers") -> None:
    reg_vals = " ".join(f"{reg.name}={regs.get(reg):06X}" for reg in _LCD_LOOP_REGS)
    flag_vals = " ".join(
        f"{name}={regs.get_flag(name):01X}" for name in _LCD_LOOP_FLAGS
    )
    print(f"[lcd-loop] {prefix} pc=0x{pc:06X} {reg_vals} flags={flag_vals}")


def _log_bp_bytes(prefix: str, pc: int, memory: Memory) -> None:
    if not _LCD_TRACE_BP_ENABLED:
        return
    try:
        bp = memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.BP) & 0xFF
    except Exception:
        return
    window = []
    for offset in (3, 4, 5):
        addr = INTERNAL_MEMORY_START + ((bp + offset) & 0xFF)
        try:
            window.append(memory.read_byte(addr) & 0xFF)
        except Exception:
            window.append(0)
    print(
        "[lcd-loop-bp] {prefix} pc=0x{pc:06X} BP=0x{bp:02X} "
        "bp+3=0x{bp3:02X} bp+4=0x{bp4:02X} bp+5=0x{bp5:02X}".format(
            prefix=prefix,
            pc=pc,
            bp=bp,
            bp3=window[0],
            bp4=window[1],
            bp5=window[2],
        )
    )


def _stack_snapshot_range() -> tuple[int, int] | None:
    return None


def _stack_snapshot_len() -> int:
    return 8


def _log_stack_snapshot(
    prefix: str, pc: int, regs: "Registers", memory: Memory
) -> None:
    rng = _stack_snapshot_range()
    if not rng:
        return
    start, end = rng
    if not (start <= pc <= end):
        return
    stack = regs.get(RegisterName.S)
    length = _stack_snapshot_len()
    bytes_ = [
        memory.read_byte((stack + offset) & (ADDRESS_SPACE_SIZE - 1))
        for offset in range(length)
    ]
    byte_str = " ".join(f"{b:02X}" for b in bytes_)
    reg_str = " ".join(
        f"{name.name}={regs.get(name):06X}"
        for name in (
            RegisterName.A,
            RegisterName.B,
            RegisterName.BA,
            RegisterName.X,
            RegisterName.Y,
            RegisterName.U,
            RegisterName.S,
        )
    )
    print(
        f"[stack-snapshot] backend={prefix} pc=0x{pc:06X} S=0x{stack:06X} bytes={byte_str} {reg_str}"
    )


class Registers:
    BASE: Set[RegisterName] = {
        RegisterName.BA,
        RegisterName.I,
        RegisterName.X,
        RegisterName.Y,
        RegisterName.U,
        RegisterName.S,
        RegisterName.PC,
        RegisterName.F,
    } | {getattr(RegisterName, f"TEMP{i}") for i in range(NUM_TEMP_REGISTERS)}

    _SUBREG_INFO: Dict[RegisterName, Tuple[RegisterName, int, int]] = {
        RegisterName.A: (RegisterName.BA, 0, 0xFF),
        RegisterName.B: (RegisterName.BA, 8, 0xFF),
        RegisterName.IL: (RegisterName.I, 0, 0xFF),
        RegisterName.IH: (RegisterName.I, 8, 0xFF),
        RegisterName.FC: (RegisterName.F, 0, 0x01),
        RegisterName.FZ: (RegisterName.F, 1, 0x01),
    }

    def __init__(self) -> None:
        self._values: Dict[RegisterName, int] = {reg: 0 for reg in self.BASE}
        # Call stack tracking for Perfetto tracing
        self.call_sub_level: int = 0

    def get(self, reg: RegisterName) -> int:
        if reg in self.BASE:
            val = self._values[reg]
            if reg is RegisterName.F:
                return val & MODELED_F_MASK
            if reg in (
                RegisterName.PC,
                RegisterName.X,
                RegisterName.Y,
                RegisterName.U,
                RegisterName.S,
            ):
                return val & PC_MASK
            return val

        info = self._SUBREG_INFO.get(reg)
        if info is not None:
            base, shift, mask = info
            return (self._values[base] >> shift) & mask

        raise ValueError(f"Attempted to get unknown or non-base register: {reg}")

    def set(self, reg: RegisterName, value: int) -> None:
        if reg in self.BASE:
            mask = (1 << (REGISTER_SIZE[reg] * 8)) - 1
            if reg is RegisterName.F:
                value = validate_f_image(value)
                mask = MODELED_F_MASK
            if reg in (
                RegisterName.PC,
                RegisterName.X,
                RegisterName.Y,
                RegisterName.U,
                RegisterName.S,
            ):
                mask = PC_MASK
            self._values[reg] = value & mask
            return

        info = self._SUBREG_INFO.get(reg)
        if info is not None:
            base, shift, mask = info
            if reg is RegisterName.IL:
                # Hardware behaviour: writing IL clears IH.
                self._values[base] = value & mask
                return
            full_mask = (1 << (REGISTER_SIZE[base] * 8)) - 1
            cur = self._values[base] & full_mask
            cur &= ~(mask << shift)
            cur |= (value & mask) << shift
            self._values[base] = cur
            return

        raise ValueError(f"Attempted to set unknown or non-base register: {reg}")

    def get_by_name(self, name: str) -> int:
        return self.get(RegisterName[name])

    def set_by_name(self, name: str, value: int) -> None:
        self.set(RegisterName[name], value)

    def get_flag(self, name: str) -> int:
        reg = FLAG_TO_REGISTER.get(name)
        if reg is None:
            raise ValueError(f"Unknown flag {name}")
        return self.get(reg)

    def set_flag(self, name: str, value: int) -> None:
        reg = FLAG_TO_REGISTER.get(name)
        if reg is None:
            raise ValueError(f"Unknown flag {name}")
        self.set(reg, value)


class Emulator:
    def __init__(self, memory: Memory, reset_on_init: bool = True) -> None:
        # Register SC62015-specific intrinsics with the evaluation system
        register_sc62015_intrinsics()

        self.regs = Registers()
        self.memory = memory
        self.state = State()
        setattr(self.state, "power_state", "running")
        self._poisoned: str | None = None
        self._execution_may_have_side_effects = False
        self._pending_vector_transfer: _ValidatedVectorTransfer | None = None
        self._pending_scheduled_opcode: tuple[int, int] | None = None

        # Track last PC for tracing
        self._last_pc: int = 0
        self._current_pc: int = 0
        self._perfetto_path: str | None = None

        # Perform power-on reset if requested
        if reset_on_init:
            self.power_on_reset()

    def set_perfetto_trace(self, path: str | None) -> None:
        if path is None:
            try:
                from pce500.tracing.perfetto_tracing import tracer as perfetto_tracer
            except Exception:
                return
            perfetto_tracer.stop()
            return
        try:
            from pce500.tracing.perfetto_tracing import tracer as perfetto_tracer
        except Exception:
            return
        perfetto_tracer.start(path)
        self._perfetto_path = path

    def flush_perfetto(self) -> None:
        try:
            from pce500.tracing.perfetto_tracing import tracer as perfetto_tracer
        except Exception:
            return
        perfetto_tracer.stop()

    def decode_instruction(self, address: int, read_fn=None) -> Instruction:
        # Allow an override fetch function (used for KIO tracing); default to memory.read_byte.
        fetched_opcode: int | None = None

        def fecher(offset: int) -> int:
            nonlocal fetched_opcode
            # Instruction fetch uses the same 20-bit program address bus as PC.
            # In particular, operands after an opcode at FFFFF continue at
            # 00000 rather than spilling into the synthetic IMEM window.
            addr = (address + offset) & PC_MASK
            if addr == INTERNAL_MEMORY_START + IMEMRegisters.KIL:
                pc_val = self.regs.get(RegisterName.PC)
                val = self.memory.read_byte(addr)
                try:
                    tracer = getattr(self.memory, "_perf_tracer", None)
                    if tracer is not None and hasattr(tracer, "instant"):
                        tracer.instant(
                            "KIO",
                            "read@KIL",
                            {
                                "pc": pc_val & PC_MASK
                                if isinstance(pc_val, int)
                                else None,
                                "offset": IMEMRegisters.KIL,
                                "value": val & 0xFF,
                            },
                        )
                except Exception:
                    pass
                try:
                    from pce500.tracing import trace_dispatcher

                    trace_dispatcher.record_instant(
                        "KIO",
                        "read@KIL",
                        {
                            "pc": f"0x{pc_val & PC_MASK:06X}"
                            if isinstance(pc_val, int)
                            else "N/A",
                            "offset": f"0x{IMEMRegisters.KIL:02X}",
                            "value": f"0x{val & 0xFF:02X}",
                        },
                    )
                except Exception:
                    pass
                return val
            # Generic IMEM read hook: always emit a KIO event for internal addresses.
            if addr >= INTERNAL_MEMORY_START:
                pc_val = self.regs.get(RegisterName.PC)
                val = self.memory.read_byte(addr)
                try:
                    tracer = getattr(self.memory, "_perf_tracer", None)
                    if tracer is not None and hasattr(tracer, "instant"):
                        tracer.instant(
                            "KIO",
                            f"read@0x{addr - INTERNAL_MEMORY_START:02X}",
                            {
                                "pc": pc_val & PC_MASK
                                if isinstance(pc_val, int)
                                else None,
                                "offset": addr - INTERNAL_MEMORY_START,
                                "value": val & 0xFF,
                            },
                        )
                except Exception:
                    pass
                return val
            if read_fn is not None:
                value = read_fn(addr)
            else:
                value = self.memory.read_byte(addr)
            if offset == 0:
                fetched_opcode = int(value) & 0xFF
            return value

        # Use cached decoder if available for better performance
        if USE_CACHED_DECODER:
            decoder = CachedFetchDecoder(fecher, ADDRESS_SPACE_SIZE)
        else:
            decoder = FetchDecoder(fecher, ADDRESS_SPACE_SIZE)
        instr = decode(decoder, address, OPCODES)  # type: ignore
        if instr is None or isinstance(instr, UnknownInstruction):
            opcode = fetched_opcode
            if opcode is None:
                opcode = (
                    read_fn(address & PC_MASK)
                    if read_fn is not None
                    else self.memory.read_byte(address & PC_MASK)
                ) & 0xFF
            raise InvalidInstruction(
                f"Invalid, reserved, or truncated opcode 0x{opcode:02X} "
                f"at 0x{address & PC_MASK:05X}"
            )
        return cast(Instruction, instr)

    def prepare_vector_transfer(
        self,
        vector_address: int,
        *,
        source_pc: int,
        require_immutable: bool = False,
        scope: str = "instruction",
    ) -> int:
        """Architecturally fetch one vector for the next matching operation."""

        if self._pending_vector_transfer is not None:
            raise RuntimeError("an SC62015 vector transfer is already prepared")
        transfer = fetch_validated_vector_transfer(
            self.memory,
            self.regs,
            vector_address,
            source_pc=source_pc,
            require_immutable=require_immutable,
            scope=scope,
            require_stability_metadata=True,
        )
        self._pending_vector_transfer = transfer
        return transfer.target

    def _take_prepared_vector_transfer(
        self, vector_address: int, source_pc: int, *, scope: str = "instruction"
    ) -> _ValidatedVectorTransfer | None:
        transfer = self._pending_vector_transfer
        self._pending_vector_transfer = None
        if transfer is None:
            return None
        if transfer._scope != scope:
            transfer.invalidate()
            return None
        if not transfer.matches(self.memory, vector_address, source_pc):
            transfer.consume(self.memory, vector_address, source_pc)
        return transfer

    def cancel_prepared_vector_transfer(self) -> None:
        transfer = self._pending_vector_transfer
        self._pending_vector_transfer = None
        if transfer is not None:
            transfer.invalidate()
        self._pending_scheduled_opcode = None

    def prepare_scheduled_opcode(self, address: int, opcode: int) -> None:
        if self._pending_scheduled_opcode is not None:
            raise RuntimeError("an SC62015 scheduled opcode is already prepared")
        self._pending_scheduled_opcode = (
            int(address) & PC_MASK,
            int(opcode) & 0xFF,
        )

    def execute_instruction(self, address: int) -> InstructionEvalInfo:
        if self._poisoned is not None:
            raise RuntimeError(
                "SC62015 CPU is poisoned after a failed side-effecting "
                f"instruction; reset required: {self._poisoned}"
            )

        register_snapshot = dict(self.regs._values)
        call_sub_level_snapshot = self.regs.call_sub_level
        state_snapshot = dict(vars(self.state))
        last_pc_snapshot = self._last_pc
        current_pc_snapshot = self._current_pc
        self._execution_may_have_side_effects = False
        pending_transfer = self._pending_vector_transfer
        self._pending_vector_transfer = None
        pending_opcode = self._pending_scheduled_opcode
        self._pending_scheduled_opcode = None

        # Check if performance tracing is available through memory context
        tracer = getattr(self.memory, "_perf_tracer", None)
        try:
            if tracer and hasattr(tracer, "slice"):
                with tracer.slice(
                    "Lifting", "execute_instruction", {"pc": f"0x{address:06X}"}
                ):
                    result = self._execute_instruction_impl(
                        address,
                        pending_transfer=pending_transfer,
                        pending_opcode=pending_opcode,
                    )
            else:
                result = self._execute_instruction_impl(
                    address,
                    pending_transfer=pending_transfer,
                    pending_opcode=pending_opcode,
                )
        except Exception as exc:
            may_have_external_side_effects = self._execution_may_have_side_effects
            self.regs._values.clear()
            self.regs._values.update(register_snapshot)
            self.regs.call_sub_level = call_sub_level_snapshot
            vars(self.state).clear()
            vars(self.state).update(state_snapshot)
            self._last_pc = last_pc_snapshot
            self._current_pc = current_pc_snapshot
            if may_have_external_side_effects:
                # Host memory/timer callbacks are not transactionally reversible.
                # Restore native CPU state, then require RESET before retrying so
                # a partially completed instruction cannot run twice.
                self._poisoned = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._execution_may_have_side_effects = False
            if pending_transfer is not None:
                pending_transfer.invalidate()
        return result

    def _execute_instruction_impl(
        self,
        address: int,
        read_fn: Optional[Callable[[int], int]] = None,
        *,
        pending_transfer: _ValidatedVectorTransfer | None = None,
        pending_opcode: tuple[int, int] | None = None,
    ) -> InstructionEvalInfo:
        # Track PC history for tracing
        pc_value = address & PC_MASK
        if _should_trace_lcd(pc_value):
            _log_lcd_loop_state("python", pc_value, self.regs)
            _log_bp_bytes("python", pc_value, self.memory)
        if _stack_snapshot_range():
            _log_stack_snapshot("python", pc_value, self.regs, self.memory)
        self._last_pc = self._current_pc
        self._current_pc = pc_value

        self.regs.set(RegisterName.PC, pc_value)
        if pending_opcode is not None:
            prepared_pc, prepared_byte = pending_opcode
            if prepared_pc != pc_value:
                raise RuntimeError(
                    "prepared SC62015 opcode does not match the current instruction"
                )
            opcode_requested = False

            def prepared_read(byte_address: int) -> int:
                nonlocal opcode_requested
                if (byte_address & PC_MASK) == pc_value:
                    opcode_requested = True
                    return prepared_byte
                return _read_byte_with_pc(self.memory, byte_address, pc_value)

            instr = self.decode_instruction(address, read_fn=prepared_read)
            if not opcode_requested:
                raise RuntimeError("SC62015 decoder did not consume prepared opcode")
        else:
            instr = self.decode_instruction(address, read_fn=read_fn)
        assert instr is not None, f"Failed to decode instruction at {address:04X}"

        # Silicon behavior at I=0 is not established for any counted
        # instruction (including WAIT).  Reject it before call-stack,
        # pointer, flag, memory, or timing side effects.  The lifted forms
        # carry the same validation intrinsic so direct LLIL evaluation
        # cannot bypass this dispatch guard.
        if (
            isinstance(instr, I_COUNTED_INSTRUCTIONS)
            and (self.regs.get(RegisterName.I) & 0xFFFF) == 0
        ):
            raise NotImplementedError(
                "SC62015 I=0 counted-instruction semantics require "
                "real-hardware tracing"
            )

        # The decoder already fetched and cached the instruction bytes. Re-read
        # of the opcode is observable on callback-backed buses and can consume
        # a device value twice, so use the decoded opcode identity directly.
        if instr.opcode is None:
            raise InvalidInstruction(
                f"Decoded instruction at 0x{address & PC_MASK:05X} has no opcode"
            )
        opcode = instr.opcode

        prepared_transfer: _ValidatedVectorTransfer | None = None
        if isinstance(instr, IR):
            if pending_transfer is not None:
                if pending_transfer._scope != "instruction":
                    pending_transfer.invalidate()
                    raise RuntimeError(
                        "prepared SC62015 vector transfer has the wrong scope"
                    )
                if not pending_transfer.matches(
                    self.memory, INTERRUPT_VECTOR_ADDR, address
                ):
                    pending_transfer.consume(
                        self.memory, INTERRUPT_VECTOR_ADDR, address
                    )
                prepared_transfer = pending_transfer
        elif isinstance(instr, RESET):
            if pending_transfer is not None:
                if pending_transfer._scope != "instruction":
                    pending_transfer.invalidate()
                    raise RuntimeError(
                        "prepared SC62015 vector transfer has the wrong scope"
                    )
                if not pending_transfer.matches(self.memory, ENTRY_POINT_ADDR, address):
                    pending_transfer.consume(self.memory, ENTRY_POINT_ADDR, address)
                prepared_transfer = pending_transfer
        elif pending_transfer is not None:
            pending_transfer.invalidate()
            raise RuntimeError(
                "prepared SC62015 vector transfer does not match the current instruction"
            )

        # TCL has timer-phase side effects controlled by LCC.STCL/MTCL.  Until
        # the peripheral hook is implemented and hardware-traced, stop before
        # advancing PC rather than silently executing it as a NOP.
        if isinstance(instr, TCL):
            raise NotImplementedError(
                "TCL timer-clear side effects are not implemented; "
                "hardware trace required"
            )

        # Vector-bearing synchronous transfers must fail before the generic
        # execution path is marked side-effecting.  Their lifted forms repeat
        # this validation for direct LLIL evaluators.
        if isinstance(instr, IR) and prepared_transfer is None:
            validate_vector_transfer(
                self.memory,
                self.regs,
                INTERRUPT_VECTOR_ADDR,
                source_pc=address,
            )
        elif isinstance(instr, RESET) and prepared_transfer is None:
            validate_vector_transfer(
                self.memory,
                self.regs,
                ENTRY_POINT_ADDR,
                source_pc=address,
            )

        # A prepared software IR must not execute the lifted architectural
        # vector load after the wrapper's timer tick.  Consume the exact
        # pre-tick fetch and perform the documented frame writes directly.
        if isinstance(instr, IR) and prepared_transfer is not None:
            target = prepared_transfer.consume(
                self.memory, INTERRUPT_VECTOR_ADDR, address
            )
            info = InstructionInfo()
            instr.analyze(info, address)
            self._execution_may_have_side_effects = True
            saved_pc = address & PC_MASK
            s = (self.regs.get(RegisterName.S) - 3) & PC_MASK
            self.regs.set(RegisterName.S, s)
            for byte_index in range(3):
                _write_byte_with_pc(
                    self.memory,
                    (s + byte_index) & PC_MASK,
                    (saved_pc >> (8 * byte_index)) & 0xFF,
                    saved_pc,
                )
            s = (self.regs.get(RegisterName.S) - 1) & PC_MASK
            self.regs.set(RegisterName.S, s)
            _write_byte_with_pc(
                self.memory,
                s,
                self.regs.get(RegisterName.F),
                saved_pc,
            )
            imr_address = INTERNAL_MEMORY_START + IMEMRegisters.IMR
            imr = _read_byte_with_pc(self.memory, imr_address, saved_pc)
            s = (self.regs.get(RegisterName.S) - 1) & PC_MASK
            self.regs.set(RegisterName.S, s)
            _write_byte_with_pc(self.memory, s, imr, saved_pc)
            _write_byte_with_pc(self.memory, imr_address, imr & 0x7F, saved_pc)
            self.regs.set(RegisterName.PC, target)
            self.regs.call_sub_level += 1
            return InstructionEvalInfo(instruction_info=info, instruction=instr)

        # Monitor specific opcodes for call stack tracking
        call_stack_delta = CALL_STACK_EFFECTS.get(opcode)
        if call_stack_delta is not None:
            new_level = self.regs.call_sub_level + call_stack_delta
            self.regs.call_sub_level = max(0, new_level)

        # Fast-path: optimize WAIT (opcode 0xEF) to avoid long LLIL loops.
        # Semantics: WAIT performs an idle loop, decrementing I until zero.
        if isinstance(instr, WAIT):
            wait_hook = getattr(self.memory, "wait_cycles", None)
            if not callable(wait_hook):
                raise NotImplementedError(
                    "WAIT requires a memory.wait_cycles timing hook; refusing "
                    "to advance PC or clear I without accounting for elapsed cycles"
                )

            # Build minimal instruction info/length via analyze, and set I to 0.
            il = MockLowLevelILFunction()
            info = InstructionInfo()
            instr.analyze(info, address)
            current_instr_length = cast(int, info.length)
            assert current_instr_length is not None, (
                "InstructionInfo.length was not set by analyze()"
            )
            wait_cycles = self.regs.get(RegisterName.I) & 0xFFFF
            self._execution_may_have_side_effects = True
            wait_hook(int(wait_cycles))
            # Advance PC only after timing has been accounted for (we return
            # early and skip the common PC update).
            self.regs.set(RegisterName.PC, address + current_instr_length)
            # Emulate loop effect: I decremented to 0 (flags unchanged).
            self.regs.set(RegisterName.I, 0)
            # Return without evaluating any LLIL
            return InstructionEvalInfo(instruction_info=info, instruction=instr)

        il = MockLowLevelILFunction()
        instr.lift(il, address)

        info = InstructionInfo()
        instr.analyze(info, address)

        # Type checker fix: Cast info.length to int.
        # Although type-hinted as int, type checker might not be able to prove it in all contexts.
        current_instr_length = cast(int, info.length)
        assert current_instr_length is not None, (
            "InstructionInfo.length was not set by analyze()"
        )
        self._execution_may_have_side_effects = True
        self.regs.set(RegisterName.PC, address + current_instr_length)

        if isinstance(instr, RESET) and prepared_transfer is not None:
            setattr(self.state, "_sc62015_prepared_reset_transfer", prepared_transfer)
            setattr(self.state, "_sc62015_prepared_reset_source_pc", address & PC_MASK)

        label_to_index: Dict[Any, int] = {}
        for idx, node in enumerate(il.ils):
            if isinstance(node, MockLabel):
                label_to_index[node.label] = idx

        pc_llil = 0
        while pc_llil < len(il.ils):
            node = il.ils[pc_llil]

            if isinstance(node, MockLabel):
                pc_llil += 1
                continue

            if isinstance(node, MockIfExpr):
                # Type checker fix: Ensure node.cond is MockLLIL for eval
                assert isinstance(node.cond, MockLLIL), (
                    "Condition for IF expression must be MockLLIL"
                )
                cond_val, _ = self.evaluate(node.cond)
                assert cond_val is not None, (
                    "Condition for IF expression evaluated to None"
                )
                target_label = node.t if cond_val else node.f
                assert target_label in label_to_index, f"Unknown label {target_label}"
                pc_llil = label_to_index[target_label]
                continue

            if isinstance(node, MockGoto):
                assert node.label in label_to_index, (
                    f"Unknown goto target label {node.label}"
                )
                pc_llil = label_to_index[node.label]
                continue

            assert isinstance(node, MockLLIL), f"Expected MockLLIL, got {type(node)}"
            self.evaluate(node)
            pc_llil += 1

        if hasattr(self.state, "_sc62015_prepared_reset_transfer"):
            # Successful RESET consumes and removes these in the intrinsic.
            # Reaching here with them intact indicates the lift did not invoke
            # RESET as promised; do not permit later reuse.
            transfer = getattr(self.state, "_sc62015_prepared_reset_transfer")
            delattr(self.state, "_sc62015_prepared_reset_transfer")
            delattr(self.state, "_sc62015_prepared_reset_source_pc")
            if isinstance(transfer, _ValidatedVectorTransfer):
                transfer.consume(self.memory, ENTRY_POINT_ADDR, address)
            raise RuntimeError("RESET lift did not consume its prepared vector")

        return InstructionEvalInfo(instruction_info=info, instruction=instr)

    def evaluate(self, llil: MockLLIL) -> Tuple[Optional[int], Optional[ResultFlags]]:
        if llil.bare_op() == "CALL":
            # CALL/CALLF lifters emit the exact wrapped SC62015 stack writes
            # explicitly, followed by LLIL_CALL so real Binary Ninja analysis
            # retains call/return semantics.  binja-test-mocks historically
            # gives LLIL_CALL an SC62015-specific implicit push; bypass that
            # test-only behavior here or the explicit frame would be written
            # twice.
            assert len(llil.ops) == 1 and isinstance(llil.ops[0], MockLLIL)
            target, _ = self.evaluate(llil.ops[0])
            assert isinstance(target, int)
            self.regs.set(RegisterName.PC, target)
            return None, None
        return evaluate_llil(
            llil,
            self.regs,
            self.memory,
            self.state,
            self.regs.get_flag,
            self.regs.set_flag,
        )

    def power_on_reset(
        self,
    ) -> None:
        """Perform power-on reset per SC62015 spec.

        This method calls the RESET intrinsic evaluator directly to avoid duplicating
        the reset logic. The RESET intrinsic performs all necessary operations:
        - LCC (FEH) bit 7 is reset to 0 (documented as ACM bit 7)
        - UCR (F7H) is reset to 0
        - USR (F8H) bits 0 to 2/5 are reset to 0, bits 3 and 4 are set to 1
        - ISR (FCH) is reset to 0 (clears interrupt status)
        - SCR (FDH) is reset to 0
        - SSR (FFH) bit 2 is reset to 0
        - PC reads the reset vector at 0xFFFFD (3 bytes, little-endian)
        - Other registers retain their values (initialized to 0)
        - Flags (C/Z) are retained (initialized to 0)
        """
        # Directly call the RESET intrinsic evaluator. Host writes are not
        # transactionally reversible, so retain a native-state snapshot and
        # keep the CPU poisoned unless the entire reset completes.
        from .intrinsics import eval_intrinsic_reset

        self._pending_scheduled_opcode = None

        # Keep vector-policy failures outside the side-effecting recovery
        # transaction.  They perform only explicit safe peeks and therefore do
        # not poison an otherwise inspectable CPU.
        prepared_transfer = self._take_prepared_vector_transfer(
            ENTRY_POINT_ADDR,
            self.regs.get(RegisterName.PC),
            scope="machine_reset",
        )
        if prepared_transfer is None:
            validate_vector_transfer(
                self.memory,
                self.regs,
                ENTRY_POINT_ADDR,
            )

        register_snapshot = dict(self.regs._values)
        call_sub_level_snapshot = self.regs.call_sub_level
        state_snapshot = dict(vars(self.state))
        original_poison = self._poisoned
        try:
            if prepared_transfer is None:
                prepared_transfer = fetch_validated_vector_transfer(
                    self.memory,
                    self.regs,
                    ENTRY_POINT_ADDR,
                )
            setattr(self.state, "_sc62015_prepared_reset_transfer", prepared_transfer)
            setattr(
                self.state,
                "_sc62015_prepared_reset_source_pc",
                self.regs.get(RegisterName.PC),
            )
            eval_intrinsic_reset(
                None,  # llil not needed
                None,  # size not needed
                self.regs,
                self.memory,
                self.state,
                self.regs.get_flag,
                self.regs.set_flag,
            )
        except Exception as exc:
            self.regs._values.clear()
            self.regs._values.update(register_snapshot)
            self.regs.call_sub_level = call_sub_level_snapshot
            vars(self.state).clear()
            vars(self.state).update(state_snapshot)
            # A recovery attempt must not erase the first side-effecting fault:
            # that is the only reliable description of what may already have
            # committed to the host. Record RESET itself only when it created
            # the poisoned state.
            if original_poison is None:
                self._poisoned = f"RESET failed: {type(exc).__name__}: {exc}"
            raise
        self._poisoned = None

        # Clear halted state (RESET doesn't set this, but power-on should clear it)
        self.state.halted = False
