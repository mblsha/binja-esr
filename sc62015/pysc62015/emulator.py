from typing import Dict, Set, Optional, Any, cast, Tuple, Callable
import enum
import os
from dataclasses import dataclass
from binja_test_mocks.coding import FetchDecoder

try:
    from .cached_decoder import CachedFetchDecoder

    USE_CACHED_DECODER = True
except ImportError:
    USE_CACHED_DECODER = False
from .constants import PC_MASK, ADDRESS_SPACE_SIZE, INTERNAL_MEMORY_START

from .instr.opcode_table import OPCODES
from .instr.opcodes import IMEMRegisters
from .instr import (
    decode,
    Instruction,
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


NUM_TEMP_REGISTERS = 14


CALL_STACK_EFFECTS = {
    0x04: 1,  # CALL mn
    0x05: 1,  # CALLF lmn
    0xFE: 1,  # IR - Interrupt entry
    0x06: -1,  # RET
    0x07: -1,  # RETF
    0x01: -1,  # RETI - Return from interrupt
}


@dataclass
class InstructionEvalInfo:
    instruction_info: InstructionInfo
    instruction: Instruction


class _FallbackInstruction:
    """Minimal stand-in when decode fails so execution can continue."""

    def __init__(self, opcode: int, length: int = 1) -> None:
        self._opcode = opcode & 0xFF
        self._length = max(1, length)

    def name(self) -> str:
        return f"UNK_{self._opcode:02X}"

    def length(self) -> int:
        return self._length

    def analyze(
        self, info: InstructionInfo, addr: int
    ) -> None:  # pragma: no cover - defensive
        info.length += self._length

    def lift(
        self, il: MockLowLevelILFunction, addr: int
    ) -> None:  # pragma: no cover - defensive
        il.append(il.nop())

    def render(self):
        return []


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
    # 24-bit (3 bytes)
    X = "X"
    Y = "Y"
    U = "U"
    S = "S"
    # 20-bit (stored in 3 bytes, masked)
    PC = "PC"
    # Flags
    FC = "FC"  # Carry
    FZ = "FZ"  # Zero
    F = "F"
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
    RegisterName.X: 3,  # 24-bit
    RegisterName.Y: 3,  # 24-bit
    RegisterName.U: 3,  # 24-bit
    RegisterName.S: 3,  # 24-bit
    RegisterName.PC: 3,  # 20-bit (stored in 3 bytes)
    RegisterName.FC: 1,  # 1-bit
    RegisterName.FZ: 1,  # 1-bit
    RegisterName.F: 1,  # 8-bit (general flags register)
    **{getattr(RegisterName, f"TEMP{i}"): 3 for i in range(NUM_TEMP_REGISTERS)},
}

# Mapping from generic flag names to architecture specific registers
FLAG_TO_REGISTER: Dict[str, RegisterName] = {
    "C": RegisterName.FC,
    "Z": RegisterName.FZ,
}


_LCD_LOOP_TRACE_ENABLED: bool = os.getenv("LCD_LOOP_TRACE") == "1"
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
_LCD_TRACE_BP_ENABLED: bool = os.getenv("LCD_TRACE_BP") == "1"

_STACK_SNAPSHOT_RANGE: tuple[int, int] | None = None
_STACK_SNAPSHOT_LEN: int | None = None


def _lcd_loop_range() -> tuple[int, int]:
    global _LCD_LOOP_RANGE
    if _LCD_LOOP_RANGE is not None:
        return _LCD_LOOP_RANGE
    env = os.getenv("LCD_LOOP_RANGE")
    if env:
        parts = env.strip().split("-")
        try:
            start = int(parts[0], 0)
        except ValueError:
            start = _LCD_LOOP_RANGE_DEFAULT[0]
        if len(parts) > 1:
            try:
                end = int(parts[1], 0)
            except ValueError:
                end = _LCD_LOOP_RANGE_DEFAULT[1]
        else:
            end = start
        if start > end:
            start, end = end, start
        _LCD_LOOP_RANGE = (start, end)
    else:
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
    global _STACK_SNAPSHOT_RANGE
    if _STACK_SNAPSHOT_RANGE is not None:
        return _STACK_SNAPSHOT_RANGE
    env = os.getenv("STACK_SNAPSHOT_RANGE")
    if not env:
        return None
    parts = env.strip().split("-")
    if not parts:
        return None
    try:
        start = int(parts[0], 0)
        end = int(parts[1], 0) if len(parts) > 1 else start
    except ValueError:
        return None
    if start > end:
        start, end = end, start
    _STACK_SNAPSHOT_RANGE = (start, end)
    return _STACK_SNAPSHOT_RANGE


def _stack_snapshot_len() -> int:
    global _STACK_SNAPSHOT_LEN
    if _STACK_SNAPSHOT_LEN is not None:
        return _STACK_SNAPSHOT_LEN
    length = 8
    env = os.getenv("STACK_SNAPSHOT_LEN")
    if env:
        try:
            candidate = int(env, 0)
            if candidate > 0:
                length = candidate
        except ValueError:
            pass
    _STACK_SNAPSHOT_LEN = length
    return length


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
            if reg is RegisterName.PC:
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
            if reg is RegisterName.PC:
                mask = PC_MASK
            self._values[reg] = value & mask
            return

        info = self._SUBREG_INFO.get(reg)
        if info is not None:
            base, shift, mask = info
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

        # Track last PC for tracing
        self._last_pc: int = 0
        self._current_pc: int = 0

        # Perform power-on reset if requested
        if reset_on_init:
            self.power_on_reset()

    def decode_instruction(self, address: int, read_fn=None) -> Instruction:
        # Allow an override fetch function (used for KIO tracing); default to memory.read_byte.
        def fecher(offset: int) -> int:
            addr = address + offset
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
                                "pc": pc_val & 0xFFFFFF
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
                            "pc": f"0x{pc_val & 0xFFFFFF:06X}"
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
                                "pc": pc_val & 0xFFFFFF
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
                return read_fn(addr)
            return self.memory.read_byte(addr)

        # Use cached decoder if available for better performance
        if USE_CACHED_DECODER:
            decoder = CachedFetchDecoder(fecher, ADDRESS_SPACE_SIZE)
        else:
            decoder = FetchDecoder(fecher, ADDRESS_SPACE_SIZE)
        instr = decode(decoder, address, OPCODES)  # type: ignore
        if instr is None:
            opcode = self.memory.read_byte(address) & 0xFF
            instr = _FallbackInstruction(opcode)
        return cast(Instruction, instr)

    def execute_instruction(self, address: int) -> InstructionEvalInfo:
        # Check if performance tracing is available through memory context
        tracer = getattr(self.memory, "_perf_tracer", None)
        if tracer and hasattr(tracer, "slice"):
            with tracer.slice(
                "Lifting", "execute_instruction", {"pc": f"0x{address:06X}"}
            ):
                return self._execute_instruction_impl(address)
        else:
            return self._execute_instruction_impl(address)

    def _execute_instruction_impl(
        self, address: int, read_fn: Optional[Callable[[int], int]] = None
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
        instr = self.decode_instruction(address)
        assert instr is not None, f"Failed to decode instruction at {address:04X}"

        # Provide a unified byte reader to honor any injected read_fn.
        def _read_byte_fn(addr: int) -> int:
            if read_fn is not None:
                return read_fn(addr)
            return self.memory.read_byte(addr)

        # Track call stack depth based on opcode
        opcode = _read_byte_fn(address)

        # Monitor specific opcodes for call stack tracking
        call_stack_delta = CALL_STACK_EFFECTS.get(opcode)
        if call_stack_delta is not None:
            new_level = self.regs.call_sub_level + call_stack_delta
            self.regs.call_sub_level = max(0, new_level)

        # Fast-path: optimize WAIT (opcode 0xEF) to avoid long LLIL loops
        # Semantics: WAIT performs an idle loop, decrementing I until zero.
        # This has no side effects other than I reaching 0, so we can skip
        # lifting/evaluating the loop and set I:=0 directly.
        if opcode == 0xEF:  # WAIT
            # Build minimal instruction info/length via analyze, and set I to 0
            il = MockLowLevelILFunction()
            info = InstructionInfo()
            instr.analyze(info, address)
            current_instr_length = cast(int, info.length)
            assert current_instr_length is not None, (
                "InstructionInfo.length was not set by analyze()"
            )
            # Advance PC (we return early and skip common PC update)
            self.regs.set(RegisterName.PC, address + current_instr_length)
            # Emulate loop effect: I decremented to 0
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
        self.regs.set(RegisterName.PC, address + current_instr_length)

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

        return InstructionEvalInfo(instruction_info=info, instruction=instr)

    def evaluate(self, llil: MockLLIL) -> Tuple[Optional[int], Optional[ResultFlags]]:
        return evaluate_llil(
            llil,
            self.regs,
            self.memory,
            self.state,
            self.regs.get_flag,
            self.regs.set_flag,
        )

    def power_on_reset(self) -> None:
        """Perform power-on reset per SC62015 spec.

        This method calls the RESET intrinsic evaluator directly to avoid duplicating
        the reset logic. The RESET intrinsic performs all necessary operations:
        - LCC (FEH) bit 7 is reset to 0 (documented as ACM bit 7)
        - UCR (F7H) is reset to 0
        - USR (F8H) bits 0 to 2/5 are reset to 0, bits 3 and 4 are set to 1
        - ISR (FCH) is reset to 0 (clears interrupt status)
        - SCR (FDH) is reset to 0
        - SSR (FFH) bit 2 is reset to 0
        - PC reads the reset vector at 0xFFFFA (3 bytes, little-endian)
        - Other registers retain their values (initialized to 0)
        - Flags (C/Z) are retained (initialized to 0)
        """
        # Directly call the RESET intrinsic evaluator
        from .intrinsics import eval_intrinsic_reset

        eval_intrinsic_reset(
            None,  # llil not needed
            None,  # size not needed
            self.regs,
            self.memory,
            self.state,
            self.regs.get_flag,
            self.regs.set_flag,
        )

        # Clear halted state (RESET doesn't set this, but power-on should clear it)
        self.state.halted = False
