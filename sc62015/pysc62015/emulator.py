from typing import Dict, Set, Optional, Any, cast, Tuple
import enum
from dataclasses import dataclass
from binja_test_mocks.coding import FetchDecoder

try:
    from .cached_decoder import CachedFetchDecoder

    USE_CACHED_DECODER = True
except ImportError:
    USE_CACHED_DECODER = False
from .constants import PC_MASK, ADDRESS_SPACE_SIZE

from .instr.opcode_table import OPCODES
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

    def decode_instruction(self, address: int) -> Instruction:
        def fecher(offset: int) -> int:
            return self.memory.read_byte(address + offset)

        # Use cached decoder if available for better performance
        if USE_CACHED_DECODER:
            decoder = CachedFetchDecoder(fecher, ADDRESS_SPACE_SIZE)
        else:
            decoder = FetchDecoder(fecher, ADDRESS_SPACE_SIZE)
        return decode(decoder, address, OPCODES)  # type: ignore

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

    def _execute_instruction_impl(self, address: int) -> InstructionEvalInfo:
        # Track PC history for tracing
        self._last_pc = self._current_pc
        self._current_pc = address

        self.regs.set(RegisterName.PC, address)
        instr = self.decode_instruction(address)
        assert instr is not None, f"Failed to decode instruction at {address:04X}"

        # Track call stack depth based on opcode
        opcode = self.memory.read_byte(address)

        # Monitor specific opcodes for call stack tracking
        if opcode == 0x04:  # CALL mn
            self.regs.call_sub_level += 1
        elif opcode == 0x05:  # CALLF lmn
            self.regs.call_sub_level += 1
        elif opcode == 0xFE:  # IR - Interrupt entry
            self.regs.call_sub_level += 1
        elif opcode == 0x06:  # RET
            self.regs.call_sub_level = max(0, self.regs.call_sub_level - 1)
        elif opcode == 0x07:  # RETF
            self.regs.call_sub_level = max(0, self.regs.call_sub_level - 1)
        elif opcode == 0x01:  # RETI - Return from interrupt
            self.regs.call_sub_level = max(0, self.regs.call_sub_level - 1)

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
