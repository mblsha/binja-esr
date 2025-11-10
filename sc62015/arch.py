import os
from typing import List

from binaryninja import (
    Architecture,
    RegisterInfo,
    IntrinsicInfo,
    InstructionInfo,
    CallingConvention,
)
from binaryninja.enums import Endianness, FlagRole
from binaryninja.log import log_error

from .pysc62015.instr import decode, encode, OPCODES
from .pysc62015.instr.opcodes import InvalidInstruction
from binja_test_mocks.tokens import asm
from .decoding.dispatcher import CompatDispatcher
from .decoding.compat_il import emit_instruction as emit_compat_instruction
from .scil import from_decoded
from .scil.backend_llil import emit_llil as emit_scil_llil
from .scil.compat_builder import CompatLLILBuilder

try:
    from binja_test_mocks.mock_llil import MockLowLevelILFunction
except ImportError:
    MockLowLevelILFunction = None


class SC62015(Architecture):
    name = "SC62015"
    endianness = Endianness.LittleEndian
    address_size = 3
    default_int_size = 1

    # registers from page 32 of the book
    regs = {
        "BA": RegisterInfo("BA", 2),
        "A": RegisterInfo("BA", 1, 1),  # accumulator
        "B": RegisterInfo("BA", 1, 0),  # axiliary
        "I": RegisterInfo("I", 2),  # counter
        "IL": RegisterInfo("I", 1, 1),
        "IH": RegisterInfo("I", 1, 0),
        "X": RegisterInfo("X", 3),  # pointer
        "Y": RegisterInfo("Y", 3),  # pointer
        "U": RegisterInfo("U", 3),  # user stack
        "S": RegisterInfo("S", 3),  # system stack
        "PC": RegisterInfo("PC", 3),  # program counter
        "PS": RegisterInfo("PC", 1, 2),  # actually 4 bits, page segment
    }
    stack_pointer = "S"

    flags = [
        "Z",  # zero
        "C",  # carry
    ]
    flag_roles = {
        "Z": FlagRole.ZeroFlagRole,
        "C": FlagRole.CarryFlagRole,
    }
    flag_write_types = [
        "Z",
        "C",
        "CZ",
    ]
    flags_written_by_flag_write_type = {
        "Z": ["Z"],
        "C": ["C"],
        "CZ": ["Z", "C"],
    }

    intrinsics = {
        "TCL": IntrinsicInfo(inputs=[], outputs=[]),
        "HALT": IntrinsicInfo(inputs=[], outputs=[]),
        "OFF": IntrinsicInfo(inputs=[], outputs=[]),
        "RESET": IntrinsicInfo(inputs=[], outputs=[]),
    }

    def __init__(self) -> None:
        super().__init__()
        self._compat_dispatcher = CompatDispatcher()
        self._scil_shadow_enabled = bool(int(os.getenv("SC62015_SCIL_SHADOW", "0")))
        self._scil_allow = {
            "MV A,n",
            "JRZ ±n",
            "JP mn",
            "MV A,[lmn]",
        }

    def get_instruction_info(self, data, addr):
        try:
            if decoded := decode(data, addr, OPCODES):
                info = InstructionInfo()
                decoded.analyze(info, addr)
                return info
        except (AssertionError, InvalidInstruction):
            # Invalid instruction encoding, return None to mark as data
            return None
        except Exception as exc:
            log_error(f"SC62015.get_instruction_info() failed at {addr:#x}: {exc}")
            raise

    def get_instruction_text(self, data, addr):
        try:
            if decoded := decode(data, addr, OPCODES):
                encoded = data[: decoded.length()]
                recoded = encode(decoded, addr)
                if encoded != recoded:
                    # Roundtrip failed - this can happen with consecutive PRE instructions
                    # or other invalid encodings. Treat as data.
                    return None
                return asm(decoded.render()), decoded.length()
        except (AssertionError, InvalidInstruction):
            # Invalid instruction encoding, return None to mark as data
            return None
        except Exception as exc:
            log_error(f"SC62015.get_instruction_text() failed at {addr:#x}: {exc}")
            raise

    def get_instruction_low_level_il(self, data, addr, il):
        try:
            compat_result = self._compat_dispatcher.try_emit(bytes(data), addr, il)
            if compat_result is not None:
                length, decoded_instr = compat_result
                if decoded_instr is not None:
                    self._run_scil_shadow(decoded_instr, addr)
                return length

            if decoded := decode(data, addr, OPCODES):
                decoded.lift(il, addr)
                return decoded.length()
        except (AssertionError, InvalidInstruction):
            # Invalid instruction encoding, return None to mark as data
            return None
        except Exception as exc:
            log_error(
                f"SC62015.get_instruction_low_level_il() failed at {addr:#x}: {exc}"
            )
            raise

    def _run_scil_shadow(self, decoded_instr, addr: int) -> None:
        if not self._scil_shadow_enabled:
            return
        if MockLowLevelILFunction is None:
            return
        if decoded_instr.mnemonic not in self._scil_allow:
            return

        scil_instr, binder = from_decoded.build(decoded_instr)

        compat_il = MockLowLevelILFunction()
        emit_compat_instruction(decoded_instr, compat_il, addr)

        scil_il = MockLowLevelILFunction()
        emit_scil_llil(scil_il, scil_instr, binder, CompatLLILBuilder(scil_il), addr)

        if self._canonical_il(compat_il) != self._canonical_il(scil_il):
            raise AssertionError(
                f"SCIL mismatch for {decoded_instr.mnemonic} at {addr:#x}"
            )

    @staticmethod
    def _canonical_il(il_func: MockLowLevelILFunction) -> List[str]:
        out: List[str] = []
        for node in il_func.ils:
            if getattr(node, "op", "") == "LABEL":
                continue
            text = repr(node)
            text = text.replace("object at 0x", "object at 0x?")
            out.append(text)
        return out


class SC62015CallingConvention(CallingConvention):
    # caller_saved_regs = ["R7", "R6"]
    # int_arg_regs = ["R5", "R4", "R3", "R2", "R1", "R0"]
    int_return_reg = "A"
    # high_int_return_reg = "R1"
