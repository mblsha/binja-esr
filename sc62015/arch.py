import os
from collections import Counter

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
from .scil import from_decoded
from .scil.backend_llil import emit_llil as emit_scil_llil
from .scil.compat_builder import CompatLLILBuilder
from .pysc62015 import config as scil_config


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
        skip_bn = bool(int(os.getenv("SC62015_SKIP_BN_INIT", "0")))
        if not skip_bn:
            super().__init__()
        self._compat_dispatcher = CompatDispatcher()
        self._scil_config = scil_config.load_scil_config()
        self._scil_counters: Counter[str] = Counter()
        self._legacy_warning_emitted = False

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
            compat_result = self._compat_dispatcher.try_decode(bytes(data), addr)
            if compat_result is not None:
                length, decoded_instr = compat_result
                if decoded_instr is not None:
                    self._emit_scil(decoded_instr, addr, il)
                return length
            if self._scil_config.allow_legacy:
                return self._emit_legacy(bytes(data), addr, il)
        except (AssertionError, InvalidInstruction):
            # Invalid instruction encoding, return None to mark as data
            return None
        except Exception as exc:
            self._scil_counters["scil_error"] += 1
            if self._scil_config.allow_legacy:
                log_error(
                    f"SCIL emit failed at {addr:#x}, falling back to legacy: {exc}"
                )
                return self._emit_legacy(bytes(data), addr, il)
            log_error(f"SC62015.get_instruction_low_level_il() failed at {addr:#x}: {exc}")
            raise

    def get_scil_counters(self) -> dict[str, int]:
        return dict(self._scil_counters)

    def _emit_scil(self, decoded_instr, addr: int, il) -> None:
        payload = from_decoded.build(decoded_instr)
        emit_scil_llil(
            il,
            payload.instr,
            payload.binder,
            CompatLLILBuilder(il),
            addr,
            pre_applied=payload.pre_applied,
        )
        self._scil_counters["scil_ok"] += 1

    def _emit_legacy(self, data: bytes, addr: int, il, exc: Exception | None = None):
        if not self._legacy_warning_emitted:
            log_error("BN_ALLOW_LEGACY=1 set: using deprecated legacy lifter path")
            self._legacy_warning_emitted = True
        if decoded := decode(data, addr, OPCODES):
            decoded.lift(il, addr)
            self._scil_counters["legacy_rescue"] += 1
            return decoded.length()
        if exc:
            raise exc
        return None


class SC62015CallingConvention(CallingConvention):
    # caller_saved_regs = ["R7", "R6"]
    # int_arg_regs = ["R5", "R4", "R3", "R2", "R1", "R0"]
    int_return_reg = "A"
    # high_int_return_reg = "R1"
