from binaryninja import (
    Architecture,
    RegisterInfo,
    IntrinsicInfo,
    InstructionInfo,
    CallingConvention,
    Type,
)
from binaryninja.enums import Endianness, FlagRole
from binaryninja.log import log_error

from .pysc62015.instr import (
    OPCODES,
    PRE,
    UnknownInstruction,
    decode,
    encode,
)
from .pysc62015.instr.opcodes import InvalidInstruction
from binja_test_mocks.tokens import asm


class SC62015(Architecture):
    name = "SC62015"
    endianness = Endianness.LittleEndian
    address_size = 3
    default_int_size = 1

    # registers from page 32 of the book
    regs = {
        "BA": RegisterInfo("BA", 2),
        # Binary Ninja subregister offsets are little-endian: offset 0 is the LSB.
        # SC62015 docs define BA as (B:A) where A is the low byte and B is the high byte.
        "A": RegisterInfo("BA", 1, 0),  # accumulator (LSB of BA)
        "B": RegisterInfo("BA", 1, 1),  # auxiliary (MSB of BA)
        "I": RegisterInfo("I", 2),  # counter
        # I is (IH:IL) where IL is the low byte and IH is the high byte.
        "IL": RegisterInfo("I", 1, 0),  # LSB of I
        "IH": RegisterInfo("I", 1, 1),  # MSB of I
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
        "WAIT": IntrinsicInfo(inputs=[], outputs=[]),
        "TCL": IntrinsicInfo(inputs=[], outputs=[]),
        "HALT": IntrinsicInfo(inputs=[], outputs=[]),
        "OFF": IntrinsicInfo(inputs=[], outputs=[]),
        "RESET": IntrinsicInfo(inputs=[], outputs=[]),
        "VALIDATE_F": IntrinsicInfo(inputs=[Type.int(1, False)], outputs=[]),
        "VALIDATE_VECTOR_TRANSFER": IntrinsicInfo(
            inputs=[
                Type.int(3, False),
                Type.int(3, False),
                Type.int(3, False),
            ],
            outputs=[],
        ),
    }

    @staticmethod
    def _decode_canonical(data, addr):
        """Decode an executable instruction, rejecting reserved and partial forms."""

        decoded = decode(data, addr, OPCODES)
        if decoded is None or isinstance(decoded, (PRE, UnknownInstruction)):
            return None

        encoded = bytes(data[: decoded.length()])
        recoded = bytes(encode(decoded, addr))
        if encoded != recoded:
            # Every accepted raw alias (currently the documented ED/FD selector
            # aliases) is preserved by the decoder's operand objects. Any
            # mismatch here is therefore a malformed form or an implementation
            # defect, not permission to broaden an alias.
            return None
        return decoded

    def get_instruction_info(self, data, addr):
        try:
            if decoded := self._decode_canonical(data, addr):
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
            if decoded := self._decode_canonical(data, addr):
                return asm(decoded.render()), decoded.length()
        except (AssertionError, InvalidInstruction):
            # Invalid instruction encoding, return None to mark as data
            return None
        except Exception as exc:
            log_error(f"SC62015.get_instruction_text() failed at {addr:#x}: {exc}")
            raise

    def get_instruction_low_level_il(self, data, addr, il):
        try:
            if decoded := self._decode_canonical(data, addr):
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


class SC62015CallingConvention(CallingConvention):
    # caller_saved_regs = ["R7", "R6"]
    # int_arg_regs = ["R5", "R4", "R3", "R2", "R1", "R0"]
    int_return_reg = "A"
    # high_int_return_reg = "R1"
