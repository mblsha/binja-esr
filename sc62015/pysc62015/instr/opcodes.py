# based on https://github.com/whitequark/binja-avnera/blob/main/mc/instr.py
from binja_test_mocks.tokens import (
    Token,
    TInstr,
    TText,
    TSep,
    TInt,
    TReg,
    TBegMem,
    TEndMem,
    TAddr,
    MemType,
)
from binja_test_mocks.coding import (
    Decoder,
    Encoder,
    BufferTooShortErrorError as BufferTooShort,
)

from binja_test_mocks.mock_llil import MockLLIL
from ..constants import INTERNAL_MEMORY_START
from .traits import HasWidth

import copy
from dataclasses import dataclass
from typing import (
    Optional,
    List,
    Generator,
    Iterator,
    Dict,
    Tuple,
    Union,
    Type,
    Literal,
    Any,
)
import enum
from enum import IntEnum
from contextlib import contextmanager


from binja_test_mocks import binja_api  # noqa: F401  # pyright: ignore
from binaryninja import (  # type: ignore
    InstructionInfo,
)
from binaryninja import (  # type: ignore
    RegisterName,
    IntrinsicName,
    FlagName,
)
from binaryninja.lowlevelil import (  # type: ignore
    LowLevelILFunction,
)
from binaryninja.lowlevelil import (  # type: ignore
    LowLevelILLabel,
    LLIL_TEMP,
    ExpressionIndex,
)


class InvalidInstruction(Exception):
    pass


# This table defines the hexadecimal value of the PRE (Prefix) byte required for
# certain complex internal RAM addressing modes, based on the combination of
# addressing calculations used for the first and second address components.
#
# Rows indicate the addressing mode calculation specified by the first operand
# byte (e.g., in MV (m) (n), this corresponds to (m)).
#
# Columns indicate the addressing mode calculation specified by the second
# operand byte (e.g., in MV (m) (n), this corresponds to (n)).
#
# +---------------+-------+--------+--------+--------+
# | 1st op \ 2nd  | (n)   | (BP+n) | (PY+n) |(BP+PY) |
# +---------------+-------+--------+--------+--------+
# | (n)           | 32H   | 30H    | 33H    | 31H    |
# +---------------+-------+--------+--------+--------+
# | (BP+n)        | 22H   |        | 23H    | 21H    |
# +---------------+-------+--------+--------+--------+
# | (PX+n)        | 36H   | 34H    | 37H    | 35H    |
# +---------------+-------+--------+--------+--------+
# | (BP+PX)       | 26H   | 24H    | 27H    | 25H    |
# +---------------+-------+--------+--------+--------+


class AddressingMode(enum.Enum):
    N = "(n)"
    BP_N = "(BP+n)"
    PX_N = "(PX+n)"
    PY_N = "(PY+n)"
    BP_PX = "(BP+PX)"
    BP_PY = "(BP+PY)"


PRE_TABLE = {
    1: {  # 1st operand (row, left)
        0x32: AddressingMode.N,
        0x30: AddressingMode.N,
        0x33: AddressingMode.N,
        0x31: AddressingMode.N,
        0x22: AddressingMode.BP_N,
        0x23: AddressingMode.BP_N,
        0x21: AddressingMode.BP_N,
        0x36: AddressingMode.PX_N,
        0x34: AddressingMode.PX_N,
        0x37: AddressingMode.PX_N,
        0x35: AddressingMode.PX_N,
        0x26: AddressingMode.BP_PX,
        0x24: AddressingMode.BP_PX,
        0x27: AddressingMode.BP_PX,
        0x25: AddressingMode.BP_PX,
    },
    2: {  # 2nd operand (column, top)
        0x32: AddressingMode.N,
        0x22: AddressingMode.N,
        0x36: AddressingMode.N,
        0x26: AddressingMode.N,
        0x30: AddressingMode.BP_N,
        0x34: AddressingMode.BP_N,
        0x24: AddressingMode.BP_N,
        0x33: AddressingMode.PY_N,
        0x23: AddressingMode.PY_N,
        0x37: AddressingMode.PY_N,
        0x27: AddressingMode.PY_N,
        0x31: AddressingMode.BP_PY,
        0x21: AddressingMode.BP_PY,
        0x35: AddressingMode.BP_PY,
        0x25: AddressingMode.BP_PY,
    },
}

REVERSE_PRE_TABLE: Dict[Tuple[AddressingMode, AddressingMode], int] = {
    # 1st Op \ 2nd Op: (n)
    (AddressingMode.N, AddressingMode.N): 0x32,
    (AddressingMode.BP_N, AddressingMode.N): 0x22,
    (AddressingMode.PX_N, AddressingMode.N): 0x36,
    (AddressingMode.BP_PX, AddressingMode.N): 0x26,
    # 1st Op \ 2nd Op: (BP+n)
    (AddressingMode.N, AddressingMode.BP_N): 0x30,
    (AddressingMode.PX_N, AddressingMode.BP_N): 0x34,
    (AddressingMode.BP_PX, AddressingMode.BP_N): 0x24,
    # 1st Op \ 2nd Op: (PY+n)
    (AddressingMode.N, AddressingMode.PY_N): 0x33,
    (AddressingMode.BP_N, AddressingMode.PY_N): 0x23,
    (AddressingMode.PX_N, AddressingMode.PY_N): 0x37,
    (AddressingMode.BP_PX, AddressingMode.PY_N): 0x27,
    # 1st Op \ 2nd Op: (BP+PY)
    (AddressingMode.N, AddressingMode.BP_PY): 0x31,
    (AddressingMode.BP_N, AddressingMode.BP_PY): 0x21,
    (AddressingMode.PX_N, AddressingMode.BP_PY): 0x35,
}

# Lookup table for instructions that operate on a single internal
# memory operand requiring a PRE prefix.  The mapping is based on the
# addressing mode used for that operand.  Simple `(n)` addressing does
# not require a prefix and therefore isn't included here.
SINGLE_OPERAND_PRE_LOOKUP: Dict[AddressingMode, int] = {
    AddressingMode.BP_N: 0x22,
    AddressingMode.PX_N: 0x36,
    AddressingMode.PY_N: 0x33,
    AddressingMode.BP_PX: 0x26,
    AddressingMode.BP_PY: 0x31,
}


def get_addressing_mode(pre_value: Optional[int], operand_index: int) -> AddressingMode:
    """
    Returns the addressing mode for the given PRE byte and operand index (1 or 2).
    If pre_value is None, returns BP_N as the default addressing mode.
    """
    if pre_value is None:
        return AddressingMode.BP_N
    try:
        return PRE_TABLE[operand_index][pre_value]
    except KeyError:
        raise ValueError(
            f"Unknown PRE value {pre_value:02X}H for operand index {operand_index}"
        )


TCLIntrinsic = IntrinsicName("TCL")
HALTIntrinsic = IntrinsicName("HALT")
OFFIntrinsic = IntrinsicName("OFF")
RESETIntrinsic = IntrinsicName("RESET")

# Use distinct temporary registers for various operations in order to avoid
# overlap in case of multiple operations being performed in the same instruction.
TempRegF = LLIL_TEMP(0)
TempIncDecHelper = LLIL_TEMP(1)
TempMvlSrc = LLIL_TEMP(2)
TempMvlDst = LLIL_TEMP(3)
TempMultiByte1 = LLIL_TEMP(4)
TempMultiByte2 = LLIL_TEMP(5)
TempExchange = LLIL_TEMP(6)
TempBcdAddEmul = LLIL_TEMP(7)
TempBcdSubEmul = LLIL_TEMP(8)
TempBcdLowNibbleProcessing = LLIL_TEMP(9)
TempBcdHighNibbleProcessing = LLIL_TEMP(10)
TempOverallZeroAcc = LLIL_TEMP(11)
TempLoopByteResult = LLIL_TEMP(12)
TempBcdDigitCarry = LLIL_TEMP(13)

# Single addressable operand opcodes - these should only use PRE1
SINGLE_ADDRESSABLE_OPCODES = set(
    [
        0x10,
        0x41,
        0x42,
        0x43,
        0x47,
        0x49,
        0x4A,
        0x4B,
        0x51,
        0x52,
        0x53,
        0x55,
        0x57,
        0x59,
        0x5A,
        0x5B,
        0x5D,
        0x61,
        0x62,
        0x63,
        0x65,
        0x66,
        0x67,
        0x69,
        0x6A,
        0x6B,
        0x6D,
        0x6F,
        0x71,
        0x72,
        0x73,
        0x77,
        0x79,
        0x7A,
        0x7B,
        0x7D,
        0x7F,
        0x80,
        0x81,
        0x82,
        0x83,
        0x84,
        0x85,
        0x86,
        0x87,
        0x88,
        0x89,
        0x8A,
        0x8B,
        0x8C,
        0x8D,
        0x8E,
        0x8F,
        0x98,
        0x99,
        0x9A,
        0x9B,
        0x9C,
        0x9D,
        0x9E,
        0xA0,
        0xA1,
        0xA2,
        0xA3,
        0xA4,
        0xA5,
        0xA6,
        0xA7,
        0xA8,
        0xA9,
        0xAA,
        0xAB,
        0xAC,
        0xAD,
        0xAE,
        0xAF,
        0xB8,
        0xB9,
        0xBA,
        0xBB,
        0xBC,
        0xBD,
        0xBE,
        0xC5,
        0xCC,
        0xCD,
        0xD5,
        0xD6,
        0xD7,
        0xDC,
        0xE3,
        0xE5,
        0xE7,
        0xEB,
        0xEC,
        0xF5,
        0xF7,
        0xFC,
    ]
)


# mapping to size, page 67 of the book
REGISTERS = [
    # r1
    (RegisterName("A"), 1),
    (RegisterName("IL"), 1),
    # r2
    (RegisterName("BA"), 2),
    (RegisterName("I"), 2),
    # r3
    (RegisterName("X"), 4),  # r4, actually 3 bytes
    (RegisterName("Y"), 4),  # r4, actually 3 bytes
    (RegisterName("U"), 4),  # r4, actually 3 bytes
    (RegisterName("S"), 3),
]

CFlag = FlagName("C")
ZFlag = FlagName("Z")
CZFlag = FlagName("CZ")

REG_NAMES = [reg[0] for reg in REGISTERS]
REG_SIZES = {reg[0]: min(3, reg[1]) for reg in REGISTERS}


INTERRUPT_VECTOR_ADDR = 0xFFFFA
ENTRY_POINT_ADDR = 0xFFFFD

# Hitachi LCD Driver
SH26_ADDR_START = 0x00000
SH26_ADDR_END = 0x3FFFF

# TENRI LCD Segment Driver
LH5073A1_ADDR_START = 0x40000
LH5073A1_ADDR_END = 0x7FFFF

CE1_ADDR_START = 0x80000
CE1_ADDR_END = 0x9FFFF
CE0_ADDR_START = 0xA0000
CE0_ADDR_END = 0xBFFFF

# Map internal RAM to start immediately after the 1MB external space. The
# internal region occupies addresses
#   [INTERNAL_MEMORY_START, ADDRESS_SPACE_SIZE - 1].


class IMEMRegisters(IntEnum):
    """Internal Memory-mapped registers for SC62015.

    Using IntEnum provides type safety and autocomplete while still
    allowing the values to be used directly as integers.
    """

    # RAM Pointers
    BP = 0xEC  # RAM Base Pointer
    PX = 0xED  # RAM PX Pointer
    PY = 0xEE  # RAM PY Pointer

    # A system with two RAM card slots may have two discontinuous
    # physical address windows (CE1 and CE0).  This register lets
    # you virtually join them into one contiguous block when enabled.
    #
    # When AME (bit 7) = 1:
    #   - The end of the CE1 window is linked to the start of the
    #     CE0 window in the software's virtual address space.
    #
    # Bitfields:
    #   AME     (bit 7)    = 1 to enable address‐modify
    #   AM5–AM0 (bits 6–1) = CE0 RAM size code:
    #     000000 =   2 KB
    #     000001 =   4 KB
    #     000011 =   8 KB
    #     000111 =  16 KB
    #     001111 =  32 KB
    #     011111 =  64 KB
    #     111111 = 128 KB
    #
    # Notes:
    #   • Virtual CE1 region follows directly after CE1's physical
    #     end.
    #   • Virtual CE0 region begins at CE0's physical base.
    AMC = 0xEF  # ADR Modify Control

    # Key I/O ports
    # Controls KO0-KO15 output pins
    KOL = 0xF0  # Key Output Buffer H
    KOH = 0xF1  # Key Output Buffer L

    # Controls KI0-KI7 input pins
    KIL = 0xF2  # Key Input Buffer

    # E Port I/O
    # Controls E0-E15 pins
    EOL = 0xF3  # E Port Output Buffer H
    EOH = 0xF4  # E Port Output Buffer L
    # Controls E0-E15 pins
    EIL = 0xF5  # E Port Input Buffer H
    EIH = 0xF6  # E Port Input Buffer L

    #     7     6     5     4     3     2     1     0
    #   +-----+-----+-----+-----+-----+-----+-----+-----+
    #   | BOE | BR2 | BR1 | BR0 | PA1 | PA0 |  DL |  ST |
    #   +-----+-----+-----+-----+-----+-----+-----+-----+
    #
    #  BOE  (bit 7)  – Break Output Enable.
    #                  When '1', TXD is driven low ("0") continuously.
    #
    #  Baud Rate Factor (bits 6–4 = BR2,BR1,BR0):
    #    000 → 0    (resets UART)
    #    001 → 300  bps
    #    010 → 600  bps
    #    011 → 1200 bps
    #    100 → 2400 bps
    #    101 → 4800 bps
    #    110 → 9600 bps
    #    111 → 19200 bps
    #
    #  Parity Select (bits 3–2 = PA1,PA0):
    #    00 → EVEN
    #    01 → ODD
    #    1x → NONE
    #
    #  Character Length (bit 1 = DL):
    #    0 →  8-bit data
    #    1 →  7-bit data
    #
    #  Stop Bits (bit 0 = ST):
    #    0 → 1 stop bit
    #    1 → 2 stop bits
    UCR = 0xF7  # UART Control Register

    #     7     6     5     4     3     2     1     0
    #   +-----+-----+-----+-----+-----+-----+-----+-----+
    #   |     |     | RXR | TXE | TXR |  FE |  OE |  PE |
    #   +-----+-----+-----+-----+-----+-----+-----+-----+
    #
    #  RXR (bit 5) – Receiver Ready:
    #     '1' when a character has been fully received;
    #     clears to '0' once RX buffer is read.
    #
    #  TXE (bit 4) – Transmitter Empty:
    #     '0' while UART is shifting bits out;
    #     '1' when transmitter is idle.
    #
    #  TXR (bit 3) – Transmitter Ready:
    #     '0' immediately after software writes TXD;
    #     becomes '1' once data has moved into the shift register.
    #
    #  FE  (bit 2) – Framing Error:
    #     '0' if stop-bit framing was incorrect; '1' otherwise.
    #     Updated on each receive completion.
    #
    #  OE  (bit 1) – Overrun Error:
    #     '1' if new character completes while RXR='1'.
    #     Updated on each receive completion.
    #
    #  PE  (bit 0) – Parity Error:
    #     '1' if received parity does not match.
    #     Updated on each receive completion.
    USR = 0xF8  # UART Status Register

    # Holds the 8-bit data of the last received character.
    RXD = 0xF9  # UART Receive Buffer

    # – Write data here for transmission.
    # – When TXE (USR[4]) goes '1', the byte moves to the transmitter.
    # – You may queue a new byte even while prior is sending;
    #   TXR (USR[3]) tells you when it's been accepted.
    TXD = 0xFA  # UART Transmit Buffer

    #    7     6     5      4      3      2     1     0
    #  +-----+-----+------+-------+------+-----+-----+-----+
    #  | IRM | EXM | RXRM | TXRM  | ONKM | KEYM| STM | MTM |
    #  +-----+-----+------+-------+------+-----+-----+-----+
    #
    # IRM  (bit 7) – Global interrupt mask:
    #    Write '0' to disable all sources.
    #
    # EXM  (bit 6) – External Interrupt Mask.
    # RXRM (bit 5) – Receiver Ready Interrupt Mask.
    # TXRM (bit 4) – Transmitter Ready Interrupt Mask.
    # ONKM (bit 3) – On-Key Interrupt Mask.
    # KEYM (bit 2) – Key Interrupt Mask.
    # STM  (bit 1) – SEC Timer Interrupt Mask.
    # MTM  (bit 0) – MSEC Timer Interrupt Mask.
    #
    # Writing '0' to any bit inhibits that individual interrupt source.
    # On interrupt entry, the current IMR is pushed to system/user stack
    # and IRM (bit 7) is cleared.
    IMR = 0xFB  # Interrupt Mask Register

    #     7    6     5     4      3      2     1     0
    #   +----+-----+-----+------+-------+-----+-----+-----+
    #   |    | EXI | RXRI| TXRI | ONKI  | KEYI| STI | MTI |
    #   +----+-----+-----+------+-------+-----+-----+-----+
    #
    #  Bit 7  – Reserved.
    #  EXI    (bit 6) – External Interrupt:
    #        '1' when an IRQ request arrives on the external pin.
    #  RXRI   (bit 5) – Receiver Ready Interrupt:
    #        '1' when UART has completed receiving one character.
    #  TXRI   (bit 4) – Transmitter Ready Interrupt:
    #        '1' when TX buffer (FAH) is ready for new data.
    #  ONKI   (bit 3) – On-Key Interrupt:
    #        '1' when a high level is input to the ON pin.
    #  KEYI   (bit 2) – Key Interrupt:
    #        '1' if any configured KI pin goes high.
    #  STI    (bit 1) – SEC Timer Interrupt:
    #        '1' when the sub-CG timer requests an interrupt.
    #  MTI    (bit 0) – MSEC Timer Interrupt:
    #        '1' when the main CG timer requests an interrupt.
    ISR = 0xFC  # Interrupt Status Register

    #     7    6    5    4    3    2    1     0
    #   +----+----+----+----+-----+----+----+-----+
    #   | ISE| BZ2| BZ1| BZ0| VDDC| STS| MTS| DISC|
    #   +----+----+----+----+-----+----+----+-----+
    #
    #  ISE   (bit 7) – IRQ Start Enable:
    #               '1' allows an external IRQ to resume the CPU from HALT/OFF.
    #
    #  BZ2–BZ0 (bits 6–4) – CO/CI pin Control Factors:
    #     000: CO=low,    CI=0 (input disallowed)
    #     001: CO=high,   CI=0 (input disallowed)
    #     010: CO=2 kHz,  CI=0 (input disallowed)
    #     011: CO=4 kHz,  CI=0 (input disallowed)
    #     100: CO=low,      CI=0/1 (input allowed)
    #     101: CO=high,     CI=0/1 (input allowed)
    #     11x: CO=CI level, CI=0/1 (input allowed)
    #
    #  VDDC  (bit 3) – VDD Control:
    #               0 = low (VCC),  1 = high (GND).
    #
    #  STS   (bit 2) – SEC Timer Select:
    #               0 = longer sub-CG interval, 1 = shorter.
    #               Change must occur just after STI=1 or after TCL.
    #
    #  MTS   (bit 1) – MSEC Timer Select:
    #               0 = shorter main CG interval, 1 = longer.
    #               Change must occur just after MTI=1 or after TCL.
    #
    #  DISC  (bit 0) – LCD Driver Control:
    #               0 = DIS pin low → display OFF;
    #               1 = DIS pin high → display ON.
    #               To synchronize: set DISC=1, wait >1 cycle, set DISC=0.
    SCR = 0xFD  # System Control Register

    #     7     6    5    4    3    2    1     0
    #   +----+----+----+----+----+----+-----+------+
    #   |LCC4|LCC3|LCC2|LCC1|LCC0| KSD| STCL| MTCL |
    #   +----+----+----+----+----+----+-----+------+
    #
    #  LCC4–LCC0 (bits 7–3) – Contrast level (0–31):
    #     00000 = min … 11111 = max
    #
    #  KSD    (bit 2) – Key Strobe Disable:
    #               '1' forces KO pins low; key outputs can be read.
    #
    #  STCL   (bit 1) – SEC Timer Clear:
    #               If '1' when TCL executes, resets sub-CG timer.
    #
    #  MTCL   (bit 0) – MSEC Timer Clear:
    #               If '1' when TCL executes, resets main CG timer.
    LCC = 0xFE  # LCD Contrast Control

    #     7    6    5    4    3    2    1     0
    #   +----+----+----+----+----+----+----+------+
    #   |    |    |    |    | ONK| RSF| CI | TEST |
    #   +----+----+----+----+----+----+----+------+
    #
    #  Bits 7–4 – Reserved.
    #
    #  ONK   (bit 3) – ON-Key input:
    #               '0' when ON pin is low, '1' when high.
    #
    #  RSF   (bit 2) – Reset-Start Flag:
    #               '0' when RESET pin is high, '1' when HALT/OFF.
    #
    #  CI    (bit 1) – CMT Input:
    #               '0' when CI pin is low, '1' when high.
    #
    #  TEST  (bit 0) – Test Input:
    #               '0' when TEST pin is low, '1' when high.
    SSR = 0xFF  # System Status Control


class Operand:
    def render(self, pre: Optional[AddressingMode] = None) -> List[Token]:
        return [TText("unimplemented")]

    def decode(self, decoder: Decoder, addr: int) -> None:
        pass

    def encode(self, encoder: Encoder, addr: int) -> None:
        pass

    # expand physical-encoding of operands into virtual printable operands
    def operands(self) -> Generator["Operand", None, None]:
        yield self

    def lift(
        self,
        il: LowLevelILFunction,
        pre: Optional[AddressingMode] = None,
        side_effects: bool = True,
    ) -> ExpressionIndex:
        return il.unimplemented()

    def lift_assign(
        self,
        il: LowLevelILFunction,
        value: ExpressionIndex,
        pre: Optional[AddressingMode] = None,
    ) -> None:
        il.append(value)
        il.append(il.unimplemented())


# used by Operands to help render / lift values
class OperandHelper(Operand):
    def render(self, pre: Optional[AddressingMode] = None) -> List[Token]:
        raise NotImplementedError(
            f"render() not implemented for {self.__class__.__name__} helper"
        )


@dataclass
class Opts:
    # useful when logical operands order is different from physical opcode encoding order
    ops_reversed: Optional[bool] = None
    # for conditional instructions
    cond: Optional[str] = None
    # override name
    name: Optional[str] = None
    # ops is short for operands
    ops: Optional[List[Operand]] = None


def iter_encode(instrs: List["Instruction"], addr: int) -> bytearray:
    encoder = Encoder()
    for instr in instrs:
        instr.encode(encoder, addr)
        addr += instr.length()
    return encoder.buf


def encode(instr: "Instruction", addr: int) -> bytearray:
    return iter_encode([instr], addr)


InstrOptsType = Tuple[Type["Instruction"], Opts]
OpcodesType = Union[Type["Instruction"], InstrOptsType]


def create_instruction(
    decoder: Decoder, opcodes: Dict[int, OpcodesType]
) -> Optional["Instruction"]:
    if decoder is None:
        return None

    opcode = decoder.peek(0)
    if opcode not in opcodes:
        return None

    definition = opcodes[opcode]
    cls, opts = definition if isinstance(definition, tuple) else (definition, Opts())

    name = opts.name or cls.__name__.split("_")[0]
    # since the operands are values and not constructors, we need to copy them
    ops = [copy.deepcopy(op) for op in (opts.ops or [])]
    return cls(name, operands=ops, cond=opts.cond, ops_reversed=opts.ops_reversed)


def iter_decode(
    decoder: Decoder, addr: int, opcodes: Dict[int, OpcodesType]
) -> Iterator[Tuple["Instruction", int]]:
    while True:
        try:
            instr = create_instruction(decoder, opcodes)
            if instr is None:
                raise NotImplementedError(
                    f"Cannot decode opcode at address {addr + decoder.pos:#06x}"
                )
            start_pos = decoder.get_pos()
            opcode = decoder.peek(0)
            instr.decode(decoder, addr)
            instr.set_length(decoder.get_pos() - start_pos)
            yield instr, addr
            addr += instr.length()
        except BufferTooShort:
            break
        except InvalidInstruction:
            break
        except AssertionError as e:
            raise AssertionError(
                f"Assertion failed while decoding opcode {opcode:02X} "
                f"at address {addr:#06x}: {e}"
            ) from e


def fusion(
    instr_iter: Iterator[Tuple["Instruction", int]],
) -> Iterator[Tuple["Instruction", int]]:
    try:
        instr1, addr1 = next(instr_iter)
    except StopIteration:
        return
    while True:
        try:
            instr2, addr2 = next(instr_iter)
        except (StopIteration, NotImplementedError):
            yield instr1, addr1
            break

        if instr12 := instr1.fuse(instr2):
            yield instr12, addr1
            try:
                instr1, addr1 = next(instr_iter)
            except (StopIteration, NotImplementedError):
                break
        else:
            yield instr1, addr1
            instr1, addr1 = instr2, addr2


def _create_decoder(
    decoder: Decoder, addr: int, opcodes: Dict[int, OpcodesType]
) -> Iterator[Tuple["Instruction", int]]:
    # TODO: Investigate why double fusion is needed for PRE instructions
    # For now, keeping double fusion but need to optimize memory reads elsewhere
    return fusion(fusion(iter_decode(decoder, addr, opcodes)))


def decode(
    decoder: Decoder | bytes | bytearray,
    addr: int,
    opcodes: Dict[int, OpcodesType],
) -> Optional["Instruction"]:
    """Decode one instruction from ``decoder``.

    ``decoder`` may be either an existing :class:`Decoder` instance or raw
    bytes.  The Binary Ninja Architecture API supplies raw bytes to the
    ``get_instruction_*`` hooks, so supporting that here avoids an
    ``AttributeError`` when running under the real application.
    """

    if not isinstance(decoder, Decoder):
        decoder = Decoder(bytearray(decoder))

    try:
        instr, _ = next(_create_decoder(decoder, addr, opcodes))

        return instr
    except StopIteration:
        return None
    # except NotImplementedError as e:
    #     binaryninja.log_warn(e)


class Instruction:
    opcode: Optional[int]
    _length: Optional[int]
    _pre: Optional[int] = None

    def __init__(
        self,
        name: str,
        operands: List[Operand],
        cond: Optional[str],
        ops_reversed: Optional[bool],
    ) -> None:
        self.instr_name = name
        self.ops_reversed = ops_reversed
        self._operands = operands
        self._cond = cond

    def length(self) -> int:
        assert self._length is not None, "Length not set"
        return self._length

    def name(self) -> str:
        return self.instr_name

    def decode(self, decoder: Decoder, addr: int) -> None:
        self.opcode = decoder.unsigned_byte()
        for op in self.operands_coding():
            op.decode(decoder, addr)
            # Set width for operands that support it based on instruction name
            set_width_fn = getattr(op, "set_width_from_instruction", None)
            if callable(set_width_fn):
                set_width_fn(self)

    def set_length(self, length: int) -> None:
        self._length = length

    def encode(self, encoder: Encoder, addr: int) -> None:
        assert self.opcode is not None, "Opcode not set"
        if self._pre is not None:
            encoder.unsigned_byte(self._pre)
        encoder.unsigned_byte(self.opcode)
        for op in self.operands_coding():
            op.encode(encoder, addr)

    def fuse(self, sister: "Instruction") -> Optional["Instruction"]:
        return None

    # logical operands order
    def operands(self) -> Generator[Operand, None, None]:
        if self._operands is None:
            return

        def _expand(op: Operand) -> Generator[Operand, None, None]:
            for sub in op.operands():
                if sub is op:
                    yield sub
                else:
                    yield from _expand(sub)

        for operand in self._operands:
            yield from _expand(operand)

    # physical opcode encoding order
    def operands_coding(self) -> Iterator[Operand]:
        if not self.ops_reversed:
            return iter(self._operands)
        # self.operands() is a generator
        # so we need to convert it to a list
        ops = list(self._operands)
        assert len(ops) == 2, "Expected 2 operands"
        return reversed(ops)

    def render(self) -> List[Token]:
        dst_mode = get_addressing_mode(self._pre, 1)
        src_mode = get_addressing_mode(self._pre, 2)

        # For single addressable operand instructions, always use PRE1
        if self.opcode in SINGLE_ADDRESSABLE_OPCODES:
            src_mode = dst_mode

        tokens: List[Token] = [TInstr(self.name())]
        if len(self._operands) > 0:
            tokens.append(TSep(" " * (6 - len(self.name()))))

        for index, operand in enumerate(self.operands()):
            if index > 0:
                tokens.append(TSep(", "))
            assert index < 2, "Expected up to 2 operands"
            mode = dst_mode if index == 0 else src_mode
            tokens += operand.render(mode)
        return tokens

    def analyze(self, info: InstructionInfo, addr: int) -> None:
        info.length += self.length()

    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        dst_mode = get_addressing_mode(self._pre, 1)
        src_mode = get_addressing_mode(self._pre, 2)

        # For single addressable operand instructions, always use PRE1
        if self.opcode in SINGLE_ADDRESSABLE_OPCODES:
            src_mode = dst_mode

        operands = tuple(self.operands())
        if len(operands) == 0:
            il.append(il.unimplemented())
        else:
            # For destination operand, disable side effects on first lift() to avoid double increment
            op1 = operands[0].lift(il, dst_mode, side_effects=False)
            if len(operands) == 1:
                il_value = self.lift_operation1(il, op1)
            elif len(operands) == 2:
                op2 = operands[1].lift(il, src_mode)
                il_value = self.lift_operation2(il, op1, op2)
            else:
                raise NotImplementedError("lift() not implemented for this instruction")
            operands[0].lift_assign(il, il_value, dst_mode)

    def lift_operation1(
        self, il: LowLevelILFunction, arg1: ExpressionIndex
    ) -> ExpressionIndex:
        raise NotImplementedError(
            f"lift_operation1() not implemented for {self.__class__.__name__} instruction"
        )
        return il.unimplemented()

    def lift_operation2(
        self, il: LowLevelILFunction, arg1: ExpressionIndex, arg2: ExpressionIndex
    ) -> ExpressionIndex:
        raise NotImplementedError(
            f"lift_operation2() not implemented for {self.__class__.__name__} instruction"
        )
        return il.unimplemented()


# HasOperands is used to indicate that the operand expects other operands to be
# used instead.
class HasOperands:
    def lift(
        self,
        il: LowLevelILFunction,
        pre: Optional[AddressingMode] = None,
        side_effects: bool = True,
    ) -> ExpressionIndex:
        raise NotImplementedError("lift not implemented for HasOperands")

    def lift_assign(
        self,
        il: LowLevelILFunction,
        value: ExpressionIndex,
        pre: Optional[AddressingMode] = None,
    ) -> None:
        raise NotImplementedError("lift_assign not implemented for HasOperands")


class IMemOperand(Operand, HasWidth):
    def __init__(self, mode: AddressingMode, n: Optional[int] = None):
        self.mode = mode
        self.n_val = n
        self.helper = IMemHelper(width=1, value=self)

    def __repr__(self) -> str:
        return f"IMemOperand(mode={self.mode}, n={self.n_val})"

    def width(self) -> int:
        return 1

    def render(self, pre: Optional[AddressingMode] = None) -> List[Token]:
        # The 'pre' argument is ignored here because the operand itself
        # already knows its addressing mode.
        return self.helper.render(pre=self.mode)

    def encode(self, encoder: Encoder, addr: int) -> None:
        # The 'n' value is encoded only if the mode requires it.
        if self.mode in [
            AddressingMode.N,
            AddressingMode.BP_N,
            AddressingMode.PX_N,
            AddressingMode.PY_N,
        ]:
            assert isinstance(self.n_val, int)
            encoder.unsigned_byte(self.n_val)

    def lift(
        self,
        il: LowLevelILFunction,
        pre: Optional[AddressingMode] = None,
        side_effects: bool = True,
    ) -> ExpressionIndex:
        return self.helper.lift(il, self.mode, side_effects)

    def lift_assign(
        self,
        il: LowLevelILFunction,
        value: ExpressionIndex,
        pre: Optional[AddressingMode] = None,
    ) -> None:
        self.helper.lift_assign(il, value, self.mode)


class ImmOperand(Operand, HasWidth):
    value: Optional[int]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def lift(
        self,
        il: LowLevelILFunction,
        pre: Optional[AddressingMode] = None,
        side_effects: bool = True,
    ) -> ExpressionIndex:
        assert self.value is not None, "Value not set"
        return il.const(self.width(), self.value)


# n: encoded as `n`
class Imm8(ImmOperand):
    def __init__(self, value: Optional[int] = None) -> None:
        super().__init__()
        self.value = value

    def width(self) -> int:
        return 1

    def decode(self, decoder: Decoder, addr: int) -> None:
        self.value = decoder.unsigned_byte()

    def encode(self, encoder: Encoder, addr: int) -> None:
        assert self.value is not None, "Value not set"
        encoder.unsigned_byte(self.value)

    def render(self, pre: Optional[AddressingMode] = None) -> List[Token]:
        return [TInt(f"{self.value:02X}")]


# mn: encoded as `n m`
class Imm16(ImmOperand):
    def __init__(self) -> None:
        super().__init__()
        self.value = None

    def width(self) -> int:
        return 2

    def decode(self, decoder: Decoder, addr: int) -> None:
        self.value = decoder.unsigned_word_le()

    def encode(self, encoder: Encoder, addr: int) -> None:
        assert self.value is not None, "Value not set"
        encoder.unsigned_word_le(self.value)

    def render(self, pre: Optional[AddressingMode] = None) -> List[Token]:
        return [TInt(f"{self.value:04X}")]


# lmn: encoded as `n m l`
class Imm20(ImmOperand):
    extra_hi: Optional[int]

    def __init__(self) -> None:
        super().__init__()
        self.value = None
        self.extra_hi = None

    def width(self) -> int:
        return 3

    def decode(self, decoder: Decoder, addr: int) -> None:
        lo = decoder.unsigned_byte()
        mid = decoder.unsigned_byte()
        self.extra_hi = decoder.unsigned_byte()
        hi = self.extra_hi & 0x0F
        self.value = lo | (mid << 8) | (hi << 16)

    def encode(self, encoder: Encoder, addr: int) -> None:
        assert self.value is not None, "Value not set"
        assert self.extra_hi is not None, "Extra high byte not set"
        encoder.unsigned_byte(self.value & 0xFF)
        encoder.unsigned_byte((self.value >> 8) & 0xFF)
        encoder.unsigned_byte(self.extra_hi)

    def render(self, pre: Optional[AddressingMode] = None) -> List[Token]:
        return [TInt(f"{self.value:05X}")]


# Offset sign is encoded as part of the instruction opcode, and the actual
# offset is Imm8.
class ImmOffset(Imm8):
    def __init__(self, sign: Literal["+", "-"]) -> None:
        super().__init__()
        self.sign = sign

    def offset_value(self) -> int:
        assert self.value is not None, "Value not set"
        return -self.value if self.sign == "-" else self.value

    def render(self, pre: Optional[AddressingMode] = None) -> List[Token]:
        return [TInt(f"{self.sign}{self.value:02X}")]

    def lift(
        self,
        il: LowLevelILFunction,
        pre: Optional[AddressingMode] = None,
        side_effects: bool = True,
    ) -> ExpressionIndex:
        raise NotImplementedError("lift() not implemented for ImmOffset")

    def lift_offset(
        self, il: LowLevelILFunction, value: ExpressionIndex
    ) -> ExpressionIndex:
        # Determine the width of the value we're adding to
        # For external memory addresses, this should be 3 bytes (20-bit)
        width = 3  # External memory addresses are 20-bit
        offset = il.const(width, self.offset_value())
        return il.add(width, value, offset)


# Utility mixin for operands that support optional +/- byte offsets based on
# their addressing mode.  Several operand types share the same logic for
# parsing/encoding these offsets, so centralize it here.
class OffsetOperandMixin:
    offset: Optional[ImmOffset] = None

    def _decode_offset(self, decoder: Decoder, addr: int) -> None:
        mode = getattr(self, "mode", None)
        if mode is None:
            return
        mode_enum = type(mode)
        positive = getattr(mode_enum, "POSITIVE_OFFSET", None)
        negative = getattr(mode_enum, "NEGATIVE_OFFSET", None)
        if mode in (positive, negative):
            sign_lit: Literal["+", "-"] = "+" if mode == positive else "-"
            self.offset = ImmOffset(sign_lit)
            self.offset.decode(decoder, addr)

    def _encode_offset(self, encoder: Encoder, addr: int) -> None:
        if self.offset:
            self.offset.encode(encoder, addr)


# Internal Memory Addressing Modes:
# 1. Direct
# 2. BP-indexed
# 3. PX/PY-indexed
# 4. BP-indexed with PX/PY offset
class IMemHelper(Operand):
    def __init__(self, width: int, value: Operand) -> None:
        super().__init__()
        self._width = width
        self.value = value

    def width(self) -> int:
        return self._width

    def render(self, pre: Optional[AddressingMode] = None) -> List[Token]:
        # Convert None to BP_N for consistent behavior
        if pre is None:
            pre = AddressingMode.BP_N

        result: List[Token] = [TBegMem(MemType.INTERNAL)]
        match pre:
            case AddressingMode.N:
                result.extend(self.value.render())
            case AddressingMode.BP_N:
                result.append(TText("BP"))
                result.append(TSep("+"))
                result.extend(self.value.render())
            case AddressingMode.PX_N:
                result.append(TText("PX"))
                result.append(TSep("+"))
                result.extend(self.value.render())
            case AddressingMode.PY_N:
                result.append(TText("PY"))
                result.append(TSep("+"))
                result.extend(self.value.render())
            case AddressingMode.BP_PX:
                result.append(TText("BP"))
                result.append(TSep("+"))
                result.append(TText("PX"))
            case AddressingMode.BP_PY:
                result.append(TText("BP"))
                result.append(TSep("+"))
                result.append(TText("PY"))
            case _:
                raise NotImplementedError(f"Unknown addressing mode {pre}")
        result.append(TEndMem(MemType.INTERNAL))
        return result

    # they're not real registers, but we treat them as such
    @staticmethod
    def _reg_value(name: str, il: LowLevelILFunction) -> ExpressionIndex:
        # Use the IntEnum for dynamic lookup
        addr = IMEMRegisters[name]
        return il.load(1, il.const_pointer(3, INTERNAL_MEMORY_START + addr))

    def _imem_offset(
        self, il: LowLevelILFunction, pre: Optional[AddressingMode]
    ) -> ExpressionIndex:
        # Convert None to BP_N for consistent behavior
        if pre is None:
            pre = AddressingMode.BP_N

        n_val: int = 0
        if isinstance(self.value, ImmOperand):
            if self.value.value is not None:
                n_val = self.value.value
        elif isinstance(self.value, IMemOperand):
            if self.value.n_val is not None:
                n_val = self.value.n_val

        n_lifted = il.const(1, n_val)

        match pre:
            case AddressingMode.N:
                return n_lifted
            case AddressingMode.BP_N:
                return il.add(1, self._reg_value("BP", il), n_lifted)
            case AddressingMode.PX_N:
                return il.add(1, self._reg_value("PX", il), n_lifted)
            case AddressingMode.PY_N:
                return il.add(1, self._reg_value("PY", il), n_lifted)
            case AddressingMode.BP_PX:
                return il.add(1, self._reg_value("BP", il), self._reg_value("PX", il))
            case AddressingMode.BP_PY:
                return il.add(1, self._reg_value("BP", il), self._reg_value("PY", il))
            case _:
                raise NotImplementedError(f"Unknown addressing mode {pre}")

    def imem_addr(
        self, il: LowLevelILFunction, pre: Optional[AddressingMode]
    ) -> ExpressionIndex:
        # Convert None to BP_N for consistent behavior
        if pre is None:
            pre = AddressingMode.BP_N

        if isinstance(self.value, TempReg):
            if pre == AddressingMode.N:
                # The register is assumed to hold the complete address.
                return self.value.lift(il)

        if isinstance(self.value, Reg):
            if pre == AddressingMode.N:
                return il.add(
                    3, self.value.lift(il), il.const(3, INTERNAL_MEMORY_START)
                )

        if isinstance(self.value, ImmOperand) and pre == AddressingMode.N:
            assert self.value.value is not None, "Value not set"
            raw_addr = INTERNAL_MEMORY_START + self.value.value
            return il.const_pointer(3, raw_addr)

        offset = self._imem_offset(il, pre)
        return il.add(3, offset, il.const(3, INTERNAL_MEMORY_START))

    def lift(
        self,
        il: LowLevelILFunction,
        pre: Optional[AddressingMode] = None,
        side_effects: bool = True,
    ) -> ExpressionIndex:
        return il.load(self.width(), self.imem_addr(il, pre))

    def lift_assign(
        self,
        il: LowLevelILFunction,
        value: ExpressionIndex,
        pre: Optional[AddressingMode] = None,
    ) -> None:
        assert isinstance(value, (MockLLIL, int)), (
            f"Expected MockLLIL or int, got {type(value)}"
        )
        il.append(il.store(self.width(), self.imem_addr(il, pre), value))


class EMemHelper(Operand):
    def __init__(self, width: int, value: Operand) -> None:
        super().__init__()
        self._width = width
        self.value = value

    def width(self) -> int:
        return self._width

    def render(self, pre: Optional[AddressingMode] = None) -> List[Token]:
        result: List[Token] = [TBegMem(MemType.EXTERNAL)]
        result.extend(self.value.render())
        result.append(TEndMem(MemType.EXTERNAL))
        return result

    def emem_addr(self, il: LowLevelILFunction) -> ExpressionIndex:
        if isinstance(self.value, ImmOperand):
            assert self.value.value is not None, "Value not set"
            raw_addr = self.value.value
            return il.const_pointer(3, raw_addr)

        return self.value.lift(il)

    def lift(
        self,
        il: LowLevelILFunction,
        pre: Optional[AddressingMode] = None,
        side_effects: bool = True,
    ) -> ExpressionIndex:
        return il.load(
            self.width(),
            self.emem_addr(il),
        )

    def lift_assign(
        self,
        il: LowLevelILFunction,
        value: ExpressionIndex,
        pre: Optional[AddressingMode] = None,
    ) -> None:
        assert isinstance(value, (MockLLIL, int)), (
            f"Expected MockLLIL or int, got {type(value)}"
        )
        il.append(il.store(self.width(), self.emem_addr(il), value))


class Pointer:
    def lift_current_addr(
        self,
        il: LowLevelILFunction,
        pre: Optional[AddressingMode] = None,
        side_effects: bool = True,
    ) -> ExpressionIndex:
        raise NotImplementedError(
            f"lift_current_addr() not implemented for {type(self)}"
        )

    def memory_helper(self) -> Type[Union[IMemHelper, EMemHelper]]:
        raise NotImplementedError(f"memory_helper() not implemented for {type(self)}")


# Read 8 bits from internal memory based on Imm8 address.
class IMem8(Imm8, Pointer):
    def width(self) -> int:
        return 1

    def lift_current_addr(
        self,
        il: LowLevelILFunction,
        pre: Optional[AddressingMode] = None,
        side_effects: bool = True,
    ) -> ExpressionIndex:
        return self._helper().imem_addr(il, pre)

    def memory_helper(self) -> Type[IMemHelper]:
        return IMemHelper

    def _helper(self) -> IMemHelper:
        return IMemHelper(self.width(), Imm8(self.value))

    def render(self, pre: Optional[AddressingMode] = None) -> List[Token]:
        return self._helper().render(pre)

    # We need to extract the raw address from IMem8 for MVL / MVLD,
    # so can't return the helper directly.
    #
    # def operands(self):
    #     yield self._helper()

    def lift(
        self,
        il: LowLevelILFunction,
        pre: Optional[AddressingMode] = None,
        side_effects: bool = True,
    ) -> ExpressionIndex:
        return self._helper().lift(il, pre, side_effects=side_effects)

    def lift_assign(
        self,
        il: LowLevelILFunction,
        value: ExpressionIndex,
        pre: Optional[AddressingMode] = None,
    ) -> None:
        return self._helper().lift_assign(il, value, pre)


# Read 16 bits from internal memory based on Imm8 address.
class IMem16(IMem8):
    def width(self) -> int:
        return 2


# Read 20 bits from internal memory based on Imm8 address.
class IMem20(IMem8):
    def width(self) -> int:
        return 3


# Register operand encoded as part of the instruction opcode
class RegLiftMixin(HasWidth):
    """Mixin providing common register lifting helpers."""

    reg: Any

    def lift(
        self,
        il: LowLevelILFunction,
        pre: Optional[AddressingMode] = None,
        side_effects: bool = True,
    ) -> ExpressionIndex:
        return il.reg(self.width(), self.reg)

    def lift_assign(
        self,
        il: LowLevelILFunction,
        value: ExpressionIndex,
        pre: Optional[AddressingMode] = None,
    ) -> None:
        il.append(il.set_reg(self.width(), self.reg, value))


class Reg(RegLiftMixin, Operand, HasWidth):
    def __init__(self, reg: Any) -> None:
        super().__init__()
        self.reg = reg

    def __repr__(self) -> str:
        return f"Reg(reg={self.reg!r})"

    def render(self, pre: Optional[AddressingMode] = None) -> List[Token]:
        return [TReg(self.reg)]

    def width(self) -> int:
        return REG_SIZES[self.reg]


class TempReg(RegLiftMixin, Operand):
    def __init__(self, reg: Any, width: int = 3) -> None:
        super().__init__()
        self.reg = reg
        self._width = width

    def render(self, pre: Optional[AddressingMode] = None) -> List[Token]:
        raise NotImplementedError("render() not implemented for TempReg")

    def width(self) -> int:
        return self._width

    # lift() and lift_assign() provided by RegLiftMixin


# only makes sense for PUSHU / POPU
class RegIL(Reg):
    """Special IL register that clears the entire I register when assigned."""

    def __init__(self) -> None:
        super().__init__("IL")

    def width(self) -> int:
        return 1

    def lift_assign(
        self,
        il: LowLevelILFunction,
        value: ExpressionIndex,
        pre: Optional[AddressingMode] = None,
    ) -> None:
        # When assigning to IL, clear the entire I register first, then set the low byte
        # This matches the hardware behavior where MV IL, XX clears IH
        il.append(
            il.set_reg(2, RegisterName("I"), il.and_expr(2, value, il.const(2, 0xFF)))
        )


class RegIMR(Reg):
    def __init__(self) -> None:
        super().__init__("IMR")

    def width(self) -> int:
        return 1

    def lift(
        self,
        il: LowLevelILFunction,
        pre: Optional[AddressingMode] = None,
        side_effects: bool = True,
    ) -> ExpressionIndex:
        # Always use direct addressing (N) for IMR, ignoring PRE mode
        imem = IMem8(IMEMRegisters.IMR)
        return imem.lift(il, AddressingMode.N, side_effects)

    def lift_assign(
        self,
        il: LowLevelILFunction,
        value: ExpressionIndex,
        pre: Optional[AddressingMode] = None,
    ) -> None:
        # Always use direct addressing (N) for IMR, ignoring PRE mode
        imem = IMem8(IMEMRegisters.IMR)
        imem.lift_assign(il, value, AddressingMode.N)


# Special case: only makes sense for MV, special case since B is not in the REGISTERS
class RegB(Reg):
    def __init__(self) -> None:
        super().__init__("B")

    def width(self) -> int:
        return 1


class RegPC(Reg):
    def __init__(self) -> None:
        super().__init__("PC")

    def width(self) -> int:
        return 3


# only makes sense for PUSHU / POPU / PUSHS / POPS
class RegF(Reg):
    def __init__(self) -> None:
        super().__init__("F")

    def width(self) -> int:
        return 1

    def lift(
        self,
        il: LowLevelILFunction,
        pre: Optional[AddressingMode] = None,
        side_effects: bool = True,
    ) -> ExpressionIndex:
        zbit = il.shift_left(1, il.flag(ZFlag), il.const(1, 1))
        return il.or_expr(1, il.flag(CFlag), zbit)

    def lift_assign(
        self,
        il: LowLevelILFunction,
        value: ExpressionIndex,
        pre: Optional[AddressingMode] = None,
    ) -> None:
        tmp = TempReg(TempRegF, width=self.width())
        tmp.lift_assign(il, value)
        il.append(il.set_flag(CFlag, il.and_expr(1, tmp.lift(il), il.const(1, 1))))
        il.append(il.set_flag(ZFlag, il.and_expr(1, tmp.lift(il), il.const(1, 2))))


class Reg3(RegLiftMixin, Operand, HasWidth):
    reg: Optional[RegisterName]
    reg_raw: Optional[int]
    high4: Optional[int]

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def reg_name(cls, idx: int) -> RegisterName:
        return REG_NAMES[idx]

    @classmethod
    def reg_idx(cls, name: RegisterName) -> int:
        return REG_NAMES.index(name)

    def width(self) -> int:
        assert self.reg is not None, "Register not set"
        return REG_SIZES[self.reg]

    def assert_r3(self) -> None:
        try:
            assert self.width() >= 3, (
                f"Want r3 register, got r{self.width()} ({self.reg}) instead"
            )
        except AssertionError as e:
            raise InvalidInstruction("Invalid register for r3 instruction") from e

    def decode(self, decoder: Decoder, addr: int) -> None:
        byte = decoder.unsigned_byte()
        self.reg_raw = byte
        self.reg = self.reg_name(byte & 7)
        # store high 4 bits from byte for later reference
        self.high4 = (byte >> 4) & 0x0F

    def encode(self, encoder: Encoder, addr: int) -> None:
        assert self.reg_raw is not None, "Register raw value not set"
        assert self.high4 is not None, "High 4 bits not set"
        byte = self.reg_raw | (self.high4 << 4)
        encoder.unsigned_byte(byte)

    def render(self, pre: Optional[AddressingMode] = None) -> List[Token]:
        assert self.reg is not None, "Register not set"
        return [TReg(self.reg)]

    # lift() and lift_assign() provided by RegLiftMixin


# External Memory: Absolute Addressing using 20-bit address
# [lmn]: encoded as `[n m l]`
class EMemAddr(Imm20, Pointer):
    def __init__(self, width: int) -> None:
        super().__init__()
        self._width = width
        # Ensure extra_hi exists so assembler can populate it
        self.extra_hi = 0

    def width(self) -> int:
        return self._width

    def lift_current_addr(
        self,
        il: LowLevelILFunction,
        pre: Optional[AddressingMode] = None,
        side_effects: bool = True,
    ) -> ExpressionIndex:
        assert self.value is not None, "Value not set"
        return il.const_pointer(3, self.value)

    def memory_helper(self) -> Type[EMemHelper]:
        return EMemHelper

    def render(self, pre: Optional[AddressingMode] = None) -> List[Token]:
        assert self.value is not None, "Value not set"
        return [
            TBegMem(MemType.EXTERNAL),
            TAddr(self.value),
            TEndMem(MemType.EXTERNAL),
        ]

    def lift(
        self,
        il: LowLevelILFunction,
        pre: Optional[AddressingMode] = None,
        side_effects: bool = True,
    ) -> ExpressionIndex:
        assert self.value is not None, "Value not set"
        return il.load(self.width(), il.const_pointer(3, self.value))

    def lift_assign(
        self,
        il: LowLevelILFunction,
        value: ExpressionIndex,
        pre: Optional[AddressingMode] = None,
    ) -> None:
        assert self.value is not None, "Value not set"
        assert isinstance(value, (MockLLIL, int)), (
            f"Expected MockLLIL or int, got {type(value)}"
        )
        il.append(il.store(self.width(), il.const_pointer(3, self.value), value))


class EMemValueOffsetHelper(OperandHelper, Pointer):
    def __init__(
        self, value: Operand, offset: Optional[ImmOffset], width: int = 1
    ) -> None:
        super().__init__()
        self.value = value
        self.offset = offset
        self._width = width

    def width(self) -> int:
        return self._width

    def lift_current_addr(
        self,
        il: LowLevelILFunction,
        pre: Optional[AddressingMode] = None,
        side_effects: bool = True,
    ) -> ExpressionIndex:
        # For indirect external memory addressing through internal memory,
        # we need to read a 20-bit address from internal memory
        if isinstance(self.value, IMem8):
            # For indirect addressing, the IMem8 value points to a location in internal memory
            # that contains a 20-bit external memory address. We need to read 3 bytes from there.
            # First get the internal memory address
            imem_addr = self.value._helper().imem_addr(il, pre)
            # Now load 3 bytes (20-bit address) from that location
            addr = il.load(3, imem_addr)
        else:
            addr = self.value.lift(il, pre=pre, side_effects=side_effects)

        if self.offset:
            addr = self.offset.lift_offset(il, addr)
        return addr

    def memory_helper(self) -> Type[EMemHelper]:
        return EMemHelper

    def render(self, pre: Optional[AddressingMode] = None) -> List[Token]:
        result: List[Token] = [TBegMem(MemType.EXTERNAL)]
        result.extend(
            self.value.render(pre)
        )  # Pass pre to render if self.value supports it
        if self.offset:
            result.extend(self.offset.render())
        result.append(TEndMem(MemType.EXTERNAL))
        return result

    def lift(
        self,
        il: LowLevelILFunction,
        pre: Optional[AddressingMode] = None,
        side_effects: bool = True,
    ) -> ExpressionIndex:
        # width is determined by the context in which this helper is used
        return il.load(
            self.width(), self.lift_current_addr(il, pre=pre, side_effects=side_effects)
        )

    def lift_assign(
        self,
        il: LowLevelILFunction,
        value: ExpressionIndex,
        pre: Optional[AddressingMode] = None,
    ) -> None:
        addr = self.lift_current_addr(il, pre=pre, side_effects=True)
        il.append(il.store(self.width(), addr, value))


# page 74 of the book
# External Memory: Register Indirect
# 0: [r3']:     Register Indirect
# 2: [r3'++]:   Register Indirect with post-increment
# 3: [--r3']:   Register Indirect with pre-decrement
# 8: [r3+imm8]: Register Indirect with positive offset
# C: [r3-imm8]: Register Indirect with negative offset
class EMemRegMode(enum.Enum):
    SIMPLE = 0x0
    POST_INC = 0x2
    PRE_DEC = 0x3
    POSITIVE_OFFSET = 0x8
    NEGATIVE_OFFSET = 0xC


def get_emem_reg_mode(val: Optional[int], addr: int) -> EMemRegMode:
    try:
        return EMemRegMode(val)
    except Exception:
        raise InvalidInstruction(
            f"Invalid EMemRegMode {val:02X} at address {addr:#06x}"
        )


class RegIncrementDecrementHelper(OperandHelper):
    def __init__(self, width: int, reg: Reg3, mode: EMemRegMode) -> None:
        super().__init__()
        self.width = width  # This width is the increment/decrement amount, typically data size (1, 2, or 3)
        self.reg = reg
        self.mode = mode
        assert mode in (EMemRegMode.SIMPLE, EMemRegMode.POST_INC, EMemRegMode.PRE_DEC)

    def render(self, pre: Optional[AddressingMode] = None) -> List[Token]:
        result = []
        if self.mode == EMemRegMode.SIMPLE:
            result.extend(self.reg.render())
        elif self.mode == EMemRegMode.POST_INC:
            result.extend(self.reg.render())
            result.append(TText("++"))
        elif self.mode == EMemRegMode.PRE_DEC:
            result.append(TText("--"))
            result.extend(self.reg.render())
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        return result

    def lift(
        self,
        il: LowLevelILFunction,
        pre: Optional[AddressingMode] = None,
        side_effects: bool = True,
    ) -> ExpressionIndex:
        value = self.reg.lift(il)

        if side_effects and self.mode == EMemRegMode.POST_INC:
            # create LLIL_TEMP to hold the value since we're supposed to
            # increment it after using it
            tmp = TempReg(TempIncDecHelper, width=self.reg.width())
            tmp.lift_assign(il, value)
            self.reg.lift_assign(
                il,
                il.add(self.reg.width(), value, il.const(self.reg.width(), self.width)),
            )
            value = tmp.lift(il)

        if side_effects and self.mode == EMemRegMode.PRE_DEC:
            # For pre-decrement with side effects:
            # 1. Calculate the decremented value expression
            # 2. Store it in a temp register to capture the value
            # 3. Update the actual register with the same expression
            # 4. Return the temp register

            # Calculate the decremented value
            new_value = il.sub(
                self.reg.width(), value, il.const(self.reg.width(), self.width)
            )

            # Store the decremented value in a temp register
            # This captures the value at this point in time
            tmp = TempReg(TempIncDecHelper, width=self.reg.width())
            tmp.lift_assign(il, new_value)

            # Update the actual register with the same expression
            self.reg.lift_assign(il, new_value)

            # Return the temp register's value
            value = tmp.lift(il)
        elif self.mode == EMemRegMode.PRE_DEC:
            # No side effects - just return the decremented value expression
            value = il.sub(
                self.reg.width(), value, il.const(self.reg.width(), self.width)
            )

        return value


class EMemRegOffsetHelper(HasOperands, OperandHelper):
    def __init__(
        self, width: int, reg: Reg3, mode: EMemRegMode, offset: Optional[ImmOffset]
    ) -> None:
        super().__init__()
        self.width = width
        self.reg = reg
        self.mode = mode
        self.offset = offset

    def operands(self) -> Generator[Operand, None, None]:
        reg: Operand
        if self.mode in (EMemRegMode.SIMPLE, EMemRegMode.POST_INC, EMemRegMode.PRE_DEC):
            # Create the helper only once and cache it
            if not hasattr(self, "_cached_helper"):
                self._cached_helper = RegIncrementDecrementHelper(
                    self.width, self.reg, self.mode
                )
            reg = self._cached_helper
        else:
            reg = self.reg

        op = EMemValueOffsetHelper(reg, self.offset, width=self.width)
        yield op


class RegIMemOffsetOrder(enum.Enum):
    DEST_IMEM = 0
    DEST_REG_OFFSET = 1


# 0x56: page 77 of the book
# (m), [r3±n]: encoded as `56 (8 r3 | C r3) m n
#
# 0x5E: page 77 of the book
# [r3±m], (n): encoded as 5E (8 r3 | C r3) n m
#
# 0xE0: page 75 of the book
# (n), [r3], : encoded as E0 (0 r3) n
# (n), [r3++]: encoded as E0 (2 r3) n
# (n), [--r3]: encoded as E0 (3 r3) n
# (n), [r3±m]: encoded as E0 (8 r3 | C r3) n m
#
# 0xE8: page 75 of the book
# [r3],   (n): encoded as E8 (0 r3) n
# [r3++], (n): encoded as E8 (2 r3) n
# [--r3], (n): encoded as E8 (3 r3) n
# [r3±m], (n): encoded as E8 (8 r3 | C r3) n m
class RegIMemOffset(OffsetOperandMixin, HasOperands, Operand):
    reg: Optional[Reg3]
    imem: Optional[IMem8]
    mode: Optional[EMemRegMode]
    offset: Optional[ImmOffset] = None

    def __init__(
        self,
        order: RegIMemOffsetOrder,
        allowed_modes: Optional[List[EMemRegMode]] = None,
    ) -> None:
        self.order = order
        self.allowed_modes = allowed_modes
        self.width = 1  # Default width, will be updated based on instruction name

    def __repr__(self) -> str:
        return (
            f"RegIMemOffset(order={self.order}, mode={getattr(self, 'mode', None)},"
            f" offset={getattr(self, 'offset', None)})"
        )

    def operands(self) -> Generator[Operand, None, None]:
        assert self.reg is not None, "Register not set"
        assert self.imem is not None, "IMem not set"
        assert self.mode is not None, "Mode not set"
        assert isinstance(self.imem, HasWidth), (
            f"Expected HasWidth, got {type(self.imem)}"
        )

        # Create the appropriate IMem operand based on width
        if self.width == 2:
            # For MVW, we need IMem16 instead of IMem8
            imem_operand = IMem16()
            imem_operand.value = self.imem.value  # Copy the value
        elif self.width == 3:
            # For MVP, we need IMem20
            imem_operand = IMem20()
            imem_operand.value = self.imem.value  # Copy the value
        else:
            imem_operand = self.imem

        op = EMemRegOffsetHelper(self.width, self.reg, self.mode, self.offset)
        if self.order == RegIMemOffsetOrder.DEST_REG_OFFSET:
            yield op
            yield imem_operand
        else:
            yield imem_operand
            yield op

    def set_width_from_instruction(self, instr: "Instruction") -> None:
        """Set width based on the instruction name (MVW=2, MVP=3, otherwise 1)."""
        if instr.name() == "MVW":
            self.width = 2
        elif instr.name() == "MVP":
            self.width = 3
        else:
            self.width = 1

    def decode(self, decoder: Decoder, addr: int) -> None:
        super().decode(decoder, addr)
        self.reg = Reg3()
        self.reg.decode(decoder, addr)
        self.reg.assert_r3()

        # For now, always decode as IMem8, we'll handle width in operands()
        self.imem = IMem8()
        self.imem.decode(decoder, addr)

        self.mode = get_emem_reg_mode(self.reg.high4, addr)
        if self.allowed_modes is not None:
            assert self.mode in self.allowed_modes
        self._decode_offset(decoder, addr)

    def encode(self, encoder: Encoder, addr: int) -> None:
        super().encode(encoder, addr)
        assert self.reg is not None, "Register not set"
        self.reg.encode(encoder, addr)
        assert self.imem is not None, "IMem not set"
        self.imem.encode(encoder, addr)
        self._encode_offset(encoder, addr)


class EMemReg(OffsetOperandMixin, HasOperands, Operand):
    mode: Optional[EMemRegMode]
    offset: Optional[ImmOffset] = None

    def __init__(
        self, width: int, allowed_modes: Optional[List[EMemRegMode]] = None
    ) -> None:
        super().__init__()
        self.width = width
        self.reg = Reg3()
        self.allowed_modes = allowed_modes

    def __repr__(self) -> str:
        return (
            f"EMemReg(width={self.width}, mode={getattr(self, 'mode', None)}, "
            f"offset={getattr(self, 'offset', None)})"
        )

    def decode(self, decoder: Decoder, addr: int) -> None:
        super().decode(decoder, addr)
        self.reg.decode(decoder, addr)
        self.reg.assert_r3()
        self.mode = get_emem_reg_mode(self.reg.high4, addr)
        if self.allowed_modes is not None:
            assert self.mode in self.allowed_modes, (
                f"Invalid mode: {self.mode}, allowed: {self.allowed_modes}"
            )
        self._decode_offset(decoder, addr)

    def encode(self, encoder: Encoder, addr: int) -> None:
        # super().encode(encoder, addr)
        self.reg.encode(encoder, addr)
        self._encode_offset(encoder, addr)

    def operands(self) -> Generator[Operand, None, None]:
        assert self.mode is not None, "Mode not set"
        op = EMemRegOffsetHelper(self.width, self.reg, self.mode, self.offset)
        yield op


# page 74 of the book
# External Memory: Internal Memory indirect
# 00: [(n)]
# 80: [(m)+n]
# C0: [(m)-n]
class EMemIMemMode(enum.Enum):
    SIMPLE = 0x00
    POSITIVE_OFFSET = 0x80
    NEGATIVE_OFFSET = 0xC0


def get_emem_imem_mode(val: Optional[int], addr: int) -> EMemIMemMode:
    try:
        return EMemIMemMode(val)
    except Exception:
        raise InvalidInstruction(f"Invalid EMemIMemMode {val:02X} at {addr:04X}")


class EMemIMem(OffsetOperandMixin, HasOperands, Imm8):
    mode: Optional[EMemIMemMode]
    offset: Optional[ImmOffset] = None

    def __init__(self, width: Optional[int] = None) -> None:
        super().__init__()
        # Allow both decoded IMem8 values and parsed IMemOperand objects
        self.imem: Union[IMem8, IMemOperand] = IMem8()
        self._width = width if width is not None else 1

    def __repr__(self) -> str:
        return (
            f"EMemIMem(mode={getattr(self, 'mode', None)}, "
            f"offset={getattr(self, 'offset', None)})"
        )

    def decode(self, decoder: Decoder, addr: int) -> None:
        super().decode(decoder, addr)
        self.imem.decode(decoder, addr)

        self.mode = get_emem_imem_mode(self.value, addr)
        self._decode_offset(decoder, addr)

    def encode(self, encoder: Encoder, addr: int) -> None:
        super().encode(encoder, addr)
        self.imem.encode(encoder, addr)

        self._encode_offset(encoder, addr)

    def operands(self) -> Generator[Operand, None, None]:
        op = EMemValueOffsetHelper(self.imem, self.offset, width=self._width)
        yield op

    def set_width_from_instruction(self, instr: "Instruction") -> None:
        """Set width based on the source register for MV instructions."""
        # For MV EMemIMem, Reg - determine width from the source register
        # The opcode table should set the width based on the register size
        pass  # Width should be set in __init__ from opcode table


class EMemIMemOffsetOrder(enum.Enum):
    DEST_INT_MEM = 0
    DEST_EXT_MEM = 1


# page 75 of the book
# (m), [(n)]:   encoded as F0 00 m n
# (l), [(m)+n]: encoded as F0 80 l m n
# (l), [(m)-n]: encoded as F0 C0 l m n
#
# page 77 of the book
# (m), [(n)]:   encoded as F3 00 m n
# (l), [(m)+n]: encoded as F3 80 l m n
# (l), [(m)-n]: encoded as F3 C0 l m n
#
# page 75 of the book
# [(m)], (n):   encoded as F8 00 m n
# [(l)+m], (n): encoded as F8 80 l m n
# [(l)-m], (n): encoded as F8 C0 l m n
#
# page 77 of the book
# [(m)], (n):   encoded as FB 00 m n
# [(l)+m], (n): encoded as FB 80 l n m
# [(l)-m], (n): encoded as FB C0 l n m
class EMemIMemOffset(OffsetOperandMixin, HasOperands, Operand):
    mode: Optional[EMemIMemMode]
    offset: Optional[ImmOffset] = None

    def __init__(self, order: EMemIMemOffsetOrder, width: int = 1) -> None:
        self.order = order
        self.width = width
        self.mode_imm = Imm8()
        self.imem1 = IMem8()
        self.imem2 = IMem8()
        self._parent_instruction = None

    def __repr__(self) -> str:
        return (
            f"EMemIMemOffset(order={self.order}, width={self.width}, "
            f"mode={getattr(self, 'mode', None)}, offset={getattr(self, 'offset', None)})"
        )

    def operands(self) -> Generator[Operand, None, None]:
        # Create the appropriate IMem operands based on width
        if self.width == 2:
            # For MVW, we need IMem16
            imem1_operand = IMem16()
            imem2_operand = IMem16()
            imem1_operand.value = self.imem1.value
            imem2_operand.value = self.imem2.value
        elif self.width == 3:
            # For MVP, we need IMem20
            imem1_operand = IMem20()
            imem2_operand = IMem20()
            imem1_operand.value = self.imem1.value
            imem2_operand.value = self.imem2.value
        else:
            imem1_operand = self.imem1
            imem2_operand = self.imem2

        if self.order == EMemIMemOffsetOrder.DEST_INT_MEM:
            yield imem1_operand
            op = EMemValueOffsetHelper(imem2_operand, self.offset, width=self.width)
            yield op
        else:
            op = EMemValueOffsetHelper(imem1_operand, self.offset, width=self.width)
            yield op
            yield imem2_operand

    def set_width_from_instruction(self, instr: "Instruction") -> None:
        """Set width based on the instruction name (MVW=2, MVP=3, otherwise 1)."""
        if instr.name() == "MVW":
            self.width = 2
        elif instr.name() == "MVP":
            self.width = 3
        else:
            self.width = 1

    def decode(self, decoder: Decoder, addr: int) -> None:
        super().decode(decoder, addr)
        self.mode_imm = Imm8()
        self.mode_imm.decode(decoder, addr)

        self.imem1 = IMem8()
        self.imem1.decode(decoder, addr)

        self.imem2 = IMem8()
        self.imem2.decode(decoder, addr)

        self.mode = get_emem_imem_mode(self.mode_imm.value, addr)
        self._decode_offset(decoder, addr)

    def encode(self, encoder: Encoder, addr: int) -> None:
        super().encode(encoder, addr)
        self.mode_imm.encode(encoder, addr)
        self.imem1.encode(encoder, addr)
        self.imem2.encode(encoder, addr)
        self._encode_offset(encoder, addr)


# ADD/SUB can use various-sized register pairs
class RegPair(HasOperands, Reg3):
    reg_raw: Optional[int]
    reg1: Optional[Reg]
    reg2: Optional[Reg]

    def __init__(self, size: Optional[int] = None) -> None:
        super().__init__()
        self.size = size

    def decode(self, decoder: Decoder, addr: int) -> None:
        self.reg_raw = decoder.unsigned_byte()
        self.reg1 = Reg(REG_NAMES[(self.reg_raw >> 4) & 7])
        self.reg2 = Reg(REG_NAMES[self.reg_raw & 7])

        try:
            # high-bits of both halves must be zero: 0x80 and 0x08 must not be set
            assert (self.reg_raw & 0x80) == 0, (
                f"Invalid reg1 high bit: {self.reg_raw:02X}"
            )
            assert (self.reg_raw & 0x08) == 0, (
                f"Invalid reg2 high bit: {self.reg_raw:02X}"
            )
        except AssertionError as e:
            raise InvalidInstruction(f"Invalid reg pair at {addr:04X}") from e

    def operands(self) -> Generator[Operand, None, None]:
        assert self.reg1 is not None, "Register 1 not set"
        assert self.reg2 is not None, "Register 2 not set"
        yield self.reg1
        yield self.reg2

    def encode(self, encoder: Encoder, addr: int) -> None:
        assert self.reg_raw is not None, "Register raw value not set"
        encoder.unsigned_byte(self.reg_raw)

    def render(self, pre: Optional[AddressingMode] = None) -> List[Token]:
        assert self.reg1 is not None, "Register 1 not set"
        assert self.reg2 is not None, "Register 2 not set"
        result = self.reg1.render()
        result.append(TSep(", "))
        result.extend(self.reg2.render())
        return result


@contextmanager
def lift_loop(il: LowLevelILFunction) -> Generator[None, None, None]:
    if_true = LowLevelILLabel()
    if_false = LowLevelILLabel()

    loop_reg = Reg("I")
    width = loop_reg.width()

    # If I is zero, skip the loop entirely
    il.append(
        il.if_expr(
            il.compare_equal(width, loop_reg.lift(il), il.const(width, 0)),
            if_true,
            if_false,
        )
    )
    il.mark_label(if_false)

    # loop iteration
    yield

    loop_reg.lift_assign(il, il.sub(width, loop_reg.lift(il), il.const(1, 1)))
    cond = il.compare_equal(width, loop_reg.lift(il), il.const(width, 0))
    il.append(il.if_expr(cond, if_true, if_false))
    il.mark_label(if_true)
