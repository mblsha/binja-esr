# based on https://github.com/whitequark/binja-avnera/blob/main/mc/instr.py
from .tokens import Token, TInstr, TText, TSep, TInt, TReg, TBegMem, TEndMem, MemType
from .coding import Decoder, Encoder, BufferTooShort
from .mock_analysis import BranchType
from .mock_llil import MockLLIL
from .constants import INTERNAL_MEMORY_START

import copy
from dataclasses import dataclass
from typing import Optional, List, Generator, Iterator, Dict, Tuple, Union, Type, Literal, Any, Callable, cast
import enum
from contextlib import contextmanager


from . import binja_api # noqa: F401
from binaryninja import (
    InstructionInfo,
)
from binaryninja.architecture import (
    RegisterName,
    IntrinsicName,
    FlagName,
)
from binaryninja.lowlevelil import (
    LowLevelILFunction,
)
from binaryninja.lowlevelil import (
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
    N = '(n)'
    BP_N = '(BP+n)'
    PX_N = '(PX+n)'
    PY_N = '(PY+n)'
    BP_PX = '(BP+PX)'
    BP_PY = '(BP+PY)'

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
    }
}

REVERSE_PRE_TABLE: Dict[Tuple[AddressingMode, AddressingMode], int] = {
    # 1st Op \ 2nd Op: (n)
    (AddressingMode.N,     AddressingMode.N):     0x32,
    (AddressingMode.BP_N,  AddressingMode.N):     0x22,
    (AddressingMode.PX_N,  AddressingMode.N):     0x36,
    (AddressingMode.BP_PX, AddressingMode.N):     0x26,
    # 1st Op \ 2nd Op: (BP+n)
    (AddressingMode.N,     AddressingMode.BP_N):  0x30,
    (AddressingMode.PX_N,  AddressingMode.BP_N):  0x34,
    (AddressingMode.BP_PX, AddressingMode.BP_N):  0x24,
    # 1st Op \ 2nd Op: (PY+n)
    (AddressingMode.N,     AddressingMode.PY_N):  0x33,
    (AddressingMode.BP_N,  AddressingMode.PY_N):  0x23,
    (AddressingMode.PX_N,  AddressingMode.PY_N):  0x37,
    (AddressingMode.BP_PX, AddressingMode.PY_N):  0x27,
    # 1st Op \ 2nd Op: (BP+PY)
    (AddressingMode.N,     AddressingMode.BP_PY): 0x31,
    (AddressingMode.BP_N,  AddressingMode.BP_PY): 0x21,
    (AddressingMode.PX_N,  AddressingMode.BP_PY): 0x35,
}


def get_addressing_mode(pre_value: int, operand_index: int) -> AddressingMode:
    """
    Returns the addressing mode for the given PRE byte and operand index (1 or 2).
    """
    try:
        return PRE_TABLE[operand_index][pre_value]
    except KeyError:
        raise ValueError(f"Unknown PRE value {pre_value:02X}H for operand index {operand_index}")

TCLIntrinsic = IntrinsicName("TCL")
HALTIntrinsic = IntrinsicName("HALT")
OFFIntrinsic = IntrinsicName("OFF")

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

CFlag = FlagName('C')
ZFlag = FlagName('Z')
CZFlag = FlagName('CZ')

REG_NAMES = [reg[0] for reg in REGISTERS]
REG_SIZES = {reg[0]: min(3, reg[1]) for reg in REGISTERS}


INTERRUPT_VECTOR_ADDR = 0xFFFFA
ENTRY_POINT_ADDR = 0xFFFFD

# Hitachi LCD Driver
SH26_ADDR_START = 0x00000
SH26_ADDR_END   = 0x3FFFF

# TENRI LCD Segment Driver
LH5073A1_ADDR_START = 0x40000
LH5073A1_ADDR_END   = 0x7FFFF

CE1_ADDR_START = 0x80000
CE1_ADDR_END   = 0x9FFFF
CE0_ADDR_START = 0xA0000
CE0_ADDR_END   = 0xBFFFF

# Map internal RAM to start immediately after the 1MB external space. The
# internal region occupies addresses
#   [INTERNAL_MEMORY_START, ADDRESS_SPACE_SIZE - 1].

IMEM_NAMES = {
    "BP":  0xEC, # RAM Base Pointer
    "PX":  0xED, # RAM PX Pointer
    "PY":  0xEE, # RAM PY Pointer

    # A system with two RAM card slots may have two discontinuous
    # physical address windows (CE1 and CE0).  This register lets
    # you virtually join them into one contiguous block when enabled.
    #
    # When AME (bit 7) = 1:
    #   - The end of the CE1 window is linked to the start of the
    #     CE0 window in the software’s virtual address space.
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
    #   • Virtual CE1 region follows directly after CE1’s physical
    #     end.
    #   • Virtual CE0 region begins at CE0’s physical base.
    "AMC": 0xEF, # ADR Modify Control

    # Controls KO0-KO15 output pins
    "KOL": 0xF0, # Key Output Buffer H
    "KOH": 0xF1, # Key Output Buffer L

    # Controls KI0-KI7 input pins
    "KIL": 0xF2, # Key Input Buffer

    # Controls E0-E15 pins
    "EOL": 0xF3, # E Port Output Buffer H
    "EOH": 0xF4, # E Port Output Buffer L
    # Controls E0-E15 pins
    "EIL": 0xF5, # E Port Input Buffer H
    "EIH": 0xF6, # E Port Input Buffer L

    #     7     6     5     4     3     2     1     0
    #   +-----+-----+-----+-----+-----+-----+-----+-----+
    #   | BOE | BR2 | BR1 | BR0 | PA1 | PA0 |  DL |  ST |
    #   +-----+-----+-----+-----+-----+-----+-----+-----+
    #
    #  BOE  (bit 7)  – Break Output Enable.
    #                  When ‘1’, TXD is driven low (“0”) continuously.
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
    "UCR": 0xF7, # UART Control Register

    #     7     6     5     4     3     2     1     0
    #   +-----+-----+-----+-----+-----+-----+-----+-----+
    #   |     |     | RXR | TXE | TXR |  FE |  OE |  PE |
    #   +-----+-----+-----+-----+-----+-----+-----+-----+
    #
    #  RXR (bit 5) – Receiver Ready:
    #     ‘1’ when a character has been fully received;
    #     clears to ‘0’ once RX buffer is read.
    #
    #  TXE (bit 4) – Transmitter Empty:
    #     ‘0’ while UART is shifting bits out;
    #     ‘1’ when transmitter is idle.
    #
    #  TXR (bit 3) – Transmitter Ready:
    #     ‘0’ immediately after software writes TXD;
    #     becomes ‘1’ once data has moved into the shift register.
    #
    #  FE  (bit 2) – Framing Error:
    #     ‘0’ if stop-bit framing was incorrect; ‘1’ otherwise.
    #     Updated on each receive completion.
    #
    #  OE  (bit 1) – Overrun Error:
    #     ‘1’ if new character completes while RXR=‘1’.
    #     Updated on each receive completion.
    #
    #  PE  (bit 0) – Parity Error:
    #     ‘1’ if received parity does not match.
    #     Updated on each receive completion.
    "USR": 0xF8, # UART Status Register

    # Holds the 8-bit data of the last received character.
    "RXD": 0xF9, # UART Receive Buffer

    # – Write data here for transmission.
    # – When TXE (USR[4]) goes ‘1’, the byte moves to the transmitter.
    # – You may queue a new byte even while prior is sending;
    #   TXR (USR[3]) tells you when it’s been accepted.
    "TXD": 0xFA, # UART Transmit Buffer

    #    7     6     5      4      3      2     1     0
    #  +-----+-----+------+-------+------+-----+-----+-----+
    #  | IRM | EXM | RXRM | TXRM  | ONKM | KEYM| STM | MTM |
    #  +-----+-----+------+-------+------+-----+-----+-----+
    #
    # IRM  (bit 7) – Global interrupt mask:
    #    Write ‘0’ to disable all sources.
    #
    # EXM  (bit 6) – External Interrupt Mask.
    # RXRM (bit 5) – Receiver Ready Interrupt Mask.
    # TXRM (bit 4) – Transmitter Ready Interrupt Mask.
    # ONKM (bit 3) – On-Key Interrupt Mask.
    # KEYM (bit 2) – Key Interrupt Mask.
    # STM  (bit 1) – SEC Timer Interrupt Mask.
    # MTM  (bit 0) – MSEC Timer Interrupt Mask.
    #
    # Writing ‘0’ to any bit inhibits that individual interrupt source.
    # On interrupt entry, the current IMR is pushed to system/user stack
    # and IRM (bit 7) is cleared.
    "IMR": 0xFB, # Interrupt Mask Register

    #     7    6     5     4      3      2     1     0
    #   +----+-----+-----+------+-------+-----+-----+-----+
    #   |    | EXI | RXRI| TXRI | ONKI  | KEYI| STI | MTI |
    #   +----+-----+-----+------+-------+-----+-----+-----+
    #
    #  Bit 7  – Reserved.
    #  EXI    (bit 6) – External Interrupt:
    #        ‘1’ when an IRQ request arrives on the external pin.
    #  RXRI   (bit 5) – Receiver Ready Interrupt:
    #        ‘1’ when UART has completed receiving one character.
    #  TXRI   (bit 4) – Transmitter Ready Interrupt:
    #        ‘1’ when TX buffer (FAH) is ready for new data.
    #  ONKI   (bit 3) – On-Key Interrupt:
    #        ‘1’ when a high level is input to the ON pin.
    #  KEYI   (bit 2) – Key Interrupt:
    #        ‘1’ if any configured KI pin goes high.
    #  STI    (bit 1) – SEC Timer Interrupt:
    #        ‘1’ when the sub-CG timer requests an interrupt.
    #  MTI    (bit 0) – MSEC Timer Interrupt:
    #        ‘1’ when the main CG timer requests an interrupt.
    "ISR": 0xFC, # Interrupt Status Register

    #     7    6    5    4    3    2    1     0
    #   +----+----+----+----+-----+----+----+-----+
    #   | ISE| BZ2| BZ1| BZ0| VDDC| STS| MTS| DISC|
    #   +----+----+----+----+-----+----+----+-----+
    #
    #  ISE   (bit 7) – IRQ Start Enable:
    #               ‘1’ allows an external IRQ to resume the CPU from HALT/OFF.
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
    "SCR": 0xFD, # System Control Register

    #     7     6    5    4    3    2    1     0
    #   +----+----+----+----+----+----+-----+------+
    #   |LCC4|LCC3|LCC2|LCC1|LCC0| KSD| STCL| MTCL |
    #   +----+----+----+----+----+----+-----+------+
    #
    #  LCC4–LCC0 (bits 7–3) – Contrast level (0–31):
    #     00000 = min … 11111 = max
    #
    #  KSD    (bit 2) – Key Strobe Disable:
    #               ‘1’ forces KO pins low; key outputs can be read.
    #
    #  STCL   (bit 1) – SEC Timer Clear:
    #               If ‘1’ when TCL executes, resets sub-CG timer.
    #
    #  MTCL   (bit 0) – MSEC Timer Clear:
    #               If ‘1’ when TCL executes, resets main CG timer.
    "LCC": 0xFE, # LCD Contrast Control

    #     7    6    5    4    3    2    1     0
    #   +----+----+----+----+----+----+----+------+
    #   |    |    |    |    | ONK| RSF| CI | TEST |
    #   +----+----+----+----+----+----+----+------+
    #
    #  Bits 7–4 – Reserved.
    #
    #  ONK   (bit 3) – ON-Key input:
    #               ‘0’ when ON pin is low, ‘1’ when high.
    #
    #  RSF   (bit 2) – Reset-Start Flag:
    #               ‘0’ when RESET pin is high, ‘1’ when HALT/OFF.
    #
    #  CI    (bit 1) – CMT Input:
    #               ‘0’ when CI pin is low, ‘1’ when high.
    #
    #  TEST  (bit 0) – Test Input:
    #               ‘0’ when TEST pin is low, ‘1’ when high.
    "SSR": 0xFF, # System Status Control
}

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

    def lift(self, il: LowLevelILFunction, pre: Optional[AddressingMode] = None, side_effects: bool = True) -> ExpressionIndex:
        return il.unimplemented()

    def lift_assign(self, il: LowLevelILFunction, value: ExpressionIndex, pre:
                    Optional[AddressingMode] = None) -> None:
        il.append(value)
        il.append(il.unimplemented())


# used by Operands to help render / lift values
class OperandHelper(Operand):
    def render(self, pre: Optional[AddressingMode] = None) -> List[Token]:
        raise NotImplementedError(f"render() not implemented for {self.__class__.__name__} helper")


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


def iter_encode(iter: List['Instruction'], addr: int) -> bytearray:
    encoder = Encoder()
    for instr in iter:
        instr.encode(encoder, addr)
        addr += instr.length()
    return encoder.buf


def encode(instr: 'Instruction', addr: int) -> bytearray:
    return iter_encode([instr], addr)


InstrOptsType = Tuple[Type['Instruction'], Opts]
OpcodesType = Union[Type['Instruction'], InstrOptsType]
def create_instruction(decoder: Decoder, opcodes: Dict[int, OpcodesType]) -> Optional['Instruction']:
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


def iter_decode(decoder: Decoder, addr: int, opcodes: Dict[int, OpcodesType]) -> Iterator[Tuple['Instruction', int]]:
    while True:
        try:
            instr = create_instruction(decoder, opcodes)
            if instr is None:
                raise NotImplementedError(
                    f"Cannot decode opcode "
                    f"at address {addr + decoder.pos:#06x}"
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


def fusion(iter: Iterator[Tuple['Instruction', int]]) -> Iterator[Tuple['Instruction', int]]:
    try:
        instr1, addr1 = next(iter)
    except StopIteration:
        return
    while True:
        try:
            instr2, addr2 = next(iter)
        except (StopIteration, NotImplementedError):
            yield instr1, addr1
            break

        if instr12 := instr1.fuse(instr2):
            yield instr12, addr1
            try:
                instr1, addr1 = next(iter)
            except (StopIteration, NotImplementedError):
                break
        else:
            yield instr1, addr1
            instr1, addr1 = instr2, addr2


def _create_decoder(decoder: Decoder, addr: int, opcodes: Dict[int, OpcodesType]) -> Iterator[Tuple['Instruction', int]]:
    return fusion(fusion(iter_decode(decoder, addr, opcodes)))


def decode(decoder: Decoder, addr: int, opcodes: Dict[int, OpcodesType]) -> Optional['Instruction']:
    try:
        instr, _ = next(_create_decoder(decoder, addr, opcodes))

        return instr
    except StopIteration:
        print("StopIteration: No instruction found")
        return None
    # except NotImplementedError as e:
    #     binaryninja.log_warn(e)


class Instruction:
    opcode: Optional[int]
    _length: Optional[int]
    _pre: Optional[int] = None

    def __init__(self, name: str, operands: List[Operand], cond: Optional[str],
                 ops_reversed: Optional[bool]) -> None:
        self.instr_name = name
        self.ops_reversed = ops_reversed
        self._operands = operands
        self._cond = cond
        self.doinit()

    def doinit(self) -> None:
        pass

    def length(self) -> int:
        assert self._length is not None, "Length not set"
        return self._length

    def name(self) -> str:
        return self.instr_name

    def decode(self, decoder: Decoder, addr: int) -> None:
        self.opcode = decoder.unsigned_byte()
        for op in self.operands_coding():
            op.decode(decoder, addr)

    def set_length(self, length: int) -> None:
        self._length = length


    def encode(self, encoder: Encoder, addr: int) -> None:
        assert self.opcode is not None, "Opcode not set"
        if self._pre is not None:
            encoder.unsigned_byte(self._pre)
        encoder.unsigned_byte(self.opcode)
        for op in self.operands_coding():
            op.encode(encoder, addr)

    def fuse(self, sister: 'Instruction') -> Optional['Instruction']:
        return None

    # logical operands order
    def operands(self) -> Generator[Operand, None, None]:
        if self._operands is None:
            yield from ()
        else:
            # three levels of indirection SURELY is enough to handle cases
            # where operands expand to other operands
            for op1 in self._operands:
                for op2 in op1.operands():
                    for op3 in op2.operands():
                        yield op3

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
        dst_mode = get_addressing_mode(self._pre, 1) if self._pre else None
        src_mode = get_addressing_mode(self._pre, 2) if self._pre else None

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

    def display(self, addr: int) -> None:
        print(f"{addr:04X}:\t" + "".join(str(token) for token in self.render()))

    def analyze(self, info: InstructionInfo, addr: int) -> None:
        info.length += self.length()

    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        dst_mode = get_addressing_mode(self._pre, 1) if self._pre else None
        src_mode = get_addressing_mode(self._pre, 2) if self._pre else None

        operands = tuple(self.operands())
        if len(operands) == 0:
            il.append(il.unimplemented())
        else:
            op1 = operands[0].lift(il, dst_mode)
            if len(operands) == 1:
                il_value = self.lift_operation1(il, op1)
            elif len(operands) == 2:
                op2 = operands[1].lift(il, src_mode)
                il_value = self.lift_operation2(il, op1, op2)
            else:
                raise NotImplementedError("lift() not implemented for this instruction")
            operands[0].lift_assign(il, il_value, dst_mode)

    def lift_operation1(self, il: LowLevelILFunction, arg1: ExpressionIndex) -> ExpressionIndex:
        raise NotImplementedError(f"lift_operation1() not implemented for {self.__class__.__name__} instruction")
        return il.unimplemented()

    def lift_operation2(self, il: LowLevelILFunction, arg1: ExpressionIndex, arg2: ExpressionIndex) -> ExpressionIndex:
        raise NotImplementedError(f"lift_operation2() not implemented for {self.__class__.__name__} instruction")
        return il.unimplemented()


class HasWidth:
    def width(self) -> int:
        raise NotImplementedError("width not implemented for HasWidth")


# HasOperands is used to indicate that the operand expects other operands to be
# used instead.
class HasOperands:
    def lift(self, il: LowLevelILFunction, pre: Optional[AddressingMode] = None, side_effects: bool = True) -> ExpressionIndex:
        raise NotImplementedError("lift not implemented for HasOperands")

    def lift_assign(self, il: LowLevelILFunction, value: ExpressionIndex, pre:
                    Optional[AddressingMode] = None) -> None:
        raise NotImplementedError("lift_assign not implemented for HasOperands")


class IMemOperand(Operand, HasWidth):
    def __init__(self, mode: AddressingMode, n: Optional[Union[str, int]] = None):
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
        if self.mode in [AddressingMode.N, AddressingMode.BP_N, AddressingMode.PX_N, AddressingMode.PY_N]:
            assert self.n_val is not None
            encoder.unsigned_byte(cast(int, self.n_val))  # Assumes n_val is already an int

    def lift(self, il: LowLevelILFunction, pre: Optional[AddressingMode] = None, side_effects: bool = True) -> ExpressionIndex:
        return self.helper.lift(il, self.mode, side_effects)

    def lift_assign(self, il: LowLevelILFunction, value: ExpressionIndex, pre: Optional[AddressingMode] = None) -> None:
        self.helper.lift_assign(il, value, self.mode)


class ImmOperand(Operand, HasWidth):
    value: Optional[Union[str, int]]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def lift(self, il: LowLevelILFunction, pre: Optional[AddressingMode] = None, side_effects: bool = True) -> ExpressionIndex:
        assert self.value is not None, "Value not set"
        return il.const(self.width(), self.value)


# n: encoded as `n`
class Imm8(ImmOperand):
    def __init__(self, value: Optional[Union[str, int]] = None) -> None:
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
    def __init__(self, sign: Literal['+', '-']) -> None:
        super().__init__()
        self.sign = sign

    def offset_value(self) -> int:
        assert self.value is not None, "Value not set"
        return -self.value if self.sign == '-' else self.value

    def render(self, pre: Optional[AddressingMode] = None) -> List[Token]:
        return [TInt(f"{self.sign}{self.value:02X}")]

    def lift(self, il: LowLevelILFunction, pre: Optional[AddressingMode] = None, side_effects: bool = True) -> ExpressionIndex:
        raise NotImplementedError("lift() not implemented for ImmOffset")

    def lift_offset(self, il: LowLevelILFunction, value: ExpressionIndex) -> ExpressionIndex:
        offset = il.const(self.width(), self.offset_value())
        return il.add(self.width(), value, offset)


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
        result: List[Token] = [TBegMem(MemType.INTERNAL)]
        match pre:
            case None | AddressingMode.N:
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
        addr = IMEM_NAMES[name]
        return il.load(1, il.const_pointer(3, INTERNAL_MEMORY_START + addr))

    def _imem_offset(self, il: LowLevelILFunction, pre: Optional[AddressingMode]) -> ExpressionIndex:
        n_val: Union[str, int] = 0
        if isinstance(self.value, ImmOperand):
            if self.value.value is not None:
                n_val = self.value.value
        elif isinstance(self.value, IMemOperand):
            if self.value.n_val is not None:
                n_val = self.value.n_val

        n_lifted = il.const(1, int(n_val))

        match pre:
            case None | AddressingMode.N:
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

    def imem_addr(self, il: LowLevelILFunction, pre: Optional[AddressingMode]) -> ExpressionIndex:
        if isinstance(self.value, TempReg):
            if pre is None or pre == AddressingMode.N:
                # The register is assumed to hold the complete address.
                return self.value.lift(il)

        if isinstance(self.value, Reg):
            if pre is None or pre == AddressingMode.N:
                return il.add(3, self.value.lift(il), il.const(3, INTERNAL_MEMORY_START))

        if isinstance(self.value, ImmOperand) and (pre is None or pre == AddressingMode.N):
            assert self.value.value is not None, "Value not set"
            raw_addr = INTERNAL_MEMORY_START + self.value.value
            return il.const_pointer(3, raw_addr)

        offset = self._imem_offset(il, pre)
        return il.add(3, offset, il.const(3, INTERNAL_MEMORY_START))

    def lift(self, il: LowLevelILFunction, pre: Optional[AddressingMode] = None, side_effects: bool = True) -> ExpressionIndex:
        return il.load(self.width(), self.imem_addr(il, pre))

    def lift_assign(self, il: LowLevelILFunction, value: ExpressionIndex, pre:
                    Optional[AddressingMode] = None) -> None:
        assert isinstance(value, (MockLLIL, int)), f"Expected MockLLIL or int, got {type(value)}"
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

    def lift(self, il: LowLevelILFunction, pre: Optional[AddressingMode] = None, side_effects: bool = True) -> ExpressionIndex:
        return il.load(self.width(), self.emem_addr(il),)

    def lift_assign(self, il: LowLevelILFunction, value: ExpressionIndex, pre:
                    Optional[AddressingMode] = None) -> None:
        assert isinstance(value, (MockLLIL, int)), f"Expected MockLLIL or int, got {type(value)}"
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

    def lift(self, il: LowLevelILFunction, pre: Optional[AddressingMode] = None, side_effects: bool = True) -> ExpressionIndex:
        return self._helper().lift(il, pre, side_effects=side_effects)

    def lift_assign(self, il: LowLevelILFunction, value: ExpressionIndex, pre:
                    Optional[AddressingMode] = None) -> None:
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
class Reg(Operand, HasWidth):
    def __init__(self, reg: Any) -> None:
        super().__init__()
        self.reg = reg

    def __repr__(self) -> str:
        return f"Reg(reg={self.reg!r})"

    def render(self, pre: Optional[AddressingMode] = None) -> List[Token]:
        return [TReg(self.reg)]

    def width(self) -> int:
        return REG_SIZES[self.reg]

    def lift(self, il: LowLevelILFunction, pre: Optional[AddressingMode] = None, side_effects: bool = True) -> ExpressionIndex:
        return il.reg(self.width(), self.reg)

    def lift_assign(self, il: LowLevelILFunction, value: ExpressionIndex, pre:
                    Optional[AddressingMode] = None) -> None:
        il.append(il.set_reg(self.width(), self.reg, value))

class TempReg(Operand):
    def __init__(self, reg: Any, width: int=3) -> None:
        super().__init__()
        self.reg = reg
        self._width = width

    def render(self, pre: Optional[AddressingMode] = None) -> List[Token]:
        raise NotImplementedError("render() not implemented for TempReg")

    def width(self) -> int:
        return self._width

    def lift(self, il: LowLevelILFunction, pre: Optional[AddressingMode] = None, side_effects: bool = True) -> ExpressionIndex:
        return il.reg(self.width(), self.reg)

    def lift_assign(self, il: LowLevelILFunction, value: ExpressionIndex, pre:
                    Optional[AddressingMode] = None) -> None:
        il.append(il.set_reg(self.width(), self.reg, value))

# only makes sense for PUSHU / POPU
class RegIMR(HasOperands, Reg):
    def __init__(self) -> None:
        super().__init__("IMR")

    def width(self) -> int:
        return 1

    def operands(self) -> Generator[Operand, None, None]:
        yield IMem8(IMEM_NAMES["IMR"])

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

    def lift(self, il: LowLevelILFunction, pre: Optional[AddressingMode] = None, side_effects: bool = True) -> ExpressionIndex:
        zbit = il.shift_left(1, il.flag(ZFlag), il.const(1, 1))
        return il.or_expr(1, il.flag(CFlag), zbit)

    def lift_assign(self, il: LowLevelILFunction, value: ExpressionIndex, pre:
                    Optional[AddressingMode] = None) -> None:
        tmp = TempReg(TempRegF, width=self.width())
        tmp.lift_assign(il, value)
        il.append(il.set_flag(CFlag, il.and_expr(1, tmp.lift(il), il.const(1, 1))))
        il.append(il.set_flag(ZFlag, il.and_expr(1, tmp.lift(il), il.const(1, 2))))

class Reg3(Operand, HasWidth):
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
            assert self.width() >= 3, f"Want r3 register, got r{self.width()} ({self.reg}) instead"
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

    def lift(self, il: LowLevelILFunction, pre: Optional[AddressingMode] = None, side_effects: bool = True) -> ExpressionIndex:
        assert self.reg is not None, "Register not set"
        return il.reg(self.width(), self.reg)

    def lift_assign(self, il: LowLevelILFunction, value: ExpressionIndex, pre:
                    Optional[AddressingMode] = None) -> None:
        assert self.reg is not None, "Register not set"
        il.append(il.set_reg(self.width(), self.reg, value))

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
        return [TBegMem(MemType.EXTERNAL), TInt(f"{self.value:05X}"), TEndMem(MemType.EXTERNAL)]

    def lift(self, il: LowLevelILFunction, pre: Optional[AddressingMode] = None, side_effects: bool = True) -> ExpressionIndex:
        assert self.value is not None, "Value not set"
        return il.load(self.width(), il.const_pointer(3, self.value))

    def lift_assign(self, il: LowLevelILFunction, value: ExpressionIndex, pre:
                    Optional[AddressingMode] = None) -> None:
        assert self.value is not None, "Value not set"
        # FIXME: should use width of value
        il.append(il.store(self.width(), il.const_pointer(3, self.value), value))


class EMemValueOffsetHelper(OperandHelper, Pointer):
    def __init__(self, value: Operand, offset: Optional[ImmOffset], width: int = 1) -> None:
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
        addr = self.value.lift(il, pre=pre, side_effects=side_effects)
        if self.offset:
            addr = self.offset.lift_offset(il, addr)
        return addr

    def memory_helper(self) -> Type[EMemHelper]:
        return EMemHelper

    def render(self, pre: Optional[AddressingMode] = None) -> List[Token]:
        result: List[Token] = [TBegMem(MemType.EXTERNAL)]
        result.extend(self.value.render(pre)) # Pass pre to render if self.value supports it
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
        return il.load(self.width(), self.lift_current_addr(il, pre=pre, side_effects=side_effects))

    def lift_assign(
        self,
        il: LowLevelILFunction,
        value: ExpressionIndex,
        pre: Optional[AddressingMode] = None,
    ) -> None:
        il.append(
            il.store(self.width(), self.lift_current_addr(il, pre=pre, side_effects=True), value)
        )

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
        raise InvalidInstruction(f"Invalid EMemRegMode {val:02X} at address {addr:#06x}")

class RegIncrementDecrementHelper(OperandHelper):
    def __init__(self, width: int, reg: Reg3, mode: EMemRegMode) -> None:
        super().__init__()
        self.width = width # This width is the increment/decrement amount, typically data size (1, 2, or 3)
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

    def lift(self, il: LowLevelILFunction, pre: Optional[AddressingMode] = None, side_effects: bool = True) -> ExpressionIndex:
        value = self.reg.lift(il)

        if side_effects and self.mode == EMemRegMode.POST_INC:
            # create LLIL_TEMP to hold the value since we're supposed to
            # increment it after using it
            tmp = TempReg(TempIncDecHelper, width=self.reg.width())
            tmp.lift_assign(il, value)
            self.reg.lift_assign(il, il.add(self.reg.width(), value, il.const(1,
                                                                              self.width)))
            value = tmp.lift(il)

        if self.mode == EMemRegMode.PRE_DEC:
            value = il.sub(self.reg.width(), value, il.const(1, self.width))
            if side_effects:
                self.reg.lift_assign(il, value)

        return value


class EMemRegOffsetHelper(HasOperands, OperandHelper):
    def __init__(self, width: int, reg: Reg3, mode: EMemRegMode, offset: Optional[ImmOffset]) -> None:
        super().__init__()
        self.width = width
        self.reg = reg
        self.mode = mode
        self.offset = offset

    def operands(self) -> Generator[Operand, None, None]:
        reg: Operand
        if self.mode in (EMemRegMode.SIMPLE,
                         EMemRegMode.POST_INC,
                         EMemRegMode.PRE_DEC):
            reg = RegIncrementDecrementHelper(self.width, self.reg, self.mode)
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
class RegIMemOffset(HasOperands, Operand):
    reg: Optional[Reg3]
    imem: Optional[IMem8]
    mode: Optional[EMemRegMode]
    offset: Optional[ImmOffset] = None

    def __init__(self, order: RegIMemOffsetOrder, allowed_modes:
                 Optional[List[EMemRegMode]] = None) -> None:
        self.order = order
        self.allowed_modes = allowed_modes

    def operands(self) -> Generator[Operand, None, None]:
        assert self.reg is not None, "Register not set"
        assert self.imem is not None, "IMem not set"
        assert self.mode is not None, "Mode not set"
        # FIXME: is width=1 here?
        op = EMemRegOffsetHelper(1, self.reg, self.mode, self.offset)
        if self.order == RegIMemOffsetOrder.DEST_REG_OFFSET:
            yield op
            yield self.imem
        else:
            yield self.imem
            yield op

    def decode(self, decoder: Decoder, addr: int) -> None:
        super().decode(decoder, addr)
        self.reg = Reg3()
        self.reg.decode(decoder, addr)
        self.reg.assert_r3()

        self.imem = IMem8()
        self.imem.decode(decoder, addr)

        self.mode = get_emem_reg_mode(self.reg.high4, addr)
        if self.allowed_modes is not None:
            assert self.mode in self.allowed_modes

        if self.mode in (EMemRegMode.POSITIVE_OFFSET, EMemRegMode.NEGATIVE_OFFSET):
            self.offset = ImmOffset('+' if self.mode == EMemRegMode.POSITIVE_OFFSET else '-')
            self.offset.decode(decoder, addr)

    def encode(self, encoder: Encoder, addr: int) -> None:
        super().encode(encoder, addr)
        assert self.reg is not None, "Register not set"
        self.reg.encode(encoder, addr)
        assert self.imem is not None, "IMem not set"
        self.imem.encode(encoder, addr)
        if self.offset:
            self.offset.encode(encoder, addr)

class EMemReg(HasOperands, Operand):
    mode: Optional[EMemRegMode]
    offset: Optional[ImmOffset] = None

    def __init__(self, width: int, allowed_modes: Optional[List[EMemRegMode]]=None) -> None:
        super().__init__()
        self.width = width
        self.reg = Reg3()
        self.allowed_modes = allowed_modes

    def decode(self, decoder: Decoder, addr: int) -> None:
        super().decode(decoder, addr)
        self.reg.decode(decoder, addr)
        self.reg.assert_r3()
        self.mode = get_emem_reg_mode(self.reg.high4, addr)
        if self.allowed_modes is not None:
            assert self.mode in self.allowed_modes, f"Invalid mode: {self.mode}, allowed: {self.allowed_modes}"

        if self.mode in (EMemRegMode.POSITIVE_OFFSET, EMemRegMode.NEGATIVE_OFFSET):
            self.offset = ImmOffset('+' if self.mode == EMemRegMode.POSITIVE_OFFSET else '-')
            self.offset.decode(decoder, addr)

    def encode(self, encoder: Encoder, addr: int) -> None:
        # super().encode(encoder, addr)
        self.reg.encode(encoder, addr)
        if self.offset:
            self.offset.encode(encoder, addr)

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

class EMemIMem(HasOperands, Imm8):
    mode: Optional[EMemIMemMode]
    offset: Optional[ImmOffset] = None

    def __init__(self) -> None:
        super().__init__()
        self.imem = IMem8()

    def decode(self, decoder: Decoder, addr: int) -> None:
        super().decode(decoder, addr)
        self.imem.decode(decoder, addr)

        self.mode = get_emem_imem_mode(self.value, addr)
        if self.mode in (EMemIMemMode.POSITIVE_OFFSET, EMemIMemMode.NEGATIVE_OFFSET):
            self.offset = ImmOffset('+' if self.mode == EMemIMemMode.POSITIVE_OFFSET else '-')
            self.offset.decode(decoder, addr)

    def encode(self, encoder: Encoder, addr: int) -> None:
        super().encode(encoder, addr)
        self.imem.encode(encoder, addr)

        if self.offset:
            self.offset.encode(encoder, addr)

    def operands(self) -> Generator[Operand, None, None]:
        op = EMemValueOffsetHelper(self.imem, self.offset, width=1)
        yield op

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
class EMemIMemOffset(HasOperands, Operand):
    mode: Optional[EMemIMemMode]
    offset: Optional[ImmOffset] = None

    def __init__(self, order: EMemIMemOffsetOrder) -> None:
        self.order = order
        self.mode_imm = Imm8()
        self.imem1 = IMem8()
        self.imem2 = IMem8()

    def operands(self) -> Generator[Operand, None, None]:
        if self.order == EMemIMemOffsetOrder.DEST_INT_MEM:
            yield self.imem1
            op = EMemValueOffsetHelper(self.imem2, self.offset, width=1)
            yield op
        else:
            op = EMemValueOffsetHelper(self.imem1, self.offset, width=1)
            yield op
            yield self.imem2

    def decode(self, decoder: Decoder, addr: int) -> None:
        super().decode(decoder, addr)
        self.mode_imm = Imm8()
        self.mode_imm.decode(decoder, addr)

        self.imem1 = IMem8()
        self.imem1.decode(decoder, addr)

        self.imem2 = IMem8()
        self.imem2.decode(decoder, addr)

        self.mode = get_emem_imem_mode(self.mode_imm.value, addr)
        if self.mode in (EMemIMemMode.POSITIVE_OFFSET, EMemIMemMode.NEGATIVE_OFFSET):
            self.offset = ImmOffset('+' if self.mode == EMemIMemMode.POSITIVE_OFFSET else '-')
            self.offset.decode(decoder, addr)

    def encode(self, encoder: Encoder, addr: int) -> None:
        super().encode(encoder, addr)
        self.mode_imm.encode(encoder, addr)
        self.imem1.encode(encoder, addr)
        self.imem2.encode(encoder, addr)
        if self.offset:
            self.offset.encode(encoder, addr)


# ADD/SUB can use various-sized register pairs
class RegPair(HasOperands, Reg3):
    reg_raw: Optional[int]
    reg1: Optional[Reg]
    reg2: Optional[Reg]

    def __init__(self, size: Optional[int]=None) -> None:
        super().__init__()
        self.size = size

    def decode(self, decoder: Decoder, addr: int) -> None:
        self.reg_raw = decoder.unsigned_byte()
        self.reg1 = Reg(REG_NAMES[(self.reg_raw >> 4) & 7])
        self.reg2 = Reg(REG_NAMES[self.reg_raw & 7])

        try:
            # high-bits of both halves must be zero: 0x80 and 0x08 must not be set
            assert (self.reg_raw & 0x80) == 0, f"Invalid reg1 high bit: {self.reg_raw:02X}"
            assert (self.reg_raw & 0x08) == 0, f"Invalid reg2 high bit: {self.reg_raw:02X}"
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
    il.mark_label(if_false)

    # loop iteration
    yield

    width = loop_reg.width()
    loop_reg.lift_assign(il, il.sub(width, loop_reg.lift(il), il.const(1, 1)))
    cond = il.compare_equal(width, loop_reg.lift(il), il.const(width, 0))
    il.append(il.if_expr(cond, if_true, if_false))
    il.mark_label(if_true)


class NOP(Instruction):
     def lift(self, il: LowLevelILFunction, addr: int) -> None:
        il.append(il.nop())

class JumpInstruction(Instruction):
    def lift_jump_addr(self, il: LowLevelILFunction, addr: int) -> ExpressionIndex:
        raise NotImplementedError("lift_jump_addr() not implemented")

    def analyze(self, info: InstructionInfo, addr: int) -> None:
        super().analyze(info, addr)
        # expect TrueBranch to be handled by subclasses as it might require
        # llil logic to calculate the address
        info.add_branch(BranchType.FalseBranch, addr + self.length())


    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        if_true  = LowLevelILLabel()
        if_false = LowLevelILLabel()

        if self._cond:
            zero = il.const(1, 0)
            one  = il.const(1, 1)
            flag = il.flag(ZFlag) if "Z" in self._cond else il.flag(CFlag)
            value = zero if "N" in self._cond else one

            cond = il.compare_equal(1, flag, value)
            il.append(il.if_expr(cond, if_true, if_false))

        il.mark_label(if_true)
        il.append(il.jump(self.lift_jump_addr(il, addr)))
        il.mark_label(if_false)


class JP_Abs(JumpInstruction):
    def name(self) -> str:
        return super().name() + (self._cond if self._cond else "")

    def lift_jump_addr(self, il: LowLevelILFunction, addr: int) -> ExpressionIndex:
        first, *rest = self.operands()
        assert len(rest) == 0, "Expected no extra operands"
        assert isinstance(first, HasWidth), f"Expected HasWidth, got {type(first)}"
        if first.width() >= 3:
            return first.lift(il)
        high_addr = addr & 0xFF0000
        return il.or_expr(3, first.lift(il), il.const(3, high_addr))

    def analyze(self, info: InstructionInfo, addr: int) -> None:
        super().analyze(info, addr)

        first, *rest = self.operands()
        assert len(rest) == 0, "Expected no extra operands"
        if isinstance(first, ImmOperand):
            # absolute address
            assert first.value is not None, "Value not set"
            dest = first.value
            info.add_branch(BranchType.TrueBranch, dest)

class JP_Rel(JumpInstruction):
    def name(self) -> str:
        return "JR" + (self._cond if self._cond else "")

    def lift_jump_addr(self, il: LowLevelILFunction, addr: int) -> ExpressionIndex:
        first, *rest = self.operands()
        assert len(rest) == 0, "Expected no extra operands"
        assert isinstance(first, ImmOffset), f"Expected ImmOffset, got {type(first)}"
        return il.const(3, addr + self.length() + first.offset_value())

    def analyze(self, info: InstructionInfo, addr: int) -> None:
        super().analyze(info, addr)
        first, *rest = self.operands()
        assert len(rest) == 0, "Expected no extra operands"
        assert isinstance(first, ImmOffset), f"Expected ImmOffset, got {type(first)}"
        dest = addr + self.length() + first.offset_value()
        info.add_branch(BranchType.TrueBranch, dest)

class CALL(Instruction):
    def _dest(self) -> ImmOperand:
        dest, *rest = self.operands()
        assert len(rest) == 0, "Expected no extra operands"
        assert isinstance(dest, ImmOperand), "Expected ImmOperand"
        return dest

    def dest_addr(self, addr: int) -> int:
        dest = self._dest()
        result = dest.value
        assert result is not None, "Value not set"
        if dest.width() != 3:
            assert dest.width() == 2
            result = addr & 0xFF0000 | result
        return result

    def analyze(self, info: InstructionInfo, addr: int) -> None:
        super().analyze(info, addr)
        info.add_branch(BranchType.CallDestination, self.dest_addr(addr))

    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        dest = self._dest()
        if dest.width() == 3:
            il.append(il.call(il.const_pointer(3, self.dest_addr(addr))))
            return
        # manually push 2 bytes of address + self.length()
        il.append(il.push(2, il.const(2, addr + self.length())))
        il.append(il.jump(il.const_pointer(3, self.dest_addr(addr))))


class RetInstruction(Instruction):
    def addr_size(self) -> int:
        return 2

    def analyze(self, info: InstructionInfo, addr: int) -> None:
        super().analyze(info, addr)
        info.add_branch(BranchType.FunctionReturn)

    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        # FIXME: should add bitmask for 2-byte pop?
        il.append(il.ret(il.pop(self.addr_size())))

class RET(RetInstruction): pass
class RETF(RetInstruction):
    def addr_size(self) -> int:
        return 3
class RETI(RetInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        imr, *rest = RegIMR().operands()
        imr.lift_assign(il, il.pop(1))
        RegF().lift_assign(il, il.pop(1))
        il.append(il.ret(il.pop(3)))


class MoveInstruction(Instruction):
    pass

class MV(MoveInstruction):
    def encode(self, encoder: Encoder, addr: int) -> None:
        # Handle special case for imem-to-imem moves that need a PRE byte
        op1, op2 = self.operands()
        if isinstance(op1, IMemOperand) and isinstance(op2, IMemOperand):
            pre_key = (op1.mode, op2.mode)
            pre_byte = REVERSE_PRE_TABLE.get(pre_key)
            if pre_byte is None:
                raise ValueError(f"Invalid addressing mode combination for MV: {op1.mode.value} and {op2.mode.value}")
            self._pre = pre_byte
            self.opcode = 0xC8 # Base opcode for MV (m),(n)
            # Fall through to the generic Instruction.encode

        super().encode(encoder, addr)

    def lift_operation2(self, il: LowLevelILFunction, il_arg1: ExpressionIndex, il_arg2: ExpressionIndex) -> ExpressionIndex:
        return il_arg2

class MVL(MoveInstruction):
    def modify_addr_il(self, il: LowLevelILFunction) -> Callable[[int, ExpressionIndex, ExpressionIndex], ExpressionIndex]:
        return il.add

    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        dst, src = self.operands()
        assert isinstance(dst, Pointer), f"Expected Pointer, got {type(dst)}"
        assert isinstance(src, Pointer), f"Expected Pointer, got {type(src)}"
        # 0xCB and 0xCF variants use IMem8, IMem8
        dst_reg = TempReg(TempMvlDst)
        dst_mode = get_addressing_mode(self._pre, 1) if self._pre else None
        src_mode = get_addressing_mode(self._pre, 2) if self._pre else None

        dst_reg.lift_assign(
            il, dst.lift_current_addr(il, pre=dst_mode, side_effects=False)
        )
        src_reg = TempReg(TempMvlSrc)
        src_reg.lift_assign(
            il, src.lift_current_addr(il, pre=src_mode, side_effects=False)
        )

        with lift_loop(il):
            src_mem = src.memory_helper()(1, src_reg)
            dst_mem = dst.memory_helper()(1, dst_reg)
            dst_mem.lift_assign(il, src_mem.lift(il))

            # +1 index
            func = self.modify_addr_il(il)
            dst_reg.lift_assign(il, func(dst_reg.width(), dst_reg.lift(il), il.const(1, 1)))
            src_reg.lift_assign(il, func(src_reg.width(), src_reg.lift(il), il.const(1, 1)))

            # in case we have POST_INC or PRE_DEC, we need to update the
            # register by lifting it and not assigning it
            dst.lift_current_addr(il, pre=dst_mode)
            src.lift_current_addr(il, pre=src_mode)

class MVLD(MVL):
    def modify_addr_il(self, il: LowLevelILFunction) -> Callable[[int, ExpressionIndex, ExpressionIndex], ExpressionIndex]:
        return il.sub

class PRE(Instruction):
    def name(self) -> str:
        return f"PRE{self.opcode:02x}"

    def fuse(self, sister: 'Instruction') -> Optional['Instruction']:
        if isinstance(sister, PRE):
            return None
        sister._pre = self.opcode
        sister.set_length(self.length() + sister.length())
        return sister

class StackInstruction(Instruction):
    def reg(self) -> Operand:
        r, *rest = self.operands()
        assert len(rest) == 0, "Expected no extra operands"
        return r
class StackPushInstruction(StackInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        r = self.reg()
        assert isinstance(r, HasWidth), f"Expected HasWidth, got {type(r)}"
        il.append(il.push(r.width(), r.lift(il)))
class StackPopInstruction(StackInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        r = self.reg()
        assert isinstance(r, HasWidth), f"Expected HasWidth, got {type(r)}"
        r.lift_assign(il, il.pop(r.width()))

# FIXME: should use U pointer, not S
class PUSHU(StackInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        r = self.reg()
        assert isinstance(r, HasWidth)
        size = r.width()
        # save the original U so the store uses the pre-decremented value
        old_u = TempReg(TempIncDecHelper, width=3)
        old_u.lift_assign(il, il.reg(3, RegisterName("U")))
        new_u = il.sub(3, old_u.lift(il), il.const(3, size))
        il.append(il.set_reg(3, RegisterName("U"), new_u))
        il.append(il.store(size, new_u, r.lift(il)))
        if isinstance(r, RegIMR):
            r.lift_assign(il, il.and_expr(1, r.lift(il), il.const(1, 0x7F)))

class POPU(StackInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        r = self.reg()
        assert isinstance(r, HasWidth)
        size = r.width()
        # preserve the pointer prior to increment so the load happens at
        # the original U value
        old_u = TempReg(TempIncDecHelper, width=3)
        old_u.lift_assign(il, il.reg(3, RegisterName("U")))
        r.lift_assign(il, il.load(size, old_u.lift(il)))
        il.append(
            il.set_reg(3, RegisterName("U"), il.add(3, old_u.lift(il), il.const(3, size)))
        )

class PUSHS(StackPushInstruction): pass
class POPS(StackPopInstruction): pass

class ArithmeticInstruction(Instruction):
    def width(self) -> int:
        first, second = self.operands()
        assert isinstance(first, HasWidth), f"Expected HasWidth, got {type(first)}"
        return first.width()
class ADD(ArithmeticInstruction):
    def lift_operation2(self, il: LowLevelILFunction, il_arg1: ExpressionIndex, il_arg2: ExpressionIndex) -> ExpressionIndex:
        return il.add(self.width(), il_arg1, il_arg2, CZFlag)
class ADC(ArithmeticInstruction):
    def lift_operation2(self, il: LowLevelILFunction, il_arg1: ExpressionIndex, il_arg2: ExpressionIndex) -> ExpressionIndex:
        return il.add(self.width(), il_arg1, il.add(self.width(), il_arg2,
                                                    il.flag(CFlag)), CZFlag)
class SUB(ArithmeticInstruction):
    def lift_operation2(self, il: LowLevelILFunction, il_arg1: ExpressionIndex, il_arg2: ExpressionIndex) -> ExpressionIndex:
        return il.sub(self.width(), il_arg1, il_arg2, CZFlag)
class SBC(ArithmeticInstruction):
    def lift_operation2(self, il: LowLevelILFunction, il_arg1: ExpressionIndex, il_arg2: ExpressionIndex) -> ExpressionIndex:
        return il.sub(self.width(), il_arg1, il.add(self.width(), il_arg2,
                                                    il.flag(CFlag)), CZFlag)


def bcd_add_emul(il: LowLevelILFunction, w: int, a: ExpressionIndex, b: ExpressionIndex) -> Operand:
    assert w == 1, "BCD add currently only supports 1-byte operands"

    # Incoming CFlag is the BCD carry from the previous byte's BCD addition
    incoming_carry = il.flag(CFlag)

    # Low nibble addition: (a & 0xF) + (b & 0xF) + incoming_carry_byte
    a_low = il.and_expr(1, a, il.const(1, 0x0F))
    b_low = il.and_expr(1, b, il.const(1, 0x0F))
    sum_low_nibbles_val = il.add(1, a_low, b_low)
    sum_low_with_carry_val = il.add(1, sum_low_nibbles_val, incoming_carry) # Max val 9+9+1 = 19 (0x13)

    # Adjust if low nibble sum > 9
    temp_sum_low_final_reg = TempReg(TempBcdLowNibbleProcessing, width=1)
    adj_low_needed = il.compare_unsigned_greater_than(1, sum_low_with_carry_val, il.const(1, 9))
    sum_low_adjusted_val = il.add(1, sum_low_with_carry_val, il.const(1, 0x06))

    label_adj_low_true = LowLevelILLabel()
    label_adj_low_false = LowLevelILLabel()
    label_after_adj_low = LowLevelILLabel()
    il.append(il.if_expr(adj_low_needed, label_adj_low_true, label_adj_low_false))
    il.mark_label(label_adj_low_true)
    temp_sum_low_final_reg.lift_assign(il, sum_low_adjusted_val)
    il.append(il.goto(label_after_adj_low))
    il.mark_label(label_adj_low_false)
    temp_sum_low_final_reg.lift_assign(il, sum_low_with_carry_val)
    il.mark_label(label_after_adj_low)

    current_sum_low_final = temp_sum_low_final_reg.lift(il)
    result_low_nibble_val = il.and_expr(1, current_sum_low_final, il.const(1, 0x0F))
    carry_to_high_nibble_val = il.logical_shift_right(1, current_sum_low_final, il.const(1, 4)) # 0 or 1

    # High nibble addition: (a >> 4) + (b >> 4) + carry_to_high_nibble_val
    a_high = il.logical_shift_right(1, a, il.const(1, 4))
    b_high = il.logical_shift_right(1, b, il.const(1, 4))
    sum_high_nibbles_val = il.add(1, a_high, b_high)
    sum_high_with_carry_val = il.add(1, sum_high_nibbles_val, carry_to_high_nibble_val) # Max 9+9+1 = 19 (0x13)

    # Adjust if high nibble sum > 9
    temp_sum_high_final_reg = TempReg(TempBcdHighNibbleProcessing, width=1)
    adj_high_needed = il.compare_unsigned_greater_than(1, sum_high_with_carry_val, il.const(1, 9))
    sum_high_adjusted_val = il.add(1, sum_high_with_carry_val, il.const(1, 0x06))

    label_adj_high_true = LowLevelILLabel()
    label_adj_high_false = LowLevelILLabel()
    label_after_adj_high = LowLevelILLabel()
    il.append(il.if_expr(adj_high_needed, label_adj_high_true, label_adj_high_false))
    il.mark_label(label_adj_high_true)
    temp_sum_high_final_reg.lift_assign(il, sum_high_adjusted_val)
    il.append(il.goto(label_after_adj_high))
    il.mark_label(label_adj_high_false)
    temp_sum_high_final_reg.lift_assign(il, sum_high_with_carry_val)
    il.mark_label(label_after_adj_high)

    current_sum_high_final = temp_sum_high_final_reg.lift(il)
    result_high_nibble_val = il.and_expr(1, current_sum_high_final, il.const(1, 0x0F))
    new_bcd_carry_out_byte_val = il.logical_shift_right(1, current_sum_high_final, il.const(1, 4)) # 0 or 1

    result_byte_val = il.or_expr(1, il.shift_left(1, result_high_nibble_val, il.const(1, 4)), result_low_nibble_val)

    output_reg = TempReg(TempBcdAddEmul, width=1)
    output_reg.lift_assign(il, result_byte_val)
    il.append(il.set_flag(CFlag, new_bcd_carry_out_byte_val))
    # Z flag for current byte (overall Z handled by lift_multi_byte)
    il.append(il.set_flag(ZFlag, il.compare_equal(1, result_byte_val, il.const(1,0))))

    return output_reg

def bcd_sub_emul(il: LowLevelILFunction, w: int, a: ExpressionIndex, b: ExpressionIndex) -> Operand:
    assert w == 1, "BCD sub currently only supports 1-byte operands"

    incoming_borrow = il.flag(CFlag) # 0 for no borrow, 1 for borrow

    # Low nibble subtraction: (a_low) - (b_low) - incoming_borrow
    a_low = il.and_expr(1, a, il.const(1, 0x0F))
    b_low = il.and_expr(1, b, il.const(1, 0x0F))

    sub_val_low = il.add(1, b_low, incoming_borrow) # bL + Cin
    temp_sub_low_val = il.sub(1, a_low, sub_val_low)

    # Check for borrow from low nibble
    borrow_from_low_val = il.compare_signed_less_than(1, temp_sub_low_val, il.const(1, 0))

    final_low_nibble_reg = TempReg(TempBcdLowNibbleProcessing, width=1)
    adj_val_low = il.sub(1, temp_sub_low_val, il.const(1, 0x06)) # Subtract 6 if borrow

    label_adj_low_true_s = LowLevelILLabel()
    label_adj_low_false_s = LowLevelILLabel()
    label_after_adj_low_s = LowLevelILLabel()
    il.append(il.if_expr(borrow_from_low_val, label_adj_low_true_s, label_adj_low_false_s))
    il.mark_label(label_adj_low_true_s)
    final_low_nibble_reg.lift_assign(il, adj_val_low)
    il.append(il.goto(label_after_adj_low_s))
    il.mark_label(label_adj_low_false_s)
    final_low_nibble_reg.lift_assign(il, temp_sub_low_val)
    il.mark_label(label_after_adj_low_s)

    result_low_nibble_val = il.and_expr(1, final_low_nibble_reg.lift(il), il.const(1, 0x0F))

    # High nibble subtraction: (a_high) - (b_high) - borrow_from_low_val
    a_high = il.logical_shift_right(1, a, il.const(1, 4))
    b_high = il.logical_shift_right(1, b, il.const(1, 4))

    sub_val_high = il.add(1, b_high, borrow_from_low_val) # bH + borrow_low
    temp_sub_high_val = il.sub(1, a_high, sub_val_high)

    new_bcd_borrow_out_byte_val = il.compare_signed_less_than(1, temp_sub_high_val, il.const(1, 0))
    final_high_nibble_reg = TempReg(TempBcdHighNibbleProcessing, width=1)
    adj_val_high = il.sub(1, temp_sub_high_val, il.const(1, 0x06))

    label_adj_high_true_s = LowLevelILLabel()
    label_adj_high_false_s = LowLevelILLabel()
    label_after_adj_high_s = LowLevelILLabel()
    il.append(il.if_expr(new_bcd_borrow_out_byte_val, label_adj_high_true_s, label_adj_high_false_s))
    il.mark_label(label_adj_high_true_s)
    final_high_nibble_reg.lift_assign(il, adj_val_high)
    il.append(il.goto(label_after_adj_high_s))
    il.mark_label(label_adj_high_false_s)
    final_high_nibble_reg.lift_assign(il, temp_sub_high_val)
    il.mark_label(label_after_adj_high_s)

    result_high_nibble_val = il.and_expr(1, final_high_nibble_reg.lift(il), il.const(1, 0x0F))
    result_byte_val = il.or_expr(1, il.shift_left(1, result_high_nibble_val, il.const(1, 4)), result_low_nibble_val)

    output_reg = TempReg(TempBcdSubEmul, width=1)
    output_reg.lift_assign(il, result_byte_val)
    il.append(il.set_flag(CFlag, new_bcd_borrow_out_byte_val)) # C=1 if borrow
    il.append(il.set_flag(ZFlag, il.compare_equal(1, result_byte_val, il.const(1,0))))

    return output_reg


def lift_multi_byte(
    il: LowLevelILFunction,
    op1: Operand,
    op2: Operand,
    clear_carry: bool = False,
    reverse: bool = False,
    bcd: bool = False,
    subtract: bool = False,
    pre: Optional[int] = None,
) -> None:
    assert isinstance(op1, HasWidth), f"Expected HasWidth, got {type(op1)}"

    dst_mode = get_addressing_mode(pre, 1) if pre else None
    src_mode = get_addressing_mode(pre, 2) if pre else None

    # Helper to create load/store/advance logic for operands
    def make_handlers(
        op: Operand,
        is_dest_op: bool,
        mode: Optional[AddressingMode],
    ) -> Tuple[Callable[[], ExpressionIndex],
                     Callable[[ExpressionIndex], None],
                     Callable[[], None]]:
        if isinstance(op, Pointer):
            # Temp reg to hold the iterating pointer for memory operands
            ptr_temp_reg_const = TempMultiByte1 if is_dest_op else TempMultiByte2
            ptr = TempReg(ptr_temp_reg_const, width=3) # Addresses are 3 bytes (20/24 bit)

            # Initialize the pointer temp reg with the initial address from the operand
            # side_effects=False for source, potentially True for dest if pre/post inc/dec
            ptr.lift_assign(
                il,
                op.lift_current_addr(il, pre=mode, side_effects=is_dest_op),
            )

            def load() -> ExpressionIndex:
                # Use width 'w' (e.g. 1 for byte) for memory load/store element size
                assert isinstance(op, Pointer)
                return op.memory_helper()(w, ptr).lift(il)
            def store(val: ExpressionIndex) -> None:
                assert isinstance(op, Pointer)
                op.memory_helper()(w, ptr).lift_assign(il, val)
            def advance() -> None:
                op_il_math = il.sub if reverse else il.add
                # Advance pointer by element width 'w'
                ptr.lift_assign(il, op_il_math(3, ptr.lift(il), il.const(3, w))) # ptr is 3 bytes
        else: # Register operand
            def load() -> ExpressionIndex:
                return op.lift(il)
            def store(val: ExpressionIndex) -> None:
                op.lift_assign(il, val)
            def advance() -> None: # No advancement for direct register operands in a loop
                pass
        return load, store, advance

    w = op1.width()

    load1, store1, adv1 = make_handlers(op1, True, dst_mode)
    load2, store2, adv2 = make_handlers(op2, False, src_mode)

    if clear_carry:
        il.append(il.set_flag(CFlag, il.const(1, 0)))

    overall_zero_acc_reg = TempReg(TempOverallZeroAcc, width=w)
    overall_zero_acc_reg.lift_assign(il, il.const(w, 0))

    # TempReg to store the result of the current byte's main arithmetic operation
    byte_op_result_holder = TempReg(TempLoopByteResult, width=w)

    with lift_loop(il): # loop_reg is 'I', controls number of iterations (bytes)
        a = load1() # ExpressionIndex for current byte of op1
        b = load2() # ExpressionIndex for current byte of op2

        # This will hold the evaluated result of the current byte's operation
        # before it's stored or used in overall_zero_acc.
        current_byte_calculated_value_expr: ExpressionIndex

        if bcd:
            # BCD operations are complex; they read il.flag(CFlag) internally for incoming carry,
            # perform BCD arithmetic, set CFlag and ZFlag (for the byte) based on BCD logic,
            # and return an Operand (specifically a TempReg like TempBcdAddEmul or TempBcdSubEmul)
            # which holds the BCD result of the current byte.
            bcd_op_result_operand: Operand
            if subtract: # DSBL
                bcd_op_result_operand = bcd_sub_emul(il, w, a, b)
            else: # DADL
                bcd_op_result_operand = bcd_add_emul(il, w, a, b)

            # The expression for the result of this byte's BCD operation
            current_byte_calculated_value_expr = bcd_op_result_operand.lift(il)
            # No need to assign to byte_op_result_holder if flags are fully set by bcd_emul
            # and result is self-contained in its returned TempReg.
            # The flags (C and Z for the byte) are set by set_flag calls within bcd_xxx_emul.
        else: # Binary: ADCL, SBCL
            # These operations use il.flag(CFlag) for incoming carry and set CZFlag for outgoing.
            main_op_llil: ExpressionIndex

            # Capture the incoming carry flag *before* this byte's main operation
            initial_c_flag_expr = il.flag(CFlag)

            if subtract: # SBCL: m = m - n - C_in. Implemented as m - (n + C_in)
                         # The inner add (n + C_in) must NOT alter flags.
                term_to_subtract = il.add(w, b, initial_c_flag_expr)
                main_op_llil = il.sub(w, a, term_to_subtract, CZFlag) # This SUB sets C and Z flags
            else: # ADCL: m = m + n + C_in. Implemented as m + (n + C_in)
                  # The inner add (n + C_in) must NOT alter flags.
                term_to_add = il.add(w, b, initial_c_flag_expr)
                main_op_llil = il.add(w, a, term_to_add, CZFlag) # This ADD sets C and Z flags

            # Execute the main operation and store its result in byte_op_result_holder.
            # The flags (C and Z) are set when main_op_llil is evaluated as part of this set_reg.
            byte_op_result_holder.lift_assign(il, main_op_llil)
            current_byte_calculated_value_expr = byte_op_result_holder.lift(il) # = REG(TempLoopByteResult)

        # Store the result for the current byte using the calculated value
        store1(current_byte_calculated_value_expr)

        # Accumulate for overall Zero flag check. This OR must not affect C/Z flags.
        overall_zero_acc_reg.lift_assign(il, il.or_expr(w, overall_zero_acc_reg.lift(il), current_byte_calculated_value_expr))

        adv1()
        adv2()

    # After loop, set the final Zero flag based on the accumulator
    il.append(il.set_flag(ZFlag, il.compare_equal(w, overall_zero_acc_reg.lift(il), il.const(w, 0))))
    # The Carry flag (FC) will hold the carry/borrow from the last byte's operation.


class ADCL(ArithmeticInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        dst, src = self.operands()
        # ADCL uses the incoming carry flag for the first byte.
        lift_multi_byte(il, dst, src, clear_carry=False, pre=self._pre)

class SBCL(ArithmeticInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        dst, src = self.operands()
        # SBCL uses the incoming carry (borrow) flag for the first byte.
        lift_multi_byte(il, dst, src, subtract=True, clear_carry=False, pre=self._pre)

class DADL(ArithmeticInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        dst, src = self.operands()
        # DADL does not use incoming carry for the first byte (implicitly 0).
        lift_multi_byte(il, dst, src, clear_carry=True, bcd=True, reverse=True, pre=self._pre)

class DSBL(ArithmeticInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        dst, src = self.operands()
        # DSBL uses the incoming carry (borrow) flag for the first byte.
        lift_multi_byte(
            il,
            dst,
            src,
            bcd=True,
            subtract=True,
            reverse=True,
            clear_carry=False,
            pre=self._pre,
        )


class LogicInstruction(Instruction): pass
class AND(LogicInstruction):
    def lift_operation2(self, il: LowLevelILFunction, il_arg1: ExpressionIndex, il_arg2: ExpressionIndex) -> ExpressionIndex:
        return il.and_expr(1, il_arg1, il_arg2, ZFlag)
class OR(LogicInstruction):
    def lift_operation2(self, il: LowLevelILFunction, il_arg1: ExpressionIndex, il_arg2: ExpressionIndex) -> ExpressionIndex:
        return il.or_expr(1, il_arg1, il_arg2, ZFlag)
class XOR(LogicInstruction):
    def lift_operation2(self, il: LowLevelILFunction, il_arg1: ExpressionIndex, il_arg2: ExpressionIndex) -> ExpressionIndex:
        return il.xor_expr(1, il_arg1, il_arg2, ZFlag)

class CompareInstruction(Instruction): pass
class TEST(CompareInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        dst_mode = get_addressing_mode(self._pre, 1) if self._pre else None
        src_mode = get_addressing_mode(self._pre, 2) if self._pre else None
        first, second = self.operands()
        # FIXME: does it set the Z flag if any bit is set?
        il.append(
            il.set_flag(
                ZFlag,
                il.and_expr(3, first.lift(il, dst_mode), second.lift(il, src_mode)),
            )
        )

class CMP(CompareInstruction):
    def width(self) -> int:
        return 1
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        dst_mode = get_addressing_mode(self._pre, 1) if self._pre else None
        src_mode = get_addressing_mode(self._pre, 2) if self._pre else None
        first, second = self.operands()
        # FIXME: what's the proper width?
        il.append(
            il.sub(
                self.width(),
                first.lift(il, dst_mode),
                second.lift(il, src_mode),
                CZFlag,
            )
        )
class CMPW(CMP):
    def width(self) -> int:
        return 2
class CMPP(CMP):
    def width(self) -> int:
        return 3

# FIXME: verify on real hardware, likely wrong
class ShiftRotateInstruction(Instruction):
    def shift_by(self, il: LowLevelILFunction) -> ExpressionIndex:
        return il.const(1, 1)
# bit rotation
class ROR(ShiftRotateInstruction):
    def lift_operation1(self, il: LowLevelILFunction, il_arg1: ExpressionIndex) -> ExpressionIndex:
        return il.rotate_right(1, il_arg1, self.shift_by(il), CZFlag)
class ROL(ShiftRotateInstruction):
    def lift_operation1(self, il: LowLevelILFunction, il_arg1: ExpressionIndex) -> ExpressionIndex:
        return il.rotate_left(1, il_arg1, self.shift_by(il), CZFlag)
# bit shift
class SHL(ShiftRotateInstruction):
    def lift_operation1(self, il: LowLevelILFunction, il_arg1: ExpressionIndex) -> ExpressionIndex:
        return il.rotate_left_carry(1, il_arg1, self.shift_by(il),
                                    il.flag(CFlag), CZFlag)
class SHR(ShiftRotateInstruction):
    def lift_operation1(self, il: LowLevelILFunction, il_arg1: ExpressionIndex) -> ExpressionIndex:
        return il.rotate_right_carry(1, il_arg1, self.shift_by(il),
                                     il.flag(CFlag), CZFlag)

# digit shift
class DSLL(Instruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        imem_op, = self.operands()
        # Ensure the operand is IMem8 as corrected in OPCODES
        assert isinstance(imem_op, IMem8), f"DSLL operand should be IMem8, got {type(imem_op)}"

        # current_addr_reg holds the internal memory address (e.g.,
        # INTERNAL_MEMORY_START + n_offset)
        # For DSLL, (n) is the MSB address.
        current_addr_reg = TempReg(TempMultiByte1, width=3)  # Addresses are 3 bytes (20/24 bit)
        mode = get_addressing_mode(self._pre, 1) if self._pre else None
        current_addr_reg.lift_assign(
            il, imem_op.lift_current_addr(il, pre=mode, side_effects=False)
        )

        # digit_carry_reg stores the high nibble of T (current byte) to be OR'd
        # into the low nibble of S (shifted byte) in the next (less significant) iteration.
        digit_carry_reg = TempReg(TempBcdDigitCarry, width=1)
        # The first S is (T_low << 4) | 0. So initial carry is 0.
        digit_carry_reg.lift_assign(il, il.const(1, 0))

        overall_zero_acc_reg = TempReg(TempOverallZeroAcc, width=1)
        overall_zero_acc_reg.lift_assign(il, il.const(1, 0))

        mem_accessor = IMemHelper(width=1, value=current_addr_reg) # For load/store via current_addr_reg

        with lift_loop(il): # loop_reg is 'I', decrements from initial value
            current_byte_T = mem_accessor.lift(il)

            # S = (T_low_nibble << 4) | digit_carry_from_PREVIOUS_T_high_nibble
            T_low_nibble = il.and_expr(1, current_byte_T, il.const(1, 0x0F))
            T_high_nibble = il.logical_shift_right(1, current_byte_T, il.const(1, 4))
            # T_high_nibble = il.and_expr(1, T_high_nibble, il.const(1, 0x0F)) # Mask if needed

            shifted_byte_S = il.or_expr(1,
                                        il.shift_left(1, T_low_nibble, il.const(1, 4)),
                                        digit_carry_reg.lift(il))

            mem_accessor.lift_assign(il, shifted_byte_S)

            # Update digit_carry_reg with T_high_nibble for the next iteration
            digit_carry_reg.lift_assign(il, T_high_nibble)

            overall_zero_acc_reg.lift_assign(il, il.or_expr(1, overall_zero_acc_reg.lift(il), shifted_byte_S))

            # Decrement address pointer for DSLL (MSB to LSB)
            current_addr_reg.lift_assign(il, il.sub(3, current_addr_reg.lift(il), il.const(3, 1)))

        il.append(il.set_flag(ZFlag, il.compare_equal(1, overall_zero_acc_reg.lift(il), il.const(1, 0))))
        # FC is not affected.

class DSRL(Instruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        imem_op, = self.operands()
        assert isinstance(imem_op, IMem8), f"DSRL operand should be IMem8, got {type(imem_op)}"

        # For DSRL, (n) is the LSB address.
        current_addr_reg = TempReg(TempMultiByte1, width=3)
        mode = get_addressing_mode(self._pre, 1) if self._pre else None
        current_addr_reg.lift_assign(
            il, imem_op.lift_current_addr(il, pre=mode, side_effects=False)
        )

        # digit_carry_reg stores the low nibble of T (current byte) to be OR'd
        # into the high nibble of S (shifted byte) in the next (more significant) iteration.
        digit_carry_reg = TempReg(TempBcdDigitCarry, width=1)
        # The first S is T >> 4. So initial carry is 0.
        digit_carry_reg.lift_assign(il, il.const(1, 0))

        overall_zero_acc_reg = TempReg(TempOverallZeroAcc, width=1)
        overall_zero_acc_reg.lift_assign(il, il.const(1, 0))

        mem_accessor = IMemHelper(width=1, value=current_addr_reg)

        with lift_loop(il):
            current_byte_T = mem_accessor.lift(il)

            # S = T_high_nibble | (digit_carry_from_PREVIOUS_T_low_nibble << 4)
            T_low_nibble = il.and_expr(1, current_byte_T, il.const(1, 0x0F))
            T_high_nibble = il.logical_shift_right(1, current_byte_T, il.const(1, 4))
            # T_high_nibble = il.and_expr(1, T_high_nibble, il.const(1, 0x0F))

            shifted_byte_S = il.or_expr(1,
                                        T_high_nibble,
                                        il.shift_left(1, digit_carry_reg.lift(il), il.const(1, 4)))

            mem_accessor.lift_assign(il, shifted_byte_S)

            # Update digit_carry_reg with T_low_nibble for the next iteration
            digit_carry_reg.lift_assign(il, T_low_nibble)

            overall_zero_acc_reg.lift_assign(il, il.or_expr(1, overall_zero_acc_reg.lift(il), shifted_byte_S))

            # Increment address pointer for DSRL (LSB to MSB)
            current_addr_reg.lift_assign(il, il.add(3, current_addr_reg.lift(il), il.const(3, 1)))

        il.append(il.set_flag(ZFlag, il.compare_equal(1, overall_zero_acc_reg.lift(il), il.const(1, 0))))
        # FC is not affected.


class IncDecInstruction(Instruction): pass
class INC(IncDecInstruction):
    def lift_operation1(self, il: LowLevelILFunction, il_arg: ExpressionIndex) -> ExpressionIndex:
        return il.add(1, il_arg, il.const(1, 1), ZFlag)
class DEC(IncDecInstruction):
    def lift_operation1(self, il: LowLevelILFunction, il_arg: ExpressionIndex) -> ExpressionIndex:
        return il.sub(1, il_arg, il.const(1, 1), ZFlag)

class ExchangeInstruction(Instruction):
    def lift_single_exchange(self, il: LowLevelILFunction, addr: int) -> None:
        first, second = self.operands()
        assert isinstance(first, HasWidth), f"Expected HasWidth, got {type(first)}"
        width = first.width()
        tmp = TempReg(TempExchange, width=width)
        tmp.lift_assign(il, first.lift(il))
        first.lift_assign(il, second.lift(il))
        second.lift_assign(il, tmp.lift(il))
class EX(ExchangeInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        self.lift_single_exchange(il, addr)
# uses counter
class EXL(ExchangeInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        with lift_loop(il):
            self.lift_single_exchange(il, addr)

class MiscInstruction(Instruction): pass
class WAIT(MiscInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        with lift_loop(il):
            # Wait is just an idle loop
            pass

class PMDF(MiscInstruction):
    # FIXME: verify
    def lift_operation2(self, il: LowLevelILFunction, il_arg1: ExpressionIndex, il_arg2: ExpressionIndex) -> ExpressionIndex:
        return il.add(1, il_arg1, il_arg2)

class SWAP(MiscInstruction):
    def lift_operation1(self, il: LowLevelILFunction, il_arg1: ExpressionIndex) -> ExpressionIndex:
        low = il.and_expr(1, il_arg1, il.const(1, 0x0F))
        low = il.shift_left(1, low, il.const(1, 4))
        high = il.and_expr(1, il_arg1, il.const(1, 0xF0))
        high = il.logical_shift_right(1, high, il.const(1, 4))
        return il.or_expr(1, low, high, ZFlag)

class SC(MiscInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        il.append(il.set_flag(CFlag, il.const(1, 1)))
class RC(MiscInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        il.append(il.set_flag(CFlag, il.const(1, 0)))

# Timer Clear: sub-CG or main-CG timers are reset when STCL / MTCL of LCC are
# set.
# Divider ← D
class TCL(MiscInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        il.append(il.intrinsic([], TCLIntrinsic, []))

# System Clock Stop: halts main-CG of CPU
# Execution can continue past HALT: ON, IRQ, KI pins
# USR resets bits 0 to 2/5 to 0
# SSR bit 2 and USR 3 and 4 are set to 1
class HALT(MiscInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        il.append(il.intrinsic([], HALTIntrinsic, []))

# System Clock Stop; Sub Clock Stop: main-CG and sub-CG of CPU are stopped
# Execution can continue past OFF: ON, IRQ, KI pins
class OFF(MiscInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        il.append(il.intrinsic([], OFFIntrinsic, []))

# AKA `INT / Interrupt`
# 1. Save context to system stack (S-stack), in this strict order:
#      - PS  (Program Status)
#      - PC  (Program Counter, high byte first, then low byte)
#      - FLAG (Status Flags)
#      - IMR (Interrupt Mask Register at FBH)
#    (Total pushed = 5 bytes)
#
# 2. Load new PC and PS from fixed memory locations:
#      - PC high-byte loaded from address FFFFBH
#      - PC low-byte  loaded from address FFFFAH
#      - PS loaded from address FFFFCH
#
# 3. After pushing IMR, bit 7 (IRM) of IMR is forcibly cleared to 0.
class IR(MiscInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        il.append(il.push(3, RegPC().lift(il)))
        il.append(il.push(1, RegF().lift(il)))
        imr, *rest = RegIMR().operands()
        il.append(il.push(1, imr.lift(il)))
        imr.lift_assign(il, il.and_expr(1, imr.lift(il), il.const(1, 0x7F)))

        mem = EMemAddr(width=3)
        mem.value = INTERRUPT_VECTOR_ADDR
        il.append(il.jump(mem.lift(il)))

# ACM bit 7, UCR + USR bits 0 to 2/5, IMR, SCR, SSR bit 2 are all reset to 0
# USR bits 3 and 4 are set to 1
class RESET(MiscInstruction):
    def analyze(self, info: InstructionInfo, addr: int) -> None:
        super().analyze(info, addr)
        info.add_branch(BranchType.FunctionReturn)

    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        mem = EMemAddr(width=3)
        mem.value = ENTRY_POINT_ADDR
        il.append(il.jump(mem.lift(il)))

class UnknownInstruction(Instruction):
    def name(self) -> str:
        return f"??? ({self.opcode:02X})"


OPCODES = {
    # 00h
    0x00: NOP,
    0x01: RETI,
    0x02: (JP_Abs, Opts(ops=[Imm16()])),
    0x03: (JP_Abs, Opts(name="JPF", ops=[Imm20()])),
    0x04: (CALL, Opts(ops=[Imm16()])),
    0x05: (CALL, Opts(name="CALLF", ops=[Imm20()])),
    0x06: RET,
    0x07: RETF,
    0x08: (MV, Opts(ops=[Reg("A"), Imm8()])),
    0x09: (MV, Opts(ops=[Reg("IL"), Imm8()])),
    0x0A: (MV, Opts(ops=[Reg("BA"), Imm16()])),
    0x0B: (MV, Opts(ops=[Reg("I"), Imm16()])),
    0x0C: (MV, Opts(ops=[Reg("X"), Imm20()])),
    0x0D: (MV, Opts(ops=[Reg("Y"), Imm20()])),
    0x0E: (MV, Opts(ops=[Reg("U"), Imm20()])),
    0x0F: (MV, Opts(ops=[Reg("S"), Imm20()])),
    # 10h
    0x10: (JP_Abs, Opts(ops=[IMem20()])),
    0x11: (JP_Abs, Opts(ops=[Reg3()])),
    0x12: (JP_Rel, Opts(ops=[ImmOffset("+")])),
    0x13: (JP_Rel, Opts(ops=[ImmOffset("-")])),
    0x14: (JP_Abs, Opts(cond="Z", ops=[Imm16()])),
    0x15: (JP_Abs, Opts(cond="NZ", ops=[Imm16()])),
    0x16: (JP_Abs, Opts(cond="C", ops=[Imm16()])),
    0x17: (JP_Abs, Opts(cond="NC", ops=[Imm16()])),
    0x18: (JP_Rel, Opts(cond="Z", ops=[ImmOffset("+")])),
    0x19: (JP_Rel, Opts(cond="Z", ops=[ImmOffset("-")])),
    0x1A: (JP_Rel, Opts(cond="NZ", ops=[ImmOffset("+")])),
    0x1B: (JP_Rel, Opts(cond="NZ", ops=[ImmOffset("-")])),
    0x1C: (JP_Rel, Opts(cond="C", ops=[ImmOffset("+")])),
    0x1D: (JP_Rel, Opts(cond="C", ops=[ImmOffset("-")])),
    0x1E: (JP_Rel, Opts(cond="NC", ops=[ImmOffset("+")])),
    0x1F: (JP_Rel, Opts(cond="NC", ops=[ImmOffset("-")])),
    # 20h
    0x20: UnknownInstruction,
    0x21: PRE,
    0x22: PRE,
    0x23: PRE,
    0x24: PRE,
    0x25: PRE,
    0x26: PRE,
    0x27: PRE,
    0x28: (PUSHU, Opts(ops=[Reg("A")])),
    0x29: (PUSHU, Opts(ops=[Reg("IL")])),
    0x2A: (PUSHU, Opts(ops=[Reg("BA")])),
    0x2B: (PUSHU, Opts(ops=[Reg("I")])),
    0x2C: (PUSHU, Opts(ops=[Reg("X")])),
    0x2D: (PUSHU, Opts(ops=[Reg("Y")])),
    0x2E: (PUSHU, Opts(ops=[RegF()])),
    0x2F: (PUSHU, Opts(ops=[RegIMR()])),
    # 30h
    0x30: PRE,
    0x31: PRE,
    0x32: PRE,
    0x33: PRE,
    0x34: PRE,
    0x35: PRE,
    0x36: PRE,
    0x37: PRE,
    0x38: (POPU, Opts(ops=[Reg("A")])),
    0x39: (POPU, Opts(ops=[Reg("IL")])),
    0x3A: (POPU, Opts(ops=[Reg("BA")])),
    0x3B: (POPU, Opts(ops=[Reg("I")])),
    0x3C: (POPU, Opts(ops=[Reg("X")])),
    0x3D: (POPU, Opts(ops=[Reg("Y")])),
    0x3E: (POPU, Opts(ops=[RegF()])),
    0x3F: (POPU, Opts(ops=[RegIMR()])),
    # 40h
    0x40: (ADD, Opts(ops=[Reg("A"), Imm8()])),
    0x41: (ADD, Opts(ops=[IMem8(), Imm8()])),
    0x42: (ADD, Opts(ops=[Reg("A"), IMem8()])),
    0x43: (ADD, Opts(ops=[IMem8(), Reg("A")])),
    0x44: (ADD, Opts(ops=[RegPair(size=2)])),
    0x45: (ADD, Opts(ops=[RegPair(size=3)])),
    0x46: (ADD, Opts(ops=[RegPair(size=1)])),
    0x47: (PMDF, Opts(ops=[IMem8(), Imm8()])),
    0x48: (SUB, Opts(ops=[Reg("A"), Imm8()])),
    0x49: (SUB, Opts(ops=[IMem8(), Imm8()])),
    0x4A: (SUB, Opts(ops=[Reg("A"), IMem8()])),
    0x4B: (SUB, Opts(ops=[IMem8(), Reg("A")])),
    0x4C: (SUB, Opts(ops=[RegPair(size=2)])),
    0x4D: (SUB, Opts(ops=[RegPair(size=3)])),
    0x4E: (SUB, Opts(ops=[RegPair(size=1)])),
    0x4F: (PUSHS, Opts(ops=[RegF()])),
    # 50h
    0x50: (ADC, Opts(ops=[Reg("A"), Imm8()])),
    0x51: (ADC, Opts(ops=[IMem8(), Imm8()])),
    0x52: (ADC, Opts(ops=[Reg("A"), IMem8()])),
    0x53: (ADC, Opts(ops=[IMem8(), Reg("A")])),
    0x54: (ADCL, Opts(ops=[IMem8(), IMem8()])),
    0x55: (ADCL, Opts(ops=[IMem8(), Reg("A")])),
    0x56: (MVL, Opts(ops=[RegIMemOffset(order=RegIMemOffsetOrder.DEST_IMEM,
                                        allowed_modes=[EMemRegMode.POSITIVE_OFFSET, EMemRegMode.NEGATIVE_OFFSET])])),
    0x57: (PMDF, Opts(ops=[IMem8(), Reg("A")])),
    0x58: (SBC, Opts(ops=[Reg("A"), Imm8()])),
    0x59: (SBC, Opts(ops=[IMem8(), Imm8()])),
    0x5A: (SBC, Opts(ops=[Reg("A"), IMem8()])),
    0x5B: (SBC, Opts(ops=[IMem8(), Reg("A")])),
    0x5C: (SBCL, Opts(ops=[IMem8(), IMem8()])),
    0x5D: (SBCL, Opts(ops=[IMem8(), Reg("A")])),
    0x5E: (MVL, Opts(ops=[RegIMemOffset(order=RegIMemOffsetOrder.DEST_REG_OFFSET,
                                        allowed_modes=[EMemRegMode.POSITIVE_OFFSET, EMemRegMode.NEGATIVE_OFFSET])])),
    0x5F: (POPS, Opts(ops=[RegF()])),
    # 60h
    0x60: (CMP, Opts(ops=[Reg("A"), Imm8()])),
    0x61: (CMP, Opts(ops=[IMem8(), Imm8()])),
    0x62: (CMP, Opts(ops=[EMemAddr(width=1), Imm8()])),
    0x63: (CMP, Opts(ops=[IMem8(), Reg("A")])),
    0x64: (TEST, Opts(ops=[Reg("A"), Imm8()])),
    0x65: (TEST, Opts(ops=[IMem8(), Imm8()])),
    0x66: (TEST, Opts(ops=[EMemAddr(width=1), Imm8()])),
    0x67: (TEST, Opts(ops=[IMem8(), Reg("A")])),
    0x68: (XOR, Opts(ops=[Reg("A"), Imm8()])),
    0x69: (XOR, Opts(ops=[IMem8(), Imm8()])),
    0x6A: (XOR, Opts(ops=[EMemAddr(width=1), Imm8()])),
    0x6B: (XOR, Opts(ops=[IMem8(), Reg("A")])),
    0x6C: (INC, Opts(ops=[Reg3()])),
    0x6D: (INC, Opts(ops=[IMem8()])),
    0x6E: (XOR, Opts(ops=[IMem8(), IMem8()])),
    0x6F: (XOR, Opts(ops=[Reg("A"), IMem8()])),
    # 70h
    0x70: (AND, Opts(ops=[Reg("A"), Imm8()])),
    0x71: (AND, Opts(ops=[IMem8(), Imm8()])),
    0x72: (AND, Opts(ops=[EMemAddr(width=1), Imm8()])),
    0x73: (AND, Opts(ops=[IMem8(), Reg("A")])),
    0x74: (MV, Opts(ops=[Reg("A"), RegB()])),
    0x75: (MV, Opts(ops=[RegB(), Reg("A")])),
    0x76: (AND, Opts(ops=[IMem8(), IMem8()])),
    0x77: (AND, Opts(ops=[Reg("A"), IMem8()])),
    0x78: (OR, Opts(ops=[Reg("A"), Imm8()])),
    0x79: (OR, Opts(ops=[IMem8(), Imm8()])),
    0x7A: (OR, Opts(ops=[EMemAddr(width=1), Imm8()])),
    0x7B: (OR, Opts(ops=[IMem8(), Reg("A")])),
    0x7C: (DEC, Opts(ops=[Reg3()])),
    0x7D: (DEC, Opts(ops=[IMem8()])),
    0x7E: (OR, Opts(ops=[IMem8(), IMem8()])),
    0x7F: (OR, Opts(ops=[Reg("A"), IMem8()])),
    # 80h
    0x80: (MV, Opts(ops=[Reg("A"), IMem8()])),
    0x81: (MV, Opts(ops=[Reg("IL"), IMem8()])),
    0x82: (MV, Opts(ops=[Reg("BA"), IMem16()])),
    0x83: (MV, Opts(ops=[Reg("I"), IMem16()])),
    0x84: (MV, Opts(ops=[Reg("X"), IMem20()])),
    0x85: (MV, Opts(ops=[Reg("Y"), IMem20()])),
    0x86: (MV, Opts(ops=[Reg("U"), IMem20()])),
    0x87: (MV, Opts(ops=[Reg("S"), IMem20()])),
    0x88: (MV, Opts(ops=[Reg("A"), EMemAddr(width=1)])),
    0x89: (MV, Opts(ops=[Reg("IL"), EMemAddr(width=1)])),
    0x8A: (MV, Opts(ops=[Reg("BA"), EMemAddr(width=2)])),
    0x8B: (MV, Opts(ops=[Reg("I"), EMemAddr(width=2)])),
    0x8C: (MV, Opts(ops=[Reg("X"), EMemAddr(width=3)])),
    0x8D: (MV, Opts(ops=[Reg("Y"), EMemAddr(width=3)])),
    0x8E: (MV, Opts(ops=[Reg("U"), EMemAddr(width=3)])),
    0x8F: (MV, Opts(ops=[Reg("S"), EMemAddr(width=3)])),
    # 90h
    0x90: (MV, Opts(ops=[Reg("A"), EMemReg(width=1)])),
    0x91: (MV, Opts(ops=[Reg("IL"), EMemReg(width=1)])),
    0x92: (MV, Opts(ops=[Reg("BA"), EMemReg(width=2)])),
    0x93: (MV, Opts(ops=[Reg("I"), EMemReg(width=2)])),
    0x94: (MV, Opts(ops=[Reg("X"), EMemReg(width=3)])),
    0x95: (MV, Opts(ops=[Reg("Y"), EMemReg(width=3)])),
    0x96: (MV, Opts(ops=[Reg("U"), EMemReg(width=3)])),
    0x97: SC,
    0x98: (MV, Opts(ops=[Reg("A"), EMemIMem()])),
    0x99: (MV, Opts(ops=[Reg("IL"), EMemIMem()])),
    0x9A: (MV, Opts(ops=[Reg("BA"), EMemIMem()])),
    0x9B: (MV, Opts(ops=[Reg("I"), EMemIMem()])),
    0x9C: (MV, Opts(ops=[Reg("X"), EMemIMem()])),
    0x9D: (MV, Opts(ops=[Reg("Y"), EMemIMem()])),
    0x9E: (MV, Opts(ops=[Reg("U"), EMemIMem()])),
    0x9F: RC,
    # A0h
    0xA0: (MV, Opts(ops=[IMem8(), Reg("A")])),
    0xA1: (MV, Opts(ops=[IMem8(), Reg("IL")])),
    0xA2: (MV, Opts(ops=[IMem16(), Reg("BA")])),
    0xA3: (MV, Opts(ops=[IMem16(), Reg("I")])),
    0xA4: (MV, Opts(ops=[IMem20(), Reg("X")])),
    0xA5: (MV, Opts(ops=[IMem20(), Reg("Y")])),
    0xA6: (MV, Opts(ops=[IMem20(), Reg("U")])),
    0xA7: (MV, Opts(ops=[IMem20(), Reg("S")])),
    0xA8: (MV, Opts(ops=[EMemAddr(width=1), Reg("A")])),
    0xA9: (MV, Opts(ops=[EMemAddr(width=1), Reg("IL")])),
    0xAA: (MV, Opts(ops=[EMemAddr(width=2), Reg("BA")])),
    0xAB: (MV, Opts(ops=[EMemAddr(width=2), Reg("I")])),
    0xAC: (MV, Opts(ops=[EMemAddr(width=3), Reg("X")])),
    0xAD: (MV, Opts(ops=[EMemAddr(width=3), Reg("Y")])),
    0xAE: (MV, Opts(ops=[EMemAddr(width=3), Reg("U")])),
    0xAF: (MV, Opts(ops=[EMemAddr(width=3), Reg("S")])),
    # B0h
    0xB0: (MV, Opts(ops=[EMemReg(width=1), Reg("A")])),
    0xB1: (MV, Opts(ops=[EMemReg(width=1), Reg("IL")])),
    0xB2: (MV, Opts(ops=[EMemReg(width=2), Reg("BA")])),
    0xB3: (MV, Opts(ops=[EMemReg(width=2), Reg("I")])),
    0xB4: (MV, Opts(ops=[EMemReg(width=3), Reg("X")])),
    0xB5: (MV, Opts(ops=[EMemReg(width=3), Reg("Y")])),
    0xB6: (MV, Opts(ops=[EMemReg(width=3), Reg("U")])),
    0xB7: (CMP, Opts(ops=[IMem8(), IMem8()])),
    0xB8: (MV, Opts(ops=[EMemIMem(), Reg("A")])),
    0xB9: (MV, Opts(ops=[EMemIMem(), Reg("IL")])),
    0xBA: (MV, Opts(ops=[EMemIMem(), Reg("BA")])),
    0xBB: (MV, Opts(ops=[EMemIMem(), Reg("I")])),
    0xBC: (MV, Opts(ops=[EMemIMem(), Reg("X")])),
    0xBD: (MV, Opts(ops=[EMemIMem(), Reg("Y")])),
    0xBE: (MV, Opts(ops=[EMemIMem(), Reg("U")])),
    0xBF: UnknownInstruction,
    # C0h
    0xC0: (EX, Opts(ops=[IMem8(), IMem8()])),
    0xC1: (EX, Opts(name="EXW", ops=[IMem16(), IMem16()])),
    0xC2: (EX, Opts(name="EXP", ops=[IMem20(), IMem20()])),
    0xC3: (EXL, Opts(ops=[IMem8(), IMem8()])),
    0xC4: (DADL, Opts(ops=[IMem8(), IMem8()])),
    0xC5: (DADL, Opts(ops=[IMem8(), Reg("A")])),
    0xC6: (CMPW, Opts(ops=[IMem16(), IMem16()])),
    0xC7: (CMPP, Opts(ops=[IMem20(), IMem20()])),
    0xC8: (MV, Opts(ops=[IMem8(), IMem8()])),
    0xC9: (MV, Opts(name="MVW", ops=[IMem16(), IMem16()])),
    0xCA: (MV, Opts(name="MVP", ops=[IMem20(), IMem20()])),
    0xCB: (MVL, Opts(ops=[IMem8(), IMem8()])),
    0xCC: (MV, Opts(ops=[IMem8(), Imm8()])),
    0xCD: (MV, Opts(name="MVW", ops=[IMem16(), Imm16()])),
    0xCE: TCL,
    0xCF: (MVLD, Opts(ops=[IMem8(), IMem8()])),
    # D0h
    0xD0: (MV, Opts(ops=[IMem8(), EMemAddr(width=1)])),
    0xD1: (MV, Opts(name="MVW", ops=[IMem16(), EMemAddr(width=2)])),
    0xD2: (MV, Opts(name="MVP", ops=[IMem20(), EMemAddr(width=3)])),
    0xD3: (MVL, Opts(ops=[IMem8(), EMemAddr(width=1)])),
    0xD4: (DSBL, Opts(ops=[IMem8(), IMem8()])),
    0xD5: (DSBL, Opts(ops=[IMem8(), Reg("A")])),
    0xD6: (CMPW, Opts(ops_reversed=True, ops=[IMem16(), Reg3()])),
    0xD7: (CMPP, Opts(ops_reversed=True, ops=[IMem20(), Reg3()])),
    0xD8: (MV, Opts(ops=[EMemAddr(width=1), IMem8()])),
    0xD9: (MV, Opts(name="MVW", ops=[EMemAddr(width=2), IMem16()])),
    0xDA: (MV, Opts(name="MVP", ops=[EMemAddr(width=3), IMem20()])),
    0xDB: (MVL, Opts(ops=[EMemAddr(width=1), IMem8()])),
    0xDC: (MV, Opts(name="MVP", ops=[IMem20(), Imm20()])),
    0xDD: (EX, Opts(ops=[Reg("A"), RegB()])),
    0xDE: HALT,
    0xDF: OFF,
    # E0h
    0xE0: (MV, Opts(ops=[RegIMemOffset(order=RegIMemOffsetOrder.DEST_IMEM)])),
    0xE1: (MV, Opts(name="MVW",
                    ops=[RegIMemOffset(order=RegIMemOffsetOrder.DEST_IMEM)])),
    0xE2: (MV, Opts(name="MVP",
                    ops=[RegIMemOffset(order=RegIMemOffsetOrder.DEST_IMEM)])),
    # FIXME: verify width
    0xE3: (MVL, Opts(ops_reversed=True, ops=[IMem8(),
                                             EMemReg(width=1,
                                                     allowed_modes=[EMemRegMode.POST_INC,
                                                                    EMemRegMode.PRE_DEC])])),
    0xE4: (ROR, Opts(ops=[Reg("A")])),
    0xE5: (ROR, Opts(ops=[IMem8()])),
    0xE6: (ROL, Opts(ops=[Reg("A")])),
    0xE7: (ROL, Opts(ops=[IMem8()])),
    0xE8: (MV, Opts(ops=[RegIMemOffset(order=RegIMemOffsetOrder.DEST_REG_OFFSET)])),
    0xE9: (MV, Opts(name="MVW",
                    ops=[RegIMemOffset(order=RegIMemOffsetOrder.DEST_REG_OFFSET)])),
    0xEA: (MV, Opts(name="MVP",
                    ops=[RegIMemOffset(order=RegIMemOffsetOrder.DEST_REG_OFFSET)])),
    # FIXME: verify width
    0xEB: (MVL, Opts(ops=[EMemReg(width=1), IMem8()])),
    0xEC: (DSLL, Opts(ops=[IMem8()])),
    0xED: (EX, Opts(ops=[RegPair(size=2)])),
    0xEE: (SWAP, Opts(ops=[Reg("A")])),
    0xEF: WAIT,
    # F0h
    0xF0: (MV, Opts(ops=[EMemIMemOffset(order=EMemIMemOffsetOrder.DEST_INT_MEM)])),
    0xF1: (MV, Opts(name='MVW', ops=[EMemIMemOffset(order=EMemIMemOffsetOrder.DEST_INT_MEM)])),
    0xF2: (MV, Opts(name='MVP', ops=[EMemIMemOffset(order=EMemIMemOffsetOrder.DEST_INT_MEM)])),
    0xF3: (MVL, Opts(ops=[EMemIMemOffset(order=EMemIMemOffsetOrder.DEST_INT_MEM)])),
    0xF4: (SHR, Opts(ops=[Reg("A")])),
    0xF5: (SHR, Opts(ops=[IMem8()])),
    0xF6: (SHL, Opts(ops=[Reg("A")])),
    0xF7: (SHL, Opts(ops=[IMem8()])),
    0xF8: (MV, Opts(ops=[EMemIMemOffset(order=EMemIMemOffsetOrder.DEST_EXT_MEM)])),
    0xF9: (MV, Opts(name='MVW', ops=[EMemIMemOffset(order=EMemIMemOffsetOrder.DEST_EXT_MEM)])),
    0xFA: (MV, Opts(name='MVP', ops=[EMemIMemOffset(order=EMemIMemOffsetOrder.DEST_EXT_MEM)])),
    0xFB: (MVL, Opts(ops=[EMemIMemOffset(order=EMemIMemOffsetOrder.DEST_EXT_MEM)])),
    0xFC: (DSRL, Opts(ops=[IMem8()])),
    0xFD: (MV, Opts(ops=[RegPair(size=2)])),
    0xFE: IR,
    0xFF: RESET,
}

