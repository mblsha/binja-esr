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
from ..constants import INTERNAL_MEMORY_START, PC_MASK
from .traits import HasWidth

import copy
from typing import (
    Optional,
    List,
    Iterable,
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

from dataclasses import dataclass


class InvalidInstruction(Exception):
    pass


class IntAddrCalc(str, enum.Enum):
    """Enumeration of supported internal memory addressing calculations."""

    N = "(n)"
    BP_N = "(BP+n)"
    PX_N = "(PX+n)"
    PY_N = "(PY+n)"
    BP_PX = "(BP+PX)"
    BP_PY = "(BP+PY)"


@dataclass(frozen=True)
class PreLatch:
    first: IntAddrCalc
    second: IntAddrCalc


@dataclass(frozen=True)
class PreMode:
    """Represents a single PRE opcode and the IMEM mappings it enforces."""

    opcode: int
    latch: PreLatch


_PRE_OPCODE_MATRIX = {
    # First-op rows: (n), (BP+n), (PX+n), (BP+PX)
    # Second-op cols: (n), (BP+n), (PY+n), (BP+PY)
    (IntAddrCalc.N, IntAddrCalc.N): 0x32,
    (IntAddrCalc.N, IntAddrCalc.BP_N): 0x30,
    (IntAddrCalc.N, IntAddrCalc.PY_N): 0x33,
    (IntAddrCalc.N, IntAddrCalc.BP_PY): 0x31,
    (IntAddrCalc.BP_N, IntAddrCalc.N): 0x22,
    (IntAddrCalc.BP_N, IntAddrCalc.PY_N): 0x23,
    (IntAddrCalc.BP_N, IntAddrCalc.BP_PY): 0x21,
    (IntAddrCalc.PX_N, IntAddrCalc.N): 0x36,
    (IntAddrCalc.PX_N, IntAddrCalc.BP_N): 0x34,
    (IntAddrCalc.PX_N, IntAddrCalc.PY_N): 0x37,
    (IntAddrCalc.PX_N, IntAddrCalc.BP_PY): 0x35,
    (IntAddrCalc.BP_PX, IntAddrCalc.N): 0x26,
    (IntAddrCalc.BP_PX, IntAddrCalc.BP_N): 0x24,
    (IntAddrCalc.BP_PX, IntAddrCalc.PY_N): 0x27,
    (IntAddrCalc.BP_PX, IntAddrCalc.BP_PY): 0x25,
}


def _build_modes() -> Tuple[
    Dict[int, PreMode], Dict[Tuple[IntAddrCalc, IntAddrCalc], int]
]:
    by_opcode: Dict[int, PreMode] = {}
    by_pair: Dict[Tuple[IntAddrCalc, IntAddrCalc], int] = {}

    for (row, col), opcode in _PRE_OPCODE_MATRIX.items():
        if opcode == 0x20:
            continue
        if opcode in by_opcode:
            raise ValueError(f"Duplicate PRE opcode {opcode:#x}")
        latch = PreLatch(row, col)
        by_opcode[opcode] = PreMode(opcode=opcode, latch=latch)
        by_pair[(row, col)] = opcode

    return by_opcode, by_pair


PRE_BY_OPCODE, PRE_BY_PAIR = _build_modes()


def opcode_for_modes(first: IntAddrCalc, second: IntAddrCalc) -> int | None:
    """Return the PRE opcode byte for a given pair of IMEM addressing modes."""

    return PRE_BY_PAIR.get((first, second))


def iter_pre_modes() -> Iterable[PreMode]:
    """Yield all supported PRE modes sorted by opcode."""

    for opcode in sorted(PRE_BY_OPCODE):
        yield PRE_BY_OPCODE[opcode]


# `AddressingMode` historically lived in this module; now it aliases the shared
# IntAddrCalc enum kept in sync with opcode helpers and potential codegen.
AddressingMode = IntAddrCalc


def _build_pre_tables() -> Tuple[
    Dict[int, Dict[int, AddressingMode]],
    Dict[Tuple[AddressingMode, AddressingMode], int],
]:
    per_operand: Dict[int, Dict[int, AddressingMode]] = {1: {}, 2: {}}
    reverse: Dict[Tuple[AddressingMode, AddressingMode], int] = {}
    for mode in iter_pre_modes():
        per_operand[1][mode.opcode] = mode.latch.first
        per_operand[2][mode.opcode] = mode.latch.second
        reverse[(mode.latch.first, mode.latch.second)] = mode.opcode
    return per_operand, reverse


PRE_TABLE, REVERSE_PRE_TABLE = _build_pre_tables()


# Canonical PRE bytes for instructions with one internal-memory selector. Such
# instructions consume PRE1; PRE2 remains at its default BP+n setting. Modes
# that exist only in PRE2 (PY+n and BP+PY) therefore cannot be represented by a
# single-selector instruction.
SINGLE_OPERAND_PRE_LOOKUP: Dict[AddressingMode, int] = {
    AddressingMode.N: opcode_for_modes(AddressingMode.N, AddressingMode.BP_N) or 0x30,
    AddressingMode.PX_N: opcode_for_modes(AddressingMode.PX_N, AddressingMode.BP_N)
    or 0x34,
    AddressingMode.BP_PX: opcode_for_modes(AddressingMode.BP_PX, AddressingMode.BP_N)
    or 0x24,
}

# Exact one-selector PRE aliases exercised on a real PC-E500. The second latch
# is not consulted when the opcode has only one addressable IMEM operand.
# Keep the encoder canonical; this table exists only for decoding/lifting raw
# silicon-accepted byte streams.
SILICON_PROVEN_SINGLE_PRE_ALIASES: Dict[AddressingMode, frozenset[int]] = {
    AddressingMode.BP_N: frozenset({0x22}),
    AddressingMode.N: frozenset({0x30, 0x31, 0x32, 0x33}),
    AddressingMode.PX_N: frozenset({0x34, 0x36}),
    AddressingMode.BP_PX: frozenset({0x24, 0x26}),
}


# No ignored-PRE pair currently has a proved executable boundary. PC-E500
# 0xF0002 was previously accepted as PRE23; SUB A,3F, but fresh decoding from
# the real 0xF0000 memory-card entry shows that 0x23 is the second byte of
# ``FD 23`` (MV BA,I) and 0x48 starts the following instruction at 0xF0003.
# Keep the registry so a future silicon- or boundary-proved pair can be added
# explicitly; all present PRE-insensitive combinations fail closed.
ROM_PROVEN_IGNORED_PRE_PAIRS: frozenset[tuple[int, int]] = frozenset()


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


WAITIntrinsic = IntrinsicName("WAIT")
TCLIntrinsic = IntrinsicName("TCL")
HALTIntrinsic = IntrinsicName("HALT")
OFFIntrinsic = IntrinsicName("OFF")
RESETIntrinsic = IntrinsicName("RESET")
ValidateFIntrinsic = IntrinsicName("VALIDATE_F")
PreflightVectorTransferIntrinsic = IntrinsicName("PREFLIGHT_VECTOR_TRANSFER")
ValidateVectorTransferIntrinsic = IntrinsicName("VALIDATE_VECTOR_TRANSFER")

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
TempWideMemoryValue = LLIL_TEMP(14)
TempWideMemoryAddress = LLIL_TEMP(15)
# Counted instructions cannot also execute a wide-memory helper inside their
# LLIL, so TEMP15 is safe to reuse for the initial-I snapshot.
TempInitialICount = TempWideMemoryAddress
# IR does not execute exchange logic, so this per-instruction temporary can be
# safely reused without expanding the public snapshot/native-register layout.
TempVectorTarget = TempExchange

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

    # ---------------------------------------------------------------------
    # IOCS/FCS "logic registers" (TRM internal RAM scratchpad at 0xD4–0xDF).
    #
    # These are not hardware MMIO registers; they are conventional parameter
    # slots used by higher-level ROM services (FCS/IOCS) when passing values
    # that don't fit in the CPU registers.
    #
    # Layout (little-endian):
    #   (bx) = [BH:BL] at 0xD5:0xD4
    #   (cx) = [CH:CL] at 0xD7:0xD6
    #   (dx) = [DH:DL] at 0xD9:0xD8
    #   (si) = raw three-byte image at 0xDA..0xDC
    #   (di) = raw three-byte image at 0xDD..0xDF
    # ---------------------------------------------------------------------
    BL = 0xD4
    BH = 0xD5
    CL = 0xD6
    CH = 0xD7
    DL = 0xD8
    DH = 0xD9
    SI = 0xDA
    SI1 = 0xDB
    SI2 = 0xDC
    DI = 0xDD
    DI1 = 0xDE
    DI2 = 0xDF

    # ---------------------------------------------------------------------
    # IOCS workspace base pointer (TRM "(E6)" style parameter).
    #
    # PC-E500 ROM code keeps a three-byte pointer image here and indexes IOCS
    # state as `[(E6)+offset]`. External consumers use bits 19-0. The bytes are
    # little-endian: LO/MID/HI.
    # ---------------------------------------------------------------------
    IOCS_WS = 0xE6
    IOCS_WS1 = 0xE7
    IOCS_WS2 = 0xE8

    # Alternative spellings (older name; keep working in asm/code).
    IOCS_WORKSPACE = IOCS_WS
    IOCS_WORKSPACE1 = IOCS_WS1
    IOCS_WORKSPACE2 = IOCS_WS2

    # Shorthand aliases (keeps older disassembly-style `E6`/`E7`/`E8` usable in asm).
    E6 = IOCS_WS
    E7 = IOCS_WS1
    E8 = IOCS_WS2

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
        # Only PRE can fuse with the following instruction. Looking ahead for
        # every ordinary opcode performs an architecturally spurious fetch of
        # the next instruction and can consume a callback-backed bus value.
        if instr1.opcode not in PRE_BY_OPCODE:
            yield instr1, addr1
            try:
                instr1, addr1 = next(instr_iter)
            except (StopIteration, NotImplementedError):
                break
            continue

        try:
            instr2, addr2 = next(instr_iter)
        except (StopIteration, NotImplementedError):
            yield instr1, addr1
            break

        if instr12 := instr1.fuse(instr2):
            instr1 = instr12
            continue

        yield instr1, addr1
        instr1, addr1 = instr2, addr2


def _create_decoder(
    decoder: Decoder, addr: int, opcodes: Dict[int, OpcodesType]
) -> Iterator[Tuple["Instruction", int]]:
    return fusion(iter_decode(decoder, addr, opcodes))


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
            # Some operands (e.g. RegPair) require instruction context while decoding.
            setattr(op, "_parent_instruction", self)
            op.decode(decoder, addr)
            # Set width for operands that support it based on instruction name
            set_width_fn = getattr(op, "set_width_from_instruction", None)
            if callable(set_width_fn):
                set_width_fn(self)

    def set_length(self, length: int) -> None:
        self._length = length

    def encode(self, encoder: Encoder, addr: int) -> None:
        assert self.opcode is not None, "Opcode not set"
        self._validate_pre_for_encode()
        start_pos = len(encoder.buf)
        if self._pre is not None:
            encoder.unsigned_byte(self._pre)
        encoder.unsigned_byte(self.opcode)
        for op in self.operands_coding():
            # Keep encode-time validation under the same instruction context as
            # decode-time validation.  Composite encodings such as RegPair need
            # the opcode to distinguish legal selector classes; without this,
            # the assembler could emit bytes that this decoder immediately
            # rejects (for example ``ADD BA, X`` -> ``44 24``).
            setattr(op, "_parent_instruction", self)
            op.encode(encoder, addr)
        encoded_length = len(encoder.buf) - start_pos
        decoded_length = getattr(self, "_length", None)
        if decoded_length is not None and encoded_length != decoded_length:
            raise InvalidInstruction(
                f"Encoded {self.name()} consumes {encoded_length} bytes, "
                f"but its decoded length is {decoded_length}"
            )

    @staticmethod
    def _declared_pre_mode(operand: Operand) -> Optional[AddressingMode]:
        if isinstance(operand, IMemOperand):
            return operand.mode
        if isinstance(operand, IMem8):
            mode = getattr(operand, "_asm_addressing_mode", None)
            return mode if isinstance(mode, AddressingMode) else None
        if isinstance(operand, EMemValueOffsetHelper):
            return Instruction._declared_pre_mode(operand.value)
        return None

    @staticmethod
    def _ignored_selector(operand: Operand) -> Optional[int]:
        if isinstance(operand, IMemOperand):
            # Logical BP+PX/BP+PY syntax has no selector expression; its
            # encoder deliberately materializes the required canonical zero.
            return 0 if operand.n_val is None else operand.n_val
        if isinstance(operand, IMem8):
            return operand.value
        if isinstance(operand, EMemValueOffsetHelper):
            return Instruction._ignored_selector(operand.value)
        return None

    def _validate_pre_for_encode(self) -> None:
        """Require the one canonical PRE byte for the encoded semantics.

        PRE fusion applies this rule while decoding.  Enforce the same contract
        for callers that construct or mutate :class:`Instruction` objects
        directly, otherwise encode() can manufacture bytes that decode() rejects.
        """

        if self._pre is not None and self._pre not in PRE_BY_OPCODE:
            raise InvalidInstruction(
                f"Unknown PRE value {self._pre!r} for {self.name()}"
            )

        operands = tuple(self.operands())
        pre_operand_indexes = [
            index
            for index, operand in enumerate(operands)
            if self._operand_uses_pre_mode(operand)
        ]
        if not pre_operand_indexes:
            if self._pre is None:
                return
            if (self._pre, self.opcode) in ROM_PROVEN_IGNORED_PRE_PAIRS:
                return
            raise InvalidInstruction(
                f"PRE{self._pre:02X} cannot prefix PRE-insensitive {self.name()}"
            )

        dst_mode, src_mode = self._addressing_modes()
        modes = (dst_mode, src_mode)

        for operand_index in pre_operand_indexes:
            effective_mode = modes[0 if operand_index == 0 else 1]
            declared_mode = self._declared_pre_mode(operands[operand_index])
            if declared_mode is not None and declared_mode != effective_mode:
                raise InvalidInstruction(
                    f"{self.name()} operand {operand_index + 1} declares "
                    f"{declared_mode.value}, but its PRE encoding selects "
                    f"{effective_mode.value}"
                )

        if len(pre_operand_indexes) == 1:
            operand_index = pre_operand_indexes[0]
            effective_mode = modes[0 if operand_index == 0 else 1]
            canonical_pre = SINGLE_OPERAND_PRE_LOOKUP.get(effective_mode)
        else:
            canonical_pre = REVERSE_PRE_TABLE.get((dst_mode, src_mode))

        if canonical_pre != self._pre:
            actual = "no prefix" if self._pre is None else f"PRE{self._pre:02X}"
            expected = (
                "no prefix" if canonical_pre is None else f"PRE{canonical_pre:02X}"
            )
            raise InvalidInstruction(
                f"Noncanonical {actual} for {self.name()}; use {expected}"
            )

        for operand_index in pre_operand_indexes:
            effective_mode = modes[0 if operand_index == 0 else 1]
            if effective_mode not in (AddressingMode.BP_PX, AddressingMode.BP_PY):
                continue
            selector = self._ignored_selector(operands[operand_index])
            if selector != 0:
                raise InvalidInstruction(
                    f"Nonzero ignored selector {selector!r} for {effective_mode.value}"
                )

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

    def _addressing_modes(self) -> Tuple[AddressingMode, AddressingMode]:
        dst_mode = get_addressing_mode(self._pre, 1)
        src_mode = get_addressing_mode(self._pre, 2)

        # For single addressable operand instructions, always use PRE1.
        if self.opcode in SINGLE_ADDRESSABLE_OPCODES:
            src_mode = dst_mode

        # Absolute external-memory forms such as MVW [addr], (n) have two
        # logical operands but only one operand that consumes the PRE latch.
        # Hardware applies PRE1 to that lone internal-memory selector.
        if self._pre is not None:
            operands = tuple(self.operands())
            pre_operand_indexes = [
                index
                for index, operand in enumerate(operands)
                if self._operand_uses_pre_mode(operand)
            ]
            if len(pre_operand_indexes) == 1 and pre_operand_indexes[0] == 1:
                src_mode = dst_mode

        # RegIMemOffset IMEM selector follows PRE1 as well.
        # Keep no-PRE behavior unchanged (operand 2 defaults to BP+n rendering).
        if (
            self._pre is not None
            and len(self._operands) == 1
            and self._operands[0].__class__.__name__ == "RegIMemOffset"
        ):
            src_mode = dst_mode

        return dst_mode, src_mode

    def _operand_uses_pre_mode(self, operand: Operand) -> bool:
        if isinstance(operand, (IMem8, IMemOperand)):
            return True
        if isinstance(operand, EMemValueOffsetHelper):
            return isinstance(operand.value, (IMem8, IMemOperand))
        return False

    def render(self) -> List[Token]:
        dst_mode, src_mode = self._addressing_modes()

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
        dst_mode, src_mode = self._addressing_modes()

        operands = tuple(self.operands())
        if not operands:
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
        # The instruction format always carries the operand byte. BP+PX and
        # BP+PY ignore it during address calculation, so emit a canonical zero.
        if self.mode in [
            AddressingMode.N,
            AddressingMode.BP_N,
            AddressingMode.PX_N,
            AddressingMode.PY_N,
        ]:
            if not isinstance(self.n_val, int) or not 0 <= self.n_val <= 0xFF:
                raise InvalidInstruction(
                    f"Internal-memory selector out of range at {addr:04X}: "
                    f"{self.n_val!r}"
                )
            encoder.unsigned_byte(self.n_val)
        else:
            if self.n_val not in (None, 0):
                raise InvalidInstruction(
                    f"Nonzero ignored selector {self.n_val!r} for {self.mode.value}"
                )
            encoder.unsigned_byte(0)

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
        if not 0 <= self.value <= 0xFF:
            raise ValueError(
                f"8-bit immediate out of range: {self.value:#x}; expected 0..0xff"
            )
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
        if not 0 <= self.value <= 0xFFFF:
            raise ValueError(
                f"16-bit immediate out of range: {self.value:#x}; expected 0..0xffff"
            )
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
        raw_hi = decoder.unsigned_byte()
        if raw_hi & 0xF0:
            raise InvalidInstruction(
                f"20-bit immediate has reserved upper bits at {addr:04X}: "
                f"0x{raw_hi:02X}"
            )
        self.extra_hi = raw_hi
        self.value = lo | (mid << 8) | (raw_hi << 16)

    def encode(self, encoder: Encoder, addr: int) -> None:
        assert self.value is not None, "Value not set"
        assert self.extra_hi is not None, "Extra high byte not set"
        if not 0 <= self.value <= PC_MASK:
            raise ValueError(
                f"20-bit immediate out of range: {self.value:#x}; "
                f"expected 0..{PC_MASK:#x}"
            )
        if not 0 <= self.extra_hi <= 0x0F:
            raise ValueError(
                f"20-bit immediate high byte out of range: {self.extra_hi:#x}; "
                "expected 0..0x0f"
            )
        expected_hi = (self.value >> 16) & 0x0F
        if self.extra_hi != expected_hi:
            raise InvalidInstruction(
                f"20-bit immediate value {self.value:#x} disagrees with encoded "
                f"high byte {self.extra_hi:#x}"
            )
        encoder.unsigned_byte(self.value & 0xFF)
        encoder.unsigned_byte((self.value >> 8) & 0xFF)
        encoder.unsigned_byte(self.extra_hi)

    def render(self, pre: Optional[AddressingMode] = None) -> List[Token]:
        return [TInt(f"{self.value:05X}")]


class RegisterImm20(Imm20):
    """Three-byte immediate loaded into X/Y/U/S as a 20-bit value.

    The instruction fetch still consumes all three encoded bytes, but the
    register keeps only bits 19-0. PC-E500 capture reports show that
    ``MV X/Y, 0x3C5AA5`` subsequently pushes ``A5 5A 0C``. The report does not
    isolate whether hardware masks at load or push; normalization at register
    write is the architectural emulator model. This is not the same contract
    as an address/control-flow operand, whose nonzero encoded upper nibble
    remains unverified and is rejected by :class:`Imm20`.

    Fresh assembly stays canonical: source values must fit in 20 bits and the
    encoder writes a zero upper nibble.  Decoding a hardware-accepted alias
    normalizes it to that effective value instead of preserving ignored bits.
    """

    def decode(self, decoder: Decoder, addr: int) -> None:
        lo = decoder.unsigned_byte()
        mid = decoder.unsigned_byte()
        raw_hi = decoder.unsigned_byte()
        self.extra_hi = raw_hi
        self.value = lo | (mid << 8) | ((raw_hi & 0x0F) << 16)

    def encode(self, encoder: Encoder, addr: int) -> None:
        assert self.value is not None, "Value not set"
        if not 0 <= self.value <= PC_MASK:
            raise ValueError(
                f"20-bit register immediate out of range: {self.value:#x}; "
                f"expected 0..{PC_MASK:#x}"
            )
        encoder.unsigned_byte(self.value & 0xFF)
        encoder.unsigned_byte((self.value >> 8) & 0xFF)
        encoder.unsigned_byte((self.value >> 16) & 0x0F)


class FarControlImm20(RegisterImm20):
    """JPF/CALLF target whose encoded upper nibble is ignored by silicon.

    PC-E500 HW-009 pairs executed otherwise-identical JPF and CALLF opcodes
    with high operand bytes ``0x01`` and ``0x81``. Both variants transferred
    to ``0x101E0``; CALLF also produced the same return frame and RETF path.
    Decode therefore consumes all 24 encoded bits but retains bits 19-0.
    Fresh assembly remains canonical through :class:`RegisterImm20.encode`.

    This policy is intentionally opcode-specific. It does not relax vector
    decoding or unmeasured 20-bit operand families.
    """


# Raw three-byte immediate encoded as `n m l`.  Unlike Imm20, every bit is
# data: opcode DC uses this operand to seed an internal-memory triple, and real
# ROM/hardware examples rely on the upper nibble of `l` being preserved.
class Imm24(ImmOperand):
    def __init__(self) -> None:
        super().__init__()
        self.value = None

    def width(self) -> int:
        return 3

    def decode(self, decoder: Decoder, addr: int) -> None:
        lo = decoder.unsigned_byte()
        mid = decoder.unsigned_byte()
        hi = decoder.unsigned_byte()
        self.value = lo | (mid << 8) | (hi << 16)

    def encode(self, encoder: Encoder, addr: int) -> None:
        assert self.value is not None, "Value not set"
        if not 0 <= self.value <= 0xFFFFFF:
            raise ValueError(
                f"24-bit immediate out of range: {self.value:#x}; expected 0..0xffffff"
            )
        encoder.unsigned_byte(self.value & 0xFF)
        encoder.unsigned_byte((self.value >> 8) & 0xFF)
        encoder.unsigned_byte((self.value >> 16) & 0xFF)

    def render(self, pre: Optional[AddressingMode] = None) -> List[Token]:
        return [TInt(f"{self.value:06X}")]


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
        self.offset = None
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

    def _validate_offset_shape(self, addr: int) -> None:
        mode = getattr(self, "mode", None)
        if mode is None:
            raise InvalidInstruction(f"Offset addressing mode is not set at {addr:04X}")
        mode_enum = type(mode)
        positive = getattr(mode_enum, "POSITIVE_OFFSET", None)
        negative = getattr(mode_enum, "NEGATIVE_OFFSET", None)
        expected_sign: Optional[Literal["+", "-"]] = None
        if mode == positive:
            expected_sign = "+"
        elif mode == negative:
            expected_sign = "-"

        if expected_sign is None:
            if self.offset is not None:
                raise InvalidInstruction(
                    f"External-memory mode {mode.name} must not encode an offset "
                    f"at {addr:04X}"
                )
            return
        if self.offset is None:
            raise InvalidInstruction(
                f"External-memory mode {mode.name} requires one offset byte "
                f"at {addr:04X}"
            )
        if self.offset.sign != expected_sign:
            raise InvalidInstruction(
                f"External-memory mode {mode.name} requires a {expected_sign} offset, "
                f"got {self.offset.sign} at {addr:04X}"
            )

    def _encode_offset(self, encoder: Encoder, addr: int) -> None:
        self._validate_offset_shape(addr)
        if self.offset is None:
            return
        self.offset.encode(encoder, addr)


def _wrapped_memory_addr(
    il: LowLevelILFunction,
    base_addr: ExpressionIndex,
    byte_offset: int,
    *,
    address_mask: int,
    region_base: int = 0,
) -> ExpressionIndex:
    """Return one byte address wrapped inside an architectural memory region."""
    relative = base_addr
    if region_base:
        relative = il.sub(3, relative, il.const(3, region_base))
    relative = il.add(3, relative, il.const(3, byte_offset))
    relative = il.and_expr(3, relative, il.const(3, address_mask))
    if region_base:
        relative = il.add(3, il.const(3, region_base), relative)
    return relative


def _zero_extend_byte(
    il: LowLevelILFunction, width: int, value: ExpressionIndex
) -> ExpressionIndex:
    if width == 1:
        return value
    # Binary Ninja exposes ZERO_EXT, while the lightweight test LLIL currently
    # does not.  Its width-selected AND has the same integer semantics here.
    zero_extend = getattr(il, "zero_extend", None)
    if callable(zero_extend):
        return zero_extend(width, value)
    return il.and_expr(width, value, il.const(width, 0xFF))


def _resize_unsigned(
    il: LowLevelILFunction,
    value: ExpressionIndex,
    source_width: int,
    target_width: int,
) -> ExpressionIndex:
    """Resize an unsigned LLIL value without relying on implicit coercion.

    Real Binary Ninja rejects arithmetic, register writes, and control-flow
    expressions whose child widths do not match the parent operation.  The
    lightweight evaluator is intentionally more permissive and does not expose
    either constructor, so it retains the value expression while focused
    width-strict tests provide those constructors and assert the real shape.
    """

    if source_width == target_width:
        return value
    if source_width < target_width:
        zero_extend = getattr(il, "zero_extend", None)
        if callable(zero_extend):
            return zero_extend(target_width, value)
        return value

    low_part = getattr(il, "low_part", None)
    if callable(low_part):
        return low_part(target_width, value)
    return value


def _low_byte(
    il: LowLevelILFunction, width: int, value: ExpressionIndex
) -> ExpressionIndex:
    if width == 1:
        return value
    # See _zero_extend_byte: LOW_PART is available in Binary Ninja but not in
    # the mock LLIL used by the executor tests.
    low_part = getattr(il, "low_part", None)
    if callable(low_part):
        return low_part(1, value)
    return il.and_expr(1, value, il.const(1, 0xFF))


def _lift_wrapped_memory_load(
    il: LowLevelILFunction,
    width: int,
    base_addr: ExpressionIndex,
    *,
    address_mask: int,
    region_base: int = 0,
) -> ExpressionIndex:
    """Load a little-endian value with per-byte address-bus wrapping."""
    if width == 1:
        return il.load(1, base_addr)

    result = il.const(width, 0)
    for byte_offset in range(width):
        byte = il.load(
            1,
            _wrapped_memory_addr(
                il,
                base_addr,
                byte_offset,
                address_mask=address_mask,
                region_base=region_base,
            ),
        )
        part = _zero_extend_byte(il, width, byte)
        if byte_offset:
            part = il.shift_left(width, part, il.const(1, byte_offset * 8))
        result = il.or_expr(width, result, part)
    return result


def _lift_wrapped_memory_store(
    il: LowLevelILFunction,
    width: int,
    base_addr: ExpressionIndex,
    value: ExpressionIndex,
    *,
    address_mask: int,
    region_base: int = 0,
) -> None:
    """Store a little-endian value with per-byte address-bus wrapping."""
    if width == 1:
        il.append(il.store(1, base_addr, value))
        return

    # Snapshot both expressions before the first byte store.  A memory-backed
    # source must not be re-evaluated after an overlapping destination changes
    # it, and an IMEM destination can overwrite BP/PX/PY while those registers
    # still participate in its effective-address expression.
    address_snapshot = TempReg(TempWideMemoryAddress, width=3)
    address_snapshot.lift_assign(il, base_addr)
    value_snapshot = TempReg(TempWideMemoryValue, width=width)
    value_snapshot.lift_assign(il, value)
    for byte_offset in range(width):
        part = value_snapshot.lift(il)
        if byte_offset:
            part = il.logical_shift_right(width, part, il.const(1, byte_offset * 8))
        byte = _low_byte(il, width, part)
        il.append(
            il.store(
                1,
                _wrapped_memory_addr(
                    il,
                    address_snapshot.lift(il),
                    byte_offset,
                    address_mask=address_mask,
                    region_base=region_base,
                ),
                byte,
            )
        )


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

        def _render_named_imem_addr(n_val: int) -> List[Token]:
            try:
                name = IMEMRegisters(n_val).name
            except ValueError:
                return [TInt(f"{n_val:02X}")]
            return [TText(name)]

        result: List[Token] = [TBegMem(MemType.INTERNAL)]
        match pre:
            case AddressingMode.N:
                if isinstance(self.value, IMemOperand) and isinstance(
                    self.value.n_val, int
                ):
                    result.extend(_render_named_imem_addr(self.value.n_val))
                elif isinstance(self.value, Imm8) and isinstance(self.value.value, int):
                    result.extend(_render_named_imem_addr(self.value.value))
                else:
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
                    3,
                    _resize_unsigned(
                        il,
                        self.value.lift(il),
                        self.value.width(),
                        3,
                    ),
                    il.const(3, INTERNAL_MEMORY_START),
                )

        if isinstance(self.value, ImmOperand) and pre == AddressingMode.N:
            assert self.value.value is not None, "Value not set"
            raw_addr = INTERNAL_MEMORY_START + self.value.value
            return il.const_pointer(3, raw_addr)

        # BP/PX/PY arithmetic wraps in the byte-wide IMEM offset space.  Lift
        # that byte explicitly into the three-byte address space before adding
        # the synthetic internal-memory base; mixed ADD.l(ADD.b, CONST.l) is
        # rejected by real Binary Ninja even though the test evaluator accepts
        # it.
        offset = _resize_unsigned(il, self._imem_offset(il, pre), 1, 3)
        return il.add(3, offset, il.const(3, INTERNAL_MEMORY_START))

    def lift(
        self,
        il: LowLevelILFunction,
        pre: Optional[AddressingMode] = None,
        side_effects: bool = True,
    ) -> ExpressionIndex:
        return _lift_wrapped_memory_load(
            il,
            self.width(),
            self.imem_addr(il, pre),
            address_mask=0xFF,
            region_base=INTERNAL_MEMORY_START,
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
        _lift_wrapped_memory_store(
            il,
            self.width(),
            self.imem_addr(il, pre),
            value,
            address_mask=0xFF,
            region_base=INTERNAL_MEMORY_START,
        )


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

        value = self.value.lift(il)
        width_fn = getattr(self.value, "width", None)
        if callable(width_fn):
            value = _resize_unsigned(il, value, width_fn(), 3)
        # X/Y/U/S are stored in three bytes but every external-memory consumer
        # observes the documented 20-bit bus, matching the runtime register
        # facade even if upstream dataflow has not yet proved the high nibble
        # clear.
        return il.and_expr(3, value, il.const(3, PC_MASK))

    def lift(
        self,
        il: LowLevelILFunction,
        pre: Optional[AddressingMode] = None,
        side_effects: bool = True,
    ) -> ExpressionIndex:
        return _lift_wrapped_memory_load(
            il,
            self.width(),
            self.emem_addr(il),
            address_mask=PC_MASK,
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
        _lift_wrapped_memory_store(
            il,
            self.width(),
            self.emem_addr(il),
            value,
            address_mask=PC_MASK,
        )


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


# Read three bytes from internal memory. ``IMem20`` is a historical name, not
# a declaration that every consumer truncates the loaded bytes. A register or
# external-address consumer explicitly keeps bits 19-0; MVP copies all three
# bytes; CMPP compares three-byte values; and EXP exchanges all three bytes.
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
        il.append(il.set_reg(2, RegisterName("I"), _resize_unsigned(il, value, 1, 2)))


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
        # Validate the raw byte before changing either modeled flag so a
        # dubious stack frame cannot become a plausible-looking C/Z result.
        il.append(il.intrinsic([], ValidateFIntrinsic, [tmp.lift(il)]))
        self.lift_assign_validated(il, tmp.lift(il))

    def lift_assign_validated(
        self,
        il: LowLevelILFunction,
        value: ExpressionIndex,
    ) -> None:
        """Assign a byte already checked or normalized to modeled C/Z bits.

        Stack pops need validation to occur before the stack pointer advances.
        Keeping the flag writes separate avoids either validating twice or
        moving the pointer update ahead of validation.
        """

        il.append(il.set_flag(CFlag, il.and_expr(1, value, il.const(1, 1))))
        il.append(il.set_flag(ZFlag, il.and_expr(1, value, il.const(1, 2))))


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

    @staticmethod
    def _validate_selector(byte: int, parent_opcode: Optional[int], addr: int) -> None:
        """Validate opcode-specific register classes on decode and encode."""

        if parent_opcode == 0x11 and byte not in range(0x04, 0x08):
            raise InvalidInstruction(
                f"JP register selector must be X, Y, U, or S at {addr:04X}, "
                f"got {byte:02X}"
            )
        if parent_opcode in (0x6C, 0x7C) and byte not in range(0x00, 0x08):
            raise InvalidInstruction(
                f"INC/DEC register selector has reserved upper bits at {addr:04X}, "
                f"got {byte:02X}"
            )
        if parent_opcode == 0xD6 and byte not in (0x02, 0x03):
            raise InvalidInstruction(
                f"CMPW register selector must be BA or I at {addr:04X}, got {byte:02X}"
            )
        if parent_opcode == 0xD7 and byte not in range(0x04, 0x08):
            raise InvalidInstruction(
                f"CMPP register selector must be X, Y, U, or S at {addr:04X}, "
                f"got {byte:02X}"
            )

    def decode(self, decoder: Decoder, addr: int) -> None:
        byte = decoder.unsigned_byte()
        self.reg_raw = byte
        self.reg = self.reg_name(byte & 7)
        # store high 4 bits from byte for later reference
        self.high4 = (byte >> 4) & 0x0F

        parent = getattr(self, "_parent_instruction", None)
        parent_opcode = getattr(parent, "opcode", None)
        self._validate_selector(byte, parent_opcode, addr)

    def encode(self, encoder: Encoder, addr: int) -> None:
        assert self.reg_raw is not None, "Register raw value not set"
        assert self.high4 is not None, "High 4 bits not set"
        if not 0 <= self.reg_raw <= 0xFF or not 0 <= self.high4 <= 0x0F:
            raise InvalidInstruction(
                f"Register selector fields out of range at {addr:04X}: "
                f"raw={self.reg_raw!r}, high4={self.high4!r}"
            )
        raw_high4 = (self.reg_raw >> 4) & 0x0F
        if raw_high4 not in (0, self.high4):
            raise InvalidInstruction(
                f"Register selector raw high nibble {raw_high4:X} disagrees with "
                f"mode nibble {self.high4:X} at {addr:04X}"
            )
        if self.reg is None:
            raise InvalidInstruction(
                f"Register selector has no semantic register at {addr:04X}"
            )
        expected_selector = self.reg_idx(self.reg)
        raw_selector = self.reg_raw & 0x0F
        if raw_selector != expected_selector:
            raise InvalidInstruction(
                f"Register selector encodes {self.reg_name(self.reg_raw & 0x07)} "
                f"but operand names {self.reg} at {addr:04X}"
            )
        byte = raw_selector | (self.high4 << 4)
        parent = getattr(self, "_parent_instruction", None)
        self._validate_selector(byte, getattr(parent, "opcode", None), addr)
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
        return _lift_wrapped_memory_load(
            il,
            self.width(),
            il.const_pointer(3, self.value),
            address_mask=PC_MASK,
        )

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
        _lift_wrapped_memory_store(
            il,
            self.width(),
            il.const_pointer(3, self.value),
            value,
            address_mask=PC_MASK,
        )


class DirectReadAbsoluteAddr(EMemAddr):
    """Absolute data address used by opcodes 88-8F.

    Paired PC-E500 HW-009 captures exercised every direct-register load width
    with canonical high byte ``01`` and raw high byte ``81``. Both forms
    consumed all three address bytes and read the same one-, two-, or
    three-byte value at 0x101F0. Decode therefore ignores bits 23-20 for this
    direct-read block. Re-encoding stays canonical, and the policy does not
    extend to control flow, vectors, or other absolute-memory opcode families.
    """

    def decode(self, decoder: Decoder, addr: int) -> None:
        lo = decoder.unsigned_byte()
        mid = decoder.unsigned_byte()
        raw_hi = decoder.unsigned_byte()
        self.extra_hi = raw_hi & 0x0F
        self.value = lo | (mid << 8) | (self.extra_hi << 16)


class DirectWriteAbsoluteAddr(EMemAddr):
    """Absolute data address used by opcodes A8-AF.

    A paired PC-E500 HW-009 matrix captured every A8-AF register width with
    encoded high bytes ``04`` and ``84``. Both forms consumed all three
    address bytes and emitted the same one-, two-, or three-byte CE1 write
    sequence at 0x406D0. Decode therefore ignores bits 23-20 for these direct
    stores. Encoding remains canonical and this policy does not extend to
    control flow, vectors, or other absolute-memory opcode families.
    """

    def decode(self, decoder: Decoder, addr: int) -> None:
        lo = decoder.unsigned_byte()
        mid = decoder.unsigned_byte()
        raw_hi = decoder.unsigned_byte()
        self.extra_hi = raw_hi & 0x0F
        self.value = lo | (mid << 8) | (self.extra_hi << 16)


class AbsoluteByteOpAddr(EMemAddr):
    """Absolute byte address used by opcodes 62/66/6A/72/7A.

    Paired PC-E500 HW-009 captures used high bytes ``04`` and ``84`` for CMP,
    TEST, XOR, AND, and OR at 0x406D0. Both forms consumed the complete
    five-byte instruction. CMP/TEST emitted the same one-byte read; the three
    identity read/modify/write cases emitted the same read then write address.
    Decode therefore ignores bits 23-20 only for this measured opcode family.
    """

    def decode(self, decoder: Decoder, addr: int) -> None:
        lo = decoder.unsigned_byte()
        mid = decoder.unsigned_byte()
        raw_hi = decoder.unsigned_byte()
        self.extra_hi = raw_hi & 0x0F
        self.value = lo | (mid << 8) | (self.extra_hi << 16)


class AbsoluteTransferAddr(EMemAddr):
    """Absolute data address used by opcodes D0-D3 and D8-DB.

    Paired PC-E500 HW-009 captures exercised fixed one-, two-, and three-byte
    moves plus I=3 MVL in both directions. Encoded high bytes 01/81 selected
    the same experiment-ROM source for D0-D3; 04/84 emitted the same bounded
    CE1 destination sequence for D8-DB. Decode therefore ignores bits 23-20
    only for these eight measured transfer opcodes. Encoding remains
    canonical and control-flow/vector operands remain strict.
    """

    def decode(self, decoder: Decoder, addr: int) -> None:
        lo = decoder.unsigned_byte()
        mid = decoder.unsigned_byte()
        raw_hi = decoder.unsigned_byte()
        self.extra_hi = raw_hi & 0x0F
        self.value = lo | (mid << 8) | (self.extra_hi << 16)


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
            # Now load the three-byte pointer, wrapping FF -> 00 inside IMEM.
            addr = _lift_wrapped_memory_load(
                il,
                3,
                imem_addr,
                address_mask=0xFF,
                region_base=INTERNAL_MEMORY_START,
            )
        else:
            addr = self.value.lift(il, pre=pre, side_effects=side_effects)

        if self.offset:
            addr = self.offset.lift_offset(il, addr)
        # External addresses are carried in three-byte LLIL values but the
        # SC62015 address bus is 20 bits.  Canonicalize both an indirect
        # pointer's ignored upper nibble and +/- offset wraparound.
        return il.and_expr(3, addr, il.const(3, PC_MASK))

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
        return _lift_wrapped_memory_load(
            il,
            self.width(),
            self.lift_current_addr(il, pre=pre, side_effects=side_effects),
            address_mask=PC_MASK,
        )

    def lift_assign(
        self,
        il: LowLevelILFunction,
        value: ExpressionIndex,
        pre: Optional[AddressingMode] = None,
    ) -> None:
        addr = self.lift_current_addr(il, pre=pre, side_effects=True)
        _lift_wrapped_memory_store(
            il,
            self.width(),
            addr,
            value,
            address_mask=PC_MASK,
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


EMEM_REG_OFFSET_MODES = (
    EMemRegMode.POSITIVE_OFFSET,
    EMemRegMode.NEGATIVE_OFFSET,
)
EMEM_REG_BLOCK_MODES = (
    EMemRegMode.POST_INC,
    EMemRegMode.PRE_DEC,
)


def get_emem_reg_mode(val: Optional[int], addr: int) -> EMemRegMode:
    try:
        return EMemRegMode(val)
    except Exception:
        raise InvalidInstruction(
            f"Invalid EMemRegMode {val:02X} at address {addr:#06x}"
        )


def _validate_emem_reg_mode(
    *,
    reg: Reg3,
    mode: Optional[EMemRegMode],
    allowed_modes: Optional[Iterable[EMemRegMode]],
    parent_opcode: Optional[int],
    addr: int,
) -> None:
    """Keep external-register mode metadata, encoded bits, and opcode policy aligned."""

    if mode is None:
        raise InvalidInstruction(f"External-memory mode is not set at {addr:04X}")
    assert reg.reg_raw is not None, "Register raw value not set"
    assert reg.high4 is not None, "Register high nibble not set"

    # Reg3 preserves the original raw byte for byte-exact ROM round trips, while
    # assemblers normally keep only the low selector bits in ``reg_raw``.  Check
    # the byte that encode() will actually emit, not either field in isolation.
    encoded_byte = reg.reg_raw | (reg.high4 << 4)
    encoded_mode = get_emem_reg_mode((encoded_byte >> 4) & 0x0F, addr)
    if encoded_mode != mode:
        raise InvalidInstruction(
            f"External-memory mode metadata {mode.name} does not match encoded "
            f"mode {encoded_mode.name} at {addr:04X}"
        )

    opcode_modes = {
        0x56: EMEM_REG_OFFSET_MODES,
        0x5E: EMEM_REG_OFFSET_MODES,
        0xE3: EMEM_REG_BLOCK_MODES,
        0xEB: EMEM_REG_BLOCK_MODES,
    }.get(parent_opcode)
    declared_modes = tuple(allowed_modes) if allowed_modes is not None else None
    if opcode_modes is None:
        effective_modes = declared_modes
    elif declared_modes is None:
        # Direct Instruction construction must not bypass opcode restrictions
        # merely by omitting the table operand's allowed_modes metadata.
        effective_modes = opcode_modes
    else:
        effective_modes = tuple(mode for mode in opcode_modes if mode in declared_modes)

    if effective_modes is not None and mode not in effective_modes:
        raise InvalidInstruction(
            f"Invalid external-memory mode {mode.name} at {addr:04X}; "
            f"allowed modes: {', '.join(item.name for item in effective_modes)}"
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
                il.and_expr(
                    self.reg.width(),
                    il.add(
                        self.reg.width(),
                        value,
                        il.const(self.reg.width(), self.width),
                    ),
                    il.const(self.reg.width(), PC_MASK),
                ),
            )
            value = tmp.lift(il)

        if side_effects and self.mode == EMemRegMode.PRE_DEC:
            # For pre-decrement with side effects:
            # 1. Calculate the decremented value expression
            # 2. Store it in a temp register to capture the value
            # 3. Update the actual register with the same expression
            # 4. Return the temp register

            # Calculate the decremented value
            new_value = il.and_expr(
                self.reg.width(),
                il.sub(self.reg.width(), value, il.const(self.reg.width(), self.width)),
                il.const(self.reg.width(), PC_MASK),
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
            value = il.and_expr(
                self.reg.width(),
                il.sub(self.reg.width(), value, il.const(self.reg.width(), self.width)),
                il.const(self.reg.width(), PC_MASK),
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

        asm_mode = getattr(self.imem, "_asm_addressing_mode", None)
        if asm_mode is not None:
            setattr(imem_operand, "_asm_addressing_mode", asm_mode)

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
        parent = getattr(self, "_parent_instruction", None)
        _validate_emem_reg_mode(
            reg=self.reg,
            mode=self.mode,
            allowed_modes=self.allowed_modes,
            parent_opcode=getattr(parent, "opcode", None),
            addr=addr,
        )
        self._decode_offset(decoder, addr)

    def encode(self, encoder: Encoder, addr: int) -> None:
        assert self.reg is not None, "Register not set"
        self.reg.assert_r3()
        parent = getattr(self, "_parent_instruction", None)
        _validate_emem_reg_mode(
            reg=self.reg,
            mode=self.mode,
            allowed_modes=self.allowed_modes,
            parent_opcode=getattr(parent, "opcode", None),
            addr=addr,
        )
        self._validate_offset_shape(addr)
        setattr(
            self.reg,
            "_parent_instruction",
            parent,
        )
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
        parent = getattr(self, "_parent_instruction", None)
        _validate_emem_reg_mode(
            reg=self.reg,
            mode=self.mode,
            allowed_modes=self.allowed_modes,
            parent_opcode=getattr(parent, "opcode", None),
            addr=addr,
        )
        self._decode_offset(decoder, addr)

    def encode(self, encoder: Encoder, addr: int) -> None:
        # super().encode(encoder, addr)
        self.reg.assert_r3()
        parent = getattr(self, "_parent_instruction", None)
        _validate_emem_reg_mode(
            reg=self.reg,
            mode=self.mode,
            allowed_modes=self.allowed_modes,
            parent_opcode=getattr(parent, "opcode", None),
            addr=addr,
        )
        self._validate_offset_shape(addr)
        setattr(
            self.reg,
            "_parent_instruction",
            parent,
        )
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
        if self.mode is None:
            raise InvalidInstruction(f"External-memory mode is not set at {addr:04X}")
        if self.value != self.mode.value:
            raise InvalidInstruction(
                f"External-memory mode metadata {self.mode.name} does not match "
                f"encoded mode byte {self.value!r} at {addr:04X}"
            )
        # Validate arity before writing the mode or selector bytes, so malformed
        # direct operands cannot leave a plausibly truncated instruction behind.
        self._validate_offset_shape(addr)
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

        for source, expanded in (
            (self.imem1, imem1_operand),
            (self.imem2, imem2_operand),
        ):
            asm_mode = getattr(source, "_asm_addressing_mode", None)
            if asm_mode is not None:
                setattr(expanded, "_asm_addressing_mode", asm_mode)

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
        if self.mode is None:
            raise InvalidInstruction(f"External-memory mode is not set at {addr:04X}")
        if self.mode_imm.value != self.mode.value:
            raise InvalidInstruction(
                f"External-memory mode metadata {self.mode.name} does not match "
                f"encoded mode byte {self.mode_imm.value!r} at {addr:04X}"
            )
        self._validate_offset_shape(addr)
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

    @staticmethod
    def _regpair_name(code: int, use_r2: bool) -> RegisterName:
        idx = code & 0x7
        if use_r2:
            if idx in (0, 2):
                return RegisterName("BA")
            if idx in (1, 3):
                return RegisterName("I")
            if idx == 4:
                return RegisterName("X")
            if idx == 5:
                return RegisterName("Y")
            if idx == 6:
                return RegisterName("U")
            return RegisterName("S")

        if idx == 0:
            return RegisterName("A")
        if idx == 1:
            return RegisterName("IL")
        if idx == 2:
            return RegisterName("BA")
        if idx == 3:
            return RegisterName("I")
        if idx == 4:
            return RegisterName("X")
        if idx == 5:
            return RegisterName("Y")
        if idx == 6:
            return RegisterName("U")
        return RegisterName("S")

    @staticmethod
    def _regpair_is_20bit(reg: RegisterName) -> bool:
        return reg in (
            RegisterName("X"),
            RegisterName("Y"),
            RegisterName("U"),
            RegisterName("S"),
        )

    @staticmethod
    def _uses_r2_mapping(parent: Optional[Any]) -> bool:
        if parent is None:
            return False
        return parent.name() in ("MV", "EX")

    def bit_width(self) -> int:
        if self.size == 1:
            return 8
        if self.size == 2:
            parent = getattr(self, "_parent_instruction", None)
            if (
                self._uses_r2_mapping(parent)
                and self.reg1 is not None
                and self.reg2 is not None
                and (
                    self._regpair_is_20bit(self.reg1.reg)
                    or self._regpair_is_20bit(self.reg2.reg)
                )
            ):
                return 20
            return 16
        if self.size == 3:
            return 20
        if self.reg1 is not None:
            return self.reg1.width() * 8
        return 8

    @staticmethod
    def _validate_selector(raw: int, parent_opcode: Optional[int], addr: int) -> None:
        """Validate one physical register-pair selector.

        This is deliberately shared by decode and encode so the assembler
        cannot manufacture an instruction that the architecture refuses to
        decode.  Arithmetic opcodes use different destination/source register
        classes; mixed-width forms remain legal where the ISA and stock ROM use
        them (for example ``ADD Y, BA`` at PC-E500 ROM F2B62).
        """

        if raw & 0x80:
            raise InvalidInstruction(
                f"Invalid reg1 high bit in register pair {raw:02X} at {addr:04X}"
            )
        if raw & 0x08:
            raise InvalidInstruction(
                f"Invalid reg2 high bit in register pair {raw:02X} at {addr:04X}"
            )

        reg1_code = (raw >> 4) & 7
        reg2_code = raw & 7
        arithmetic_classes = {
            0x44: (range(2, 4), range(0, 4)),  # ADD r2,r1 or r2,r2
            0x45: (range(4, 8), range(0, 8)),  # ADD r3,r
            0x46: (range(0, 2), range(0, 2)),  # ADD r1,r1
            0x4C: (range(2, 4), range(0, 4)),  # SUB r2,r1 or r2,r2
            0x4D: (range(4, 8), range(0, 8)),  # SUB r3,r
            0x4E: (range(0, 2), range(0, 2)),  # SUB r1,r1
        }
        allowed = arithmetic_classes.get(parent_opcode)
        if allowed is None:
            return
        allowed_dest, allowed_src = allowed
        if reg1_code not in allowed_dest or reg2_code not in allowed_src:
            raise InvalidInstruction(
                f"Invalid arithmetic register pair {raw:02X} "
                f"for opcode {parent_opcode:02X} at {addr:04X}"
            )

    def _validate_encode_semantics(
        self, parent_opcode: Optional[int], addr: int
    ) -> None:
        """Reject source operands that collapse through the ED/FD r2 aliases.

        Decoder aliases remain byte-preserving: a decoded alias already carries
        the architectural BA/I names even when its raw selector nibble is 0/1.
        Fresh assembly using A/IL, however, must not silently change meaning.
        """

        if parent_opcode not in (0xED, 0xFD):
            return
        invalid = {RegisterName("A"), RegisterName("IL")}
        semantic_regs = {
            getattr(self.reg1, "reg", None),
            getattr(self.reg2, "reg", None),
        }
        if semantic_regs & invalid:
            raise InvalidInstruction(
                f"{parent_opcode:02X} register pairs require BA, I, X, Y, U, or S; "
                f"A/IL would change meaning at {addr:04X}"
            )

    def decode(self, decoder: Decoder, addr: int) -> None:
        reg_raw = decoder.unsigned_byte()
        self.reg_raw = reg_raw
        parent = getattr(self, "_parent_instruction", None)
        parent_opcode = getattr(parent, "opcode", None)
        self._validate_selector(reg_raw, parent_opcode, addr)
        use_r2 = self._uses_r2_mapping(parent)
        reg1_code = (reg_raw >> 4) & 7
        reg2_code = reg_raw & 7
        self.reg1 = Reg(self._regpair_name(reg1_code, use_r2))
        self.reg2 = Reg(self._regpair_name(reg2_code, use_r2))

    def operands(self) -> Generator[Operand, None, None]:
        assert self.reg1 is not None, "Register 1 not set"
        assert self.reg2 is not None, "Register 2 not set"
        yield self.reg1
        yield self.reg2

    def encode(self, encoder: Encoder, addr: int) -> None:
        assert self.reg_raw is not None, "Register raw value not set"
        parent = getattr(self, "_parent_instruction", None)
        parent_opcode = getattr(parent, "opcode", None)
        self._validate_selector(self.reg_raw, parent_opcode, addr)
        self._validate_encode_semantics(parent_opcode, addr)
        if self.reg1 is None or self.reg2 is None:
            raise InvalidInstruction(f"Register pair is incomplete at {addr:04X}")
        use_r2 = self._uses_r2_mapping(parent)
        encoded_reg1 = self._regpair_name((self.reg_raw >> 4) & 7, use_r2)
        encoded_reg2 = self._regpair_name(self.reg_raw & 7, use_r2)
        if self.reg1.reg != encoded_reg1 or self.reg2.reg != encoded_reg2:
            raise InvalidInstruction(
                f"Register pair selector {self.reg_raw:02X} encodes "
                f"{encoded_reg1},{encoded_reg2} but operand names "
                f"{self.reg1.reg},{self.reg2.reg} at {addr:04X}"
            )
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
    loop_label = LowLevelILLabel()
    exit_label = LowLevelILLabel()

    loop_reg = Reg("I")
    width = loop_reg.width()

    # PC-E500 HW-002 establishes a 16-bit do-while countdown: an initial
    # I=0 wraps through 0xffff and performs 65,536 iterations.
    il.mark_label(loop_label)
    yield

    loop_reg.lift_assign(il, il.sub(width, loop_reg.lift(il), il.const(width, 1)))
    cond = il.compare_equal(width, loop_reg.lift(il), il.const(width, 0))
    il.append(il.if_expr(cond, exit_label, loop_label))
    il.mark_label(exit_label)
