# based on https://github.com/whitequark/binja-avnera/blob/main/mc/instr.py
from .tokens import Token, TInstr, TText, TSep, TInt, TAddr, TReg, TBegMem, TEndMem, MemType
from .coding import Decoder, Encoder, BufferTooShort
from .mock_analysis import BranchType

import copy
from dataclasses import dataclass
from typing import Optional, List, Literal
import enum

from .binja_api import *
from binaryninja.lowlevelil import (
    LowLevelILLabel,
    LLIL_TEMP,
)


# mapping to size, page 67 of the book
REGISTERS = [
    # r1
    ("A", 1),
    ("IL", 1),
    # r2
    ("BA", 2),
    ("I", 2),
    # r3
    ("X", 4),  # r4, actually 3 bytes
    ("Y", 4),  # r4, actually 3 bytes
    ("U", 4),  # r4, actually 3 bytes
    ("S", 3),
]

REG_NAMES = [reg[0] for reg in REGISTERS]
REG_SIZES = {reg[0]: reg[1] for reg in REGISTERS}

# map internal memory to start at this address
INTERNAL_MEMORY_START = 0xFFFFF + 1
IMEM_NAMES = {
    0xEC: "BP", # RAM Base Pointer
    0xED: "PX", # RAM PX Pointer
    0xEE: "PY", # RAM PY Pointer
    0xEF: "AMC", # ADR Modify Control
    0xF0: "KOL", # Key Output Buffer H
    0xF1: "KOH", # Key Output Buffer L
    0xF2: "KIL", # Key Input Buffer
    0xF3: "EOL", # E Port Output Buffer H
    0xF4: "EOH", # E Port Output Buffer L
    0xF5: "EIL", # E Port Input Buffer H
    0xF6: "EIH", # E Port Input Buffer L
    0xF7: "UCR", # UART Control Register
    0xF8: "USR", # UART Status Register
    0xF9: "RXD", # UART Receive Buffer
    0xFA: "TXD", # UART Transmit Buffer
    0xFB: "IMR", # Interrupt Mask Register
    0xFC: "ISR", # Interrupt Status Register
    0xFD: "SCR", # System Control Register
    0xFE: "LCC", # LCD Contrast Control
    0xFF: "SSR", # System Status Control
}

class Operand:
    def render(self):
        return [TText("unimplemented")]

    def decode(self, decoder, addr):
        pass

    def encode(self, encoder, addr):
        pass

    # expand physical-encoding of operands into virtual printable operands
    def operands(self):
        yield self

    def lift(self, il):
        return il.unimplemented()

    def lift_assign(self, il, value):
        il.append(value)
        il.append(il.unimplemented())


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


def iter_encode(iter, addr):
    encoder = Encoder()
    for instr in iter:
        instr.encode(encoder, addr)
        addr += instr.length()
    return encoder.buf


def encode(instr, addr):
    return iter_encode([instr], addr)


def create_instruction(decoder, opcodes):
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


def iter_decode(data, addr, opcodes):
    decoder = Decoder(data)
    while True:
        try:
            instr = create_instruction(decoder, opcodes)
            if instr is None:
                raise NotImplementedError(
                    f"Cannot decode opcode {data[decoder.pos]:#04x} "
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
        except AssertionError as e:
            raise AssertionError(
                f"Assertion failed while decoding opcode {opcode:02X} "
                f"at address {addr:#06x}: {e}"
            ) from e


def _create_decoder(data, addr, opcodes):
    # useful for instruction fusing
    # return fusion(fusion(iter_decode(data, addr, opcodes)))
    return iter_decode(data, addr, opcodes)


def decode(data, addr, opcodes):
    try:
        instr, _ = next(_create_decoder(data, addr, opcodes))

        return instr
    except StopIteration:
        print("StopIteration: No instruction found")
        return None
    # except NotImplementedError as e:
    #     binaryninja.log_warn(e)


class Instruction:
    def __init__(self, name, operands, cond, ops_reversed):
        self.instr_name = name
        self.opcode = None
        self.ops_reversed = ops_reversed
        self._operands = operands
        self._cond = cond
        self._length = None
        self.doinit()

    def doinit(self):
        pass

    def length(self):
        return self._length

    def name(self):
        return self.instr_name

    def decode(self, decoder, addr):
        self.opcode = decoder.unsigned_byte()
        for op in self.operands_coding():
            op.decode(decoder, addr)

    def set_length(self, length):
        self._length = length


    def encode(self, encoder, addr):
        encoder.unsigned_byte(self.opcode)
        for op in self.operands_coding():
            op.encode(encoder, addr)

    def fuse(self, sister):
        return None

    # logical operands order
    def operands(self):
        if self._operands is None:
            yield from ()
        else:
            for operand in self._operands:
                for op in operand.operands():
                    yield op

    # physical opcode encoding order
    def operands_coding(self):
        if not self.ops_reversed:
            return self._operands
        # self.operands() is a generator
        # so we need to convert it to a list
        ops = list(self._operands)
        assert len(ops) == 2, "Expected 2 operands"
        return reversed(ops)

    def render(self):
        tokens = [TInstr(self.name())]
        if len(self._operands) > 0:
            tokens.append(TSep(" " * (6 - len(self.name()))))
        for index, operand in enumerate(self.operands()):
            if index > 0:
                tokens.append(TSep(", "))
            tokens += operand.render()
        return tokens

    def display(self, addr):
        print(f"{addr:04X}:\t" + "".join(str(token) for token in self.render()))

    def analyze(self, info, addr):
        info.length += self.length()

    def lift(self, il, addr):
        operands = tuple(self.operands())
        if len(operands) == 0:
            il.append(il.unimplemented())
        else:
            il_value = self.lift_operation(
                il, *(operand.lift(il) for operand in operands)
            )
            operands[0].lift_assign(il, il_value)

    def lift_operation(self, il, *il_operands):
        return il.unimplemented()


class ImmOperand(Operand):
    def width(self):
        raise NotImplementedError("width not implemented for ImmOperand")

    def lift(self, il):
        return il.const(self.width(), self.value)


# n: encoded as `n`
class Imm8(ImmOperand):
    def __init__(self):
        super().__init__()
        self.value = None

    def width(self):
        return 1

    def decode(self, decoder, addr):
        self.value = decoder.unsigned_byte()

    def encode(self, encoder, addr):
        encoder.unsigned_byte(self.value)

    def render(self):
        return [TInt(f"{self.value:02X}")]

# mn: encoded as `n m`
class Imm16(ImmOperand):
    def __init__(self):
        super().__init__()
        self.value = None

    def width(self):
        return 2

    def decode(self, decoder, addr):
        self.value = decoder.unsigned_word_le()

    def encode(self, encoder, addr):
        encoder.unsigned_word_le(self.value)

    def render(self):
        return [TInt(f"{self.value:04X}")]


# lmn: encoded as `n m l`
class Imm20(ImmOperand):
    def __init__(self):
        super().__init__()
        self.value = None
        self.extra_hi = None

    def width(self):
        return 3

    def decode(self, decoder, addr):
        lo = decoder.unsigned_byte()
        mid = decoder.unsigned_byte()
        self.extra_hi = decoder.unsigned_byte()
        hi = self.extra_hi & 0x0F
        self.value = lo | (mid << 8) | (hi << 16)

    def encode(self, encoder, addr):
        encoder.unsigned_byte(self.value & 0xFF)
        encoder.unsigned_byte((self.value >> 8) & 0xFF)
        encoder.unsigned_byte(self.extra_hi)

    def render(self):
        return [TInt(f"{self.value:05X}")]


# Offset sign is encoded as part of the instruction opcode, and the actual
# offset is Imm8.
class ImmOffset(Imm8):
    def __init__(self, sign):
        super().__init__()
        self.sign = sign

    def offset_value(self):
        return -self.value if self.sign == '-' else self.value

    def render(self):
        return [TInt(f"{self.sign}{self.value:02X}")]


# Read 8 bits from internal memory based on Imm8 address.
class IMem8(Imm8):
    def render(self):
        return [TBegMem(MemType.INTERNAL), TInt(f"{self.value:02X}"),
                TEndMem(MemType.INTERNAL)]

    # FIXME: this depends on the PRE
    def imem_addr(self):
        return INTERNAL_MEMORY_START + self.value

    def lift(self, il):
        return il.load(self.width(), il.const_pointer(3, self.imem_addr()))

    def lift_assign(self, il, value):
        il.append(il.store(self.width(), il.const_pointer(3, self.imem_addr()), value))

# Read 16 bits from internal memory based on Imm8 address.
class IMem16(IMem8):
    def render(self):
        return [TBegMem(MemType.INTERNAL), TInt(f"{self.value:02X}"),
                TEndMem(MemType.INTERNAL)]

# Read 20 bits from internal memory based on Imm8 address.
class IMem20(IMem8):
    def render(self):
        return [TBegMem(MemType.INTERNAL), TInt(f"{self.value:02X}"), TEndMem(MemType.INTERNAL)]


# Register operand encoded as part of the instruction opcode
class Reg(Operand):
    def __init__(self, reg):
        super().__init__()
        self.reg = reg

    def render(self):
        return [TReg(self.reg)]

    def width(self):
        return REG_SIZES[self.reg]

    def lift(self, il):
        return il.reg(self.width(), self.reg)

    def lift_assign(self, il, value):
        il.append(il.set_reg(self.width(), self.reg, value))

# only makes sense for PUSHU / POPU
class RegIMR(Reg):
    def __init__(self):
        super().__init__("IMR")

    def width(self):
        return 1

    def operands(self):
        imem = IMem8()
        imem.value = 0xFB
        yield imem

# only makes sense for MV
class RegB(Reg):
    def __init__(self):
        super().__init__("B")

    def width(self):
        return 1

# only makes sense for PUSHU / POPU / PUSHS / POPS
class RegF(Reg):
    def __init__(self):
        super().__init__("F")

    def width(self):
        return 1

    def lift(self, il):
        # FIXME: likely wrong
        return il.or_expr(1, il.flag("C"), il.shift_left(1, 1, il.flag("Z")))

    def lift_assign(self, il, value):
        # FIXME: likely wrong
        il.append(il.set_reg(self.width(), LLIL_TEMP(0), value))
        tmp = il.reg(self.width(), LLIL_TEMP(0))
        il.append(il.set_flag("C", il.and_expr(1, tmp, il.const(1, 1))))
        il.append(il.set_flag("Z", il.and_expr(1, tmp, il.const(1, 2))))

class Reg3(Operand):
    def __init__(self):
        super().__init__()
        self.reg = None
        self.reg_raw = None
        self.high4 = None

    @classmethod
    def reg_name(cls, idx):
        return REG_NAMES[idx]

    @classmethod
    def reg_idx(cls, name):
        return REG_NAMES.index(name)

    def width(self):
        return REG_SIZES[self.reg]

    def assert_r3(self):
        assert self.width() >= 3, f"Want r3 register, got r{self.width()} ({self.reg}) instead"

    def decode(self, decoder, addr):
        byte = decoder.unsigned_byte()
        self.reg_raw = byte
        self.reg = self.reg_name(byte & 7)
        # store high 4 bits from byte for later reference
        self.high4 = (byte >> 4) & 0x0F

    def encode(self, encoder, addr):
        byte = self.reg_raw | (self.high4 << 4)
        encoder.unsigned_byte(byte)

    def render(self):
        return [TReg(self.reg)]

# External Memory: Absolute Addressing using 20-bit address
# [lmn]: encoded as `[n m l]`
class EMemAddr(Imm20):
    def render(self):
        return [TBegMem(MemType.EXTERNAL), TInt(f"{self.value:05X}"), TEndMem(MemType.EXTERNAL)]

    def lift(self, il):
        return il.load(self.width(), il.const_pointer(3, self.value))

    def lift_assign(self, il, value):
        il.append(il.store(self.width(), il.const_pointer(3, self.value), value))

class OperandHelper: pass

class EMemIMemOffsetHelper(OperandHelper):
    # offset could be None
    def __init__(self, imem, offset):
        super().__init__()
        self.imem = imem
        self.offset = offset

    def render(self):
        result = [TBegMem(MemType.EXTERNAL),]
        result.extend(self.imem.render())
        if self.offset:
            result.extend(self.offset.render())
        result.append(TEndMem(MemType.EXTERNAL))
        return result

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

class EMemRegOffsetHelper(OperandHelper):
    # offset could be None
    def __init__(self, reg, mode, offset):
        super().__init__()
        self.reg = reg
        self.mode = mode
        self.offset = offset

    def render(self):
        result = [TBegMem(MemType.EXTERNAL)]
        # switch based on self.mode
        if self.mode == EMemRegMode.SIMPLE:
            result.extend(self.reg.render())
        elif self.mode == EMemRegMode.POST_INC:
            result.extend(self.reg.render())
            result.append(TText("++"))
        elif self.mode == EMemRegMode.PRE_DEC:
            result.append(TText("--"))
            result.extend(self.reg.render())
        elif self.mode in (
            EMemRegMode.POSITIVE_OFFSET,
            EMemRegMode.NEGATIVE_OFFSET,
        ):
            result.extend(self.reg.render())
            result.extend(self.offset.render())
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        result.append(TEndMem(MemType.EXTERNAL))
        return result


class EMemReg(Operand):
    def __init__(self, allowed_modes=None):
        super().__init__()
        self.reg = Reg3()
        self.mode = None
        self.offset = None
        self.allowed_modes = allowed_modes

    def decode(self, decoder, addr):
        super().decode(decoder, addr)
        self.reg.decode(decoder, addr)
        self.reg.assert_r3()
        self.mode = EMemRegMode(self.reg.high4)
        if self.allowed_modes is not None:
            assert self.mode in self.allowed_modes, f"Invalid mode: {self.mode}, allowed: {self.allowed_modes}"

        if self.mode in (EMemRegMode.POSITIVE_OFFSET, EMemRegMode.NEGATIVE_OFFSET):
            self.offset = ImmOffset('+' if self.mode == EMemRegMode.POSITIVE_OFFSET else '-')
            self.offset.decode(decoder, addr)

    def encode(self, encoder, addr):
        # super().encode(encoder, addr)
        self.reg.encode(encoder, addr)
        if self.offset:
            self.offset.encode(encoder, addr)

    def render(self):
        op = EMemRegOffsetHelper(self.reg, self.mode, self.offset)
        return op.render()

# page 74 of the book
# External Memory: Internal Memory indirect
# 00: [(n)]
# 80: [(m)+n]
# C0: [(m)-n]
class EMemIMemMode(enum.Enum):
    SIMPLE = 0x00
    POSITIVE_OFFSET = 0x80
    NEGATIVE_OFFSET = 0xC0

class EMemIMem(Imm8):
    def __init__(self):
        super().__init__()
        self.mode = None
        self.imem = None
        self.offset = None

    def decode(self, decoder, addr):
        super().decode(decoder, addr)
        self.imem = IMem8()
        self.imem.decode(decoder, addr)

        self.mode = EMemIMemMode(self.value)
        if self.mode in (EMemIMemMode.POSITIVE_OFFSET, EMemIMemMode.NEGATIVE_OFFSET):
            self.offset = ImmOffset('+' if self.mode == EMemIMemMode.POSITIVE_OFFSET else '-')
            self.offset.decode(decoder, addr)

    def encode(self, encoder, addr):
        super().encode(encoder, addr)
        self.imem.encode(encoder, addr)

        if self.offset:
            self.offset.encode(encoder, addr)

    def render(self):
        op = EMemIMemOffsetHelper(self.imem, self.offset)
        return op.render()


# ADD/SUB can use various-sized register pairs
class RegPair(Reg3):
    def __init__(self, size=None):
        super().__init__()
        self.reg_raw = None
        self.reg1 = None
        self.reg2 = None
        self.size = size

    def decode(self, decoder, addr):
        self.reg_raw = decoder.unsigned_byte()
        self.reg1 = Reg(REG_NAMES[(self.reg_raw >> 4) & 7])
        self.reg2 = Reg(REG_NAMES[self.reg_raw & 7])

        # high-bits of both halves must be zero: 0x80 and 0x08 must not be set
        assert (self.reg_raw & 0x80) == 0, f"Invalid reg1 high bit: {self.reg1}"
        assert (self.reg_raw & 0x08) == 0, f"Invalid reg2 high bit: {self.reg2}"

    def operands(self):
        yield self.reg1
        yield self.reg2

    def encode(self, encoder, addr):
        encoder.unsigned_byte(self.reg_raw)

    def render(self):
        result = self.reg1.render()
        result.append(TSep(", "))
        result.extend(self.reg2.render())
        return result

class NOP(Instruction):
     def lift(self, il, addr):
        il.append(il.nop())

class JumpInstruction(Instruction):
    def lift_jump_addr(self, il, addr):
        raise NotImplementedError("lift_jump_addr() not implemented")

    def analyze(self, info, addr):
        super().analyze(info, addr)
        # expect TrueBranch to be handled by subclasses as it might require
        # llil logic to calculate the address
        info.add_branch(BranchType.FalseBranch, addr + self.length())


    def lift(self, il, addr):
        if_true  = LowLevelILLabel()
        if_false = LowLevelILLabel()

        if self._cond:
            zero = il.const(1, 0)
            one  = il.const(1, 1)
            flag = il.flag("Z") if "Z" in self._cond else il.flag("C")
            value = zero if "N" in self._cond else one

            cond = il.compare_equal(1, flag, value)
            il.append(il.if_expr(cond, if_true, if_false))

        il.mark_label(if_true)
        il.append(il.jump(self.lift_jump_addr(il, addr)))
        il.mark_label(if_false)


class JP_Abs(JumpInstruction):
    def name(self):
        return super().name() + (self._cond if self._cond else "")

    def lift_jump_addr(self, il,  addr):
        first, *rest = self.operands()
        assert len(rest) == 0, "Expected no extra operands"
        return first.lift(il)

    def analyze(self, info, addr):
        super().analyze(info, addr)

        first, *rest = self.operands()
        assert len(rest) == 0, "Expected no extra operands"
        if isinstance(first, ImmOperand):
            # absolute address
            dest = first.value
            info.add_branch(BranchType.TrueBranch, dest)

class JP_Rel(JumpInstruction):
    def name(self):
        return "JR" + (self._cond if self._cond else "")

    def lift_jump_addr(self, il, addr):
        first, *rest = self.operands()
        assert len(rest) == 0, "Expected no extra operands"
        return il.const_pointer(3, addr + self.length() + first.value)

    def analyze(self, info, addr):
        super().analyze(info, addr)
        first, *rest = self.operands()
        assert len(rest) == 0, "Expected no extra operands"
        dest = addr + self.length() + first.offset_value()
        info.add_branch(BranchType.TrueBranch, dest)

class CALL(Instruction):
    def dest_addr(self, addr):
        dest, *rest = self.operands()
        assert len(rest) == 0, "Expected no extra operands"

        result = dest.value
        if dest.width() != 3:
            assert dest.width() == 2
            result = addr & 0xFF0000 | result
        return result

    def analyze(self, info, addr):
        super().analyze(info, addr)
        info.add_branch(BranchType.CallDestination, self.dest_addr(addr))

    def lift(self, il, addr):
        il.append(il.call(il.const_pointer(3, self.dest_addr(addr))))


class RetInstruction(Instruction):
    def addr_size(self):
        return 2

    def analyze(self, info, addr):
        super().analyze(info, addr)
        info.add_branch(BranchType.FunctionReturn)

    def lift(self, il, addr):
        # FIXME: should add bitmask for 2-byte pop?
        il.append(il.ret(il.pop(self.addr_size())))

class RET(RetInstruction): pass
class RETF(RetInstruction):
    def addr_size(self):
        return 3
class RETI(Instruction): pass

class MoveInstruction(Instruction):
    def dst(self):
        raise NotImplementedError("dst() not implemented")
    def src(self):
        raise NotImplementedError("src() not implemented")

class MV(MoveInstruction):
    def dst(self):
        first, *rest = self.operands()
        assert len(rest) == 1, f"Expected no extra operands, got: {rest}"
        return first
    def src(self):
        _, second, *rest = self.operands()
        assert len(rest) == 0, f"Expected no extra operands, got: {rest}"
        return second
    def lift(self, il, addr):
        self.dst().lift_assign(il, self.src().lift(il))

class MVP(MoveInstruction): pass # 20-bit move
class MVLD(MoveInstruction): pass # Block move decrementing
class MVL(MoveInstruction): pass

# page 77 of the book
# (m), [r3±n]: encoded as `56 (8 r3 | C r3) m n
class MVL_56(MoveInstruction):
    def doinit(self):
        super().doinit()
        self.reg = None
        self.dst = None
        self.mode = None
        self.offset = None

    def decode(self, decoder, addr):
        super().decode(decoder, addr)
        self.reg = Reg3()
        self.reg.decode(decoder, addr)
        self.reg.assert_r3()

        self.dst = IMem8()
        self.dst.decode(decoder, addr)

        self.mode = EMemRegMode(self.reg.high4)
        assert self.mode in (EMemRegMode.POSITIVE_OFFSET, EMemRegMode.NEGATIVE_OFFSET)
        self.offset = ImmOffset('+' if self.mode == EMemRegMode.POSITIVE_OFFSET else '-')
        self.offset.decode(decoder, addr)

    def encode(self, encoder, addr):
        super().encode(encoder, addr)
        self.reg.encode(encoder, addr)
        self.dst.encode(encoder, addr)
        self.offset.encode(encoder, addr)

    def render(self):
        result = super().render()
        result.append(TSep(" "))

        result.extend(self.dst.render())
        result.append(TSep(", "))

        op = EMemRegOffsetHelper(self.reg, self.mode, self.offset)
        result.extend(op.render())
        return result

# page 77 of the book
# [r3±m], (n): encoded as 5E (8 r3 | C r3) n m
class MVL_5E(MoveInstruction):
    def doinit(self):
        super().doinit()
        self.reg = None
        self.mode = None
        self.offset = None
        self.src = None

    def decode(self, decoder, addr):
        super().decode(decoder, addr)
        self.reg = Reg3()
        self.reg.decode(decoder, addr)
        self.reg.assert_r3()

        self.src = IMem8()
        self.src.decode(decoder, addr)

        self.mode = EMemRegMode(self.reg.high4)
        assert self.mode in (EMemRegMode.POSITIVE_OFFSET, EMemRegMode.NEGATIVE_OFFSET)
        self.offset = ImmOffset('+' if self.mode == EMemRegMode.POSITIVE_OFFSET else '-')
        self.offset.decode(decoder, addr)

    def encode(self, encoder, addr):
        super().encode(encoder, addr)
        self.reg.encode(encoder, addr)
        self.src.encode(encoder, addr)
        self.offset.encode(encoder, addr)

    def render(self):
        result = super().render()
        result.append(TSep(" "))

        op = EMemRegOffsetHelper(self.reg, self.mode, self.offset)
        result.extend(op.render())

        result.append(TSep(", "))
        result.extend(self.src.render())
        return result

# page 77 of the book
# (m), [(n)]:   encoded as F3 00 m n
# (l), [(m)+n]: encoded as F3 80 l m n
# (l), [(m)-n]: encoded as F3 C0 l m n
class MVL_F3(MoveInstruction):
    def doinit(self):
        super().doinit()
        self.mode_imm = None
        self.dst = None
        self.mode = None
        self.src = None
        self.offset = None

    def decode(self, decoder, addr):
        super().decode(decoder, addr)
        self.mode_imm = Imm8()
        self.mode_imm.decode(decoder, addr)

        self.dst = IMem8()
        self.dst.decode(decoder, addr)

        self.src = IMem8()
        self.src.decode(decoder, addr)

        self.mode = EMemIMemMode(self.mode_imm.value)
        if self.mode in (EMemIMemMode.POSITIVE_OFFSET, EMemIMemMode.NEGATIVE_OFFSET):
            self.offset = ImmOffset('+' if self.mode == EMemIMemMode.POSITIVE_OFFSET else '-')
            self.offset.decode(decoder, addr)

    def encode(self, encoder, addr):
        super().encode(encoder, addr)
        self.mode_imm.encode(encoder, addr)
        self.dst.encode(encoder, addr)
        self.src.encode(encoder, addr)
        if self.offset:
            self.offset.encode(encoder, addr)

    def render(self):
        result = super().render()
        result.append(TSep(" "))
        result.extend(self.dst.render())
        result.append(TSep(", "))

        op = EMemIMemOffsetHelper(self.src, self.offset)
        result.extend(op.render())
        return result

# page 77 of the book
# [(m)], (n):   encoded as FB 00 m n
# [(l)+m], (n): encoded as FB 80 l n m
# [(l)-m], (n): encoded as FB C0 l n m
class MVL_FB(MoveInstruction):
    def doinit(self):
        super().doinit()
        self.mode_imm = None
        self.dst = None
        self.mode = None
        self.offset = None
        self.src = None

    def decode(self, decoder, addr):
        super().decode(decoder, addr)
        self.mode_imm = Imm8()
        self.mode_imm.decode(decoder, addr)

        self.dst = IMem8()
        self.dst.decode(decoder, addr)

        self.src = IMem8()
        self.src.decode(decoder, addr)

        self.mode = EMemIMemMode(self.mode_imm.value)
        if self.mode in (EMemIMemMode.POSITIVE_OFFSET, EMemIMemMode.NEGATIVE_OFFSET):
            self.offset = ImmOffset('+' if self.mode == EMemIMemMode.POSITIVE_OFFSET else '-')
            self.offset.decode(decoder, addr)

    def encode(self, encoder, addr):
        super().encode(encoder, addr)
        self.mode_imm.encode(encoder, addr)
        self.dst.encode(encoder, addr)
        self.src.encode(encoder, addr)
        if self.offset:
            self.offset.encode(encoder, addr)

    def render(self):
        result = super().render()
        result.append(TSep(" "))

        op = EMemIMemOffsetHelper(self.dst, self.offset)
        result.extend(op.render())

        result.append(TSep(", "))
        result.extend(self.src.render())
        return result


# page 75 of the book
# (n), [r3], : encoded as E0 (0 r3) n
# (n), [r3++]: encoded as E0 (2 r3) n
# (n), [--r3]: encoded as E0 (3 r3) n
# (n), [r3±m]: encoded as E0 (8 r3 | C r3) n m
class MV_E0(MoveInstruction):
    def doinit(self):
        super().doinit()
        self.reg = None
        self.dst = None
        self.mode = None
        self.offset = None

    def decode(self, decoder, addr):
        super().decode(decoder, addr)
        self.reg = Reg3()
        self.reg.decode(decoder, addr)
        self.reg.assert_r3()

        self.dst = IMem8() # FIXME: size is wrong depending on the opcode!
        self.dst.decode(decoder, addr)

        self.mode = EMemRegMode(self.reg.high4)
        if self.mode in (EMemRegMode.POSITIVE_OFFSET,
                         EMemRegMode.NEGATIVE_OFFSET):
            self.offset = ImmOffset('+' if self.mode == EMemRegMode.POSITIVE_OFFSET else '-')
            self.offset.decode(decoder, addr)

    def encode(self, encoder, addr):
        super().encode(encoder, addr)
        self.reg.encode(encoder, addr)
        self.dst.encode(encoder, addr)
        if self.offset:
            self.offset.encode(encoder, addr)

    def render(self):
        result = super().render()
        result.append(TSep(" "))
        result.extend(self.dst.render())
        result.append(TSep(", "))

        op = EMemRegOffsetHelper(self.reg, self.mode, self.offset)
        result.extend(op.render())
        return result

# page 75 of the book
# [r3],   (n): encoded as E8 (0 r3) n
# [r3++], (n): encoded as E8 (2 r3) n
# [--r3], (n): encoded as E8 (3 r3) n
# [r3±m], (n): encoded as E8 (8 r3 | C r3) n m
class MV_E8(MoveInstruction):
    def doinit(self):
        super().doinit()
        self.reg = None
        self.mode = None
        self.offset = None
        self.src = None

    def decode(self, decoder, addr):
        super().decode(decoder, addr)
        self.reg = Reg3()
        self.reg.decode(decoder, addr)
        self.reg.assert_r3()

        self.src = IMem8() # FIXME: size is wrong depending on the opcode!
        self.src.decode(decoder, addr)

        self.mode = EMemRegMode(self.reg.high4)
        if self.mode in (EMemRegMode.POSITIVE_OFFSET,
                         EMemRegMode.NEGATIVE_OFFSET):
            self.offset = ImmOffset('+' if self.mode == EMemRegMode.POSITIVE_OFFSET else '-')
            self.offset.decode(decoder, addr)

    def encode(self, encoder, addr):
        super().encode(encoder, addr)
        self.reg.encode(encoder, addr)
        self.src.encode(encoder, addr)
        if self.offset:
            self.offset.encode(encoder, addr)

    def render(self):
        result = super().render()
        result.append(TSep(" "))

        op = EMemRegOffsetHelper(self.reg, self.mode, self.offset)
        result.extend(op.render())
        result.append(TSep(", "))

        result.extend(self.src.render())
        return result


# page 75 of the book
# (m), [(n)]:   encoded as F0 00 m n
# (l), [(m)+n]: encoded as F0 80 l m n
# (l), [(m)-n]: encoded as F0 C0 l m n
class MV_F0(MoveInstruction):
    def doinit(self):
        super().doinit()
        self.mode_imm = None
        self.dst = None
        self.mode = None
        self.src = None
        self.offset = None

    def decode(self, decoder, addr):
        super().decode(decoder, addr)
        self.mode_imm = Imm8()
        self.mode_imm.decode(decoder, addr)

        self.dst = IMem8() # FIXME: size is wrong depending on the opcode!
        self.dst.decode(decoder, addr)

        self.src = IMem8() # FIXME: size is wrong depending on the opcode!
        self.src.decode(decoder, addr)

        self.mode = EMemIMemMode(self.mode_imm.value)
        if self.mode in (EMemIMemMode.POSITIVE_OFFSET,
                         EMemIMemMode.NEGATIVE_OFFSET):
            self.offset = ImmOffset('+' if self.mode == EMemIMemMode.POSITIVE_OFFSET else '-')
            self.offset.decode(decoder, addr)

    def encode(self, encoder, addr):
        super().encode(encoder, addr)
        self.mode_imm.encode(encoder, addr)
        self.dst.encode(encoder, addr)
        self.src.encode(encoder, addr)
        if self.offset:
            self.offset.encode(encoder, addr)

    def render(self):
        result = super().render()
        result.append(TSep(" "))
        result.extend(self.dst.render())
        result.append(TSep(", "))

        op = EMemIMemOffsetHelper(self.src, self.offset)
        result.extend(op.render())
        return result

# page 75 of the book
# [(m)], (n):   encoded as F8 00 m n
# [(l)+m], (n): encoded as F8 80 l m n
# [(l)-m], (n): encoded as F8 C0 l m n
class MV_F8(MoveInstruction):
    def doinit(self):
        super().doinit()
        self.mode_imm = None
        self.dst = None
        self.mode = None
        self.offset = None
        self.src = None

    def decode(self, decoder, addr):
        super().decode(decoder, addr)
        self.mode_imm = Imm8()
        self.mode_imm.decode(decoder, addr)

        self.dst = IMem8() # FIXME: size is wrong depending on the opcode!
        self.dst.decode(decoder, addr)

        self.src = IMem8() # FIXME: size is wrong depending on the opcode!
        self.src.decode(decoder, addr)

        self.mode = EMemIMemMode(self.mode_imm.value)
        if self.mode in (EMemIMemMode.POSITIVE_OFFSET,
                         EMemIMemMode.NEGATIVE_OFFSET):
            self.offset = ImmOffset('+' if self.mode == EMemIMemMode.POSITIVE_OFFSET else '-')
            self.offset.decode(decoder, addr)

    def encode(self, encoder, addr):
        super().encode(encoder, addr)
        self.mode_imm.encode(encoder, addr)
        self.dst.encode(encoder, addr)
        self.src.encode(encoder, addr)
        if self.offset:
            self.offset.encode(encoder, addr)

    def render(self):
        result = super().render()
        result.append(TSep(" "))

        op = EMemIMemOffsetHelper(self.dst, self.offset)
        result.extend(op.render())
        result.append(TSep(", "))

        result.extend(self.src.render())
        return result


class PRE(Instruction):
    def name(self):
        return f"PRE{self.opcode:02x}"
    def lift(self, il, addr):
        # FIXME: ignore PRE for now
        pass

class StackInstruction(Instruction):
    def reg(self):
        r, *rest = self.operands()
        assert len(rest) == 0, "Expected no extra operands"
        return r
class StackPushInstruction(StackInstruction):
    def lift(self, il, addr):
        r = self.reg()
        il.append(il.push(r.width(), r.lift(il)))
class StackPopInstruction(StackInstruction):
    def lift(self, il, addr):
        r = self.reg()
        r.lift_assign(il, il.pop(r.width()))

# FIXME: should use U pointer, not S
class PUSHU(StackPushInstruction): pass
class POPU(StackPopInstruction): pass

class PUSHS(StackPushInstruction): pass
class POPS(StackPopInstruction): pass

class ArithmeticInstruction(Instruction):
    def width(self):
        first, second = self.operands()
        return first.width()
class ADD(ArithmeticInstruction):
    def lift_operation(self, il, il_arg1, il_arg2):
        return il.add(self.width(), il_arg1, il_arg2, 'CZ')
class ADC(ArithmeticInstruction):
    def lift_operation(self, il, il_arg1, il_arg2):
        return il.add(self.width(), il_arg1, il.add(self.width(), il_arg2, il.flag('C')), 'CZ')
class ADCL(ArithmeticInstruction): pass
class SUB(ArithmeticInstruction):
    def lift_operation(self, il, il_arg1, il_arg2):
        return il.sub(self.width(), il_arg1, il_arg2, 'CZ')
class SBC(ArithmeticInstruction):
    def lift_operation(self, il, il_arg1, il_arg2):
        return il.sub(self.width(), il_arg1, il.add(self.width(), il_arg2, il.flag('C')), 'CZ')
class SBCL(ArithmeticInstruction): pass
class DADL(ArithmeticInstruction): pass
class DSBL(ArithmeticInstruction): pass

class LogicInstruction(Instruction): pass
class AND(LogicInstruction):
    def lift_operation(self, il, il_arg1, il_arg2):
        return il.and_expr(1, il_arg1, il_arg2, "Z")
class OR(LogicInstruction):
    def lift_operation(self, il, il_arg1, il_arg2):
        return il.or_expr(1, il_arg1, il_arg2, "Z")
class XOR(LogicInstruction):
    def lift_operation(self, il, il_arg1, il_arg2):
        return il.xor_expr(1, il_arg1, il_arg2, "Z")

class CompareInstruction(Instruction): pass
class TEST(CompareInstruction):
    def lift(self, il, addr):
        first, second = self.operands()
        il.append(il.set_flag("Z", il.and_expr(3, first.lift(il), second.lift(il))))

class CMP(CompareInstruction): pass
class CMPW(CompareInstruction): pass
class CMPP(CompareInstruction): pass

class ShiftRotateInstruction(Instruction): pass
class ROR(ShiftRotateInstruction): pass
class ROL(ShiftRotateInstruction): pass
class SHR(ShiftRotateInstruction): pass
class SHL(ShiftRotateInstruction): pass
class DSRL(ShiftRotateInstruction): pass
class DSLL(ShiftRotateInstruction): pass

class IncDecInstruction(Instruction): pass
class INC(IncDecInstruction): pass
class DEC(IncDecInstruction): pass

class ExchangeInstruction(Instruction): pass
class EX(ExchangeInstruction): pass
class EXW(ExchangeInstruction): pass
class EXP(ExchangeInstruction): pass
class EXL(ExchangeInstruction): pass

class MiscInstruction(Instruction): pass
class WAIT(MiscInstruction):
    def lift(self, il, addr):
        t = LowLevelILLabel()
        f = LowLevelILLabel()

        reg = Reg("I")
        il.mark_label(f)
        reg.lift_assign(il, il.sub(2, reg.lift(il), il.const(1, 1)))
        cond = il.compare_equal(2, reg.lift(il), il.const(2, 0))
        il.append(il.if_expr(cond, t, f))
        il.mark_label(t)

class PMDF(MiscInstruction): pass
class SWAP(MiscInstruction): pass
class HALT(MiscInstruction): pass
class OFF(MiscInstruction): pass
class IR(MiscInstruction): pass
class RESET(MiscInstruction): pass
class SC(MiscInstruction): pass
class RC(MiscInstruction): pass
class TCL(MiscInstruction): pass

class UnknownInstruction(Instruction):
    def name(self):
        return f"??? ({self.opcode:02X})"


OPCODES = {
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
    0x56: MVL_56,
    0x57: (PMDF, Opts(ops=[IMem8(), Reg("A")])),
    0x58: (SBC, Opts(ops=[Reg("A"), Imm8()])),
    0x59: (SBC, Opts(ops=[IMem8(), Imm8()])),
    0x5A: (SBC, Opts(ops=[Reg("A"), IMem8()])),
    0x5B: (SBC, Opts(ops=[IMem8(), Reg("A")])),
    0x5C: (SBCL, Opts(ops=[IMem8(), IMem8()])),
    0x5D: (SBCL, Opts(ops=[IMem8(), Reg("A")])),
    0x5E: MVL_5E,
    0x5F: (POPS, Opts(ops=[RegF()])),
    # 60h
    0x60: (CMP, Opts(ops=[Reg("A"), Imm8()])),
    0x61: (CMP, Opts(ops=[IMem8(), Imm8()])),
    0x62: (CMP, Opts(ops=[EMemAddr(), Imm8()])),
    0x63: (CMP, Opts(ops=[IMem8(), Reg("A")])),
    0x64: (TEST, Opts(ops=[Reg("A"), Imm8()])),
    0x65: (TEST, Opts(ops=[IMem8(), Imm8()])),
    0x66: (TEST, Opts(ops=[EMemAddr(), Imm8()])),
    0x67: (TEST, Opts(ops=[IMem8(), Reg("A")])),
    0x68: (XOR, Opts(ops=[Reg("A"), Imm8()])),
    0x69: (XOR, Opts(ops=[IMem8(), Imm8()])),
    0x6A: (XOR, Opts(ops=[EMemAddr(), Imm8()])),
    0x6B: (XOR, Opts(ops=[IMem8(), Reg("A")])),
    0x6C: (INC, Opts(ops=[Reg3()])),
    0x6D: (INC, Opts(ops=[IMem8()])),
    0x6E: (XOR, Opts(ops=[IMem8(), IMem8()])),
    0x6F: (XOR, Opts(ops=[Reg("A"), IMem8()])),
    # 70h
    0x70: (AND, Opts(ops=[Reg("A"), Imm8()])),
    0x71: (AND, Opts(ops=[IMem8(), Imm8()])),
    0x72: (AND, Opts(ops=[EMemAddr(), Imm8()])),
    0x73: (AND, Opts(ops=[IMem8(), Reg("A")])),
    0x74: (MV, Opts(ops=[Reg("A"), RegB()])),
    0x75: (MV, Opts(ops=[RegB(), Reg("A")])),
    0x76: (AND, Opts(ops=[IMem8(), IMem8()])),
    0x77: (AND, Opts(ops=[Reg("A"), IMem8()])),
    0x78: (OR, Opts(ops=[Reg("A"), Imm8()])),
    0x79: (OR, Opts(ops=[IMem8(), Imm8()])),
    0x7A: (OR, Opts(ops=[EMemAddr(), Imm8()])),
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
    0x88: (MV, Opts(ops=[Reg("A"), EMemAddr()])),
    0x89: (MV, Opts(ops=[Reg("IL"), EMemAddr()])),
    0x8A: (MV, Opts(ops=[Reg("BA"), EMemAddr()])),
    0x8B: (MV, Opts(ops=[Reg("I"), EMemAddr()])),
    0x8C: (MV, Opts(ops=[Reg("X"), EMemAddr()])),
    0x8D: (MV, Opts(ops=[Reg("Y"), EMemAddr()])),
    0x8E: (MV, Opts(ops=[Reg("U"), EMemAddr()])),
    0x8F: (MV, Opts(ops=[Reg("S"), EMemAddr()])),
    # 90h
    0x90: (MV, Opts(ops=[Reg("A"), EMemReg()])),
    0x91: (MV, Opts(ops=[Reg("IL"), EMemReg()])),
    0x92: (MV, Opts(ops=[Reg("BA"), EMemReg()])),
    0x93: (MV, Opts(ops=[Reg("I"), EMemReg()])),
    0x94: (MV, Opts(ops=[Reg("X"), EMemReg()])),
    0x95: (MV, Opts(ops=[Reg("Y"), EMemReg()])),
    0x96: (MV, Opts(ops=[Reg("U"), EMemReg()])),
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
    0xA8: (MV, Opts(ops=[EMemAddr(), Reg("A")])),
    0xA9: (MV, Opts(ops=[EMemAddr(), Reg("IL")])),
    0xAA: (MV, Opts(ops=[EMemAddr(), Reg("BA")])),
    0xAB: (MV, Opts(ops=[EMemAddr(), Reg("I")])),
    0xAC: (MV, Opts(ops=[EMemAddr(), Reg("X")])),
    0xAD: (MV, Opts(ops=[EMemAddr(), Reg("Y")])),
    0xAE: (MV, Opts(ops=[EMemAddr(), Reg("U")])),
    0xAF: (MV, Opts(ops=[EMemAddr(), Reg("S")])),
    # B0h
    0xB0: (MV, Opts(ops=[EMemReg(), Reg("A")])),
    0xB1: (MV, Opts(ops=[EMemReg(), Reg("IL")])),
    0xB2: (MV, Opts(ops=[EMemReg(), Reg("BA")])),
    0xB3: (MV, Opts(ops=[EMemReg(), Reg("I")])),
    0xB4: (MV, Opts(ops=[EMemReg(), Reg("X")])),
    0xB5: (MV, Opts(ops=[EMemReg(), Reg("Y")])),
    0xB6: (MV, Opts(ops=[EMemReg(), Reg("U")])),
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
    0xC1: (EXW, Opts(ops=[IMem16(), IMem16()])),
    0xC2: (EXP, Opts(ops=[IMem20(), IMem20()])),
    0xC3: (EXL, Opts(ops=[IMem8(), IMem8()])),
    0xC4: (DADL, Opts(ops=[IMem8(), IMem8()])),
    0xC5: (DADL, Opts(ops=[IMem8(), Reg("A")])),
    0xC6: (CMPW, Opts(ops=[IMem16(), IMem16()])),
    0xC7: (CMPP, Opts(ops=[IMem20(), IMem20()])),
    0xC8: (MV, Opts(ops=[IMem8(), IMem8()])),
    0xC9: (MV, Opts(name="MVW", ops=[IMem16(), IMem16()])),
    0xCA: (MVP, Opts(ops=[IMem20(), IMem20()])),
    0xCB: (MVL, Opts(ops=[IMem8(), IMem8()])),
    0xCC: (MV, Opts(ops=[IMem8(), Imm8()])),
    0xCD: (MV, Opts(name="MVW", ops=[IMem16(), Imm16()])),
    0xCE: TCL,
    0xCF: (MVLD, Opts(ops=[IMem8(), IMem8()])),
    # D0h
    0xD0: (MV, Opts(ops=[IMem8(), EMemAddr()])),
    0xD1: (MV, Opts(name="MVW", ops=[IMem16(), EMemAddr()])),
    0xD2: (MVP, Opts(ops=[IMem20(), EMemAddr()])),
    0xD3: (MVL, Opts(ops=[IMem8(), EMemAddr()])),
    0xD4: (DSBL, Opts(ops=[IMem8(), IMem8()])),
    0xD5: (DSBL, Opts(ops=[IMem8(), Reg("A")])),
    0xD6: (CMPW, Opts(ops_reversed=True, ops=[IMem16(), Reg3()])),
    0xD7: (CMPP, Opts(ops_reversed=True, ops=[IMem20(), Reg3()])),
    0xD8: (MV, Opts(ops=[EMemAddr(), IMem8()])),
    0xD9: (MV, Opts(name="MVW", ops=[EMemAddr(), IMem16()])),
    0xDA: (MVP, Opts(ops=[EMemAddr(), IMem20()])),
    0xDB: (MVL, Opts(ops=[EMemAddr(), IMem8()])),
    0xDC: (MVP, Opts(ops=[IMem20(), Imm20()])),
    0xDD: (EX, Opts(ops=[Reg("A"), RegB()])),
    0xDE: HALT,
    0xDF: OFF,
    # E0h
    0xE0: (MV_E0, Opts(name="MV")),
    0xE1: (MV_E0, Opts(name="MVW")),
    0xE2: (MV_E0, Opts(name="MVP")),
    0xE3: (MVL, Opts(ops_reversed=True, ops=[IMem8(),
                                             EMemReg(allowed_modes=[EMemRegMode.POST_INC,
                                                                    EMemRegMode.PRE_DEC])])),
    0xE4: (ROR, Opts(ops=[Reg("A")])),
    0xE5: (ROR, Opts(ops=[IMem8()])),
    0xE6: (ROL, Opts(ops=[Reg("A")])),
    0xE7: (ROL, Opts(ops=[IMem8()])),
    0xE8: (MV_E8, Opts(name="MV")),
    0xE9: (MV_E8, Opts(name="MVW")),
    0xEA: (MV_E8, Opts(name="MVP")),
    0xEB: (MVL, Opts(ops=[EMemReg(), IMem8()])),
    0xEC: (DSLL, Opts(ops=[IMem8()])),
    0xED: (EX, Opts(ops=[RegPair(size=2)])),
    0xEE: (SWAP, Opts(ops=[Reg("A")])),
    0xEF: WAIT,
    # F0h
    0xF0: (MV_F0, Opts(name='MV')),
    0xF1: (MV_F0, Opts(name='MVW')),
    0xF2: (MV_F0, Opts(name='MVP')),
    0xF3: MVL_F3,
    0xF4: (SHR, Opts(ops=[Reg("A")])),
    0xF5: (SHR, Opts(ops=[IMem8()])),
    0xF6: (SHL, Opts(ops=[Reg("A")])),
    0xF7: (SHL, Opts(ops=[IMem8()])),
    0xF8: (MV_F8, Opts(name='MV')),
    0xF9: (MV_F8, Opts(name='MVW')),
    0xFA: (MV_F8, Opts(name='MVP')),
    0xFB: MVL_FB,
    0xFC: (DSRL, Opts(ops=[IMem8()])),
    0xFD: (MV, Opts(ops=[RegPair(size=2)])),
    0xFE: IR,
    0xFF: RESET,
}
