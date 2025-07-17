from typing import Any, List, Optional, Union, cast, TypedDict, Type, Literal
from binja_helpers import binja_api  # noqa: F401  # pyright: ignore
from binaryninja import RegisterName  # type: ignore
from lark import Lark, Transformer, Token
from .instr import (
    Instruction,
    Opts,
    NOP,
    RETI,
    RET,
    RETF,
    SC,
    RC,
    TCL,
    HALT,
    OFF,
    WAIT,
    IR,
    RESET,
    SWAP,
    ROR,
    ROL,
    SHR,
    SHL,
    MV,
    MVL,
    MVLD,
    EX,
    EXL,
    PUSHS,
    POPS,
    PUSHU,
    POPU,
    AND,
    OR,
    XOR,
    ADD,
    ADC,
    SUB,
    SBC,
    ADCL,
    SBCL,
    DADL,
    DSBL,
    DSLL,
    DSRL,
    PMDF,
    TEST,
    CMP,
    CMPW,
    CMPP,
    CALL,
    JP_Abs,
    JP_Rel,
    Imm16,
    Imm20,
    ImmOffset,
    IMem20,
    IMem16,
    INC,
    DEC,
    Reg3,
    Reg,
    RegB,
    RegF,
    RegIMR,
    RegPair,
    Operand,
    IMemOperand,
    IMem8,
    ImmOperand,
    Imm8,
    EMemAddr,
    EMemReg,
    EMemRegMode,
    EMemIMem,
    EMemIMemMode,
    RegIMemOffset,
    RegIMemOffsetOrder,
    EMemIMemOffset,
    EMemIMemOffsetOrder,
    AddressingMode,
)

import os

grammar_path = os.path.join(os.path.dirname(__file__), "asm.lark")
with open(grammar_path, "r") as f:
    asm_grammar = f.read()

asm_parser = Lark(asm_grammar, parser="earley", maybe_placeholders=False)


class LabelNode(TypedDict):
    label: str


class SectionNode(TypedDict):
    section: str


class DataDirectiveNode(TypedDict):
    type: str  # "defb", "defw", "defl", "defs", "defm"
    args: Union[List[str], int, str]


class ParsedInstruction(TypedDict):
    instr_class: Type[Instruction]
    instr_opts: Opts


class InstructionNode(TypedDict):
    instruction: ParsedInstruction


class LineNode(TypedDict, total=False):
    label: Optional[str]
    statement: Optional[Union[SectionNode, DataDirectiveNode, InstructionNode]]


class ProgramNode(TypedDict):
    lines: List[LineNode]


class AsmTransformer(Transformer):
    def _make_instr(
        self,
        instr_class: Type[Instruction],
        *ops: Any,
        name: Optional[str] = None,
        cond: Optional[str] = None,
    ) -> InstructionNode:
        return {
            "instruction": {
                "instr_class": instr_class,
                "instr_opts": Opts(name=name, cond=cond, ops=list(ops) if ops else None),
            }
        }
    def start(self, items: List[LineNode]) -> ProgramNode:
        # Filter out empty lines and stray NEWLINE tokens
        return {"lines": [line for line in items if isinstance(line, dict) and line]}

    def line(self, items: List[Any]) -> LineNode:
        # Filter out any stray NEWLINE tokens that might be children of the line rule
        items = [
            item
            for item in items
            if not (isinstance(item, Token) and item.type == "NEWLINE")
        ]

        # line: label? statement?
        out: LineNode = {}
        if not items:
            return out

        if len(items) == 2:
            out["label"] = cast(str, items[0])
            out["statement"] = cast(
                Union[SectionNode, DataDirectiveNode, InstructionNode, None], items[1]
            )
        elif len(items) == 1:
            item = items[0]
            if isinstance(item, str):
                out["label"] = item
            elif isinstance(item, dict):
                out["statement"] = cast(
                    Union[SectionNode, DataDirectiveNode, InstructionNode, None], item
                )
        return out

    def label(self, items: List[Token]) -> str:
        return str(items[0])

    def section_decl(self, items: List[Any]) -> SectionNode:
        return {"section": str(items[-1])}

    def data_directive(self, items: List[Any]) -> DataDirectiveNode:
        return items[0]  # type: ignore

    def org_directive(self, items: List[Any]) -> DataDirectiveNode:
        return {"type": "org", "args": str(items[0])}

    def defb_directive(self, items: List[Any]) -> DataDirectiveNode:
        return {"type": "defb", "args": items}

    def defw_directive(self, items: List[Any]) -> DataDirectiveNode:
        return {"type": "defw", "args": items}

    def defl_directive(self, items: List[Any]) -> DataDirectiveNode:
        return {"type": "defl", "args": items}

    def defs_directive(self, items: List[Any]) -> DataDirectiveNode:
        return {"type": "defs", "args": int(items[0])}

    def defm_directive(self, items: List[Any]) -> DataDirectiveNode:
        return {"type": "defm", "args": items[0]}

    def nop(self, _: List[Any]) -> InstructionNode:
        return self._instr_node(NOP)

    def reti(self, _: List[Any]) -> InstructionNode:
        return self._instr_node(RETI)

    def ret(self, _: List[Any]) -> InstructionNode:
        return self._instr_node(RET)

    def retf(self, _: List[Any]) -> InstructionNode:
        return self._instr_node(RETF)

    def sc(self, _: List[Any]) -> InstructionNode:
        return self._instr_node(SC)

    def rc(self, _: List[Any]) -> InstructionNode:
        return self._instr_node(RC)

    def tcl(self, _: List[Any]) -> InstructionNode:
        return self._instr_node(TCL)

    def halt(self, _: List[Any]) -> InstructionNode:
        return self._instr_node(HALT)

    def off(self, _: List[Any]) -> InstructionNode:
        return self._instr_node(OFF)

    def wait(self, _: List[Any]) -> InstructionNode:
        return self._instr_node(WAIT)

    def ir(self, _: List[Any]) -> InstructionNode:
        return self._instr_node(IR)

    def reset(self, _: List[Any]) -> InstructionNode:
        return self._instr_node(RESET)

    def swap_a(self, _: List[Any]) -> InstructionNode:
        return self._instr_node(SWAP, Reg("A"))

    def ror_a(self, _: List[Any]) -> InstructionNode:
        return self._instr_node(ROR, Reg("A"))

    def rol_a(self, _: List[Any]) -> InstructionNode:
        return self._instr_node(ROL, Reg("A"))

    def shr_a(self, _: List[Any]) -> InstructionNode:
        return self._instr_node(SHR, Reg("A"))

    def shl_a(self, _: List[Any]) -> InstructionNode:
        return self._instr_node(SHL, Reg("A"))

    def ror_imem(self, items: List[Any]) -> InstructionNode:
        op = cast(IMemOperand, items[0])
        return self._instr_node(ROR, op)

    def rol_imem(self, items: List[Any]) -> InstructionNode:
        op = cast(IMemOperand, items[0])
        return self._instr_node(ROL, op)

    def shr_imem(self, items: List[Any]) -> InstructionNode:
        op = cast(IMemOperand, items[0])
        return self._instr_node(SHR, op)

    def shl_imem(self, items: List[Any]) -> InstructionNode:
        op = cast(IMemOperand, items[0])
        return self._instr_node(SHL, op)

    def mv_a_b(self, _: List[Any]) -> InstructionNode:
        return self._instr_node(MV, Reg("A"), RegB())

    def mv_b_a(self, _: List[Any]) -> InstructionNode:
        return self._instr_node(MV, RegB(), Reg("A"))

    def ex_a_b(self, _: List[Any]) -> InstructionNode:
        return self._instr_node(EX, Reg("A"), RegB())

    def pushs_f(self, _: List[Any]) -> InstructionNode:
        return self._instr_node(PUSHS, RegF())

    def pops_f(self, _: List[Any]) -> InstructionNode:
        return self._instr_node(POPS, RegF())

    def pushu_f(self, _: List[Any]) -> InstructionNode:
        return self._instr_node(PUSHU, RegF())

    def popu_f(self, _: List[Any]) -> InstructionNode:
        return self._instr_node(POPU, RegF())

    def pushu_imr(self, _: List[Any]) -> InstructionNode:
        return self._instr_node(PUSHU, RegIMR())

    def popu_imr(self, _: List[Any]) -> InstructionNode:
        return self._instr_node(POPU, RegIMR())

    def pushu_reg(self, items: List[Any]) -> InstructionNode:
        reg = cast(Reg, items[0])
        return self._instr_node(PUSHU, reg)

    def popu_reg(self, items: List[Any]) -> InstructionNode:
        reg = cast(Reg, items[0])
        return self._instr_node(POPU, reg)

    def call(self, items: List[Any]) -> InstructionNode:
        imm = self._imm16(items[0])
        return self._instr_node(CALL, imm)

    def callf(self, items: List[Any]) -> InstructionNode:
        imm = self._imm20(items[0])
        return self._instr_node(CALL, imm, name="CALLF")

    def jp_abs(self, items: List[Any]) -> InstructionNode:
        imm = self._imm16(items[0])
        return self._instr_node(JP_Abs, imm)

    def jpf_abs(self, items: List[Any]) -> InstructionNode:
        imm = self._imm20(items[0])
        return self._instr_node(JP_Abs, imm, name="JPF")

    def jpz_abs(self, items: List[Any]) -> InstructionNode:
        imm = self._imm16(items[0])
        return self._instr_node(JP_Abs, imm, cond="Z")

    def jpnz_abs(self, items: List[Any]) -> InstructionNode:
        imm = self._imm16(items[0])
        return self._instr_node(JP_Abs, imm, cond="NZ")

    def jpc_abs(self, items: List[Any]) -> InstructionNode:
        imm = self._imm16(items[0])
        return self._instr_node(JP_Abs, imm, cond="C")

    def jpnc_abs(self, items: List[Any]) -> InstructionNode:
        imm = self._imm16(items[0])
        return self._instr_node(JP_Abs, imm, cond="NC")

    def jp_reg(self, items: List[Any]) -> InstructionNode:
        reg = cast(Reg, items[0])
        try:
            idx = Reg3.reg_idx(cast(RegisterName, reg.reg))
        except (ValueError, TypeError):
            # Not a known register, treat as absolute jump to symbol
            return self.jp_abs([str(reg.reg)])
        r = Reg3()
        r.reg = cast(RegisterName, reg.reg)
        r.reg_raw = idx
        r.high4 = 0
        return self._instr_node(JP_Abs, r)

    def jp_imem(self, items: List[Any]) -> InstructionNode:
        op = cast(IMemOperand, items[0])
        imm = IMem20()
        imm.value = cast(int, op.n_val)
        return self._instr_node(JP_Abs, imm)

    def jr_plus(self, items: List[Any]) -> InstructionNode:
        imm = self._imm_offset("+", items[0])
        return self._instr_node(JP_Rel, imm)

    def jr_minus(self, items: List[Any]) -> InstructionNode:
        imm = self._imm_offset("-", items[0])
        return self._instr_node(JP_Rel, imm)

    def jrz_plus(self, items: List[Any]) -> InstructionNode:
        imm = self._imm_offset("+", items[0])
        return self._instr_node(JP_Rel, imm, cond="Z")

    def jrz_minus(self, items: List[Any]) -> InstructionNode:
        imm = self._imm_offset("-", items[0])
        return self._instr_node(JP_Rel, imm, cond="Z")

    def jrnz_plus(self, items: List[Any]) -> InstructionNode:
        imm = self._imm_offset("+", items[0])
        return self._instr_node(JP_Rel, imm, cond="NZ")

    def jrnz_minus(self, items: List[Any]) -> InstructionNode:
        imm = self._imm_offset("-", items[0])
        return self._instr_node(JP_Rel, imm, cond="NZ")

    def jrc_plus(self, items: List[Any]) -> InstructionNode:
        imm = self._imm_offset("+", items[0])
        return self._instr_node(JP_Rel, imm, cond="C")

    def jrc_minus(self, items: List[Any]) -> InstructionNode:
        imm = self._imm_offset("-", items[0])
        return self._instr_node(JP_Rel, imm, cond="C")

    def jrnc_plus(self, items: List[Any]) -> InstructionNode:
        imm = self._imm_offset("+", items[0])
        return self._instr_node(JP_Rel, imm, cond="NC")

    def jrnc_minus(self, items: List[Any]) -> InstructionNode:
        imm = self._imm_offset("-", items[0])
        return self._instr_node(JP_Rel, imm, cond="NC")

    def inc_reg(self, items: List[Any]) -> InstructionNode:
        reg = cast(Reg, items[0])
        r = Reg3()
        r.reg = reg.reg
        r.reg_raw = Reg3.reg_idx(reg.reg)
        r.high4 = 0
        return self._instr_node(INC, r)

    def inc_imem(self, items: List[Any]) -> InstructionNode:
        op = cast(IMemOperand, items[0])
        return self._instr_node(INC, op)

    def dec_reg(self, items: List[Any]) -> InstructionNode:
        reg = cast(Reg, items[0])
        r = Reg3()
        r.reg = reg.reg
        r.reg_raw = Reg3.reg_idx(reg.reg)
        r.high4 = 0
        return self._instr_node(DEC, r)

    def dec_imem(self, items: List[Any]) -> InstructionNode:
        op = cast(IMemOperand, items[0])
        return self._instr_node(DEC, op)

    def reg(self, items: List[Token]) -> Reg:
        reg_name = str(items[0]).upper()
        # Specific register types are handled by their rules (_A, _B, etc.)
        # This is a fallback for general-purpose registers.
        if reg_name == "B":
            return RegB()
        return Reg(reg_name)

    def _make_reg_pair(self, reg1: Reg, reg2: Reg) -> RegPair:
        rp = RegPair()
        rp.reg1 = reg1
        rp.reg2 = reg2
        rp.reg_raw = (RegPair.reg_idx(reg1.reg) << 4) | RegPair.reg_idx(reg2.reg)

        if reg1.width() >= 3:
            rp.size = 3
        else:
            if reg1.width() == reg2.width():
                rp.size = reg1.width()
            else:
                rp.size = reg1.width()
        return rp

    # --- Helper utilities for building InstructionNodes ---

    def _instr_node(
        self,
        cls: type[Instruction],
        *operands: Operand,
        name: str | None = None,
        cond: str | None = None,
    ) -> InstructionNode:
        """Create an InstructionNode with the given class and operands."""
        return {
            "instruction": {
                "instr_class": cls,
                "instr_opts": Opts(name=name, cond=cond, ops=list(operands)),
            }
        }

    @staticmethod
    def _imm8(value: Any) -> Imm8:
        imm = Imm8()
        imm.value = value
        return imm

    @staticmethod
    def _imm16(value: Any) -> Imm16:
        imm = Imm16()
        imm.value = value
        return imm

    @staticmethod
    def _imm20(value: Any) -> Imm20:
        imm = Imm20()
        imm.value = value
        return imm

    def _op_imm(
        self,
        instr_cls: type[Instruction],
        op: Operand,
        val: Any,
        *,
        name: str | None = None,
    ) -> InstructionNode:
        """Helper to build ``instr_cls op, #imm`` instructions."""
        imm = self._imm8(val)
        return self._instr_node(instr_cls, op, imm, name=name)

    def _reg_imm(
        self,
        instr_cls: type[Instruction],
        reg_name: str,
        val: Any,
    ) -> InstructionNode:
        """Helper for ``instr reg, #imm`` style instructions."""
        return self._op_imm(instr_cls, Reg(reg_name), val)

    @staticmethod
    def _imm_offset(sign: Literal['+', '-'], value: Any) -> ImmOffset:
        imm = ImmOffset(sign)
        imm.value = value
        return imm

    def atom(self, items: List[Any]) -> str:
        # This will return a number as a string, or a symbol name.
        # The assembler will resolve it later.
        return str(items[0])

    def expression(self, items: List[Any]) -> str:
        # For now, expressions are just atoms.
        return str(items[0])

    # --- Internal Memory Operand Rules ---

    def imem_n(self, items: List[Any]) -> IMemOperand:
        value = int(str(items[0]), 0)
        return IMemOperand(AddressingMode.N, n=value)

    def imem_bp_n(self, items: List[Any]) -> IMemOperand:
        value = items[0]
        if isinstance(value, str):
            upper = value.upper()
            if upper == "PX":
                return IMemOperand(AddressingMode.BP_PX)
            if upper == "PY":
                return IMemOperand(AddressingMode.BP_PY)
            value = int(value, 0)
        return IMemOperand(AddressingMode.BP_N, n=int(value))

    def imem_px_n(self, items: List[Any]) -> IMemOperand:
        value = int(str(items[0]), 0)
        return IMemOperand(AddressingMode.PX_N, n=value)

    def imem_py_n(self, items: List[Any]) -> IMemOperand:
        value = int(str(items[0]), 0)
        return IMemOperand(AddressingMode.PY_N, n=value)

    def imem_bp_px(self, items: List[Any]) -> IMemOperand:
        return IMemOperand(AddressingMode.BP_PX)

    def imem_bp_py(self, items: List[Any]) -> IMemOperand:
        return IMemOperand(AddressingMode.BP_PY)

    def imem_operand(self, items: List[Any]) -> IMemOperand:
        # This rule just passes through the IMemOperand object created by the more specific rules.
        return cast(IMemOperand, items[0])

    def emem_addr(self, items: List[Any]) -> EMemAddr:
        addr = EMemAddr(width=1)
        addr.value = items[0]
        return addr

    def emem_operand(self, items: List[Any]) -> EMemAddr:
        return cast(EMemAddr, items[0])

    def _build_emem_reg(
        self,
        reg: Reg,
        mode: EMemRegMode,
        offset: Optional[int] = None,
        sign: str = "+",
    ) -> EMemReg:
        r = Reg3()
        r.reg = reg.reg
        r.reg_raw = Reg3.reg_idx(reg.reg)
        r.high4 = mode.value
        op = EMemReg(width=reg.width())
        op.reg = r
        op.mode = mode
        if offset is not None:
            off = ImmOffset(sign)
            off.value = offset
            op.offset = off
        return op

    def _cmp_imem_reg(
        self,
        op1: IMemOperand,
        reg: Reg,
        mem_cls: Type[IMemOperand],
        instr: Type[Instruction],
    ) -> InstructionNode:
        mem = cast(Any, mem_cls())
        mem.value = op1.n_val
        r = Reg3()
        r.reg = reg.reg
        r.reg_raw = Reg3.reg_idx(cast(RegisterName, r.reg))
        r.high4 = 0
        return self._instr_node(instr, mem, r)

    def emem_reg_simple(self, items: List[Any]) -> EMemReg:
        reg = cast(Reg, items[0])
        return self._build_emem_reg(reg, EMemRegMode.SIMPLE)

    def emem_reg_post_inc(self, items: List[Any]) -> EMemReg:
        reg = cast(Reg, items[0])
        return self._build_emem_reg(reg, EMemRegMode.POST_INC)

    def emem_reg_pre_dec(self, items: List[Any]) -> EMemReg:
        reg = cast(Reg, items[0])
        return self._build_emem_reg(reg, EMemRegMode.PRE_DEC)

    def emem_reg_plus(self, items: List[Any]) -> EMemReg:
        reg = cast(Reg, items[0])
        value = items[1]
        return self._build_emem_reg(reg, EMemRegMode.POSITIVE_OFFSET, value, "+")

    def emem_reg_minus(self, items: List[Any]) -> EMemReg:
        reg = cast(Reg, items[0])
        value = items[1]
        return self._build_emem_reg(reg, EMemRegMode.NEGATIVE_OFFSET, value, "-")

    def emem_reg_operand(self, items: List[Any]) -> EMemReg:
        return cast(EMemReg, items[0])

    def emem_imem_simple(self, items: List[Any]) -> EMemIMem:
        im = cast(IMemOperand, items[0])
        op = EMemIMem()
        op.imem = im
        op.value = EMemIMemMode.SIMPLE.value
        op.mode = EMemIMemMode.SIMPLE
        return op

    def emem_imem_plus(self, items: List[Any]) -> EMemIMem:
        im, val = items
        im = cast(IMemOperand, im)
        offset = ImmOffset("+")
        offset.value = val
        op = EMemIMem()
        op.imem = im
        op.value = EMemIMemMode.POSITIVE_OFFSET.value
        op.mode = EMemIMemMode.POSITIVE_OFFSET
        op.offset = offset
        return op

    def emem_imem_minus(self, items: List[Any]) -> EMemIMem:
        im, val = items
        im = cast(IMemOperand, im)
        offset = ImmOffset("-")
        offset.value = val
        op = EMemIMem()
        op.imem = im
        op.value = EMemIMemMode.NEGATIVE_OFFSET.value
        op.mode = EMemIMemMode.NEGATIVE_OFFSET
        op.offset = offset
        return op

    def emem_imem_operand(self, items: List[Any]) -> EMemIMem:
        return cast(EMemIMem, items[0])

    # --- Instruction Rules ---

    def mv_imem_imem(self, items: List[Any]) -> InstructionNode:
        op1, op2 = items
        return {
            "instruction": {"instr_class": MV, "instr_opts": Opts(ops=[op1, op2])}
        }

    def mv_reg_reg(self, items: List[Any]) -> InstructionNode:
        reg1 = cast(Reg, items[0])
        reg2 = cast(Reg, items[1])
        rp = self._make_reg_pair(reg1, reg2)
        rp.size = 2
        return {
            "instruction": {"instr_class": MV, "instr_opts": Opts(ops=[rp])}
        }

    def mv_reg_imm(self, items: List[Any]) -> InstructionNode:
        reg = cast(Reg, items[0])
        val = items[1]
        width = reg.width()
        imm: ImmOperand
        if width == 1:
            imm = Imm8()
        elif width == 2:
            imm = Imm16()
        else:
            imm = Imm20()
        imm.value = val
        return {
            "instruction": {"instr_class": MV, "instr_opts": Opts(ops=[reg, imm])}
        }

    def mv_reg_imem(self, items: List[Any]) -> InstructionNode:
        reg = cast(Reg, items[0])
        mem = cast(IMemOperand, items[1])
        return {
            "instruction": {"instr_class": MV, "instr_opts": Opts(ops=[reg, mem])}
        }

    def mv_reg_emem(self, items: List[Any]) -> InstructionNode:
        reg = cast(Reg, items[0])
        mem = cast(EMemAddr | EMemReg, items[1])
        # The grammar for ``emem_operand`` also matches ``emem_reg_operand`` which
        # results in this handler receiving ``EMemReg`` instances as ``mem``.
        # For ``MV`` instructions the width of the external memory operand is
        # defined by the destination register width, so adjust it here to avoid
        # mismatches during opcode lookup.
        if isinstance(mem, EMemReg):
            mem.width = reg.width()
        return {
            "instruction": {"instr_class": MV, "instr_opts": Opts(ops=[reg, mem])}
        }

    def mv_imem_reg(self, items: List[Any]) -> InstructionNode:
        mem = cast(IMemOperand, items[0])
        reg = cast(Reg, items[1])
        return {
            "instruction": {"instr_class": MV, "instr_opts": Opts(ops=[mem, reg])}
        }

    def mv_emem_reg(self, items: List[Any]) -> InstructionNode:
        mem = cast(EMemAddr | EMemReg, items[0])
        reg = cast(Reg, items[1])
        if isinstance(mem, EMemReg):
            mem.width = reg.width()
        return {
            "instruction": {"instr_class": MV, "instr_opts": Opts(ops=[mem, reg])}
        }

    def mv_reg_ememreg(self, items: List[Any]) -> InstructionNode:
        reg = cast(Reg, items[0])
        mem = cast(EMemReg, items[1])
        mem.width = reg.width()
        return {
            "instruction": {"instr_class": MV, "instr_opts": Opts(ops=[reg, mem])}
        }

    def mv_ememreg_reg(self, items: List[Any]) -> InstructionNode:
        mem = cast(EMemReg, items[0])
        reg = cast(Reg, items[1])
        mem.width = reg.width()
        return {
            "instruction": {"instr_class": MV, "instr_opts": Opts(ops=[mem, reg])}
        }

    def mv_reg_ememimem(self, items: List[Any]) -> InstructionNode:
        reg = cast(Reg, items[0])
        mem = cast(EMemIMem, items[1])
        return {
            "instruction": {"instr_class": MV, "instr_opts": Opts(ops=[reg, mem])}
        }

    def mv_ememimem_reg(self, items: List[Any]) -> InstructionNode:
        mem = cast(EMemIMem, items[0])
        reg = cast(Reg, items[1])
        return {
            "instruction": {"instr_class": MV, "instr_opts": Opts(ops=[mem, reg])}
        }

    def mv_imem_imm(self, items: List[Any]) -> InstructionNode:
        mem, val = items
        return self._op_imm(MV, mem, val)

    def mv_emem_imm(self, items: List[Any]) -> InstructionNode:
        mem, val = items
        return self._op_imm(MV, mem, val)

    def mv_imem_emem(self, items: List[Any]) -> InstructionNode:
        imem = cast(IMemOperand, items[0])
        emem = cast(EMemAddr, items[1])
        return {
            "instruction": {"instr_class": MV, "instr_opts": Opts(ops=[imem, emem])}
        }

    def mv_emem_imem(self, items: List[Any]) -> InstructionNode:
        emem = cast(EMemAddr, items[0])
        imem = cast(IMemOperand, items[1])
        return self._instr_node(MV, emem, imem)

    def mv_imem_ememreg(self, items: List[Any]) -> InstructionNode:
        imem = cast(IMemOperand, items[0])
        regop = cast(EMemReg, items[1])
        op = RegIMemOffset(order=RegIMemOffsetOrder.DEST_IMEM)
        im = IMem8()
        im.value = imem.n_val
        op.imem = im
        op.reg = regop.reg
        op.mode = regop.mode
        op.offset = regop.offset
        return self._instr_node(MV, op)

    def mv_ememreg_imem(self, items: List[Any]) -> InstructionNode:
        regop = cast(EMemReg, items[0])
        imem = cast(IMemOperand, items[1])
        op = RegIMemOffset(order=RegIMemOffsetOrder.DEST_REG_OFFSET)
        im = IMem8()
        im.value = imem.n_val
        op.imem = im
        op.reg = regop.reg
        op.mode = regop.mode
        op.offset = regop.offset
        return self._instr_node(MV, op)

    def mv_imem_ememimem(self, items: List[Any]) -> InstructionNode:
        imem = cast(IMemOperand, items[0])
        src = cast(EMemIMem, items[1])
        op = EMemIMemOffset(EMemIMemOffsetOrder.DEST_INT_MEM)
        op.mode_imm.value = src.value
        im1 = IMem8()
        im1.value = imem.n_val
        op.imem1 = im1
        im2 = IMem8()
        src_val = cast(IMemOperand, src.imem).n_val if isinstance(src.imem, IMemOperand) else src.imem.value
        im2.value = src_val
        op.imem2 = im2
        op.mode = src.mode
        op.offset = src.offset
        return self._instr_node(MV, op)

    def mv_ememimem_imem(self, items: List[Any]) -> InstructionNode:
        src = cast(EMemIMem, items[0])
        imem = cast(IMemOperand, items[1])
        op = EMemIMemOffset(EMemIMemOffsetOrder.DEST_EXT_MEM)
        op.mode_imm.value = src.value
        im1 = IMem8()
        src_val = cast(IMemOperand, src.imem).n_val if isinstance(src.imem, IMemOperand) else src.imem.value
        im1.value = src_val
        op.imem1 = im1
        im2 = IMem8()
        im2.value = imem.n_val
        op.imem2 = im2
        op.mode = src.mode
        op.offset = src.offset
        return self._instr_node(MV, op)

    def mvw_imem_imem(self, items: List[Any]) -> InstructionNode:
        op1, op2 = items
        m1 = IMem16()
        m1.value = op1.n_val
        m2 = IMem16()
        m2.value = op2.n_val
        return self._instr_node(MV, m1, m2, name="MVW")

    def mvw_imem_imm(self, items: List[Any]) -> InstructionNode:
        mem, val = items
        dst = IMem16()
        dst.value = mem.n_val
        imm = Imm16()
        imm.value = val
        return self._instr_node(MV, dst, imm, name="MVW")

    def mvp_imem_imem(self, items: List[Any]) -> InstructionNode:
        op1, op2 = items
        m1 = IMem20()
        m1.value = op1.n_val
        m2 = IMem20()
        m2.value = op2.n_val
        return self._instr_node(MV, m1, m2, name="MVP")

    def mvp_imem_imm(self, items: List[Any]) -> InstructionNode:
        mem, val = items
        dst = IMem20()
        dst.value = mem.n_val
        imm = Imm20()
        imm.value = val
        return self._instr_node(MV, dst, imm, name="MVP")

    def mvw_imem_emem(self, items: List[Any]) -> InstructionNode:
        imem, emem_src = items
        dst = IMem16()
        dst.value = imem.n_val
        src = EMemAddr(width=2)
        src.value = emem_src.value
        return self._instr_node(MV, dst, src, name="MVW")

    def mvw_emem_imem(self, items: List[Any]) -> InstructionNode:
        emem_src, imem = items
        src = EMemAddr(width=2)
        src.value = emem_src.value
        dst = IMem16()
        dst.value = imem.n_val
        return self._instr_node(MV, src, dst, name="MVW")

    def mvw_imem_ememreg(self, items: List[Any]) -> InstructionNode:
        imem = cast(IMemOperand, items[0])
        regop = cast(EMemReg, items[1])
        op = RegIMemOffset(order=RegIMemOffsetOrder.DEST_IMEM)
        im = IMem8()
        im.value = int(imem.n_val, 0) if isinstance(imem.n_val, str) else imem.n_val
        op.imem = im
        op.reg = regop.reg
        op.mode = regop.mode
        op.offset = regop.offset
        return self._instr_node(MV, op, name="MVW")

    def mvw_ememreg_imem(self, items: List[Any]) -> InstructionNode:
        regop = cast(EMemReg, items[0])
        imem = cast(IMemOperand, items[1])
        op = RegIMemOffset(order=RegIMemOffsetOrder.DEST_REG_OFFSET)
        im = IMem8()
        im.value = int(imem.n_val, 0) if isinstance(imem.n_val, str) else imem.n_val
        op.imem = im
        op.reg = regop.reg
        op.mode = regop.mode
        op.offset = regop.offset
        return self._instr_node(MV, op, name="MVW")

    def mvw_imem_ememimem(self, items: List[Any]) -> InstructionNode:
        imem = cast(IMemOperand, items[0])
        src = cast(EMemIMem, items[1])
        op = EMemIMemOffset(EMemIMemOffsetOrder.DEST_INT_MEM, width=2)
        op.mode_imm.value = src.value
        im1 = IMem8()
        im1.value = imem.n_val
        op.imem1 = im1
        im2 = IMem8()
        src_val = cast(IMemOperand, src.imem).n_val if isinstance(src.imem, IMemOperand) else src.imem.value
        im2.value = src_val
        op.imem2 = im2
        op.mode = src.mode
        op.offset = src.offset
        return self._instr_node(MV, op, name="MVW")

    def mvw_ememimem_imem(self, items: List[Any]) -> InstructionNode:
        src = cast(EMemIMem, items[0])
        imem = cast(IMemOperand, items[1])
        op = EMemIMemOffset(EMemIMemOffsetOrder.DEST_EXT_MEM, width=2)
        op.mode_imm.value = src.value
        im1 = IMem8()
        src_val = cast(IMemOperand, src.imem).n_val if isinstance(src.imem, IMemOperand) else src.imem.value
        im1.value = src_val
        op.imem1 = im1
        im2 = IMem8()
        im2.value = imem.n_val
        op.imem2 = im2
        op.mode = src.mode
        op.offset = src.offset
        return self._instr_node(MV, op, name="MVW")

    def mvp_imem_emem(self, items: List[Any]) -> InstructionNode:
        imem, emem_src = items
        dst = IMem20()
        dst.value = imem.n_val
        src = EMemAddr(width=3)
        src.value = emem_src.value
        return self._instr_node(MV, dst, src, name="MVP")

    def mvp_emem_imem(self, items: List[Any]) -> InstructionNode:
        emem_src, imem = items
        src = EMemAddr(width=3)
        src.value = emem_src.value
        dst = IMem20()
        dst.value = imem.n_val
        return self._instr_node(MV, src, dst, name="MVP")

    def mvp_imem_ememreg(self, items: List[Any]) -> InstructionNode:
        imem = cast(IMemOperand, items[0])
        regop = cast(EMemReg, items[1])
        op = RegIMemOffset(order=RegIMemOffsetOrder.DEST_IMEM)
        im = IMem8()
        im.value = int(imem.n_val, 0) if isinstance(imem.n_val, str) else imem.n_val
        op.imem = im
        op.reg = regop.reg
        op.mode = regop.mode
        op.offset = regop.offset
        return self._instr_node(MV, op, name="MVP")

    def mvp_ememreg_imem(self, items: List[Any]) -> InstructionNode:
        regop = cast(EMemReg, items[0])
        imem = cast(IMemOperand, items[1])
        op = RegIMemOffset(order=RegIMemOffsetOrder.DEST_REG_OFFSET)
        im = IMem8()
        im.value = int(imem.n_val, 0) if isinstance(imem.n_val, str) else imem.n_val
        op.imem = im
        op.reg = regop.reg
        op.mode = regop.mode
        op.offset = regop.offset
        return self._instr_node(MV, op, name="MVP")

    def mvp_imem_ememimem(self, items: List[Any]) -> InstructionNode:
        imem = cast(IMemOperand, items[0])
        src = cast(EMemIMem, items[1])
        op = EMemIMemOffset(EMemIMemOffsetOrder.DEST_INT_MEM)
        op.mode_imm.value = src.value
        im1 = IMem8()
        im1.value = imem.n_val
        op.imem1 = im1
        im2 = IMem8()
        src_val = cast(IMemOperand, src.imem).n_val if isinstance(src.imem, IMemOperand) else src.imem.value
        im2.value = src_val
        op.imem2 = im2
        op.mode = src.mode
        op.offset = src.offset
        return self._instr_node(MV, op, name="MVP")

    def mvp_ememimem_imem(self, items: List[Any]) -> InstructionNode:
        src = cast(EMemIMem, items[0])
        imem = cast(IMemOperand, items[1])
        op = EMemIMemOffset(EMemIMemOffsetOrder.DEST_EXT_MEM)
        op.mode_imm.value = src.value
        im1 = IMem8()
        src_val = cast(IMemOperand, src.imem).n_val if isinstance(src.imem, IMemOperand) else src.imem.value
        im1.value = src_val
        op.imem1 = im1
        im2 = IMem8()
        im2.value = imem.n_val
        op.imem2 = im2
        op.mode = src.mode
        op.offset = src.offset
        return self._instr_node(MV, op, name="MVP")

    def mvl_imem_emem(self, items: List[Any]) -> InstructionNode:
        imem, emem_src = items
        return self._instr_node(MVL, imem, emem_src)

    def mvl_emem_imem(self, items: List[Any]) -> InstructionNode:
        emem_src, imem = items
        return self._instr_node(MVL, emem_src, imem)

    def mvl_imem_ememreg(self, items: List[Any]) -> InstructionNode:
        imem = cast(IMemOperand, items[0])
        regop = cast(EMemReg, items[1])
        # For simple, post-increment and pre-decrement modes the opcode expects
        # an ``EMemReg`` operand directly.  Only the ``+n``/``-n`` forms are
        # encoded using ``RegIMemOffset``.
        operand: Union[RegIMemOffset, EMemReg]
        if regop.mode in (
            EMemRegMode.POSITIVE_OFFSET,
            EMemRegMode.NEGATIVE_OFFSET,
        ):
            op = RegIMemOffset(
                order=RegIMemOffsetOrder.DEST_IMEM,
                allowed_modes=[
                    EMemRegMode.POSITIVE_OFFSET,
                    EMemRegMode.NEGATIVE_OFFSET,
                ],
            )
            im = IMem8()
            im.value = imem.n_val
            op.imem = im
            op.reg = regop.reg
            op.mode = regop.mode
            op.offset = regop.offset
            operand = op
        else:
            regop.width = 1
            operand = regop
        if operand is regop:
            return self._instr_node(MVL, imem, operand)
        return self._instr_node(MVL, operand)

    def mvl_ememreg_imem(self, items: List[Any]) -> InstructionNode:
        regop = cast(EMemReg, items[0])
        imem = cast(IMemOperand, items[1])
        operand: Union[RegIMemOffset, EMemReg]
        if regop.mode in (
            EMemRegMode.POSITIVE_OFFSET,
            EMemRegMode.NEGATIVE_OFFSET,
        ):
            op = RegIMemOffset(
                order=RegIMemOffsetOrder.DEST_REG_OFFSET,
                allowed_modes=[
                    EMemRegMode.POSITIVE_OFFSET,
                    EMemRegMode.NEGATIVE_OFFSET,
                ],
            )
            im = IMem8()
            im.value = imem.n_val
            op.imem = im
            op.reg = regop.reg
            op.mode = regop.mode
            op.offset = regop.offset
            operand = op
        else:
            # When using simple, post-increment or pre-decrement modes the
            # operands are encoded in the logical order.
            regop.width = 1
            operand = regop
        if operand is regop:
            return self._instr_node(MVL, operand, imem)
        return self._instr_node(MVL, operand)

    def mvl_imem_ememimem(self, items: List[Any]) -> InstructionNode:
        imem = cast(IMemOperand, items[0])
        src = cast(EMemIMem, items[1])
        op = EMemIMemOffset(EMemIMemOffsetOrder.DEST_INT_MEM)
        op.mode_imm.value = src.value
        im1 = IMem8()
        im1.value = imem.n_val
        op.imem1 = im1
        im2 = IMem8()
        src_val = cast(IMemOperand, src.imem).n_val if isinstance(src.imem, IMemOperand) else src.imem.value
        im2.value = src_val
        op.imem2 = im2
        op.mode = src.mode
        op.offset = src.offset
        return self._instr_node(MVL, op)

    def mvl_ememimem_imem(self, items: List[Any]) -> InstructionNode:
        src = cast(EMemIMem, items[0])
        imem = cast(IMemOperand, items[1])
        op = EMemIMemOffset(EMemIMemOffsetOrder.DEST_EXT_MEM)
        op.mode_imm.value = src.value
        im1 = IMem8()
        src_val = cast(IMemOperand, src.imem).n_val if isinstance(src.imem, IMemOperand) else src.imem.value
        im1.value = src_val
        op.imem1 = im1
        im2 = IMem8()
        im2.value = imem.n_val
        op.imem2 = im2
        op.mode = src.mode
        op.offset = src.offset
        return self._instr_node(MVL, op)

    def mvl_imem_imem(self, items: List[Any]) -> InstructionNode:
        op1, op2 = items
        return self._instr_node(MVL, op1, op2)

    def mvld_imem_imem(self, items: List[Any]) -> InstructionNode:
        op1, op2 = items
        return self._instr_node(MVLD, op1, op2)

    def ex_imem_imem(self, items: List[Any]) -> InstructionNode:
        op1, op2 = items
        return self._instr_node(EX, op1, op2)

    def ex_reg_reg(self, items: List[Any]) -> InstructionNode:
        reg1 = cast(Reg, items[0])
        reg2 = cast(Reg, items[1])
        rp = RegPair(size=2)
        rp.reg1 = reg1
        rp.reg2 = reg2
        rp.reg_raw = (RegPair.reg_idx(reg1.reg) << 4) | RegPair.reg_idx(reg2.reg)
        return self._instr_node(EX, rp)

    def exw_imem_imem(self, items: List[Any]) -> InstructionNode:
        op1, op2 = items
        return self._instr_node(EX, op1, op2, name="EXW")

    def exp_imem_imem(self, items: List[Any]) -> InstructionNode:
        op1, op2 = items
        return self._instr_node(EX, op1, op2, name="EXP")

    def exl_imem_imem(self, items: List[Any]) -> InstructionNode:
        op1, op2 = items
        return self._instr_node(EXL, op1, op2)

    def and_a_imm(self, items: List[Any]) -> InstructionNode:
        return self._reg_imm(AND, "A", items[0])

    def and_imem_imm(self, items: List[Any]) -> InstructionNode:
        op1, val = items
        return self._op_imm(AND, op1, val)

    def and_emem_imm(self, items: List[Any]) -> InstructionNode:
        op1, val = items
        return self._op_imm(AND, op1, val)

    def and_imem_a(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return self._instr_node(AND, op1, Reg("A"))

    def and_a_imem(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return self._instr_node(AND, Reg("A"), op1)

    def and_imem_imem(self, items: List[Any]) -> InstructionNode:
        op1, op2 = items
        return self._instr_node(AND, op1, op2)

    def add_a_imm(self, items: List[Any]) -> InstructionNode:
        return self._reg_imm(ADD, "A", items[0])

    def add_imem_imm(self, items: List[Any]) -> InstructionNode:
        op1, val = items
        return self._op_imm(ADD, op1, val)

    def add_a_imem(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return self._instr_node(ADD, Reg("A"), op1)

    def add_imem_a(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return self._instr_node(ADD, op1, Reg("A"))

    def add_reg_reg(self, items: List[Any]) -> InstructionNode:
        if len(items) == 1:
            reg1 = Reg("A")
            reg2 = cast(Reg, items[0])
        else:
            reg1 = cast(Reg, items[0])
            reg2 = cast(Reg, items[1])
        rp = self._make_reg_pair(reg1, reg2)
        return self._instr_node(ADD, rp)

    def adc_a_imm(self, items: List[Any]) -> InstructionNode:
        return self._reg_imm(ADC, "A", items[0])

    def adc_imem_imm(self, items: List[Any]) -> InstructionNode:
        op1, val = items
        return self._op_imm(ADC, op1, val)

    def adc_a_imem(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return self._instr_node(ADC, Reg("A"), op1)

    def adc_imem_a(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return self._instr_node(ADC, op1, Reg("A"))

    def sub_a_imm(self, items: List[Any]) -> InstructionNode:
        return self._reg_imm(SUB, "A", items[0])

    def sub_imem_imm(self, items: List[Any]) -> InstructionNode:
        op1, val = items
        return self._op_imm(SUB, op1, val)

    def sub_a_imem(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return self._instr_node(SUB, Reg("A"), op1)

    def sub_imem_a(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return self._instr_node(SUB, op1, Reg("A"))

    def sub_reg_reg(self, items: List[Any]) -> InstructionNode:
        if len(items) == 1:
            reg1 = Reg("A")
            reg2 = cast(Reg, items[0])
        else:
            reg1 = cast(Reg, items[0])
            reg2 = cast(Reg, items[1])
        rp = self._make_reg_pair(reg1, reg2)
        return self._instr_node(SUB, rp)

    def sbc_a_imm(self, items: List[Any]) -> InstructionNode:
        return self._reg_imm(SBC, "A", items[0])

    def sbc_imem_imm(self, items: List[Any]) -> InstructionNode:
        op1, val = items
        return self._op_imm(SBC, op1, val)

    def sbc_a_imem(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return self._instr_node(SBC, Reg("A"), op1)

    def sbc_imem_a(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return self._instr_node(SBC, op1, Reg("A"))

    def adcl_imem_imem(self, items: List[Any]) -> InstructionNode:
        dst, src = items
        return self._instr_node(ADCL, dst, src)

    def adcl_imem_a(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return self._instr_node(ADCL, op1, Reg("A"))

    def sbcl_imem_imem(self, items: List[Any]) -> InstructionNode:
        dst, src = items
        return self._instr_node(SBCL, dst, src)

    def sbcl_imem_a(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return self._instr_node(SBCL, op1, Reg("A"))

    def dadl_imem_imem(self, items: List[Any]) -> InstructionNode:
        dst, src = items
        return self._instr_node(DADL, dst, src)

    def dadl_imem_a(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return self._instr_node(DADL, op1, Reg("A"))

    def dsbl_imem_imem(self, items: List[Any]) -> InstructionNode:
        dst, src = items
        return self._instr_node(DSBL, dst, src)

    def dsbl_imem_a(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return self._instr_node(DSBL, op1, Reg("A"))

    def dsll_imem(self, items: List[Any]) -> InstructionNode:
        op = cast(IMemOperand, items[0])
        return self._instr_node(DSLL, op)

    def dsrl_imem(self, items: List[Any]) -> InstructionNode:
        op = cast(IMemOperand, items[0])
        return self._instr_node(DSRL, op)

    def pmdf_imem_imm(self, items: List[Any]) -> InstructionNode:
        op1, val = items
        return self._op_imm(PMDF, op1, val)

    def pmdf_imem_a(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return self._instr_node(PMDF, op1, Reg("A"))

    def or_a_imm(self, items: List[Any]) -> InstructionNode:
        return self._reg_imm(OR, "A", items[0])

    def or_imem_imm(self, items: List[Any]) -> InstructionNode:
        op1, val = items
        return self._op_imm(OR, op1, val)

    def or_emem_imm(self, items: List[Any]) -> InstructionNode:
        op1, val = items
        return self._op_imm(OR, op1, val)

    def or_imem_a(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return self._instr_node(OR, op1, Reg("A"))

    def or_a_imem(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return self._instr_node(OR, Reg("A"), op1)

    def or_imem_imem(self, items: List[Any]) -> InstructionNode:
        op1, op2 = items
        return self._instr_node(OR, op1, op2)

    def xor_a_imm(self, items: List[Any]) -> InstructionNode:
        return self._reg_imm(XOR, "A", items[0])

    def xor_imem_imm(self, items: List[Any]) -> InstructionNode:
        op1, val = items
        return self._op_imm(XOR, op1, val)

    def xor_emem_imm(self, items: List[Any]) -> InstructionNode:
        op1, val = items
        return self._op_imm(XOR, op1, val)

    def xor_imem_a(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return self._instr_node(XOR, op1, Reg("A"))

    def xor_a_imem(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return self._instr_node(XOR, Reg("A"), op1)

    def xor_imem_imem(self, items: List[Any]) -> InstructionNode:
        op1, op2 = items
        return self._instr_node(XOR, op1, op2)

    def cmp_a_imm(self, items: List[Any]) -> InstructionNode:
        return self._reg_imm(CMP, "A", items[0])

    def cmp_imem_imm(self, items: List[Any]) -> InstructionNode:
        op1, val = items
        return self._op_imm(CMP, op1, val)

    def cmp_emem_imm(self, items: List[Any]) -> InstructionNode:
        op1, val = items
        return self._op_imm(CMP, op1, val)

    def cmp_imem_a(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return self._instr_node(CMP, op1, Reg("A"))

    def cmp_imem_imem(self, items: List[Any]) -> InstructionNode:
        op1, op2 = items
        return self._instr_node(CMP, op1, op2)

    def cmpw_imem_imem(self, items: List[Any]) -> InstructionNode:
        op1, op2 = items
        m1 = IMem16()
        m1.value = op1.n_val
        m2 = IMem16()
        m2.value = op2.n_val
        return self._instr_node(CMPW, m1, m2)

    def cmpp_imem_imem(self, items: List[Any]) -> InstructionNode:
        op1, op2 = items
        m1 = IMem20()
        m1.value = op1.n_val
        m2 = IMem20()
        m2.value = op2.n_val
        return self._instr_node(CMPP, m1, m2)

    def cmpw_imem_reg(self, items: List[Any]) -> InstructionNode:
        op1, reg = items
        return self._cmp_imem_reg(op1, cast(Reg, reg), IMem16, CMPW)

    def cmpp_imem_reg(self, items: List[Any]) -> InstructionNode:
        op1, reg = items
        return self._cmp_imem_reg(op1, cast(Reg, reg), IMem20, CMPP)

    def test_a_imm(self, items: List[Any]) -> InstructionNode:
        return self._reg_imm(TEST, "A", items[0])

    def test_imem_imm(self, items: List[Any]) -> InstructionNode:
        op1, val = items
        return self._op_imm(TEST, op1, val)

    def test_emem_imm(self, items: List[Any]) -> InstructionNode:
        op1, val = items
        return self._op_imm(TEST, op1, val)

    def test_imem_a(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return self._instr_node(TEST, op1, Reg("A"))

    def def_arg(self, items: List[Any]) -> str:
        return str(items[0])

    def NUMBER(self, token: Token) -> str:
        return str(token)

    def string_literal(self, items: List[Token]) -> str:
        return str(items[0])[1:-1]  # Remove quotes

    def CNAME(self, token: Token) -> str:
        return str(token)

    # --- Instruction Aggregation ---
    def instruction(self, items: List[Any]) -> InstructionNode:
        """Pass through the single parsed instruction node."""
        assert len(items) == 1
        return cast(InstructionNode, items[0])

