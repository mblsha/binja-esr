from typing import Any, List, Optional, Union, cast, TypedDict, Type
from . import binja_api  # noqa: F401
from binaryninja.architecture import RegisterName
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
    IMemOperand,
    ImmOperand,
    Imm8,
    EMemAddr,
    EMemReg,
    EMemRegMode,
    EMemIMem,
    EMemIMemMode,
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
        return {"instruction": {"instr_class": NOP, "instr_opts": Opts()}}

    def reti(self, _: List[Any]) -> InstructionNode:
        return {"instruction": {"instr_class": RETI, "instr_opts": Opts()}}

    def ret(self, _: List[Any]) -> InstructionNode:
        return {"instruction": {"instr_class": RET, "instr_opts": Opts()}}

    def retf(self, _: List[Any]) -> InstructionNode:
        return {"instruction": {"instr_class": RETF, "instr_opts": Opts()}}

    def sc(self, _: List[Any]) -> InstructionNode:
        return {"instruction": {"instr_class": SC, "instr_opts": Opts()}}

    def rc(self, _: List[Any]) -> InstructionNode:
        return {"instruction": {"instr_class": RC, "instr_opts": Opts()}}

    def tcl(self, _: List[Any]) -> InstructionNode:
        return {"instruction": {"instr_class": TCL, "instr_opts": Opts()}}

    def halt(self, _: List[Any]) -> InstructionNode:
        return {"instruction": {"instr_class": HALT, "instr_opts": Opts()}}

    def off(self, _: List[Any]) -> InstructionNode:
        return {"instruction": {"instr_class": OFF, "instr_opts": Opts()}}

    def wait(self, _: List[Any]) -> InstructionNode:
        return {"instruction": {"instr_class": WAIT, "instr_opts": Opts()}}

    def ir(self, _: List[Any]) -> InstructionNode:
        return {"instruction": {"instr_class": IR, "instr_opts": Opts()}}

    def reset(self, _: List[Any]) -> InstructionNode:
        return {"instruction": {"instr_class": RESET, "instr_opts": Opts()}}

    def swap_a(self, _: List[Any]) -> InstructionNode:
        return {
            "instruction": {"instr_class": SWAP, "instr_opts": Opts(ops=[Reg("A")])}
        }

    def ror_a(self, _: List[Any]) -> InstructionNode:
        return {"instruction": {"instr_class": ROR, "instr_opts": Opts(ops=[Reg("A")])}}

    def rol_a(self, _: List[Any]) -> InstructionNode:
        return {"instruction": {"instr_class": ROL, "instr_opts": Opts(ops=[Reg("A")])}}

    def shr_a(self, _: List[Any]) -> InstructionNode:
        return {"instruction": {"instr_class": SHR, "instr_opts": Opts(ops=[Reg("A")])}}

    def shl_a(self, _: List[Any]) -> InstructionNode:
        return {"instruction": {"instr_class": SHL, "instr_opts": Opts(ops=[Reg("A")])}}

    def ror_imem(self, items: List[Any]) -> InstructionNode:
        op = cast(IMemOperand, items[0])
        return {"instruction": {"instr_class": ROR, "instr_opts": Opts(ops=[op])}}

    def rol_imem(self, items: List[Any]) -> InstructionNode:
        op = cast(IMemOperand, items[0])
        return {"instruction": {"instr_class": ROL, "instr_opts": Opts(ops=[op])}}

    def shr_imem(self, items: List[Any]) -> InstructionNode:
        op = cast(IMemOperand, items[0])
        return {"instruction": {"instr_class": SHR, "instr_opts": Opts(ops=[op])}}

    def shl_imem(self, items: List[Any]) -> InstructionNode:
        op = cast(IMemOperand, items[0])
        return {"instruction": {"instr_class": SHL, "instr_opts": Opts(ops=[op])}}

    def mv_a_b(self, _: List[Any]) -> InstructionNode:
        return {
            "instruction": {
                "instr_class": MV,
                "instr_opts": Opts(ops=[Reg("A"), RegB()]),
            }
        }

    def mv_b_a(self, _: List[Any]) -> InstructionNode:
        return {
            "instruction": {
                "instr_class": MV,
                "instr_opts": Opts(ops=[RegB(), Reg("A")]),
            }
        }

    def ex_a_b(self, _: List[Any]) -> InstructionNode:
        return {
            "instruction": {
                "instr_class": EX,
                "instr_opts": Opts(ops=[Reg("A"), RegB()]),
            }
        }

    def pushs_f(self, _: List[Any]) -> InstructionNode:
        return {"instruction": {"instr_class": PUSHS, "instr_opts": Opts(ops=[RegF()])}}

    def pops_f(self, _: List[Any]) -> InstructionNode:
        return {"instruction": {"instr_class": POPS, "instr_opts": Opts(ops=[RegF()])}}

    def pushu_f(self, _: List[Any]) -> InstructionNode:
        return {"instruction": {"instr_class": PUSHU, "instr_opts": Opts(ops=[RegF()])}}

    def popu_f(self, _: List[Any]) -> InstructionNode:
        return {"instruction": {"instr_class": POPU, "instr_opts": Opts(ops=[RegF()])}}

    def pushu_imr(self, _: List[Any]) -> InstructionNode:
        return {
            "instruction": {"instr_class": PUSHU, "instr_opts": Opts(ops=[RegIMR()])}
        }

    def popu_imr(self, _: List[Any]) -> InstructionNode:
        return {
            "instruction": {"instr_class": POPU, "instr_opts": Opts(ops=[RegIMR()])}
        }

    def pushu_reg(self, items: List[Any]) -> InstructionNode:
        reg = cast(Reg, items[0])
        return {"instruction": {"instr_class": PUSHU, "instr_opts": Opts(ops=[reg])}}

    def popu_reg(self, items: List[Any]) -> InstructionNode:
        reg = cast(Reg, items[0])
        return {"instruction": {"instr_class": POPU, "instr_opts": Opts(ops=[reg])}}

    def call(self, items: List[Any]) -> InstructionNode:
        imm = Imm16()
        imm.value = items[0]
        return {
            "instruction": {
                "instr_class": CALL,
                "instr_opts": Opts(ops=[imm]),
            }
        }

    def callf(self, items: List[Any]) -> InstructionNode:
        imm = Imm20()
        imm.value = items[0]
        return {
            "instruction": {
                "instr_class": CALL,
                "instr_opts": Opts(name="CALLF", ops=[imm]),
            }
        }

    def jp_abs(self, items: List[Any]) -> InstructionNode:
        imm = Imm16()
        imm.value = items[0]
        return {
            "instruction": {
                "instr_class": JP_Abs,
                "instr_opts": Opts(ops=[imm]),
            }
        }

    def jpf_abs(self, items: List[Any]) -> InstructionNode:
        imm = Imm20()
        imm.value = items[0]
        return {
            "instruction": {
                "instr_class": JP_Abs,
                "instr_opts": Opts(name="JPF", ops=[imm]),
            }
        }

    def jpz_abs(self, items: List[Any]) -> InstructionNode:
        imm = Imm16()
        imm.value = items[0]
        return {
            "instruction": {
                "instr_class": JP_Abs,
                "instr_opts": Opts(cond="Z", ops=[imm]),
            }
        }

    def jpnz_abs(self, items: List[Any]) -> InstructionNode:
        imm = Imm16()
        imm.value = items[0]
        return {
            "instruction": {
                "instr_class": JP_Abs,
                "instr_opts": Opts(cond="NZ", ops=[imm]),
            }
        }

    def jpc_abs(self, items: List[Any]) -> InstructionNode:
        imm = Imm16()
        imm.value = items[0]
        return {
            "instruction": {
                "instr_class": JP_Abs,
                "instr_opts": Opts(cond="C", ops=[imm]),
            }
        }

    def jpnc_abs(self, items: List[Any]) -> InstructionNode:
        imm = Imm16()
        imm.value = items[0]
        return {
            "instruction": {
                "instr_class": JP_Abs,
                "instr_opts": Opts(cond="NC", ops=[imm]),
            }
        }

    def jp_reg(self, items: List[Any]) -> InstructionNode:
        reg = cast(Reg, items[0])
        r = Reg3()
        r.reg = reg.reg
        r.reg_raw = Reg3.reg_idx(reg.reg)
        r.high4 = 0
        return {
            "instruction": {"instr_class": JP_Abs, "instr_opts": Opts(ops=[r])}}

    def jp_imem(self, items: List[Any]) -> InstructionNode:
        op = cast(IMemOperand, items[0])
        imm = IMem20()
        imm.value = cast(int, op.n_val)
        return {
            "instruction": {"instr_class": JP_Abs, "instr_opts": Opts(ops=[imm])}}

    def jr_plus(self, items: List[Any]) -> InstructionNode:
        imm = ImmOffset("+")
        imm.value = items[0]
        return {
            "instruction": {"instr_class": JP_Rel, "instr_opts": Opts(ops=[imm])}}

    def jr_minus(self, items: List[Any]) -> InstructionNode:
        imm = ImmOffset("-")
        imm.value = items[0]
        return {
            "instruction": {"instr_class": JP_Rel, "instr_opts": Opts(ops=[imm])}}

    def jrz_plus(self, items: List[Any]) -> InstructionNode:
        imm = ImmOffset("+")
        imm.value = items[0]
        return {
            "instruction": {
                "instr_class": JP_Rel,
                "instr_opts": Opts(cond="Z", ops=[imm]),
            }
        }

    def jrz_minus(self, items: List[Any]) -> InstructionNode:
        imm = ImmOffset("-")
        imm.value = items[0]
        return {
            "instruction": {
                "instr_class": JP_Rel,
                "instr_opts": Opts(cond="Z", ops=[imm]),
            }
        }

    def jrnz_plus(self, items: List[Any]) -> InstructionNode:
        imm = ImmOffset("+")
        imm.value = items[0]
        return {
            "instruction": {
                "instr_class": JP_Rel,
                "instr_opts": Opts(cond="NZ", ops=[imm]),
            }
        }

    def jrnz_minus(self, items: List[Any]) -> InstructionNode:
        imm = ImmOffset("-")
        imm.value = items[0]
        return {
            "instruction": {
                "instr_class": JP_Rel,
                "instr_opts": Opts(cond="NZ", ops=[imm]),
            }
        }

    def jrc_plus(self, items: List[Any]) -> InstructionNode:
        imm = ImmOffset("+")
        imm.value = items[0]
        return {
            "instruction": {
                "instr_class": JP_Rel,
                "instr_opts": Opts(cond="C", ops=[imm]),
            }
        }

    def jrc_minus(self, items: List[Any]) -> InstructionNode:
        imm = ImmOffset("-")
        imm.value = items[0]
        return {
            "instruction": {
                "instr_class": JP_Rel,
                "instr_opts": Opts(cond="C", ops=[imm]),
            }
        }

    def jrnc_plus(self, items: List[Any]) -> InstructionNode:
        imm = ImmOffset("+")
        imm.value = items[0]
        return {
            "instruction": {
                "instr_class": JP_Rel,
                "instr_opts": Opts(cond="NC", ops=[imm]),
            }
        }

    def jrnc_minus(self, items: List[Any]) -> InstructionNode:
        imm = ImmOffset("-")
        imm.value = items[0]
        return {
            "instruction": {
                "instr_class": JP_Rel,
                "instr_opts": Opts(cond="NC", ops=[imm]),
            }
        }

    def inc_reg(self, items: List[Any]) -> InstructionNode:
        reg = cast(Reg, items[0])
        r = Reg3()
        r.reg = reg.reg
        r.reg_raw = Reg3.reg_idx(reg.reg)
        r.high4 = 0
        return {"instruction": {"instr_class": INC, "instr_opts": Opts(ops=[r])}}

    def inc_imem(self, items: List[Any]) -> InstructionNode:
        op = cast(IMemOperand, items[0])
        return {"instruction": {"instr_class": INC, "instr_opts": Opts(ops=[op])}}

    def dec_reg(self, items: List[Any]) -> InstructionNode:
        reg = cast(Reg, items[0])
        r = Reg3()
        r.reg = reg.reg
        r.reg_raw = Reg3.reg_idx(reg.reg)
        r.high4 = 0
        return {"instruction": {"instr_class": DEC, "instr_opts": Opts(ops=[r])}}

    def dec_imem(self, items: List[Any]) -> InstructionNode:
        op = cast(IMemOperand, items[0])
        return {"instruction": {"instr_class": DEC, "instr_opts": Opts(ops=[op])}}

    def reg(self, items: List[Token]) -> Reg:
        reg_name = str(items[0]).upper()
        # Specific register types are handled by their rules (_A, _B, etc.)
        # This is a fallback for general-purpose registers.
        if reg_name == "B":
            return RegB()
        return Reg(reg_name)

    def atom(self, items: List[Any]) -> str:
        # This will return a number as a string, or a symbol name.
        # The assembler will resolve it later.
        return str(items[0])

    def expression(self, items: List[Any]) -> str:
        # For now, expressions are just atoms.
        return str(items[0])

    # --- Internal Memory Operand Rules ---

    def imem_n(self, items: List[Any]) -> IMemOperand:
        return IMemOperand(AddressingMode.N, n=items[0])

    def imem_bp_n(self, items: List[Any]) -> IMemOperand:
        value = items[0]
        if isinstance(value, str):
            upper = value.upper()
            if upper == "PX":
                return IMemOperand(AddressingMode.BP_PX)
            if upper == "PY":
                return IMemOperand(AddressingMode.BP_PY)
        return IMemOperand(AddressingMode.BP_N, n=value)

    def imem_px_n(self, items: List[Any]) -> IMemOperand:
        return IMemOperand(AddressingMode.PX_N, n=items[0])

    def imem_py_n(self, items: List[Any]) -> IMemOperand:
        return IMemOperand(AddressingMode.PY_N, n=items[0])

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

    def emem_reg_simple(self, items: List[Any]) -> EMemReg:
        reg = cast(Reg, items[0])
        r = Reg3()
        r.reg = reg.reg
        r.reg_raw = Reg3.reg_idx(reg.reg)
        r.high4 = 0
        op = EMemReg(width=reg.width())
        op.reg = r
        op.mode = EMemRegMode.SIMPLE
        return op

    def emem_reg_post_inc(self, items: List[Any]) -> EMemReg:
        reg = cast(Reg, items[0])
        r = Reg3()
        r.reg = reg.reg
        r.reg_raw = Reg3.reg_idx(reg.reg)
        r.high4 = 0
        op = EMemReg(width=reg.width())
        op.reg = r
        op.mode = EMemRegMode.POST_INC
        return op

    def emem_reg_pre_dec(self, items: List[Any]) -> EMemReg:
        reg = cast(Reg, items[0])
        r = Reg3()
        r.reg = reg.reg
        r.reg_raw = Reg3.reg_idx(reg.reg)
        r.high4 = 0
        op = EMemReg(width=reg.width())
        op.reg = r
        op.mode = EMemRegMode.PRE_DEC
        return op

    def emem_reg_plus(self, items: List[Any]) -> EMemReg:
        reg = cast(Reg, items[0])
        value = items[1]
        r = Reg3()
        r.reg = reg.reg
        r.reg_raw = Reg3.reg_idx(reg.reg)
        r.high4 = 0
        offset = ImmOffset("+")
        offset.value = value
        op = EMemReg(width=reg.width())
        op.reg = r
        op.mode = EMemRegMode.POSITIVE_OFFSET
        op.offset = offset
        return op

    def emem_reg_minus(self, items: List[Any]) -> EMemReg:
        reg = cast(Reg, items[0])
        value = items[1]
        r = Reg3()
        r.reg = reg.reg
        r.reg_raw = Reg3.reg_idx(reg.reg)
        r.high4 = 0
        offset = ImmOffset("-")
        offset.value = value
        op = EMemReg(width=reg.width())
        op.reg = r
        op.mode = EMemRegMode.NEGATIVE_OFFSET
        op.offset = offset
        return op

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
        return {
            "instruction": {"instr_class": MV, "instr_opts": Opts(ops=[reg1, reg2])}
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
        mem = cast(EMemAddr, items[1])
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
        mem = cast(EMemAddr, items[0])
        reg = cast(Reg, items[1])
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
        imm = Imm8()
        imm.value = val
        return {
            "instruction": {"instr_class": MV, "instr_opts": Opts(ops=[mem, imm])}
        }

    def mv_emem_imm(self, items: List[Any]) -> InstructionNode:
        mem, val = items
        imm = Imm8()
        imm.value = val
        return {
            "instruction": {"instr_class": MV, "instr_opts": Opts(ops=[mem, imm])}
        }

    def mvw_imem_imem(self, items: List[Any]) -> InstructionNode:
        op1, op2 = items
        m1 = IMem16()
        m1.value = op1.n_val
        m2 = IMem16()
        m2.value = op2.n_val
        return {
            "instruction": {"instr_class": MV, "instr_opts": Opts(name="MVW", ops=[m1, m2])}
        }

    def mvw_imem_imm(self, items: List[Any]) -> InstructionNode:
        mem, val = items
        dst = IMem16()
        dst.value = mem.n_val
        imm = Imm16()
        imm.value = val
        return {
            "instruction": {"instr_class": MV, "instr_opts": Opts(name="MVW", ops=[dst, imm])}
        }

    def mvp_imem_imem(self, items: List[Any]) -> InstructionNode:
        op1, op2 = items
        m1 = IMem20()
        m1.value = op1.n_val
        m2 = IMem20()
        m2.value = op2.n_val
        return {
            "instruction": {"instr_class": MV, "instr_opts": Opts(name="MVP", ops=[m1, m2])}
        }

    def mvp_imem_imm(self, items: List[Any]) -> InstructionNode:
        mem, val = items
        dst = IMem20()
        dst.value = mem.n_val
        imm = Imm20()
        imm.value = val
        return {
            "instruction": {"instr_class": MV, "instr_opts": Opts(name="MVP", ops=[dst, imm])}
        }

    def mvl_imem_imem(self, items: List[Any]) -> InstructionNode:
        op1, op2 = items
        return {
            "instruction": {"instr_class": MVL, "instr_opts": Opts(ops=[op1, op2])}
        }

    def mvld_imem_imem(self, items: List[Any]) -> InstructionNode:
        op1, op2 = items
        return {
            "instruction": {"instr_class": MVLD, "instr_opts": Opts(ops=[op1, op2])}
        }

    def ex_imem_imem(self, items: List[Any]) -> InstructionNode:
        op1, op2 = items
        return {
            "instruction": {"instr_class": EX, "instr_opts": Opts(ops=[op1, op2])}
        }

    def ex_reg_reg(self, items: List[Any]) -> InstructionNode:
        reg1 = cast(Reg, items[0])
        reg2 = cast(Reg, items[1])
        rp = RegPair(size=2)
        rp.reg1 = reg1
        rp.reg2 = reg2
        rp.reg_raw = (RegPair.reg_idx(reg1.reg) << 4) | RegPair.reg_idx(reg2.reg)
        return {
            "instruction": {"instr_class": EX, "instr_opts": Opts(ops=[rp])}
        }

    def exw_imem_imem(self, items: List[Any]) -> InstructionNode:
        op1, op2 = items
        return {
            "instruction": {"instr_class": EX, "instr_opts": Opts(name="EXW", ops=[op1, op2])}
        }

    def exp_imem_imem(self, items: List[Any]) -> InstructionNode:
        op1, op2 = items
        return {
            "instruction": {"instr_class": EX, "instr_opts": Opts(name="EXP", ops=[op1, op2])}
        }

    def exl_imem_imem(self, items: List[Any]) -> InstructionNode:
        op1, op2 = items
        return {
            "instruction": {"instr_class": EXL, "instr_opts": Opts(ops=[op1, op2])}
        }

    def and_a_imm(self, items: List[Any]) -> InstructionNode:
        imm = Imm8()
        imm.value = items[0]
        return {
            "instruction": {"instr_class": AND, "instr_opts": Opts(ops=[Reg("A"), imm])}
        }

    def and_imem_imm(self, items: List[Any]) -> InstructionNode:
        op1, val = items
        imm = Imm8()
        imm.value = val
        return {
            "instruction": {"instr_class": AND, "instr_opts": Opts(ops=[op1, imm])}
        }

    def and_emem_imm(self, items: List[Any]) -> InstructionNode:
        op1, val = items
        imm = Imm8()
        imm.value = val
        return {
            "instruction": {"instr_class": AND, "instr_opts": Opts(ops=[op1, imm])}
        }

    def and_imem_a(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return {
            "instruction": {"instr_class": AND, "instr_opts": Opts(ops=[op1, Reg("A")])}
        }

    def and_a_imem(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return {
            "instruction": {"instr_class": AND, "instr_opts": Opts(ops=[Reg("A"), op1])}
        }

    def and_imem_imem(self, items: List[Any]) -> InstructionNode:
        op1, op2 = items
        return {
            "instruction": {"instr_class": AND, "instr_opts": Opts(ops=[op1, op2])}
        }

    def add_a_imm(self, items: List[Any]) -> InstructionNode:
        imm = Imm8()
        imm.value = items[0]
        return {
            "instruction": {"instr_class": ADD, "instr_opts": Opts(ops=[Reg("A"), imm])}
        }

    def add_imem_imm(self, items: List[Any]) -> InstructionNode:
        op1, val = items
        imm = Imm8()
        imm.value = val
        return {
            "instruction": {"instr_class": ADD, "instr_opts": Opts(ops=[op1, imm])}
        }

    def add_a_imem(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return {
            "instruction": {"instr_class": ADD, "instr_opts": Opts(ops=[Reg("A"), op1])}
        }

    def add_imem_a(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return {
            "instruction": {"instr_class": ADD, "instr_opts": Opts(ops=[op1, Reg("A")])}
        }

    def adc_a_imm(self, items: List[Any]) -> InstructionNode:
        imm = Imm8()
        imm.value = items[0]
        return {
            "instruction": {"instr_class": ADC, "instr_opts": Opts(ops=[Reg("A"), imm])}
        }

    def adc_imem_imm(self, items: List[Any]) -> InstructionNode:
        op1, val = items
        imm = Imm8()
        imm.value = val
        return {
            "instruction": {"instr_class": ADC, "instr_opts": Opts(ops=[op1, imm])}
        }

    def adc_a_imem(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return {
            "instruction": {"instr_class": ADC, "instr_opts": Opts(ops=[Reg("A"), op1])}
        }

    def adc_imem_a(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return {
            "instruction": {"instr_class": ADC, "instr_opts": Opts(ops=[op1, Reg("A")])}
        }

    def sub_a_imm(self, items: List[Any]) -> InstructionNode:
        imm = Imm8()
        imm.value = items[0]
        return {
            "instruction": {"instr_class": SUB, "instr_opts": Opts(ops=[Reg("A"), imm])}
        }

    def sub_imem_imm(self, items: List[Any]) -> InstructionNode:
        op1, val = items
        imm = Imm8()
        imm.value = val
        return {
            "instruction": {"instr_class": SUB, "instr_opts": Opts(ops=[op1, imm])}
        }

    def sub_a_imem(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return {
            "instruction": {"instr_class": SUB, "instr_opts": Opts(ops=[Reg("A"), op1])}
        }

    def sub_imem_a(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return {
            "instruction": {"instr_class": SUB, "instr_opts": Opts(ops=[op1, Reg("A")])}
        }

    def sbc_a_imm(self, items: List[Any]) -> InstructionNode:
        imm = Imm8()
        imm.value = items[0]
        return {
            "instruction": {"instr_class": SBC, "instr_opts": Opts(ops=[Reg("A"), imm])}
        }

    def sbc_imem_imm(self, items: List[Any]) -> InstructionNode:
        op1, val = items
        imm = Imm8()
        imm.value = val
        return {
            "instruction": {"instr_class": SBC, "instr_opts": Opts(ops=[op1, imm])}
        }

    def sbc_a_imem(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return {
            "instruction": {"instr_class": SBC, "instr_opts": Opts(ops=[Reg("A"), op1])}
        }

    def sbc_imem_a(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return {
            "instruction": {"instr_class": SBC, "instr_opts": Opts(ops=[op1, Reg("A")])}
        }

    def adcl_imem_imem(self, items: List[Any]) -> InstructionNode:
        dst, src = items
        return {
            "instruction": {"instr_class": ADCL, "instr_opts": Opts(ops=[dst, src])}
        }

    def adcl_imem_a(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return {
            "instruction": {"instr_class": ADCL, "instr_opts": Opts(ops=[op1, Reg("A")])}
        }

    def sbcl_imem_imem(self, items: List[Any]) -> InstructionNode:
        dst, src = items
        return {
            "instruction": {"instr_class": SBCL, "instr_opts": Opts(ops=[dst, src])}
        }

    def sbcl_imem_a(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return {
            "instruction": {"instr_class": SBCL, "instr_opts": Opts(ops=[op1, Reg("A")])}
        }

    def dadl_imem_imem(self, items: List[Any]) -> InstructionNode:
        dst, src = items
        return {
            "instruction": {"instr_class": DADL, "instr_opts": Opts(ops=[dst, src])}
        }

    def dadl_imem_a(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return {
            "instruction": {"instr_class": DADL, "instr_opts": Opts(ops=[op1, Reg("A")])}
        }

    def dsbl_imem_imem(self, items: List[Any]) -> InstructionNode:
        dst, src = items
        return {
            "instruction": {"instr_class": DSBL, "instr_opts": Opts(ops=[dst, src])}
        }

    def dsbl_imem_a(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return {
            "instruction": {"instr_class": DSBL, "instr_opts": Opts(ops=[op1, Reg("A")])}
        }

    def dsll_imem(self, items: List[Any]) -> InstructionNode:
        op = cast(IMemOperand, items[0])
        return {
            "instruction": {"instr_class": DSLL, "instr_opts": Opts(ops=[op])}
        }

    def dsrl_imem(self, items: List[Any]) -> InstructionNode:
        op = cast(IMemOperand, items[0])
        return {
            "instruction": {"instr_class": DSRL, "instr_opts": Opts(ops=[op])}
        }

    def pmdf_imem_imm(self, items: List[Any]) -> InstructionNode:
        op1, val = items
        imm = Imm8()
        imm.value = val
        return {
            "instruction": {"instr_class": PMDF, "instr_opts": Opts(ops=[op1, imm])}
        }

    def pmdf_imem_a(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return {
            "instruction": {"instr_class": PMDF, "instr_opts": Opts(ops=[op1, Reg("A")])}
        }

    def or_a_imm(self, items: List[Any]) -> InstructionNode:
        imm = Imm8()
        imm.value = items[0]
        return {
            "instruction": {"instr_class": OR, "instr_opts": Opts(ops=[Reg("A"), imm])}
        }

    def or_imem_imm(self, items: List[Any]) -> InstructionNode:
        op1, val = items
        imm = Imm8()
        imm.value = val
        return {
            "instruction": {"instr_class": OR, "instr_opts": Opts(ops=[op1, imm])}
        }

    def or_emem_imm(self, items: List[Any]) -> InstructionNode:
        op1, val = items
        imm = Imm8()
        imm.value = val
        return {
            "instruction": {"instr_class": OR, "instr_opts": Opts(ops=[op1, imm])}
        }

    def or_imem_a(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return {
            "instruction": {"instr_class": OR, "instr_opts": Opts(ops=[op1, Reg("A")])}
        }

    def or_a_imem(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return {
            "instruction": {"instr_class": OR, "instr_opts": Opts(ops=[Reg("A"), op1])}
        }

    def or_imem_imem(self, items: List[Any]) -> InstructionNode:
        op1, op2 = items
        return {
            "instruction": {"instr_class": OR, "instr_opts": Opts(ops=[op1, op2])}
        }

    def xor_a_imm(self, items: List[Any]) -> InstructionNode:
        imm = Imm8()
        imm.value = items[0]
        return {
            "instruction": {"instr_class": XOR, "instr_opts": Opts(ops=[Reg("A"), imm])}
        }

    def xor_imem_imm(self, items: List[Any]) -> InstructionNode:
        op1, val = items
        imm = Imm8()
        imm.value = val
        return {
            "instruction": {"instr_class": XOR, "instr_opts": Opts(ops=[op1, imm])}
        }

    def xor_emem_imm(self, items: List[Any]) -> InstructionNode:
        op1, val = items
        imm = Imm8()
        imm.value = val
        return {
            "instruction": {"instr_class": XOR, "instr_opts": Opts(ops=[op1, imm])}
        }

    def xor_imem_a(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return {
            "instruction": {"instr_class": XOR, "instr_opts": Opts(ops=[op1, Reg("A")])}
        }

    def xor_a_imem(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return {
            "instruction": {"instr_class": XOR, "instr_opts": Opts(ops=[Reg("A"), op1])}
        }

    def xor_imem_imem(self, items: List[Any]) -> InstructionNode:
        op1, op2 = items
        return {
            "instruction": {"instr_class": XOR, "instr_opts": Opts(ops=[op1, op2])}
        }

    def cmp_a_imm(self, items: List[Any]) -> InstructionNode:
        imm = Imm8()
        imm.value = items[0]
        return {
            "instruction": {"instr_class": CMP, "instr_opts": Opts(ops=[Reg("A"), imm])}
        }

    def cmp_imem_imm(self, items: List[Any]) -> InstructionNode:
        op1, val = items
        imm = Imm8()
        imm.value = val
        return {
            "instruction": {"instr_class": CMP, "instr_opts": Opts(ops=[op1, imm])}
        }

    def cmp_emem_imm(self, items: List[Any]) -> InstructionNode:
        op1, val = items
        imm = Imm8()
        imm.value = val
        return {
            "instruction": {"instr_class": CMP, "instr_opts": Opts(ops=[op1, imm])}
        }

    def cmp_imem_a(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return {
            "instruction": {"instr_class": CMP, "instr_opts": Opts(ops=[op1, Reg("A")])}
        }

    def cmp_imem_imem(self, items: List[Any]) -> InstructionNode:
        op1, op2 = items
        return {
            "instruction": {"instr_class": CMP, "instr_opts": Opts(ops=[op1, op2])}
        }

    def cmpw_imem_imem(self, items: List[Any]) -> InstructionNode:
        op1, op2 = items
        m1 = IMem16()
        m1.value = op1.n_val
        m2 = IMem16()
        m2.value = op2.n_val
        return {
            "instruction": {"instr_class": CMPW, "instr_opts": Opts(ops=[m1, m2])}
        }

    def cmpp_imem_imem(self, items: List[Any]) -> InstructionNode:
        op1, op2 = items
        m1 = IMem20()
        m1.value = op1.n_val
        m2 = IMem20()
        m2.value = op2.n_val
        return {
            "instruction": {"instr_class": CMPP, "instr_opts": Opts(ops=[m1, m2])}
        }

    def cmpw_imem_reg(self, items: List[Any]) -> InstructionNode:
        op1, reg = items
        m = IMem16()
        m.value = op1.n_val
        r = Reg3()
        r.reg = cast(Reg, reg).reg
        r.reg_raw = Reg3.reg_idx(cast(RegisterName, r.reg))
        r.high4 = 0
        return {
            "instruction": {"instr_class": CMPW, "instr_opts": Opts(ops=[m, r])}
        }

    def cmpp_imem_reg(self, items: List[Any]) -> InstructionNode:
        op1, reg = items
        m = IMem20()
        m.value = op1.n_val
        r = Reg3()
        r.reg = cast(Reg, reg).reg
        r.reg_raw = Reg3.reg_idx(cast(RegisterName, r.reg))
        r.high4 = 0
        return {
            "instruction": {"instr_class": CMPP, "instr_opts": Opts(ops=[m, r])}
        }

    def test_a_imm(self, items: List[Any]) -> InstructionNode:
        imm = Imm8()
        imm.value = items[0]
        return {
            "instruction": {"instr_class": TEST, "instr_opts": Opts(ops=[Reg("A"), imm])}
        }

    def test_imem_imm(self, items: List[Any]) -> InstructionNode:
        op1, val = items
        imm = Imm8()
        imm.value = val
        return {
            "instruction": {"instr_class": TEST, "instr_opts": Opts(ops=[op1, imm])}
        }

    def test_emem_imm(self, items: List[Any]) -> InstructionNode:
        op1, val = items
        imm = Imm8()
        imm.value = val
        return {
            "instruction": {"instr_class": TEST, "instr_opts": Opts(ops=[op1, imm])}
        }

    def test_imem_a(self, items: List[Any]) -> InstructionNode:
        op1 = items[0]
        return {
            "instruction": {"instr_class": TEST, "instr_opts": Opts(ops=[op1, Reg("A")])}
        }

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

