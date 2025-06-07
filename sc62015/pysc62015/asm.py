from typing import Any, List, Optional, Union, cast, TypedDict, Type
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
    EX,
    PUSHS,
    POPS,
    PUSHU,
    POPU,
    Reg,
    RegB,
    RegF,
    RegIMR,
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
        # Filter out empty lines that might result from parsing only newlines
        return {"lines": [line for line in items if line]}

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

    def def_arg(self, items: List[Any]) -> str:
        return str(items[0])

    def NUMBER(self, token: Token) -> str:
        return str(token)

    def string_literal(self, items: List[Token]) -> str:
        return str(items[0])[1:-1]  # Remove quotes

    def CNAME(self, token: Token) -> str:
        return str(token)

    def atom(self, items: List[Any]) -> str:
        return str(items[0])

    def expression(self, items: List[Any]) -> List[Any]:
        return items
