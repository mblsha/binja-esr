from typing import Any, List, Optional, Union, cast, TypedDict
from lark import Lark, Transformer, Token

import os
grammar_path = os.path.join(os.path.dirname(__file__), 'asm.lark')
with open(grammar_path, 'r') as f:
    asm_grammar = f.read()

asm_parser = Lark(asm_grammar, parser='earley', maybe_placeholders=False)

class LabelNode(TypedDict):
    label: str

class SectionNode(TypedDict):
    section: str

class DataDirectiveNode(TypedDict):
    type: str  # "defb", "defw", "defl", "defs", "defm"
    args: Union[List[str], int, str]

class InstructionNode(TypedDict):
    instruction: str

class LineNode(TypedDict, total=False):
    label: Optional[str]
    statement: Optional[Union[SectionNode, DataDirectiveNode, InstructionNode]]

class ProgramNode(TypedDict):
    lines: List[LineNode]

class AsmTransformer(Transformer):

    def start(self, items: List[LineNode]) -> ProgramNode:
        return {"lines": items}

    def line(self, items: List[Any]) -> LineNode:
        # line: label? statement? NEWLINE | NEWLINE
        out: LineNode = {}
        if not items:
            return out

        if len(items) == 2:
            out["label"] = cast(str, items[0])
            out["statement"] = cast(Union[SectionNode, DataDirectiveNode, InstructionNode], items[1])
        elif len(items) == 1:
            # Could be only label or only statement
            item = items[0]
            if isinstance(item, str):  # label only
                out["label"] = item
            elif isinstance(item, dict):
                out["statement"] = item
        return out

    def label(self, items: List[Token]) -> str:
        return str(items[0])

    def section_decl(self, items: List[Any]) -> SectionNode:
        return {"section": str(items[-1])}

    def defb_directive(self, items: List[str]) -> DataDirectiveNode:
        return {"type": "defb", "args": items}

    def defw_directive(self, items: List[str]) -> DataDirectiveNode:
        return {"type": "defw", "args": items}

    def defl_directive(self, items: List[str]) -> DataDirectiveNode:
        return {"type": "defl", "args": items}

    def defs_directive(self, items: List[str]) -> DataDirectiveNode:
        return {"type": "defs", "args": int(items[0])}

    def defm_directive(self, items: List[str]) -> DataDirectiveNode:
        return {"type": "defm", "args": items[0]}

    def instruction(self, items: List[Token]) -> InstructionNode:
        return {"instruction": str(items[0])}

    def INSTRUCTION(self, token: Token) -> Token:
        return token

    def defb_arg(self, items: List[Token]) -> str:
        return str(items[0])

    def defw_arg(self, items: List[Token]) -> str:
        return str(items[0])

    def defl_arg(self, items: List[Token]) -> str:
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

