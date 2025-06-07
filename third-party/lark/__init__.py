from __future__ import annotations
import re
from typing import Any, List

__all__ = ["Lark", "Transformer", "Token", "Tree"]

class Token(str):
    def __new__(cls, type_: str, value: str):
        obj = str.__new__(cls, value)
        obj.type = type_
        obj.value = value
        return obj

class Tree:
    def __init__(self, data: str, children: List[Any]):
        self.data = data
        self.children = list(children)

    def __iter__(self):
        return iter(self.children)

    def __repr__(self) -> str:
        return f"Tree({self.data!r}, {self.children!r})"

    @classmethod
    def __class_getitem__(cls, item: Any) -> "Tree":
        return cls

class Transformer:
    pass

class Lark:
    def __init__(self, grammar: str, parser: str = "earley", maybe_placeholders: bool = False):
        # grammar is ignored; parser settings unused
        pass

    token_re = re.compile(
        r"(?P<COMMENT>;[^\n]*)"
        r"|(?P<NEWLINE>\r?\n)"
        r"|(?P<WS_INLINE>[ \t]+)"
        r"|(?P<SECTION>SECTION\b)"
        r"|(?P<DEFB>defb\b)"
        r"|(?P<DEFW>defw\b)"
        r"|(?P<DEFL>defl\b)"
        r"|(?P<DEFS>defs\b)"
        r"|(?P<DEFM>defm\b)"
        r"|(?P<COMMA>,)"
        r"|(?P<COLON>:)"
        r"|(?P<ESCAPED_STRING>\"(?:\\\\.|[^\"\\\\])*\")"
        r"|(?P<BIN>0b[01]+|[01]+[bB])"
        r"|(?P<HEX>0x[0-9a-fA-F]+|[0-9a-fA-F]+[hH])"
        r"|(?P<INT>[0-9]+)"
        r"|(?P<CNAME>[A-Za-z_][A-Za-z0-9_]*)",
    )

    def lex(self, text: str) -> List[Token]:
        tokens: List[Token] = []
        pos = 0
        while pos < len(text):
            m = self.token_re.match(text, pos)
            if not m:
                raise ValueError(f"Unexpected character at position {pos}")
            kind = m.lastgroup
            value = m.group(kind)
            pos = m.end()
            if kind == "COMMENT":
                continue  # ignore comments
            if kind in {"INT", "HEX", "BIN"}:
                kind = "NUMBER"
            tokens.append(Token(kind, value))
        return tokens

    def parse(self, text: str) -> Tree:
        tokens = self.lex(text)
        lines = []
        line_tokens: List[Token] = []
        for t in tokens:
            if t.type == "NEWLINE":
                if line_tokens:
                    node = self._parse_line(line_tokens)
                    if node is not None:
                        lines.append(node)
                    line_tokens = []
                else:
                    line_tokens = []
            else:
                line_tokens.append(t)
        if line_tokens:
            node = self._parse_line(line_tokens)
            if node is not None:
                lines.append(node)
        return Tree("start", lines)

    def _parse_line(self, tokens: List[Token]) -> Tree | None:
        if not tokens:
            return None
        i = 0
        children: List[Any] = []
        # label
        if i + 1 < len(tokens) and tokens[i].type == "CNAME" and tokens[i + 1].type == "COLON":
            children.append(Tree("label", [tokens[i]]))
            i += 2
            while i < len(tokens) and tokens[i].type == "WS_INLINE":
                i += 1
        if i >= len(tokens):
            return Tree("line", children)
        tok = tokens[i]
        if tok.type == "SECTION":
            sec_children: List[Any] = [tok]
            i += 1
            if i < len(tokens) and tokens[i].type == "WS_INLINE":
                sec_children.append(tokens[i])
                i += 1
            if i < len(tokens) and tokens[i].type == "CNAME":
                sec_children.append(tokens[i])
                i += 1
            while i < len(tokens) and tokens[i].type == "WS_INLINE":
                sec_children.append(tokens[i])
                i += 1
            children.append(Tree("section_decl", sec_children))
        elif tok.type in {"DEFB", "DEFW", "DEFL", "DEFS", "DEFM"}:
            dir_name = tok.type.lower()
            dir_children: List[Any] = [tok]
            i += 1
            if i < len(tokens) and tokens[i].type == "WS_INLINE":
                dir_children.append(tokens[i])
                i += 1
            if dir_name == "defs":
                if i < len(tokens) and tokens[i].type == "NUMBER":
                    dir_children.append(tokens[i])
                    i += 1
            elif dir_name == "defm":
                if i < len(tokens) and tokens[i].type == "ESCAPED_STRING":
                    dir_children.append(Tree("string_literal", [tokens[i]]))
                    i += 1
            else:
                while i < len(tokens) and tokens[i].type == "NUMBER":
                    dir_children.append(tokens[i])
                    i += 1
                    if i < len(tokens) and tokens[i].type == "COMMA":
                        dir_children.append(tokens[i])
                        i += 1
                        if i < len(tokens) and tokens[i].type == "WS_INLINE":
                            dir_children.append(tokens[i])
                            i += 1
                    else:
                        break
            children.append(Tree(f"{dir_name}_directive", dir_children))
        else:
            children.append(Tree("instruction", tokens[i:]))
        return Tree("line", children)

