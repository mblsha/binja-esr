import pytest
from .asm import asm_parser, AsmTransformer
from .instr import Instruction
from .instr import (
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
from lark import Tree, Token
from typing import Any, Callable, List, Union, Tuple, Type


def get_token_tuples(text: str) -> List[Tuple[str, str]]:
    # Returns a list of (type, value) for easy assertion
    return [(token.type, token.value) for token in asm_parser.lex(text)]


def find_all_tokens_by_type(
    node: Union[Tree[Any], Token], target_type: str
) -> List[Token]:
    """Recursively find all Tokens of a specific type within a Tree or from a Token list."""
    found_tokens: List[Token] = []
    if isinstance(node, Token):
        if node.type == target_type:
            found_tokens.append(node)
    elif isinstance(node, Tree):
        for child in node.children:
            found_tokens.extend(find_all_tokens_by_type(child, target_type))
    return found_tokens


def test_label_tokenization() -> None:
    tokens = get_token_tuples("mylabel:")
    assert tokens == [("CNAME", "mylabel"), ("COLON", ":")]


def test_section_decl_tokenization() -> None:
    tokens = get_token_tuples("SECTION data")
    # "SECTION" is a literal, "data" is CNAME
    assert tokens == [("SECTION", "SECTION"), ("CNAME", "data")]


def test_defb_tokenization() -> None:
    tokens = get_token_tuples("defb 1,2,0xFF")
    assert tokens == [
        ("DEFB", "defb"),
        ("NUMBER", "1"),
        ("COMMA", ","),
        ("NUMBER", "2"),
        ("COMMA", ","),
        ("NUMBER", "0xFF"),
    ]


def test_instruction_tokenization() -> None:
    tokens = get_token_tuples("NOP")
    assert tokens == [("NOP", "NOP")]


def test_string_literal_tokenization() -> None:
    tokens = get_token_tuples('defm "Hello"')
    assert tokens == [
        ("DEFM", "defm"),
        ("ESCAPED_STRING", '"Hello"'),
    ]


def test_comment_ignored() -> None:
    tokens = get_token_tuples("; this is a comment\n")
    # COMMENT is ignored by the grammar, so only NEWLINE
    assert tokens == [("NEWLINE", "\n")]


def test_mixed_line_tokenization() -> None:
    tokens = get_token_tuples("label1: defb 1, 2 ; comment\n")
    assert ("CNAME", "label1") in tokens


def find_pred(tree: Tree[Any], pred: Callable[[Tree[Any]], bool]) -> List[Tree[Any]]:
    "Recursively collect all subtrees matching predicate."
    found = []
    if isinstance(tree, Tree) and pred(tree):
        found.append(tree)
    for child in tree.children:
        if isinstance(child, Tree):
            found.extend(find_pred(child, pred))
    return found


def pretty_print_tree(tree: Union[Tree[Any], Token], indent: int = 0) -> None:
    """Recursively pretty-print a Lark parse tree or token, with indentation.

    Args:
        tree: The parse tree (Tree or Token).
        indent: Current indentation level (default 0).
    """
    prefix = "    " * indent
    if isinstance(tree, Tree):
        print(f"{prefix}{tree.data}")
        for child in tree.children:
            pretty_print_tree(child, indent + 1)
    elif isinstance(tree, Token):
        print(f"{prefix}{tree.type}: {tree.value}")


def test_parse_section_decl() -> None:
    src = "SECTION data\n"
    tree: Tree[Any] = asm_parser.parse(src)
    assert tree.data == "start"
    # Find section_decl anywhere in tree
    section_decls = find_pred(tree, lambda t: t.data == "section_decl")
    assert section_decls
    sec = section_decls[0]
    assert any(
        isinstance(ch, Token) and ch.type == "CNAME" and ch == "data"
        for ch in sec.children
    )


def test_parse_label_and_defb() -> None:
    src = "label1: defb 1, 2, 3\n"
    tree: Tree[Any] = asm_parser.parse(src)
    labels = find_pred(tree, lambda t: t.data == "label")
    assert labels
    label = labels[0]
    assert label.children and isinstance(label.children[0], Token)
    assert label.children[0] == "label1"

    defbs = find_pred(tree, lambda t: t.data == "defb_directive")
    assert defbs
    defb = defbs[0]

    nums_tokens = find_all_tokens_by_type(defb, "NUMBER")
    expected_values = ["1", "2", "3"]
    assert [t.value for t in nums_tokens] == expected_values


def test_parse_defs_and_defm() -> None:
    src = 'defs 42\ndefm "Hello, World"\n'
    tree: Tree[Any] = asm_parser.parse(src)
    defs = find_pred(tree, lambda t: t.data == "defs_directive")
    assert defs
    defnode = defs[0]
    num = [c for c in defnode.children if isinstance(c, Token) and c.type == "NUMBER"]
    assert num and num[0] == "42"
    defms = find_pred(tree, lambda t: t.data == "defm_directive")
    assert defms
    defm = defms[0]
    string_lits = find_pred(defm, lambda t: t.data == "string_literal")
    assert string_lits
    lit = string_lits[0]
    assert lit.children and isinstance(lit.children[0], Token)
    assert lit.children[0] == '"Hello, World"'


def test_parse_comment_and_newline() -> None:
    src = "; just a comment\n\n"
    tree: Tree[Any] = asm_parser.parse(src)
    assert tree.data == "start"
    # The grammar likely collapses multiple newlines between statements.
    # The error message shows the parser produces one empty line node.
    assert len(tree.children) == 1
    assert all(child.data == "line" for child in tree.children)


@pytest.mark.parametrize(
    "instr_str, instr_class",
    [
        ("NOP", NOP),
        ("RETI", RETI),
        ("RET", RET),
        ("RETF", RETF),
        ("SC", SC),
        ("RC", RC),
        ("TCL", TCL),
        ("HALT", HALT),
        ("OFF", OFF),
        ("WAIT", WAIT),
        ("IR", IR),
        ("RESET", RESET),
    ],
)
def test_parse_zero_operand_instructions(
    instr_str: str, instr_class: Type[Instruction]
) -> None:
    tree = asm_parser.parse(f"{instr_str}\n")
    result = AsmTransformer().transform(tree)
    stmt = result["lines"][0]["statement"]
    parsed_instr = stmt["instruction"]
    assert parsed_instr["instr_class"] is instr_class
    assert not parsed_instr["instr_opts"].ops


@pytest.mark.parametrize(
    "instr_str, instr_class",
    [("SWAP A", SWAP), ("ROR A", ROR), ("ROL A", ROL), ("SHR A", SHR), ("SHL A", SHL)],
)
def test_parse_single_a_operand_instructions(
    instr_str: str, instr_class: Type[Instruction]
) -> None:
    tree = asm_parser.parse(f"{instr_str}\n")
    result = AsmTransformer().transform(tree)
    stmt = result["lines"][0]["statement"]
    parsed_instr = stmt["instruction"]
    assert parsed_instr["instr_class"] is instr_class
    ops = parsed_instr["instr_opts"].ops
    assert len(ops) == 1
    assert isinstance(ops[0], Reg) and ops[0].reg == "A"


def test_parse_mv_reg_reg() -> None:
    tree = asm_parser.parse("MV A, B\n")
    result = AsmTransformer().transform(tree)
    stmt = result["lines"][0]["statement"]
    parsed_instr = stmt["instruction"]
    assert parsed_instr["instr_class"] is MV
    ops = parsed_instr["instr_opts"].ops
    assert len(ops) == 2
    assert isinstance(ops[0], Reg) and ops[0].reg == "A"
    assert isinstance(ops[1], RegB)

    tree = asm_parser.parse("MV B, A\n")
    result = AsmTransformer().transform(tree)
    stmt = result["lines"][0]["statement"]
    parsed_instr = stmt["instruction"]
    assert parsed_instr["instr_class"] is MV
    ops = parsed_instr["instr_opts"].ops
    assert len(ops) == 2
    assert isinstance(ops[0], RegB)
    assert isinstance(ops[1], Reg) and ops[1].reg == "A"


def test_parse_ex_a_b() -> None:
    tree = asm_parser.parse("EX A, B\n")
    result = AsmTransformer().transform(tree)
    stmt = result["lines"][0]["statement"]
    parsed_instr = stmt["instruction"]
    assert parsed_instr["instr_class"] is EX
    ops = parsed_instr["instr_opts"].ops
    assert len(ops) == 2
    assert isinstance(ops[0], Reg) and ops[0].reg == "A"
    assert isinstance(ops[1], RegB)


@pytest.mark.parametrize(
    "instr_str, instr_class, reg_class",
    [
        ("PUSHS F", PUSHS, RegF),
        ("POPS F", POPS, RegF),
        ("PUSHU F", PUSHU, RegF),
        ("POPU F", POPU, RegF),
        ("PUSHU IMR", PUSHU, RegIMR),
        ("POPU IMR", POPU, RegIMR),
    ],
)
def test_parse_stack_f_imr_instructions(
    instr_str: str, instr_class: Type[Instruction], reg_class: Type[Reg]
) -> None:
    tree = asm_parser.parse(f"{instr_str}\n")
    result = AsmTransformer().transform(tree)
    stmt = result["lines"][0]["statement"]
    parsed_instr = stmt["instruction"]
    assert parsed_instr["instr_class"] is instr_class
    ops = parsed_instr["instr_opts"].ops
    assert len(ops) == 1
    assert isinstance(ops[0], reg_class)
