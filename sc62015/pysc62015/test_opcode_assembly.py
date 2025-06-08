import re
import pytest

from .sc_asm import Assembler, AssemblerError
from .test_instr import opcode_generator

REGS = {"A","B","IL","IH","I","BA","X","Y","U","S","PC","IMR","F","FC","FZ"}
PRE_PREFIXES = set(range(0x20, 0x38))
PATTERN = re.compile(r"(?<![A-Za-z0-9])([+-]?)([A-F0-9]+)(?![A-Za-z0-9])")


def _transform(instr: str) -> str:
    instr = instr.replace("(FB)", "IMR")
    parts = instr.split(maxsplit=1)
    mnemonic = parts[0]
    rest = parts[1] if len(parts) > 1 else ""

    def repl(match: re.Match[str]) -> str:
        sign, token = match.groups()
        if token.isalpha():
            if token in REGS:
                return match.group(0)
            if token == "FCDAB":
                return f"{sign}0xEFCDAB"
        return f"{sign}0x{token}"

    rest = PATTERN.sub(repl, rest)
    return f"{mnemonic} {rest}".strip()


def _strip_pre(data: bytes) -> bytes:
    index = 0
    while index < len(data) and data[index] in PRE_PREFIXES:
        index += 1
    return data[index:]


@pytest.mark.xfail(reason="Assembler and opcode table currently diverge")
def test_opcode_table_roundtrip() -> None:
    assembler = Assembler()
    mismatches = []

    for idx, (expected_bytes, asm_text) in enumerate(opcode_generator()):
        if expected_bytes is None or asm_text is None:
            continue
        if asm_text.startswith("PRE") or asm_text.startswith("???"):
            continue

        source = _transform(asm_text)
        try:
            output = assembler.assemble(source).as_binary()
        except AssemblerError as exc:
            mismatches.append(f"Line {idx+1}: failed to assemble '{asm_text}': {exc}")
            continue

        if _strip_pre(output) != _strip_pre(expected_bytes):
            mismatches.append(
                f"Line {idx+1}: {asm_text} -> {output.hex()} expected {expected_bytes.hex()}"
            )

    if mismatches:
        pytest.fail("\n".join(mismatches))
