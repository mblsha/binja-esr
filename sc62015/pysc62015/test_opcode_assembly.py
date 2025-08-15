import re
import pytest

from .sc_asm import Assembler
from .test_instr import opcode_generator, decode

REGS = {
    "A",
    "B",
    "IL",
    "IH",
    "I",
    "BA",
    "X",
    "Y",
    "U",
    "S",
    "PC",
    "IMR",
    "F",
    "FC",
    "FZ",
}
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
            # ``token`` may look like a register name but occasionally represents
            # a numeric constant inside addressing modes, e.g. ``[(BA)]`` or
            # ``[(CD)+BA]``. When preceded by "(", "+" or "-" treat it as a
            # constant rather than a register. ``match.start()`` points to the
            # beginning of the whole match, which may include the ``+`` or ``-``
            # sign. ``token_start`` tracks the actual token location so we can
            # inspect the character immediately before it.
            start = match.start()
            token_start = start + len(sign)
            prev = rest[token_start - 1] if token_start > 0 else ""
            if token in REGS and prev not in {"(", "+", "-"}:
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


def test_opcode_table_roundtrip() -> None:
    assembler = Assembler()
    mismatches = []

    seen: set[str] = set()
    for idx, (expected_bytes, asm_text) in enumerate(opcode_generator()):
        if expected_bytes is None or asm_text is None:
            continue
        if asm_text.startswith("PRE") or asm_text.startswith("???"):
            continue
        if asm_text in seen:
            continue
        seen.add(asm_text)

        source = _transform(asm_text)
        try:
            output = assembler.assemble(source).as_binary()
        except Exception as e:
            text = str(e).replace("\n", " ")
            mismatches.append(f"Line {idx + 1}: {asm_text} -> Exception: {text}")
            continue

        if _strip_pre(output) != _strip_pre(expected_bytes):
            try:
                expected_instr = decode(expected_bytes, 0)
                output_instr = decode(output, 0)
            except Exception:
                mismatches.append(
                    f"Line {idx + 1}: {asm_text} -> {output.hex()} expected {expected_bytes.hex()}"
                )
                continue

            if expected_instr.render() != output_instr.render():
                mismatches.append(
                    f"Line {idx + 1}: {asm_text} -> {output.hex()} expected {expected_bytes.hex()}"
                )

    if mismatches:
        print("\n".join(mismatches))
        pytest.fail(
            f"Opcode table divergence detected. {len(mismatches)} mismatches found."
        )
