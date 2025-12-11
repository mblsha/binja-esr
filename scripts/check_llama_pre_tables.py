"""Check LLAMA PRE mode and single-addressable opcode tables against Python sources.

Run with:
    uv run python scripts/check_llama_pre_tables.py

Exits non-zero on mismatch.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sc62015.pysc62015.instr.opcodes import (  # type: ignore  # noqa: E402
    AddressingMode,
    SINGLE_ADDRESSABLE_OPCODES as PY_SINGLE,
    opcode_for_modes,
)

RUST_EVAL = ROOT / "sc62015" / "core" / "src" / "llama" / "eval.rs"


@dataclass(frozen=True)
class PreEntry:
    opcode: int
    first: AddressingMode
    second: AddressingMode


def parse_rust_pre_modes(text: str) -> List[PreEntry]:
    """Parse PRE_MODES tuples from the Rust eval.rs source."""
    enum_map: Dict[str, AddressingMode] = {
        "N": AddressingMode.N,
        "BpN": AddressingMode.BP_N,
        "PxN": AddressingMode.PX_N,
        "PyN": AddressingMode.PY_N,
        "BpPx": AddressingMode.BP_PX,
        "BpPy": AddressingMode.BP_PY,
    }
    pre_block = re.search(
        r"const PRE_MODES: &\[\(u8, AddressingMode, AddressingMode\)\] = &\[(.*?)\];",
        text,
        re.S,
    )
    if not pre_block:
        raise SystemExit("Could not find PRE_MODES in eval.rs")
    entries: List[PreEntry] = []
    for line in pre_block.group(1).splitlines():
        m = re.search(
            r"\((0x[0-9A-Fa-f]+),\s*AddressingMode::(\w+),\s*AddressingMode::(\w+)\)",
            line,
        )
        if not m:
            continue
        opcode = int(m.group(1), 16)
        first = enum_map[m.group(2)]
        second = enum_map[m.group(3)]
        entries.append(PreEntry(opcode, first, second))
    return entries


def parse_rust_single(text: str) -> List[int]:
    block = re.search(
        r"const SINGLE_ADDRESSABLE_OPCODES: &\[u8\] = &\[(.*?)\];", text, re.S
    )
    if not block:
        raise SystemExit("Could not find SINGLE_ADDRESSABLE_OPCODES in eval.rs")
    opcodes: List[int] = []
    for token in block.group(1).replace("\n", " ").split(","):
        token = token.strip()
        if not token:
            continue
        opcodes.append(int(token, 16))
    return opcodes


def python_pre_modes() -> List[PreEntry]:
    entries: List[PreEntry] = []
    for a in (
        AddressingMode.N,
        AddressingMode.BP_N,
        AddressingMode.PX_N,
        AddressingMode.PY_N,
        AddressingMode.BP_PX,
        AddressingMode.BP_PY,
    ):
        for b in (
            AddressingMode.N,
            AddressingMode.BP_N,
            AddressingMode.PX_N,
            AddressingMode.PY_N,
            AddressingMode.BP_PX,
            AddressingMode.BP_PY,
        ):
            op = opcode_for_modes(a, b)
            if op is not None:
                entries.append(PreEntry(op, a, b))
    return entries


def diff_sets(label: str, rust: Iterable[int], py: Iterable[int]) -> int:
    rust_set = set(rust)
    py_set = set(py)
    missing = py_set - rust_set
    extra = rust_set - py_set
    if not missing and not extra:
        print(f"{label}: OK ({len(rust_set)})")
        return 0
    if missing:
        print(f"{label}: missing in Rust -> {[hex(x) for x in sorted(missing)]}")
    if extra:
        print(f"{label}: extra in Rust -> {[hex(x) for x in sorted(extra)]}")
    return 1


def main() -> int:
    text = RUST_EVAL.read_text()
    rust_pre = parse_rust_pre_modes(text)
    rust_single = parse_rust_single(text)
    py_pre = python_pre_modes()
    py_single = list(PY_SINGLE)

    status = 0
    # Compare PRE opcode coverage only (mode pairs are validated by grouping tuples per opcode).
    status |= diff_sets(
        "PRE opcodes", (e.opcode for e in rust_pre), (e.opcode for e in py_pre)
    )
    # Compare full mode pairs per opcode.
    rust_pairs: Dict[int, List[Tuple[AddressingMode, AddressingMode]]] = {}
    for e in rust_pre:
        rust_pairs.setdefault(e.opcode, []).append((e.first, e.second))
    py_pairs: Dict[int, List[Tuple[AddressingMode, AddressingMode]]] = {}
    for e in py_pre:
        py_pairs.setdefault(e.opcode, []).append((e.first, e.second))
    for opcode in sorted(set(rust_pairs) | set(py_pairs)):
        r_modes = set(rust_pairs.get(opcode, []))
        p_modes = set(py_pairs.get(opcode, []))
        if r_modes != p_modes:
            print(
                f"PRE mode mismatch for 0x{opcode:02X}: rust={sorted(r_modes)} python={sorted(p_modes)}"
            )
            status = 1
    status |= diff_sets("SINGLE_ADDRESSABLE_OPCODES", rust_single, py_single)
    return status


if __name__ == "__main__":
    sys.exit(main())
