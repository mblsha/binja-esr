"""
Parity sweep runner for LLAMA vs Python backend.

This iterates the opcode encodings emitted by ``opcode_generator`` and executes
each instruction once on both the Python emulator and the LLAMA backend,
comparing register snapshots and memory writes. It is intentionally lenient:
exceptions or mismatches are collected and summarised instead of aborting the
run. Use `--limit` to restrict the number of cases while bringing LLAMA up to
parity.

Run with:

    FORCE_BINJA_MOCK=1 uv run python tools/llama_parity_sweep.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, List, Tuple

from binja_test_mocks.eval_llil import Memory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sc62015.pysc62015 import CPU, RegisterName, available_backends
from sc62015.pysc62015.constants import ADDRESS_SPACE_SIZE
from sc62015.pysc62015.stepper import CPURegistersSnapshot
from sc62015.pysc62015.test_instr import opcode_generator


class LoggingMemory(Memory):
    """Memory adapter that records writes and serves a flat address space."""

    def __init__(self, backing: bytearray) -> None:
        self._backing = backing
        self.writes: list[tuple[int, int]] = []
        super().__init__(self._read_byte, self._write_byte)

    def _read_byte(self, address: int) -> int:
        address &= 0xFFFFFF
        if address < 0 or address >= len(self._backing):
            raise IndexError(f"Read address {address:#x} out of bounds")
        return self._backing[address]

    def _write_byte(self, address: int, value: int) -> None:
        address &= 0xFFFFFF
        if address < 0 or address >= len(self._backing):
            raise IndexError(f"Write address {address:#x} out of bounds")
        self._backing[address] = value & 0xFF
        self.writes.append((address, value & 0xFF))

    def snapshot(self) -> dict[int, int]:
        return {idx: byte for idx, byte in enumerate(self._backing) if byte}


@dataclass
class ParityResult:
    opcode: int
    bytes_hex: str
    reg_diff: dict[str, tuple[int, int]]
    writes_diff: tuple[Tuple[int, int], Tuple[int, int]]
    python_error: str | None = None
    llama_error: str | None = None


def _make_memory(instr_bytes: bytes, pc: int) -> LoggingMemory:
    backing = bytearray(ADDRESS_SPACE_SIZE)
    backing[pc : pc + len(instr_bytes)] = instr_bytes
    return LoggingMemory(backing)


def _snapshot_registers(cpu: CPU) -> dict[str, int]:
    snap = cpu.snapshot_registers()
    return snap.to_dict()


def _compare_writes(
    lhs: list[tuple[int, int]], rhs: list[tuple[int, int]]
) -> tuple[Tuple[int, int], Tuple[int, int]] | None:
    if lhs == rhs:
        return None
    # Compare last write per address
    lmap: dict[int, int] = {addr: val for addr, val in lhs}
    rmap: dict[int, int] = {addr: val for addr, val in rhs}
    if lmap == rmap:
        return None
    return ((-1, -1), (-1, -1))  # placeholder; detailed diffs logged via reg_diff


def run_case(instr_bytes: bytes, pc: int) -> ParityResult | None:
    reg_init = CPURegistersSnapshot(pc=pc)

    # Python backend
    mem_py = _make_memory(instr_bytes, pc)
    cpu_py = CPU(mem_py, reset_on_init=False, backend="python")
    cpu_py.apply_snapshot(reg_init)
    py_err = None
    try:
        cpu_py.execute_instruction(pc)
        regs_py = _snapshot_registers(cpu_py)
    except Exception as exc:  # pragma: no cover - defensive
        py_err = f"{type(exc).__name__}: {exc}"
        regs_py = {}

    # LLAMA backend
    mem_ll = _make_memory(instr_bytes, pc)
    cpu_ll = CPU(mem_ll, reset_on_init=False, backend="llama")
    cpu_ll.apply_snapshot(reg_init)
    ll_err = None
    try:
        cpu_ll.execute_instruction(pc)
        regs_ll = _snapshot_registers(cpu_ll)
    except Exception as exc:  # pragma: no cover - defensive
        ll_err = f"{type(exc).__name__}: {exc}"
        regs_ll = {}

    opcode = instr_bytes[0]
    if py_err or ll_err:
        return ParityResult(
            opcode=opcode,
            bytes_hex=instr_bytes.hex(),
            reg_diff={},
            writes_diff=(tuple(), tuple()),
            python_error=py_err,
            llama_error=ll_err,
        )

    if regs_py != regs_ll or mem_py.writes != mem_ll.writes:
        reg_diff: dict[str, tuple[int, int]] = {}
        keys = set(regs_py) | set(regs_ll)
        for key in sorted(keys):
            lp = regs_py.get(key, 0)
            rp = regs_ll.get(key, 0)
            if lp != rp:
                reg_diff[key] = (lp, rp)
        writes_diff = _compare_writes(mem_py.writes, mem_ll.writes)
        return ParityResult(
            opcode=opcode,
            bytes_hex=instr_bytes.hex(),
            reg_diff=reg_diff,
            writes_diff=writes_diff if writes_diff else (tuple(), tuple()),
        )

    return None


def sweep(limit: int | None) -> list[ParityResult]:
    failures: list[ParityResult] = []
    for idx, encoding in enumerate(opcode_generator()):
        if limit is not None and idx >= limit:
            break
        if isinstance(encoding, tuple):
            raw = encoding[0]
        else:
            raw = encoding
        if raw is None:
            continue
        instr_bytes = bytes(raw)
        result = run_case(instr_bytes, pc=0)
        if result is not None:
            failures.append(result)
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="LLAMA/Python parity sweep")
    parser.add_argument("--limit", type=int, default=None, help="limit cases")
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit JSON report instead of human-readable summary",
    )
    args = parser.parse_args()

    if "llama" not in available_backends():
        raise SystemExit("LLAMA backend not available; build rustcore first.")

    failures = sweep(args.limit)

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "opcode": f"0x{f.opcode:02X}",
                        "bytes": f.bytes_hex,
                        "reg_diff": f.reg_diff,
                        "writes_diff": f.writes_diff,
                        "python_error": f.python_error,
                        "llama_error": f.llama_error,
                    }
                    for f in failures
                ],
                indent=2,
            )
        )
    else:
        if not failures:
            print("Parity sweep passed (no mismatches).")
        else:
            print(f"Found {len(failures)} mismatches:")
            for f in failures:
                print(
                    f"- opcode 0x{f.opcode:02X} bytes={f.bytes_hex} "
                    f"py_err={f.python_error} llama_err={f.llama_error} "
                    f"reg_diff={f.reg_diff} writes_diff={f.writes_diff}"
                )

    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
