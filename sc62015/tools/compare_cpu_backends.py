#!/usr/bin/env python3
"""Cross-check SC62015 CPU backends by executing individual opcodes."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from binja_test_mocks.eval_llil import Memory

from sc62015.pysc62015 import CPU, RegisterName, available_backends


def _construct_memory(opcode: int) -> Tuple[Memory, bytearray]:
    raw = bytearray([opcode]) + bytearray(8)

    def read(addr: int) -> int:
        if addr < 0 or addr >= len(raw):
            raise IndexError(f"Read address {addr:#x} out of bounds")
        return raw[addr]

    def write(addr: int, value: int) -> None:
        if addr < 0 or addr >= len(raw):
            raise IndexError(f"Write address {addr:#x} out of bounds")
        raw[addr] = value & 0xFF

    return Memory(read, write), raw


def _snapshot_registers(cpu) -> Dict[str, int]:
    snapshot: Dict[str, int] = {}
    for reg in RegisterName:
        try:
            snapshot[reg.name] = cpu.regs.get(reg)
        except Exception:  # pragma: no cover - backend divergence
            snapshot[reg.name] = None
    snapshot["call_sub_level"] = getattr(cpu.regs, "call_sub_level", None)
    return snapshot


@dataclass
class ExecutionResult:
    backend: str
    opcode: int
    registers: Dict[str, int]
    pc: int
    instruction: str


def run_opcode(backend: str, opcode: int) -> ExecutionResult:
    memory, raw = _construct_memory(opcode)

    cpu = CPU(memory, reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.PC, 0)
    info = cpu.execute_instruction(0)

    return ExecutionResult(
        backend=backend,
        opcode=opcode,
        registers=_snapshot_registers(cpu),
        pc=cpu.regs.get(RegisterName.PC),
        instruction=info.instruction.name(),
    )


def compare_opcode(opcode: int) -> Tuple[ExecutionResult, ExecutionResult, List[str]]:
    python_result = run_opcode("python", opcode)
    rust_result = run_opcode("rust", opcode)

    differences: List[str] = []
    if python_result.instruction != rust_result.instruction:
        differences.append(
            f"instruction mismatch: python={python_result.instruction}, rust={rust_result.instruction}"
        )
    if python_result.pc != rust_result.pc:
        differences.append(f"pc mismatch: python=0x{python_result.pc:05X}, rust=0x{rust_result.pc:05X}")
    if python_result.registers != rust_result.registers:
        differences.append("register snapshot mismatch")

    return python_result, rust_result, differences


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start",
        type=lambda v: int(v, 0),
        default=0x00,
        help="First opcode to compare (inclusive, default: 0x00).",
    )
    parser.add_argument(
        "--end",
        type=lambda v: int(v, 0),
        default=0xFF,
        help="Last opcode to compare (inclusive, default: 0xFF).",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    backends = set(available_backends())
    if "rust" not in backends:
        print(
            "Rust backend is unavailable; build it with "
            "`uv run maturin develop --manifest-path sc62015/rustcore/Cargo.toml` to enable comparisons.",
            file=sys.stderr,
        )
        return 2

    failures = 0
    for opcode in range(args.start, args.end + 1):
        try:
            python_result, rust_result, diff = compare_opcode(opcode)
        except NotImplementedError:
            print(f"[SKIP] opcode 0x{opcode:02X}: Rust backend not implemented yet")
            continue
        except RuntimeError as exc:
            print(f"[ERROR] opcode 0x{opcode:02X}: {exc}", file=sys.stderr)
            failures += 1
            continue

        if diff:
            failures += 1
            print(f"[FAIL] opcode 0x{opcode:02X}: {'; '.join(diff)}")
        else:
            print(f"[ OK ] opcode 0x{opcode:02X}: {python_result.instruction}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
