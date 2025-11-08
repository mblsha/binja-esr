#!/usr/bin/env python3
"""Cross-check SC62015 CPU backends by executing opcodes or short programs."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from binja_test_mocks.eval_llil import Memory

from sc62015.pysc62015 import CPU, RegisterName, available_backends


def _construct_opcode_memory(opcode: int) -> Tuple[Memory, bytearray]:
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


@dataclass
class ProgramExecutionResult:
    backend: str
    registers: Dict[str, int]
    memory: bytes
    halted: bool
    steps: int
    last_instruction: str
    trace: List[Tuple[int, int, str]]


def run_opcode(backend: str, opcode: int) -> ExecutionResult:
    memory, raw = _construct_opcode_memory(opcode)

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
        differences.append(
            f"pc mismatch: python=0x{python_result.pc:05X}, rust=0x{rust_result.pc:05X}"
        )
    if python_result.registers != rust_result.registers:
        differences.append("register snapshot mismatch")

    return python_result, rust_result, differences


def _load_program(path: Path) -> bytearray:
    data = path.read_bytes()
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError:
        return bytearray(data)
    cleaned = "".join(ch for ch in text if ch.strip())
    if cleaned and all(ch in "0123456789abcdefABCDEF" for ch in cleaned):
        if len(cleaned) % 2 != 0:
            cleaned = cleaned[:-1]
        return bytearray(int(cleaned[i : i + 2], 16) for i in range(0, len(cleaned), 2))
    return bytearray(data)


def _construct_program_memory(program: Sequence[int]) -> Tuple[Memory, bytearray]:
    raw = bytearray(program)
    if len(raw) < 0x200:
        raw.extend(b"\x00" * (0x200 - len(raw)))

    def read(addr: int) -> int:
        if addr < 0 or addr >= len(raw):
            raise IndexError(f"Read address {addr:#x} out of bounds")
        return raw[addr]

    def write(addr: int, value: int) -> None:
        if addr < 0 or addr >= len(raw):
            raise IndexError(f"Write address {addr:#x} out of bounds")
        raw[addr] = value & 0xFF

    return Memory(read, write), raw


def run_program(
    backend: str,
    program: Sequence[int],
    pc: int,
    max_steps: int,
    snapshot_len: int,
    capture_trace: bool,
) -> ProgramExecutionResult:
    memory, raw = _construct_program_memory(program)
    cpu = CPU(memory, reset_on_init=False, backend=backend)
    cpu.regs.set(RegisterName.PC, pc)
    steps = 0
    last_instruction = ""
    trace_log: List[Tuple[int, int, str]] = []
    while steps < max_steps:
        current_pc = cpu.regs.get(RegisterName.PC)
        info = cpu.execute_instruction(current_pc)
        steps += 1
        last_instruction = info.instruction.name()
        if capture_trace:
            trace_log.append((steps, current_pc, last_instruction))
        if getattr(cpu.state, "halted", False):
            break
    registers = _snapshot_registers(cpu)
    halted = getattr(cpu.state, "halted", False)
    return ProgramExecutionResult(
        backend=backend,
        registers=registers,
        memory=bytes(raw[:snapshot_len]),
        halted=halted,
        steps=steps,
        last_instruction=last_instruction,
        trace=trace_log,
    )


def compare_program(
    program: Sequence[int],
    pc: int,
    max_steps: int,
    snapshot_len: int,
    capture_trace: bool,
) -> Tuple[ProgramExecutionResult, ProgramExecutionResult, List[str]]:
    python_result = run_program("python", program, pc, max_steps, snapshot_len, capture_trace)
    rust_result = run_program("rust", program, pc, max_steps, snapshot_len, capture_trace)

    differences: List[str] = []
    if python_result.registers != rust_result.registers:
        differences.append("register snapshot mismatch")
    if python_result.memory != rust_result.memory:
        differences.append("memory snapshot mismatch")
    if python_result.halted != rust_result.halted:
        differences.append(
            f"halted mismatch: python={python_result.halted} rust={rust_result.halted}"
        )
    if python_result.steps != rust_result.steps:
        differences.append(
            f"step count mismatch: python={python_result.steps} rust={rust_result.steps}"
        )
    if python_result.last_instruction != rust_result.last_instruction:
        differences.append(
            "last instruction mismatch: "
            f"python={python_result.last_instruction} rust={rust_result.last_instruction}"
        )
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
    parser.add_argument(
        "--program",
        type=Path,
        help="Optional binary/hex program file to execute instead of opcode sweep.",
    )
    parser.add_argument(
        "--pc",
        type=lambda v: int(v, 0),
        default=0,
        help="Initial PC for program execution mode (default: 0).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=32,
        help="Maximum instructions to execute for program mode (default: 32).",
    )
    parser.add_argument(
        "--snapshot-len",
        type=int,
        default=64,
        help="Bytes of memory to compare/dump in program mode (default: 64).",
    )
    parser.add_argument(
        "--dump-on-fail",
        action="store_true",
        help="Dump register and memory state when a mismatch is detected.",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Emit instruction traces for program mode.",
    )
    return parser.parse_args(list(argv))


def _dump_registers(result: ProgramExecutionResult, prefix: str) -> None:
    print(f"{prefix} registers:")
    for name in sorted(result.registers):
        value = result.registers[name]
        print(f"  {name:>4}: 0x{value:06X}" if value is not None else f"  {name:>4}: <n/a>")
    print(f"  halted: {result.halted}  steps: {result.steps} last={result.last_instruction}")


def _dump_memory(result: ProgramExecutionResult, prefix: str) -> None:
    print(f"{prefix} memory (first {len(result.memory)} bytes):")
    for offset in range(0, len(result.memory), 16):
        chunk = result.memory[offset : offset + 16]
        hex_bytes = " ".join(f"{byte:02X}" for byte in chunk)
        print(f"  0x{offset:04X}: {hex_bytes}")


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

    if args.program:
        program = _load_program(args.program)
        try:
            python_result, rust_result, diff = compare_program(
                program,
                pc=args.pc,
                max_steps=args.max_steps,
                snapshot_len=args.snapshot_len,
                capture_trace=args.trace,
            )
        except RuntimeError as exc:
            print(f"[ERROR] program execution failed: {exc}", file=sys.stderr)
            return 1

        if diff:
            print(f"[FAIL] program mismatch: {'; '.join(diff)}")
            if args.dump_on_fail:
                _dump_registers(python_result, "[python]")
                _dump_memory(python_result, "[python]")
                _dump_registers(rust_result, "[rust]")
                _dump_memory(rust_result, "[rust]")
            return 1

        print(
            f"[ OK ] program matched after {python_result.steps} steps (last={python_result.last_instruction})"
        )
        if args.trace:
            for backend_result in (python_result, rust_result):
                print(f"Trace ({backend_result.backend}):")
                for idx, pc, name in backend_result.trace:
                    print(f"  step {idx:02d} pc=0x{pc:05X} {name}")
        return 0

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
