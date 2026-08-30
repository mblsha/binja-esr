# ruff: noqa: E402
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
import copy
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import sys
from typing import Iterable

from binja_test_mocks.eval_llil import Memory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sc62015.pysc62015 import CPU, available_backends
from sc62015.pysc62015.constants import ADDRESS_SPACE_SIZE, INTERNAL_MEMORY_START
from sc62015.pysc62015.stepper import CPURegistersSnapshot
from sc62015.pysc62015.test_instr import opcode_generator, decode as decode_instr
from sc62015.pysc62015.instr.opcodes import (
    Imm8,
    Imm16,
    Imm20,
    ImmOffset,
    IMEMRegisters,
    Instruction,
    encode as encode_instr,
    Operand,
)

STACK_SEED = 0x80000
# These opcodes are supposed to reject on both backends.  0x20/0xBF are
# reserved encodings; TCL (0xCE) is deliberately quarantined until its timer
# side effects are implemented from hardware evidence.
EXPECTED_REJECTION_OPCODES = frozenset(
    {0x20, 0xBF, 0xCE, *range(0x21, 0x28), *range(0x30, 0x38)}
)
FAIL_CLOSED_CONTRACT_MARKERS = (
    "SC62015 I=0 counted-instruction semantics require real-hardware tracing",
    "TCL timer-clear side effects are not implemented",
)


class LoggingMemory(Memory):
    """Memory adapter that records writes and serves a flat address space."""

    def __init__(self, backing: bytearray) -> None:
        self._backing = backing
        self.writes: list[tuple[int, int]] = []
        self.waits: list[int] = []
        super().__init__(self._read_byte, self._write_byte)

    def wait_cycles(self, cycles: int) -> None:
        """Record WAIT timing so neither backend can silently omit it."""
        self.waits.append(int(cycles))

    def _read_byte(self, address: int) -> int:
        address &= 0xFFFFFF
        if address < 0 or address >= len(self._backing):
            raise IndexError(f"Read address {address:#x} out of bounds")
        return self._backing[address]

    def peek_byte_for_preflight(self, address: int, _pc: int | None = None) -> int:
        """Return backing data without recording an architectural bus read."""

        return self._backing[address & 0xFFFFFF]

    def _write_byte(self, address: int, value: int) -> None:
        address &= 0xFFFFFF
        if address < 0 or address >= len(self._backing):
            raise IndexError(f"Write address {address:#x} out of bounds")
        value &= 0xFF
        self._backing[address] = value
        # A same-value write is still an architectural bus transaction and may
        # trigger SFR/peripheral side effects. Preserve the complete sequence.
        self.writes.append((address, value))

    def snapshot(self) -> dict[int, int]:
        return {idx: byte for idx, byte in enumerate(self._backing) if byte}


@dataclass
class ParityResult:
    opcode: int
    bytes_hex: str
    reg_diff: dict[str, tuple[int, int]]
    # Full ordered write transactions for Python vs LLAMA runs.
    writes_diff: tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]
    waits_diff: tuple[tuple[int, ...], tuple[int, ...]] = (tuple(), tuple())
    python_error: str | None = None
    llama_error: str | None = None
    expected_error: bool = False


def _make_memory(instr_bytes: bytes, pc: int) -> LoggingMemory:
    backing = bytearray(ADDRESS_SPACE_SIZE)
    backing[pc : pc + len(instr_bytes)] = instr_bytes
    return LoggingMemory(backing)


def _snapshot_registers(cpu: CPU) -> dict[str, int]:
    snap = cpu.snapshot_registers()
    raw = snap.to_dict()
    # Ignore TEMP* internal scratch registers; packed F is architectural state.
    return {k: v for k, v in raw.items() if not k.startswith("TEMP")}


def _compare_writes(
    lhs: list[tuple[int, int]], rhs: list[tuple[int, int]]
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]] | None:
    if lhs == rhs:
        return None
    return (tuple(lhs), tuple(rhs))


def _error_message(error: str) -> str:
    """Remove only transport/type wrappers, preserving the contract text."""

    message = error.split(": ", 1)[1] if ": " in error else error
    prefix = "llama execute: "
    if message.startswith(prefix):
        message = message[len(prefix) :]
    return message


def _matching_expected_rejection(
    opcode: int, python_error: str | None, llama_error: str | None
) -> bool:
    """Recognize narrowly defined two-sided fail-closed outcomes.

    PyO3 necessarily wraps a Rust execution error as ``RuntimeError`` while the
    Python backend can expose the architecture-specific exception type.  That
    type boundary is not a semantic divergence when both messages name the
    exact same quarantined contract.  Arbitrary two-sided exceptions must still
    fail the sweep.
    """

    if python_error is None or llama_error is None:
        return False
    python_message = _error_message(python_error)
    llama_message = _error_message(llama_error)
    if opcode in EXPECTED_REJECTION_OPCODES and python_message == llama_message:
        return True
    return any(
        marker in python_message and marker in llama_message
        for marker in FAIL_CLOSED_CONTRACT_MARKERS
    )


def run_case(instr_bytes: bytes, pc: int) -> ParityResult | None:
    # X/Y/U/S are masked to the 20-bit external address space. Seeding them at
    # INTERNAL_MEMORY_START (0x100000) wraps to zero, so negative offsets and
    # pushes underflow to 0xFFFFFF. Keep all pointer registers in valid scratch
    # memory instead.
    reg_init = CPURegistersSnapshot(
        pc=pc,
        x=STACK_SEED,
        y=STACK_SEED,
        u=STACK_SEED,
        s=STACK_SEED,
    )

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
            expected_error=_matching_expected_rejection(opcode, py_err, ll_err),
        )

    if (
        regs_py != regs_ll
        or mem_py.writes != mem_ll.writes
        or mem_py.waits != mem_ll.waits
    ):
        reg_diff: dict[str, tuple[int, int]] = {}
        keys = set(regs_py) | set(regs_ll)
        for key in sorted(keys):
            lp = regs_py.get(key, 0)
            rp = regs_ll.get(key, 0)
            if lp != rp:
                reg_diff[key] = (lp, rp)
        writes_diff = _compare_writes(mem_py.writes, mem_ll.writes)
        waits_diff = (
            (tuple(mem_py.waits), tuple(mem_ll.waits))
            if mem_py.waits != mem_ll.waits
            else (tuple(), tuple())
        )
        return ParityResult(
            opcode=opcode,
            bytes_hex=instr_bytes.hex(),
            reg_diff=reg_diff,
            writes_diff=writes_diff if writes_diff else (tuple(), tuple()),
            waits_diff=waits_diff,
        )

    return None


def emit_perfetto_traces(prefix: Path) -> None:
    """Emit Perfetto traces for a small deterministic program on both backends."""

    from sc62015.pysc62015 import CPU
    from pce500.tracing.perfetto_tracing import tracer as perfetto_tracer

    # Small deterministic sequence with simple ALU and IMEM writes (all supported by both backends):
    # 0: NOP
    # 1: MV A,0x12
    # 3: ADD A,0x01
    # 5: MV IMem8,0x34 (offset 0x10)
    # 8: MV IMem8,A   (offset 0x11)
    program = bytes(
        [
            0x00,  # NOP
            0x08,
            0x12,  # MV A,0x12
            0x40,
            0x01,  # ADD A,0x01
            0xCC,
            0x10,
            0x34,  # MV IMem8,0x34 @0x10
            0xA0,
            0x11,  # MV IMem8,A @0x11
        ]
    )

    def _run_trace(backend: str, path: Path) -> None:
        perfetto_tracer.stop()
        perfetto_tracer.start(str(path))
        try:
            mem = _make_memory(program, pc=0)
            cpu = CPU(mem, reset_on_init=False, backend=backend)
            idx = 0
            while True:
                pc = cpu.regs.get(RegisterName.PC)
                if pc == len(program):
                    break
                if not 0 <= pc < len(program):
                    raise RuntimeError(
                        f"{backend} synthetic trace escaped program at PC {pc:#x}"
                    )
                if idx >= len(program):
                    raise RuntimeError(
                        f"{backend} synthetic trace did not reach the program end"
                    )

                opcode = mem.read_byte(pc)
                # Canonical parity semantics are pre-instruction PC/opcode/regs,
                # followed by post-instruction IMR/ISR. This matches the full
                # machine tracers while keeping the instruction index atomic.
                registers = _snapshot_registers(cpu)
                cpu.execute_instruction(pc)
                annotations = {
                    "pc": pc & 0xFFFFFF,
                    "opcode": opcode & 0xFF,
                    "op_index": idx,
                    "mem_imr": mem.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.IMR)
                    & 0xFF,
                    "mem_isr": mem.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR)
                    & 0xFF,
                }
                annotations.update(
                    {
                        f"reg_{name.lower()}": value & 0xFF_FFFF
                        for name, value in registers.items()
                    }
                )
                perfetto_tracer.instant("InstructionTrace", "Instruction", annotations)
                idx += 1
        finally:
            perfetto_tracer.stop()

    from sc62015.pysc62015.emulator import RegisterName

    py_path = prefix.with_name(f"{prefix.name}_python.trace")
    ll_path = prefix.with_name(f"{prefix.name}_llama.trace")
    _run_trace("python", py_path)
    _run_trace("llama", ll_path)


def _edge_values_for(op: Operand) -> list[int] | None:
    if isinstance(op, ImmOffset):
        # ImmOffset.value stores the magnitude; sign is fixed on the instance.
        base = [0, 1, 0x7F, 0x80]
        if op.sign == "-":
            return [abs(v) for v in base]
        else:
            return [v for v in base if v >= 0]
    if isinstance(op, Imm20):
        return [0x00000, 0x00001, 0x7FFFF, 0x80000, 0xFFFFF]
    if isinstance(op, Imm16):
        return [0x0000, 0x0001, 0x7FFF, 0x8000, 0xFFFF]
    if isinstance(op, Imm8):
        return [0x00, 0x01, 0x7F, 0x80, 0xFF]
    return None


def _mutated_encodings(instr: Instruction) -> list[bytes]:
    operands = list(instr.operands())
    immediate_positions: list[tuple[int, Operand]] = [
        (index, op)
        for index, op in enumerate(operands)
        if _edge_values_for(op) is not None
    ]
    if not immediate_positions:
        return [bytes(encode_instr(instr, 0))]

    value_sets: list[list[int]] = []
    for _, op in immediate_positions:
        choices = _edge_values_for(op)
        if not choices:
            raise AssertionError(
                f"stress operand {type(op).__name__} has no edge-value choices"
            )
        value_sets.append(choices)

    expected_count = 1
    for choices in value_sets:
        expected_count *= len(choices)

    encodings: list[bytes] = []
    for combo in product(*value_sets):
        inst = copy.deepcopy(instr)
        copied_operands = list(inst.operands())
        for (operand_index, _), val in zip(immediate_positions, combo):
            target = copied_operands[operand_index]
            if isinstance(target, Imm20):
                target.value = val & 0xFFFFF
                target.extra_hi = (val >> 16) & 0x0F
            elif isinstance(target, Imm16):
                target.value = val & 0xFFFF
            elif isinstance(target, ImmOffset):
                target.value = abs(val) & 0xFF
            elif isinstance(target, Imm8):
                target.value = val & 0xFF
        try:
            encodings.append(bytes(encode_instr(inst, 0)))
        except Exception as exc:
            raise RuntimeError(
                f"failed to encode stress variant {combo!r} for {instr.name()}"
            ) from exc

    if len(encodings) != expected_count:
        raise AssertionError(
            f"generated {len(encodings)} stress variants, expected {expected_count}"
        )
    return encodings


def sweep(limit: int | None, stress_immediates: bool) -> list[ParityResult]:
    failures: list[ParityResult] = []
    for idx, encoding in enumerate(opcode_generator()):
        if limit is not None and idx >= limit:
            break
        raw = encoding[0] if isinstance(encoding, tuple) else encoding
        if raw is None:
            continue
        variants: Iterable[bytes]
        if stress_immediates:
            try:
                instr = decode_instr(bytearray(raw), 0)
            except Exception:
                instr = None
            if instr is None:
                variants = [bytes(raw)]
            else:
                variants = _mutated_encodings(instr)
        else:
            variants = [bytes(raw)]

        for instr_bytes in variants:
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
    parser.add_argument(
        "--stress-immediates",
        action="store_true",
        help="mutate immediate/offset operands across edge values",
    )
    parser.add_argument(
        "--allow-mismatches",
        action="store_true",
        help="exit 0 even if mismatches are found (useful for non-gating runs)",
    )
    parser.add_argument(
        "--emit-traces",
        action="store_true",
        help="Emit Perfetto traces for a small deterministic program (Python and LLAMA).",
    )
    parser.add_argument(
        "--trace-prefix",
        default="trace_ref",
        help="Prefix for emitted trace files when --emit-traces is used.",
    )
    args = parser.parse_args()

    if "llama" not in available_backends():
        raise SystemExit("LLAMA backend not available; build rustcore first.")

    findings = sweep(args.limit, args.stress_immediates)
    failures = [finding for finding in findings if not finding.expected_error]
    expected_errors = [finding for finding in findings if finding.expected_error]

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "opcode": f"0x{f.opcode:02X}",
                        "bytes": f.bytes_hex,
                        "reg_diff": f.reg_diff,
                        "writes_diff": f.writes_diff,
                        "waits_diff": f.waits_diff,
                        "python_error": f.python_error,
                        "llama_error": f.llama_error,
                        "expected_error": f.expected_error,
                    }
                    for f in findings
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
                    f"reg_diff={f.reg_diff} writes_diff={f.writes_diff} "
                    f"waits_diff={f.waits_diff}"
                )
        if expected_errors:
            rendered = ", ".join(
                f"0x{finding.opcode:02X}" for finding in expected_errors
            )
            print(f"Expected fail-closed rejections: {rendered}")

    if args.emit_traces:
        emit_perfetto_traces(Path(args.trace_prefix))

    should_fail = bool(failures) and not args.allow_mismatches
    raise SystemExit(1 if should_fail else 0)


if __name__ == "__main__":
    main()
