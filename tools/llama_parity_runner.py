# ruff: noqa: E402
"""Helper to run a single LLAMA vs Python parity case.

This stays SCIL/LLIL-free: the Python side uses the existing emulator,
while the Rust side should call this script as a subprocess to get oracle
state for a synthetic instruction stream. When retrobus-perfetto is available,
the runner also emits a Perfetto trace with InstructionTrace/MemoryWrites
tracks so `scripts/compare_perfetto_traces.py` can diff against the Rust trace.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure repo root is importable when invoked via subprocess
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from binja_test_mocks.eval_llil import Memory
from sc62015.pysc62015.constants import (
    ADDRESS_SPACE_SIZE,
    INTERNAL_MEMORY_START,
    INTERNAL_MEMORY_LENGTH,
)

try:
    from retrobus_perfetto import PerfettoTraceBuilder

    HAVE_PERFETTO = True
except ImportError:  # pragma: no cover - optional dependency
    PerfettoTraceBuilder = None  # type: ignore
    HAVE_PERFETTO = False
from sc62015.pysc62015.instr import decode
from sc62015.pysc62015.cpu import CPU


@dataclass
class Snapshot:
    regs: Dict[str, int]
    mem_writes: List[Tuple[int, int, int, str]]  # addr, bits, value, space
    perfetto_path: str | None = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "regs": self.regs,
                "mem_writes": self.mem_writes,
            },
            sort_keys=True,
        )


class TrackedMemory(Memory):
    def __init__(self):
        self._backing = bytearray(ADDRESS_SPACE_SIZE)
        self._writes: List[Tuple[int, int, int, str]] = []
        super().__init__(self._read_byte, self._write_byte)

    def _read_byte(self, address: int) -> int:
        address &= 0xFFFFFF
        if address >= len(self._backing):
            return 0
        return self._backing[address]

    def write_byte(self, address: int, value: int) -> None:
        self._writes.append(
            (
                address & 0xFFFFFF,
                8,
                value & 0xFF,
                self._space_for(address),
            )
        )
        self._write_byte(address, value)

    def write_word(self, address: int, value: int) -> None:
        self._writes.append(
            (
                address & 0xFFFFFF,
                16,
                value & 0xFFFF,
                self._space_for(address),
            )
        )
        self._write_byte(address, value & 0xFF)
        self._write_byte(address + 1, (value >> 8) & 0xFF)

    def _write_byte(self, address: int, value: int) -> None:
        address &= 0xFFFFFF
        if address >= len(self._backing):
            return
        self._backing[address] = value & 0xFF

    def _space_for(self, address: int) -> str:
        if (
            INTERNAL_MEMORY_START
            <= address
            < INTERNAL_MEMORY_START + INTERNAL_MEMORY_LENGTH
        ):
            return "internal"
        return "external"

    def writes(self) -> List[Tuple[int, int, int, str]]:
        return list(self._writes)


def run_once(payload: str) -> Snapshot:
    data = json.loads(payload)
    bytes_in = bytes(data["bytes"])
    regs_in: Dict[str, int] = data.get("regs", {})
    pc = data.get("pc", 0)

    mem = TrackedMemory()
    for offset, b in enumerate(bytes_in):
        mem._backing[pc + offset] = b

    cpu = CPU(mem, reset_on_init=False)
    # Seed registers
    for name, value in regs_in.items():
        cpu.write_register(name, value)
    cpu.write_register("PC", pc)

    instr = decode.decode_at(pc, mem)
    cpu.emulator.evaluate_one(instr)

    regs_out = {}
    for name in ("A", "B", "BA", "IL", "IH", "I", "X", "Y", "U", "S", "PC", "F", "IMR"):
        regs_out[name] = cpu.read_register(name)
    regs_out["FC"] = 1 if cpu.read_flag("C") else 0
    regs_out["FZ"] = 1 if cpu.read_flag("Z") else 0

    perfetto_out: str | None = None
    if HAVE_PERFETTO:
        builder = (
            PerfettoTraceBuilder.new("PythonParity")
            if hasattr(PerfettoTraceBuilder, "new")
            else PerfettoTraceBuilder("PythonParity")
        )
        instr_track = builder.add_thread("InstructionTrace")
        mem_track = builder.add_thread("MemoryWrites")
        ts = 0
        ev = builder.add_instant_event(instr_track, f"Exec@0x{pc:06X}", ts)
        ev.add_annotations(
            [
                ("backend", "python"),
                ("pc", pc),
                ("opcode", bytes_in[0] if bytes_in else 0),
                ("op_index", 0),
            ]
        )
        for name, value in regs_out.items():
            ev.add_annotation(f"reg_{name.lower()}", value & 0xFF_FFFF)
        ev.finish()
        for addr, bits, value, space in mem.writes():
            mev = builder.add_instant_event(mem_track, f"Write@0x{addr:06X}", ts + 1)
            mev.add_annotations(
                [
                    ("backend", "python"),
                    ("pc", pc),
                    ("address", addr),
                    ("value", value & 0xFF_FFFF),
                    ("size", bits),
                    ("op_index", 0),
                    ("space", space),
                ]
            )
            mev.finish()
        out_path = Path("python_parity.pftrace")
        builder.save(out_path)
        perfetto_out = str(out_path)

    return Snapshot(regs=regs_out, mem_writes=mem.writes(), perfetto_path=perfetto_out)


def main() -> None:
    payload = sys.stdin.read()
    snap = run_once(payload)
    sys.stdout.write(snap.to_json())


if __name__ == "__main__":
    main()
# Ensure repo root and binja_test_mocks are importable.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BINJA_MOCK_ROOT = ROOT / "vendor" / "binja-test-mocks"
if BINJA_MOCK_ROOT.exists():
    mock_path = BINJA_MOCK_ROOT
    if str(mock_path) not in sys.path:
        sys.path.insert(0, str(mock_path))
