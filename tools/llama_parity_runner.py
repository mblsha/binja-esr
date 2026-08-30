# ruff: noqa: E402
"""Helper to run a single LLAMA vs Python parity case.

This stays SCIL/LLIL-free: the Python side uses the existing emulator,
while the Rust side should call this script as a subprocess to get oracle
state for a synthetic instruction stream. When ``perfetto_path`` is requested,
the runner requires retrobus-perfetto and emits InstructionTrace/MemoryWrites
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
BINJA_MOCK_ROOT = ROOT / "vendor" / "binja-test-mocks"
if BINJA_MOCK_ROOT.exists() and str(BINJA_MOCK_ROOT) not in sys.path:
    sys.path.insert(0, str(BINJA_MOCK_ROOT))

from binja_test_mocks.eval_llil import Memory
from sc62015.pysc62015.constants import (
    ADDRESS_SPACE_SIZE,
    INTERNAL_MEMORY_START,
    INTERNAL_MEMORY_LENGTH,
)

try:
    from retrobus_perfetto.builder import PerfettoTraceBuilder

    HAVE_PERFETTO = True
except ImportError:  # pragma: no cover - optional dependency
    PerfettoTraceBuilder = None  # type: ignore
    HAVE_PERFETTO = False
from sc62015.pysc62015.cpu import CPU
from sc62015.pysc62015.stepper import CPURegistersSnapshot


@dataclass
class Snapshot:
    regs: Dict[str, int]
    mem_writes: List[Tuple[int, int, int, str]]  # addr, bits, value, space
    mem_imr: int
    mem_isr: int
    backend: str
    perfetto_path: str | None = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "regs": self.regs,
                "mem_writes": self.mem_writes,
                "mem_imr": self.mem_imr,
                "mem_isr": self.mem_isr,
                "backend": self.backend,
                "perfetto_path": self.perfetto_path,
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


IMR_ADDRESS = INTERNAL_MEMORY_START + 0xFB
ISR_ADDRESS = INTERNAL_MEMORY_START + 0xFC


def _required_u8(data: dict[str, object], field: str) -> int:
    """Read one mandatory byte field without accepting JSON booleans."""

    if field not in data:
        raise ValueError(f"parity input missing required {field}")
    value = data[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"parity input {field} must be an integer byte")
    parsed = value
    if not 0 <= parsed <= 0xFF:
        raise ValueError(f"parity input {field} must fit in an unsigned byte")
    return parsed


def _seeded_byte(mem_seed: object, address: int) -> int | None:
    """Return the last byte seeded at ``address``, matching load order."""

    if not isinstance(mem_seed, list):
        raise TypeError("parity input mem must be an array")
    result: int | None = None
    for index, entry in enumerate(mem_seed):
        if not isinstance(entry, list) or len(entry) != 2:
            raise ValueError(f"parity input mem[{index}] must be [address, value]")
        raw_address, raw_value = entry
        if (
            isinstance(raw_address, bool)
            or not isinstance(raw_address, int)
            or isinstance(raw_value, bool)
            or not isinstance(raw_value, int)
        ):
            raise TypeError(f"parity input mem[{index}] values must be integers")
        entry_address = raw_address & 0xFFFFFF
        entry_value = raw_value
        if not 0 <= entry_value <= 0xFF:
            raise ValueError(f"parity input mem[{index}] value must fit in a byte")
        if entry_address == address:
            result = entry_value
    return result


def _validate_interrupt_seeds(
    regs_in: Dict[str, int], mem_seed: object, mem_imr: int, mem_isr: int
) -> None:
    """Reject contradictory aliases before executing the oracle."""

    reg_imr = regs_in.get("IMR")
    if reg_imr is not None:
        if isinstance(reg_imr, bool) or not isinstance(reg_imr, int):
            raise TypeError("parity input register IMR must be an integer byte")
        if not 0 <= reg_imr <= 0xFF:
            raise ValueError("parity input register IMR must fit in a byte")
        if reg_imr != mem_imr:
            raise ValueError(
                f"parity input IMR aliases disagree: regs.IMR={reg_imr:#04x}, "
                f"mem_imr={mem_imr:#04x}"
            )

    for field, address, required in (
        ("IMR", IMR_ADDRESS, mem_imr),
        ("ISR", ISR_ADDRESS, mem_isr),
    ):
        seeded = _seeded_byte(mem_seed, address)
        if seeded is not None and seeded != required:
            raise ValueError(
                f"parity input {field} aliases disagree: mem seed={seeded:#04x}, "
                f"mem_{field.lower()}={required:#04x}"
            )


def _trace_register_key(name: str) -> str:
    """Match the Rust parity writer's annotation names exactly."""

    if name == "FC":
        return "flag_c"
    if name == "FZ":
        return "flag_z"
    return f"reg_{name.lower()}"


def run_once(payload: str) -> Snapshot:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise TypeError("parity input must be a JSON object")
    bytes_in = bytes(data["bytes"])
    regs_in: Dict[str, int] = data.get("regs", {})
    pc = data.get("pc", 0)
    mem_seed = data.get("mem", [])
    mem_imr_in = _required_u8(data, "mem_imr")
    mem_isr_in = _required_u8(data, "mem_isr")
    _validate_interrupt_seeds(regs_in, mem_seed, mem_imr_in, mem_isr_in)

    mem = TrackedMemory()
    for offset, b in enumerate(bytes_in):
        mem._backing[pc + offset] = b
    for addr, val in mem_seed:
        mem._backing[addr & 0xFFFFFF] = val & 0xFF
    # IMR/ISR are mandatory, named inputs. Seed them last so the byte image and
    # the explicit parity contract cannot drift apart.
    mem._backing[IMR_ADDRESS] = mem_imr_in
    mem._backing[ISR_ADDRESS] = mem_isr_in

    # This process is the independent Python oracle. Never inherit a caller's
    # SC62015_CPU_BACKEND=llama setting and accidentally compare LLAMA to itself.
    cpu = CPU(mem, reset_on_init=False, backend="python")
    if cpu.backend != "python":  # pragma: no cover - defensive contract guard
        raise RuntimeError(f"parity oracle selected unexpected backend {cpu.backend!r}")
    # Seed registers via snapshot
    ba = regs_in.get("BA", 0)
    a = regs_in.get("A")
    b = regs_in.get("B")
    if a is not None or b is not None:
        a_value = a if a is not None else ba & 0xFF
        b_value = b if b is not None else (ba >> 8) & 0xFF
        ba = (b_value << 8) | a_value
    i_val = regs_in.get("I", 0)
    il = regs_in.get("IL")
    ih = regs_in.get("IH")
    if il is not None or ih is not None:
        il_value = il if il is not None else i_val & 0xFF
        ih_value = ih if ih is not None else (i_val >> 8) & 0xFF
        i_val = (ih_value << 8) | il_value
    f_val = regs_in.get("F")
    if f_val is None:
        fc = regs_in.get("FC", 0) & 1
        fz = (regs_in.get("FZ", 0) & 1) << 1
        f_val = fc | fz
    snap = CPURegistersSnapshot(
        pc=pc,
        ba=ba,
        i=i_val,
        x=regs_in.get("X", 0),
        y=regs_in.get("Y", 0),
        u=regs_in.get("U", 0),
        s=regs_in.get("S", 0),
        f=f_val,
    )
    cpu.apply_snapshot(snap)

    cpu.execute_instruction(pc)

    snap = cpu.snapshot_registers().to_dict()
    ba = snap.get("ba", 0)
    i_val = snap.get("i", 0)
    f_val = snap.get("f", 0)
    mem_imr = mem._backing[IMR_ADDRESS]
    mem_isr = mem._backing[ISR_ADDRESS]
    regs_out = {
        "A": ba & 0xFF,
        "B": (ba >> 8) & 0xFF,
        "BA": ba,
        "IL": i_val & 0xFF,
        "IH": (i_val >> 8) & 0xFF,
        "I": i_val,
        "X": snap.get("x", 0),
        "Y": snap.get("y", 0),
        "U": snap.get("u", 0),
        "S": snap.get("s", 0),
        "PC": snap.get("pc", 0),
        "F": f_val,
        "FC": f_val & 0x1,
        "FZ": (f_val >> 1) & 0x1,
        "IMR": mem_imr,
    }

    perfetto_out: str | None = None
    requested_perfetto = data.get("perfetto_path")
    if requested_perfetto is not None:
        if not isinstance(requested_perfetto, str) or not requested_perfetto:
            raise TypeError("parity input perfetto_path must be a non-empty string")
        if not HAVE_PERFETTO:
            raise RuntimeError(
                "perfetto_path requested but retrobus-perfetto is unavailable"
            )
        assert PerfettoTraceBuilder is not None
        builder = PerfettoTraceBuilder("PythonParity")
        instr_track = builder.add_thread("InstructionTrace")
        mem_track = builder.add_thread("MemoryWrites")
        ts = 0
        ev = builder.add_instant_event(instr_track, f"Exec@0x{pc:06X}", ts)
        ev.add_annotations(
            {
                "backend": "python",
                "pc": pc,
                "opcode": bytes_in[0] if bytes_in else 0,
                "op_index": 0,
                "mem_imr": mem_imr,
                "mem_isr": mem_isr,
            }
        )
        ev.add_annotations(
            {
                _trace_register_key(name): value & 0xFF_FFFF
                for name, value in regs_out.items()
            }
        )
        for addr, bits, value, space in mem.writes():
            mev = builder.add_instant_event(mem_track, f"Write@0x{addr:06X}", ts + 1)
            mev.add_annotations(
                {
                    "backend": "python",
                    "pc": pc,
                    "address": addr,
                    "value": value & 0xFF_FFFF,
                    "size": bits,
                    "op_index": 0,
                    "space": space,
                }
            )
        out_path = Path(requested_perfetto)
        builder.save(str(out_path))
        perfetto_out = str(out_path)

    return Snapshot(
        regs=regs_out,
        mem_writes=mem.writes(),
        mem_imr=mem_imr,
        mem_isr=mem_isr,
        backend=cpu.backend,
        perfetto_path=perfetto_out,
    )


def main() -> None:
    payload = sys.stdin.read()
    snap = run_once(payload)
    sys.stdout.write(snap.to_json())


if __name__ == "__main__":
    main()
