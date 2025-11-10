from __future__ import annotations

import json
import subprocess
from typing import Dict, Iterable, Tuple

import pytest

from sc62015.decoding import decode_map
from sc62015.decoding.reader import StreamCtx
from sc62015.scil import from_decoded, serde
from sc62015.scil.pyemu import CPUState, MemoryBus, execute_decoded
from sc62015.pysc62015.instr.opcodes import IMEMRegisters, INTERRUPT_VECTOR_ADDR

CARGO_CMD = [
    "cargo",
    "run",
    "--quiet",
    "--manifest-path",
    "emulators/rust_scil/Cargo.toml",
    "--bin",
    "scil-run",
]


def _decode(opcode: int, operands: Iterable[int], pc: int) -> decode_map.DecodedInstr:
    data = bytes([opcode, *operands])
    ctx = StreamCtx(pc=pc, data=data[1:], base_len=1)
    return decode_map.decode_opcode(opcode, ctx)


_REG_WIDTHS = {
    "A": 8,
    "B": 8,
    "IL": 8,
    "IH": 8,
    "BA": 16,
    "I": 16,
    "X": 24,
    "Y": 24,
    "U": 24,
    "S": 24,
    "F": 8,
}


def _snapshot(cpu: CPUState, bus: MemoryBus) -> Dict[str, Dict]:
    regs = {"PC": cpu.pc}
    for name, width in _REG_WIDTHS.items():
        regs[name] = cpu.get_reg(name, width)
    flags = {"C": cpu.get_flag("C"), "Z": cpu.get_flag("Z")}
    return {
        "state": {"pc": cpu.pc, "halted": cpu.halted, "regs": regs, "flags": flags},
        "int_mem": bus.dump_internal(),
        "ext_mem": bus.dump_external(),
    }


def _build_initial_state(pc: int = 0) -> Tuple[CPUState, MemoryBus]:
    cpu = CPUState()
    cpu.pc = pc
    bus = MemoryBus()
    return cpu, bus


def _apply_snapshot(cpu: CPUState, bus: MemoryBus, snapshot: Dict[str, Dict]) -> None:
    cpu.pc = snapshot["state"]["pc"] & 0xFFFFF
    cpu.halted = snapshot["state"].get("halted", False)
    for name, value in snapshot["state"]["regs"].items():
        if name == "PC":
            continue
        width = _REG_WIDTHS.get(name, 24)
        cpu.set_reg(name, value, width)
    for flag, value in snapshot["state"]["flags"].items():
        cpu.set_flag(flag, value)
    bus.preload_internal(snapshot["int_mem"].items())
    bus.preload_external(snapshot["ext_mem"].items())


def _run_rust(payload: Dict) -> Dict:
    try:
        proc = subprocess.run(
            CARGO_CMD,
            input=json.dumps(payload).encode("utf-8"),
            capture_output=True,
            check=True,
        )
    except FileNotFoundError:
        pytest.skip("cargo not available")
    return json.loads(proc.stdout.decode("utf-8"))


def _compare_states(py: Dict, rust: Dict) -> None:
    assert rust["state"]["pc"] == py["state"]["pc"]
    assert rust["state"].get("halted", False) == py["state"].get("halted", False)
    for name, value in py["state"]["regs"].items():
        if name == "PC":
            continue
        assert rust["state"]["regs"].get(name, 0) == value
    assert rust["state"]["flags"] == py["state"]["flags"]
    assert dict(rust["int_mem"]) == py["int_mem"]
    assert dict(rust["ext_mem"]) == py["ext_mem"]


@pytest.mark.parametrize(
    "opcode,operands,setup",
    [
        (0x08, [0x5A], lambda cpu, bus: None),
        (0x18, [0x04], lambda cpu, bus: cpu.set_flag("Z", 1)),
        (0x02, [0x34, 0x12], lambda cpu, bus: None),
        (
            0x88,
            [0x10, 0x20, 0x00],
            lambda cpu, bus: bus.preload_external([(0x002010, 0xAA)]),
        ),
        (
            0xA8,
            [0x12, 0x20, 0x00],
            lambda cpu, bus: cpu.set_reg("A", 0xCC, 8),
        ),
        (
            0xCB,
            [0x10, 0x20],
            lambda cpu, bus: (
                cpu.set_reg("I", 2, 16),
                bus.preload_internal(
                    [
                        (0x10, 0x00),
                        (0x11, 0x00),
                        (0x20, 0x11),
                        (0x21, 0x22),
                    ]
                ),
            ),
        ),
        (
            0x54,
            [0x30, 0x40],
            lambda cpu, bus: (
                cpu.set_reg("I", 2, 16),
                cpu.set_flag("C", 1),
                bus.preload_internal(
                    [
                        (0x30, 0x05),
                        (0x31, 0x06),
                        (0x40, 0x01),
                        (0x41, 0x02),
                    ]
                ),
            ),
        ),
        (
            0xC4,
            [0x50, 0x60],
            lambda cpu, bus: (
                cpu.set_reg("I", 2, 16),
                bus.preload_internal(
                    [
                        (0x50, 0x12),
                        (0x51, 0x34),
                        (0x60, 0x11),
                        (0x61, 0x22),
                    ]
                ),
            ),
        ),
        (
            0xEC,
            [0x70],
            lambda cpu, bus: (
                cpu.set_reg("I", 3, 16),
                bus.preload_internal(
                    [
                        (0x70, 0x09),
                        (0x6F, 0x00),
                        (0x6E, 0x00),
                    ]
                ),
            ),
        ),
        (
            0xDE,
            [],
            lambda cpu, bus: bus.preload_internal(
                [
                    (IMEMRegisters.USR, 0xFF),
                    (IMEMRegisters.SSR, 0x00),
                ]
            ),
        ),
        (
            0xFE,
            [],
            lambda cpu, bus: (
                cpu.set_reg("S", 0x0200, 24),
                cpu.set_reg("F", 0xA5, 8),
                bus.preload_internal([(IMEMRegisters.IMR, 0xFF)]),
                bus.preload_external(
                    [
                        (INTERRUPT_VECTOR_ADDR, 0xAA),
                        (INTERRUPT_VECTOR_ADDR + 1, 0xBB),
                        (INTERRUPT_VECTOR_ADDR + 2, 0x01),
                    ]
                ),
            ),
        ),
    ],
)
def test_rust_cli_matches_pyemu(opcode: int, operands: Iterable[int], setup) -> None:
    pc = 0x1000
    base_cpu, base_bus = _build_initial_state(pc)
    setup(base_cpu, base_bus)
    base_snapshot = _snapshot(base_cpu, base_bus)

    cpu_py, bus_py = _build_initial_state()
    _apply_snapshot(cpu_py, bus_py, base_snapshot)
    decoded = _decode(opcode, operands, pc)

    execute_decoded(cpu_py, bus_py, decoded)
    py_snapshot = _snapshot(cpu_py, bus_py)

    build = from_decoded.build(decoded)
    instr_dict = serde.instr_to_dict(build.instr)
    binder_dict = {name: serde.expr_to_dict(expr) for name, expr in build.binder.items()}
    pre_applied = (
        {"first": build.pre_applied.first, "second": build.pre_applied.second}
        if build.pre_applied
        else None
    )
    payload = {
        "state": base_snapshot["state"],
        "int_mem": list(base_snapshot["int_mem"].items()),
        "ext_mem": list(base_snapshot["ext_mem"].items()),
        "instr": instr_dict,
        "binder": binder_dict,
        "pre_applied": pre_applied,
    }
    rust_output = _run_rust(payload)

    _compare_states(py_snapshot, rust_output)
