from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, cast

import pytest

import tools.llama_parity_runner as llama_parity_runner
import tools.llama_parity_sweep as llama_parity_sweep


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tools" / "llama_parity_runner.py"
COMPARATOR = ROOT / "scripts" / "compare_perfetto_traces.py"


def _oracle_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "bytes": [0x00],
        "pc": 0,
        "mem_imr": 0,
        "mem_isr": 0,
    }
    payload.update(updates)
    return payload


class _Snapshot:
    def __init__(self, *, f: int = 0) -> None:
        self._f = f

    def to_dict(self) -> dict[str, int]:
        return {"pc": 1, "f": self._f, "TEMP0": 0xAA}


class _FakeCPU:
    def __init__(self, backend: str, *, error_backend: str | None = None) -> None:
        self.backend = backend
        self.error_backend = error_backend

    def apply_snapshot(self, _snapshot: object) -> None:
        pass

    def execute_instruction(self, _pc: int) -> None:
        if self.backend == self.error_backend:
            raise RuntimeError(f"{self.backend} exploded")

    def snapshot_registers(self) -> _Snapshot:
        return _Snapshot(f=1 if self.backend == "python" else 0)


def test_snapshot_registers_preserves_architectural_flags() -> None:
    assert llama_parity_sweep._snapshot_registers(cast(Any, _FakeCPU("python"))) == {
        "pc": 1,
        "f": 1,
    }


def test_run_case_reports_backend_exceptions(monkeypatch) -> None:
    monkeypatch.setattr(
        llama_parity_sweep,
        "CPU",
        lambda _memory, *, reset_on_init, backend: _FakeCPU(
            backend, error_backend="python"
        ),
    )

    result = llama_parity_sweep.run_case(b"\x00", pc=0)

    assert result is not None
    assert result.python_error == "RuntimeError: python exploded"
    assert result.llama_error is None


@pytest.mark.parametrize("opcode", (0xFE, 0xFF))
def test_run_case_prepares_vector_opcode_once_before_execution(
    monkeypatch, opcode: int
) -> None:
    calls: list[tuple[str, str, int]] = []

    class TrackingCPU(_FakeCPU):
        def prepare_instruction_before_scheduling(self, pc: int) -> None:
            calls.append((self.backend, "prepare", pc))

        def execute_instruction(self, pc: int) -> None:
            calls.append((self.backend, "execute", pc))

    monkeypatch.setattr(
        llama_parity_sweep,
        "CPU",
        lambda _memory, *, reset_on_init, backend: TrackingCPU(backend),
    )

    llama_parity_sweep.run_case(bytes((opcode,)), pc=0)
    assert calls == [
        ("python", "prepare", 0),
        ("python", "execute", 0),
        ("llama", "prepare", 0),
        ("llama", "execute", 0),
    ]


def test_run_case_seeds_stacks_in_valid_external_memory(monkeypatch) -> None:
    snapshots = []

    class CapturingCPU(_FakeCPU):
        def apply_snapshot(self, snapshot: object) -> None:
            snapshots.append(snapshot)

    monkeypatch.setattr(
        llama_parity_sweep,
        "CPU",
        lambda _memory, *, reset_on_init, backend: CapturingCPU(backend),
    )

    llama_parity_sweep.run_case(b"\x00", pc=0)

    assert len(snapshots) == 2
    assert all(
        (snapshot.x, snapshot.y, snapshot.u, snapshot.s)
        == (llama_parity_sweep.STACK_SEED,) * 4
        for snapshot in snapshots
    )


def test_run_case_classifies_two_sided_reserved_rejection(monkeypatch) -> None:
    class RejectingCPU(_FakeCPU):
        def execute_instruction(self, _pc: int) -> None:
            raise ValueError("reserved")

    monkeypatch.setattr(
        llama_parity_sweep,
        "CPU",
        lambda _memory, *, reset_on_init, backend: RejectingCPU(backend),
    )

    result = llama_parity_sweep.run_case(b"\x20", pc=0)

    assert result is not None
    assert result.expected_error
    assert result.python_error == "ValueError: reserved"
    assert result.llama_error == "ValueError: reserved"


def test_run_case_classifies_two_sided_quarantined_tcl_rejection(monkeypatch) -> None:
    class RejectingCPU(_FakeCPU):
        def execute_instruction(self, _pc: int) -> None:
            raise NotImplementedError("timer hardware trace required")

    monkeypatch.setattr(
        llama_parity_sweep,
        "CPU",
        lambda _memory, *, reset_on_init, backend: RejectingCPU(backend),
    )

    result = llama_parity_sweep.run_case(b"\xce", pc=0)

    assert result is not None
    assert result.expected_error


def test_fail_closed_i_zero_wrappers_are_semantically_equivalent() -> None:
    marker = "SC62015 I=0 counted-instruction semantics require real-hardware tracing"

    assert llama_parity_sweep._matching_expected_rejection(
        0x54,
        f"NotImplementedError: {marker}",
        f"RuntimeError: llama execute: {marker}",
    )


def test_expected_rejection_does_not_hide_arbitrary_two_sided_errors() -> None:
    assert not llama_parity_sweep._matching_expected_rejection(
        0x20,
        "ValueError: decoder bug",
        "RuntimeError: llama execute: different bug",
    )
    assert not llama_parity_sweep._matching_expected_rejection(
        0x54,
        "NotImplementedError: SC62015 I=0 counted-instruction semantics require real-hardware tracing",
        None,
    )


def test_compare_writes_preserves_order_and_same_value_transactions() -> None:
    assert llama_parity_sweep._compare_writes([(0x10, 0), (0x10, 1)], [(0x10, 1)]) == (
        ((0x10, 0), (0x10, 1)),
        ((0x10, 1),),
    )


def test_stress_immediates_mutates_the_actual_immediate_operand() -> None:
    instr = llama_parity_sweep.decode_instr(bytearray([0x40, 0x00]), 0)

    generated = llama_parity_sweep._mutated_encodings(instr)

    assert len(generated) == 5
    assert set(generated) == {
        bytes([0x40, 0x00]),
        bytes([0x40, 0x01]),
        bytes([0x40, 0x7F]),
        bytes([0x40, 0x80]),
        bytes([0x40, 0xFF]),
    }


def test_stress_immediate_generation_surfaces_encoder_failures(monkeypatch) -> None:
    instr = llama_parity_sweep.decode_instr(bytearray([0x40, 0x00]), 0)

    def reject_encode(_instr, _addr):
        raise ValueError("encoder bug")

    monkeypatch.setattr(llama_parity_sweep, "encode_instr", reject_encode)

    with pytest.raises(
        RuntimeError, match="failed to encode stress variant"
    ) as exc_info:
        llama_parity_sweep._mutated_encodings(instr)

    assert isinstance(exc_info.value.__cause__, ValueError)


def test_python_oracle_ignores_llama_backend_environment(monkeypatch) -> None:
    monkeypatch.setenv("SC62015_CPU_BACKEND", "llama")
    monkeypatch.setattr(llama_parity_runner, "HAVE_PERFETTO", False)

    snapshot = llama_parity_runner.run_once(json.dumps(_oracle_payload()))

    assert snapshot.backend == "python"
    assert json.loads(snapshot.to_json())["backend"] == "python"


def test_python_oracle_preserves_explicit_zero_subregister_seeds(monkeypatch) -> None:
    monkeypatch.setattr(llama_parity_runner, "HAVE_PERFETTO", False)
    snapshot = llama_parity_runner.run_once(
        json.dumps(
            _oracle_payload(
                regs={
                    "BA": 0xFFFF,
                    "A": 0,
                    "B": 0,
                    "I": 0xFFFF,
                    "IL": 0,
                    "IH": 0,
                }
            )
        )
    )

    assert snapshot.regs["BA"] == 0
    assert snapshot.regs["I"] == 0


@pytest.mark.parametrize("missing", ["mem_imr", "mem_isr"])
def test_python_oracle_requires_interrupt_image(missing: str, monkeypatch) -> None:
    monkeypatch.setattr(llama_parity_runner, "HAVE_PERFETTO", False)
    payload = _oracle_payload()
    payload.pop(missing)

    with pytest.raises(ValueError, match=missing):
        llama_parity_runner.run_once(json.dumps(payload))


def test_python_oracle_rejects_contradictory_imr_aliases(monkeypatch) -> None:
    monkeypatch.setattr(llama_parity_runner, "HAVE_PERFETTO", False)

    with pytest.raises(ValueError, match="IMR aliases disagree"):
        llama_parity_runner.run_once(
            json.dumps(_oracle_payload(mem_imr=0x80, regs={"IMR": 0x81}))
        )


def test_python_oracle_subprocess_trace_contract(tmp_path: Path) -> None:
    """Exercise the actual JSON/Perfetto/comparator process boundary."""

    assert llama_parity_runner.HAVE_PERFETTO, (
        "retrobus-perfetto is required for the parity subprocess smoke"
    )
    trace_path = tmp_path / "python-oracle.perfetto-trace"
    payload = _oracle_payload(
        mem_imr=0x81,
        mem_isr=0x04,
        regs={"IMR": 0x81},
        perfetto_path=str(trace_path),
    )
    env = os.environ.copy()
    env["FORCE_BINJA_MOCK"] = "1"
    # The subprocess must remain an independent Python oracle even when its
    # caller selects LLAMA globally.
    env["SC62015_CPU_BACKEND"] = "llama"
    oracle = subprocess.run(
        [sys.executable, str(RUNNER)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        check=False,
    )
    assert oracle.returncode == 0, oracle.stderr
    result = json.loads(oracle.stdout)
    assert result["backend"] == "python"
    assert result["mem_imr"] == 0x81
    assert result["mem_isr"] == 0x04
    assert result["regs"]["IMR"] == result["mem_imr"]
    assert result["perfetto_path"] == str(trace_path)
    assert trace_path.is_file()

    compared = subprocess.run(
        [sys.executable, str(COMPARATOR), str(trace_path), str(trace_path)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        check=False,
    )
    assert compared.returncode == 0, (
        f"stdout:\n{compared.stdout}\nstderr:\n{compared.stderr}"
    )
