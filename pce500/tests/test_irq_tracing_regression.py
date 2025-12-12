from __future__ import annotations

from pce500.emulator import PCE500Emulator as Emulator
from pce500.tracing.perfetto_tracing import tracer as new_tracer
from sc62015.pysc62015 import RegisterName


def _collect_imem_reads(emu: Emulator) -> list[str]:
    reads: list[str] = []

    def callback(pc: int, reg_name: str, access_type: str, value: int) -> None:
        if access_type == "read":
            reads.append(reg_name)

    emu.memory.set_imem_access_callback(callback)
    emu.memory.write_byte(0x0000, 0x00)  # NOP
    emu.cpu.regs.set(RegisterName.PC, 0x0000)
    emu.step()
    return reads


def test_irq_probe_skipped_without_tracing() -> None:
    # Ensure tracer state is clean between runs.
    try:
        new_tracer.safe_stop()
    except Exception:
        pass

    emu = Emulator(
        trace_enabled=False,
        perfetto_trace=False,
        enable_new_tracing=False,
        keyboard_columns_active_high=True,
    )
    reads = _collect_imem_reads(emu)

    assert "IMR" not in reads
    assert "ISR" not in reads


def test_irq_probe_runs_with_tracing_enabled() -> None:
    try:
        new_tracer.safe_stop()
    except Exception:
        pass

    emu = Emulator(
        trace_enabled=False,
        perfetto_trace=True,
        enable_new_tracing=True,
        keyboard_columns_active_high=True,
    )
    try:
        reads = _collect_imem_reads(emu)
    finally:
        try:
            new_tracer.safe_stop()
        except Exception:
            pass

    assert "IMR" in reads
    assert "ISR" in reads
