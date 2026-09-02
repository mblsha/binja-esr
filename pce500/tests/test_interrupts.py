"""Synthetic SC62015 interrupt-model tests using generated handler programs.

These pin emulator delivery, wake, and accounting behavior. They do not execute
the stock ROM dispatcher or establish real-device peripheral timing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from textwrap import dedent
from typing import Dict, Iterable, Optional, Set

import pytest

from sc62015.pysc62015.sc_asm import Assembler
from sc62015.pysc62015.emulator import RegisterName
from sc62015.pysc62015.instr.opcodes import IMEMRegisters

from pce500 import PCE500Emulator
from pce500.emulator import IRQSource
from pce500.keyboard_matrix import KEY_LOCATIONS
from pce500.tests.vector_fixtures import install_static_machine_vectors
from sc62015.pysc62015.constants import IMRFlag, ISRFlag
from sc62015.pysc62015 import available_backends


# ----------------------------- Types & constants -----------------------------

INTERNAL_MEMORY_START: int = 0x100000
TEST_MTI_PERIOD: int = 200
TEST_STI_PERIOD: int = 2000


class Trigger(Enum):
    KEY_F1 = "KEY_F1"
    KEY_ON = "KEY_ON"
    KEY_OFF = "KEY_OFF"
    MTI = "MTI"  # Main timer
    STI = "STI"  # Sub timer
    NONE = "NONE"  # No source


class Program(Enum):
    HALT = "HALT"
    WAIT = "WAIT"
    OFF = "OFF"


TRIGGER_MASK: Dict[Trigger, int] = {
    Trigger.MTI: int(ISRFlag.MTI),
    Trigger.STI: int(ISRFlag.STI),
    Trigger.KEY_F1: int(ISRFlag.KEYI),
    Trigger.KEY_ON: int(ISRFlag.ONKI),
}


@dataclass(frozen=True)
class ProgramConfig:
    entry: int
    handler: int
    marker: int = 0x42


PROGRAM = ProgramConfig(entry=0xB8000, handler=0xB9000)


# Assembly templates (same logic as the unified tests)
ENTRY_HALT = """
.ORG 0x{entry:05X}
entry:
    HALT
    NOP
"""

ENTRY_WAIT = """
.ORG 0x{entry:05X}
entry:
{wait_body}
    NOP
"""

ENTRY_OFF = """
.ORG 0x{entry:05X}
entry:
    OFF
    NOP
"""

HANDLER_TEMPLATE = """
.ORG 0x{handler:05X}
handler:
    MV A, 0x{marker:02X}
    PUSHU A
    MV A, (ISR)
    PUSHU A
    MV (ISR), 0x00
    RETI
"""


# ----------------------------- Strongly-typed I/O -----------------------------


@dataclass(frozen=True)
class Given:
    """Input state for a scenario."""

    name: str
    trigger: Trigger
    program: Program
    imr: Set[IMRFlag]
    wait_delta: Optional[int] = None  # WAIT-only: delta vs period (cycles)
    isolate_timers: bool = False  # disable the other timer
    timers_enabled: Optional[bool] = None  # HALT-only: None→off by default


@dataclass(frozen=True)
class Expect:
    """What the test intends to check (contract)."""

    deliver: bool
    halt_canceled: Optional[bool] = None  # Only meaningful for HALT/OFF program


@dataclass(frozen=True)
class Observed:
    """Golden output snapshot for a scenario."""

    halted_after: bool
    u_before: int
    u_after: int
    u_delta: int
    delivered: bool
    isr_stack: Optional[int]
    marker_stack: Optional[int]
    last_irq_src_present: bool


@dataclass(frozen=True)
class InterruptSpec:
    """A full test scenario = Given + Expect + inline Observed snapshot."""

    given: Given
    expect: Expect
    observed: Observed  # inline, strongly-typed golden


# ----------------------------- Helpers (unchanged core) -----------------------------


def _wait_chunks(wait_count: int) -> list[int]:
    if wait_count <= 0:
        return [0]
    chunks: list[int] = []
    remaining = wait_count
    while remaining > 0:
        chunk = min(0xFFFF, remaining)
        chunks.append(chunk)
        remaining -= chunk
    return chunks


def _assemble_program(
    emu: PCE500Emulator, program: Program, wait_count: Optional[int]
) -> int:
    asm = Assembler()
    if program is Program.HALT:
        entry_src = ENTRY_HALT.format(entry=PROGRAM.entry)
    elif program is Program.WAIT:
        chunks = _wait_chunks(wait_count or 0)
        wait_body = "\n".join(f"    MV I, 0x{chunk:04X}\n    WAIT" for chunk in chunks)
        entry_src = ENTRY_WAIT.format(entry=PROGRAM.entry, wait_body=wait_body)
    else:
        entry_src = ENTRY_OFF.format(entry=PROGRAM.entry)
    handler_src = HANDLER_TEMPLATE.format(
        handler=PROGRAM.handler, marker=PROGRAM.marker
    )
    source = dedent(entry_src + handler_src)
    binfile = asm.assemble(source)
    for seg in binfile.segments:
        for i, b in enumerate(seg.data):
            emu.memory.write_byte(seg.address + i, b)
    # Interrupt vector → handler (3-byte little-endian 20-bit pointer)
    rom = bytearray(b"\xff" * 0x40000)
    off = 0x3FFFA
    vec = PROGRAM.handler & 0xFFFFF
    rom[off : off + 3] = bytes([vec & 0xFF, (vec >> 8) & 0xFF, (vec >> 16) & 0xFF])
    emu.load_rom(bytes(rom))
    return len(chunks) if program is Program.WAIT else 0


def _isolate_other_timer(emu: PCE500Emulator, trig: Trigger) -> None:
    # For non-timer triggers, disable timers entirely to avoid spurious IRQs
    if trig not in (Trigger.MTI, Trigger.STI):
        emu._timer_enabled = False  # type: ignore[attr-defined]
        return
    if trig == Trigger.MTI:
        emu._timer_sti_period = 10**9  # type: ignore[attr-defined]
        emu._timer_next_sti = emu.cycle_count + emu._timer_sti_period  # type: ignore[attr-defined]
    elif trig == Trigger.STI:
        emu._timer_mti_period = 10**9  # type: ignore[attr-defined]
        emu._timer_next_mti = emu.cycle_count + emu._timer_mti_period  # type: ignore[attr-defined]


def _run_and_observe(g: Given, e: Expect) -> Observed:
    emu = PCE500Emulator(perfetto_trace=False, save_lcd_on_exit=False)

    # Speed up timer-driven scenarios to avoid CI timeouts while keeping semantics intact.
    emu._timer_mti_period = TEST_MTI_PERIOD  # type: ignore[attr-defined]
    emu._timer_sti_period = TEST_STI_PERIOD  # type: ignore[attr-defined]
    emu._timer_next_mti = emu.cycle_count + TEST_MTI_PERIOD  # type: ignore[attr-defined]
    emu._timer_next_sti = emu.cycle_count + TEST_STI_PERIOD  # type: ignore[attr-defined]

    # Optional isolation
    if g.isolate_timers:
        emu._timer_enabled = True  # type: ignore[attr-defined]
        _isolate_other_timer(emu, g.trigger)

    # WAIT: compute wait_count = period + delta
    wait_count: Optional[int] = None
    mti_period = int(getattr(emu, "_timer_mti_period", 2048))
    sti_period = int(getattr(emu, "_timer_sti_period", 512000))
    if g.program is Program.WAIT:
        period = mti_period if g.trigger == Trigger.MTI else sti_period
        wait_count = max(0, period + int(g.wait_delta or 0))

    wait_chunks = _assemble_program(emu, g.program, wait_count)

    # Initialize CPU/internal registers
    emu.cpu.regs.set(RegisterName.PC, PROGRAM.entry)
    emu.cpu.regs.set(RegisterName.S, 0xBFF00)
    emu.cpu.regs.set(RegisterName.U, 0xBFE00)
    emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR, 0x00)
    imr_val = 0
    for flag in g.imr:
        imr_val |= int(flag.value)
    emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.IMR, imr_val)

    # Prelude for HALT/OFF
    if g.program in (Program.HALT, Program.OFF):
        if g.program is Program.OFF:
            timers_on = False
        else:
            timers_on = False if g.timers_enabled is None else bool(g.timers_enabled)
        emu._timer_enabled = timers_on  # type: ignore[attr-defined]
        emu.step()
        assert emu.cpu.state.halted is True
        # Stable halt check only when timers are disabled
        if not timers_on:
            u_stable = emu.cpu.regs.get(RegisterName.U)
            for _ in range(3):
                emu.step()
                assert emu.cpu.state.halted is True
                assert emu.cpu.regs.get(RegisterName.U) == u_stable

    u_before = emu.cpu.regs.get(RegisterName.U)

    # Trigger
    if g.trigger in (Trigger.KEY_F1, Trigger.KEY_ON):
        if g.trigger is Trigger.KEY_F1:
            loc = KEY_LOCATIONS[g.trigger.value]
            kol = 0
            koh = 0
            if loc.column < 8:
                kol = 1 << loc.column
            else:
                koh = 1 << (loc.column - 8)
            emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.KOL, kol)
            emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.KOH, koh)
        assert emu.press_key(g.trigger.value) is True
    elif g.trigger is Trigger.KEY_OFF:
        emu.press_key(g.trigger.value)

    # These synthetic scenarios test one interrupt delivery, not held-key
    # retriggering.  Give F1 one scheduler sample at the selected KIL level,
    # then release it.  The handler acknowledges ISR after recording it so a
    # latched source cannot manufacture additional deliveries.  Dedicated
    # hardware-backed keyboard tests cover the real held-level behaviour.
    key_pulse_pending = g.trigger is Trigger.KEY_F1

    def step_once() -> None:
        nonlocal key_pulse_pending
        emu.step()
        if key_pulse_pending:
            emu.release_key(g.trigger.value)
            key_pulse_pending = False

    # Execute program to the delivery point
    if g.program is Program.HALT:
        if g.timers_enabled:
            if g.trigger is Trigger.MTI:
                steps = mti_period + 50
            elif g.trigger is Trigger.STI:
                steps = sti_period + 200
            else:
                steps = 24
            for _ in range(int(steps)):
                step_once()
        else:
            for _ in range(24):
                step_once()
    elif g.program is Program.OFF:
        # Timers remain off; ON unmasked should wake, others shouldn't
        if e.deliver:
            # One step to cancel OFF and deliver, plus a few for handler
            step_once()
            for _ in range(5):
                step_once()
        else:
            # Exceed the longest timer period to be robust to off-by-one
            longest = max(mti_period, sti_period)
            for _ in range(int(longest + 200)):
                step_once()
    else:
        # WAIT program
        for _ in range(max(1, wait_chunks)):
            step_once()  # MV I
            step_once()  # WAIT
        step_once()  # Attempt delivery
        if e.deliver:
            for _ in range(5):  # Full handler
                step_once()
        else:
            step_once()  # NOP

    u_after = emu.cpu.regs.get(RegisterName.U)
    u_delta = u_after - u_before
    delivered = u_delta == -2

    isr_stack = emu.memory.read_byte(u_after) if delivered else None
    marker_stack = emu.memory.read_byte(u_after + 1) if delivered else None
    last_irq_src_present = bool(emu.last_irq.get("src"))

    return Observed(
        halted_after=emu.cpu.state.halted,
        u_before=u_before,
        u_after=u_after,
        u_delta=u_delta,
        delivered=delivered,
        isr_stack=isr_stack,
        marker_stack=marker_stack,
        last_irq_src_present=last_irq_src_present,
    )


def _observed_snippet(name: str, obs: Observed) -> str:
    """Pretty-print an Observed(...) code snippet for quick copy-paste into SPECS."""

    def lit(v: object) -> str:
        if isinstance(v, bool) or v is None:
            return repr(v)
        if isinstance(v, int):
            return hex(v)  # hex int literals keep strong typing but read nicely
        return repr(v)

    return (
        f"Observed(\n"
        f"    halted_after={lit(obs.halted_after)},\n"
        f"    u_before={lit(obs.u_before)},\n"
        f"    u_after={lit(obs.u_after)},\n"
        f"    u_delta={lit(obs.u_delta)},\n"
        f"    delivered={lit(obs.delivered)},\n"
        f"    isr_stack={lit(obs.isr_stack)},\n"
        f"    marker_stack={lit(obs.marker_stack)},\n"
        f"    last_irq_src_present={lit(obs.last_irq_src_present)},\n"
        f")"
    )


def _assert_contract(g: Given, e: Expect, o: Observed) -> None:
    # Contract checks (readable failures)
    assert o.delivered is e.deliver
    if g.program in (Program.HALT, Program.OFF) and e.halt_canceled is not None:
        assert o.halted_after == (not e.halt_canceled)
    if o.delivered:
        mask = TRIGGER_MASK.get(g.trigger, 0)
        assert o.isr_stack is not None
        assert (o.isr_stack & mask) == mask
        assert o.marker_stack == PROGRAM.marker
        assert o.last_irq_src_present
    else:
        assert not o.last_irq_src_present


def _assert_observed_equals(expected: Observed, actual: Observed, name: str) -> None:
    if expected == actual:
        return
    # Helpful diff + ready-to-paste snippet to update the snapshot inline.
    lines = [
        f"[inline snapshot mismatch] scenario={name}",
        "Expected:",
        _observed_snippet(name, expected),
        "Actual (paste this into the scenario if intentional):",
        _observed_snippet(name, actual),
    ]
    pytest.fail("\n".join(lines))


# ----------------------------- Scenarios (typed + inline observed) -----------------------------
# The u_before is fixed by setup (0xBFE00). When delivery occurs, u_after = 0xBFDFE (two PUSHU).
# isr_stack is the ISR byte pushed by the handler; marker_stack is always 0x42 on delivery.

SPECS: tuple[InterruptSpec, ...] = (
    # HALT + keyboard/ON
    InterruptSpec(
        Given("key_unmasked", Trigger.KEY_F1, Program.HALT, {IMRFlag.IRM, IMRFlag.KEY}),
        Expect(deliver=True, halt_canceled=True),
        Observed(
            halted_after=False,
            u_before=0xBFE00,
            u_after=0xBFDFE,
            u_delta=-2,
            delivered=True,
            isr_stack=0x04,
            marker_stack=0x42,
            last_irq_src_present=True,
        ),
    ),
    InterruptSpec(
        Given("key_masked", Trigger.KEY_F1, Program.HALT, {IMRFlag.IRM}),
        Expect(deliver=False, halt_canceled=True),
        Observed(
            halted_after=False,
            u_before=0xBFE00,
            u_after=0xBFE00,
            u_delta=0,
            delivered=False,
            isr_stack=None,
            marker_stack=None,
            last_irq_src_present=False,
        ),
    ),
    InterruptSpec(
        Given("on_unmasked", Trigger.KEY_ON, Program.HALT, {IMRFlag.IRM, IMRFlag.ONK}),
        Expect(deliver=True, halt_canceled=True),
        Observed(
            halted_after=False,
            u_before=0xBFE00,
            u_after=0xBFDFE,
            u_delta=-2,
            delivered=True,
            isr_stack=0x08,
            marker_stack=0x42,
            last_irq_src_present=True,
        ),
    ),
    InterruptSpec(
        Given("on_masked", Trigger.KEY_ON, Program.HALT, {IMRFlag.IRM}),
        Expect(deliver=False, halt_canceled=True),
        Observed(
            halted_after=False,
            u_before=0xBFE00,
            u_after=0xBFE00,
            u_delta=0,
            delivered=False,
            isr_stack=None,
            marker_stack=None,
            last_irq_src_present=False,
        ),
    ),
    InterruptSpec(
        Given("no_trigger_halts", Trigger.NONE, Program.HALT, {IMRFlag.IRM}),
        Expect(deliver=False, halt_canceled=False),
        Observed(
            halted_after=True,
            u_before=0xBFE00,
            u_after=0xBFE00,
            u_delta=0,
            delivered=False,
            isr_stack=None,
            marker_stack=None,
            last_irq_src_present=False,
        ),
    ),
    # WAIT + timers
    InterruptSpec(
        Given(
            "mti_unmasked",
            Trigger.MTI,
            Program.WAIT,
            {IMRFlag.IRM, IMRFlag.MTI},
            wait_delta=50,
            isolate_timers=True,
        ),
        Expect(deliver=True),
        Observed(
            halted_after=False,
            u_before=0xBFE00,
            u_after=0xBFDFE,
            u_delta=-2,
            delivered=True,
            isr_stack=0x01,
            marker_stack=0x42,
            last_irq_src_present=True,
        ),
    ),
    InterruptSpec(
        Given(
            "mti_masked",
            Trigger.MTI,
            Program.WAIT,
            {IMRFlag.IRM},
            wait_delta=50,
            isolate_timers=True,
        ),
        Expect(deliver=False),
        Observed(
            halted_after=False,
            u_before=0xBFE00,
            u_after=0xBFE00,
            u_delta=0,
            delivered=False,
            isr_stack=None,
            marker_stack=None,
            last_irq_src_present=False,
        ),
    ),
    InterruptSpec(
        Given(
            "mti_unmasked_not_enough",
            Trigger.MTI,
            Program.WAIT,
            {IMRFlag.IRM, IMRFlag.MTI},
            wait_delta=-20,
            isolate_timers=True,
        ),
        Expect(deliver=False),
        Observed(
            halted_after=False,
            u_before=0xBFE00,
            u_after=0xBFE00,
            u_delta=0,
            delivered=False,
            isr_stack=None,
            marker_stack=None,
            last_irq_src_present=False,
        ),
    ),
    InterruptSpec(
        Given(
            "sti_unmasked",
            Trigger.STI,
            Program.WAIT,
            {IMRFlag.IRM, IMRFlag.STI},
            wait_delta=200,
            isolate_timers=True,
        ),
        Expect(deliver=True),
        Observed(
            halted_after=False,
            u_before=0xBFE00,
            u_after=0xBFDFE,
            u_delta=-2,
            delivered=True,
            isr_stack=0x02,
            marker_stack=0x42,
            last_irq_src_present=True,
        ),
    ),
    InterruptSpec(
        Given(
            "sti_masked",
            Trigger.STI,
            Program.WAIT,
            {IMRFlag.IRM},
            wait_delta=200,
            isolate_timers=True,
        ),
        Expect(deliver=False),
        Observed(
            halted_after=False,
            u_before=0xBFE00,
            u_after=0xBFE00,
            u_delta=0,
            delivered=False,
            isr_stack=None,
            marker_stack=None,
            last_irq_src_present=False,
        ),
    ),
    InterruptSpec(
        Given(
            "sti_unmasked_not_enough",
            Trigger.STI,
            Program.WAIT,
            {IMRFlag.IRM, IMRFlag.STI},
            wait_delta=-50,
            isolate_timers=True,
        ),
        Expect(deliver=False),
        Observed(
            halted_after=False,
            u_before=0xBFE00,
            u_after=0xBFE00,
            u_delta=0,
            delivered=False,
            isr_stack=None,
            marker_stack=None,
            last_irq_src_present=False,
        ),
    ),
    # WAIT + keyboard
    InterruptSpec(
        Given(
            "wait_key_unmasked",
            Trigger.KEY_F1,
            Program.WAIT,
            {IMRFlag.IRM, IMRFlag.KEY},
            isolate_timers=True,
        ),
        Expect(deliver=True),
        Observed(
            halted_after=False,
            u_before=0xBFE00,
            u_after=0xBFDFE,
            u_delta=-2,
            delivered=True,
            isr_stack=0x04,
            marker_stack=0x42,
            last_irq_src_present=True,
        ),
    ),
    InterruptSpec(
        Given(
            "wait_key_masked",
            Trigger.KEY_F1,
            Program.WAIT,
            {IMRFlag.IRM},
            isolate_timers=True,
        ),
        Expect(deliver=False),
        Observed(
            halted_after=False,
            u_before=0xBFE00,
            u_after=0xBFE00,
            u_delta=0,
            delivered=False,
            isr_stack=None,
            marker_stack=None,
            last_irq_src_present=False,
        ),
    ),
    # HALT + timers (unmasked; timers enabled)
    InterruptSpec(
        Given(
            "halt_mti_unmasked",
            Trigger.MTI,
            Program.HALT,
            {IMRFlag.IRM, IMRFlag.MTI},
            isolate_timers=True,
            timers_enabled=True,
        ),
        Expect(deliver=True, halt_canceled=True),
        Observed(
            halted_after=False,
            u_before=0xBFE00,
            u_after=0xBFDFE,
            u_delta=-2,
            delivered=True,
            isr_stack=0x01,
            marker_stack=0x42,
            last_irq_src_present=True,
        ),
    ),
    InterruptSpec(
        Given(
            "halt_sti_unmasked",
            Trigger.STI,
            Program.HALT,
            {IMRFlag.IRM, IMRFlag.STI},
            isolate_timers=True,
            timers_enabled=True,
        ),
        Expect(deliver=True, halt_canceled=True),
        Observed(
            halted_after=False,
            u_before=0xBFE00,
            u_after=0xBFDFE,
            u_delta=-2,
            delivered=True,
            isr_stack=0x02,
            marker_stack=0x42,
            last_irq_src_present=True,
        ),
    ),
    # OFF-state behavior
    InterruptSpec(
        Given("off_no_exec_on_longest_timer", Trigger.NONE, Program.OFF, {IMRFlag.IRM}),
        Expect(deliver=False, halt_canceled=False),
        Observed(
            halted_after=True,
            u_before=0xBFE00,
            u_after=0xBFE00,
            u_delta=0,
            delivered=False,
            isr_stack=None,
            marker_stack=None,
            last_irq_src_present=False,
        ),
    ),
    InterruptSpec(
        Given(
            "off_on_unmasked_wakes_and_delivers",
            Trigger.KEY_ON,
            Program.OFF,
            {IMRFlag.IRM, IMRFlag.ONK},
        ),
        Expect(deliver=True, halt_canceled=True),
        Observed(
            halted_after=False,
            u_before=0xBFE00,
            u_after=0xBFDFE,
            u_delta=-2,
            delivered=True,
            isr_stack=0x08,
            marker_stack=0x42,
            last_irq_src_present=True,
        ),
    ),
    InterruptSpec(
        Given("off_press_off_key_no_wake", Trigger.KEY_OFF, Program.OFF, {IMRFlag.IRM}),
        Expect(deliver=False, halt_canceled=False),
        Observed(
            halted_after=True,
            u_before=0xBFE00,
            u_after=0xBFE00,
            u_delta=0,
            delivered=False,
            isr_stack=None,
            marker_stack=None,
            last_irq_src_present=False,
        ),
    ),
)


@pytest.mark.parametrize("backend", available_backends())
def test_off_wake_source_model_is_distinct_and_non_destructive(
    monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    """OFF ignores non-ONKI status; this provisional policy needs hardware trace."""
    monkeypatch.setenv("SC62015_CPU_BACKEND", backend)
    emu = PCE500Emulator(perfetto_trace=False, save_lcd_on_exit=False)
    emu._timer_enabled = False  # type: ignore[attr-defined]
    emu._irq_pending = False  # type: ignore[attr-defined]
    emu.cpu.regs.set(RegisterName.PC, 0x1000)
    emu.memory.write_byte(0x1000, 0x00)  # NOP, executed only after a wake
    emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.IMR, 0)
    emu.cpu.state.halted = True
    emu.cpu.state.power_state = "off"

    emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR, int(ISRFlag.KEYI))
    emu.step()
    assert emu.cpu.state.halted is True
    assert emu.cpu.state.power_state == "off"
    assert emu.cpu.regs.get(RegisterName.PC) == 0x1000
    assert emu.memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR) == int(
        ISRFlag.KEYI
    )

    combined = int(ISRFlag.KEYI | ISRFlag.ONKI)
    emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR, combined)
    emu.step()
    assert emu.cpu.state.halted is False
    assert emu.cpu.state.power_state == "running"
    assert emu.memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR) == combined


def test_wait_scheduler_failure_does_not_advance_pc_or_i(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken host timing hook must stop WAIT before architectural mutation."""
    monkeypatch.setenv("SC62015_CPU_BACKEND", "python")
    emu = PCE500Emulator(perfetto_trace=False, save_lcd_on_exit=False)
    emu._timer_enabled = False  # type: ignore[attr-defined]
    emu.load_rom(bytes([0xEF]), start_address=0x1000)
    emu.cpu.regs.set(RegisterName.PC, 0x1000)
    emu.cpu.regs.set(RegisterName.I, 3)

    def fail_wait(_cycles: int) -> None:
        raise RuntimeError("scheduler failed")

    monkeypatch.setattr(emu, "_simulate_wait", fail_wait)

    with pytest.raises(RuntimeError, match="scheduler failed"):
        emu.step()

    assert emu.cpu.regs.get(RegisterName.PC) == 0x1000
    assert emu.cpu.regs.get(RegisterName.I) == 3


def test_wait_timer_failure_does_not_advance_pc_or_i(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failures inside the real WAIT scheduler are not converted into success."""
    monkeypatch.setenv("SC62015_CPU_BACKEND", "python")
    emu = PCE500Emulator(perfetto_trace=False, save_lcd_on_exit=False)
    emu._timer_enabled = True  # type: ignore[attr-defined]
    emu.load_rom(bytes([0xEF]), start_address=0x1000)
    emu.cpu.regs.set(RegisterName.PC, 0x1000)
    emu.cpu.regs.set(RegisterName.I, 3)
    monkeypatch.setattr(
        emu,
        "_tick_timers",
        lambda: (_ for _ in ()).throw(RuntimeError("timer failed")),
    )

    with pytest.raises(RuntimeError, match="timer failed"):
        emu.step()

    assert emu.cpu.regs.get(RegisterName.PC) == 0x1000
    assert emu.cpu.regs.get(RegisterName.I) == 3


@pytest.mark.parametrize("failure_site", ["scan", "isr"])
def test_low_power_host_failure_never_executes_next_opcode(
    monkeypatch: pytest.MonkeyPatch, failure_site: str
) -> None:
    monkeypatch.setenv("SC62015_CPU_BACKEND", "python")
    emu = PCE500Emulator(perfetto_trace=False, save_lcd_on_exit=False)
    emu._timer_enabled = False  # type: ignore[attr-defined]
    emu.cpu.regs.set(RegisterName.PC, 0x1000)
    emu.memory.write_byte(0x1000, 0x00)
    emu.cpu.state.halted = True
    emu.cpu.state.power_state = "halted"

    if failure_site == "scan":
        monkeypatch.setattr(
            emu,
            "_scan_keyboard_per_instruction",
            lambda: (_ for _ in ()).throw(RuntimeError("scan failed")),
        )
    else:
        original_read = emu.memory.read_byte
        isr_addr = INTERNAL_MEMORY_START + IMEMRegisters.ISR

        def fail_isr_read(address: int, *args, **kwargs):
            if address == isr_addr:
                raise RuntimeError("ISR read failed")
            return original_read(address, *args, **kwargs)

        monkeypatch.setattr(emu.memory, "read_byte", fail_isr_read)

    with pytest.raises(RuntimeError, match="failed"):
        emu.step()

    assert emu.cpu.state.halted is True
    assert emu.cpu.state.power_state == "halted"
    assert emu.cpu.regs.get(RegisterName.PC) == 0x1000


def _arm_key_irq_for_delivery(
    emu: PCE500Emulator, *, stack: int = 0x0200, install_vectors: bool = True
) -> None:
    if install_vectors:
        install_static_machine_vectors(emu)
    emu._timer_enabled = False  # type: ignore[attr-defined]
    emu._in_interrupt = False  # type: ignore[attr-defined]
    emu._irq_pending = True  # type: ignore[attr-defined]
    emu._irq_source = IRQSource.KEY  # type: ignore[attr-defined]
    emu.cpu.regs.set(RegisterName.PC, 0x1000)
    emu.cpu.regs.set(RegisterName.S, stack)
    emu.memory.write_byte(0x1000, 0x00)  # NOP
    emu.memory.write_byte(
        INTERNAL_MEMORY_START + IMEMRegisters.IMR,
        int(IMRFlag.IRM | IMRFlag.KEYM),
    )
    emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR, int(ISRFlag.KEYI))


def test_pending_irq_functional_read_failure_never_executes_opcode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SC62015_CPU_BACKEND", "python")
    emu = PCE500Emulator(perfetto_trace=False, save_lcd_on_exit=False)
    _arm_key_irq_for_delivery(emu)

    original_read = emu.memory.read_byte
    imr_addr = INTERNAL_MEMORY_START + IMEMRegisters.IMR
    imr_reads = 0

    def fail_delivery_imr_read(address: int, *args, **kwargs):
        nonlocal imr_reads
        if address == imr_addr:
            imr_reads += 1
            if imr_reads == 2:
                raise RuntimeError("delivery IMR read failed")
        return original_read(address, *args, **kwargs)

    monkeypatch.setattr(emu.memory, "read_byte", fail_delivery_imr_read)

    with pytest.raises(RuntimeError, match="delivery IMR read failed"):
        emu.step()

    assert emu.cpu.regs.get(RegisterName.PC) == 0x1000
    assert emu.cpu.regs.get(RegisterName.S) == 0x0200
    assert emu.instruction_count == 0
    assert emu._irq_pending is True  # type: ignore[attr-defined]
    assert emu._poisoned is None  # type: ignore[attr-defined]


def test_partial_irq_frame_failure_poisons_until_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SC62015_CPU_BACKEND", "python")
    emu = PCE500Emulator(perfetto_trace=False, save_lcd_on_exit=False)
    _arm_key_irq_for_delivery(emu)

    original_write_bytes = emu.memory.write_bytes
    writes = 0

    def fail_second_frame_write(length: int, address: int, value: int) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise RuntimeError("interrupt frame write failed")
        original_write_bytes(length, address, value)

    monkeypatch.setattr(emu.memory, "write_bytes", fail_second_frame_write)

    with pytest.raises(RuntimeError, match="interrupt frame write failed"):
        emu.step()

    assert emu.cpu.regs.get(RegisterName.PC) == 0x1000
    # PC high/middle/low commit first, followed by F.  The injected failure is
    # the IMR frame write, so four bytes have observably reached the stack.
    assert emu.cpu.regs.get(RegisterName.S) == 0x01FC
    assert emu.instruction_count == 0
    assert "IRQ delivery failed" in str(emu._poisoned)  # type: ignore[attr-defined]

    monkeypatch.setattr(emu.memory, "write_bytes", original_write_bytes)
    with pytest.raises(RuntimeError, match="poisoned.*reset required"):
        emu.step()

    emu.reset()
    assert emu._poisoned is None  # type: ignore[attr-defined]


def test_pending_irq_stack_underflow_wraps_in_20_bit_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SC62015_CPU_BACKEND", "python")
    emu = PCE500Emulator(perfetto_trace=False, save_lcd_on_exit=False)
    _arm_key_irq_for_delivery(emu, stack=2, install_vectors=False)
    handler = 0x23456
    # Keep the wrapped frame bytes distinct from the PCE memory shim's compact
    # internal-register backing, and supply the vector independently.  A real
    # ROM overlay provides the same separation in full-machine runs.
    emu.memory.add_ram(0xFFFFD, 3, "wrapped_irq_frame")
    emu.memory.add_rom(
        0xFFFFA,
        handler.to_bytes(3, "little"),
        "wrapped_stack_irq_vector",
    )
    emu.memory.add_rom(handler, b"\x00", "wrapped_stack_irq_target")
    external_writes: list[int] = []
    original_write = emu.memory.write_byte

    def record_write(address: int, value: int, *args, **kwargs) -> None:
        canonical = address & 0xFFFFFF
        if canonical < 0x100000:
            external_writes.append(canonical & 0xFFFFF)
        original_write(address, value, *args, **kwargs)

    monkeypatch.setattr(emu.memory, "write_byte", record_write)

    emu.step()

    assert external_writes[:5] == [0x00001, 0x00000, 0xFFFFF, 0xFFFFE, 0xFFFFD]
    assert emu.memory.read_byte(0xFFFFF) == 0x00  # saved PC low
    assert emu.memory.read_byte(0x00000) == 0x10  # saved PC middle
    assert emu.memory.read_byte(0x00001) == 0x00  # saved PC high
    assert emu.memory.read_byte(0xFFFFE) == 0x00  # saved modeled F
    assert emu.memory.read_byte(0xFFFFD) == int(IMRFlag.IRM | IMRFlag.KEYM)
    assert emu.cpu.regs.get(RegisterName.PC) == handler + 1
    assert emu.cpu.regs.get(RegisterName.S) == 0xFFFFD
    assert emu.instruction_count == 1
    assert emu._irq_pending is False  # type: ignore[attr-defined]
    assert emu._in_interrupt is True  # type: ignore[attr-defined]
    assert emu._poisoned is None  # type: ignore[attr-defined]


def test_pre_instruction_timer_failure_poisons_without_executing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SC62015_CPU_BACKEND", "python")
    emu = PCE500Emulator(perfetto_trace=False, save_lcd_on_exit=False)
    emu._timer_enabled = True  # type: ignore[attr-defined]
    emu.cpu.regs.set(RegisterName.PC, 0x1000)
    emu.memory.write_byte(0x1000, 0x00)  # NOP
    original_tick = emu._tick_timers  # type: ignore[attr-defined]

    def fail_timer_tick() -> None:
        raise RuntimeError("pre-instruction timer failed")

    monkeypatch.setattr(emu, "_tick_timers", fail_timer_tick)

    with pytest.raises(RuntimeError, match="pre-instruction timer failed"):
        emu.step()

    assert emu.cpu.regs.get(RegisterName.PC) == 0x1000
    assert emu.instruction_count == 0
    assert "timer tick failed" in str(emu._poisoned)  # type: ignore[attr-defined]

    monkeypatch.setattr(emu, "_tick_timers", original_tick)
    with pytest.raises(RuntimeError, match="poisoned.*reset required"):
        emu.step()

    install_static_machine_vectors(emu)
    emu.reset()
    emu._timer_enabled = False  # type: ignore[attr-defined]
    emu.cpu.regs.set(RegisterName.PC, 0x1000)
    emu.step()
    assert emu.cpu.regs.get(RegisterName.PC) == 0x1001


# Focused regression: ensure reset fully reinitializes timers and IRQ state
def test_reset_reinitializes_timers_and_interrupt_state() -> None:
    emu = PCE500Emulator(perfetto_trace=False, save_lcd_on_exit=False)
    install_static_machine_vectors(emu)

    # Simulate in-flight timer and interrupt state before reset
    emu._timer_next_mti = 12345  # type: ignore[attr-defined]
    emu._timer_next_sti = 67890  # type: ignore[attr-defined]
    emu._irq_pending = True  # type: ignore[attr-defined]
    emu._irq_source = IRQSource.KEY  # type: ignore[attr-defined]
    emu._in_interrupt = True  # type: ignore[attr-defined]
    emu._interrupt_stack.append(99)  # type: ignore[attr-defined]
    emu._next_interrupt_id = 42  # type: ignore[attr-defined]
    emu._kb_irq_count = 5  # type: ignore[attr-defined]

    emu.reset()

    assert emu._timer_next_mti == emu._timer_mti_period  # type: ignore[attr-defined]
    assert emu._timer_next_sti == emu._timer_sti_period  # type: ignore[attr-defined]
    assert emu._irq_pending is False  # type: ignore[attr-defined]
    assert emu._irq_source is None  # type: ignore[attr-defined]
    assert emu._in_interrupt is False  # type: ignore[attr-defined]
    assert emu._interrupt_stack == []  # type: ignore[attr-defined]
    assert emu._next_interrupt_id == 1  # type: ignore[attr-defined]
    assert emu._kb_irq_count == 0  # type: ignore[attr-defined]


# ----------------------------- The test -----------------------------


def _ids(specs: Iterable[InterruptSpec]) -> list[str]:
    return [s.given.name for s in specs]


@pytest.mark.parametrize("spec", SPECS, ids=_ids(SPECS))
def test_interrupts_typed_inline_snapshots(spec: InterruptSpec) -> None:
    # Run scenario and observe
    actual = _run_and_observe(spec.given, spec.expect)

    # THEN: contract assertions (fast, crisp)
    _assert_contract(spec.given, spec.expect, actual)

    # THEN: inline golden snapshot comparison (strongly-typed)
    _assert_observed_equals(spec.observed, actual, spec.given.name)
