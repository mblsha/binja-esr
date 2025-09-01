"""Dataclass-driven interrupt tests for HALT, WAIT, keyboard, ON-key, and timers.

Everything flows through a single parameterized test. Scenarios describe
triggers, IMR masks, program shape, and expected delivery succinctly.
"""

from dataclasses import dataclass
from textwrap import dedent
import pytest
from enum import Enum, auto

from sc62015.pysc62015.sc_asm import Assembler
from sc62015.pysc62015.emulator import RegisterName
from sc62015.pysc62015.instr.opcodes import IMEMRegisters

from pce500 import PCE500Emulator


INTERNAL_MEMORY_START = 0x100000


class Trigger(Enum):
    KEY_F1 = "KEY_F1"
    KEY_ON = "KEY_ON"
    KEY_OFF = "KEY_OFF"
    MTI = "MTI"  # Main timer
    STI = "STI"  # Sub timer
    NONE = "NONE"  # No source


class Program(Enum):
    HALT = auto()
    WAIT = auto()
    OFF = auto()


TRIGGER_MASK = {
    Trigger.MTI: 0x01,
    Trigger.STI: 0x02,
    Trigger.KEY_F1: 0x04,
    Trigger.KEY_ON: 0x08,
}


@dataclass(frozen=True)
class ProgramConfig:
    entry: int
    handler: int
    marker: int = 0x42


PROGRAM = ProgramConfig(entry=0xB8000, handler=0xB9000)


# Reusable assembly templates
ENTRY_HALT = """
.ORG 0x{entry:05X}
entry:
    HALT
    NOP
"""

ENTRY_WAIT = """
.ORG 0x{entry:05X}
entry:
    MV I, 0x{wait_count:04X}
    WAIT
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
    RETI
"""


@dataclass(frozen=True)
class InterruptScenario:
    name: str
    trigger: Trigger
    imr: int
    program: Program  # HALT or WAIT
    # WAIT-only: relative offset from period in cycles (positive → after; negative → before)
    wait_delta: int | None = None
    # Concise expectations
    expect_deliver: bool = False
    expect_halt_canceled: bool | None = None  # Only meaningful for HALT program
    # Harness knobs
    isolate_timers: bool = False
    timers_enabled: bool | None = None  # HALT: override default (None→disable timers)


SCENARIOS: list[InterruptScenario] = [
    # HALT + keyboard/ON
    InterruptScenario(
        "key_unmasked",
        Trigger.KEY_F1,
        0x80 | 0x04,
        Program.HALT,
        expect_deliver=True,
        expect_halt_canceled=True,
    ),
    InterruptScenario(
        "key_masked",
        Trigger.KEY_F1,
        0x80,
        Program.HALT,
        expect_deliver=False,
        expect_halt_canceled=True,
    ),
    InterruptScenario(
        "on_unmasked",
        Trigger.KEY_ON,
        0x80 | 0x08,
        Program.HALT,
        expect_deliver=True,
        expect_halt_canceled=True,
    ),
    InterruptScenario(
        "on_masked",
        Trigger.KEY_ON,
        0x80,
        Program.HALT,
        expect_deliver=False,
        expect_halt_canceled=True,
    ),
    InterruptScenario(
        "no_trigger_halts",
        Trigger.NONE,
        0x80,
        Program.HALT,
        expect_deliver=False,
        expect_halt_canceled=False,
    ),
    # WAIT + timers
    InterruptScenario(
        "mti_unmasked",
        Trigger.MTI,
        0x80 | 0x01,
        Program.WAIT,
        wait_delta=50,
        expect_deliver=True,
        isolate_timers=True,
    ),
    InterruptScenario(
        "mti_masked",
        Trigger.MTI,
        0x80,
        Program.WAIT,
        wait_delta=50,
        expect_deliver=False,
        isolate_timers=True,
    ),
    InterruptScenario(
        "mti_unmasked_not_enough",
        Trigger.MTI,
        0x80 | 0x01,
        Program.WAIT,
        wait_delta=-20,
        expect_deliver=False,
        isolate_timers=True,
    ),
    InterruptScenario(
        "sti_unmasked",
        Trigger.STI,
        0x80 | 0x02,
        Program.WAIT,
        wait_delta=200,
        expect_deliver=True,
        isolate_timers=True,
    ),
    InterruptScenario(
        "sti_masked",
        Trigger.STI,
        0x80,
        Program.WAIT,
        wait_delta=200,
        expect_deliver=False,
        isolate_timers=True,
    ),
    InterruptScenario(
        "sti_unmasked_not_enough",
        Trigger.STI,
        0x80 | 0x02,
        Program.WAIT,
        wait_delta=-50,
        expect_deliver=False,
        isolate_timers=True,
    ),
    # HALT + timers (verify timers wake and deliver when masked in IMR)
    InterruptScenario(
        "halt_mti_unmasked",
        Trigger.MTI,
        0x80 | 0x01,
        Program.HALT,
        expect_deliver=True,
        expect_halt_canceled=True,
        isolate_timers=True,
        timers_enabled=True,
    ),
    InterruptScenario(
        "halt_sti_unmasked",
        Trigger.STI,
        0x80 | 0x02,
        Program.HALT,
        expect_deliver=True,
        expect_halt_canceled=True,
        isolate_timers=True,
        timers_enabled=True,
    ),
    # OFF-state behavior
    InterruptScenario(
        "off_no_exec_on_longest_timer",
        Trigger.NONE,
        0x80,
        Program.OFF,
        expect_deliver=False,
        expect_halt_canceled=False,
    ),
    InterruptScenario(
        "off_on_unmasked_wakes_and_delivers",
        Trigger.KEY_ON,
        0x80 | 0x08,
        Program.OFF,
        expect_deliver=True,
        expect_halt_canceled=True,
    ),
    InterruptScenario(
        "off_press_off_key_no_wake",
        Trigger.KEY_OFF,
        0x80,
        Program.OFF,
        expect_deliver=False,
        expect_halt_canceled=False,
    ),
]


def _assemble_program(
    emu: PCE500Emulator, program: Program, wait_count: int | None
) -> None:
    asm = Assembler()
    if program is Program.HALT:
        entry = ENTRY_HALT.format(entry=PROGRAM.entry)
    elif program is Program.WAIT:
        entry = ENTRY_WAIT.format(entry=PROGRAM.entry, wait_count=wait_count or 0)
    else:
        entry = ENTRY_OFF.format(entry=PROGRAM.entry)
    handler = HANDLER_TEMPLATE.format(handler=PROGRAM.handler, marker=PROGRAM.marker)
    source = dedent(entry + handler)
    binfile = asm.assemble(source)
    for seg in binfile.segments:
        for i, b in enumerate(seg.data):
            emu.memory.write_byte(seg.address + i, b)
    # Interrupt vector → handler
    rom = bytearray(b"\xff" * 0x40000)
    off = 0x3FFFA
    vec = PROGRAM.handler & 0xFFFFF
    rom[off : off + 3] = bytes([vec & 0xFF, (vec >> 8) & 0xFF, (vec >> 16) & 0xFF])
    emu.load_rom(bytes(rom))


def _isolate_other_timer(emu: PCE500Emulator, trig: Trigger) -> None:
    if trig == Trigger.MTI:
        emu._timer_sti_period = 10**9  # type: ignore[attr-defined]
        emu._timer_next_sti = emu.cycle_count + emu._timer_sti_period  # type: ignore[attr-defined]
    elif trig == Trigger.STI:
        emu._timer_mti_period = 10**9  # type: ignore[attr-defined]
        emu._timer_next_mti = emu.cycle_count + emu._timer_mti_period  # type: ignore[attr-defined]


@pytest.mark.parametrize("sc", SCENARIOS, ids=[s.name for s in SCENARIOS])
def test_interrupts(sc: InterruptScenario) -> None:
    emu = PCE500Emulator(perfetto_trace=False, save_lcd_on_exit=False)

    if sc.isolate_timers:
        emu._timer_enabled = True  # type: ignore[attr-defined]
        _isolate_other_timer(emu, sc.trigger)

    # WAIT: compute wait_count = period + delta
    wait_count = None
    if sc.program is Program.WAIT:
        period = (
            int(getattr(emu, "_timer_mti_period", 500))
            if sc.trigger == Trigger.MTI
            else int(getattr(emu, "_timer_sti_period", 5000))
        )
        wait_count = max(0, period + int(sc.wait_delta or 0))

    _assemble_program(emu, sc.program, wait_count)

    # Initialize CPU/internal registers
    emu.cpu.regs.set(RegisterName.PC, PROGRAM.entry)
    emu.cpu.regs.set(RegisterName.S, 0xBFF00)
    emu.cpu.regs.set(RegisterName.U, 0xBFE00)
    emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR, 0x00)
    emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.IMR, sc.imr)

    # HALT/OFF prelude
    off_pc_after_off: int | None = None
    if sc.program in (Program.HALT, Program.OFF):
        # For OFF: timers never work; for HALT: allow override
        if sc.program is Program.OFF:
            timers_on = False
        else:
            timers_on = False if sc.timers_enabled is None else bool(sc.timers_enabled)
        emu._timer_enabled = timers_on  # type: ignore[attr-defined]
        emu.step()  # HALT/OFF
        assert emu.cpu.state.halted is True
        if sc.program is Program.OFF:
            off_pc_after_off = emu.cpu.regs.get(RegisterName.PC)
        # Stable halt check only when timers are disabled
        if not timers_on:
            u_stable = emu.cpu.regs.get(RegisterName.U)
            for _ in range(3):
                emu.step()
                assert emu.cpu.state.halted is True
                assert emu.cpu.regs.get(RegisterName.U) == u_stable

    u_before = emu.cpu.regs.get(RegisterName.U)

    # Trigger
    if sc.trigger in (Trigger.KEY_F1, Trigger.KEY_ON):
        assert emu.press_key(sc.trigger.value) is True
    elif sc.trigger is Trigger.KEY_OFF:
        # OFF key may not be part of matrix; just attempt without asserting
        emu.press_key(sc.trigger.value)

    # Execute program to the point of delivery
    if sc.program in (Program.HALT, Program.OFF):
        if sc.timers_enabled:
            # Let timers advance enough cycles to fire
            if sc.trigger is Trigger.MTI:
                period = int(getattr(emu, "_timer_mti_period", 500))
                steps = period + 50
            elif sc.trigger is Trigger.STI:
                period = int(getattr(emu, "_timer_sti_period", 5000))
                steps = period + 200
            else:
                steps = 24
            for _ in range(int(steps)):
                emu.step()
        else:
            # For OFF, explicitly run enough steps to exceed the longest timer period
            if sc.program is Program.OFF:
                mti = int(getattr(emu, "_timer_mti_period", 500))
                sti = int(getattr(emu, "_timer_sti_period", 5000))
                longest = max(mti, sti)
                for _ in range(int(longest)):
                    emu.step()
            else:
                for _ in range(24):
                    emu.step()
    else:
        emu.step()  # MV I
        emu.step()  # WAIT
        emu.step()  # Attempt delivery
        if sc.expect_deliver:
            for _ in range(5):  # Full handler
                emu.step()
        else:
            emu.step()  # NOP

    u_after = emu.cpu.regs.get(RegisterName.U)
    expected_u_delta = -2 if sc.expect_deliver else 0
    assert u_after - u_before == expected_u_delta

    if (
        sc.program in (Program.HALT, Program.OFF)
        and sc.expect_halt_canceled is not None
    ):
        assert emu.cpu.state.halted == (not sc.expect_halt_canceled)

    if sc.expect_deliver:
        mask = TRIGGER_MASK.get(sc.trigger, 0)
        assert (emu.memory.read_byte(u_after) & mask) == mask
        assert emu.memory.read_byte(u_after + 1) == PROGRAM.marker
        assert emu.last_irq.get("src") is not None
    else:
        assert emu.last_irq.get("src") in (None, "")
        # In OFF, ensure PC did not advance beyond OFF if no delivery
        if sc.program is Program.OFF:
            pc_now = emu.cpu.regs.get(RegisterName.PC)
            assert pc_now == (
                off_pc_after_off if off_pc_after_off is not None else PROGRAM.entry
            )
