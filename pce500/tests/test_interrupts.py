"""Dataclass-driven interrupt tests for HALT, WAIT, keyboard, ON-key, and timers.

All cases flow through a single parameterized test with concise, declarative
scenarios. Program assembly uses reusable entry/handler templates.
"""

from dataclasses import dataclass
from textwrap import dedent
import pytest
from enum import Enum

from sc62015.pysc62015.sc_asm import Assembler
from sc62015.pysc62015.emulator import RegisterName
from sc62015.pysc62015.instr.opcodes import IMEMRegisters

from pce500 import PCE500Emulator


INTERNAL_MEMORY_START = 0x100000


class Trigger(Enum):
    KEY_F1 = "KEY_F1"
    KEY_ON = "KEY_ON"
    MTI = "MTI"  # Main timer
    STI = "STI"  # Sub timer
    NONE = "NONE"  # No trigger


@dataclass(frozen=True)
class ProgramConfig:
    entry: int
    handler: int
    marker: int = 0x42


PROGRAM = ProgramConfig(entry=0xB8000, handler=0xB9000)


# Assembly templates
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
    mode: str  # "HALT" or "WAIT"
    # WAIT-only configuration: relative offset from period
    wait_extra: int | None = None
    # Expectations
    expect_u_delta: int = 0
    expect_isr_mask: int | None = None
    expect_halt_canceled: bool | None = None  # Only relevant for HALT
    # Test harness knobs
    isolate_timers: bool = False


SCENARIOS: list[InterruptScenario] = [
    # HALT + keyboard/ON
    InterruptScenario(
        name="key_unmasked",
        trigger=Trigger.KEY_F1,
        imr=0x80 | 0x04,  # IRM=1, KEYM=1
        mode="HALT",
        expect_u_delta=-2,
        expect_isr_mask=0x04,
        expect_halt_canceled=True,
    ),
    InterruptScenario(
        name="key_masked",
        trigger=Trigger.KEY_F1,
        imr=0x80,  # IRM=1, KEYM=0 → HALT cancels but no delivery
        mode="HALT",
        expect_u_delta=0,
        expect_isr_mask=None,
        expect_halt_canceled=True,
    ),
    InterruptScenario(
        name="on_unmasked",
        trigger=Trigger.KEY_ON,
        imr=0x80 | 0x08,  # IRM=1, ONKM=1
        mode="HALT",
        expect_u_delta=-2,
        expect_isr_mask=0x08,
        expect_halt_canceled=True,
    ),
    InterruptScenario(
        name="on_masked",
        trigger=Trigger.KEY_ON,
        imr=0x80,  # IRM=1, ONKM=0 → HALT cancels but no delivery
        mode="HALT",
        expect_u_delta=0,
        expect_isr_mask=None,
        expect_halt_canceled=True,
    ),
    InterruptScenario(
        name="no_trigger_halts",
        trigger=Trigger.NONE,
        imr=0x80,
        mode="HALT",
        expect_u_delta=0,
        expect_isr_mask=None,
        expect_halt_canceled=False,
    ),
    # WAIT + timers (unmasked, masked, and not-enough)
    InterruptScenario(
        name="mti_unmasked",
        trigger=Trigger.MTI,
        imr=0x80 | 0x01,  # IRM=1, MTM=1
        mode="WAIT",
        wait_extra=50,
        expect_u_delta=-2,
        expect_isr_mask=0x01,
        isolate_timers=True,
    ),
    InterruptScenario(
        name="mti_masked",
        trigger=Trigger.MTI,
        imr=0x80,  # IRM=1, MTM=0
        mode="WAIT",
        wait_extra=50,
        expect_u_delta=0,
        expect_isr_mask=None,
        isolate_timers=True,
    ),
    InterruptScenario(
        name="mti_unmasked_not_enough",
        trigger=Trigger.MTI,
        imr=0x80 | 0x01,
        mode="WAIT",
        wait_extra=-20,
        expect_u_delta=0,
        expect_isr_mask=None,
        isolate_timers=True,
    ),
    InterruptScenario(
        name="sti_unmasked",
        trigger=Trigger.STI,
        imr=0x80 | 0x02,  # IRM=1, STM=1
        mode="WAIT",
        wait_extra=200,
        expect_u_delta=-2,
        expect_isr_mask=0x02,
        isolate_timers=True,
    ),
    InterruptScenario(
        name="sti_masked",
        trigger=Trigger.STI,
        imr=0x80,  # IRM=1, STM=0
        mode="WAIT",
        wait_extra=200,
        expect_u_delta=0,
        expect_isr_mask=None,
        isolate_timers=True,
    ),
    InterruptScenario(
        name="sti_unmasked_not_enough",
        trigger=Trigger.STI,
        imr=0x80 | 0x02,
        mode="WAIT",
        wait_extra=-50,
        expect_u_delta=0,
        expect_isr_mask=None,
        isolate_timers=True,
    ),
]


def _assemble_program(emu: PCE500Emulator, mode: str, wait_count: int | None) -> None:
    asm = Assembler()
    entry = (
        ENTRY_HALT.format(entry=PROGRAM.entry)
        if mode == "HALT"
        else ENTRY_WAIT.format(entry=PROGRAM.entry, wait_count=wait_count or 0)
    )
    handler = HANDLER_TEMPLATE.format(handler=PROGRAM.handler, marker=PROGRAM.marker)
    source = dedent(entry + handler)
    binfile = asm.assemble(source)
    for seg in binfile.segments:
        for i, b in enumerate(seg.data):
            emu.memory.write_byte(seg.address + i, b)
    # Interrupt vector -> handler
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
    # Build emulator; enable WAIT timing by default
    emu = PCE500Emulator(perfetto_trace=False, save_lcd_on_exit=False)

    # Optional timer isolation for WAIT scenarios
    if sc.isolate_timers:
        emu._timer_enabled = True  # type: ignore[attr-defined]
        _isolate_other_timer(emu, sc.trigger)

    # Compute wait_count from period+extra when needed
    wait_count = None
    if sc.mode == "WAIT":
        period = (
            int(getattr(emu, "_timer_mti_period", 500))
            if sc.trigger == Trigger.MTI
            else int(getattr(emu, "_timer_sti_period", 5000))
        )
        wait_count = max(0, period + int(sc.wait_extra or 0))

    _assemble_program(emu, sc.mode, wait_count)

    # Init CPU state
    emu.cpu.regs.set(RegisterName.PC, PROGRAM.entry)
    emu.cpu.regs.set(RegisterName.S, 0xBFF00)
    emu.cpu.regs.set(RegisterName.U, 0xBFE00)
    emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR, 0x00)
    emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.IMR, sc.imr)

    # HALT-specific prelude
    if sc.mode == "HALT":
        # To focus on keyboard/ON behavior, disable periodic timers
        emu._timer_enabled = False  # type: ignore[attr-defined]
        emu.step()  # Execute HALT
        assert emu.cpu.state.halted is True
        # Confirm stable halt
        u_stable = emu.cpu.regs.get(RegisterName.U)
        for _ in range(3):
            emu.step()
            assert emu.cpu.state.halted is True
            assert emu.cpu.regs.get(RegisterName.U) == u_stable

    u_before = emu.cpu.regs.get(RegisterName.U)

    # Trigger
    if sc.trigger == Trigger.KEY_F1 or sc.trigger == Trigger.KEY_ON:
        assert emu.press_key(sc.trigger.value) is True
    elif sc.trigger == Trigger.MTI or sc.trigger == Trigger.STI:
        # Nothing to do; WAIT controls timing
        pass
    elif sc.trigger == Trigger.NONE:
        # No source
        pass

    # Execute
    if sc.mode == "HALT":
        # Allow pending IRQ path and handler to run if applicable
        for _ in range(24):
            emu.step()
    else:
        # WAIT path: MV I; WAIT; then either deliver+handler or execute NOP
        emu.step()  # MV I
        emu.step()  # WAIT
        emu.step()  # Delivery attempt
        if sc.expect_isr_mask is not None:
            for _ in range(5):
                emu.step()
        else:
            emu.step()  # NOP

    u_after = emu.cpu.regs.get(RegisterName.U)
    assert u_after - u_before == sc.expect_u_delta

    if sc.mode == "HALT" and sc.expect_halt_canceled is not None:
        assert emu.cpu.state.halted == (not sc.expect_halt_canceled)

    if sc.expect_isr_mask is not None:
        # Top of U is ISR value; next is marker
        assert (
            emu.memory.read_byte(u_after) & sc.expect_isr_mask
        ) == sc.expect_isr_mask
        assert emu.memory.read_byte(u_after + 1) == PROGRAM.marker
        assert emu.last_irq.get("src") is not None
    else:
        assert emu.last_irq.get("src") in (None, "")


# -------- WAIT-based timer tests (dataclass-driven) --------


@dataclass(frozen=True)
class TimerScenario:
    name: str
    imr: int
    trigger: Trigger  # MTI or STI
    wait_extra: int  # extra instructions beyond period
    expect_isr_mask: int | None


TIMER_SCENARIOS: list[TimerScenario] = [
    TimerScenario(
        name="mti_unmasked",
        imr=0x80 | 0x01,
        trigger=Trigger.MTI,
        wait_extra=50,
        expect_isr_mask=0x01,
    ),
    TimerScenario(
        name="mti_masked",
        imr=0x80,
        trigger=Trigger.MTI,
        wait_extra=50,
        expect_isr_mask=None,
    ),
    TimerScenario(
        name="sti_unmasked",
        imr=0x80 | 0x02,
        trigger=Trigger.STI,
        wait_extra=200,
        expect_isr_mask=0x02,
    ),
    TimerScenario(
        name="sti_masked",
        imr=0x80,
        trigger=Trigger.STI,
        wait_extra=200,
        expect_isr_mask=None,
    ),
]


def _assemble_wait_program(emu: PCE500Emulator, wait_count: int) -> None:
    asm = Assembler()
    source = dedent(
        f"""
        .ORG 0x{PROGRAM.entry:05X}
        entry:
            MV I, 0x{wait_count:04X}
            WAIT
            NOP

        .ORG 0x{PROGRAM.handler:05X}
        handler:
            MV A, 0x{PROGRAM.marker:02X}
            PUSHU A
            MV A, (ISR)
            PUSHU A
            RETI
        """
    )
    binfile = asm.assemble(source)
    for seg in binfile.segments:
        for i, b in enumerate(seg.data):
            emu.memory.write_byte(seg.address + i, b)
    # Interrupt vector
    rom = bytearray(b"\xff" * 0x40000)
    off = 0x3FFFA
    vec = PROGRAM.handler & 0xFFFFF
    rom[off : off + 3] = bytes([vec & 0xFF, (vec >> 8) & 0xFF, (vec >> 16) & 0xFF])
    emu.load_rom(bytes(rom))


def _isolate_other_timer(emu: PCE500Emulator, trigger: Trigger) -> None:
    # Push the non-target timer far into the future to avoid cross-triggering
    if trigger == Trigger.MTI:
        emu._timer_sti_period = 10**9  # type: ignore[attr-defined]
        emu._timer_next_sti = emu.cycle_count + emu._timer_sti_period  # type: ignore[attr-defined]
    elif trigger == Trigger.STI:
        emu._timer_mti_period = 10**9  # type: ignore[attr-defined]
        emu._timer_next_mti = emu.cycle_count + emu._timer_mti_period  # type: ignore[attr-defined]


@pytest.mark.parametrize("ts", TIMER_SCENARIOS, ids=[t.name for t in TIMER_SCENARIOS])
@pytest.mark.timeout(10)
def test_wait_timer_interrupts(ts: TimerScenario) -> None:
    emu = PCE500Emulator(
        perfetto_trace=False, save_lcd_on_exit=False, enable_new_tracing=True
    )
    emu._timer_enabled = True  # type: ignore[attr-defined]
    _isolate_other_timer(emu, ts.trigger)

    # Determine period and compute wait_count
    if ts.trigger == Trigger.MTI:
        period = int(getattr(emu, "_timer_mti_period", 500))
        mask_bit = 0x01
    else:
        period = int(getattr(emu, "_timer_sti_period", 5000))
        mask_bit = 0x02
    wait_count = period + ts.wait_extra

    _assemble_wait_program(emu, wait_count)

    # Init state
    emu.cpu.regs.set(RegisterName.PC, PROGRAM.entry)
    emu.cpu.regs.set(RegisterName.S, 0xBFF00)
    emu.cpu.regs.set(RegisterName.U, 0xBFE00)
    emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR, 0x00)
    emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.IMR, ts.imr)

    u_before = emu.cpu.regs.get(RegisterName.U)

    # Execute: MV I; WAIT; delivery; then either handler (5 steps) or NOP
    emu.step()  # MV I
    emu.step()  # WAIT
    emu.step()  # Delivery attempt

    if ts.expect_isr_mask is not None:
        for _ in range(5):
            emu.step()
        u_after = emu.cpu.regs.get(RegisterName.U)
        assert u_after - u_before == -2
        assert (emu.memory.read_byte(u_after) & mask_bit) == mask_bit
        assert emu.memory.read_byte(u_after + 1) == PROGRAM.marker
        assert emu.last_irq.get("src") is not None
    else:
        # Masked: should execute NOP next, no delivery
        emu.step()
        u_after = emu.cpu.regs.get(RegisterName.U)
        assert u_after == u_before
        assert emu.last_irq.get("src") in (None, "")


def _common_setup_with_halt(emu: PCE500Emulator) -> int:
    """Assemble program, reset, execute HALT and return U before value."""
    emu.reset()
    assemble_and_load(emu, PROGRAM)
    emu.cpu.regs.set(RegisterName.PC, PROGRAM.entry)
    emu.cpu.regs.set(RegisterName.S, 0xBFF00)
    emu.cpu.regs.set(RegisterName.U, 0xBFE00)
    # Execute HALT
    emu.step()
    assert emu.cpu.state.halted is True
    return emu.cpu.regs.get(RegisterName.U)


@pytest.mark.timeout(10)
def test_timer_mti_unmasked_wakes_and_delivers() -> None:
    """MTI (bit 0) fires after WAIT-ing enough instructions and delivers when unmasked."""
    emu = PCE500Emulator(
        perfetto_trace=False, save_lcd_on_exit=False, enable_new_tracing=True
    )
    # Isolate MTI by pushing STI far away
    emu._timer_enabled = True  # type: ignore[attr-defined]
    emu._timer_sti_period = 10**9  # type: ignore[attr-defined]
    emu._timer_next_sti = emu.instruction_count + emu._timer_sti_period  # type: ignore[attr-defined]

    asm = Assembler()
    mti_period = int(getattr(emu, "_timer_mti_period", 500))
    wait_count = mti_period + 50
    source = dedent(
        f"""
        .ORG 0x{PROGRAM.entry:05X}
        entry:
            MV I, 0x{wait_count:04X}
            WAIT
            NOP

        .ORG 0x{PROGRAM.handler:05X}
        handler:
            MV A, 0x{PROGRAM.marker:02X}
            PUSHU A
            MV A, (ISR)
            PUSHU A
            RETI
        """
    )
    binfile = asm.assemble(source)
    for seg in binfile.segments:
        for i, b in enumerate(seg.data):
            emu.memory.write_byte(seg.address + i, b)
    rom = bytearray(b"\xff" * 0x40000)
    off = 0x3FFFA
    vec = PROGRAM.handler & 0xFFFFF
    rom[off : off + 3] = bytes([vec & 0xFF, (vec >> 8) & 0xFF, (vec >> 16) & 0xFF])
    emu.load_rom(bytes(rom))
    emu.cpu.regs.set(RegisterName.PC, PROGRAM.entry)
    emu.cpu.regs.set(RegisterName.S, 0xBFF00)
    emu.cpu.regs.set(RegisterName.U, 0xBFE00)
    emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR, 0x00)
    emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.IMR, 0x80 | 0x01)

    u_before = emu.cpu.regs.get(RegisterName.U)
    emu.step()  # MV I
    emu.step()  # MV I+1
    emu.step()  # WAIT → sets ISR[0]
    emu.step()  # Deliver
    # Execute full handler: MV A; PUSHU; MV A,(ISR); PUSHU; RETI
    for _ in range(5):
        emu.step()
    u_after = emu.cpu.regs.get(RegisterName.U)
    assert u_after - u_before == -2
    assert (emu.memory.read_byte(u_after) & 0x01) == 0x01
    assert emu.memory.read_byte(u_after + 1) == PROGRAM.marker


@pytest.mark.timeout(10)
def test_timer_mti_masked_wakes_but_no_delivery() -> None:
    """MTI masked: WAIT sets ISR[0]; no delivery when masked."""
    emu = PCE500Emulator(
        perfetto_trace=False, save_lcd_on_exit=False, enable_new_tracing=True
    )
    emu._timer_enabled = True  # type: ignore[attr-defined]
    emu._timer_sti_period = 10**9  # type: ignore[attr-defined]
    emu._timer_next_sti = emu.instruction_count + emu._timer_sti_period  # type: ignore[attr-defined]

    asm = Assembler()
    mti_period = int(getattr(emu, "_timer_mti_period", 500))
    wait_count = mti_period + 50
    source = dedent(
        f"""
        .ORG 0x{PROGRAM.entry:05X}
        entry:
            MV I, 0x{wait_count:04X}
            WAIT
            NOP

        .ORG 0x{PROGRAM.handler:05X}
        handler:
            MV A, 0x{PROGRAM.marker:02X}
            PUSHU A
            MV A, (ISR)
            PUSHU A
            RETI
        """
    )
    binfile = asm.assemble(source)
    for seg in binfile.segments:
        for i, b in enumerate(seg.data):
            emu.memory.write_byte(seg.address + i, b)
    rom = bytearray(b"\xff" * 0x40000)
    off = 0x3FFFA
    vec = PROGRAM.handler & 0xFFFFF
    rom[off : off + 3] = bytes([vec & 0xFF, (vec >> 8) & 0xFF, (vec >> 16) & 0xFF])
    emu.load_rom(bytes(rom))
    emu.cpu.regs.set(RegisterName.PC, PROGRAM.entry)
    emu.cpu.regs.set(RegisterName.S, 0xBFF00)
    emu.cpu.regs.set(RegisterName.U, 0xBFE00)
    emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR, 0x00)
    # Mask MTI
    emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.IMR, 0x80)

    u_before = emu.cpu.regs.get(RegisterName.U)
    emu.step()  # MV I
    emu.step()  # MV I+1
    emu.step()  # WAIT
    # Next step executes NOP (no delivery)
    emu.step()
    u_after = emu.cpu.regs.get(RegisterName.U)
    assert u_after == u_before
    assert emu.last_irq.get("src") in (None, "")


@pytest.mark.timeout(10)
def test_timer_mti_unmasked_not_enough_does_not_trigger() -> None:
    """MTI unmasked but WAIT < period: should not trigger delivery."""
    emu = PCE500Emulator(
        perfetto_trace=False, save_lcd_on_exit=False, enable_new_tracing=True
    )
    emu._timer_enabled = True  # type: ignore[attr-defined]
    emu._timer_sti_period = 10**9  # type: ignore[attr-defined]
    emu._timer_next_sti = emu.cycle_count + emu._timer_sti_period  # type: ignore[attr-defined]

    asm = Assembler()
    mti_period = int(getattr(emu, "_timer_mti_period", 500))
    wait_count = max(1, mti_period - 3)
    source = dedent(
        f"""
        .ORG 0x{PROGRAM.entry:05X}
        entry:
            MV I, 0x{wait_count:04X}
            WAIT
            NOP

        .ORG 0x{PROGRAM.handler:05X}
        handler:
            MV A, 0x{PROGRAM.marker:02X}
            PUSHU A
            MV A, (ISR)
            PUSHU A
            RETI
        """
    )
    binfile = asm.assemble(source)
    for seg in binfile.segments:
        for i, b in enumerate(seg.data):
            emu.memory.write_byte(seg.address + i, b)
    rom = bytearray(b"\xff" * 0x40000)
    off = 0x3FFFA
    vec = PROGRAM.handler & 0xFFFFF
    rom[off : off + 3] = bytes([vec & 0xFF, (vec >> 8) & 0xFF, (vec >> 16) & 0xFF])
    emu.load_rom(bytes(rom))
    emu.cpu.regs.set(RegisterName.PC, PROGRAM.entry)
    emu.cpu.regs.set(RegisterName.S, 0xBFF00)
    emu.cpu.regs.set(RegisterName.U, 0xBFE00)
    emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR, 0x00)
    # Unmask MTI
    emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.IMR, 0x80 | 0x01)

    u_before = emu.cpu.regs.get(RegisterName.U)
    emu.step()  # MV I
    emu.step()  # WAIT (sets less than period)
    emu.step()  # Next executes NOP; no delivery
    u_after = emu.cpu.regs.get(RegisterName.U)
    assert u_after == u_before
    assert emu.last_irq.get("src") in (None, "")


@pytest.mark.timeout(10)
def test_timer_sti_unmasked_wakes_and_delivers() -> None:
    """STI (bit 1) fires after WAIT-ing enough instructions and delivers when unmasked."""
    emu = PCE500Emulator(
        perfetto_trace=False, save_lcd_on_exit=False, enable_new_tracing=True
    )
    emu._timer_enabled = True  # type: ignore[attr-defined]
    # Isolate STI: push MTI far away
    emu._timer_mti_period = 10**9  # type: ignore[attr-defined]
    emu._timer_next_mti = emu.instruction_count + emu._timer_mti_period  # type: ignore[attr-defined]

    asm = Assembler()
    sti_period = int(getattr(emu, "_timer_sti_period", 5000))
    wait_count = sti_period + 200
    source = dedent(
        f"""
        .ORG 0x{PROGRAM.entry:05X}
        entry:
            MV I, 0x{wait_count:04X}
            WAIT
            NOP

        .ORG 0x{PROGRAM.handler:05X}
        handler:
            MV A, 0x{PROGRAM.marker:02X}
            PUSHU A
            MV A, (ISR)
            PUSHU A
            RETI
        """
    )
    binfile = asm.assemble(source)
    for seg in binfile.segments:
        for i, b in enumerate(seg.data):
            emu.memory.write_byte(seg.address + i, b)
    rom = bytearray(b"\xff" * 0x40000)
    off = 0x3FFFA
    vec = PROGRAM.handler & 0xFFFFF
    rom[off : off + 3] = bytes([vec & 0xFF, (vec >> 8) & 0xFF, (vec >> 16) & 0xFF])
    emu.load_rom(bytes(rom))
    emu.cpu.regs.set(RegisterName.PC, PROGRAM.entry)
    emu.cpu.regs.set(RegisterName.S, 0xBFF00)
    emu.cpu.regs.set(RegisterName.U, 0xBFE00)
    emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR, 0x00)
    emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.IMR, 0x80 | 0x02)

    u_before = emu.cpu.regs.get(RegisterName.U)
    emu.step()  # MV I
    emu.step()  # MV I+1
    emu.step()  # WAIT
    emu.step()  # Deliver
    for _ in range(5):
        emu.step()
    u_after = emu.cpu.regs.get(RegisterName.U)
    assert u_after - u_before == -2
    assert (emu.memory.read_byte(u_after) & 0x02) == 0x02
    assert emu.memory.read_byte(u_after + 1) == PROGRAM.marker
    assert emu.last_irq.get("src") is not None


@pytest.mark.timeout(10)
def test_timer_sti_masked_wakes_but_no_delivery() -> None:
    """STI masked: WAIT sets ISR[1], but no delivery when masked."""
    emu = PCE500Emulator(
        perfetto_trace=False, save_lcd_on_exit=False, enable_new_tracing=True
    )
    emu._timer_enabled = True  # type: ignore[attr-defined]
    emu._timer_mti_period = 10**9  # type: ignore[attr-defined]
    emu._timer_next_mti = emu.instruction_count + emu._timer_mti_period  # type: ignore[attr-defined]

    asm = Assembler()
    sti_period = int(getattr(emu, "_timer_sti_period", 5000))
    wait_count = sti_period + 200
    source = dedent(
        f"""
        .ORG 0x{PROGRAM.entry:05X}
        entry:
            MV I, 0x{wait_count:04X}
            WAIT
            NOP

        .ORG 0x{PROGRAM.handler:05X}
        handler:
            MV A, 0x{PROGRAM.marker:02X}
            PUSHU A
            MV A, (ISR)
            PUSHU A
            RETI
        """
    )
    binfile = asm.assemble(source)
    for seg in binfile.segments:
        for i, b in enumerate(seg.data):
            emu.memory.write_byte(seg.address + i, b)
    rom = bytearray(b"\xff" * 0x40000)
    off = 0x3FFFA
    vec = PROGRAM.handler & 0xFFFFF
    rom[off : off + 3] = bytes([vec & 0xFF, (vec >> 8) & 0xFF, (vec >> 16) & 0xFF])
    emu.load_rom(bytes(rom))
    emu.cpu.regs.set(RegisterName.PC, PROGRAM.entry)
    emu.cpu.regs.set(RegisterName.S, 0xBFF00)
    emu.cpu.regs.set(RegisterName.U, 0xBFE00)
    emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR, 0x00)
    emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.IMR, 0x80)

    u_before = emu.cpu.regs.get(RegisterName.U)
    emu.step()  # MV I
    emu.step()  # MV I+1
    emu.step()  # WAIT
    emu.step()  # Execute NOP (no delivery)
    u_after = emu.cpu.regs.get(RegisterName.U)
    assert u_after == u_before
    assert emu.last_irq.get("src") in (None, "")


@pytest.mark.timeout(10)
def test_timer_sti_unmasked_not_enough_does_not_trigger() -> None:
    """STI unmasked but WAIT < period: should not trigger delivery."""
    emu = PCE500Emulator(
        perfetto_trace=False, save_lcd_on_exit=False, enable_new_tracing=True
    )
    emu._timer_enabled = True  # type: ignore[attr-defined]
    emu._timer_mti_period = 10**9  # type: ignore[attr-defined]
    emu._timer_next_mti = emu.cycle_count + emu._timer_mti_period  # type: ignore[attr-defined]

    asm = Assembler()
    sti_period = int(getattr(emu, "_timer_sti_period", 5000))
    wait_count = max(1, sti_period - 3)
    source = dedent(
        f"""
        .ORG 0x{PROGRAM.entry:05X}
        entry:
            MV I, 0x{wait_count:04X}
            WAIT
            NOP

        .ORG 0x{PROGRAM.handler:05X}
        handler:
            MV A, 0x{PROGRAM.marker:02X}
            PUSHU A
            MV A, (ISR)
            PUSHU A
            RETI
        """
    )
    binfile = asm.assemble(source)
    for seg in binfile.segments:
        for i, b in enumerate(seg.data):
            emu.memory.write_byte(seg.address + i, b)
    rom = bytearray(b"\xff" * 0x40000)
    off = 0x3FFFA
    vec = PROGRAM.handler & 0xFFFFF
    rom[off : off + 3] = bytes([vec & 0xFF, (vec >> 8) & 0xFF, (vec >> 16) & 0xFF])
    emu.load_rom(bytes(rom))
    emu.cpu.regs.set(RegisterName.PC, PROGRAM.entry)
    emu.cpu.regs.set(RegisterName.S, 0xBFF00)
    emu.cpu.regs.set(RegisterName.U, 0xBFE00)
    emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR, 0x00)
    # Unmask STI
    emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.IMR, 0x80 | 0x02)

    u_before = emu.cpu.regs.get(RegisterName.U)
    emu.step()  # MV I
    emu.step()  # WAIT (less than period)
    emu.step()  # Next executes NOP; no delivery
    u_after = emu.cpu.regs.get(RegisterName.U)
    assert u_after == u_before
    assert emu.last_irq.get("src") in (None, "")


"""Dataclass-driven interrupt tests for HALT, WAIT, keyboard, ON-key, and timers.

Everything flows through a single parameterized test. Scenarios describe
triggers, IMR masks, program shape, and expected delivery succinctly.
"""

from dataclasses import dataclass
from textwrap import dedent
import pytest
from enum import Enum

from sc62015.pysc62015.sc_asm import Assembler
from sc62015.pysc62015.emulator import RegisterName
from sc62015.pysc62015.instr.opcodes import IMEMRegisters

from pce500 import PCE500Emulator


INTERNAL_MEMORY_START = 0x100000


class Trigger(Enum):
    KEY_F1 = "KEY_F1"
    KEY_ON = "KEY_ON"
    MTI = "MTI"  # Main timer
    STI = "STI"  # Sub timer
    NONE = "NONE"  # No source


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
    program: str  # "HALT" or "WAIT"
    # WAIT-only: relative offset from period in cycles (positive → after; negative → before)
    wait_delta: int | None = None
    # Concise expectations
    expect_deliver: bool = False
    expect_halt_canceled: bool | None = None  # Only meaningful for HALT program
    # Harness knobs
    isolate_timers: bool = False


SCENARIOS: list[InterruptScenario] = [
    # HALT + keyboard/ON
    InterruptScenario(
        "key_unmasked",
        Trigger.KEY_F1,
        0x80 | 0x04,
        "HALT",
        expect_deliver=True,
        expect_halt_canceled=True,
    ),
    InterruptScenario(
        "key_masked",
        Trigger.KEY_F1,
        0x80,
        "HALT",
        expect_deliver=False,
        expect_halt_canceled=True,
    ),
    InterruptScenario(
        "on_unmasked",
        Trigger.KEY_ON,
        0x80 | 0x08,
        "HALT",
        expect_deliver=True,
        expect_halt_canceled=True,
    ),
    InterruptScenario(
        "on_masked",
        Trigger.KEY_ON,
        0x80,
        "HALT",
        expect_deliver=False,
        expect_halt_canceled=True,
    ),
    InterruptScenario(
        "no_trigger_halts",
        Trigger.NONE,
        0x80,
        "HALT",
        expect_deliver=False,
        expect_halt_canceled=False,
    ),
    # WAIT + timers
    InterruptScenario(
        "mti_unmasked",
        Trigger.MTI,
        0x80 | 0x01,
        "WAIT",
        wait_delta=50,
        expect_deliver=True,
        isolate_timers=True,
    ),
    InterruptScenario(
        "mti_masked",
        Trigger.MTI,
        0x80,
        "WAIT",
        wait_delta=50,
        expect_deliver=False,
        isolate_timers=True,
    ),
    InterruptScenario(
        "mti_unmasked_not_enough",
        Trigger.MTI,
        0x80 | 0x01,
        "WAIT",
        wait_delta=-20,
        expect_deliver=False,
        isolate_timers=True,
    ),
    InterruptScenario(
        "sti_unmasked",
        Trigger.STI,
        0x80 | 0x02,
        "WAIT",
        wait_delta=200,
        expect_deliver=True,
        isolate_timers=True,
    ),
    InterruptScenario(
        "sti_masked",
        Trigger.STI,
        0x80,
        "WAIT",
        wait_delta=200,
        expect_deliver=False,
        isolate_timers=True,
    ),
    InterruptScenario(
        "sti_unmasked_not_enough",
        Trigger.STI,
        0x80 | 0x02,
        "WAIT",
        wait_delta=-50,
        expect_deliver=False,
        isolate_timers=True,
    ),
]


def _assemble_program(
    emu: PCE500Emulator, program: str, wait_count: int | None
) -> None:
    asm = Assembler()
    entry = (
        ENTRY_HALT.format(entry=PROGRAM.entry)
        if program == "HALT"
        else ENTRY_WAIT.format(entry=PROGRAM.entry, wait_count=wait_count or 0)
    )
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
    if sc.program == "WAIT":
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

    # HALT prelude
    if sc.program == "HALT":
        emu._timer_enabled = False  # focus on keyboard/ON behavior
        emu.step()  # HALT
        assert emu.cpu.state.halted is True
        # Stable halt check
        u_stable = emu.cpu.regs.get(RegisterName.U)
        for _ in range(3):
            emu.step()
            assert emu.cpu.state.halted is True
            assert emu.cpu.regs.get(RegisterName.U) == u_stable

    u_before = emu.cpu.regs.get(RegisterName.U)

    # Trigger
    if sc.trigger in (Trigger.KEY_F1, Trigger.KEY_ON):
        assert emu.press_key(sc.trigger.value) is True

    # Execute program to the point of delivery
    if sc.program == "HALT":
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

    if sc.program == "HALT" and sc.expect_halt_canceled is not None:
        assert emu.cpu.state.halted == (not sc.expect_halt_canceled)

    if sc.expect_deliver:
        mask = TRIGGER_MASK.get(sc.trigger, 0)
        assert (emu.memory.read_byte(u_after) & mask) == mask
        assert emu.memory.read_byte(u_after + 1) == PROGRAM.marker
        assert emu.last_irq.get("src") is not None
    else:
        assert emu.last_irq.get("src") in (None, "")
