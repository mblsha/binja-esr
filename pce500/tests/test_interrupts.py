"""Generic interrupt handling tests using HALT and ISR semantics.

This file verifies:
- HALT prevents instruction execution until an interrupt.
- Any bit set in ISR cancels the HALT state (CPU wakes).
- Interrupt delivery obeys IMR masks after HALT cancellation.
- ON key asserts ISR.ONKI and can wake/interrupt when unmasked.
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
    NONE = "NONE"  # No trigger; CPU should remain halted


@dataclass(frozen=True)
class ProgramConfig:
    entry: int
    handler: int
    marker: int = 0x42


PROGRAM = ProgramConfig(entry=0xB8000, handler=0xB9000)


def assemble_and_load(emu: PCE500Emulator, cfg: ProgramConfig) -> None:
    """Assemble minimal entry (with HALT) and interrupt handler (pushes ISR)."""
    asm = Assembler()
    source = dedent(
        f"""
        .ORG 0x{cfg.entry:05X}
        entry:
            HALT            ; stop until interrupt
            NOP             ; should not execute before interrupt

        .ORG 0x{cfg.handler:05X}
        handler:
            MV A, 0x{cfg.marker:02X}
            PUSHU A
            MV A, (ISR)
            PUSHU A
            RETI
        """
    )
    binfile = asm.assemble(source)

    # Write segments into emulator memory
    for seg in binfile.segments:
        base = seg.address
        for i, b in enumerate(seg.data):
            emu.memory.write_byte(base + i, b)

    # Minimal ROM overlay with interrupt vector at 0xFFFFA → handler
    rom_size = 0x40000
    rom = bytearray(b"\xff" * rom_size)
    vec_off = 0x3FFFA  # 0xFFFFA - 0xC0000
    vec = cfg.handler & 0xFFFFF
    rom[vec_off : vec_off + 3] = bytes(
        [vec & 0xFF, (vec >> 8) & 0xFF, (vec >> 16) & 0xFF]
    )
    emu.load_rom(bytes(rom))


@dataclass(frozen=True)
class Scenario:
    name: str
    imr: int
    trigger: Trigger  # which source to trigger (or NONE)
    expect_u_delta: int
    expect_isr_mask: int | None  # pushed ISR value when interrupt delivered
    expect_halt_canceled: bool = True  # whether HALT should cancel after trigger


SCENARIOS: list[Scenario] = [
    Scenario(
        name="key_unmasked",
        imr=0x80 | 0x04,  # IRM=1, KEYM=1
        trigger=Trigger.KEY_F1,
        expect_u_delta=-2,
        expect_isr_mask=0x04,  # ISR[2]
    ),
    Scenario(
        name="key_masked",
        imr=0x80,  # IRM=1, KEYM=0 → HALT cancels but no interrupt delivery
        trigger=Trigger.KEY_F1,
        expect_u_delta=0,
        expect_isr_mask=None,
        expect_halt_canceled=True,
    ),
    Scenario(
        name="on_unmasked",
        imr=0x80 | 0x08,  # IRM=1, ONKM=1
        trigger=Trigger.KEY_ON,
        expect_u_delta=-2,
        expect_isr_mask=0x08,  # ISR[3]
    ),
    Scenario(
        name="on_masked",
        imr=0x80,  # IRM=1, ONKM=0 → HALT cancels but no interrupt delivery
        trigger=Trigger.KEY_ON,
        expect_u_delta=0,
        expect_isr_mask=None,
        expect_halt_canceled=True,
    ),
    Scenario(
        name="no_trigger_halts",
        imr=0x80,  # IRM on (mask values irrelevant since no source)
        trigger=Trigger.NONE,
        expect_u_delta=0,
        expect_isr_mask=None,
        expect_halt_canceled=False,
    ),
]


@pytest.mark.parametrize("sc", SCENARIOS, ids=[s.name for s in SCENARIOS])
def test_interrupt_delivery_with_halt(sc: Scenario) -> None:
    emu = PCE500Emulator(perfetto_trace=False, save_lcd_on_exit=False)
    # Disable periodic timers to avoid interference
    emu._timer_enabled = False  # type: ignore[attr-defined]
    emu.reset()

    assemble_and_load(emu, PROGRAM)

    # Initialize PC and stacks
    emu.cpu.regs.set(RegisterName.PC, PROGRAM.entry)
    emu.cpu.regs.set(RegisterName.S, 0xBFF00)
    emu.cpu.regs.set(RegisterName.U, 0xBFE00)

    # Configure IMR and clear ISR
    emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.IMR, sc.imr)
    emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR, 0x00)

    # Execute HALT (CPU becomes halted)
    emu.step()
    assert emu.cpu.state.halted is True

    # While halted and with no ISR bits, additional steps do not execute instructions
    u_before = emu.cpu.regs.get(RegisterName.U)
    for _ in range(5):
        emu.step()
        assert emu.cpu.state.halted is True
        # Stack unchanged while halted
        assert emu.cpu.regs.get(RegisterName.U) == u_before

    if sc.trigger is Trigger.NONE:
        # With no trigger, CPU must remain halted and U unchanged
        for _ in range(20):
            emu.step()
            assert emu.cpu.state.halted is True
            assert emu.cpu.regs.get(RegisterName.U) == u_before
        return
    elif sc.trigger in (Trigger.KEY_F1, Trigger.KEY_ON):
        # Trigger specific source
        key_code = sc.trigger.value
        assert emu.press_key(key_code) is True

        # Step enough times to allow delivery (pending + entry)
        for _ in range(20):
            emu.step()
    else:
        pytest.skip("Timer triggers are validated in WAIT-based tests below")

    u_after = emu.cpu.regs.get(RegisterName.U)
    assert u_after - u_before == sc.expect_u_delta

    # Validate HALT cancellation semantics
    assert (
        emu.cpu.state.halted is (not sc.expect_halt_canceled)
        if False
        else emu.cpu.state.halted == (not sc.expect_halt_canceled)
    )

    # Delivery vs masking: last_irq should be set only when delivered
    delivered = sc.expect_isr_mask is not None and sc.trigger is not Trigger.NONE

    if delivered:
        # Top of U is ISR value; next is marker
        assert emu.memory.read_byte(u_after) & sc.expect_isr_mask == sc.expect_isr_mask
        assert emu.memory.read_byte(u_after + 1) == PROGRAM.marker
        # Interrupt delivery recorded
        assert emu.last_irq.get("src") is not None
    else:
        # Interrupt masked: handler not executed; HALT cancelled by ISR but no pushes
        assert u_after == u_before
        # Interrupt not recorded
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
        emu._timer_next_sti = emu.instruction_count + emu._timer_sti_period  # type: ignore[attr-defined]
    elif trigger == Trigger.STI:
        emu._timer_mti_period = 10**9  # type: ignore[attr-defined]
        emu._timer_next_mti = emu.instruction_count + emu._timer_mti_period  # type: ignore[attr-defined]


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
