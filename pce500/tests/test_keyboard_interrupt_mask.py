"""Test keyboard interrupt delivery obeys IMR mask bits using dataclasses.

We assemble a tiny program with an interrupt handler that pushes 0x42 onto the
U stack. We verify behavior across scenarios defined as dataclasses.
"""

from dataclasses import dataclass
from textwrap import dedent
import pytest

from sc62015.pysc62015.sc_asm import Assembler
from sc62015.pysc62015.emulator import RegisterName
from sc62015.pysc62015.instr.opcodes import IMEMRegisters

from pce500 import PCE500Emulator


INTERNAL_MEMORY_START = 0x100000


@dataclass(frozen=True)
class ProgramConfig:
    entry: int
    handler: int
    handler_push_value: int = 0x42
    steps_after_press: int = 20


@dataclass(frozen=True)
class Expectation:
    name: str
    imr_value: int
    key_code: str
    expect_u_delta: int
    expect_pushed: int | None  # KIL (top of U) when enabled
    kol_value: int = 0x00
    koh_value: int = 0x00


PROGRAM = ProgramConfig(entry=0xB8000, handler=0xB9000)

SCENARIOS: list[Expectation] = [
    # KO10 strobed via KOH bit 2; F1 row 6, F2 row 5
    Expectation(
        name="key_enabled_f1",
        imr_value=0x80 | 0x04,  # IRM=1, KEYM=1
        key_code="KEY_F1",
        expect_u_delta=-2,  # pushes marker then KIL
        expect_pushed=0x40,  # KIL bit for row 6
        koh_value=0x04,
    ),
    Expectation(
        name="key_enabled_f2",
        imr_value=0x80 | 0x04,  # IRM=1, KEYM=1
        key_code="KEY_F2",
        expect_u_delta=-2,
        expect_pushed=0x20,  # KIL bit for row 5
        koh_value=0x04,
    ),
    Expectation(
        name="key_masked",
        imr_value=0x80,  # IRM=1, KEYM=0
        key_code="KEY_F1",
        expect_u_delta=0,
        expect_pushed=None,
        koh_value=0x04,
    ),
]


def assemble_and_load(emu: PCE500Emulator, cfg: ProgramConfig) -> None:
    """Assemble minimal entry and interrupt handler and load into memory.

    entry:   two NOPs (and optionally set KOL/KOH in test)
    handler: MV A,0x42; PUSHU A; MV A,(KIL); PUSHU A; RETI
    """
    asm = Assembler()
    source = dedent(
        f"""
        .ORG 0x{cfg.entry:05X}
        entry:
            NOP
            NOP

        .ORG 0x{cfg.handler:05X}
        handler:
            MV A, 0x{cfg.handler_push_value:02X}
            PUSHU A
            ; Debounce compat keyboard: read KIL multiple times while strobed
            MV A, (0xF2)
            MV A, (0xF2)
            MV A, (0xF2)
            MV A, (0xF2)
            MV A, (0xF2)
            MV A, (0xF2)
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
    rom = bytearray(b"\xFF" * rom_size)
    vec_off = 0x3FFFA  # 0xFFFFA - 0xC0000
    vec = cfg.handler & 0xFFFFF
    rom[vec_off : vec_off + 3] = bytes([vec & 0xFF, (vec >> 8) & 0xFF, (vec >> 16) & 0xFF])
    emu.load_rom(bytes(rom))


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.name for s in SCENARIOS])
def test_keyboard_interrupt_masking_dataclass(scenario: Expectation) -> None:
    emu = PCE500Emulator(perfetto_trace=False, save_lcd_on_exit=False)
    emu._timer_enabled = False  # type: ignore[attr-defined]
    emu.reset()

    assemble_and_load(emu, PROGRAM)

    # Initialize PC and stacks
    emu.cpu.regs.set(RegisterName.PC, PROGRAM.entry)
    emu.cpu.regs.set(RegisterName.S, 0xBFF00)
    emu.cpu.regs.set(RegisterName.U, 0xBFE00)

    # Set IMR per scenario and clear ISR
    emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.IMR, scenario.imr_value)
    emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR, 0x00)
    # Program strobe selection (compat mapping: KOL bits for KO0..7, KOH bits 0..3 for KO8..KO11)
    emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.KOL, scenario.kol_value & 0xFF)
    emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.KOH, scenario.koh_value & 0xFF)

    u_before = emu.cpu.regs.get(RegisterName.U)

    # Press the key and run for some steps
    assert emu.press_key(scenario.key_code) is True
    for _ in range(PROGRAM.steps_after_press):
        emu.step()

    # Check U delta
    u_after = emu.cpu.regs.get(RegisterName.U)
    assert u_after - u_before == scenario.expect_u_delta

    # Check pushed value when applicable: top of stack should be KIL, next should be marker
    if scenario.expect_pushed is not None:
        assert emu.memory.read_byte(u_after) == scenario.expect_pushed
        assert emu.memory.read_byte(u_after + 1) == PROGRAM.handler_push_value
    else:
        # No push occurred; U unchanged. Optionally confirm previous byte left as reset value (0x00)
        assert u_after == u_before
