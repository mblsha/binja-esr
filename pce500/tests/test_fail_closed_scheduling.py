"""Machine-wrapper scheduling must not run ahead of quarantined opcodes."""

from __future__ import annotations

import pytest

from pce500.emulator import IRQSource, PCE500Emulator
from pce500.memory_bus import MemoryOverlay, UnsafePreflightRead
from sc62015.pysc62015 import RegisterName, available_backends
from sc62015.pysc62015.constants import IMRFlag, ISRFlag
from sc62015.pysc62015.instr import InvalidInstruction
from sc62015.pysc62015.instr.opcodes import IMEMRegisters, INTERNAL_MEMORY_START


@pytest.mark.parametrize("backend", available_backends())
@pytest.mark.parametrize(
    ("opcode", "i_value", "error"),
    (
        (0xEF, 0, "I=0 counted-instruction semantics require real-hardware tracing"),
        (0xCE, 1, "TCL timer-clear side effects are not implemented"),
    ),
)
def test_quarantined_opcode_rejects_before_pce_timer_mutation(
    backend: str,
    opcode: int,
    i_value: int,
    error: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SC62015_CPU_BACKEND", backend)
    emu = PCE500Emulator(perfetto_trace=False, save_lcd_on_exit=False)
    try:
        emu.memory.write_byte(0, opcode)
        emu.cpu.regs.set(RegisterName.PC, 0)
        emu.cpu.regs.set(RegisterName.I, i_value)
        emu._timer_enabled = True
        emu._timer_next_mti = 0
        emu._timer_next_sti = 0
        emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR, 0)
        before = (
            emu.cycle_count,
            emu._timer_next_mti,
            emu._timer_next_sti,
            emu.memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR),
        )

        with pytest.raises(NotImplementedError, match=error):
            emu.step()

        assert emu.cpu.regs.get(RegisterName.PC) == 0
        assert emu.cpu.regs.get(RegisterName.I) == i_value
        assert (
            emu.cycle_count,
            emu._timer_next_mti,
            emu._timer_next_sti,
            emu.memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR),
        ) == before
    finally:
        emu.close()


@pytest.mark.parametrize("backend", available_backends())
@pytest.mark.parametrize(
    "program",
    (
        bytes.fromhex("303100"),  # consecutive PRE bytes
        bytes.fromhex("211100"),  # PRE followed by reserved JP selector
        bytes.fromhex("21E31000"),  # PRE followed by malformed E3 mode
        bytes.fromhex("257C01"),  # mid-instruction bytes at PC-E500 EFE2B
    ),
)
def test_malformed_pre_rejects_before_pce_scheduler_mutation(
    backend: str,
    program: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SC62015_CPU_BACKEND", backend)
    emu = PCE500Emulator(perfetto_trace=False, save_lcd_on_exit=False)
    try:
        for address, value in enumerate(program):
            emu.memory.write_byte(address, value)
        emu.cpu.regs.set(RegisterName.PC, 0)
        emu.cpu.regs.set(RegisterName.I, 1)
        emu._timer_enabled = True
        emu._timer_next_mti = 0
        emu._timer_next_sti = 0
        emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR, 0)
        before = (
            emu.cycle_count,
            emu.instruction_count,
            emu._timer_next_mti,
            emu._timer_next_sti,
            emu.memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR),
            emu.cpu.regs.get(RegisterName.PC),
            emu.cpu.regs.get(RegisterName.I),
        )

        with pytest.raises(InvalidInstruction, match="PRE"):
            emu.step()

        assert (
            emu.cycle_count,
            emu.instruction_count,
            emu._timer_next_mti,
            emu._timer_next_sti,
            emu.memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR),
            emu.cpu.regs.get(RegisterName.PC),
            emu.cpu.regs.get(RegisterName.I),
        ) == before
    finally:
        emu.close()


@pytest.mark.parametrize("backend", available_backends())
@pytest.mark.parametrize(
    ("program", "stack_register", "error"),
    (
        (
            bytes.fromhex("32C22030"),
            None,
            "EXP high-nibble behavior requires real-hardware tracing",
        ),
        (bytes.fromhex("3E"), RegisterName.U, "bits 2-7 require real-hardware"),
        (bytes.fromhex("5F"), RegisterName.S, "bits 2-7 require real-hardware"),
        (bytes.fromhex("01"), RegisterName.S, "bits 2-7 require real-hardware"),
    ),
)
def test_data_dependent_quarantine_rejects_before_pce_scheduler_mutation(
    backend: str,
    program: bytes,
    stack_register: RegisterName | None,
    error: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SC62015_CPU_BACKEND", backend)
    emu = PCE500Emulator(perfetto_trace=False, save_lcd_on_exit=False)
    try:
        for address, value in enumerate(program):
            emu.memory.write_byte(address, value)
        emu.cpu.regs.set(RegisterName.PC, 0)
        emu.cpu.regs.set(RegisterName.F, 0x01)
        if program[0] == 0x32:  # PRE32; EXP (20),(30), both direct.
            for index, value in enumerate(bytes.fromhex("1122A8334409")):
                selector = 0x20 + index if index < 3 else 0x30 + index - 3
                emu.memory.write_byte(INTERNAL_MEMORY_START + selector, value)
        else:
            assert stack_register is not None
            emu.cpu.regs.set(stack_register, 0x100)
            f_offset = 1 if program[0] == 0x01 else 0
            emu.memory.write_byte(0x100 + f_offset, 0x80)

        emu._timer_enabled = True
        emu._timer_next_mti = 0
        emu._timer_next_sti = 0
        emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR, 0)
        before = (
            emu.cycle_count,
            emu.instruction_count,
            emu._timer_next_mti,
            emu._timer_next_sti,
            emu.memory.peek_byte_for_preflight(
                INTERNAL_MEMORY_START + IMEMRegisters.ISR
            ),
            emu.cpu.regs.get(RegisterName.PC),
            emu.cpu.regs.get(RegisterName.F),
            emu.cpu.regs.get(stack_register) if stack_register is not None else None,
        )

        with pytest.raises((RuntimeError, NotImplementedError), match=error):
            emu.step()

        assert (
            emu.cycle_count,
            emu.instruction_count,
            emu._timer_next_mti,
            emu._timer_next_sti,
            emu.memory.peek_byte_for_preflight(
                INTERNAL_MEMORY_START + IMEMRegisters.ISR
            ),
            emu.cpu.regs.get(RegisterName.PC),
            emu.cpu.regs.get(RegisterName.F),
            emu.cpu.regs.get(stack_register) if stack_register is not None else None,
        ) == before
    finally:
        emu.close()


@pytest.mark.parametrize("backend", available_backends())
def test_dynamic_instruction_overlay_without_safe_peek_fails_before_read(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SC62015_CPU_BACKEND", backend)
    emu = PCE500Emulator(perfetto_trace=False, save_lcd_on_exit=False)
    normal_reads = 0

    def read_handler(_address: int, _pc: int | None) -> int:
        nonlocal normal_reads
        normal_reads += 1
        return 0x00

    try:
        emu.memory.add_overlay(
            MemoryOverlay(
                start=0x100,
                end=0x100,
                name="side_effecting_instruction_source",
                read_handler=read_handler,
            )
        )
        emu.cpu.regs.set(RegisterName.PC, 0x100)
        before = (emu.cycle_count, emu.instruction_count)

        with pytest.raises(UnsafePreflightRead, match="side-effect-free preflight"):
            emu.step()

        assert normal_reads == 0
        assert (emu.cycle_count, emu.instruction_count) == before
        assert emu.cpu.regs.get(RegisterName.PC) == 0x100
    finally:
        emu.close()


@pytest.mark.parametrize("backend", available_backends())
def test_invalid_pending_pc_rejects_before_key_latch_or_irq_state_mutation(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SC62015_CPU_BACKEND", backend)
    emu = PCE500Emulator(perfetto_trace=False, save_lcd_on_exit=False)
    try:
        emu.memory.write_byte(0, 0xCE)  # TCL
        emu.cpu.regs.set(RegisterName.PC, 0)
        emu.cpu.regs.set(RegisterName.S, 0x300)
        emu._key_irq_latched = True
        emu._irq_pending = False
        emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR, 0)
        before = (
            emu.cpu.regs.get(RegisterName.PC),
            emu.cpu.regs.get(RegisterName.S),
            emu._irq_pending,
            emu._in_interrupt,
            emu.memory.peek_byte_for_preflight(
                INTERNAL_MEMORY_START + IMEMRegisters.ISR
            ),
        )

        with pytest.raises(NotImplementedError, match="TCL timer-clear"):
            emu.step()

        assert (
            emu.cpu.regs.get(RegisterName.PC),
            emu.cpu.regs.get(RegisterName.S),
            emu._irq_pending,
            emu._in_interrupt,
            emu.memory.peek_byte_for_preflight(
                INTERNAL_MEMORY_START + IMEMRegisters.ISR
            ),
        ) == before
        assert emu._key_irq_latched is True
    finally:
        emu.close()


@pytest.mark.parametrize("backend", available_backends())
def test_invalid_irq_handler_rejects_before_interrupt_frame_mutation(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SC62015_CPU_BACKEND", backend)
    emu = PCE500Emulator(perfetto_trace=False, save_lcd_on_exit=False)
    try:
        handler = 0x200
        stack = 0x300
        emu._timer_enabled = False
        emu._irq_pending = True
        emu._irq_source = IRQSource.KEY
        emu._in_interrupt = False
        emu.memory.write_byte(0x1000, 0x00)  # pending NOP
        emu.memory.write_byte(handler, 0xCE)  # quarantined TCL handler
        # Keep the external vector distinct from the compact internal-RAM
        # backing used by this test memory shim.
        emu.memory.add_rom(0xFFFFA, handler.to_bytes(3, "little"), "test_irq_vector")
        emu.cpu.regs.set(RegisterName.PC, 0x1000)
        emu.cpu.regs.set(RegisterName.S, stack)
        emu.memory.write_byte(
            INTERNAL_MEMORY_START + IMEMRegisters.IMR,
            int(IMRFlag.IRM | IMRFlag.KEYM),
        )
        emu.memory.write_byte(
            INTERNAL_MEMORY_START + IMEMRegisters.ISR, int(ISRFlag.KEYI)
        )
        before_stack = bytes(
            emu.memory.peek_byte_for_preflight(stack - offset) for offset in range(1, 6)
        )
        before_imr = emu.memory.peek_byte_for_preflight(
            INTERNAL_MEMORY_START + IMEMRegisters.IMR
        )

        with pytest.raises(NotImplementedError, match="TCL timer-clear"):
            emu.step()

        assert emu.cpu.regs.get(RegisterName.PC) == 0x1000
        assert emu.cpu.regs.get(RegisterName.S) == stack
        assert emu._irq_pending is True
        assert emu._in_interrupt is False
        assert emu._poisoned is None
        assert (
            emu.memory.peek_byte_for_preflight(
                INTERNAL_MEMORY_START + IMEMRegisters.IMR
            )
            == before_imr
        )
        assert (
            bytes(
                emu.memory.peek_byte_for_preflight(stack - offset)
                for offset in range(1, 6)
            )
            == before_stack
        )
    finally:
        emu.close()
