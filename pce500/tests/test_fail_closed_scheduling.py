"""Machine-wrapper scheduling must not run ahead of quarantined opcodes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pce500.emulator import IRQSource, PCE500Emulator, new_tracer
from pce500.memory import PCE500Memory
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


def test_software_ir_vector_rejects_before_scheduler_or_frame_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SC62015_CPU_BACKEND", "python")
    emu = PCE500Emulator(perfetto_trace=False, save_lcd_on_exit=False)
    try:
        stack = 0x400
        emu.memory.write_byte(0x1000, 0xFE)  # software IR
        emu.memory.add_rom(
            0xFFFFA,
            bytes.fromhex("0001F0"),  # raw F00100, not a canonical PC image
            "noncanonical_software_ir_vector",
        )
        emu.cpu.regs.set(RegisterName.PC, 0x1000)
        emu.cpu.regs.set(RegisterName.S, stack)
        emu.cpu.regs.set(RegisterName.F, 0x03)
        emu.memory.write_byte(
            INTERNAL_MEMORY_START + IMEMRegisters.IMR,
            int(IMRFlag.IRM | IMRFlag.KEYM),
        )
        emu._timer_enabled = True
        emu._timer_next_mti = 0
        emu._timer_next_sti = 0
        before = (
            emu.cycle_count,
            emu.instruction_count,
            emu._timer_next_mti,
            emu._timer_next_sti,
            emu.memory_read_count,
            emu.cpu.regs.get(RegisterName.PC),
            emu.cpu.regs.get(RegisterName.S),
            emu.memory.peek_byte_for_preflight(
                INTERNAL_MEMORY_START + IMEMRegisters.IMR
            ),
            bytes(
                emu.memory.peek_byte_for_preflight(stack - offset)
                for offset in range(1, 6)
            ),
        )

        with pytest.raises(NotImplementedError, match="noncanonical vector 0xF00100"):
            emu.step()

        assert (
            emu.cycle_count,
            emu.instruction_count,
            emu._timer_next_mti,
            emu._timer_next_sti,
            emu.memory_read_count,
            emu.cpu.regs.get(RegisterName.PC),
            emu.cpu.regs.get(RegisterName.S),
            emu.memory.peek_byte_for_preflight(
                INTERNAL_MEMORY_START + IMEMRegisters.IMR
            ),
            bytes(
                emu.memory.peek_byte_for_preflight(stack - offset)
                for offset in range(1, 6)
            ),
        ) == before
        assert emu._poisoned is None
    finally:
        emu.close()


def test_machine_reset_bad_vector_preserves_ram_lcd_and_wrapper_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SC62015_CPU_BACKEND", "python")
    emu = PCE500Emulator(perfetto_trace=False, save_lcd_on_exit=False)
    try:
        emu.memory.add_rom(
            0xFFFFD,
            bytes.fromhex("0001F0"),
            "noncanonical_reset_vector",
        )
        emu.memory.write_byte(0xB8000, 0xA5)
        emu.memory.write_byte(0xA008, 0x3F)  # display on
        emu.cpu.regs.set(RegisterName.PC, 0x23456)
        emu.cpu.regs.set(RegisterName.S, 0x34567)
        emu.cycle_count = 123
        emu.instruction_count = 45
        emu._poisoned = "prior machine fault"
        before = (
            emu.memory.peek_byte_for_preflight(0xB8000),
            tuple(emu.lcd.display_on),
            emu.cpu.regs.get(RegisterName.PC),
            emu.cpu.regs.get(RegisterName.S),
            emu.cycle_count,
            emu.instruction_count,
            emu._poisoned,
        )

        with pytest.raises(NotImplementedError, match="noncanonical vector 0xF00100"):
            emu.reset()

        assert (
            emu.memory.peek_byte_for_preflight(0xB8000),
            tuple(emu.lcd.display_on),
            emu.cpu.regs.get(RegisterName.PC),
            emu.cpu.regs.get(RegisterName.S),
            emu.cycle_count,
            emu.instruction_count,
            emu._poisoned,
        ) == before
    finally:
        emu._poisoned = None
        emu.close()


def test_bootstrap_bad_vector_rejects_before_ram_or_imem_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SC62015_CPU_BACKEND", "python")
    emu = PCE500Emulator(perfetto_trace=False, save_lcd_on_exit=False)
    try:
        emu.memory.add_rom(
            0xFFFFD,
            bytes.fromhex("0001F0"),
            "noncanonical_bootstrap_vector",
        )
        emu.memory.write_byte(emu.INTERNAL_RAM_START, 0xA5)
        imr_addr = INTERNAL_MEMORY_START + IMEMRegisters.IMR
        emu.memory.write_byte(imr_addr, 0x81)
        emu.cpu.regs.set(RegisterName.PC, 0x34567)
        before = (
            emu.memory.peek_byte_for_preflight(emu.INTERNAL_RAM_START),
            emu.memory.peek_byte_for_preflight(imr_addr),
            emu.cpu.regs.get(RegisterName.PC),
        )

        with pytest.raises(NotImplementedError, match="noncanonical vector 0xF00100"):
            emu.bootstrap_from_rom_image(bytes(0x100000), reset=False)

        assert (
            emu.memory.peek_byte_for_preflight(emu.INTERNAL_RAM_START),
            emu.memory.peek_byte_for_preflight(imr_addr),
            emu.cpu.regs.get(RegisterName.PC),
        ) == before
    finally:
        emu.close()


def _install_static_vector_rom(
    emu: PCE500Emulator,
    vector_address: int,
    target: int,
    *,
    source: tuple[int, int] | None = None,
) -> None:
    """Install a callback-free ROM containing one vector and a NOP target."""

    rom_start = 0xC0000
    rom = bytearray(0x40000)
    rom[target - rom_start] = 0x00
    if source is not None:
        source_address, source_opcode = source
        rom[source_address - rom_start] = source_opcode & 0xFF
    vector_offset = vector_address - rom_start
    rom[vector_offset : vector_offset + 3] = target.to_bytes(3, "little")
    emu.memory.load_rom(rom, start_address=rom_start)


@pytest.mark.parametrize("backend", available_backends())
@pytest.mark.parametrize(
    ("power_state", "isr_flag", "imr_flag"),
    (
        ("halted", ISRFlag.KEYI, IMRFlag.KEYM),
        ("off", ISRFlag.ONKI, IMRFlag.ONKM),
    ),
)
def test_low_power_wake_is_idle_boundary_before_dormant_pc_fetch(
    backend: str,
    power_state: str,
    isr_flag: ISRFlag,
    imr_flag: IMRFlag,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deliverable wake IRQ may replace an unfetched dormant PC."""

    monkeypatch.setenv("SC62015_CPU_BACKEND", backend)
    emu = PCE500Emulator(perfetto_trace=False, save_lcd_on_exit=False)
    dormant_pc = 0x1000
    handler = 0xC0200
    normal_reads: list[int] = []

    def dormant_read(address: int, _pc: int | None) -> int:
        normal_reads.append(address)
        return 0xCE  # quarantined TCL if this dormant PC were executed

    try:
        _install_static_vector_rom(emu, 0xFFFFA, handler)
        emu.memory.add_overlay(
            MemoryOverlay(
                start=dormant_pc,
                end=dormant_pc,
                name="unfetched_dormant_pc",
                read_handler=dormant_read,
            )
        )
        monkeypatch.setattr(emu, "_scan_keyboard_per_instruction", lambda: None)
        emu.cpu.regs.set(RegisterName.PC, dormant_pc)
        emu.cpu.regs.set(RegisterName.S, 0x400)
        emu.cpu.state.halted = True
        emu.cpu.state.power_state = power_state
        emu._timer_enabled = False
        emu.memory.write_byte(
            INTERNAL_MEMORY_START + IMEMRegisters.IMR,
            int(IMRFlag.IRM | imr_flag),
        )
        emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR, int(isr_flag))
        before = (emu.cycle_count, emu.instruction_count)

        # Wake is its own idle step; it does not fetch/decode the dormant PC.
        assert emu.step() is True
        assert emu.cpu.state.halted is False
        assert emu.cpu.state.power_state == "running"
        assert emu._irq_pending is True
        assert emu.cpu.regs.get(RegisterName.PC) == dormant_pc
        assert (emu.cycle_count, emu.instruction_count) == (before[0] + 1, before[1])
        assert normal_reads == []

        # The next scheduling pass takes the already-unmasked interrupt and
        # executes its valid handler without ever reading the replaced PC.
        assert emu.step() is True
        assert emu._in_interrupt is True
        assert emu.cpu.regs.get(RegisterName.PC) == handler + 1
        assert normal_reads == []
    finally:
        emu.close()


@pytest.mark.parametrize("backend", available_backends())
def test_off_ignores_non_onki_without_fetching_dormant_pc(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SC62015_CPU_BACKEND", backend)
    emu = PCE500Emulator(perfetto_trace=False, save_lcd_on_exit=False)
    dormant_pc = 0x1000
    handler = 0xC0200
    normal_reads: list[int] = []

    def dormant_read(address: int, _pc: int | None) -> int:
        normal_reads.append(address)
        return 0x00

    try:
        _install_static_vector_rom(emu, 0xFFFFA, handler)
        emu.memory.add_overlay(
            MemoryOverlay(
                start=dormant_pc,
                end=dormant_pc,
                name="off_dormant_pc",
                read_handler=dormant_read,
            )
        )
        monkeypatch.setattr(emu, "_scan_keyboard_per_instruction", lambda: None)
        emu.cpu.regs.set(RegisterName.PC, dormant_pc)
        emu.cpu.state.halted = True
        emu.cpu.state.power_state = "off"
        emu._timer_enabled = False
        emu.memory.write_byte(
            INTERNAL_MEMORY_START + IMEMRegisters.IMR,
            int(IMRFlag.IRM | IMRFlag.KEYM),
        )
        emu.memory.write_byte(
            INTERNAL_MEMORY_START + IMEMRegisters.ISR, int(ISRFlag.KEYI)
        )
        before = (emu.cycle_count, emu.instruction_count)

        assert emu.step() is True

        assert emu.cpu.state.halted is True
        assert emu.cpu.state.power_state == "off"
        assert emu.cpu.regs.get(RegisterName.PC) == dormant_pc
        assert (emu.cycle_count, emu.instruction_count) == (before[0] + 1, before[1])
        assert normal_reads == []
    finally:
        emu.close()


def test_breakpoint_precedes_opcode_vector_fetch_and_timer_tick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SC62015_CPU_BACKEND", "python")
    emu = PCE500Emulator(perfetto_trace=False, save_lcd_on_exit=False)
    try:
        source = 0xC0100
        target = 0xC0200
        _install_static_vector_rom(
            emu,
            0xFFFFD,
            target,
            source=(source, 0xFF),  # software RESET
        )
        original_read = emu.memory.read_byte
        bus_reads: list[int] = []

        def instrumented_read(address: int, cpu_pc: int | None = None) -> int:
            bus_reads.append(address)
            return original_read(address, cpu_pc)

        emu.cpu.regs.set(RegisterName.PC, source)
        emu.breakpoints.add(source)
        emu._timer_enabled = True
        emu._timer_next_mti = 0
        emu._timer_next_sti = 0
        emu._key_irq_latched = True
        emu._irq_pending = False
        emu.memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR, 0)
        monkeypatch.setattr(emu.memory, "read_byte", instrumented_read)
        before = (
            emu.cycle_count,
            emu._timer_next_mti,
            emu._timer_next_sti,
            emu.instruction_count,
            emu._key_irq_latched,
            emu._irq_pending,
            emu.memory.peek_byte_for_preflight(
                INTERNAL_MEMORY_START + IMEMRegisters.ISR
            ),
        )

        assert emu.step() is False

        assert bus_reads == []
        assert (
            emu.cycle_count,
            emu._timer_next_mti,
            emu._timer_next_sti,
            emu.instruction_count,
            emu._key_irq_latched,
            emu._irq_pending,
            emu.memory.peek_byte_for_preflight(
                INTERNAL_MEMORY_START + IMEMRegisters.ISR
            ),
        ) == before
    finally:
        emu.close()


def test_stub_trace_diagnostic_is_silent_before_breakpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trace-only stub context must not perform normal bus reads when stopped."""

    monkeypatch.setenv("SC62015_CPU_BACKEND", "python")
    emu = PCE500Emulator(perfetto_trace=False, save_lcd_on_exit=False)
    source = 0xF205C
    stack = 0x300
    watched_reads: list[int] = []

    try:
        emu.memory.add_rom(source, b"\x00", "stub_breakpoint_source")
        original_read = emu.memory.read_byte

        def instrumented_read(address: int, cpu_pc: int | None = None) -> int:
            if address == source or stack <= address < stack + 5:
                watched_reads.append(address)
                raise AssertionError("trace diagnostic performed a normal bus read")
            return original_read(address, cpu_pc)

        monkeypatch.setattr(emu.memory, "read_byte", instrumented_read)
        monkeypatch.setattr(new_tracer, "_enabled", True)
        monkeypatch.setattr(new_tracer, "instant", lambda *_args, **_kwargs: None)
        emu.cpu.regs.set(RegisterName.PC, source)
        emu.cpu.regs.set(RegisterName.S, stack)
        emu.breakpoints.add(source)

        assert emu.step() is False
        assert watched_reads == []
        assert emu.cpu.regs.get(RegisterName.PC) == source
    finally:
        emu.close()


def test_hardware_irq_callback_vector_rejects_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A callback-backed IRQ vector is a silent, non-poisoning rejection."""

    monkeypatch.setenv("SC62015_CPU_BACKEND", "python")
    emu = PCE500Emulator(perfetto_trace=False, save_lcd_on_exit=False)
    try:
        target = 0xC0200
        vector = target.to_bytes(3, "little")
        normal_reads: list[int] = []

        def normal_read(address: int, _pc: int | None) -> int:
            normal_reads.append(address)
            return vector[address - 0xFFFFA]

        emu.memory.add_overlay(
            MemoryOverlay(
                start=0xFFFFA,
                end=0xFFFFC,
                name="dynamic_irq_vector",
                read_handler=normal_read,
                preflight_read_handler=lambda address, _pc: vector[address - 0xFFFFA],
            )
        )
        emu.memory.add_rom(target, b"\x00", "irq_target")
        emu.cpu.regs.set(RegisterName.PC, 0x1000)
        emu.cpu.regs.set(RegisterName.S, 0x0400)
        emu.memory.write_byte(
            INTERNAL_MEMORY_START + IMEMRegisters.IMR,
            int(IMRFlag.IRM | IMRFlag.KEYM),
        )
        emu.memory.write_byte(
            INTERNAL_MEMORY_START + IMEMRegisters.ISR,
            int(ISRFlag.KEYI),
        )
        emu._irq_pending = True
        before = (emu.cycle_count, emu.instruction_count, emu._poisoned)

        with pytest.raises(RuntimeError, match="callback-free.*dynamic"):
            emu.step()

        assert normal_reads == []
        assert (emu.cycle_count, emu.instruction_count, emu._poisoned) == before
    finally:
        emu.close()


@pytest.mark.parametrize("failure_mode", ("read_error", "mismatch"))
def test_hardware_irq_architectural_vector_failure_is_atomic_and_poisoning(
    failure_mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SC62015_CPU_BACKEND", "python")
    emu = PCE500Emulator(perfetto_trace=False, save_lcd_on_exit=False)
    try:
        handler = 0xC0200
        stack = 0x400
        _install_static_vector_rom(emu, 0xFFFFA, handler)
        original_read = emu.memory.read_byte
        architectural_vector_reads: list[int] = []

        def normal_read(address: int, cpu_pc: int | None = None) -> int:
            if 0xFFFFA <= address <= 0xFFFFC:
                architectural_vector_reads.append(address)
            if failure_mode == "read_error" and address == 0xFFFFB:
                raise RuntimeError("architectural IRQ-vector read failed")
            value = original_read(address, cpu_pc)
            if failure_mode == "mismatch" and address == 0xFFFFA:
                value ^= 0x01
            return value

        monkeypatch.setattr(emu.memory, "read_byte", normal_read)
        emu.memory.write_byte(0x1000, 0x00)
        emu.cpu.regs.set(RegisterName.PC, 0x1000)
        emu.cpu.regs.set(RegisterName.S, stack)
        emu.cpu.regs.set(RegisterName.F, 0x03)
        emu.memory.write_byte(
            INTERNAL_MEMORY_START + IMEMRegisters.IMR,
            int(IMRFlag.IRM | IMRFlag.KEYM),
        )
        emu.memory.write_byte(
            INTERNAL_MEMORY_START + IMEMRegisters.ISR, int(ISRFlag.KEYI)
        )
        emu._timer_enabled = False
        emu._irq_pending = True
        # Deliberately differ from the pending ISR source so the assertion
        # proves failed vector delivery does not relabel wrapper metadata.
        emu._irq_source = IRQSource.MTI
        emu._in_interrupt = False
        before = (
            emu.cpu.regs.get(RegisterName.PC),
            emu.cpu.regs.get(RegisterName.S),
            emu.cpu.regs.get(RegisterName.F),
            emu.memory.peek_byte_for_preflight(
                INTERNAL_MEMORY_START + IMEMRegisters.IMR
            ),
            bytes(
                emu.memory.peek_byte_for_preflight(stack - offset)
                for offset in range(1, 6)
            ),
            emu._irq_pending,
            emu._in_interrupt,
            emu._irq_source,
            emu.cycle_count,
            emu.instruction_count,
            dict(emu.irq_counts),
            dict(emu.last_irq),
        )
        error = (
            "architectural IRQ-vector read failed"
            if failure_mode == "read_error"
            else "fetch disagrees with safe preflight"
        )

        with pytest.raises(RuntimeError, match=error):
            emu.step()

        assert (
            emu.cpu.regs.get(RegisterName.PC),
            emu.cpu.regs.get(RegisterName.S),
            emu.cpu.regs.get(RegisterName.F),
            emu.memory.peek_byte_for_preflight(
                INTERNAL_MEMORY_START + IMEMRegisters.IMR
            ),
            bytes(
                emu.memory.peek_byte_for_preflight(stack - offset)
                for offset in range(1, 6)
            ),
            emu._irq_pending,
            emu._in_interrupt,
            emu._irq_source,
            emu.cycle_count,
            emu.instruction_count,
            dict(emu.irq_counts),
            dict(emu.last_irq),
        ) == before
        assert architectural_vector_reads
        assert emu._poisoned is not None
    finally:
        emu.close()


def test_machine_reset_callback_vector_rejects_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dynamic reset vector is rejected before normal reads or reset writes."""

    monkeypatch.setenv("SC62015_CPU_BACKEND", "python")
    emu = PCE500Emulator(perfetto_trace=False, save_lcd_on_exit=False)
    try:
        target = 0xC0200
        vector = target.to_bytes(3, "little")
        normal_reads: list[int] = []

        def normal_read(address: int, _pc: int | None) -> int:
            normal_reads.append(address)
            return vector[address - 0xFFFFD]

        emu.memory.add_overlay(
            MemoryOverlay(
                start=0xFFFFD,
                end=0xFFFFF,
                name="dynamic_reset_vector",
                read_handler=normal_read,
                preflight_read_handler=lambda address, _pc: vector[address - 0xFFFFD],
            )
        )
        emu.memory.add_rom(target, b"\x00", "reset_target")
        emu.memory.write_byte(emu.INTERNAL_RAM_START, 0xA5)
        before = (
            emu.memory.peek_byte_for_preflight(emu.INTERNAL_RAM_START),
            emu.cpu.regs.get(RegisterName.PC),
            emu._poisoned,
        )

        with pytest.raises(RuntimeError, match="immutable.*dynamic"):
            emu.reset()

        assert normal_reads == []
        assert (
            emu.memory.peek_byte_for_preflight(emu.INTERNAL_RAM_START),
            emu.cpu.regs.get(RegisterName.PC),
            emu._poisoned,
        ) == before
    finally:
        emu.close()


@pytest.mark.parametrize("failure_mode", ("read_error", "mismatch"))
def test_machine_reset_architectural_vector_failure_precedes_machine_reset_and_poisons(
    failure_mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SC62015_CPU_BACKEND", "python")
    emu = PCE500Emulator(perfetto_trace=False, save_lcd_on_exit=False)
    try:
        target = 0xC0200
        _install_static_vector_rom(emu, 0xFFFFD, target)
        original_read = emu.memory.read_byte
        architectural_vector_reads: list[int] = []

        def normal_read(address: int, cpu_pc: int | None = None) -> int:
            if 0xFFFFD <= address <= 0xFFFFF:
                architectural_vector_reads.append(address)
            if failure_mode == "read_error" and address == 0xFFFFE:
                raise RuntimeError("architectural RESET-vector read failed")
            value = original_read(address, cpu_pc)
            if failure_mode == "mismatch" and address == 0xFFFFD:
                value ^= 0x01
            return value

        monkeypatch.setattr(emu.memory, "read_byte", normal_read)
        emu.memory.write_byte(emu.INTERNAL_RAM_START, 0xA5)
        emu.memory.write_byte(0xA008, 0x3F)
        emu.cpu.regs.set(RegisterName.PC, 0x34567)
        emu.cpu.regs.set(RegisterName.S, 0x45678)
        emu.cycle_count = 123
        emu.instruction_count = 45
        before = (
            emu.memory.peek_byte_for_preflight(emu.INTERNAL_RAM_START),
            tuple(emu.lcd.display_on),
            emu.cpu.regs.get(RegisterName.PC),
            emu.cpu.regs.get(RegisterName.S),
            emu.cycle_count,
            emu.instruction_count,
        )
        error = (
            "architectural RESET-vector read failed"
            if failure_mode == "read_error"
            else "fetch disagrees with safe preflight"
        )

        with pytest.raises(RuntimeError, match=error):
            emu.reset()

        assert (
            emu.memory.peek_byte_for_preflight(emu.INTERNAL_RAM_START),
            tuple(emu.lcd.display_on),
            emu.cpu.regs.get(RegisterName.PC),
            emu.cpu.regs.get(RegisterName.S),
            emu.cycle_count,
            emu.instruction_count,
        ) == before
        assert architectural_vector_reads
        assert emu._poisoned is not None
    finally:
        emu.close()


def test_software_ir_control_flow_trace_reuses_executed_pc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SC62015_CPU_BACKEND", "python")
    emu = PCE500Emulator(perfetto_trace=False, save_lcd_on_exit=False)
    normal_reads = 0

    def forbidden_trace_read(_address: int, _pc: int | None) -> int:
        nonlocal normal_reads
        normal_reads += 1
        raise RuntimeError("trace re-read the interrupt vector")

    try:
        emu.memory.add_overlay(
            MemoryOverlay(
                start=0xFFFFA,
                end=0xFFFFC,
                name="trace_vector_read_guard",
                read_handler=forbidden_trace_read,
                preflight_read_handler=lambda _address, _pc: 0,
            )
        )
        emu.memory.write_byte(0x1000, 0xFE)
        instr = emu.cpu.decode_instruction(
            0x1000,
            read_fn=lambda address: emu.memory.peek_byte_for_preflight(address, 0x1000),
        )
        emu.cpu.regs.set(RegisterName.PC, 0x00200)

        emu._trace_control_flow(0x1000, SimpleNamespace(instruction=instr))

        assert normal_reads == 0
    finally:
        emu.close()


def test_native_machine_reset_reuses_wrapper_prefetched_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if "llama" not in available_backends():
        pytest.skip("native LLAMA backend is unavailable")
    monkeypatch.setenv("SC62015_CPU_BACKEND", "llama")
    vector_reads: list[int] = []
    target = 0xC0200
    original_read = PCE500Memory.read_byte
    armed = False

    def one_fetch_only(
        memory: PCE500Memory,
        address: int,
        cpu_pc: int | None = None,
    ) -> int:
        if armed and 0xFFFFD <= address <= 0xFFFFF:
            vector_reads.append(address)
            if len(vector_reads) > 3:
                raise RuntimeError("native reset re-read wrapper-prefetched vector")
        return original_read(memory, address, cpu_pc)

    # The native backend binds the host reader when its CPU is constructed,
    # so instrument the class before creating the machine rather than
    # replacing the instance attribute afterward.
    monkeypatch.setattr(PCE500Memory, "read_byte", one_fetch_only)
    emu = PCE500Emulator(perfetto_trace=False, save_lcd_on_exit=False)

    try:
        _install_static_vector_rom(emu, 0xFFFFD, target)
        armed = True

        emu.reset()

        assert vector_reads == [0xFFFFD, 0xFFFFE, 0xFFFFF]
        assert emu.cpu.regs.get(RegisterName.PC) == target
        assert emu.cpu.backend == "llama"
    finally:
        emu.close()
