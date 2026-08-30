"""Test HALT/OFF/RESET instruction behavior."""

import pytest
from binja_test_mocks.eval_llil import Memory
from .emulator import Emulator, RegisterName, validate_vector_transfer
from .sc_asm import Assembler
from .constants import ADDRESS_SPACE_SIZE, INTERNAL_MEMORY_START
from .instr import InvalidInstruction
from .instr.opcodes import IMEMRegisters


def create_memory():
    """Create a memory object with backing storage."""
    raw = bytearray([0x00] * ADDRESS_SPACE_SIZE)

    def read_mem(addr: int) -> int:
        if addr < 0 or addr >= len(raw):
            raise IndexError(f"Read address {addr:04x} out of bounds")
        return raw[addr]

    def write_mem(addr: int, value: int) -> None:
        if addr < 0 or addr >= len(raw):
            raise IndexError(f"Write address {addr:04x} out of bounds")
        raw[addr] = value & 0xFF

    memory = Memory(read_mem, write_mem)

    def peek_byte_for_preflight(addr: int, _pc: int | None = None) -> int:
        if addr < 0 or addr >= len(raw):
            raise IndexError(f"Preflight address {addr:04x} out of bounds")
        return raw[addr]

    setattr(memory, "peek_byte_for_preflight", peek_byte_for_preflight)
    return memory, raw


def test_reset_disabled_starts_in_running_power_state():
    memory, _ = create_memory()
    emu = Emulator(memory, reset_on_init=False)
    assert emu.state.halted is False
    assert getattr(emu.state, "power_state") == "running"


def test_halt_model_contract():
    """Pin the provisional HALT register model; this is not hardware proof."""
    memory, _ = create_memory()

    # Set up initial values
    memory.write_byte(
        INTERNAL_MEMORY_START + IMEMRegisters.USR, 0xFF
    )  # USR with all bits set
    memory.write_byte(
        INTERNAL_MEMORY_START + IMEMRegisters.SSR, 0x00
    )  # SSR with all bits clear

    # Write HALT instruction at address 0x1000
    assembler = Assembler()
    bin_file = assembler.assemble("HALT")
    for i, byte in enumerate(bin_file.segments[0].data):
        memory.write_byte(0x1000 + i, byte)

    # Create emulator without reset
    emu = Emulator(memory, reset_on_init=False)
    emu.regs.set(RegisterName.PC, 0x1000)

    # Execute HALT
    emu.execute_instruction(0x1000)

    # Check register modifications
    usr = memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.USR)
    assert (usr & 0x3F) == 0x18  # Bits 0-5: only bits 3,4 should be set

    ssr = memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.SSR)
    assert (ssr & 0x04) == 0x04  # Bit 2 should be set

    # Check halted state
    assert emu.state.halted is True
    assert getattr(emu.state, "power_state") == "halted"


def test_off_model_contract():
    """Pin the provisional OFF register model and keep it distinct from HALT."""
    memory, _ = create_memory()

    # Set up initial values
    memory.write_byte(
        INTERNAL_MEMORY_START + IMEMRegisters.USR, 0xFF
    )  # USR with all bits set
    memory.write_byte(
        INTERNAL_MEMORY_START + IMEMRegisters.SSR, 0x00
    )  # SSR with all bits clear

    # Write OFF instruction at address 0x1000
    assembler = Assembler()
    bin_file = assembler.assemble("OFF")
    for i, byte in enumerate(bin_file.segments[0].data):
        memory.write_byte(0x1000 + i, byte)

    # Create emulator without reset
    emu = Emulator(memory, reset_on_init=False)
    emu.regs.set(RegisterName.PC, 0x1000)

    # Execute OFF
    emu.execute_instruction(0x1000)

    # Check register modifications (same as HALT)
    usr = memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.USR)
    assert (usr & 0x3F) == 0x18  # Bits 0-5: only bits 3,4 should be set

    ssr = memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.SSR)
    assert (ssr & 0x04) == 0x04  # Bit 2 should be set

    # Check halted state
    assert emu.state.halted is True
    assert getattr(emu.state, "power_state") == "off"


def test_reset_model_contract_and_distinct_vector_slot():
    """Pin the synthetic reset model and its distinct vector-slot contract."""
    memory, _ = create_memory()

    # Set up initial values
    memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.UCR, 0xFF)  # UCR
    memory.write_byte(
        INTERNAL_MEMORY_START + IMEMRegisters.USR, 0xFF
    )  # USR with all bits set
    memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR, 0xFF)  # ISR
    memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.SCR, 0xFF)  # SCR
    memory.write_byte(
        INTERNAL_MEMORY_START + IMEMRegisters.LCC, 0xFF
    )  # LCC with bit 7 set
    memory.write_byte(
        INTERNAL_MEMORY_START + IMEMRegisters.SSR, 0xFF
    )  # SSR with bit 2 set

    # Keep a distinct interrupt-vector decoy at FFFFA and place a synthetic
    # reset target in the separate FFFFD vector slot.
    memory.write_byte(0xFFFFA, 0xAA)
    memory.write_byte(0xFFFFB, 0xBB)
    memory.write_byte(0xFFFFC, 0x0C)
    memory.write_byte(0xFFFFD, 0x45)  # Low byte
    memory.write_byte(0xFFFFE, 0x23)  # Middle byte
    memory.write_byte(0xFFFFF, 0x01)  # High byte

    # Write RESET instruction at address 0x1000
    assembler = Assembler()
    bin_file = assembler.assemble("RESET")
    for i, byte in enumerate(bin_file.segments[0].data):
        memory.write_byte(0x1000 + i, byte)

    # Create emulator without reset
    emu = Emulator(memory, reset_on_init=False)
    emu.regs.set(RegisterName.PC, 0x1000)

    # Set some register values that should be retained
    emu.regs.set(RegisterName.A, 0x55)
    emu.regs.set(RegisterName.B, 0xAA)
    emu.regs.set(RegisterName.FC, 1)
    emu.regs.set(RegisterName.FZ, 1)

    # Execute RESET
    emu.execute_instruction(0x1000)

    # Check register modifications
    assert (
        memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.UCR) == 0x00
    )  # UCR reset
    assert (
        memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR) == 0x00
    )  # ISR reset (clears interrupt status)
    assert (
        memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.SCR) == 0x00
    )  # SCR reset

    usr = memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.USR)
    assert (usr & 0x3F) == 0x18  # Bits 0-5: only bits 3,4 should be set

    lcc = memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.LCC)
    assert (lcc & 0x80) == 0x00  # Bit 7 should be clear

    ssr = memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.SSR)
    assert (ssr & 0x04) == 0x00  # Bit 2 should be clear

    # Check PC set to reset vector
    assert emu.regs.get(RegisterName.PC) == 0x12345

    # Check retained registers
    assert emu.regs.get(RegisterName.A) == 0x55
    assert emu.regs.get(RegisterName.B) == 0xAA
    assert emu.regs.get(RegisterName.FC) == 1
    assert emu.regs.get(RegisterName.FZ) == 1
    assert emu.state.halted is False
    assert getattr(emu.state, "power_state") == "running"


def test_power_on_reset_model_contract():
    """Pin the provisional power-on register model and reset-vector fetch."""
    memory, _ = create_memory()

    # Set up initial values
    memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.UCR, 0xFF)  # UCR
    memory.write_byte(
        INTERNAL_MEMORY_START + IMEMRegisters.USR, 0xFF
    )  # USR with all bits set
    memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR, 0xFF)  # ISR
    memory.write_byte(INTERNAL_MEMORY_START + IMEMRegisters.SCR, 0xFF)  # SCR
    memory.write_byte(
        INTERNAL_MEMORY_START + IMEMRegisters.LCC, 0xFF
    )  # LCC with bit 7 set
    memory.write_byte(
        INTERNAL_MEMORY_START + IMEMRegisters.SSR, 0xFF
    )  # SSR with bit 2 set

    # FFFFA is the interrupt vector; power-on RESET reads FFFFD instead.
    memory.write_byte(0xFFFFA, 0xAA)
    memory.write_byte(0xFFFFB, 0xBB)
    memory.write_byte(0xFFFFC, 0x0C)
    memory.write_byte(0xFFFFD, 0x21)  # Low byte
    memory.write_byte(0xFFFFE, 0x43)  # Middle byte
    memory.write_byte(0xFFFFF, 0x05)  # High byte

    # Create emulator with reset
    emu = Emulator(memory, reset_on_init=True)

    assert emu.state.halted is False
    assert getattr(emu.state, "power_state") == "running"

    # Check register modifications
    assert (
        memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.UCR) == 0x00
    )  # UCR reset
    assert (
        memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.ISR) == 0x00
    )  # ISR reset (clears interrupt status)
    assert (
        memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.SCR) == 0x00
    )  # SCR reset

    usr = memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.USR)
    assert (usr & 0x3F) == 0x18  # Bits 0-5: only bits 3,4 should be set

    lcc = memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.LCC)
    assert (lcc & 0x80) == 0x00  # Bit 7 should be clear

    ssr = memory.read_byte(INTERNAL_MEMORY_START + IMEMRegisters.SSR)
    assert (ssr & 0x04) == 0x00  # Bit 2 should be clear

    # Check PC set to reset vector
    assert emu.regs.get(RegisterName.PC) == 0x54321

    # Check halted state is false
    assert emu.state.halted is False


def test_failed_power_on_reset_restores_native_state_and_requires_complete_reset():
    raw = bytearray([0x00] * ADDRESS_SPACE_SIZE)
    raw[0] = 0x00  # NOP used to prove poisoned execution is blocked.
    raw[0xFFFFD:0x100000] = bytes.fromhex("452301")
    fail_reset_write = True

    def read_mem(addr: int) -> int:
        return raw[addr]

    def write_mem(addr: int, value: int) -> None:
        nonlocal fail_reset_write
        if fail_reset_write and addr == INTERNAL_MEMORY_START + IMEMRegisters.UCR:
            raise RuntimeError("reset write failed")
        raw[addr] = value & 0xFF

    memory = Memory(read_mem, write_mem)
    setattr(
        memory,
        "peek_byte_for_preflight",
        lambda addr, _pc=None: raw[addr],
    )
    emu = Emulator(memory, reset_on_init=False)
    emu.regs.set(RegisterName.PC, 0x22222)
    emu.regs.set(RegisterName.A, 0x5A)
    emu.regs.call_sub_level = 4
    emu.state.halted = True
    setattr(emu.state, "power_state", "halted")

    with pytest.raises(RuntimeError, match="reset write failed"):
        emu.power_on_reset()

    assert emu.regs.get(RegisterName.PC) == 0x22222
    assert emu.regs.get(RegisterName.A) == 0x5A
    assert emu.regs.call_sub_level == 4
    assert emu.state.halted is True
    assert getattr(emu.state, "power_state") == "halted"
    with pytest.raises(RuntimeError, match="poisoned.*reset required"):
        emu.execute_instruction(0)

    fail_reset_write = False
    emu.power_on_reset()
    assert emu.regs.get(RegisterName.PC) == 0x12345
    assert emu.state.halted is False
    assert getattr(emu.state, "power_state") == "running"
    emu.execute_instruction(0)
    assert emu.regs.get(RegisterName.PC) == 1


def test_ir_noncanonical_vector_fails_atomically_via_safe_peek():
    memory, raw = create_memory()
    raw[0x1000] = 0xFE  # IR
    raw[0xFFFFA:0xFFFFD] = bytes.fromhex("0001F0")  # raw F00100
    raw[INTERNAL_MEMORY_START + IMEMRegisters.IMR] = 0xA5

    emu = Emulator(memory, reset_on_init=False)
    emu.regs.set(RegisterName.PC, 0x1000)
    emu.regs.set(RegisterName.S, 0x00400)
    emu.regs.set(RegisterName.F, 0x03)
    emu.regs.call_sub_level = 7
    before_regs = dict(emu.regs._values)
    before_memory = bytes(raw)

    with pytest.raises(NotImplementedError, match="noncanonical vector 0xF00100"):
        emu.execute_instruction(0x1000)

    assert dict(emu.regs._values) == before_regs
    assert emu.regs.call_sub_level == 7
    assert bytes(raw) == before_memory
    assert emu._poisoned is None


def test_reset_invalid_destination_fails_before_any_sfr_write():
    memory, raw = create_memory()
    raw[0x1000] = 0xFF  # RESET
    raw[0xFFFFD:0x100000] = bytes.fromhex("000200")
    raw[0x00200] = 0x20  # reserved opcode
    for offset, value in (
        (IMEMRegisters.LCC, 0xFF),
        (IMEMRegisters.UCR, 0xA1),
        (IMEMRegisters.USR, 0xB2),
        (IMEMRegisters.ISR, 0xC3),
        (IMEMRegisters.SCR, 0xD4),
        (IMEMRegisters.SSR, 0xE5),
    ):
        raw[INTERNAL_MEMORY_START + offset] = value

    emu = Emulator(memory, reset_on_init=False)
    emu.regs.set(RegisterName.PC, 0x1000)
    emu.regs.set(RegisterName.A, 0x5A)
    emu.state.halted = True
    setattr(emu.state, "power_state", "halted")
    before_regs = dict(emu.regs._values)
    before_state = dict(vars(emu.state))
    before_memory = bytes(raw)

    with pytest.raises(InvalidInstruction, match="opcode 0x20.*0x00200"):
        emu.execute_instruction(0x1000)

    assert dict(emu.regs._values) == before_regs
    assert dict(vars(emu.state)) == before_state
    assert bytes(raw) == before_memory
    assert emu._poisoned is None


def test_power_on_reset_requires_safe_peek_before_normal_bus_reads():
    raw = bytearray([0x00] * ADDRESS_SPACE_SIZE)
    raw[0xFFFFD:0x100000] = bytes.fromhex("000100")
    normal_reads: list[int] = []
    writes: list[tuple[int, int]] = []

    def read_mem(addr: int) -> int:
        normal_reads.append(addr)
        return raw[addr]

    def write_mem(addr: int, value: int) -> None:
        writes.append((addr, value))
        raw[addr] = value & 0xFF

    emu = Emulator(Memory(read_mem, write_mem), reset_on_init=False)
    before_regs = dict(emu.regs._values)

    with pytest.raises(RuntimeError, match="peek_byte_for_preflight"):
        emu.power_on_reset()

    assert normal_reads == []
    assert writes == []
    assert dict(emu.regs._values) == before_regs
    assert emu._poisoned is None


@pytest.mark.parametrize("target_opcode", (0xFE, 0xFF, 0xEF))
def test_vector_static_preflight_does_not_recurse_or_apply_i_dependent_checks(
    target_opcode: int,
):
    memory, raw = create_memory()
    target = 0x00200
    raw[0xFFFFD:0x100000] = target.to_bytes(3, "little")
    raw[target] = target_opcode  # IR, RESET, or I-counted WAIT
    emu = Emulator(memory, reset_on_init=False)
    emu.regs.set(RegisterName.I, 0)

    assert validate_vector_transfer(memory, emu.regs, 0xFFFFD) == target


@pytest.mark.parametrize(
    ("target_bytes", "error_type", "error"),
    (
        (bytes.fromhex("20"), InvalidInstruction, "opcode 0x20"),
        (bytes.fromhex("313100"), InvalidInstruction, "PRE"),
        (bytes.fromhex("CE"), NotImplementedError, "TCL timer-clear"),
    ),
)
def test_vector_static_preflight_rejects_invalid_pre_and_tcl_targets(
    target_bytes: bytes,
    error_type: type[Exception],
    error: str,
):
    memory, raw = create_memory()
    target = 0x00200
    raw[0xFFFFD:0x100000] = target.to_bytes(3, "little")
    raw[target : target + len(target_bytes)] = target_bytes
    emu = Emulator(memory, reset_on_init=False)

    with pytest.raises(error_type, match=error):
        validate_vector_transfer(memory, emu.regs, 0xFFFFD)


def test_ir_architectural_vector_mismatch_fails_before_frame_or_imr_write():
    raw = bytearray([0x00] * ADDRESS_SPACE_SIZE)
    source = 0x1000
    target = 0x0200
    raw[source] = 0xFE
    raw[0xFFFFA:0xFFFFD] = target.to_bytes(3, "little")
    raw[INTERNAL_MEMORY_START + IMEMRegisters.IMR] = 0xA5
    normal_vector = (target + 1).to_bytes(3, "little")
    writes: list[tuple[int, int]] = []

    def read_mem(addr: int) -> int:
        if 0xFFFFA <= addr <= 0xFFFFC:
            return normal_vector[addr - 0xFFFFA]
        return raw[addr]

    def write_mem(addr: int, value: int) -> None:
        writes.append((addr, value))
        raw[addr] = value & 0xFF

    memory = Memory(read_mem, write_mem)
    setattr(memory, "peek_byte_for_preflight", lambda addr, _pc=None: raw[addr])
    emu = Emulator(memory, reset_on_init=False)
    emu.regs.set(RegisterName.PC, source)
    emu.regs.set(RegisterName.S, 0x400)
    before_regs = dict(emu.regs._values)
    before_memory = bytes(raw)

    with pytest.raises(RuntimeError, match="fetch disagrees with safe preflight"):
        emu.execute_instruction(source)

    assert dict(emu.regs._values) == before_regs
    assert bytes(raw) == before_memory
    assert writes == []
    assert emu._poisoned is not None


def test_power_on_reset_vector_read_failure_precedes_all_sfr_writes():
    raw = bytearray([0x00] * ADDRESS_SPACE_SIZE)
    target = 0x0200
    raw[0xFFFFD:0x100000] = target.to_bytes(3, "little")
    for offset in (
        IMEMRegisters.LCC,
        IMEMRegisters.UCR,
        IMEMRegisters.USR,
        IMEMRegisters.ISR,
        IMEMRegisters.SCR,
        IMEMRegisters.SSR,
    ):
        raw[INTERNAL_MEMORY_START + offset] = 0xA5
    writes: list[tuple[int, int]] = []

    def read_mem(addr: int) -> int:
        if addr == 0xFFFFE:
            raise RuntimeError("architectural reset-vector read failed")
        return raw[addr]

    def write_mem(addr: int, value: int) -> None:
        writes.append((addr, value))
        raw[addr] = value & 0xFF

    memory = Memory(read_mem, write_mem)
    setattr(memory, "peek_byte_for_preflight", lambda addr, _pc=None: raw[addr])
    emu = Emulator(memory, reset_on_init=False)
    emu.regs.set(RegisterName.PC, 0x12345)
    before_regs = dict(emu.regs._values)
    before_memory = bytes(raw)

    with pytest.raises(RuntimeError, match="reset-vector read failed"):
        emu.power_on_reset()

    assert dict(emu.regs._values) == before_regs
    assert bytes(raw) == before_memory
    assert writes == []
    assert emu._poisoned is not None


if __name__ == "__main__":
    pytest.main([__file__])
