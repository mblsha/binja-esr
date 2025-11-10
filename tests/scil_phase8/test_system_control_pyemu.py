from __future__ import annotations

from sc62015.decoding import decode_map
from sc62015.decoding.reader import StreamCtx
from sc62015.scil.pyemu import CPUState, MemoryBus, execute_decoded
from sc62015.pysc62015.instr.opcodes import IMEMRegisters, INTERRUPT_VECTOR_ADDR


def _decode(opcode: int, pc: int):
    data = bytes([opcode])
    ctx = StreamCtx(pc=pc, data=data[1:], base_len=1)
    return decode_map.decode_opcode(opcode, ctx)


def _write_internal(bus: MemoryBus, name: str, value: int) -> None:
    bus.store("int", IMEMRegisters[name].value & 0xFF, value, 8)


def _read_internal(bus: MemoryBus, name: str) -> int:
    return bus.load("int", IMEMRegisters[name].value & 0xFF, 8) & 0xFF


def test_halt_updates_usr_ssr_and_sets_halted() -> None:
    state = CPUState()
    state.pc = 0x4000
    bus = MemoryBus()
    _write_internal(bus, "USR", 0xFF)
    _write_internal(bus, "SSR", 0x00)

    decoded = _decode(0xDE, state.pc)
    execute_decoded(state, bus, decoded)

    assert _read_internal(bus, "USR") & 0x3F == 0x18
    assert _read_internal(bus, "SSR") & 0x04
    assert state.halted is True


def test_off_behaves_like_halt() -> None:
    state = CPUState()
    state.pc = 0x4010
    bus = MemoryBus()
    _write_internal(bus, "USR", 0xFF)
    _write_internal(bus, "SSR", 0x00)

    decoded = _decode(0xDF, state.pc)
    execute_decoded(state, bus, decoded)

    assert _read_internal(bus, "USR") & 0x3F == 0x18
    assert _read_internal(bus, "SSR") & 0x04
    assert state.halted is True


def test_reset_clears_control_registers_and_sets_pc() -> None:
    state = CPUState()
    state.pc = 0x4020
    bus = MemoryBus()
    for reg in ("UCR", "ISR", "SCR", "USR", "LCC", "SSR"):
        _write_internal(bus, reg, 0xFF)
    bus.preload_external(
        [
            (INTERRUPT_VECTOR_ADDR, 0x45),
            (INTERRUPT_VECTOR_ADDR + 1, 0x23),
            (INTERRUPT_VECTOR_ADDR + 2, 0x01),
        ]
    )

    decoded = _decode(0xFF, state.pc)
    execute_decoded(state, bus, decoded)

    assert _read_internal(bus, "UCR") == 0x00
    assert _read_internal(bus, "ISR") == 0x00
    assert _read_internal(bus, "SCR") == 0x00
    assert _read_internal(bus, "LCC") & 0x80 == 0
    assert state.get_reg("PC", 24) == 0x12345 & 0xFFFFF
    assert state.halted is False


def test_wait_leaves_state_unchanged() -> None:
    state = CPUState()
    state.pc = 0x4030
    bus = MemoryBus()
    decoded = _decode(0xEF, state.pc)
    original_pc = state.get_reg("PC", 24)
    execute_decoded(state, bus, decoded)
    assert state.get_reg("PC", 24) == (original_pc + decoded.length) & 0xFFFFF


def test_ir_pushes_context_and_jumps_to_vector() -> None:
    state = CPUState()
    state.pc = 0x4040
    original_pc = state.pc
    state.set_reg("S", 0x000200, 24)
    state.set_reg("F", 0xA5, 8)
    bus = MemoryBus()
    bus.preload_external(
        [
            (INTERRUPT_VECTOR_ADDR, 0xAA),
            (INTERRUPT_VECTOR_ADDR + 1, 0xBB),
            (INTERRUPT_VECTOR_ADDR + 2, 0x01),
        ]
    )
    _write_internal(bus, "IMR", 0xFF)

    decoded = _decode(0xFE, state.pc)
    execute_decoded(state, bus, decoded)

    new_sp = state.get_reg("S", 24)
    assert new_sp == (0x000200 - 5) & ((1 << 24) - 1)
    stack_bytes = [
        bus.load("ext", (new_sp + i) & ((1 << 24) - 1), 8) & 0xFF for i in range(5)
    ]
    assert stack_bytes[0] == (original_pc & 0xFF)
    assert stack_bytes[1] == ((original_pc >> 8) & 0xFF)
    assert stack_bytes[2] == ((original_pc >> 16) & 0xFF)
    assert stack_bytes[3] == 0xA5
    assert stack_bytes[4] == 0xFF
    assert state.get_reg("PC", 24) == 0x01BBAA & 0xFFFFF
