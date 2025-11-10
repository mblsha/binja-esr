from __future__ import annotations

from sc62015.decoding import decode_map
from sc62015.decoding.reader import StreamCtx
from sc62015.scil.pyemu import CPUState, MemoryBus, execute_decoded


def _decode(hex_bytes: str, pc: int):
    data = bytes.fromhex(hex_bytes)
    ctx = StreamCtx(pc=pc, data=data[1:], base_len=1)
    return decode_map.decode_opcode(data[0], ctx)


def test_adcl_mem_mem_clears_carry_and_zero_flags() -> None:
    state = CPUState()
    state.pc = 0x2100
    state.set_reg("I", 2, 16)
    state.set_flag("C", 0)
    bus = MemoryBus()
    bus.preload_internal(
        [
            (0x10, 0xFF),
            (0x11, 0x01),
            (0x20, 0x01),
            (0x21, 0x02),
        ]
    )

    decoded = _decode("541020", state.pc)
    execute_decoded(state, bus, decoded)

    mem = bus.dump_internal()
    assert mem[0x10] == 0x00
    assert mem[0x11] == 0x04
    assert state.get_reg("I", 16) == 0
    assert state.get_flag("C") == 0
    assert state.get_flag("Z") == 0


def test_adcl_mem_reg_uses_register_source_each_byte() -> None:
    state = CPUState()
    state.pc = 0x2200
    state.set_reg("I", 1, 16)
    state.set_reg("A", 0x34, 8)
    state.set_flag("C", 0)
    bus = MemoryBus()
    bus.preload_internal([(0x40, 0x12)])

    decoded = _decode("5540", state.pc)
    execute_decoded(state, bus, decoded)

    assert bus.dump_internal()[0x40] == 0x46
    assert state.get_flag("C") == 0
    assert state.get_flag("Z") == 0


def test_sbcl_mem_mem_sets_borrow_flag() -> None:
    state = CPUState()
    state.pc = 0x2300
    state.set_reg("I", 1, 16)
    state.set_flag("C", 0)
    bus = MemoryBus()
    bus.preload_internal(
        [
            (0x50, 0x10),
            (0x60, 0x20),
        ]
    )

    decoded = _decode("5C5060", state.pc)
    execute_decoded(state, bus, decoded)

    assert bus.dump_internal()[0x50] == 0xF0
    assert state.get_flag("C") == 1
    assert state.get_reg("I", 16) == 0
