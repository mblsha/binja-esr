from __future__ import annotations

from sc62015.decoding import decode_map
from sc62015.decoding.reader import StreamCtx
from sc62015.scil.pyemu import CPUState, MemoryBus, execute_decoded


def _decode(hex_bytes: str, pc: int):
    data = bytes.fromhex(hex_bytes)
    ctx = StreamCtx(pc=pc, data=data[1:], base_len=1)
    return decode_map.decode_opcode(data[0], ctx)


def test_pmdf_immediate_adds_value_to_internal_byte() -> None:
    state = CPUState()
    state.pc = 0x3000
    state.set_reg("I", 0, 16)
    bus = MemoryBus()
    bus.preload_internal([(0x10, 0x12)])

    decoded = _decode("471003", state.pc)
    execute_decoded(state, bus, decoded)

    assert bus.dump_internal()[0x10] == 0x15


def test_pmdf_register_uses_a_register_value() -> None:
    state = CPUState()
    state.pc = 0x3010
    state.set_reg("I", 0, 16)
    state.set_reg("A", 0x20, 8)
    bus = MemoryBus()
    bus.preload_internal([(0x07, 0x11)])

    decoded = _decode("5707", state.pc)
    execute_decoded(state, bus, decoded)

    assert bus.dump_internal()[0x07] == 0x31
