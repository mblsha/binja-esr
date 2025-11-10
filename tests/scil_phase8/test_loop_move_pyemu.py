from __future__ import annotations

from sc62015.decoding import decode_map
from sc62015.decoding.reader import StreamCtx
from sc62015.scil.pyemu import CPUState, MemoryBus, execute_decoded


def _decode(hex_bytes: str, pc: int):
    data = bytes.fromhex(hex_bytes)
    ctx = StreamCtx(pc=pc, data=data[1:], base_len=1)
    return decode_map.decode_opcode(data[0], ctx)


def test_mvl_imem_wraparound_and_counter_clear() -> None:
    state = CPUState()
    state.pc = 0x8000
    state.set_reg("I", 4, 16)
    bus = MemoryBus()
    bus.preload_internal(
        [
            (0xF0, 0x11),
            (0xF1, 0x22),
            (0xF2, 0x33),
            (0xF3, 0x44),
        ]
    )

    decoded = _decode("CBFEF0", state.pc)
    execute_decoded(state, bus, decoded)

    mem = bus.dump_internal()
    assert mem[0xFE] == 0x11
    assert mem[0xFF] == 0x22
    assert mem[0x00] == 0x33
    assert mem[0x01] == 0x44
    assert state.get_reg("I", 16) == 0


def test_mvld_imem_moves_backwards_without_clobber() -> None:
    state = CPUState()
    state.pc = 0x9000
    state.set_reg("I", 3, 16)
    bus = MemoryBus()
    bus.preload_internal(
        [
            (0x50, 0xAA),
            (0x4F, 0xBB),
            (0x4E, 0xCC),
        ]
    )

    decoded = _decode("CF5150", state.pc)
    execute_decoded(state, bus, decoded)

    mem = bus.dump_internal()
    assert mem[0x51] == 0xAA
    assert mem[0x50] == 0xBB
    assert mem[0x4F] == 0xCC
    assert state.get_reg("I", 16) == 0
