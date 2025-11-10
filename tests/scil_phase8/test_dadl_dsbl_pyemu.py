from __future__ import annotations

from sc62015.decoding import decode_map
from sc62015.decoding.reader import StreamCtx
from sc62015.scil.pyemu import CPUState, MemoryBus, execute_decoded


def _decode(hex_bytes: str, pc: int):
    data = bytes.fromhex(hex_bytes)
    ctx = StreamCtx(pc=pc, data=data[1:], base_len=1)
    return decode_map.decode_opcode(data[0], ctx)


def test_dadl_mem_mem_bcd_addition() -> None:
    state = CPUState()
    state.pc = 0x2400
    state.set_reg("I", 2, 16)
    state.set_flag("C", 0)
    bus = MemoryBus()
    bus.preload_internal(
        [
            (0x11, 0x50),
            (0x10, 0x01),
            (0x21, 0x50),
            (0x20, 0x02),
        ]
    )

    decoded = _decode("C41121", state.pc)
    execute_decoded(state, bus, decoded)

    mem = bus.dump_internal()
    assert mem[0x11] == 0x00
    assert mem[0x10] == 0x04
    assert state.get_flag("C") == 0
    assert state.get_flag("Z") == 0
    assert state.get_reg("I", 16) == 0


def test_dadl_mem_reg_clears_carry() -> None:
    state = CPUState()
    state.pc = 0x2500
    state.set_reg("I", 2, 16)
    state.set_reg("A", 0x01, 8)
    state.set_flag("C", 1)
    bus = MemoryBus()
    bus.preload_internal(
        [
            (0x11, 0x99),
            (0x10, 0x01),
        ]
    )

    decoded = _decode("C511", state.pc)
    execute_decoded(state, bus, decoded)

    mem = bus.dump_internal()
    assert mem[0x11] == 0x00
    assert mem[0x10] == 0x03
    assert state.get_flag("C") == 0
    assert state.get_flag("Z") == 0


def test_dsbl_mem_mem_sets_borrow() -> None:
    state = CPUState()
    state.pc = 0x2600
    state.set_reg("I", 1, 16)
    state.set_flag("C", 0)
    bus = MemoryBus()
    bus.preload_internal(
        [
            (0x10, 0x20),
            (0x20, 0x05),
        ]
    )

    decoded = _decode("D41020", state.pc)
    execute_decoded(state, bus, decoded)

    assert bus.dump_internal()[0x10] == 0x15
    assert state.get_flag("C") == 0
    assert state.get_flag("Z") == 0
