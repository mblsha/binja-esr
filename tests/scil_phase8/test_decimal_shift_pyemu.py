from __future__ import annotations

from sc62015.decoding import decode_map
from sc62015.decoding.reader import StreamCtx
from sc62015.scil.pyemu import CPUState, MemoryBus, execute_decoded
from sc62015.pysc62015.test_emulator import compute_expected_dsll, compute_expected_dsrl


def _decode(opcode: int, operand: int, pc: int):
    data = bytes([opcode, operand])
    ctx = StreamCtx(pc=pc, data=data[1:], base_len=1)
    return decode_map.decode_opcode(opcode, ctx)


def test_dsll_shifts_digits_left_and_wraps_addresses() -> None:
    state = CPUState()
    state.pc = 0x2A00
    state.set_reg("I", 2, 16)
    bus = MemoryBus()
    # DSLL treats operand as MSB address; data stored descending
    bus.preload_internal(
        [
            (0x11, 0x12),  # MSB
            (0x10, 0x34),  # LSB
        ]
    )

    decoded = _decode(0xEC, 0x11, state.pc)
    execute_decoded(state, bus, decoded)

    expected = compute_expected_dsll([0x12, 0x34])
    mem = bus.dump_internal()
    assert mem[0x11] == expected[0]
    assert mem[0x10] == expected[1]
    assert state.get_flag("Z") == 0
    assert state.get_reg("I", 16) == 0


def test_dsrl_shifts_digits_right_and_sets_zero_flag() -> None:
    state = CPUState()
    state.pc = 0x2B00
    state.set_reg("I", 2, 16)
    bus = MemoryBus()
    # For DSRL, operand is LSB address; data stored ascending
    bus.preload_internal(
        [
            (0x10, 0x34),  # LSB
            (0x11, 0x12),  # MSB
        ]
    )

    decoded = _decode(0xFC, 0x10, state.pc)
    execute_decoded(state, bus, decoded)

    expected = compute_expected_dsrl([0x34, 0x12])
    mem = bus.dump_internal()
    assert mem[0x10] == expected[0]
    assert mem[0x11] == expected[1]
    assert state.get_flag("Z") == 0
    assert state.get_reg("I", 16) == 0
