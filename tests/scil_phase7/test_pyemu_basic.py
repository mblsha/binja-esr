from __future__ import annotations

from typing import Iterable, List, Tuple

from sc62015.decoding import decode_map
from sc62015.decoding.dispatcher import CompatDispatcher
from sc62015.decoding.reader import StreamCtx
from sc62015.scil.pyemu import CPUState, MemoryBus, execute_decoded
from sc62015.scil import from_decoded
from sc62015.pysc62015.constants import PC_MASK
from sc62015.pysc62015.instr.opcodes import IMEMRegisters


def _decode(opcode: int, operands: Iterable[int], pc: int) -> decode_map.DecodedInstr:
    data = bytes([opcode, *operands])
    ctx = StreamCtx(pc=pc, data=data[1:], base_len=1)
    return decode_map.decode_opcode(opcode, ctx)


def _decode_sequence(chunks: List[bytes], pc: int) -> List[Tuple[int, decode_map.DecodedInstr]]:
    disp = CompatDispatcher()
    records: List[Tuple[int, decode_map.DecodedInstr]] = []
    cursor = pc
    for raw in chunks:
        res = disp.try_decode(raw, cursor)
        assert res is not None
        length, decoded = res
        if decoded is not None:
            records.append((cursor, decoded))
        cursor = (cursor + length) & PC_MASK
    return records


def test_pyemu_moves_immediate_into_a() -> None:
    state = CPUState()
    state.pc = 0x1000
    bus = MemoryBus()

    decoded = _decode(0x08, [0x5A], state.pc)
    execute_decoded(state, bus, decoded)

    assert state.get_reg("A", 8) == 0x5A
    assert state.pc == 0x1002


def test_pyemu_jrz_updates_pc_when_taken() -> None:
    state = CPUState()
    state.pc = 0x2000
    state.set_flag("Z", 1)
    bus = MemoryBus()

    decoded = _decode(0x18, [0x04], state.pc)
    execute_decoded(state, bus, decoded)

    assert state.pc == 0x2006  # fallthrough (0x2002) + disp (0x4)


def test_pyemu_jp_absolute_uses_page_join() -> None:
    state = CPUState()
    state.pc = 0x34567
    bus = MemoryBus()
    decoded = _decode(0x02, [0x34, 0x12], state.pc)

    execute_decoded(state, bus, decoded)

    assert state.pc == 0x31234


def test_pyemu_external_load_and_store() -> None:
    state = CPUState()
    state.pc = 0x4000
    bus = MemoryBus()
    bus.preload_external([(0x00_20_10, 0xAA)])

    load = _decode(0x88, [0x10, 0x20, 0x00], state.pc)
    execute_decoded(state, bus, load)
    assert state.get_reg("A", 8) == 0xAA

    store = _decode(0xA8, [0x12, 0x20, 0x00], state.pc)
    state.set_reg("A", 0xCC, 8)
    execute_decoded(state, bus, store)
    assert bus.dump_external()[0x00_20_12] == 0xCC


def test_pyemu_pre_sequence_consumes_exactly_once() -> None:
    state = CPUState()
    state.pc = 0x1800
    bus = MemoryBus()
    bus.preload_internal(
        [
            (IMEMRegisters.BP.value, 0x20),
            (0x05, 0x5A),  # direct byte (used when PRE => (n))
            (0x25, 0x11),  # base + offset (used without PRE)
        ]
    )

    records = _decode_sequence([b"\x32", b"\x80\x05"], state.pc)
    # Only the consumer appears in records
    assert len(records) == 1
    consumer_pc, decoded = records[0]
    assert consumer_pc == (state.pc + 1) & PC_MASK

    # Manually advance PC for the PRE prefix
    state.pc = consumer_pc
    execute_decoded(state, bus, decoded)

    assert state.get_reg("A", 8) == 0x5A

    # Second execution without PRE should fall back to BP+n (0x25 -> 0x11)
    state.set_reg("A", 0x00, 8)
    state.pc = consumer_pc
    fresh = _decode(0x80, [0x05], state.pc)
    execute_decoded(state, bus, fresh)
    assert state.get_reg("A", 8) == 0x11
