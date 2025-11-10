from binja_test_mocks.mock_llil import MockLowLevelILFunction

from sc62015.decoding.dispatcher import CompatDispatcher


def test_pre_instruction_sets_pending_latch() -> None:
    dispatcher = CompatDispatcher()
    il = MockLowLevelILFunction()
    length, decoded = dispatcher.try_emit(bytes.fromhex("32"), 0x1000, il)
    assert length == 1
    assert decoded is None
    assert dispatcher.pending_pre is not None
    assert il.ils == []


def test_pre_consumed_by_next_pilot_opcode() -> None:
    dispatcher = CompatDispatcher()
    il = MockLowLevelILFunction()
    dispatcher.try_emit(bytes.fromhex("32"), 0x1000, il)
    length, decoded = dispatcher.try_emit(bytes.fromhex("085A"), 0x1000, il)
    assert length == 2
    assert decoded is not None
    assert dispatcher.pending_pre is None
    assert il.ils  # MV emitted once
