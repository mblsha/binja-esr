from sc62015.decoding.dispatcher import CompatDispatcher


def test_pre_instruction_sets_pending_latch() -> None:
    dispatcher = CompatDispatcher()
    length, decoded = dispatcher.try_decode(bytes.fromhex("32"), 0x1000)
    assert length == 1
    assert decoded is None
    assert dispatcher.pending_pre is not None


def test_pre_consumed_by_next_pilot_opcode() -> None:
    dispatcher = CompatDispatcher()
    dispatcher.try_decode(bytes.fromhex("32"), 0x1000)
    length, decoded = dispatcher.try_decode(bytes.fromhex("085A"), 0x1000)
    assert length == 2
    assert decoded is not None
    assert dispatcher.pending_pre is None
    assert decoded.pre_applied is not None
