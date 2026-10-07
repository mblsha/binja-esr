"""UART reference semantics and operation-by-operation Rust parity fixture."""

import json
import random
from copy import deepcopy
from pathlib import Path

from pce500.peripherals.uart import RxByte, Uart


def test_holding_and_shift_ready_are_separate_from_whole_frame_completion():
    u = Uart()
    u.write_control(0x48)
    assert u.baud() == 1200 and u.status() == 0x18
    assert u.write_tx(0x55) and u.status() == 0x10
    assert u.advance(2 * u.bit_units() - 1) == []
    assert u.advance(1) == [("TxReady", 0x55)] and u.status() == 8
    assert u.write_tx(0xAA) and u.status() == 0
    assert not u.write_tx(0xCC)
    assert u.advance(u.frame_units()) == [("TxComplete", 0x55), ("TxReady", 0xAA)]
    assert u.advance(u.frame_units()) == [("TxComplete", 0xAA)]
    assert u.status() == 0x18
    assert (u.take_tx(), u.take_tx(), u.take_tx()) == (0x55, 0xAA, None)


def test_unread_latch_overrun_seven_bit_mode_and_error_persistence():
    u = Uart()
    u.write_control(0x4A)
    u.queue_rx(RxByte(255, parity_error=True, framing_error=True))
    assert u.advance(u.frame_units()) == [("RxReady", 127)]
    assert u.status() == 0x3D and u.read_rx() == 127 and u.status() == 0x1D
    u.queue_rx(RxByte(0x31))
    u.queue_rx(RxByte(0x32))
    u.advance(2 * u.frame_units())
    assert u.status() == 0x3A and u.read_rx() == 0x32 and u.status() == 0x1A
    u.queue_rx(RxByte(0x33))
    u.advance(u.frame_units())
    assert u.status() == 0x38


def test_elapsed_time_slicing_and_clone_preserve_every_event():
    whole = Uart()
    whole.write_control(0x74)
    whole.write_tx(0x5A)
    for byte in range(10):
        whole.queue_rx(RxByte(byte))
    whole.advance(53)
    sliced = deepcopy(whole)
    events = whole.advance(50_000)
    split_events = [event for _ in range(500) for event in sliced.advance(100)]
    assert split_events == events and vars(whole) == vars(sliced)


def test_reset_break_and_bounded_host_queues():
    u = Uart()
    assert not u.write_tx(1) and not u.queue_rx(RxByte(1))
    u.write_control(0xC8)
    u.write_tx(0x44)
    u.advance(2 * u.bit_units() + u.frame_units())
    assert u.suppressed_break_frames == 1 and u.take_tx() is None
    u.write_control(0x48)
    u.write_tx(1)
    u.queue_rx(RxByte(2))
    u.write_control(0)
    assert u.advance(1_000_000) == [] and u.status() == 0x18
    u.write_control(0x48)
    for i in range(4097):
        u.write_tx(i & 255)
        u.advance(2 * u.bit_units() + u.frame_units())
    assert len(u.completed_tx) == 4096 and u.dropped_tx == 1
    for _ in range(4096):
        assert u.queue_rx(RxByte(3))
    assert not u.queue_rx(RxByte(4))


def uart_corpus():
    rng = random.Random(9600707)
    ops = [["tx", 1], ["rx", 2, False, False, False], ["control", 0x48]]
    ops += [["tx", 0x55], ["advance", 1707], ["advance", 1], ["tx", 0xAA]]
    ops += [["advance", 8539], ["advance", 1], ["advance", 8540], ["take"], ["take"]]
    ops += [["rx", 255, True, False, True], ["advance", 8540], ["read"]]
    for _ in range(160):
        choice = rng.randrange(6)
        if choice == 0:
            ops.append(["control", rng.randrange(256)])
        elif choice == 1:
            ops.append(["tx", rng.randrange(256)])
        elif choice == 2:
            ops.append(
                [
                    "rx",
                    rng.randrange(256),
                    bool(rng.randrange(2)),
                    False,
                    bool(rng.randrange(2)),
                ]
            )
        elif choice == 3:
            ops.append(["advance", rng.randrange(50_000)])
        else:
            ops.append(["read" if choice == 4 else "take"])
    ops += [["clone"], ["advance", 100_000], ["restore"], ["advance", 100_000]]
    ops += [["control", 0], ["advance", 100_000], ["read"], ["take"]]
    u, clone = Uart(), None
    replies = []
    for op in ops:
        result = None
        if op[0] == "control":
            u.write_control(op[1])
        elif op[0] == "tx":
            result = u.write_tx(op[1])
        elif op[0] == "rx":
            result = u.queue_rx(RxByte(*op[1:]))
        elif op[0] == "advance":
            result = u.advance(op[1])
        elif op[0] == "read":
            result = u.read_rx()
        elif op[0] == "take":
            result = u.take_tx()
        elif op[0] == "clone":
            clone = deepcopy(u)
        elif op[0] == "restore":
            u = deepcopy(clone)
        replies.append(dict(result=result, state=u.report()))
    return json.loads(json.dumps(dict(operations=ops, replies=replies)))


def test_rust_fixture_is_current_python_reference():
    path = Path(__file__).parents[2] / "sc62015/core/data/uart_reference.json"
    assert json.loads(path.read_text()) == uart_corpus()
