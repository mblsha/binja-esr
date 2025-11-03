"""Tests for the LCD pipeline snapshot helpers."""

from __future__ import annotations

from typing import List, Tuple

from pce500.display.pipeline import (
    LCDOperation,
    LCDPipeline,
    replay_operations,
)
from pce500.display.hd61202 import parse_command


def _op(address: int, value: int, pc: int | None = None) -> LCDOperation:
    return LCDOperation(command=parse_command(address, value), pc=pc)


def test_pipeline_replay_updates_both_chips():
    pipeline = LCDPipeline()
    pipeline.apply(_op(0x2000, 0x3F))  # ON
    pipeline.apply(_op(0x2000, 0xB8))  # SET PAGE 0
    pipeline.apply(_op(0x2000, 0x40))  # SET Y = 0
    pipeline.apply(_op(0x2002, 0x55, pc=0xF20000))

    snapshot = pipeline.snapshot
    assert snapshot.chips[0].vram[0][0] == 0x55
    assert snapshot.chips[1].vram[0][0] == 0x55
    # Instruction/data counters should reflect the writes.
    assert snapshot.chips[0].instruction_count == 3
    assert snapshot.chips[0].data_write_count == 1


def test_pipeline_observer_receives_events_and_snapshot():
    pipeline = LCDPipeline()
    events: List[Tuple[dict, int]] = []

    def _observer(event: dict, snapshot) -> None:
        events.append((event, snapshot.chips[event["chip"]].y_address))

    pipeline.subscribe(_observer)
    pipeline.apply(_op(0x2008, 0x3F))  # LEFT chip only
    pipeline.apply(_op(0x2008, 0xB8))
    pipeline.apply(_op(0x2008, 0x40))
    pipeline.apply(_op(0x200A, 0xAA))

    assert any(evt["type"] == "data" for evt, _ in events)
    last_event, y_address = events[-1]
    assert last_event["type"] == "data"
    assert last_event["chip"] == 0  # left chip
    assert y_address == 1  # column advanced after the write


def test_replay_operations_helper_collects_events():
    operations = [
        _op(0x2000, 0x3F),
        _op(0x2004, 0xB8),  # RIGHT chip page set
        _op(0x2004, 0x40),
        _op(0x2006, 0x7E),
    ]
    snapshot, events = replay_operations(operations)

    assert snapshot.chips[1].vram[0][0] == 0x7E
    assert len(events) == 5
    assert events[-1]["type"] == "data"
    assert events[-1]["chip"] == 1
