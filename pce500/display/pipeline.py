"""Event-driven LCD pipeline with snapshot support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .hd61202 import ChipSelect, Command, HD61202, parse_command

Observer = Callable[[Dict[str, object], "LCDSnapshot"], None]


@dataclass(frozen=True)
class LCDChipSnapshot:
    """Immutable representation of a single HD61202 chip."""

    on: bool
    start_line: int
    page: int
    y_address: int
    vram: Tuple[Tuple[int, ...], ...]
    instruction_count: int
    data_write_count: int


@dataclass(frozen=True)
class LCDSnapshot:
    """Snapshot of the full PC-E500 display (two chips)."""

    chips: Tuple[LCDChipSnapshot, ...]


@dataclass(frozen=True)
class LCDOperation:
    """One parsed LCD command coupled with optional metadata."""

    command: Command
    pc: Optional[int] = None


def _chip_indices(cs: ChipSelect) -> Sequence[int]:
    if cs == ChipSelect.BOTH:
        return (0, 1)
    if cs == ChipSelect.RIGHT:
        return (1,)
    if cs == ChipSelect.LEFT:
        return (0,)
    return ()


def _snapshot_from_chips(chips: Sequence[HD61202]) -> LCDSnapshot:
    capture = []
    for chip in chips:
        capture.append(
            LCDChipSnapshot(
                on=chip.state.on,
                start_line=chip.state.start_line,
                page=chip.state.page,
                y_address=chip.state.y_address,
                vram=tuple(tuple(row) for row in chip.vram),
                instruction_count=chip.instruction_count,
                data_write_count=chip.data_write_count,
            )
        )
    return LCDSnapshot(tuple(capture))


class LCDPipeline:
    """Replay LCD commands, emit events, and expose VRAM snapshots."""

    def __init__(self, chips: Optional[Sequence[HD61202]] = None):
        self._chips: List[HD61202] = (
            list(chips)
            if chips is not None
            else [
                HD61202(),
                HD61202(),
            ]
        )
        self._observers: List[Observer] = []

    @property
    def chips(self) -> List[HD61202]:
        return self._chips

    @property
    def snapshot(self) -> LCDSnapshot:
        return _snapshot_from_chips(self._chips)

    def subscribe(self, observer: Observer) -> None:
        self._observers.append(observer)

    def apply(self, operation: LCDOperation) -> List[Dict[str, object]]:
        return self._apply_command(operation.command, operation.pc)

    def apply_raw(
        self, address: int, value: int, pc: Optional[int] = None
    ) -> List[Dict[str, object]]:
        command = parse_command(address, value)
        return self._apply_command(command, pc)

    def replay(self, operations: Iterable[LCDOperation]) -> LCDSnapshot:
        for operation in operations:
            self.apply(operation)
        return self.snapshot

    def _dispatch(self, events: List[Dict[str, object]]) -> None:
        if not events or not self._observers:
            return
        snap = self.snapshot
        for event in events:
            for observer in self._observers:
                observer(dict(event), snap)

    def _apply_command(
        self, command: Command, pc: Optional[int]
    ) -> List[Dict[str, object]]:
        events: List[Dict[str, object]] = []
        for chip_idx in _chip_indices(command.cs):
            chip = self._chips[chip_idx]
            if command.instr is not None:
                column_snapshot = chip.state.y_address
                chip.write_instruction(command.instr, command.data)
                events.append(
                    {
                        "type": "instruction",
                        "chip": chip_idx,
                        "instruction": command.instr.name,
                        "instruction_name": command.instr.name.lower(),
                        "instruction_code": command.instr.value,
                        "data": command.data,
                        "page": chip.state.page,
                        "column": column_snapshot,
                        "pc": pc,
                    }
                )
            else:
                column_raw = chip.state.y_address & 0xFF
                page_before = chip.state.page
                chip.write_data(command.data, pc_source=pc)
                column_written = column_raw % chip.LCD_WIDTH_PIXELS
                next_column = chip.state.y_address & 0xFF
                events.append(
                    {
                        "type": "data",
                        "chip": chip_idx,
                        "page": page_before,
                        "column": column_written,
                        "column_raw": column_raw,
                        "column_next": next_column,
                        "data": command.data,
                        "pc": pc,
                    }
                )
        self._dispatch(events)
        return events


def replay_operations(
    operations: Iterable[LCDOperation],
) -> Tuple[LCDSnapshot, List[Dict[str, object]]]:
    """Convenience wrapper for test code."""
    pipeline = LCDPipeline()
    emitted: List[Dict[str, object]] = []

    def _collect(event: Dict[str, object], _snapshot: LCDSnapshot) -> None:
        emitted.append(event)

    pipeline.subscribe(_collect)
    pipeline.replay(operations)
    return pipeline.snapshot, emitted
