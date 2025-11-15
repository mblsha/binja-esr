from __future__ import annotations

from typing import Iterable

from sc62015.decoding import decode_map
from sc62015.decoding.reader import LayoutEntry, StreamCtx


def _capture_layout(
    opcode: int, data: Iterable[int] | None = None
) -> tuple[LayoutEntry, ...]:
    payload = bytes(data or [0x00] * 16)
    ctx = StreamCtx(pc=0, data=payload, base_len=1, record_layout=True)
    decode_map.decode_with_pre_variants(opcode, ctx)
    return ctx.snapshot_layout()


def test_loop_arith_mem_layout_has_offsets() -> None:
    layout = _capture_layout(0x54, [0x00, 0x00, 0x00])
    assert len(layout) == 2
    first, second = layout
    assert first.key == "dst"
    assert first.kind == "imm8"
    assert first.meta["offset"] == 0
    assert first.meta["length_bytes"] == 1
    assert second.key == "src"
    assert second.meta["offset"] == 1
    assert second.meta["length_bytes"] == 1


def test_call_layout_records_addr16_span() -> None:
    layout = _capture_layout(0x04, [0xAA, 0xBB])
    assert len(layout) == 1
    (entry,) = layout
    assert entry.kind == "addr16_page"
    assert entry.meta["offset"] == 0
    assert entry.meta["length_bytes"] == 2
    assert entry.meta["order"] == "mn"


def test_callf_layout_records_addr24_span() -> None:
    layout = _capture_layout(0x05, [0x01, 0x02, 0x03])
    assert len(layout) == 1
    (entry,) = layout
    assert entry.kind == "addr24"
    assert entry.meta["offset"] == 0
    assert entry.meta["length_bytes"] == 3
    assert entry.meta["order"] == "lmn"


def test_ext_reg_ptr_layout_captures_displacement() -> None:
    layout = _capture_layout(0x90, [0x84, 0x02])
    assert len(layout) == 1
    (entry,) = layout
    assert entry.kind == "ext_reg_ptr"
    assert entry.meta["offset"] == 0
    assert entry.meta["length_bytes"] == 2
    assert entry.meta["width_bytes"] >= 1


def test_imem_ptr_layout_handles_variable_length() -> None:
    layout = _capture_layout(0x98, [0x80, 0x10, 0x04])
    assert len(layout) == 1
    (entry,) = layout
    assert entry.kind == "imem_ptr"
    assert entry.meta["offset"] == 0
    assert entry.meta["length_bytes"] == 3
