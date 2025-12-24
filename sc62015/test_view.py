from __future__ import annotations

from binja_test_mocks import binja_api  # noqa: F401  # pyright: ignore

from .pysc62015.constants import INTERNAL_MEMORY_LENGTH, INTERNAL_MEMORY_START
from .view import SC62015FullView


def test_full_view_internal_ram_is_virtual() -> None:
    internal_ram = next(
        seg for seg in SC62015FullView.SEGMENTS if seg.name == "Internal RAM"
    )
    assert internal_ram.start == INTERNAL_MEMORY_START
    assert internal_ram.length == INTERNAL_MEMORY_LENGTH
    assert internal_ram.file_offset is None
