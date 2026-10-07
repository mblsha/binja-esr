"""Shared framebuffer geometry and row packing, independent of ROM fonts."""

import pytest

from pce500.display.lcd_frame import LcdFrame


def test_row_padding_and_logical_polarity_match_rust_fixture() -> None:
    frame = LcdFrame(
        9, 2, bytes([1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    )
    assert frame.pbm() == b"P4\n9 2\n\x81\x80\x40\x80"


def test_full_oz9600_geometry_preserves_last_row_and_column() -> None:
    pixels = bytearray(336 * 240)
    for x, y in [(0, 0), (239, 31), (319, 239), (335, 239)]:
        pixels[y * 336 + x] = 1
    frame = LcdFrame(336, 240, bytes(pixels))
    pbm = frame.pbm()
    header = b"P4\n336 240\n"
    assert pbm.startswith(header)
    assert len(pbm) == len(header) + 42 * 240
    assert pbm[len(header)] == 0x80
    assert pbm[len(header) + 31 * 42 + 29] == 1
    assert pbm[len(header) + 239 * 42 + 39] == 1
    assert pbm[len(header) + 239 * 42 + 41] == 1
    assert sum(frame.pixels) == 4


def test_frame_owns_mutable_input_and_keeps_pixel_polarity() -> None:
    pixels = bytearray([0, 1])
    frame = LcdFrame(2, 1, pixels)  # type: ignore[arg-type]
    pixels[1] = 0
    assert frame.pixels == b"\x00\x01"
    assert frame.to_rows() == [[0, 1]]


@pytest.mark.parametrize(
    ("cols", "rows", "pixels"),
    [(0, 1, b""), (1, 0, b""), (2, 2, b"\0" * 3), (2**64 - 1, 2, b""), (1, 1, b"\x02")],
)
def test_invalid_shape_and_nonbinary_pixels(
    cols: int, rows: int, pixels: bytes
) -> None:
    with pytest.raises(ValueError):
        LcdFrame(cols, rows, pixels)
