"""Data-format and geometry checks; these tests do not execute a CPU emulator."""

import math
import struct

import pytest

from pce500.handwriting import HYBRID_MAGIC, Model, shape


def base_model():
    payload = "い".encode().ljust(16, b"\0") + struct.pack("<fIf", 2.5, 0xFFFFFFFF, 0.0)
    return struct.pack("<III", (12 + len(payload)) ^ 0x0EF71821, 1, 1) + payload


def hybrid_model(templates):
    base = base_model()
    payload = bytearray(
        HYBRID_MAGIC + struct.pack("<IIIf", len(base), len(templates), 32, 0.75) + base
    )
    for label, horizontal in templates:
        points = [
            (i / 31 - 0.5, 0.0) if horizontal else (0.0, i / 31 - 0.5)
            for i in range(32)
        ]
        payload.extend(label.encode().ljust(16, b"\0") + struct.pack("<I", 1))
        payload.extend(struct.pack("<64f", *(v for p in points for v in p)))
    return payload


def test_clear_latin_route_and_ambiguous_or_missing_stroke_fallback():
    model = Model(hybrid_model([("A", True), ("A", True), ("い", False)]))
    assert model.class_count() == 2
    assert Model(hybrid_model([("A", True), ("字", False)])).class_count() == 2
    result = model.classify([[[10, 20], [30, 20]]], 100, 100)
    assert len(result) == 1 and result[0][0] == "A" and abs(result[0][1]) < 1e-12
    base = Model(base_model())
    for strokes in ([[[20, 10], [20, 30]]], [[[0, 0]], [[1, 1]]]):
        assert model.classify(strokes, 100, 100) == base.classify(strokes, 100, 100)
    tie = Model(hybrid_model([("A", True), ("い", True)]))
    assert tie.classify([[[0, 0], [100, 0]]], 100, 100) == [("い", 2.5)]


def test_arc_length_shape_with_duplicate_and_degenerate_points():
    points = shape([[[10, 20], [10, 20], [30, 20]]], 100, 100)
    assert len(points) == 32
    for i, (x, y) in enumerate(points):
        assert x == pytest.approx(i / 31 - 0.5, abs=1e-7) and y == 0
    assert shape([[[10, 20]]], 100, 100) == [[0, 0]] * 32


def test_bad_hybrid_configuration_truncation_and_nonfinite_points():
    data = hybrid_model([("A", True), ("い", False)])
    for length in (8, 23, 24, 30, len(data) - 1):
        with pytest.raises(ValueError):
            Model(data[:length])
    for offset, value in (
        (12, 0),
        (16, 31),
        (20, struct.unpack("<I", struct.pack("<f", math.nan))[0]),
        (24 + len(base_model()) + 16, 65),
    ):
        bad = data.copy()
        struct.pack_into("<I", bad, offset, value)
        with pytest.raises(ValueError):
            Model(bad)
    struct.pack_into("<f", data, 24 + len(base_model()) + 20, math.inf)
    with pytest.raises(ValueError):
        Model(data)
    with pytest.raises(ValueError):
        Model(hybrid_model([("A", True)]) + b"\0")
