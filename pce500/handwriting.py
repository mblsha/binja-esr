"""Portable Zinnia-compatible features/inference; host supplies model data.

Feature extraction adapted from Taku Kudo's BSD-licensed Zinnia (2008),
commit 581faa8f6f15e4a7b21964be3a5ec36265c80e5b. See third_party/zinnia/COPYING.
"""

import math
import struct

HYBRID_MAGIC = b"PVHW02\0\0"
SHAPE_POINTS = 32


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def features(strokes, width, height):
    if (
        not math.isfinite(width)
        or not math.isfinite(height)
        or width <= 0
        or height <= 0
        or not strokes
        or len(strokes) > 64
        or any(
            not s or len(s) > 16384 or any(not math.isfinite(v) for p in s for v in p)
            for s in strokes
        )
    ):
        raise ValueError("invalid handwriting geometry/strokes")
    out = [(0, 1.0)]

    def basic(offset, a, b):
        dx, dy = f32(b[0] - a[0]), f32(b[1] - a[1])
        vals = [
            10 * math.sqrt(f32(f32(dx * dx) + f32(dy * dy))),
            math.atan2(dy, dx),
            10 * (a[0] - 0.5),
            10 * (a[1] - 0.5),
            10 * (b[0] - 0.5),
            10 * (b[1] - 0.5),
            math.atan2(a[1] - 0.5, a[0] - 0.5),
            math.atan2(b[1] - 0.5, b[0] - 0.5),
            10 * math.hypot(a[0] - 0.5, a[1] - 0.5),
            10 * math.hypot(b[0] - 0.5, b[1] - 0.5),
            5 * dx,
            5 * dy,
        ]
        out.extend((offset + i + 1, f32(v)) for i, v in enumerate(vals))

    def vertex(points, first, last, node, sid):
        if node > 50:
            return
        basic(sid * 1000 + node * 20, points[first], points[last])
        if first == last:
            return
        a, b = (
            f32(points[last][0] - points[first][0]),
            f32(points[last][1] - points[first][1]),
        )
        c = f32(
            f32(points[last][1] * points[first][0])
            - f32(points[last][0] * points[first][1])
        )
        distances = [
            abs(f32(f32(a * p[1]) - f32(b * p[0]) + c)) for p in points[first:last]
        ]
        distance = max(distances)
        best = first + distances.index(distance)
        denominator = f32(f32(a * a) + f32(b * b))
        if (
            denominator
            and f32(distance * distance) / denominator > 0.001
            and first < best < last
        ):
            vertex(points, first, best, 2 * node + 1, sid)
            vertex(points, best, last, 2 * node + 2, sid)

    previous = None
    for sid, stroke in enumerate(strokes):
        points = [[f32(x / width), f32(y / height)] for x, y in stroke]
        vertex(points, 0, len(points) - 1, 0, sid)
        if previous is not None:
            basic(100000 + sid * 1000, previous, points[0])
        previous = points[-1]
    out += [(2000000, float(len(strokes))), (2000000 + len(strokes), 10.0)]
    if any(not math.isfinite(v) for _, v in out):
        raise ValueError("overflowed handwriting features")
    return sorted(out, key=lambda v: v[0])


class Model:
    def __init__(self, data):
        self.hybrid = None
        if data.startswith(HYBRID_MAGIC):
            self._load_hybrid(data)
            return
        at = 0

        def word():
            nonlocal at
            if at + 4 > len(data):
                raise ValueError("truncated handwriting model")
            value = struct.unpack_from("<I", data, at)[0]
            at += 4
            return value

        if word() ^ 0x0EF71821 != len(data) or word() != 1:
            raise ValueError("handwriting model identity/version mismatch")
        count = word()
        if not 0 < count <= 65536:
            raise ValueError("invalid handwriting class count")
        self.classes = []
        for _ in range(count):
            label = data[at : at + 16]
            at += 16
            if len(label) != 16 or b"\0" not in label:
                raise ValueError("unterminated model label")
            label = label.split(b"\0")[0].decode("utf-8")
            bias = struct.unpack("<f", struct.pack("<I", word()))[0]
            if not label or not math.isfinite(bias):
                raise ValueError("invalid model label/bias")
            weights, previous = [], None
            while True:
                index = word()
                value = struct.unpack("<f", struct.pack("<I", word()))[0]
                if index == 0xFFFFFFFF:
                    break
                if (
                    index > 2100000
                    or not math.isfinite(value)
                    or (previous is not None and index <= previous)
                ):
                    raise ValueError("invalid model feature")
                previous = index
                weights.append((index, value))
            self.classes.append((label, bias, weights))
        if at != len(data):
            raise ValueError("trailing handwriting model bytes")

    def _load_hybrid(self, data):
        if len(data) < 24:
            raise ValueError("truncated hybrid model")
        base_len, count, points, ratio = struct.unpack_from("<IIIf", data, 8)
        if not 0 < count <= 65536 or points != SHAPE_POINTS or not 0 < ratio < 1:
            raise ValueError("invalid hybrid model configuration")
        at = 24
        base = data[at : at + base_len]
        if len(base) != base_len or base.startswith(HYBRID_MAGIC):
            raise ValueError("invalid hybrid base model")
        self.classes = Model(base).classes
        at += base_len
        templates = []
        for _ in range(count):
            raw = data[at : at + 16]
            at += 16
            if len(raw) != 16 or b"\0" not in raw:
                raise ValueError("invalid hybrid label")
            label = raw.split(b"\0")[0].decode("utf-8")
            if len(label) != 1 or at + 4 > len(data):
                raise ValueError("invalid hybrid label/strokes")
            strokes = struct.unpack_from("<I", data, at)[0]
            at += 4
            size = strokes * SHAPE_POINTS * 8
            if not 1 <= strokes <= 64 or at + size > len(data):
                raise ValueError("invalid hybrid stroke count/points")
            values = struct.unpack_from(f"<{strokes * SHAPE_POINTS * 2}f", data, at)
            if any(not math.isfinite(v) or abs(v) > 1 for v in values):
                raise ValueError("invalid hybrid shape point")
            at += size
            templates.append((label, strokes, list(zip(values[::2], values[1::2]))))
        if at != len(data):
            raise ValueError("trailing hybrid model bytes")
        self.hybrid = (ratio, templates)

    def class_count(self):
        labels = {label for label, _, _ in self.classes}
        return len(self.classes) + (
            len(
                {
                    label
                    for label, _, _ in self.hybrid[1]
                    if label.isascii() and label.isalnum()
                }
                - labels
            )
            if self.hybrid
            else 0
        )

    def classify(self, strokes, width, height, limit=10):
        values = features(strokes, width, height)
        if self.hybrid:
            ratio, templates = self.hybrid
            query = shape(strokes, width, height)
            latin, japanese = [], math.inf
            for label, count, points in templates:
                if count != len(strokes):
                    continue
                distance = sum(
                    (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
                    for a, b in zip(query, points)
                ) / (2 * len(query))
                if label.isascii() and label.isalnum():
                    latin.append((label, distance))
                else:
                    japanese = min(japanese, distance)
            latin.sort(key=lambda v: (v[1], v[0]))
            if latin and latin[0][1] < ratio * japanese:
                seen, result = set(), []
                for label, distance in latin:
                    if label not in seen:
                        seen.add(label)
                        result.append((label, f32(-distance)))
                return result[:limit]
        scored = []
        for label, bias, weights in self.classes:
            a, b, total = 0, 0, 0.0
            while a < len(weights) and b < len(values):
                if weights[a][0] == values[b][0]:
                    total += f32(weights[a][1] * values[b][1])
                    a, b = a + 1, b + 1
                elif weights[a][0] < values[b][0]:
                    a += 1
                else:
                    b += 1
            scored.append((label, f32(bias + total)))
        return sorted(scored, key=lambda v: (v[1], v[0]), reverse=True)[:limit]


def shape(strokes, width, height):
    """Aspect-preserving, centered, arc-length samples of ordered strokes.

    Callers validate geometry through features before using this representation.
    """
    normalized = [[(x / width, y / height) for x, y in stroke] for stroke in strokes]
    low = [min(p[j] for s in normalized for p in s) for j in (0, 1)]
    high = [max(p[j] for s in normalized for p in s) for j in (0, 1)]
    span = max(high[0] - low[0], high[1] - low[1], 1e-6)
    center = [(low[j] + high[j]) / 2 for j in (0, 1)]
    result = []
    for stroke in normalized:
        points, distances, length = [], [], 0.0
        for raw in stroke:
            point = [(raw[j] - center[j]) / span for j in (0, 1)]
            if points:
                step = math.hypot(point[0] - points[-1][0], point[1] - points[-1][1])
                if step == 0:
                    continue
                length += step
            points.append(point)
            distances.append(length)
        segment = 0
        for i in range(SHAPE_POINTS):
            target = length * i / (SHAPE_POINTS - 1)
            while segment + 1 < len(distances) and distances[segment + 1] < target:
                segment += 1
            if segment + 1 == len(points):
                point = points[segment]
            else:
                fraction = (target - distances[segment]) / (
                    distances[segment + 1] - distances[segment]
                )
                point = [
                    points[segment][j]
                    + (points[segment + 1][j] - points[segment][j]) * fraction
                    for j in (0, 1)
                ]
            result.append([f32(v) for v in point])
    if any(not math.isfinite(v) for p in result for v in p):
        raise ValueError("overflowed handwriting shape")
    return result
