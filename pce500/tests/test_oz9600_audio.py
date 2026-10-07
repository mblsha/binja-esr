"""Independent rational integration and bounded transport checks."""

import hashlib
import json
import random
import struct
from pathlib import Path

from pce500.oz9600.audio import AudioCapture, QUEUE_CAPACITY, SAMPLE_RATE, TIMEBASE_HZ


def test_area_integration_has_no_per_instruction_rounding_or_drain_phase_reset():
    a = AudioCapture(True)
    a.advance(32, 0x90)  # ISE is independent; exactly 1.5 samples high.
    assert a.take()["samples"] == [8192]
    a.advance(32, 0x80)
    assert a.take()["samples"] == [4096, 0]
    assert a.status()["total_samples"] == 3
    a.advance(TIMEBASE_HZ - 64, 0)
    assert len(a.take()["samples"]) == SAMPLE_RATE - 3


def test_integer_oracle_and_arbitrary_slices_preserve_sample_areas():
    rng = random.Random(9600)
    waveform = [(rng.randrange(1, 300), rng.randrange(2)) for _ in range(200)]
    whole, sliced = AudioCapture(True), AudioCapture(True)
    levels = []
    collected = []
    for units, high in waveform:
        whole.advance(units, high << 4)
        levels.extend([8192 * high] * units)
        while units:
            count = min(units, rng.randrange(1, 50))
            sliced.advance(count, high << 4)
            collected.extend(sliced.take()["samples"])
            units -= count
    # Scale reduced by gcd(48000,1024000)=16000: three ticks per CPU
    # unit, 64 ticks per sample. This brute oracle shares no phase loop.
    ticks = [level for level in levels for _ in range(3)]
    expected = [sum(ticks[i : i + 64]) // 64 for i in range(0, len(ticks) - 63, 64)]
    assert whole.take()["samples"] == collected == expected


def test_backlog_is_bounded_and_loss_reported_while_phase_remains_continuous():
    a = AudioCapture(True)
    a.advance(TIMEBASE_HZ * 3, 0x10)
    chunk = a.take()
    assert len(chunk["samples"]) == QUEUE_CAPACITY
    assert chunk["first_sample"] == chunk["dropped_samples"] == SAMPLE_RATE
    assert chunk["total_samples"] == SAMPLE_RATE * 3
    assert set(chunk["samples"]) == {8192}
    assert a.take()["first_sample"] == SAMPLE_RATE * 3


def test_unsupported_modes_and_off_are_silent_and_capture_does_not_run_by_default():
    a = AudioCapture()
    a.advance(TIMEBASE_HZ, 0x10)
    assert a.status()["elapsed_units"] == 0 and not a.take()["samples"]
    a.set_enabled(True)
    for mode in (4, 6, 7):
        a.advance(64, mode << 4)
    a.advance(64, 0x10, off=True)
    assert a.take()["samples"] == [0] * 12
    assert a.status()["unsupported_units"] == 128
    assert a.status()["provisional_units"] == 192
    a.advance(64, 0x10)
    a.reset()
    assert a.enabled and a.status()["total_samples"] == 0
    a.set_enabled(False)
    assert not a.enabled and not a.take()["samples"]


def test_clocked_modes_count_every_edge_and_preserve_phase_across_arbitrary_slices():
    for mode, half in ((2, 256), (3, 128)):
        whole, sliced = AudioCapture(True), AudioCapture(True)
        units = half * 7 + 19
        whole.advance(units, mode << 4)
        pcm = []
        rng = random.Random(mode)
        left = units
        while left:
            part = min(left, rng.randrange(1, half * 2))
            sliced.advance(part, (mode << 4) | 0x8F)
            pcm.extend(sliced.take()["samples"])
            left -= part
        assert pcm == whole.take()["samples"]
        run = half * 3 // 64
        assert pcm[: run * 7] == ([8192] * run + [0] * run) * 3 + [8192] * run
        whole.advance(64, 0, off=True)
        whole.take()
        whole.advance(half, mode << 4)
        assert any(whole.take()["samples"])


def test_provisional_constant_and_ci_modes_use_explicit_input_without_ssr_aliases():
    for mode, ci, high in (
        (4, False, False),
        (5, False, True),
        (6, False, False),
        (6, True, True),
        (7, True, True),
    ):
        a = AudioCapture(True)
        a.advance(64, (mode << 4) | 128, ci=ci)
        assert a.take()["samples"] == [8192 if high else 0] * 3
        assert a.provisional_units == 64


def audio_corpus():
    rng = random.Random(7079600)
    ops = [["advance", 123, 0x10, False, False], ["enable", True]]
    ops += [
        ["advance", 32, 0x90, False, False],
        ["take"],
        ["advance", 32, 0x80, False, False],
    ]
    for _ in range(120):
        ops += [
            [
                "advance",
                rng.randrange(300),
                rng.randrange(256),
                bool(rng.randrange(8) == 0),
                bool(rng.randrange(2)),
            ]
        ]
        if rng.randrange(4) == 0:
            ops += [["take"]]
    ops += [
        ["advance", TIMEBASE_HZ * 3, 0x10, False, False],
        ["take"],
        ["reset"],
        ["take"],
    ]
    ops += [["advance", 64, 0x10, False, False], ["enable", False], ["take"]]
    a = AudioCapture()
    replies = []
    for op in ops:
        result = None
        if op[0] == "advance":
            a.advance(*op[1:])
        elif op[0] == "enable":
            a.set_enabled(op[1])
        elif op[0] == "reset":
            a.reset()
        elif op[0] == "take":
            result = a.take()
            samples = result.pop("samples")
            result["sample_count"] = len(samples)
            result["pcm_sha256"] = hashlib.sha256(
                struct.pack(f"<{len(samples)}h", *samples)
            ).hexdigest()
        replies.append(dict(chunk=result, status=a.status()))
    return dict(operations=ops, replies=replies)


def test_rust_fixture_is_the_current_independent_python_corpus():
    path = Path(__file__).parents[2] / "sc62015/core/data/oz9600_audio_reference.json"
    assert json.loads(path.read_text()) == audio_corpus()
