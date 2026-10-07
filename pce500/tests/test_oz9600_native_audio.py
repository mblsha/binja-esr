"""Host sound backlog, discontinuity and clearing regression controls."""

import pytest

from pce500.oz9600.native_audio import CAPACITY, Queue


def test_backlog_and_discontinuity_do_not_replay_old_tone():
    queue = Queue(48_000)
    queue.push(0, [8192] * (CAPACITY + 17))
    assert len(queue.samples) == CAPACITY
    assert queue.dropped == 17
    assert queue.next() > 0.2
    queue.push(100_000, [0] * 16)
    assert queue.next() == 0
    assert len(queue.samples) == 14
    queue.clear()
    assert queue.next() == 0
    assert not queue.samples


@pytest.mark.parametrize("rate", [24_000, 44_100, 48_000, 96_000])
def test_dc_decay_rate_conversion_and_empty_queue_emit_silence(rate):
    queue = Queue(rate)
    queue.push(0, [8192] * CAPACITY)
    output = [queue.next() for _ in range(rate // 10)]
    assert output[0] > 0.2
    assert abs(output[-1]) < 0.0001
    assert queue.next() == 0
    assert not queue.samples
    assert all(-1 <= value <= 1 for value in output)


def test_split_source_chunks_preserve_phase_and_filter_history():
    samples = [8192 if n % 37 < 19 else 0 for n in range(100)]
    whole, split = Queue(44_100), Queue(44_100)
    whole.push(0, samples)
    split.push(0, samples[:50])
    a = [whole.next() for _ in range(25)]
    b = [split.next() for _ in range(25)]
    split.push(50, samples[50:])
    a += [whole.next() for _ in range(100)]
    b += [split.next() for _ in range(100)]
    assert a == b
    assert split.next() == 0


def test_mismatched_source_rate_clears_previous_tone():
    queue = Queue(48_000)
    queue.push(0, [8192] * 16)
    assert queue.next() > 0.2
    queue.push(16, [8192] * 16, sample_rate=44_100)
    assert queue.next() == 0
    assert not queue.samples
