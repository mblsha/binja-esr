"""Pure component profile contracts; no alternate CPU execution."""

from pce500.digital_audio import AudioCapture, TIMEBASE_HZ


def test_pvf1_clocked_key_tone_and_explicit_unwired_ci_policy():
    capture = AudioCapture(software_only=True, allow_clocked=True)
    capture.set_enabled(True)
    capture.advance(TIMEBASE_HZ // 100, 0x30)  # SCR.BZ1 | BZ0.
    samples = capture.take()["samples"]
    assert len(samples) == 480
    assert max(samples) > 0 and min(samples) == 0
    assert capture.status()["unsupported_units"] == 0
    capture.reset()
    capture.advance(TIMEBASE_HZ // 100, 0x70, ci=True)
    assert capture.take()["samples"] == [0] * 480
    assert capture.status()["unsupported_units"] == TIMEBASE_HZ // 100


def test_software_only_policy_is_preserved_when_capture_is_reenabled():
    capture = AudioCapture(software_only=True)
    capture.set_enabled(True)
    capture.advance(TIMEBASE_HZ // 100, 0x90)  # ISE | BZ0.
    assert capture.take()["samples"] == [8192] * 480
    capture.set_enabled(False)
    capture.set_enabled(True)
    capture.advance(TIMEBASE_HZ // 100, 0x30)
    assert capture.take()["samples"] == [0] * 480
