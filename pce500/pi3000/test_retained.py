"""Corruption, model identity, raw RTC and pre-execution restore contracts."""

import hashlib

import pytest

from pce500.pi3000.hardware import Hardware
from pce500.pi3000.retained import IMAGE_SIZE, retained_state, restore_retained_state


def fixture():
    external = bytearray(0x100000)
    external[0xE0000:] = bytes([0xFF]) * 0x20000
    external[0xF0000:0xF0008] = bytes(8)
    external[0xFFFFD:] = bytes([0, 0, 15])
    return external, Hardware({})


def test_round_trip_keeps_raw_rtc_and_fresh_devices():
    external, source = fixture()
    source.ram[0], source.ram[-1] = 0x12, 0x34
    source.scratch[0], source.scratch[-1] = 0x56, 0x78
    source.rtc.write(12, 0x80)
    source.rtc.write(13, 0)
    source.rtc.write(10, 4)
    source.gate[0x12], source.gpio[0] = 0x40, 0x91
    image = retained_state(external, source)
    assert len(image) == IMAGE_SIZE and image[:8] == b"PIRAM01\0"
    assert hashlib.sha256(image).hexdigest() == (
        "7c2d76d7983487b93dfd076b2410a0e5561780f6eed09c688e23e57b25625118"
    )
    assert source.gate[0x12] == 0x40 and source.cycle == 0
    _, target = fixture()
    restore_retained_state(external, target, image)
    assert retained_state(external, target) == image
    assert (target.ram[0], target.ram[-1]) == (0x12, 0x34)
    assert (target.scratch[0], target.scratch[-1]) == (0x56, 0x78)
    assert target.rtc.backing[13] == 0 and target.rtc.read(13) == 0x80
    assert target.gate[0x12] == 0 and target.gpio[0] == 0
    assert target.selector == 0xFD and target.tablet.data_reads == 0


def test_corruption_and_cross_rom_bank_identity_rejection_is_atomic():
    external, source = fixture()
    image = retained_state(external, source)
    _, target = fixture()
    target.ram[7] = 0x93
    before = retained_state(external, target)
    for offset in (0, 8, 40, 72, IMAGE_SIZE - 1):
        bad = bytearray(image)
        bad[offset] ^= 1
        with pytest.raises(ValueError):
            restore_retained_state(external, target, bad)
        assert retained_state(external, target) == before
    for bad in (image[:-1], image + b"\0"):
        with pytest.raises(ValueError, match="version/length"):
            restore_retained_state(external, target, bad)
        assert retained_state(external, target) == before
    external[0xF0000] = 1
    with pytest.raises(ValueError, match="identity"):
        restore_retained_state(external, target, image)
    external[0xF0000] = 0
    target.banks[3] = bytes([0xA5]) * 0x20000
    bank_before = retained_state(external, target)
    with pytest.raises(ValueError, match="identity"):
        restore_retained_state(external, target, image)
    assert retained_state(external, target) == bank_before


@pytest.mark.parametrize("instructions,timing_units", [(1, 0), (0, 1)])
def test_post_execution_restore_rejection(instructions, timing_units):
    external, target = fixture()
    image = retained_state(external, target)
    with pytest.raises(ValueError, match="before the first CPU boundary"):
        restore_retained_state(
            external,
            target,
            image,
            instructions=instructions,
            timing_units=timing_units,
        )
    assert retained_state(external, target) == image
