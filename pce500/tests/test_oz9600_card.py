"""Logical cartridge hypotheses and independent media validation reference."""

import pytest

from pce500.oz9600.boundary import ExecutionState
from pce500.oz9600.card import Oz707Card, ROM_SIZE, SRAM_SIZE
from pce500.oz9600.hardware import Hardware


def fixture_card():
    # Component fixture deliberately bypasses capture identity; ordinary media
    # construction rejects these bytes. This is not an application boot proof.
    card = Oz707Card.__new__(Oz707Card)
    card.rom = bytes(range(256)) * (ROM_SIZE // 256)
    card.sram = bytearray([0x34] * SRAM_SIZE)
    return card


def test_canonical_sram_read_mirror_and_readonly_rom_share_one_backing():
    hw = Hardware()
    assert hw.read(0x40000) == 255 and hw.read(0x30000) is None
    hw.card = fixture_card()
    assert hw.read(0x40123) == hw.read(0x60123) == 0x23
    hw.write(0x38013, 0xA5)
    assert hw.read(0x30013) == hw.read(0x38013) == 0xA5
    hw.write(0x30013, 0x77)
    hw.write(0x40123, 0x77)
    assert hw.read(0x30013) == 0xA5 and hw.read(0x40123) == 0x23
    assert hw.card.read(0x80000) is None


def test_paired_physical_native_upper_samples_override_card_rom_view():
    hw = Hardware()
    hw.card = fixture_card()
    for mode in (0x20, 0x30):
        hw.gate[0x20], hw.selector, hw.gate[0x22], hw.gate[0x23], hw.gpio[5] = (
            mode,
            7,
            0xFF,
            1,
            0,
        )
        for address, expected in (
            (0x42000, 0xE8),
            (0x4A000, 0xE8),
            (0x52000, 0xEA),
            (0x5A000, 0xEA),
            (0x62000, 0xEC),
            (0x6A000, 0xEC),
            (0x72000, 0xEE),
            (0x7A000, 0xEE),
        ):
            assert hw.read(address) == expected
    hw.gate[0x20], hw.selector = 0x20, 0xFE
    for address, expected in (
        (0x42000, 0xE8),
        (0x4A000, 0xE8),
        (0x52000, 0xEA),
        (0x5A000, 0xEA),
        (0x62000, 0xEC),
        (0x6A000, 0xEC),
        (0x72000, 0xEE),
        (0x7A000, 0xEE),
    ):
        assert hw.read(address) == expected
    hw.gate[0x20] = 0x30
    assert hw.read(0x42000) == 0x00  # 30/FE was not physically sampled.
    hw.gate[0x20], hw.gate[0x23], hw.gpio[5] = 0x35, 0, 2
    assert hw.read(0x42000) == 0x00  # Synthetic card ROM offset 0x2000.


def test_bad_media_identity_and_sizes_are_rejected():
    with pytest.raises(ValueError, match="128 KiB"):
        Oz707Card(b"", b"")
    with pytest.raises(ValueError, match="identity"):
        Oz707Card(bytes(ROM_SIZE), bytes(SRAM_SIZE))


def test_boundary_preserves_other_ssr_bits_and_reasserts_card_input_after_reset():
    class Memory:
        ssr = 0xA4

        def read_internal_byte_silent(self, offset):
            assert offset == 0xFF
            return self.ssr

        def write_internal_byte(self, offset, value):
            assert offset == 0xFF
            self.ssr = value

    hw, memory, state = Hardware(), Memory(), ExecutionState()
    state.before_boundary(hw, 0, memory=memory)
    assert memory.ssr == 0xA4
    hw.card = fixture_card()
    state.before_boundary(hw, 1, memory=memory)
    assert memory.ssr == 0xA6
    memory.ssr = 0
    state.before_boundary(hw, 2, memory=memory)
    assert memory.ssr == 2
