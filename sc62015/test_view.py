from __future__ import annotations

from binja_test_mocks import binja_api  # noqa: F401  # pyright: ignore
from binaryninja.enums import SectionSemantics, SegmentFlag

from .pysc62015.constants import INTERNAL_MEMORY_LENGTH, INTERNAL_MEMORY_START
from .view import IOCS_TYPES_C, SC62015FullView, SC62015Iq7000FullView


def test_full_view_internal_ram_is_virtual() -> None:
    internal_ram = next(
        seg for seg in SC62015FullView.SEGMENTS if seg.name == "Internal RAM"
    )
    assert internal_ram.start == INTERNAL_MEMORY_START
    assert internal_ram.length == INTERNAL_MEMORY_LENGTH
    assert internal_ram.file_offset is None


def test_full_view_maps_complete_external_space_without_gaps() -> None:
    file_backed = sorted(
        (seg for seg in SC62015FullView.SEGMENTS if seg.file_offset is not None),
        key=lambda seg: seg.start,
    )

    assert file_backed[0].start == 0
    assert file_backed[-1].start + file_backed[-1].length == 0x100000
    assert len({seg.name for seg in file_backed}) == len(file_backed)
    for current, following in zip(file_backed, file_backed[1:]):
        assert current.start + current.length == following.start
    assert all(seg.file_offset == seg.start for seg in file_backed)


def test_pc_e500_full_view_separates_bundled_disk_data_from_code() -> None:
    segments = {seg.name: seg for seg in SC62015FullView.SEGMENTS}

    disk = segments["Bundled Program Disk"]
    headers = segments["IOCS Header Data"]
    # These byte ranges are ROM tables/assets, not executable code.  Lock the
    # disassembly-grounded boundaries while allowing executable spans between
    # them to be split further as more functions are recovered.
    data_ranges = {
        "Bundled Program Disk": (0xC0000, 0xDF820),
        "IOCS Header Data": (0xDF820, 0xDF8B8),
        "IOCS main dispatch table": (0xE0037, 0xE0043),
        "SYSTM dispatch table": (0xE01A3, 0xE01B1),
        "FCS dispatch table": (0xE06DC, 0xE0700),
        "Packed format-record offset table": (0xE13D1, 0xE13E6),
        "Packed format-operation records": (0xE13E6, 0xE1457),
        "Push BREAK to Stop text": (0xE1463, 0xE1476),
        "Near-return dispatch table": (0xE1A7D, 0xE1AA1),
        "Powers-of-ten table": (0xE1DAA, 0xE1DB4),
        "Key/input translation lookup": (0xE246E, 0xE24D0),
        "E:F:G: dispatch table": (0xE491E, 0xE493E),
        "Character/input computed dispatch table (32 words)": (0xE52D1, 0xE5311),
        "Character/input computed dispatch table (212 words)": (0xE5311, 0xE54B9),
        "Q15 cosine 0-to-45-degree table": (0xE73BC, 0xE7418),
        "Q15 sine 0-to-45-degree table": (0xE7418, 0xE7474),
        "BASIC/FCS error-message table": (0xE968B, 0xE99B1),
        "CAS special-file dispatch table": (0xE99D6, 0xE99F6),
        "CAS device-command dispatch table": (0xE99F6, 0xE9A04),
        "Printer standard-character dispatch table": (0xEA7C4, 0xEA7D2),
        "Printer device-command dispatch table": (0xEA7D2, 0xEA7DC),
        "COM special-file dispatch table": (0xEAD5F, 0xEAD7F),
        "COM device-command dispatch table": (0xEAD7F, 0xEAD99),
        "Function-driver numeric dispatch table": (0xEF33D, 0xEF373),
        "Function-driver string dispatch table": (0xEF373, 0xEF387),
        "Function-driver conversion dispatch table": (0xEF387, 0xEF38B),
        "Memory-card special-file dispatch table": (0xF002F, 0xF004F),
        "Memory-card management dispatch table": (0xF004F, 0xF0063),
        "Keyboard standard-character dispatch table": (0xF1693, 0xF16A3),
        "Keyboard device-command dispatch table": (0xF16A3, 0xF16B3),
        "PC-E500 font and bitmap atlas": (0xF2155, 0xF28F9),
        "STDO dispatch tables": (0xF2928, 0xF296A),
        "BASIC token-to-keyword offset table": (0xF5581, 0xF5781),
        "BASIC token descriptor table": (0xF5781, 0xF5A81),
        "BASIC keyword dictionary": (0xF5A8C, 0xF5E75),
        "Public-vector filler": (0xFFFE0, 0xFFFE4),
        "ROM metadata and CPU vectors": (0xFFFEC, 0x100000),
    }
    assert (disk.start, disk.start + disk.length) == (0xC0000, 0xDF820)
    assert (headers.start, headers.start + headers.length) == (0xDF820, 0xDF8B8)

    assert {
        name: (segments[name].start, segments[name].start + segments[name].length)
        for name in data_ranges
    } == data_ranges

    for name in data_ranges:
        data = segments[name]
        assert data.flags & SegmentFlag.SegmentReadable
        assert not data.flags & SegmentFlag.SegmentExecutable
        assert not data.flags & SegmentFlag.SegmentWritable
        assert data.semantics != SectionSemantics.ReadOnlyCodeSectionSemantics

    for name, entry in segments.items():
        if entry.file_offset is None or entry.start < 0xC0000 or name in data_ranges:
            continue
        entry = segments[name]
        assert entry.flags & SegmentFlag.SegmentReadable
        assert entry.flags & SegmentFlag.SegmentExecutable
        assert not entry.flags & SegmentFlag.SegmentWritable
        assert entry.semantics == SectionSemantics.ReadOnlyCodeSectionSemantics


def test_pc_e500_full_view_non_rom_external_segments_are_writable_data() -> None:
    for seg in SC62015FullView.SEGMENTS:
        if seg.file_offset is None or seg.start >= 0xC0000:
            continue

        assert seg.flags & SegmentFlag.SegmentReadable
        assert seg.flags & SegmentFlag.SegmentWritable
        assert not seg.flags & SegmentFlag.SegmentExecutable
        assert seg.semantics == SectionSemantics.ReadWriteDataSectionSemantics


def test_iq7000_full_view_maps_complete_external_space_without_gaps() -> None:
    file_backed = sorted(
        (seg for seg in SC62015Iq7000FullView.SEGMENTS if seg.file_offset is not None),
        key=lambda seg: seg.start,
    )

    assert [seg.name for seg in file_backed] == [
        "SH26",
        "Lower ROM header",
        "Lower ROM before BASIC error strings",
        "BASIC error strings",
        "Lower ROM before S-C12 dispatch tables",
        "S-C12 dispatch tables",
        "Lower ROM before keyboard secondary maps",
        "Keyboard alternate secondary maps",
        "Lower ROM before IOCS headers",
        "S-C12 IOCS header data",
        "Lower ROM after IOCS headers",
        "BASIC keyword letter offsets",
        "BASIC token-to-keyword offset table",
        "BASIC token dispatch table",
        "BASIC keyword table",
        "Lower ROM tail",
        "Lower ROM mirror",
        "CE1",
        "CE0",
        "System ROM mirror",
        "System ROM before app labels",
        "App name strings",
        "App menu wrappers",
        "TEL submenu labels",
        "TEL submenu handler",
        "SCHEDULE submenu labels",
        "SCHEDULE submenu handler",
        "MEMO submenu label",
        "System ROM before IOCS headers",
        "System IOCS header data",
        "IOCS dispatch prelude",
        "IOCS command-family table",
        "System ROM services",
        "IQ font and glyph assets",
        "RTC dispatcher",
        "RTC command table",
        "System ROM before keyboard dispatch",
        "Keyboard command table",
        "System ROM before keyboard default state",
        "Keyboard default-state block",
        "System ROM before keyboard translation tables",
        "Keyboard character-transform tables",
        "Primary keyboard keycode map",
        "Default secondary keyboard map",
        "System ROM services after keyboard tables",
        "CARD SECRET DATA string",
        "RAM-disk dispatcher",
        "RAM-disk command table",
        "System ROM storage services",
        "RAM-disk template",
        "System ROM before WORLD city data",
        "WORLD city database",
        "System ROM communications and services",
        "Built-in demo records",
        "Factory diagnostics and system ROM tail",
        "Public-vector filler",
        "Interrupt-vector seed entry",
        "Public-vector separator",
        "IOCS public entry",
        "ROM metadata and CPU vectors",
    ]
    assert [
        (seg.start, seg.start + seg.length, seg.file_offset) for seg in file_backed
    ] == [
        (0x00000, 0x40000, 0x00000),
        (0x40000, 0x40100, 0x40000),
        (0x40100, 0x4631E, 0x40100),
        (0x4631E, 0x46649, 0x4631E),
        (0x46649, 0x50046, 0x46649),
        (0x50046, 0x5007C, 0x50046),
        (0x5007C, 0x50DB7, 0x5007C),
        (0x50DB7, 0x50DDF, 0x50DB7),
        (0x50DDF, 0x51729, 0x50DDF),
        (0x51729, 0x517C6, 0x51729),
        (0x517C6, 0x58E1C, 0x517C6),
        (0x58E1C, 0x58E50, 0x58E1C),
        (0x58E50, 0x59050, 0x58E50),
        (0x59050, 0x59350, 0x59050),
        (0x59350, 0x596FB, 0x59350),
        (0x596FB, 0x60000, 0x596FB),
        (0x60000, 0x80000, 0x60000),
        (0x80000, 0xA0000, 0x80000),
        (0xA0000, 0xC0000, 0xA0000),
        (0xC0000, 0xE0000, 0xC0000),
        (0xE0000, 0xEEC87, 0xE0000),
        (0xEEC87, 0xEECA3, 0xEEC87),
        (0xEECA3, 0xEECE2, 0xEECA3),
        (0xEECE2, 0xEECF1, 0xEECE2),
        (0xEECF1, 0xEECFF, 0xEECF1),
        (0xEECFF, 0xEED12, 0xEECFF),
        (0xEED12, 0xEED20, 0xEED12),
        (0xEED20, 0xEED25, 0xEED20),
        (0xEED25, 0xF03F5, 0xEED25),
        (0xF03F5, 0xF04A5, 0xF03F5),
        (0xF04A5, 0xF04C4, 0xF04A5),
        (0xF04C4, 0xF0510, 0xF04C4),
        (0xF0510, 0xF1B45, 0xF0510),
        (0xF1B45, 0xF31EF, 0xF1B45),
        (0xF31EF, 0xF320F, 0xF31EF),
        (0xF320F, 0xF322D, 0xF320F),
        (0xF322D, 0xF362A, 0xF322D),
        (0xF362A, 0xF3650, 0xF362A),
        (0xF3650, 0xF3952, 0xF3650),
        (0xF3952, 0xF3969, 0xF3952),
        (0xF3969, 0xF3C19, 0xF3969),
        (0xF3C19, 0xF4019, 0xF3C19),
        (0xF4019, 0xF4081, 0xF4019),
        (0xF4081, 0xF4095, 0xF4081),
        (0xF4095, 0xF7C73, 0xF4095),
        (0xF7C73, 0xF7C83, 0xF7C73),
        (0xF7C83, 0xF7CA9, 0xF7C83),
        (0xF7CA9, 0xF7CD3, 0xF7CA9),
        (0xF7CD3, 0xF82DA, 0xF7CD3),
        (0xF82DA, 0xF8308, 0xF82DA),
        (0xF8308, 0xF9409, 0xF8308),
        (0xF9409, 0xFA5DD, 0xF9409),
        (0xFA5DD, 0xFD4AF, 0xFA5DD),
        (0xFD4AF, 0xFDAA0, 0xFD4AF),
        (0xFDAA0, 0xFFFE0, 0xFDAA0),
        (0xFFFE0, 0xFFFE4, 0xFFFE0),
        (0xFFFE4, 0xFFFE7, 0xFFFE4),
        (0xFFFE7, 0xFFFE8, 0xFFFE7),
        (0xFFFE8, 0xFFFEC, 0xFFFE8),
        (0xFFFEC, 0x100000, 0xFFFEC),
    ]


def test_iq7000_full_view_marks_only_canonical_rom_ranges_executable() -> None:
    segments = {seg.name: seg for seg in SC62015Iq7000FullView.SEGMENTS}

    for name in (
        "Lower ROM before BASIC error strings",
        "Lower ROM before S-C12 dispatch tables",
        "Lower ROM before keyboard secondary maps",
        "Lower ROM before IOCS headers",
        "Lower ROM after IOCS headers",
        "Lower ROM tail",
        "System ROM before app labels",
        "App menu wrappers",
        "TEL submenu handler",
        "SCHEDULE submenu handler",
        "System ROM before IOCS headers",
        "IOCS dispatch prelude",
        "System ROM services",
        "RTC dispatcher",
        "System ROM before keyboard dispatch",
        "System ROM before keyboard default state",
        "System ROM before keyboard translation tables",
        "System ROM services after keyboard tables",
        "RAM-disk dispatcher",
        "System ROM storage services",
        "System ROM before WORLD city data",
        "System ROM communications and services",
        "Factory diagnostics and system ROM tail",
        "Interrupt-vector seed entry",
        "IOCS public entry",
    ):
        rom = segments[name]
        assert rom.flags & SegmentFlag.SegmentReadable
        assert rom.flags & SegmentFlag.SegmentExecutable
        assert not rom.flags & SegmentFlag.SegmentWritable
        assert rom.semantics == SectionSemantics.ReadOnlyCodeSectionSemantics

    for name in (
        "Lower ROM mirror",
        "Lower ROM header",
        "S-C12 IOCS header data",
        "BASIC error strings",
        "S-C12 dispatch tables",
        "Keyboard alternate secondary maps",
        "BASIC keyword letter offsets",
        "BASIC token-to-keyword offset table",
        "BASIC token dispatch table",
        "BASIC keyword table",
        "System ROM mirror",
        "System IOCS header data",
        "App name strings",
        "TEL submenu labels",
        "SCHEDULE submenu labels",
        "MEMO submenu label",
        "IOCS command-family table",
        "IQ font and glyph assets",
        "RTC command table",
        "Keyboard command table",
        "Keyboard default-state block",
        "Keyboard character-transform tables",
        "Primary keyboard keycode map",
        "Default secondary keyboard map",
        "CARD SECRET DATA string",
        "RAM-disk command table",
        "RAM-disk template",
        "WORLD city database",
        "Built-in demo records",
        "Public-vector filler",
        "Public-vector separator",
        "ROM metadata and CPU vectors",
    ):
        mirror = segments[name]
        assert mirror.flags & SegmentFlag.SegmentReadable
        assert not mirror.flags & SegmentFlag.SegmentExecutable
        assert not mirror.flags & SegmentFlag.SegmentWritable
        assert mirror.semantics != SectionSemantics.ReadOnlyCodeSectionSemantics


def test_model_specific_public_entry_points_are_seeded() -> None:
    assert (0xEAF69, "sio_close") in SC62015FullView.EXTRA_ENTRY_POINTS
    for entry in (
        (0xF16B5, "stdi_unsupported_write_error"),
        (0xF20D8, "lcd_power_toggle_then_subtimer"),
        (0xF210C, "subtimer_display_countdown_tick"),
        (0xF28F9, "iocs_stdo_screen_handler"),
        (0xF5A81, "basic_token61_command_stub"),
        (0xF5A84, "basic_token61_function_stub"),
        (0xF5E79, "basic_execute_intermediate_code"),
    ):
        assert entry in SC62015FullView.EXTRA_ENTRY_POINTS
    assert (
        0xFFFD8,
        "legacy_basic_iocs_staging_entry",
    ) in SC62015FullView.EXTRA_ENTRY_POINTS
    assert (
        0xFFFDC,
        "legacy_basic_iocs_entry",
    ) in SC62015FullView.EXTRA_ENTRY_POINTS
    assert (0xFFFE4, "public_fcs_entry") in SC62015FullView.EXTRA_ENTRY_POINTS
    assert (
        0xFFFE4,
        "public_interrupt_vector_seed",
    ) in SC62015Iq7000FullView.EXTRA_ENTRY_POINTS
    assert (
        0x5FFF4,
        "s_c12_card_iocs_entry",
    ) in SC62015Iq7000FullView.EXTRA_ENTRY_POINTS
    assert (
        0xF7C83,
        "iocs_ram_disk_handler",
    ) in SC62015Iq7000FullView.EXTRA_ENTRY_POINTS
    assert (
        0xF5247,
        "irq_onki_default_handler",
    ) in SC62015Iq7000FullView.EXTRA_ENTRY_POINTS
    for entry in (
        (0x50000, "s_c12_storage_handler"),
        (0xE000F, "daily_alarm_service_keyc00a_e000f"),
        (0xEECF1, "app_tel_submenu_handler_eecf1"),
        (0xEED12, "app_schedule_submenu_handler_eed12"),
        (0xEED25, "app_memo_submenu_handler_eed25"),
        (0xF04C0, "iocs_error_stub_a3_sc_f04c0"),
        (0xF3969, "keyboard_clear_buffers_f3969"),
        (0xF8308, "ram_disk_copy_record_to_caller_f8308"),
        (0xFBA38, "pacom_handshake_send_a5_and_read2_fba38"),
        (0xFDAA0, "factory_diagnostic_boot_gate_fdaa0"),
    ):
        assert entry in SC62015Iq7000FullView.EXTRA_ENTRY_POINTS
    seeded_addresses = {
        address for address, _name in SC62015Iq7000FullView.EXTRA_ENTRY_POINTS
    }
    iocs_command_targets = {
        0xF04C0,
        0xF069D,
        0xF07A3,
        0xF08AE,
        0xF0934,
        0xF0940,
        0xF0A62,
        0xF0BEB,
        0xF0C36,
        0xF0CBA,
        0xF0E09,
        0xF0E92,
        0xF0F01,
        0xF0F0E,
        0xF0F2D,
        0xF100E,
        0xF1023,
        0xF11E4,
        0xF129B,
        0xF12B0,
        0xF13C0,
        0xF148C,
        0xF174A,
        0xF1774,
        0xF1868,
        0xF18C5,
        0xF1988,
        0xF19A7,
        0xF1AD3,
    }
    assert iocs_command_targets <= seeded_addresses
    assert not {0xF3AA6, 0xF3BD5, 0xFB585} & seeded_addresses


def test_iq7000_proven_inline_data_ranges_are_exact() -> None:
    segments = {seg.name: seg for seg in SC62015Iq7000FullView.SEGMENTS}
    expected = {
        "App name strings": (0xEEC87, 0xEECA3),
        "TEL submenu labels": (0xEECE2, 0xEECF1),
        "SCHEDULE submenu labels": (0xEECFF, 0xEED12),
        "MEMO submenu label": (0xEED20, 0xEED25),
        "IOCS command-family table": (0xF04C4, 0xF0510),
        "BASIC error strings": (0x4631E, 0x46649),
        "S-C12 dispatch tables": (0x50046, 0x5007C),
        "Keyboard alternate secondary maps": (0x50DB7, 0x50DDF),
        "BASIC keyword letter offsets": (0x58E1C, 0x58E50),
        "BASIC token-to-keyword offset table": (0x58E50, 0x59050),
        "BASIC token dispatch table": (0x59050, 0x59350),
        "BASIC keyword table": (0x59350, 0x596FB),
        "IQ font and glyph assets": (0xF1B45, 0xF31EF),
        "RTC command table": (0xF320F, 0xF322D),
        "Keyboard command table": (0xF362A, 0xF3650),
        "Keyboard default-state block": (0xF3952, 0xF3969),
        "Keyboard character-transform tables": (0xF3C19, 0xF4019),
        "Primary keyboard keycode map": (0xF4019, 0xF4081),
        "Default secondary keyboard map": (0xF4081, 0xF4095),
        "CARD SECRET DATA string": (0xF7C73, 0xF7C83),
        "RAM-disk command table": (0xF7CA9, 0xF7CD3),
        "RAM-disk template": (0xF82DA, 0xF8308),
        "WORLD city database": (0xF9409, 0xFA5DD),
        "Built-in demo records": (0xFD4AF, 0xFDAA0),
    }
    assert {
        name: (segments[name].start, segments[name].start + segments[name].length)
        for name in expected
    } == expected


def test_iq7000_full_view_non_rom_windows_are_writable_data() -> None:
    segments = {seg.name: seg for seg in SC62015Iq7000FullView.SEGMENTS}

    for name in ("SH26", "CE1", "CE0", "Internal RAM"):
        data = segments[name]
        assert data.flags & SegmentFlag.SegmentReadable
        assert data.flags & SegmentFlag.SegmentWritable
        assert not data.flags & SegmentFlag.SegmentExecutable
        assert data.semantics == SectionSemantics.ReadWriteDataSectionSemantics


class _BytesData:
    def __init__(self, payload: bytes):
        self.payload = payload

    def read(self, offset: int, length: int) -> bytes:
        return self.payload[offset : offset + length]


def test_full_view_signatures_select_the_correct_model() -> None:
    pc_e500 = _BytesData(b"\x00\x00\x0c\x18\xf8\x0d\x40\x00\x24\x07\x00\x00")
    iq7000 = _BytesData(b"\x01\x01\x01\x01\x01\x01\x01\x01\x06\x60\x00\x34")

    assert SC62015FullView.is_valid_for_data(pc_e500)
    assert not SC62015Iq7000FullView.is_valid_for_data(pc_e500)
    assert SC62015Iq7000FullView.is_valid_for_data(iq7000)
    assert not SC62015FullView.is_valid_for_data(iq7000)


def test_iocs_entry_uses_one_byte_attribute_at_offset_four() -> None:
    assert "uint8_t device_attr" in IOCS_TYPES_C
    assert "enum IOCSAttribute device_attr" not in IOCS_TYPES_C
    assert "uint8_t entry_address[3]" in IOCS_TYPES_C
