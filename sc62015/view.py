from dataclasses import dataclass

from binaryninja.architecture import Architecture
from binaryninja.binaryview import BinaryView
from binaryninja.enums import Endianness, SectionSemantics, SegmentFlag, SymbolType
from binaryninja.log import log_warn
from binaryninja.types import Symbol

# Import architecture-specific constants
from .pysc62015.constants import (
    INTERNAL_MEMORY_START,
    INTERNAL_MEMORY_LENGTH,
)
from .pysc62015.instr import (
    IMEMRegisters,
    INTERRUPT_VECTOR_ADDR,
    ENTRY_POINT_ADDR,
    SH26_ADDR_START,
    SH26_ADDR_END,
    LH5073A1_ADDR_START,
    LH5073A1_ADDR_END,
    CE1_ADDR_START,
    CE1_ADDR_END,
    CE0_ADDR_START,
    CE0_ADDR_END,
)


_READ_ONLY_DATA_SECTION_SEMANTICS = getattr(
    SectionSemantics,
    "ReadOnlyDataSectionSemantics",
    SectionSemantics.ReadWriteDataSectionSemantics,
)


IOCS_TYPES_C = """enum IOCSAttribute {
    DEVICE_READ_ENABLE = 1,
    DEVICE_WRITE_ENABLE = 2,
    DEVICE_NO_SIMULTANEOUS_RW = 4,
    DEVICE_ASCII_DEFAULT = 16,
    DEVICE_CHARACTER = 32,
    DEVICE_SPECIAL_FILE = 64,
    DEVICE_FILE_CONTROL = 128
};

struct IOCSEntry {
    uint8_t next_header_addr[3];      // +0: little-endian 20-bit next pointer
    uint8_t device_number;            // +3: device number
    uint8_t device_attr;              // +4: IOCSAttribute bit field (one byte)
    uint8_t entry_address[3];         // +5: little-endian 20-bit handler
    // +8: NUL-terminated variable-length device name follows
};"""


@dataclass(frozen=True)
class SegmentDef:
    name: str
    start: int
    length: int
    file_offset: int | None
    flags: SegmentFlag
    semantics: SectionSemantics


class SC62015BaseView(BinaryView):
    """
    Base class for SC62015 BinaryViews. Subclasses must define:
      - name
      - long_name
      - SEGMENTS: list[SegmentDef]
      - is_valid_for_data()
    """

    EXTRA_ENTRY_POINTS: tuple[tuple[int, str], ...] = ()

    def __init__(self, data):
        super().__init__(parent_view=data, file_metadata=data.file)
        self.data = data
        self._interrupt_vector = 0
        self._entry_point = 0

    def init(self) -> bool:
        # Set architecture + platform
        arch = Architecture["SC62015"]
        self.arch = arch
        self.platform = arch.standalone_platform

        # Add segments and sections
        for seg in self.SEGMENTS:
            data_off = seg.file_offset if seg.file_offset is not None else 0
            data_len = seg.length if seg.file_offset is not None else 0
            self.add_auto_segment(
                seg.start,
                seg.length,
                data_off,
                data_len,
                seg.flags,
            )
            self.add_auto_section(
                seg.name,
                seg.start,
                seg.length,
                seg.semantics,
            )

        # Define named internal-memory variables
        for reg in IMEMRegisters:
            addr = INTERNAL_MEMORY_START + reg.value
            self.define_data_var(addr, "uint8_t", reg.name)

        # Read vectors and define entry points
        self._interrupt_vector = self.read_int(INTERRUPT_VECTOR_ADDR, 3)
        self._entry_point = self.read_int(ENTRY_POINT_ADDR, 3)

        self.define_auto_symbol(
            Symbol(
                SymbolType.FunctionSymbol, self._interrupt_vector, "interrupt_vector"
            )
        )
        self.add_function(self._interrupt_vector)

        self.define_auto_symbol(
            Symbol(SymbolType.FunctionSymbol, self._entry_point, "entry_point")
        )
        self.add_function(self._entry_point)

        for address, name in self.EXTRA_ENTRY_POINTS:
            self.define_auto_symbol(Symbol(SymbolType.FunctionSymbol, address, name))
            self.add_function(address)

        # Define types
        self._define_types()

        return True

    def _define_types(self):
        """Define SC62015-specific types in Binary Ninja"""

        # Parse and define the types
        try:
            types_result = self.platform.parse_types_from_source(IOCS_TYPES_C)
            if types_result.types:
                for name, type_obj in types_result.types.items():
                    self.define_user_type(name, type_obj)
        except Exception as e:
            # Log error but don't fail initialization
            log_warn(f"Failed to define types: {e}")

    def perform_get_address_size(self) -> int:
        return 3

    def perform_get_default_endianness(self) -> Endianness:
        return Endianness.LittleEndian

    def perform_get_entry_point(self):
        return self._entry_point


class SC62015RomView(SC62015BaseView):
    """
    View for standalone ROM dumps (only the 0x20_000 bytes at 0xE0000).
    """

    name = "SC62015:ROM"
    long_name = "SC62015 ROM-only View"

    # Only the ROM is file-backed; internal RAM is virtual
    EXTRA_ENTRY_POINTS = (
        (0xFFFD8, "legacy_basic_iocs_staging_entry"),
        (0xFFFDC, "legacy_basic_iocs_entry"),
        (0xFFFE4, "public_fcs_entry"),
        (0xFFFE8, "public_iocs_entry"),
    )

    SEGMENTS: list[SegmentDef] = [
        SegmentDef(
            name="ROM",
            start=0xE0000,
            length=0x1FFD8,
            file_offset=0,
            flags=(SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable),
            semantics=SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "Legacy BASIC IOCS entries",
            0xFFFD8,
            0x8,
            0x1FFD8,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "Public-vector filler",
            0xFFFE0,
            0x4,
            0x1FFE0,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "FCS public entry",
            0xFFFE4,
            0x4,
            0x1FFE4,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "IOCS public entry",
            0xFFFE8,
            0x4,
            0x1FFE8,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "ROM metadata and CPU vectors",
            0xFFFEC,
            0x14,
            0x1FFEC,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        # SC62015 CPU internal RAM - not file-backed
        SegmentDef(
            name="Internal RAM",
            start=INTERNAL_MEMORY_START,  # 0x100000
            length=INTERNAL_MEMORY_LENGTH,  # 0x100 (256 bytes)
            file_offset=None,  # Virtual segment, not in file
            flags=(SegmentFlag.SegmentReadable | SegmentFlag.SegmentWritable),
            semantics=SectionSemantics.ReadWriteDataSectionSemantics,
        ),
    ]

    @classmethod
    def is_valid_for_data(cls, data) -> bool:
        # Expect at least 4 bytes and the ROM signature
        buf = data.read(0, 4)
        return buf == b"\x2a\x0a\x00\x00"


# Binary Ninja does not infer page-local indirect edges once the word arrays
# below are correctly marked non-executable.  Seed every unique handler word
# referenced by those tables so a clean analysis still starts at each real
# computed destination.  Names intentionally state only the provenance of the
# boundary; semantic names belong in the curated analysis after inspection.
_PC_E500_DISPATCH_TABLE_TARGET_ADDRESSES = tuple(
    sorted(
        {
            # SYSTM and FCS.
            0xE0043,
            0xE00BB,
            0xE00BF,
            0xE00C3,
            0xE012D,
            0xE0159,
            0xE01B1,
            0xE01B8,
            0xE01BC,
            0xE01C0,
            0xE0214,
            0xE04AD,
            0xE0700,
            0xE0838,
            0xE0858,
            0xE08A2,
            0xE0935,
            0xE09BF,
            0xE0A93,
            0xE0B38,
            0xE0B74,
            0xE0BC2,
            0xE0CC4,
            0xEF537,
            # Near-return and character/input computed dispatchers.
            0xE923A,
            0xE9243,
            0xE933D,
            0xE9346,
            0xE936B,
            0xE939C,
            0xE93F1,
            0xE9457,
            0xE945C,
            0xE9461,
            0xE9466,
            0xE946B,
            0xE9470,
            0xE9478,
            0xE947D,
            0xE948F,
            0xE94A3,
            0xE9612,
            # E:F:G:, CAS, printer, and COM.
            0xE4981,
            0xE49A0,
            0xE49E7,
            0xE4A70,
            0xE4A94,
            0xE4AE6,
            0xE4B2D,
            0xE4B51,
            0xE4B6D,
            0xE528D,
            0xE54B9,
            0xE59B7,
            0xE59EB,
            0xE5A89,
            0xE5A9D,
            0xE5AB2,
            0xE5AB6,
            0xE5ABA,
            0xE5ABF,
            0xE5AC5,
            0xE5B89,
            0xE5BCF,
            0xE5BD9,
            0xE5BDD,
            0xE5BE1,
            0xE5C7A,
            0xE5CB8,
            0xE5CCD,
            0xE5CDD,
            0xE5CE4,
            0xE5CFB,
            0xE5D2E,
            0xE5D50,
            0xE5D8A,
            0xE5DAB,
            0xE9A06,
            0xE9A0A,
            0xE9A85,
            0xE9B15,
            0xE9B86,
            0xE9BB0,
            0xE9BEA,
            0xE9BEE,
            0xE9C0E,
            0xE9C1C,
            0xE9C3A,
            0xE9C3E,
            0xE9C89,
            0xE9CAD,
            0xE9DA8,
            0xE9DD8,
            0xE9E0A,
            0xE9E52,
            0xE9EB0,
            0xEA7DC,
            0xEA7E0,
            0xEA7E8,
            0xEA7ED,
            0xEA7F1,
            0xEA7F9,
            0xEA816,
            0xEA88C,
            0xEA8A3,
            0xEA944,
            0xEACE1,
            0xEACE6,
            0xEACEB,
            0xEACF0,
            0xEACF5,
            0xEACFA,
            0xEACFF,
            0xEAD09,
            0xEAD99,
            0xEADA1,
            0xEADAB,
            0xEADE3,
            0xEAE37,
            0xEAE98,
            0xEAEED,
            0xEAF2D,
            0xEAF34,
            0xEAF94,
            0xEAFB0,
            0xEB030,
            0xEB2A9,
            0xEB33D,
            # Function driver.
            0xECE93,
            0xECE9E,
            0xECEA9,
            0xECEC2,
            0xED0EE,
            0xED25C,
            0xED265,
            0xED26E,
            0xED6E7,
            0xED6F0,
            0xED6F9,
            0xED99A,
            0xEDDDE,
            0xEDDE7,
            0xEDDFF,
            0xEE7A9,
            0xEE7B2,
            0xEEB19,
            0xEEB46,
            0xEEB4F,
            0xEECC3,
            0xEEF92,
            0xEF38B,
            0xEF392,
            0xEF399,
            0xEF862,
            0xEF866,
            0xEF86A,
            0xEF86E,
            0xEF872,
            0xEF876,
            0xEF87A,
            0xEF87E,
            0xEF882,
            0xEF886,
            0xEF88A,
            0xEF88E,
            0xEFD99,
            0xEFE34,
            # Memory-card and keyboard drivers.
            0xF0063,
            0xF00C3,
            0xF01C1,
            0xF02DF,
            0xF034B,
            0xF039E,
            0xF043E,
            0xF0460,
            0xF0470,
            0xF0517,
            0xF0570,
            0xF05F9,
            0xF061C,
            0xF0633,
            0xF067C,
            0xF073A,
            0xF0752,
            0xF07AD,
            0xF0821,
            0xF0859,
            0xF08DC,
            0xF0948,
            0xF09D8,
            0xF0A19,
            0xF0A63,
            0xF16B3,
            0xF16B9,
            0xF16FA,
            0xF176A,
            0xF192C,
            0xF1982,
            0xF19FC,
            0xF1B59,
            0xF1B8C,
            0xF1BF9,
            0xF1C3A,
            0xF1C6C,
        }
    )
)


class SC62015FullView(SC62015BaseView):
    """
    View for PC-E500 full-memory images.

    The checked-in PC-E500 images contain the complete 256 KiB system ROM at
    0xC0000-0xFFFFF.  The probe at 0xF0C11 recognizes the read-only 0x1210
    signature found at 0xC0000, and coherent firmware functions occupy the D
    bank.  The RAMFILE.! block in the C bank is a bundled read-only program-disk
    image, not proof that the bank is writable; treating that half as CE2 RAM
    makes real ROM code writable in analysis.
    The IQ-7000 has a distinct image signature and is handled by
    :class:`SC62015Iq7000FullView` below.
    """

    name = "SC62015:Memory"
    long_name = "SC62015 PC-E500 Full 1MB Memory View"

    # Map the complete 20-bit external address space.  File offsets equal
    # virtual addresses in the 1 MiB capture.
    EXTRA_ENTRY_POINTS = tuple(
        (address, f"dispatch_table_target_{address:05x}")
        for address in _PC_E500_DISPATCH_TABLE_TARGET_ADDRESSES
    ) + (
        # Preserve executable roots immediately following data islands when
        # no direct edge is guaranteed to cross the non-executable boundary.
        (0xE1457, "data_island_resume_e1457"),
        (0xE1476, "data_island_resume_e1476"),
        (0xE1AA1, "data_island_resume_e1aa1"),
        (0xE1DB4, "data_island_resume_e1db4"),
        (0xE24D0, "data_island_resume_e24d0"),
        # The ten IOCS device headers live in non-executable data and carry
        # 20-bit far-entry pointers.  Seed the roots that are not already
        # guaranteed by a computed-dispatch table or another explicit entry.
        (0xE0183, "iocs_header_entry_systm_e0183"),
        (0xE493E, "iocs_header_entry_efg_e493e"),
        (0xEA7A7, "iocs_header_entry_prn_ea7a7"),
        (0xEB956, "iocs_header_entry_xy_eb956"),
        (0xEF2EE, "iocs_header_entry_dev09_ef2ee"),
        (0xF0000, "iocs_header_entry_memcard_f0000"),
        (0xF1664, "iocs_header_entry_stdi_f1664"),
        (0xE7474, "data_island_resume_e7474"),
        (0xE9A04, "data_island_resume_e9a04"),
        (0xE99B1, "data_island_resume_e99b1"),
        # IOCS device headers store far entry pointers in the non-executable
        # header list.  These four-byte CALL/RETF ABI adapters therefore need
        # explicit roots even though their near dispatchers are discovered.
        (0xE99D2, "iocs_header_entry_cas_e99d2"),
        (0xEAD5B, "iocs_header_entry_com_ead5b"),
        (0xEAF69, "sio_close"),
        # STDI command-table target for unsupported write-style operations.
        # The indirect word table references this four-byte A=8/SC/RETF stub,
        # but recursive analysis does not reliably recover the target.
        (0xF16B5, "stdi_unsupported_write_error"),
        # This complete STDO/LCD sub-timer prelude has no known static ROM
        # caller, so normal recursive analysis cannot discover it.  It reads
        # the right LCD status, asks STDO command 0x50 to invert the display
        # power state, then falls through to the regular handler at 0xF210C.
        (0xF20D8, "lcd_power_toggle_then_subtimer"),
        # Boot installs this address directly in sub-timer vector slot 1.  It
        # is also the shared tail of the optional F20D8 LCD-power prelude, so
        # both overlapping function boundaries are intentional.
        (0xF210C, "subtimer_display_countdown_tick"),
        (0xF28F9, "iocs_stdo_screen_handler"),
        (0xF5A81, "basic_token61_command_stub"),
        (0xF5A84, "basic_token61_function_stub"),
        (0xF5E79, "basic_execute_intermediate_code"),
        (0xFFFD8, "legacy_basic_iocs_staging_entry"),
        (0xFFFDC, "legacy_basic_iocs_entry"),
        (0xFFFE4, "public_fcs_entry"),
        (0xFFFE8, "public_iocs_entry"),
    )

    SEGMENTS: list[SegmentDef] = [
        # The extension header and bundled RAMFILE.! filesystem are data.
        SegmentDef(
            "Bundled Program Disk",
            0xC0000,
            0x1F820,
            0xC0000,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        # Variable-length IOCS records immediately following RAMFILE.!.
        SegmentDef(
            "IOCS Header Data",
            0xDF820,
            0x98,
            0xDF820,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        # Coherent executable firmware begins at 0xDF8B8. Split the page-local
        # dispatch tables, font atlas, and packed BASIC tables out of the
        # executable map: broad-ROM analysis invented functions in those byte
        # arrays and obscured the real code that resumes after each island.
        SegmentDef(
            "System ROM before IOCS main dispatch table",
            0xDF8B8,
            0x77F,
            0xDF8B8,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "IOCS main dispatch table",
            0xE0037,
            0xC,
            0xE0037,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "System ROM before SYSTM dispatch table",
            0xE0043,
            0x160,
            0xE0043,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "SYSTM dispatch table",
            0xE01A3,
            0xE,
            0xE01A3,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "System ROM before FCS dispatch table",
            0xE01B1,
            0x52B,
            0xE01B1,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "FCS dispatch table",
            0xE06DC,
            0x24,
            0xE06DC,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "System ROM before packed format tables",
            0xE0700,
            0xCD1,
            0xE0700,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "Packed format-record offset table",
            0xE13D1,
            0x15,
            0xE13D1,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "Packed format-operation records",
            0xE13E6,
            0x71,
            0xE13E6,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "System ROM before BREAK prompt",
            0xE1457,
            0xC,
            0xE1457,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "Push BREAK to Stop text",
            0xE1463,
            0x13,
            0xE1463,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "System ROM before near-return dispatch table",
            0xE1476,
            0x607,
            0xE1476,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "Near-return dispatch table",
            0xE1A7D,
            0x24,
            0xE1A7D,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "System ROM before powers-of-ten table",
            0xE1AA1,
            0x309,
            0xE1AA1,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "Powers-of-ten table",
            0xE1DAA,
            0xA,
            0xE1DAA,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "System ROM before key/input translation lookup",
            0xE1DB4,
            0x6BA,
            0xE1DB4,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "Key/input translation lookup",
            0xE246E,
            0x62,
            0xE246E,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "System ROM before E:F:G: dispatch table",
            0xE24D0,
            0x244E,
            0xE24D0,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "E:F:G: dispatch table",
            0xE491E,
            0x20,
            0xE491E,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "System ROM before computed dispatch tables",
            0xE493E,
            0x993,
            0xE493E,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "Character/input computed dispatch table (32 words)",
            0xE52D1,
            0x40,
            0xE52D1,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "Character/input computed dispatch table (212 words)",
            0xE5311,
            0x1A8,
            0xE5311,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "System ROM before Q15 trigonometry tables",
            0xE54B9,
            0x1F03,
            0xE54B9,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "Q15 cosine 0-to-45-degree table",
            0xE73BC,
            0x5C,
            0xE73BC,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "Q15 sine 0-to-45-degree table",
            0xE7418,
            0x5C,
            0xE7418,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "System ROM before BASIC/FCS error messages",
            0xE7474,
            0x2217,
            0xE7474,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "BASIC/FCS error-message table",
            0xE968B,
            0x326,
            0xE968B,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "System ROM before CAS dispatch tables",
            0xE99B1,
            0x25,
            0xE99B1,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "CAS special-file dispatch table",
            0xE99D6,
            0x20,
            0xE99D6,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "CAS device-command dispatch table",
            0xE99F6,
            0xE,
            0xE99F6,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "System ROM before printer dispatch tables",
            0xE9A04,
            0xDC0,
            0xE9A04,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "Printer standard-character dispatch table",
            0xEA7C4,
            0xE,
            0xEA7C4,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "Printer device-command dispatch table",
            0xEA7D2,
            0xA,
            0xEA7D2,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "System ROM before COM dispatch tables",
            0xEA7DC,
            0x583,
            0xEA7DC,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "COM special-file dispatch table",
            0xEAD5F,
            0x20,
            0xEAD5F,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "COM device-command dispatch table",
            0xEAD7F,
            0x1A,
            0xEAD7F,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "System ROM before function-driver dispatch tables",
            0xEAD99,
            0x45A4,
            0xEAD99,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "Function-driver numeric dispatch table",
            0xEF33D,
            0x36,
            0xEF33D,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "Function-driver string dispatch table",
            0xEF373,
            0x14,
            0xEF373,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "Function-driver conversion dispatch table",
            0xEF387,
            0x4,
            0xEF387,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "System ROM before memory-card dispatch tables",
            0xEF38B,
            0xCA4,
            0xEF38B,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "Memory-card special-file dispatch table",
            0xF002F,
            0x20,
            0xF002F,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "Memory-card management dispatch table",
            0xF004F,
            0x14,
            0xF004F,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "System ROM before keyboard dispatch tables",
            0xF0063,
            0x1630,
            0xF0063,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "Keyboard standard-character dispatch table",
            0xF1693,
            0x10,
            0xF1693,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "Keyboard device-command dispatch table",
            0xF16A3,
            0x10,
            0xF16A3,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "System ROM before font atlas",
            0xF16B3,
            0xAA2,
            0xF16B3,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "PC-E500 font and bitmap atlas",
            0xF2155,
            0x7A4,
            0xF2155,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "System ROM before STDO dispatch tables",
            0xF28F9,
            0x2F,
            0xF28F9,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "STDO dispatch tables",
            0xF2928,
            0x42,
            0xF2928,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "System ROM before BASIC token-to-keyword offsets",
            0xF296A,
            0x2C17,
            0xF296A,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "BASIC token-to-keyword offset table",
            0xF5581,
            0x200,
            0xF5581,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "BASIC token descriptor table",
            0xF5781,
            0x300,
            0xF5781,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "BASIC token trampolines",
            0xF5A81,
            0xB,
            0xF5A81,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "BASIC keyword dictionary",
            0xF5A8C,
            0x3E9,
            0xF5A8C,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "System ROM after BASIC dictionary",
            0xF5E75,
            0xA163,
            0xF5E75,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "Legacy BASIC IOCS entries",
            0xFFFD8,
            0x8,
            0xFFFD8,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "Public-vector filler",
            0xFFFE0,
            0x4,
            0xFFFE0,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "FCS public entry",
            0xFFFE4,
            0x4,
            0xFFFE4,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "IOCS public entry",
            0xFFFE8,
            0x4,
            0xFFFE8,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "ROM metadata and CPU vectors",
            0xFFFEC,
            0x14,
            0xFFFEC,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        # SC62015 CPU Internal RAM at 0x100000-0x1000FF (256 bytes)
        # Note: This is the CPU's built-in RAM, separate from external memory space. Full
        # memory dumps are typically 1MB (0x00000-0xFFFFF) and do not include this region.
        SegmentDef(
            "Internal RAM",
            INTERNAL_MEMORY_START,
            INTERNAL_MEMORY_LENGTH,
            None,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentWritable,
            SectionSemantics.ReadWriteDataSectionSemantics,
        ),
        # SH26 device registers
        SegmentDef(
            "SH26",
            SH26_ADDR_START,
            SH26_ADDR_END - SH26_ADDR_START + 1,
            SH26_ADDR_START,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentWritable,
            SectionSemantics.ReadWriteDataSectionSemantics,
        ),
        # LH5073A1 registers
        SegmentDef(
            "LH5073A1",
            LH5073A1_ADDR_START,
            LH5073A1_ADDR_END - LH5073A1_ADDR_START + 1,
            LH5073A1_ADDR_START,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentWritable,
            SectionSemantics.ReadWriteDataSectionSemantics,
        ),
        # CE1 registers
        SegmentDef(
            "CE1",
            CE1_ADDR_START,
            CE1_ADDR_END - CE1_ADDR_START + 1,
            CE1_ADDR_START,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentWritable,
            SectionSemantics.ReadWriteDataSectionSemantics,
        ),
        # CE0 registers
        SegmentDef(
            "CE0",
            CE0_ADDR_START,
            CE0_ADDR_END - CE0_ADDR_START + 1,
            CE0_ADDR_START,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentWritable,
            SectionSemantics.ReadWriteDataSectionSemantics,
        ),
    ]

    @classmethod
    def is_valid_for_data(cls, data) -> bool:
        buf = data.read(0, 4 * 3)
        return buf == b"\x00\x00\x0c\x18\xf8\x0d\x40\x00\x24\x07\x00\x00"


class SC62015Iq7000FullView(SC62015BaseView):
    """View for IQ-7000 full-memory images with 256 KiB of unique ROM.

    The checked-in capture contains two exact 128 KiB mirror pairs:
    ``0x40000-0x5FFFF`` is repeated at ``0x60000-0x7FFFF``, and
    ``0xE0000-0xFFFFF`` is repeated at ``0xC0000-0xDFFFF``.  Existing ROM
    references and curated analysis use the first range of the lower pair and
    the second range of the upper pair. The S-C12 descriptor/filler occupies
    ``0x40000-0x400FF`` and executable card code begins at ``0x40100``. The
    other copies remain readable but non-executable to prevent duplicate
    function discovery.

    This describes the bytes in the full-memory capture; it does not by itself
    prove whether real hardware aliases those addresses or the acquisition
    process duplicated separately selected ROM windows.
    """

    name = "SC62015:IQ-7000 Memory"
    long_name = "SC62015 IQ-7000 Full 1MB Memory View"

    EXTRA_ENTRY_POINTS = (
        (0x40100, "s_c12_card_iocs_dispatcher"),
        (0x4026A, "s_c12_card_iocs_broadcast_il40"),
        (0x4345E, "s_c12_iocs_efg_handler"),
        (0x44C2E, "sync_workspace_coordinate_pairs_from_px_flags?_44c2e"),
        (0x45235, "s_c12_iocs_cassette_handler"),
        (0x4574A, "s_c12_iocs_printer_handler"),
        (0x458CB, "s_c12_iocs_com_handler"),
        (0x48C31, "s_c12_iocs_dev09_handler"),
        (0x49C5B, "s_c12_iocs_systm_handler"),
        (0x50000, "s_c12_storage_handler"),
        (0x50AE2, "s_c12_iocs_stdi_handler"),
        (0x50DDF, "s_c12_iocs_stdo_handler"),
        (0x5A3F8, "parse_prefixed_text_field?_5a3f8"),
        (0x5E7EB, "s_c12_card_ap_entry"),
        (0x5F868, "s_c12_iocs_pacom_handler"),
        (0x5FFE0, "lower_rom_jpf_5d384_entry_5ffe0"),
        (0x5FFE4, "lower_rom_jpf_48c22_entry_5ffe4"),
        (0x5FFE8, "lower_rom_jpf_49deb_entry_5ffe8"),
        (0x5FFF0, "lower_rom_jpf_4038f_entry_5fff0"),
        (0x5FFF4, "s_c12_card_iocs_entry"),
        (0xE000F, "daily_alarm_service_keyc00a_e000f"),
        (0xE08E1, "error_selector_descriptor_table_driver_e08e1"),
        (0xE0A29, "plan_service_keyc00b_e0a29"),
        (0xE1A46, "anniversary_date_format_service_key400c_e1a46"),
        (0xE4571, "ui_shell_service_dispatcher_key5000_e4571"),
        (0xE4760, "ui_shell_state_dispatcher_e4760"),
        (0xE4768, "ui_shell_state0_init_e4768"),
        (0xE478A, "ui_shell_state0_event_loop_e478a"),
        (0xE48BB, "ui_shell_state1_event_loop_e48bb"),
        (0xE4ACB, "ui_shell_state2_init_e4acb"),
        (0xE4AE8, "ui_shell_state2_event_loop_e4ae8"),
        (0xE508F, "error_selector_mode_dispatch_carry_clear_e508f"),
        (0xE70CA, "schedule_alarm_list_manager_e70ca"),
        (0xEBA84, "compute_3byte_record_offset_from_ba_eba84"),
        (0xEECA3, "app_menu_entry_calendar_default_eeca3"),
        (0xEECF1, "app_tel_submenu_handler_eecf1"),
        (0xEED12, "app_schedule_submenu_handler_eed12"),
        (0xEED25, "app_memo_submenu_handler_eed25"),
        (0xF04A5, "iocs_stdo_screen_handler"),
        (0xF04C0, "iocs_error_stub_a3_sc_f04c0"),
        (0xF069D, "iq7000_font_lookup_ptr_far_f069d"),
        (0xF069F, "ret_stub_f069f"),
        (0xF07A3, "iocs_init_slot0_f07a3"),
        (0xF08AE, "font_select_table_set_f08ae"),
        (0xF0934, "iocs_char_io_one_byte_f0934"),
        (0xF0940, "iocs_char_io_loop_f0940"),
        (0xF0A62, "iocs_sync_workspace_e6_08_f0a62"),
        (0xF0BEB, "iocs_fetch_workspace_ptr_words_f0beb"),
        (0xF0C36, "iocs_init_tail_state_sync_f0c36"),
        (0xF0CBA, "iocs_block_io_f0cba"),
        (0xF0E09, "iocs_block_io_with_header_f0e09"),
        (0xF0E92, "iocs_state_sync_masked_f0e92"),
        (0xF0F01, "iocs_cache_d4_to_1fd9b_f0f01"),
        (0xF0F0E, "iocs_status_flag_helper_f0f0e"),
        (0xF0F2D, "iocs_command_slot25_handler?_f0f2d"),
        (0xF100E, "iocs_command_slot27_handler?_f100e"),
        (0xF1023, "iocs_sync_workspace_e6_09_f1023"),
        (0xF11E4, "iocs_command_slot28_handler?_f11e4"),
        (0xF129B, "iocs_command_slot29_handler?_f129b"),
        (0xF12B0, "iocs_command_slot30_handler?_f12b0"),
        (0xF13C0, "iocs_command_slot31_handler?_f13c0"),
        (0xF148C, "iocs_display_ctrl_slot14_f148c"),
        (0xF174A, "iocs_display_ctrl_slot12_f174a"),
        (0xF1774, "iocs_display_ctrl_slot13_f1774"),
        (0xF1868, "iocs_display_ctrl_slot11_f1868"),
        (0xF18C5, "iocs_display_ctrl_slot10_f18c5"),
        (0xF1988, "iocs_display_ctrl_slot33?_f1988"),
        (0xF19A7, "display_ctrl_prepare_vram_base_entry_f19a7"),
        (0xF1AD3, "iocs_display_ctrl_slot35?_f1ad3"),
        (0xF31EF, "iocs_rtc_handler"),
        (0xF355E, "rtc_cmd_fd_status_f355e"),
        (0xF360A, "iocs_stdi_keyboard_handler"),
        (0xF3969, "keyboard_clear_buffers_f3969"),
        (0xF3A7D, "keyboard_buffer_walk_wrapper_il1_f3a7d"),
        (0xF4095, "keyboard_char_transform_dispatch?_f4095"),
        (0xF523E, "irq_keyi_default_handler"),
        (0xF5247, "irq_onki_default_handler"),
        (0xF525D, "irq_txri_default_handler"),
        (0xF525E, "irq_mti_default_handler"),
        (0xF527F, "irq_exi_default_handler"),
        (0xF52FF, "irq_sti_default_handler"),
        (0xF563C, "irq_rxri_default_handler"),
        (0xF5702, "iocs_systm_handler"),
        (0xF5C88, "iocs_dev0e_handler"),
        (0xF5CAC, "dev0e_noop_success_f5cac"),
        (0xF5CAE, "dev0e_timed_scr_strobe_f5cae"),
        (0xF6777, "iocs_dev0d_record_service_dispatcher_f6777"),
        (0xF67D7, "system_setup_noop_success_f67d7"),
        (0xF6BBF, "user_dict_cmd_next_then_mode0_f6bbf"),
        (0xF6C07, "user_dict_cmd_prev_then_mode1_f6c07"),
        (0xF6C3C, "user_dict_cmd_next_then_mode2_f6c3c"),
        (0xF6C44, "user_dict_cmd_field_match_mode2_f6c44"),
        (0xF6C6D, "user_dict_cmd_prev_then_mode3_f6c6d"),
        (0xF6C75, "user_dict_cmd_field_match_mode3_f6c75"),
        (0xF7364, "boot_init_dispatch_and_storage_f7364"),
        (0xF7C83, "iocs_ram_disk_handler"),
        (0xF8308, "ram_disk_copy_record_to_caller_f8308"),
        (0xF8733, "user_dict_transfer_service_f8733"),
        (0xF8914, "user_dict_transfer_service_f8914"),
        (0xFA5FE, "iocs_com_handler"),
        (0xFA668, "com_reset_or_close_dispatch_fa668"),
        (0xFA7E3, "com_rearm_line_and_flow_il0_fa7e3"),
        (0xFA7E5, "com_rearm_line_and_flow_fa7e5"),
        (0xFA81A, "com_line_close_worker_fa81a"),
        (0xFAEC0, "iocs_cassette_handler"),
        (0xFB1DC, "cas_render_status_found_fb1dc"),
        (0xFB439, "iocs_printer_handler"),
        (0xFB9ED, "prn_panet_seed_default_config_fb9ed"),
        (0xFBA38, "pacom_handshake_send_a5_and_read2_fba38"),
        (0xFBC00, "iocs_pacom_handler"),
        (0xFBC3B, "pacom_panet_noop_success_fbc3b"),
        (0xFBC3D, "iocs_panet_handler"),
        (0xFBD4B, "pacom_panet_recv_frame_checksum_ack_check_first_fbd4b"),
        (0xFBD54, "pacom_panet_recv_frame_checksum_ack_fbd54"),
        (0xFD41A, "demo_show_builtin_samples_record1_fd41a"),
        (0xFD426, "demo_show_builtin_samples_fd426"),
        (0xFD432, "demo_render_blob_fd432"),
        (0xFDAA0, "factory_diagnostic_boot_gate_fdaa0"),
        (0xFFFDC, "public_il_dispatcher"),
        (0xFFFE4, "public_interrupt_vector_seed"),
        (0xFFFE8, "public_iocs_entry"),
    )

    SEGMENTS: list[SegmentDef] = [
        # Writable data/peripheral window in the capture.
        SegmentDef(
            "SH26",
            SH26_ADDR_START,
            SH26_ADDR_END - SH26_ADDR_START + 1,
            SH26_ADDR_START,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentWritable,
            SectionSemantics.ReadWriteDataSectionSemantics,
        ),
        # Canonical lower-ROM descriptor, executable card code, and exact copy.
        SegmentDef(
            "Lower ROM header",
            0x40000,
            0x100,
            0x40000,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "Lower ROM before BASIC error strings",
            0x40100,
            0x4631E - 0x40100,
            0x40100,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "BASIC error strings",
            0x4631E,
            0x46649 - 0x4631E,
            0x4631E,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "Lower ROM before operand and FF padding",
            0x46649,
            0x49E41 - 0x46649,
            0x46649,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "Lower ROM operand and FF padding",
            0x49E41,
            0x4A000 - 0x49E41,
            0x49E41,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "Embedded RAMFILE BASIC filesystem",
            0x4A000,
            0x4EC08 - 0x4A000,
            0x4A000,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "Lower ROM FF padding 4EC08",
            0x4EC08,
            0x50000 - 0x4EC08,
            0x4EC08,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "S-C12 storage dispatcher",
            0x50000,
            0x50046 - 0x50000,
            0x50000,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "S-C12 dispatch tables",
            0x50046,
            0x5007C - 0x50046,
            0x50046,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "Lower ROM before keyboard secondary maps",
            0x5007C,
            0x50DB7 - 0x5007C,
            0x5007C,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "Keyboard alternate secondary maps",
            0x50DB7,
            0x50DDF - 0x50DB7,
            0x50DB7,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "Lower ROM before IOCS headers",
            0x50DDF,
            0x51729 - 0x50DDF,
            0x50DDF,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "S-C12 IOCS header data",
            0x51729,
            0x9D,
            0x51729,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "Lower ROM after IOCS headers",
            0x517C6,
            0x58E1C - 0x517C6,
            0x517C6,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "BASIC keyword letter offsets",
            0x58E1C,
            0x58E50 - 0x58E1C,
            0x58E1C,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "BASIC token-to-keyword offset table",
            0x58E50,
            0x59050 - 0x58E50,
            0x58E50,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "BASIC token dispatch table",
            0x59050,
            0x59350 - 0x59050,
            0x59050,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "BASIC keyword table",
            0x59350,
            0x596FB - 0x59350,
            0x59350,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "Lower ROM tail before FF padding",
            0x596FB,
            0x5FCB3 - 0x596FB,
            0x596FB,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "Lower ROM FF padding 5FCB3",
            0x5FCB3,
            0x5FFD9 - 0x5FCB3,
            0x5FCB3,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "Lower ROM vector-tail metadata before JPF stubs",
            0x5FFD9,
            0x5FFE0 - 0x5FFD9,
            0x5FFD9,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "Lower ROM first JPF stub group",
            0x5FFE0,
            0x5FFEC - 0x5FFE0,
            0x5FFE0,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "Lower ROM vector-tail FF separator",
            0x5FFEC,
            0x5FFF0 - 0x5FFEC,
            0x5FFEC,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "Lower ROM second JPF stub group",
            0x5FFF0,
            0x5FFF8 - 0x5FFF0,
            0x5FFF0,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "Lower ROM vector-tail metadata",
            0x5FFF8,
            0x60000 - 0x5FFF8,
            0x5FFF8,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "Lower ROM mirror",
            0x60000,
            0x20000,
            0x60000,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        # Writable external data windows in the capture.
        SegmentDef(
            "CE1",
            CE1_ADDR_START,
            CE1_ADDR_END - CE1_ADDR_START + 1,
            CE1_ADDR_START,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentWritable,
            SectionSemantics.ReadWriteDataSectionSemantics,
        ),
        SegmentDef(
            "CE0",
            CE0_ADDR_START,
            CE0_ADDR_END - CE0_ADDR_START + 1,
            CE0_ADDR_START,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentWritable,
            SectionSemantics.ReadWriteDataSectionSemantics,
        ),
        # The system-ROM copy at 0xE0000 is canonical in existing analysis.
        SegmentDef(
            "System ROM mirror",
            0xC0000,
            0x20000,
            0xC0000,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "System ROM before HOME label",
            0xE0000,
            0xEBA78 - 0xE0000,
            0xE0000,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "HOME title label",
            0xEBA78,
            0xEBA84 - 0xEBA78,
            0xEBA78,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "System ROM after HOME label",
            0xEBA84,
            0xEEC87 - 0xEBA84,
            0xEBA84,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "App name strings",
            0xEEC87,
            0xEECA3 - 0xEEC87,
            0xEEC87,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "App menu wrappers",
            0xEECA3,
            0xEECE2 - 0xEECA3,
            0xEECA3,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "TEL submenu labels",
            0xEECE2,
            0xEECF1 - 0xEECE2,
            0xEECE2,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "TEL submenu handler",
            0xEECF1,
            0xEECFF - 0xEECF1,
            0xEECF1,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "SCHEDULE submenu labels",
            0xEECFF,
            0xEED12 - 0xEECFF,
            0xEECFF,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "SCHEDULE submenu handler",
            0xEED12,
            0xEED20 - 0xEED12,
            0xEED12,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "MEMO submenu label",
            0xEED20,
            0xEED25 - 0xEED20,
            0xEED20,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "System ROM before late FF padding",
            0xEED25,
            0xEFFB0 - 0xEED25,
            0xEED25,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "System ROM FF padding EFFB0",
            0xEFFB0,
            0xF0000 - 0xEFFB0,
            0xEFFB0,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "System ROM before IOCS headers",
            0xF0000,
            0xF03F5 - 0xF0000,
            0xF0000,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "System IOCS header data",
            0xF03F5,
            0xB0,
            0xF03F5,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "IOCS dispatch prelude",
            0xF04A5,
            0xF04C4 - 0xF04A5,
            0xF04A5,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "IOCS command-family table",
            0xF04C4,
            0xF0510 - 0xF04C4,
            0xF04C4,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "System ROM services",
            0xF0510,
            0xF1B45 - 0xF0510,
            0xF0510,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "IQ font and glyph assets",
            0xF1B45,
            0xF31EF - 0xF1B45,
            0xF1B45,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "RTC dispatcher",
            0xF31EF,
            0xF320F - 0xF31EF,
            0xF31EF,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "RTC command table",
            0xF320F,
            0xF322D - 0xF320F,
            0xF320F,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "System ROM before keyboard dispatch",
            0xF322D,
            0xF362A - 0xF322D,
            0xF322D,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "Keyboard command table",
            0xF362A,
            0xF3650 - 0xF362A,
            0xF362A,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "System ROM before keyboard default state",
            0xF3650,
            0xF3952 - 0xF3650,
            0xF3650,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "Keyboard default-state block",
            0xF3952,
            0xF3969 - 0xF3952,
            0xF3952,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "System ROM before keyboard fallback callback records",
            0xF3969,
            0xF3A68 - 0xF3969,
            0xF3969,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "Keyboard fallback callback records",
            0xF3A68,
            0xF3A7D - 0xF3A68,
            0xF3A68,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "System ROM before keyboard translation tables",
            0xF3A7D,
            0xF3C19 - 0xF3A7D,
            0xF3A7D,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "Keyboard character-transform tables",
            0xF3C19,
            0xF4019 - 0xF3C19,
            0xF3C19,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "Primary keyboard keycode map",
            0xF4019,
            0xF4081 - 0xF4019,
            0xF4019,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "Default secondary keyboard map",
            0xF4081,
            0xF4095 - 0xF4081,
            0xF4081,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "System ROM services before dev0E dispatch table",
            0xF4095,
            0xF5CA8 - 0xF4095,
            0xF4095,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "dev0E command dispatch table",
            0xF5CA8,
            0xF5CAC - 0xF5CA8,
            0xF5CA8,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "System ROM services after dev0E dispatch table",
            0xF5CAC,
            0xF7C73 - 0xF5CAC,
            0xF5CAC,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "CARD SECRET DATA string",
            0xF7C73,
            0x10,
            0xF7C73,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "RAM-disk dispatcher",
            0xF7C83,
            0xF7CA9 - 0xF7C83,
            0xF7C83,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "RAM-disk command table",
            0xF7CA9,
            0xF7CD3 - 0xF7CA9,
            0xF7CA9,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "System ROM storage services",
            0xF7CD3,
            0xF82DA - 0xF7CD3,
            0xF7CD3,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "RAM-disk template",
            0xF82DA,
            0xF8308 - 0xF82DA,
            0xF82DA,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "System ROM before WORLD city data",
            0xF8308,
            0xF9409 - 0xF8308,
            0xF8308,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "WORLD city database",
            0xF9409,
            0xFA5DD - 0xF9409,
            0xF9409,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "System ROM communications and services",
            0xFA5DD,
            0xFD4AF - 0xFA5DD,
            0xFA5DD,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "Built-in demo records",
            0xFD4AF,
            0xFDAA0 - 0xFD4AF,
            0xFD4AF,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "Factory diagnostics before FF padding",
            0xFDAA0,
            0xFE52B - 0xFDAA0,
            0xFDAA0,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "System ROM FF padding FE52B",
            0xFE52B,
            0xFFFDC - 0xFE52B,
            0xFE52B,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "System ROM final public-dispatch code",
            0xFFFDC,
            0xFFFE0 - 0xFFFDC,
            0xFFFDC,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        # Keep filler and metadata/vector bytes out of executable analysis while
        # preserving the two public call stubs as exact code segments.
        SegmentDef(
            "Public-vector filler",
            0xFFFE0,
            0x4,
            0xFFFE0,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "Interrupt-vector seed entry",
            0xFFFE4,
            0x3,
            0xFFFE4,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "Public-vector separator",
            0xFFFE7,
            0x1,
            0xFFFE7,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        SegmentDef(
            "IOCS public entry",
            0xFFFE8,
            0x4,
            0xFFFE8,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
            SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        SegmentDef(
            "ROM metadata and CPU vectors",
            0xFFFEC,
            0x14,
            0xFFFEC,
            SegmentFlag.SegmentReadable,
            _READ_ONLY_DATA_SECTION_SEMANTICS,
        ),
        # CPU internal RAM is outside the 20-bit external capture.
        SegmentDef(
            "Internal RAM",
            INTERNAL_MEMORY_START,
            INTERNAL_MEMORY_LENGTH,
            None,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentWritable,
            SectionSemantics.ReadWriteDataSectionSemantics,
        ),
    ]

    @classmethod
    def is_valid_for_data(cls, data) -> bool:
        buf = data.read(0, 4 * 3)
        return buf == b"\x01\x01\x01\x01\x01\x01\x01\x01\x06\x60\x00\x34"
