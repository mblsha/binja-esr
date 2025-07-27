from dataclasses import dataclass
from typing import Optional, List

from binaryninja.binaryview import BinaryView
from binaryninja.architecture import Architecture
from binaryninja.enums import SegmentFlag, SectionSemantics, SymbolType, Endianness
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

@dataclass(frozen=True)
class SegmentDef:
    name: str
    start: int
    length: int
    file_offset: Optional[int]
    flags: SegmentFlag
    semantics: SectionSemantics


class SC62015BaseView(BinaryView):
    """
    Base class for SC62015 BinaryViews. Subclasses must define:
      - name
      - long_name
      - SEGMENTS: List[SegmentDef]
      - is_valid_for_data()
    """
    def __init__(self, data):
        super().__init__(parent_view=data, file_metadata=data.file)
        self.data = data
        self._interrupt_vector = 0
        self._entry_point = 0

    def init(self) -> bool:
        # Set architecture + platform
        arch = Architecture['SC62015']
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
            self.define_data_var(addr, 'uint8_t', reg.name)

        # Read vectors and define entry points
        self._interrupt_vector = self.read_int(INTERRUPT_VECTOR_ADDR, 3)
        self._entry_point = self.read_int(ENTRY_POINT_ADDR, 3)

        self.define_auto_symbol(
            Symbol(SymbolType.FunctionSymbol, self._interrupt_vector, 'interrupt_vector')
        )
        self.add_function(self._interrupt_vector)

        self.define_auto_symbol(
            Symbol(SymbolType.FunctionSymbol, self._entry_point, 'entry_point')
        )
        self.add_function(self._entry_point)

        # Define types
        self._define_types()

        return True

    def _define_types(self):
        """Define SC62015-specific types in Binary Ninja"""
        
        # Define IOCSAttribute enum
        iocs_attr_c = '''enum IOCSAttribute {
    DEVICE_READ_ENABLE = 1,
    DEVICE_WRITE_ENABLE = 2, 
    DEVICE_NO_SIMULTANEOUS_RW = 4,
    DEVICE_ASCII_DEFAULT = 16,
    DEVICE_CHARACTER = 32,
    DEVICE_SPECIAL_FILE = 64,
    DEVICE_FILE_CONTROL = 128
};'''
        
        # Define IOCSEntry struct  
        iocs_entry_c = '''struct IOCSEntry {
    uint8_t next_header_addr[3];      // +0: Address to next IOCS header (3 bytes)
    uint8_t device_number;            // +3: Device number (1 byte)
    enum IOCSAttribute device_attr;   // +4: Device attribute (1 byte)
    uint8_t entry_address[3];         // +5: Entry address of each IOCS (3 bytes)
    // +8: Drive name follows (variable length, null-terminated, max 5 bytes)
};'''
        
        # Parse and define the types
        try:
            types_result = self.platform.parse_types_from_source(iocs_attr_c + '\n' + iocs_entry_c)
            if types_result.types:
                for name, type_obj in types_result.types.items():
                    self.define_user_type(name, type_obj)
        except Exception as e:
            # Log error but don't fail initialization
            print(f"Warning: Failed to define types: {e}")

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
    name = 'SC62015:ROM'
    long_name = 'SC62015 ROM-only View'

    # Only the ROM is file-backed; internal RAM is virtual
    SEGMENTS: List[SegmentDef] = [
        SegmentDef(
            name='ROM',
            start=0xE0000,
            length=0x20000,
            file_offset=0,
            flags=(SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable),
            semantics=SectionSemantics.ReadOnlyCodeSectionSemantics,
        ),
        # SC62015 CPU internal RAM - not file-backed
        SegmentDef(
            name='Internal RAM',
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
        return buf == b"\x2A\x0A\x00\x00"


class SC62015FullView(SC62015BaseView):
    """
    View for full 1MB memory images, mapping ROM and all RAM/device regions.
    """
    name = 'SC62015:Memory'
    long_name = 'SC62015 Full 1MB Memory View'

    # Map entire memory image (1MB) file-backed for each region
    SEGMENTS: List[SegmentDef] = [
        # ROM region (file offset = virtual start)
        SegmentDef('ROM', 0xE0000, 0x20000, 0xE0000,
                   SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
                   SectionSemantics.ReadOnlyCodeSectionSemantics),

        # SC62015 CPU Internal RAM at 0x100000-0x1000FF (256 bytes)
        # Note: This is the CPU's built-in RAM, separate from external memory space
        SegmentDef('Internal RAM', INTERNAL_MEMORY_START, INTERNAL_MEMORY_LENGTH, INTERNAL_MEMORY_START,
                   SegmentFlag.SegmentReadable | SegmentFlag.SegmentWritable,
                   SectionSemantics.ReadWriteDataSectionSemantics),

        # SH26 device registers
        SegmentDef('SH26', SH26_ADDR_START, SH26_ADDR_END - SH26_ADDR_START, SH26_ADDR_START,
                   SegmentFlag.SegmentReadable | SegmentFlag.SegmentWritable,
                   SectionSemantics.ReadWriteDataSectionSemantics),

        # LH5073A1 registers
        SegmentDef('LH5073A1', LH5073A1_ADDR_START, LH5073A1_ADDR_END - LH5073A1_ADDR_START, LH5073A1_ADDR_START,
                   SegmentFlag.SegmentReadable | SegmentFlag.SegmentWritable,
                   SectionSemantics.ReadWriteDataSectionSemantics),

        # CE1 registers
        SegmentDef('CE1', CE1_ADDR_START, CE1_ADDR_END - CE1_ADDR_START, CE1_ADDR_START,
                   SegmentFlag.SegmentReadable | SegmentFlag.SegmentWritable,
                   SectionSemantics.ReadWriteDataSectionSemantics),

        # CE0 registers
        SegmentDef('CE0', CE0_ADDR_START, CE0_ADDR_END - CE0_ADDR_START, CE0_ADDR_START,
                   SegmentFlag.SegmentReadable | SegmentFlag.SegmentWritable,
                   SectionSemantics.ReadWriteDataSectionSemantics),

        # CE2 registers
        # TODO: Need to figure out the proper memory map - CE0/CE1/CE2 indexing doesn't make sense
        SegmentDef('CE2', 0xC0000, 0x20000, 0xC0000,
                   SegmentFlag.SegmentReadable | SegmentFlag.SegmentWritable | SegmentFlag.SegmentExecutable,
                   SectionSemantics.ReadWriteDataSectionSemantics),
    ]

    @classmethod
    def is_valid_for_data(cls, data) -> bool:
        buf = data.read(0, 4*3)
        return buf == b"\x01\x01\x01\x01\x01\x01\x01\x01\x06\x60\x00\x34" or buf == b"\x00\x00\x0C\x18\xF8\x0D\x40\x00\x24\x07\x00\x00"
