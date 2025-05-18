from dataclasses import dataclass
from typing import Optional, List

from binaryninja.binaryview import BinaryView
from binaryninja.architecture import Architecture
from binaryninja.enums import SegmentFlag, SectionSemantics, SymbolType, Endianness
from binaryninja.types import Symbol

# Import architecture-specific constants
from .pysc62015.instr import (
    INTERNAL_MEMORY_START,
    INTERNAL_MEMORY_LENGTH,
    IMEM_NAMES,
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
        for name, offset in IMEM_NAMES.items():
            addr = INTERNAL_MEMORY_START + offset
            self.define_data_var(addr, 'uint8_t', name)

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

        return True

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

    # Only the ROM is file-backed
    SEGMENTS: List[SegmentDef] = [
        SegmentDef(
            name='ROM',
            start=0xE0000,
            length=0x20000,
            file_offset=0,
            flags=(SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable),
            semantics=SectionSemantics.ReadOnlyCodeSectionSemantics,
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

        # Internal RAM (file contains RAM contents)
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
    ]

    @classmethod
    def is_valid_for_data(cls, data) -> bool:
        buf = data.read(0, 4*3)
        return buf == b"\x01\x01\x01\x01\x01\x01\x01\x01\x06\x60\x00\x34"
