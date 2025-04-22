from struct import unpack
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Tuple, Dict

from binaryninja.binaryview import BinaryView
from binaryninja.architecture import Architecture, IntrinsicInfo
from binaryninja.types import Symbol, Type
from binaryninja.enums import SegmentFlag, SymbolType
from binaryninja.enums import SymbolType, SegmentFlag, SectionSemantics, Endianness

from .pysc62015.instr import INTERNAL_MEMORY_START, IMEM_NAMES


@dataclass
class Segment:
    name: str
    start: int
    length: int
    data_offset: Optional[int]
    data_length: Optional[int]
    flags: SegmentFlag
    semantics: SectionSemantics


class SC62015View(BinaryView):
    name = "SC62015"
    long_name = "SC62015 ROM"

    @classmethod
    def is_valid_for_data(self, data):
        filename = data.file.filename
        buf = data.read(0, 4)
        if len(buf) < 4:
            return False
        # 2A0A0000
        result = buf[:4] == b"\x2A\x0A\x00\x00"
        return result

    def __init__(self, data):
        # data is a binaryninja.binaryview.BinaryView
        BinaryView.__init__(self, parent_view=data, file_metadata=data.file)
        self.repro_crash_on_save = False
        self.data = data
        self._interrupt_vector = None
        self._entry_point = None

    def init(self):
        arch_name = "SC62015"
        self.arch = Architecture[arch_name]
        self.platform = Architecture[arch_name].standalone_platform

        segments = []
        segments.append(
            Segment(
                f"ROM",
                0xE0000,
                0x20000,
                0,
                None,
                SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable,
                SectionSemantics.ReadOnlyCodeSectionSemantics,
            )
        )

        segments.append(
            Segment(
                f"Internal RAM",
                INTERNAL_MEMORY_START,
                0xFF,
                None,
                0,
                SegmentFlag.SegmentReadable | SegmentFlag.SegmentWritable,
                SectionSemantics.ReadWriteDataSectionSemantics,
            )
        )

        for s in segments:
            data_offset = s.data_offset if s.data_offset is not None else 0
            data_length = s.data_length if s.data_length is not None else s.length
            self.add_auto_segment(s.start, s.length, data_offset, data_length, s.flags)
            self.add_auto_section(s.name, s.start, s.length, s.semantics)

        for addr, name in IMEM_NAMES.items():
            full_addr = INTERNAL_MEMORY_START + addr
            self.define_data_var(full_addr, f"uint8_t", name)


        self._interrupt_vector = self.read_int(0xFFFFA, 3)
        self._entry_point = self.read_int(0xFFFFD, 3)

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

        return True

    def perform_get_address_size(self) -> int:
        return 3

    def perform_get_default_endianness(self) -> Endianness:
        return Endianness.LittleEndian

    def perform_get_entry_point(self):
        return self._entry_point
