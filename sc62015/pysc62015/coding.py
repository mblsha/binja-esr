# based on https://github.com/whitequark/binja-avnera/blob/main/mc/coding.py
import struct
from typing import Callable


class BufferTooShort(Exception):
    pass


class Decoder:
    def __init__(self, buf: bytearray) -> None:
        self.buf, self.pos = buf, 0

    def get_pos(self) -> int:
        return self.pos

    def peek(self, offset: int) -> int:
        if len(self.buf) - self.pos <= offset:
            raise BufferTooShort

        return self.buf[self.pos + offset]

    def _unpack(self, fmt: str) -> int:
        size = struct.calcsize(fmt)
        if len(self.buf) - self.pos < size:
            raise BufferTooShort

        fmt = "<" + fmt if fmt[0] != ">" else fmt
        items = struct.unpack_from(fmt, self.buf, self.pos)
        self.pos += size
        if len(items) == 1:
            return items[0]  # type: ignore
        else:
            raise ValueError("Unpacking more than one item is not supported")
            return items

    def unsigned_byte(self) -> int:
        return self._unpack("B")

    def unsigned_word_le(self) -> int:
        return self._unpack("H")


# FetchDecoder is similar to Decoder but uses a read_mem function to fetch memory
class FetchDecoder(Decoder):
    MAX_ADDR = 0xFFFFF + 0xFF

    def __init__(self, read_mem: Callable[[int], int]) -> None:
        self.read_mem = read_mem
        self.pos = 0

    def get_pos(self) -> int:
        return self.pos

    def peek(self, offset: int) -> int:
        if self.pos + offset > self.MAX_ADDR:
            raise BufferTooShort
        return self.read_mem(self.pos + offset)

    def _unpack(self, fmt: str) -> int:
        size = struct.calcsize(fmt)
        if self.pos + size > self.MAX_ADDR:
            raise BufferTooShort

        fmt = "<" + fmt if fmt[0] != ">" else fmt
        items = struct.unpack_from(
            fmt, bytearray(self.read_mem(self.pos + i) for i in range(size))
        )
        self.pos += size
        if len(items) == 1:
            return items[0]  # type: ignore
        else:
            raise ValueError("Unpacking more than one item is not supported")
            return items


class Encoder:
    def __init__(self) -> None:
        self.buf = bytearray()

    def _pack(self, fmt: str, item: int) -> None:
        offset = len(self.buf)
        self.buf += b"\x00" * struct.calcsize(fmt)
        fmt = "<" + fmt if fmt[0] != ">" else fmt
        struct.pack_into(fmt, self.buf, offset, item)

    def unsigned_byte(self, value: int) -> None:
        self._pack("B", value)

    def unsigned_word_le(self, value: int) -> None:
        self._pack("H", value)
