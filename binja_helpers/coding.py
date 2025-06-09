# based on https://github.com/whitequark/binja-avnera/blob/main/mc/coding.py
"""Binary decoding helpers used by multiple modules."""

import struct
from typing import Callable


class BufferTooShort(Exception):
    """Raised when attempting to read past the end of the buffer."""


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
        raise ValueError("Unpacking more than one item is not supported")

    def unsigned_byte(self) -> int:
        return self._unpack("B")

    def unsigned_word_le(self) -> int:
        return self._unpack("H")


class FetchDecoder(Decoder):
    """Decoder that fetches bytes using a callable instead of a buffer.

    This class is used by the SC62015 emulator which exposes memory through a
    callback.  The normal :class:`Decoder` expects a ``bytearray`` and therefore
    cannot operate on the emulator's ``Memory`` object.  ``address_space_size``
    is the total addressable memory and allows :class:`BufferTooShort` to be
    raised when an instruction would read past the end of memory, mirroring the
    behaviour of the real hardware where addresses wrap at the address space
    boundary.
    """

    def __init__(
        self,
        read_mem: Callable[[int], int],
        address_space_size: int,
    ) -> None:
        self.read_mem = read_mem
        self.address_space_size = address_space_size
        self.pos = 0

    def get_pos(self) -> int:
        return self.pos

    def peek(self, offset: int) -> int:
        if self.pos + offset >= self.address_space_size:
            raise BufferTooShort
        return self.read_mem(self.pos + offset)

    def _unpack(self, fmt: str) -> int:
        size = struct.calcsize(fmt)
        if self.pos + size > self.address_space_size:
            raise BufferTooShort
        fmt = "<" + fmt if fmt[0] != ">" else fmt
        items = struct.unpack_from(
            fmt, bytearray(self.read_mem(self.pos + i) for i in range(size))
        )
        self.pos += size
        if len(items) == 1:
            return items[0]  # type: ignore
        raise ValueError("Unpacking more than one item is not supported")


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
