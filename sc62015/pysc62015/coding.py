# based on https://github.com/whitequark/binja-avnera/blob/main/mc/coding.py
import struct


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
            return items[0] # type: ignore
        else:
            raise ValueError("Unpacking more than one item is not supported")
            return items

    def unsigned_byte(self) -> int:
        return self._unpack("B")

    def signed_byte(self) -> int:
        return self._unpack("b")

    def unsigned_word_le(self) -> int:
        return self._unpack("H")

    def unsigned_word_be(self) -> int:
        return self._unpack(">H")

    def signed_word_le(self) -> int:
        return self._unpack("h")

    def signed_word_be(self) -> int:
        return self._unpack(">h")


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

    def signed_byte(self, value: int) -> None:
        self._pack("b", value)

    def unsigned_word_le(self, value: int) -> None:
        self._pack("H", value)

    def unsigned_word_be(self, value: int) -> None:
        self._pack(">H", value)

    def signed_word_le(self, value: int) -> None:
        self._pack("h", value)

    def signed_word_be(self, value: int) -> None:
        self._pack(">h", value)
