"""Passive RAM media backing; device profiles own capacity and bus mapping."""


class RamMedia:
    def __init__(self, image, capacity, writable=True):
        if capacity <= 0 or len(image) != capacity:
            raise ValueError(f"RAM medium requires an exact {capacity}-byte image")
        self.bytes = bytearray(image)
        self.writable = bool(writable)

    def read(self, offset):
        return self.bytes[offset] if 0 <= offset < len(self.bytes) else None

    def write(self, offset, value):
        if not 0 <= offset < len(self.bytes):
            return False
        if self.writable:
            self.bytes[offset] = value
        return True

    def image(self):
        return bytes(self.bytes)
