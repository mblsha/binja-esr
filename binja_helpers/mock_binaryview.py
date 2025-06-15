"""Generic Mock BinaryView implementation for testing."""

from typing import Dict


class MockFile:
    """Mock file object for BinaryView."""

    def __init__(self, filename: str = "test.bin"):
        self.filename = filename


class MockBinaryView:
    """Generic mock BinaryView for testing."""

    def __init__(self, filename: str = "test.bin"):
        self.file = MockFile(filename)
        # Memory buffer for read operations
        self._memory: Dict[int, bytes] = {}
        self._default_memory = bytearray(0x100000)  # 1MB default buffer

    def read(self, addr: int, length: int) -> bytes:
        """Read bytes from the mock memory."""
        # Check if we have specific data for this address
        if addr in self._memory:
            data = self._memory[addr]
            if len(data) >= length:
                return data[:length]

        # Fall back to default memory buffer
        if addr + length <= len(self._default_memory):
            return bytes(self._default_memory[addr:addr+length])

        # Return zeros for out-of-bounds reads
        return b'\x00' * length

    def write_memory(self, addr: int, data: bytes) -> None:
        """Write data to specific memory address for testing."""
        self._memory[addr] = data

        # Also update default memory buffer if within range
        if addr + len(data) <= len(self._default_memory):
            self._default_memory[addr:addr+len(data)] = data

    def set_memory_region(self, start_addr: int, data: bytes) -> None:
        """Set a region of memory for testing."""
        self.write_memory(start_addr, data)
