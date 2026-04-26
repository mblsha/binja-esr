"""Deterministic memory-card and RAM-disk models for IOCS edge tests."""

from __future__ import annotations

from dataclasses import dataclass, field


class StorageError(RuntimeError):
    """Base error for deterministic PC-E500 storage models."""


class DuplicateBlockError(StorageError):
    """Raised when creating or renaming to an existing block name."""


class BlockNotFoundError(StorageError):
    """Raised when a named block does not exist."""


class InsufficientSpaceError(StorageError):
    """Raised when an operation would exceed card capacity."""


def _normalise_name(name: str) -> str:
    value = name.strip().upper()
    if not value:
        raise ValueError("block name must not be empty")
    if len(value) > 11:
        raise ValueError("block names are limited to 11 characters")
    return value


@dataclass
class MemoryCardBlock:
    """Single S1:S2:S3 memory block."""

    name: str
    data: bytearray
    protected: bool = False

    @property
    def size(self) -> int:
        return len(self.data)


@dataclass
class MemoryCardImage:
    """Ordered memory-card block image with E500-like edge semantics."""

    capacity: int
    blocks: list[MemoryCardBlock] = field(default_factory=list)

    @property
    def used(self) -> int:
        return sum(block.size for block in self.blocks)

    @property
    def free(self) -> int:
        return self.capacity - self.used

    def create(self, name: str, size: int, *, at_top: bool = False) -> MemoryCardBlock:
        normalised = _normalise_name(name)
        if size < 0:
            raise ValueError("block size must be non-negative")
        if self.find(normalised, missing_ok=True) is not None:
            raise DuplicateBlockError(normalised)
        if size > self.free:
            raise InsufficientSpaceError(normalised)
        block = MemoryCardBlock(normalised, bytearray(size))
        if at_top:
            self.blocks.insert(0, block)
        else:
            self.blocks.append(block)
        return block

    def find(self, name: str, *, missing_ok: bool = False) -> MemoryCardBlock | None:
        normalised = _normalise_name(name)
        for block in self.blocks:
            if block.name == normalised:
                return block
        if missing_ok:
            return None
        raise BlockNotFoundError(normalised)

    def resize(self, name: str, size: int) -> MemoryCardBlock:
        block = self.find(name)
        assert block is not None
        if size < 0:
            raise ValueError("block size must be non-negative")
        delta = size - block.size
        if delta > self.free:
            raise InsufficientSpaceError(block.name)
        if delta > 0:
            block.data.extend(b"\x00" * delta)
        elif delta < 0:
            del block.data[size:]
        return block

    def rename(self, old_name: str, new_name: str) -> MemoryCardBlock:
        block = self.find(old_name)
        assert block is not None
        normalised = _normalise_name(new_name)
        existing = self.find(normalised, missing_ok=True)
        if existing is not None and existing is not block:
            raise DuplicateBlockError(normalised)
        block.name = normalised
        return block

    def delete(self, name: str) -> None:
        block = self.find(name)
        assert block is not None
        self.blocks.remove(block)

    def condense(self) -> None:
        """Keep deterministic block order while removing zero-sized gaps."""

        self.blocks = [block for block in self.blocks if block.size > 0]


@dataclass
class RamDiskImage:
    """E:F:G RAM disk backed by the memory-card RAMFILE block."""

    card: MemoryCardImage
    backing_name: str = "RAMFILE"

    def format(self, size: int) -> MemoryCardBlock:
        block = self.card.find(self.backing_name, missing_ok=True)
        if block is None:
            return self.card.create(self.backing_name, size, at_top=True)
        return self.card.resize(self.backing_name, size)

    @property
    def size(self) -> int:
        block = self.card.find(self.backing_name, missing_ok=True)
        return 0 if block is None else block.size
