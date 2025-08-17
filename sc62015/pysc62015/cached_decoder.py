"""Cached FetchDecoder for improved performance."""

from typing import Callable
from binja_test_mocks.coding import Decoder, BufferTooShortErrorError


class CachedFetchDecoder(Decoder):
    """FetchDecoder with byte caching to reduce redundant memory reads."""
    
    def __init__(self, read_mem: Callable[[int], int], address_space_size: int):
        self.read_mem = read_mem
        self.address_space_size = address_space_size
        self.pos = 0
        # Cache for recently read bytes
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def peek(self, offset: int) -> int:
        """Peek at a byte without advancing position."""
        addr = self.pos + offset
        if addr >= self.address_space_size:
            raise BufferTooShortErrorError
        
        # Check cache first
        if addr in self._cache:
            self._cache_hits += 1
            return self._cache[addr]
        
        # Cache miss - read from memory
        self._cache_misses += 1
        value = self.read_mem(offset)
        self._cache[addr] = value
        
        # Limit cache size to prevent memory bloat
        if len(self._cache) > 32:
            # Keep the most recent 16 entries
            sorted_keys = sorted(self._cache.keys())
            for key in sorted_keys[:-16]:
                del self._cache[key]
        
        return value
    
    def unsigned_byte(self) -> int:
        """Read an unsigned byte and advance position."""
        if self.pos >= self.address_space_size:
            raise BufferTooShortErrorError
        
        # Try cache first
        if self.pos in self._cache:
            self._cache_hits += 1
            value = self._cache[self.pos]
        else:
            self._cache_misses += 1
            value = self.read_mem(0)
            self._cache[self.pos] = value
            
            # Limit cache size
            if len(self._cache) > 32:
                sorted_keys = sorted(self._cache.keys())
                for key in sorted_keys[:-16]:
                    del self._cache[key]
        
        self.pos += 1
        return value
    
    def advance(self, count: int) -> None:
        """Advance position by count bytes."""
        self.pos += count
        if self.pos > self.address_space_size:
            raise BufferTooShortErrorError
    
    def _unpack(self, fmt: str) -> tuple:
        """Unpack binary data (not used in SC62015)."""
        raise NotImplementedError("_unpack not implemented for CachedFetchDecoder")
    
    def get_cache_stats(self) -> tuple[int, int]:
        """Get cache hit/miss statistics."""
        return self._cache_hits, self._cache_misses
    
    def clear_cache(self):
        """Clear the byte cache."""
        self._cache.clear()