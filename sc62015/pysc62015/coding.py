from binja_helpers.coding import (
    Decoder as _Decoder,
    FetchDecoder as _FetchDecoder,
    Encoder as _Encoder,
    BufferTooShort as _BufferTooShort,
)
from .constants import ADDRESS_SPACE_SIZE  # noqa: F401 - re-export

# Re-export the generic helpers to maintain compatibility with existing imports
Decoder = _Decoder
FetchDecoder = _FetchDecoder
Encoder = _Encoder
BufferTooShort = _BufferTooShort
