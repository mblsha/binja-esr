from binja_helpers.coding import (
    Decoder as _Decoder,
    FetchDecoder as _FetchDecoder,
    Encoder as _Encoder,
    BufferTooShort as _BufferTooShort,
    ADDRESS_SPACE_SIZE as ADDRESS_SPACE_SIZE,
)

# Re-export the generic helpers to maintain compatibility with existing imports
Decoder = _Decoder
FetchDecoder = _FetchDecoder
Encoder = _Encoder
BufferTooShort = _BufferTooShort
