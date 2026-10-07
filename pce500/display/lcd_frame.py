"""Owned logical LCD pixels; mirrors sc62015/core/src/lcd_frame.rs."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LcdFrame:
    """Row-major zero/one pixels with device-supplied geometry."""

    cols: int
    rows: int
    pixels: bytes

    def __post_init__(self) -> None:
        if (
            self.cols <= 0
            or self.rows <= 0
            or self.cols * self.rows != len(self.pixels)
        ):
            raise ValueError(
                "LCD frame requires nonzero geometry matching its pixel count"
            )
        if any(pixel > 1 for pixel in self.pixels):
            raise ValueError("LCD frame pixels must be zero or one")
        # Own the input, including when the caller supplies a mutable bytearray.
        object.__setattr__(self, "pixels", bytes(self.pixels))

    def to_rows(self) -> list[list[int]]:
        return [
            list(self.pixels[start : start + self.cols])
            for start in range(0, len(self.pixels), self.cols)
        ]

    def pbm(self) -> bytes:
        """Pack separate rows MSB-first, with clear unused row-padding bits."""
        stride = (self.cols + 7) // 8
        output = bytearray(stride * self.rows)
        for index, pixel in enumerate(self.pixels):
            y, x = divmod(index, self.cols)
            output[y * stride + x // 8] |= pixel << (7 - x % 8)
        return f"P4\n{self.cols} {self.rows}\n".encode("ascii") + output
