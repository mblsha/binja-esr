from typing import Protocol, runtime_checkable


@runtime_checkable
class HasWidth(Protocol):
    def width(self) -> int: ...
