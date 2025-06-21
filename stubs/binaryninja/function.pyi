"""Type stubs for binaryninja.function module."""

from typing import Any
from binaryninja.architecture import Architecture

class Function:
    """Binary Ninja Function class."""
    
    @property
    def arch(self) -> Architecture:
        """The architecture of this function."""
        ...
    
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...