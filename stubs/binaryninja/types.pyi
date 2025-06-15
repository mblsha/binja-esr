from typing import Any

class Symbol:
    type: Any
    addr: int
    name: str
    
    def __init__(self, symbol_type: Any, addr: int, name: str) -> None: ...
