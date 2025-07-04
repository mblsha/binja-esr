from typing import Any

class Symbol:
    type: Any
    addr: int
    name: str
    
    def __init__(self, symbol_type: Any, addr: int, name: str) -> None: ...

class Type:
    @staticmethod
    def array(element_type: 'Type', count: int) -> 'Type': ...
    
    @staticmethod
    def int(width: int, sign: bool = True, altname: str = "") -> 'Type': ...
