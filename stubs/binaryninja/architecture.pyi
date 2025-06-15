from typing import Any, Dict, List
from . import RegisterInfo, RegisterName, FlagWriteTypeName

class Architecture:
    name: str | None
    regs: Dict[RegisterName, RegisterInfo]
    stack_pointer: str | None
    flag_write_types: List[FlagWriteTypeName]
    standalone_platform: Any

    def __getitem__(self, name: str) -> "Architecture":
        ...
    
    @classmethod
    def __class_getitem__(cls, name: str) -> "Architecture":
        ...


class FlagName(str):
    ...
