from typing import Any, Dict, List, Union
from . import RegisterInfo, RegisterName, FlagWriteTypeName

class Architecture(object):
    address_size: int
    name: str
    regs: Dict[RegisterName, RegisterInfo]
    stack_pointer: str
    flag_write_types: List[Union[FlagWriteTypeName, str]]
    standalone_platform: Any

    def __getitem__(self, name: str) -> "Architecture":
        ...
    
    @classmethod
    def __class_getitem__(cls, name: str) -> "Architecture":
        ...


class FlagName(str):
    ...
