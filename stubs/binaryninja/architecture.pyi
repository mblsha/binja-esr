from typing import Any, Dict, List

class Architecture:
    name: str
    regs: Dict[str, Any]
    stack_pointer: str
    flag_write_types: List[str]
    standalone_platform: Any

    def __getitem__(self, name: str) -> "Architecture":
        ...
    
    @classmethod
    def __class_getitem__(cls, name: str) -> "Architecture":
        ...


class RegisterName(str):
    ...


class IntrinsicName(str):
    ...


class FlagName(str):
    ...
