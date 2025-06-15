from typing import Any, Callable

class ArchitectureMeta(type):
    def __getitem__(cls, name: str) -> "Architecture":
        ...

class Architecture(metaclass=ArchitectureMeta):
    name: str
    regs: dict[str, Any]
    stack_pointer: str
    flag_write_types: list[str]
    standalone_platform: Any

class BinaryView:
    file: Any
    end: int
    
    def read(self, addr: int, length: int) -> bytes: ...
    def add_auto_segment(self, start: int, length: int, data_offset: int = 0, data_length: int = 0, flags: Any = None) -> Any: ...
    def add_user_section(self, name: str, start: int, length: int, semantics: Any = None) -> Any: ...
    def parse_type_string(self, type_string: str) -> tuple[Any, ...]: ...
    def define_user_symbol(self, symbol: Any) -> None: ...
    def define_user_data_var(self, addr: int, var_type: Any) -> None: ...
    def get_function_at(self, addr: int) -> Any | None: ...
    def create_user_function(self, addr: int) -> Any | None: ...
    def __init__(self, parent_view: "BinaryView | None" = None, file_metadata: Any = None) -> None: ...

InstructionInfo: Any
RegisterInfo: Any
IntrinsicInfo: Any
CallingConvention: Any
InstructionTextToken: Any
UIContext: Any

log_error: Callable[[str], None]
