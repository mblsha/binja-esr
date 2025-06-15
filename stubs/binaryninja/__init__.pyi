from typing import Any, Callable, Union

# Type aliases for register and intrinsic names
RegisterName = str
IntrinsicName = str 
FlagWriteTypeName = str

class RegisterInfo:
    def __init__(self, name: Union[RegisterName, str], size: int, offset: int = 0, extend: Any = None) -> None: ...
    name: RegisterName
    size: int
    offset: int

class IntrinsicInfo:
    def __init__(self, inputs: list[Any], outputs: list[Any]) -> None: ...
    inputs: list[Any]
    outputs: list[Any]

class Architecture:
    name: str
    regs: dict[Union[RegisterName, str], RegisterInfo]
    stack_pointer: str
    flag_write_types: list[Union[FlagWriteTypeName, str]]
    standalone_platform: Any
    
    # Workaround for mypy not understanding __getitem__ on metaclass
    @classmethod
    def __class_getitem__(cls, name: str) -> "Architecture": ...

# Unfortunately mypy has issues with dynamic indexing, so we need this workaround
def __getattr__(name: str) -> Any: ...

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
CallingConvention: Any
InstructionTextToken: Any
UIContext: Any

log_error: Callable[[str], None]
