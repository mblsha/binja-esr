from typing import Any, Callable, NewType, ClassVar, Optional, Dict, List

# Type aliases for register and intrinsic names - these need to be distinct types
RegisterName = NewType('RegisterName', str)
IntrinsicName = NewType('IntrinsicName', str) 
FlagWriteTypeName = NewType('FlagWriteTypeName', str)
ILRegister = Any
RegisterIndex = Any
ILIntrinsic = Any
IntrinsicIndex = Any

class RegisterInfo:
    def __init__(self, name: RegisterName, size: int, offset: int = 0, extend: Any = None) -> None: ...
    name: str
    size: int
    offset: int

class IntrinsicInfo:
    def __init__(self, inputs: list[Any], outputs: list[Any]) -> None: ...
    inputs: list[Any]
    outputs: list[Any]

class Architecture:
    name: ClassVar[Optional[str]] = None
    endianness: Any = None
    address_size: int = 8
    default_int_size: int = 4
    instr_alignment: int = 1
    max_instr_length: int = 16
    opcode_display_length: int = 8
    regs: ClassVar[Dict[RegisterName, RegisterInfo]] = {}
    stack_pointer: ClassVar[Optional[str]] = None
    link_reg: str | None = None
    global_regs: list[str] = []
    system_regs: list[str] = []
    flags: list[str] = []
    flag_write_types: ClassVar[List[FlagWriteTypeName]] = []
    flag_roles: dict[str, Any] = {}
    flags_written_by_flag_write_type: dict[str, list[str]] = {}
    intrinsics: dict[str, IntrinsicInfo] = {}
    standalone_platform: Any = None
    
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
