from typing import Any, Callable, Union
from . import RegisterName, IntrinsicName

ExpressionIndex = int
ILRegister = Union[RegisterName, ExpressionIndex]
RegisterIndex = int
IntrinsicIndex = int
ILIntrinsic = Union[IntrinsicName, ExpressionIndex]

def LLIL_TEMP(n: int) -> ExpressionIndex:
    ...

class LowLevelILFunction:
    def expr(self, *args: Any, size: int | None, flags: Any | None = None) -> ExpressionIndex:
        ...

    def reg(self, size: int, reg: Union[RegisterName, ILRegister, RegisterIndex, str]) -> ExpressionIndex:
        ...
    
    def set_reg(self, size: int, reg: Union[RegisterName, ILRegister, RegisterIndex, str], value: Any) -> ExpressionIndex:
        ...
    
    def intrinsic(self, outputs: list[Any], name: Union[IntrinsicName, ILIntrinsic, IntrinsicIndex, str], inputs: list[Any]) -> ExpressionIndex:
        ...

    def __getattr__(self, name: str) -> Callable[..., ExpressionIndex]:
        ...

class LowLevelILLabel:
    ...

class ILSourceLocation:
    instr_index: int
