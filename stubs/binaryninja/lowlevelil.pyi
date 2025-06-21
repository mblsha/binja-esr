from typing import Any, Callable, Union
from . import RegisterName, ILRegister, RegisterIndex, IntrinsicName, ILIntrinsic, IntrinsicIndex
from .function import Function
from .architecture import Architecture

ExpressionIndex = int

def LLIL_TEMP(n: int) -> ExpressionIndex:
    ...

class LowLevelILFunction:
    @property
    def source_function(self) -> Function:
        """The source function that this LLIL belongs to."""
        ...
    
    @property 
    def arch(self) -> Architecture:
        """The architecture for this LLIL function."""
        ...
    
    def expr(self, *args: Any, size: int | None, flags: Any | None = None) -> ExpressionIndex:
        ...

    def reg(self, size: int, reg: Union[RegisterName, ILRegister, RegisterIndex]) -> ExpressionIndex:
        ...
    
    def set_reg(self, size: int, reg: Union[RegisterName, ILRegister, RegisterIndex], value: Any) -> ExpressionIndex:
        ...
    
    def intrinsic(self, outputs: list[Any], name: Union[IntrinsicName, ILIntrinsic, IntrinsicIndex], inputs: list[Any]) -> ExpressionIndex:
        ...

    def __getattr__(self, name: str) -> Callable[..., ExpressionIndex]:
        ...

class LowLevelILLabel:
    ...

class ILSourceLocation:
    instr_index: int
