from typing import Any, Callable, Union

ExpressionIndex = int

def LLIL_TEMP(n: int) -> ExpressionIndex:
    ...

class LowLevelILFunction:
    def expr(self, *args: Any, size: int | None, flags: Any | None = None) -> ExpressionIndex:
        ...

    def reg(self, size: int, reg: Union[str, Any]) -> ExpressionIndex:
        ...
    
    def set_reg(self, size: int, reg: Union[str, Any], value: Any) -> ExpressionIndex:
        ...
    
    def intrinsic(self, outputs: list[Any], name: Union[str, Any], inputs: list[Any]) -> ExpressionIndex:
        ...

    def __getattr__(self, name: str) -> Callable[..., ExpressionIndex]:
        ...

class LowLevelILLabel:
    ...

class ILSourceLocation:
    instr_index: int
