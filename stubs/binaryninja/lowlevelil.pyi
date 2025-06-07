from typing import Any, Callable

ExpressionIndex = int

def LLIL_TEMP(n: int) -> ExpressionIndex:
    ...

class LowLevelILFunction:
    def expr(self, *args: Any, size: int | None, flags: Any | None = None) -> Any:
        ...

    def __getattr__(self, name: str) -> Callable[..., Any]:
        ...

class LowLevelILLabel:
    ...

class ILSourceLocation:
    instr_index: int
