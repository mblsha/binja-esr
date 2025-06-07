from typing import Any

ExpressionIndex = int

class LowLevelILFunction:
    def expr(self, *args: Any, size: int | None, flags: Any | None = None) -> Any:
        ...

class LowLevelILLabel:
    ...

class ILSourceLocation:
    instr_index: int
