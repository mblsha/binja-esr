from . import binja_api # noqa: F401

from binaryninja.enums import BranchType
from binaryninja import InstructionInfo, Architecture
from typing import List, Optional, Tuple

class MockAnalysisInfo(InstructionInfo):
    def __init__(self) -> None:
        self.length = 0
        self.mybranches: List[Tuple[BranchType, Optional[int]]] = []

    def add_branch(self, branch_type: BranchType, target: Optional[int] = None,
                   arch: Optional[Architecture] = None) -> None:
        self.mybranches.append((branch_type, target))

