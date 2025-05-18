from .binja_api import *

from binaryninja.enums import BranchType
from typing import List, Optional, Tuple

class MockAnalysisInfo:
    def __init__(self) -> None:
        self.length = 0
        self.branches: List[Tuple[BranchType, int]] = []

    def add_branch(self, branch_type: BranchType, addr: Optional[int] = None) -> None:
        self.branches.append((branch_type, addr))

