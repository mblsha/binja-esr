from .binja_api import *

from binaryninja.enums import BranchType

class MockAnalysisInfo:
    def __init__(self):
        self.length = 0
        self.branches = []

    def add_branch(self, branch_type, addr=None):
        self.branches.append((branch_type, addr))

