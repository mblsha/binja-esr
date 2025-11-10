from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence


@dataclass
class PropState:
    """Structured initial state for property tests."""

    regs: Dict[str, int]
    flags: Dict[str, int]
    internal: Dict[int, int]
    external: Dict[int, int]

    def clone(self) -> "PropState":
        return PropState(
            regs=dict(self.regs),
            flags=dict(self.flags),
            internal=dict(self.internal),
            external=dict(self.external),
        )


@dataclass
class Scenario:
    """Describes a generated instruction scenario."""

    bytes_seq: Sequence[bytes]
    family: str
    info: Dict[str, int]
    description: str = ""
    expect_pre_sequence: bool = False

    def flattened(self) -> bytes:
        return b"".join(self.bytes_seq)


@dataclass
class ExecutionResult:
    regs: Dict[str, int]
    flags: Dict[str, int]
    internal: Dict[int, int]
    external: Dict[int, int]
    mem_log: List[Dict[str, int]] = field(default_factory=list)
