"""Strict ESR-H startup component, mirrored by Rust pa9500/compat_probe.rs.

The caller owns fetch/data/WRIT decoding. Scalar WRIT and 7B far transfer are
explicit PA compatibility hypotheses; no general ESR-H machine is claimed.
"""

from dataclasses import dataclass, field
from copy import deepcopy


@dataclass
class StartupState:
    pc: int
    dp: int = 0
    p: int = 0
    internal: bytearray = field(default_factory=lambda: bytearray(128))
    carry: bool = False
    zero: bool = False


def step(state, bus):
    """One component boundary; return a native target or None to continue."""
    candidate = deepcopy(state)

    def byte():
        value = bus.fetch(candidate.pc)
        if not 0 <= value <= 255:
            raise ValueError("fetch did not return a byte")
        candidate.pc = (candidate.pc + 1) & 65535
        return value

    opcode = byte()
    transfer, output = None, None
    if 0 <= opcode <= 3:
        candidate.internal[opcode] = byte()
    elif opcode == 0x10:
        candidate.dp = (byte() << 8) | byte()
    elif opcode == 0x4D:
        pass
    elif opcode == 0x57:
        candidate.internal[2] = bus.read(candidate.dp)
    elif opcode == 0x64:
        candidate.internal[2] &= byte()
        candidate.zero = candidate.internal[2] == 0
    elif opcode == 0x67:
        value = byte()
        candidate.carry = candidate.internal[2] < value
        candidate.zero = candidate.internal[2] == value
    elif opcode == 0x7C:
        target = (byte() << 8) | byte()
        if not candidate.zero:
            candidate.pc = target
    elif 0x80 <= opcode <= 0xBF:
        candidate.p = opcode & 63
    elif opcode == 0xD3:
        output = candidate.dp, candidate.p, candidate.internal[candidate.p]
    elif opcode == 0x7B:
        high, middle, low = byte(), byte(), byte()
        transfer = (high << 16) | (middle << 8) | low
        if high > 15 or transfer < 65536:
            raise ValueError(
                f"invalid PA compatibility native transfer at {state.pc:04X}"
            )
    else:
        raise ValueError(
            f"unqualified ESR-H startup opcode {opcode:02X} at {state.pc:04X}"
        )
    if output is not None:
        bus.writ(*output)
    state.__dict__.update(candidate.__dict__)
    return transfer
