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
    q: int = 0
    r: int = 0
    table_count: int = 0
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

    def push(value):
        candidate.r = (candidate.r - 1) & 127
        candidate.internal[candidate.r] = value

    def pop():
        value = candidate.internal[candidate.r]
        candidate.r = (candidate.r + 1) & 127
        return value

    opcode = byte()
    transfer, output, data_output, f_output, fill_output = None, None, None, None, None
    if 0 <= opcode <= 3:
        candidate.internal[opcode] = byte()
    elif opcode == 0x10:
        candidate.dp = (byte() << 8) | byte()
    elif opcode == 0x11:
        candidate.dp = (candidate.dp & 0xFF00) | byte()
    elif opcode in (4, 5, 6, 7, 0x24, 0x25, 0x26, 0x27):
        reg = 4 if opcode in (4, 5, 0x24, 0x25) else 6
        old = candidate.internal[reg] | (candidate.internal[reg + 1] << 8)
        value = (old + (1 if opcode & 1 == 0 else -1)) & 65535
        candidate.internal[reg : reg + 2] = value.to_bytes(2, "little")
        candidate.dp, candidate.q = value, reg + 1
        if opcode in (0x24, 0x25):
            candidate.internal[2] = bus.read(value)
        if opcode in (0x26, 0x27):
            data_output = value, candidate.internal[2]
    elif opcode in (0x12, 0x13):
        value = byte() & 127
        if opcode == 0x12:
            candidate.p = value
        else:
            candidate.q = value
    elif 0x28 <= opcode <= 0x2D or 0x38 <= opcode <= 0x3B:
        displacement = byte()
        condition = {
            0x28: not candidate.zero,
            0x29: not candidate.zero,
            0x2A: not candidate.carry,
            0x2B: not candidate.carry,
            0x38: candidate.zero,
            0x39: candidate.zero,
            0x3A: candidate.carry,
            0x3B: candidate.carry,
        }.get(opcode, True)
        if condition:
            candidate.pc = (
                state.pc + 1 + (displacement if opcode & 1 == 0 else -displacement)
            ) & 65535
    elif opcode == 0x1F:
        count = candidate.internal[0] + 1
        fill_output = candidate.dp, count, candidate.internal[2]
        candidate.dp = (candidate.dp + count - 1) & 65535
    elif 0x08 <= opcode <= 0x0B:
        count = candidate.internal[0 if opcode < 0x0A else 1] + 1
        if candidate.p + count > 127 or candidate.q + count > 127:
            raise ValueError("unqualified legacy internal block pointer range")
        for _ in range(count):
            p, q = candidate.p, candidate.q
            if opcode & 1:
                candidate.internal[p], candidate.internal[q] = (
                    candidate.internal[q],
                    candidate.internal[p],
                )
            else:
                candidate.internal[p] = candidate.internal[q]
            candidate.p += 1
            candidate.q += 1
    elif 0x20 <= opcode <= 0x22:
        candidate.internal[2] = (candidate.p, candidate.q, candidate.r)[opcode - 0x20]
    elif opcode in (
        0x40,
        0x41,
        0x42,
        0x43,
        0x48,
        0x49,
        0x4A,
        0x4B,
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC8,
        0xC9,
        0xCA,
        0xCB,
    ):
        reg = {
            0x40: 0,
            0x42: 2,
            0x48: 8,
            0x4A: 10,
            0xC0: 1,
            0xC2: 3,
            0xC8: 9,
            0xCA: 11,
        }[opcode & 0xFE]
        old = candidate.internal[reg]
        candidate.internal[reg] = (old + (1 if opcode & 1 == 0 else -1)) & 255
        candidate.q = reg
        candidate.zero = candidate.internal[reg] == 0
        candidate.carry = old == (255 if opcode & 1 == 0 else 0)
    elif 0x30 <= opcode <= 0x32:
        value = candidate.internal[2] & 127
        if opcode == 0x30:
            candidate.p = value
        elif opcode == 0x31:
            candidate.q = value
        else:
            candidate.r = value
    elif opcode == 0x34:
        push(candidate.internal[2])
    elif opcode == 0x37:
        low = pop()
        candidate.pc = (pop() << 8) | low
    elif opcode in (0x4D, 0xCE):
        pass
    elif opcode == 0x55:
        candidate.internal[candidate.p] = bus.read(candidate.dp)
    elif opcode in (0x50, 0x51):
        p = candidate.p + (1 if opcode == 0x50 else -1)
        if not 0 <= p < 128:
            raise ValueError("unqualified legacy internal pointer boundary")
        candidate.p = p
    elif opcode == 0x58:
        a = candidate.internal[2]
        candidate.internal[2] = ((a << 4) | (a >> 4)) & 255
    elif opcode in (0x62, 0x66):
        mask = byte()
        candidate.zero = (
            candidate.internal[candidate.p if opcode == 0x62 else 2] & mask == 0
        )
    elif opcode == 0x65:
        candidate.internal[2] |= byte()
        candidate.zero = candidate.internal[2] == 0
    elif opcode == 0x57:
        candidate.internal[2] = bus.read(candidate.dp)
    elif opcode in (0x52, 0x53):
        data_output = (
            candidate.dp,
            candidate.internal[2 if opcode == 0x52 else candidate.p],
        )
    elif opcode == 0x59:
        candidate.internal[2] = candidate.internal[candidate.p]
    elif opcode == 0x5B:
        candidate.internal[2] = pop()
    elif opcode == 0x5F:
        f_output = candidate.internal[0x5E]
        candidate.q = 0x5E
    elif opcode in (0x60, 0x61):
        value = byte()
        slot = candidate.p
        candidate.internal[slot] = (
            candidate.internal[slot] & value
            if opcode == 0x60
            else candidate.internal[slot] | value
        )
        candidate.zero = candidate.internal[slot] == 0
    elif opcode == 0x63:
        value = byte()
        candidate.carry = candidate.internal[candidate.p] < value
        candidate.zero = candidate.internal[candidate.p] == value
    elif opcode == 0x64:
        candidate.internal[2] &= byte()
        candidate.zero = candidate.internal[2] == 0
    elif opcode == 0x67:
        value = byte()
        candidate.carry = candidate.internal[2] < value
        candidate.zero = candidate.internal[2] == value
    elif opcode == 0x69:
        selected = None
        for _ in range(candidate.table_count):
            test, target = byte(), (byte() << 8) | byte()
            candidate.zero = candidate.internal[2] == test
            if candidate.zero:
                selected = target
                break
        candidate.pc = selected if selected is not None else (byte() << 8) | byte()
    elif opcode in (0x78, 0x79, 0x7C, 0x7D, 0x7E, 0x7F):
        target = (byte() << 8) | byte()
        if opcode == 0x78:
            push(candidate.pc >> 8)
            push(candidate.pc & 255)
        condition = {
            0x7C: not candidate.zero,
            0x7D: not candidate.carry,
            0x7E: candidate.zero,
            0x7F: candidate.carry,
        }.get(opcode, True)
        if condition:
            candidate.pc = target
    elif opcode == 0x7A:
        candidate.table_count = byte()
        high, low = byte(), byte()
        push(high)
        push(low)
    elif 0x80 <= opcode <= 0xBF:
        candidate.p = opcode & 63
    elif opcode == 0xD3:
        output = candidate.dp, candidate.p, candidate.internal[candidate.p]
    elif opcode in (0xD4, 0xD5, 0xD6):
        mask = byte()
        old = bus.read(candidate.dp)
        value = old | mask if opcode == 0xD5 else old & mask
        candidate.zero = value == 0
        if opcode != 0xD6:
            data_output = candidate.dp, value
    elif opcode == 0xDA:
        candidate.internal[2], candidate.internal[3] = (
            candidate.internal[3],
            candidate.internal[2],
        )
    elif opcode == 0xDB:
        slot = candidate.p
        candidate.internal[2], candidate.internal[slot] = (
            candidate.internal[slot],
            candidate.internal[2],
        )
    elif 0xE0 <= opcode <= 0xFF:
        target = ((opcode & 31) << 8) | byte()
        push(candidate.pc >> 8)
        push(candidate.pc & 255)
        candidate.pc = target
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
    if data_output is not None:
        bus.write(*data_output)
    if f_output is not None:
        bus.outf(f_output)
    if fill_output is not None:
        bus.fill(*fill_output)
    state.__dict__.update(candidate.__dict__)
    return transfer
