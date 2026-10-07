"""Deterministic cross-language bus/controller corpus, independent of ROM files."""

import hashlib
import json
import random
from .hardware import Hardware
from .rtc import Clock


def actions():
    rng = random.Random(9600)
    ops = [
        ["write", 0x4021, 0xF0],
        ["peek", 0xC0000],
        ["read", 0xDFFFF],
        ["write", 0xC0000, 255],
        ["write", 0x4021, 0xF1],
        ["read", 0xC0000],
        ["write", 0x4021, 7],
        ["peek", 0xC0000],
        ["read", 0xC0000],
        ["clear_fault"],
    ]
    # Full-area bypass, mask and valid stream increments.
    ops += [
        ["write", 0x843F, 2],
        ["write", 0x8423, 255],
        ["write", 0x8424, 0x90],
        ["write", 0x8425, 0x90],
    ]
    for mode in range(8):
        for vertical in (0, 128):
            ops += [["write", 0x8428, mode | vertical]]
            for register, value in (
                (16, 0),
                (17, 0),
                (18, mode * 16),
                (19, 0),
                (20, 0),
                (21, 0),
                (22, mode * 16),
                (23, 0),
            ):
                ops.append(["write", 0x8420 + register, value])
            for _ in range(24):
                ops += [
                    ["write", 0x8422, rng.randrange(256)],
                    ["peek", 0x8402],
                    ["read", 0x8402],
                ]
    # Final backing row/column, clipping, signed coordinates, window banks,
    # block fill/copy, mode inversion and invalid increments before mutation.
    for window in range(16):
        ops += [["write", 0x8429, window]]
        for n in range(8):
            ops += [["write", 0x8438 + n, rng.randrange(256)]]
        ops += [["peek", 0x8418 + n] for n in range(8)]
    ops += [
        ["write", 0x8429, 0],
        ["write", 0x843F, 2],
        ["write", 0x8428, 0],
        ["write", 0x8425, 0],
        ["write", 0x8430, 79],
        ["write", 0x8431, 1],
        ["write", 0x8432, 239],
        ["write", 0x8433, 0],
        ["write", 0x8422, 128],
        ["write", 0x8426, 3],
        ["write", 0x8425, 0x53],
        ["write", 0x8422, 255],
        ["read", 0x8400],
        ["clear_fault"],
        ["write", 0x8425, 0x90],
        ["write", 0x8420, 0x81],
        ["read", 0x8400],
        ["read", 0x8402],
    ]
    for cycle in (0, 69, 140, 32767, 32768, 65535, 65536, 1000000):
        ops += [["cycle", cycle], ["peek", 0x8401], ["read", 0x8401], ["read", 0x8401]]
    for x, y in [(403, 601), (404, 602), (1023, 0), (0, 1023)]:
        ops += [
            ["contact", x, y, True],
            ["write", 0x8B04, 0xA8],
            ["write", 0x8800, 6],
            ["peek", 0x8C00],
            ["read", 0x8C00],
            ["contact", x ^ 1, y ^ 1, True],
            ["peek", 0x8C00],
            ["read", 0x8C00],
            ["write", 0x8B04, 0x8A],
            ["write", 0x8800, 7],
            ["read", 0x8C00],
            ["read", 0x8C00],
            ["contact", x, y, False],
        ]
    ops += [["contact", 1024, 0, True], ["write", 0x8800, 5], ["clear_fault"]]
    for clock in [
        Clock(1900, 2, 28, 23, 59, 59, 2),
        Clock(1993, 1, 1, 12, 30, 15, 4),
        Clock(2000, 2, 28, 23, 59, 59, 3),
        Clock(2099, 12, 31, 23, 59, 59, 5),
    ]:
        ops += [
            ["write", 0x8000 + n, value]
            for n, value in enumerate(clock.encode(bytes(5)))
        ]
        ops += [
            ["seconds", 2],
            ["peek", 0x8000],
            ["write", 0x800C, 128],
            ["peek", 0x832D],
            ["seconds", 1],
            ["write", 0x800C, 0],
        ]
    ops += [
        ["write", 0x8008, 4],
        ["causes", 4, 2],
        ["write", 0x4010, 3],
        ["irq"],
        ["write", 0x801A, 0],
        ["write", 0x4012, 0],
        ["irq"],
        ["write", 0x8B02, 87],
        ["peek", 0x8B1A],
        ["write", 0x8B05, 165],
        ["peek", 0x8B1D],
    ]
    for _ in range(64):
        address = 0x80000 + rng.randrange(0x40000)
        ops += [["write", address, rng.randrange(256)], ["read", address]]
    for low, high in [(0x10000, 0xB0000), (0x1ADCE, 0xBADCE), (0x1FFFF, 0xBFFFF)]:
        ops += [
            ["write", low, 0x5A],
            ["peek", high],
            ["read", high],
            ["write", high, 0xA5],
            ["peek", low],
            ["read", low],
        ]
    return ops


def apply(hw, operation):
    command, *args = operation
    if command == "write":
        return hw.write(*args)
    if command == "read":
        return hw.architectural_read(*args)
    if command == "peek":
        return hw.read(*args)
    if command == "cycle":
        hw.cycle = args[0]
    elif command == "clear_fault":
        hw.fault = None
    elif command == "causes":
        hw.rtc.raise_causes(*args)
    elif command == "irq":
        return hw.refresh_irq_inputs()
    elif command == "contact":
        if hw.tablet.set_contact(*args):
            hw.gate[18] |= 2
    elif command == "seconds":
        hw.rtc.advance_seconds(*args)
    else:
        raise ValueError(f"unknown operation {command}")
    return None


def observe(hw):
    return dict(
        selector=hw.selector,
        gate=list(hw.gate),
        gpio=list(hw.gpio),
        rtc=list(hw.rtc.registers()),
        lcd=list(hw.lcd.registers),
        windows=[list(hw.lcd.window_descriptor(n)) for n in range(16)],
        lcd_counts=[hw.lcd.data_writes, hw.lcd.data_reads, hw.lcd.block_operations],
        tablet=[
            hw.tablet.x,
            hw.tablet.y,
            hw.tablet.pressed,
            hw.tablet.conversion_control,
            hw.tablet.drive_control,
            hw.tablet.data_reads,
        ],
        fault=hw.fault is not None,
        pbm_sha256=hashlib.sha256(hw.lcd.pbm()).hexdigest(),
        ram_sha256=hashlib.sha256(hw.ram).hexdigest(),
        events_sha256=hashlib.sha256(
            json.dumps(hw.events, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    )


def corpus():
    hw = Hardware()
    hw.banks = {0xF0: bytes([0x21]) * 0x20000, 0xF1: bytes([0x43]) * 0x20000}
    ops = actions()
    replies = []
    checkpoints = []
    for index, op in enumerate(ops):
        try:
            replies.append(apply(hw, op))
        except ValueError:
            replies.append({"error": True})
        if index % 128 == 127 or index == len(ops) - 1:
            checkpoints.append(dict(index=index, state=observe(hw)))
    return dict(
        schema="oz9600-controller-reference-1",
        operations=ops,
        replies=replies,
        checkpoints=checkpoints,
    )


if __name__ == "__main__":
    print(json.dumps(corpus(), indent=2))
