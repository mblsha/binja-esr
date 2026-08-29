# Test evidence levels

Tests in this repository make different kinds of claims. A passing parity or
model test is useful, but it is not automatically evidence of real firmware or
hardware behaviour. Use these labels when adding or interpreting correctness
tests.

| Evidence level | What it establishes | Representative coverage |
| --- | --- | --- |
| ROM-grounded | Executes stock ROM instructions and asserts their observable state changes. The external ROM fixture must be present. | `pce500/tests/test_keyboard_interrupt_rom.py` |
| Architecture/ISA contract | Pins documented instruction or register semantics without depending on a particular ROM. | RETI and interrupt layout tests in `pce500/tests/test_memory_alias_and_irq.py` and `sc62015/core/src/lib.rs` |
| Cross-backend parity | Shows that two implementations agree. Agreement alone does not prove either implementation matches hardware. | CPU backend and WAIT/Perfetto parity tests |
| ROM-informed model | Pins a higher-level model whose addresses or data flow follow firmware, while timing or host-event policy remains emulated. | Keyboard handler/FIFO tests |
| Emulator model contract | Protects intentionally synthetic scheduling, delivery, wake, peripheral, or host-bridge behaviour. | `test_interrupts.py`, `test_key_irq_latch.py`, and `test_peripherals.py` |
| Smoke/instrumentation | Establishes that a path runs or emits trace data, not that the represented device behaviour is correct. | IRQ and execution tracing tests |

## Interrupt ground truth currently encoded

The optional PC-E500 ROM fixture is executed directly by the dispatcher tests.
They establish these firmware facts:

- `interrupt_vector` at `0xF1FB5` selects `ISR & IMR` in RX, EX, TX, ON,
  KEY, ST, MT order.
- The selected handler returns through the dispatcher epilogue, which
  acknowledges the source with `AND (ISR), A` at `0xF2059`.
- RETI at `0xF2060` restores IMR, F, and PC; it does not acknowledge ISR.
- The default KEY handler at `0xF2061` clears KEYM in the saved IMR frame.

The architecture-level tests therefore pin TX=`0x10`, RX=`0x20`, EX=`0x40`
and require RETI to preserve ISR. The Python scheduler, debounce, key-event
latch, and peripheral-manager policies remain model contracts unless a test is
explicitly promoted with ROM or hardware evidence.

## Review rule

Name and document each test at the narrowest evidence level it actually proves.
Do not use “hardware”, “ROM”, “timer cadence”, or “parity” in a test claim when
the test only injects a register value, calls a shared host scheduler, or runs a
synthetic handler. When ROM-grounded and model tests cover the same boundary,
keep both: the former establishes the behaviour and the latter provides a fast
regression test.
