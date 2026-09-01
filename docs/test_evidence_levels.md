# Test evidence levels

Tests in this repository make different kinds of claims. A passing parity or
model test is useful, but it is not automatically evidence of real firmware or
hardware behaviour. Use these labels when adding or interpreting correctness
tests.

| Evidence level | What it establishes | Representative coverage |
| --- | --- | --- |
| Real-device-derived | Encodes a claim derived from a deliberately executed physical-PC-E500 probe whose source, raw FT600 capture, execution window, and hashes are archived in the paired private evidence repository. The raw capture carries the device evidence; passing the regression test is not a new hardware observation. The claim is limited to the tested operands and observations. | Regressions derived from wrapping/flag-preserving `PMDF`; zero-WAIT's 65,536-count wrap; nonzero-WAIT I clearing and architectural C/Z preservation; tested-state BA/I `ED`/`FD` aliases; raw-width `C7`, `D7`, and `EXP`; isolated internal-direct overlap final states plus I/F observations, tested BP-relative source/destination selection, and tested counted-base selection; tested upper-address aliases for 62/66/6A/72/7A, 88-8F reads, A8-AF writes, and D0-D3/D8-DB absolute transfers; low-two-bit `POPU F`/`POPS F` normalization; read-side boundary wrapping; and X/Y/U/S 20-bit consumer behavior |
| ROM-grounded | Executes stock ROM instructions and asserts their observable state changes. The external ROM fixture must be present. | `pce500/tests/test_keyboard_interrupt_rom.py` |
| Architecture/ISA contract | Pins documented instruction or register semantics without depending on a particular ROM. | Interrupt-bit layout and 20-bit register-width tests in `pce500/tests/test_memory_alias_and_irq.py`, `sc62015/pysc62015/test_llama_raw24.py`, and `sc62015/core/src/lib.rs` |
| Cross-backend parity | Shows that two implementations agree. Agreement alone does not prove either implementation matches hardware. | CPU backend and WAIT/Perfetto parity tests |
| ROM-informed model | Pins a higher-level model whose addresses or data flow follow firmware, while timing or host-event policy remains emulated. | Keyboard handler/FIFO tests |
| Emulator model contract | Protects intentionally synthetic scheduling, delivery, wake, peripheral, or host-bridge behaviour. | RETI's conservative no-implicit-ack rule, `test_interrupts.py`, `test_key_irq_latch.py`, and `test_peripherals.py` |
| Implementation integrity | Proves that malformed input or a failed callback is rejected or contained visibly; it makes no silicon claim. | Vector-transfer preflight/fetch atomicity, snapshot candidate validation and atomic save, native-mirror rollback, poisoned-CPU recovery, and ON-key transaction tests |
| Smoke/instrumentation | Establishes that a path runs or emits trace data, not that the represented device behaviour is correct. | IRQ and execution tracing tests |

Assembler round trips and a full 256-opcode decode sweep are smoke/consistency
checks.  They prove that tables agree and that canonical encodings are stable;
they do not prove instruction semantics.  Cross-backend LLIL parity has the
same limitation.  See `sc62015_asm_llil_audit.md` for the current fail-closed
policy and the real-hardware trace queue.

The 2026-08-30 raw campaign record and follow-up are
`docs/pc-e500/en/analysis/sc62015_hardware_campaign_2026-08-30.md` in the
paired private repository, with the follow-up at
`sc62015_hardware_followup_2026-08-30.md`; later dated notes preserve the
WAIT-F, zero-count WAIT, overlap/base-selection, and upper-address matrices,
including the corrected D0-D3/D8-DB transfer capture.
The zero-count result is recorded in
`sc62015_hardware_zero_wait_2026-08-30.md`. Together they
verify complete HW-005, HW-010, HW-012, and observable HW-015 claims, and
partial subcases for zero and nonzero `WAIT`, overlap final states and base
selection, read-side boundary wrapping, and explicit user/system-stack F
normalization.
Public regression tests derived from those
results may be called real-device-derived, but a passing test without its
paired capture is not new hardware evidence. In particular, the captures do
not settle WAIT timer phase, complete countdown/bus interpretation, or
interrupt acceptance; invisible IMEM micro-order or untested relative/external
overlap modes; write-side boundaries; or interrupt-entry/RETI F images. The
connected unit's hardware revision was not recorded, so do not generalize the
results across revisions. The tools were invoked from a relocated path, but their
private-recorded hashes match
byte-identical files at tracked legacy paths in the gateware base.

Snapshot round trips are implementation-integrity tests. Version 4 rejects
legacy v3 outright, requires an exact schema, rejects duplicate or wrong-typed
fields, validates every represented candidate before commit, preserves
read-only ROM, and restores the represented scheduler/keyboard/LCD/runtime
state. A typed card contract and `memory_card.bin` preserve exact mode,
capacity, writability, and payload, including latent media while the slot is
absent. Core and standalone Rust reject remaining active device, callback, or
generic-overlay state that v4 cannot encode. The Python PCE wrapper similarly
rejects queued serial/cassette state, custom handler/writable overlays,
ambiguous or out-of-image static overlays, and custom IMEM callbacks. Rust
saves refresh `fast_mode` and range metadata from the live runtime, and loads
require the destination machine's fallback/read-only range contract to match
exactly. An
unexpected late native bridge hook failure poisons execution instead of
pretending that the partial commit was rolled back. These tests do not
establish that saved timer cadence, interrupt policy, or peripheral state
matches real hardware.

## Interrupt ground truth currently encoded

The optional PC-E500 ROM fixture is executed directly by the dispatcher tests.
They establish these firmware facts:

- `interrupt_vector` at `0xF1FB5` selects `ISR & IMR` in RX, EX, TX, ON,
  KEY, ST, MT order.
- The selected handler returns through the dispatcher epilogue, which
  acknowledges the source with `AND (ISR), A` at `0xF2059`.
- RETI at `0xF2060` follows the explicit firmware acknowledgement. The ROM path
  establishes the restored frame but cannot isolate whether RETI would also
  acknowledge an ISR bit that firmware deliberately left set.
- The default KEY handler at `0xF2061` clears KEYM in the saved IMR frame.

The architecture-level tests therefore pin TX=`0x10`, RX=`0x20`, EX=`0x40`.
The shared RETI implementation conservatively leaves ISR acknowledgement to
firmware because that matches both ROMs, but the unacknowledged case remains an
emulator model contract until hardware isolates it. The Python scheduler,
debounce, key-event latch, and peripheral-manager policies likewise remain
model contracts unless a test is explicitly promoted with ROM or hardware
evidence.

## Review rule

Name and document each test at the narrowest evidence level it actually proves.
Do not use “hardware”, “ROM”, “timer cadence”, or “parity” in a test claim when
the test only injects a register value, calls a shared host scheduler, or runs a
synthetic handler. When ROM-grounded and model tests cover the same boundary,
keep both: the former establishes the behaviour and the latter provides a fast
regression test.
