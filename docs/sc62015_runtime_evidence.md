# SC62015 runtime evidence boundary

This is the short, discoverable status page for claims made by the shared Rust
`CoreRuntime`. The detailed instruction audit remains in
[`sc62015_asm_llil_audit.md`](sc62015_asm_llil_audit.md), and the rules for
labelling tests are in [`test_evidence_levels.md`](test_evidence_levels.md).

## Closed instruction-description scope

The paired private repository's archived 2026-08-30 through 2026-09-02 device
campaign resolves every remaining row that selected between competing
descriptions of a valid, implemented opcode. The public evaluator keeps
unsupported or malformed encodings fail-closed. This does **not** promote
unobserved peripheral timing or reserved encodings into ISA facts.

## 2026-09-03 post-audit qualification

- A real Binary Ninja session, isolated from the older installed plugin copy,
  accepted and finalized all 905 canonical manifest rows, rejected all 17
  reserved/prefix-only rows, and emitted no root `LLIL_UNIMPL`. Distributed
  128-entry ROM samples passed for both PC-E500 and IQ-7000.
- The PC-E500 full-ROM boot and PF1 flow passed on the Python and shared Rust
  runtime paths. The IQ-7000 full-ROM Function Runner reached `LINK READY`,
  returned a checksummed seven-entry directory, imported two MEMOs through the
  paced COM/SIO bridge, exited through the ROM's ON-key path, and recalled the
  final MEMO through the normal UI.
- The final local suites passed 1,162 Python tests with two skips, 510 shared
  core Rust tests with six explicitly ignored cases, 26 PyO3 bridge tests, and
  31 PC-Link host-tool tests.
- That IQ-7000 proof exposed a real machine-profile bug rather than an ISA
  defect: PC-E500 hardware observes the physical ON level at `SSR.3` (`0x08`),
  while IQ-7000 ROM helper `F54EF` explicitly samples `SSR.1` (`0x02`). Both
  complete Rust runtimes now select the input mask from the device profile.
  A held input remains external to raw SSR storage.
- The private FCS/IOCS trace sweep completed all 123 PC-E500 entries without a
  runtime error, missing trace, or generic call-graph name. IQ-7000 produced
  traces for 104 of 105 applicable IOCS entries; command `0x61` is unreachable
  through that ROM dispatcher. One device-specific `0x18` return code and
  uncertain ROM symbol names remain analysis/spec work, not CPU-core failures.

## Runtime contract table

| Area | Current basis | Runtime policy |
| --- | --- | --- |
| Valid implemented instruction semantics | Real-device-derived, manual, and ROM evidence as itemized by the instruction audit | Execute normally; retain narrow regressions with their evidence level |
| Reserved opcodes, malformed PRE/modes/selectors, unsupported `F` bits, noncanonical address encodings | No valid-ISA claim | Reject atomically before observable scheduling or device mutation |
| Interrupt bit layout and RX, EX, TX, ON, KEY, ST, MT dispatcher priority | Stock-ROM control flow | Match both ROM dispatchers; do not claim that priority is hard-wired in silicon |
| Raw selected-matrix `KEYI` and MTI/STI status latching during an active handler | Archived PC-E500 device probes | Preserve the measured narrow behavior; host translated events remain separate |
| `RETI` acknowledgement | Archived PC-E500 probes with each defined ISR bit, and all bits together, deliberately left asserted | Restore the frame and leave `ISR` unchanged; acknowledgement remains handler/peripheral work |
| ON-key input and assertion/re-latch timing | PC-E500 hardware level/latch traces plus IQ-7000 ROM `F54EF` | Use model-specific SSR input bits (`0x08` PC-E500, `0x02` IQ-7000) with a shared `ISR.ONKI` latch; exact latency and debounce remain unverified |
| Neutral external `EXI` input | Functional test hook only | Level-style emulator contract with no claimed connector or peripheral meaning |
| SIO RX/TX ready interrupts and delays | ROM-compatible functional model | Advance from relative instruction timing; do not claim measured baud/status latency |
| PC-E500 timer cadence | Published nominal periods mapped onto a compatibility timebase | Mark absolute cadence and SCR divider phase provisional |
| IQ-7000 timer cadence | PC-E500 compatibility fallback | Mark the entire machine cadence uncalibrated; never present it as an IQ measurement |
| HALT/OFF clock domains | Manual/ROM-informed machine model | Freeze the modeled system-clock domain in HALT, keep the subclock running, stop both in OFF; retain as machine-level rather than opcode-timing evidence |
| Snapshot, callback rollback, poison state, trace ordering | Implementation integrity | Enforce exactness/fail-stop guarantees without describing them as silicon behavior |

## Default quarantine

- Historical PC-E500 serial ROM replacements and manufactured peer responses
  are disabled unless a diagnostic caller opts in explicitly.
- Trace-derived reset/turn-on overlays remain behind `--runtime legacy`; they
  are replay tools, not an alternative production scheduler.
- Raw external-bus transaction logs and snapshots remain on the legacy runner
  until every state or observation they require has an exact `CoreRuntime`
  contract. Exact bounded LCD-write capture is available on `CoreRuntime`.
- The shared runtime may expose neutral test hooks for ONK, EXI, and SIO, but a
  test that calls one proves the model contract only.

## Remaining useful real-hardware work

These measurements would improve machine accuracy without reopening valid
instruction descriptions:

1. Calibrate PC-E500 oscillator-to-relative-timing conversion and both
   `SCR.MTS`/`SCR.STS` periods, including divider phase after an SCR write.
2. Repeat timer calibration on IQ-7000 rather than inheriting the PC-E500
   compatibility mapping.
3. Measure ON-key assertion, release, debounce, wake, and re-latch latency at
   scheduler boundaries.
4. Identify the physical source and level/edge policy of `EXI` on each machine.
5. Measure SIO TX-ready, RX-ready, handshake, timeout, and interrupt latency at
   representative UCR settings.
6. Capture write data/partial visibility for external boundary-crossing writes;
   existing gateware established address/count/order but not trustworthy data.

Until a capture with source, raw output, hashes, and tested scope is archived,
these rows remain provisional model behavior.
