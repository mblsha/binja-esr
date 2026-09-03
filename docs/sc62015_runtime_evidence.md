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

## Runtime contract table

| Area | Current basis | Runtime policy |
| --- | --- | --- |
| Valid implemented instruction semantics | Real-device-derived, manual, and ROM evidence as itemized by the instruction audit | Execute normally; retain narrow regressions with their evidence level |
| Reserved opcodes, malformed PRE/modes/selectors, unsupported `F` bits, noncanonical address encodings | No valid-ISA claim | Reject atomically before observable scheduling or device mutation |
| Interrupt bit layout and RX, EX, TX, ON, KEY, ST, MT dispatcher priority | Stock-ROM control flow | Match both ROM dispatchers; do not claim that priority is hard-wired in silicon |
| Raw selected-matrix `KEYI` and MTI/STI status latching during an active handler | Archived PC-E500 device probes | Preserve the measured narrow behavior; host translated events remain separate |
| `RETI` acknowledgement | ROM explicitly acknowledges ISR before `RETI`; isolated silicon case is unmeasured | Do not invent an implicit acknowledgement |
| ON-key assertion/re-latch timing | Functional model plus ROM acknowledgement path | Level-style emulator contract; exact latency and debounce remain unverified |
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
- Raw bus/LCD transaction logs and snapshots remain on the legacy runner until
  every state or observation they require has an exact `CoreRuntime` contract.
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
6. Isolate `RETI` with an unacknowledged ISR bit to determine whether silicon
   performs any acknowledgement beyond restoring the frame.
7. Capture write data/partial visibility for external boundary-crossing writes;
   existing gateware established address/count/order but not trustworthy data.

Until a capture with source, raw output, hashes, and tested scope is archived,
these rows remain provisional model behavior.
