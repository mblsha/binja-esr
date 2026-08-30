# SC62015 assembler and LLIL audit

Date: 2026-08-30

This audit covers the Python decoder, assembler, LLIL evaluator, the Rust
LLAMA decoder/evaluator, the Python-to-Rust bridge, v4 checkpoints, timer
scheduling, and parity evidence paths.  Its purpose is to separate
ROM-supported behavior from emulator convention and to stop invalid or
unverified encodings from silently executing as plausible instructions.

## Evidence standard

Evidence is ranked as follows:

1. real-device bus, stack, and readback evidence from deliberately executed
   probes, distinguishing archived raw captures from decoded reports whose raw
   artifacts live elsewhere;
2. current ROM bytes decoded with the SC62015 decoder and BNIDA annotations;
3. PC-E500 firmware ABI and register-use behavior in the available technical
   reference, which is not a standalone SC62015 instruction-semantics source;
4. checked-in Binary Ninja disassembly/HLIL as a secondary cross-check; and
5. exact assembler/decode round trips, cross-backend parity, and isolated
   emulator-model tests.

Level 5 is regression evidence, not hardware evidence. Two backends can agree
on the same wrong behavior. ROM use sites are strong evidence that an encoding
is real, but a dedicated device capture outranks an inference from firmware.
The paired private repository currently contains decoded FT600 event tables,
not the named raw JSON artifacts, so those results are hardware reports rather
than independently reprocessable captures from a clean checkout.
The live Binary Ninja session was not
available for this pass, so no conclusion is presented as a live-decompiler
finding.  The current decoder was preferred over stale text exports.

## Disposition

The execution policy after this audit is:

- reserved opcodes, malformed operands, consecutive prefixes, and unsupported
  execution shapes fail without advancing PC;
- decoded metadata, text, and LLIL share the same canonicality checks;
- semantics supported by ROM use sites and the reference are implemented and
  covered in both cores;
- snapshots, native shadows, callbacks, timers, and parity traces use strict
  fail-closed integrity contracts that are not counted as silicon evidence;
  and
- ambiguous silicon edge cases are named as model contracts or quarantined,
  rather than being disguised as hardware-correct tests.

## Re-evaluation after device-evidence review

The second pass changed two audit classifications:

- The original blanket upper-nibble rejection for every `Imm20` was wrong.
  The X/Y device-capture reports execute a third byte of `3C` and expose `0C`
  through the following push. Register-immediate opcodes now have a distinct 20-bit
  normalization policy; control/address operands remain quarantined.
- RETI preserving ISR is a conservative, ROM-consistent emulator contract, not
  a hardware-proven instruction side effect. Both stock ROMs acknowledge ISR
  before RETI, so only a deliberately unacknowledged device probe can decide it.

The raw-data split survives re-evaluation: `MVP` is supported by both executable
ROM use and a device round trip; `C7 CMPP (m),(n)` remains a raw three-byte
comparison whose upper-nibble flags are model-derived; `D7 CMPP (m),r3` is a
20-bit pointer comparison, corroborated only by the baseline emulator's
best-guess pending hardware; and upper-nibble `EXP` remains quarantined. No
parity-only result is promoted to an ISA fact.

## Corrected defects

| Area | Previous behavior | Audit disposition |
| --- | --- | --- |
| Reserved opcodes `20`/`BF` | Python/Rust could manufacture placeholder execution | Reject as invalid; ROM byte sightings are not treated as executable proof |
| PRE decoding | Consecutive or noncanonical prefixes could reach execution, and mid-instruction byte pairs at `EFE2B` and `F0002` were incorrectly promoted to executable overlaps | Reject stacked/noncanonical PRE before scheduler mutation; no ignored-PRE pair currently has a proved executable boundary |
| Instruction observers and lookahead | Python fused-decode inspected the following opcode after every non-PRE instruction, the facade and PCE wrapper could fetch the current opcode repeatedly for rendering, WAIT detection, and tracing, and device reads therefore changed merely by enabling observers | Only PRE fetches its sister opcode. Scheduled execution is allowed only when every byte inspected by preflight is explicitly callback-free; the single normal opcode fetch is bound to the current PC in an owner-held prepared operation and reused by execution, WAIT inspection, rendering, and tracing. PRE carries its fused length and bytes in the same proof, so observers cannot add bus reads or reinterpret an operand as an opcode |
| Assembler PRE selection | Mode metadata could be lost and prefixes deduplicated by bytes rather than semantics | Preserve addressing provenance and verify all 16 two-operand PRE combinations |
| 20-bit assembler literals | Values above `FFFFF` were emitted and silently decoded as a different semantic value | Text assembly rejects values above `FFFFF` and emits a canonical low-nibble high byte. Executable decoding is instruction-specific: the reported X/Y device load-to-push path fetches the full third byte but exposes only bits 19-0, while unverified control/address operands reject bits 23-20 |
| Raw three-byte data | Rust treated the historical `IMem20` operand name as a universal 20-bit transfer, while both cores reused address-style `Imm20` for `MVP (k),lmn` | `MVP` preserves all 24 data bits and opcode `DC` has a distinct raw immediate. `C7 CMPP (m),(n)` compares raw three-byte images but its upper-nibble flags are not hardware-isolated. `EXP` rejects noncanonical upper nibbles until traced |
| `CMPP` width split | Python and Rust used one 24-bit comparison for both `C7 (m),(n)` and `D7 (m),r3`, so cross-backend parity confirmed the same mistake | Keep `C7` at 24 bits; mask both operands and set flags at 20-bit width for `D7`, whose other operand is the 20-bit X/Y/U/S class. A discriminating `F00080` versus `000080` regression now prevents parity from hiding the distinction. The baseline emulator independently makes the same split, but remains only a best-guess against hardware |
| `56`/`5E` and `E3`/`EB` external modes | Unsupported external-address modes could be accepted, and direct operand encoding could emit bytes that the decoder rejected | Share encoded-mode validation between decode and encode; require offset modes for `56`/`5E` and post-increment/pre-decrement for `E3`/`EB`, including direct `Instruction` construction |
| Register selectors | Exact `JP`/`CMPW`/`CMPP` forms and reserved selectors were too permissive; `JP A` also produced conflicting page semantics in Python and Rust; the assembler could emit values its own decoder rejected; an intermediate generic rule then rejected legal mixed-width `r3,r` arithmetic | Require exact `04..07` selectors for `JP X/Y/U/S`, exact low-three-bit selectors for `INC`/`DEC`, and opcode-specific `CMPW`/`CMPP` classes across Python encode/decode and Rust; preserve ROM-proven mixed-width forms such as `45 52` = `ADD Y, BA` |
| LLIL operand widths and masks | The mock evaluator accepted mixed-width near jumps, returns, dynamic IMEM addresses, `MV IL`, ROM-valid `FD 24`/`FD 42`, and `ED` exchanges that real Binary Ninja rejects; some control/address consumers retained unmodeled high bits | Emit explicit unsigned `ZERO_EXT`/`LOW_PART` conversions, snapshot both exchange directions, and apply the 8-bit IMEM or 20-bit external/control mask at every consumer; verify through a width-strict LLIL facade |
| Control/address immediate high byte | Text assembly could turn an out-of-range literal into a self-fulfilling raw alias; table bytes at PC-E500 `F003A` were incorrectly treated as executed `CALLF` evidence | Keep `JPF`/`CALLF`, direct external addresses, and synthetic vectors canonical pending targeted hardware evidence; this restriction does not apply to `MV X/Y/U/S,lmn`, whose register consumer discards bits 23-20 |
| Reset-vector selection | Python read the interrupt vector at `FFFFA` | Read the distinct reset-vector slot at `FFFFD`; whether software opcode `FF` exactly matches external/power-on reset, including stack and register effects, stays model-only |
| Vector-transfer failure ordering | Software `IR`, RESET, and synthetic IRQ paths could push a frame, clear `IMR.IRM`, reset SFRs, advance timers, or clear wrapper state before discovering a noncanonical vector, rejected destination, failed bus read, volatile mismatch, or late observer failure | Require explicit callback-free metadata for every scheduled vector and target byte, bind the one matching architectural vector fetch plus target length and memory provenance into a non-forgeable, one-shot prepared operation, and consume it before any stack, SFR, RAM, LCD, timer, scheduler, trace, or IRQ-metadata mutation. Machine reset is stricter: the reset-vector and decoded target bytes must be declared immutable before RAM/LCD reset can begin. Destination `I`, stack, and other data-dependent checks remain the destination instruction's job, and vector targets that are themselves `IR`/RESET do not recurse during preflight |
| Software interrupt `IR` | The saved PC pointed after the opcode | Save the `IR` opcode address; the ROM dispatcher tests `FE` there and advances the saved frame |
| Stacked `F` byte | Rust preserved six opaque upper bits while Python modeled only carry/zero; `POPS F` advanced `S` before its lazy validation node | Keep the one-byte stack layout and accept only the modeled `C`/`Z` image (`0..3`); snapshot and reject raw images with bits 2-7 before `S`, flags, or PC can mutate |
| `TEST` | Some LLIL paths used a 24-bit operation for a byte instruction | Use byte-width logic |
| `ADC`/`SBC`, `ADCL`/`SBCL` | Carry/borrow propagation could use an unbounded intermediate | Apply width wrapping at each arithmetic stage |
| `EXL` | Only part of the `I`-byte exchange was modeled | Exchange exactly `I` bytes, with independent wrapped internal pointers |
| `MVL` family | Several Rust composite forms copied one extra byte or only one byte | Execute exactly `I` ordered transfers, update both pointers, then clear `I` |
| Decimal shifts | Direction, pointer walk, and carried nibble disagreed | `DSLL` starts at the LSB and decrements; `DSRL` starts at the MSB and increments |
| Wide memory access | A word/pointer could spill outside 8-bit IMEM or 20-bit external memory, an overlapping store could mutate the register used to calculate later destination bytes, and bridge buses could collapse a wide store into one callback | Snapshot the complete destination and source, then load/store and invoke hardware/host hooks once per byte in architectural order with 8-bit IMEM or 20-bit external wrapping at every byte |
| `CALL`/`CALLF` LLIL | The lifter performed the explicit wrapped stack-frame writes but represented the control-flow edge as a jump | Keep the explicit architectural frame writes and emit a Binary Ninja call edge; the mock evaluator treats that call node as control flow only so it cannot add a second synthetic frame |
| Interrupt-frame boundary access | Synthetic delivery paths could read or write a five-byte system frame past the 20-bit external bus when `S < 5` | Snapshot the frame inputs and wrap every byte at `FFFFF -> 00000`; exact silicon bus order at that boundary remains hardware-trace work |
| External pointers and control flow | Some paths advanced in a 24-bit space, while LLIL register consumers could retain a high nibble that the runtime facade masks | Mask effective addresses, pointer updates, register jumps, and far-return targets to 20 bits; near control flow explicitly widens its 16-bit component before page composition |
| `PMDF` | Implemented as packed BCD | Use 8-bit wrapping binary pointer addition; flag preservation remains a model contract |
| `TCL` | Could execute as a silent no-op | Fail closed until `LCC.STCL`/`LCC.MTCL` timer-phase effects are implemented and traced |
| `I=0` counted instructions | Empty-block and 65,536-iteration interpretations had no decisive ROM or hardware evidence | Reject `ADCL`/`SBCL`/`DADL`/`DSBL`/`MVL`/`MVLD`/`EXL`/`DSLL`/`DSRL`/`WAIT` before architectural or timing mutation; require a real-hardware trace to choose semantics |
| `WAIT` | Timing could be omitted by direct LLIL evaluation; irrelevant PRE could reach a different path | For nonzero `I`, use a WAIT intrinsic/decoded fast path for exact model cycles, fail closed without a working timing implementation, and reject PRE+WAIT as noncanonical |
| Scheduler preflight | Device wrappers could advance timers before discovering that `TCL`, an `I=0` counted operation, an unfused/consecutive PRE, or another invalid encoding was quarantined | Decode and validate the pending instruction before cycle, timer, or ISR mutation attributable to scheduling it; preflight the replacement PC again after interrupt delivery, and repeat validation in each backend so direct callers also fail closed |
| Host-authoritative PCE/native shadow | Some host-originated writes bypassed the Rust mirror, while native snapshot defaults could replace live host SFR/IRQ state | Treat the PCE memory image and its `IMR`/`ISR` bytes as authoritative; synchronize every host write into the native shadow, reject split host/native latch or scheduler state, and never let native defaults overwrite the host image. This is bridge-regression evidence (level 5), not hardware evidence |
| Wide-write/callback atomicity | WAIT/timer writes could remain pending, retrying callback signatures after a body `TypeError` could invoke a side effect twice, and a later byte failure could leave native counters, mirror state, or trace ordering partially advanced | Invoke each callback once after arity inspection, flush deferred writes in order before reporting success, and roll back native CPU/device state, mirror/dirty queues, counters, and global trace ordering on failure. Because an external host byte may already have committed, record uncertain addresses and poison mutation until RESET rereads them from authoritative host memory |
| Poisoned key APIs | Key injection/release could mutate native keyboard, IRQ, or mirror state and call the host after an earlier callback failure | Gate all four Rust key APIs while poisoned; make each key operation a mirror/keyboard/timer transaction, stop after the first failed callback, roll back native state, and retain the original poison reason. This is bridge-regression evidence (level 5), not hardware evidence |
| ON-key transaction | Host ON press/release could update SSR, ISR, pending-source metadata, or the native keyboard in different orders and retain a partial state when a callback failed | Treat SSR.ONK, ISR.ONKI, pending-source selection, native keyboard state, and timer bookkeeping as one fail-closed transaction. Release reselects any remaining asserted source instead of inventing or discarding one |
| Rust bridge image extraction | A malformed or partial Python memory export could leave the active native mirror partly replaced or produce a padded/truncated snapshot; a rejected LLAMA snapshot could then be silently retried through a different serializer | Require an exact three-item export, exact 1 MiB external and 256-byte internal images, and validated ranges for both mirror initialization and snapshot capture; build a complete candidate off to the side and commit or serialize it only after every extraction and shape check succeeds. When LLAMA is explicitly active, propagate its native saver errors instead of falling back. This is bridge-regression evidence (level 5), not hardware evidence |
| Snapshot v4 validation and restoration | A malformed late field could mutate a live machine before rejection; Rust JSON duplicate keys were collapsed; LLAMA output could replace scheduler/call state with defaults, truncate timer targets, lose a physically held ON-key level, save stale runtime/range metadata, overwrite read-only ROM, or be silently retried through another serializer; v3 omitted memory-card payload/configuration and carried no proof that its producer had rejected other unrepresented host state; standalone load could add PC-E500 read-only ranges to an IQ-7000 | Reject legacy v3 outright and require a duplicate-free v4 archive with recursively duplicate-free JSON, exact memory/register shapes, strict external/ON-key input-level booleans, internally consistent call/timer/IRQ/keyboard/LCD state, an exact live fallback/read-only range contract, matching device model and active read-only bytes, host-authoritative `IMR`/`ISR`, and a typed card mode/capacity/writability contract plus exact `memory_card.bin`. Core round-trips its held input levels; the PCE host refuses asserted native-only levels rather than erasing them. Refresh metadata from live runtime, validate represented subsystem candidates before commit, save to a temporary file, strict-read it back, sync, and atomically replace the destination. Core/standalone still refuse active SIO, peripheral, RTC, callback, trace-resume, or generic overlay state that v4 does not encode; the Python PCE wrapper likewise refuses custom handler/writable overlays, ambiguous or out-of-image static overlays, custom IMEM callbacks, and queued serial/cassette state. Never change serializer after an explicitly selected LLAMA path fails. These are format/integrity guarantees (level 5), not hardware evidence |
| Wrapping timer catch-up | Repeatedly adding a period could take unbounded time for a stale deadline or hang near the host `u64` cycle-counter wrap | Compare deadlines with half-range wrapping order and advance missed periods with one widened division, preserving phase in O(1); reject ambiguous periods at or above half-range. This is scheduler-integrity behavior, not a claim about the silicon counter width or timer phase |
| Bridge/parity tools | Callback errors, broken host-memory attachment, encoder-generation failures, explicit backend fallback, missing flags/interrupt bytes, duplicate indices, or unequal trace lengths could be hidden | Require explicit host-memory/backend selection and complete modeled flags; every instruction event must contain `pc`, `opcode`, `op_index`, `mem_imr`, and `mem_isr`; reject duplicate or contradictory indices and compare the union of indices so a missing event is a divergence. Exercise the independent Python oracle and real comparator across a subprocess boundary |
| Perfetto current-instruction context | Process-global current-PC/op/substep fields let executors on different OS threads publish transient context into each other's callbacks and made a rejection test order-dependent | Keep the global monotonic instruction sequence, but make active instruction context and substep local to the executor thread. This does not provide per-machine isolation for same-thread nesting: the tracer/counter remain global and re-entrant execution is outside the guarantee. This is trace/test isolation, not silicon evidence |
| Binary Ninja full-memory view | `C0000-DFFFF` was mislabeled writable CE2 and several regions omitted their final byte | Map the complete `C0000-FFFFF` ROM read-only and make all external segments contiguous |

The `JP r3` restriction is grounded in coherent stock-ROM dispatch paths, not
just emulator agreement. PC-E500 `E2F5F` and IQ-7000 `E4766` use `11 04`
(`JP X`); PC-E500 `F2053` and IQ-7000 `F5230` use `11 05` (`JP Y`). Apparent
narrow-register or upper-bit forms occur in table/misaligned regions and are
therefore quarantined rather than promoted to instruction encodings.

Three apparent ROM “aliases” were removed after rechecking instruction boundaries
and BNIDA entries. At PC-E500 `EFE28`, the stream is `90 24`
(`MV A,[X++]`), `B0 25` (`MV [Y++],A`), then `7C 01` (`DEC IL`);
`EFE2B` is the second move's operand and has no analyzed entry.  At `F003A`,
`05 3A 07 7C` lies inside the function-dispatch table between named table
regions and likewise has no analyzed entry. At `F0000`, the real memory-card
entry decodes as `2A` (`PUSHU BA`), `FD 23` (`MV BA,I`), then `48 3F`
(`SUB A,3F`). The former `F0002` start overlaps the operand byte of `FD 23`, so
`23 48 3F` is not an ignored-PRE instruction. Byte-pattern coincidence alone
is not execution evidence.

Three encoded bytes do not imply a 24-bit register. PC-E500 capture reports
deliberately fetched `MV X,0x3C5AA5` and `MV Y,0x3C5AA5`; the following pushes
wrote `A5 5A 0C`, not `A5 5A 3C`. That is direct silicon evidence that the
observable X/Y load-to-push semantics expose only bits 19-0. The capture does
not, by itself, distinguish an internal mask at `MV` from one at `PUSHU`; the
architectural emulator model normalizes at the register write. The decoder
therefore accepts the raw third-byte alias
for register-load opcodes `0C..0F` and normalizes it to the effective 20-bit
value. Text assembly still emits only canonical values. U and S share that
architectural operand class, although an equally direct U/S upper-nibble
capture remains queued. This result must not be generalized to `JPF`, `CALLF`,
external-address operands, or interrupt vectors without their own probe.

## Emulator integrity boundaries

Snapshot version 4 is an exact contract for the state represented by the
format, not a best-effort import and not a promise to capture arbitrary host
attachments.  Loaders reject duplicate, missing, or unexpected archive/schema
fields; recursively duplicate JSON members; wrong register or memory sizes;
inconsistent internal-memory mirrors; invalid call stacks, timer phase,
interrupt flow IDs, keyboard state, or LCD payloads; a mismatched device model;
and any attempt to replace the active read-only image.  Represented subsystem
candidates are validated before commit.  Savers write a temporary archive,
strict-read it, sync it, and only then replace the destination.

Version 4 replaces v3 rather than silently reinterpreting it: an old v3 archive
does not attest that its producer rejected omitted callbacks, queues, or card
state, so exact loaders reject it by default. A typed, non-forgeable memory-card
contract records present/absent mode, supported capacity, writability, and the
exact payload in `memory_card.bin`; an absent card retains its latent medium so
later reinsertion cannot reveal lost state. Generic overlay names never count
as proof of that contract.

Core and standalone Rust fail closed when active SIO queues/lines, peripheral
bridge state, RTC protocol state, host callbacks, trace-resume state, or generic
overlays cannot be represented by v4. The Python PCE wrapper also refuses save
and load while serial queues, cassette tape state, custom handler-backed or
writable overlays, overlapping/out-of-image static overlays, or a custom IMEM
callback would be omitted. Disjoint static ROM overlays wholly represented by
the flattened image remain allowed. Snapshot metadata is refreshed from
the live `fast_mode`, fallback-range, and read-only-range configuration; load
requires the destination machine's range contract to match exactly and never
adds another model's defaults. The PCE bridge validates native candidates
before host mutation. An
unexpected failure in a native restore hook after commit begins cannot
honestly be called atomic; it poisons the wrapper so the partially restored
machine cannot execute until RESET. Diagnostic tracing sinks and external
device configuration remain outside the checkpoint payload.

The PCE host image remains authoritative when LLAMA is selected.  Host byte
writes, including SFR writes, synchronize the native shadow; wide native stores
cross the bridge as ordered byte transactions.  If a callback fails, reversible
native state and accounting roll back.  A byte already committed by external
host code cannot be undone honestly, so its address is retained as uncertain
and mutation stays poisoned until RESET rereads authoritative host memory.

Timer and parity hardening protects the evidence pipeline itself.  Stale timer
deadlines advance in constant time with wrapping arithmetic instead of a loop.
Parity emitters, the independent Python oracle, and the comparator require the
same instruction identity and interrupt-image fields; duplicate indices and
missing events are errors.  These guarantees make failures visible, but a
matching trace still establishes only cross-implementation agreement.

## Explicit model contracts

These behaviors are deliberately testable, but a passing test must not be
described as silicon proof:

- For nonzero `I`, `WAIT` requests exactly `I` idle cycles in addition to the
  opcode step, clears `I`, and preserves `C`/`Z`.
- `PMDF` preserves arithmetic flags.
- External/power-on reset and software opcode `FF` are not treated as proven
  equivalents. Their UCR/USR/ISR/SCR/SSR/LCC mutations, stack behavior, and
  general-register/flag retention are current manual-derived model contracts;
  only the distinct reset-vector slot is established here.
- HALT and OFF register mutations and wake sources are provisional.  The two
  power states must remain distinguishable even where a host API exposes a
  common `halted` boolean.  The current model preserves pending status that is
  not a wake source; OFF wakes only for ONKI, while HALT may wake for any
  already asserted ISR source even if it is masked or host-side generation of
  that source is disabled. Wake is modeled as an idle-step boundary: it changes
  power/IRQ state without fetching the dormant PC, and an already-unmasked IRQ
  may replace that saved PC on the next scheduling pass.
- Multi-byte overlapping writes are ordered low byte first in the current
  emulator contract, with the full destination and source snapshotted before
  the first byte is written.
- `F` occupies one byte in stack/interrupt frames, but only `C` and `Z` are
  currently modeled.  Locally generated images are therefore `0..3`; incoming
  images with any of bits 2-7 set fail closed before registers, stack pointers,
  memory, or interrupt state can change.

## Quarantined or hardware-trace work

Strict v4 snapshots, host/native shadow synchronization, callback rollback,
O(1) host-timer arithmetic, and parity trace validation are emulator-integrity
requirements.  They do not need to be promoted to ISA facts, and they do not
resolve any silicon question below.

The following cases must be cross-validated on a real SC62015 target before
they are promoted to architecture facts:

1. `TCL` timer-divider phase changes for every `LCC.STCL`/`LCC.MTCL`
   combination.  Execution currently raises an error.
2. Silicon behavior for `I=0` on every counted instruction, including `WAIT`.
   All ten instruction families currently fail atomically before PC, `I`,
   pointers, flags, memory, or timing mutation.
3. For nonzero `I`, the total `1 + I` WAIT timing, when `I` clears, `C`/`Z`/`F`
   preservation, timer and bus-countdown progression, and whether interrupts
   are accepted during or only after the wait.
4. Probe software opcode `FF` separately from external/power-on reset: vector
   fetch and byte order, stack effects, resume PC, `S`/`U`/`F`/`C`/`Z`
   retention, and SFR/timer mutations. For HALT and OFF, also capture resume
   PC, flags, active clock domains, pending status, and masked/unmasked wake
   sources.
5. `PMDF` carry/zero flag preservation.
6. Reserved `20`/`BF`, illegal `E3`/`EB` modes, unproven redundant PRE,
   consecutive PRE, and reserved register-selector encodings. These currently
   fail closed, including the disproved overlapping `23 48` stream at `F0002`.
7. Whether ignored bytes in BP+PX/BP+PY encodings and `ED`/`FD` aliases have
   observable silicon behavior.  Noncanonical ignored PRE selector bytes are
   rejected; documented register-pair aliases remain decodable.
8. Silicon behavior for a deliberately executed `JPF`/`CALLF`, direct external
   address, or interrupt vector whose encoded high byte has bits 7-4 set.
   Existing ROM sightings are table/data bytes, not executed proof, so those
   control/address consumers currently reject the form. Register-immediate
   opcodes `0C..0F` are excluded: X/Y captures already prove a low-20-bit
   observable load-to-push path.
9. Byte order and final state for self-overlapping `MVW`/`MVP` and other wide
   writes, including whether source/destination addresses are snapshotted or
   reread after each byte. Separately validate `MVL`/`MVLD`/`EXL` overlap,
   pointer/`I` update order, cycle counts, bus ordering, and address-space
   boundary crossings.
10. Whether `POPU/POPS F`, `PUSHU/PUSHS F`, and interrupt entry/`RETI`
   preserve, clear, or otherwise expose bits 2-7 of the byte-wide `F` image.
   Both cores currently reject such raw images before any architectural
   mutation rather than inventing normalization or retention semantics.
11. Exact silicon behavior of `EXP (m),(n)` and `C7 CMPP (m),(n)` when either
    source high byte has bits 7-4 set, plus a discriminating upper-nibble probe
    for `D7 CMPP (m),r3`. The sibling `MVP` path preserves all three data bytes,
    but neither stock ROM provides a credible executed `EXP` use site and no
    capture isolates either comparison's flags. `EXP` therefore rejects either
    nonzero upper nibble before its first write; `C7` remains a model-derived
    raw three-byte comparison and `D7` a model-derived 20-bit comparison.
12. ON-key electrical/status ordering: when SSR.ONK and ISR.ONKI assert and
    clear relative to press/release, whether ONKI re-latches while masked or
    in service, and how another pending source is selected after release.
    Transactional host/native behavior is an emulator integrity rule only.
13. System-interrupt frame bus order when `S < 5`, including whether each byte
    wraps independently at `FFFFF -> 00000` and whether any partial write is
    externally visible before vector fetch.
14. Direct upper-nibble load/readback for U and S, plus an upper-bit-sensitive
    ALU/compare probe for X/Y that can distinguish load-time normalization from
    push-time truncation. U/S use the same 20-bit architectural register class,
    so the emulator masks them identically, but current captures do not
    independently exercise opcodes `0E` and `0F`.
15. RETI entered with the delivered ISR bit deliberately still asserted. Both
    stock ROMs acknowledge before RETI, so the emulator's no-implicit-ack rule
    is a conservative, ROM-consistent model rather than an isolated silicon
    result.
16. Archive the raw FT600 JSON and exact probe source/configuration for every
    capture report cited by the paired private audit, then re-run their
    decoders. The checked-in event tables are detailed but cannot currently be
    regenerated from this checkout alone.

Static byte matches in a ROM are insufficient evidence: data and misaligned
instruction streams contain opcode-looking bytes.  A control-flow xref,
executed trace, or dedicated hardware program is required.

## Reproduction

Run the table and PRE consistency checks, then the Python and Rust suites:

```bash
python scripts/check_llama_opcodes.py
python scripts/check_llama_pre_tables.py
FORCE_BINJA_MOCK=1 pytest -q sc62015/pysc62015 sc62015/test_arch.py sc62015/test_view.py
cargo test --manifest-path sc62015/core/Cargo.toml --all-features
cargo test --manifest-path sc62015/rustcore/Cargo.toml
```

The private ROM-evidence repository contains exact decoder commands and
addresses.  Those ROM observations establish use-site intent; the hardware
trace queue above deliberately remains separate.
