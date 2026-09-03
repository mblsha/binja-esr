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
For the 2026-08-30 PC-E500 campaign, the paired private repository preserves
the exact generated probe source, raw FT600 JSON, payload and artifact hashes,
strict execution windows, and decoded result tables. Its dated campaign record
is `docs/pc-e500/en/analysis/sc62015_hardware_campaign_2026-08-30.md` in that
repository; `sc62015_hardware_followup_2026-08-30.md` records the same-day
boundary, system-stack-F, and Y/S follow-up, and
`sc62015_hardware_overlap_matrix_2026-08-30.md` records the later isolated
HW-007 matrix, `sc62015_hardware_ce1_pointer_overlap_2026-08-31.md` the
read-only direct/displaced `F1`/`F2`/`F3` external-pointer trace,
`sc62015_hardware_ce1_write_order_2026-08-31.md` the address-only
direct/displaced `F8`/`F9`/`FA`/`FB` external-write trace,
`sc62015_hardware_upper_address_2026-08-30.md` the HW-009 direct-read partial,
`sc62015_hardware_upper_read_matrix_2026-08-31.md` the paired 89-8F
direct-read width matrix,
`sc62015_hardware_upper_write_2026-08-31.md` the paired A8-AF direct-write
address matrix,
`sc62015_hardware_upper_absolute_byte_2026-08-31.md` the paired
62/66/6A/72/7A absolute-byte matrix,
`sc62015_hardware_upper_absolute_transfer_2026-08-31.md` the corrected paired
D0-D3/D8-DB transfer matrix,
and `sc62015_hardware_wait_flags_2026-08-30.md` the HW-003 C/Z matrix. The later
`sc62015_hardware_zero_wait_2026-08-30.md` records the HW-002 zero-count WAIT
result. The 2026-09-02 follow-up records low-S frame wrapping and
frame-before-vector order in `sc62015_hardware_interrupt_frame_wrap_2026-09-02.md`,
raw matrix KEYI level/latch behavior in `sc62015_hardware_key_irq_2026-09-02.md`,
and continued MTI/STI latching during interrupt service in
`sc62015_hardware_mti_during_interrupt_2026-09-02.md` and
`sc62015_hardware_sti_during_interrupt_2026-09-02.md`. The private
`sc62015_hardware_instruction_closure_2026-09-01.md`
reconciles the later TCL, zero-count block, far-control, PRE, RETI, IR,
WAIT/IRQ, HALT/OFF, and software RESET captures and separates valid-instruction
closure from residual peripheral or malformed-encoding work. The connected
unit's PCB, package, and unit revisions were not
recorded, so claims are scoped to that unit. The
capture tools were invoked from a relocated path, but their private-recorded
hashes match byte-identical files at tracked legacy paths in the gateware base.
The live Binary Ninja session was not available for this pass, so no conclusion
is presented as a live-decompiler finding. The current decoder was preferred
over stale text exports.

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

The device-evidence passes changed these audit classifications:

- The original blanket upper-nibble rejection for every `Imm20` was wrong.
  Archived X/Y/U/S probes execute a third byte of `3C` and expose only the low
  nibble across discriminating D7 comparison and copy/serialization consumers.
  Register-immediate opcodes now have a distinct 20-bit normalization policy;
  control/address operands remain quarantined. The physical circuit point at
  which bits 23–20 disappear is unobservable and is not an architectural
  subcase.
- A 2026-08-30 PC-E500 campaign directly established wrapping binary `PMDF`
  with preservation of deliberately contradictory incoming `C`/`Z`. In the
  tested BA/I and seeded-`F=03` (`C=1`, `Z=1`) state, `ED`/`FD` selectors `01`,
  `03`, `21`, and canonical `23` matched in result and observed event shape.
- The campaign and follow-up established the operand-width split: `C7 CMPP
  (m),(n)` is raw-24 versus raw-24, `D7 CMPP (m),r3` is raw-24 memory versus a
  zero-extended 20-bit register across directly tested X/Y/U/S selectors, and
  `EXP` swaps both raw 24-bit triples. `EXP` preserved
  `F=03` in the tested case, but that single case is not a complete flag matrix.
- `POPU F`, `POPS F`, and `RETI` normalize tested arbitrary stack bytes to
  their low two `C`/`Z` bits; the corresponding pushes and normal interrupt
  frame therefore emit the normalized byte.
- A guarded opcode `88` pair establishes that raw encoded address `8100FD`
  consumes all four bytes and reads the same low-20-bit `100FD` byte as
  canonical `0100FD`. A later paired 89-8F matrix extends that result across
  every remaining direct register-load width: high bytes `81` and `01` load
  the same one/two/three sentinel bytes from `101F0`. The paired A8-AF matrix
  shows raw encoded destination `8406D0` consumes four bytes and emits the same
  width-sized low-to-high write-address burst as canonical `0406D0` for every
  direct register-store width. A further absolute-byte matrix shows CMP/TEST
  reads and XOR/AND/OR identity read-then-write sequences match between high
  bytes `84` and `04`. The corrected D0-D3/D8-DB matrix shows high bytes
  `81`/`01` select the same external source for transfers into IMEM and high
  bytes `84`/`04` emit the same bounded CE1 destination sequence for transfers
  out of IMEM. The loaded gateware does not expose write data. These results are
  scoped to 62/66/6A/72/7A, 88-8F, A8-AF, D0-D3, D8-DB, and tested upper nibble
  `8`; they do not establish a general control/address alias rule.
- A deliberately unacknowledged real-device matrix now establishes that RETI
  leaves each defined ISR bit, and all seven together, unchanged. A separate
  arbitrary stacked-F matrix establishes `F = raw & 03` during RETI and the
  five-byte frame order. The former preserve-ISR contract is now hardware-backed.

The campaign also resolved the remaining valid-instruction alternatives.
Nonzero `WAIT` with `I=1,2,4` cleared I and added one captured
countdown/address unit per requested idle unit. A later `I=2` matrix shows all
four architectural C/Z images `00..03` survive. A human-attended exact `WAIT`
with `I=0000` produces 65,536 CE-inactive address/countdown samples covering
one full 16-bit wrap, then resumes with I still zero and tested `F=03`
(`C=1`, `Z=1`) preserved. Exact 4,096- and 65,535-unit runs with an independently
arising MTI show that WAIT completes atomically and the IRQ is delivered at the
next boundary after the fall-through fetch. A separate matrix establishes the
same 65,536-iteration do-while interpretation for all nine counted block
families. The isolated overlap matrix covers
exact aliases and both ±1 directions for
fixed-width moves, counted moves, `EXL`, and `EXP`. Fixed-width move final
states are snapshot-equivalent, counted moves expose direction-specific
cascades, and `EXL`/`EXP` are sequential-byte-exchange-equivalent. Overlapping
`EXP` contradicts whole-triple snapshot behavior. Final state does not expose
invisible IMEM micro-order. A BP-relative companion establishes
snapshot-equivalent source and initial-destination selection for tested
`MVW`/`MVP` even though the first write overwrites BP. Counted companions fix
the tested initial BP, PX, and BP+PX destination bases and PY source base before
iteration. The follow-up verifies read-side boundary wrapping but not external
partial-write visibility. Separate device matrices establish TCL's independent
named timer-phase restarts, every high-nibble JPF/CALLF alias, measured PRE
aliases/boundaries, software IR frame semantics, HALT/OFF fall-through resume,
and software RESET's distinct vector/no-frame transfer. No parity-only result
is promoted to an ISA fact.

## Corrected defects

| Area | Previous behavior | Audit disposition |
| --- | --- | --- |
| Reserved opcodes `20`/`BF` | Python/Rust could manufacture placeholder execution | Reject as invalid; ROM byte sightings are not treated as executable proof |
| PRE decoding | Consecutive or noncanonical prefixes could reach execution, and mid-instruction byte pairs at `EFE2B` and `F0002` were incorrectly promoted to executable overlaps | Reject unsupported stacked/noncanonical PRE before scheduler mutation. Device probes now pin the accepted redundant aliases and consecutive-prefix boundary; only the named `F0002` entry `23 48 3F` remains exact ROM-backed redundant-PRE evidence |
| Instruction observers and lookahead | Python fused-decode inspected the following opcode after every non-PRE instruction, the facade and PCE wrapper could fetch the current opcode repeatedly for rendering, WAIT detection, and tracing, and device reads therefore changed merely by enabling observers | Only PRE fetches its sister opcode. Scheduled execution is allowed only when every byte inspected by preflight is explicitly callback-free; the single normal opcode fetch is bound to the current PC in an owner-held prepared operation and reused by execution, WAIT inspection, rendering, and tracing. PRE carries its fused length and bytes in the same proof, so observers cannot add bus reads or reinterpret an operand as an opcode |
| Assembler PRE selection | Mode metadata could be lost and prefixes deduplicated by bytes rather than semantics | Preserve addressing provenance and verify all 16 two-operand PRE combinations |
| 20-bit assembler literals | Values above `FFFFF` were emitted and silently decoded as a different semantic value | Text assembly rejects values above `FFFFF` and emits a canonical low-nibble high byte. Executable decoding is instruction-specific: X/Y/U/S register-immediate loads expose only bits 19-0; guarded 62/66/6A/72/7A, 88-8F, A8-AF, D0-D3, and D8-DB probes map tested upper nibble `8` to the low-20-bit data address. Unverified vectors and other untested absolute-memory families still reject bits 23-20 |
| Raw three-byte data | Rust treated the historical `IMem20` operand name as a universal 20-bit transfer, while both cores reused address-style `Imm20` for `MVP (k),lmn`; `EXP` then snapshotted both complete triples before writing | `MVP` preserves all 24 data bits and opcode `DC` has a distinct raw immediate. Real-device probes establish raw 24-bit operands for `C7 CMPP (m),(n)` and both sides of `EXP`. The isolated ±1 overlap results contradict whole-triple snapshot behavior and require sequential-byte-exchange-equivalent final states; final state alone does not establish invisible micro-order |
| `CMPP` width split | Python and Rust used one 24-bit comparison for both `C7 (m),(n)` and `D7 (m),r3`, then an intermediate audit incorrectly masked D7's memory operand to 20 bits | Keep `C7` as a 24-bit memory-to-memory comparison. For `D7`, compare the raw 24-bit memory image with the zero-extended 20-bit X/Y/U/S register and set flags at 24-bit width. Discriminating PC-E500 cases `F00080` versus `000080`, plus `0C5AA5`/`3C5AA5` against the same raw-loaded X, establish this split |
| `56`/`5E` and `E3`/`EB` external modes | Unsupported external-address modes could be accepted, and direct operand encoding could emit bytes that the decoder rejected | Share encoded-mode validation between decode and encode; require offset modes for `56`/`5E` and post-increment/pre-decrement for `E3`/`EB`, including direct `Instruction` construction |
| Register selectors | Exact `JP`/`CMPW`/`CMPP` forms and reserved selectors were too permissive; `JP A` also produced conflicting page semantics in Python and Rust; the assembler could emit values its own decoder rejected; an intermediate generic rule then rejected legal mixed-width `r3,r` arithmetic | Require exact `04..07` selectors for `JP X/Y/U/S`, exact low-three-bit selectors for `INC`/`DEC`, and opcode-specific `CMPW`/`CMPP` classes across Python encode/decode and Rust; preserve ROM-proven mixed-width forms such as `45 52` = `ADD Y, BA`. For the BA/I pair, PC-E500 probes show selectors `01`, `03`, `21`, and canonical `23` match in the tested BA/I and seeded-`F=03` state, including observed event shape |
| LLIL operand widths and masks | The mock evaluator accepted mixed-width near jumps, returns, dynamic IMEM addresses, `MV IL`, ROM-valid `FD 24`/`FD 42`, and `ED` exchanges that real Binary Ninja rejects; some control/address consumers retained unmodeled high bits | Emit explicit unsigned `ZERO_EXT`/`LOW_PART` conversions, snapshot both exchange directions, and apply the 8-bit IMEM or 20-bit external/control mask at every consumer; verify through a width-strict LLIL facade |
| Control/address immediate high byte | Text assembly could turn an out-of-range literal into a self-fulfilling raw alias; table bytes at PC-E500 `F003A` were incorrectly treated as executed `CALLF` evidence | Raw JPF/CALLF probes for every upper nibble `1..F` transfer to the same low-20-bit target as canonical form. Raw 62/66/6A/72/7A, 88-8F, A8-AF, D0-D3, and D8-DB forms with tested upper nibble `8` are likewise device-verified low-20-bit data-address aliases. Fresh text assembly remains canonical; untested absolute-memory families and synthetic vectors remain fail-closed rather than promoted by analogy. Register-immediate `MV X/Y/U/S,lmn` separately discards bits 23-20 |
| Reset-vector selection | Python read the interrupt vector at `FFFFA` | Read the distinct reset-vector slot at `FFFFD`. Software-RESET capture verifies low-first reads through `FFFFF`, no system-stack frame, and the first target fetch. Detailed SFR/general-register retention remains manual-derived because reset ROM initialization starts immediately |
| Vector-transfer and frame ordering | IR/IRQ paths fetched the interrupt vector before constructing the frame, while some runtimes decoded a discarded fall-through instruction or omitted its fetch | For software `IR` and asynchronous IRQ delivery, silently validate the vector/destination first, then perform the measured bus sequence: one opcode-byte fetch for asynchronous delivery without operand decode, five independently wrapped frame writes (PC high/middle/low, F, IMR), and exactly one low-to-high `FFFFA..FFFFC` architectural vector fetch. A failed or changed post-frame vector poisons the machine because the hardware-visible frame is already committed. RESET remains distinct and fetches/validates `FFFFD..FFFFF` before reset mutation. The HALT/OFF wake step, masked status, active-handler nested check, and timer-only pass do not speculatively read the IRQ vector |
| Raw matrix KEYI | Python and Rust tied KEYI assertion/reassertion to debounced events, FIFO contents, host enable policy, and sometimes `in_interrupt` | Sample selected physical KIL independently of debounce/repeat/FIFO bookkeeping. `KIL != 0` asserts or reasserts `ISR.KEYI` regardless of `IMR`, host event generation, or handler-in-service state; release lowers KIL without acknowledging an existing status bit, and a firmware clear remains clear after release. FIFO-only/input-event injection cannot manufacture raw KEYI. Nested delivery remains deferred |
| Timers during interrupt service | Generic Rust and several Python timing paths froze MTI/STI phase while `in_interrupt` | Continue MTI/STI phase advancement and status latching while a handler is active. Preserve pending status but defer nested delivery until service returns and the normal mask checks select it |
| Software interrupt `IR` | The saved PC pointed after the opcode | Save the `IR` opcode address; the ROM dispatcher tests `FE` there and advances the saved frame |
| Stacked `F` byte | Rust preserved six opaque upper bits while Python modeled only carry/zero; `POPS F` advanced `S` before its lazy validation node; an initial quarantine also rejected arbitrary `POPU F`/`RETI` input | PC-E500 matrices establish `POPU F`, `POPS F`, and `RETI` as `F = raw & 03`, followed by their normal pointer/frame advance. Pushes and interrupt entry can therefore emit only the normalized architectural C/Z byte. Keep the one-byte frame field and snapshot each input once |
| `TEST` | Some LLIL paths used a 24-bit operation for a byte instruction | Use byte-width logic |
| `ADC`/`SBC`, `ADCL`/`SBCL` | Carry/borrow propagation could use an unbounded intermediate | Apply width wrapping at each arithmetic stage |
| `EXL` | Only part of the `I`-byte exchange was modeled | Exchange exactly `I` bytes, with independent wrapped internal pointers |
| `MVL` family | Several Rust composite forms copied one extra byte or only one byte | Execute exactly `I` ordered transfers, update both pointers, then clear `I` |
| Decimal shifts | Direction, pointer walk, and carried nibble disagreed | `DSLL` starts at the LSB and decrements; `DSRL` starts at the MSB and increments |
| Wide memory access | A word/pointer could spill outside 8-bit IMEM or 20-bit external memory, an overlapping store could mutate the register used to calculate later destination bytes, and bridge buses could collapse a wide access into one callback | Fix effective bases before writes; fixed-width moves also fix their source values, while `EXP` applies its separately documented ordered byte-exchange semantics. Expand every wide load/store into ordered wrapped byte operations so hardware/host hooks run once per byte. Device captures verify wrapped IMEM byte mapping and external read bus order for the tested `MVW`/`MVP` and `[X++]` forms. Read-only CE1 overlap traces further verify that direct and both displaced `F1`/`F2`/`F3` modes fix their initial effective source and read two/three ordered external bytes from it before overlapping writes. An address-only write trace verifies all direct and displaced `F8`/`F9`/`FA`/`FB` modes select the expected effective destination and emit exactly one/two/three/I low-to-high write phases. These traces exposed and now prevent Rust's former non-boundary wide-callback collapse. The loaded gateware sampled and read every write as zero, so exact write data, partial visibility, boundary order, and invisible IMEM temporal order remain model contracts |
| `CALL`/`CALLF` LLIL | The lifter performed the explicit wrapped stack-frame writes but represented the control-flow edge as a jump | Keep the explicit architectural frame writes and emit a Binary Ninja call edge; the mock evaluator treats that call node as control flow only so it cannot add a second synthetic frame |
| Interrupt-frame boundary access | Synthetic delivery paths could read or write a five-byte system frame past the 20-bit external bus when `S < 5` | Snapshot the frame inputs and decrement/wrap each byte independently on the 20-bit bus. A real S=5..0 matrix verifies PC-high/middle/low, F, IMR time order, modulo-20-bit addresses, and completion of all five writes before the vector read |
| External pointers and control flow | Some paths advanced in a 24-bit space, while LLIL register consumers could retain a high nibble that the runtime facade masks | Mask effective addresses, pointer updates, register jumps, and far-return targets to 20 bits; near control flow explicitly widens its 16-bit component before page composition |
| `PMDF` | Implemented as packed BCD | Use 8-bit wrapping binary pointer addition and preserve incoming `C`/`Z`; both immediate and A-source forms, wrap/zero cases, and deliberately contradictory incoming flags are verified on a PC-E500 |
| `TCL` | Could execute as a silent no-op | Device traces establish independent main/sub divider-phase restart selected by `LCC.MTCL`/`LCC.STCL`, without clearing LCC or already-latched ISR; require the timer-phase-clear hook and fail closed if a host cannot implement it |
| `I=0` counted block instructions | Empty-block and 65,536-iteration interpretations were both plausible | Device matrices establish 65,536 do-while iterations for `ADCL`/`SBCL`/`DADL`/`DSBL`/`MVL`/`MVLD`/`EXL`/`DSLL`/`DSRL`; both cores implement the effective count without unbounded host looping |
| `WAIT` | Timing could be omitted by direct LLIL evaluation; irrelevant PRE could reach a different path; `I=0` was formerly quarantined without evidence | Use a WAIT intrinsic/decoded fast path with an effective count of `I` when nonzero and 65,536 when zero, fail closed without a working timing implementation, and reject PRE+WAIT as noncanonical. Device probes establish countdown length, `I=0`, all architectural C/Z images, the zero-count wrap, and interrupt-atomic execution with pending delivery at the next boundary after the fall-through fetch. Mapping host units to oscillator time remains machine timing, not an alternate instruction result |
| Scheduler preflight | Device wrappers could advance timers before discovering that `TCL`, a quarantined `I=0` counted block operation, an unfused/consecutive PRE, or another invalid encoding had been reached | Decode and validate the pending instruction before cycle, timer, or ISR mutation attributable to scheduling it; preflight the replacement PC again after interrupt delivery, and repeat validation in each backend so direct callers also fail closed |
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
value. Text assembly still emits only canonical values. A later U probe found
the same low-20-bit observable result through both comparison and U-to-Y/push
consumers. A follow-up Y/S matrix found the same result through discriminating
D7 comparisons and copy/serialization, so all four X/Y/U/S architectural
register consumers are covered. The physical point at which bits 23-20
disappear is unobservable. These results must not be generalized to `JPF`,
`CALLF`, external-address operands, or interrupt vectors without their own
probe.

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
the flattened image remain allowed. Snapshot metadata is refreshed from the
live fallback-range and read-only-range configuration; the legacy `fast_mode`
JSON member is always false because CoreRuntime has one execution path. Load
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

IRQ-boundary parity remains an emulator-integrity follow-up for source-specific
peripheral edges. Hardware establishes HALT/OFF fall-through resume,
interrupt-atomic WAIT, the asynchronous discarded-opcode fetch, complete
frame-before-vector order with 20-bit stack wrapping, raw selected-KIL KEYI
re-latching (including in-service), and continued MTI/STI latching in a
handler. ON-key, external, and SIO edge latency still benefit from small
cross-runtime traces; those are peripheral-scheduler questions rather than
unresolved instruction semantics.

## Explicit model contracts

These behaviors are deliberately testable, but a passing test must not be
described as silicon proof:

- `WAIT` requests an effective host idle count of `I` when nonzero and 65,536
  when zero, in addition to the opcode step in the emulator. Countdown length,
  clearing I, architectural C/Z preservation, the zero-count wrap, and
  interrupt-atomic delivery after the fall-through fetch are hardware-backed.
  Mapping one host idle unit to oscillator time remains a machine-timing
  contract. Both backends make one bounded timing-hook request rather than
  simulating an unbounded host loop.
- External/power-on reset and software opcode `FF` are not treated as identical
  reset causes. Device capture proves software FF's low-first
  `FFFFD..FFFFF` vector fetch, no interrupt frame, and first target fetch.
  UCR/USR/ISR/SCR/SSR/LCC mutations and general-register/flag retention remain
  manual-derived because reset-ROM initialization begins immediately.
- HALT and OFF remain distinct power states even where a host API exposes a
  common `halted` boolean. Device capture proves exact fall-through resume,
  STI wake after HALT, and ordinary ON/BREAK wake after ROM-prepared OFF. The
  Python/PCE policy that OFF wakes only for ONKI and HALT can wake for any
  asserted ISR source is the deterministic peripheral integration contract for
  source combinations not directly exercised on the connected unit.
- An isolated PC-E500 matrix covers exact aliases and both ±1 directions for
  `MVW`, `MVP`, `MVL`, `MVLD`, `EXL`, and `EXP`. Fixed-width move final states
  are snapshot-equivalent; counted moves show direction-specific cascades;
  and `EXL`/`EXP` are sequential-byte-exchange-equivalent. Overlapping `EXP`
  contradicts whole-triple snapshot behavior. Counted I clears, fixed-width I
  and seeded `F=03` survive, and comparative total-run timing matches within
  one tick. Final state does not establish invisible IMEM micro-order;
  tested BP-relative `MVW`/`MVP` self-overlaps also retain snapshot-equivalent
  source and initial-destination selection after BP is overwritten, while
  counted companions fix the tested BP, PX, BP+PX destination bases and PY
  source base before iteration. A later read-only CE1 trace decisively verifies
  that direct and both displaced `F1`/`F2`/`F3` modes fix their initial
  effective source pointer and read two/three ordered bytes from it before
  overlapping writes. A companion CE1 trace verifies every direct and
  displaced `F8`/`F9`/`FA`/`FB` effective destination and its
  one/two/three/I ascending write-address sequence. These captures exposed
  Rust's former non-boundary wide-callback collapse; wide loads/stores now
  expand to ordered wrapped bytes. The loaded gateware could not expose the
  driven byte values, so write data and partial visibility remain model
  contracts. Read-side
  IMEM/external boundary wrapping is separately device-verified; write-side
  boundary order remains model-only.
- `F` occupies one byte in stack/interrupt frames, but only `C` and `Z` are
  architectural. `POPU F`, `POPS F`, and `RETI` are hardware-verified to
  normalize arbitrary stack bytes with `raw & 03`; locally generated pushes
  and interrupt frames therefore emit only `0..3`.

## Quarantined or hardware-trace work

Strict v4 snapshots, host/native shadow synchronization, callback rollback,
O(1) host-timer arithmetic, and parity trace validation are emulator-integrity
requirements.  They do not need to be promoted to ISA facts, and they do not
resolve any silicon question below.

The paired private repository's 2026-09-01 closure record concludes that no
remaining hardware row selects between competing architectural descriptions of
a valid implemented opcode. The following residual work remains intentionally
outside that closure:

1. Reserved `20`/`BF`, reserved register selectors, and malformed mode/PRE
   encodings. Device probes pin the accepted PRE aliases/boundary and show that
   displacement-shaped `E3` is not a documented alias. All other malformed
   cases remain fail-closed; silicon behavior here is not a valid-ISA claim.
2. Synthetic interrupt/reset vectors with nonzero bits 23–20 and untested
   absolute-memory opcode families. `JPF`/`CALLF` are resolved for every upper
   nibble, and the named data-address families are resolved for tested nibble
   `8`; untested consumers are not promoted by analogy.
3. External partial-write visibility and invisible IMEM micro-order. Current
   gateware establishes write addresses/count/order but not trustworthy write
   data. Architectural values and effective-base rules are already covered by
   device round trips and the documented instruction contract.
4. Detailed software/external/power-on RESET SFR retention. Software FF's
   vector order, no-frame transfer, and target fetch are device-verified; reset
   ROM code immediately overwrites SFRs, so remaining retention stays labeled
   manual-derived.
5. ON-key, external/SIO peripheral-ready, debounce/repeat, and exact
   asynchronous latency questions. Raw matrix KEYI and MTI/STI-in-handler
   latching are resolved; the remaining items belong to device scheduling and
   peripherals, not alternate meanings of HALT/OFF/WAIT/RETI.
6. Preserve equivalent raw artifacts for any older cited report that still has
   only a decoded event table. This is evidence reproducibility, not a silicon
   semantic question.

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

The paired private ROM-evidence repository contains exact decoder commands,
addresses, raw campaign artifacts, and the canonical hardware trace queue.
Those ROM observations establish use-site intent; only rows backed by the
archived device campaign or follow-up are described above as
real-device-derived.
