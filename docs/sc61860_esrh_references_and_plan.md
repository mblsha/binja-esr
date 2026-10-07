# ESR-H / SC61860 references and implementation plan

ESR-H is the older instruction family represented by SC61860. The current
plugin and Rust/Python CPU implementations target SC62015 / ESR-L; a public
ESR-H decoder, Binary Ninja architecture and CPU runtime remain to be built.
This page records available references and the work needed to add them.

## Existing references

The links below were reviewed on 2026-10-05. Source links using `master` can
change; pin a full revision and record file hashes before using an external
implementation as a reproducible test oracle. Baseline emulator behavior is
a best guess against real hardware, not independent confirmation of silicon
semantics. The additional ESR-H reference does not replace the project's
existing baseline emulator or its current parity checks.

| Reference | Useful material | Limits / provenance |
| --- | --- | --- |
| [Baseline ESR-H disassembler](https://github.com/mamedev/mame/blob/master/src/devices/cpu/sc61860/scdasm.cpp) and [interface](https://github.com/mamedev/mame/blob/master/src/devices/cpu/sc61860/scdasm.h) | Mnemonics, operand decoding, instruction lengths, relative targets and compact LP/CAL forms | Files identify BSD-3-Clause and Peter Trauner; audit against execution and manuals before adoption |
| [Baseline CPU device](https://github.com/mamedev/mame/blob/master/src/devices/cpu/sc61860/sc61860.cpp) and [state/interface](https://github.com/mamedev/mame/blob/master/src/devices/cpu/sc61860/sc61860.h) | Registers, internal RAM, reset, callbacks, timers and debugger state | Same file-level license; device assumptions require model qualification |
| [Opcode dispatch](https://github.com/mamedev/mame/blob/master/src/devices/cpu/sc61860/sctable.hxx) and [instruction operations](https://github.com/mamedev/mame/blob/master/src/devices/cpu/sc61860/scops.hxx) | Actual execution, flag/register side effects, stack order, block/BCD operations and timing estimates | Implementation evidence; unsupported encodings currently log an error rather than providing a strict execution contract |
| [Baseline pocket-computer driver](https://github.com/mamedev/mame/blob/master/src/mame/sharp/pocketc.cpp) | Model ROM regions, memory maps, keyboard/LCD wiring and system status | GPL-2.0+ file, separately licensed from the CPU files; use as a reference for device work |
| [ESR-H architecture and instruction notes](https://www.oit.ac.jp/labs/rd/rssrv/kobayashi-lab/~yagshi/old_web/misc/pocketcom/sc61860op.html) | Internal/external memory conventions, register map and CALL/JP/PTC/CASE encodings | Researcher's documentation; some sections are incomplete |
| [SC61860 instruction-set research](https://github.com/utz82/SC61860-Instruction-Set) | Opcode/effect table, undocumented-instruction caveats and links to manuals/research | Repository declares CC0-1.0; preserve the distinction between documented facts and author hypotheses |

The CPU, dispatch and disassembler sources above provide both disassembly and
execution references for SC61860. The reviewed
[CPU build registrations](https://github.com/mamedev/mame/blob/master/scripts/src/cpu.lua)
and [standalone disassembler registrations](https://github.com/mamedev/mame/blob/master/src/tools/unidasm.cpp)
contain no SC62015 / ESR-L or LH5806 / ESR-P implementation.

The pocket-computer driver registers PC-1250, PC-1251, PC-1255, PC-1350,
PC-1401, PC-1402 and the TRS-80 PC-3 without a `MACHINE_NOT_WORKING` flag.
PC-1245, PC-1260, PC-1261/1262, PC-1360, PC-1403, PC-1403H and PC-1450 are
marked not working. All are marked no sound. A system registration does not
qualify an organizer host, a card ABI or every ESR-H chip variant; PC-1360's
entry is especially insufficient as an execution oracle for a BASIC card.

## Instruction audit before implementation

Keep ESR-H encoding and state separate from ESR-L. ESR-H's 16-bit external
addressing, P/Q/R internal-memory pointers and compact calls differ from the
existing SC62015 representation. Similar BASIC bytecode or register names do
not establish native-code compatibility.

The references expose concrete questions for the initial opcode audit:

- Absolute CALL/JP/LIDP operands are high-byte first; X/Y register pairs in
  internal RAM use low-byte first. Verify each operand's byte order explicitly.
  The standalone disassembler registers SC61860 with little-endian buffers,
  while its decoder uses `r16` and the execution core reads high-byte first.
  Check a known CALL/JP byte sequence before trusting standalone output.
- The disassembler labels `READM` (`0x54`) and `READ` (`0x56`) as implicit
  one-byte forms; the execution dispatch consumes a following byte. Resolve
  the length and effects before instruction walking or lifting.
- The disassembler describes alternate `LIDP`/`LIDL` encodings at `0x16` and
  `0x17`, but the reviewed execution switch has no corresponding cases.
  Track decode support separately from execution support.
- `0xD3` / WRIT and other undocumented encodings need a chip/profile decision.
  The reviewed CPU dispatch does not implement `0xD3`; its use in an older
  organizer ROM must not silently inherit a no-op interpretation.
- Audit flags, internal pointer updates, overlapping block operations,
  `PTC`/`ETC` tables and packed BCD with focused cases. Check `SC`/`RC` zero
  flag effects as well as carry. Keep timing and port behavior provisional
  until the target machine supplies evidence.

Record conflicting descriptions in an opcode/evidence matrix instead of
choosing the behavior needed to make one ROM continue.

## Work to do

Every item below is pending. Reuse the existing project's trace, test and
frontend infrastructure where appropriate, with a separate ESR-H CPU state.

1. **Pin and audit the references.** Record revisions, hashes and file-level
   licenses. Build a matrix of opcode length, operands, effects, timing,
   variant, evidence and unresolved discrepancies. Obtain the applicable
   primary machine-language/chip documentation for the chosen target.
2. **Add a standalone decoder and Binary Ninja architecture.** Start in a
   separate `sc61860/` module with explicit target/variant selection. Decode
   every first-byte value, truncated operands, compact LP/CAL, relative jumps
   and PTC/ETC table data. Verify high-byte-first addresses using synthetic
   cases such as `78 95 F6` → `CALL 0x95F6` and `79 C0 01` → `JP 0xC001`.
   Add branch/call metadata and instruction text before LLIL; implement and
   test LLIL only for qualified effects. Retain unknown encodings explicitly.
3. **Build bounded CPU execution.** Define internal RAM/register aliases,
   stack and external-memory/port contracts. Add single-step execution with
   instruction budgets and explicit unsupported-opcode errors. Implement
   scalar/control instructions first, then tables, block transfers and BCD.
   Keep a Rust core primary and a Python reference in lockstep, following
   the project's existing parity conventions.
4. **Add differential and hardware qualification.** Compare synthetic
   instruction cases against the pinned baseline CPU reference and stop at
   the first divergence. Include PC/DP/P/Q/R, internal RAM, flags and bus
   effects with explicit pre/post trace semantics. Classify disagreements
   using [the evidence levels](test_evidence_levels.md); agreement between
   software implementations is not hardware proof.
5. **Qualify an older organizer/card profile.** Establish the actual chip,
   host ROM services, selected ROM/RAM banks and card entry ABI before full
   card execution. Use PA-7C18 research as a candidate workload, with capture
   hashes and analytical mappings kept explicit. A repeated captured address
   window does not establish the complete logical ROM or SRAM capacity.
   Resolve WRIT and host-mediated bank helpers before using the card as a
   whole-program acceptance test. Keep ROM/capture fixtures in the paired
   private repository; public tests can use synthetic instruction bytes.
6. **Add machine and BASIC acceptance cases.** Once the host profile is
   qualified, drive normal startup, keyboard/LCD and card activation paths.
   Recover the older BASIC token dictionary, line framing and SRAM layout;
   then test decompilation/recompilation and execution of known programs.
   Matching ESR-L token markers alone does not qualify keyword IDs or saved
   file compatibility. Expose an ESR-H frontend only after its supported
   profile and remaining limits are documented.

The first useful deliverable is a strict decoder plus the evidence matrix.
CPU differential tests are the next milestone; organizer/card integration
depends on the host and banking work, rather than decoder availability alone.
