# Phase 8 Plan & Status

Phase 8 must land every “hard” semantic family so SCIL fully mirrors the legacy lifter/emulator: control & stacks, looped transfers, carry-chain loops, packed‑BCD ops/shifts, and system-control instructions. Below is the original scope with current completion markers, aligned with the original plan so nothing gets dropped.

---

## Plan Audit (high-level goals → status)

| Plan Item | Status | Notes |
| --- | --- | --- |
| Control/stack effects (CALL*/RET*/PUSH*/POP*/JPF/interrupts) | ✅ Completed | Specs, effect nodes, compat LLIL + PyEMU covered; RETI/interrupt exit semantics implemented. |
| Looped transfers (`MVL/MVLD` & addressing variants) | ✅ Completed | All addressing forms decoded/spec’d/bound; compat LLIL + PyEMU honor PRE + wrapping semantics. |
| Carry-chain loops (`ADCL/SBCL`) | ✅ Completed | Effects/backends/tests landed for mem+reg variants. |
| Packed‑BCD loops & decimal shifts (`DADL/DSBL`, `DSLL/DSRL`, `PMDF`) | ✅ Completed | Decimal-aware effects plus tests + parity checks done. |
| System control (`HALT/OFF/RESET/WAIT/IR`) | ✅ Completed | Effects with IMEM side-effects wired through decode/spec/backends + PyEMU tests. |
| Test/property expansion + Rust parity | ✅ Completed | Property harness now includes loop/carry/BCD/system invariants and Rust parity tests cover them. |

---

## 1. Control / Stack Effects

| Feature | Status | Notes |
| --- | --- | --- |
| `CALL`, `CALLF`, `RET`, `RETF`, `RETI`, `JPF` (ret pushes, page joins) | ✅ Done | SCIL specs + compat LLIL + PyEMU match legacy MockLLIL shapes (OR.l joins, little-endian stack stores). |
| System stack `PUSHS/POPS` variants | ✅ Done | Covered earlier; revalidated during Phase 8 work. |
| User stack `PUSHU/POPU` for A/IL/BA/I/X/Y/F/IMR | ✅ Done | Decoders, specs, `push_bytes`/`pop_bytes` effects, LLIL+PyEMU parity, regression tests in `tests/scil_phase3`. |
| Remaining call/control ops | ⚪ Pending review | Once system-control effects land, re-audit `JPF`/interrupt enter semantics to ensure no gaps. |

---

## 2. Looped Transfers (`MVL/MVLD` + variants)

| Feature | Status | Notes |
| --- | --- | --- |
| Decoders/pilots for 0xCB/0xCF IMEM↔IMEM | ✅ Done | Decode map + dispatcher entries + builders landed. |
| `loop_move` SCIL effect + specs/binders for each addressing form | ✅ Done | All IMEM↔IMEM/external variants emit loop_move with signed strides + PRE-aware bindings. |
| Compat LLIL lowering (TempMvlSrc/Dst, `lift_loop`-style labels, PRE handling) | ✅ Done | Matches legacy shapes for every addressing family, including wraparound + PRE consumers. |
| PyEMU interpreter + future Rust emitter | ✅ Done | Python + Rust interpreters both handle all looped transfers with shared semantics. |
| Regression/property tests | ✅ Done | Behavior suites + property harness cover IMEM/EXT permutations and invariants. |

---

## 3. Loop Arithmetic & Packed‑BCD

| Feature | Status | Notes |
| --- | --- | --- |
| `ADCL/SBCL` (`loop_add_carry` / `loop_sub_borrow`) | ✅ Done | SCIL effects + compat LLIL + PyEMU for IMEM↔IMEM + `(m),A`; tests cover carry/borrow cases. |
| `DADL/DSBL` packed‑BCD loops | ✅ Done | SCIL BCD effects + PyEMU/LLIL parity with tests for mem/reg forms. |
| `DSLL/DSRL` decimal shifts | ✅ Done | Decimal shift effect wired through SCIL + tests for both directions. |
| `PMDF` | ✅ Done | New SCIL effect + decode/builder + tests. |

---

## 4. System Control

| Feature | Status | Notes |
| --- | --- | --- |
| `HALT`, `OFF`, `RESET`, `WAIT`, `IR` (interrupt enter/exit) | ✅ Done | Specs + effects emit IMEM side-effects, stack order, vector jumps, and PyEMU tests verify behavior. |
| Compat LLIL + PyEMU + Rust parity | ✅ Done | All backends share the same stack/IMEM side-effects; Rust CLI tests exercise HALT/OFF/RESET/IR. |

---

## 5. Testing & Tooling

| Requirement | Status | Notes |
| --- | --- | --- |
| Dedicated suites (`tests/effects_*`, `tests/loops_bcd_*`) | ✅ Done | Behavior suites for loop/carry/BCD/system effects live under `tests/scil_phase8`. |
| Phase 6 fuzz harness extensions (loops/BCD/system) | ✅ Done | Prop generators emit looped transfers, carry-chain ops, packed-BCD data, and system-control opcodes with invariants. |
| PyEMU ↔ Rust parity for new effects | ✅ Done | CLI parity suite now includes MVL/ADCL/DADL/DSLL/HALT/IR cases ensuring Rust matches PyEMU. |

---

---

## Files / Components Checklist (from original plan)

| Component | Status | Notes |
| --- | --- | --- |
| `sc62015/scil/effects.py` definitions | ✅ In place | Control/stack + loop/BCD/system effect variants defined per plan. |
| `sc62015/scil/specs/*` extensions | ✅ Done | Specs now cover all Phase 8 families (looped transfers, carry-chain, BCD, system control). |
| `sc62015/scil/from_decoded.py` bindings | ✅ Done | Builders bind every new family; PRE-aware wiring preserved. |
| `backend_llil_compat.py` lowering | ✅ Done | Lowers all new effects with legacy shapes, incl. interrupt stack order + HALT/OFF intrinsics. |
| `pyemu` effect eval | ✅ Done | Python interpreter handles control/stack + loop/BCD/system behavior incl. IMEM side-effects. |
| `emulators/rust_scil` effect eval | ✅ Done | Interpreter handles control/stack + loop/BCD/system effects with matching semantics. |
| Tests (`tests/effects_*`, `tests/loops_bcd_*`) | ✅ Done | Suites under `tests/scil_phase8/` exercise loops/carry/BCD/system instructions. |
| Property harness updates | ✅ Done | Strategies generate looped transfers, carry/BCD ops, and system-control sequences with invariants. |
| Rust parity tests | ✅ Done | CLI suite compares Rust vs PyEMU across new effect families. |

---

## Invariants (remain enforced)

- `PC` masked to 20 bits; external addresses 24-bit; internal addresses 8-bit; PRE applies once.
- Paged control flow uses `(PC & 0xF0000) OR lo16`; JR family uses `PC+len ± sext8`.
- Completed control/stack work preserves these invariants and existing MockLLIL shapes.

---

## Immediate Next Steps
✅ Phase 8 and its gating/telemetry docs are complete. Next up: Phase 9 (SCIL-only lifter) planning.
