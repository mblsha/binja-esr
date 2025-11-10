# Phase 8 Plan & Status

Phase 8 must land every “hard” semantic family so SCIL fully mirrors the legacy lifter/emulator: control & stacks, looped transfers, carry-chain loops, packed‑BCD ops/shifts, and system-control instructions. Below is the original scope with current completion markers, aligned with the original plan so nothing gets dropped.

---

## Plan Audit (high-level goals → status)

| Plan Item | Status | Notes |
| --- | --- | --- |
| Control/stack effects (CALL*/RET*/PUSH*/POP*/JPF/interrupts) | ✅ Completed | Specs, effect nodes, compat LLIL + PyEMU covered; RETI/interrupt exit semantics implemented. |
| Looped transfers (`MVL/MVLD` & addressing variants) | ⛔ Outstanding | No decode/spec/backend work yet; remains top priority. |
| Carry-chain loops (`ADCL/SBCL`) | ⛔ Outstanding | Effects/tests/backends pending. |
| Packed‑BCD loops & decimal shifts (`DADL/DSBL`, `DSLL/DSRL`, `PMDF`) | ⛔ Outstanding | Need nibble-aware semantics everywhere. |
| System control (`HALT/OFF/RESET/WAIT/IR`) | ⛔ Outstanding | Effects + IMEM side-effects still to implement. |
| Test/property expansion + Rust parity | ⛔ Outstanding | Dedicated suites + prop harness updates + Rust backend support for new effects still missing. |

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
| `loop_move` SCIL effect + specs/binders for each addressing form | 🟡 Partial | IMEM↔IMEM effect/specs live; external/int-mixed forms still pending. |
| Compat LLIL lowering (TempMvlSrc/Dst, `lift_loop`-style labels, PRE handling) | 🟡 Partial | IMEM↔IMEM lowering matches legacy shapes; remaining addressing modes TBD. |
| PyEMU interpreter + future Rust emitter | 🟡 Partial | PyEMU handles IMEM↔IMEM loops; Rust backend still missing coverage. |
| Regression/property tests | 🟡 Partial | Shape + basic behavior tests for IMEM↔IMEM added; PRE/other variants + fuzzing outstanding. |

---

## 3. Loop Arithmetic & Packed‑BCD

| Feature | Status | Notes |
| --- | --- | --- |
| `ADCL/SBCL` (`loop_add_carry` / `loop_sub_borrow`) | ✅ Done | SCIL effects + compat LLIL + PyEMU for IMEM↔IMEM + `(m),A`; tests cover carry/borrow cases. |
| `DADL/DSBL` packed‑BCD loops | ✅ Done | SCIL BCD effects + PyEMU/LLIL parity with tests for mem/reg forms. |
| `DSLL/DSRL` decimal shifts | ❌ Pending | Effect must shift decimal digits (directional, zero fill). |
| `PMDF` | ❌ Pending | Single-byte packed modifier still to encode as effect. |

---

## 4. System Control

| Feature | Status | Notes |
| --- | --- | --- |
| `HALT`, `OFF`, `RESET`, `WAIT`, `IR` (interrupt enter/exit) | ❌ Pending | Must encode IMEM side-effects (USR/SSR/UCR/etc.), PC updates, emulator state flags. |
| Compat LLIL + PyEMU + Rust parity | ❌ Pending | Emit exact MockLLIL shapes asserted by tests; emulator needs matching semantics. |

---

## 5. Testing & Tooling

| Requirement | Status | Notes |
| --- | --- | --- |
| Dedicated suites (`tests/effects_*`, `tests/loops_bcd_*`) | ❌ Pending | Currently only Phase‑3 shape tests exist. Need richer behavior/property coverage. |
| Phase 6 fuzz harness extensions (loops/BCD/system) | ❌ Pending | Must add generators for loop counts, packed-BCD data, HALT/OFF sequences; nightly corpora need to stay green. |
| PyEMU ↔ Rust parity for new effects | ❌ Pending | Python interpreter already handles control/stack; Rust engine needs the same effect set before flipping prod. |

---

---

## Files / Components Checklist (from original plan)

| Component | Status | Notes |
| --- | --- | --- |
| `sc62015/scil/effects.py` definitions | ✅ In place | Control/stack effects landed; loop/BCD/system variants still to add. |
| `sc62015/scil/specs/*` extensions | 🟡 Partial | Control/stack specs merged; need loop/BCD/system specs. |
| `sc62015/scil/from_decoded.py` bindings | 🟡 Partial | Bindings exist for control/stack; loop/BCD/system bindings pending. |
| `backend_llil_compat.py` lowering | 🟡 Partial | Supports control/stack; must add loop/BCD/system effects with legacy shapes. |
| `pyemu` effect eval | 🟡 Partial | Handles call/ret stacks; needs loop/BCD/system logic. |
| `emulators/rust_scil` effect eval | 🟡 Partial | Control/stack not implemented yet; future work must mirror Python semantics. |
| Tests (`tests/effects_*`, `tests/loops_bcd_*`) | ❌ Missing | Need dedicated suites per plan. |
| Property harness updates | ❌ Missing | Strategies/cases for new families pending. |
| Rust parity tests | ❌ Missing | CLI parity must cover new effects once implemented. |

---

## Invariants (remain enforced)

- `PC` masked to 20 bits; external addresses 24-bit; internal addresses 8-bit; PRE applies once.
- Paged control flow uses `(PC & 0xF0000) OR lo16`; JR family uses `PC+len ± sext8`.
- Completed control/stack work preserves these invariants and existing MockLLIL shapes.

---

## Immediate Next Steps
1. **Looped transfers – IMEM↔IMEM (0xCB/0xCF)**  
   - Add decoder/builder/spec entries.  
   - Introduce `loop_move` effect capturing dst/src AddrSpecs, stride (+1/-1), width, PRE data.  
   - Implement compat LLIL/PyEMU support plus regression tests.
2. **Expand loop coverage** to IMEM↔[lmn], IMEM↔[r3±n], IMEM↔[(n)]/EMEM forms.  
3. **Carry-chain & BCD loops** (`ADCL/SBCL`, `DADL/DSBL`, `DSLL/DSRL`, `PMDF`).  
4. **System control effects** (`HALT/OFF/RESET/WAIT/IR`).  
5. **Testing & parity**: add dedicated suites, extend property tests, and keep compat + PyEMU + Rust aligned.

Only once these items are completed and all tests (pytest + property + Rust parity) are green can we declare Phase 8 done and move on to Phase 9 (SCIL-only lifter).
