# Phase 8 Plan & Status

Phase 8 must land every “hard” semantic family so SCIL fully mirrors the legacy lifter/emulator: control & stacks, looped transfers, carry-chain loops, packed‑BCD ops/shifts, and system-control instructions. Below is the original scope with current completion markers.

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
| Decoders/pilots for 0xCB/0xCF IMEM↔IMEM | ❌ Not started | Need decode map + dispatcher entries and builders. |
| `loop_move` SCIL effect + specs/binders for each addressing form | ❌ Not started | Must cover IMEM↔IMEM, IMEM↔[lmn], IMEM↔[r3±n], IMEM↔[(n)], EMEM counterparts. |
| Compat LLIL lowering (TempMvlSrc/Dst, `lift_loop`-style labels, PRE handling) | ❌ Not started | Must preserve MockLLIL shape, IMEM wrapping, signed stride (+1/-1), pointer side-effects. |
| PyEMU interpreter + future Rust emitter | ❌ Not started | Needs identical semantics (I register countdown, PRE single-use, address wrap). |
| Regression/property tests | ❌ Not started | Shape tests per addressing class, behavior tests (I=0, wraparound, PRE), Phase 6 property coverage. |

---

## 3. Loop Arithmetic & Packed‑BCD

| Feature | Status | Notes |
| --- | --- | --- |
| `ADCL/SBCL` (`loop_add_carry` / `loop_sub_borrow`) | ❌ Pending | Requires new effect + backend/emu/test coverage. |
| `DADL/DSBL` packed‑BCD loops | ❌ Pending | Need nibble-aware effect; Z flag from final byte. |
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
