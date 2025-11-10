# Phase 9 Plan & Status

Phase 9 retires the legacy lifter so SCIL is the sole source of truth from decode to LLIL/emulators. We keep the Compat backend for shape parity, add a one-release rescue flag, and make the validation rules fatal. This doc tracks the original goals vs. current status.

---

## Plan Audit

| Item | Status | Notes |
| --- | --- | --- |
| Remove BoundStream + handwritten emitters | ✅ Done | Replay helpers/tests deleted; `compat_il.emit_instruction` now wraps SCIL. |
| SCIL-only dispatch in `arch.py` | ✅ Done | All instructions decode through `CompatDispatcher`; SCIL emit is mandatory; `BN_ALLOW_LEGACY` is the only fallback hook. |
| Config/env cleanup | ✅ Done | Deprecated `BN_USE_SCIL`/allow/block/family flags in favor of `BN_ALLOW_LEGACY`; tests updated. |
| Docs: rollout guidance & rescue flag | ✅ Done | `AGENTS.md` describes the new default + telemetry counters. |
| Property/CI hardening (shape snapshots, nightly prop gate) | ✅ Completed | Property harness now uses a preserved legacy snapshot emitter (`tests/prop/legacy_compat_il.py`) and continues to gate shapes + behavior. |
| Perf + telemetry guards | ⛔ Outstanding | Need CI perf check and documentation of counter expectations. |

---

## Completed Work

1. **Legacy entry points removed**
   - `sc62015/decoding/replay.py` and `tests/decoding/test_replay.py` deleted.
   - `sc62015/decoding/compat_il.py` now routes through SCIL for every opcode.
2. **SCIL-only dispatcher**
   - `CompatDispatcher` decodes every opcode (PRE latch still honored).
   - BinaryNinja arch always uses SCIL; counters renamed (`scil_ok`, `scil_error`, `legacy_rescue`).
3. **Rescue flag + docs**
   - `BN_ALLOW_LEGACY=1` temporarily re-enables the legacy lifter; warnings + counters make it obvious.
   - `AGENTS.md` updated with the new rollout guidance.

---

## Remaining Work

1. **Perf/telemetry**
   - Add perf guards (fixed corpus lift time) and document `scil_ok/scil_error/legacy_rescue` expectations.
2. **Release notes + cleanup**
   - Finalize release messaging (legacy removed, rescue flag removal date).
   - Ensure CI no longer exports deprecated env vars.

Once the above are complete—and after one release window with the rescue flag—the legacy path can be removed entirely with confidence.
