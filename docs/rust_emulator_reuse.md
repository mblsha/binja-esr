# Rust Emulator Reuse Plan (No SCIL)

What can be reused for a full-Rust emulator that bypasses SCIL, and what cannot.

## Reusable as-is or with light adaptation

- Memory & addressing
  - `sc62015/core/src/memory.rs`: `MemoryImage`, address helpers, readonly/fallback ranges.
  - Address masks/constants in `sc62015/core/src/lib.rs` (e.g., `INTERNAL_MEMORY_START`, `ADDRESS_MASK`).
- Devices
  - `sc62015/core/src/lcd.rs`: LCD controller, VRAM, `display_buffer` export.
  - `sc62015/core/src/keyboard.rs`: keyboard matrix handling.
  - `sc62015/core/src/timer.rs`: timer scaffolding (if kept).
- Tracing/telemetry
  - `sc62015/core/src/perfetto.rs`: Perfetto tracer for instruction/memory instants/counters.
  - Event schema patterns from `pce500` tracing glue (if matching dispatcher/Perfetto).
- Snapshots
  - `sc62015/core/src/snapshot.rs`: archive format, range deserialization, register packing/unpacking (adjust registers if model changes).
- Glue/runtime scaffolding
  - HybridBus/host-device wiring patterns in `sc62015/core/src/exec.rs` (minus SCIL eval hook).
  - Perfetto emission patterns in `pce500/emulator.py` as reference for payload shapes.
- Build/interop
  - pyo3 glue structure in `sc62015/rustcore/src/lib.rs` (for a Python extension), minus SCIL payload loading.
  - CLI scaffolding in `sc62015/rustcore/src/bin/pce500-cli.rs` (arg parsing, snapshot load/save, tracing hooks), swapping in your core execute step.

## Reusable concepts but needs refactor

- Opcode lookup scaffolding (`OpcodeLookup`, `OpcodeIndexView`) in `sc62015/core/src/exec.rs`: map/retrieval pattern is useful; source data must come from your new decoder/dispatch table.
- Perfetto event schemas: tracer is reusable; adjust annotations to your core.
- Snapshot register packing: current layout matches SCIL regs (A/B/BA, I/IL/IH, etc.); update if your register model differs.

## Not reusable (SCIL-bound)

- SCIL interpreter/AST: `emulators/rust_scil` (state/eval/ast) and generated payload (`handlers.rs`, `manifest.json`, `opcode_index.rs`).
- SCIL decode/codegen: `tools/scil_codegen_*`, `sc62015/decoding/*`, `sc62015/scil/*` bound_repr/serde.

## Summary

Keep the device layer (memory, LCD, keyboard), snapshot container/format (tweak registers as needed), tracing infrastructure, bus/device wiring patterns, and pyo3/CLI scaffolding. Replace SCIL decode/eval and generated payloads with your new core’s decode/execute, reusing the surrounding runtime and tooling.
