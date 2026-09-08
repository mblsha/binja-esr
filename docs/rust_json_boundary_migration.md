# Rust JSON boundary migration

Status: implemented and verified, 2026-09-08. Changes await human review; no PR
or commit is implied by this report.

The target is typed emulator state and typed internal snapshots, with JSON
restricted to explicit compatibility, file, and tool boundaries. Serde-derived
Rust structures and direct JavaScript object conversion are not JSON parsing.

## Completed implementation stages

- Physical key definitions generate static Rust tables at build time; the shared
  JSON source remains authoritative for native/browser mappings.
- Screenshot capture copies pixels directly and uses structured cloning for
  metadata rather than stringify/parse.
- IRQ execution histories have no dynamic JSON fallback. Unsupported shapes
  fail import before timer mutation; canonical histories use fixed arrays.
- Interrupt capture/restore has an owned typed representation. Legacy counters,
  last-IRQ information, and history conversion live in `interrupt_codec.rs`.
- LCD controllers capture/restore typed variants, validating before mutation.
  Machine snapshot metadata now holds those typed variants.
- Keyboard state and metrics are typed machine snapshot fields. Native save and
  restore no longer roundtrip those fields through dynamic JSON values.
- Structured Function Runner finish/cancel endpoints exist, alongside unchanged
  JSON compatibility endpoints. Node/WASM and Chromium equivalence tests cover
  both devices, cancellation, ownership and callback re-entry. The browser's
  bounded-call path prefers structured results; old bundles retain a fallback.
- All interrupt machine metadata is typed. The core builds and passes strict
  Clippy without default features or a runtime `serde_json` dependency. JSON
  compatibility methods are feature-gated; snapshot/CLI features opt in.
- Buffered fixed-record bus tracing and an offline JSONL converter are available.
  The existing JSONL option is unchanged. Writer/flush errors reach the command
  result; truncated records and reordered indices fail conversion.

## Verification recorded so far

- Core: 466 tests pass, six ignored; strict all-target/all-feature Clippy passes.
- CLI binaries: 74 tests pass, including snapshot restoration and latched trace
  flush failure. Rust formatting and diff-whitespace checks pass.
- JSON-free core: strict no-default-feature library Clippy passes; the normal
  dependency graph has no `serde_json` (build-host key generation still uses it).
- WASM: 32 Node tests; web: 143 tests. Chromium tests exercise structured result
  conversion and the existing Function Runner aliasing regression.
- Saved optimized WASM baseline versus candidate: 21 IQ-7000 and 11 PC-E500
  checkpoints match registers, memory hashes, LCD, timing and input state.
- PC-E500 legacy diagnostic bus trace: 59,983 events are byte-identical between
  direct JSONL and converted binary output.
- IQ-7000 legacy diagnostic trace: the same comparison passes for 48,051 events.
- Supported v4 snapshots generated using pre-migration commit `1f221338` load
  in the candidate. Identical one-boundary continuations match all seven archive
  entries (JSON compared structurally, binary entries byte-for-byte) on both
  devices. IQ-7000 RTC is explicitly disabled, as required by existing snapshot
  scope. An old v2 artifact remains unsupported, as before.
- The Python compatibility adapter compiles using Python 3.11. This is an adapter
  build check, not a new full Python-emulator parity run.

## Measurements and interpretation

Binary output uses 2,219,379 versus 8,809,060 bytes for the PC-E500 sample, and
1,777,895 versus 6,989,784 for IQ-7000: approximately 75% less output. Conversion
is offline; this does not measure trace-writing CPU speed.

Three real-ROM 445-character MEMO runs each produced median typing times of
16,342 ms for the saved optimized WASM baseline and 16,158 ms for the candidate.
All runs verified save/reopen with identical retired-instruction and timing-unit
counts. These are exploratory sequential measurements on a working host (some
baseline execution overlapped build activity), not a controlled speedup claim.
The observed ~1% difference is insufficient evidence of an execution improvement.
This migration primarily removes representation/copy overhead at boundaries and
makes runtime state statically checkable; it does not change guest timing.

Private local evidence is under `artifacts/dejson/`: `differential`, `trace`,
`bench-base`, `bench-candidate`, and pre-change/current snapshot captures. ROM and
snapshot payloads are deliberately not included in the public repository.

## Compatibility decisions to verify in final review

Unsupported IRQ history schemas are now rejected rather than retained as mutable
JSON during execution. IRQ numeric imports reject overflow rather than truncating.
LCD typed restore rejects wrong geometry and out-of-range chip selectors before
mutation. Keyboard unknown fields remain rejected, including per-key state.
Canonical pre-change v4 files were checked as described above. Deliberately
malformed imports have targeted rejection tests; this is not a promise that
every permissive historical diagnostic object remains accepted.

The Function Runner JSON endpoints remain for external callers and old bundles;
they are no longer the preferred interactive path. Snapshot duplicate-key
rejection and candidate-first restoration remain in place.

## Deliberately retained JSON boundaries

- Snapshot archives keep their existing JSON metadata format. Typed state is
  converted only on file import/export, not while stepping the machine.
- CLI scenarios, symbol maps, capture reports and offline JSONL are file/tool
  interfaces. Python/parity conversion stays in the adapter and compatibility
  feature; this work does not expand the Python emulator.
- Function Runner options and compatibility exports remain tool interfaces.
  The disposable script worker's synchronous shared-memory RPC mailbox uses
  bounded JSON encoding: it is not emulated memory or an instruction-loop path.
  Replacing that protocol needs a separately versioned binary codec, not an
  unsafe switch to asynchronous messaging for synchronous script calls.
- Keyboard debug JSON is formatted only for explicit debug requests. User print
  output is formatted on demand; screenshot copies no longer serialize JSON.

Binary bus tracing remains on the existing legacy diagnostic runtime. It does
not add bus tracing or snapshot support to the shared CLI runtime, nor does it
resolve the existing IQ-7000 RTC snapshot-scope restriction.
