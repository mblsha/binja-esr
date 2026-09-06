# Rust interactive-emulator responsiveness

Status: active, staged implementation. Applies to the PC-E500 and IQ-7000
Rust machines, native terminal frontend, and browser/WASM frontend. Python
emulator implementation is explicitly outside this work's scope.

## Contract

CPU throughput is not a responsiveness guarantee. A guest ROM may ignore a
button or wait for an unsupported peripheral, but host controls must remain
usable and explain what is happening. Do not force interrupts, skip ROM loops,
stub device calls, or silently reset the machine to simulate responsiveness.

Keep three distinct coordinates: machine-relative timing units, host monotonic
execution budgets, and display presentation cadence. Interactive pacing is not
proof of physically calibrated instruction timing.

## Deliverables

1. **Bounded execution foundation:** shared Rust `CoreRuntime::run_slice`,
   native terminal adoption, opt-in expensive loop diagnostics, and deterministic
   chunking regressions. Implemented and verified locally on 2026-09-06;
   browser adoption and end-to-end response guarantees remain pending.
2. **Browser control lifecycle:** bounded normal/explicit/script execution,
   acknowledged Pause, finite request lifetimes, startup/load serialization,
   failure handling, and isolation of user JavaScript from machine ownership.
3. **Reliable input:** correct release/repress/focus-loss behavior, complete
   model-specific controls, and distinct raw-contact versus timed synthetic-tap
   contracts. Never conflate scheduler boundaries with retired instructions.
4. **Pacing and presentation:** explicit interactive/turbo/deterministic modes,
   bounded catch-up policy, latest-frame delivery, and cheap normal-play status.
5. **Qualification:** actual browser input against both ROMs; HALT/OFF/wake,
   long counted instructions, busy loops, slow execution, script cancellation,
   tab focus/background and reload stress; architectural chunking/bus-order
   checks; measured foreground control latency on a declared reference host.

Land these as focused local commits. No PRs or hardware operations are part of
this goal without a separate user request. Do not describe the whole goal as
complete merely because an individual stage passes.

## Shared bounded-run API

`run_slice(boundary_budget, should_yield)` submits batches of at most 64
scheduler boundaries to the existing machine scheduler. The host predicate is
checked before the first batch and between batches, using counters only. It may
check a deadline or cancellation flag, but must not access the emulated bus.
Yielding neither changes the guest clock nor delivers an interrupt.

The result distinguishes a completed boundary budget from a host-requested
yield. It reports submitted boundary budget, retired instructions, and the
relative timing-counter delta separately. Inert OFF execution can consume the
submitted budget without advancing that counter; the IQ-7000's independent OFF
RTC is also not represented by the CPU timing-counter delta.

This is cooperative scheduling, **not hard real-time preemption**. One long
instruction or synchronous host callback can exceed the deadline. No partial
instruction, write sequence or interrupt entry is split. Browser callers still
need an actual event-loop yield to receive messages; an `async` wrapper around a
synchronous call is insufficient. Existing raw `step` and Function Runner calls
are not automatically bounded by adding this API.

The terminal uses a 4 ms host target checked between these batches; expensive
instruction-history/loop detection requires `--loop-diagnostics`. Terminal
rendering, input work and synchronous I/O are not yet covered by that target.
An explicit `--loop-report PATH` also enables loop detection.

## First-stage validation (2026-09-06)

- `cargo test --offline --manifest-path sc62015/core/Cargo.toml --all-targets
  --all-features --quiet`: 547 passed, six existing ignored tests and one new
  explicitly opt-in private-ROM comparison.
- `cargo clippy --offline --manifest-path sc62015/core/Cargo.toml --all-targets
  --all-features -- -D warnings`: passed.
- Seven focused core regressions cover no-work/no-bus yield, resume, chunk-size
  equivalence for both models in RUN/HALT/OFF, honest budget/time counters,
  fail-closed error propagation, and IRQ discarded-fetch/frame preservation.
- The opt-in two-ROM comparison was explicitly run in release mode and passed:
  one million submitted scheduler boundaries per machine, comparing direct
  execution to host-deadline slices. CPU registers, RAM, timer deadlines, RTC
  status, power state and actual LCD pixel buffers matched. Both ROMs drew
  nonblank displays. This is startup/chunking evidence, not app-input coverage.
- Indicative native timings on this macOS host: PC-E500 direct 9.64 ms versus
  sliced 13.78 ms (mostly idle); IQ-7000 direct 317.81 ms versus sliced 321.89 ms.
  Maximum observed slices were 4.005 ms and 4.051 ms respectively. These short
  measurements are not a performance guarantee, browser responsiveness result,
  physical clock calibration, or proof of worst-case counted-instruction cost.

Reproduce the explicit ROM comparison with both licensed ROM files available
through the usual `data/` links:

```bash
cargo test --offline --release --manifest-path sc62015/core/Cargo.toml \
  --test run_control_rom -- --ignored --nocapture
```

## Exit criteria still to establish

- Record foreground input-application and Pause acknowledgement latency
  distributions (initial p99 Pause target: under 50 ms on a reference host).
- Distinguish input applied to the matrix from input consumed by firmware.
- Never leave a control promise pending indefinitely after a worker failure.
- Verify identical architectural results for the same emulated-time input
  schedule across chunk sizes; do not inject control polling as bus reads.
- A hard worker abort, if needed, must explicitly warn that unsaved state is
  lost; it is not an automatic recovery or a successful graceful Pause.

These are targets, not measured guarantees or claims that historical freezes
have all been reproduced.
