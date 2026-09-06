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
   browser adoption is also implemented; end-to-end response qualification
   remains incomplete.
2. **Browser control lifecycle:** bounded normal/explicit/script execution,
   acknowledged Pause, finite request lifetimes, startup/load serialization,
   failure handling, and isolation of user JavaScript from machine ownership.
   Normal running, explicit stepping and Function Runner `e.step()` are bounded;
   control acknowledgements, request failure handling and ROM-load generation
   checks are implemented. Function calls and JavaScript isolation remain pending.
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

## Browser controls implemented (2026-09-06)

The WASM binding exposes the same cooperative Rust scheduler with a monotonic
host deadline. Browser normal running, explicit Step, and Function Runner
`e.step()` use a 4 ms target and yield to the worker event loop between slices.
Even repeated tiny awaited steps yield; a chain of resolved promises would not
let the worker receive Stop. Synthetic key taps release their contact in a
`finally` block if stepping fails or is cancelled.

Only one asynchronous execution/load job owns the machine. Stop cancels that
job and waits for ownership to be released before replying. The UI shows
`PAUSE REQUESTED` until that reply, and final display decoding happens after
the acknowledgement. Requests have finite lifetimes; a timeout reports
`UNRESPONSIVE — STATE UNCONFIRMED`, not a successful pause. Worker failure
rejects outstanding requests and explains that replacement loses unsaved state.
WASM initialization is shared, and ROM loads/frames carry generation and model
identity so a delayed old fetch cannot install or display the wrong machine.

Local validation:

- Svelte checks: zero errors/warnings; 68 frontend unit/component tests passed.
- WASM binding tests: 17 passed, including budget validation and sliced/direct
  state equivalence. WASM crate Clippy passed; the WASM dependency build still
  emits 18 existing feature-gated core warnings.
- Full public Chromium suite: nine passed, three opt-in ROM tests skipped.
  The synthetic call fixture now supplies a writable stack; the emulator's
  rejection of a sentinel pushed into ROM was correct and was not weakened.
- Five browser-control tests passed with real-ROM mode explicitly enabled.
  These check control behaviour, not application completeness. The load-race
  test deliberately uses synthetic ROMs in either mode.
- Twenty foreground Run/Stop cycles per model measured actual worker reply
  arrival, not an optimistic UI status. On this macOS/Chromium host the maxima
  were 36.6/45.3 ms (PC-E500/IQ-7000 synthetic fixtures) and 52.3/51.8 ms (real
  ROMs). With only twenty samples, the empirical p99 equals the maximum; this
  is a small smoke measurement, not a statistical latency guarantee. A repeated
  synthetic run peaked at 52.2/52.9 ms, demonstrating host-run variability.
  The observations do **not** consistently meet the initial sub-50 ms target.
  Test attachments include the sample list and host/browser identification.

The public PR workflow runs only three short synthetic browser-control
regressions, excluding the repeated latency measurements; the full browser
suite runs on pushes. Both reuse CI's explicitly built app. Private ROM and
hardware runs are not new mandatory CI requirements.

Reproduce locally from `web/` (the default builds current WASM and Vite first):

```bash
npm run check:ci
npm run test:ci
npm run wasm:test
CI=1 PCE500_E2E_PORT=4197 npm run e2e -- --workers=1
CI=1 PCE500_E2E_PORT=4197 PCE500_E2E_REAL_ROM=1 \
  npm run e2e -- e2e/responsiveness.spec.ts --workers=1
```

The real-ROM invocation requires both licensed files through the usual `data/`
links. A missing ROM is a failure, not permission to substitute a fixture.

### Remaining limitations

- Function Runner `e.call()` still invokes synchronous `call_function_ex`;
  arbitrary user JavaScript still runs in the machine worker. Either can block
  message delivery. Timeouts now describe that truthfully, but do not preempt
  them. Isolated scripts and resumable function calls are required next.
- The no-Worker fallback has bounded stepping but is not a qualified isolated
  production frontend; arbitrary scripts can still freeze its UI.
- Virtual-key release timing, repress/focus-loss handling and model-specific
  controls still need the input stage. Pause latency does not measure a key
  reaching the matrix or being consumed by ROM firmware.
- Rendering/debug snapshots can still delay subsequent controls; latest-only
  frame delivery and measured rendering overhead remain pending.
- Pacing/turbo/deterministic modes, long-instruction and fault stress tests,
  native terminal input/render isolation, and sustained app-input testing are
  not finished. The core's 64-boundary polling is cooperative, not a hard
  upper bound on any individual instruction or callback.
