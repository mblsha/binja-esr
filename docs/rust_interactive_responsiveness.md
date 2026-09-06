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
   Normal running, explicit stepping and Function Runner `e.step()`/`e.call()` use
   bounded Rust execution;
   control acknowledgements, request failure handling and ROM-load generation
   checks and browser JavaScript isolation are implemented. Bounded artifact
   processing and broader runtime fault classification remain pending.
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

The public PR workflow runs only eighteen short synthetic browser-control
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

## Resumable Function Runner calls (2026-09-06)

The WASM debugger-call helper now has explicit begin, advance, stub-response,
finish and cancel operations. Each invocation has an ownership ID; stale replies,
overlapping calls, reset/load and conflicting register/memory/step operations
are rejected while it is active. Browser and Node Function Runners use these
operations with the same 4 ms cooperative host target and real event-loop
yields. The old synchronous helper remains an explicitly offline compatibility
API; the interactive worker no longer calls it.

The browser reports actual call progress, and Stop can cancel an executing
four-billion-step busy-loop request. Cancellation retains a partial call result
and reports an error to the script instead of pretending it returned normally.
`report.steps` preserves the historical call budget (scheduler boundaries plus
explicit debugger stub actions); `report.scheduler_boundaries` excludes stubs.
Neither is an oscillator-cycle count.

Stubs are handed to JavaScript only after Rust releases its mutable WASM borrow.
Their memory readers use current-machine debugger peeks, not pointers captured
before reset. Promise-returning stub handlers and Map/array call options are
rejected rather than silently becoming empty patches/options. The callback API
still requires a synchronous patch; isolation of arbitrary callback code is not
yet implemented.

This remains an **invasive debugger invocation**, not transparent foreground
execution. Finish/cancel restores the original PC, S, three sentinel stack
bytes and call bookkeeping. Other registers, guest writes, power/interrupt
state, timers, RTC and peripheral effects remain changed. A cancelled or timed
out stateful routine may therefore require a deliberate reset; do not treat
that result as a resumable application snapshot or automatically reset away
unsaved data. Restoration now precedes fallible trace/artifact encoding, so an
encoding failure cannot strand the debugger sentinel.

Validation of this stage:

- 24 WASM tests passed. New cases compare synchronous versus sliced calls for
  both models, four chunk sizes, timeout and actual RET/RETF instructions;
  compare registers, RAM, bus-access counters, timers and RTC status; and cover
  cancellation cleanup, trace ownership, stale call IDs and per-stub response
  sequences, invalid inputs, distinct
  fault/HALT outcomes and non-executing stub handoffs.
- 79 frontend tests passed, including cancellation artifacts, late stub replies,
  failure cleanup, overlapping script calls, and reset-safe memory readers.
- The full Chromium suite passed 13 tests with three opt-in ROM tests skipped.
  Both models' busy-loop tests wait for actual Rust call progress before Stop,
  then verify restored stack bytes and subsequent execution. Separate tests
  perform traced stub calls across reset in both models. These deliberately
  synthetic fixtures are not app or hardware evidence.
- Svelte checks, formatting and WASM-crate Clippy passed. Existing dependency
  feature-gating warnings remain as recorded above.
- Explicit Node Function Runner runs loaded both local licensed ROMs and invoked
  their reset-entry code through the resumable API with a diagnostic RAM stack.
  PC-E500 executed 12,116 call-budget steps before HALT; IQ-7000 exhausted the
  100,000-step budget without a fault. Both produced nonblank ROM-driven LCD
  captures. This is bounded ROM-execution evidence, not proof of a completed
  IQ-7000 startup or application-safe continuation after an invasive call.
- A repeated normal-run latency smoke on Apple M1 Ultra / Darwin 25.6.0 /
  Chromium 143.0.7499.4 measured maxima of 24 ms (PC-E500) and 22 ms (IQ-7000),
  twenty synthetic-ROM samples each. This does not supersede the earlier
  slower observations or qualify function-call/trace-finalization latency.

## Isolated browser scripts (2026-09-06)

Browser user code, stub handlers, probe handlers and trace-body callbacks now
execute in a disposable child worker. The Rust machine and `EvalApi` artifact
store stay in the machine owner. Stop terminates only the script worker, rejects
outstanding callback requests, cancels bounded execution, and drains pending
machine operations and their cleanup before acknowledging. Completed and partial
call/trace artifacts remain available. Late requests/replies cannot mutate the
next session. Unawaited overlapping machine mutations are rejected.

Synchronous register and debugger memory APIs use a request/reply mailbox, with
`Atomics.wait` **only in the script worker**. Each read queries the current owner;
there is no approximate machine snapshot or shared guest RAM. Artifact reads are
copies. Data crossing this boundary is JSON-shaped; nonfinite numbers fail
closed, and BigInts become decimal strings. Explicit callback APIs retain their
closures and result ordering. User object getters/serialization execute on the
script side, not in a machine-owned WASM borrow.

The shared mailbox requires a secure, cross-origin-isolated context. Vite
development/preview now sets headers before static worker serving; SvelteKit
sets them on its responses too. Production proxies/CDNs must set them on worker
assets as well as HTML. Plain LAN HTTP is not sufficient. Missing isolation
refuses Function Runner; the main-thread script fallback has been removed.
See [web setup](../web/README.md#function-runner-stubs) and the
[browser requirements](https://developer.mozilla.org/en-US/docs/Web/API/WorkerGlobalScope/crossOriginIsolated).

Limits: 16 MiB RPC replies, 30-second synchronous RPC waits, 64 outstanding RPC
jobs and a 270-second browser-script host deadline. Limits report failure and
cancel pending work; they are not changes to CPU/RTC timing or a security
sandbox. The CLI still uses offline user-script execution.

Validation:

- 89 frontend tests passed; Svelte checks and formatting passed. After final
  UTF-8 payload-limit hardening, all four targeted stub/isolation browser checks
  passed again.
- Full Chromium suite: 24 passed, three opt-in tests skipped. Eight tests run
  actual infinite JavaScript loops (script, stub, probe and trace callback, both
  models), wait for a real callback/print RPC, Stop, verify surviving RAM and
  debugger-stack bytes, and run another trace. Additional checks cover closure
  ordering, current registers, copied artifacts, missing isolation headers,
  mutation rejection during a pending step and no-Worker refusal. These are
  synthetic fixtures using the actual Rust/WASM runtime, not app/hardware proof.
- On the same Apple M1 Ultra / Chromium 143 reference host, those eight
  click-to-stopped-UI measurements were 134–153 ms (one per case). This includes
  Playwright click action and UI settlement; it is **not** the worker-ack metric.
  Twenty ordinary Run/Stop samples per model peaked at 16.66/15.57 ms. Neither
  limited sample set establishes a p99 guarantee or supersedes earlier slower
  real-ROM observations.
- Three existing opt-in tests passed using the actual local licensed ROMs:
  PC-E500 PF1 boot-menu navigation, its reset/PF1/traced-call sequence, and
  IQ-7000 MEMO followed by SHIFT with ROM-driven CAPS/key-beep/SHIFT segments.
  The IQ test saved the actual live canvas. This is narrow ROM-driven flow
  evidence, not full application/input qualification or hardware tracing.

### Remaining limitations

- Browser arbitrary JavaScript is isolated; native Rust execution and the Node
  CLI remain separate surfaces. Trace serialization, artifact growth and large
  result processing in the machine owner are not host-time bounded yet.
- The no-Worker fallback has bounded stepping but is not a qualified isolated
  production frontend; Function Runner is now refused there.
- Virtual-key release timing, repress/focus-loss handling and model-specific
  controls still need the input stage. Pause latency does not measure a key
  reaching the matrix or being consumed by ROM firmware. Scripted taps execute
  their owner-side release cleanup on cancellation, but explicit raw key-downs
  remain explicit state, including across a cancelled script; source-aware
  contact ownership/release still needs work.
  The browser adapter also still needs explicit `onKey` press/release wiring
  to the existing WASM exports (the optional adapter methods currently permit a
  silent no-op); cover ON wake with the input-stage regression tests.
- Rendering/debug snapshots can still delay subsequent controls; latest-only
  frame delivery and measured rendering overhead remain pending.
- Pacing/turbo/deterministic modes, long-instruction and fault stress tests,
  native terminal input/render isolation, and sustained app-input testing are
  not finished. The core's 64-boundary polling is cooperative, not a hard
  upper bound on any individual instruction or callback.
