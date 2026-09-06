# Rust interactive-emulator responsiveness

Status: active, staged implementation. Applies to the PC-E500 and IQ-7000
Rust machines, native terminal frontend, and browser/WASM frontend. Python
emulator implementation is explicitly outside this work's scope.

## Contract

CPU throughput is not a responsiveness guarantee. A guest ROM may ignore a
button or wait for an unsupported peripheral, but host controls must remain
usable and explain what is happening. Do not force interrupts, skip ROM loops,
stub device calls, or silently reset the machine to simulate responsiveness.

Keep CPU-relative timing, independently elapsed OFF time, host monotonic
execution budgets, and display presentation cadence distinct. Interactive
pacing is not proof of physically calibrated instruction timing.

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
   contracts. Browser ownership, cleanup, raw/assisted timing and an initial
   model-specific control subset are implemented; full character mapping and
   native terminal input still need qualification. Never conflate scheduler
   boundaries with retired instructions.
4. **Pacing and presentation:** explicit interactive/turbo/deterministic modes,
   bounded catch-up policy, latest-frame delivery, and cheap normal-play status.
   Browser latest-only delivery and shared Rust pacing/mode controls are
   implemented. Native pacing adoption, cheap-default diagnostics and
   per-capture cost qualification remain unfinished.
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
RTC is also not represented by the CPU timing-counter delta. The newer
`elapsed_timing_units_advanced` observation includes OFF idle separately for
pacing; it does not replace the CPU counter or drive architectural timers.

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

The public PR workflow initially ran eighteen short synthetic browser-control
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
- Browser ownership and ON contact wiring are implemented below. Complete
  model-specific text input and native terminal input remain unfinished. Pause
  latency does not measure a key being consumed by ROM firmware. ON contact
  assertions are tested, but sustained OFF/wake UI qualification remains open.
- Individual rendering/debug captures can still delay subsequent controls;
  browser delivery is latest-only, but measured rendering overhead and bounded
  trace/artifact processing remain pending.
- Pacing/turbo/deterministic modes, long-instruction and fault stress tests,
  native terminal input/render isolation, and sustained app-input testing are
  not finished. The core's 64-boundary polling is cooperative, not a hard
  upper bound on any individual instruction or callback.

## Browser contact ownership (2026-09-06)

The machine owner now arbitrates separate physical-keyboard, virtual-pointer,
script-physical and legacy diagnostic sources. Multiple host keys/pointers may
own one contact; releasing one source cannot release another. Rust's raw matrix
and ON setters remain the only normal-input path: no FIFO insertion, forced
KEYI or altered CPU scheduling. `input_contacts()` is a side-effect-free view
of actual Rust contacts, distinct from the host ownership record.

Virtual controls offer two explicit contracts:

- **Raw:** DOWN/UP change contacts immediately when the owner handles them.
- **Assisted (default):** a contact stays down for at least 40,000 submitted
  scheduler boundaries **from DOWN**, not another 40,000 after UP. This is a
  convenience policy, not physical keyboard timing or an instruction count.
  It advances only with requested machine execution, including inert OFF
  boundary budgets. Normal runs, explicit steps and debugger calls all stop
  their slices at pending input deadlines. Debugger stub actions do not count.

A repress cancels its old delayed release. If no electrical UP occurred, it is
one continuous contact, not proof of two typed characters. Several assisted
taps made while paused may overlap when execution resumes; this interface is
not a queued text-entry protocol. Full character-entry sequencing is still a
separate qualification item.

Pointer capture keeps a drag held until UP; cancellation, lost capture, blur,
hidden documents, disabling controls and model replacement release contacts
without running the guest. Distinct pointer/key identities survive overlapping
presses. Host keyboard events do not type into the guest while editing a text
field, using host shortcuts or composing text. Both Shift keys are independent
owners of one guest contact. UI input requests have finite acknowledgements and
machine-generation checks. An acknowledgement explicitly does **not** claim the
ROM consumed the key.

Browser scripts own their explicit matrix/ON presses **only for one script
invocation**. Success, cancellation, exceptions and unawaited-operation cleanup
all release those contacts after pending machine work drains; another host
owner remains held. This changes the earlier cross-script raw-hold behaviour.
Use one script invocation for a multi-step held-key experiment. Legacy
`keys.event` / `keyboard.injectEvent` deliberately still perturb debounce/FIFO
state, unlike `keys.phys`; their compatibility path restores the composed raw
contact level after injection. They are not normal-UI or silicon input proof.
Physical input codes reject invalid values instead of silently wrapping to a
different byte. Missing physical/ON adapters report an error instead of no-op.

Initial model-specific controls replace the PC-only six-button layout:
PC-E500 PF1–PF5, cursor, SHIFT/CAPS/ENTER/ON; IQ-7000 app keys,
SHIFT/CAPS/Search/Return/ENTER/ON. Full alpha/numeric/navigation mapping remains
pending. PC LEFT is corrected to physical `0x1F` (old `0x27` was ENTER).
IQ SHIFT is physical `0x02`; CAPS is `0x24`, not physical HOME `0x09`.
The initial IQ map follows ROM scanner/app evidence, not a complete independently
hardware-traced keyboard matrix.

Validation for this stage:

- 103 frontend unit/component tests passed; Svelte reported zero errors or
  warnings, and Prettier checks passed. Chunked-input tests cover direct steps
  and resumable calls, including debugger actions that must not advance holds.
- 25 Rust/WASM tests passed, including actual matrix/ON observation without
  guest execution or implicit interrupt acknowledgement.
- The native Rust suite passed 547 tests, with six existing ignored cases and
  the separately opt-in two-ROM chunking test ignored in this invocation.
  Native core Clippy passed.
- The final full public Chromium suite passed 34 tests, with four opt-in ROM
  cases skipped (49.4 seconds including the app rebuild). WASM-crate Clippy
  passed; its 18 pre-existing feature-gated core dependency warnings remain.
- Eleven compiled-browser tests passed in the focused input/physical-keyboard
  run, with one private-ROM test explicitly skipped. Both models covered
  overlapping owners, repress timing, focus/text-field cleanup, script success
  and runaway-script cancellation, ON ownership and stale-generation rejection.
- Four opt-in actual-ROM browser checks passed separately. The new IQ test
  clicks the real browser MEMO and CAPS controls, executes the Rust/WASM ROM,
  observes `MEMO ?` and the ROM-driven CAPS change, and captures the live LCD.
  It uses no input-FIFO or framebuffer injection. Earlier PC-E500 PF1/trace and
  IQ annunciator checks also passed. This is not a claim of all-app completeness.
- The short PR browser guard now includes the ten synthetic input tests;
  real-ROM and repeated latency measurement runs remain opt-in, not new
  mandatory private-data/hour-long CI jobs.

Reproduce the new private-ROM input check from `web/`:

```bash
CI=1 PCE500_E2E_PORT=4197 IQ7000_E2E_REAL_ROM=1 \
  npm run e2e -- e2e/input_ownership.spec.ts --workers=1 --grep 'real IQ'
```

This stage does not qualify native TUI input, full text entry, end-to-end OFF
wake, or a p99 input-to-firmware latency target. Those remain active goal work.

## Latest-only browser display delivery (2026-09-06)

The machine worker may have **one transferred display frame in flight** and
**one lazy request for the newest state**. The UI returns that frame's sequence
credit after applying the Svelte/canvas update. Until then, further render or
refresh requests replace the lazy request: they neither capture pixels nor
allocate a queue of stale screenshots/debug snapshots. Capture reads the current
machine when credit becomes available, not when the request was enqueued.

Credits are sequence-specific and survive ROM model changes. Old/duplicate
credits cannot release a newer frame, and an ignored old-generation frame is
still acknowledged so the current model can display. Pause, input and ROM-load
acknowledgements are independent of frame credit. Deferred capture is scheduled
in a later host turn; a failed capture reports a display error, not a fictional
CPU pause/reset. Machine faults discard deferred captures. Function Runner LCD
artifacts are separate, explicit captures and are not dropped by live-display
coalescing.

The two compiled-browser backpressure regressions, one per model, withhold real
frame credit, issue 200 refresh requests, verify that only one frame was sent,
and then Pause, assert/release ON, and replace the model before returning credit.
The next frame is from the replacement model; no stale queue drains afterward.
Unit tests cover lazy/coalesced capture, current-state capture, invalid credits,
fault discard and capture/transport error recovery. The two short synthetic
browser regressions are included in the PR guard, not private-ROM CI.

Final local validation: 108 frontend unit/component tests passed, Svelte had
zero errors/warnings, and Prettier passed. The full public Chromium suite passed
36 tests with four opt-in ROM cases skipped (45.4 seconds including rebuild).
After the last UI error-reporting change, the current app was rebuilt and both
backpressure tests plus four actual-ROM checks passed together (six tests,
12.2 seconds). The synthetic twenty-sample Pause smoke observed maxima of
15.65 ms for PC-E500 and 17.23 ms for IQ-7000 on the recorded M1 Ultra/Chromium
host; this is not a sustained p99, input-consumption or per-capture timing
guarantee, and does not supersede earlier slower observations.

This bounds **display backlog**, not a single Rust renderer, text decoder,
snapshot or artifact serialization. Those costs, native terminal rendering,
background-tab pacing and sustained latency qualification remain active work.

## Shared Rust pacing policy (2026-09-06)

`pacing::Pacer` and `CoreRuntime::run_automatic_slice` share the host policy
between native and WASM adapters. The pacer takes a supplied monotonic host
timestamp; it never reads the guest bus, skips firmware, jumps a timer, or
mutates RTC state. Explicit `run_slice`/step/call budgets remain unpaced.

- **Interactive:** accumulate nominal timing credit using the model timer
  profile (currently 1,024,000 compatibility units/s). This is not measured
  hardware MHz; IQ-7000 still uses the explicitly uncalibrated PC fallback.
- **Turbo:** execute without a host-speed throttle, still using bounded
  slices and polling host controls between slices.
- **Deterministic:** refuse autonomous Run; require an explicit scheduler
  boundary budget. Repeatability additionally requires identical initial
  machine/RTC state and boundary-scheduled inputs. Merely selecting this mode
  does not make a host-time seed or live human keyboard input deterministic.

Pacing charges actual elapsed timing from successful slices, not requested
boundaries or retired instructions. The CPU counter remains frozen in OFF.
A separate, nonarchitectural OFF-idle observation makes mixed RUN/OFF slices
account for **CPU timing plus OFF idle**, including the existing independently
advancing IQ RTC. The observation is not serialized as architectural snapshot
state; reset/load and Pause/Resume establish new host pacing epochs.

Positive catch-up credit is capped at 50 ms of nominal host time. Discarded
backlog is reported, not silently injected into guest timers or the RTC. Atomic
instruction overshoot remains debt; it is not truncated to fake the target
rate. Suggested host sleeps never exceed 4 ms. One expensive instruction or
callback is still not preemptible by this cooperative design.

The WASM adapter exposes mode selection, pacing status, rebase and automatic
slices, rejecting mode changes during an owned debugger call. Model reload
preserves the chosen mode but resets pacing credit and uses the new model's
timebase. Status includes CPU timing, total elapsed timing, retired instructions
and discarded host backlog without adding guest reads.

Validation: 555 native Rust tests passed (six existing ignored cases plus the
separate opt-in two-ROM test); 26 WASM tests passed. Core and WASM Clippy passed;
the WASM dependency retains its 18 pre-existing feature-gated warnings. Tests
cover fractional credit, host stalls, instruction debt, backward-clock failure,
rebasing, mode changes, explicit-budget preservation, OFF/RTC elapsed accounting,
and equal machine state under varied paced host schedules for both models in
RUN/HALT/OFF. Native terminal adoption remains the next separate stage.

The opt-in two-ROM release regression also passed: one million submitted
boundaries per model, comparing direct execution with deadline slices and with
both interactive and turbo pacing. Injected monotonic host jitter includes
one-second stalls without waiting in real time. Registers, power state, timing
counters, internal/external RAM, timer deadlines, RTC state and actual nonblank
LCD buffers matched exactly. This qualifies chunking/pacing, not calibrated
speed or all application behavior.

## Browser pacing controls (2026-09-06)

Continuous browser Run now calls the shared Rust automatic-slice API. The JS
adapter only limits the submitted slice to the next input-release boundary,
accounts for the actual boundary progress, and yields for the suggested host
delay. Pauses do not expire assisted input holds. Explicit Step and Function
Runner calls continue to use unthrottled, cancellable boundary budgets in every
mode. Rendering retains its independent FPS loop and latest-only delivery.

The mode selector is available only while paused and acknowledges the actual
worker operation before changing UI state. The worker independently rejects
mode changes while running or while another operation owns the machine. Invalid
mode names fail without mutating the machine. Deterministic mode refuses Run
at both UI and worker boundaries; it does not silently reset guest state or
change the default host-derived RTC seed. The UI explicitly explains the fixed
seed/input-schedule requirement. Stop and Start rebase pacing so paused host
time is never replayed as catch-up. An old display frame cannot overwrite an
acknowledged mode selection. The no-worker fallback uses the same WASM pacing
adapter, while browser user scripts still require worker isolation.

Pacing status reports the nominal timebase, lack of hardware calibration, and
discarded host backlog. Background browser throttling may therefore slow guest
elapsed time; this deliberately is not an always-wall-clock RTC. Turbo removes
the speed throttle, not execution/host yielding, input ownership or fault checks.

Validation: 111 frontend tests passed, Svelte reported zero errors/warnings,
and formatting passed. The full compiled Chromium suite passed 38 tests with
four opt-in ROM cases skipped (50.8 seconds including rebuild). Two new synthetic
mode regressions cover both models, actual worker state/progress, invalid modes,
explicit steps, frozen paused state and mode-change rejection during a huge
cancellable explicit request. They join the short PR control guard; no new
long-running mandatory ROM suite was added.

Four private-ROM browser regressions passed separately against the current
build (10.6 seconds): PC-E500 PF1 and traced function call, IQ-7000 annunciators
and actual MEMO/CAPS browser input. No framebuffer/FIFO injection is used as
evidence for those input captures. The twenty-sample synthetic foreground Pause
smoke observed maxima of 25.09 ms (PC-E500) and 21.50 ms (IQ-7000) on M1 Ultra /
Chromium 143. These short samples are not a sustained p99, hardware-speed proof
or input-to-firmware latency guarantee. Native input/render isolation, complete
model key maps, expensive artifact processing and broader qualification remain
active goal work.
