# Device-first WASM UI

Completed bounded UI deliverable, 2026-09-07. Hardware-dependent accuracy and
unsupported snapshot restoration remain explicitly outside the verified claims.

## Interaction contract

- The normal surface is the device, Power / ON, Pause / Resume and an options
  menu. Advanced contains the bench controls, ROM/model selection and Function
  Runner. The menu offers Device and LCD-only views of the same emulator pixels.
- Power / ON presses the real ON input. It is not a reset, an emulator pause
  toggle, or an invented OFF key. Unqualified OFF contacts stay disabled.
- Pausing freezes emulated time. Device OFF must remain distinct from Pause:
  the scheduler can still advance the IQ RTC and deliver its alarm wake.
- Device pace defaults to nominal interactive pacing without typing acceleration.
  Responsive temporarily accelerates buffered scan work, including guest RTC
  and peripherals. Neither is hardware-calibrated. Turbo and deterministic
  explicit execution remain Advanced options.
- Debug observations are not restorable machine snapshots. The existing worker
  `snapshot` request obtains display/debug information only. Do not label this
  session backup or promise restoration from it.

## Delivery checklist

- [x] Move existing bench controls into a collapsed Advanced section.
- [x] Add a compact toolbar and menu with pacing presets and Device/LCD-only views.
- [x] Default both host paths to non-accelerated Device pace.
- [x] Complete power-state reporting and priority ON interaction tests.
- [x] Focus-on-device, delivered-contact feedback, shortcut overlay.
- [x] Qualified, cancellable paste with explicit unsupported-character handling.
- [x] Destructive-action confirmation, snapshot-scope disclosure, observation export and fault actions.
- [x] Native-resolution LCD screenshot export with optional provenance metadata.
- [x] Refine visible shell clutter and consolidate accuracy disclosures.
- [x] Full bounded browser regression and real-ROM checks for both models.
- [x] Visual review and coherent public/private commits.

## Final acceptance and limits

Final validation: 139 frontend tests, 27 WASM tests, 54 bounded Chromium checks
(13 opt-in cases skipped), and 12 selected private-ROM browser checks passed.
Svelte/TypeScript reported zero errors/warnings; formatting and production build
passed. Both final browser runs recorded `passed` with no failed tests. Real-ROM
full-page IQ MEMO and PC calculator captures were visually inspected.

| Requirement | Verification |
| --- | --- |
| Minimal toolbar, Advanced, Device/LCD-only, shortcut and accuracy panels | Device layout and input-ownership browser tests |
| Device OFF distinct from host Pause; real ON input | Rust power-state test and browser OFF fixture, including explicit paused stepping and ON hold |
| Focus, delivered contacts, cancellable qualified paste | Host-input unit tests, browser ownership tests, real-ROM MEMO and calculator input |
| Nominal default and disclosed Responsive acceleration | Pacing/input tests and real-ROM repeated-character checks with acceleration off/on |
| Actual LCD/annunciator export and matching metadata | Decoded downloaded PNG equality for both models; no DOM or text reconstruction |
| Reset/ROM/model protection and fault recovery | Cancellation preserves the session; confirmed reset replaces the worker; injected transport fault exports diagnostics and recovers |
| Snapshot scope | Explicit session-safety disclosure; no unsupported restoration or hidden partial autosave |

Power / ON submits the real contact with a minimum assisted hold even when raw
virtual taps are selected. It does not implicitly resume the host or reset the
machine. Fault recovery reloads the last successfully installed ROM in a fresh
worker only after confirmation; old-worker messages cannot mutate the new session.

**Full browser session restoration is not implemented.** Native Core snapshot
save/load is excluded from WASM, and its snapshot guards reject unrepresented
runtime/peripheral state, including the IQ RTC profile. The browser's debug
`snapshot` is not a complete backup. Current browser data is memory-only; reload,
reset or replacement can lose it. Diagnostics explicitly contain last-observed
state, potentially preceding a fault, not restorable RAM/peripheral/RTC state.

Remaining follow-ups require separate scope/evidence: measured/photo-scanned
cases, qualification of disabled physical keys and provisional annunciator bits,
hardware pacing calibration (IQ still uses a compatibility fallback), full
snapshot serialization/restoration, and broader character/IME composition.
Paste preserves actual CAPS/application semantics and rejects unsupported
characters instead of silently inserting or dropping them. This UI work does
not resolve existing ROM/input or silicon-correctness gaps by changing artwork.

The UI-structure regression uses a labelled synthetic ROM, not app proof. The
real-ROM checks must separately exercise actual application input and capture
the actual framebuffer. Future scans and unverified glass mappings remain
external evidence gaps, not grounds to fabricate a more complete device.

## First milestone validation (2026-09-07)

- 134 frontend tests pass; Svelte/TypeScript reports no errors or warnings.
- Production web build and formatting checks pass.
- 48 bounded Chromium checks pass (11 opt-in cases skipped), including both
  models' Device/LCD-only pixel equivalence and the non-accelerated default.
- 10 selected private-ROM browser checks pass, covering PC calculator input,
  IQ MEMO edit/store/reopen, CAPS/SHIFT annunciators, cursor editing and live
  zero-delay repeated-character typing with acceleration both off and on.

These results cover this first layout/preset change, not the later
milestones below. Decoded pixel equality is tested rather than equality of PNG
encoder output, which can differ while representing identical RGBA pixels.

## Input feedback and capture milestone

### Display-path optimization follow-up (2026-09-07)

The WASM `lcd_capture()` export now returns an owned `Uint8Array` using bulk
byte serialization rather than one JavaScript assignment per pixel. Its geometry,
gray levels and annunciator metadata are unchanged. The Function Runner's
`e.lcd.capture()` explicitly converts this to an owned plain JSON array, preserving
that scripting API. No view into growable WASM memory is retained or transferred.

Normal presentation uses `lcd_capture_if_changed(false)`. It compares the exact
logical matrix, LCD kind and annunciator sources before rendering, without hashes
or new mutation hooks. This small comparison catches debugger writes, rendered
controller changes and restored LCD contents. Reset/model replacement clears the
cache. Explicit captures remain independent, and an explicit worker `snapshot`
forces pixels even if unchanged. Forced requests survive frame coalescing, and
transfer/capture failures force a retry. Existing one-frame credit remains intact.

Unchanged frames still report fresh PC/instruction, input, power and pacing
observations; the UI retains the matching pixels. Chip-debug images are generated
only while their panel is open. Canvas presentation skips unchanged immutable
pixel references and reuses ImageData for changed frames of the same geometry.
Pixel generation, CPU execution, RTC progression and screenshot provenance are
not replaced by guessed text, host clocks or direct application state writes.

Chromium 143 release-WASM microbenchmarks, median of three repeated batches:

| Display operation | PC-E500 | IQ-7000 |
| --- | ---: | ---: |
| Full capture before | 0.507 ms | 7.611 ms |
| Full capture after | 0.070 ms | 0.634 ms |
| Unchanged-display check after | 0.010 ms | 0.0069 ms |

These are isolated display timings, not whole-emulator speedups. The idle cache
avoids rendering and transferring pixels; lightweight status updates continue.
The comparison still allocates a small logical matrix, and changed displays still
render the full glass at its original resolution. No CPU optimization or pacing
change is included.

Validation: 142 frontend tests, 31 WASM tests, 54 bounded Chromium checks (13
opt-in cases skipped), and 12 selected private-ROM browser checks passed. Tests
cover owned bulk pixels against the unchanged core renderer, all annunciator
source bytes, PC start-line/restore, forced capture/reset/model replacement,
suppressed pixel payloads with fresh status, debug-panel gating, PNG equivalence,
real-ROM input/paste and bounded cancellation. Type checks and formatting passed.

Clicking the device focuses keyboard input without stealing the dedicated pan
region's keyboard navigation. The menu exposes a compact shortcut reference.
Host-held cyan outlines remain distinct from depressed keycaps, which follow
Rust's side-effect-free `input_contacts()` observations on input changes and
ordinary frames. This is sampled electrical-contact feedback, not ROM-consumed
character acknowledgement; short contacts between display frames may not be
visible. It does not require opening debug panels.

**Save LCD PNG** downloads a copy of the currently observed emulator capture at
its intrinsic resolution, including IQ fixed segments. No DOM case screenshot,
OCR replacement, or framebuffer rewriting is involved. **Last capture metadata**
optionally downloads a same-named JSON sidecar with model/source/build, capture
geometry, PC/instruction observation, raw annunciator bytes and accuracy limits.
Metadata is copied before asynchronous PNG encoding, so subsequent frames cannot
relabel it. An old generation's image cannot be exported under a new ROM/model.
These are display observations, not restorable snapshots. Export does not pause
or advance the CPU.

Unit coverage distinguishes delivered contacts from host holds. Bounded browser
coverage verifies focus, queued A→B delivery/release, shortcut opening/closing,
and decoded downloaded-PNG equality to live pixels for both models, with matching
metadata filenames and geometry.

Milestone validation: 135 frontend tests, 49 bounded Chromium tests (11 opt-in
cases skipped), and 10 selected private-ROM checks passed. Svelte/TypeScript,
formatting and production build passed. Full-page real-ROM captures of IQ MEMO
and PC calculator were visually inspected; further shell decluttering and
accuracy-panel consolidation remain on the checklist.

## Paste milestone

Paste is previewed and explicitly confirmed. Unsupported characters reject the
whole plan without changing contacts. The same compiler validates the preview
and worker submission against the selected model's qualified key map; stale
machine generations are rejected. Up to 4,096 characters (up to 8,192 contacts
for IQ comma composition) feed the existing 128-key buffer incrementally.
Pause never advances the paste. Cancel/ON/lifecycle cleanup discard the full
plan, and competing key-downs are rejected during delivery. CAPS/app-state and
PC ENTER execution consequences are disclosed before submission.

The real-ROM acceptance adds IQ MEMO `PASTE ONE,2` plus a newline and `SECOND`,
and a fresh PC calculator receiving confirmed `11+22` plus ENTER and producing
`33.`. These tests use preview UI and ordinary paced Run, not record writers or
direct application calls. Details and limits are in
[Rust input correctness](rust_input_correctness.md#previewed-paste).

Validation: 139 frontend tests, 51 bounded Chromium checks (13 opt-in cases
skipped), and 12 selected private-ROM tests passed. Type checks, formatting,
and production build passed. Additional host tests verify clipboard preview
without immediate input, preserved host-editor paste, and stale-generation
rejection before any contact mutation.
