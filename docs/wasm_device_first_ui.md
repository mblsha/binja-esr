# Device-first WASM UI

Work in progress, 2026-09-07. This document tracks implementation against the
full UI goal; it is not a claim that all milestones have passed validation.

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
- [ ] Complete power-state reporting and priority ON interaction tests.
- [x] Focus-on-device, delivered-contact feedback, shortcut overlay.
- [x] Qualified, cancellable paste with explicit unsupported-character handling.
- [ ] Destructive-action confirmation, supported restoration/export and fault actions.
- [x] Native-resolution LCD screenshot export with optional provenance metadata.
- [ ] Refine visible shell clutter and consolidate accuracy disclosures.
- [ ] Full bounded browser regression and real-ROM checks for both models.
- [ ] Visual review and coherent public/private commits.

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

These results cover this first layout/preset change, not the unfinished
milestones above. Decoded pixel equality is tested rather than equality of PNG
encoder output, which can differ while representing identical RGBA pixels.

## Input feedback and capture milestone

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
