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
- [ ] Focus-on-device, delivered-contact feedback, shortcut overlay.
- [ ] Qualified, cancellable paste with explicit unsupported-character handling.
- [ ] Destructive-action confirmation, supported restoration/export and fault actions.
- [ ] Native LCD screenshot export with optional provenance metadata.
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
