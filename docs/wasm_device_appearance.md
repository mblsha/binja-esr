# WASM device appearance and scan handoff

Status: reference-based 2D case layouts, 2026-09-06. **Not measured replicas,
photo textures, 3D models, or scan-derived assets.** This stage changes the web
presentation only; it does not change Rust, input contacts, RTC, LCD decoding,
or the unresolved input findings in [Rust input correctness](rust_input_correctness.md).

## UI audit and implemented changes

| Previous UI issue | Current treatment |
| --- | --- |
| Both devices used the same generic six-column key grid | IQ book-style layout with ABCDEF rows and a lower numeric pad; PC wide case with QWERTY, PF, scientific and numeric groups |
| The disconnected LCD and keys did not resemble a device | Actual framebuffer sits in a case-mounted viewport, with the IQ fixed-segment area still part of the emulator's image |
| Long timing/build paragraphs displaced the device below the first screen | Compact host toolbar; session/timing disclosure below the case; errors remain above the case |
| Many physical keycaps were missing from the visual surface | Show reference-photo keycaps, but disable unqualified contacts and mark them with an amber dot and explanatory accessible name/title |
| Generic key reflow lost physical relationships | Preserve the reference layout, allow horizontal panning on narrow screens instead of squeezing all keys into tiny targets |
| No clear upgrade path to scanned cases | Versioned geometry, stable semantic key IDs, and presentation-only shell components |

The visual references inspected were the collector-owned
[IQ-7000 front photograph](https://vintagecomputer.net/sharp/IQ-7000/Sharp_IQ-7000.jpg)
and [PC-E500 front photograph](https://sharp.ledudu.com/images/pockets/sharp/machines/SHARPPCE500.jpg).
These support approximate key groups, relative positions, coloring and case
features—not exact dimensions or keyboard electrical mappings. Their image
bytes are not copied, hotlinked or redistributed by the UI. Case artwork,
mode pictograms and materials are original CSS/SVG approximations. Printed
legends and regional/model revisions still need comparison to our actual units.

## Boundaries that must survive a skin change

- [`device_layout.ts`](../web/src/lib/device_layout.ts): viewBox-space geometry,
  LCD rectangle, stable key IDs, face/shift legends, visual tones and provenance.
  Coordinates are converted to percentages, **not millimeters**.
- [`physical_keys.json`](../sc62015/core/data/physical_keys.json): authoritative
  implemented keycap-to-matrix mapping shared with native Rust. A skin resolves
  a semantic ID through this map; it must never invent an event or FIFO write.
- [`DeviceShell.svelte`](../web/src/lib/components/DeviceShell.svelte): case,
  hinge, bezel, decorative card bay and LCD slot. No machine ownership or clock.
- [`VirtualKeyboard.svelte`](../web/src/lib/components/VirtualKeyboard.svelte):
  the same pointer capture, keyboard/assistive activation, cancellation and
  source-owner handling used before. The geometry is optional; its compact
  layout remains available to component consumers.
- [`LcdCanvas.svelte`](../web/src/lib/components/LcdCanvas.svelte): same framebuffer
  bytes, intrinsic dimensions and pixel conversion; fit within the case while
  preserving aspect ratio. No text replacement, color filter or synthetic app
  screen. LCD-only proof captures and full-device proof captures are distinct.

The IQ card bay is explicitly labelled as an artwork placeholder, not a second
display or a claim about which card is emulated. It must remain replaceable;
scanning a particular inserted card must not bake that card into the base case.

Unmapped case buttons (including OFF and currently unqualified scientific or
memory keys) do not work merely because they are drawn. They are disabled, do
not emit input, and are tested as such. ON still uses the power-key input. Host
Run/Stop, execution mode, ROM loading and diagnostics stay **outside** the case.

## Proposed capture package for our own units

This is a project capture checklist, not a claim that capture/calibration has
already happened. First take a small pilot set and verify its scale, sharpness
and glare before committing to a full scan session.

1. **Identify each exact unit.** Model, language/revision, inserted card and
   condition. Decide whether the target is this specimen's wear or a clean
   factory-like reconstruction. Record image ownership and permission to ship
   derived assets; keep serial numbers or private reflections out of releases.
2. **Measure the anchors.** Case width/height/depth; key centers and pitch;
   LCD glass and actual active matrix/annunciator rectangles; hinge axis and
   open angle; rest positions and travel of keys/latches. Include an in-plane
   scale reference. Never infer physical scale from an arbitrary photo crop.
3. **Photograph artwork separately.** Straight-on high-resolution views of
   each panel, plus macro coverage of every key legend, status icon, badge,
   card insert and connector label. Keep original captures and metadata; use
   consistent lighting and color references, and check glare on black plastic
   and the LCD before the full set. Do not bake a running app image into glass.
4. **Capture geometry separately.** Exterior from multiple views, including
   rear, edges, hinge, keycap sides, latches and accessible ports. Capture the
   IQ panels and hinge as distinct moving parts. No disassembly is necessary
   for this initial external-case goal. Do not use sprays or coatings on the
   vintage units without a separately approved preservation-safe procedure.
5. **Keep raw and runtime assets separate.** Archive original scans/photos
   privately; derive a lightweight mesh, texture atlas and measured anchors
   for the browser. Validate those against the original images and measurements
   before labelling anything scan-faithful.

## Future 3D asset contract (not implemented yet)

Use a calibrated scene with a documented physical unit and axis convention,
separate case/key/hinge nodes, and explicit provenance. A suitable delivery
container is [glTF/GLB](https://www.khronos.org/gltf/), which supports meshes,
materials, textures and animations; this is a proposed asset target, not an
existing importer or dependency in our app.

Proposed node roles are `case.left`, `case.right`, `hinge`, `lcd.surface`,
`card.insert` and one key node per stable `DeviceKey.id` (for example `PF1`,
`MEMO`, `ENTER`). Ray hits would resolve those IDs through the **same** input
adapter. Key travel is visual animation, never another trigger for a press.
The LCD would use the live emulator image as a texture, with calibrated UVs,
not a screenshot baked into the case material. Keep a native-resolution,
untinted LCD capture path for correctness comparisons.

Keep the present 2D keyboard as an accessible/low-GPU fallback. A renderer may
drop a frame under load, but must not block Pause/Stop or key release. Choose
mesh and texture budgets only after testing on the intended devices; full
resolution source scans should not become mandatory runtime downloads.

## Acceptance and remaining work

Local validation: 116 frontend unit/integration tests, 38 bounded Chromium
regressions and six selected private-ROM browser checks passed; Svelte reported
zero errors/warnings and formatting/build checks passed. The private-ROM checks
use the actual compiled Rust/WASM machine, not a mocked display. No Rust source
changed in this presentation stage.

Unit coverage checks every supported key is present exactly once, stable IDs,
in-bounds/nonoverlapping hit regions, correct alphabet grouping, and disabled
unmapped contacts. Browser coverage checks narrow-screen panning plus the
existing focus/owner/cancellation tests. Private-ROM acceptance still types,
edits, saves and recalls IQ MEMO and evaluates PC CAL `2+2` as `4`, now through
the case buttons; full-device PNGs accompany the existing LCD-only proof PNGs.

Still approximate: dimensions, LCD physical pixel aspect/optics, fonts, colors,
legends, exact key silhouettes/travel, hinges/latches/ports and card artwork.
Not implemented: 3D rendering, scanning/import tooling, photo texture loading,
folding animation, a complete electrical map for all displayed keycaps, or any
fix for the existing BASIC/colon/early-boot input blockers. Case appearance
must not be presented as additional silicon-correctness evidence.
